// malachitectl <fwdir> [start|stop|disp|show|pmcycle|shot <file>]... -- hand Gpu its firmware and drive it; output goes to the kernel log.
// "shot <file.ppm>" saves what the lit head shows, "h264 <clip.h264> <dir>" decodes a clip on NVDEC into dir.
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <zlib.h>

#include "../kext/abi.h"

static int shot(io_connect_t conn, const char *path)
{
    size_t len = sizeof(struct malachite_shot) + (64u << 20);
    uint8_t *buf = malloc(len);
    uint64_t args[2] = { (uintptr_t)buf, len };
    kern_return_t kr = IOConnectCallScalarMethod(conn, MALACHITE_DISP_READ, args, 2, NULL, NULL);
    if (kr) {
        fprintf(stderr, "shot: %#x\n", kr);
        free(buf);
        return -1;
    }
    const struct malachite_shot *h = (const void *)buf;
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        free(buf);
        return -1;
    }
    fprintf(f, "P6\n%u %u\n255\n", h->width, h->height);
    for (uint32_t y = 0; y < h->height; y++) {
        const uint8_t *row = buf + sizeof(*h) + (size_t)y * h->pitch;
        for (uint32_t x = 0; x < h->width; x++) {
            uint8_t rgb[3] = { row[4 * x + 2], row[4 * x + 1], row[4 * x] };
            fwrite(rgb, 1, 3, f);
        }
    }
    fclose(f);
    free(buf);
    printf("shot: %ux%u -> %s\n", h->width, h->height, path);
    return 0;
}

static uint8_t *slurp(const char *path, size_t *len)
{
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) {
        perror(path);
        return NULL;
    }
    uint8_t *buf = malloc(st.st_size);
    *len = read(fd, buf, st.st_size) == st.st_size ? (size_t)st.st_size : 0;
    close(fd);
    if (!*len) {
        free(buf);
        return NULL;
    }
    return buf;
}

static void write_ppm(const char *path, const struct malachite_frame *f, const uint8_t *nv12)
{
    FILE *out = fopen(path, "wb");
    if (!out) {
        perror(path);
        return;
    }
    fprintf(out, "P6\n%u %u\n255\n", f->width, f->height);
    const uint8_t *uv = nv12 + (size_t)f->width * f->height;
    for (uint32_t y = 0; y < f->height; y++) {
        for (uint32_t x = 0; x < f->width; x++) {
            // BT.601, video range
            const uint8_t *chroma = uv + (size_t)(y / 2) * f->width + (x & ~1u);
            int c = 298 * (nv12[(size_t)y * f->width + x] - 16) + 128, d = chroma[0] - 128, e = chroma[1] - 128;
            int rgb[3] = { (c + 409 * e) >> 8, (c - 100 * d - 208 * e) >> 8, (c + 516 * d) >> 8 };
            for (int i = 0; i < 3; i++)
                fputc(rgb[i] < 0 ? 0 : rgb[i] > 255 ? 255 : rgb[i], out);
        }
    }
    fclose(out);
}

static int by_display(const void *a, const void *b)
{
    const struct malachite_frame *x = *(const struct malachite_frame *const *)a;
    const struct malachite_frame *y = *(const struct malachite_frame *const *)b;
    if (x->idr_count != y->idr_count)
        return x->idr_count < y->idr_count ? -1 : 1;
    return (x->poc > y->poc) - (x->poc < y->poc);
}

// Decode clip on NVDEC, write its pictures in display order to dir as frameNN.ppm and frames.nv12, and check each
// picture's CRC-32 against the lines of clip with .crc for .h264, if there is one (made from a software decoder's
// NV12, ffmpeg -i clip.h264 -f rawvideo -pix_fmt nv12)
static int h264(io_connect_t conn, const char *clip, const char *dir)
{
    size_t len;
    uint8_t *stream = slurp(clip, &len);
    if (!stream)
        return -1;
    size_t out_len = 256u << 20;
    uint8_t *out = calloc(1, out_len);
    uint64_t args[5] = { (uintptr_t)stream, len, (uintptr_t)out, out_len, 300 };
    kern_return_t kr = IOConnectCallScalarMethod(conn, MALACHITE_H264_DECODE, args, 5, NULL, NULL);
    free(stream);
    const struct malachite_frames *hdr = (const void *)out;
    uint32_t n = hdr->count;  // what the kernel got out, failure or not
    if (kr)
        fprintf(stderr, "h264: %#x after %u pictures\n", kr, n);

    const struct malachite_frame **frames = calloc(n + 1, sizeof(*frames));
    size_t off = sizeof(*hdr);
    for (uint32_t i = 0; i < n; i++) {
        frames[i] = (const void *)(out + off);
        off += sizeof(**frames) + (size_t)frames[i]->width * frames[i]->height * 3 / 2;
    }
    qsort(frames, n, sizeof(*frames), by_display);

    char path[1024];
    mkdir(dir, 0755);
    snprintf(path, sizeof(path), "%s/frames.nv12", dir);
    FILE *raw = fopen(path, "wb");
    snprintf(path, sizeof(path), "%.*s.crc", (int)(strlen(clip) > 5 ? strlen(clip) - 5 : strlen(clip)), clip);
    FILE *crcs = fopen(path, "r");
    uint32_t matched = 0, checked = 0;
    for (uint32_t i = 0; i < n; i++) {
        const struct malachite_frame *f = frames[i];
        const uint8_t *nv12 = (const uint8_t *)(f + 1);
        size_t bytes = (size_t)f->width * f->height * 3 / 2;
        uint32_t crc = (uint32_t)crc32(0, nv12, (uInt)bytes);
        unsigned want = 0;
        int have = crcs && fscanf(crcs, "%x", &want) == 1;
        printf("frame %2u: picture %2u, poc %3d, %ux%u, crc %08x%s\n", i, f->index, f->poc, f->width, f->height, crc,
               !have ? "" : crc == want ? " ok" : " MISMATCH");
        checked += have;
        matched += have && crc == want;
        if (raw)
            fwrite(nv12, 1, bytes, raw);
        snprintf(path, sizeof(path), "%s/frame%02u.ppm", dir, i);
        write_ppm(path, f, nv12);
    }
    if (raw)
        fclose(raw);
    if (crcs)
        fclose(crcs);
    printf("h264: %u pictures into %s", n, dir);
    if (checked)
        printf(", %u of %u match the reference", matched, checked);
    printf("\n");
    free(frames);
    free(out);
    return kr || !n || matched != checked ? -1 : 0;
}

static int load(io_connect_t conn, const char *dir, int index, const char *name)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) {
        perror(path);
        return -1;
    }
    void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t args[3] = { index, (uintptr_t)data, st.st_size };
    kern_return_t kr = IOConnectCallScalarMethod(conn, MALACHITE_LOAD_FIRMWARE, args, 3, NULL, NULL);
    munmap(data, st.st_size);
    if (kr) {
        fprintf(stderr, "loading %s: %#x\n", name, kr);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <fwdir> [start|stop|disp|show|pmcycle|shot <file>]...\n", argv[0]);
        return 1;
    }
    io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("MalachiteGpu"));
    io_connect_t conn;
    if (!svc || IOServiceOpen(svc, mach_task_self(), 0, &conn)) {
        fprintf(stderr, "Malachite is not loaded\n");
        return 1;
    }

    static const char *const names[] = MALACHITE_FW_NAMES;
    for (int i = 0; i < MALACHITE_FW_COUNT; i++)
        if (load(conn, argv[1], i, names[i]))
            return 1;

    static const struct { const char *name; uint32_t selector; } cmds[] = {
        { "start", MALACHITE_GSP_START },
        { "stop", MALACHITE_GSP_STOP },
        { "disp", MALACHITE_DISP_PROBE },
        { "show", MALACHITE_DISP_SHOW },
        { "pmcycle", MALACHITE_PM_CYCLE },
    };
    int ret = 0;
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], "shot") && a + 1 < argc) {
            ret |= shot(conn, argv[++a]) != 0;
            continue;
        }
        if (!strcmp(argv[a], "h264") && a + 2 < argc) {
            ret |= h264(conn, argv[a + 1], argv[a + 2]) != 0;
            a += 2;
            continue;
        }
        unsigned i;
        for (i = 0; i < sizeof(cmds) / sizeof(cmds[0]) && strcmp(argv[a], cmds[i].name); i++)
            ;
        if (i == sizeof(cmds) / sizeof(cmds[0])) {
            fprintf(stderr, "unknown command %s\n", argv[a]);
            return 1;
        }
        kern_return_t kr = IOConnectCallScalarMethod(conn, cmds[i].selector, NULL, 0, NULL, NULL);
        printf("%s: %s (%#x)\n", argv[a], kr ? "failed" : "ok", kr);
        if (kr)
            ret = 1;
    }
    IOServiceClose(conn);
    return ret;
}
