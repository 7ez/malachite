// GSP-RM r570.144 boot on GA102, following nouveau (subdev/gsp/tu102.c, ga102.c, rm/r535/gsp.c, rpc.c,
// rm/r570/gsp.c) and nova-core (gsp/boot.rs, fb.rs)
#include "chip.h"
#include "../nvrm/disp.h"
#include "../nvrm/engine.h"
#include "../nvrm/fbsr.h"
#include "../nvrm/gr.h"
#include "../nvrm/fifo.h"
#include "../nvrm/gsp.h"
#include "../nvrm/msgfn.h"
#include "../nvrm/rpcfn.h"
#include "booter.h"
#include "falcon.h"
#include "fwsec.h"
#include "gr.h"
#include "gsp.h"
#include "mem.h"
#include "pwr.h"

#define PAGE 0x1000ull
#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((uint64_t)(a) - 1))
#define ALIGN_DOWN(x, a) ((x) & ~((uint64_t)(a) - 1))

#define QUEUE_SIZE 0x40000
#define GSP_FW_FLAGS_CLOCK_BOOST 1  // GspFwWprMeta.flags: GSP-RM boots with boost clocks

struct msg_elem {
    uint8_t auth_tag[16], aad[16];
    uint32_t checksum, sequence, elem_count, pad;
    rpc_message_header_v rpc;
};

struct elf64_ehdr {
    uint8_t ident[16];
    uint16_t type, machine;
    uint32_t version;
    uint64_t entry, phoff, shoff;
    uint32_t flags;
    uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
};

struct elf64_shdr {
    uint32_t name, type;
    uint64_t flags, addr, offset, size;
    uint32_t link, info;
    uint64_t addralign, entsize;
};

// a buffer gsp_dma() handed out, on the list gsp_stop() frees
struct owned_dma {
    struct dma d;
    struct owned_dma *next;
};

struct gsp {
    struct gpu *g;
    struct dma flush, fw, lvl0, lvl1, lvl2, sig, boot, libos, loginit, logintr, logrm, rmargs, shm, wpr_meta;
    struct owned_dma *owned;  // extra buffers handed out by gsp_dma(), freed by gsp_stop()
    uint64_t fw_size;
    uint32_t app_version;
    msgqTxHeader *cmdq_tx, *msgq_tx;
    msgqRxHeader *cmdq_rx, *msgq_rx;
    uint8_t *cmdq, *msgq;
    uint32_t seq;
    uint32_t init_result;
    struct msg_elem *last;  // most recent message, kept until the next one arrives
    uint32_t last_size;
    uint32_t ctrl_status;  // of the last control, ~0 when GSP-RM didn't answer
    struct rm_client internal;  // GSP-RM's own client, for internal controls
    struct gsp_info info;
    struct vram vram;         // every VRAM allocation made while GSP-RM runs
    struct gr_global *gr;     // the graphics context buffers kept from golden context init
    uint32_t clients;         // rm_client()s handed out
    struct {
        gsp_rc_fn fn;
        void *ctx;
    } rc[2];                  // told about channel recovery: the kernel's own channels', the frontend's clients'
    gsp_event_fn event_fn;    // and about posted events
    void *event_ctx;
    uint32_t vec_gsp, vec_disp;  // CPU interrupt vectors: GSP-RM's messages, head timing
    uint32_t vec_ns[16], n_ns;   // and the nonstall ones of graphics and the copy engines (work done)
    uint32_t irqs;
    uint32_t chids[GSP_CHIDS / 32];  // channel IDs handed out (the first few are the core's own, fixed)
    uint64_t fb_rsvd;  // VRAM GSP-RM reserves between the last usable region and its non-WPR heap
    // held from gsp_suspend() to gsp_resume(): WPR2 as Booter-Unload saved it (radix3, described by meta) and
    // GSP-RM's VRAM outside WPR2, saved by GSP-RM itself (FBSR)
    struct {
        struct dma data, lvl0, lvl1, lvl2, meta, fbsr;
    } sr;
};

static int dma(struct gsp *s, struct dma *d, size_t size, int contiguous)
{
    if (dma_alloc(s->g, d, size, contiguous)) {
        log_msg("gsp: DMA allocation of %llu bytes failed\n", (unsigned long long)size);
        return -1;
    }
    return 0;
}

static const void *elf_section(const uint8_t *img, const char *name, uint64_t *size)
{
    const struct elf64_ehdr *eh = (const void *)img;
    const struct elf64_shdr *sh = (const void *)(img + eh->shoff);
    const char *names = (const char *)img + sh[eh->shstrndx].offset;
    for (int i = 0; i < eh->shnum; i++) {
        if (!strcmp(names + sh[i].name, name)) {
            *size = sh[i].size;
            return img + sh[i].offset;
        }
    }
    log_msg("gsp: no ELF section %s\n", name);
    return NULL;
}

// radix3 over data's pages: level 0 -> level 1 -> level 2 -> data pages
static int radix3(struct gsp *s, const struct dma *data, uint64_t size, struct dma *lvl0, struct dma *lvl1,
                  struct dma *lvl2)
{
    uint64_t pages = size / PAGE;
    if (dma(s, lvl0, PAGE, 1) || dma(s, lvl1, PAGE, 1) || dma(s, lvl2, ALIGN_UP(pages * 8, PAGE), 0))
        return -1;
    uint64_t *l2 = lvl2->cpu, *l1 = lvl1->cpu, *l0 = lvl0->cpu;
    for (uint64_t i = 0; i < pages; i++)
        l2[i] = dma_addr(data, i * PAGE);
    for (uint64_t i = 0; i < lvl2->size / PAGE; i++)
        l1[i] = dma_addr(lvl2, i * PAGE);
    l0[0] = lvl1->addr;
    return 0;
}

static int load_firmware(struct gsp *s)
{
    size_t size;
    const uint8_t *elf = fw_get(s->g, "gsp.bin", &size);
    uint64_t n;
    const void *img = elf ? elf_section(elf, ".fwimage", &n) : NULL;
    if (!img || dma(s, &s->fw, n, 0))
        return -1;
    s->fw_size = n;
    memcpy(s->fw.cpu, img, n);

    const void *sig = elf_section(elf, s->g->chip->fwsig, &n);
    if (!sig || dma(s, &s->sig, ALIGN_UP(n, 256), 1))
        return -1;
    memcpy(s->sig.cpu, sig, n);
    s->sig.size = n;

    if (radix3(s, &s->fw, s->fw_size, &s->lvl0, &s->lvl1, &s->lvl2))
        return -1;

    const uint8_t *bl = fw_get(s->g, "bootloader.bin", &size);
    if (!bl)
        return -1;
    const uint32_t *hdr = (const void *)bl;  // nvfw_bin_hdr
    const RM_RISCV_UCODE_DESC *desc = (const void *)(bl + hdr[3]);
    if (dma(s, &s->boot, hdr[5], 1))
        return -1;
    memcpy(s->boot.cpu, bl + hdr[4], hdr[5]);
    s->boot.size = hdr[5];
    s->app_version = desc->appVersion;

    if (dma(s, &s->wpr_meta, sizeof(GspFwWprMeta), 1))
        return -1;
    GspFwWprMeta *meta = s->wpr_meta.cpu;
    meta->bootloaderCodeOffset = desc->monitorCodeOffset;
    meta->bootloaderDataOffset = desc->monitorDataOffset;
    meta->bootloaderManifestOffset = desc->manifestOffset;
    log_msg("gsp: firmware %llu MB, bootloader %llu KB\n", (unsigned long long)(s->fw_size >> 20),
            (unsigned long long)(s->boot.size >> 10));
    return 0;
}

static uint64_t libos_id8(const char *name)
{
    uint64_t id = 0;
    for (; *name; name++)
        id = (id << 8) | (uint8_t)*name;
    return id;
}

static int log_region(struct gsp *s, LibosMemoryRegionInitArgument *arg, struct dma *d, const char *name)
{
    if (dma(s, d, 0x10000, 1))
        return -1;
    uint64_t *pte = (uint64_t *)d->cpu + 1;
    for (uint64_t i = 0; i < d->size / PAGE; i++)
        pte[i] = d->addr + i * PAGE;
    arg->id8 = libos_id8(name);
    arg->pa = d->addr;
    arg->size = d->size;
    arg->kind = LIBOS_MEMORY_REGION_CONTIGUOUS;
    arg->loc = LIBOS_MEMORY_REGION_LOC_SYSMEM;
    return 0;
}

static int init_libos(struct gsp *s)
{
    if (dma(s, &s->libos, PAGE, 1))
        return -1;
    LibosMemoryRegionInitArgument *args = s->libos.cpu;
    if (log_region(s, &args[0], &s->loginit, "LOGINIT") || log_region(s, &args[1], &s->logintr, "LOGINTR") ||
        log_region(s, &args[2], &s->logrm, "LOGRM"))
        return -1;

    // shared memory: page table for itself, then command queue, then status (message) queue
    uint64_t nr = (2 * QUEUE_SIZE) / PAGE;
    nr += ALIGN_UP(nr * 8, PAGE) / PAGE;
    uint64_t ptes_size = ALIGN_UP(nr * 8, PAGE);
    if (dma(s, &s->shm, ptes_size + 2 * QUEUE_SIZE, 1))
        return -1;
    uint64_t *ptes = s->shm.cpu;
    for (uint64_t i = 0; i < nr; i++)
        ptes[i] = s->shm.addr + i * PAGE;
    s->cmdq = (uint8_t *)s->shm.cpu + ptes_size;
    s->msgq = s->cmdq + QUEUE_SIZE;

    s->cmdq_tx = (msgqTxHeader *)s->cmdq;
    s->cmdq_rx = (msgqRxHeader *)(s->cmdq + sizeof(msgqTxHeader));
    s->msgq_tx = (msgqTxHeader *)s->msgq;
    s->msgq_rx = (msgqRxHeader *)(s->msgq + sizeof(msgqTxHeader));

    if (dma(s, &s->rmargs, PAGE, 1))
        return -1;
    GSP_ARGUMENTS_CACHED *rm = s->rmargs.cpu;
    rm->messageQueueInitArguments.sharedMemPhysAddr = s->shm.addr;
    rm->messageQueueInitArguments.pageTableEntryCount = nr;
    rm->messageQueueInitArguments.cmdQueueOffset = ptes_size;
    rm->messageQueueInitArguments.statQueueOffset = ptes_size + QUEUE_SIZE;
    rm->bDmemStack = 1;

    args[3].id8 = libos_id8("RMARGS");
    args[3].pa = s->rmargs.addr;
    args[3].size = s->rmargs.size;
    args[3].kind = LIBOS_MEMORY_REGION_CONTIGUOUS;
    args[3].loc = LIBOS_MEMORY_REGION_LOC_SYSMEM;
    return 0;
}

// empty queues for a boot: a GSP-RM booted before left its pointers and messages in them
static void queue_reset(struct gsp *s)
{
    memset(s->cmdq, 0, 2 * QUEUE_SIZE);
    s->cmdq_tx->size = QUEUE_SIZE;
    s->cmdq_tx->entryOff = PAGE;
    s->cmdq_tx->msgSize = PAGE;
    s->cmdq_tx->msgCount = (QUEUE_SIZE - PAGE) / PAGE;
    s->cmdq_tx->flags = 1;
    s->cmdq_tx->rxHdrOff = sizeof(msgqTxHeader);
    s->seq = 0;
}

static void cmdq_elem(struct gsp *s, uint32_t fn, const void *payload, uint32_t len)
{
    uint32_t rpc_len = sizeof(rpc_message_header_v) + len;
    uint32_t total = ALIGN_UP(__builtin_offsetof(struct msg_elem, rpc) + ALIGN_UP(rpc_len, 8), PAGE);
    struct msg_elem *m = mem_alloc(total);
    m->rpc.header_version = 0x03000000;
    m->rpc.signature = ('C' << 24) | ('P' << 16) | ('R' << 8) | 'V';
    m->rpc.length = rpc_len;
    m->rpc.function = fn;
    m->rpc.rpc_result = 0xffffffff;
    m->rpc.rpc_result_private = 0xffffffff;
    memcpy(m->rpc.rpc_message_data, payload, len);
    m->sequence = s->seq++;
    m->elem_count = total / PAGE;

    uint64_t csum = 0;
    for (uint64_t *p = (uint64_t *)m; p < (uint64_t *)((uint8_t *)m + total); p++)
        csum ^= *p;
    m->checksum = (uint32_t)(csum >> 32) ^ (uint32_t)csum;

    uint32_t cnt = s->cmdq_tx->msgCount, wptr = s->cmdq_tx->writePtr;
    // GSP-RM's read pointer into the command queue sits in the message queue's header
    uint64_t end = time_us() + 1000000;
    while ((s->msgq_rx->readPtr + cnt - wptr - 1) % cnt < m->elem_count) {
        if (time_us() > end) {
            log_msg("gsp: command queue full, rpc %u dropped\n", fn);
            mem_free(m);
            return;
        }
        delay_us(10);
    }
    for (uint32_t off = 0; off < total; off += PAGE) {
        memcpy(s->cmdq + PAGE + wptr * PAGE, (uint8_t *)m + off, PAGE);
        if (++wptr == cnt)
            wptr = 0;
    }
    __sync_synchronize();
    s->cmdq_tx->writePtr = wptr;
    __sync_synchronize();
    flcn_wr(s->g, &gsp_falcon, 0xc00, 0);
    mem_free(m);
}

// a queue element holds 16 pages at most, headers included; a longer payload goes on in continuation records
static void cmdq_push(struct gsp *s, uint32_t fn, const void *payload, uint32_t len)
{
    const uint32_t max = 16 * PAGE - __builtin_offsetof(struct msg_elem, rpc.rpc_message_data);
    uint32_t off = 0;
    do {
        uint32_t n = len - off < max ? len - off : max;
        cmdq_elem(s, off ? NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD : fn, (const uint8_t *)payload + off, n);
        off += n;
    } while (off < len);
}

static void queue_init_rpcs(struct gsp *s)
{
    GspSystemInfo *info = mem_alloc(sizeof(*info));
    info->gpuPhysAddr = bar_phys(s->g, 0);
    info->gpuPhysFbAddr = bar_phys(s->g, 1);
    info->gpuPhysInstAddr = bar_phys(s->g, 3);
    info->nvDomainBusDeviceFunc = pci_bdf(s->g);
    info->maxUserVa = 0x7ffffffff000ull;
    info->pciConfigMirrorBase = 0x088000;
    info->pciConfigMirrorSize = 0x1000;
    info->PCIDeviceID = (rd32(s->g, 0x088000) >> 16) << 16 | 0x10de;
    info->PCISubDeviceID = rd32(s->g, 0x08802c);
    info->PCIRevisionID = rd32(s->g, 0x088008) & 0xff;
    info->bIsPrimary = is_boot_vga(s->g);
    cmdq_push(s, NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, info, sizeof(*info));
    mem_free(info);

    static const struct { const char *name; uint32_t value; } keys[] = {
        { "RMSecBusResetEnable", 1 },
        { "RMForcePcieConfigSave", 1 },
        { "RMDevidCheckIgnore", 1 },
    };
    uint8_t reg[512] = { 0 };
    PACKED_REGISTRY_TABLE *t = (void *)reg;
    PACKED_REGISTRY_ENTRY *e = (void *)(reg + sizeof(*t));
    uint32_t str = sizeof(*t) + 3 * sizeof(*e);
    for (int i = 0; i < 3; i++) {
        e[i].nameOffset = str;
        e[i].type = 1;  // DWORD
        e[i].data = keys[i].value;
        e[i].length = 4;
        uint32_t n = strlen(keys[i].name) + 1;
        memcpy(reg + str, keys[i].name, n);
        str += n;
    }
    t->numEntries = 3;
    t->size = str;
    cmdq_push(s, NV_VGPU_MSG_FUNCTION_SET_REGISTRY, reg, str);
}

static void fb_layout(struct gsp *s, uint64_t *frts_addr, uint64_t *frts_size)
{
    GspFwWprMeta *m = s->wpr_meta.cpu;
    uint64_t fb = (uint64_t)rd32(s->g, 0x1183a4) << 20;
    uint64_t vga = fb - 0x100000, reg = rd32(s->g, 0x625f04);
    if (reg & 8) {
        uint64_t addr = (reg & 0xffffff00) << 8;
        vga = addr < vga ? fb - 0x20000 : addr;
    }

    uint64_t frts_sz = 0x100000;
    uint64_t frts = ALIGN_DOWN(vga, 0x20000) - frts_sz;
    uint64_t boot = ALIGN_DOWN(frts - s->boot.size, 0x1000);
    uint64_t elf = ALIGN_DOWN(boot - s->fw_size, 0x10000);

    // nouveau's sum (128 MB on 12 GB); nova-core's 170 MB minimum breaks GSP-RM's graphics init on GA106
    uint64_t gb = (fb + (1ull << 30) - 1) >> 30;
    uint64_t heap_size = (22ull << 20) + (8ull << 20) + ALIGN_UP((96ull << 10) * gb, 1 << 20) +
                         ALIGN_UP((48ull << 10) * 2048, 1 << 20);
    uint64_t heap = ALIGN_DOWN(elf - heap_size, 0x100000);
    heap_size = ALIGN_DOWN(elf - heap, 0x100000);
    uint64_t wpr2 = ALIGN_DOWN(heap - sizeof(GspFwWprMeta), 0x100000);
    uint64_t nonwpr = wpr2 - 0x100000;

    m->magic = GSP_FW_WPR_META_MAGIC;
    m->revision = GSP_FW_WPR_META_REVISION;
    m->sysmemAddrOfRadix3Elf = s->lvl0.addr;
    m->sizeOfRadix3Elf = s->fw_size;
    m->sysmemAddrOfBootloader = s->boot.addr;
    m->sizeOfBootloader = s->boot.size;
    m->sysmemAddrOfSignature = s->sig.addr;
    m->sizeOfSignature = s->sig.size;
    m->gspFwRsvdStart = nonwpr;
    m->nonWprHeapOffset = nonwpr;
    m->nonWprHeapSize = 0x100000;
    m->gspFwWprStart = wpr2;
    m->gspFwHeapOffset = heap;
    m->gspFwHeapSize = heap_size;
    m->gspFwOffset = elf;
    m->bootBinOffset = boot;
    m->frtsOffset = frts;
    m->frtsSize = frts_sz;
    m->gspFwWprEnd = ALIGN_DOWN(vga, 0x20000);
    m->fbSize = fb;
    m->vgaWorkspaceOffset = vga;
    m->vgaWorkspaceSize = fb - vga;
    // as the Linux driver boots Turing to Ada: without it GSP-RM sits in P0 when idle, with it P8 (P0 under load)
    m->flags |= GSP_FW_FLAGS_CLOCK_BOOST;

    log_msg("gsp: fb %llu MB, wpr2 %#llx, heap %#llx (+%llu MB), elf %#llx, frts %#llx\n",
            (unsigned long long)(fb >> 20), (unsigned long long)wpr2, (unsigned long long)heap,
            (unsigned long long)(heap_size >> 20), (unsigned long long)elf, (unsigned long long)frts);
    *frts_addr = frts;
    *frts_size = frts_sz;
}

static int reset_into_riscv(struct gsp *s)
{
    if (falcon_reset_eng(s->g, &gsp_falcon))
        return -1;
    wr32(s->g, gsp_falcon.base2 + 0x668, rd32(s->g, gsp_falcon.base2 + 0x668) | 0x111);
    flcn_wr(s->g, &gsp_falcon, 0x040, (uint32_t)s->libos.addr);
    flcn_wr(s->g, &gsp_falcon, 0x044, s->libos.addr >> 32);
    return 0;
}

// GSP-RM hands control back to SEC2's booter, which reloads it; wait for the handoff
static int core_resume(struct gsp *s)
{
    if (reset_into_riscv(s))
        return -1;
    falcon_start(s->g, &sec2_falcon);
    if (poll32(s->g, 0x1180f8, 1 << 26, 1 << 26, 2000000)) {
        log_msg("seq: GSP reload did not complete\n");
        return -1;
    }
    uint32_t mbox0 = flcn_rd(s->g, &sec2_falcon, 0x040);
    if (mbox0) {
        log_msg("seq: SEC2 reload error %#x\n", mbox0);
        return -1;
    }
    flcn_wr(s->g, &gsp_falcon, 0x080, s->app_version);
    if (!falcon_riscv_active(s->g, &gsp_falcon)) {
        log_msg("seq: RISC-V core not running after reload\n");
        return -1;
    }
    return 0;
}

static int run_sequencer(struct gsp *s, rpc_run_cpu_sequencer_v17_00 *seq)
{
    static const int payload[] = { 2, 3, 5, 1, 2, 0, 0, 0, 0 };
    struct gpu *g = s->g;
    for (uint32_t ptr = 0; ptr < seq->cmdIndex;) {
        GSP_SEQUENCER_BUFFER_CMD *cmd = (void *)&seq->commandBuffer[ptr];
        if (cmd->opCode > GSP_SEQ_BUF_OPCODE_CORE_RESUME) {
            log_msg("seq: unknown opcode %u\n", cmd->opCode);
            return -1;
        }
        ptr += 1 + payload[cmd->opCode];
        switch (cmd->opCode) {
        case GSP_SEQ_BUF_OPCODE_REG_WRITE:
            wr32(g, cmd->payload.regWrite.addr, cmd->payload.regWrite.val);
            break;
        case GSP_SEQ_BUF_OPCODE_REG_MODIFY: {
            uint32_t a = cmd->payload.regModify.addr;
            wr32(g, a, (rd32(g, a) & ~cmd->payload.regModify.mask) | cmd->payload.regModify.val);
            break;
        }
        case GSP_SEQ_BUF_OPCODE_REG_POLL: {
            GSP_SEQ_BUF_PAYLOAD_REG_POLL *p = &cmd->payload.regPoll;
            if (poll32(g, p->addr, p->mask, p->val, p->timeout ? p->timeout : 4000000))
                log_msg("seq: poll %06x & %08x == %08x timed out\n", p->addr, p->mask, p->val);
            break;
        }
        case GSP_SEQ_BUF_OPCODE_DELAY_US:
            delay_us(cmd->payload.delayUs.val);
            break;
        case GSP_SEQ_BUF_OPCODE_REG_STORE:
            seq->regSaveArea[cmd->payload.regStore.index] = rd32(g, cmd->payload.regStore.addr);
            break;
        case GSP_SEQ_BUF_OPCODE_CORE_RESET:
            if (falcon_reset(g, &gsp_falcon))
                return -1;
            falcon_dma_reset(g, &gsp_falcon);
            break;
        case GSP_SEQ_BUF_OPCODE_CORE_START:
            falcon_start(g, &gsp_falcon);
            break;
        case GSP_SEQ_BUF_OPCODE_CORE_WAIT_FOR_HALT:
            if (falcon_wait_halt(g, &gsp_falcon))
                return -1;
            break;
        case GSP_SEQ_BUF_OPCODE_CORE_RESUME:
            if (core_resume(s))
                return -1;
            break;
        }
    }
    log_msg("gsp: ran CPU sequencer (%u dwords)\n", seq->cmdIndex);
    return 0;
}

// RmRCCommonJournal_RECORD then RmRcDiag_RECORD (rmcd.h) as GSP-RM lays them out, pNext a 64-bit pointer: what a
// channel recovery's rcJournalBuffer holds, one after another. data starts at 0x44.
struct rc_diag {
    uint8_t group, type;  // NVCD_RECORD; RmRcDiagReport is type 147
    uint16_t size;        // of the whole record
    uint32_t gpu_tag;
    uint64_t cpu_tag, time, state_mask, next;
    uint16_t idx;
    uint32_t seconds;
    uint16_t diag_type;
    uint32_t flags;
    uint16_t count;
    uint32_t owner, pid;
    struct {
        uint32_t offset, tag, value, attribute;
    } data[];
};

// NV_PFAULT_FAULT_TYPE_*, NV_PFAULT_ACCESS_TYPE_* and NV_PFAULT_CLIENT_HUB_* from open-gpu's
// hwref/ampere/ga100/dev_fault.h, the nearest published to GA10x
static const char *const fault_types[] = {
    "PDE", "PDE size", "PTE", "VA limit", "unbound instance block", "privilege", "read-only", "write-only",
    "pitch mask", "work creation", "unsupported aperture", "compression", "unsupported kind", "region", "poisoned",
    "atomic",
};

static const char *const fault_access[] = {
    "read", "write", "atomic", "prefetch", "weak atomic", NULL, NULL, NULL,
    "physical read", "physical write", "physical atomic", "physical prefetch",
};

static const char *const hub_clients[] = {
    "VIP", "CE0", "CE1", "DNISO", "FE", "FECS", "HOST", "HOST_CPU",
    "HOST_CPU_NB", "ISO", "MMU", "NVDEC0", NULL, "NVENC1", "NISO", "P2P",
    "PD", "PERF", "PMU", "RASTERTWOD", "SCC", "SCC_NB", "SEC", "SSYNC",
    "CE2", "XV", "MMU_NB", "NVENC0", "DFALCON", "SKED", "AFALCON", "DONT_CARE",
    "HSCE0", "HSCE1", "HSCE2", "HSCE3", "HSCE4", "HSCE5", "HSCE6", "HSCE7",
    "HSCE8", "HSCE9", "HSHUB", "PTP_X0", "PTP_X1", "PTP_X2", "PTP_X3", "PTP_X4",
    "PTP_X5", "PTP_X6", "PTP_X7", "NVENC2", "VPR_SCRUBBER0", "VPR_SCRUBBER1", "DWBIF", "FBFALCON",
    "CE_SHIM", "GSP", "NVDEC1", "NVDEC2", "NVJPG0", "NVDEC3", "NVDEC4", "OFA0",
};

static const char *fault_name(const char *const *names, uint32_t n, uint32_t i)
{
    return i < n && names[i] ? names[i] : "unknown";
}

// NV_PFAULT_CLIENT_GPC_*: the unit and, for the per-TPC ones, which (else -1). dev_fault.h names 0x31-0x38
// LTP_UTLB_8-15 as well; they're taken as the per-TPC units here.
static const char *gpc_unit(uint32_t c, int *n)
{
    *n = -1;
    if (c < 0x10) {
        *n = c % 8;
        return c < 8 ? "T1" : "PE";
    }
    if (c >= 0x13 && c <= 0x16) {
        *n = c - 0x13;
        return "PROP";
    }
    if (c >= 0x18 && c <= 0x1f) {
        *n = c - 0x18;
        return "LTP_UTLB";
    }
    if (c >= 0x21 && c <= 0x28) {
        *n = c - 0x21 + 8;
        return "T1";
    }
    if (c >= 0x29 && c <= 0x30) {
        *n = c - 0x29;
        return "TPCCS";
    }
    if (c >= 0x31 && c <= 0x60) {  // from TPC 8 on, eight per pair of TPCs: PE, PE, TPCCS, TPCCS, then four T1
        uint32_t pair = (c - 0x31) / 8, k = (c - 0x31) % 8;
        *n = k < 4 ? 8 + 2 * pair + k % 2 : 16 + 4 * pair + k - 4;
        return k < 2 ? "PE" : k < 4 ? "TPCCS" : "T1";
    }
    if (c >= 0x70 && c <= 0x73) {
        *n = c - 0x70;
        return "ROP";
    }
    switch (c) {
    case 0x10: return "RAST";
    case 0x11: return "GCC";
    case 0x12: return "GPCCS";
    case 0x17: return "GPM";
    case 0x20: return "RGG_UTLB";
    }
    return NULL;
}

static char *put_str(char *p, const char *s)
{
    while (*s)
        *p++ = *s++;
    *p = 0;
    return p;
}

static char *put_num(char *p, uint32_t v)
{
    char d[10];
    int n = 0;
    do
        d[n++] = '0' + v % 10;
    while (v /= 10);
    while (n)
        *p++ = d[--n];
    *p = 0;
    return p;
}

// "GPC 1 PE_2", "hub CE0", "hub client 12"; s holds 24
static void fault_client(char *s, int gpc, uint32_t gpc_id, uint32_t client)
{
    int n = -1;
    const char *unit = gpc ? gpc_unit(client, &n) : client < 0x40 ? hub_clients[client] : NULL;
    if (gpc) {
        s = put_str(s, "GPC ");
        s = put_num(s, gpc_id);
        s = put_str(s, " ");
    } else {
        s = put_str(s, "hub ");
    }
    if (!unit) {
        s = put_str(s, "client ");
        put_num(s, client);
        return;
    }
    s = put_str(s, unit);
    if (n >= 0) {
        s = put_str(s, "_");
        put_num(s, n);
    }
}

// GSP-RM handles a graphics channel's MMU fault itself: it copies to a client's shadow buffer only faults UVM
// services (copy engine or PBDMA faults of channels with replayable faults on, _kgmmuFaultEntryRmServiceable_GV100),
// so registering one gets nothing here. What it does pass on is an RC diagnostic record in the recovery's journal:
// entries at offset 0 tagged 0x22d-0x233 holding FIFO_MMU_EXCEPTION_DATA's addrLo, addrHi, faultType, clientId (with
// bGpc as its attribute), gpcId, accessType and faultEngineId, in that order. The tags aren't in the open sources;
// the order and the values (engine 0x40 graphics, GPC 0-2 on GA106) give them away. The last such record wins;
// 0 when there is none.
static int rc_fault(const uint8_t *journal, uint32_t size, struct gsp_fault *f)
{
    const uint32_t head = offsetof(struct rc_diag, data);
    uint32_t v[7], gpc = 0, seen = 0;
    for (uint32_t off = 0; off + head <= size;) {
        const struct rc_diag *r = (const void *)(journal + off);
        if (r->size < head || r->size > size - off)
            break;
        for (uint32_t i = 0; r->type == 147 && i < r->count && head + (i + 1) * sizeof(r->data[0]) <= r->size; i++) {
            uint32_t k = r->data[i].tag - 0x22d;
            if (r->data[i].offset || k >= 7)
                continue;
            v[k] = r->data[i].value;
            seen |= 1u << k;
            if (k == 3)
                gpc = r->data[i].attribute & 1;
        }
        off += r->size;
    }
    if (seen != 0x7f)
        return 0;
    f->va = (uint64_t)v[1] << 32 | v[0];
    f->type = fault_name(fault_types, sizeof(fault_types) / sizeof(fault_types[0]), v[2]);
    f->access = fault_name(fault_access, sizeof(fault_access) / sizeof(fault_access[0]), v[5]);
    fault_client(f->client, gpc, v[4], v[3]);
    f->engine = v[6];
    return 1;
}

// returns the function of the message handled, -1 on timeout, -2 when the sequencer failed
static int msgq_recv(struct gsp *s, int timeout_ms)
{
    uint32_t cnt = s->cmdq_tx->msgCount;
    uint64_t end = time_us() + (uint64_t)timeout_ms * 1000;
    while (s->msgq_tx->writePtr == s->cmdq_rx->readPtr) {
        if (time_us() > end)
            return -1;
        delay_us(100);
    }
    __sync_synchronize();

    uint32_t rptr = s->cmdq_rx->readPtr;
    struct msg_elem *first = (void *)(s->msgq + PAGE + rptr * PAGE);
    uint32_t size = first->elem_count * PAGE;
    uint8_t *buf = mem_alloc(size);
    for (uint32_t off = 0; off < size; off += PAGE) {
        memcpy(buf + off, s->msgq + PAGE + rptr * PAGE, PAGE);
        if (++rptr == cnt)
            rptr = 0;
    }
    __sync_synchronize();
    s->cmdq_rx->readPtr = rptr;

    struct msg_elem *m = (void *)buf;
    int fn = m->rpc.function;
    void *data = m->rpc.rpc_message_data;
    switch (fn) {
    case NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER:
        if (run_sequencer(s, data))
            fn = -2;
        break;
    case NV_VGPU_MSG_EVENT_OS_ERROR_LOG:
    {
        const uint32_t *h = data;  // rpc_os_error_log_v17_00: exceptType, runlistId, chid, errString
        log_msg("gsp: os error log (type %#x runlist %u chid %u): %.256s\n", h[0], h[1], h[2], (char *)data + 12);
    }
        break;
    case NV_VGPU_MSG_EVENT_GSP_INIT_DONE:
        s->init_result = m->rpc.rpc_result;
        log_msg("gsp: GSP_INIT_DONE (result %#x)\n", m->rpc.rpc_result);
        break;
    case NV_VGPU_MSG_EVENT_RC_TRIGGERED: {
        const rpc_rc_triggered_v17_02 *rc = data;
        uint32_t room = size - (uint32_t)(rc->rcJournalBuffer - buf);
        uint32_t journal = rc->rcJournalBufferSize < room ? rc->rcJournalBufferSize : room;
        log_msg("gsp: channel recovery: engine %u chid %u exception %u level %u scope %u, fault %#llx type %u\n",
                rc->nv2080EngineType, rc->chid, rc->exceptType, rc->exceptLevel, rc->scope,
                (unsigned long long)rc->mmuFaultAddrHi << 32 | rc->mmuFaultAddrLo, rc->mmuFaultType);
        struct gsp_fault fault = { 0 }, *f = NULL;
        int recorded = rc_fault(rc->rcJournalBuffer, journal, &fault);
        if (recorded) {
            f = &fault;
            log_msg("gsp: MMU fault at %#llx: %s fault on a %s by %s, engine %#x\n", (unsigned long long)f->va,
                    f->type, f->access, f->client, f->engine);
        } else if (rc->exceptType == 31) {  // ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT without a record
            f = &fault;
            f->va = (uint64_t)rc->mmuFaultAddrHi << 32 | rc->mmuFaultAddrLo;
            f->type = fault_name(fault_types, sizeof(fault_types) / sizeof(fault_types[0]), rc->mmuFaultType);
            f->access = "unknown";
            put_str(f->client, "unknown");
        }
        // the registers GSP-RM read for a class error, and anything else in a journal we couldn't read
        if (rc->exceptType == 69 || (rc->exceptType == 31 && !recorded)) {
            if (rc->exceptType == 69) {  // the method and data it trapped on, if still latched
                uint32_t addr = rd32(s->g, 0x400704);
                log_msg("gsp: class error code %#x subchannel %u method %#x data %#x %#x\n", rd32(s->g, 0x400110),
                        (addr >> 16) & 7, addr & 0x3ffc, rd32(s->g, 0x400708), rd32(s->g, 0x40070c));
            }
            for (uint32_t i = 0; i < journal && i < 512; i += 32) {
                const uint32_t *w = (const uint32_t *)(rc->rcJournalBuffer + i);
                log_msg("gsp: rc journal %03x: %08x %08x %08x %08x %08x %08x %08x %08x\n", i, w[0], w[1], w[2], w[3],
                        w[4], w[5], w[6], w[7]);
            }
        }
        // the journal is binary; its text records are what's worth reading
        for (uint32_t i = 0, start = 0; i <= journal && i < 0x4000; i++) {
            uint8_t ch = i < journal ? rc->rcJournalBuffer[i] : 0;
            if (ch >= 0x20 && ch < 0x7f)
                continue;
            if (i - start >= 12)
                log_msg("gsp: rc journal: %.*s\n", (int)(i - start), (const char *)rc->rcJournalBuffer + start);
            start = i + 1;
        }
        for (int i = 0; i < 2; i++)
            if (s->rc[i].fn)
                s->rc[i].fn(s->rc[i].ctx, rc->nv2080EngineType, rc->chid, rc->exceptType, f);
        break;
    }
    case NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED: {
        // GSP-RM took a fault off the hub's non-replayable buffer (gmcapiMmuFaultQueued); the channel recovery that
        // follows carries its details (rc_fault). The hub's own latch (nouveau's tu102_fault_info_fault) only holds
        // a fault that went to no buffer, otherwise stale bits with valid clear.
        uint32_t lo = rd32(s->g, 0xb83080), hi = rd32(s->g, 0xb83084), info0 = rd32(s->g, 0xb83088),
                 insthi = rd32(s->g, 0xb8308c), info1 = rd32(s->g, 0xb83090);
        if (!(info1 & 0x80000000))
            break;
        log_msg("gsp: hub latched an MMU fault at %#llx, inst %#llx, engine %#x, %s client %#x access %u reason %u\n",
                (unsigned long long)hi << 32 | lo, (unsigned long long)insthi << 32 | (info0 & 0xfffff000),
                info0 & 0xff, info1 & 0x100000 ? "hub" : "gpc", (info1 >> 8) & 0x7f, (info1 >> 16) & 0xf, info1 & 0x1f);
        break;
    }
    case NV_VGPU_MSG_EVENT_POST_EVENT: {
        const rpc_post_event_v17_00 *e = data;
        log_msg("gsp: event posted: client %#x event %#x index %#x, %u bytes\n", e->hClient, e->hEvent, e->notifyIndex,
                e->eventDataSize);
        if (s->event_fn && sizeof(*e) + e->eventDataSize <= m->rpc.length)
            s->event_fn(s->event_ctx, e->hClient, e->hEvent, e->eventData, e->eventDataSize);
        break;
    }
    case NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD:
    case NV_VGPU_MSG_EVENT_UCODE_LIBOS_PRINT:
        break;
    default:
        if (fn >= 0x1000)
            log_msg("gsp: event %#x (%u bytes)\n", fn, m->rpc.length);
        break;
    }
    mem_free(s->last);
    s->last = m;
    s->last_size = size;
    return fn;
}

// sends an RPC and waits for its reply, servicing events meanwhile; the reply payload is in s->last
static void *rpc_call(struct gsp *s, uint32_t fn, const void *payload, uint32_t len)
{
    cmdq_push(s, fn, payload, len);
    // 10 s in all, not 5 s from the last message: freeing a wedged display went 27 s without a reply
    uint64_t t0 = time_us(), end = t0 + 10000000;
    for (;;) {
        uint64_t now = time_us();
        int got = now < end ? msgq_recv(s, (int)((end - now) / 1000)) : -1;
        if (got < 0) {
            log_msg("gsp: no reply to rpc %u\n", fn);
            return NULL;
        }
        if ((uint32_t)got != fn)
            continue;
        if (time_us() - t0 > 1000000)
            log_msg("gsp: rpc %u took %llu ms\n", fn, (unsigned long long)(time_us() - t0) / 1000);
        if (s->last->rpc.rpc_result) {
            log_msg("gsp: rpc %u failed: %#x\n", fn, s->last->rpc.rpc_result);
            return NULL;
        }
        return s->last->rpc.rpc_message_data;
    }
}

int rm_alloc(struct gsp *s, uint32_t client, uint32_t parent, uint32_t object, uint32_t cls,
             const void *params, uint32_t size)
{
    uint32_t len = sizeof(rpc_gsp_rm_alloc_v03_00) + size;
    rpc_gsp_rm_alloc_v03_00 *rpc = mem_alloc(len);
    rpc->hClient = client;
    rpc->hParent = parent;
    rpc->hObject = object;
    rpc->hClass = cls;
    rpc->paramsSize = size;
    memcpy(rpc->params, params, size);
    rpc_gsp_rm_alloc_v03_00 *rep = rpc_call(s, NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, rpc, len);
    mem_free(rpc);
    if (!rep || rep->status) {
        log_msg("gsp: RM_ALLOC class %#x as %#x: status %#x\n", cls, object, rep ? rep->status : 0);
        return -1;
    }
    log_msg("gsp: allocated class 0x%04x as %#010x\n", cls, object);
    return 0;
}

int rm_free(struct gsp *s, uint32_t client, uint32_t object)
{
    rpc_free_v03_00 rpc = { .params = { .hRoot = client, .hObjectOld = object } };
    return rpc_call(s, NV_VGPU_MSG_FUNCTION_FREE, &rpc, sizeof(rpc)) ? 0 : -1;
}

void *rm_control(struct gsp *s, uint32_t client, uint32_t object, uint32_t cmd, const void *params, uint32_t size)
{
    uint32_t len = sizeof(rpc_gsp_rm_control_v03_00) + size;
    rpc_gsp_rm_control_v03_00 *rpc = mem_alloc(len);
    rpc->hClient = client;
    rpc->hObject = object;
    rpc->cmd = cmd;
    rpc->paramsSize = size;
    memcpy(rpc->params, params, size);
    rpc_gsp_rm_control_v03_00 *rep = rpc_call(s, NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, rpc, len);
    mem_free(rpc);
    s->ctrl_status = rep ? rep->status : ~0u;
    if (!rep || rep->status) {
        log_msg("gsp: RM_CONTROL %#x on %#x: status %#x\n", cmd, object, rep ? rep->status : 0);
        return NULL;
    }
    return rep->params;
}

// our own RM client, device and subdevice, the root every later object hangs off
int rm_client(struct gsp *s, struct rm_client *c)
{
    uint32_t n = ++s->clients;  // from 1: the display engine takes a client ID of 0 as none
    *c = (struct rm_client){ 0xc1e00000 + n, 0xde1e0000 + n, 0x5d1e0000 + n };
    NV0000_ALLOC_PARAMETERS root = { .hClient = c->client, .processID = ~0u };
    NV0080_ALLOC_PARAMETERS dev = { .hClientShare = c->client };
    NV2080_ALLOC_PARAMETERS sub = { 0 };
    return rm_alloc(s, c->client, c->client, c->client, NV01_ROOT, &root, sizeof(root)) ||
           rm_alloc(s, c->client, c->client, c->device, NV01_DEVICE_0, &dev, sizeof(dev)) ||
           rm_alloc(s, c->client, c->device, c->subdevice, NV20_SUBDEVICE_0, &sub, sizeof(sub)) ? -1 : 0;
}

uint32_t gsp_ctrl_status(const struct gsp *s)
{
    return s->ctrl_status;
}

const struct rm_client *gsp_internal(struct gsp *s)
{
    return &s->internal;
}

// the CPU's interrupt tree (nouveau's tu102_vfn_intr): leaves of 32 vectors, two leaves per top bit
#define INTR_LEAF(v) (0xb81000 + ((v) / 32) * 4)
#define INTR_ALLOW(v) (0xb81200 + ((v) / 32) * 4)
#define INTR_TOP_REARM 0xb81608
#define INTR_TOP_UNARM 0xb81610
#define VBIT(v) (1u << ((v) % 32))
#define GSP_IRQSTAT 0x110008
#define GSP_IRQSCLR 0x110004
#define GSP_IRQMASK 0x111528  // riscv_irqmask
#define GSP_RETRIGGER 0x1103e8

static uint32_t top_mask(struct gsp *s)
{
    uint32_t m = 1u << (s->vec_gsp / 64) | (s->vec_disp != ~0u ? 1u << (s->vec_disp / 64) : 0);
    for (uint32_t i = 0; i < s->n_ns; i++)
        m |= 1u << (s->vec_ns[i] / 64);
    return m;
}

// graphics and copy engines: their nonstall interrupts are semaphore releases NVK sleeps on (several engines may
// share a vector)
static int is_work_engine(uint32_t idx)
{
    return idx == MC_ENGINE_IDX_GR0 || (idx >= MC_ENGINE_IDX_CE0 && idx <= MC_ENGINE_IDX_CE13);
}

static void add_nonstall(struct gsp *s, uint32_t vec)
{
    for (uint32_t i = 0; i < s->n_ns; i++)
        if (s->vec_ns[i] == vec)
            return;
    if (s->n_ns < sizeof(s->vec_ns) / sizeof(s->vec_ns[0]))
        s->vec_ns[s->n_ns++] = vec;
}

static int intr_init(struct gsp *s)
{
    s->vec_gsp = s->vec_disp = ~0u;
    s->n_ns = 0;
    uint32_t disp_low = ~0u;
    NV2080_CTRL_INTERNAL_INTR_GET_KERNEL_TABLE_PARAMS *req = mem_alloc(sizeof(*req));
    NV2080_CTRL_INTERNAL_INTR_GET_KERNEL_TABLE_PARAMS *t =
        rm_control(s, s->internal.client, s->internal.subdevice, NV2080_CTRL_CMD_INTERNAL_INTR_GET_KERNEL_TABLE, req,
                   sizeof(*req));
    mem_free(req);
    for (uint32_t i = 0; t && i < t->tableLen; i++) {
        if (t->table[i].engineIdx == MC_ENGINE_IDX_GSP)
            s->vec_gsp = t->table[i].vectorStall;
        if (is_work_engine(t->table[i].engineIdx) && t->table[i].vectorNonStall != ~0u)
            add_nonstall(s, t->table[i].vectorNonStall);
        if (t->table[i].engineIdx == MC_ENGINE_IDX_DISP)
            s->vec_disp = t->table[i].vectorStall;
        if (t->table[i].engineIdx == MC_ENGINE_IDX_DISP_LOW)
            disp_low = t->table[i].vectorStall;
    }
    if (disp_low != ~0u)
        s->vec_disp = disp_low;  // chips with a low-latency line raise head timing there (nouveau's r535 disp)
    if (s->vec_gsp == ~0u) {
        log_msg("gsp: no interrupt vector for GSP-RM\n");
        return -1;
    }
    log_msg("gsp: interrupt vectors: GSP-RM %u, display %d, %u nonstall (graphics, copy engines) from %d\n",
            s->vec_gsp, (int)s->vec_disp, s->n_ns, s->n_ns ? (int)s->vec_ns[0] : -1);
    return 0;
}

void gsp_irq_enable(struct gsp *s, int on)
{
    if (s->vec_gsp == ~0u)
        return;
    wr32(s->g, INTR_TOP_UNARM, top_mask(s));
    if (!on)
        return;
    wr32(s->g, INTR_ALLOW(s->vec_gsp), VBIT(s->vec_gsp));
    for (uint32_t i = 0; i < s->n_ns; i++)
        wr32(s->g, INTR_ALLOW(s->vec_ns[i]), VBIT(s->vec_ns[i]));
    if (s->vec_disp != ~0u)
        wr32(s->g, INTR_ALLOW(s->vec_disp), VBIT(s->vec_disp));
    wr32(s->g, INTR_TOP_REARM, top_mask(s));
}

int gsp_irq(struct gsp *s)
{
    int events = 0;
    if (s->vec_gsp == ~0u)
        return 0;
    wr32(s->g, INTR_TOP_UNARM, top_mask(s));
    if (rd32(s->g, INTR_LEAF(s->vec_gsp)) & VBIT(s->vec_gsp)) {
        wr32(s->g, INTR_LEAF(s->vec_gsp), VBIT(s->vec_gsp));
        uint32_t stat = rd32(s->g, GSP_IRQSTAT) & rd32(s->g, GSP_IRQMASK);
        if (stat & 0x40) {  // the message queue has something
            wr32(s->g, GSP_IRQSCLR, 0x40);
            gsp_poll(s);
            events |= GSP_IRQ_MSG;
        }
        if (stat & ~0x40u)
            wr32(s->g, GSP_IRQSCLR, stat & ~0x40u);
        wr32(s->g, GSP_RETRIGGER, 1);
    }
    for (uint32_t i = 0; i < s->n_ns; i++) {
        uint32_t v = s->vec_ns[i];
        if (rd32(s->g, INTR_LEAF(v)) & VBIT(v)) {
            wr32(s->g, INTR_LEAF(v), VBIT(v));
            events |= GSP_IRQ_GR;
        }
    }
    if (s->vec_disp != ~0u && rd32(s->g, INTR_LEAF(s->vec_disp)) & VBIT(s->vec_disp)) {
        wr32(s->g, INTR_LEAF(s->vec_disp), VBIT(s->vec_disp));
        // head timing (tu102_disp_intr): which heads, then each one's vblank, acked where it was raised
        uint32_t heads = rd32(s->g, 0x611ec0) & 0xff;
        for (int h = 0; h < 8; h++) {
            if ((heads & (1u << h)) && (rd32(s->g, 0x611c00 + h * 4) & 2)) {
                wr32(s->g, 0x611800 + h * 4, 2);
                events |= GSP_IRQ_VBLANK(h);
            }
        }
    }
    if (events && s->irqs++ < 3)
        log_msg("gsp: interrupt (%s%s)\n", events & GSP_IRQ_MSG ? "messages " : "", events & GSP_IRQ_GR ? "graphics" : "");
    wr32(s->g, INTR_TOP_REARM, top_mask(s));
    return events;
}

int gsp_pending(const struct gsp *s)
{
    return s->msgq_tx->writePtr != s->cmdq_rx->readPtr;
}

void gsp_poll(struct gsp *s)
{
    while (s->msgq_tx->writePtr != s->cmdq_rx->readPtr)
        msgq_recv(s, 0);
}

int gsp_chid_get(struct gsp *s, uint32_t first)
{
    for (uint32_t i = first; i < GSP_CHIDS; i++)
        if (!(s->chids[i / 32] & (1u << (i % 32)))) {
            s->chids[i / 32] |= 1u << (i % 32);
            return (int)i;
        }
    return -1;
}

void gsp_chid_put(struct gsp *s, uint32_t chid)
{
    s->chids[chid / 32] &= ~(1u << (chid % 32));
}

int gsp_on_rc(struct gsp *s, gsp_rc_fn fn, void *ctx)
{
    for (int i = 0; i < 2; i++)
        if (!s->rc[i].fn) {
            s->rc[i].fn = fn;
            s->rc[i].ctx = ctx;
            return 0;
        }
    return -1;
}

void gsp_off_rc(struct gsp *s, void *ctx)
{
    for (int i = 0; i < 2; i++)
        if (s->rc[i].ctx == ctx)
            s->rc[i].fn = NULL;
}

void gsp_on_event(struct gsp *s, gsp_event_fn fn, void *ctx)
{
    s->event_fn = fn;
    s->event_ctx = ctx;
}

struct vram *gsp_vram(struct gsp *s)
{
    return &s->vram;
}

struct gr_global **gsp_gr(struct gsp *s)
{
    return &s->gr;
}

struct gpu *gsp_gpu(struct gsp *s)
{
    return s->g;
}

struct dma *gsp_dma(struct gsp *s, size_t size, int contiguous)
{
    struct owned_dma *o = mem_alloc(sizeof(*o));
    if (dma(s, &o->d, size, contiguous)) {
        mem_free(o);
        return NULL;
    }
    o->next = s->owned;
    s->owned = o;
    return &o->d;
}

void gsp_dma_free(struct gsp *s, struct dma *d)
{
    for (struct owned_dma **p = &s->owned; *p; p = &(*p)->next) {
        struct owned_dma *o = *p;
        if (&o->d == d) {
            *p = o->next;
            dma_free(s->g, d);
            mem_free(o);
            return;
        }
    }
}

// the graphics info table GSP-RM keeps for the CPU-side RM, which answers GR_GET_INFO from it
static int gr_info(struct gsp *s)
{
    NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS *req = mem_alloc(sizeof(*req));
    NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS *p = rm_control(
        s, s->internal.client, s->internal.subdevice, NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO, req, sizeof(*req));
    mem_free(req);
    if (!p)
        return -1;
    for (int i = 0; i < NV2080_CTRL_INTERNAL_GR_INFO_MAX_SIZE; i++) {
        NV2080_CTRL_INTERNAL_GR_INFO *e = &p->engineInfo[0].infoList[i];
        if (e->index < GSP_GR_INFO_MAX)
            s->info.gr[e->index] = e->data;
    }
    NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS zcull = {};
    const NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS *z =
        rm_control(s, s->internal.client, s->internal.subdevice, NV2080_CTRL_CMD_GR_GET_ZCULL_INFO, &zcull, sizeof(zcull));
    if (z && z->subregionCount <= GSP_ZCULL_SUBREGIONS)
        memcpy(&s->info.zcull, z, sizeof(s->info.zcull));
    return 0;
}

const struct gsp_info *gsp_get_info(struct gsp *s)
{
    return &s->info;
}

static int gsp_query(struct gsp *s)
{
    GspStaticConfigInfo *req = mem_alloc(sizeof(*req));
    GspStaticConfigInfo *info = rpc_call(s, NV_VGPU_MSG_FUNCTION_GET_GSP_STATIC_INFO, req, sizeof(*req));
    mem_free(req);
    if (!info)
        return -1;
    log_msg("gsp: %s, VBIOS valid %d\n", (char *)info->gpuNameString, info->bVbiosValid);
    s->internal = (struct rm_client){ info->hInternalClient, info->hInternalDevice, info->hInternalSubdevice };
    memcpy(s->info.name, info->gpuNameString, sizeof(s->info.name) - 1);
    memcpy(s->info.short_name, info->gpuShortNameString, sizeof(s->info.short_name) - 1);
    s->info.fb_size = (uint64_t)rd32(s->g, 0x1183a4) << 20;
    s->info.bar1_pdb = info->bar1PdeBase;
    NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS *fb = &info->fbRegionInfoParams;
    uint64_t usable_end = 0;
    for (uint32_t i = 0; i < fb->numFBRegions; i++) {
        NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO *r = &fb->fbRegion[i];
        log_msg("gsp: fb region %u: %#llx..%#llx%s\n", i, (unsigned long long)r->base, (unsigned long long)r->limit,
                r->supportCompressed ? " compressible" : "");
        if (!r->reserved && !r->bProtected)
            usable_end = r->limit + 1;
    }
    const GspFwWprMeta *m = s->wpr_meta.cpu;
    s->fb_rsvd = usable_end && usable_end < m->nonWprHeapOffset ? m->nonWprHeapOffset - usable_end : 0;
    return gr_info(s);
}

// FWSEC-SB restores the pre-OS state on the GSP falcon, Booter-Unload releases WPR2, saving it first to the
// suspend copy arg points at (0xff/0xff: none)
static int teardown(struct gpu *g, uint64_t arg)
{
    int ret = 0;
    if (falcon_reset(g, &gsp_falcon) || fwsec_run(g, FWSEC_SB, 0, 0))
        ret = -1;
    if (!rd32(g, 0x1fa828))
        return ret;
    uint32_t mbox0;
    if (booter_run(g, "booter_unload.bin", arg, &mbox0) || mbox0) {
        log_msg("gsp: booter-unload failed\n");
        return -1;
    }
    if (rd32(g, 0x1fa828)) {
        log_msg("gsp: WPR2 still set after booter-unload\n");
        return -1;
    }
    log_msg("gsp: WPR2 released\n");
    return ret;
}

int gsp_cleanup(struct gpu *g)
{
    return teardown(g, 0x000000ff000000ffull);
}

static void sr_free(struct gsp *s)
{
    struct dma *sr[] = { &s->sr.data, &s->sr.lvl0, &s->sr.lvl1, &s->sr.lvl2, &s->sr.meta, &s->sr.fbsr };
    for (unsigned i = 0; i < sizeof(sr) / sizeof(sr[0]); i++)
        if (sr[i]->cpu)
            dma_free(s->g, sr[i]);
    memset(&s->sr, 0, sizeof(s->sr));
}

static void free_all(struct gsp *s)
{
    struct dma *mine[] = { &s->flush, &s->fw, &s->lvl0, &s->lvl1, &s->lvl2, &s->sig, &s->boot, &s->libos,
                           &s->loginit, &s->logintr, &s->logrm, &s->rmargs, &s->shm, &s->wpr_meta };
    for (unsigned i = 0; i < sizeof(mine) / sizeof(mine[0]); i++)
        if (mine[i]->cpu)
            dma_free(s->g, mine[i]);
    sr_free(s);
    while (s->owned)
        gsp_dma_free(s, &s->owned->d);
    vram_destroy(&s->vram);
    mem_free(s->gr);
    mem_free(s->last);
    mem_free(s);
}

// UNLOADING_GUEST_DRIVER, then GSP-RM's core halting with LIBOS_INTERRUPT_PROCESSOR_SUSPENDED in mailbox0
static void unload_rpc(struct gsp *s, int pm)
{
    gsp_irq_enable(s, 0);
    rpc_unloading_guest_driver_v1F_07 rpc = { 0 };
    if (pm) {
        rpc.bInPMTransition = 1;
        rpc.newLevel = NV2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_3;
    }
    uint64_t t0 = time_us();
    cmdq_push(s, NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, &rpc, sizeof(rpc));
    for (;;) {
        int fn = msgq_recv(s, 5000);
        if (fn < 0) {
            log_msg("gsp: no reply to UNLOADING_GUEST_DRIVER\n");
            break;
        }
        if (fn == NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER)
            break;
    }
    if (poll32(s->g, gsp_falcon.base + 0x040, 0xffffffff, 0x80000000, 2000000))
        log_msg("gsp: did not suspend (mailbox0 %#x)\n", flcn_rd(s->g, &gsp_falcon, 0x040));
    else
        log_msg("gsp: GSP-RM %s in %llu ms\n", pm ? "suspended" : "unloaded",
                (unsigned long long)(time_us() - t0) / 1000);
}

int gsp_unload(struct gsp *s)
{
    // a boot that failed has torn GSP-RM down already: nobody would answer
    if (falcon_riscv_active(s->g, &gsp_falcon))
        unload_rpc(s, 0);
    int ret = gsp_cleanup(s->g);
    gr_forget(s);
    return ret;
}

void gsp_stop(struct gsp *s)
{
    gsp_unload(s);
    free_all(s);
}

// after a power-up the GPU's firmware runs devinit itself (GFW boot), and nothing else may touch it until it's done
// (nouveau's tu102_devinit_wait); a GPU that kept power reads done already
static void gfw_wait(struct gpu *g)
{
    uint64_t end = time_us() + 2000000;
    while (!(rd32(g, 0x118128) & 1) || (rd32(g, 0x118234) & 0xff) != 0xff) {
        if (time_us() > end) {
            log_msg("gsp: GFW boot not done (%#x, progress %#x), booting anyway\n", rd32(g, 0x118128),
                    rd32(g, 0x118234) & 0xff);
            return;
        }
        delay_us(1000);
    }
}

// the system memory a boot needs, kept until gsp_stop(): flush page, firmware, libos arguments and queues
static struct gsp *gsp_alloc(struct gpu *g)
{
    struct gsp *s = mem_alloc(sizeof(*s));
    s->g = g;
    vram_init(&s->vram, (uint64_t)rd32(g, 0x1183a4) << 20);
    if (dma(s, &s->flush, PAGE, 1) || load_firmware(s) || init_libos(s)) {
        free_all(s);
        return NULL;
    }
    return s;
}

// Booter-Load with arg (the WPR metadata, or the suspend copy's), then GSP-RM's core started and its initialisation
// waited for
static int run_rm(struct gsp *s, uint64_t arg)
{
    struct gpu *g = s->g;
    uint32_t mbox0;
    if (booter_run(g, "booter_load.bin", arg, &mbox0) || mbox0) {
        log_msg("gsp: booter-load failed\n");
        return -1;
    }
    flcn_wr(g, &gsp_falcon, 0x080, s->app_version);
    if (!falcon_riscv_active(g, &gsp_falcon)) {
        log_msg("gsp: RISC-V core is not running (mailbox0 %#x)\n", flcn_rd(g, &gsp_falcon, 0x040));
        return -1;
    }
    for (;;) {
        int fn = msgq_recv(s, 10000);
        if (fn == -1)
            log_msg("gsp: no message for 10 s (mailbox0 %#x)\n", flcn_rd(g, &gsp_falcon, 0x040));
        if (fn < 0)
            return -1;
        if (fn == NV_VGPU_MSG_EVENT_GSP_INIT_DONE)
            return s->init_result ? -1 : 0;
    }
}

int gsp_boot(struct gsp *s)
{
    struct gpu *g = s->g;
    gfw_wait(g);
    // GSP-RM left loaded, by a wake that failed or a restart that didn't reset the GPU: FWSEC-FRTS won't run under it
    if (rd32(g, 0x1fa828)) {
        log_msg("gsp: WPR2 still up (GSP mailbox0 %#x), unloading what holds it\n", flcn_rd(g, &gsp_falcon, 0x040));
        if (gsp_cleanup(g))
            return -1;
    }

    // page the GPU writes to when flushing to system memory
    wr32(g, 0x100c40, s->flush.addr >> 40);
    wr32(g, 0x100c10, s->flush.addr >> 8);
    flcn_wr(g, &gsp_falcon, 0x004, 1 << 6);  // clear SWGEN0

    queue_reset(s);
    queue_init_rpcs(s);

    uint64_t frts_addr, frts_size;
    fb_layout(s, &frts_addr, &frts_size);
    if (fwsec_run(g, FWSEC_FRTS, frts_addr, frts_size) || reset_into_riscv(s))
        return -1;

    if (run_rm(s, s->wpr_meta.addr) || gsp_query(s)) {
        gsp_unload(s);
        return -1;
    }
    intr_init(s);  // without it the platform keeps polling
    return 0;
}

static int schedule_channels(struct gsp *s, int on)
{
    NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING_PARAMS p = { .bDisableActiveChannels = !on };
    return rm_control(s, s->internal.client, s->internal.subdevice,
                      NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING, &p, sizeof(p)) ? 0 : -1;
}

#define FBSR_MEMLIST 0xcaf00003

// GSP-RM's VRAM outside WPR2 (its non-WPR heap, what it reserves below that, the VGA workspace, the compression
// backing store) is GSP-RM's own to save: it gets system memory for it as a memory list, as nouveau's r570 FBSR does
static int fbsr_init(struct gsp *s)
{
    const GspFwWprMeta *m = s->wpr_meta.cpu;
    NV0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS req = { 0 };
    const NV0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS *cbc = rm_control(
        s, s->internal.client, s->internal.device, NV0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO, &req, sizeof(req));
    if (!cbc)
        return -1;
    uint64_t cbc_size = cbc->Size;
    uint64_t size = ALIGN_UP(m->nonWprHeapSize + s->fb_rsvd + m->vgaWorkspaceSize + cbc_size, PAGE);
    uint64_t pages = size / PAGE;
    log_msg("gsp: FBSR %llu MB (reserved %llu MB, compression backing store %llu MB)\n",
            (unsigned long long)(size >> 20), (unsigned long long)(s->fb_rsvd >> 20),
            (unsigned long long)(cbc_size >> 20));
    if (pages > 0xffff) {  // pteDesc.length is 16 bits
        log_msg("gsp: FBSR needs more than a memory list describes\n");
        return -1;
    }
    // the list is a client's of ours (ogkm's memory manager client, nouveau's internal one), not GSP-RM's
    struct rm_client c;
    if (dma(s, &s->sr.fbsr, size, 0) || rm_client(s, &c))
        return -1;

    uint32_t len = sizeof(rpc_alloc_memory_v13_01) + pages * sizeof(uint64_t);
    rpc_alloc_memory_v13_01 *rpc = mem_alloc(len);
    rpc->hClient = c.client;
    rpc->hDevice = c.device;
    rpc->hMemory = FBSR_MEMLIST;
    rpc->hClass = NV01_MEMORY_LIST_SYSTEM;
    rpc->flags = NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS | NVOS02_FLAGS_LOCATION_PCI | NVOS02_FLAGS_MAPPING_NO_MAP;
    rpc->length = size;
    rpc->pageCount = pages;
    rpc->pteDesc.length = pages;
    for (uint64_t i = 0; i < pages; i++)
        rpc->pteDesc.pte_pde[i].pte = dma_addr(&s->sr.fbsr, i * PAGE) >> 12;
    void *done = rpc_call(s, NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY, rpc, len);
    mem_free(rpc);
    if (done) {
        NV2080_CTRL_INTERNAL_FBSR_INIT_PARAMS p = {
            .hClient = c.client,
            .hSysMem = FBSR_MEMLIST,
            // ogkm's for S3; nouveau's 1 announces GC-OFF, which ogkm only enters with the display unused
            .bEnteringGcoffState = 0,
            .sysmemAddrOfSuspendResumeData = s->sr.meta.addr,
        };
        done = rm_control(s, s->internal.client, s->internal.subdevice, NV2080_CTRL_CMD_INTERNAL_FBSR_INIT, &p,
                          sizeof(p));
    }
    rm_free(s, c.client, c.client);  // GSP-RM keeps the pages it was told about, not the list
    return done ? 0 : -1;
}

int gsp_suspend(struct gsp *s)
{
    const GspFwWprMeta *m = s->wpr_meta.cpu;
    uint64_t size = m->frtsOffset + m->frtsSize - (m->nonWprHeapOffset + m->nonWprHeapSize);
    if (dma(s, &s->sr.data, size, 0) || radix3(s, &s->sr.data, size, &s->sr.lvl0, &s->sr.lvl1, &s->sr.lvl2) ||
        dma(s, &s->sr.meta, sizeof(GspFwSRMeta), 1)) {
        sr_free(s);
        return -1;
    }
    GspFwSRMeta *sr = s->sr.meta.cpu;
    sr->magic = GSP_FW_SR_META_MAGIC;
    sr->revision = GSP_FW_SR_META_REVISION;
    sr->sysmemAddrOfSuspendResumeData = s->sr.lvl0.addr;
    sr->sizeOfSuspendResumeData = size;
    sr->flags = m->flags;
    log_msg("gsp: suspending, WPR2 copy %llu MB\n", (unsigned long long)(size >> 20));

    if (schedule_channels(s, 0)) {
        sr_free(s);
        return -1;
    }
    if (fbsr_init(s)) {
        schedule_channels(s, 1);
        sr_free(s);
        return -1;
    }
    delay_us(200000);  // nouveau's: without it Turing resets every channel after resume
    unload_rpc(s, 1);
    teardown(s->g, s->sr.meta.addr);
    return 0;
}

int gsp_resume(struct gsp *s)
{
    if (!s->sr.meta.cpu)
        return gsp_boot(s) ? -1 : 1;
    struct gpu *g = s->g;
    gfw_wait(g);
    wr32(g, 0x100c40, s->flush.addr >> 40);
    wr32(g, 0x100c10, s->flush.addr >> 8);

    // FRTS, the core's boot arguments and the queues come back from the copy (nouveau's tu102_gsp_init)
    GSP_ARGUMENTS_CACHED *rm = s->rmargs.cpu;
    rm->srInitArguments = (GSP_SR_INIT_ARGUMENTS){
        .oldLevel = NV2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_3,
        .flags = GPU_STATE_FLAGS_PRESERVING | GPU_STATE_FLAGS_PM_TRANSITION,
        .bInPMTransition = 1,
    };
    int ret = run_rm(s, s->sr.meta.addr);
    rm->srInitArguments = (GSP_SR_INIT_ARGUMENTS){ 0 };
    sr_free(s);
    if (!ret && !schedule_channels(s, 1)) {
        log_msg("gsp: GSP-RM resumed with its state\n");
        if (gr_resume(s))
            log_msg("gsp: no golden context after the resume, graphics channels may fault\n");
        return 0;
    }
    log_msg("gsp: GSP-RM did not resume, booting it afresh\n");
    gsp_unload(s);
    return gsp_boot(s) ? -1 : 1;
}

struct gsp *gsp_start(struct gpu *g)
{
    struct gsp *s = gsp_alloc(g);
    if (s && gsp_boot(s)) {
        free_all(s);
        return NULL;
    }
    return s;
}
