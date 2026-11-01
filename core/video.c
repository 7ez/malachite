// NVDEC through GSP-RM, following nouveau's GSP-mode falcon engines (rm/r535/fifo.c r535_flcn_ctor, rm/r570/fifo.c
// r570_fifo_ectx_size, rm/r535/nvdec.c); methods from NVIDIA's clc7b0.h, which C9B0 keeps
#include "chip.h"
#include "../nvrm/engine.h"
#include "../nvrm/fifo.h"
#include "../nvrm/gr.h"
#include "chan.h"
#include "gsp.h"
#include "mem.h"
#include "nvdec.h"
#include "video.h"

#define PAGE 0x1000ull
#define SUBCH 4

#define NOP 0x100
#define SET_APPLICATION_ID 0x200
#define APPLICATION_ID_H264 3
#define SEMAPHORE_A 0x240
#define EXECUTE 0x300
#define SEMAPHORE_D 0x304
#define SET_CONTROL_PARAMS 0x400
#define CONTROL_H264 3
#define CONTROL_GPTIMER_ON (1u << 4)
#define CONTROL_ERR_CONCEAL_ON (1u << 6)
#define CONTROL_ERROR_FRM_IDX(i) ((uint32_t)(i) << 7)
#define CONTROL_MBTIMER_ON (1u << 13)
#define SET_DRV_PIC_SETUP_OFFSET 0x404
#define SET_IN_BUF_BASE_OFFSET 0x408
#define SET_PICTURE_INDEX 0x40c
#define SET_SLICE_OFFSETS_BUF_OFFSET 0x410
#define SET_COLOC_DATA_OFFSET 0x414
#define SET_HISTORY_OFFSET 0x418
#define SET_NVDEC_STATUS_OFFSET 0x424
#define SET_PICTURE_LUMA_OFFSET0 0x430
#define SET_PICTURE_CHROMA_OFFSET0 0x474
#define H264_SET_MBHIST_BUF_OFFSET 0x500

// the first NVDEC in the device info table: its RM engine type, runlist and engine descriptor
static int find_nvdec(struct gsp *s, const struct rm_client *c, uint32_t *engine, uint32_t *runlist, uint32_t *desc)
{
    NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS *req = mem_alloc(sizeof(*req));
    NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS *t =
        rm_control(s, c->client, c->subdevice, NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE, req, sizeof(*req));
    mem_free(req);
    for (uint32_t i = 0; t && i < t->numEntries; i++) {
        const NvU32 *d = t->entries[i].engineData;
        if (d[ENGINE_INFO_TYPE_RM_ENGINE_TYPE] - RM_ENGINE_TYPE_NVDEC0 < 8) {
            *engine = d[ENGINE_INFO_TYPE_RM_ENGINE_TYPE];
            *runlist = d[ENGINE_INFO_TYPE_RUNLIST];
            *desc = d[ENGINE_INFO_TYPE_ENG_DESC];
            return 0;
        }
    }
    log_msg("video: no NVDEC engine\n");
    return -1;
}

// the engine context GSP-RM wants for the falcon `desc` names, 0 for none; only the internal client gets an answer
static uint32_t ctx_size(struct gsp *s, uint32_t desc)
{
    const struct rm_client *in = gsp_internal(s);
    NV2080_CTRL_GPU_GET_CONSTRUCTED_FALCON_INFO_PARAMS *req = mem_alloc(sizeof(*req));
    NV2080_CTRL_GPU_GET_CONSTRUCTED_FALCON_INFO_PARAMS *info = rm_control(
        s, in->client, in->subdevice, NV2080_CTRL_CMD_GPU_GET_CONSTRUCTED_FALCON_INFO, req, sizeof(*req));
    mem_free(req);
    for (uint32_t i = 0; info && i < info->numConstructedFalcons; i++)
        if (info->constructedFalconsTable[i].engDesc == desc)
            return info->constructedFalconsTable[i].ctxBufferSize;
    return 0;
}

// the engine context in VRAM, mapped with small pages, and promoted: a falcon's is one buffer, given by VA alone
static int promote(struct gsp *s, const struct rm_client *c, uint32_t desc, struct video *v)
{
    uint32_t size = ctx_size(s, desc);
    if (!size) {
        log_msg("video: GSP-RM has no context size for NVDEC%u (engine %#x)\n", v->inst, desc);
        return -1;
    }
    v->ctx_size = (size + PAGE - 1) & ~(PAGE - 1);
    v->ctx_vram = vram_alloc(gsp_vram(s), v->ctx_size, PAGE);
    if (!v->ctx_vram)
        return -1;
    vram_zero(gsp_gpu(s), v->ctx_vram, v->ctx_size);
    v->ctx_va = vmm_alloc(v->vm, v->ctx_size, PAGE);
    if (!v->ctx_va || vmm_map_vram(v->vm, v->ctx_va, v->ctx_vram, v->ctx_size, 0))
        return -1;
    vmm_flush(gsp_gpu(s));

    NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS *p = mem_alloc(sizeof(*p));
    p->engineType = NV2080_ENGINE_TYPE_NVDEC0 + v->inst;
    p->hClient = c->client;
    p->hChanClient = c->client;
    p->hObject = v->ch.handle;
    p->ChID = v->ch.chid;
    p->virtAddress = v->ctx_va;
    p->size = size;
    void *ok = rm_control(s, c->client, c->subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, p, sizeof(*p));
    mem_free(p);
    return ok ? 0 : -1;
}

int video_create(struct gsp *s, const struct rm_client *c, struct vmm *vm, uint32_t chid, int privileged,
                 struct video *v)
{
    const struct chip *k = gsp_gpu(s)->chip;
    uint32_t engine, runlist, desc;
    *v = (struct video){ .vm = vm };
    if (find_nvdec(s, c, &engine, &runlist, &desc))
        return -1;
    v->inst = engine - RM_ENGINE_TYPE_NVDEC0;

    // RM rejects the parameters unless size says what they are (kernel_nvdec_engdesc.c)
    struct { uint32_t size, prohibitMultipleInstances, engineInstance; } dec = { sizeof(dec), 0, v->inst };
    if (chan_create(s, c, vm, engine, runlist, chid, 0xc56f0000 | chid, privileged, &v->ch) ||
        chan_schedule(&v->ch) || promote(s, c, desc, v) ||
        rm_alloc(s, c->client, v->ch.handle, 0xdec00000, k->nvdec, &dec, sizeof(dec)))
        goto fail;

    v->sem = gsp_dma(s, PAGE, 1);
    v->semva = v->sem ? vmm_alloc(vm, PAGE, PAGE) : 0;
    if (!v->semva || vmm_map_sys(vm, v->semva, v->sem, 0, PAGE, 0))
        goto fail;
    vmm_flush(gsp_gpu(s));

    uint32_t bind[] = { MTHD(SUBCH, 0x000, 1), k->nvdec };
    chan_push(&v->ch, bind, sizeof(bind) / 4);  // goes to the engine with the first submit
    log_msg("video: NVDEC%u on runlist %u, class %#x, %llu KB engine context\n", v->inst, runlist, k->nvdec,
            (unsigned long long)(v->ctx_size >> 10));
    return 0;
fail:
    rm_free(s, c->client, v->ch.handle);
    video_free(v);
    return -1;
}

void video_release_queue(struct video *v, uint64_t semva, uint32_t payload)
{
    uint32_t push[] = {
        MTHD(SUBCH, SEMAPHORE_A, 3), (uint32_t)(semva >> 32) & 0xff, (uint32_t)semva, payload,
        MTHD(SUBCH, SEMAPHORE_D, 1), 0,  // release, one word, flushed first
    };
    chan_submit(&v->ch, push, sizeof(push) / 4);
}

static int vbuf_alloc(struct video *v, struct vbuf *b, uint64_t size)
{
    b->size = (size + PAGE - 1) & ~(PAGE - 1);
    b->mem = gsp_dma(v->ch.s, b->size, 0);
    b->va = b->mem ? vmm_alloc(v->vm, b->size, PAGE) : 0;
    if (!b->va || vmm_map_sys(v->vm, b->va, b->mem, 0, b->size, 0))
        return -1;
    vmm_flush(gsp_gpu(v->ch.s));
    return 0;
}

static void vbuf_free(struct video *v, struct vbuf *b)
{
    if (b->va)
        vmm_unmap(v->vm, b->va, b->size);
    if (b->mem)
        gsp_dma_free(v->ch.s, b->mem);
    *b = (struct vbuf){ 0 };
}

int video_h264_open(struct video *v, struct video_h264 *h, uint32_t width_mbs, uint32_t height_mbs, uint32_t slots,
                    uint64_t bits)
{
    struct nvdec_h264_sizes sizes;
    *h = (struct video_h264){ .width_mbs = width_mbs, .height_mbs = height_mbs, .slots = slots };
    if (!width_mbs || width_mbs > 512 || !height_mbs || height_mbs > 512 || !slots || slots > H264_SLOTS ||
        !bits || bits > 64u << 20)
        return -1;
    nvdec_h264_sizes(width_mbs, height_mbs, &sizes);
    nvdec_h264_surface(width_mbs, height_mbs, &h->layout);
    if (vbuf_alloc(v, &h->setup, NVDEC_SETUP_SIZE) || vbuf_alloc(v, &h->bits, bits) ||
        vbuf_alloc(v, &h->coloc, (uint64_t)sizes.coloc * slots) || vbuf_alloc(v, &h->mbhist, sizes.mbhist) ||
        vbuf_alloc(v, &h->history, sizes.history))
        goto fail;
    for (uint32_t i = 0; i < slots; i++)
        if (vbuf_alloc(v, &h->surf[i], h->layout.size))
            goto fail;
    return 0;
fail:
    video_h264_close(v, h);
    return -1;
}

void video_h264_close(struct video *v, struct video_h264 *h)
{
    vbuf_free(v, &h->setup);
    vbuf_free(v, &h->bits);
    vbuf_free(v, &h->coloc);
    vbuf_free(v, &h->mbhist);
    vbuf_free(v, &h->history);
    for (int i = 0; i < H264_SLOTS; i++)
        vbuf_free(v, &h->surf[i]);
    vmm_flush(gsp_gpu(v->ch.s));
    *h = (struct video_h264){ 0 };
}

int video_h264_queue(struct video *v, struct video_h264 *h, uint32_t slot, uint32_t index, uint32_t error_idx,
                     uint32_t *payload)
{
    const struct nvdec_h264_pic *pic = h->setup.mem->cpu;
    if (slot >= h->slots || pic->curr_pic_idx != slot)
        return -1;
    memset((uint8_t *)h->setup.mem->cpu + NVDEC_SETUP_STATUS, 0, 64);

    uint32_t push[64], n = 0;
#define M(mthd, val) (push[n++] = MTHD(SUBCH, mthd, 1), push[n++] = (val))
    M(SET_APPLICATION_ID, APPLICATION_ID_H264);
    M(SET_CONTROL_PARAMS, CONTROL_H264 | CONTROL_GPTIMER_ON | CONTROL_ERR_CONCEAL_ON |
                              CONTROL_ERROR_FRM_IDX(error_idx & 0x3f) | CONTROL_MBTIMER_ON);
    M(SET_DRV_PIC_SETUP_OFFSET, (uint32_t)(h->setup.va >> 8));
    M(SET_IN_BUF_BASE_OFFSET, (uint32_t)(h->bits.va >> 8));
    M(SET_PICTURE_INDEX, index);
    M(SET_SLICE_OFFSETS_BUF_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_SLICES) >> 8));
    M(SET_COLOC_DATA_OFFSET, (uint32_t)(h->coloc.va >> 8));
    M(SET_HISTORY_OFFSET, (uint32_t)(h->history.va >> 8));
    M(SET_NVDEC_STATUS_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_STATUS) >> 8));
    M(H264_SET_MBHIST_BUF_OFFSET, (uint32_t)(h->mbhist.va >> 8));
#undef M
    push[n++] = MTHD(SUBCH, SET_PICTURE_LUMA_OFFSET0, H264_SLOTS);
    for (int i = 0; i < H264_SLOTS; i++)
        push[n++] = (uint32_t)(h->surf[i].va >> 8);
    push[n++] = MTHD(SUBCH, SET_PICTURE_CHROMA_OFFSET0, H264_SLOTS);
    for (int i = 0; i < H264_SLOTS; i++)
        push[n++] = h->surf[i].va ? (uint32_t)((h->surf[i].va + h->layout.chroma_offset) >> 8) : 0;
    push[n++] = MTHD(SUBCH, EXECUTE, 1);
    push[n++] = 0;  // no notify, no awaken

    *payload = ++v->seq;
    chan_push(&v->ch, push, n);
    video_release_queue(v, v->semva, *payload);
    return 0;
}

int video_h264_wait(struct video *v, struct video_h264 *h, uint32_t payload, uint32_t index)
{
    volatile uint32_t *sem = v->sem->cpu;
    const volatile uint32_t *status = (const volatile uint32_t *)((uint8_t *)h->setup.mem->cpu + NVDEC_SETUP_STATUS);
    if (wait_word(sem, payload, 2000)) {
        log_msg("video: picture %u not decoded (GP_GET %u, word %#x, status %#x)\n", index, chan_gp_get(&v->ch),
                *sem, status[3]);
        return -1;
    }
    if (status[1] || status[3])
        log_msg("video: picture %u: %u macroblocks decoded, %u in error, error status %#x\n", index, status[0],
                status[1], status[3]);
    return 0;
}

static int h264_picture(struct video *v, const struct h264_dec *d, const struct h264_picture *pic,
                        struct video_h264 *h, const uint8_t *stream)
{
    uint8_t *setup = h->setup.mem->cpu, *bits = h->bits.mem->cpu;
    uint32_t refs = pic->sps->max_num_ref_frames, payload;
    h264_slices(pic, stream, bits, (uint32_t *)(setup + NVDEC_SETUP_SLICES));
    memset(bits + pic->bytes, 0, 64);
    nvdec_h264_setup(d, pic, &h->layout, (struct nvdec_h264_pic *)setup);
    if (video_h264_queue(v, h, pic->slot, pic->index, refs ? pic->index % refs : 0, &payload))
        return -1;
    return video_h264_wait(v, h, payload, pic->index);
}

int video_decode_h264(struct video *v, const uint8_t *stream, size_t len, uint32_t max,
                      int (*emit)(void *arg, const struct video_frame *f), void *arg)
{
    struct h264_dec *d = mem_alloc(sizeof(*d));
    struct h264_picture *pic = mem_alloc(sizeof(*pic));
    struct video_h264 *h = mem_alloc(sizeof(*h));
    uint8_t *frame = NULL;
    size_t pos = 0, frame_size = 0;
    uint32_t n = 0;
    uint64_t t0 = time_us();
    int r = 0;
    h264_init(d);
    while (n < max && (r = h264_next(d, stream, len, &pos, pic)) > 0) {
        const struct h264_sps *s = pic->sps;
        uint32_t w = nvdec_crop_width(s), ht = nvdec_crop_height(s);
        if (!h->width_mbs) {
            frame_size = (size_t)w * ht * 3 / 2;
            // the slices with their start codes take no more than the stream did
            if (video_h264_open(v, h, s->width_mbs, s->height_mbs, H264_SLOTS, len + 64) ||
                !(frame = mem_alloc(frame_size))) {
                r = -1;
                break;
            }
        }
        if (s->width_mbs != h->width_mbs || s->height_mbs != h->height_mbs) {
            log_msg("video: H.264 stream changes size at picture %u\n", pic->index);
            r = -1;
            break;
        }
        if (h264_picture(v, d, pic, h, stream)) {
            r = -1;
            break;
        }
        nvdec_detile_nv12(h->surf[pic->slot].mem->cpu, &h->layout, s, frame, w, frame + (size_t)w * ht, w);
        struct video_frame f = {
            .index = pic->index,
            .idr_count = pic->idr_count,
            .poc = pic->poc[0] < pic->poc[1] ? pic->poc[0] : pic->poc[1],
            .width = w,
            .height = ht,
            .nv12 = frame,
        };
        h264_done(d, pic);
        n++;
        if (emit(arg, &f))
            break;
    }
    if (r >= 0)
        log_msg("video: decoded %u H.264 pictures in %llu us\n", n, (unsigned long long)(time_us() - t0));
    else
        log_msg("video: H.264 decode stopped at picture %u, stream byte %llu\n", n, (unsigned long long)pos);
    if (frame)
        mem_free(frame);
    video_h264_close(v, h);
    mem_free(h);
    mem_free(pic);
    mem_free(d);
    return r < 0 ? -1 : (int)n;
}

void video_free(struct video *v)
{
    struct gsp *s = v->ch.s;
    if (v->ctx_va)
        vmm_unmap(v->vm, v->ctx_va, v->ctx_size);
    if (v->ctx_vram)
        vram_free(gsp_vram(s), v->ctx_vram, v->ctx_size);
    if (v->semva)
        vmm_unmap(v->vm, v->semva, PAGE);
    if (v->sem)
        gsp_dma_free(s, v->sem);
    vmm_flush(gsp_gpu(s));
    chan_free(&v->ch);
    *v = (struct video){ 0 };
}
