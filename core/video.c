// NVDEC through GSP-RM, following nouveau's GSP-mode falcon engines (rm/r535/fifo.c r535_flcn_ctor, rm/r570/fifo.c
// r570_fifo_ectx_size, rm/r535/nvdec.c); methods from NVIDIA's clc7b0.h, which C9B0 keeps
#include "chip.h"
#include "../nvrm/engine.h"
#include "../nvrm/fifo.h"
#include "../nvrm/gr.h"
#include "av1cdf.h"
#include "chan.h"
#include "gsp.h"
#include "mem.h"
#include "nvdec.h"
#include "video.h"
#include "vp9prob.h"

#define PAGE 0x1000ull
#define SUBCH 4
#define CLASS_NVDEC_AV1 0xc7b0  // the first video decoder class with AV1 methods (clc7b0.h)
// VP9 needs SET_INTRA_TOP_BUF_OFFSET, or the engine faults at address 0 (mickys/nvenc), which clc7b0.h has first
#define CLASS_NVDEC_VP9 0xc7b0
#define CLASS_NVDEC_VP8 0xc5b0  // the oldest of NVIDIA's video decoder class headers in Mesa, which has VP8 methods

#define NOP 0x100
#define SET_APPLICATION_ID 0x200
#define APPLICATION_ID_H264 3
#define APPLICATION_ID_VP8 5
#define APPLICATION_ID_HEVC 7
#define APPLICATION_ID_VP9 9  // the one without VP9_WITH_PARSER's header parser, which mesa-nova uses
#define APPLICATION_ID_AV1 0x12  // "AVD" in clc7b0.h: the one that runs (VP9's and the others hang the engine)
#define SEMAPHORE_A 0x240
#define EXECUTE 0x300
#define SEMAPHORE_D 0x304
#define SET_CONTROL_PARAMS 0x400
#define CONTROL_H264 3
#define CONTROL_VP8 5
#define CONTROL_HEVC 7
#define CONTROL_VP9 9
#define CONTROL_AV1 10
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
#define SET_INTRA_TOP_BUF_OFFSET 0x4c8
#define SET_TILE_SIZE_BUF_OFFSET 0x4cc
#define SET_FILTER_BUFFER_OFFSET 0x4d0
#define H264_SET_MBHIST_BUF_OFFSET 0x500
#define VP8_SET_PROB_DATA_OFFSET 0x540
#define HEVC_SET_SCALING_LIST_OFFSET 0x580
#define HEVC_SET_TILE_SIZES_OFFSET 0x584
#define HEVC_SET_FILTER_BUFFER_OFFSET 0x588
#define VP9_SET_PROB_TAB_BUF_OFFSET 0x5c0
#define VP9_SET_CTX_COUNTER_BUF_OFFSET 0x5c4
#define VP9_SET_SEGMENT_READ_BUF_OFFSET 0x5c8
#define VP9_SET_SEGMENT_WRITE_BUF_OFFSET 0x5cc
#define VP9_SET_TILE_SIZE_BUF_OFFSET 0x5d0
#define VP9_SET_COL_MVWRITE_BUF_OFFSET 0x5d4
#define VP9_SET_COL_MVREAD_BUF_OFFSET 0x5d8
#define VP9_SET_FILTER_BUFFER_OFFSET 0x5dc
#define AV1_SET_PROB_TAB_READ_BUF_OFFSET 0x640
#define AV1_SET_PROB_TAB_WRITE_BUF_OFFSET 0x644
#define AV1_SET_SEGMENT_READ_BUF_OFFSET 0x648
#define AV1_SET_SEGMENT_WRITE_BUF_OFFSET 0x64c
#define AV1_SET_COL_MV0_READ_BUF_OFFSET 0x650  // and 1 and 2 after it
#define AV1_SET_COL_MVWRITE_BUF_OFFSET 0x65c
#define AV1_SET_GLOBAL_MODEL_BUF_OFFSET 0x660
#define AV1_SET_FILM_GRAIN_BUF_OFFSET 0x664

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

// The probabilities each reference slot's frame ended in. The specification saves them with the frame (7.20) and a
// later frame starts from the ones its primary reference has; NVDEC writes a frame's into a table of its own, which
// the CPU copies to the slots the frame refreshes, or the table it started from when it doesn't adapt them.
struct av1_tables {
    uint8_t slot[AV1_REFS][AV1_PROBS_SIZE];
    uint32_t refresh;
    int adapt;
};

// The four probability contexts of a VP9 stream (load_probs() and save_probs(), 7.20), which NVDEC leaves to the
// driver, and the frame being decoded: the probabilities it got, the context they came from, which the counts NVDEC
// writes adapt and refresh_frame_context saves them back to, and its picture, which predicts the next frames
struct vp9_tables {
    struct vp9_probs ctx[VP9_CONTEXTS], cur;
    struct vp9_prob_frame frame;
    uint32_t context, slot, width, height;
    int refresh, adapt;
};

int video_supports(const struct video *v, enum nvdec_codec codec)
{
    uint32_t cls = gsp_gpu(v->ch.s)->chip->nvdec;
    switch (codec) {
    case NVDEC_AV1:
        return cls >= CLASS_NVDEC_AV1;
    case NVDEC_VP9:
        return cls >= CLASS_NVDEC_VP9;
    case NVDEC_VP8:
        return cls >= CLASS_NVDEC_VP8;
    default:
        return 1;
    }
}

int video_dec_open(struct video *v, struct video_dec *h, enum nvdec_codec codec, uint32_t width_mbs,
                   uint32_t height_mbs, uint32_t bps, uint32_t slots, uint64_t bits)
{
    *h = (struct video_dec){ .codec = codec, .width_mbs = width_mbs, .height_mbs = height_mbs, .slots = slots };
    if (!video_supports(v, codec) || !width_mbs || width_mbs > 512 || !height_mbs || height_mbs > 512 || !slots ||
        slots > NVDEC_SLOTS || (bps != 1 && bps != 2) || !bits || bits > 64u << 20)
        return -1;
    nvdec_surface_init(width_mbs, height_mbs, bps, &h->layout);
    if (vbuf_alloc(v, &h->setup, NVDEC_SETUP_SIZE) || vbuf_alloc(v, &h->bits, bits))
        goto fail;
    if (codec == NVDEC_H264) {
        struct nvdec_h264_sizes sizes;
        nvdec_h264_sizes(width_mbs, height_mbs, &sizes);
        if (vbuf_alloc(v, &h->coloc, (uint64_t)sizes.coloc * slots) || vbuf_alloc(v, &h->mbhist, sizes.mbhist) ||
            vbuf_alloc(v, &h->history, sizes.history))
            goto fail;
    } else if (codec == NVDEC_H265) {
        struct nvdec_hevc_sizes sizes;
        nvdec_hevc_sizes(&h->layout, &sizes);
        if (vbuf_alloc(v, &h->coloc, (uint64_t)sizes.coloc * slots) || vbuf_alloc(v, &h->filter, sizes.filter) ||
            vbuf_alloc(v, &h->intra_top, sizes.intra_top))
            goto fail;
    } else if (codec == NVDEC_AV1) {
        struct nvdec_av1_sizes sizes;
        nvdec_av1_sizes(&h->layout, &sizes);
        h->av1 = mem_alloc(sizeof(*h->av1));
        // a segmentation map to each surface, then one of zeros
        if (!h->av1 || vbuf_alloc(v, &h->coloc, (uint64_t)sizes.coloc * slots) ||
            vbuf_alloc(v, &h->segmap, (uint64_t)sizes.segmap * (slots + 1)) || vbuf_alloc(v, &h->prob_in, sizes.prob) ||
            vbuf_alloc(v, &h->prob_out, sizes.prob) || vbuf_alloc(v, &h->filter, sizes.filter) ||
            vbuf_alloc(v, &h->intra_top, sizes.intra_top))
            goto fail;
        memset(h->segmap.mem->cpu, 0, h->segmap.size);
    } else if (codec == NVDEC_VP9) {
        struct nvdec_vp9_sizes sizes;
        nvdec_vp9_sizes(&h->layout, &sizes);
        h->vp9 = mem_alloc(sizeof(*h->vp9));
        if (!h->vp9 || vbuf_alloc(v, &h->coloc, 2ull * sizes.coloc) || vbuf_alloc(v, &h->segmap, 2ull * sizes.segmap) ||
            vbuf_alloc(v, &h->prob_in, sizes.prob) || vbuf_alloc(v, &h->counts, sizes.counts) ||
            vbuf_alloc(v, &h->filter, sizes.filter) || vbuf_alloc(v, &h->intra_top, sizes.intra_top))
            goto fail;
        memset(h->segmap.mem->cpu, 0, h->segmap.size);
        for (int i = 0; i < VP9_CONTEXTS; i++)
            vp9_default_probs(&h->vp9->ctx[i]);
    } else {
        struct nvdec_vp8_sizes sizes;
        nvdec_vp8_sizes(&h->layout, &sizes);
        // NVDEC keeps the probabilities from the defaults on, and stops with status 0x10 on a buffer of zeros (nvenc)
        if (vbuf_alloc(v, &h->history, sizes.history) || vbuf_alloc(v, &h->prob_in, sizes.prob))
            goto fail;
        memset(h->history.mem->cpu, 0, h->history.size);
        memset(h->prob_in.mem->cpu, 0, h->prob_in.size);
        vp8_default_probs(h->prob_in.mem->cpu);
    }
    for (uint32_t i = 0; i < slots; i++)
        if (vbuf_alloc(v, &h->surf[i], h->layout.size))
            goto fail;
    return 0;
fail:
    video_dec_close(v, h);
    return -1;
}

void video_dec_close(struct video *v, struct video_dec *h)
{
    vbuf_free(v, &h->setup);
    vbuf_free(v, &h->bits);
    vbuf_free(v, &h->coloc);
    vbuf_free(v, &h->mbhist);
    vbuf_free(v, &h->history);
    vbuf_free(v, &h->filter);
    vbuf_free(v, &h->intra_top);
    vbuf_free(v, &h->segmap);
    vbuf_free(v, &h->prob_in);
    vbuf_free(v, &h->prob_out);
    vbuf_free(v, &h->counts);
    if (h->av1)
        mem_free(h->av1);
    if (h->vp9)
        mem_free(h->vp9);
    for (int i = 0; i < NVDEC_SLOTS; i++)
        vbuf_free(v, &h->surf[i]);
    vmm_flush(gsp_gpu(v->ch.s));
    *h = (struct video_dec){ 0 };
}

// the table the picture reads, from the slot it starts from or the defaults
static void av1_load(struct video_dec *h, const struct nvdec_av1_pic *pic, const struct nvdec_av1_frame *fr)
{
    struct av1_tables *t = h->av1;
    t->adapt = pic->refresh_frame_context;
    t->refresh = fr->refresh;
    if (fr->primary == NVDEC_AV1_DEFAULT)
        av1_default_probs(pic->base_qindex, h->prob_in.mem->cpu);
    else
        memcpy(h->prob_in.mem->cpu, t->slot[fr->primary], AV1_PROBS_SIZE);
}

// the picture is decoded: the table it ends in goes to the slots it refreshed
static void av1_store(struct video_dec *h)
{
    struct av1_tables *t = h->av1;
    const void *table = t->adapt ? h->prob_out.mem->cpu : h->prob_in.mem->cpu;
    for (int i = 0; i < AV1_REFS; i++)
        if (t->refresh >> i & 1)
            memcpy(t->slot[i], table, AV1_PROBS_SIZE);
}

// The probabilities the frame decodes with: its context, reset first if it says so, with the compressed header's
// updates, which the kernel parses again rather than take the client's word for them
static int vp9_load(struct video_dec *h, uint32_t slot, const struct nvdec_vp9_pic *pic,
                    const struct nvdec_vp9_frame *fr)
{
    struct vp9_tables *t = h->vp9;
    const uint8_t *bias = pic->ref_sign_bias;
    t->frame = (struct vp9_prob_frame){
        .intra = pic->key_frame || pic->intra_only,
        .lossless = pic->lossless,
        .switchable = pic->interp_filter == VP9_SWITCHABLE,
        .allow_hp = pic->allow_hp,
        .compound = bias[2] != bias[1] || bias[3] != bias[1],
        .after_key = pic->prev_key_frame,
    };
    for (int i = 0; i < VP9_CONTEXTS; i++)
        if (fr->reset >> i & 1)
            vp9_default_probs(&t->ctx[i]);
    t->cur = t->ctx[fr->context];
    if (vp9_compressed(&t->frame, (const uint8_t *)h->setup.mem->cpu + NVDEC_SETUP_VP9_HEADER, fr->header_len,
                       &t->cur))
        return -1;
    vp9_pack(&t->cur, fr->seg_tree, fr->seg_pred, h->prob_in.mem->cpu);
    memset(h->counts.mem->cpu, 0, h->counts.size);
    t->context = fr->context;
    t->refresh = (int)fr->refresh_context;
    t->adapt = (int)fr->adapt;
    t->slot = slot;
    t->width = pic->current.width;
    t->height = pic->current.height;
    return 0;
}

// the frame is decoded: its context saved back, adapted by what NVDEC counted, and its picture extended for the
// frames that predict from it
static void vp9_store(struct video_dec *h)
{
    struct vp9_tables *t = h->vp9;
    if (t->refresh) {
        if (t->adapt)
            vp9_adapt(&t->frame, &t->ctx[t->context], h->counts.mem->cpu, &t->cur);
        t->ctx[t->context] = t->cur;
    }
    nvdec_extend(h->surf[t->slot].mem->cpu, &h->layout, t->width, t->height);
}

int video_dec_queue(struct video *v, struct video_dec *h, uint32_t slot, uint32_t index, uint32_t error_idx,
                    uint32_t *payload)
{
    static const uint32_t application[] = { APPLICATION_ID_H264, APPLICATION_ID_HEVC, APPLICATION_ID_AV1,
                                            APPLICATION_ID_VP9, APPLICATION_ID_VP8 };
    static const uint32_t control[] = { CONTROL_H264, CONTROL_HEVC, CONTROL_AV1, CONTROL_VP9, CONTROL_VP8 };
    const struct nvdec_h264_pic *h264 = h->setup.mem->cpu;
    const struct nvdec_hevc_pic *hevc = h->setup.mem->cpu;
    const uint8_t *page = h->setup.mem->cpu;
    struct nvdec_av1_pic av1 = { 0 };
    struct nvdec_av1_frame fr = { 0 };
    struct nvdec_av1_sizes av1_sizes = { 0 };
    struct nvdec_vp9_pic vp9 = { 0 };
    struct nvdec_vp9_frame vp9_fr = { 0 };
    struct nvdec_vp9_sizes vp9_sizes = { 0 };
    struct nvdec_vp8_frame vp8_fr = { 0 };
    int is_h264 = h->codec == NVDEC_H264, is_av1 = h->codec == NVDEC_AV1;
    int is_vp9 = h->codec == NVDEC_VP9, is_vp8 = h->codec == NVDEC_VP8;
    // the page is the client's to write: what the kernel goes by is copied out of it first
    if (is_av1) {
        av1 = *(const struct nvdec_av1_pic *)page;
        fr = *(const struct nvdec_av1_frame *)(page + NVDEC_SETUP_AV1);
    } else if (is_vp9) {
        vp9 = *(const struct nvdec_vp9_pic *)page;
        vp9_fr = *(const struct nvdec_vp9_frame *)(page + NVDEC_SETUP_VP9);
    } else if (is_vp8) {
        vp8_fr = *(const struct nvdec_vp8_frame *)(page + NVDEC_SETUP_VP8);
    }
    uint32_t current = is_h264 ? h264->curr_pic_idx : h->codec == NVDEC_H265 ? hevc->curr_pic_idx : slot;
    if (slot >= h->slots || current != slot)
        return -1;
    // the surface each picture offset names: H.264's and H.265's by index, AV1's references LAST_FRAME to
    // ALTREF_FRAME in 0 to 6 and the picture in 8, whatever the setup's frame_buffer_idx say (mesa-nova av1.rs),
    // VP9's LAST_FRAME to ALTREF_FRAME and VP8's golden, altref and last in 0 to 2 and the picture in 3
    // (VP9_FRAME_SFC_ID and VP8_FRAME_SFC_ID of nvdec_drv.h)
    int pics[NVDEC_SLOTS];
    for (int i = 0; i < NVDEC_SLOTS; i++)
        pics[i] = is_h264 || h->codec == NVDEC_H265 ? i : -1;
    if (is_vp9 || is_vp8) {
        const uint32_t *refs = is_vp9 ? vp9_fr.refs : vp8_fr.refs;
        for (int i = 0; i < 3; i++) {
            if (refs[i] >= h->slots)
                return -1;
            pics[i] = (int)refs[i];
        }
        pics[3] = (int)slot;
    }
    if (is_vp9) {
        uint32_t w = vp9.current.width, ht = vp9.current.height;
        // the edges are extended out to the superblocks, which the surfaces have to hold
        if (vp9_fr.context >= VP9_CONTEXTS || vp9_fr.reset > 0xf || vp9_fr.mv_map > 1 || vp9_fr.seg_map > 1 ||
            vp9_fr.header_len > NVDEC_SETUP_SIZE - NVDEC_SETUP_VP9_HEADER || !w || !ht ||
            (w + 63) / 64 * 4 > h->width_mbs || (ht + 63) / 64 * 4 > h->height_mbs || vp9_load(h, slot, &vp9, &vp9_fr))
            return -1;
        nvdec_vp9_sizes(&h->layout, &vp9_sizes);
        if (vp9_fr.clear_seg_map)
            memset((uint8_t *)h->segmap.mem->cpu + (uint64_t)vp9_sizes.segmap * vp9_fr.seg_map, 0, vp9_sizes.segmap);
    }
    if (is_av1) {
        if ((fr.primary >= AV1_REFS && fr.primary != NVDEC_AV1_DEFAULT) || fr.refresh > 0xff ||
            (fr.segmap >= h->slots && fr.segmap != NVDEC_AV1_DEFAULT) || fr.coloc[0] >= h->slots ||
            fr.coloc[1] >= h->slots || fr.coloc[2] >= h->slots)
            return -1;
        for (int i = 0; i < AV1_REFS_PER_FRAME; i++) {
            if (fr.refs[i] >= h->slots)
                return -1;
            pics[i] = (int)fr.refs[i];
        }
        pics[8] = (int)slot;
        if (fr.segmap == NVDEC_AV1_DEFAULT)
            fr.segmap = h->slots;  // the map of zeros
        nvdec_av1_sizes(&h->layout, &av1_sizes);
        av1_load(h, &av1, &fr);
        if (!av1.segment_enabled)  // the map the next picture predicts from is zero, which the engine doesn't write
            memset((uint8_t *)h->segmap.mem->cpu + (uint64_t)av1_sizes.segmap * slot, 0, av1_sizes.segmap);
    }
    memset((uint8_t *)h->setup.mem->cpu + NVDEC_SETUP_STATUS, 0, 64);

    uint32_t push[128], n = 0;
#define M(mthd, val) (push[n++] = MTHD(SUBCH, mthd, 1), push[n++] = (val))
    M(SET_APPLICATION_ID, application[h->codec]);
    M(SET_CONTROL_PARAMS, control[h->codec] | CONTROL_GPTIMER_ON | CONTROL_ERR_CONCEAL_ON |
                              CONTROL_ERROR_FRM_IDX(error_idx & 0x3f) | CONTROL_MBTIMER_ON);
    M(SET_DRV_PIC_SETUP_OFFSET, (uint32_t)(h->setup.va >> 8));
    M(SET_IN_BUF_BASE_OFFSET, (uint32_t)(h->bits.va >> 8));
    M(SET_PICTURE_INDEX, index);
    if (is_h264 || h->codec == NVDEC_H265)
        M(SET_COLOC_DATA_OFFSET, (uint32_t)(h->coloc.va >> 8));
    M(SET_NVDEC_STATUS_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_STATUS) >> 8));
    if (is_vp9) {
        uint64_t coloc = h->coloc.va, segmap = h->segmap.va;
        M(VP9_SET_PROB_TAB_BUF_OFFSET, (uint32_t)(h->prob_in.va >> 8));
        M(VP9_SET_CTX_COUNTER_BUF_OFFSET, (uint32_t)(h->counts.va >> 8));
        M(VP9_SET_SEGMENT_READ_BUF_OFFSET, (uint32_t)((segmap + (uint64_t)vp9_sizes.segmap * vp9_fr.seg_map) >> 8));
        M(VP9_SET_SEGMENT_WRITE_BUF_OFFSET,
          (uint32_t)((segmap + (uint64_t)vp9_sizes.segmap * (vp9_fr.seg_map ^ 1)) >> 8));
        M(VP9_SET_TILE_SIZE_BUF_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_TILES) >> 8));
        M(VP9_SET_COL_MVWRITE_BUF_OFFSET, (uint32_t)((coloc + (uint64_t)vp9_sizes.coloc * (vp9_fr.mv_map ^ 1)) >> 8));
        M(VP9_SET_COL_MVREAD_BUF_OFFSET, (uint32_t)((coloc + (uint64_t)vp9_sizes.coloc * vp9_fr.mv_map) >> 8));
        M(VP9_SET_FILTER_BUFFER_OFFSET, (uint32_t)(h->filter.va >> 8));
        M(SET_INTRA_TOP_BUF_OFFSET, (uint32_t)(h->intra_top.va >> 8));
    } else if (is_vp8) {
        M(SET_HISTORY_OFFSET, (uint32_t)(h->history.va >> 8));
        M(VP8_SET_PROB_DATA_OFFSET, (uint32_t)(h->prob_in.va >> 8));
    } else if (is_av1) {
        uint64_t coloc = h->coloc.va, segmap = h->segmap.va;
        M(AV1_SET_PROB_TAB_READ_BUF_OFFSET, (uint32_t)(h->prob_in.va >> 8));
        M(AV1_SET_PROB_TAB_WRITE_BUF_OFFSET, (uint32_t)(h->prob_out.va >> 8));
        M(AV1_SET_SEGMENT_READ_BUF_OFFSET, (uint32_t)((segmap + (uint64_t)av1_sizes.segmap * fr.segmap) >> 8));
        M(AV1_SET_SEGMENT_WRITE_BUF_OFFSET, (uint32_t)((segmap + (uint64_t)av1_sizes.segmap * slot) >> 8));
        for (int i = 0; i < 3; i++)
            M(AV1_SET_COL_MV0_READ_BUF_OFFSET + 4 * i, (uint32_t)((coloc + (uint64_t)av1_sizes.coloc * fr.coloc[i]) >> 8));
        M(AV1_SET_COL_MVWRITE_BUF_OFFSET, (uint32_t)((coloc + (uint64_t)av1_sizes.coloc * slot) >> 8));
        M(AV1_SET_GLOBAL_MODEL_BUF_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_GLOBAL) >> 8));
        M(AV1_SET_FILM_GRAIN_BUF_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_GRAIN) >> 8));
        M(SET_TILE_SIZE_BUF_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_TILES) >> 8));
        M(SET_FILTER_BUFFER_OFFSET, (uint32_t)(h->filter.va >> 8));
        M(SET_INTRA_TOP_BUF_OFFSET, (uint32_t)(h->intra_top.va >> 8));
    } else if (is_h264) {
        M(SET_SLICE_OFFSETS_BUF_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_SLICES) >> 8));
        M(SET_HISTORY_OFFSET, (uint32_t)(h->history.va >> 8));
        M(H264_SET_MBHIST_BUF_OFFSET, (uint32_t)(h->mbhist.va >> 8));
    } else {
        M(HEVC_SET_SCALING_LIST_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_SCALING) >> 8));
        M(HEVC_SET_TILE_SIZES_OFFSET, (uint32_t)((h->setup.va + NVDEC_SETUP_TILES) >> 8));
        M(HEVC_SET_FILTER_BUFFER_OFFSET, (uint32_t)(h->filter.va >> 8));
        M(SET_INTRA_TOP_BUF_OFFSET, (uint32_t)(h->intra_top.va >> 8));
    }
#undef M
    push[n++] = MTHD(SUBCH, SET_PICTURE_LUMA_OFFSET0, NVDEC_SLOTS);
    for (int i = 0; i < NVDEC_SLOTS; i++)
        push[n++] = pics[i] >= 0 ? (uint32_t)(h->surf[pics[i]].va >> 8) : 0;
    push[n++] = MTHD(SUBCH, SET_PICTURE_CHROMA_OFFSET0, NVDEC_SLOTS);
    for (int i = 0; i < NVDEC_SLOTS; i++)
        push[n++] = pics[i] >= 0 && h->surf[pics[i]].va
                        ? (uint32_t)((h->surf[pics[i]].va + h->layout.chroma_offset) >> 8)
                        : 0;
    push[n++] = MTHD(SUBCH, EXECUTE, 1);
    push[n++] = 0;  // no notify, no awaken

    *payload = ++v->seq;
    chan_push(&v->ch, push, n);
    video_release_queue(v, v->semva, *payload);
    return 0;
}

int video_dec_wait(struct video *v, struct video_dec *h, uint32_t payload, uint32_t index)
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
    if (h->av1)
        av1_store(h);
    if (h->vp9)
        vp9_store(h);
    return 0;
}

static int decode_picture(struct video *v, struct nvdec_stream *s, const struct nvdec_pic *pic, struct video_dec *h,
                          const uint8_t *stream)
{
    uint32_t payload;
    nvdec_stream_prepare(s, stream, &h->layout, h->setup.mem->cpu, h->bits.mem->cpu);
    if (video_dec_queue(v, h, pic->slot, pic->index, pic->error_idx, &payload))
        return -1;
    return video_dec_wait(v, h, payload, pic->index);
}

int video_decode(struct video *v, enum nvdec_codec codec, const uint8_t *stream, size_t len, uint32_t max,
                 int (*emit)(void *arg, const struct video_frame *f), void *arg)
{
    const char *name = nvdec_codec_name(codec);
    struct nvdec_stream s;
    struct nvdec_pic pic;
    struct video_dec *h = mem_alloc(sizeof(*h));
    uint8_t *frame = NULL;
    size_t pos = 0;
    uint32_t n = 0;
    uint64_t t0 = time_us();
    int r = nvdec_stream_open(&s, codec);
    while (r >= 0 && n < max && (r = nvdec_stream_next(&s, stream, len, &pos, &pic)) > 0) {
        uint32_t w = pic.crop.width, ht = pic.crop.height, pitch = w * pic.bps;
        if (!h->width_mbs) {
            // the slices with their start codes take no more than the stream did; an H.264 stream may keep more
            // references than it says
            uint32_t slots = codec == NVDEC_H264 ? NVDEC_SLOTS : pic.slots;
            // AV1 and VP9 frames can be smaller than the surfaces
            if (video_dec_open(v, h, codec, pic.width_mbs, pic.height_mbs, pic.bps, slots, len + 64) ||
                !(frame = mem_alloc((size_t)pic.width_mbs * pic.height_mbs * 384 * pic.bps))) {
                r = -1;
                break;
            }
        }
        if (pic.width_mbs != h->width_mbs || pic.height_mbs != h->height_mbs || pic.bps != h->layout.bps) {
            log_msg("video: %s stream changes size at picture %u\n", name, pic.index);
            r = -1;
            break;
        }
        if (!pic.existing && decode_picture(v, &s, &pic, h, stream)) {
            r = -1;
            break;
        }
        nvdec_detile(h->surf[pic.slot].mem->cpu, &h->layout, &pic.crop, frame, pitch, frame + (size_t)pitch * ht,
                     pitch);
        struct video_frame f = {
            .index = pic.index,
            .idr_count = pic.idr_count,
            .poc = pic.poc,
            .width = w,
            .height = ht,
            .bps = pic.bps,
            .nv12 = frame,
        };
        nvdec_stream_done(&s);
        n++;
        if (pic.output && emit(arg, &f))
            break;
    }
    if (r >= 0)
        log_msg("video: decoded %u %s pictures in %llu us\n", n, name, (unsigned long long)(time_us() - t0));
    else
        log_msg("video: %s decode stopped at picture %u, stream byte %llu\n", name, n, (unsigned long long)pos);
    if (frame)
        mem_free(frame);
    video_dec_close(v, h);
    mem_free(h);
    nvdec_stream_close(&s);
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
