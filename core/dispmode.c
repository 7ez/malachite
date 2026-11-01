// Lighting displays: the root display object, the core channel and each head's window and window-immediate
// channels, DisplayPort link training through GSP-RM and the methods nvkms and nouveau send for a head with one
// window (head h, window 2h)
#include "chip.h"
#include "../hw/clc37a.h"
#include "../hw/clc37b.h"
#include "../hw/clc37d.h"
#include "../hw/clc37e.h"
#include "../hw/clc57d.h"
#include "../hw/clc57e.h"
#include "../nvrm/disp.h"
#include "../nvrm/gsp.h"
#include "cursor.h"
#include "disp.h"
#include "disp_nouveau.h"
#include "gsp.h"
#include "mem.h"

#define DRF_LO(f) (0 ? f)
#define DRF_HI(f) (1 ? f)
#define NVV(d, f, v) ((((uint32_t)(v)) & (uint32_t)((1ull << (DRF_HI(d##_##f) - DRF_LO(d##_##f) + 1)) - 1)) << DRF_LO(d##_##f))
#define NVD(d, f, c) NVV(d, f, d##_##f##_##c)


#define DISP_ROOT 0xd1500000u
#define FB_CTXDMA 0xfb000001u
#define RAMHT_BITS 10      // GA102's 0x2000-byte RAMHT, at the start of the instance memory
#define CTXDMA_OFF 0x8000  // well clear of anything GSP-RM might put at the start of object memory
#define NTFY_CTXDMA 0xfb000002u
#define NTFY_OFF (CTXDMA_OFF + 0x20)

// display channels: a push buffer in system memory, PUT and GET in the channel's user area
static int dchan_new(struct disp *d, struct dchan *ch, uint32_t cls, uint32_t inst, uint32_t user)
{
    struct gsp *s = d->s;
    const struct rm_client *in = gsp_internal(s);
    *ch = (struct dchan){ .user = user, .cls = cls };
    if (dma_alloc(gsp_gpu(s), &ch->pb, 0x1000, 1))
        return -1;
    NV2080_CTRL_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER_PARAMS pb = {
        .addressSpace = 1,  // ADDR_SYSMEM
        .physicalAddr = ch->pb.addr,
        .limit = 0xfff,
        .cacheSnoop = 1,
        .hclass = cls,
        .channelInstance = inst,
        .valid = 1,
        .pbTargetAperture = 3,  // PHYS_PCI_COHERENT; 0 (IOVA) has the channel fetch from VRAM
        .subDeviceId = 1,       // one-hot, as nouveau's r570 sets it here and in the allocation
    };
    if (!rm_control(s, in->client, in->subdevice, NV2080_CTRL_CMD_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER, &pb, sizeof(pb)))
        return -1;
    NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS a = { .channelInstance = inst, .subDeviceId = 1 };
    if (rm_alloc(s, d->c.client, DISP_ROOT, cls << 16 | inst, cls, &a, sizeof(a))) {
        struct gpu *g = gsp_gpu(s);
        log_msg("disp: channel %#x refused, core channel state %#x, exception %#x %#x %#x\n", cls, rd32(g, 0x610630),
                rd32(g, 0x611020), rd32(g, 0x611024), rd32(g, 0x611028));
        return -1;
    }
    ch->handle = cls << 16 | inst;
    return 0;
}

#define PB_DWORDS (0x1000 / 4)

static void dmthd(struct dchan *ch, uint32_t mthd, uint32_t data)
{
    uint32_t *pb = ch->pb.cpu;
    // back to the start, keeping a dword for the jump; GET sits at the last kick, past what gets overwritten
    if (ch->cur + 3 > PB_DWORDS) {
        pb[ch->cur] = 0x20000000;  // NV507C_DMA_OPCODE_JUMP to 0
        ch->cur = 0;
    }
    pb[ch->cur++] = 1u << 18 | (mthd & 0x3ffc);  // count 27:18, offset 13:2 (NVC37D_DMA_METHOD_*)
    pb[ch->cur++] = data;
}

static int dkick(struct disp *d, struct dchan *ch)
{
    struct gpu *g = gsp_gpu(d->s);
    __sync_synchronize();
    wr32(g, ch->user + 0x0, ch->cur * 4);  // PUT, bytes
    for (int i = 0; i < 2000; i++) {
        if (rd32(g, ch->user + 0x4) == ch->cur * 4)  // GET caught up
            return 0;
        delay_us(1000);
    }
    // GET runs a few methods ahead of the one the channel is stuck on
    uint32_t get = rd32(g, ch->user + 0x4), chid = ch->user == 0x680000 ? 0 : (ch->user - 0x690000) / 0x1000 + 1;
    const uint32_t *pb = ch->pb.cpu;
    log_msg("disp: channel %#x stuck, GET %#x PUT %#x, status %#x, exception %#x %#x %#x\n", ch->cls, get,
            ch->cur * 4, rd32(g, chid ? 0x610664 : 0x610630), rd32(g, 0x611020 + chid * 12),
            rd32(g, 0x611024 + chid * 12), rd32(g, 0x611028 + chid * 12));
    for (uint32_t i = get / 4 >= 4 ? get / 4 - 4 : 0; i < get / 4 + 4 && i < ch->cur; i += 2)
        log_msg("disp: pb %#x: mthd %#x data %#x\n", i * 4, pb[i] & 0x3ffc, pb[i + 1]);
    return -1;
}

static void dchan_free(struct disp *d, struct dchan *ch)
{
    if (ch->pb.cpu)
        dma_free(gsp_gpu(d->s), &ch->pb);
}

// a context DMA over [base, limit] of VRAM in the instance memory (RM's instmemCommitContextDma_v03_00)
static void ctxdma_write(struct disp *d, uint32_t off, uint64_t base, uint64_t limit)
{
    struct gpu *g = gsp_gpu(d->s);
    vram_wr32(g, d->inst + off + 0x00, 0x00000045);  // as nouveau has them: small pages, rw, VRAM, pitch
    vram_wr32(g, d->inst + off + 0x04, (uint32_t)(base >> 8));
    vram_wr32(g, d->inst + off + 0x08, (uint32_t)(base >> 40));
    vram_wr32(g, d->inst + off + 0x0c, (uint32_t)(limit >> 8));
    vram_wr32(g, d->inst + off + 0x10, (uint32_t)(limit >> 40));
}

// makes the context DMA at off findable as handle by channel chid, through RAMHT
static void ctxdma_insert(struct disp *d, uint32_t chid, uint32_t handle, uint32_t off)
{
    struct gpu *g = gsp_gpu(d->s);

    // nouveau's nvkm_ramht_hash, which it runs Ampere under GSP-RM with (RM's instmemHashFunc_v03_00 also folds
    // in the client's low bits)
    uint32_t hash = 0;
    for (uint32_t h = handle; h; h >>= RAMHT_BITS)
        hash ^= h & ((1u << RAMHT_BITS) - 1);
    hash ^= chid << (RAMHT_BITS - 4);
    for (uint32_t i = 0; i < (1u << RAMHT_BITS); i++) {
        uint32_t slot = (hash + i) & ((1u << RAMHT_BITS) - 1);
        if (vram_rd32(g, d->inst + slot * 8))
            continue;
        vram_wr32(g, d->inst + slot * 8, handle);
        vram_wr32(g, d->inst + slot * 8 + 4, chid << 25 | (d->c.client & 0x3fff) | off << 9);
        return;
    }
}

// U0.16 to FP16, for the window's input LUT (nouveau's fixedU0_16_FP16)
static uint16_t u16_to_fp16(uint32_t v)
{
    if (v < 2)
        return 0;
    int p = 31 - __builtin_clz(v);  // value is v / 2^16 = 1.m * 2^(p-16)
    uint32_t mant = p >= 10 ? (v >> (p - 10)) & 0x3ff : (v << (10 - p)) & 0x3ff;
    return (uint16_t)((p - 1) << 10 | mant);
}

#define LUT_ENTRIES 1025  // 1024 and the last one again, for interpolation
#define LUT_SIZE 0x4000

// identity LUTs: a 32-byte header, then r, g, b, 0 as 16-bit values per entry (fp16 for the input LUT)
static void lut_identity(struct gpu *g, uint64_t at, int fp16)
{
    for (uint32_t o = 0; o < 0x20; o += 4)
        vram_wr32(g, at + o, 0);
    for (uint32_t i = 0; i < LUT_ENTRIES; i++) {
        uint32_t v = (i < 1024 ? i : 1023) * 64;
        uint16_t c = fp16 ? u16_to_fp16(v) : (uint16_t)v;
        vram_wr32(g, at + 0x20 + i * 8, c | (uint32_t)c << 16);
        vram_wr32(g, at + 0x20 + i * 8 + 4, c);
    }
}

// an EDID detailed timing in the hardware's terms, which start one unit into the sync pulse
static void raster(const struct disp_mode *t, struct disp_raster *m)
{
    uint32_t hss = t->hactive + t->hsync_off, vss = t->vactive + t->vsync_off;
    m->clock = t->clock;
    m->htotal = t->hactive + t->hblank;
    m->hsynce = t->hsync_len - 1;
    m->hblanke = m->htotal - hss - 1;
    m->hblanks = m->hblanke + t->hactive;
    m->vtotal = t->vactive + t->vblank;
    m->vsynce = t->vsync_len - 1;
    m->vblanke = m->vtotal - vss - 1;
    m->vblanks = m->vblanke + t->vactive;
    m->nhsync = !(t->flags & 0x2);
    m->nvsync = !(t->flags & 0x4);
}

// how far the head scales in down to out, in 1024ths, rounded up (nvComputeScalingUsageBounds)
static uint32_t scale_factor(uint32_t in, uint32_t out)
{
    return (in * 1024 + out - 1) / out;
}

// one interlocked update of the core and a window, as nvkms does (UpdateWindowC3, UpdateCoreC3): the window
// waits for the core; returns once the core channel is idle again, i.e. GSP-RM's supervisors have run
static int commit(struct disp *d, int head)
{
    struct gpu *g = gsp_gpu(d->s);
    struct disp_head *hd = &d->head[head];
    const int wndw = head * 2;
    dmthd(&hd->wimm, NVC37B_UPDATE, 1);
    dmthd(&hd->wndw, NVC37E_SET_INTERLOCK_FLAGS, 1);  // INTERLOCK_WITH_CORE
    dmthd(&hd->wndw, NVC37E_SET_WINDOW_INTERLOCK_FLAGS, 0);
    dmthd(&hd->wndw, NVC37E_UPDATE, 1);
    if (dkick(d, &hd->wimm) || dkick(d, &hd->wndw))
        return -1;
    dmthd(&d->core, NVC37D_SET_INTERLOCK_FLAGS, 0);
    dmthd(&d->core, NVC37D_SET_WINDOW_INTERLOCK_FLAGS, 1u << wndw);
    dmthd(&d->core, NVC37D_UPDATE, 1);
    if (dkick(d, &d->core))
        return -1;
    uint32_t st = 0;
    for (int i = 0; i < 500; i++) {
        st = rd32(g, 0x610630);
        if ((st >> 16 & 0x1f) == 0xb && !(st >> 31))  // NV_PDISP_FE_CHNSTATUS_CORE_STATE_IDLE, no method executing
            return 0;
        delay_us(10000);
    }
    log_msg("disp: update didn't finish: core %#x window %#x\n", st, rd32(g, 0x610664 + wndw * 4));
    return -1;
}

// once per display: the root object, the core channel, the notifier memory, the context DMAs
static int setup(struct disp *d)
{
    if (d->set_up)
        return 0;
    if (rm_alloc(d->s, d->c.client, d->c.device, DISP_ROOT, gsp_gpu(d->s)->chip->disp, NULL, 0))
        return -1;
    if (dchan_new(d, &d->core, gsp_gpu(d->s)->chip->core, 0, 0x680000))
        return -1;
    d->ntfy = vram_alloc(gsp_vram(d->s), 0x1000, 0x1000);
    if (!d->ntfy)
        return -1;
    vram_zero(gsp_gpu(d->s), d->ntfy, 0x1000);
    ctxdma_write(d, CTXDMA_OFF, 0, gsp_get_info(d->s)->fb_size - 1);
    ctxdma_write(d, NTFY_OFF, d->ntfy, d->ntfy + 0xfff);
    ctxdma_insert(d, 0, NTFY_CTXDMA, NTFY_OFF);
    ctxdma_insert(d, 0, FB_CTXDMA, CTXDMA_OFF);  // the core, for the output LUTs
    d->set_up = 1;
    return 0;
}

// once per head: its window's channels, which find the context DMAs too, and its LUTs
static int head_setup(struct disp *d, int head)
{
    struct disp_head *hd = &d->head[head];
    const int wndw = head * 2;
    if (hd->set_up)
        return 0;
    if (dchan_new(d, &hd->wndw, gsp_gpu(d->s)->chip->wndw, wndw, 0x690000 + wndw * 0x1000) ||
        dchan_new(d, &hd->wimm, gsp_gpu(d->s)->chip->wimm, wndw, 0x6b0000 + wndw * 0x1000))
        return -1;
    ctxdma_insert(d, 1 + wndw, NTFY_CTXDMA, NTFY_OFF);
    ctxdma_insert(d, 1 + wndw, FB_CTXDMA, CTXDMA_OFF);  // window channels are user channels 1..32
    hd->luts = vram_alloc(gsp_vram(d->s), 3 * LUT_SIZE, 0x1000);
    if (!hd->luts)
        return -1;
    lut_identity(gsp_gpu(d->s), hd->luts, 1);
    lut_identity(gsp_gpu(d->s), hd->luts + LUT_SIZE, 0);
    hd->olut = 0;
    hd->wndw_ntfy = head * 0x40 + 0x10;
    hd->set_up = 1;
    return 0;
}

static int is_dp(uint32_t protocol)
{
    return protocol == NVC37D_SOR_SET_CONTROL_PROTOCOL_DP_A || protocol == NVC37D_SOR_SET_CONTROL_PROTOCOL_DP_B;
}

// no OR on the head, no surface in the window
static int detach(struct disp *d, int head, uint32_t sor)
{
    dmthd(&d->core, NVC37D_SOR_SET_CONTROL(sor), 0);
    dmthd(&d->head[head].wndw, NVC57E_SET_CONTEXT_DMA_ISO(0), 0);
    return commit(d, head);
}

// display sleep: the head off its OR and the sink told to power down (DPCD SET_POWER D3), so the monitor sees no
// signal and goes to standby; disp_modeset() brings it back
int disp_blank(struct disp *d, int head, int out)
{
    const struct disp_output *o = &d->out[out];
    if (!d->head[head].set_up || detach(d, head, d->head[head].sor))
        return -1;
    const uint8_t d3 = 2;
    if (is_dp(o->protocol)) {
        dp_aux_write(d, o->id, 0x600, &d3, 1);
        d->head[head].dp_lanes = 0;
    }
    log_msg("disp: display %#x asleep\n", o->id);
    return 0;
}

static int is_tmds(uint32_t protocol)
{
    return protocol == NVC37D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_A ||
           protocol == NVC37D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_B ||
           protocol == NVC37D_SOR_SET_CONTROL_PROTOCOL_DUAL_TMDS;
}

// an HDMI sink, not DVI: a CEA-861 extension with the HDMI vendor-specific data block (OUI 00-0C-03)
static int edid_hdmi(const struct disp_output *o)
{
    if (o->edid_size < 256 || o->edid[128] != 0x02)
        return 0;
    const uint8_t *c = o->edid + 128;
    for (uint32_t i = 4; i < c[2] && i < 127;) {
        uint32_t tag = c[i] >> 5, len = c[i] & 0x1f;
        if (tag == 3 && len >= 3 && c[i + 1] == 0x03 && c[i + 2] == 0x0c && c[i + 3] == 0x00)
            return 1;
        i += 1 + len;
    }
    return 0;
}

int disp_modeset(struct disp *d, int head, int out, const struct disp_mode *t, uint64_t fb, uint32_t pitch)
{
    if (head < 0 || head >= (int)d->heads)
        return -1;
    struct disp_output *o = &d->out[out];
    struct disp_head *hd = &d->head[head];
    struct disp_raster m;
    raster(t, &m);
    uint32_t w = mode_w(t), h = mode_h(t);
    if (setup(d) || head_setup(d, head))
        return -1;

    // not a SOR another head's display has (nouveau's r535_outp_acquire)
    NV0073_CTRL_DFP_ASSIGN_SOR_PARAMS as = { .displayId = o->id };
    for (int i = 0; i < (int)d->heads; i++)
        if (i != head && d->head[i].display && d->head[i].display != o->id)
            as.sorExcludeMask |= 1u << d->head[i].sor;
    NV0073_CTRL_DFP_ASSIGN_SOR_PARAMS *ar =
        rm_control(d->s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DFP_ASSIGN_SOR, &as, sizeof(as));
    uint32_t sor = ~0u;
    for (uint32_t i = 0; ar && i < NV0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS; i++)
        if (ar->sorAssignListWithTag[i].displayMask & o->id)
            sor = i;
    if (sor == ~0u) {
        log_msg("disp: no SOR for display %#x (other heads' %#x)\n", o->id, as.sorExcludeMask);
        return DISP_NO_OUTPUT;
    }
    // the protocol on the OR it now has (the probe asked before any was assigned: a TMDS port may be on link B)
    NV0073_CTRL_SPECIFIC_OR_GET_INFO_PARAMS orq = { .displayId = o->id };
    NV0073_CTRL_SPECIFIC_OR_GET_INFO_PARAMS *ori =
        rm_control(d->s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_SPECIFIC_OR_GET_INFO, &orq, sizeof(orq));
    if (ori && ori->protocol != o->protocol) {
        log_msg("disp: display %#x: protocol %u on SOR%u, not %u\n", o->id, ori->protocol, ori->index, o->protocol);
        o->protocol = ori->protocol;
    }
    if (is_tmds(o->protocol) && edid_hdmi(o)) {
        NV0073_CTRL_SPECIFIC_SET_HDMI_ENABLE_PARAMS he = { .displayId = o->id, .enable = 1 };
        if (!rm_control(d->s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_SPECIFIC_SET_HDMI_ENABLE, &he, sizeof(he)))
            log_msg("disp: display %#x: HDMI not enabled\n", o->id);
    }
    uint64_t centi = (uint64_t)m.clock * 100000 / ((uint64_t)m.htotal * m.vtotal);  // Hz x 100
    log_msg("disp: display %#x on SOR%u protocol %u, head %d: %ux%u at %u.%02u Hz, raster %ux%u at %u kHz\n", o->id,
            sor, o->protocol, head, w, h, (unsigned)(centi / 100), (unsigned)(centi % 100), m.htotal, m.vtotal,
            m.clock);
    hd->sor = sor;
    hd->display = o->id;
    // take the head off whatever the EFI firmware lit first (nvkms shuts heads down before reattaching them)
    struct dchan *c = &d->core, *wc = &hd->wndw;
    if (detach(d, head, sor))
        return -1;
    log_msg("disp: head %d detached\n", head);

    if (is_dp(o->protocol)) {
        const uint8_t d0 = 1;  // awake, should disp_blank() have put it to sleep
        dp_aux_write(d, o->id, 0x600, &d0, 1);
        if (disp_dp_link(d, head, sor, o, t))
            return DISP_NO_OUTPUT;
    } else if (o->dp) {
        log_msg("disp: display %#x: DisplayPort, but protocol %u on SOR%u\n", o->id, o->protocol, sor);
    }

    // core: window bounds and owners, then the head (corec57d_init, headc57d, sorc37d_ctrl)
    for (int i = 0; i < 8; i++) {
        dmthd(c, NVC37D_WINDOW_SET_CONTROL(i), NVV(NVC37D, WINDOW_SET_CONTROL_OWNER, i >> 1));
        dmthd(c, NVC57D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS(i),
              NVD(NVC57D, WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED1BPP, TRUE) |
                  NVD(NVC57D, WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED2BPP, TRUE) |
                  NVD(NVC57D, WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED4BPP, TRUE) |
                  NVD(NVC57D, WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS_RGB_PACKED8BPP, TRUE));
        dmthd(c, NVC57D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS(i), 0);
        dmthd(c, NVC57D_WINDOW_SET_WINDOW_USAGE_BOUNDS(i),
              NVV(NVC57D, WINDOW_SET_WINDOW_USAGE_BOUNDS_MAX_PIXELS_FETCHED_PER_LINE, 0x7fff) |
                  NVD(NVC57D, WINDOW_SET_WINDOW_USAGE_BOUNDS_ILUT_ALLOWED, TRUE) |
                  NVD(NVC57D, WINDOW_SET_WINDOW_USAGE_BOUNDS_INPUT_SCALER_TAPS, TAPS_2) |
                  NVD(NVC57D, WINDOW_SET_WINDOW_USAGE_BOUNDS_UPSCALING_ALLOWED, FALSE));
    }
    for (unsigned i = 0; i < sizeof(nouveau_head0) / sizeof(nouveau_head0[0]); i++)
        dmthd(c, nouveau_head0[i][0] + head * 0x400, nouveau_head0[i][1]);
    dmthd(c, NVC57D_HEAD_SET_CONTROL_OUTPUT_RESOURCE(head),
          NVV(NVC57D, HEAD_SET_CONTROL_OUTPUT_RESOURCE_HSYNC_POLARITY, m.nhsync) |
              NVV(NVC57D, HEAD_SET_CONTROL_OUTPUT_RESOURCE_VSYNC_POLARITY, m.nvsync) |
              NVD(NVC57D, HEAD_SET_CONTROL_OUTPUT_RESOURCE_PIXEL_DEPTH, BPP_24_444) |
              NVD(NVC57D, HEAD_SET_CONTROL_OUTPUT_RESOURCE_EXT_PACKET_WIN, NONE));  // 0 is window 0; GSP-RM logs Xid 56
    dmthd(c, NVC57D_HEAD_SET_PROCAMP(head), 0);  // RGB, VESA range
    dmthd(c, NVC37D_HEAD_SET_DITHER_CONTROL(head), 0);
    dmthd(c, NVC57D_HEAD_SET_RASTER_SIZE(head), m.htotal | m.vtotal << 16);
    dmthd(c, NVC57D_HEAD_SET_RASTER_SYNC_END(head), m.hsynce | m.vsynce << 16);
    dmthd(c, NVC57D_HEAD_SET_RASTER_BLANK_END(head), m.hblanke | m.vblanke << 16);
    dmthd(c, NVC57D_HEAD_SET_RASTER_BLANK_START(head), m.hblanks | m.vblanks << 16);
    dmthd(c, 0x2074 + head * 0x400, 0 << 16 | 1);  // blank2, for interlaced modes; unpublished, nouveau sends it
    dmthd(c, 0x2008 + head * 0x400, 0);            // HEAD_SET_CONTROL: progressive, no locking
    dmthd(c, NVC57D_HEAD_SET_PIXEL_CLOCK_FREQUENCY(head), m.clock * 1000);
    dmthd(c, NVC57D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX(head), m.clock * 1000);
    dmthd(c, NVC57D_HEAD_SET_HEAD_USAGE_BOUNDS(head),
          NVD(NVC57D, HEAD_SET_HEAD_USAGE_BOUNDS_CURSOR, USAGE_W256_H256) |
              NVD(NVC57D, HEAD_SET_HEAD_USAGE_BOUNDS_OLUT_ALLOWED, TRUE) |
              NVD(NVC57D, HEAD_SET_HEAD_USAGE_BOUNDS_OUTPUT_SCALER_TAPS, TAPS_2) |
              NVD(NVC57D, HEAD_SET_HEAD_USAGE_BOUNDS_UPSCALING_ALLOWED, TRUE));
    // Turing and later want both LUTs, identity here (headc57d_olut_set)
    dmthd(c, 0x2280 + head * 0x400, 1 | 2 << 2 | (LUT_ENTRIES + 4) << 8);  // OLUT: interpolate, DIRECT10, size
    dmthd(c, 0x2284 + head * 0x400, 0xffffffff);                            // FP_NORM_SCALE
    dmthd(c, 0x2288 + head * 0x400, FB_CTXDMA);
    dmthd(c, 0x228c + head * 0x400, (uint32_t)((hd->luts + LUT_SIZE * (1 + hd->olut)) >> 8));
    dmthd(c, NVC37D_HEAD_SET_VIEWPORT_SIZE_IN(head), w | h << 16);
    dmthd(c, NVC37D_HEAD_SET_VIEWPORT_SIZE_OUT(head), (uint32_t)t->hactive | (uint32_t)t->vactive << 16);
    dmthd(c, NVC37D_HEAD_SET_MAX_OUTPUT_SCALE_FACTOR(head),
          scale_factor(w, t->hactive) | scale_factor(h, t->vactive) << 16);
    dmthd(c, 0x2020 + head * 0x400, o->id);  // HEAD_SET_DISPLAY_ID(head, 0)
    dmthd(c, NVC37D_SOR_SET_CONTROL(sor), 1u << head | NVV(NVC37D, SOR_SET_CONTROL_PROTOCOL, o->protocol));

    // the window: the framebuffer, opaque, full size (wndwc67e_image_set, wndwc37e_blend_set)
    for (unsigned i = 0; i < sizeof(nouveau_wndw0) / sizeof(nouveau_wndw0[0]); i++)
        dmthd(wc, nouveau_wndw0[i][0], nouveau_wndw0[i][1]);
    // no semaphore to acquire before a flip: the assembly state isn't reset when the channel is allocated
    dmthd(wc, 0x218, 0);  // SET_CONTEXT_DMA_SEMAPHORE
    dmthd(wc, 0x20c, 0);  // SET_SEMAPHORE_CONTROL
    dmthd(wc, NVC57E_SET_PRESENT_CONTROL, NVV(NVC57E, SET_PRESENT_CONTROL_MIN_PRESENT_INTERVAL, 1));
    dmthd(wc, NVC57E_SET_SIZE, w | h << 16);
    dmthd(wc, NVC57E_SET_STORAGE, 0);  // pitch: the context DMA's kind says so
    dmthd(wc, NVC57E_SET_PARAMS, NVD(NVC57E, SET_PARAMS_FORMAT, A8R8G8B8));
    dmthd(wc, NVC57E_SET_PLANAR_STORAGE(0), pitch >> 6);
    dmthd(wc, NVC57E_SET_CONTEXT_DMA_ISO(0), FB_CTXDMA);
    dmthd(wc, NVC57E_SET_OFFSET(0), (uint32_t)(fb >> 8));
    dmthd(wc, NVC57E_SET_POINT_IN(0), 0);
    dmthd(wc, NVC57E_SET_SIZE_IN, w | h << 16);
    dmthd(wc, NVC57E_SET_SIZE_OUT, w | h << 16);
    dmthd(wc, 0x440, 2 << 2 | (LUT_ENTRIES + 4) << 8);  // ILUT: DIRECT10, no interpolation, size
    dmthd(wc, 0x444, FB_CTXDMA);
    dmthd(wc, 0x448, (uint32_t)(hd->luts >> 8));
    static const uint32_t identity[12] = { 0x10000, 0, 0, 0, 0, 0x10000, 0, 0, 0, 0, 0x10000, 0 };
    for (int i = 0; i < 12; i++)
        dmthd(wc, NVC57E_SET_FMT_COEFFICIENT_C00 + i * 4, identity[i]);
    dmthd(wc, NVC37E_SET_COMPOSITION_CONTROL, NVV(NVC37E, SET_COMPOSITION_CONTROL_DEPTH, 255));
    dmthd(wc, NVC37E_SET_COMPOSITION_CONSTANT_ALPHA, NVV(NVC37E, SET_COMPOSITION_CONSTANT_ALPHA_K1, 255));
    dmthd(wc, NVC37E_SET_COMPOSITION_FACTOR_SELECT,
          NVD(NVC37E, SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_MATCH_SELECT, K1) |
              NVD(NVC37E, SET_COMPOSITION_FACTOR_SELECT_SRC_COLOR_FACTOR_NO_MATCH_SELECT, K1) |
              NVD(NVC37E, SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_MATCH_SELECT, NEG_K1) |
              NVD(NVC37E, SET_COMPOSITION_FACTOR_SELECT_DST_COLOR_FACTOR_NO_MATCH_SELECT, NEG_K1));
    dmthd(&hd->wimm, NVC37B_SET_POINT_OUT(0), 0);

    vram_flush(gsp_gpu(d->s));  // RAMHT, context DMAs, LUTs and the framebuffer were written through PRAMIN
    // completion notifiers: the core's at +0, the window's at +0x40 * head + 0x10 or 0x20 (status in bits 31:30,
    // 2 = finished), cleared so a later modeset doesn't read the last one's. The window writes none where its last
    // update wrote one, so take the one the last flip didn't use
    uint32_t wn = hd->wndw_ntfy ^ 0x30;
    vram_wr32(gsp_gpu(d->s), d->ntfy, 0);
    vram_wr32(gsp_gpu(d->s), d->ntfy + wn, 0);
    vram_flush(gsp_gpu(d->s));
    dmthd(c, 0x208, NTFY_CTXDMA);
    dmthd(c, 0x20c, 0 << 4 | 1 << 12);
    dmthd(wc, 0x21c, NTFY_CTXDMA);
    dmthd(wc, 0x220, wn);
    hd->wndw_ntfy = wn;
    if (commit(d, head))
        return -1;
    uint32_t done = vram_rd32(gsp_gpu(d->s), d->ntfy);
    if (done >> 30 != 2) {
        log_msg("disp: modeset didn't complete (notifiers: core %#x window %#x)\n", done,
                vram_rd32(gsp_gpu(d->s), d->ntfy + wn));
        return -1;
    }
    log_msg("disp: head %d lit, showing %#llx (SOR%u pwr %#x)\n", head, (unsigned long long)fb, sor,
            rd32(gsp_gpu(d->s), 0x61c004 + sor * 0x800));
    return 0;
}

// shows fb (same size and pitch as the modeset's) from the next vblank on, window channel only (wndwc37e_image_set);
// returns the microseconds until the window's notifier said begun, or -1
int disp_flip_queue(struct disp *d, int head, uint64_t fb, uint32_t acquire, unsigned k)
{
    struct gpu *g = gsp_gpu(d->s);
    struct dchan *wc = &d->head[head].wndw;
    uint32_t slot = d->head[head].wndw_ntfy ^= 0x30;
    vram_wr32(g, d->ntfy + slot, 0);
    vram_flush(g);
    dmthd(wc, 0x21c, NTFY_CTXDMA);
    dmthd(wc, 0x220, (slot >> 4) << 4);
    // the window waits for the semaphore to read `acquire`, and leaves it so once flipped (wndwc37e_sema_set)
    dmthd(wc, NVC37E_SET_SEMAPHORE_CONTROL, DISP_FLIP_SEMA(head, k) >> 4);
    dmthd(wc, NVC37E_SET_SEMAPHORE_ACQUIRE, acquire);
    dmthd(wc, NVC37E_SET_SEMAPHORE_RELEASE, acquire);
    dmthd(wc, NVC37E_SET_CONTEXT_DMA_SEMAPHORE, NTFY_CTXDMA);
    dmthd(wc, NVC57E_SET_OFFSET(0), (uint32_t)(fb >> 8));
    dmthd(wc, NVC37E_SET_INTERLOCK_FLAGS, 0);
    dmthd(wc, NVC37E_SET_WINDOW_INTERLOCK_FLAGS, 0);
    dmthd(wc, NVC37E_UPDATE, 1);
    // not waited for: the channel holds the update until its semaphore and a vblank, and the caller has the lock
    __sync_synchronize();
    wr32(g, wc->user + 0x0, wc->cur * 4);
    return 0;
}

int disp_flip_taken(struct disp *d, int head)
{
    return vram_rd32(gsp_gpu(d->s), d->ntfy + d->head[head].wndw_ntfy) >> 30 != 0;
}

// written into the output LUT the head isn't reading, then the head pointed at it by a core update of its own
int disp_gamma(struct disp *d, int head, const uint16_t *rgb)
{
    struct gpu *g = gsp_gpu(d->s);
    struct disp_head *hd = &d->head[head];
    if (!hd->set_up)
        return -1;
    int next = hd->olut ^ 1;
    uint64_t at = hd->luts + LUT_SIZE * (1 + next);
    for (uint32_t i = 0; i < LUT_ENTRIES; i++) {
        const uint16_t *e = rgb + i * 3;
        vram_wr32(g, at + 0x20 + i * 8, e[0] | (uint32_t)e[1] << 16);
        vram_wr32(g, at + 0x20 + i * 8 + 4, e[2]);
    }
    vram_flush(g);
    dmthd(&d->core, 0x228c + head * 0x400, (uint32_t)(at >> 8));
    dmthd(&d->core, NVC37D_SET_INTERLOCK_FLAGS, 0);
    dmthd(&d->core, NVC37D_SET_WINDOW_INTERLOCK_FLAGS, 0);
    dmthd(&d->core, NVC37D_UPDATE, 1);
    if (dkick(d, &d->core))
        return -1;
    uint32_t st = 0;
    for (int us = 0; us < 100000; us += 100) {
        st = rd32(g, 0x610630);
        if ((st >> 16 & 0x1f) == 0xb && !(st >> 31)) {
            hd->olut = next;
            return 0;
        }
        delay_us(100);
    }
    log_msg("disp: output LUT update didn't finish: core %#x\n", st);
    return -1;
}

#define CURS_MAX 256  // W256_H256, the largest size the head takes
#define CURS_USER(head) (0x6d8000 + (head) * 0x1000)

// the head's cursor: a PIO channel for its position (no push buffer: nouveau's r535_curs_init) and room in VRAM
// for its image; after the first modeset
int disp_cursor_init(struct disp *d, int head)
{
    struct disp_head *hd = &d->head[head];
    if (hd->curs_img)
        return 0;
    const struct rm_client *in = gsp_internal(d->s);
    NV2080_CTRL_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER_PARAMS pb = {
        .hclass = gsp_gpu(d->s)->chip->curs, .channelInstance = (NvU32)head, .subDeviceId = 1,
    };
    if (!rm_control(d->s, in->client, in->subdevice, NV2080_CTRL_CMD_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER, &pb,
                    sizeof(pb)))
        return -1;
    NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS a = { .channelInstance = (NvV32)head };
    if (rm_alloc(d->s, d->c.client, DISP_ROOT, 0xc67a0000 | head, gsp_gpu(d->s)->chip->curs, &a, sizeof(a)))
        return -1;
    hd->curs_img = vram_alloc(gsp_vram(d->s), 2 * CURS_MAX * CURS_MAX * 4, 0x1000);
    // the memory isn't cleared: the first image written to each takes the whole plane
    hd->curs_w[0] = hd->curs_w[1] = hd->curs_h[0] = hd->curs_h[1] = CURS_MAX;
    return hd->curs_img ? 0 : -1;
}

struct curs_plane {
    struct gpu *g;
    uint64_t img;
    uint64_t window;  // the 64 KB of VRAM PRAMIN shows: set when a pixel is past it, not for every pixel
};

static void curs_put(void *ctx, uint32_t index, uint32_t px)
{
    struct curs_plane *p = ctx;
    uint64_t at = p->img + index * 4;
    if (at >> 16 != p->window) {
        p->window = at >> 16;
        wr32(p->g, 0x1700, p->window);
    }
    wr32(p->g, 0x700000 + (at & 0xffff), px);
}

// a w x h A8R8G8B8 cursor image of straight alpha; the head shows it at the next position update
int disp_cursor_image(struct disp *d, int head, const uint32_t *argb, uint32_t w, uint32_t h)
{
    struct gpu *g = gsp_gpu(d->s);
    struct disp_head *hd = &d->head[head];
    if (!hd->curs_img || !w || !h || w > CURS_MAX || h > CURS_MAX)
        return -1;
    // the head scans its image out all the time, and writing one through PRAMIN takes long enough to be caught half
    // done: the new image goes into the other one, then the head moves
    hd->curs_back ^= 1;
    int back = hd->curs_back;
    struct curs_plane plane = { g, hd->curs_img + (uint64_t)back * CURS_MAX * CURS_MAX * 4, ~0ull };
    // the plane is always the head's largest size, so the size class never changes together with the offset; the
    // image sits in its top left corner with clear pixels around it
    cursor_pack(curs_put, &plane, CURS_MAX, hd->curs_w[back], hd->curs_h[back], argb, w, h);
    hd->curs_w[back] = w;
    hd->curs_h[back] = h;
    vram_flush(g);
    struct dchan *c = &d->core;
    dmthd(c, NVC37D_HEAD_SET_CONTROL_CURSOR(head),
          NVD(NVC37D, HEAD_SET_CONTROL_CURSOR_ENABLE, ENABLE) | NVD(NVC37D, HEAD_SET_CONTROL_CURSOR_FORMAT, A8R8G8B8) |
              NVD(NVC37D, HEAD_SET_CONTROL_CURSOR_SIZE, W256_H256));
    dmthd(c, NVC37D_HEAD_SET_CONTROL_CURSOR_COMPOSITION(head),
          NVV(NVC37D, HEAD_SET_CONTROL_CURSOR_COMPOSITION_K1, 0xff) |
              NVD(NVC37D, HEAD_SET_CONTROL_CURSOR_COMPOSITION_CURSOR_COLOR_FACTOR_SELECT, K1) |
              NVD(NVC37D, HEAD_SET_CONTROL_CURSOR_COMPOSITION_VIEWPORT_COLOR_FACTOR_SELECT, NEG_K1_TIMES_SRC) |
              NVD(NVC37D, HEAD_SET_CONTROL_CURSOR_COMPOSITION_MODE, BLEND));
    dmthd(c, NVC37D_HEAD_SET_CONTEXT_DMA_CURSOR(head, 0), FB_CTXDMA);
    dmthd(c, NVC37D_HEAD_SET_OFFSET_CURSOR(head, 0), (uint32_t)(plane.img >> 8));
    return commit(d, head);
}

// the cursor's top left at (x, y) on the head, which may be off it; two register writes, safe without the lock.
// With the channel's FIFO full the move is dropped (nouveau's curs507a_space): the next one carries a newer position
void disp_cursor_move(struct disp *d, int head, int x, int y)
{
    struct gpu *g = gsp_gpu(d->s);
    if ((rd32(g, CURS_USER(head) + NVC37A_FREE) & 0x3f) < 4)
        return;
    wr32(g, CURS_USER(head) + NVC37A_SET_CURSOR_HOT_SPOT_POINT_OUT(0), (uint32_t)(y & 0xffff) << 16 | (x & 0xffff));
    wr32(g, CURS_USER(head) + NVC37A_UPDATE, 1);
}

// a head's vblank interrupt on or off (tu102_head_vblank_get/put); gsp_irq() reports and acks them
void disp_vblank(struct disp *d, int head, int on)
{
    struct gpu *g = gsp_gpu(d->s);
    uint32_t en = rd32(g, 0x611d80 + head * 4);
    if (on)
        wr32(g, 0x611800 + head * 4, 2);  // drop a stale one
    wr32(g, 0x611d80 + head * 4, on ? en | 2 : en & ~2u);
}

void disp_modeset_fini(struct disp *d)
{
    dchan_free(d, &d->core);
    d->core = (struct dchan){ 0 };
    if (d->ntfy)
        vram_free(gsp_vram(d->s), d->ntfy, 0x1000);
    d->ntfy = 0;
    for (int h = 0; h < (int)d->heads; h++) {
        struct disp_head *hd = &d->head[h];
        dchan_free(d, &hd->wimm);
        dchan_free(d, &hd->wndw);
        if (hd->luts)
            vram_free(gsp_vram(d->s), hd->luts, 3 * LUT_SIZE);
        if (hd->curs_img)
            vram_free(gsp_vram(d->s), hd->curs_img, 2 * CURS_MAX * CURS_MAX * 4);
        *hd = (struct disp_head){ 0 };
    }
    d->set_up = 0;
}

// after a modeset the display engine refused, its channels stay stuck: they go with their RM objects and memory,
// and the next disp_modeset() sets everything up afresh
void disp_reset(struct disp *d)
{
    if (!d->core.pb.cpu)
        return;
    rm_free(d->s, d->c.client, DISP_ROOT);
    disp_modeset_fini(d);
}

void disp_release_channels(struct gpu *g)
{
    uint32_t core = rd32(g, 0x6104e0), n = 0;
    // NV_PDISP_FE_CHNCTL_CORE, _WIN(32), _WINIM(32) from 0x6104e0, then _CURS(8) from 0x610604
    for (uint32_t i = 0; i <= 80; i++) {
        if (i > 64 && i < 73)
            continue;
        uint32_t ctl = rd32(g, 0x6104e0 + i * 4);
        if (!(ctl & 1))  // ALLOCATION
            continue;
        wr32(g, 0x6104e0 + i * 4, ctl & ~0x10u);  // PUTPTR_WRITE off, then the channel deallocated
        wr32(g, 0x6104e0 + i * 4, 0);
        n++;
    }
    uint32_t st = 0;
    for (int i = 0; i < 100; i++) {
        st = rd32(g, 0x610630);
        if (!(st >> 16 & 0x1f))  // NV_PDISP_FE_CHNSTATUS_CORE_STATE_DEALLOC
            break;
        delay_us(1000);
    }
    log_msg("disp: %u hardware channels deallocated, core control %#x, state now %#x\n", n, core, st);
}

// what the EFI firmware is showing (its window 0 is still armed before our modeset), copied into fb scaled to fit
// and centred, the rest black. -1 if the firmware's window doesn't look like a plain A8R8G8B8 surface.
int disp_copy_firmware(struct gpu *g, uint64_t fb, uint32_t pitch, uint32_t w, uint32_t h)
{
    uint32_t size = rd32(g, 0x690800 + 0x224), params = rd32(g, 0x690800 + 0x22c);
    uint32_t fw_pitch = rd32(g, 0x690800 + 0x230) << 6;
    uint64_t fw = (uint64_t)rd32(g, 0x690800 + 0x260) << 8;  // into the firmware's context DMA, at VRAM 0
    uint32_t fw_w = size & 0xffff, fw_h = size >> 16;
    if ((params & 0xff) != 0xcf || !fw_w || !fw_h || fw_pitch < fw_w * 4 || fw + (uint64_t)fw_pitch * fw_h > fb)
        return -1;
    // fit, keeping the aspect: dw x dh at (x0, y0)
    uint32_t dw = w, dh = (uint32_t)((uint64_t)fw_h * w / fw_w);
    if (dh > h) {
        dh = h;
        dw = (uint32_t)((uint64_t)fw_w * h / fw_h);
    }
    uint32_t x0 = (w - dw) / 2, y0 = (h - dh) / 2;
    log_msg("disp: keeping the firmware's %ux%u screen, at %ux%u\n", fw_w, fw_h, dw, dh);
    for (uint32_t y = 0; y < h; y++) {
        int row = y >= y0 && y < y0 + dh;
        uint64_t src = fw + (uint64_t)((y - y0) * fw_h / dh) * fw_pitch;
        for (uint32_t x = 0; x < w; x++) {
            uint32_t px = 0;
            if (row && x >= x0 && x < x0 + dw)
                px = vram_rd32(g, src + (x - x0) * fw_w / dw * 4);
            vram_wr32(g, fb + (uint64_t)y * pitch + x * 4, px | 0xff000000);
        }
    }
    return 0;
}

// colour bars with a white frame, so geometry and colour order are both visible
void disp_test_pattern(struct gpu *g, uint64_t fb, uint32_t pitch, uint32_t w, uint32_t h)
{
    static const uint32_t bars[8] = { 0xffffffff, 0xffffff00, 0xff00ffff, 0xff00ff00,
                                      0xffff00ff, 0xffff0000, 0xff0000ff, 0xff000000 };
    uint32_t window = ~0u;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint64_t a = fb + (uint64_t)y * pitch + x * 4;
            if ((uint32_t)(a >> 16) != window) {
                window = (uint32_t)(a >> 16);
                wr32(g, 0x1700, window);
            }
            int edge = x < 8 || y < 8 || x >= w - 8 || y >= h - 8;
            wr32(g, 0x700000 + (uint32_t)(a & 0xffff), edge ? 0xffffffff : bars[x * 8 / w]);
        }
    }
}
