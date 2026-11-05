// The display engine through GSP-RM, after nouveau's rm/r535/disp.c: instance memory, NV04_DISPLAY_COMMON,
// and what is connected where
#include "../nvrm/disp.h"
#include "../nvrm/gsp.h"
#include "disp.h"
#include "gsp.h"
#include "mem.h"

#define DISP_INST_SIZE 0x10000

static const char *or_name(uint32_t type)
{
    static const char *const names[] = { "none", "DAC", "SOR", "PIOR" };
    return type < 4 ? names[type] : "?";
}

// an 18-byte detailed timing descriptor, the same in the base block and in CEA-861 extensions
static void dtd(const uint8_t *b, struct disp_mode *t)
{
    t->clock = (b[0] | b[1] << 8) * 10;  // kHz
    t->hactive = b[2] | (b[4] & 0xf0) << 4;
    t->hblank = b[3] | (b[4] & 0x0f) << 8;
    t->vactive = b[5] | (b[7] & 0xf0) << 4;
    t->vblank = b[6] | (b[7] & 0x0f) << 8;
    t->hsync_off = b[8] | (b[11] & 0xc0) << 2;
    t->hsync_len = b[9] | (b[11] & 0x30) << 4;
    t->vsync_off = (b[10] >> 4) | (b[11] & 0x0c) << 2;
    t->vsync_len = (b[10] & 0xf) | (b[11] & 0x03) << 4;
    t->flags = b[17];
}

static void keep_mode(struct disp_output *o, struct disp_mode t)
{
    if (!t.hactive || !t.vactive || (t.flags & 0x80))  // interlaced: not for us
        return;
    for (uint32_t i = 0; i < o->nmodes; i++)
        if (!memcmp(&o->modes[i], &t, sizeof(t)))
            return;
    if (o->nmodes < DISP_MAX_MODES)
        o->modes[o->nmodes++] = t;
}

static void add_mode(struct disp_output *o, const uint8_t *b)
{
    struct disp_mode t = { 0 };  // padding included: modes are compared whole
    dtd(b, &t);
    keep_mode(o, t);
}

// a 20-byte DisplayID detailed timing (Type I in 1.x, clock in 10 kHz units; Type VII in 2.0, in 1 kHz), every
// field stored minus one; polarities in the top bits of the sync offsets
static void displayid_timing(struct disp_output *o, const uint8_t *b, uint32_t clock_unit)
{
#define F16(i) ((uint32_t)(b[i] | b[i + 1] << 8))
    struct disp_mode t = { 0 };
    t.clock = ((uint32_t)(b[0] | b[1] << 8 | b[2] << 16) + 1) * clock_unit;
    t.hactive = (uint16_t)(F16(4) + 1);
    t.hblank = (uint16_t)(F16(6) + 1);
    t.hsync_off = (uint16_t)((F16(8) & 0x7fff) + 1);
    t.hsync_len = (uint16_t)(F16(10) + 1);
    t.vactive = (uint16_t)(F16(12) + 1);
    t.vblank = (uint16_t)(F16(14) + 1);
    t.vsync_off = (uint16_t)((F16(16) & 0x7fff) + 1);
    t.vsync_len = (uint16_t)(F16(18) + 1);
    // as an EDID detailed timing's flags: digital separate sync, the polarities, interlace
    t.flags = 0x18 | (b[9] & 0x80 ? 0x2 : 0) | (b[17] & 0x80 ? 0x4 : 0) | (b[3] & 0x10 ? 0x80 : 0);
#undef F16
    keep_mode(o, t);
}

// a DisplayID section in an EDID extension block: data blocks of tag, revision, length, payload from byte 5
static void displayid(struct disp_output *o, const uint8_t *c)
{
    uint32_t end = 5 + c[2] < 127 ? 5 + c[2] : 127;
    for (uint32_t p = 5; p + 3 <= end;) {
        uint8_t tag = c[p], len = c[p + 2];
        if (p + 3 + len > end)
            break;
        uint32_t unit = tag == 0x03 ? 10 : tag == 0x22 ? 1 : 0;  // Type I / Type VII detailed timings
        for (uint32_t q = 0; unit && q + 20 <= len; q += 20)
            displayid_timing(o, c + p + 3 + q, unit);
        p += 3 + len;
    }
}

// a progressive mode with CVT's reduced blanking (drm_cvt_mode); positive hsync, negative vsync
static struct disp_mode cvt_rb_mode(uint32_t w, uint32_t h, uint32_t hz, uint32_t vsync)
{
    uint32_t hperiod = (1000000000 / hz - 460000) / h;  // ns
    uint32_t vblank = 460000 / hperiod + 1;
    if (vblank < 3 + vsync + 6)
        vblank = 3 + vsync + 6;
    struct disp_mode t = { 0 };
    t.clock = (w + 160) * (h + vblank) * hz / 1000;  // no CVT 250 kHz step: it turns 75 Hz into 74.97
    t.hactive = (uint16_t)w;
    t.hblank = 160;
    t.hsync_off = 48;
    t.hsync_len = 32;
    t.vactive = (uint16_t)h;
    t.vblank = (uint16_t)vblank;
    t.vsync_off = 3;
    t.vsync_len = (uint16_t)vsync;
    t.flags = 0x1a;
    return t;
}

// a progressive mode by the default GTF, in integers as drm_gtf_mode_complex has it, and clocked for the exact rate as
// NVIDIA has it; negative hsync, positive vsync
static struct disp_mode gtf_mode(uint32_t w, uint32_t h, uint32_t hz)
{
    uint32_t hfreq = (h + 1) * 2 * 1000 * hz / ((1000000 - 550 * hz) / 500);  // Hz
    uint32_t sync_bp = (550 * hfreq / 1000 + 500) / 1000;                     // lines of vsync and back porch
    uint32_t duty = 30000 - 300 * 1000000 / hfreq;                            // blanking, in thousandths of a percent
    uint32_t hblank = (w * duty / (100000 - duty) + 8) / 16 * 16;
    uint32_t total = w + hblank, hsync = (8 * total / 100 + 4) / 8 * 8;
    struct disp_mode t = { 0 };
    t.clock = total * (h + sync_bp + 1) * hz / 1000;
    t.hactive = (uint16_t)w;
    t.hblank = (uint16_t)hblank;
    t.hsync_off = (uint16_t)(hblank / 2 - hsync);
    t.hsync_len = (uint16_t)hsync;
    t.vactive = (uint16_t)h;
    t.vblank = (uint16_t)(sync_bp + 1);
    t.vsync_off = 1;
    t.vsync_len = 3;
    t.flags = 0x1c;
    return t;
}

// a standard timing at the preferred size faster than any detailed one: where a 75 Hz monitor may keep its top rate.
// Before EDID 1.4 one outside DMT means GTF (parseEdidStandardTimingDescriptor); the other sizes stay out of the list
static void add_standard(struct disp_output *o, const uint8_t *e)
{
    // 16:10, 4:3, 5:4, 16:9
    static const uint8_t num[] = { 5, 3, 4, 9 }, den[] = { 8, 4, 5, 16 }, vsync[] = { 6, 4, 7, 5 };
    if (!o->nmodes)
        return;
    uint32_t w = o->modes[0].hactive, h = o->modes[0].vactive, top = 0;
    for (uint32_t i = 0; i < o->nmodes; i++) {
        const struct disp_mode *m = &o->modes[i];
        uint32_t lines = (uint32_t)(m->hactive + m->hblank) * (m->vactive + m->vblank);
        uint32_t rate = (m->clock * 1000 + lines / 2) / lines;
        if (m->hactive == w && m->vactive == h && rate > top)
            top = rate;
    }
    for (int i = 38; i < 54; i += 2) {
        uint32_t hz = (e[i + 1] & 0x3f) + 60, width = (e[i] + 31) * 8, aspect = e[i + 1] >> 6;
        if (e[i] > 1 && width == w && width * num[aspect] / den[aspect] == h && hz > top)
            keep_mode(o, e[19] >= 4 ? cvt_rb_mode(w, h, hz, vsync[aspect]) : gtf_mode(w, h, hz));
    }
}

// every refresh rate of the panel's own size again for each image that suits it: bigger, of the same shape, and no
// more than the twice the 2-tap scaler takes (NV_EVO_SCALER_2TAPS)
static void add_scaled(struct disp_output *o, const struct disp_size *sizes, uint32_t n)
{
    if (!o->nmodes)
        return;
    uint32_t w = o->modes[0].hactive, h = o->modes[0].vactive, real = o->nmodes;
    for (uint32_t s = 0; s < n; s++) {
        if (sizes[s].w <= w || sizes[s].w > 2 * w || sizes[s].w * h != sizes[s].h * w)
            continue;
        for (uint32_t i = 0; i < real; i++) {
            struct disp_mode t = o->modes[i];
            if (t.hactive != w || t.vactive != h)
                continue;
            t.view_w = (uint16_t)sizes[s].w;
            t.view_h = (uint16_t)sizes[s].h;
            keep_mode(o, t);
        }
    }
}

void disp_set_scaled(struct disp *d, const struct disp_size *sizes, uint32_t n)
{
    d->nscaled = n;
    memcpy(d->scaled, sizes, n * sizeof(*sizes));
    for (int i = 0; i < d->n; i++) {
        struct disp_output *o = &d->out[i];
        while (o->nmodes && o->modes[o->nmodes - 1].view_w)
            o->nmodes--;
        add_scaled(o, sizes, n);
    }
}

// manufacturer, name and the detailed timings out of an EDID: the base block's (the first is the preferred mode)
// and those of CEA-861 extension blocks
static void edid_describe(const uint8_t *e, uint32_t size, struct disp_output *o)
{
    if (size < 128 || e[0] != 0 || e[1] != 0xff) {
        memcpy(o->name, "no EDID", 8);
        return;
    }
    uint16_t m = e[8] << 8 | e[9];
    char name[14] = "";
    o->nmodes = 0;
    for (int d = 54; d < 126; d += 18) {
        const uint8_t *b = e + d;
        if (b[0] || b[1]) {
            add_mode(o, b);
        } else if (b[3] == 0xfc) {  // monitor name
            for (int i = 0; i < 13 && b[5 + i] != '\n'; i++)
                name[i] = (char)b[5 + i];
        }
    }
    for (uint32_t x = 128; x + 128 <= size; x += 128) {
        const uint8_t *c = e + x;
        if (c[0] == 0x70) {  // DisplayID: where high refresh rates usually are
            displayid(o, c);
            continue;
        }
        if (c[0] != 0x02 || c[2] < 4)  // CEA-861: detailed timings from byte c[2] on
            continue;
        // data blocks before them: high refresh rates may be in a DisplayID Type VII timing block (extended tag 0x22)
        for (uint32_t i = 4, len; i < c[2] && i < 127; i += 1 + len) {
            len = c[i] & 0x1f;
            if (c[i] >> 5 != 7 || len < 2 + 20 || i + 1 + len > c[2] || c[i + 1] != 0x22 || (c[i + 2] & 0x70))
                continue;
            for (uint32_t q = i + 3; q + 20 <= i + 1 + len; q += 20)
                displayid_timing(o, c + q, 1);
        }
        for (uint32_t d = c[2]; d + 18 <= 127 && (c[d] || c[d + 1]); d += 18)
            add_mode(o, c + d);
    }
    add_standard(o, e);
    if (o->nmodes)
        o->mode = o->modes[0];
    o->name[0] = (char)('@' + ((m >> 10) & 0x1f));
    o->name[1] = (char)('@' + ((m >> 5) & 0x1f));
    o->name[2] = (char)('@' + (m & 0x1f));
    o->name[3] = ' ';
    memcpy(o->name + 4, name, sizeof(name));
}

// message processing: only note which displays changed, the RPCs to look at them come later
static void hotplug_event(void *ctx, uint32_t client, uint32_t event, const void *data, uint32_t size)
{
    struct disp *d = ctx;
    if (client != d->c.client)
        return;
    if (event == DISP_DP_IRQ && size >= sizeof(Nv2080DpIrqNotification)) {
        const Nv2080DpIrqNotification *irq = data;
        log_msg("disp: DisplayPort IRQ_HPD from %#x\n", irq->displayId);
        d->dp_irq |= irq->displayId;
        return;
    }
    const Nv2080HotplugNotification *h = data;
    if (event != DISP_HOTPLUG || size < sizeof(*h))
        return;
    log_msg("disp: hotplug: plug %#x unplug %#x\n", h->plugDisplayMask, h->unplugDisplayMask);
    d->hotplug |= h->plugDisplayMask | h->unplugDisplayMask;
}

// an event object under our subdevice that GSP-RM posts notifier index's events to, again and again
static int listen(struct disp *d, uint32_t handle, uint32_t index)
{
    NV0005_ALLOC_PARAMETERS ev = {
        .hParentClient = d->c.client,
        .hClass = NV01_EVENT_KERNEL_CALLBACK_EX,
        .notifyIndex = NV01_EVENT_CLIENT_RM | index,
    };
    NV2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS sn = {
        .event = index,
        .action = NV2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT,
    };
    if (rm_alloc(d->s, d->c.client, d->c.subdevice, handle, NV01_EVENT_KERNEL_CALLBACK_EX, &ev, sizeof(ev)))
        return -1;
    void *r = rm_control(d->s, d->c.client, d->c.subdevice, NV2080_CTRL_CMD_EVENT_SET_NOTIFICATION, &sn, sizeof(sn));
    return r ? 0 : -1;
}

int disp_probe(struct gsp *s, struct disp *d, const struct disp_size *scaled, uint32_t nscaled)
{
    const struct rm_client *in = gsp_internal(s);
    *d = (struct disp){ .s = s, .nscaled = nscaled };
    memcpy(d->scaled, scaled, nscaled * sizeof(*scaled));

    // the display engine's instance memory: RAMHT and channel contexts live here
    d->inst = vram_alloc(gsp_vram(s), DISP_INST_SIZE, DISP_INST_SIZE);
    if (!d->inst)
        return -1;
    vram_zero(gsp_gpu(s), d->inst, DISP_INST_SIZE);
    NV2080_CTRL_INTERNAL_DISPLAY_WRITE_INST_MEM_PARAMS im = {
        .instMemPhysAddr = d->inst, .instMemSize = DISP_INST_SIZE,
        .instMemAddrSpace = 2,     // ADDR_FBMEM
        .instMemCpuCacheAttr = 2,  // NV_MEMORY_WRITECOMBINED
    };
    if (!rm_control(s, in->client, in->subdevice, NV2080_CTRL_CMD_INTERNAL_DISPLAY_WRITE_INST_MEM, &im, sizeof(im)))
        return -1;
    struct gpu *g = gsp_gpu(s);
    log_msg("disp: ownership %#x, core channel state %#x control %#x, capabilities %#x %#x %#x\n", rd32(g, 0x6254e8),
            rd32(g, 0x610630), rd32(g, 0x6104e0), rd32(g, 0x640000), rd32(g, 0x640004), rd32(g, 0x64000c));

    if (rm_client(s, &d->c) || rm_alloc(s, d->c.client, d->c.device, DISP_COMMON, NV04_DISPLAY_COMMON, NULL, 0))
        return -1;

    NV2080_CTRL_INTERNAL_DISPLAY_GET_STATIC_INFO_PARAMS *si =
        rm_control(s, in->client, in->subdevice, NV2080_CTRL_CMD_INTERNAL_DISPLAY_GET_STATIC_INFO, NULL, 0);
    if (!si) {
        NV2080_CTRL_INTERNAL_DISPLAY_GET_STATIC_INFO_PARAMS *req = mem_alloc(sizeof(*req));
        si = rm_control(s, in->client, in->subdevice, NV2080_CTRL_CMD_INTERNAL_DISPLAY_GET_STATIC_INFO, req,
                        sizeof(*req));
        mem_free(req);
    }
    if (si)
        d->windows = si->windowPresentMask;

    // no ACPI backlight data, but GSP-RM waits to be told so before it finishes loading display state
    NV2080_CTRL_INTERNAL_INIT_BRIGHTC_STATE_LOAD_PARAMS *bl = mem_alloc(sizeof(*bl));
    bl->status = 0x56;  // NV_ERR_NOT_SUPPORTED
    rm_control(s, in->client, in->subdevice, NV2080_CTRL_CMD_INTERNAL_INIT_BRIGHTC_STATE_LOAD, bl, sizeof(*bl));
    mem_free(bl);

    NV0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT_PARAMS man = { 0 };
    rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT, &man, sizeof(man));

    NV0073_CTRL_SYSTEM_GET_NUM_HEADS_PARAMS nh = { 0 };
    NV0073_CTRL_SYSTEM_GET_NUM_HEADS_PARAMS *h =
        rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_SYSTEM_GET_NUM_HEADS, &nh, sizeof(nh));
    d->heads = h ? h->numHeads : 0;
    if (d->heads > DISP_MAX_HEADS)
        d->heads = DISP_MAX_HEADS;

    // what the EFI firmware lit where, so its screen can stay on the head that shows it
    for (uint32_t i = 0; i < d->heads; i++) {
        NV0073_CTRL_SYSTEM_GET_ACTIVE_PARAMS aq = { .head = i };
        NV0073_CTRL_SYSTEM_GET_ACTIVE_PARAMS *ar =
            rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_SYSTEM_GET_ACTIVE, &aq, sizeof(aq));
        d->fw_display[i] = ar ? ar->displayId : 0;
        if (d->fw_display[i])
            log_msg("disp: head %u: the firmware lit display %#x\n", i, d->fw_display[i]);
    }

    // the GPU's DisplayPort limits: its fastest link rate, and which SST watermark limits its SORs take
    NV0073_CTRL_CMD_DP_GET_CAPS_PARAMS cq = { .sorIndex = ~0u };
    NV0073_CTRL_CMD_DP_GET_CAPS_PARAMS *cr =
        rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DP_GET_CAPS, &cq, sizeof(cq));
    if (cr) {
        static const uint8_t bw[] = { 0, 0x06, 0x0a, 0x14, 0x1e };  // NV0073_CTRL_CMD_DP_GET_CAPS_MAX_LINK_RATE_*
        uint32_t rate = cr->maxLinkRate & 7;  // NV0073_CTRL_CMD_DP_GET_CAPS_MAX_LINK_RATE, 2:0
        d->dp_max_bw = rate < sizeof(bw) ? bw[rate] : 0;
        d->dp_wm_increased = cr->bHasIncreasedWatermarkLimits;
        log_msg("disp: DisplayPort up to %u Mbps per lane%s%s\n", d->dp_max_bw * 270,
                cr->bIsMultistreamSupported ? ", MST" : "", cr->bHasIncreasedWatermarkLimits ? ", increased watermarks" : "");
    }

    // hotplug and DisplayPort IRQ_HPD: GSP-RM posts them to event objects under our subdevice
    if (listen(d, DISP_HOTPLUG, NV2080_NOTIFIERS_HOTPLUG))
        log_msg("disp: no hotplug events\n");
    if (listen(d, DISP_DP_IRQ, NV2080_NOTIFIERS_DP_IRQ))
        log_msg("disp: no DisplayPort IRQ_HPD events\n");
    gsp_on_event(s, hotplug_event, d);

    disp_read_outputs(d);
    return 0;
}

// a DisplayPort sink's modes its fastest link can't carry left out, so neither the first lighting nor macOS picks one
static void drop_unfit_modes(const struct disp *d, struct disp_output *o, int quiet)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < o->nmodes; i++) {
        const struct disp_mode *t = &o->modes[i];
        if (disp_dp_fits(d, o, t))
            o->modes[n++] = *t;
        else if (!quiet)
            log_msg("disp: display %#x: %ux%u at %u kHz left out, more than %u x %u Mbps carries\n", o->id,
                    t->hactive, t->vactive, t->clock, o->dp_lanes, o->dp_bw * 270);
    }
    o->nmodes = n;
    o->mode = n ? o->modes[0] : (struct disp_mode){ 0 };
}

void disp_read_outputs(struct disp *d)
{
    struct gsp *s = d->s;
    NV0073_CTRL_SYSTEM_GET_SUPPORTED_PARAMS sup = { 0 };
    NV0073_CTRL_SYSTEM_GET_SUPPORTED_PARAMS *sp =
        rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_SYSTEM_GET_SUPPORTED, &sup, sizeof(sup));
    uint32_t supported = sp ? sp->displayMask : 0;

    NV0073_CTRL_SYSTEM_GET_CONNECT_STATE_PARAMS cs = { .displayMask = supported };
    NV0073_CTRL_SYSTEM_GET_CONNECT_STATE_PARAMS *cp =
        rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_SYSTEM_GET_CONNECT_STATE, &cs, sizeof(cs));
    uint32_t connected = cp ? cp->displayMask : 0;
    int quiet = d->n && connected == d->connected;  // asked again with nothing changed: no need to say it all
    d->connected = connected;
    if (!quiet)
        log_msg("disp: %u heads, windows %#x, displays %#x, connected %#x\n", d->heads, d->windows, supported,
                connected);

    d->n = 0;
    for (int i = 0; i < 32 && d->n < DISP_MAX_OUTPUTS; i++) {
        uint32_t id = 1u << i;
        if (!(supported & id))
            continue;
        struct disp_output *o = &d->out[d->n++];
        *o = (struct disp_output){ .id = id, .connected = !!(connected & id) };
        NV0073_CTRL_SPECIFIC_OR_GET_INFO_PARAMS orq = { .displayId = id };
        NV0073_CTRL_SPECIFIC_OR_GET_INFO_PARAMS *ori =
            rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_SPECIFIC_OR_GET_INFO, &orq, sizeof(orq));
        if (ori) {
            o->or_type = ori->type;
            o->or_index = ori->index;
            o->protocol = ori->protocol;
        }
        o->dp = o->protocol == NV0073_CTRL_SPECIFIC_OR_PROTOCOL_SOR_DP_A ||
                o->protocol == NV0073_CTRL_SPECIFIC_OR_PROTOCOL_SOR_DP_B;
        if (o->connected) {
            // the signal the connector carries is GSP-RM's to say; the protocol is a guess before a SOR is assigned
            NV0073_CTRL_DFP_GET_INFO_PARAMS fq = { .displayId = id };
            NV0073_CTRL_DFP_GET_INFO_PARAMS *fi =
                rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DFP_GET_INFO, &fq, sizeof(fq));
            if (fi)
                o->dp = (fi->flags & 7) == NV0073_CTRL_DFP_FLAGS_SIGNAL_DISPLAYPORT;
            if (o->dp)
                disp_dp_caps(d, o, quiet);
            NV0073_CTRL_SPECIFIC_GET_EDID_V2_PARAMS *eq = mem_alloc(sizeof(*eq));
            eq->displayId = id;
            eq->bufferSize = sizeof(eq->edidBuffer);
            NV0073_CTRL_SPECIFIC_GET_EDID_V2_PARAMS *er =
                rm_control(s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_SPECIFIC_GET_EDID_V2, eq, sizeof(*eq));
            uint32_t size = er ? er->bufferSize : 0;
            const uint8_t *e = er ? er->edidBuffer : NULL;
            // GSP-RM may leave a DisplayPort sink's EDID to the DisplayPort library, which reads it over AUX
            if (o->dp && (size < 128 || e[0] || e[1] != 0xff)) {
                size = disp_dp_edid(d, id, eq->edidBuffer, sizeof(eq->edidBuffer));
                e = eq->edidBuffer;
            }
            if (e) {
                edid_describe(e, size, o);
                add_scaled(o, d->scaled, d->nscaled);
                o->edid_size = size < sizeof(o->edid) ? size : sizeof(o->edid);
                memcpy(o->edid, e, o->edid_size);
            }
            mem_free(eq);
            if (o->dp_lanes)  // not when AUX is silent, as behind a DP++ adapter to HDMI, which runs as TMDS
                drop_unfit_modes(d, o, quiet);
        }
        if (quiet)
            continue;
        log_msg("disp: display %#x: %s%u protocol %u%s%s%s%s, %ux%u %u kHz\n", id, or_name(o->or_type), o->or_index,
                o->protocol, o->dp ? " (DisplayPort)" : "", o->connected ? ", connected" : "", o->connected ? ": " : "",
                o->connected ? o->name : "", o->mode.hactive, o->mode.vactive, o->mode.clock);
        for (uint32_t i = 0; i < o->nmodes; i++) {
            const struct disp_mode *t = &o->modes[i];
            uint64_t total = (uint64_t)(t->hactive + t->hblank) * (t->vactive + t->vblank);
            uint64_t centi = total ? (uint64_t)t->clock * 100000 / total : 0;  // Hz x 100
            log_msg("disp:   mode %u: %ux%u at %u.%02u Hz (%u kHz)\n", i, mode_w(t), mode_h(t), (unsigned)(centi / 100),
                    (unsigned)(centi % 100), t->clock);
        }
    }
}

void disp_fini(struct disp *d)
{
    gsp_on_event(d->s, NULL, NULL);
    if (d->c.client)
        rm_free(d->s, d->c.client, d->c.client);
    if (d->inst)
        vram_free(gsp_vram(d->s), d->inst, DISP_INST_SIZE);
}

void disp_forget(struct disp *d)
{
    gsp_on_event(d->s, NULL, NULL);
    disp_modeset_fini(d);
    if (d->inst)
        vram_free(gsp_vram(d->s), d->inst, DISP_INST_SIZE);
}
