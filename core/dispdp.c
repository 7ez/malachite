// DisplayPort through GSP-RM, after nouveau's r535 DP code and NVIDIA's DisplayPort library: AUX transfers, the
// sink's capabilities and EDID, the smallest link that carries a mode, its training, and the SST stream's
// watermark and blanking symbols. GSP-RM does the training itself (NV0073_CTRL_CMD_DP_CTRL); single stream only.
#include "../nvrm/disp.h"
#include "../nvrm/gsp.h"
#include "disp.h"
#include "gsp.h"
#include "mem.h"

#define DRF_LO(f) (0 ? f)
#define DRF_HI(f) (1 ? f)
#define NVV(d, f, v) ((((uint32_t)(v)) & (uint32_t)((1ull << (DRF_HI(d##_##f) - DRF_LO(d##_##f) + 1)) - 1)) << DRF_LO(d##_##f))
#define NVD(d, f, c) NVV(d, f, d##_##f##_##c)

static const uint8_t dp_rates[] = { 0x06, 0x0a, 0x14, 0x1e };  // RBR, HBR, HBR2, HBR3, in 0.27 Gbps

int dp_aux_read(struct disp *d, uint32_t id, uint32_t addr, uint8_t *buf, uint32_t size)
{
    NV0073_CTRL_DP_AUXCH_CTRL_PARAMS a = {
        .displayId = id,
        .cmd = NVD(NV0073_CTRL, DP_AUXCH_CMD_TYPE, AUX) | NVD(NV0073_CTRL, DP_AUXCH_CMD_REQ_TYPE, READ),
        .addr = addr,
        .size = size - 1,
    };
    NV0073_CTRL_DP_AUXCH_CTRL_PARAMS *r =
        rm_control(d->s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DP_AUXCH_CTRL, &a, sizeof(a));
    if (!r)
        return -1;
    memcpy(buf, r->data, size);
    return 0;
}

int dp_aux_write(struct disp *d, uint32_t id, uint32_t addr, const uint8_t *buf, uint32_t size)
{
    NV0073_CTRL_DP_AUXCH_CTRL_PARAMS a = {
        .displayId = id,
        .cmd = NVD(NV0073_CTRL, DP_AUXCH_CMD_TYPE, AUX) | NVD(NV0073_CTRL, DP_AUXCH_CMD_REQ_TYPE, WRITE),
        .addr = addr,
        .size = size - 1,
    };
    memcpy(a.data, buf, size);
    return rm_control(d->s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DP_AUXCH_CTRL, &a, sizeof(a)) ? 0 : -1;
}

// one I2C-over-AUX transfer on the sink's DDC bus; mot leaves the I2C transaction open, no bytes is a bare stop.
// Reads return how many bytes came back, writes 0.
static int aux_i2c(struct disp *d, uint32_t id, uint32_t addr, int read, int mot, uint8_t *buf, uint32_t size)
{
    NV0073_CTRL_DP_AUXCH_CTRL_PARAMS a = {
        .displayId = id,
        .bAddrOnly = !size,
        .cmd = NVD(NV0073_CTRL, DP_AUXCH_CMD_TYPE, I2C) | NVV(NV0073_CTRL, DP_AUXCH_CMD_I2C_MOT, mot) |
               (read ? NVD(NV0073_CTRL, DP_AUXCH_CMD_REQ_TYPE, READ) : NVD(NV0073_CTRL, DP_AUXCH_CMD_REQ_TYPE, WRITE)),
        .addr = addr,
        .size = size ? size - 1 : 0,
    };
    if (!read && size)
        memcpy(a.data, buf, size);
    NV0073_CTRL_DP_AUXCH_CTRL_PARAMS *r =
        rm_control(d->s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DP_AUXCH_CTRL, &a, sizeof(a));
    if (!r)
        return -1;
    if (!read)
        return 0;
    uint32_t n = r->size < size ? r->size : size;
    memcpy(buf, r->data, n);
    return (int)n;
}

// EDID block b: E-DDC segment b / 2 (pointer at 0x30), then 16 bytes at a time from DDC address 0x50, the offset
// written before each read so a short reply can't leave us out of step with the sink's
static int edid_block(struct disp *d, uint32_t id, uint32_t b, uint8_t *out)
{
    uint8_t seg = (uint8_t)(b / 2), sum = 0;
    for (uint32_t off = 0; off < 128;) {
        uint8_t at = (uint8_t)((b & 1) * 128 + off);
        int n = 0;
        for (int tries = 0; tries < 3 && n <= 0; tries++) {
            if ((!seg || !aux_i2c(d, id, 0x30, 0, 1, &seg, 1)) && !aux_i2c(d, id, 0x50, 0, 1, &at, 1))
                n = aux_i2c(d, id, 0x50, 1, 1, out + off, 128 - off < 16 ? 128 - off : 16);
            aux_i2c(d, id, 0x50, 0, 0, NULL, 0);
        }
        if (n <= 0)
            return -1;
        off += (uint32_t)n;
    }
    for (int i = 0; i < 128; i++)
        sum += out[i];
    return sum ? -1 : 0;
}

uint32_t disp_dp_edid(struct disp *d, uint32_t id, uint8_t *buf, uint32_t size)
{
    uint32_t blocks = 1, b;
    for (b = 0; b < blocks && (b + 1) * 128 <= size; b++) {
        if (edid_block(d, id, b, buf + b * 128))
            break;
        if (!b)
            blocks = 1u + buf[126];
    }
    log_msg("disp: display %#x: EDID over AUX, %u of %u blocks\n", id, b, blocks);
    return b * 128;
}

int disp_dp_caps(struct disp *d, struct disp_output *o, int quiet)
{
    uint8_t *c = o->dpcd, ext[16];
    o->dp_bw = o->dp_lanes = 0;
    if (dp_aux_read(d, o->id, 0, c, 16))
        return -1;
    // DP 1.4 sinks keep DP 1.2 capabilities here for old sources, the real ones (HBR3) at 0x2200
    if ((c[0x0e] & 0x80) && !dp_aux_read(d, o->id, 0x2200, ext, sizeof(ext)) && ext[1] >= c[1])
        memcpy(c, ext, sizeof(ext));
    uint32_t max = d->dp_max_bw && d->dp_max_bw < c[1] ? d->dp_max_bw : c[1], lanes = c[2] & 0x1f;
    for (unsigned i = 0; i < sizeof(dp_rates) && dp_rates[i] <= max; i++)
        o->dp_bw = dp_rates[i];
    o->dp_lanes = lanes >= 4 ? 4 : lanes >= 2 ? 2 : lanes;
    if (!quiet)
        log_msg("disp: display %#x: DPCD %u.%u, sink %u lanes up to %u Mbps%s%s%s%s; links up to %u x %u Mbps\n",
                o->id, c[0] >> 4, c[0] & 0xf, lanes, c[1] * 270, c[2] & 0x80 ? ", enhanced framing" : "",
                c[2] & 0x40 ? ", TPS3" : "", c[3] & 0x80 ? ", TPS4" : "", c[3] & 1 ? ", downspread" : "",
                o->dp_lanes, o->dp_bw * 270);
    return o->dp_bw && o->dp_lanes ? 0 : -1;
}

struct dp_stream {
    uint32_t watermark, hblank_sym, vblank_sym;
};

#define PREC 100000ull  // the fixed point NVIDIA's DisplayPort library works the stream out in

// the SST stream for mode t at 24 bpp on lanes x bw, with 64-symbol transfer units: the watermark, and the
// symbols left in horizontal and vertical blanking for secondary data (the library's and nouveau's
// nv50_sor_dp_watermark_sst arithmetic); -1 if the link can't carry the mode
static int dp_stream(const struct disp *d, const struct disp_mode *t, uint32_t bw, uint32_t lanes, int ef,
                     struct dp_stream *st)
{
    const uint64_t depth = 24, link = (uint64_t)bw * 27000000, pclk = (uint64_t)t->clock * 1000;
    const uint64_t w = t->hactive, htotal = (uint64_t)t->hactive + t->hblank;
    const uint64_t adjust = d->dp_wm_increased ? 8 : 2, least = d->dp_wm_increased ? 22 : 20;
    // the link is 8 bits per symbol per lane, and down-spreading slows it by up to 0.5%
    if (!pclk || w <= 60 || pclk * depth * 1000 > link * 8 * lanes * 995)
        return -1;
    uint64_t ratio = pclk * depth * PREC / (link * 8 * lanes);
    uint64_t wm = adjust + (2 * (depth * PREC / (8 * lanes)) + ratio * 64 * (PREC - ratio) / PREC) / PREC;
    if (wm > 39 || wm > w * depth / (8 * lanes))
        return -1;
    if (wm < least)
        wm = least;
    // blanking start and end (twice with enhanced framing), VB-ID, Mvid and Maud four times each, and the
    // padding when the active width doesn't split evenly over the lanes
    uint64_t bits = 3 * 8 * lanes * (ef ? 2 : 1) + 3 * 8 * 4 + (w % lanes ? (lanes - w % lanes) * depth : 0);
    uint64_t min_hblank = bits * PREC / (8 * lanes) * pclk / link / PREC + 12;
    if (min_hblank > htotal - w)
        return -1;
    // less the stuffer's and the secondary packet path's latencies
    int64_t hsym = (int64_t)((htotal - w - min_hblank) * link / pclk) - 4 - (lanes == 1 ? 9 : lanes == 2 ? 6 : 3);
    int64_t vsym = (int64_t)((w - 40) * link / pclk) - 1 - (lanes == 1 ? 39 : lanes == 2 ? 21 : 12);
    st->watermark = (uint32_t)wm;
    st->hblank_sym = hsym > 0 ? (uint32_t)hsym : 0;
    st->vblank_sym = vsym > 0 ? (uint32_t)vsym : 0;
    return 0;
}

int disp_dp_fits(const struct disp *d, const struct disp_output *o, const struct disp_mode *t)
{
    struct dp_stream st;
    return o->dp_lanes && !dp_stream(d, t, o->dp_bw, o->dp_lanes, o->dpcd[2] >> 7, &st);
}

// DPCD lane status (0x202, 0x203) and alignment (0x204): clock recovery, equalisation and symbol lock on every lane
static int lanes_ok(const uint8_t *st, uint32_t lanes)
{
    for (uint32_t l = 0; l < lanes; l++)
        if ((st[l / 2] >> (l % 2 * 4) & 7) != 7)
            return 0;
    return st[2] & 1;
}

static int dp_train(struct disp *d, const struct disp_output *o, uint32_t bw, uint32_t lanes, int ef)
{
    NV0073_CTRL_DP_CTRL_PARAMS t = {
        .displayId = o->id,
        .cmd = NVD(NV0073_CTRL, DP_CMD_SET_LANE_COUNT, TRUE) | NVD(NV0073_CTRL, DP_CMD_SET_LINK_BW, TRUE) |
               NVD(NV0073_CTRL, DP_CMD_TRAIN_PHY_REPEATER, YES) |
               (ef ? NVD(NV0073_CTRL, DP_CMD_SET_ENHANCED_FRAMING, TRUE) : 0),
        .data = NVV(NV0073_CTRL, DP_DATA_SET_LANE_COUNT, lanes) | NVV(NV0073_CTRL, DP_DATA_SET_LINK_BW, bw) |
                NVD(NV0073_CTRL, DP_DATA_TARGET, SINK),
    };
    NV0073_CTRL_DP_CTRL_PARAMS *r = rm_control(d->s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DP_CTRL, &t, sizeof(t));
    if (!r || r->err) {
        log_msg("disp: display %#x: training %u lanes at %u Mbps failed (err %#x)\n", o->id, lanes, bw * 270,
                r ? r->err : ~0u);
        return -1;
    }
    uint8_t st[3] = { 0 };
    dp_aux_read(d, o->id, 0x202, st, sizeof(st));
    log_msg("disp: display %#x: link trained, %u lanes at %u Mbps%s, lane status %02x %02x align %02x%s\n", o->id,
            lanes, bw * 270, ef ? ", enhanced framing" : "", st[0], st[1], st[2],
            lanes_ok(st, lanes) ? "" : " (sink not locked)");
    return 0;
}

// most lanes at the lowest rate first, as nouveau picks; a config that fails to train gives way to the next
int disp_dp_link(struct disp *d, int head, uint32_t sor, struct disp_output *o, const struct disp_mode *t)
{
    struct disp_head *hd = &d->head[head];
    hd->dp_lanes = 0;
    if (disp_dp_caps(d, o, 0)) {
        log_msg("disp: display %#x: no DPCD over AUX\n", o->id);
        return -1;
    }
    int ef = o->dpcd[2] >> 7;
    log_msg("disp: display %#x: %ux%u at %u kHz needs %u Mbps\n", o->id, t->hactive, t->vactive, t->clock,
            t->clock * 24 / 1000);
    for (uint32_t lanes = o->dp_lanes; lanes; lanes >>= 1) {
        for (unsigned i = 0; i < sizeof(dp_rates) && dp_rates[i] <= o->dp_bw; i++) {
            struct dp_stream st;
            uint32_t bw = dp_rates[i];
            if (dp_stream(d, t, bw, lanes, ef, &st) || dp_train(d, o, bw, lanes, ef))
                continue;
            NV0073_CTRL_CMD_DP_CONFIG_STREAM_PARAMS c = {
                .head = (NvU32)head,
                .sorIndex = sor,
                .dpLink = o->protocol == NV0073_CTRL_SPECIFIC_OR_PROTOCOL_SOR_DP_B,
                .bEnableOverride = 1,
                .hBlankSym = st.hblank_sym,
                .vBlankSym = st.vblank_sym,
                .SST = { .bEnhancedFraming = (NvBool)ef, .tuSize = 64, .waterMark = st.watermark },
            };
            if (!rm_control(d->s, d->c.client, DISP_COMMON, NV0073_CTRL_CMD_DP_CONFIG_STREAM, &c, sizeof(c))) {
                log_msg("disp: head %d: SST stream not configured\n", head);
                return -1;
            }
            log_msg("disp: head %d: SST on SOR%u link %c, watermark %u, blanking symbols h %u v %u\n", head, sor,
                    c.dpLink ? 'B' : 'A', st.watermark, st.hblank_sym, st.vblank_sym);
            hd->dp_bw = bw;
            hd->dp_lanes = lanes;
            hd->dp_ef = ef;
            return 0;
        }
    }
    log_msg("disp: display %#x: no link carries the mode\n", o->id);
    return -1;
}

void disp_dp_service(struct disp *d, int head, const struct disp_output *o)
{
    const struct disp_head *hd = &d->head[head];
    uint8_t st[5];  // sink count, service IRQ vector, lane status, alignment
    if (dp_aux_read(d, o->id, 0x200, st, sizeof(st)))
        return;
    if (st[1])
        dp_aux_write(d, o->id, 0x201, &st[1], 1);  // written back to acknowledge
    log_msg("disp: display %#x: IRQ_HPD, sink count %u, service %#x, lane status %02x %02x align %02x\n", o->id,
            st[0] & 0xbf, st[1], st[2], st[3], st[4]);
    if (!hd->dp_lanes || lanes_ok(st + 2, hd->dp_lanes))
        return;
    log_msg("disp: display %#x: link lost, training it again\n", o->id);
    dp_train(d, o, hd->dp_bw, hd->dp_lanes, hd->dp_ef);
}
