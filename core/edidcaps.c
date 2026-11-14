#include "edidcaps.h"

// 2^(i/32) in 1/1024ths
static const uint16_t pow2_32[32] = { 1024, 1046, 1069, 1093, 1117, 1141, 1166, 1192, 1218, 1244, 1272,
                                      1300, 1328, 1357, 1387, 1417, 1448, 1480, 1512, 1545, 1579, 1614,
                                      1649, 1685, 1722, 1760, 1798, 1838, 1878, 1919, 1961, 2004 };

// the code's 50 * 2^(code/32) nits
static uint32_t nits_of(uint8_t code)
{
    return (uint32_t)((50u * pow2_32[code & 31] << (code >> 5)) >> 10);
}

static void hdr_block(const uint8_t *p, uint32_t len, struct edid_caps *c)
{
    if (len < 3)
        return;
    c->hdr_pq = p[1] >> 2 & 1;
    c->hdr_hlg = p[1] >> 3 & 1;
    c->hdr_type1 = p[2] & 1;
    if (len >= 4 && p[3])
        c->max_nits = nits_of(p[3]);
    if (len >= 5 && p[4])
        c->avg_nits = nits_of(p[4]);
    // the minimum is the maximum times (code / 255)^2 / 100
    if (len >= 6 && p[5] && c->max_nits)
        c->min_nits_e4 = (uint32_t)((uint64_t)c->max_nits * p[5] * p[5] * 100 / 65025);
}

static void forum_block(const uint8_t *p, uint32_t len, struct edid_caps *c)
{
    // the OUI is C4-5D-D8, least significant byte first; the VRR range is the 9th and 10th payload bytes
    if (len < 10 || p[0] != 0xd8 || p[1] != 0x5d || p[2] != 0xc4)
        return;
    c->hdmi_vrr_min = p[8] & 0x3f;
    c->hdmi_vrr_max = (uint32_t)(p[8] >> 6) << 8 | p[9];
}

static void range_limits(const uint8_t *d, struct edid_caps *c)
{
    if (d[0] || d[1] || d[2] || d[3] != 0xfd)
        return;
    uint32_t lo = d[5], hi = d[6];
    if ((d[4] & 3) == 2) {
        hi += 255;
    } else if ((d[4] & 3) == 3) {
        lo += 255;
        hi += 255;
    }
    c->range_min = lo;
    c->range_max = hi;
}

static void cta_block(const uint8_t *b, struct edid_caps *c)
{
    uint32_t end = b[2];
    if (b[0] != 0x02 || end < 4 || end > 127)
        return;
    for (uint32_t p = 4; p < end;) {
        uint32_t tag = b[p] >> 5, len = b[p] & 31;
        if (p + 1 + len > end)
            break;
        const uint8_t *x = b + p + 1;
        if (tag == 7 && len >= 1) {
            if (x[0] == 0x06)
                hdr_block(x, len, c);
            else if (x[0] == 0x05 && len >= 2)
                c->bt2020_rgb = x[1] >> 7 & 1;
        } else if (tag == 3) {
            forum_block(x, len, c);
        }
        p += 1 + len;
    }
    // detailed timings and descriptors follow the data blocks
    for (uint32_t p = end; p + 18 <= 127; p += 18)
        range_limits(b + p, c);
}

void edid_caps_parse(const uint8_t *edid, uint32_t size, struct edid_caps *out)
{
    *out = (struct edid_caps){ 0 };
    if (size < 128)
        return;
    for (uint32_t p = 54; p <= 108; p += 18)
        range_limits(edid + p, out);
    for (uint32_t i = 1; i * 128 + 128 <= size && i <= 3; i++)
        cta_block(edid + i * 128, out);
}
