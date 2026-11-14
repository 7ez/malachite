// What a display's EDID says about HDR and variable refresh: CTA-861's HDR static metadata and colorimetry blocks, the
// HDMI Forum block's VRR range and the range limits descriptor of the base block (Adaptive-Sync, FreeSync).
#pragma once
#include <stdint.h>

struct edid_caps {
    int hdr_pq, hdr_hlg;  // the EOTFs the display lists besides SDR
    int hdr_type1;        // static metadata type 1, the one HDR10 sends
    uint32_t max_nits, avg_nits;  // 0 when the block leaves them out
    uint32_t min_nits_e4;         // in 0.0001 nits
    int bt2020_rgb;
    uint32_t range_min, range_max;  // Hz, from a range limits descriptor, 0 when there is none
    uint32_t hdmi_vrr_min, hdmi_vrr_max;  // Hz, from the HDMI Forum block, 0 when there is none
};

// size counts the bytes there are: a 128-byte base block, then 128-byte extensions
void edid_caps_parse(const uint8_t *edid, uint32_t size, struct edid_caps *out);
