// NVDEC's film grain tables (AV1FilmGrainMemory in nvdec_drv.h): 7.18.3.3 grain cropped to what a frame uses, and
// the scaling functions as LUTs.
#pragma once
#include "av1.h"

struct av1_grain_tables {
    uint8_t scaling_lut_y[256], scaling_lut_cb[256], scaling_lut_cr[256];
    int16_t luma[64 * 64], cb[32 * 32], cr[32 * 32];
};

_Static_assert(sizeof(struct av1_grain_tables) == 13056, "AV1FilmGrainMemory is 13056 bytes");

void av1_grain_tables(const struct av1_grain *g, uint32_t bit_depth, struct av1_grain_tables *out);
