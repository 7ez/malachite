// Cursor image handling that needs no hardware: undoing the padding IOFramebuffer's converter may add, and laying
// a premultiplied image out in the head's fixed-size plane.
#pragma once
#include <stdint.h>

// The converter may space rows by pad (the descriptor's width). px was cleared first, so a set pixel past w * h means
// padded rows; they are packed to rows of w. px holds pad * h pixels. Returns 1 if it repacked.
static inline int cursor_unpad(uint32_t *px, uint32_t w, uint32_t h, uint32_t pad)
{
    if (pad <= w)
        return 0;
    int padded = 0;
    for (uint32_t i = w * h; i < pad * h && !padded; i++)
        padded = px[i] != 0;
    if (!padded)
        return 0;
    for (uint32_t y = 1; y < h; y++)
        for (uint32_t x = 0; x < w; x++)
            px[y * w + x] = px[y * pad + x];
    return 1;
}

// straight alpha premultiplied, as the head blends it (NEG_K1_TIMES_SRC); clear outside the image
static inline uint32_t cursor_pixel(const uint32_t *argb, uint32_t w, uint32_t h, uint32_t x, uint32_t y)
{
    if (x >= w || y >= h)
        return 0;
    uint32_t p = argb[y * w + x], a = p >> 24;
    uint32_t r = ((p >> 16 & 255) * a + 127) / 255, g = ((p >> 8 & 255) * a + 127) / 255,
             b = ((p & 255) * a + 127) / 255;
    return a << 24 | r << 16 | g << 8 | b;
}

typedef void (*cursor_put)(void *ctx, uint32_t index, uint32_t px);

// into a dim x dim plane; only the corner the new or old (old_w x old_h) image covers is written, the rest is clear
static inline void cursor_pack(cursor_put put, void *ctx, uint32_t dim, uint32_t old_w, uint32_t old_h,
                               const uint32_t *argb, uint32_t w, uint32_t h)
{
    uint32_t cols = w > old_w ? w : old_w, rows = h > old_h ? h : old_h;
    for (uint32_t y = 0; y < rows; y++)
        for (uint32_t x = 0; x < cols; x++)
            put(ctx, y * dim + x, cursor_pixel(argb, w, h, x, y));
}
