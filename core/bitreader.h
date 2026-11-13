// Bit reader for H.264/H.265 RBSP and AV1/VP9 headers, plus the VP8/VP9 boolean decoder
#pragma once
#include "plat.h"

// emulation prevention bytes skipped, unless raw; reads past the end give zeros and set over
struct bits {
    const uint8_t *p;
    size_t len, pos;  // pos: the next byte of the NAL to load
    uint32_t cur;
    int left, zeros;
    int over;
    int raw;  // AV1 has no emulation prevention
    uint32_t skipped;  // emulation prevention bytes passed
    uint64_t stop;     // the bit position of rbsp_stop_one_bit
};

void bits_init(struct bits *b, const uint8_t *p, size_t len);
void bits_init_raw(struct bits *b, const uint8_t *p, size_t len);
uint32_t bits_u(struct bits *b, int n);
uint32_t bits_ue(struct bits *b);
int32_t bits_se(struct bits *b);
int bits_more_rbsp_data(struct bits *b);
// bits read so far, not counting emulation prevention bytes
uint64_t bits_pos(const struct bits *b);

// the next NAL unit from *pos: its payload and length, start code skipped; 0 at the end
int bits_next_nal(const uint8_t *s, size_t len, size_t *pos, size_t *start, size_t *size);
// the next VP8/VP9 frame from *pos: the next IVF frame (32-byte file header skipped at 0), or the whole input if it
// isn't IVF (one container sample); 0 at the end, -1 for a cut frame
int bits_next_ivf(const uint8_t *s, size_t len, size_t *pos, size_t *start, size_t *size);

// VP8's boolean entropy decoder (RFC 6386 7.3), which VP9 keeps (9.2); reads past the end give zeros
struct boolreader {
    struct bits b;
    uint32_t value, range;
};

void bool_init(struct boolreader *r, const uint8_t *p, size_t len);
int bool_read(struct boolreader *r, uint32_t prob);
uint32_t bool_literal(struct boolreader *r, int n);
