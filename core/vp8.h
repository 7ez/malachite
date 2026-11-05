// VP8 streams, an IVF file or a container sample (a frame) at a time, down to what NVDEC needs from the driver: each
// frame's size, its first partition's and the reference surfaces it uses. NVDEC parses the frame header itself and
// keeps the probabilities in a buffer of its own; the header is parsed here as far as the references need it
// (RFC 6386 9.2 to 9.8).
#pragma once
#include "plat.h"

#define VP8_SLOTS 4  // golden, altref and last, and the frame being decoded
#define VP8_PROBS_SIZE 0x4b0  // NVDEC's probability buffer, as a stream starts it

enum { VP8_GOLDEN, VP8_ALTREF, VP8_LAST };

struct vp8_frame {
    int key, show;
    uint32_t version, first_part_size;
    uint32_t width, height;
    int seg_enabled, seg_update_data;
    int refresh_golden, refresh_alt, refresh_last;
    uint32_t copy_to_golden, copy_to_alt;  // 1 from last, 2 from altref or golden
};

struct vp8_picture {
    struct vp8_frame hdr;
    int output;
    uint32_t slot;
    uint32_t index;  // in decoding order
    uint32_t pos, bytes;  // what NVDEC reads: the frame from its first partition on
};

struct vp8_dec {
    uint32_t width, height;  // of the last key frame, or what the container says
    int valid;               // a key frame was decoded
    uint32_t refs[3];        // the surfaces of the golden, altref and last frames
    uint32_t pictures;
};

void vp8_init(struct vp8_dec *d);
// the next frame from stream position *pos, which moves past it: 1 for a frame, 0 at the end, -1 for a stream it
// can't decode. pic->slot is a surface no reference holds.
int vp8_next(struct vp8_dec *d, const uint8_t *s, size_t len, size_t *pos, struct vp8_picture *pic);
// pic is decoded: the references it refreshes or copies are updated
void vp8_done(struct vp8_dec *d, const struct vp8_picture *pic);
// the probability buffer NVDEC starts a stream from and keeps up to date itself: the defaults of RFC 6386, the
// coefficients' eleven to a row of twelve, the luma and chroma mode probabilities, then the two motion vector
// components' nineteen to a row of twenty
void vp8_default_probs(uint8_t *table);
