// VP8 for NVDEC, IVF or one container sample at a time: frame size, data from the first partition on, references.
// NVDEC parses the header and keeps the probabilities; this parses only as far as the references (RFC 6386 9.2-9.8).
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
    uint32_t width, height;  // last key frame's, or the container's
    int valid;               // a key frame was decoded
    uint32_t refs[3];        // surfaces of golden, altref and last
    uint32_t pictures;
};

void vp8_init(struct vp8_dec *d);
// next frame at *pos, advancing it: 1 frame, 0 end, -1 undecodable; pic->slot is a surface no reference holds
int vp8_next(struct vp8_dec *d, const uint8_t *s, size_t len, size_t *pos, struct vp8_picture *pic);
// updates the references pic refreshes or copies
void vp8_done(struct vp8_dec *d, const struct vp8_picture *pic);
// RFC 6386 defaults NVDEC starts a stream from and then maintains: coefficient rows of 11 padded to 12, y and uv mode
// probabilities, then both MV components' 19 padded to 20
void vp8_default_probs(uint8_t *table);
