#include "vp8.h"
#include "bitreader.h"

void vp8_init(struct vp8_dec *d)
{
    *d = (struct vp8_dec){ 0 };
}

// a flag, then a magnitude and its sign
static void skip_delta(struct boolreader *r, int bits)
{
    if (bool_literal(r, 1))
        bool_literal(r, bits + 1);
}

// 19.2: the frame header up to the references it refreshes
static void frame_header(struct boolreader *r, struct vp8_frame *f)
{
    if (f->key)
        bool_literal(r, 2);  // color_space, clamping_type
    f->seg_enabled = (int)bool_literal(r, 1);
    if (f->seg_enabled) {
        int update_map = (int)bool_literal(r, 1);
        f->seg_update_data = (int)bool_literal(r, 1);
        if (f->seg_update_data) {
            bool_literal(r, 1);  // segment_feature_mode
            for (int i = 0; i < 4; i++)
                skip_delta(r, 7);
            for (int i = 0; i < 4; i++)
                skip_delta(r, 6);
        }
        for (int i = 0; update_map && i < 3; i++)
            if (bool_literal(r, 1))
                bool_literal(r, 8);
    }
    bool_literal(r, 1 + 6 + 3);  // filter_type, loop_filter_level, sharpness_level
    if (bool_literal(r, 1) && bool_literal(r, 1))  // loop_filter_adj_enable, mode_ref_lf_delta_update
        for (int i = 0; i < 8; i++)
            skip_delta(r, 6);
    bool_literal(r, 2);  // log2_nbr_of_dct_partitions
    bool_literal(r, 7);  // y_ac_qi
    for (int i = 0; i < 5; i++)
        skip_delta(r, 4);
    if (f->key) {
        f->refresh_golden = f->refresh_alt = f->refresh_last = 1;
        return;
    }
    f->refresh_golden = (int)bool_literal(r, 1);
    f->refresh_alt = (int)bool_literal(r, 1);
    if (!f->refresh_golden)
        f->copy_to_golden = bool_literal(r, 2);
    if (!f->refresh_alt)
        f->copy_to_alt = bool_literal(r, 2);
    bool_literal(r, 2);  // sign_bias_golden, sign_bias_alternate
    bool_literal(r, 1);  // refresh_entropy_probs
    f->refresh_last = (int)bool_literal(r, 1);
}

static uint32_t free_slot(const struct vp8_dec *d)
{
    uint32_t used = d->valid ? 1u << d->refs[0] | 1u << d->refs[1] | 1u << d->refs[2] : 0, slot = 0;
    while (used >> slot & 1)
        slot++;
    return slot;
}

int vp8_next(struct vp8_dec *d, const uint8_t *s, size_t len, size_t *pos, struct vp8_picture *pic)
{
    size_t start, size;
    int r = bits_next_ivf(s, len, pos, &start, &size);
    if (r <= 0)
        return r;
    const uint8_t *p = s + start;
    struct vp8_frame *f = &pic->hdr;
    *pic = (struct vp8_picture){ .index = d->pictures };
    if (size < 3)
        return -1;
    uint32_t tag = p[0] | p[1] << 8 | p[2] << 16, header = 3;
    f->key = !(tag & 1);
    f->version = tag >> 1 & 7;
    f->show = (int)(tag >> 4 & 1);
    f->first_part_size = tag >> 5;
    if (f->key) {
        if (size < 10 || p[3] != 0x9d || p[4] != 0x01 || p[5] != 0x2a)
            return -1;
        f->width = (p[6] | p[7] << 8) & 0x3fff;
        f->height = (p[8] | p[9] << 8) & 0x3fff;
        if (!f->width || !f->height)
            return -1;
        d->width = f->width;
        d->height = f->height;
        header = 10;
    } else if (!d->valid) {
        return -1;
    } else {
        f->width = d->width;
        f->height = d->height;
    }
    if (f->first_part_size > size - header)
        return -1;
    struct boolreader b;
    bool_init(&b, p + header, f->first_part_size);
    frame_header(&b, f);
    pic->output = f->show;
    pic->slot = free_slot(d);
    pic->pos = (uint32_t)(start + header);
    pic->bytes = (uint32_t)(size - header);
    return 1;
}

// The copies take the references from before the frame, as FFmpeg's decoder does (vp8.c ref_to_update)
void vp8_done(struct vp8_dec *d, const struct vp8_picture *pic)
{
    const struct vp8_frame *f = &pic->hdr;
    uint32_t old[3];
    memcpy(old, d->refs, sizeof(old));
    d->pictures++;
    if (f->key)
        d->valid = 1;
    if (f->copy_to_golden)
        d->refs[VP8_GOLDEN] = old[f->copy_to_golden == 1 ? VP8_LAST : VP8_ALTREF];
    if (f->copy_to_alt)
        d->refs[VP8_ALTREF] = old[f->copy_to_alt == 1 ? VP8_LAST : VP8_GOLDEN];
    if (f->refresh_golden)
        d->refs[VP8_GOLDEN] = pic->slot;
    if (f->refresh_alt)
        d->refs[VP8_ALTREF] = pic->slot;
    if (f->refresh_last)
        d->refs[VP8_LAST] = pic->slot;
}
