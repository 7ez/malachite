#include "vp9.h"
#include "bitreader.h"

#define SYNC_CODE 0x498342
#define CS_RGB 7
#define MIN_TILE_WIDTH_B64 4u
#define MAX_TILE_WIDTH_B64 64u

static const uint8_t seg_feature_bits[4] = { 8, 6, 2, 0 };
static const uint8_t seg_feature_signed[4] = { 1, 1, 0, 0 };

void vp9_init(struct vp9_dec *d)
{
    *d = (struct vp9_dec){ 0 };
}

void vp9_tile_starts(uint32_t mi, uint32_t log2, uint32_t *starts)
{
    uint32_t sbs = (mi + 7) >> 3;
    for (uint32_t n = 0; n <= 1u << log2; n++) {
        uint32_t at = (n * sbs) >> log2;
        starts[n] = at < sbs ? at : sbs;
    }
}

static int32_t su(struct bits *b, int n)
{
    int32_t v = (int32_t)bits_u(b, n);
    return bits_u(b, 1) ? -v : v;
}

static uint8_t read_prob(struct bits *b)
{
    return bits_u(b, 1) ? (uint8_t)bits_u(b, 8) : 255;
}

// profiles 1 and 3 are 4:4:4, 4:2:2 or 4:4:0, which NVDEC doesn't decode
static int color_config(struct bits *b, struct vp9_frame *f)
{
    f->bit_depth = f->profile >= 2 ? (bits_u(b, 1) ? 12 : 10) : 8;
    if (bits_u(b, 3) == CS_RGB || f->bit_depth == 12)
        return -1;
    bits_u(b, 1);  // color_range
    return 0;
}

static void frame_size(struct bits *b, struct vp9_frame *f)
{
    f->width = bits_u(b, 16) + 1;
    f->height = bits_u(b, 16) + 1;
}

static void render_size(struct bits *b)
{
    if (bits_u(b, 1)) {
        bits_u(b, 16);
        bits_u(b, 16);
    }
}

// setup_past_independence(), but for the probabilities, which are the kernel's
static void past_independence(struct vp9_frame *f)
{
    static const int32_t ref_deltas[4] = { 1, 0, -1, -1 };
    memset(f->seg_feature_enabled, 0, sizeof(f->seg_feature_enabled));
    memset(f->seg_feature_data, 0, sizeof(f->seg_feature_data));
    f->seg_abs = 0;
    f->lf_delta_enabled = 1;
    memcpy(f->lf_ref_deltas, ref_deltas, sizeof(ref_deltas));
    f->lf_mode_deltas[0] = f->lf_mode_deltas[1] = 0;
    memset(f->sign_bias, 0, sizeof(f->sign_bias));
}

static void loop_filter_params(struct bits *b, struct vp9_frame *f)
{
    f->lf_level = bits_u(b, 6);
    f->lf_sharpness = bits_u(b, 3);
    f->lf_delta_enabled = (int)bits_u(b, 1);
    if (!f->lf_delta_enabled || !bits_u(b, 1))
        return;
    for (int i = 0; i < 4; i++)
        if (bits_u(b, 1))
            f->lf_ref_deltas[i] = su(b, 6);
    for (int i = 0; i < 2; i++)
        if (bits_u(b, 1))
            f->lf_mode_deltas[i] = su(b, 6);
}

static int32_t read_delta_q(struct bits *b)
{
    return bits_u(b, 1) ? su(b, 4) : 0;
}

static void segmentation_params(struct bits *b, struct vp9_frame *f)
{
    memset(f->seg_tree, 255, sizeof(f->seg_tree));
    memset(f->seg_pred, 255, sizeof(f->seg_pred));
    f->seg_enabled = (int)bits_u(b, 1);
    if (!f->seg_enabled)
        return;
    f->seg_update_map = (int)bits_u(b, 1);
    if (f->seg_update_map) {
        for (int i = 0; i < 7; i++)
            f->seg_tree[i] = read_prob(b);
        f->seg_temporal_update = (int)bits_u(b, 1);
        for (int i = 0; f->seg_temporal_update && i < 3; i++)
            f->seg_pred[i] = read_prob(b);
    }
    if (!bits_u(b, 1))  // segmentation_update_data
        return;
    f->seg_abs = (int)bits_u(b, 1);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            int16_t v = 0;
            f->seg_feature_enabled[i][j] = (uint8_t)bits_u(b, 1);
            if (f->seg_feature_enabled[i][j]) {
                v = (int16_t)bits_u(b, seg_feature_bits[j]);
                if (seg_feature_signed[j] && bits_u(b, 1))
                    v = (int16_t)-v;
            }
            f->seg_feature_data[i][j] = v;
        }
    }
}

static void tile_info(struct bits *b, struct vp9_frame *f)
{
    uint32_t sb_cols = ((f->width + 7) / 8 + 7) / 8, min_log2 = 0, max_log2 = 1;
    while (MAX_TILE_WIDTH_B64 << min_log2 < sb_cols)
        min_log2++;
    while (sb_cols >> max_log2 >= MIN_TILE_WIDTH_B64)
        max_log2++;
    max_log2--;
    f->tile_cols_log2 = min_log2;
    while (f->tile_cols_log2 < max_log2 && bits_u(b, 1))
        f->tile_cols_log2++;
    f->tile_rows_log2 = bits_u(b, 1);
    if (f->tile_rows_log2)
        f->tile_rows_log2 += bits_u(b, 1);
}

// 6.2: the uncompressed header, from what the frame before left
static int uncompressed_header(struct vp9_dec *d, struct bits *b, struct vp9_frame *f)
{
    *f = (struct vp9_frame){
        .bit_depth = d->bit_depth,
        .lf_delta_enabled = d->lf_delta_enabled,
        .seg_abs = d->seg_abs,
    };
    memcpy(f->lf_ref_deltas, d->lf_ref_deltas, sizeof(f->lf_ref_deltas));
    memcpy(f->lf_mode_deltas, d->lf_mode_deltas, sizeof(f->lf_mode_deltas));
    memcpy(f->seg_feature_enabled, d->seg_feature_enabled, sizeof(f->seg_feature_enabled));
    memcpy(f->seg_feature_data, d->seg_feature_data, sizeof(f->seg_feature_data));

    if (bits_u(b, 2) != 2)  // frame_marker
        return -1;
    f->profile = bits_u(b, 1);
    f->profile |= bits_u(b, 1) << 1;
    if (f->profile == 1 || f->profile == 3)
        return -1;
    f->show_existing = (int)bits_u(b, 1);
    if (f->show_existing) {
        f->existing_idx = bits_u(b, 3);
        return d->refs[f->existing_idx].valid ? 0 : -1;
    }
    f->type = bits_u(b, 1);
    f->show = (int)bits_u(b, 1);
    f->error_resilient = (int)bits_u(b, 1);
    if (f->type == VP9_KEY_FRAME) {
        if (bits_u(b, 24) != SYNC_CODE || color_config(b, f))
            return -1;
        frame_size(b, f);
        render_size(b);
        f->refresh = 0xff;
        f->intra = 1;
    } else {
        f->intra_only = f->show ? 0 : (int)bits_u(b, 1);
        f->intra = f->intra_only;
        f->reset_context = f->error_resilient ? 0 : bits_u(b, 2);
        if (f->intra_only) {
            if (bits_u(b, 24) != SYNC_CODE)
                return -1;
            if (f->profile > 0 && color_config(b, f))
                return -1;
            if (f->profile == 0)
                f->bit_depth = 8;
            f->refresh = bits_u(b, 8);
            frame_size(b, f);
            render_size(b);
        } else {
            f->refresh = bits_u(b, 8);
            for (int i = 0; i < 3; i++) {
                f->ref_idx[i] = bits_u(b, 3);
                f->sign_bias[VP9_LAST_FRAME + i] = (int)bits_u(b, 1);
                if (!d->refs[f->ref_idx[i]].valid)
                    return -1;
            }
            int found = 0;
            for (int i = 0; i < 3 && !found; i++) {
                if ((found = (int)bits_u(b, 1))) {
                    f->width = d->refs[f->ref_idx[i]].width;
                    f->height = d->refs[f->ref_idx[i]].height;
                }
            }
            if (!found)
                frame_size(b, f);
            render_size(b);
            f->allow_hp = (int)bits_u(b, 1);
            f->interp_filter = bits_u(b, 1) ? VP9_SWITCHABLE : bits_u(b, 2);
        }
    }
    if (!f->error_resilient) {
        f->refresh_context = (int)bits_u(b, 1);
        f->parallel = (int)bits_u(b, 1);
    } else {
        f->parallel = 1;
    }
    f->context = bits_u(b, 2);
    if (f->intra || f->error_resilient)
        past_independence(f);
    loop_filter_params(b, f);
    f->base_q = bits_u(b, 8);
    f->dq_y_dc = read_delta_q(b);
    f->dq_uv_dc = read_delta_q(b);
    f->dq_uv_ac = read_delta_q(b);
    f->lossless = !f->base_q && !f->dq_y_dc && !f->dq_uv_dc && !f->dq_uv_ac;
    segmentation_params(b, f);
    tile_info(b, f);
    return b->over ? -1 : 0;
}

// vp9_setup_compound_reference_mode(): the reference whose sign bias the other two don't share is the fixed one
static void compound_refs(struct vp9_frame *f)
{
    const int *s = f->sign_bias;
    if (s[VP9_LAST_FRAME] == s[VP9_GOLDEN_FRAME]) {
        f->comp_fixed_ref = VP9_ALTREF_FRAME;
        f->comp_var_ref[0] = VP9_LAST_FRAME;
        f->comp_var_ref[1] = VP9_GOLDEN_FRAME;
    } else if (s[VP9_LAST_FRAME] == s[VP9_ALTREF_FRAME]) {
        f->comp_fixed_ref = VP9_GOLDEN_FRAME;
        f->comp_var_ref[0] = VP9_LAST_FRAME;
        f->comp_var_ref[1] = VP9_ALTREF_FRAME;
    } else {
        f->comp_fixed_ref = VP9_LAST_FRAME;
        f->comp_var_ref[0] = VP9_GOLDEN_FRAME;
        f->comp_var_ref[1] = VP9_ALTREF_FRAME;
    }
}

// Annex B: the frames of a sample, by the superframe index at its end; without one the sample is a frame
static int superframe(struct vp9_dec *d, const uint8_t *s, size_t start, size_t size)
{
    uint8_t marker = size ? s[start + size - 1] : 0;
    uint32_t n = (marker & 7) + 1, bytes = (marker >> 3 & 3) + 1, index = 2 + bytes * n;
    d->frames = 1;
    d->next_frame = 0;
    d->frame_pos[0] = (uint32_t)start;
    d->frame_len[0] = (uint32_t)size;
    if ((marker & 0xe0) != 0xc0 || size < index || s[start + size - index] != marker)
        return 0;
    const uint8_t *p = s + start + size - index + 1;
    size_t at = start;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = 0;
        for (uint32_t j = 0; j < bytes; j++)
            len |= (uint32_t)*p++ << (8 * j);
        if (len > start + size - index - at)
            return -1;
        d->frame_pos[i] = (uint32_t)at;
        d->frame_len[i] = len;
        at += len;
    }
    d->frames = n;
    return 0;
}

// the slot no reference holds
static uint32_t free_slot(const struct vp9_dec *d)
{
    uint32_t used = 0;
    for (int i = 0; i < VP9_REFS; i++)
        if (d->refs[i].valid)
            used |= 1u << d->refs[i].slot;
    uint32_t slot = 0;
    while (used >> slot & 1)
        slot++;
    return slot;
}

static int parse_frame(struct vp9_dec *d, const uint8_t *s, uint32_t start, uint32_t size, struct vp9_picture *pic)
{
    struct vp9_frame *f = &pic->hdr;
    struct bits b;
    bits_init_raw(&b, s + start, size);
    *pic = (struct vp9_picture){ .index = d->pictures };
    if (uncompressed_header(d, &b, f))
        return -1;
    if (f->show_existing) {
        const struct vp9_ref *r = &d->refs[f->existing_idx];
        pic->existing = pic->output = 1;
        pic->slot = r->slot;
        f->width = r->width;
        f->height = r->height;
        return 1;
    }
    // a stream starts with an intra frame; the surfaces are as big as the last key frame
    if (f->type == VP9_KEY_FRAME) {
        d->width = f->width;
        d->height = f->height;
        d->bit_depth = f->bit_depth;
    } else if (!d->width || f->width > d->width || f->height > d->height || f->bit_depth != d->bit_depth) {
        return -1;
    }

    uint32_t compressed = bits_u(&b, 16), header = (uint32_t)((bits_pos(&b) + 7) / 8);
    if (b.over || !compressed || compressed >= size - header)  // the tile data follows
        return -1;
    pic->output = f->show;
    pic->slot = free_slot(d);
    pic->header_pos = start + header;
    pic->header_len = compressed;
    pic->tiles_pos = pic->header_pos + compressed;
    pic->bytes = size - header - compressed;
    if (!f->intra)
        compound_refs(f);

    struct vp9_prob_frame pf = {
        .intra = f->intra,
        .lossless = f->lossless,
        .switchable = f->interp_filter == VP9_SWITCHABLE,
        .allow_hp = f->allow_hp,
        .compound = f->sign_bias[2] != f->sign_bias[1] || f->sign_bias[3] != f->sign_bias[1],
    };
    vp9_default_probs(&d->scratch);
    if (vp9_compressed(&pf, s + pic->header_pos, compressed, &d->scratch))
        return -1;
    f->tx_mode = pf.tx_mode;
    f->reference_mode = pf.reference_mode;

    d->lf_delta_enabled = f->lf_delta_enabled;
    memcpy(d->lf_ref_deltas, f->lf_ref_deltas, sizeof(d->lf_ref_deltas));
    memcpy(d->lf_mode_deltas, f->lf_mode_deltas, sizeof(d->lf_mode_deltas));
    d->seg_abs = f->seg_abs;
    memcpy(d->seg_feature_enabled, f->seg_feature_enabled, sizeof(d->seg_feature_enabled));
    memcpy(d->seg_feature_data, f->seg_feature_data, sizeof(d->seg_feature_data));
    return 1;
}

int vp9_next(struct vp9_dec *d, const uint8_t *s, size_t len, size_t *pos, struct vp9_picture *pic)
{
    if (!*pos)  // a new stream or sample: what is left of a superframe that broke off is in another buffer
        d->frames = d->next_frame = 0;
    while (d->next_frame == d->frames) {
        size_t start, size;
        int r = bits_next_ivf(s, len, pos, &start, &size);
        if (r <= 0 || superframe(d, s, start, size))
            return r < 0 ? -1 : r;
    }
    uint32_t i = d->next_frame++;
    return parse_frame(d, s, d->frame_pos[i], d->frame_len[i], pic);
}

void vp9_done(struct vp9_dec *d, const struct vp9_picture *pic)
{
    const struct vp9_frame *f = &pic->hdr;
    d->pictures++;
    if (pic->existing)
        return;
    struct vp9_ref r = { 1, pic->slot, f->width, f->height };
    for (int i = 0; i < VP9_REFS; i++)
        if (f->refresh >> i & 1)
            d->refs[i] = r;
    if (f->seg_enabled && f->seg_update_map)
        d->seg_map ^= 1;
    d->mv_map ^= 1;
    d->have_last = 1;
    d->last_key = f->type == VP9_KEY_FRAME;
    d->last_show = f->show;
    d->last_width = f->width;
    d->last_height = f->height;
}
