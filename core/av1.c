#include "av1.h"
#include "bitreader.h"

enum {
    OBU_SEQUENCE_HEADER = 1,
    OBU_TEMPORAL_DELIMITER,
    OBU_FRAME_HEADER,
    OBU_TILE_GROUP,
    OBU_METADATA,
    OBU_FRAME,
    OBU_REDUNDANT_FRAME_HEADER,
};

#define SELECT_SCREEN_CONTENT_TOOLS 2
#define SELECT_INTEGER_MV 2
#define SWITCHABLE_FILTER 4
#define WARPEDMODEL_PREC_BITS 16
#define MAX_TILE_WIDTH 4096
#define MAX_TILE_AREA (4096 * 2304)
#define SEG_LVL_REF_FRAME 5

static const uint8_t seg_feature_bits[8] = { 8, 6, 6, 6, 6, 3, 0, 0 };
static const uint8_t seg_feature_signed[8] = { 1, 1, 1, 1, 1, 0, 0, 0 };
static const uint8_t seg_feature_max[8] = { 255, 63, 63, 63, 63, 7, 0, 0 };
static const int8_t default_ref_deltas[8] = { 1, 0, 0, 0, -1, 0, -1, -1 };
static const uint8_t remap_lr_type[4] = { 0, 3, 1, 2 };

void av1_init(struct av1_dec *d)
{
    *d = (struct av1_dec){ 0 };
}

int av1_dist(const struct av1_seq *s, int a, int b)
{
    if (!s->order_hint)
        return 0;
    int diff = a - b, m = 1 << (s->order_hint_bits - 1);
    return (diff & (m - 1)) - (diff & m);
}

static int32_t su(struct bits *b, int n)
{
    int32_t v = (int32_t)bits_u(b, n);
    int32_t sign = 1 << (n - 1);
    return v & sign ? v - 2 * sign : v;
}

static uint32_t ns(struct bits *b, uint32_t n)
{
    int w = 0;
    for (uint32_t x = n; x; x >>= 1)
        w++;
    uint32_t m = (1u << w) - n, v = bits_u(b, w - 1);
    if (v < m)
        return v;
    return (v << 1) - m + bits_u(b, 1);
}

static uint32_t uvlc(struct bits *b)
{
    int zeros = 0;
    while (!b->over && !bits_u(b, 1))
        zeros++;
    if (zeros >= 32)
        return 0xffffffff;
    return bits_u(b, zeros) + (uint32_t)((1ull << zeros) - 1);
}

static uint32_t tile_log2(uint32_t blk, uint32_t target)
{
    uint32_t k = 0;
    while ((blk << k) < target)
        k++;
    return k;
}

static int parse_sequence(struct av1_seq *s, const uint8_t *p, size_t len)
{
    struct bits b;
    bits_init_raw(&b, p, len);
    *s = (struct av1_seq){ 0 };
    uint32_t profile = bits_u(&b, 3);
    bits_u(&b, 1);  // still_picture
    s->reduced_still_picture = (int)bits_u(&b, 1);
    if (s->reduced_still_picture) {
        s->nops = 1;
        bits_u(&b, 5);  // seq_level_idx
    } else {
        if (bits_u(&b, 1)) {  // timing_info_present_flag
            bits_u(&b, 32);
            bits_u(&b, 32);
            s->equal_picture_interval = (int)bits_u(&b, 1);
            if (s->equal_picture_interval)
                uvlc(&b);
            s->decoder_model_info = (int)bits_u(&b, 1);
            if (s->decoder_model_info) {
                s->buffer_delay_length = bits_u(&b, 5) + 1;
                bits_u(&b, 32);
                s->buffer_removal_time_length = bits_u(&b, 5) + 1;
                s->frame_presentation_time_length = bits_u(&b, 5) + 1;
            }
        }
        int display_delay = (int)bits_u(&b, 1);
        s->nops = bits_u(&b, 5) + 1;
        for (uint32_t i = 0; i < s->nops; i++) {
            s->op_idc[i] = bits_u(&b, 12);
            if (bits_u(&b, 5) > 7)
                bits_u(&b, 1);  // seq_tier
            if (s->decoder_model_info) {
                s->op_decoder_model[i] = (int)bits_u(&b, 1);
                if (s->op_decoder_model[i]) {
                    bits_u(&b, (int)s->buffer_delay_length);
                    bits_u(&b, (int)s->buffer_delay_length);
                    bits_u(&b, 1);
                }
            }
            if (display_delay && bits_u(&b, 1))
                bits_u(&b, 4);
        }
    }
    s->width_bits = bits_u(&b, 4) + 1;
    s->height_bits = bits_u(&b, 4) + 1;
    s->max_width = bits_u(&b, (int)s->width_bits) + 1;
    s->max_height = bits_u(&b, (int)s->height_bits) + 1;
    if (!s->reduced_still_picture && (s->frame_ids = (int)bits_u(&b, 1))) {
        s->delta_frame_id_bits = bits_u(&b, 4) + 2;
        s->additional_frame_id_bits = bits_u(&b, 3) + 1;
    }
    s->sb128 = (int)bits_u(&b, 1);
    s->filter_intra = (int)bits_u(&b, 1);
    s->intra_edge_filter = (int)bits_u(&b, 1);
    s->screen_content_tools = s->force_integer_mv = SELECT_SCREEN_CONTENT_TOOLS;
    if (!s->reduced_still_picture) {
        s->interintra = (int)bits_u(&b, 1);
        s->masked_compound = (int)bits_u(&b, 1);
        s->warped_motion = (int)bits_u(&b, 1);
        s->dual_filter = (int)bits_u(&b, 1);
        s->order_hint = (int)bits_u(&b, 1);
        if (s->order_hint) {
            s->jnt_comp = (int)bits_u(&b, 1);
            s->ref_frame_mvs = (int)bits_u(&b, 1);
        }
        if (!bits_u(&b, 1))  // seq_choose_screen_content_tools
            s->screen_content_tools = bits_u(&b, 1);
        if (s->screen_content_tools > 0 && !bits_u(&b, 1))  // seq_choose_integer_mv
            s->force_integer_mv = bits_u(&b, 1);
        if (s->order_hint)
            s->order_hint_bits = bits_u(&b, 3) + 1;
    }
    s->superres = (int)bits_u(&b, 1);
    s->cdef = (int)bits_u(&b, 1);
    s->restoration = (int)bits_u(&b, 1);

    s->bit_depth = bits_u(&b, 1) ? 10 : 8;
    int mono = (int)bits_u(&b, 1);
    if (bits_u(&b, 1)) {  // color_description_present_flag
        uint32_t primaries = bits_u(&b, 8), transfer = bits_u(&b, 8), matrix = bits_u(&b, 8);
        if (primaries == 1 && transfer == 13 && matrix == 0)
            return -1;  // sRGB, which is 4:4:4
    }
    if (profile || mono)
        return -1;
    bits_u(&b, 1);  // color_range
    bits_u(&b, 2);  // chroma_sample_position
    s->separate_uv_delta_q = (int)bits_u(&b, 1);
    s->film_grain_present = (int)bits_u(&b, 1);
    s->valid = !b.over;
    return b.over ? -1 : 0;
}

struct header {
    const struct av1_dec *d;
    const struct av1_seq *s;
    struct av1_frame *f;
    struct bits *b;
    int intra, override_size;
    int err;
    int32_t prev_gm[AV1_REFS_PER_FRAME + 1][6];
};

static void reset_gm(int32_t gm[][6])
{
    for (int r = 0; r <= AV1_REFS_PER_FRAME; r++)
        for (int i = 0; i < 6; i++)
            gm[r][i] = i % 3 == 2 ? 1 << WARPEDMODEL_PREC_BITS : 0;
}

static const struct av1_ref *ref_of(const struct header *h, uint32_t i)
{
    return &h->d->refs[h->f->ref_idx[i]];
}

static void compute_image_size(struct av1_frame *f)
{
    f->mi_cols = 2 * ((f->width + 7) >> 3);
    f->mi_rows = 2 * ((f->height + 7) >> 3);
}

// superres changes the size NVDEC upscales to, which the setup doesn't do
static void superres_params(struct header *h)
{
    struct av1_frame *f = h->f;
    f->upscaled_width = f->width;
    if (h->s->superres && bits_u(h->b, 1))
        h->err = 1;
    compute_image_size(f);
}

static void frame_size(struct header *h)
{
    struct av1_frame *f = h->f;
    if (h->override_size) {
        f->width = bits_u(h->b, (int)h->s->width_bits) + 1;
        f->height = bits_u(h->b, (int)h->s->height_bits) + 1;
    } else {
        f->width = h->s->max_width;
        f->height = h->s->max_height;
    }
    superres_params(h);
}

static void render_size(struct header *h)
{
    struct av1_frame *f = h->f;
    if (bits_u(h->b, 1)) {
        f->render_width = bits_u(h->b, 16) + 1;
        f->render_height = bits_u(h->b, 16) + 1;
    } else {
        f->render_width = f->upscaled_width;
        f->render_height = f->height;
    }
}

static void frame_size_with_refs(struct header *h)
{
    struct av1_frame *f = h->f;
    for (uint32_t i = 0; i < AV1_REFS_PER_FRAME; i++) {
        if (bits_u(h->b, 1)) {
            const struct av1_ref *r = ref_of(h, i);
            f->upscaled_width = r->upscaled_width;
            f->width = r->upscaled_width;
            f->height = r->height;
            f->render_width = r->render_width;
            f->render_height = r->render_height;
            superres_params(h);
            return;
        }
    }
    frame_size(h);
    render_size(h);
}

// the unused slot on one side of the current frame (backward: not before it) with the latest or earliest hint
static int find_ref(const int *hint, const int *used, int cur, int backward, int latest)
{
    int found = -1, best = 0;
    for (int i = 0; i < AV1_REFS; i++) {
        if (used[i] || (hint[i] >= cur) != backward)
            continue;
        if (found < 0 || (latest ? hint[i] >= best : hint[i] < best)) {
            found = i;
            best = hint[i];
        }
    }
    return found;
}

// 7.8: the reference slots a frame names by the order hints of just two
static void set_frame_refs(struct header *h, uint32_t last, uint32_t gold)
{
    struct av1_frame *f = h->f;
    int idx[AV1_REFS_PER_FRAME], used[AV1_REFS] = { 0 }, hint[AV1_REFS];
    int cur = 1 << (h->s->order_hint_bits - 1);
    for (int i = 0; i < AV1_REFS_PER_FRAME; i++)
        idx[i] = -1;
    idx[0] = (int)last;
    idx[3] = (int)gold;
    used[last] = used[gold] = 1;
    for (int i = 0; i < AV1_REFS; i++)
        hint[i] = cur + av1_dist(h->s, (int)h->d->refs[i].order_hint, (int)f->order_hint);

    int ref;
    static const uint8_t backward_slots[3] = { 6, 4, 5 };  // ALTREF, BWDREF, ALTREF2
    for (int i = 0; i < 3; i++) {
        if ((ref = find_ref(hint, used, cur, 1, i == 0)) >= 0) {
            idx[backward_slots[i]] = ref;
            used[ref] = 1;
        }
    }
    static const uint8_t rest[5] = { 1, 2, 4, 5, 6 };  // LAST2, LAST3, BWDREF, ALTREF2, ALTREF
    for (int i = 0; i < 5; i++) {
        if (idx[rest[i]] < 0 && (ref = find_ref(hint, used, cur, 0, 1)) >= 0) {
            idx[rest[i]] = ref;
            used[ref] = 1;
        }
    }
    ref = -1;
    for (int i = 0, earliest = 0; i < AV1_REFS; i++) {
        if (ref < 0 || hint[i] < earliest) {
            ref = i;
            earliest = hint[i];
        }
    }
    for (int i = 0; i < AV1_REFS_PER_FRAME; i++)
        f->ref_idx[i] = (uint32_t)(idx[i] < 0 ? ref : idx[i]);
}

static void tile_info(struct header *h)
{
    struct av1_frame *f = h->f;
    struct bits *b = h->b;
    uint32_t sb_shift = h->s->sb128 ? 5 : 4, sb_size = sb_shift + 2;
    uint32_t sb_cols = (f->mi_cols + (1u << sb_shift) - 1) >> sb_shift;
    uint32_t sb_rows = (f->mi_rows + (1u << sb_shift) - 1) >> sb_shift;
    uint32_t max_tile_width = MAX_TILE_WIDTH >> sb_size, max_tile_area = MAX_TILE_AREA >> (2 * sb_size);
    uint32_t min_log2_cols = tile_log2(max_tile_width, sb_cols);
    uint32_t max_log2_cols = tile_log2(1, sb_cols < AV1_MAX_TILE_COLS ? sb_cols : AV1_MAX_TILE_COLS);
    uint32_t max_log2_rows = tile_log2(1, sb_rows < AV1_MAX_TILE_ROWS ? sb_rows : AV1_MAX_TILE_ROWS);
    uint32_t min_log2_tiles = tile_log2(max_tile_area, sb_rows * sb_cols);
    if (min_log2_cols > min_log2_tiles)
        min_log2_tiles = min_log2_cols;

    if (bits_u(b, 1)) {  // uniform_tile_spacing_flag
        f->tile_cols_log2 = min_log2_cols;
        while (f->tile_cols_log2 < max_log2_cols && bits_u(b, 1))
            f->tile_cols_log2++;
        uint32_t width = (sb_cols + (1u << f->tile_cols_log2) - 1) >> f->tile_cols_log2;
        f->tile_cols = 0;
        for (uint32_t start = 0; start < sb_cols; start += width)
            f->col_starts[f->tile_cols++] = start;
        f->col_starts[f->tile_cols] = sb_cols;

        uint32_t min_log2_rows = min_log2_tiles > f->tile_cols_log2 ? min_log2_tiles - f->tile_cols_log2 : 0;
        f->tile_rows_log2 = min_log2_rows;
        while (f->tile_rows_log2 < max_log2_rows && bits_u(b, 1))
            f->tile_rows_log2++;
        uint32_t height = (sb_rows + (1u << f->tile_rows_log2) - 1) >> f->tile_rows_log2;
        f->tile_rows = 0;
        for (uint32_t start = 0; start < sb_rows; start += height)
            f->row_starts[f->tile_rows++] = start;
        f->row_starts[f->tile_rows] = sb_rows;
    } else {
        uint32_t widest = 0, start = 0;
        f->tile_cols = 0;
        while (start < sb_cols && f->tile_cols < AV1_MAX_TILE_COLS) {
            f->col_starts[f->tile_cols++] = start;
            uint32_t max_width = sb_cols - start < max_tile_width ? sb_cols - start : max_tile_width;
            uint32_t size = ns(b, max_width) + 1;
            widest = size > widest ? size : widest;
            start += size;
        }
        f->col_starts[f->tile_cols] = sb_cols;
        h->err |= start != sb_cols;
        f->tile_cols_log2 = tile_log2(1, f->tile_cols);

        uint32_t area = min_log2_tiles > 0 ? (sb_rows * sb_cols) >> (min_log2_tiles + 1) : sb_rows * sb_cols;
        uint32_t max_height = widest ? area / widest : 1;
        max_height = max_height ? max_height : 1;
        start = 0;
        f->tile_rows = 0;
        while (start < sb_rows && f->tile_rows < AV1_MAX_TILE_ROWS) {
            f->row_starts[f->tile_rows++] = start;
            uint32_t max_h = sb_rows - start < max_height ? sb_rows - start : max_height;
            start += ns(b, max_h) + 1;
        }
        f->row_starts[f->tile_rows] = sb_rows;
        h->err |= start != sb_rows;
        f->tile_rows_log2 = tile_log2(1, f->tile_rows);
    }
    if (f->tile_cols_log2 || f->tile_rows_log2) {
        f->context_update_tile_id = bits_u(b, (int)(f->tile_rows_log2 + f->tile_cols_log2));
        f->tile_size_bytes = bits_u(b, 2) + 1;
    }
    h->err |= f->tile_cols * f->tile_rows > AV1_MAX_TILES;
}

static int32_t delta_q(struct bits *b)
{
    return bits_u(b, 1) ? su(b, 7) : 0;
}

static void quantization_params(struct header *h)
{
    struct av1_frame *f = h->f;
    struct bits *b = h->b;
    f->base_qindex = bits_u(b, 8);
    f->dq_y_dc = delta_q(b);
    int diff_uv_delta = h->s->separate_uv_delta_q ? (int)bits_u(b, 1) : 0;
    f->dq_u_dc = delta_q(b);
    f->dq_u_ac = delta_q(b);
    if (diff_uv_delta) {
        f->dq_v_dc = delta_q(b);
        f->dq_v_ac = delta_q(b);
    } else {
        f->dq_v_dc = f->dq_u_dc;
        f->dq_v_ac = f->dq_u_ac;
    }
    f->using_qmatrix = (int)bits_u(b, 1);
    if (f->using_qmatrix) {
        f->qm_y = bits_u(b, 4);
        f->qm_u = bits_u(b, 4);
        f->qm_v = h->s->separate_uv_delta_q ? bits_u(b, 4) : f->qm_u;
    }
}

static void segmentation_params(struct header *h)
{
    struct av1_frame *f = h->f;
    struct bits *b = h->b;
    f->seg_enabled = (int)bits_u(b, 1);
    if (!f->seg_enabled) {
        memset(f->seg_feature_enabled, 0, sizeof(f->seg_feature_enabled));
        memset(f->seg_feature_data, 0, sizeof(f->seg_feature_data));
        return;
    }
    int update_data = 1;
    if (f->primary_ref == AV1_PRIMARY_REF_NONE) {
        f->seg_update_map = 1;
    } else {
        f->seg_update_map = (int)bits_u(b, 1);
        if (f->seg_update_map)
            f->seg_temporal_update = (int)bits_u(b, 1);
        update_data = (int)bits_u(b, 1);
    }
    if (update_data) {
        for (int i = 0; i < 8; i++) {
            f->seg_feature_enabled[i] = 0;
            for (int j = 0; j < 8; j++) {
                int32_t value = 0;
                if (bits_u(b, 1)) {
                    f->seg_feature_enabled[i] |= (uint8_t)(1 << j);
                    int32_t limit = seg_feature_max[j];
                    if (seg_feature_signed[j]) {
                        value = su(b, 1 + seg_feature_bits[j]);
                        value = value < -limit ? -limit : value > limit ? limit : value;
                    } else {
                        value = (int32_t)bits_u(b, seg_feature_bits[j]);
                        value = value > limit ? limit : value;
                    }
                }
                f->seg_feature_data[i][j] = (int16_t)value;
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (f->seg_feature_enabled[i] >> j & 1) {
                f->seg_last_active = (uint32_t)i;
                f->seg_preskip |= j >= SEG_LVL_REF_FRAME;
            }
        }
    }
}

static int qindex(const struct av1_frame *f, int segment)
{
    int q = (int)f->base_qindex;
    if (f->seg_feature_enabled[segment] & 1) {
        q += f->seg_feature_data[segment][0];
        q = q < 0 ? 0 : q > 255 ? 255 : q;
    }
    return q;
}

static void loop_filter_params(struct header *h)
{
    struct av1_frame *f = h->f;
    struct bits *b = h->b;
    if (f->coded_lossless || f->allow_intrabc) {
        f->lf_level[0] = f->lf_level[1] = 0;
        for (int i = 0; i < 8; i++)
            f->lf_ref_deltas[i] = default_ref_deltas[i];
        f->lf_mode_deltas[0] = f->lf_mode_deltas[1] = 0;
        return;
    }
    f->lf_level[0] = bits_u(b, 6);
    f->lf_level[1] = bits_u(b, 6);
    if (f->lf_level[0] || f->lf_level[1]) {
        f->lf_level[2] = bits_u(b, 6);
        f->lf_level[3] = bits_u(b, 6);
    }
    f->lf_sharpness = bits_u(b, 3);
    f->lf_delta_enabled = (int)bits_u(b, 1);
    if (f->lf_delta_enabled && bits_u(b, 1)) {  // loop_filter_delta_update
        for (int i = 0; i < 8; i++)
            if (bits_u(b, 1))
                f->lf_ref_deltas[i] = su(b, 7);
        for (int i = 0; i < 2; i++)
            if (bits_u(b, 1))
                f->lf_mode_deltas[i] = su(b, 7);
    }
}

static void cdef_params(struct header *h)
{
    struct av1_frame *f = h->f;
    struct bits *b = h->b;
    if (f->coded_lossless || f->allow_intrabc || !h->s->cdef)
        return;
    f->cdef_damping = bits_u(b, 2);
    f->cdef_bits = bits_u(b, 2);
    for (uint32_t i = 0; i < 1u << f->cdef_bits; i++) {
        f->cdef_y_pri[i] = bits_u(b, 4);
        f->cdef_y_sec[i] = bits_u(b, 2);
        f->cdef_uv_pri[i] = bits_u(b, 4);
        f->cdef_uv_sec[i] = bits_u(b, 2);
    }
}

static void lr_params(struct header *h)
{
    struct av1_frame *f = h->f;
    struct bits *b = h->b;
    if (f->all_lossless || f->allow_intrabc || !h->s->restoration)
        return;
    int chroma = 0;
    for (int i = 0; i < 3; i++) {
        f->lr_type[i] = remap_lr_type[bits_u(b, 2)];
        f->uses_lr |= f->lr_type[i] != 0;
        chroma |= i > 0 && f->lr_type[i] != 0;
    }
    if (!f->uses_lr)
        return;
    if (h->s->sb128) {
        f->lr_unit_shift = bits_u(b, 1) + 1;
    } else {
        f->lr_unit_shift = bits_u(b, 1);
        if (f->lr_unit_shift)
            f->lr_unit_shift += bits_u(b, 1);
    }
    if (chroma)
        f->lr_uv_shift = bits_u(b, 1);
}

static void skip_mode_params(struct header *h)
{
    struct av1_frame *f = h->f;
    const struct av1_seq *s = h->s;
    if (h->intra || !f->reference_select || !s->order_hint)
        return;
    int forward = -1, backward = -1, forward_hint = 0, backward_hint = 0;
    for (int i = 0; i < AV1_REFS_PER_FRAME; i++) {
        int hint = (int)ref_of(h, (uint32_t)i)->order_hint;
        if (av1_dist(s, hint, (int)f->order_hint) < 0) {
            if (forward < 0 || av1_dist(s, hint, forward_hint) > 0) {
                forward = i;
                forward_hint = hint;
            }
        } else if (av1_dist(s, hint, (int)f->order_hint) > 0) {
            if (backward < 0 || av1_dist(s, hint, backward_hint) < 0) {
                backward = i;
                backward_hint = hint;
            }
        }
    }
    int second = -1;
    if (forward < 0)
        return;
    if (backward < 0) {
        int second_hint = 0;
        for (int i = 0; i < AV1_REFS_PER_FRAME; i++) {
            int hint = (int)ref_of(h, (uint32_t)i)->order_hint;
            if (av1_dist(s, hint, forward_hint) < 0 && (second < 0 || av1_dist(s, hint, second_hint) > 0)) {
                second = i;
                second_hint = hint;
            }
        }
        if (second < 0)
            return;
    } else {
        second = backward;
    }
    f->skip_mode_frame[0] = 1 + (uint32_t)(forward < second ? forward : second);
    f->skip_mode_frame[1] = 1 + (uint32_t)(forward < second ? second : forward);
    f->skip_mode_present = (int)bits_u(h->b, 1);
}

static uint32_t inverse_recenter(uint32_t r, uint32_t v)
{
    if (v > 2 * r)
        return v;
    return v & 1 ? r - ((v + 1) >> 1) : r + (v >> 1);
}

static uint32_t decode_subexp(struct bits *b, uint32_t num_syms)
{
    uint32_t mk = 0, k = 3;
    for (uint32_t i = 0;; i++) {
        uint32_t b2 = i ? k + i - 1 : k, a = 1u << b2;
        if (num_syms <= mk + 3 * a)
            return ns(b, num_syms - mk) + mk;
        if (!bits_u(b, 1))
            return bits_u(b, (int)b2) + mk;
        mk += a;
    }
}

static int32_t decode_signed_subexp_with_ref(struct bits *b, int32_t low, int32_t high, int32_t r)
{
    uint32_t mx = (uint32_t)(high - low), rr = (uint32_t)(r - low), v = decode_subexp(b, mx), x;
    if ((rr << 1) <= mx)
        x = inverse_recenter(rr, v);
    else
        x = mx - 1 - inverse_recenter(mx - 1 - rr, v);
    return (int32_t)x + low;
}

static void read_global_param(struct header *h, uint32_t type, int ref, int idx)
{
    int abs_bits = 12, prec_bits = 15;
    if (idx < 2) {
        if (type == AV1_WARP_TRANSLATION) {
            abs_bits = 9 - !h->f->high_precision_mv;
            prec_bits = 3 - !h->f->high_precision_mv;
        } else {
            abs_bits = 12;
            prec_bits = 6;
        }
    }
    int prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    int round = idx % 3 == 2 ? 1 << WARPEDMODEL_PREC_BITS : 0;
    int sub = idx % 3 == 2 ? 1 << prec_bits : 0;
    int32_t mx = 1 << abs_bits;
    int32_t r = (h->prev_gm[ref][idx] >> prec_diff) - sub;
    h->f->gm_params[ref][idx] = (decode_signed_subexp_with_ref(h->b, -mx, mx + 1, r) * (1 << prec_diff)) + round;
}

static void global_motion_params(struct header *h)
{
    struct av1_frame *f = h->f;
    reset_gm(f->gm_params);
    if (h->intra)
        return;
    for (int ref = 1; ref <= AV1_REFS_PER_FRAME; ref++) {
        uint32_t type = AV1_WARP_IDENTITY;
        if (bits_u(h->b, 1))
            type = bits_u(h->b, 1) ? AV1_WARP_ROTZOOM : bits_u(h->b, 1) ? AV1_WARP_TRANSLATION : AV1_WARP_AFFINE;
        f->gm_type[ref] = type;
        if (type >= AV1_WARP_ROTZOOM) {
            read_global_param(h, type, ref, 2);
            read_global_param(h, type, ref, 3);
            if (type == AV1_WARP_AFFINE) {
                read_global_param(h, type, ref, 4);
                read_global_param(h, type, ref, 5);
            } else {
                f->gm_params[ref][4] = -f->gm_params[ref][3];
                f->gm_params[ref][5] = f->gm_params[ref][2];
            }
        }
        if (type >= AV1_WARP_TRANSLATION) {
            read_global_param(h, type, ref, 0);
            read_global_param(h, type, ref, 1);
        }
    }
}

static void film_grain_points(struct bits *b, uint32_t n, uint8_t *value, uint8_t *scaling)
{
    for (uint32_t i = 0; i < n; i++) {
        value[i] = (uint8_t)bits_u(b, 8);
        scaling[i] = (uint8_t)bits_u(b, 8);
    }
}

static void film_grain_params(struct header *h)
{
    struct av1_frame *f = h->f;
    struct av1_grain *g = &f->grain;
    struct bits *b = h->b;
    *g = (struct av1_grain){ 0 };
    if (!h->s->film_grain_present || (!f->show && !f->showable) || !bits_u(b, 1))
        return;
    uint32_t seed = bits_u(b, 16);
    int update = f->type != AV1_INTER_FRAME || bits_u(b, 1);
    if (!update) {
        const struct av1_ref *r = &h->d->refs[bits_u(b, 3)];
        *g = r->grain;
        g->seed = seed;
        h->err |= !r->valid;
        return;
    }
    g->apply = 1;
    g->seed = seed;
    g->num_y = bits_u(b, 4);
    h->err |= g->num_y > 14;
    film_grain_points(b, g->num_y < 14 ? g->num_y : 14, g->y_value, g->y_scaling);
    g->chroma_scaling_from_luma = (int)bits_u(b, 1);
    if (!g->chroma_scaling_from_luma && g->num_y) {
        g->num_cb = bits_u(b, 4);
        h->err |= g->num_cb > 10;
        film_grain_points(b, g->num_cb < 10 ? g->num_cb : 10, g->cb_value, g->cb_scaling);
        g->num_cr = bits_u(b, 4);
        h->err |= g->num_cr > 10;
        film_grain_points(b, g->num_cr < 10 ? g->num_cr : 10, g->cr_value, g->cr_scaling);
    }
    g->scaling_shift = bits_u(b, 2) + 8;
    g->ar_lag = bits_u(b, 2);
    uint32_t luma = 2 * g->ar_lag * (g->ar_lag + 1), chroma = luma;
    if (g->num_y) {
        chroma++;
        for (uint32_t i = 0; i < luma; i++)
            g->ar_y[i] = (int)bits_u(b, 8) - 128;
    }
    if (g->chroma_scaling_from_luma || g->num_cb)
        for (uint32_t i = 0; i < chroma; i++)
            g->ar_cb[i] = (int)bits_u(b, 8) - 128;
    if (g->chroma_scaling_from_luma || g->num_cr)
        for (uint32_t i = 0; i < chroma; i++)
            g->ar_cr[i] = (int)bits_u(b, 8) - 128;
    g->ar_shift = bits_u(b, 2) + 6;
    g->grain_scale_shift = bits_u(b, 2);
    if (g->num_cb) {
        g->cb_mult = bits_u(b, 8);
        g->cb_luma_mult = bits_u(b, 8);
        g->cb_offset = bits_u(b, 9);
    }
    if (g->num_cr) {
        g->cr_mult = bits_u(b, 8);
        g->cr_luma_mult = bits_u(b, 8);
        g->cr_offset = bits_u(b, 9);
    }
    g->overlap = (int)bits_u(b, 1);
    g->clip_to_restricted_range = (int)bits_u(b, 1);
}

// 5.9.2. The frame header, from the first bit of the OBU payload.
static int uncompressed_header(const struct av1_dec *d, struct bits *b, uint32_t temporal_id, uint32_t spatial_id,
                               struct av1_frame *f)
{
    const struct av1_seq *s = &d->seq;
    struct header h = { .d = d, .s = s, .f = f, .b = b };
    uint32_t all_frames = (1u << AV1_REFS) - 1;
    uint32_t id_len = s->additional_frame_id_bits + s->delta_frame_id_bits;
    int override_size = 0;
    *f = (struct av1_frame){ 0 };

    if (s->reduced_still_picture) {
        f->type = AV1_KEY_FRAME;
        f->show = 1;
        f->error_resilient = 1;
    } else {
        if (bits_u(b, 1)) {  // show_existing_frame
            f->show_existing = 1;
            f->existing_idx = bits_u(b, 3);
            if (s->decoder_model_info && !s->equal_picture_interval)
                bits_u(b, (int)s->frame_presentation_time_length);
            if (s->frame_ids)
                bits_u(b, (int)id_len);
            const struct av1_ref *r = &d->refs[f->existing_idx];
            f->type = r->type;
            // a shown key frame refreshes every slot, which is state the kernel's tables don't follow
            return !r->valid || r->type == AV1_KEY_FRAME || b->over ? -1 : 0;
        }
        f->type = bits_u(b, 2);
        f->show = (int)bits_u(b, 1);
        if (f->show && s->decoder_model_info && !s->equal_picture_interval)
            bits_u(b, (int)s->frame_presentation_time_length);
        f->showable = f->show ? f->type != AV1_KEY_FRAME : (int)bits_u(b, 1);
        f->error_resilient = f->type == AV1_SWITCH_FRAME || (f->type == AV1_KEY_FRAME && f->show) ||
                             (int)bits_u(b, 1);
    }
    h.intra = f->type == AV1_KEY_FRAME || f->type == AV1_INTRA_ONLY_FRAME;
    f->disable_cdf_update = (int)bits_u(b, 1);
    f->screen_content_tools = s->screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS ? (int)bits_u(b, 1)
                                                                                    : (int)s->screen_content_tools;
    if (f->screen_content_tools)
        f->force_integer_mv = s->force_integer_mv == SELECT_INTEGER_MV ? (int)bits_u(b, 1) : (int)s->force_integer_mv;
    if (s->frame_ids)
        bits_u(b, (int)id_len);
    if (f->type == AV1_SWITCH_FRAME)
        override_size = 1;
    else if (!s->reduced_still_picture)
        override_size = (int)bits_u(b, 1);
    h.override_size = override_size;
    f->order_hint = bits_u(b, (int)s->order_hint_bits);
    f->primary_ref = h.intra || f->error_resilient ? AV1_PRIMARY_REF_NONE : bits_u(b, 3);
    if (s->decoder_model_info && bits_u(b, 1)) {  // buffer_removal_time_present_flag
        for (uint32_t i = 0; i < s->nops; i++) {
            if (!s->op_decoder_model[i])
                continue;
            uint32_t idc = s->op_idc[i];
            if (!idc || ((idc >> temporal_id & 1) && (idc >> (spatial_id + 8) & 1)))
                bits_u(b, (int)s->buffer_removal_time_length);
        }
    }
    f->refresh = f->type == AV1_SWITCH_FRAME || (f->type == AV1_KEY_FRAME && f->show) ? all_frames : bits_u(b, 8);
    if ((!h.intra || f->refresh != all_frames) && f->error_resilient && s->order_hint) {
        for (int i = 0; i < AV1_REFS; i++) {
            // a slot that doesn't hold what the stream says was lost
            h.err |= bits_u(b, (int)s->order_hint_bits) != d->refs[i].order_hint || !d->refs[i].valid;
        }
    }

    if (h.intra) {
        frame_size(&h);
        render_size(&h);
        if (f->screen_content_tools && f->upscaled_width == f->width)
            f->allow_intrabc = (int)bits_u(b, 1);
    } else {
        int short_signaling = s->order_hint && bits_u(b, 1);
        if (short_signaling) {
            uint32_t last = bits_u(b, 3), gold = bits_u(b, 3);
            set_frame_refs(&h, last, gold);
        }
        for (int i = 0; i < AV1_REFS_PER_FRAME; i++) {
            if (!short_signaling)
                f->ref_idx[i] = bits_u(b, 3);
            if (s->frame_ids)
                bits_u(b, (int)s->delta_frame_id_bits);
            h.err |= !d->refs[f->ref_idx[i]].valid;
        }
        if (h.err)
            return -1;
        if (override_size && !f->error_resilient) {
            frame_size_with_refs(&h);
        } else {
            frame_size(&h);
            render_size(&h);
        }
        f->high_precision_mv = f->force_integer_mv ? 0 : (int)bits_u(b, 1);
        f->interp_filter = bits_u(b, 1) ? SWITCHABLE_FILTER : bits_u(b, 2);
        f->motion_switchable = (int)bits_u(b, 1);
        f->use_ref_frame_mvs = !f->error_resilient && s->ref_frame_mvs ? (int)bits_u(b, 1) : 0;
    }
    f->disable_frame_end_update_cdf = s->reduced_still_picture || f->disable_cdf_update ? 1 : (int)bits_u(b, 1);

    // load_previous(): what a frame starts from without being told
    reset_gm(h.prev_gm);
    for (int i = 0; i < 8; i++)
        f->lf_ref_deltas[i] = default_ref_deltas[i];
    f->lf_delta_enabled = 1;
    if (f->primary_ref != AV1_PRIMARY_REF_NONE) {
        const struct av1_ref *r = ref_of(&h, f->primary_ref);
        memcpy(h.prev_gm, r->gm_params, sizeof(h.prev_gm));
        memcpy(f->lf_ref_deltas, r->lf_ref_deltas, sizeof(f->lf_ref_deltas));
        memcpy(f->lf_mode_deltas, r->lf_mode_deltas, sizeof(f->lf_mode_deltas));
        memcpy(f->seg_feature_enabled, r->seg_feature_enabled, sizeof(f->seg_feature_enabled));
        memcpy(f->seg_feature_data, r->seg_feature_data, sizeof(f->seg_feature_data));
    }

    tile_info(&h);
    quantization_params(&h);
    segmentation_params(&h);
    if (f->base_qindex > 0 && (f->delta_q_present = (int)bits_u(b, 1)))
        f->delta_q_res = bits_u(b, 2);
    if (f->delta_q_present && !f->allow_intrabc && (f->delta_lf_present = (int)bits_u(b, 1))) {
        f->delta_lf_res = bits_u(b, 2);
        f->delta_lf_multi = (int)bits_u(b, 1);
    }

    f->coded_lossless = 1;
    for (int i = 0; i < 8; i++) {
        int lossless = !qindex(f, i) && !f->dq_y_dc && !f->dq_u_dc && !f->dq_u_ac && !f->dq_v_dc && !f->dq_v_ac;
        f->coded_lossless &= lossless;
    }
    f->all_lossless = f->coded_lossless && f->width == f->upscaled_width;
    loop_filter_params(&h);
    cdef_params(&h);
    lr_params(&h);
    f->tx_mode = f->coded_lossless ? AV1_TX_ONLY_4X4 : bits_u(b, 1) ? AV1_TX_SELECT : AV1_TX_LARGEST;
    f->reference_select = h.intra ? 0 : (int)bits_u(b, 1);
    skip_mode_params(&h);
    f->allow_warped_motion = h.intra || f->error_resilient || !s->warped_motion ? 0 : (int)bits_u(b, 1);
    f->reduced_tx_set = (int)bits_u(b, 1);
    global_motion_params(&h);
    film_grain_params(&h);
    return h.err || b->over ? -1 : 0;
}

struct obu {
    uint32_t type, temporal_id, spatial_id;
    int extension;
    size_t payload, size, next;
};

static int parse_obu(const uint8_t *s, size_t len, size_t pos, struct obu *o)
{
    uint32_t head = s[pos];
    o->type = head >> 3 & 15;
    o->extension = head >> 2 & 1;
    o->temporal_id = o->spatial_id = 0;
    size_t at = pos + 1;
    if (head & 0x80 || (o->extension && at >= len))
        return -1;
    if (o->extension) {
        o->temporal_id = s[at] >> 5;
        o->spatial_id = s[at] >> 3 & 3;
        at++;
    }
    if (head & 2) {
        uint64_t size = 0;
        for (int i = 0; i < 8; i++) {
            if (at >= len)
                return -1;
            uint8_t byte = s[at++];
            size |= (uint64_t)(byte & 0x7f) << (7 * i);
            if (!(byte & 0x80))
                break;
        }
        if (size > len - at)
            return -1;
        o->size = (size_t)size;
    } else {
        o->size = len - at;
    }
    o->payload = at;
    o->next = at + o->size;
    return 0;
}

// 5.11.1: where the tiles of a tile group lie. The group's tile data, size fields and all, is appended to the
// picture's; *next is the first tile the group should hold.
static int tile_group(struct av1_picture *pic, const uint8_t *s, size_t at, size_t size, uint32_t *next)
{
    const struct av1_frame *f = &pic->hdr;
    struct bits b;
    bits_init_raw(&b, s + at, size);
    uint32_t start = 0, end = pic->ntiles - 1;
    if (pic->ntiles > 1 && bits_u(&b, 1)) {
        int tile_bits = (int)(f->tile_cols_log2 + f->tile_rows_log2);
        start = bits_u(&b, tile_bits);
        end = bits_u(&b, tile_bits);
    }
    size_t header = (size_t)(bits_pos(&b) + 7) / 8;
    if (b.over || header > size || start != *next || end < start || end >= pic->ntiles ||
        pic->ngroups >= AV1_MAX_TILE_GROUPS)
        return -1;
    size_t data = at + header, remaining = size - header, off = 0;
    for (uint32_t t = start; t <= end; t++) {
        size_t tile = remaining - off;
        if (t != end) {
            if (f->tile_size_bytes > remaining - off)
                return -1;
            tile = 0;
            for (uint32_t i = 0; i < f->tile_size_bytes; i++)
                tile |= (size_t)s[data + off + i] << (8 * i);
            off += f->tile_size_bytes;
            tile++;
            if (tile > remaining - off)
                return -1;
        }
        pic->tile_start[t] = pic->bytes + (uint32_t)off;
        pic->tile_end[t] = pic->bytes + (uint32_t)(off + tile);
        off += tile;
    }
    pic->group_pos[pic->ngroups] = (uint32_t)data;
    pic->group_len[pic->ngroups] = (uint32_t)remaining;
    pic->ngroups++;
    pic->bytes += (uint32_t)remaining;
    *next = end + 1;
    return 0;
}

// the slot no reference holds
static uint32_t free_slot(const struct av1_dec *d)
{
    uint32_t used = 0;
    for (int i = 0; i < AV1_REFS; i++)
        if (d->refs[i].valid)
            used |= 1u << d->refs[i].slot;
    uint32_t slot = 0;
    while (used >> slot & 1)
        slot++;
    return slot;
}

int av1_next(struct av1_dec *d, const uint8_t *s, size_t len, size_t *pos, struct av1_picture *pic)
{
    uint32_t next_tile = 0;
    int have_header = 0;
    *pic = (struct av1_picture){ 0 };
    while (*pos < len) {
        struct obu o;
        if (parse_obu(s, len, *pos, &o))
            return -1;
        *pos = o.next;
        uint32_t idc = d->seq.op_idc[0];
        if (o.type != OBU_SEQUENCE_HEADER && o.type != OBU_TEMPORAL_DELIMITER && idc && o.extension &&
            !((idc >> o.temporal_id & 1) && (idc >> (o.spatial_id + 8) & 1)))
            continue;

        size_t tiles_at = o.payload, tiles_size = o.size;
        if (o.type == OBU_SEQUENCE_HEADER) {
            if (parse_sequence(&d->seq, s + o.payload, o.size))
                return -1;
        } else if ((o.type == OBU_FRAME_HEADER || o.type == OBU_FRAME) && !have_header) {
            if (!d->seq.valid)
                return -1;
            struct bits b;
            bits_init_raw(&b, s + o.payload, o.size);
            if (uncompressed_header(d, &b, o.temporal_id, o.spatial_id, &pic->hdr))
                return -1;
            pic->seq = &d->seq;
            pic->index = d->pictures;
            if (pic->hdr.show_existing) {
                const struct av1_ref *r = &d->refs[pic->hdr.existing_idx];
                pic->existing = pic->output = 1;
                pic->slot = r->slot;
                pic->hdr.width = r->width;
                pic->hdr.upscaled_width = r->upscaled_width;
                pic->hdr.height = r->height;
                return 1;
            }
            have_header = 1;
            pic->output = pic->hdr.show;
            pic->slot = free_slot(d);
            pic->ntiles = pic->hdr.tile_cols * pic->hdr.tile_rows;
            size_t header = (size_t)(bits_pos(&b) + 7) / 8;
            tiles_at = o.payload + header;
            tiles_size = o.size - header;
            if (o.type == OBU_FRAME_HEADER)
                continue;
        } else if (o.type != OBU_TILE_GROUP || !have_header) {
            continue;
        }
        if (o.type == OBU_FRAME || o.type == OBU_TILE_GROUP) {
            if (tile_group(pic, s, tiles_at, tiles_size, &next_tile))
                return -1;
            if (next_tile == pic->ntiles)
                return 1;
        }
    }
    return 0;
}

void av1_done(struct av1_dec *d, const struct av1_picture *pic)
{
    const struct av1_frame *f = &pic->hdr;
    d->pictures++;
    if (pic->existing)
        return;
    struct av1_ref r = {
        .valid = 1,
        .slot = pic->slot,
        .width = f->width,
        .height = f->height,
        .upscaled_width = f->upscaled_width,
        .mi_cols = f->mi_cols,
        .mi_rows = f->mi_rows,
        .render_width = f->render_width,
        .render_height = f->render_height,
        .type = f->type,
        .order_hint = f->order_hint,
        .showable = f->showable,
        .grain = f->grain,
    };
    for (int i = 0; i < AV1_REFS_PER_FRAME; i++)
        r.order_hints[i + 1] = f->type == AV1_KEY_FRAME || f->type == AV1_INTRA_ONLY_FRAME
                                   ? 0
                                   : d->refs[f->ref_idx[i]].order_hint;
    memcpy(r.gm_params, f->gm_params, sizeof(r.gm_params));
    memcpy(r.lf_ref_deltas, f->lf_ref_deltas, sizeof(r.lf_ref_deltas));
    memcpy(r.lf_mode_deltas, f->lf_mode_deltas, sizeof(r.lf_mode_deltas));
    memcpy(r.seg_feature_enabled, f->seg_feature_enabled, sizeof(r.seg_feature_enabled));
    memcpy(r.seg_feature_data, f->seg_feature_data, sizeof(r.seg_feature_data));
    for (int i = 0; i < AV1_REFS; i++)
        if (f->refresh >> i & 1)
            d->refs[i] = r;
}

void av1_tiles(const struct av1_picture *pic, const uint8_t *s, uint8_t *out)
{
    for (uint32_t i = 0; i < pic->ngroups; i++) {
        memcpy(out, s + pic->group_pos[i], pic->group_len[i]);
        out += pic->group_len[i];
    }
}
