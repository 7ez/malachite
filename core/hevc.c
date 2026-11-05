// H.265 parameter sets, slice headers, picture order count and the reference picture set, ITU-T H.265 7.3, 8.3.1, 8.3.2
#include "hevc.h"
#include "bitreader.h"

// the default 8x8 lists of Table 7-6, in raster order
static const uint8_t default_intra[64] = {
    16, 16, 16, 16, 17, 18, 21, 24,
    16, 16, 16, 16, 17, 19, 22, 25,
    16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36,
    17, 17, 20, 24, 30, 35, 41, 47,
    18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88,
    24, 25, 29, 36, 47, 65, 88, 115,
};

static const uint8_t default_inter[64] = {
    16, 16, 16, 16, 17, 18, 20, 24,
    16, 16, 16, 17, 18, 20, 24, 25,
    16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33,
    17, 18, 20, 24, 25, 28, 33, 41,
    18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71,
    24, 25, 28, 33, 41, 54, 71, 91,
};

static uint32_t ceil_log2(uint32_t n)
{
    uint32_t bits = 0;
    while ((1u << bits) < n)
        bits++;
    return bits;
}

// 6.5.3: the raster position of each coefficient of a size x size list in coding order
static void diag_scan(int size, uint8_t *scan)
{
    int i = 0, x = 0, y = 0;
    while (i < size * size) {
        for (; y >= 0; y--, x++)
            if (x < size && y < size)
                scan[i++] = (uint8_t)(y * size + x);
        y = x;
        x = 0;
    }
}

static void default_list(struct h265_scaling *sl, int size, int matrix)
{
    if (!size) {
        memset(sl->list[0][matrix], 16, 16);
        return;
    }
    memcpy(sl->list[size][matrix], matrix < 3 ? default_intra : default_inter, 64);
    if (size > 1)
        sl->dc[size - 2][matrix] = 16;
}

static void default_scaling(struct h265_scaling *sl)
{
    for (int size = 0; size < 4; size++)
        for (int m = 0; m < 6; m += size == 3 ? 3 : 1)
            default_list(sl, size, m);
}

// 7.3.4
static int scaling_list_data(struct bits *b, struct h265_scaling *sl)
{
    for (int size = 0; size < 4; size++) {
        uint8_t scan[64];
        int coefs = size ? 64 : 16;
        diag_scan(size ? 8 : 4, scan);
        for (int m = 0; m < 6; m += size == 3 ? 3 : 1) {
            uint8_t *list = sl->list[size][m];
            if (!bits_u(b, 1)) {
                uint32_t delta = bits_ue(b) * (size == 3 ? 3 : 1);
                if (delta > (uint32_t)m)
                    return -1;
                if (!delta) {
                    default_list(sl, size, m);
                } else {
                    memcpy(list, sl->list[size][m - delta], (size_t)coefs);
                    if (size > 1)
                        sl->dc[size - 2][m] = sl->dc[size - 2][m - delta];
                }
                continue;
            }
            int next = 8;
            if (size > 1) {
                next = bits_se(b) + 8;
                sl->dc[size - 2][m] = (uint8_t)next;
            }
            for (int i = 0; i < coefs; i++) {
                next = (next + bits_se(b) + 256) % 256;
                list[scan[i]] = (uint8_t)next;
            }
        }
    }
    return b->over ? -1 : 0;
}

// 7.3.7 and 7.4.8: set idx of num sets, predicted from an earlier one of sets; ref_num_deltas is how many pictures
// that one has, 0 when this one isn't predicted
static int st_ref_pic_set(struct bits *b, const struct h265_rps *sets, uint32_t idx, uint32_t num,
                          struct h265_rps *out, uint32_t *ref_num_deltas)
{
    *out = (struct h265_rps){ 0 };
    if (ref_num_deltas)
        *ref_num_deltas = 0;
    if (idx && bits_u(b, 1)) {
        uint32_t delta_idx = idx == num ? bits_ue(b) + 1 : 1;
        if (delta_idx > idx)
            return -1;
        const struct h265_rps *ref = &sets[idx - delta_idx];
        uint32_t sign = bits_u(b, 1);
        int32_t abs_delta = (int32_t)bits_ue(b) + 1;
        int32_t delta_rps = sign ? -abs_delta : abs_delta;
        uint32_t n = ref->neg + (uint32_t)ref->pos;
        uint8_t used[33], use_delta[33];
        for (uint32_t j = 0; j <= n; j++) {
            used[j] = (uint8_t)bits_u(b, 1);
            use_delta[j] = used[j] ? 1 : (uint8_t)bits_u(b, 1);
        }
        if (b->over)
            return -1;
        if (ref_num_deltas)
            *ref_num_deltas = n;

        // 7-61: the negative pictures, the reference set's positive ones first, nearest the current picture first
        int i = 0;
        for (int j = ref->pos - 1; j >= 0; j--) {
            int32_t d = ref->delta[1][j] + delta_rps;
            if (d < 0 && use_delta[ref->neg + j] && i < 16) {
                out->delta[0][i] = (int16_t)d;
                out->used[0][i++] = used[ref->neg + j];
            }
        }
        if (delta_rps < 0 && use_delta[n] && i < 16) {
            out->delta[0][i] = (int16_t)delta_rps;
            out->used[0][i++] = used[n];
        }
        for (int j = 0; j < ref->neg; j++) {
            int32_t d = ref->delta[0][j] + delta_rps;
            if (d < 0 && use_delta[j] && i < 16) {
                out->delta[0][i] = (int16_t)d;
                out->used[0][i++] = used[j];
            }
        }
        out->neg = (uint8_t)i;

        // 7-62
        i = 0;
        for (int j = ref->neg - 1; j >= 0; j--) {
            int32_t d = ref->delta[0][j] + delta_rps;
            if (d > 0 && use_delta[j] && i < 16) {
                out->delta[1][i] = (int16_t)d;
                out->used[1][i++] = used[j];
            }
        }
        if (delta_rps > 0 && use_delta[n] && i < 16) {
            out->delta[1][i] = (int16_t)delta_rps;
            out->used[1][i++] = used[n];
        }
        for (int j = 0; j < ref->pos; j++) {
            int32_t d = ref->delta[1][j] + delta_rps;
            if (d > 0 && use_delta[ref->neg + j] && i < 16) {
                out->delta[1][i] = (int16_t)d;
                out->used[1][i++] = used[ref->neg + j];
            }
        }
        out->pos = (uint8_t)i;
        return out->neg + out->pos > 16 ? -1 : 0;
    }

    uint32_t neg = bits_ue(b), pos = bits_ue(b);
    if (neg > 16 || pos > 16 || neg + pos > 16)
        return -1;
    out->neg = (uint8_t)neg;
    out->pos = (uint8_t)pos;
    for (int list = 0; list < 2; list++) {
        int32_t poc = 0;
        for (uint32_t i = 0; i < (list ? pos : neg); i++) {
            int32_t step = (int32_t)bits_ue(b) + 1;
            poc += list ? step : -step;
            if (poc > 32767 || poc < -32768)
                return -1;
            out->delta[list][i] = (int16_t)poc;
            out->used[list][i] = (uint8_t)bits_u(b, 1);
        }
    }
    return b->over ? -1 : 0;
}

static void profile_tier_level(struct bits *b, uint32_t sub_layers, uint32_t *profile)
{
    bits_u(b, 3);  // profile space, tier
    *profile = bits_u(b, 5);
    bits_u(b, 32);  // compatibility flags
    bits_u(b, 16);  // source flags, the reserved bits
    bits_u(b, 16);
    bits_u(b, 16);
    bits_u(b, 8);  // level
    uint32_t profile_present = 0, level_present = 0;
    for (uint32_t i = 0; i < sub_layers; i++) {
        profile_present |= bits_u(b, 1) << i;
        level_present |= bits_u(b, 1) << i;
    }
    for (uint32_t i = sub_layers; sub_layers && i < 8; i++)
        bits_u(b, 2);
    for (uint32_t i = 0; i < sub_layers; i++) {
        if (profile_present >> i & 1) {
            bits_u(b, 32);
            bits_u(b, 32);
            bits_u(b, 24);
        }
        if (level_present >> i & 1)
            bits_u(b, 8);
    }
}

// into d->sps[id] directly: a set with its short-term sets is too big for the stack
static int parse_sps(struct h265_dec *d, struct bits *b)
{
    bits_u(b, 4);  // the VPS
    uint32_t sub_layers = bits_u(b, 3);
    bits_u(b, 1);  // temporal_id_nesting_flag
    uint32_t profile;
    profile_tier_level(b, sub_layers, &profile);
    uint32_t id = bits_ue(b);
    if (b->over || id >= H265_MAX_SPS)
        return -1;
    struct h265_sps *s = &d->sps[id];
    memset(s, 0, sizeof(*s));
    s->profile = profile;
    uint32_t chroma_format = bits_ue(b);
    if (chroma_format == 3)
        bits_u(b, 1);  // separate_colour_plane_flag
    s->width = bits_ue(b);
    s->height = bits_ue(b);
    if (bits_u(b, 1)) {
        s->crop_left = bits_ue(b) * 2;
        s->crop_right = bits_ue(b) * 2;
        s->crop_top = bits_ue(b) * 2;
        s->crop_bottom = bits_ue(b) * 2;
    }
    s->bit_depth = bits_ue(b) + 8;
    uint32_t bit_depth_chroma = bits_ue(b) + 8;
    s->log2_max_poc_lsb = bits_ue(b) + 4;
    int ordering_info = bits_u(b, 1);
    for (uint32_t i = ordering_info ? 0 : sub_layers; i <= sub_layers; i++) {
        s->max_dec_pic_buffering = bits_ue(b) + 1;
        bits_ue(b);  // sps_max_num_reorder_pics
        bits_ue(b);  // sps_max_latency_increase_plus1
    }
    s->log2_min_cb = bits_ue(b) + 3;
    s->log2_ctb = s->log2_min_cb + bits_ue(b);
    s->log2_min_tb = bits_ue(b) + 2;
    s->log2_max_tb = s->log2_min_tb + bits_ue(b);
    s->depth_inter = bits_ue(b);
    s->depth_intra = bits_ue(b);
    s->scaling_list_enabled = bits_u(b, 1);
    default_scaling(&s->scaling);
    if (s->scaling_list_enabled && bits_u(b, 1) && scaling_list_data(b, &s->scaling)) {
        log_msg("h265: SPS %u has scaling lists that don't parse\n", id);
        return -1;
    }
    s->amp = bits_u(b, 1);
    s->sao = bits_u(b, 1);
    s->pcm = bits_u(b, 1);
    if (s->pcm) {
        s->pcm_depth_luma = bits_u(b, 4) + 1;
        s->pcm_depth_chroma = bits_u(b, 4) + 1;
        s->log2_min_pcm = bits_ue(b) + 3;
        s->log2_max_pcm = s->log2_min_pcm + bits_ue(b);
        s->pcm_loop_filter_disabled = bits_u(b, 1);
    }
    s->nrps = bits_ue(b);
    if (b->over || s->nrps > H265_MAX_RPS)
        return -1;
    for (uint32_t i = 0; i < s->nrps; i++)
        if (st_ref_pic_set(b, s->rps, i, s->nrps, &s->rps[i], NULL))
            return -1;
    s->long_term_present = bits_u(b, 1);
    if (s->long_term_present) {
        s->nlt = bits_ue(b);
        if (s->nlt > H265_MAX_LT)
            return -1;
        for (uint32_t i = 0; i < s->nlt; i++) {
            s->lt_poc_lsb[i] = bits_u(b, (int)s->log2_max_poc_lsb);
            s->lt_used[i] = (uint8_t)bits_u(b, 1);
        }
    }
    s->temporal_mvp = bits_u(b, 1);
    s->strong_intra_smoothing = bits_u(b, 1);
    // what follows (the VUI and the extensions) is nothing Main and Main10 streams need

    if (b->over || s->width > 8192 || s->height > 8192 || s->width < 16 || s->height < 16 ||
        s->crop_left + s->crop_right >= s->width || s->crop_top + s->crop_bottom >= s->height ||
        s->log2_max_poc_lsb > 16 || s->max_dec_pic_buffering > H265_DPB || s->log2_ctb < 4 || s->log2_ctb > 6 ||
        s->log2_min_tb < 2 || s->log2_max_tb > 5 || s->width % (1u << s->log2_min_cb) ||
        s->height % (1u << s->log2_min_cb) || s->depth_inter > 4 || s->depth_intra > 4) {
        log_msg("h265: SPS %u is out of range\n", id);
        return -1;
    }
    if (profile < 1 || profile > 3 || chroma_format != 1 || (s->bit_depth != 8 && s->bit_depth != 10) ||
        bit_depth_chroma != s->bit_depth) {
        log_msg("h265: SPS %u is profile %u, chroma format %u, %u-bit: not Main or Main10\n", id, profile,
                chroma_format, s->bit_depth);
        return -1;
    }
    s->valid = 1;
    return 0;
}

// into d->pps[id] directly, as the SPS
static int parse_pps(struct h265_dec *d, struct bits *b)
{
    uint32_t id = bits_ue(b), sps = bits_ue(b);
    if (b->over || id >= H265_MAX_PPS || sps >= H265_MAX_SPS)
        return -1;
    struct h265_pps *p = &d->pps[id];
    memset(p, 0, sizeof(*p));
    p->sps = sps;
    p->dependent_slices = bits_u(b, 1);
    p->output_flag_present = bits_u(b, 1);
    p->extra_slice_header_bits = bits_u(b, 3);
    p->sign_data_hiding = bits_u(b, 1);
    p->cabac_init_present = bits_u(b, 1);
    p->num_ref_idx_l0 = bits_ue(b) + 1;
    p->num_ref_idx_l1 = bits_ue(b) + 1;
    p->init_qp = bits_se(b);
    p->constrained_intra_pred = bits_u(b, 1);
    p->transform_skip = bits_u(b, 1);
    p->cu_qp_delta = bits_u(b, 1);
    if (p->cu_qp_delta)
        p->diff_cu_qp_delta_depth = bits_ue(b);
    p->cb_qp_offset = bits_se(b);
    p->cr_qp_offset = bits_se(b);
    p->slice_chroma_qp_offsets_present = bits_u(b, 1);
    p->weighted_pred = bits_u(b, 1);
    p->weighted_bipred = bits_u(b, 1);
    p->transquant_bypass = bits_u(b, 1);
    p->tiles = bits_u(b, 1);
    p->entropy_coding_sync = bits_u(b, 1);
    p->tile_cols = p->tile_rows = 1;
    p->uniform_spacing = 1;
    if (p->tiles) {
        p->tile_cols = bits_ue(b) + 1;
        p->tile_rows = bits_ue(b) + 1;
        if (p->tile_cols > H265_MAX_TILE_COLS || p->tile_rows > H265_MAX_TILE_ROWS) {
            log_msg("h265: PPS %u has %ux%u tiles\n", id, p->tile_cols, p->tile_rows);
            return -1;
        }
        p->uniform_spacing = bits_u(b, 1);
        if (!p->uniform_spacing) {
            for (uint32_t i = 0; i < p->tile_cols - 1; i++)
                p->col_width[i] = bits_ue(b) + 1;
            for (uint32_t i = 0; i < p->tile_rows - 1; i++)
                p->row_height[i] = bits_ue(b) + 1;
        }
        p->loop_filter_across_tiles = bits_u(b, 1);
    }
    p->loop_filter_across_slices = bits_u(b, 1);
    p->deblocking_control = bits_u(b, 1);
    if (p->deblocking_control) {
        p->deblocking_override = bits_u(b, 1);
        p->deblocking_disabled = bits_u(b, 1);
        if (!p->deblocking_disabled) {
            p->beta_offset_div2 = bits_se(b);
            p->tc_offset_div2 = bits_se(b);
        }
    }
    p->scaling_list_present = bits_u(b, 1);
    if (p->scaling_list_present && scaling_list_data(b, &p->scaling)) {
        log_msg("h265: PPS %u has scaling lists that don't parse\n", id);
        return -1;
    }
    p->lists_modification = bits_u(b, 1);
    p->log2_parallel_merge_level = bits_ue(b) + 2;
    p->slice_header_extension = bits_u(b, 1);
    if (b->over || p->num_ref_idx_l0 > 15 || p->num_ref_idx_l1 > 15 || p->log2_parallel_merge_level > 6 ||
        p->diff_cu_qp_delta_depth > 3)
        return -1;
    p->valid = 1;
    return 0;
}

// the slice header up to slice_temporal_mvp_enabled_flag, which is all the driver needs of the first slice segment
// of a picture
static int parse_slice(const struct h265_dec *d, struct bits *b, uint32_t nal_type, uint32_t temporal_id,
                       struct h265_slice *h)
{
    *h = (struct h265_slice){ .nal_type = nal_type, .temporal_id = temporal_id, .pic_output = 1 };
    bits_u(b, 1);  // first_slice_segment_in_pic_flag
    if (nal_type >= 16 && nal_type <= 23)
        h->no_output_of_prior_pics = bits_u(b, 1);
    h->pps = bits_ue(b);
    if (h->pps >= H265_MAX_PPS || !d->pps[h->pps].valid)
        return -1;
    const struct h265_pps *p = &d->pps[h->pps];
    const struct h265_sps *s = &d->sps[p->sps];
    if (!s->valid)
        return -1;
    for (uint32_t i = 0; i < p->extra_slice_header_bits; i++)
        bits_u(b, 1);
    h->type = bits_ue(b);
    uint64_t start = bits_pos(b);
    if (p->output_flag_present)
        h->pic_output = bits_u(b, 1);
    if (nal_type != 19 && nal_type != 20) {
        h->poc_lsb = bits_u(b, (int)s->log2_max_poc_lsb);
        if (!bits_u(b, 1)) {
            uint64_t rps_start = bits_pos(b);
            if (st_ref_pic_set(b, s->rps, s->nrps, s->nrps, &h->rps, &h->rps_ref_num_deltas))
                return -1;
            h->rps_bits = (uint32_t)(bits_pos(b) - rps_start);
        } else {
            uint32_t idx = s->nrps > 1 ? bits_u(b, (int)ceil_log2(s->nrps)) : 0;
            if (idx >= s->nrps)
                return -1;
            h->rps = s->rps[idx];
        }
        if (s->long_term_present) {
            uint32_t from_sps = s->nlt ? bits_ue(b) : 0, coded = bits_ue(b);
            if (from_sps > s->nlt || from_sps + coded > H265_MAX_LT)
                return -1;
            h->nlt = from_sps + coded;
            for (uint32_t i = 0; i < h->nlt; i++) {
                if (i < from_sps) {
                    uint32_t idx = s->nlt > 1 ? bits_u(b, (int)ceil_log2(s->nlt)) : 0;
                    if (idx >= s->nlt)
                        return -1;
                    h->lt_poc_lsb[i] = s->lt_poc_lsb[idx];
                    h->lt_used[i] = s->lt_used[idx];
                } else {
                    h->lt_poc_lsb[i] = bits_u(b, (int)s->log2_max_poc_lsb);
                    h->lt_used[i] = (uint8_t)bits_u(b, 1);
                }
                h->lt_msb_present[i] = (uint8_t)bits_u(b, 1);
                uint32_t cycle = h->lt_msb_present[i] ? bits_ue(b) : 0;
                h->lt_msb_cycle[i] = i && i != from_sps ? cycle + h->lt_msb_cycle[i - 1] : cycle;
            }
        }
    }
    h->skip_bits = (uint32_t)(bits_pos(b) - start);
    return b->over ? -1 : 0;
}

// 6.5.1: n tiles over total CTBs, the explicit sizes of all but the last or even ones
static int split(uint32_t total, uint32_t n, int uniform, const uint32_t *sizes, uint32_t *out)
{
    uint32_t sum = 0;
    if (n > total)
        return -1;
    for (uint32_t i = 0; i < n - 1; i++) {
        out[i] = uniform ? (i + 1) * total / n - i * total / n : sizes[i];
        sum += out[i];
    }
    if (sum >= total)
        return -1;
    out[n - 1] = total - sum;
    return 0;
}

int h265_tiles(const struct h265_sps *s, const struct h265_pps *p, uint32_t *cols, uint32_t *rows)
{
    uint32_t ctb = 1u << s->log2_ctb;
    return split((s->width + ctb - 1) >> s->log2_ctb, p->tile_cols, p->uniform_spacing, p->col_width, cols) ||
           split((s->height + ctb - 1) >> s->log2_ctb, p->tile_rows, p->uniform_spacing, p->row_height, rows);
}

void h265_init(struct h265_dec *d)
{
    memset(d, 0, sizeof(*d));
}

static int is_irap(uint32_t type)
{
    return type >= 16 && type <= 23;
}

static int is_vcl(uint32_t type)
{
    return type <= 9 || (type >= 16 && type <= 21);
}

// 8.3.1
static void picture_order(struct h265_dec *d, struct h265_picture *pic)
{
    const struct h265_slice *h = &pic->hdr;
    int32_t max = 1 << pic->sps->log2_max_poc_lsb, lsb = (int32_t)h->poc_lsb, msb = 0;
    if (!(pic->irap && pic->no_rasl)) {
        msb = d->prev_poc_msb;
        if (lsb < d->prev_poc_lsb && d->prev_poc_lsb - lsb >= max / 2)
            msb += max;
        else if (lsb > d->prev_poc_lsb && lsb - d->prev_poc_lsb > max / 2)
            msb -= max;
    }
    pic->poc = msb + lsb;
    // the previous picture of the lowest temporal layer that isn't a leading one or a sub-layer non-reference one
    if (!h->temporal_id && !(h->nal_type >= 6 && h->nal_type <= 9) && !(h->nal_type <= 14 && !(h->nal_type & 1))) {
        d->prev_poc_lsb = lsb;
        d->prev_poc_msb = msb;
    }
}

static int find_ref(const struct h265_dec *d, int long_term, int32_t poc, int32_t mask)
{
    for (int i = 0; i < H265_DPB; i++) {
        const struct h265_ref *r = &d->dpb[i];
        if (r->used && (long_term || !r->long_term) && (r->poc & mask) == poc)
            return i;
    }
    return -1;
}

// 8.3.2: the reference picture set of the picture; every other picture stops being a reference. A reference
// picture that isn't there is only a stream error for a picture that predicts from it.
static int reference_picture_set(struct h265_dec *d, struct h265_picture *pic)
{
    const struct h265_slice *h = &pic->hdr;
    int32_t max_lsb = 1 << pic->sps->log2_max_poc_lsb;
    int keep[H265_DPB] = { 0 };
    pic->nbefore = pic->nafter = pic->nlt = 0;
    if (h->nlt + h->rps.neg + h->rps.pos > H265_DPB)
        return -1;

    for (uint32_t i = 0; i < h->nlt; i++) {
        int32_t poc = (int32_t)h->lt_poc_lsb[i], mask = max_lsb - 1;
        if (h->lt_msb_present[i]) {
            poc += pic->poc - (int32_t)h->lt_msb_cycle[i] * max_lsb - (pic->poc & (max_lsb - 1));
            mask = -1;
        }
        int j = find_ref(d, 1, poc, mask);
        if (j < 0) {
            if (h->lt_used[i] && !pic->irap)
                return -1;
            continue;
        }
        keep[j] = 1;
        d->dpb[j].long_term = 1;
        if (h->lt_used[i])
            pic->lt[pic->nlt++] = (uint8_t)j;
    }
    for (int list = 0; list < 2; list++) {
        for (uint32_t i = 0; i < (list ? h->rps.pos : h->rps.neg); i++) {
            int j = find_ref(d, 0, pic->poc + h->rps.delta[list][i], -1);
            if (j < 0) {
                if (h->rps.used[list][i] && !pic->irap)
                    return -1;
                continue;
            }
            keep[j] = 1;
            if (h->rps.used[list][i]) {
                if (list)
                    pic->after[pic->nafter++] = (uint8_t)j;
                else
                    pic->before[pic->nbefore++] = (uint8_t)j;
            }
        }
    }
    for (int i = 0; i < H265_DPB; i++)
        if (!keep[i])
            d->dpb[i].used = 0;
    return 0;
}

static int pick_slot(const struct h265_dec *d, uint32_t *slot)
{
    uint32_t busy = 0;
    for (int i = 0; i < H265_DPB; i++)
        if (d->dpb[i].used)
            busy |= 1u << d->dpb[i].slot;
    for (uint32_t i = 0; i < H265_DPB; i++) {
        if (!(busy & 1u << i)) {
            *slot = i;
            return 0;
        }
    }
    return -1;
}

int h265_next(struct h265_dec *d, const uint8_t *s, size_t len, size_t *pos, struct h265_picture *pic)
{
    size_t at = *pos, start, size;
    int ret = 0;
    pic->nslices = 0;
    while (bits_next_nal(s, len, &at, &start, &size)) {
        if (size < 3)
            continue;
        uint32_t type = s[start] >> 1 & 0x3f, layer = (s[start] & 1) << 5 | s[start + 1] >> 3;
        uint32_t temporal_id = (s[start + 1] & 7) - 1;
        struct bits b;
        bits_init(&b, s + start + 2, size - 2);
        if (layer) {
            *pos = at;
            continue;
        }
        if (is_vcl(type)) {
            int first = s[start + 2] >> 7;
            if (pic->nslices && first)
                break;  // the next picture's
            if (!pic->nslices) {
                struct h265_slice h;
                if (!first || (!is_irap(type) && !d->started) || ((type == 8 || type == 9) && d->skip_rasl)) {
                    *pos = at;
                    continue;  // nothing to predict from, or nothing to find it by
                }
                if (parse_slice(d, &b, type, temporal_id, &h)) {
                    ret = -1;
                    break;
                }
                pic->hdr = h;
                pic->bytes = 0;
            }
            if (pic->nslices == H265_MAX_SLICES) {
                ret = -1;
                break;
            }
            pic->slice_pos[pic->nslices] = (uint32_t)start;
            pic->slice_len[pic->nslices++] = (uint32_t)size;
            pic->bytes += (uint32_t)size + 3;
        } else if (pic->nslices && ((type >= 32 && type <= 37) || type == 39 || (type >= 41 && type <= 44) ||
                                    (type >= 48 && type <= 55))) {
            break;  // a new access unit, or the end of the sequence
        } else if (type == 33 && parse_sps(d, &b)) {
            ret = -1;
            break;
        } else if (type == 34 && parse_pps(d, &b)) {
            ret = -1;
            break;
        } else if (type == 36 || type == 37) {
            d->eos = 1;
        }
        *pos = at;
    }
    if (ret || !pic->nslices)
        return ret;

    pic->pps = &d->pps[pic->hdr.pps];
    pic->sps = &d->sps[pic->pps->sps];
    uint32_t cols[H265_MAX_TILE_COLS], rows[H265_MAX_TILE_ROWS];
    if (h265_tiles(pic->sps, pic->pps, cols, rows)) {
        log_msg("h265: PPS %u doesn't fit its picture\n", pic->hdr.pps);
        return -1;
    }
    pic->irap = is_irap(pic->hdr.nal_type);
    pic->idr = pic->hdr.nal_type == 19 || pic->hdr.nal_type == 20;
    pic->no_rasl = pic->irap && (pic->hdr.nal_type <= 20 || !d->started || d->eos);
    pic->output = pic->hdr.pic_output;
    if (pic->irap) {
        d->started = 1;
        d->eos = 0;
        d->skip_rasl = pic->no_rasl;
        if (pic->no_rasl) {
            d->idrs++;
            for (int i = 0; i < H265_DPB; i++)
                d->dpb[i].used = 0;
        }
    }
    picture_order(d, pic);
    if (pic->idr) {
        for (int i = 0; i < H265_DPB; i++)
            d->dpb[i].used = 0;
        pic->nbefore = pic->nafter = pic->nlt = 0;
    } else if (reference_picture_set(d, pic)) {
        log_msg("h265: picture %u predicts from a picture that isn't there\n", d->pictures);
        return -1;
    }
    if (pick_slot(d, &pic->slot) || pic->slot >= pic->sps->max_dec_pic_buffering) {
        log_msg("h265: picture %u needs more surfaces than the SPS allows for\n", d->pictures);
        return -1;
    }
    pic->index = d->pictures++;
    pic->idr_count = d->idrs;
    return 1;
}

void h265_done(struct h265_dec *d, const struct h265_picture *pic)
{
    for (int i = 0; i < H265_DPB; i++) {
        if (!d->dpb[i].used) {
            d->dpb[i] = (struct h265_ref){ .used = 1, .poc = pic->poc, .slot = pic->slot };
            return;
        }
    }
}

void h265_slices(const struct h265_picture *pic, const uint8_t *s, uint8_t *out)
{
    uint32_t at = 0;
    for (uint32_t i = 0; i < pic->nslices; i++) {
        out[at++] = 0;
        out[at++] = 0;
        out[at++] = 1;
        memcpy(out + at, s + pic->slice_pos[i], pic->slice_len[i]);
        at += pic->slice_len[i];
    }
}
