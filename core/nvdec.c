#include "nvdec.h"
#include "av1cdf.h"
#include "av1grain.h"

#define ALIGN(x, a) (((x) + (a) - 1) / (a) * (a))
#define MAX_FRAME_DISTANCE 31  // max order hint distance for projecting a reference's MVs

// two end of sequence NAL units, NVDEC's marker after the slices
static const uint8_t eos[16] = { 0, 0, 1, 0x0b, 0, 0, 0, 0, 0, 0, 1, 0x0b };

void nvdec_surface_init(uint32_t width_mbs, uint32_t height_mbs, uint32_t bps, struct nvdec_surface *surf)
{
    surf->width_mbs = width_mbs;
    surf->height_mbs = height_mbs;
    surf->bps = bps;
    surf->pitch = ALIGN(width_mbs * 16 * bps, 64);
    surf->luma_rows = height_mbs * 16;
    surf->chroma_rows = ALIGN(height_mbs * 8, 16);
    surf->chroma_offset = ALIGN((uint64_t)surf->pitch * surf->luma_rows, 0x1000);
    surf->size = surf->chroma_offset + (uint64_t)surf->pitch * surf->chroma_rows;
}

void nvdec_h264_sizes(uint32_t width_mbs, uint32_t height_mbs, struct nvdec_h264_sizes *sizes)
{
    sizes->coloc = ALIGN(ALIGN(height_mbs, 2) * width_mbs * 64 - 63, 0x100);
    sizes->mbhist = ALIGN(width_mbs * 104, 0x100);
    sizes->history = ALIGN(width_mbs * 0x300, 0x200);
}

// Filter buffer: deblocking and SAO state kept across CTB rows and tile columns. nvdec_drv.h gives part sizes
// (HEVC_DBLK_TOP_BUF_SIZE etc. for 16x16 CTBs, GIP_ASIC_*_RAM_SIZE per row per tile border, up to 20 columns) but not
// their placement, so each of the five regions the setup's offsets name is sized for its worst case.
void nvdec_hevc_sizes(const struct nvdec_surface *surf, struct nvdec_hevc_sizes *sizes)
{
    uint32_t w = surf->width_mbs, h = surf->height_mbs, rows = h * 16, borders = H265_MAX_TILE_COLS - 1;
    uint32_t dblk_top = (w + 2) * ALIGN(632, 128), dblk_left = (h + 2) * ALIGN(506, 128);
    uint32_t sao_left = (h + 2) * ALIGN(713, 128), sao_top = (w + 2) * ALIGN(713, 128);
    // SAO tile borders take 48 bytes a row as on Hantro G2, not nvdec_drv.h's 16
    uint32_t tile_filter = 16 * rows * borders * surf->bps, tile_sao = 48 * rows * borders * surf->bps;
    uint32_t tile_bsd = 4 * rows * borders;

    sizes->coloc = ALIGN(ALIGN(h, 2) * w * 64 - 63, 0x100);
    sizes->flt_above = ALIGN(dblk_left + tile_filter, 0x100);
    sizes->sao = sizes->flt_above + ALIGN(dblk_top, 0x100);
    sizes->sao_above = sizes->sao + ALIGN(sao_left + tile_sao, 0x100);
    sizes->bsd = sizes->sao_above + ALIGN(sao_top, 0x100);
    sizes->filter = sizes->bsd + ALIGN(tile_bsd, 0x100);
    // nvk gives the intra top buffer 64 KiB (h265.rs of its H.265 branch)
    sizes->intra_top = ALIGN(0x10000 + w * 16 * 32, 0x1000);
}

void nvdec_h264_setup(const struct h264_dec *d, const struct h264_picture *pic, const struct nvdec_surface *surf,
                      struct nvdec_h264_pic *out)
{
    const struct h264_sps *s = pic->sps;
    const struct h264_pps *p = pic->pps;
    struct nvdec_h264_sizes sizes;
    nvdec_h264_sizes(s->width_mbs, s->height_mbs, &sizes);

    *out = (struct nvdec_h264_pic){
        .explicit_eos = 1,
        .stream_len = pic->bytes + sizeof(eos),
        .slice_count = pic->nslices,
        .mbhist_buffer_size = sizes.mbhist,
        // with the timer on, a timeout of 0 never fires and a stuck decode hangs the engine (nvk)
        .gptimer_timeout = 81000000,
        .log2_max_poc_lsb_minus4 = s->poc_type == 0 ? (int32_t)s->log2_max_poc_lsb - 4 : 0,
        .delta_pic_order_always_zero = s->delta_pic_order_always_zero,
        .frame_mbs_only = s->frame_mbs_only,
        .width_mbs = (int32_t)s->width_mbs,
        .height_mbs = (int32_t)s->height_mbs,
        .tile_format = 1,
        .gob_height = 0,
        .cabac = p->cabac,
        .pic_order_present = p->bottom_field_pic_order,
        .num_ref_idx_l0_minus1 = (int32_t)p->num_ref_idx_l0,
        .num_ref_idx_l1_minus1 = (int32_t)p->num_ref_idx_l1,
        .deblocking_control = p->deblocking_control,
        .redundant_pic_cnt = p->redundant_pic_cnt,
        .transform_8x8 = p->transform_8x8,
        .pitch_luma = surf->pitch,
        .pitch_chroma = surf->pitch,
        .luma_bot_offset = s->width_mbs * 16,
        .chroma_bot_offset = surf->pitch / 2,
        .hist_buffer_size = sizes.history >> 8,
        .direct_8x8_inference = (uint32_t)s->direct_8x8_inference,
        .weighted_pred = (uint32_t)p->weighted_pred,
        .constrained_intra_pred = (uint32_t)p->constrained_intra_pred,
        .ref_pic = pic->hdr.nal_ref_idc != 0,
        .log2_max_frame_num_minus4 = s->log2_max_frame_num - 4,
        .chroma_format_idc = s->chroma_format,
        .poc_type = s->poc_type,
        .pic_init_qp_minus26 = p->pic_init_qp,
        .chroma_qp_index_offset = p->chroma_qp_index_offset,
        .second_chroma_qp_index_offset = p->second_chroma_qp_index_offset,
        .weighted_bipred_idc = p->weighted_bipred_idc,
        .curr_pic_idx = pic->slot,
        .curr_col_idx = pic->slot,
        .frame_num = pic->hdr.frame_num,
        .curr_field_order_cnt = { pic->poc[0], pic->poc[1] },
        .transform_bypass = (uint32_t)s->transform_bypass,
    };
    memcpy(out->eos, eos, sizeof(eos));
    memset(out->weight_scale, 16, sizeof(out->weight_scale));
    memset(out->weight_scale_8x8, 16, sizeof(out->weight_scale_8x8));

    for (int i = 0; i < H264_DPB; i++) {
        const struct h264_ref *r = &d->dpb[i];
        if (!r->used)
            continue;
        uint32_t slot = r->non_existing ? pic->slot : r->slot, marking = r->long_term ? 2 : 1;
        out->dpb[i] = (struct nvdec_dpb_entry){
            .index = slot,
            .col_idx = slot,
            .state = 3,
            .is_long_term = (uint32_t)r->long_term,
            .not_existing = (uint32_t)r->non_existing,
            .top_field_marking = marking,
            .bottom_field_marking = marking,
            .field_order_cnt = { r->poc[0], r->poc[1] },
            .frame_idx = (int32_t)(r->long_term ? r->long_term_frame_idx : r->frame_num),
        };
    }
}

// initial ref list as Hantro G2 (NVDEC's ancestor) takes it: the candidate sequence repeated to fill 16 entries
static void init_ref_list(const struct h265_dec *d, uint8_t *list, const uint8_t *first, uint32_t nfirst,
                          const uint8_t *second, uint32_t nsecond, const uint8_t *lt, uint32_t nlt)
{
    uint8_t seq[3 * H265_DPB];
    uint32_t n = 0;
    for (uint32_t i = 0; i < nfirst; i++)
        seq[n++] = (uint8_t)d->dpb[first[i]].slot;
    for (uint32_t i = 0; i < nsecond; i++)
        seq[n++] = (uint8_t)d->dpb[second[i]].slot;
    for (uint32_t i = 0; i < nlt; i++)
        seq[n++] = (uint8_t)d->dpb[lt[i]].slot;
    for (uint32_t i = 0; n && i < 16; i++)
        list[i] = seq[i % n];
}

// scaling lists as Hantro G2 takes them: 16x16 and 32x32 DC values, then each list column by column
static void scaling_lists(const struct h265_scaling *sl, uint8_t *out)
{
    for (int i = 0; i < 6; i++)
        *out++ = sl->dc[0][i];
    for (int i = 0; i < 2; i++)
        *out++ = sl->dc[1][i * 3];
    out += 8;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                *out++ = sl->list[0][i][4 * k + j];
    for (int size = 1; size < 4; size++)
        for (int i = 0; i < (size == 3 ? 2 : 6); i++)
            for (int j = 0; j < 8; j++)
                for (int k = 0; k < 8; k++)
                    *out++ = sl->list[size][size == 3 ? i * 3 : i][8 * k + j];
}

// each tile's width and height in CTBs, row by row, as Hantro G2 takes them
static void tile_sizes(const struct h265_sps *s, const struct h265_pps *p, uint16_t *out)
{
    uint32_t cols[H265_MAX_TILE_COLS], rows[H265_MAX_TILE_ROWS];
    h265_tiles(s, p, cols, rows);
    for (uint32_t i = 0; i < p->tile_rows; i++) {
        for (uint32_t j = 0; j < p->tile_cols; j++) {
            *out++ = (uint16_t)cols[j];
            *out++ = (uint16_t)rows[i];
        }
    }
}

void nvdec_hevc_setup(const struct h265_dec *d, const struct h265_picture *pic, const struct nvdec_surface *surf,
                      uint8_t *page)
{
    const struct h265_sps *s = pic->sps;
    const struct h265_pps *p = pic->pps;
    const struct h265_slice *h = &pic->hdr;
    struct nvdec_hevc_sizes sizes;
    nvdec_hevc_sizes(surf, &sizes);
    struct nvdec_hevc_pic *out = (struct nvdec_hevc_pic *)page;
    uint32_t refs = pic->nbefore + pic->nafter + pic->nlt;

    *out = (struct nvdec_hevc_pic){
        .stream_len = pic->bytes,
        // with the timer on, a timeout of 0 never fires and a stuck decode hangs the engine (nvk)
        .gptimer_timeout = 81000000,
        .tile_format = 1,
        .gob_height = 0,
        .start_code = 1,
        .disp_output_mode = s->bit_depth > 8,
        .frame_stride = { surf->pitch / surf->bps, surf->pitch / surf->bps },
        .col_mv_buffer_size = sizes.coloc >> 8,
        .sao_buffer_offset = sizes.sao >> 8,
        .bsd_ctrl_offset = sizes.bsd >> 8,
        .width = (uint16_t)s->width,
        .height = (uint16_t)s->height,
        .chroma_format_idc = 1,
        .bit_depth_luma = s->bit_depth,
        .bit_depth_chroma = s->bit_depth,
        .log2_min_cb = s->log2_min_cb,
        .log2_max_cb = s->log2_ctb,
        .log2_min_tb = s->log2_min_tb,
        .log2_max_tb = s->log2_max_tb,
        .max_transform_hierarchy_depth_inter = s->depth_inter,
        .max_transform_hierarchy_depth_intra = s->depth_intra,
        .scaling_list = (uint32_t)s->scaling_list_enabled,
        .amp = (uint32_t)s->amp,
        .sao = (uint32_t)s->sao,
        .pcm = (uint32_t)s->pcm,
        .pcm_bit_depth_luma = s->pcm_depth_luma,
        .pcm_bit_depth_chroma = s->pcm_depth_chroma,
        .log2_min_pcm = s->log2_min_pcm,
        .log2_max_pcm = s->log2_max_pcm,
        .pcm_loop_filter_disabled = (uint32_t)s->pcm_loop_filter_disabled,
        .temporal_mvp = (uint32_t)s->temporal_mvp,
        .strong_intra_smoothing = (uint32_t)s->strong_intra_smoothing,
        .dependent_slice_segments = (uint32_t)p->dependent_slices,
        .output_flag_present = (uint32_t)p->output_flag_present,
        .num_extra_slice_header_bits = p->extra_slice_header_bits,
        .sign_data_hiding = (uint32_t)p->sign_data_hiding,
        .cabac_init_present = (uint32_t)p->cabac_init_present,
        .num_ref_idx_l0_default = p->num_ref_idx_l0,
        .num_ref_idx_l1_default = p->num_ref_idx_l1,
        // 26 + init_qp_minus26 goes below 0 for high bit depths, which add 6 per bit to the QP range (nvdec_drv.h)
        .init_qp = (uint32_t)(26 + p->init_qp + 6 * (int32_t)(s->bit_depth - 8)),
        .constrained_intra_pred = (uint32_t)p->constrained_intra_pred,
        .transform_skip = (uint32_t)p->transform_skip,
        .cu_qp_delta = (uint32_t)p->cu_qp_delta,
        .diff_cu_qp_delta_depth = p->diff_cu_qp_delta_depth,
        .cb_qp_offset = (int8_t)p->cb_qp_offset,
        .cr_qp_offset = (int8_t)p->cr_qp_offset,
        .beta_offset = (int8_t)(p->beta_offset_div2 * 2),
        .tc_offset = (int8_t)(p->tc_offset_div2 * 2),
        .slice_chroma_qp_offsets_present = (uint32_t)p->slice_chroma_qp_offsets_present,
        .weighted_pred = (uint32_t)p->weighted_pred,
        .weighted_bipred = (uint32_t)p->weighted_bipred,
        .transquant_bypass = (uint32_t)p->transquant_bypass,
        .tiles = (uint32_t)p->tiles,
        .entropy_coding_sync = (uint32_t)p->entropy_coding_sync,
        .tile_cols = p->tile_cols,
        .tile_rows = p->tile_rows,
        .loop_filter_across_tiles = (uint32_t)p->loop_filter_across_tiles,
        .loop_filter_across_slices = (uint32_t)p->loop_filter_across_slices,
        .deblocking_control = (uint32_t)p->deblocking_control,
        .deblocking_override = (uint32_t)p->deblocking_override,
        .deblocking_disabled = (uint32_t)p->deblocking_disabled,
        .lists_modification = (uint32_t)p->lists_modification,
        .log2_parallel_merge_level = p->log2_parallel_merge_level,
        .slice_header_extension = (uint32_t)p->slice_header_extension,
        // non-zero even for a picture that predicts from nothing, or the engine hangs (hantro_g2_hevc_dec.c)
        .num_ref_frames = (uint8_t)(refs ? refs : 1),
        .idr = (uint8_t)pic->idr,
        .irap = (uint8_t)pic->irap,
        .curr_pic_idx = (uint8_t)pic->slot,
        // traced from NVIDIA's decoder (nvk's H.265 branch)
        .pattern_id = 2,
        .sw_hdr_skip_length = (uint16_t)h->skip_bits,
        .log2_max_poc_lsb_minus4 = s->log2_max_poc_lsb - 4,
        .num_short_term_ref_pic_sets = s->nrps,
        .num_long_term_ref_pics_sps = s->nlt,
        .num_delta_pocs_of_rps_idx = h->rps_ref_num_deltas,
        .long_term_ref_pics_present = (uint32_t)s->long_term_present,
        .num_bits_short_term_ref_pics_in_slice = h->rps_bits,
        // log2 of the largest transform-skip block, 4x4 without the range extension: at 0 the engine expects no
        // transform_skip_flag, and fails on the first block that has one
        .ext = { .pps_flags = 2, .flt_above_offset = sizes.flt_above >> 8, .sao_above_offset = sizes.sao_above >> 8 },
    };

    for (int i = 0; i < H265_DPB; i++) {
        const struct h265_ref *r = &d->dpb[i];
        if (!r->used)
            continue;
        int32_t diff = pic->poc - r->poc;
        out->ref_poc_diff[r->slot] = (int16_t)(diff < -128 ? -128 : diff > 127 ? 127 : diff);
        if (r->long_term)
            out->long_term |= (uint16_t)(0x8000 >> r->slot);
    }
    init_ref_list(d, out->init_ref_list[0], pic->before, pic->nbefore, pic->after, pic->nafter, pic->lt, pic->nlt);
    init_ref_list(d, out->init_ref_list[1], pic->after, pic->nafter, pic->before, pic->nbefore, pic->lt, pic->nlt);

    memset(page + NVDEC_SETUP_SCALING, 0, 16 * 64);
    if (s->scaling_list_enabled)
        scaling_lists(p->scaling_list_present ? &p->scaling : &s->scaling, page + NVDEC_SETUP_SCALING);
    memset(page + NVDEC_SETUP_TILES, 0, H265_MAX_TILE_COLS * H265_MAX_TILE_ROWS * 4);
    tile_sizes(s, p, (uint16_t *)(page + NVDEC_SETUP_TILES));
}

// sizes from nvdec_drv.h (AV1_TEMPORAL_MV_BUF_SIZE etc.); the six filter regions go back to back in NVIDIA's driver's
// order, superres before loop restoration
void nvdec_av1_sizes(const struct nvdec_surface *surf, struct nvdec_av1_sizes *sizes)
{
    uint32_t w = surf->width_mbs * 16, h = surf->height_mbs * 16;
    uint32_t sb_cols = ALIGN(w, 64) / 64 + 2, sb_rows = ALIGN(h, 64) / 64 + 2;
    uint32_t area = ALIGN(w, 128) * ALIGN(h, 128) / (64 * 64);
    uint32_t dblk_top = ALIGN(sb_cols * ALIGN(1920, 128), 256), dblk_left = ALIGN(sb_rows * ALIGN(1536, 128), 256);
    uint32_t cdef_left = ALIGN(sb_rows * ALIGN(1792, 128), 256);
    uint32_t hus_left = 2 * ALIGN(sb_rows * ALIGN(12544, 128), 256);  // ping-pong
    uint32_t lr_left = ALIGN(sb_rows * ALIGN(1920, 128), 256), fgs_left = ALIGN(sb_rows * ALIGN(320, 128), 256);

    sizes->coloc = ALIGN(area * 256, 4096);
    sizes->segmap = ALIGN(area * 128, 4096);
    sizes->flt_top = 0;
    sizes->flt_vert = sizes->flt_top + dblk_top;
    sizes->cdef_vert = sizes->flt_vert + dblk_left;
    sizes->hus_vert = sizes->cdef_vert + cdef_left;
    sizes->lr_vert = sizes->hus_vert + hus_left;
    sizes->fgs_vert = sizes->lr_vert + lr_left;
    sizes->filter = sizes->fgs_vert + fgs_left;
    // AV1_INTRA_TOP_BUF_SIZE is for 8-bit samples
    sizes->intra_top = 8 * 8192 * surf->bps;
    sizes->prob = ALIGN(AV1_PROBS_SIZE, 256);
}

static int32_t round2_signed(int64_t x, int n)
{
    int64_t r = (int64_t)1 << (n - 1);
    return (int32_t)(x < 0 ? -((-x + r) >> n) : (x + r) >> n);
}

static int32_t clip16(int64_t x)
{
    return (int32_t)(x < -32768 ? -32768 : x > 32767 ? 32767 : x);
}

// Div_Lut of the specification: 2^22 / (256 + i), rounded
static int div_lut(int i)
{
    return (int)(((1 << 22) + (256 + i) / 2) / (256 + i));
}

// setupShear (7.11.3.6) reduced to what the hardware keeps; 0 if the model can't warp
static int shear_params(const int32_t *m, int16_t *out)
{
    if (m[2] <= 0)
        return 0;
    int shift = 31 - __builtin_clz((uint32_t)m[2]);
    int64_t e = (int64_t)m[2] - ((int64_t)1 << shift);
    int64_t f = shift > 8 ? (e + ((int64_t)1 << (shift - 9))) >> (shift - 8) : e << (8 - shift);
    int64_t factor = div_lut((int)f);
    shift += 14;
    int32_t alpha = clip16((int64_t)m[2] - (1 << 16)), beta = clip16(m[3]);
    int32_t gamma = clip16(round2_signed((int64_t)m[4] * 65536 * factor, shift));
    int32_t delta = clip16((int64_t)m[5] - round2_signed((int64_t)m[3] * m[4] * factor, shift) - (1 << 16));
    int32_t v[4] = { alpha, beta, gamma, delta };
    for (int i = 0; i < 4; i++)
        v[i] = round2_signed(v[i], 6) * 64;
    if (4 * __builtin_abs(v[0]) + 7 * __builtin_abs(v[1]) >= 1 << 16 ||
        4 * __builtin_abs(v[2]) + 4 * __builtin_abs(v[3]) >= 1 << 16)
        return 0;
    for (int i = 0; i < 4; i++)
        out[i] = (int16_t)v[i];
    return 1;
}

// per reference: six model params, then four 16-bit shear params, all 0x8000 if the model can't warp or the
// reference is scaled; intra pictures get identity models and no shear
static void global_motion(const struct av1_frame *f, int intra, const struct nvdec_av1_pic *pic, uint8_t *out)
{
    static const int32_t identity[6] = { 0, 0, 1 << 16, 0, 0, 1 << 16 };
    for (int i = 0; i < AV1_REFS_PER_FRAME; i++, out += 32) {
        const int32_t *m = intra ? identity : f->gm_params[i + 1];
        int16_t shear[4] = { 0 };
        int scaled = pic->ref[i].x_scale != 1 << 14 || pic->ref[i].y_scale != 1 << 14;
        if (!intra && (scaled || !shear_params(m, shear)))
            shear[0] = shear[1] = shear[2] = shear[3] = INT16_MIN;
        memcpy(out, m, 24);
        memcpy(out + 24, shear, 8);
    }
}

// AV1TileInfo_OLD per tile in raster order: width and height in superblocks, then bitstream offsets of its first
// byte and one past its last (AV1_SET_TILE_STREAM_INFO_BUF_OFFSET unused)
static void av1_tiles_info(const struct av1_picture *pic, uint8_t *out)
{
    const struct av1_frame *f = &pic->hdr;
    memset(out, 0, AV1_MAX_TILES * 16);
    for (uint32_t row = 0; row < f->tile_rows; row++) {
        for (uint32_t col = 0; col < f->tile_cols; col++, out += 16) {
            uint32_t t = row * f->tile_cols + col;
            out[0] = (uint8_t)(f->col_starts[col + 1] - f->col_starts[col]);
            out[1] = (uint8_t)(f->row_starts[row + 1] - f->row_starts[row]);
            memcpy(out + 2, &pic->tile_start[t], 4);
            memcpy(out + 6, &pic->tile_end[t], 4);
        }
    }
}

static uint32_t cdef_pack(const uint32_t *v, uint32_t n, int bits)
{
    uint32_t out = 0;
    for (uint32_t i = 0; i < n; i++)
        out |= v[i] << (i * bits);
    return out;
}

// up to three references to project MVs from (libaom av1_setup_motion_field): LAST_FRAME unless an overlay, then
// BWDREF, ALTREF2, ALTREF if later, then LAST2. Ones too far away are named but not used.
static void motion_fields(const struct av1_dec *d, const struct av1_frame *f, uint32_t slot,
                          struct nvdec_av1_pic *out, struct nvdec_av1_frame *fr)
{
    const struct av1_seq *s = &d->seq;
    static const uint8_t order[5] = { 0, 4, 5, 6, 1 };
    int hint[AV1_REFS_PER_FRAME], picked[3], n = 0, stamp = 2, cur = (int)f->order_hint;
    for (int i = 0; i < AV1_REFS_PER_FRAME; i++)
        hint[i] = (int)d->refs[f->ref_idx[i]].order_hint;

    for (int k = 0; k < 5 && n < 3; k++) {
        int i = order[k];
        const struct av1_ref *r = &d->refs[f->ref_idx[i]];
        int usable = r->mi_cols == f->mi_cols && r->mi_rows == f->mi_rows && r->type != AV1_KEY_FRAME &&
                     r->type != AV1_INTRA_ONLY_FRAME;
        if (k == 0) {
            // an overlay of the golden frame
            usable &= (int)r->order_hints[AV1_REFS_PER_FRAME] != hint[3];
            stamp--;
        } else if (k <= 3 && av1_dist(s, hint[i], cur) <= 0) {
            continue;
        } else if (k >= 3 && stamp < 0) {
            continue;
        }
        if (!usable)
            continue;
        if (k)
            stamp--;
        picked[n++] = i;
    }

    int use[3] = { 0 }, type[3] = { 0 };
    for (int m = 0; m < 3; m++)
        fr->coloc[m] = slot;
    for (int m = 0; m < n; m++) {
        int i = picked[m], dist = av1_dist(s, cur, hint[i]);
        const struct av1_ref *r = &d->refs[f->ref_idx[i]];
        type[m] = i + 1;
        fr->coloc[m] = r->slot;
        if (!f->use_ref_frame_mvs || dist > MAX_FRAME_DISTANCE || dist < -MAX_FRAME_DISTANCE)
            continue;
        use[m] = 1;
        for (int k = 0; k < AV1_REFS_PER_FRAME; k++) {
            int saved = (int)r->order_hints[k + 1];
            out->mfmv_offset[m][k] = (int16_t)av1_dist(s, hint[i], saved);
            out->mfmv_side[m][k] = av1_dist(s, saved, hint[i]) > 0;
        }
    }
    out->use_temporal0_mvs = (uint32_t)use[0];
    out->use_temporal1_mvs = (uint32_t)use[1];
    out->use_temporal2_mvs = (uint32_t)use[2];
    out->mf1_type = (uint32_t)type[0];
    out->mf2_type = (uint32_t)type[1];
    out->mf3_type = (uint32_t)type[2];
}

// NVIDIA's driver allows 10000 cycles per macroblock, counting at least 8100 macroblocks
static uint32_t gptimer_timeout(uint32_t width, uint32_t height)
{
    uint64_t mbs = (uint64_t)((width + 15) / 16) * ((height + 15) / 16);
    uint64_t cycles = (mbs > 8100 ? mbs : 8100) * 10000;
    return (uint32_t)(cycles < 0x4fffffff ? cycles : 0x4fffffff);
}

void nvdec_av1_setup(const struct av1_dec *d, const struct av1_picture *pic, const struct nvdec_surface *surf,
                     uint8_t *page)
{
    const struct av1_seq *s = pic->seq;
    const struct av1_frame *f = &pic->hdr;
    const struct av1_grain *g = &f->grain;
    struct nvdec_av1_sizes sizes;
    nvdec_av1_sizes(surf, &sizes);
    struct nvdec_av1_pic *out = (struct nvdec_av1_pic *)page;
    struct nvdec_av1_frame *fr = (struct nvdec_av1_frame *)(page + NVDEC_SETUP_AV1);
    int intra = f->type == AV1_KEY_FRAME || f->type == AV1_INTRA_ONLY_FRAME;
    uint16_t stride = (uint16_t)(surf->pitch / 16);
    uint32_t tx_mode[] = { 0, 3, 4 };

    int cdef = s->cdef && !f->coded_lossless && !f->allow_intrabc &&
               (f->cdef_bits || f->cdef_y_pri[0] || f->cdef_y_sec[0] || f->cdef_uv_pri[0] || f->cdef_uv_sec[0]);
    uint32_t lr_type = 0, lr_unit = 0;
    for (int i = 0; i < 3; i++) {
        uint32_t shift = f->uses_lr ? 1 + f->lr_unit_shift - (i ? f->lr_uv_shift : 0) : 3;
        lr_type |= f->lr_type[i] << (i * 2);
        lr_unit |= shift << (i * 2);
    }

    *out = (struct nvdec_av1_pic){
        .gptimer_timeout = gptimer_timeout(f->upscaled_width, f->height),
        .stream_len = pic->bytes,
        .use_128x128_superblock = (uint32_t)s->sb128,
        .chroma_format = 1,
        .bit_depth = s->bit_depth,
        .enable_filter_intra = (uint32_t)s->filter_intra,
        .enable_intra_edge_filter = (uint32_t)s->intra_edge_filter,
        .enable_interintra_compound = (uint32_t)s->interintra,
        .enable_masked_compound = (uint32_t)s->masked_compound,
        .enable_dual_filter = (uint32_t)s->dual_filter,
        .enable_order_hint = (uint32_t)s->order_hint,
        .enable_jnt_comp = (uint32_t)s->jnt_comp,
        .enable_cdef = (uint32_t)cdef,
        // grain goes only to a separate display picture (SET_DISPLAY_BUF_LUMA_OFFSET), which isn't set up yet
        .enable_fgs = 0,
        .frame_type = f->type,
        // hidden frames too (mesa-nova)
        .show_frame = 1,
        .disable_cdf_update = (uint32_t)f->disable_cdf_update,
        .allow_screen_content_tools = (uint32_t)f->screen_content_tools,
        // intra frames code intra block copy's vectors as integers
        .force_integer_mv = (uint32_t)((f->screen_content_tools && f->force_integer_mv) || intra),
        .allow_intrabc = (uint32_t)f->allow_intrabc,
        .allow_high_precision_mv = (uint32_t)f->high_precision_mv,
        .interp_filter = f->interp_filter,
        .switchable_motion_mode = (uint32_t)f->motion_switchable,
        .use_ref_frame_mvs = (uint32_t)f->use_ref_frame_mvs,
        .refresh_frame_context = (uint32_t)!f->disable_frame_end_update_cdf,
        .delta_q_present = (uint32_t)f->delta_q_present,
        .delta_q_res = f->delta_q_res,
        .delta_lf_present = (uint32_t)f->delta_lf_present,
        .delta_lf_res = f->delta_lf_res,
        .delta_lf_multi = (uint32_t)f->delta_lf_multi,
        .coded_lossless = (uint32_t)f->coded_lossless,
        .tile_enabled = f->tile_cols * f->tile_rows > 1,
        .tile_cols = f->tile_cols,
        .tile_rows = f->tile_rows,
        .context_update_tile_id = f->context_update_tile_id,
        .cdef_damping_minus_3 = cdef ? f->cdef_damping : 0,
        .cdef_bits = cdef ? f->cdef_bits : 0,
        .tx_mode = tx_mode[f->tx_mode],
        .reference_mode = f->reference_select ? 2 : 0,
        .skip_mode = (uint32_t)f->skip_mode_present,
        .skip_ref0 = f->skip_mode_present ? f->skip_mode_frame[0] : 0,
        .skip_ref1 = f->skip_mode_present ? f->skip_mode_frame[1] : 0,
        .allow_warp = (uint32_t)f->allow_warped_motion,
        .reduced_tx_set = (uint32_t)f->reduced_tx_set,
        .superres_upscaled_width = (uint16_t)f->upscaled_width,
        // what NVIDIA's driver gives without superres (mesa-nova), which the parser turns down
        .superres_luma_step = 14,
        .superres_chroma_step = 14,
        .base_qindex = (uint8_t)f->base_qindex,
        .y_dc_delta_q = (int8_t)f->dq_y_dc,
        .u_dc_delta_q = (int8_t)f->dq_u_dc,
        .v_dc_delta_q = (int8_t)f->dq_v_dc,
        .u_ac_delta_q = (int8_t)f->dq_u_ac,
        .v_ac_delta_q = (int8_t)f->dq_v_ac,
        // level 15 is no matrix
        .qm_y = (uint8_t)(f->using_qmatrix ? f->qm_y : 15),
        .qm_u = (uint8_t)(f->using_qmatrix ? f->qm_u : 15),
        .qm_v = (uint8_t)(f->using_qmatrix ? f->qm_v : 15),
        .cdef_y_pri_strength = cdef ? cdef_pack(f->cdef_y_pri, 8, 4) : 0,
        .cdef_uv_pri_strength = cdef ? cdef_pack(f->cdef_uv_pri, 8, 4) : 0,
        .cdef_y_sec_strength = cdef ? cdef_pack(f->cdef_y_sec, 8, 2) : 0,
        .cdef_uv_sec_strength = cdef ? cdef_pack(f->cdef_uv_sec, 8, 2) : 0,
        .segment_enabled = (uint8_t)f->seg_enabled,
        .segment_update_map = (uint8_t)f->seg_update_map,
        .segment_temporal_update = (uint8_t)f->seg_temporal_update,
        .last_active_segid = (uint8_t)f->seg_last_active,
        .segid_preskip = (uint8_t)f->seg_preskip,
        .filter_level = { (uint8_t)f->lf_level[0], (uint8_t)f->lf_level[1] },
        .filter_level_u = (uint8_t)f->lf_level[2],
        .filter_level_v = (uint8_t)f->lf_level[3],
        .lf_sharpness = (uint8_t)f->lf_sharpness,
        .lr_type = (uint8_t)lr_type,
        .lr_unit_size = (uint8_t)lr_unit,
        .current = { (uint16_t)f->width, (uint16_t)f->height, { stride, stride }, 0 },
        .tile_format = 1,
        .flt_top_offset = sizes.flt_top >> 8,
        .flt_vert_offset = sizes.flt_vert >> 8,
        .cdef_vert_offset = sizes.cdef_vert >> 8,
        .lr_vert_offset = sizes.lr_vert >> 8,
        .hus_vert_offset = sizes.hus_vert >> 8,
        .fgs_vert_offset = sizes.fgs_vert >> 8,
    };
    *fr = (struct nvdec_av1_frame){
        .refresh = f->refresh,
        .primary = NVDEC_AV1_DEFAULT,
        .coloc = { pic->slot, pic->slot, pic->slot },
        .segmap = NVDEC_AV1_DEFAULT,
    };
    for (int i = 0; i < AV1_REFS_PER_FRAME; i++)
        fr->refs[i] = intra ? pic->slot : d->refs[f->ref_idx[i]].slot;

    // features from the reference frame (5) up are flags; the reference frame itself is ref + 1, 0 on key frames
    for (int i = 0; i < 8 && f->seg_enabled; i++) {
        for (int j = 0; j < 8; j++) {
            if (!(f->seg_feature_enabled[i] >> j & 1))
                continue;
            int16_t value = f->seg_feature_data[i][j];
            if (j == 0 && value < 0) {
                out->segment_quant_sign |= (uint8_t)(1 << i);
                value = (int16_t)-value;
            } else if (j == 5) {
                value = f->type == AV1_KEY_FRAME ? 0 : value + 1;
            } else if (j > 5) {
                value = 1;
            }
            out->segment_feature_data[i][j] = value;
        }
    }
    if (f->lf_delta_enabled) {
        for (int i = 0; i < 8; i++)
            out->lf_ref_deltas[i] = (int8_t)f->lf_ref_deltas[i];
        out->lf_mode_deltas[0] = (int8_t)f->lf_mode_deltas[0];
        out->lf_mode_deltas[1] = (int8_t)f->lf_mode_deltas[1];
    }

    if (f->primary_ref != AV1_PRIMARY_REF_NONE) {
        const struct av1_ref *r = &d->refs[f->ref_idx[f->primary_ref]];
        fr->primary = f->ref_idx[f->primary_ref];
        if (f->seg_enabled && r->mi_cols == f->mi_cols && r->mi_rows == f->mi_rows) {
            out->prevsegid = 1;
            fr->segmap = r->slot;
        }
    }

    if (!intra || f->allow_intrabc) {
        int scaled = 0;
        for (int i = 0; i < AV1_REFS_PER_FRAME; i++) {
            struct nvdec_av1_ref *o = &out->ref[i];
            if (f->allow_intrabc) {
                o->info = out->current;
                o->x_scale = o->y_scale = 1 << 14;
                continue;
            }
            const struct av1_ref *r = &d->refs[f->ref_idx[i]];
            int hint = (int)r->order_hint, cur = (int)f->order_hint;
            o->info = (struct nvdec_av1_frame_info){ (uint16_t)r->upscaled_width, (uint16_t)r->height,
                                                     { stride, stride }, 0 };
            o->x_scale = (int16_t)(((r->upscaled_width << 14) + f->width / 2) / f->width);
            o->y_scale = (int16_t)(((r->height << 14) + f->height / 2) / f->height);
            scaled |= o->x_scale != 1 << 14 || o->y_scale != 1 << 14;
            o->sign_bias = av1_dist(s, hint, cur) > 0;
            o->warp_type = f->gm_type[i + 1];
            o->frame_off = (int16_t)av1_dist(s, cur, hint);
            o->roffset = (int16_t)av1_dist(s, hint, cur);
        }
        out->ref_scaling = (uint32_t)scaled;
        if (!f->allow_intrabc)
            motion_fields(d, f, pic->slot, out, fr);
    }

    if (g->apply) {
        out->fgs = (struct nvdec_av1_fgs){
            .apply_grain = 1,
            .overlap = (uint16_t)g->overlap,
            .clip_to_restricted_range = (uint16_t)g->clip_to_restricted_range,
            .chroma_scaling_from_luma = (uint16_t)g->chroma_scaling_from_luma,
            .has_y_points = g->num_y > 0,
            .has_cb_points = g->num_cb > 0,
            .has_cr_points = g->num_cr > 0,
            .scaling_shift = (uint16_t)g->scaling_shift,
            .random_seed = (uint16_t)g->seed,
        };
        if (g->num_cb) {
            out->fgs.cb_mult = (int8_t)((int)g->cb_mult - 128);
            out->fgs.cb_luma_mult = (int8_t)((int)g->cb_luma_mult - 128);
            out->fgs.cb_offset = (int16_t)((int)g->cb_offset - 256);
        }
        if (g->num_cr) {
            out->fgs.cr_mult = (int8_t)((int)g->cr_mult - 128);
            out->fgs.cr_luma_mult = (int8_t)((int)g->cr_luma_mult - 128);
            out->fgs.cr_offset = (int16_t)((int)g->cr_offset - 256);
        }
        av1_grain_tables(g, s->bit_depth, (struct av1_grain_tables *)(page + NVDEC_SETUP_GRAIN));
    }

    global_motion(f, intra, out, page + NVDEC_SETUP_GLOBAL);
    av1_tiles_info(pic, page + NVDEC_SETUP_TILES);
}

// Per superblock, MVs take 1024 bytes (measured) and the segmap 32 (GIP_ASIC_VP9_CTB_SEG_SIZE). nvdec_drv.h sizes the
// deblocking lines (VP9_DBLK_*_BUF_SIZE) and per-row tile border costs (GIP_ASIC_*_RAM_SIZE) but not their placement,
// so as with H.265 each region gets its worst case: deblocking lines plus 48 bytes a row per tile border (at least
// one), as NVIDIA's driver does; above: 16 bytes a row for 64 tile columns or NVIDIA's 5776 bytes per superblock
// column, whichever is more; BSD control: 4 bytes a row for 64 tile columns.
void nvdec_vp9_sizes(const struct nvdec_surface *surf, struct nvdec_vp9_sizes *sizes)
{
    uint32_t w = surf->width_mbs * 16, h = surf->height_mbs * 16, sb_cols = w / 64, sb_rows = h / 64;
    uint32_t cols = 1;
    while (sb_cols / (cols * 2) >= 4)  // tile columns are 4 superblocks or wider
        cols *= 2;
    uint32_t dblk_top = (sb_cols + 2) * ALIGN(2000, 128), dblk_left = ALIGN((sb_rows + 2) * ALIGN(1600, 128), 256);
    uint32_t tile_filter = 48 * h * surf->bps * (cols > 1 ? cols - 1 : 1);
    uint32_t above = 16 * 64 * h > 5776 * sb_cols ? 16 * 64 * h : 5776 * sb_cols;

    sizes->coloc = ALIGN(sb_cols * sb_rows * 1024, 256);
    sizes->segmap = ALIGN(sb_cols * sb_rows * 32, 256);
    sizes->flt_above = ALIGN(dblk_top + dblk_left + tile_filter, 256);
    sizes->bsd = sizes->flt_above + ALIGN(above * surf->bps, 256);
    sizes->filter = sizes->bsd + 4 * 64 * h;
    // measured ~4 bytes per pixel of width; NVIDIA's driver gives 64 KiB
    sizes->intra_top = ALIGN(w * 8 * surf->bps, 0x10000);
    sizes->prob = ALIGN(VP9_PROBS_SIZE, 256);
    sizes->counts = sizeof(struct vp9_counts);
}

// history: 512 bytes per macroblock column; prob also holds NVDEC's state after the probabilities
void nvdec_vp8_sizes(const struct nvdec_surface *surf, struct nvdec_vp8_sizes *sizes)
{
    sizes->history = ALIGN(surf->width_mbs * 0x200 + 0x1000, 256);
    sizes->prob = 0x5000;
}

// each tile's width and height in superblocks, row by row, as for H.265
static void vp9_tile_sizes(const struct vp9_frame *f, uint16_t *out)
{
    uint32_t cols[VP9_MAX_TILE_COLS + 1], rows[VP9_MAX_TILE_ROWS + 1];
    vp9_tile_starts((f->width + 7) / 8, f->tile_cols_log2, cols);
    vp9_tile_starts((f->height + 7) / 8, f->tile_rows_log2, rows);
    memset(out, 0, VP9_MAX_TILE_COLS * VP9_MAX_TILE_ROWS * 4);
    for (uint32_t i = 0; i < 1u << f->tile_rows_log2; i++) {
        for (uint32_t j = 0; j < 1u << f->tile_cols_log2; j++) {
            *out++ = (uint16_t)(cols[j + 1] - cols[j]);
            *out++ = (uint16_t)(rows[i + 1] - rows[i]);
        }
    }
}

void nvdec_vp9_setup(const struct vp9_dec *d, const struct vp9_picture *pic, const struct nvdec_surface *surf,
                     const uint8_t *stream, uint8_t *page)
{
    const struct vp9_frame *f = &pic->hdr;
    struct nvdec_vp9_sizes sizes;
    nvdec_vp9_sizes(surf, &sizes);
    struct nvdec_vp9_pic *out = (struct nvdec_vp9_pic *)page;
    struct nvdec_vp9_frame *fr = (struct nvdec_vp9_frame *)(page + NVDEC_SETUP_VP9);
    // in samples: a byte stride puts 10-bit rows twice as far apart
    uint16_t stride = (uint16_t)(surf->pitch / surf->bps);
    int resize = !d->have_last || f->width != d->last_width || f->height != d->last_height;

    *out = (struct nvdec_vp9_pic){
        .stream_len = pic->bytes,
        .gptimer_timeout = gptimer_timeout(f->width, f->height),
        .tile_format = 1,
        .gob_height = 0,
        .bsd_ctrl_offset = sizes.bsd >> 8,
        .current = { (uint16_t)f->width, (uint16_t)f->height, { stride, stride } },
        .key_frame = f->type == VP9_KEY_FRAME,
        .prev_key_frame = (uint8_t)d->last_key,
        .resolution_change = (uint8_t)resize,
        .error_resilient = (uint8_t)f->error_resilient,
        .prev_show_frame = (uint8_t)(d->have_last && d->last_show),
        .intra_only = (uint8_t)f->intra_only,
        .lf_level = (int8_t)f->lf_level,
        .lf_sharpness = (int8_t)f->lf_sharpness,
        .base_q = (uint8_t)f->base_q,
        .dq_y_dc = (int8_t)f->dq_y_dc,
        .dq_uv_ac = (int8_t)f->dq_uv_ac,
        .dq_uv_dc = (int8_t)f->dq_uv_dc,
        .lossless = (int8_t)f->lossless,
        .tx_mode = (int8_t)f->tx_mode,
        .allow_hp = (int8_t)f->allow_hp,
        // as coded, not libvpx's remapped values, or regular frames come out smooth
        .interp_filter = (int8_t)f->interp_filter,
        .reference_mode = (int8_t)f->reference_mode,
        .comp_fixed_ref = (int8_t)f->comp_fixed_ref,
        .comp_var_ref = { (int8_t)f->comp_var_ref[0], (int8_t)f->comp_var_ref[1] },
        .log2_tile_cols = (int8_t)f->tile_cols_log2,
        .log2_tile_rows = (int8_t)f->tile_rows_log2,
        .seg_enabled = (uint8_t)f->seg_enabled,
        .seg_update_map = (uint8_t)f->seg_update_map,
        .seg_temporal_update = (uint8_t)f->seg_temporal_update,
        .seg_abs = (uint8_t)f->seg_abs,
        .lf_delta_enabled = (int8_t)f->lf_delta_enabled,
        .flt_above_offset = sizes.flt_above >> 8,
        .bit_depth = f->bit_depth,
    };
    for (int i = 0; i < 4; i++) {
        out->ref_sign_bias[i] = (uint8_t)f->sign_bias[i];
        out->lf_ref_deltas[i] = (int8_t)f->lf_ref_deltas[i];
    }
    out->lf_mode_deltas[0] = (int8_t)f->lf_mode_deltas[0];
    out->lf_mode_deltas[1] = (int8_t)f->lf_mode_deltas[1];
    memcpy(out->seg_feature_enabled, f->seg_feature_enabled, sizeof(out->seg_feature_enabled));
    memcpy(out->seg_feature_data, f->seg_feature_data, sizeof(out->seg_feature_data));

    // setup_past_independence() resets the contexts and loads the first (libvpx vp9_entropymode.c)
    uint32_t reset = 0;
    if (f->type == VP9_KEY_FRAME || f->error_resilient || (f->intra_only && f->reset_context == 3))
        reset = 0xf;
    else if (f->intra_only && f->reset_context == 2)
        reset = 1u << f->context;
    *fr = (struct nvdec_vp9_frame){
        .reset = reset,
        .context = f->intra || f->error_resilient ? 0 : f->context,
        .refresh_context = (uint32_t)f->refresh_context,
        .adapt = !f->error_resilient && !f->parallel,
        .mv_map = d->mv_map,
        .seg_map = d->seg_map,
        .clear_seg_map = f->intra || f->error_resilient || resize,
        .header_len = pic->header_len,
    };
    memcpy(fr->seg_tree, f->seg_tree, sizeof(fr->seg_tree));
    memcpy(fr->seg_pred, f->seg_pred, sizeof(fr->seg_pred));
    for (int i = 0; i < 3; i++) {
        const struct vp9_ref *r = &d->refs[f->ref_idx[i]];
        fr->refs[i] = f->intra ? pic->slot : r->slot;
        out->ref[i] = f->intra ? out->current : (struct nvdec_vp9_size){ (uint16_t)r->width, (uint16_t)r->height,
                                                                        { stride, stride } };
    }
    vp9_tile_sizes(f, (uint16_t *)(page + NVDEC_SETUP_TILES));
    memcpy(page + NVDEC_SETUP_VP9_HEADER, stream + pic->header_pos, pic->header_len);
}

// frame from its first partition on; size rounded up to whole macroblocks or the last row mispredicts; segmentation
// byte set twice as NVIDIA's Tegra driver does
void nvdec_vp8_setup(const struct vp8_dec *d, const struct vp8_picture *pic, const struct nvdec_surface *surf,
                     uint8_t *page)
{
    const struct vp8_frame *f = &pic->hdr;
    struct nvdec_vp8_sizes sizes;
    nvdec_vp8_sizes(surf, &sizes);
    struct nvdec_vp8_pic *out = (struct nvdec_vp8_pic *)page;
    struct nvdec_vp8_frame *fr = (struct nvdec_vp8_frame *)(page + NVDEC_SETUP_VP8);
    uint8_t seg_update = (uint8_t)(f->seg_enabled && f->seg_update_data);
    *out = (struct nvdec_vp8_pic){
        .gptimer_timeout = gptimer_timeout(f->width, f->height),
        .width = (uint16_t)ALIGN(f->width, 16),
        .height = (uint16_t)ALIGN(f->height, 16),
        .key_frame = (uint8_t)f->key,
        .version = (uint8_t)f->version,
        .tile_format = 1,
        .gob_height = 0,
        .error_conceal = 1,
        .first_part_size = f->first_part_size,
        .hist_buffer_size = sizes.history >> 8,
        .vld_buffer_size = pic->bytes,
        .frame_stride = { surf->pitch / 16, surf->pitch / 16 },
        .seg_feature_data_update = seg_update,
        .reserved0 = { seg_update },
    };
    *fr = (struct nvdec_vp8_frame){ { pic->slot, pic->slot, pic->slot } };
    for (int i = 0; d->valid && i < 3; i++)
        fr->refs[i] = d->refs[i];
}

void nvdec_h264_crop(const struct h264_sps *s, struct nvdec_crop *crop)
{
    *crop = (struct nvdec_crop){ s->crop_left, s->crop_top, s->width_mbs * 16 - s->crop_left - s->crop_right,
                                 s->height_mbs * 16 - s->crop_top - s->crop_bottom };
}

// where byte x of row y lies: two-GOB blocks, each GOB 64 bytes by 8 rows of 16-byte sectors
static size_t bl_offset(uint32_t pitch, uint32_t x, uint32_t y)
{
    size_t block = (size_t)(y / 16) * (pitch / 64) + x / 64;
    return block * 1024 + y % 16 / 8 * 512 + x % 64 / 32 * 256 + y % 8 / 4 * 128 + x % 32 / 16 * 64 + y % 4 * 16 +
           x % 16;
}

static void detile(const uint8_t *src, uint32_t pitch, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                   uint8_t *out, size_t out_pitch)
{
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w;) {
            uint32_t sx = x0 + x, run = 16 - sx % 16;
            if (run > w - x)
                run = w - x;
            memcpy(out + y * out_pitch + x, src + bl_offset(pitch, sx, y0 + y), run);
            x += run;
        }
    }
}

void nvdec_detile(const uint8_t *surface, const struct nvdec_surface *surf, const struct nvdec_crop *crop,
                  uint8_t *luma, size_t luma_pitch, uint8_t *chroma, size_t chroma_pitch)
{
    uint32_t x = crop->left * surf->bps, w = crop->width * surf->bps;
    detile(surface, surf->pitch, x, crop->top, w, crop->height, luma, luma_pitch);
    detile(surface + surf->chroma_offset, surf->pitch, x, crop->top / 2, w, crop->height / 2, chroma, chroma_pitch);
}

// a plane of w x h samples of size bytes each, out to aw x ah
static void extend(uint8_t *p, uint32_t pitch, uint32_t w, uint32_t h, uint32_t aw, uint32_t ah, uint32_t size)
{
    for (uint32_t y = 0; y < h; y++) {
        const uint8_t *edge = p + bl_offset(pitch, (w - 1) * size, y);  // a sample never straddles 16 bytes
        for (uint32_t x = w * size; x < aw * size; x++)
            p[bl_offset(pitch, x, y)] = edge[x % size];
    }
    for (uint32_t y = h; y < ah; y++)
        for (uint32_t x = 0; x < aw * size; x += 16)
            memcpy(p + bl_offset(pitch, x, y), p + bl_offset(pitch, x, h - 1), 16);
}

// NVDEC predicts VP9 from whole superblocks without the spec's edge clamp (8.5.2.3); repeated edges make up for it
void nvdec_extend(uint8_t *surface, const struct nvdec_surface *surf, uint32_t width, uint32_t height)
{
    uint32_t aw = ALIGN(width, 64), ah = ALIGN(height, 64), b = surf->bps;
    extend(surface, surf->pitch, width, height, aw, ah, b);
    extend(surface + surf->chroma_offset, surf->pitch, (width + 1) / 2, (height + 1) / 2, aw / 2, ah / 2, 2 * b);
}

const char *nvdec_codec_name(enum nvdec_codec codec)
{
    static const char *const names[] = { "H.264", "H.265", "AV1", "VP9", "VP8" };
    return names[codec];
}

int nvdec_stream_open(struct nvdec_stream *s, enum nvdec_codec codec)
{
    *s = (struct nvdec_stream){ .codec = codec };
    if (codec == NVDEC_H264) {
        s->h264 = mem_alloc(sizeof(*s->h264));
        s->h264_pic = mem_alloc(sizeof(*s->h264_pic));
        if (s->h264 && s->h264_pic)
            return 0;
    } else if (codec == NVDEC_H265) {
        s->h265 = mem_alloc(sizeof(*s->h265));
        s->h265_pic = mem_alloc(sizeof(*s->h265_pic));
        if (s->h265 && s->h265_pic)
            return 0;
    } else if (codec == NVDEC_AV1) {
        s->av1 = mem_alloc(sizeof(*s->av1));
        s->av1_pic = mem_alloc(sizeof(*s->av1_pic));
        if (s->av1 && s->av1_pic)
            return 0;
    } else if (codec == NVDEC_VP9) {
        s->vp9 = mem_alloc(sizeof(*s->vp9));
        s->vp9_pic = mem_alloc(sizeof(*s->vp9_pic));
        if (s->vp9 && s->vp9_pic)
            return 0;
    } else {
        s->vp8 = mem_alloc(sizeof(*s->vp8));
        s->vp8_pic = mem_alloc(sizeof(*s->vp8_pic));
        if (s->vp8 && s->vp8_pic)
            return 0;
    }
    nvdec_stream_close(s);
    return -1;
}

void nvdec_stream_close(struct nvdec_stream *s)
{
    if (s->vp9_pic)
        mem_free(s->vp9_pic);
    if (s->vp9)
        mem_free(s->vp9);
    if (s->vp8_pic)
        mem_free(s->vp8_pic);
    if (s->vp8)
        mem_free(s->vp8);
    if (s->av1_pic)
        mem_free(s->av1_pic);
    if (s->av1)
        mem_free(s->av1);
    if (s->h264_pic)
        mem_free(s->h264_pic);
    if (s->h264)
        mem_free(s->h264);
    if (s->h265_pic)
        mem_free(s->h265_pic);
    if (s->h265)
        mem_free(s->h265);
    *s = (struct nvdec_stream){ 0 };
}

static void h264_info(const struct h264_sps *sps, struct nvdec_pic *info)
{
    *info = (struct nvdec_pic){ .slots = sps->max_num_ref_frames + 1, .width_mbs = sps->width_mbs,
                                .height_mbs = sps->height_mbs, .bps = 1, .output = 1 };
    nvdec_h264_crop(sps, &info->crop);
}

static void h265_info(const struct h265_sps *sps, struct nvdec_pic *info)
{
    *info = (struct nvdec_pic){ .slots = sps->max_dec_pic_buffering,
                                .width_mbs = ALIGN(sps->width, 64) / 16,
                                .height_mbs = ALIGN(sps->height, 64) / 16,
                                .bps = sps->bit_depth > 8 ? 2 : 1,
                                .crop = { sps->crop_left, sps->crop_top, sps->width - sps->crop_left - sps->crop_right,
                                          sps->height - sps->crop_top - sps->crop_bottom } };
}

// surfaces hold whole superblocks (up to 128 samples); the sequence's max size holds every frame
static void av1_info(const struct av1_seq *seq, const struct av1_frame *f, struct nvdec_pic *info)
{
    *info = (struct nvdec_pic){ .slots = AV1_SLOTS,
                                .width_mbs = ALIGN(seq->max_width, 128) / 16,
                                .height_mbs = ALIGN(seq->max_height, 128) / 16,
                                .bps = seq->bit_depth > 8 ? 2 : 1,
                                .crop = { 0, 0, f ? f->upscaled_width : seq->max_width, f ? f->height : seq->max_height } };
}

// VP9 decodes whole 64x64 superblocks, VP8 16x16 macroblocks; the last key frame's size holds every frame
static void vp9_info(const struct vp9_dec *d, const struct vp9_frame *f, struct nvdec_pic *info)
{
    *info = (struct nvdec_pic){ .slots = VP9_SLOTS,
                                .width_mbs = ALIGN(d->width, 64) / 16,
                                .height_mbs = ALIGN(d->height, 64) / 16,
                                .bps = d->bit_depth > 8 ? 2 : 1,
                                .crop = { 0, 0, f ? f->width : d->width, f ? f->height : d->height } };
}

static void vp8_info(const struct vp8_dec *d, struct nvdec_pic *info)
{
    *info = (struct nvdec_pic){ .slots = VP8_SLOTS,
                                .width_mbs = ALIGN(d->width, 16) / 16,
                                .height_mbs = ALIGN(d->height, 16) / 16,
                                .bps = 1,
                                .crop = { 0, 0, d->width, d->height } };
}

void nvdec_stream_size(struct nvdec_stream *s, uint32_t width, uint32_t height, uint32_t bit_depth)
{
    if (s->vp9) {
        s->vp9->width = width;
        s->vp9->height = height;
        s->vp9->bit_depth = bit_depth;
    } else if (s->vp8) {
        s->vp8->width = width;
        s->vp8->height = height;
    }
}

int nvdec_stream_params(const struct nvdec_stream *s, struct nvdec_pic *info)
{
    if (s->av1 && s->av1->seq.valid) {
        av1_info(&s->av1->seq, NULL, info);
        return 1;
    }
    if (s->vp9 && s->vp9->width) {
        vp9_info(s->vp9, NULL, info);
        return 1;
    }
    if (s->vp8 && s->vp8->width) {
        vp8_info(s->vp8, info);
        return 1;
    }
    for (int i = 0; s->h264 && i < H264_MAX_SPS; i++) {
        if (s->h264->sps[i].valid) {
            h264_info(&s->h264->sps[i], info);
            return 1;
        }
    }
    for (int i = 0; s->h265 && i < H265_MAX_SPS; i++) {
        if (s->h265->sps[i].valid) {
            h265_info(&s->h265->sps[i], info);
            return 1;
        }
    }
    return 0;
}

int nvdec_stream_next(struct nvdec_stream *s, const uint8_t *stream, size_t len, size_t *pos, struct nvdec_pic *info)
{
    int r;
    if (s->av1) {
        const struct av1_picture *pic = s->av1_pic;
        if ((r = av1_next(s->av1, stream, len, pos, s->av1_pic)) <= 0)
            return r;
        av1_info(pic->seq, &pic->hdr, info);
        info->error_idx = 0;  // as NVIDIA's driver gives it (mesa-nova)
        info->poc = (int32_t)pic->index;  // frames are decoded in the order they're shown
        info->output = pic->output;
        info->existing = pic->existing;
        info->index = pic->index;
        info->slot = pic->slot;
        info->bytes = pic->bytes;
        return 1;
    }
    if (s->vp9) {
        const struct vp9_picture *pic = s->vp9_pic;
        if ((r = vp9_next(s->vp9, stream, len, pos, s->vp9_pic)) <= 0)
            return r;
        if (pic->header_len > NVDEC_SETUP_SIZE - NVDEC_SETUP_VP9_HEADER)
            return -1;
        vp9_info(s->vp9, &pic->hdr, info);
        info->poc = (int32_t)pic->index;
        info->output = pic->output;
        info->existing = pic->existing;
        info->index = pic->index;
        info->slot = pic->slot;
        info->bytes = pic->bytes;
        return 1;
    }
    if (s->vp8) {
        const struct vp8_picture *pic = s->vp8_pic;
        if ((r = vp8_next(s->vp8, stream, len, pos, s->vp8_pic)) <= 0)
            return r;
        vp8_info(s->vp8, info);
        info->poc = (int32_t)pic->index;
        info->output = pic->output;
        info->index = pic->index;
        info->slot = pic->slot;
        info->bytes = pic->bytes;
        return 1;
    }
    if (s->h264) {
        const struct h264_picture *pic = s->h264_pic;
        if ((r = h264_next(s->h264, stream, len, pos, s->h264_pic)) <= 0)
            return r;
        h264_info(pic->sps, info);
        uint32_t refs = pic->sps->max_num_ref_frames;
        info->error_idx = refs ? pic->index % refs : 0;
        info->poc = pic->poc[0] < pic->poc[1] ? pic->poc[0] : pic->poc[1];
        info->index = pic->index;
        info->slot = pic->slot;
        info->idr_count = pic->idr_count;
        info->bytes = pic->bytes;
    } else {
        const struct h265_picture *pic = s->h265_pic;
        if ((r = h265_next(s->h265, stream, len, pos, s->h265_pic)) <= 0)
            return r;
        h265_info(pic->sps, info);
        info->error_idx = pic->index % info->slots;
        info->poc = pic->poc;
        info->output = pic->output;
        info->index = pic->index;
        info->slot = pic->slot;
        info->idr_count = pic->idr_count;
        info->bytes = pic->bytes;
    }
    return 1;
}

void nvdec_stream_prepare(struct nvdec_stream *s, const uint8_t *stream, const struct nvdec_surface *surf,
                          uint8_t *page, uint8_t *bits)
{
    if (s->av1) {
        av1_tiles(s->av1_pic, stream, bits);
        memset(bits + s->av1_pic->bytes, 0, 64);
        nvdec_av1_setup(s->av1, s->av1_pic, surf, page);
    } else if (s->vp9) {
        memcpy(bits, stream + s->vp9_pic->tiles_pos, s->vp9_pic->bytes);
        memset(bits + s->vp9_pic->bytes, 0, 64);
        nvdec_vp9_setup(s->vp9, s->vp9_pic, surf, stream, page);
    } else if (s->vp8) {
        memcpy(bits, stream + s->vp8_pic->pos, s->vp8_pic->bytes);
        memset(bits + s->vp8_pic->bytes, 0, 64);
        nvdec_vp8_setup(s->vp8, s->vp8_pic, surf, page);
    } else if (s->h264) {
        h264_slices(s->h264_pic, stream, bits, (uint32_t *)(page + NVDEC_SETUP_SLICES));
        memset(bits + s->h264_pic->bytes, 0, 64);
        nvdec_h264_setup(s->h264, s->h264_pic, surf, (struct nvdec_h264_pic *)page);
    } else {
        h265_slices(s->h265_pic, stream, bits);
        memset(bits + s->h265_pic->bytes, 0, 64);
        nvdec_hevc_setup(s->h265, s->h265_pic, surf, page);
    }
}

void nvdec_stream_done(struct nvdec_stream *s)
{
    if (s->av1)
        av1_done(s->av1, s->av1_pic);
    else if (s->vp9)
        vp9_done(s->vp9, s->vp9_pic);
    else if (s->vp8)
        vp8_done(s->vp8, s->vp8_pic);
    else if (s->h264)
        h264_done(s->h264, s->h264_pic);
    else
        h265_done(s->h265, s->h265_pic);
}
