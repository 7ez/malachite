// AV1 low-overhead OBU streams parsed for NVDEC: sequence and frame headers, tile data spans, reference slots.
// Main profile (4:2:0, 8 and 10 bit), operating point 0, no scalability.
#pragma once
#include "plat.h"

#define AV1_REFS 8            // reference slots
#define AV1_REFS_PER_FRAME 7  // LAST_FRAME to ALTREF_FRAME
#define AV1_SLOTS (AV1_REFS + 1)  // surfaces: reference slots plus the frame being decoded
#define AV1_MAX_TILES 256  // what NVDEC's tile buffers hold (AV1_MAX_TILES in nvdec_drv.h)
#define AV1_MAX_TILE_COLS 64
#define AV1_MAX_TILE_ROWS 64
#define AV1_MAX_TILE_GROUPS AV1_MAX_TILES
#define AV1_MAX_OPERATING_POINTS 32
#define AV1_PRIMARY_REF_NONE 7

enum { AV1_KEY_FRAME, AV1_INTER_FRAME, AV1_INTRA_ONLY_FRAME, AV1_SWITCH_FRAME };
enum { AV1_WARP_IDENTITY, AV1_WARP_TRANSLATION, AV1_WARP_ROTZOOM, AV1_WARP_AFFINE };
enum { AV1_TX_ONLY_4X4, AV1_TX_LARGEST, AV1_TX_SELECT };

struct av1_seq {
    int valid;
    int reduced_still_picture;
    int decoder_model_info, equal_picture_interval;
    uint32_t buffer_delay_length, buffer_removal_time_length, frame_presentation_time_length;
    uint32_t nops;
    uint32_t op_idc[AV1_MAX_OPERATING_POINTS];
    int op_decoder_model[AV1_MAX_OPERATING_POINTS];
    uint32_t width_bits, height_bits, max_width, max_height;
    int frame_ids;
    uint32_t delta_frame_id_bits, additional_frame_id_bits;
    int sb128, filter_intra, intra_edge_filter, interintra, masked_compound, warped_motion, dual_filter;
    int order_hint, jnt_comp, ref_frame_mvs;
    uint32_t screen_content_tools, force_integer_mv;  // 2 lets each frame choose
    uint32_t order_hint_bits;
    int superres, cdef, restoration;
    uint32_t bit_depth;
    int separate_uv_delta_q;
    int film_grain_present;
};

struct av1_grain {
    int apply;
    uint32_t seed;
    uint32_t num_y, num_cb, num_cr;
    uint8_t y_value[14], y_scaling[14], cb_value[10], cb_scaling[10], cr_value[10], cr_scaling[10];
    int chroma_scaling_from_luma;
    uint32_t scaling_shift;  // grain_scaling_minus_8 + 8
    uint32_t ar_lag;
    int ar_y[24], ar_cb[25], ar_cr[25];  // coefficients, 128 taken off
    uint32_t ar_shift;                   // ar_coeff_shift_minus_6 + 6
    uint32_t grain_scale_shift;
    uint32_t cb_mult, cb_luma_mult, cb_offset, cr_mult, cr_luma_mult, cr_offset;
    int overlap, clip_to_restricted_range;
};

struct av1_frame {
    int show_existing;
    uint32_t existing_idx;  // the slot show_existing_frame shows
    uint32_t type;
    int show, showable, error_resilient, disable_cdf_update;
    int screen_content_tools, force_integer_mv;
    uint32_t order_hint, primary_ref, refresh;
    uint32_t width, height, upscaled_width, mi_cols, mi_rows;
    uint32_t render_width, render_height;
    int allow_intrabc, high_precision_mv, motion_switchable, use_ref_frame_mvs;
    uint32_t interp_filter;  // 4 for one chosen per block
    uint32_t ref_idx[AV1_REFS_PER_FRAME];  // slots of LAST_FRAME to ALTREF_FRAME
    int disable_frame_end_update_cdf;
    uint32_t tile_cols, tile_rows, tile_cols_log2, tile_rows_log2;
    uint32_t col_starts[AV1_MAX_TILE_COLS + 1], row_starts[AV1_MAX_TILE_ROWS + 1];  // in superblocks
    uint32_t context_update_tile_id, tile_size_bytes;
    uint32_t base_qindex;
    int32_t dq_y_dc, dq_u_dc, dq_u_ac, dq_v_dc, dq_v_ac;
    int using_qmatrix;
    uint32_t qm_y, qm_u, qm_v;
    int seg_enabled, seg_update_map, seg_temporal_update;
    uint8_t seg_feature_enabled[8];  // by segment, a bit for each feature
    int16_t seg_feature_data[8][8];
    int seg_preskip;
    uint32_t seg_last_active;
    int delta_q_present, delta_lf_present, delta_lf_multi;
    uint32_t delta_q_res, delta_lf_res;
    int coded_lossless, all_lossless;
    uint32_t lf_level[4], lf_sharpness;
    int lf_delta_enabled;
    int32_t lf_ref_deltas[8], lf_mode_deltas[2];
    uint32_t cdef_damping, cdef_bits;  // damping minus 3
    uint32_t cdef_y_pri[8], cdef_y_sec[8], cdef_uv_pri[8], cdef_uv_sec[8];  // as coded, so 3 for a strength of 4
    uint32_t lr_type[3], lr_unit_shift, lr_uv_shift;
    int uses_lr;
    uint32_t tx_mode;
    int reference_select, skip_mode_present;
    uint32_t skip_mode_frame[2];
    int allow_warped_motion, reduced_tx_set;
    uint32_t gm_type[AV1_REFS_PER_FRAME + 1];  // by reference frame, LAST_FRAME is 1
    int32_t gm_params[AV1_REFS_PER_FRAME + 1][6];
    struct av1_grain grain;
};

struct av1_ref {
    int valid;
    uint32_t slot;  // the surface it was decoded into
    uint32_t width, height, upscaled_width, mi_cols, mi_rows, render_width, render_height;
    uint32_t type, order_hint;
    uint32_t order_hints[AV1_REFS_PER_FRAME + 1];  // of its own references, by reference frame
    int showable;
    int32_t gm_params[AV1_REFS_PER_FRAME + 1][6];
    int32_t lf_ref_deltas[8], lf_mode_deltas[2];
    uint8_t seg_feature_enabled[8];
    int16_t seg_feature_data[8][8];
    struct av1_grain grain;
};

struct av1_picture {
    const struct av1_seq *seq;
    struct av1_frame hdr;
    int output;    // show_frame or show_existing_frame
    int existing;  // shows the already decoded frame in slot
    uint32_t slot;
    uint32_t index;  // in decoding order
    uint32_t ntiles, ngroups;
    // group_*: each tile group's data in the stream; tile_*: each tile's span in the groups concatenated
    uint32_t group_pos[AV1_MAX_TILE_GROUPS], group_len[AV1_MAX_TILE_GROUPS];
    uint32_t tile_start[AV1_MAX_TILES], tile_end[AV1_MAX_TILES];
    uint32_t bytes;  // of the tile data
};

struct av1_dec {
    struct av1_seq seq;
    struct av1_ref refs[AV1_REFS];
    uint32_t pictures;
};

void av1_init(struct av1_dec *d);
// next frame at *pos, advancing it: 1 frame, 0 end, -1 undecodable; pic->slot is a surface no reference holds
int av1_next(struct av1_dec *d, const uint8_t *s, size_t len, size_t *pos, struct av1_picture *pic);
// stores decoded pic in the reference slots it refreshes
void av1_done(struct av1_dec *d, const struct av1_picture *pic);
// copies pic's tile data as NVDEC reads it, pic->bytes long, to out
void av1_tiles(const struct av1_picture *pic, const uint8_t *s, uint8_t *out);
// get_relative_dist() of the spec
int av1_dist(const struct av1_seq *s, int a, int b);
