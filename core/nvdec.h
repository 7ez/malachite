// What NVDEC reads besides the bitstream: its H.264, H.265, AV1, VP9 and VP8 picture setups and the buffers and
// surfaces around them. The layouts are NVIDIA's nvdec_h264_pic_s, nvdec_hevc_pic_s, nvdec_av1_pic_s, nvdec_vp9_pic_s
// and nvdec_vp8_pic_s (nvdec_drv.h); the H.264 buffer sizes and the surface tiling are the ones nvk decodes with
// (vulkan/rust/video/decode/h264.rs, nil/tiling.rs).
#pragma once
#include "av1.h"
#include "h264.h"
#include "hevc.h"
#include "vp8.h"
#include "vp9.h"

struct nvdec_dpb_entry {
    uint32_t index : 7;  // the surface
    uint32_t col_idx : 5;  // its slice of the colocated motion vector buffer
    uint32_t state : 2;    // bit 0 the top field, bit 1 the bottom one used for reference
    uint32_t is_long_term : 1;
    uint32_t not_existing : 1;
    uint32_t is_field : 1;
    uint32_t top_field_marking : 4;  // 1 short-term, 2 long-term
    uint32_t bottom_field_marking : 4;
    uint32_t output_memory_layout : 1;
    uint32_t : 6;
    int32_t field_order_cnt[2];
    int32_t frame_idx;  // frame_num, or long_term_frame_idx for a long-term frame
};

struct nvdec_h264_pic {
    uint32_t encryption[13];
    uint8_t eos[16];
    uint8_t explicit_eos;
    uint8_t hint_dump;
    uint8_t reserved0[2];
    uint32_t stream_len;  // the slices and the end of stream marker
    uint32_t slice_count;
    uint32_t mbhist_buffer_size;
    uint32_t gptimer_timeout;
    int32_t log2_max_poc_lsb_minus4;
    int32_t delta_pic_order_always_zero;
    int32_t frame_mbs_only;
    int32_t width_mbs, height_mbs;
    uint32_t tile_format : 2;  // 0 pitch, 1 block-linear
    uint32_t gob_height : 3;   // log2 GOBs per block, minus 1
    uint32_t : 27;
    int32_t cabac;
    int32_t pic_order_present;
    int32_t num_ref_idx_l0_minus1, num_ref_idx_l1_minus1;
    int32_t deblocking_control;
    int32_t redundant_pic_cnt;
    int32_t transform_8x8;
    uint32_t pitch_luma, pitch_chroma;
    uint32_t luma_top_offset, luma_bot_offset, luma_frame_offset;
    uint32_t chroma_top_offset, chroma_bot_offset, chroma_frame_offset;
    uint32_t hist_buffer_size;  // in 256 bytes
    uint32_t mbaff : 1;
    uint32_t direct_8x8_inference : 1;
    uint32_t weighted_pred : 1;
    uint32_t constrained_intra_pred : 1;
    uint32_t ref_pic : 1;
    uint32_t field_pic : 1;
    uint32_t bottom_field : 1;
    uint32_t second_field : 1;
    uint32_t log2_max_frame_num_minus4 : 4;
    uint32_t chroma_format_idc : 2;
    uint32_t poc_type : 2;
    int32_t pic_init_qp_minus26 : 6;
    int32_t chroma_qp_index_offset : 5;
    int32_t second_chroma_qp_index_offset : 5;
    uint32_t weighted_bipred_idc : 2;
    uint32_t curr_pic_idx : 7;
    uint32_t curr_col_idx : 5;
    uint32_t frame_num : 16;
    uint32_t frame_surfaces : 1;
    uint32_t output_memory_layout : 1;  // 0 NV12
    int32_t curr_field_order_cnt[2];
    struct nvdec_dpb_entry dpb[16];
    uint8_t weight_scale[6][4][4];
    uint8_t weight_scale_8x8[2][8][8];
    uint8_t mvc[48];
    uint32_t lossless_ipred8x8_filter : 1;
    uint32_t transform_bypass : 1;
    uint32_t : 30;
    uint32_t display[7];
    uint32_t ssm[3];
};

_Static_assert(sizeof(struct nvdec_h264_pic) == 764, "nvdec_h264_pic_s is 764 bytes");

// what the H.265 Main10 and 4:4:4 extensions add to the picture setup; Main and Main10 use none of it but the two
// offsets
struct nvdec_hevc_ext {
    uint32_t sps_flags, pps_flags;
    int8_t cb_qp_adjustment[6], cr_qp_adjustment[6];
    uint32_t flt_above_offset, sao_above_offset;  // in the filter buffer, in 256 bytes
};

struct nvdec_hevc_pic {
    uint32_t encryption[12];
    uint32_t stream_len;  // the slices with their start codes
    uint32_t enable_encryption, encryption_flags;
    uint32_t gptimer_timeout;
    uint8_t tile_format : 2;  // 0 pitch, 1 block-linear
    uint8_t gob_height : 3;   // log2 GOBs per block, minus 1
    uint8_t : 3;
    uint8_t start_code;  // the stream has start codes
    uint8_t disp_output_mode;  // 1 for 10-bit
    uint8_t reserved0;
    uint32_t frame_stride[2];  // of the luma and chroma planes, in samples
    uint32_t col_mv_buffer_size;  // of one picture, in 256 bytes
    uint32_t sao_buffer_offset, bsd_ctrl_offset;  // in the filter buffer, in 256 bytes
    uint16_t width, height;
    uint32_t chroma_format_idc : 4;
    uint32_t bit_depth_luma : 4;
    uint32_t bit_depth_chroma : 4;
    uint32_t log2_min_cb : 4;
    uint32_t log2_max_cb : 4;
    uint32_t log2_min_tb : 4;
    uint32_t log2_max_tb : 4;
    uint32_t : 4;
    uint32_t max_transform_hierarchy_depth_inter : 3;
    uint32_t max_transform_hierarchy_depth_intra : 3;
    uint32_t scaling_list : 1;
    uint32_t amp : 1;
    uint32_t sao : 1;
    uint32_t pcm : 1;
    uint32_t pcm_bit_depth_luma : 4;
    uint32_t pcm_bit_depth_chroma : 4;
    uint32_t log2_min_pcm : 4;
    uint32_t log2_max_pcm : 4;
    uint32_t pcm_loop_filter_disabled : 1;
    uint32_t temporal_mvp : 1;
    uint32_t strong_intra_smoothing : 1;
    uint32_t : 3;
    uint32_t dependent_slice_segments : 1;
    uint32_t output_flag_present : 1;
    uint32_t num_extra_slice_header_bits : 3;
    uint32_t sign_data_hiding : 1;
    uint32_t cabac_init_present : 1;
    uint32_t num_ref_idx_l0_default : 4;
    uint32_t num_ref_idx_l1_default : 4;
    uint32_t init_qp : 7;  // with the bit depth's offset
    uint32_t constrained_intra_pred : 1;
    uint32_t transform_skip : 1;
    uint32_t cu_qp_delta : 1;
    uint32_t diff_cu_qp_delta_depth : 2;
    uint32_t : 5;
    int8_t cb_qp_offset, cr_qp_offset, beta_offset, tc_offset;
    uint32_t slice_chroma_qp_offsets_present : 1;
    uint32_t weighted_pred : 1;
    uint32_t weighted_bipred : 1;
    uint32_t transquant_bypass : 1;
    uint32_t tiles : 1;
    uint32_t entropy_coding_sync : 1;
    uint32_t tile_cols : 5;
    uint32_t tile_rows : 5;
    uint32_t loop_filter_across_tiles : 1;
    uint32_t loop_filter_across_slices : 1;
    uint32_t deblocking_control : 1;
    uint32_t deblocking_override : 1;
    uint32_t deblocking_disabled : 1;
    uint32_t lists_modification : 1;
    uint32_t log2_parallel_merge_level : 3;
    uint32_t slice_header_extension : 1;
    uint32_t : 6;
    uint8_t num_ref_frames;
    uint8_t reserved1;
    uint16_t long_term;  // bit 15 for the surface 0, bit 14 for 1, ...
    uint8_t init_ref_list[2][16];  // the surfaces of the initial reference lists
    int16_t ref_poc_diff[16];      // by surface: the current picture's order count minus its
    uint8_t idr, irap, curr_pic_idx, pattern_id;
    uint16_t sw_hdr_skip_length;
    uint16_t reserved2;
    uint32_t ecdma[6];
    uint32_t separate_colour_plane : 1;
    uint32_t log2_max_poc_lsb_minus4 : 4;
    uint32_t num_short_term_ref_pic_sets : 7;
    uint32_t num_long_term_ref_pics_sps : 6;
    uint32_t bit_parsing_disable : 1;
    uint32_t num_delta_pocs_of_rps_idx : 8;
    uint32_t long_term_ref_pics_present : 1;
    uint32_t : 4;
    uint32_t num_bits_short_term_ref_pics_in_slice;
    struct nvdec_hevc_ext ext;
    uint32_t ext_flags, mv_hevc, slice_decoding, slice_edge_offset;
    uint32_t ssm[3];
};

_Static_assert(sizeof(struct nvdec_hevc_pic) == 276, "nvdec_hevc_pic_s is 276 bytes");

struct nvdec_av1_frame_info {
    uint16_t width, height;
    uint16_t stride[2];  // of the luma and chroma planes, in 16 bytes
    uint32_t buffer_idx;  // 0: the decoder takes the references and the picture from fixed picture offsets
};

struct nvdec_av1_ref {
    struct nvdec_av1_frame_info info;
    int16_t x_scale, y_scale;  // 1 << 14 for the picture's size
    uint8_t sign_bias : 1;
    uint8_t warp_type : 2;
    uint8_t : 5;
    int16_t frame_off;  // the picture's order hint minus the reference's
    int16_t roffset;    // and the other way round
};

struct nvdec_av1_fgs {
    uint16_t apply_grain : 1;
    uint16_t overlap : 1;
    uint16_t clip_to_restricted_range : 1;
    uint16_t chroma_scaling_from_luma : 1;
    uint16_t has_y_points : 1;
    uint16_t has_cb_points : 1;
    uint16_t has_cr_points : 1;
    uint16_t scaling_shift : 4;
    uint16_t : 5;
    uint16_t random_seed;
    int16_t cb_offset, cr_offset;
    int8_t cb_mult, cb_luma_mult, cr_mult, cr_luma_mult;
};

struct nvdec_av1_pic {
    uint32_t encryption[13], ssm[3];
    struct nvdec_av1_fgs fgs;
    uint32_t gptimer_timeout;
    uint32_t stream_len;  // the tile data
    uint32_t skip_length;
    // the sequence header
    uint32_t use_128x128_superblock : 1;
    uint32_t chroma_format : 2;  // 1 for 4:2:0
    uint32_t bit_depth : 4;
    uint32_t enable_filter_intra : 1;
    uint32_t enable_intra_edge_filter : 1;
    uint32_t enable_interintra_compound : 1;
    uint32_t enable_masked_compound : 1;
    uint32_t enable_dual_filter : 1;
    uint32_t enable_order_hint : 1;
    uint32_t : 3;
    uint32_t enable_jnt_comp : 1;
    uint32_t : 1;
    uint32_t enable_cdef : 1;
    uint32_t : 1;
    uint32_t enable_fgs : 1;
    uint32_t enable_substream_decoding : 1;
    uint32_t : 10;
    // the frame header
    uint32_t frame_type : 2;
    uint32_t show_frame : 1;
    uint32_t : 1;
    uint32_t disable_cdf_update : 1;
    uint32_t allow_screen_content_tools : 1;
    uint32_t force_integer_mv : 1;
    uint32_t scale_denom_minus9 : 3;
    uint32_t allow_intrabc : 1;
    uint32_t allow_high_precision_mv : 1;
    uint32_t interp_filter : 3;
    uint32_t switchable_motion_mode : 1;
    uint32_t use_ref_frame_mvs : 1;
    uint32_t refresh_frame_context : 1;  // the tables it ends in are the next frames' to start from
    uint32_t delta_q_present : 1;
    uint32_t delta_q_res : 2;
    uint32_t delta_lf_present : 1;
    uint32_t delta_lf_res : 2;
    uint32_t delta_lf_multi : 1;
    uint32_t : 1;
    uint32_t coded_lossless : 1;
    uint32_t tile_enabled : 1;
    uint32_t : 2;
    uint32_t superres_is_scaled : 1;
    uint32_t : 1;
    uint32_t tile_cols : 8;
    uint32_t tile_rows : 8;
    uint32_t context_update_tile_id : 16;
    uint32_t cdef_damping_minus_3 : 2;
    uint32_t cdef_bits : 2;
    uint32_t tx_mode : 3;         // 0 only 4x4, 3 largest, 4 select
    uint32_t reference_mode : 2;  // 0 single, 2 select
    uint32_t skip_mode : 1;
    uint32_t skip_ref0 : 4;
    uint32_t skip_ref1 : 4;
    uint32_t allow_warp : 1;
    uint32_t reduced_tx_set : 1;
    uint32_t ref_scaling : 1;
    uint32_t : 11;
    uint16_t superres_upscaled_width;
    uint16_t superres_luma_step, superres_chroma_step, superres_init_luma_subpel_x, superres_init_chroma_subpel_x;
    uint8_t base_qindex;
    int8_t y_dc_delta_q, u_dc_delta_q, v_dc_delta_q, u_ac_delta_q, v_ac_delta_q;
    uint8_t qm_y, qm_u, qm_v;
    uint32_t cdef_y_pri_strength, cdef_uv_pri_strength;  // 4 bits for each strength
    uint32_t cdef_y_sec_strength : 16;                   // 2 bits
    uint32_t cdef_uv_sec_strength : 16;
    uint8_t segment_enabled, segment_update_map, reserved0, segment_temporal_update;
    int16_t segment_feature_data[8][8];
    uint8_t last_active_segid, segid_preskip, prevsegid, segment_quant_sign;
    uint8_t filter_level[2], filter_level_u, filter_level_v, lf_sharpness;
    int8_t lf_ref_deltas[8], lf_mode_deltas[2];
    uint8_t lr_type;       // 2 bits for each plane
    uint8_t lr_unit_size;  // as lr_type
    struct nvdec_av1_frame_info current;
    struct nvdec_av1_ref ref[7];  // LAST_FRAME to ALTREF_FRAME
    uint32_t use_temporal0_mvs : 1;
    uint32_t use_temporal1_mvs : 1;
    uint32_t use_temporal2_mvs : 1;
    uint32_t mf1_type : 3;  // the reference each projects motion vectors from, 1 for LAST_FRAME as AV1 names them
    uint32_t mf2_type : 3;
    uint32_t mf3_type : 3;
    uint32_t : 20;
    int16_t mfmv_offset[3][7];  // order hints from those references to their own
    int8_t mfmv_side[3][7];     // their own references come after them
    uint8_t tile_format : 2;  // 0 pitch, 1 block-linear
    uint8_t gob_height : 3;   // log2 GOBs per block, minus 1
    uint8_t : 3;
    uint8_t error_detection;
    // in the filter buffer, in 256 bytes
    uint32_t flt_top_offset, flt_vert_offset, cdef_vert_offset, lr_vert_offset, hus_vert_offset, fgs_vert_offset;
    uint32_t enable_histogram : 1;
    uint32_t sw_skip_start_length : 14;
    uint32_t : 17;
};

_Static_assert(sizeof(struct nvdec_av1_pic) == 572, "nvdec_av1_pic_s is 572 bytes");

struct nvdec_vp9_size {
    uint16_t width, height;
    uint16_t stride[2];  // of the luma and chroma planes, in samples
};

struct nvdec_vp9_pic {
    uint32_t encryption[12];
    uint32_t stream_len;  // the tile data
    uint32_t enable_encryption, encryption_flags;
    uint32_t gptimer_timeout;
    uint8_t tile_format : 2;  // 0 pitch, 1 block-linear
    uint8_t gob_height : 3;   // log2 GOBs per block, minus 1
    uint8_t : 3;
    uint8_t reserved0[3];
    uint32_t bsd_ctrl_offset;      // in the filter buffer, in 256 bytes
    struct nvdec_vp9_size ref[3];  // LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME
    struct nvdec_vp9_size current;
    uint8_t key_frame : 1;
    uint8_t prev_key_frame : 1;     // of the frame decoded before, which with the next three says whether its motion
    uint8_t resolution_change : 1;  // vectors are used
    uint8_t error_resilient : 1;
    uint8_t prev_show_frame : 1;
    uint8_t intra_only : 1;
    uint8_t : 2;
    uint8_t reserved1[3];
    uint8_t ref_sign_bias[4];  // by reference frame
    int8_t lf_level, lf_sharpness;
    uint8_t base_q;
    int8_t dq_y_dc, dq_uv_ac, dq_uv_dc;
    int8_t lossless, tx_mode, allow_hp, interp_filter, reference_mode, comp_fixed_ref, comp_var_ref[2];
    int8_t log2_tile_cols, log2_tile_rows;
    uint8_t seg_enabled, seg_update_map, seg_temporal_update, seg_abs;
    uint8_t seg_feature_enabled[8][4];  // alternative quantiser, loop filter level, reference frame, skip
    int16_t seg_feature_data[8][4];
    int8_t lf_delta_enabled, lf_ref_deltas[4], lf_mode_deltas[2];
    int8_t reserved2;
    uint32_t flt_above_offset;  // in the filter buffer, in 256 bytes
    uint32_t external_ref_mem_dis : 1;
    uint32_t bit_depth : 4;
    uint32_t error_recovery_start_pos : 2;
    uint32_t error_external_mv_en : 1;
    uint32_t : 24;
    uint32_t ssm[3];
};

_Static_assert(sizeof(struct nvdec_vp9_pic) == 256, "nvdec_vp9_pic_s is 256 bytes");

struct nvdec_vp8_pic {
    uint32_t encryption[13];
    uint32_t gptimer_timeout;
    uint16_t width, height;  // in whole macroblocks
    uint8_t key_frame, version;
    uint8_t tile_format : 2;  // 0 pitch, 1 block-linear
    uint8_t gob_height : 3;   // log2 GOBs per block, minus 1
    uint8_t : 3;
    uint8_t error_conceal;
    uint32_t first_part_size;
    uint32_t hist_buffer_size;  // in 256 bytes
    uint32_t vld_buffer_size;   // the bitstream
    uint32_t frame_stride[2];   // of the luma and chroma planes, in 16 bytes
    uint32_t luma_top_offset, luma_bot_offset, luma_frame_offset;
    uint32_t chroma_top_offset, chroma_bot_offset, chroma_frame_offset;
    uint32_t display[7];
    int8_t current_output_memory_layout, output_memory_layout[3];
    uint8_t seg_feature_data_update;
    uint8_t reserved0[3];
    uint32_t result;  // what NVDEC found in the frame header
    uint32_t partition_offset[8];
    uint32_t ssm[3];
};

_Static_assert(sizeof(struct nvdec_vp8_pic) == 0xc0, "nvdec_vp8_pic_s is 0xc0 bytes");

// a block-linear NV12 or, with two bytes a sample, P010 surface, the way NVDEC writes one: blocks a GOB (64 bytes) wide
// and two GOBs (16 rows) high
struct nvdec_surface {
    uint32_t width_mbs, height_mbs, bps;  // the pictures it holds are as big as this, a sample takes bps bytes
    uint32_t pitch;  // bytes in a row of blocks, both planes
    uint32_t luma_rows, chroma_rows;
    uint64_t chroma_offset, size;
};

// the buffers an H.264 stream's pictures share
struct nvdec_h264_sizes {
    uint32_t coloc;  // a surface's colocated motion vectors, the buffer holding one for each
    uint32_t mbhist, history;
};

// the buffers an H.265 stream's pictures share; the filter buffer holds five regions, at these offsets
struct nvdec_hevc_sizes {
    uint32_t coloc;  // as H.264's
    uint32_t filter, sao, flt_above, sao_above, bsd;
    uint32_t intra_top;
};

// the buffers an AV1 stream's pictures share. A surface has its own colocated motion vectors and segmentation map,
// and there is one more map of zeros to predict from; the filter buffer holds six regions, at these offsets.
struct nvdec_av1_sizes {
    uint32_t coloc, segmap;
    uint32_t filter, flt_top, flt_vert, cdef_vert, lr_vert, hus_vert, fgs_vert;
    uint32_t intra_top;
    uint32_t prob;  // one table of probabilities (av1cdf.h), and where NVDEC writes the next
};

// what the kernel needs of an AV1 picture besides NVDEC's setup: which of the tables and maps it carries from
// picture to picture the engine reads, and where the picture's go. Surfaces and reference slots.
struct nvdec_av1_frame {
    uint32_t refresh;  // the reference slots the picture goes in
    uint32_t primary;  // the slot whose probabilities it starts from, NVDEC_AV1_DEFAULT for the defaults
    uint32_t coloc[3];  // the surfaces its motion vectors are projected from
    uint32_t segmap;    // and its segmentation map is predicted from, NVDEC_AV1_DEFAULT for zeros
    uint32_t refs[7];   // the surfaces of LAST_FRAME to ALTREF_FRAME, the picture's own for an intra picture
};

#define NVDEC_AV1_DEFAULT 0xff

// the buffers a VP9 stream's frames share. NVDEC keeps two motion vector buffers and two segmentation maps, the last
// frame's and the one being decoded; the filter buffer holds three regions, at these offsets.
struct nvdec_vp9_sizes {
    uint32_t coloc, segmap;  // one of each
    uint32_t filter, flt_above, bsd;
    uint32_t intra_top;
    uint32_t prob, counts;
};

// what the kernel needs of a VP9 frame besides NVDEC's setup: its references' surfaces, the probabilities it starts
// from and what becomes of them, and which motion vectors and segmentation map it predicts from
struct nvdec_vp9_frame {
    uint32_t refs[3];  // the surfaces of LAST_FRAME to ALTREF_FRAME, the frame's own for an intra frame
    uint32_t reset;    // the frame contexts set to the defaults first, a bit each
    uint32_t context;  // the one it loads, and saves back with refresh_context
    uint32_t refresh_context;
    uint32_t adapt;    // the probabilities saved back are first adapted to what the frame decoded
    uint32_t mv_map, seg_map;  // the buffers holding the last frame's; the frame's go in the other ones
    uint32_t clear_seg_map;    // it predicts segment ids from zeros
    uint32_t header_len;       // the compressed header, at NVDEC_SETUP_VP9_HEADER
    uint8_t seg_tree[7], seg_pred[3];
};

// the buffers a VP8 stream's frames share
struct nvdec_vp8_sizes {
    uint32_t history, prob;
};

// the surfaces of a VP8 frame's references: golden, altref and last, the frame's own when there is none yet
struct nvdec_vp8_frame {
    uint32_t refs[3];
};

// the page a picture is set up in: the picture setup, then H.264's slice offsets, H.265's scaling lists, AV1's
// frame and global motion or VP9's and VP8's frame, then NVDEC's status, then H.265's and VP9's tile sizes or AV1's
// tiles, then AV1's film grain tables or VP9's compressed header
#define NVDEC_SETUP_SLICES 0x400
#define NVDEC_SETUP_SCALING 0x400
#define NVDEC_SETUP_AV1 0x400
#define NVDEC_SETUP_VP9 0x400
#define NVDEC_SETUP_VP8 0x400
#define NVDEC_SETUP_GLOBAL 0x500  // AV1_GLOBAL_MOTION_BUF_SIZE
#define NVDEC_SETUP_STATUS 0x800
#define NVDEC_SETUP_TILES 0x1000  // AV1: AV1TileInfo_OLD's 16 bytes for each of AV1_MAX_TILES
#define NVDEC_SETUP_GRAIN 0x2400  // AV1_FGS_BUF_SIZE
#define NVDEC_SETUP_VP9_HEADER 0x2400
#define NVDEC_SETUP_SIZE 0x6000

#define NVDEC_SLOTS (H264_DPB + 1)  // the most surfaces any stream has

enum nvdec_codec { NVDEC_H264, NVDEC_H265, NVDEC_AV1, NVDEC_VP9, NVDEC_VP8 };

struct nvdec_crop {
    uint32_t left, top, width, height;
};

void nvdec_surface_init(uint32_t width_mbs, uint32_t height_mbs, uint32_t bps, struct nvdec_surface *surf);
void nvdec_h264_sizes(uint32_t width_mbs, uint32_t height_mbs, struct nvdec_h264_sizes *sizes);
void nvdec_hevc_sizes(const struct nvdec_surface *surf, struct nvdec_hevc_sizes *sizes);
void nvdec_av1_sizes(const struct nvdec_surface *surf, struct nvdec_av1_sizes *sizes);
void nvdec_vp9_sizes(const struct nvdec_surface *surf, struct nvdec_vp9_sizes *sizes);
void nvdec_vp8_sizes(const struct nvdec_surface *surf, struct nvdec_vp8_sizes *sizes);
void nvdec_h264_setup(const struct h264_dec *d, const struct h264_picture *pic, const struct nvdec_surface *surf,
                      struct nvdec_h264_pic *out);
// the setup page of an H.265 picture: its setup, scaling lists and tile sizes
void nvdec_hevc_setup(const struct h265_dec *d, const struct h265_picture *pic, const struct nvdec_surface *surf,
                      uint8_t *page);
// the setup page of an AV1 picture: its setup, frame, global motion, tile sizes and spans, and film grain
void nvdec_av1_setup(const struct av1_dec *d, const struct av1_picture *pic, const struct nvdec_surface *surf,
                     uint8_t *page);
// the setup page of a VP9 frame: its setup, frame, tile sizes and compressed header
void nvdec_vp9_setup(const struct vp9_dec *d, const struct vp9_picture *pic, const struct nvdec_surface *surf,
                     const uint8_t *stream, uint8_t *page);
// the setup page of a VP8 frame: its setup and frame
void nvdec_vp8_setup(const struct vp8_dec *d, const struct vp8_picture *pic, const struct nvdec_surface *surf,
                     uint8_t *page);
void nvdec_h264_crop(const struct h264_sps *s, struct nvdec_crop *crop);
// the cropped picture of a decoded surface as linear NV12 or P010, the luma and chroma rows each pitch bytes apart
void nvdec_detile(const uint8_t *surface, const struct nvdec_surface *surf, const struct nvdec_crop *crop,
                  uint8_t *luma, size_t luma_pitch, uint8_t *chroma, size_t chroma_pitch);
// the edge samples of a width x height picture in a surface repeated out to its 64x64 superblocks
void nvdec_extend(uint8_t *surface, const struct nvdec_surface *surf, uint32_t width, uint32_t height);
const char *nvdec_codec_name(enum nvdec_codec codec);

// What a decoder asks of either parser: the next picture of the stream and the setup of it
struct nvdec_pic {
    uint32_t index;  // in decoding order
    uint32_t slot;   // the surface to decode into
    uint32_t slots;  // the surfaces the stream's pictures and references take
    uint32_t idr_count;  // display order: by idr_count, then poc
    int32_t poc;
    int output;  // whether it's to be shown
    int existing;  // it's shown as it was decoded, and there is nothing to decode
    uint32_t bytes;  // of the slices, without padding
    uint32_t width_mbs, height_mbs, bps;  // of the surfaces
    struct nvdec_crop crop;
    uint32_t error_idx;  // for error concealment
};

struct nvdec_stream {
    enum nvdec_codec codec;
    struct h264_dec *h264;
    struct h264_picture *h264_pic;
    struct h265_dec *h265;
    struct h265_picture *h265_pic;
    struct av1_dec *av1;
    struct av1_picture *av1_pic;
    struct vp9_dec *vp9;
    struct vp9_picture *vp9_pic;
    struct vp8_dec *vp8;
    struct vp8_picture *vp8_pic;
};

int nvdec_stream_open(struct nvdec_stream *s, enum nvdec_codec codec);
void nvdec_stream_close(struct nvdec_stream *s);
// VP9 and VP8 have no parameter sets: the size and bit depth the container gives a stream, which its first key frame
// replaces
void nvdec_stream_size(struct nvdec_stream *s, uint32_t width, uint32_t height, uint32_t bit_depth);
// the pictures a first valid parameter set describes, nothing of the picture itself set: 0 if none yet
int nvdec_stream_params(const struct nvdec_stream *s, struct nvdec_pic *info);
// the next picture from *pos in stream: 1 for one, 0 at the end, -1 for a stream it can't decode
int nvdec_stream_next(struct nvdec_stream *s, const uint8_t *stream, size_t len, size_t *pos, struct nvdec_pic *info);
// write the picture's slices to bits, which takes 64 bytes more, and its setup to the setup page
void nvdec_stream_prepare(struct nvdec_stream *s, const uint8_t *stream, const struct nvdec_surface *surf,
                          uint8_t *page, uint8_t *bits);
// the picture is decoded: it's a reference now
void nvdec_stream_done(struct nvdec_stream *s);
