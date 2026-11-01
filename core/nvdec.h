// What NVDEC reads besides the bitstream: its H.264 picture setup and the buffers and surfaces around it. The
// layout is NVIDIA's nvdec_h264_pic_s (nvdec_drv.h); the buffer sizes and surface tiling are the ones nvk
// decodes with (vulkan/rust/video/decode/h264.rs, nil/tiling.rs).
#pragma once
#include "h264.h"

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

// a block-linear NV12 surface, the way NVDEC writes one: blocks a GOB (64 bytes) wide and two GOBs (16 rows) high
struct nvdec_surface {
    uint32_t pitch;  // bytes in a row of blocks, both planes
    uint32_t luma_rows, chroma_rows;
    uint64_t chroma_offset, size;
};

// the buffers a stream's pictures share
struct nvdec_h264_sizes {
    uint32_t coloc;  // a surface's colocated motion vectors, the buffer holding one for each
    uint32_t mbhist, history;
};

// the page a picture is set up in: the picture setup, then the slice offsets, then NVDEC's status
#define NVDEC_SETUP_SLICES 0x400
#define NVDEC_SETUP_STATUS 0x800
#define NVDEC_SETUP_SIZE 0x1000

void nvdec_h264_surface(uint32_t width_mbs, uint32_t height_mbs, struct nvdec_surface *surf);
void nvdec_h264_sizes(uint32_t width_mbs, uint32_t height_mbs, struct nvdec_h264_sizes *sizes);
void nvdec_h264_setup(const struct h264_dec *d, const struct h264_picture *pic, const struct nvdec_surface *surf,
                      struct nvdec_h264_pic *out);
// the cropped picture of a decoded surface as linear NV12, the luma and chroma rows each pitch bytes apart
void nvdec_detile_nv12(const uint8_t *surface, const struct nvdec_surface *surf, const struct h264_sps *s,
                       uint8_t *luma, size_t luma_pitch, uint8_t *chroma, size_t chroma_pitch);
uint32_t nvdec_crop_width(const struct h264_sps *s);
uint32_t nvdec_crop_height(const struct h264_sps *s);
