// H.265 Annex-B parsing for what NVDEC needs from the driver: parameter sets, slices, POC, RPS and the slice header
// bits NVDEC skips (it parses the rest itself). Main and Main10, one layer; the VPS is skipped.
#pragma once
#include "plat.h"

#define H265_MAX_SPS 16
#define H265_MAX_PPS 64
#define H265_MAX_RPS 64
#define H265_MAX_SLICES 600  // slice segments in a picture at the highest level
#define H265_DPB 16          // the reference pictures and the one being decoded
#define H265_MAX_LT 32
#define H265_MAX_TILE_COLS 20  // what NVDEC's tile size buffer holds (nvdec_drv.h GIP_ASIC_TILE_SIZE)
#define H265_MAX_TILE_ROWS 22

// scaling_list_data() lists in raster order: [4x4..32x32 (32x32 only matrices 0, 3)][intra Y Cb Cr, inter Y Cb Cr]
struct h265_scaling {
    uint8_t list[4][6][64];
    uint8_t dc[2][6];  // 16x16 and 32x32
};

struct h265_rps {
    uint8_t neg, pos;  // pictures before and after the current one
    int16_t delta[2][16];  // from the current picture's order count, nearest first
    uint8_t used[2][16];   // by the current picture
};

struct h265_sps {
    int valid;
    uint32_t profile;
    uint32_t width, height;  // luma samples
    uint32_t crop_left, crop_right, crop_top, crop_bottom;  // in luma samples
    uint32_t bit_depth;
    uint32_t log2_max_poc_lsb;
    uint32_t max_dec_pic_buffering;  // of the highest temporal layer
    uint32_t log2_min_cb, log2_ctb, log2_min_tb, log2_max_tb;
    uint32_t depth_inter, depth_intra;  // max_transform_hierarchy_depth
    int scaling_list_enabled;
    struct h265_scaling scaling;  // the SPS's lists, or the defaults
    int amp, sao;
    int pcm, pcm_loop_filter_disabled;
    uint32_t pcm_depth_luma, pcm_depth_chroma, log2_min_pcm, log2_max_pcm;
    uint32_t nrps;
    struct h265_rps rps[H265_MAX_RPS];
    int long_term_present;
    uint32_t nlt;  // num_long_term_ref_pics_sps
    uint32_t lt_poc_lsb[H265_MAX_LT];
    uint8_t lt_used[H265_MAX_LT];
    int temporal_mvp, strong_intra_smoothing;
};

struct h265_pps {
    int valid;
    uint32_t sps;
    int dependent_slices, output_flag_present;
    uint32_t extra_slice_header_bits;
    int sign_data_hiding, cabac_init_present;
    uint32_t num_ref_idx_l0, num_ref_idx_l1;  // the defaults
    int32_t init_qp;  // minus 26
    int constrained_intra_pred, transform_skip, cu_qp_delta;
    uint32_t diff_cu_qp_delta_depth;
    int32_t cb_qp_offset, cr_qp_offset;
    int slice_chroma_qp_offsets_present, weighted_pred, weighted_bipred, transquant_bypass;
    int tiles, entropy_coding_sync;
    uint32_t tile_cols, tile_rows;
    int uniform_spacing;
    uint32_t col_width[H265_MAX_TILE_COLS], row_height[H265_MAX_TILE_ROWS];  // in CTBs; all but the last of each
    int loop_filter_across_tiles, loop_filter_across_slices;
    int deblocking_control, deblocking_override, deblocking_disabled;
    int32_t beta_offset_div2, tc_offset_div2;
    int scaling_list_present;
    struct h265_scaling scaling;
    int lists_modification;
    uint32_t log2_parallel_merge_level;
    int slice_header_extension;
};

struct h265_slice {
    uint32_t nal_type, temporal_id;
    uint32_t pps;
    int no_output_of_prior_pics;
    uint32_t type;  // 0 B, 1 P, 2 I
    int pic_output;
    uint32_t poc_lsb;
    struct h265_rps rps;
    uint32_t rps_bits;            // of the short_term_ref_pic_set() the slice header holds, 0 for one of the SPS's
    uint32_t rps_ref_num_deltas;  // pictures in the set a slice header's is predicted from
    uint32_t nlt;
    uint32_t lt_poc_lsb[H265_MAX_LT];
    uint8_t lt_used[H265_MAX_LT], lt_msb_present[H265_MAX_LT];
    uint32_t lt_msb_cycle[H265_MAX_LT];
    uint32_t skip_bits;  // from after slice_type to slice_temporal_mvp_enabled_flag
};

// a picture in the decoded picture buffer
struct h265_ref {
    int used;
    int long_term;
    int32_t poc;
    uint32_t slot;  // the surface it was decoded into
};

struct h265_picture {
    const struct h265_sps *sps;
    const struct h265_pps *pps;
    struct h265_slice hdr;  // the first slice's
    int irap, idr, no_rasl;
    uint32_t nslices;
    uint32_t slice_pos[H265_MAX_SLICES], slice_len[H265_MAX_SLICES];  // each NAL in the stream, without start code
    uint32_t bytes;  // the slices with three-byte start codes
    int32_t poc;
    int output;
    uint32_t slot;
    uint32_t index;      // in decoding order
    uint32_t idr_count;  // POC restarts up to and including it
    // DPB indices its reference lists are built from
    uint32_t nbefore, nafter, nlt;
    uint8_t before[H265_DPB], after[H265_DPB], lt[H265_DPB];
};

struct h265_dec {
    struct h265_sps sps[H265_MAX_SPS];
    struct h265_pps pps[H265_MAX_PPS];
    struct h265_ref dpb[H265_DPB];
    int32_t prev_poc_lsb, prev_poc_msb;  // of the previous picture of the lowest temporal layer that others can use
    uint32_t pictures, idrs;
    int started;   // seen an IRAP picture to start from
    int eos;       // after an end of sequence NAL unit, the next picture restarts
    int skip_rasl;  // the last IRAP started a sequence, so its RASL pictures can't be decoded
};

// tile column widths and row heights in CTBs; -1 if the PPS's tiles don't fit the picture
int h265_tiles(const struct h265_sps *s, const struct h265_pps *p, uint32_t *cols, uint32_t *rows);
void h265_init(struct h265_dec *d);
// next picture from *pos (advanced past it): 1 picture, 0 end, -1 undecodable; pic->slot is free of references
int h265_next(struct h265_dec *d, const uint8_t *s, size_t len, size_t *pos, struct h265_picture *pic);
// pic is decoded: it's a reference now
void h265_done(struct h265_dec *d, const struct h265_picture *pic);
// pic's slices with three-byte start codes: pic->bytes into out
void h265_slices(const struct h265_picture *pic, const uint8_t *s, uint8_t *out);
