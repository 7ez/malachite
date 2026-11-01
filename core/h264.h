// H.264 Annex-B streams down to what NVDEC needs from the driver: the parameter sets, each picture's slices,
// its picture order count and the reference pictures it may use. NVDEC parses slice data and builds the
// reference lists itself. Progressive (frame_mbs_only) 4:2:0 8-bit streams without slice groups or scaling
// matrices: Baseline, Main and High encoded that way.
#pragma once
#include "plat.h"

#define H264_MAX_SPS 32
#define H264_MAX_PPS 256
#define H264_MAX_SLICES 255
#define H264_DPB 16
#define H264_SLOTS (H264_DPB + 1)  // the reference frames and the picture being decoded

struct h264_sps {
    int valid;
    uint32_t profile, level;
    uint32_t chroma_format, bit_depth_luma, bit_depth_chroma;
    int transform_bypass;
    uint32_t log2_max_frame_num, poc_type, log2_max_poc_lsb;
    int delta_pic_order_always_zero;
    int32_t offset_for_non_ref_pic, offset_for_top_to_bottom;
    uint32_t poc_cycle;
    int32_t offset_for_ref_frame[255];
    uint32_t max_num_ref_frames;
    int gaps_allowed;
    uint32_t width_mbs, height_mbs;
    int frame_mbs_only, mbaff, direct_8x8_inference;
    uint32_t crop_left, crop_right, crop_top, crop_bottom;  // in luma samples
};

struct h264_pps {
    int valid;
    uint32_t sps;
    int cabac, bottom_field_pic_order;
    uint32_t num_ref_idx_l0, num_ref_idx_l1;  // the defaults, minus 1
    int weighted_pred;
    uint32_t weighted_bipred_idc;
    int32_t pic_init_qp, chroma_qp_index_offset, second_chroma_qp_index_offset;  // pic_init_qp minus 26
    int deblocking_control, constrained_intra_pred, redundant_pic_cnt, transform_8x8;
};

#define H264_MMCO_MAX 66

struct h264_mmco {
    uint32_t op, pic_num_diff, long_term_pic_num, long_term_frame_idx, max_long_term_frame_idx;
};

struct h264_slice {
    uint32_t nal_type, nal_ref_idc;
    uint32_t first_mb, type, pps;  // type modulo 5: 0 P, 1 B, 2 I, 3 SP, 4 SI
    uint32_t frame_num, idr_pic_id, poc_lsb;
    int32_t delta_poc_bottom, delta_poc[2];
    int long_term_reference;  // an IDR's
    int adaptive_marking;
    uint32_t nmmco;
    struct h264_mmco mmco[H264_MMCO_MAX];
};

// a frame in the decoded picture buffer
struct h264_ref {
    int used;  // marked for reference
    int long_term, non_existing;
    uint32_t frame_num, long_term_frame_idx;
    int32_t frame_num_wrap;
    int32_t poc[2];
    uint32_t slot;  // the surface it was decoded into
};

struct h264_picture {
    const struct h264_sps *sps;
    const struct h264_pps *pps;
    struct h264_slice hdr;  // the first slice's
    int idr;
    uint32_t nslices;
    uint32_t slice_pos[H264_MAX_SLICES], slice_len[H264_MAX_SLICES];  // each NAL in the stream, without start code
    uint32_t bytes;  // the slices with three-byte start codes
    int32_t poc[2];
    uint32_t slot;
    uint32_t index;  // in decoding order
    uint32_t idr_count;  // IDR pictures before it and itself: a new count of picture order
};

struct h264_dec {
    struct h264_sps sps[H264_MAX_SPS];
    struct h264_pps pps[H264_MAX_PPS];
    struct h264_ref dpb[H264_DPB];
    uint32_t max_long_term_frame_idx;  // plus 1, 0 for none
    // picture order count state from the previous (reference) picture
    int32_t prev_poc_msb, prev_poc_lsb, prev_frame_num_offset;
    uint32_t prev_frame_num, prev_ref_frame_num;
    int prev_mmco5;
    uint32_t pictures, idrs;
    int started;  // seen an IDR to start from
};

void h264_init(struct h264_dec *d);
// the next picture from stream position *pos, which moves past it: 1 for a picture, 0 at the end, -1 for a stream
// it can't decode. pic->slot is a surface no reference frame is in.
int h264_next(struct h264_dec *d, const uint8_t *s, size_t len, size_t *pos, struct h264_picture *pic);
// pic is decoded: mark the references it leaves
void h264_done(struct h264_dec *d, const struct h264_picture *pic);
// the slices of pic as NVDEC reads them, three-byte start codes and all, and where each starts: pic->bytes into out,
// pic->nslices + 1 offsets (the last one the end)
void h264_slices(const struct h264_picture *pic, const uint8_t *s, uint8_t *out, uint32_t *offsets);
