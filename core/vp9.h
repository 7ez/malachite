// VP9 streams, an IVF file or a container sample (a frame or a superframe) at a time, down to what NVDEC needs from
// the driver: each frame's uncompressed header, where its compressed header and its tile data lie, its reference slots
// and what it carries over from the frame decoded before it. NVDEC decodes the tile data; the probabilities the
// compressed header updates are the kernel's to keep (core/vp9prob.h). Profiles 0 and 2: 4:2:0, 8 and 10 bit.
#pragma once
#include "plat.h"
#include "vp9prob.h"

#define VP9_REFS 8
#define VP9_SLOTS (VP9_REFS + 1)  // surfaces: the frames in the reference slots and the one being decoded
#define VP9_MAX_FRAMES 8          // in a superframe
#define VP9_MAX_TILE_COLS 64
#define VP9_MAX_TILE_ROWS 4

enum { VP9_KEY_FRAME, VP9_NON_KEY_FRAME };
enum { VP9_LAST_FRAME = 1, VP9_GOLDEN_FRAME, VP9_ALTREF_FRAME };
#define VP9_SWITCHABLE 4

struct vp9_frame {
    int show_existing;
    uint32_t existing_idx;  // the slot show_existing_frame shows
    uint32_t profile, type;
    int show, error_resilient, intra_only, intra;  // intra: a key frame or an intra-only one
    uint32_t reset_context;
    uint32_t bit_depth;
    uint32_t refresh;
    uint32_t ref_idx[3];  // slots of LAST_FRAME, GOLDEN_FRAME and ALTREF_FRAME
    int sign_bias[4];     // by reference frame, LAST_FRAME at 1
    uint32_t width, height;
    int allow_hp;
    uint32_t interp_filter;  // as coded: 0 smooth, 1 regular, 2 sharp, 3 bilinear; VP9_SWITCHABLE by block
    int refresh_context, parallel;
    uint32_t context;  // frame_context_idx as coded, which an intra or error resilient frame doesn't load
    uint32_t lf_level, lf_sharpness;
    int lf_delta_enabled;
    int32_t lf_ref_deltas[4], lf_mode_deltas[2];
    uint32_t base_q;
    int32_t dq_y_dc, dq_uv_dc, dq_uv_ac;
    int lossless;
    int seg_enabled, seg_update_map, seg_temporal_update, seg_abs;
    uint8_t seg_tree[7], seg_pred[3];  // 255 for those not coded
    uint8_t seg_feature_enabled[8][4];
    int16_t seg_feature_data[8][4];
    uint32_t tile_cols_log2, tile_rows_log2;
    uint32_t tx_mode, reference_mode;  // of the compressed header
    uint32_t comp_fixed_ref, comp_var_ref[2];
};

// a frame in a reference slot
struct vp9_ref {
    int valid;
    uint32_t slot;  // the surface it was decoded into
    uint32_t width, height;
};

struct vp9_picture {
    struct vp9_frame hdr;
    int output;    // shown: show_frame, or show_existing_frame
    int existing;  // shows a frame already decoded, in slot
    uint32_t slot;
    uint32_t index;  // in decoding order
    uint32_t header_pos, header_len;  // the compressed header in the stream
    uint32_t tiles_pos, bytes;        // and the tile data after it
};

struct vp9_dec {
    struct vp9_ref refs[VP9_REFS];
    uint32_t width, height, bit_depth;  // of the surfaces: the last key frame's, or what the container says
    // what the next frame's header starts from
    int lf_delta_enabled;
    int32_t lf_ref_deltas[4], lf_mode_deltas[2];
    int seg_abs;
    uint8_t seg_feature_enabled[8][4];
    int16_t seg_feature_data[8][4];
    // the frame decoded last, which NVDEC predicts motion vectors and segment ids from
    int have_last, last_key, last_show;
    uint32_t last_width, last_height;
    uint32_t seg_map, mv_map;  // which of NVDEC's two segmentation maps and motion vector buffers hold its
    // the frames of the superframe being read, in the stream
    uint32_t frame_pos[VP9_MAX_FRAMES], frame_len[VP9_MAX_FRAMES];
    uint32_t frames, next_frame;
    uint32_t pictures;
    struct vp9_probs scratch;  // what compressed headers are parsed into, the probabilities being the kernel's
};

void vp9_init(struct vp9_dec *d);
// the next frame from stream position *pos, which moves past its superframe: 1 for a frame, 0 at the end, -1 for a
// stream it can't decode. pic->slot is a surface no reference slot holds.
int vp9_next(struct vp9_dec *d, const uint8_t *s, size_t len, size_t *pos, struct vp9_picture *pic);
// pic is decoded: it goes in the reference slots it refreshes, and is the frame decoded last
void vp9_done(struct vp9_dec *d, const struct vp9_picture *pic);
// the boundaries of the tiles, in 64x64 superblocks: n + 1 of them for 1 << log2 tiles across mi 8x8 blocks
void vp9_tile_starts(uint32_t mi, uint32_t log2, uint32_t *starts);
