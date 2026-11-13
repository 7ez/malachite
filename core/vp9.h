// VP9 parsed for NVDEC, from IVF or one container sample (frame or superframe) at a time: uncompressed header,
// compressed header and tile data spans, reference slots, state carried from the previous frame. The kernel keeps
// the probabilities (core/vp9prob.h). Profiles 0 and 2: 4:2:0, 8 and 10 bit.
#pragma once
#include "plat.h"
#include "vp9prob.h"

#define VP9_REFS 8
#define VP9_SLOTS (VP9_REFS + 1)  // surfaces: reference slots plus the frame being decoded
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
    int show, error_resilient, intra_only, intra;  // intra: key or intra-only
    uint32_t reset_context;
    uint32_t bit_depth;
    uint32_t refresh;
    uint32_t ref_idx[3];  // slots of LAST_FRAME, GOLDEN_FRAME and ALTREF_FRAME
    int sign_bias[4];     // by reference frame, LAST_FRAME at 1
    uint32_t width, height;
    int allow_hp;
    uint32_t interp_filter;  // as coded: 0 smooth, 1 regular, 2 sharp, 3 bilinear; VP9_SWITCHABLE by block
    int refresh_context, parallel;
    uint32_t context;  // frame_context_idx as coded; intra and error resilient frames don't load it
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
    uint32_t tx_mode, reference_mode;  // from the compressed header
    uint32_t comp_fixed_ref, comp_var_ref[2];
};

struct vp9_ref {
    int valid;
    uint32_t slot;  // the surface it was decoded into
    uint32_t width, height;
};

struct vp9_picture {
    struct vp9_frame hdr;
    int output;    // show_frame or show_existing_frame
    int existing;  // shows the already decoded frame in slot
    uint32_t slot;
    uint32_t index;  // in decoding order
    uint32_t header_pos, header_len;  // the compressed header in the stream
    uint32_t tiles_pos, bytes;        // and the tile data after it
};

struct vp9_dec {
    struct vp9_ref refs[VP9_REFS];
    uint32_t width, height, bit_depth;  // surfaces: last key frame's, or the container's
    // carried into the next frame's header
    int lf_delta_enabled;
    int32_t lf_ref_deltas[4], lf_mode_deltas[2];
    int seg_abs;
    uint8_t seg_feature_enabled[8][4];
    int16_t seg_feature_data[8][4];
    // last decoded frame, which NVDEC predicts motion vectors and segment ids from
    int have_last, last_key, last_show;
    uint32_t last_width, last_height;
    uint32_t seg_map, mv_map;  // which of NVDEC's two segmentation maps and motion vector buffers hold its
    // frames of the superframe being read
    uint32_t frame_pos[VP9_MAX_FRAMES], frame_len[VP9_MAX_FRAMES];
    uint32_t frames, next_frame;
    uint32_t pictures;
    struct vp9_probs scratch;  // compressed headers parse into this; the kernel keeps the real probabilities
};

void vp9_init(struct vp9_dec *d);
// next frame at *pos, advancing past its superframe: 1 frame, 0 end, -1 undecodable; pic->slot is unreferenced
int vp9_next(struct vp9_dec *d, const uint8_t *s, size_t len, size_t *pos, struct vp9_picture *pic);
// stores decoded pic in the slots it refreshes and as the last frame
void vp9_done(struct vp9_dec *d, const struct vp9_picture *pic);
// (1 << log2) + 1 tile boundaries in 64x64 superblocks, for mi 8x8 blocks
void vp9_tile_starts(uint32_t mi, uint32_t log2, uint32_t *starts);
