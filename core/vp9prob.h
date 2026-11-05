// VP9's probabilities: the frame contexts the driver keeps, the compressed header's updates to them and their
// backward adaptation from what a frame decoded (VP9 specification 6.3 and 8.4, as libvpx does them), and the tables
// NVDEC reads them from and counts the frame's symbols in (nvdec_vp9EntropyProbs_t and nvdec_vp9EntropyCounts_t of
// nvdec_drv.h). NVDEC decodes a frame with the probabilities it is given and leaves all of this to the driver.
#pragma once
#include "plat.h"

#define VP9_PROBS_SIZE 3744  // NVDEC's probability table
#define VP9_CONTEXTS 4

struct vp9_mv_comp {
    uint8_t sign, classes[10], class0[1], bits[10], class0_fp[2][3], fp[3], class0_hp, hp;
};

// a frame context, libvpx's FRAME_CONTEXT
struct vp9_probs {
    uint8_t y_mode[4][9], uv_mode[10][9];
    uint8_t partition[16][3];
    uint8_t coef[4][2][2][6][6][3];  // by transform size, plane, reference, band and context
    uint8_t switchable_interp[4][2];
    uint8_t inter_mode[7][3];
    uint8_t intra_inter[4], comp_inter[5], single_ref[5][2], comp_ref[5];
    uint8_t tx32[2][3], tx16[2][2], tx8[2][1];
    uint8_t skip[3];
    struct {
        uint8_t joints[3];
        struct vp9_mv_comp comps[2];
    } mv;
};

// What NVDEC counts while it decodes a frame, symbols in libvpx's order of them but for the inter modes, which are
// counted by branch of their tree: ZEROMV or not, NEARESTMV or not, NEARMV or NEWMV
struct vp9_counts {
    uint32_t inter_mode[7][3][2];
    uint32_t y_mode[4][10], uv_mode[10][10], partition[16][4], switchable_interp[4][3];
    uint32_t intra_inter[4][2], comp_inter[5][2], single_ref[5][2][2], comp_ref[5][2];
    uint32_t tx32[2][4], tx16[2][3], tx8[2][2], skip[3][2];
    uint32_t mv_joints[4], mv_sign[2][2], mv_classes[2][11], mv_class0[2][2], mv_bits[2][10][2];
    uint32_t mv_class0_fp[2][2][4], mv_fp[2][4], mv_class0_hp[2][2], mv_hp[2][2];
    uint32_t coef[4][2][2][6][6][4];  // ZERO_TOKEN, ONE_TOKEN, TWO_TOKEN and up, then no more coefficients
    uint32_t eob_branch[4][2][2][6][6];  // times it asked whether more coefficients come
};

_Static_assert(sizeof(struct vp9_counts) == 13264, "nvdec_vp9EntropyCounts_t is 13264 bytes");

enum { VP9_ONLY_4X4, VP9_ALLOW_8X8, VP9_ALLOW_16X16, VP9_ALLOW_32X32, VP9_TX_MODE_SELECT };
enum { VP9_SINGLE_REFERENCE, VP9_COMPOUND_REFERENCE, VP9_REFERENCE_MODE_SELECT };

// what of a frame's uncompressed header the compressed header and the adaptation depend on
struct vp9_prob_frame {
    int intra;       // a key frame or an intra-only one
    int lossless;    // has no tx_mode
    int switchable;  // the interpolation filter is chosen by block
    int allow_hp;
    int compound;    // the references' sign biases allow compound prediction
    int after_key;   // the frame decoded before it was a key frame, after which coefficients adapt faster
    uint32_t tx_mode, reference_mode;  // what the compressed header says
};

void vp9_default_probs(struct vp9_probs *p);
// the compressed header of a frame, data, applied to p: -1 if it doesn't start like one
int vp9_compressed(struct vp9_prob_frame *f, const uint8_t *data, size_t len, struct vp9_probs *p);
// p, the probabilities a frame decoded with, adapted from pre, the context they were loaded from, by the counts of
// the frame (refresh_probs, 6.1.2); for a frame with neither error_resilient_mode nor frame_parallel_decoding_mode
void vp9_adapt(const struct vp9_prob_frame *f, const struct vp9_probs *pre, const struct vp9_counts *c,
               struct vp9_probs *p);
// NVDEC's probability table for a frame: p and the segmentation tree's probabilities of the uncompressed header
void vp9_pack(const struct vp9_probs *p, const uint8_t *seg_tree, const uint8_t *seg_pred, uint8_t *table);
