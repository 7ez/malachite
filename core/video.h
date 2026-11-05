#pragma once
#include "plat.h"
#include "chan.h"
#include "nvdec.h"

struct gsp;
struct rm_client;
struct vmm;

// An NVDEC channel through GSP-RM: a channel on the decoder's runlist, the engine context the client allocates
// promoted to GSP-RM, and the chip's video decoder class on subchannel 4
struct video {
    struct chan ch;
    struct vmm *vm;
    uint32_t inst;                        // which NVDEC
    uint64_t ctx_vram, ctx_va, ctx_size;  // the engine context
    struct dma *sem;                      // a word in the channel's vmm for fences
    uint64_t semva;
    uint32_t seq;  // the last fence payload, counting up
};

// The first NVDEC's channel in vm, an admin channel or a user one; on failure nothing of it is left with GSP-RM
int video_create(struct gsp *s, const struct rm_client *c, struct vmm *vm, uint32_t chid, int privileged,
                 struct video *v);
// payload written to the word at semva once the work queued before it is done
void video_release_queue(struct video *v, uint64_t semva, uint32_t payload);
void video_free(struct video *v);  // after GSP-RM freed the channel

// system memory in the channel's vmm, for the CPU and NVDEC both
struct vbuf {
    struct dma *mem;
    uint64_t va, size;
};

struct av1_tables;
struct vp9_tables;

// What a stream's pictures decode with: the setup page (nvdec.h), the bitstream, the buffers NVDEC carries
// from picture to picture, and the block-linear NV12 or P010 surfaces the pictures and their references are in. An
// AV1 stream has a segmentation map to each surface, and the probabilities in and out of its pictures; a VP9 stream
// two segmentation maps and motion vector buffers, its probabilities and the symbol counts NVDEC adapts them by; a
// VP8 stream the probabilities NVDEC keeps.
struct video_dec {
    enum nvdec_codec codec;
    struct vbuf setup, bits, coloc, mbhist, history, filter, intra_top, segmap, prob_in, prob_out, counts;
    struct vbuf surf[NVDEC_SLOTS];
    uint32_t width_mbs, height_mbs, slots;
    struct nvdec_surface layout;
    struct av1_tables *av1;
    struct vp9_tables *vp9;
};

// whether the channel's video decoder class has methods for codec
int video_supports(const struct video *v, enum nvdec_codec codec);
// buffers for pictures of width_mbs x height_mbs macroblocks of bps bytes a sample in up to `slots` surfaces, bits
// bytes of bitstream
int video_dec_open(struct video *v, struct video_dec *h, enum nvdec_codec codec, uint32_t width_mbs,
                   uint32_t height_mbs, uint32_t bps, uint32_t slots, uint64_t bits);
void video_dec_close(struct video *v, struct video_dec *h);
// queue the picture set up in h->setup, its slices in h->bits, to be decoded into surface slot: the payload
// video_dec_wait waits for. index counts pictures in decoding order; error_idx goes to error concealment, nvk's
// index modulo the stream's reference frames.
int video_dec_queue(struct video *v, struct video_dec *h, uint32_t slot, uint32_t index, uint32_t error_idx,
                    uint32_t *payload);
int video_dec_wait(struct video *v, struct video_dec *h, uint32_t payload, uint32_t index);

struct video_frame {
    uint32_t index;      // in decoding order
    uint32_t idr_count;  // display order: by idr_count, then poc
    int32_t poc;
    uint32_t width, height, bps;
    const uint8_t *nv12;  // the luma rows, then the interleaved chroma rows, width * bps bytes each
};

// Up to max pictures of an Annex-B stream (AV1: of OBUs with their sizes, VP9 and VP8: an IVF file), decoded one at
// a time into block-linear surfaces in system memory and handed to emit as linear NV12 (P010 with two bytes a
// sample) in decoding order, the ones that are to be shown; emit returns nonzero to stop. How many it decoded, -1 for
// a stream it can't decode or a picture NVDEC didn't finish.
int video_decode(struct video *v, enum nvdec_codec codec, const uint8_t *stream, size_t len, uint32_t max,
                 int (*emit)(void *arg, const struct video_frame *f), void *arg);
