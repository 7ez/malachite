#pragma once
#include "plat.h"
#include "chan.h"
#include "nvdec.h"

struct gsp;
struct rm_client;
struct vmm;

// NVDEC channel through GSP-RM: client-allocated engine context promoted to GSP-RM, decoder class on subchannel 4
struct video {
    struct chan ch;
    struct vmm *vm;
    uint32_t inst;                        // which NVDEC
    uint64_t ctx_vram, ctx_va, ctx_size;  // the engine context
    struct dma *sem;                      // a word in the channel's vmm for fences
    uint64_t semva;
    uint32_t seq;  // the last fence payload, counting up
};

// channel on the first NVDEC in vm, admin or user; on failure nothing is left allocated in GSP-RM
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

// Per-stream buffers: setup page (nvdec.h), bitstream, state NVDEC carries between pictures, block-linear NV12/P010
// surfaces. AV1: a segmap per surface, probabilities in and out. VP9: two segmaps and MV buffers, probabilities,
// symbol counts. VP8: the probabilities NVDEC keeps.
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
// width_mbs x height_mbs macroblocks, bps bytes per sample, up to `slots` surfaces, `bits` bytes of bitstream
int video_dec_open(struct video *v, struct video_dec *h, enum nvdec_codec codec, uint32_t width_mbs,
                   uint32_t height_mbs, uint32_t bps, uint32_t slots, uint64_t bits);
void video_dec_close(struct video *v, struct video_dec *h);
// queue h->setup with slices in h->bits for decode into surface `slot`; *payload is for video_dec_wait. index:
// decoding order; error_idx: error concealment frame, index modulo the stream's reference frames
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

// Decode up to max pictures of an Annex-B stream (AV1: sized OBUs; VP8/VP9: IVF), passing shown ones to emit as
// linear NV12 (P010 at 2 bytes/sample) in decoding order; nonzero from emit stops. Returns pictures decoded, -1 if
// the stream can't be decoded or NVDEC didn't finish a picture.
int video_decode(struct video *v, enum nvdec_codec codec, const uint8_t *stream, size_t len, uint32_t max,
                 int (*emit)(void *arg, const struct video_frame *f), void *arg);
