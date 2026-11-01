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

// What a stream's pictures decode with: the setup page (nvdec.h), the Annex-B bitstream, the buffers NVDEC carries
// from picture to picture, and the block-linear NV12 surfaces the pictures and their references are in
struct video_h264 {
    struct vbuf setup, bits, coloc, mbhist, history, surf[H264_SLOTS];
    uint32_t width_mbs, height_mbs, slots;
    struct nvdec_surface layout;
};

// buffers for pictures of width_mbs x height_mbs macroblocks in up to `slots` surfaces, bits bytes of bitstream
int video_h264_open(struct video *v, struct video_h264 *h, uint32_t width_mbs, uint32_t height_mbs, uint32_t slots,
                    uint64_t bits);
void video_h264_close(struct video *v, struct video_h264 *h);
// queue the picture set up in h->setup, its slices in h->bits, to be decoded into surface slot: the payload
// video_h264_wait waits for. index counts pictures in decoding order; error_idx goes to error concealment, nvk's
// index modulo the stream's reference frames.
int video_h264_queue(struct video *v, struct video_h264 *h, uint32_t slot, uint32_t index, uint32_t error_idx,
                     uint32_t *payload);
int video_h264_wait(struct video *v, struct video_h264 *h, uint32_t payload, uint32_t index);

struct video_frame {
    uint32_t index;      // in decoding order
    uint32_t idr_count;  // display order: by idr_count, then poc
    int32_t poc;
    uint32_t width, height;
    const uint8_t *nv12;  // the luma rows, then the interleaved chroma rows, width bytes each
};

// Up to max pictures of an Annex-B H.264 stream, decoded one at a time into block-linear NV12 surfaces in system
// memory and handed to emit as linear NV12 in decoding order; emit returns nonzero to stop. How many it decoded,
// -1 for a stream it can't decode or a picture NVDEC didn't finish.
int video_decode_h264(struct video *v, const uint8_t *stream, size_t len, uint32_t max,
                      int (*emit)(void *arg, const struct video_frame *f), void *arg);
