#pragma once
#include "plat.h"
#include "drawbuild.h"

struct gsp;
struct gsp_zcull;
struct rm_client;
struct chan;

// subchannel 0 of compute_setup's graphics channel, compute being on 1
int render_setup(struct gsp *s, const struct rm_client *c, struct chan *ch);

// waits for idle first
void render_set_local_memory(struct chan *ch, uint64_t va, uint64_t size, uint32_t per_warp);
// as compute_set_tex_pools
void render_set_tex_pools(struct chan *ch, uint64_t header_va, uint32_t headers, uint64_t sampler_va,
                          uint32_t samplers);
// a zero page for fetches past a stream's end, which fault by default
void render_set_vertex_substitute(struct chan *ch, uint64_t va);

// SET_COLOR_TARGET_FORMAT values
#define RENDER_RGBA8 0xd5  // A8B8G8R8: bytes R, G, B, A
#define RENDER_BGRA8 0xcf  // A8R8G8B8: bytes B, G, R, A

// BEGIN values
#define RENDER_POINTS 0
#define RENDER_LINES 1
#define RENDER_LINE_STRIP 3
#define RENDER_TRIANGLES 4
#define RENDER_TRIANGLE_STRIP 5
#define RENDER_PATCHES 0xe

// SET_INDEX_BUFFER_E values
#define RENDER_INDEX_U8 0
#define RENDER_INDEX_U16 1
#define RENDER_INDEX_U32 2

// SET_VERTEX_ATTRIBUTE_A bits: stream 4:0, byte offset 20:7, component widths 26:21, type 29:27, swap R/B 31
#define RENDER_ATTR(stream, offset, widths, type) ((stream) | (offset) << 7 | (widths) << 21 | (type) << 27)
#define RENDER_ATTR_R32G32B32A32 1
#define RENDER_ATTR_FLOAT 7

// mask bit 0 depth, bit 1 stencil; with z the target is zcull region 0 until render_zcull_off
int render_clear_zs(struct chan *ch, const struct render_draw *d, const struct gsp_zcull *z, uint32_t mask, float depth,
                    uint32_t stencil, volatile uint32_t *sem, uint64_t sem_va, uint32_t payload);
void render_zcull_off(struct chan *ch);

// One method at a time with a wait after each, to find the one that hangs: its 1-based index (0 the colour target),
// or -1.
int render_zt_probe(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                    uint32_t *payload);

// as render_zt_probe, returning the method of the first group that hangs
int render_draw_probe(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                      uint32_t *payload);

// With sem NULL it returns at once, for render_wait later; render_clear and render_clear_zs do the same.
int render_draw(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                uint32_t payload);

// not kicked; the next chan_kick on the channel starts it
void render_draw_queued(struct chan *ch, const struct render_draw *d, uint64_t sem_va, uint32_t payload);

// accepts a later payload too; -1 after ms
int render_wait(volatile uint32_t *sem, uint32_t payload, int ms);
// once everything before it is done
void render_release(struct chan *ch, uint64_t sem_va, uint32_t payload);
// once earlier writes land, without holding back what follows; kicks
void render_mark(struct chan *ch, uint64_t sem_va, uint32_t payload);
// not kicked; payload then GPU time in the 16 bytes at va
void render_stamp(struct chan *ch, uint64_t va, uint32_t payload);

// target 0, or 1..7 from more[], with addresses and pitches 128-byte aligned
int render_clear(struct chan *ch, const struct render_draw *t, unsigned target, const float rgba[4],
                 volatile uint32_t *sem, uint64_t sem_va, uint32_t payload);
