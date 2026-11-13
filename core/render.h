#pragma once
#include "plat.h"
#include "drawbuild.h"

struct gsp;
struct gsp_zcull;
struct rm_client;
struct chan;

// Binds the 3D class (AMPERE_B, 0xc797) on subchannel 0 of compute_setup's graphics channel (compute is on 1). 0 on
// success.
int render_setup(struct gsp *s, const struct rm_client *c, struct chan *ch);

// shader local memory: the whole buffer and each warp's share (once the channel is idle)
void render_set_local_memory(struct chan *ch, uint64_t va, uint64_t size, uint32_t per_warp);
// texture header and sampler pools (32-byte descriptors), as compute_set_tex_pools; handle = header | sampler << 20
void render_set_tex_pools(struct chan *ch, uint64_t header_va, uint32_t headers, uint64_t sampler_va,
                          uint32_t samplers);
// a zero page that vertex fetches past a stream's end read instead; the channel's default of 0 faults
void render_set_vertex_substitute(struct chan *ch, uint64_t va);

// colour target formats, as SET_COLOR_TARGET_FORMAT takes them
#define RENDER_RGBA8 0xd5  // A8B8G8R8: bytes R, G, B, A
#define RENDER_BGRA8 0xcf  // A8R8G8B8: bytes B, G, R, A

// primitives, as BEGIN takes them
#define RENDER_POINTS 0
#define RENDER_LINES 1
#define RENDER_LINE_STRIP 3
#define RENDER_TRIANGLES 4
#define RENDER_TRIANGLE_STRIP 5
#define RENDER_PATCHES 0xe

// index sizes, as SET_INDEX_BUFFER_E takes them
#define RENDER_INDEX_U8 0
#define RENDER_INDEX_U16 1
#define RENDER_INDEX_U32 2

// SET_VERTEX_ATTRIBUTE_A fields: stream 4:0, byte offset 20:7, component widths 26:21, type 29:27, swap R/B 31
#define RENDER_ATTR(stream, offset, widths, type) ((stream) | (offset) << 7 | (widths) << 21 | (type) << 27)
#define RENDER_ATTR_R32G32B32A32 1
#define RENDER_ATTR_FLOAT 7

// Clear d's depth (mask bit 0) and stencil (bit 1) target, then wait as render_clear does. With z, the target becomes
// zcull region 0 first, which holds until render_zcull_off.
int render_clear_zs(struct chan *ch, const struct render_draw *d, const struct gsp_zcull *z, uint32_t mask, float depth,
                    uint32_t stencil, volatile uint32_t *sem, uint64_t sem_va, uint32_t payload);
void render_zcull_off(struct chan *ch);

// Bind d's colour target, then its depth/stencil target and a depth clear one method at a time, waiting for idle
// after each: the 1-based index of the first that never completes (0 the colour target), or -1 if all did.
int render_zt_probe(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                    uint32_t *payload);

// A draw's methods one group at a time, waiting for idle after each: the method of the first group that never
// completes, or -1.
int render_draw_probe(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                      uint32_t *payload);

// Draw, then wait for idle and release payload to sem_va. Returns 0 once the semaphore holds it; with sem NULL
// it returns at once, for render_wait later (so do render_clear and render_clear_zs).
int render_draw(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                uint32_t payload);

// render_draw with sem NULL, without telling the GPU: a chan_kick, the caller's or that of what's submitted next, does.
void render_draw_queued(struct chan *ch, const struct render_draw *d, uint64_t sem_va, uint32_t payload);

// Wait until sem holds payload or a later one (payloads counting up). 0 once it does, -1 after ms.
int render_wait(volatile uint32_t *sem, uint32_t payload, int ms);
// release `payload` once everything before it is done
void render_release(struct chan *ch, uint64_t sem_va, uint32_t payload);
// release `payload` once the writes before it have landed, without holding back what follows; kicks
void render_mark(struct chan *ch, uint64_t sem_va, uint32_t payload);
// queued, not kicked: the payload and then the time in the 16 bytes at va, once the draws before it are done
void render_stamp(struct chan *ch, uint64_t va, uint32_t payload);

// Clear t's colour target `target` (0, or 1..7 from more[]; addresses and pitches 128-byte aligned) to rgba, then
// wait for the engine to go idle and release payload to the semaphore at sem_va. Returns 0 once it has.
int render_clear(struct chan *ch, const struct render_draw *t, unsigned target, const float rgba[4],
                 volatile uint32_t *sem, uint64_t sem_va, uint32_t payload);
