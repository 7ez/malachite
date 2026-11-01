#pragma once
#include "plat.h"

struct gsp;
struct gsp_zcull;
struct rm_client;
struct chan;

// The 3D class (AMPERE_B, 0xc797) on a graphics channel that compute_setup already made: subchannel 0, next to
// compute on subchannel 1. Returns 0 on success.
int render_setup(struct gsp *s, const struct rm_client *c, struct chan *ch);

// the buffer the shaders' local memory comes out of: all of it, and as much a warp (once the channel is idle)
void render_set_local_memory(struct chan *ch, uint64_t va, uint64_t size, uint32_t per_warp);
// texture header and sampler pools (32-byte descriptors), as compute_set_tex_pools does for compute; a handle in a
// shader is header index | sampler index << 20
void render_set_tex_pools(struct chan *ch, uint64_t header_va, uint32_t headers, uint64_t sampler_va,
                          uint32_t samplers);

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

#define RENDER_ATTRIBUTES 32
#define RENDER_STREAMS 16

// SET_VERTEX_ATTRIBUTE_A fields: stream 4:0, byte offset 20:7, component widths 26:21, type 29:27, swap R/B 31
#define RENDER_ATTR(stream, offset, widths, type) ((stream) | (offset) << 7 | (widths) << 21 | (type) << 27)
#define RENDER_ATTR_R32G32B32A32 1
#define RENDER_ATTR_FLOAT 7

struct render_draw {
    uint64_t target;  // colour target, address and pitch 128-byte aligned
    uint32_t pitch, width, height, format;
    uint32_t target_blocklinear;  // block-linear in one-GOB blocks, as a target drawn with depth must be
    uint64_t vs, ps;  // shader headers, each followed by its code, 128-byte aligned
    uint32_t vs_regs, ps_regs;
    uint64_t vs_cbuf, ps_cbuf;  // each stage's const bank 0 (0x1000 bytes), or 0 for none
    // vertex attribute i as SET_VERTEX_ATTRIBUTE_A takes it (stream, offset, format), 0 for unused; each
    // stream is a vertex buffer, read stride bytes per vertex, 0 for none
    uint32_t attributes[RENDER_ATTRIBUTES];
    struct render_stream {
        uint64_t addr;
        uint32_t stride, size;
        uint32_t divisor;  // 0 to advance per vertex, n to advance every n instances
    } streams[RENDER_STREAMS];
    uint32_t primitive, first, count, instances;  // first and count are indices for an indexed draw
    int32_t base_vertex;  // added to each index
    uint32_t base_instance;  // the first instance's number
    uint64_t index;       // index buffer, 0 for a draw of consecutive vertices
    uint32_t index_size;  // bytes of index buffer from index
    uint32_t index_format;  // RENDER_INDEX_*
    float viewport[6];     // x, y, width, height, near, far
    uint32_t scissor[4];   // x, y, width, height
    uint64_t zt;           // block-linear depth/stencil target (blocks one GOB high), 0 for none
    uint32_t zt_format, zt_width, zt_height;  // SET_ZT_FORMAT, row stride in pixels, rows (samples, multisampled)
    uint32_t zt_compressed;  // its memory is a compressed kind
    uint32_t zt_zcull;  // zcull region 0 follows it
    // samples a pixel of the targets has (0 or 1 for one): they're laid out a grid of them per pixel (2x1, 2x2, 4x2),
    // so a block-linear target is that many times wider and higher in memory
    uint32_t samples;
    uint32_t depth_test, depth_write, depth_func;
    uint32_t stencil_test;
    uint32_t stencil[2][4];  // front, back: op fail, op zfail, op zpass, func
    uint32_t stencil_ref;  // the front's in bits 0-7, the back's in bits 8-15
    uint32_t stencil_read_mask[2], stencil_write_mask[2];
    uint32_t write_mask;  // SET_CT_WRITE: R bit 0, G bit 4, B bit 8, A bit 12
    // colour target 0's blend, as the SET_BLEND_PER_TARGET_* values: enable, then colour op, source and
    // destination factor, then the same for alpha
    uint32_t blend[7];
    // colour targets 1..7 beside target 0 (multiple render targets), the size of it; address 0 for none, as
    // target 0 can be too
    struct render_ct {
        uint64_t addr;
        uint32_t pitch, format, blocklinear, write_mask;
        uint32_t blend[7];
    } more[7];
    uint32_t cull_face;  // OGL_SET_CULL_FACE value, 0 to cull nothing
    uint32_t front_ccw;  // counter-clockwise triangles face the front, not Metal's default clockwise ones
    float point_size, line_width;  // 0 for 1
    uint32_t provoking_last;  // flat attributes from a primitive's last vertex
    uint32_t restart, restart_index;  // as nvgpu_draw's
    uint32_t window_coords;  // no viewport transform
    uint32_t depth_clamp, fill_lines;
    float depth_bias[3];  // constant, slope scale, clamp
    float blend_const[4];
    uint64_t visibility;  // where the passing samples' count goes, 0 for none
    uint32_t visibility_reset;  // the count starts from 0 rather than going on
    uint32_t wait_idle;  // earlier work finishes first: a new pass may read what the last one wrote
    uint32_t invalidate;  // the shader caches start over: code, constants or memory may have changed under them
    // the texture header and sampler pools, with this many entries, when the draw has its own; 0 keeps the bound ones
    uint64_t tex_headers, tex_samplers;
    uint32_t tex_count, sampler_count;
    // a tessellated draw's (RENDER_PATCHES, a vertex a patch) tessellation init and tessellation stages, when tess
    // (SET_TESSELLATION_PARAMETERS) is set; the tessellation stage reads vs_cbuf
    uint32_t tess;
    uint64_t ti, ts;
    uint32_t ti_regs, ts_regs;
    uint64_t ti_cbuf;
    uint32_t alpha_control;  // SET_ANTI_ALIAS_ALPHA_CONTROL
    uint32_t per_sample;     // the pixel stage runs once a sample
    uint32_t ps_depth;       // the pixel stage writes depth
    // layered targets as nvgpu_draw's: layers deep, colour target j's layer_pitch[j] bytes a layer, the vertex
    // stage picking the layer when shader_layer is set
    uint32_t layers, layer_pitch[8], zt_layer_pitch, shader_layer;
    // viewports and scissors 1..nviewports - 1 (x, y, width, height, near, far; x, y, width, height), beside 0
    uint32_t nviewports;
    const float (*viewports)[6];
    const uint32_t (*scissors)[4];
};

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

// Wait until sem holds payload or a later one (payloads counting up). 0 once it does, -1 after ms.
int render_wait(volatile uint32_t *sem, uint32_t payload, int ms);
// release `payload` once everything before it is done
void render_release(struct chan *ch, uint64_t sem_va, uint32_t payload);
// queued, not kicked: the payload and then the time in the 16 bytes at va, once the draws before it are done
void render_stamp(struct chan *ch, uint64_t va, uint32_t payload);

// Clear t's colour target `target` (0, or 1..7 from more[]; addresses and pitches 128-byte aligned) to rgba, then
// wait for the engine to go idle and release payload to the semaphore at sem_va. Returns 0 once it has.
int render_clear(struct chan *ch, const struct render_draw *t, unsigned target, const float rgba[4],
                 volatile uint32_t *sem, uint64_t sem_va, uint32_t payload);
