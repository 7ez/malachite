// The methods of a 3D draw, built from a render_draw: shared by the kernel (core/render.c) and the Metal plugin, so
// both write the same words. Plain C, no kernel or GSP headers.
#pragma once
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MTHD
#define MTHD(subch, mthd, n) (0x20000000u | ((uint32_t)(n) << 16) | ((subch) << 13) | ((mthd) >> 2))
#endif

#define R_SET_OBJECT 0x0000
#define R_MACRO_UPLOAD_POS 0x0114
#define R_MACRO_ID 0x011c
#define R_CALL_MME_MACRO(i) (0x3800 + (i) * 8)
#define R_WAIT_FOR_IDLE 0x0110
#define R_SET_SHADER_LOCAL_MEMORY_WINDOW 0x077c
#define R_SET_SHADER_LOCAL_MEMORY_A 0x0790
#define R_SET_COLOR_TARGET_A(j) (0x0800 + (j) * 64)  // A..LAYER: addr hi, addr lo, width, height, format, memory,
                                                    // third dimension, array pitch, layer
#define R_SET_CLEAR_RECT_HORIZONTAL 0x0d6c
#define R_SET_COLOR_CLEAR_VALUE(i) (0x0d80 + (i) * 4)
#define R_SET_SURFACE_CLIP_HORIZONTAL 0x0ff4
#define R_SET_CLEAR_SURFACE_CONTROL 0x10f8
#define R_SET_CT_SELECT 0x121c
#define R_CLEAR_SURFACE 0x19d0
#define R_SET_REPORT_SEMAPHORE_A 0x1b00

#define R_INVALIDATE_SHADER_CACHES_NO_WFI 0x0da4
#define R_SET_TESSELLATION_PARAMETERS 0x0320
#define R_SET_PATCH 0x0dcc
#define R_SET_VIEWPORT_SCALE_X(j) (0x0a00 + (j) * 32)  // scale x/y/z, then offset x/y/z
#define R_SET_VIEWPORT_CLIP_HORIZONTAL(j) (0x0c00 + (j) * 16)  // horizontal, vertical, min z, max z
#define R_SET_SCISSOR_ENABLE(j) (0x0e00 + (j) * 16)  // enable, horizontal, vertical
#define R_SET_INDEX_BUFFER_SIZE_A 0x0238  // Turing+: size hi, lo
#define R_SET_POINT_SIZE 0x1518
#define R_SET_POINT_SPRITE 0x1520
#define R_SET_POINT_SPRITE_SELECT 0x1604
#define R_SET_ATTRIBUTE_POINT_SIZE 0x1910
#define R_SET_DRAW_CONTROL_A 0x0260       // Turing+: topology etc., then B instance count
#define R_DRAW_INDEX_BUFFER_BEGIN_END_A 0x0268  // first index, then B index count
#define R_DRAW_VERTEX_ARRAY_BEGIN_END_A 0x0270  // first vertex, then B vertex count
#define R_SET_VERTEX_ID_BASE 0x1118
#define R_SET_GLOBAL_BASE_VERTEX_INDEX 0x1434
#define R_SET_INDEX_BUFFER_A 0x17c8            // address hi, lo
#define R_SET_INDEX_BUFFER_E 0x17d8            // index size
#define R_SET_VERTEX_STREAM_INSTANCE_A(i) (0x1880 + (i) * 4)
#define R_SET_Z_CLEAR_VALUE 0x0d90
#define R_SET_STENCIL_CLEAR_VALUE 0x0da0
#define R_SET_BACK_STENCIL_FUNC_REF 0x0f54  // then back mask, back func mask
#define R_SET_ZT_A 0x0fe0                   // address hi, lo, format, block size, array pitch
#define R_SET_ZT_SIZE_A 0x1228              // width, height, third dimension
#define R_SET_ANTI_ALIAS 0x15d0             // SAMPLES_MODE
#define R_SET_ANTI_ALIAS_ENABLE 0x1534
#define R_SET_HYBRID_ANTI_ALIAS_CONTROL 0x0754  // passes [3:0], centroid [4]
#define R_SET_ANTI_ALIAS_ALPHA_CONTROL 0x153c    // alpha to coverage [0], alpha to one [4]
#define R_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL 0x12e0
#define R_SET_POST_PS_INITIAL_COVERAGE 0x1138
#define R_SET_RT_LAYER 0x15cc  // layer [15:0], the vertex stage picks it [16]
#define R_SET_ANTI_ALIAS_SAMPLE_POSITIONS 0x11e0  // 4 words: 16 sample positions (x, y in 16ths), 4 bits each
#define R_SET_DEPTH_WRITE 0x12e8
#define R_SET_DEPTH_FUNC 0x130c
#define R_SET_STENCIL_TEST 0x1380  // then op fail, op zfail, op zpass, func, func ref, func mask, mask
#define R_SET_TWO_SIDED_STENCIL_TEST 0x1594  // then back op fail, op zfail, op zpass, func
#define R_SET_ZT_LAYER 0x179c
#define R_SET_ZCULL 0x1968
#define R_SET_ZCULL_BOUNDS 0x196c
#define R_SET_ZCULL_REGION_SIZE_A 0x07c0
#define R_SET_ZCULL_REGION_LOCATION 0x07e0  // then aliquots per layer
#define R_SET_ZCULL_STORAGE_A 0x07e8
#define R_SET_ZCULL_DIR_FORMAT 0x0dbc
#define R_SET_ZCULL_CRITERION 0x0dd8
#define R_CLEAR_ZCULL_REGION 0x12c8
#define R_SET_ACTIVE_ZCULL_REGION 0x1590
#define R_SET_ZCULL_REGION_FORMAT 0x15c8
#define R_SET_ZCULL_REGION_PIXEL_OFFSET_A 0x15fc
#define R_SET_ZCULL_SUBREGION 0x02e8
#define R_SET_ZCULL_SUBREGION_ALLOCATION 0x02f8
#define R_ASSIGN_ZCULL_SUBREGIONS 0x02fc
#define R_SET_ZCULL_SUBREGION_REPORT_TYPE 0x0370
#define R_ZT_FORMAT_Z16 0x13
#define R_SET_Z_COMPRESSION 0x19cc
#define R_SET_COLOR_COMPRESSION(i) (0x19e0 + (i) * 4)
#define R_SET_STENCIL_COMPRESSION 0x111c
#define R_SET_ZT_SPARSE 0x1208
#define R_SET_MESH_CONTROL 0x114c
#define R_SET_VERTEX_ATTRIBUTE_A(i) (0x1160 + (i) * 4)
#define R_SET_DA_OUTPUT 0x164c
#define R_SET_DA_PRIMITIVE_RESTART 0x1644  // then the restart index
#define R_SET_PROVOKING_VERTEX 0x1684
#define R_SET_FRONT_POLYGON_MODE 0x0dac  // then the back's
#define R_SET_ALIASED_LINE_WIDTH_ENABLE 0x020c
#define R_SET_FILL_VIA_TRIANGLE 0x113c
#define R_SET_LINE_MULTISAMPLE_OVERRIDE 0x1418
#define R_SET_ANTI_ALIASED_LINE 0x1570
#define R_SET_LINE_STIPPLE 0x166c
#define R_SET_EDGE_FLAG 0x15e4
#define R_SET_ZPASS_PIXEL_COUNT 0x1514
#define R_CLEAR_REPORT_VALUE 0x1530
#define R_SET_POLY_OFFSET_POINT 0x0dc0  // then line, fill
#define R_SET_DEPTH_BIAS 0x15bc
#define R_SET_SLOPE_SCALE_DEPTH_BIAS 0x156c
#define R_SET_DEPTH_BIAS_CLAMP 0x187c
#define R_SET_BLEND_CONST_RED 0x131c  // then green, blue, alpha
#define R_SET_LINE_WIDTH_FLOAT 0x13b0  // then the aliased width
#define R_SET_DEPTH_TEST 0x12cc
#define R_SET_WINDOW_ORIGIN 0x13ac
#define R_SET_ZT_SELECT 0x1538
#define R_OGL_SET_CULL 0x1918  // then front face, cull face
#define R_SET_VIEWPORT_SCALE_OFFSET 0x192c
#define R_SET_VIEWPORT_CLIP_CONTROL 0x193c
#define R_SET_VIEWPORT_Z_CLIP 0x0d7c
#define R_SET_CT_WRITE(i) (0x1a00 + (i) * 4)
#define R_SET_VERTEX_STREAM_A_FORMAT(j) (0x1c00 + (j) * 16)  // format, location hi, location lo, frequency
#define R_SET_VERTEX_STREAM_SIZE_A(j) (0x0600 + (j) * 8)  // Turing+: replaces the 0x1f00 end address
#define R_SET_PIPELINE_SHADER(j) (0x2000 + (j) * 64)
#define R_SET_PIPELINE_REGISTER_COUNT(j) (0x200c + (j) * 64)  // register count, binding, address hi, lo
#define R_SET_SAMPLER_BINDING 0x1234
#define R_SET_BLEND_STATE_PER_TARGET 0x12e4
#define R_SET_BLEND_PER_FORMAT_ENABLE 0x1140
#define R_SET_BLEND_FLOAT_OPTION 0x19c0
#define R_SET_BLEND(i) (0x1360 + (i) * 4)
#define R_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(j) (0x1e00 + (j) * 32)  // then colour op, src, dst, alpha ..
#define R_INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI 0x1288
#define R_INVALIDATE_SAMPLER_CACHE_NO_WFI 0x1424
#define R_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI 0x1428
#define R_SET_TEX_SAMPLER_POOL_A 0x155c  // address hi, address lo, maximum index
#define R_SET_TEX_HEADER_POOL_A 0x1574
#define R_SET_CONSTANT_BUFFER_SELECTOR_A 0x2380  // size, address hi, address lo
#define R_BIND_GROUP_CONSTANT_BUFFER(j) (0x2410 + (j) * 32)

#define CT_MEMORY_LAYOUT_PITCH (1u << 12)
#define PIPE_VERTEX 1
#define PIPE_TESS_INIT 2
#define PIPE_TESS 3
#define PIPE_PIXEL 5
#define GROUP_VERTEX 0  // the binding groups the stages read const banks from
#define GROUP_TESS_INIT 1
#define GROUP_TESS 2
#define GROUP_PIXEL 4
#define CBUF_BYTES 0x1000
#define ATTR_INACTIVE (1u << 6)
#define ATTR_R32G32B32A32_FLOAT ((1u << 21) | (7u << 27))

#define RENDER_ATTRIBUTES 32
#define RENDER_STREAMS 16

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
    // stage picking the layer when shader_layer is set; each target has target_layers[j] (zt_layers) of them, 0 for
    // all
    uint32_t layers, layer_pitch[8], zt_layer_pitch, shader_layer;
    uint32_t target_layers[8], zt_layers;
    // viewports and scissors 1..nviewports - 1 (x, y, width, height, near, far; x, y, width, height), beside 0
    uint32_t nviewports;
    const float (*viewports)[6];
    const uint32_t (*scissors)[4];
};

// stores, not memcpy: the kernel's is a real call here (-fno-builtin), scores of them a draw
static inline void put(uint32_t *to, const uint32_t *from, unsigned n)
{
    for (unsigned i = 0; i < n; i++)
        to[i] = from[i];
}

static inline uint32_t fbits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// SET_DA_PRIMITIVE_RESTART as nvgpu_draw's restart asks: Metal restarts indexed strips only (line strip 3, triangle
// strip 5), at the largest index the type holds; 2 restarts any indexed draw at restart_index, 1 none
static inline uint32_t draw_restarts(uint32_t primitive, uint32_t restart, int indexed)
{
    return indexed && (restart == 2 || (!restart && (primitive == 3 || primitive == 5)));
}

// the layers colour target j (8: the depth/stencil target) is drawn and cleared in: 1 unless the pass is layered
static inline uint32_t draw_target_layers(const struct render_draw *d, unsigned j)
{
    uint32_t has = j < 8 ? d->target_layers[j] : d->zt_layers;
    if (d->layers <= 1)
        return 1;
    return has && has < d->layers ? has : d->layers;
}

static inline uint32_t draw_restart_index(uint32_t restart, uint32_t restart_index, uint32_t index_format)
{
    return restart == 2 ? restart_index : index_format == 0 ? 0xff : index_format == 1 ? 0xffff : 0xffffffff;
}

// Each writes one group of d's methods into push and returns the words written.
unsigned draw_shader_words(uint32_t *push, const struct render_draw *d);
unsigned draw_depth_stencil_words(uint32_t *push, const struct render_draw *d);
unsigned draw_cull_words(uint32_t *push, const struct render_draw *d);
unsigned draw_viewport_words(uint32_t *push, const struct render_draw *d);
unsigned draw_bias_words(uint32_t *push, const struct render_draw *d);
unsigned draw_blend_const_words(uint32_t *push, const struct render_draw *d);
unsigned draw_attribute_words(uint32_t *push, const struct render_draw *d);
// the anti-aliasing state, then each colour target with its write mask and blend (interleaved, so one group)
unsigned draw_aa_words(uint32_t *push, const struct render_draw *d);
unsigned draw_target_words(uint32_t *push, const struct render_draw *d);
// viewport j (x, y, width, height, near, far) and scissor j (x, y, width, height)
unsigned viewport_bind(uint32_t *push, unsigned j, const float *vp, const uint32_t *sc);

#ifdef __cplusplus
}
#endif
