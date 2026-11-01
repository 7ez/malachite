// AMPERE_B (0xc797) on the compute bridge's graphics channel: render targets and surface clears, the start of
// the render pipeline behind MTLRenderCommandEncoder. Method layout per the sdk's cl9097.h.
#include "chip.h"
#include "chan.h"
#include "gsp.h"
#include "render.h"


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

static uint32_t fbits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// the release alone already waits for every earlier write: idle also holds back whatever comes next until then
static int release(struct chan *ch, uint32_t *push, unsigned n, volatile uint32_t *sem, uint64_t sem_va,
                   uint32_t payload, int idle)
{
    uint32_t tail[] = {
        MTHD(0, R_WAIT_FOR_IDLE, 1), 0,
        // release once every earlier write has landed, one word, from the end of the pipeline
        MTHD(0, R_SET_REPORT_SEMAPHORE_A, 4), (uint32_t)(sem_va >> 32), (uint32_t)sem_va, payload,
        (1u << 4) | (0xfu << 12) | (1u << 28),
    };
    if (n)
        chan_push(ch, push, n);
    chan_push(ch, tail + (idle ? 0 : 2), sizeof(tail) / 4 - (idle ? 0 : 2));
    chan_kick(ch);
    return sem ? wait_word(sem, payload, 2000) : 0;
}

void render_release(struct chan *ch, uint64_t sem_va, uint32_t payload)
{
    release(ch, NULL, 0, NULL, sem_va, payload, 1);
}

void render_stamp(struct chan *ch, uint64_t va, uint32_t payload)
{
    // four words: the payload, then the GPU's timestamp at +8, once every earlier write has landed
    uint32_t w[] = {
        MTHD(0, R_SET_REPORT_SEMAPHORE_A, 4), (uint32_t)(va >> 32), (uint32_t)va, payload, (1u << 4) | (0xfu << 12),
    };
    chan_push(ch, w, sizeof(w) / 4);
}

int render_wait(volatile uint32_t *sem, uint32_t payload, int ms)
{
    uint64_t end = time_us() + (uint64_t)ms * 1000;
    while ((int32_t)(*sem - payload) < 0) {  // the payloads count up, and a later release overwrites an earlier
        if (time_us() > end)
            return -1;
        delay_us(10);
    }
    return 0;
}

// The MME macro that has the GR firmware set a priv register (arg: the register, then the value and the mask), as
// Mesa's Turing+ nvc0 driver has it: it waits for idle, hands the request over in the firmware methods and waits for
// the reply.
static const uint32_t set_priv_reg[] = {
    0x8d147403, 0x18c00011, 0x31c00300, 0x00600003, 0x18c00000, 0x318c0300, 0x1a60346b, 0x19058231,
    0x300c0300, 0x00000003, 0x18c00040, 0x01804300, 0x0000000b, 0x18c02001, 0x340c7700, 0x01380103,
    0x18c00340, 0x31905b00, 0x00000007, 0x18c017ff, 0xf40c7701, 0x00000003, 0x18c02000, 0xf18c6300,
    0x0000002b, 0x18c00000, 0xb1ac5f00, 0x00d40003, 0x18c00010, 0x318c0300, 0x00000003, 0x18c00000,
    0x318c0301, 0x00000003, 0x18c00000, 0x318c0300,
};

// Without bit 3 of the SM's display control register cleared, a pixel shader's helper threads (the ones of a quad
// the primitive doesn't cover, which only feed derivatives) skip their memory loads and read zeros, so a value
// loaded from a buffer differs across the quad and its derivatives blow up along primitive edges. Volta to Ada.
#define SM_DISP_CTRL 0x419ba4
#define SM_DISP_CTRL_HELPER_LOADS_OFF 8

int render_setup(struct gsp *s, const struct rm_client *c, struct chan *ch)
{
    if (rm_alloc(s, c->client, ch->handle, 0xc7970000, gsp_gpu(s)->chip->eng3d, NULL, 0))
        return -1;
    uint32_t push[] = {
        MTHD(0, R_SET_OBJECT, 1), gsp_gpu(s)->chip->eng3d,
        // snorm8, unorm16 and snorm16 targets blend only when asked to; 0 * anything is 0 (inf and NaN too), as
        // NVK has it
        MTHD(0, R_SET_BLEND_PER_FORMAT_ENABLE, 1), 1 << 4,
        MTHD(0, R_SET_BLEND_FLOAT_OPTION, 1), 1,
        MTHD(0, R_SET_COLOR_COMPRESSION(0), 8), 1, 1, 1, 1, 1, 1, 1, 1,  // what a target's PTE kind allows
        MTHD(0, R_MACRO_ID, 2), 0, 0,
        0xa0000000u | (uint32_t)(1 + sizeof(set_priv_reg) / 4) << 16 | R_MACRO_UPLOAD_POS >> 2, 0,
    };
    uint32_t call[] = {
        0xa0000000u | 3u << 16 | R_CALL_MME_MACRO(0) >> 2, SM_DISP_CTRL, 0, SM_DISP_CTRL_HELPER_LOADS_OFF,
    };
    uint32_t all[sizeof(push) / 4 + sizeof(set_priv_reg) / 4 + sizeof(call) / 4];
    memcpy(all, push, sizeof(push));
    memcpy(all + sizeof(push) / 4, set_priv_reg, sizeof(set_priv_reg));
    memcpy(all + sizeof(push) / 4 + sizeof(set_priv_reg) / 4, call, sizeof(call));
    chan_submit(ch, all, sizeof(all) / 4);
    return 0;
}

void render_set_tex_pools(struct chan *ch, uint64_t header_va, uint32_t headers, uint64_t sampler_va,
                          uint32_t samplers)
{
    uint32_t push[] = {
        MTHD(0, R_SET_TEX_HEADER_POOL_A, 3), (uint32_t)(header_va >> 32), (uint32_t)header_va, headers - 1,
        MTHD(0, R_SET_TEX_SAMPLER_POOL_A, 3), (uint32_t)(sampler_va >> 32), (uint32_t)sampler_va, samplers - 1,
        MTHD(0, R_SET_SAMPLER_BINDING, 1), 0,  // independently: a handle names its header and sampler apart
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

void render_set_local_memory(struct chan *ch, uint64_t va, uint64_t size, uint32_t per_warp)
{
    uint32_t push[] = {
        MTHD(0, R_WAIT_FOR_IDLE, 1), 0,
        MTHD(0, R_SET_SHADER_LOCAL_MEMORY_WINDOW, 1), 0xff000000,  // where compute keeps it
        MTHD(0, R_SET_SHADER_LOCAL_MEMORY_A, 5), (uint32_t)(va >> 32), (uint32_t)va, (uint32_t)(size >> 32),
        (uint32_t)size, per_warp,
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

// the sample grid of a pixel, and SET_ANTI_ALIAS's mode for it: 2X1, 2X2, 4X2_D3D (Metal's positions are D3D's)
static uint32_t aa_mode(uint32_t samples, uint32_t *sx, uint32_t *sy)
{
    *sx = samples >= 8 ? 4 : samples >= 2 ? 2 : 1;
    *sy = samples >= 4 ? 2 : 1;
    return samples >= 8 ? 4 : samples >= 4 ? 2 : samples >= 2 ? 1 : 0;
}

// d's samples per pixel: the fragment shader runs once a pixel, its colour going to the samples it covers, or with
// per_sample once a sample (a pass each, the centroid the pass's sample). The samples sit where D3D, Vulkan and
// Metal put them (in 16ths of a pixel), the same for each pixel of a 2x2 quad; left at the hardware's default
// they'd all be at the centre and cover the same. Alpha to coverage dithers over 2x2 pixels with samples to spread,
// as an Apple GPU's does, and is a plain alpha >= 0.5 test with one.
static unsigned aa_bind(uint32_t *push, const struct render_draw *d)
{
    static const uint8_t standard[4][8][2] = {
        { { 8, 8 } },
        { { 12, 12 }, { 4, 4 } },
        { { 6, 2 }, { 14, 6 }, { 2, 10 }, { 10, 14 } },
        { { 9, 5 }, { 7, 11 }, { 13, 9 }, { 5, 3 }, { 3, 13 }, { 1, 7 }, { 11, 15 }, { 15, 1 } },
    };
    uint32_t sx, sy, mode = aa_mode(d->samples, &sx, &sy);
    unsigned set = d->samples >= 8 ? 3 : d->samples >= 4 ? 2 : d->samples >= 2 ? 1 : 0, per = 1u << set;
    uint32_t passes = d->per_sample && d->samples > 1 ? d->samples : 1;
    uint32_t w[] = {
        MTHD(0, R_SET_ANTI_ALIAS, 1), mode,
        MTHD(0, R_SET_ANTI_ALIAS_ENABLE, 1), mode != 0 || d->alpha_control,
        MTHD(0, R_SET_HYBRID_ANTI_ALIAS_CONTROL, 1), passes | (passes > 1) << 4,  // centroid per pass or fragment
        MTHD(0, R_SET_ANTI_ALIAS_ALPHA_CONTROL, 1), d->alpha_control,
        MTHD(0, R_SET_ALPHA_TO_COVERAGE_DITHER_CONTROL, 1), mode != 0,  // footprint 2x2 pixels, or 1x1
        // the coverage after the shader starts from the rasterizer's, which a [[sample_mask]] output then narrows
        MTHD(0, R_SET_POST_PS_INITIAL_COVERAGE, 1), 1,
        MTHD(0, R_SET_ANTI_ALIAS_SAMPLE_POSITIONS, 4), 0, 0, 0, 0,
    };
    for (unsigned n = 0; n < 16; n++) {  // sample n % per of the quad's pixel n / per
        const uint8_t *at = standard[set][n % per];
        w[sizeof(w) / 4 - 4 + n / 4] |= (uint32_t)(at[0] | at[1] << 4) << (n % 4 * 8);
    }
    memcpy(push, w, sizeof(w));
    return sizeof(w) / 4;
}

// d's colour targets: each bound (one absent disabled, format 0), then CT_SELECT over all of them, their write masks
// and blends; returns the words written
static unsigned ct_bind(uint32_t *push, const struct render_draw *d)
{
    uint32_t sx, sy;
    aa_mode(d->samples, &sx, &sy);
    unsigned n = aa_bind(push, d), count = 1;
    for (unsigned j = 1; j < 8; j++)
        if (d->more[j - 1].addr)
            count = j + 1;
    uint32_t select = count;
    for (unsigned j = 0; j < count; j++) {
        struct render_ct ct = { d->target, d->pitch, d->format, d->target_blocklinear, d->write_mask };
        memcpy(ct.blend, d->blend, sizeof(ct.blend));
        if (j)
            ct = d->more[j - 1];
        // block-linear (one-GOB blocks) takes the width in samples, pitch-linear the row pitch in bytes
        uint32_t w[] = {
            MTHD(0, R_SET_COLOR_TARGET_A(j), 9), (uint32_t)(ct.addr >> 32), (uint32_t)ct.addr,
            ct.blocklinear ? d->width * sx : ct.pitch, d->height * sy, ct.addr ? ct.format : 0,
            ct.blocklinear ? 0 : CT_MEMORY_LAYOUT_PITCH, d->layers > 1 ? d->layers : 1, d->layer_pitch[j] >> 2, 0,
            MTHD(0, R_SET_CT_WRITE(j), 1), ct.write_mask,
            MTHD(0, R_SET_BLEND(j), 1), ct.blend[0],
        };
        memcpy(push + n, w, sizeof(w));
        n += sizeof(w) / 4;
        if (ct.blend[0]) {  // the ops and factors only mean something (and are only valid) with blending on
            uint32_t b[] = { MTHD(0, R_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(j), 7), 1, ct.blend[1], ct.blend[2],
                             ct.blend[3], ct.blend[4], ct.blend[5], ct.blend[6] };
            memcpy(push + n, b, sizeof(b));
            n += sizeof(b) / 4;
        }
        select |= j << (4 + 3 * j);  // TARGETj: render target j writes colour target j
    }
    push[n++] = MTHD(0, R_SET_CT_SELECT, 1);
    push[n++] = select;
    push[n++] = MTHD(0, R_SET_BLEND_STATE_PER_TARGET, 1);
    push[n++] = 1;
    return n;
}

int render_clear(struct chan *ch, const struct render_draw *t, unsigned target, const float rgba[4],
                 volatile uint32_t *sem, uint64_t sem_va, uint32_t payload)
{
    uint32_t c[4], w = t->width, h = t->height, push[256];
    memcpy(c, rgba, sizeof(c));
    unsigned n = ct_bind(push, t);
    uint32_t clear[] = {
        MTHD(0, R_SET_SURFACE_CLIP_HORIZONTAL, 2), w << 16, h << 16,
        MTHD(0, R_SET_CLEAR_SURFACE_CONTROL, 1), 1u << 4,  // USE_CLEAR_RECT
        MTHD(0, R_SET_CLEAR_RECT_HORIZONTAL, 2), w << 16, h << 16,
        MTHD(0, R_SET_COLOR_CLEAR_VALUE(0), 4), c[0], c[1], c[2], c[3],
        MTHD(0, R_SET_CT_WRITE(target), 1), 0x1111,  // a clear writes every channel, whatever a draw masks
        MTHD(0, R_CLEAR_SURFACE, 1), 0xf << 2 | target << 6,  // R, G, B, A of MRT `target`
    };
    memcpy(push + n, clear, sizeof(clear));
    n += sizeof(clear) / 4;
    for (uint32_t l = 1; l < t->layers; l++) {  // a layered target's other layers
        if (n + 2 > sizeof(push) / 4) {
            chan_push(ch, push, n);
            n = 0;
        }
        push[n++] = MTHD(0, R_CLEAR_SURFACE, 1);
        push[n++] = 0xf << 2 | target << 6 | l << 10;
    }
    return release(ch, push, n, sem, sem_va, payload, 1);
}

// bind d's depth/stencil target, or none; returns the words written
static unsigned zt_bind(uint32_t *push, const struct render_draw *d)
{
    if (!d->zt) {
        push[0] = MTHD(0, R_SET_ZT_SELECT, 1);
        push[1] = 0;
        return 2;
    }
    uint32_t w[] = {
        MTHD(0, R_SET_ZT_A, 5), (uint32_t)(d->zt >> 32), (uint32_t)d->zt, d->zt_format, 0,  // blocks: one GOB
        d->zt_layer_pitch >> 2,
        MTHD(0, R_SET_ZT_SELECT, 1), 1,
        // the Z/S hardware has no tile width: the width is the row stride in pixels; one layer, or an array
        MTHD(0, R_SET_ZT_SIZE_A, 3), d->zt_width, d->zt_height, d->layers > 1 ? d->layers : (1 | 1 << 16),
        MTHD(0, R_SET_ZT_LAYER, 1), 0,
        // not sparse
        MTHD(0, R_SET_ZCULL, 1), d->zt_zcull,
        // a shader's depth isn't the plane zcull knows: no bound on it either way
        MTHD(0, R_SET_ZCULL_BOUNDS, 1), d->ps_depth ? 0x11 : 0,
        MTHD(0, R_SET_Z_COMPRESSION, 1), d->zt_compressed,
        MTHD(0, R_SET_STENCIL_COMPRESSION, 1), d->zt_compressed,
        MTHD(0, R_SET_ZT_SPARSE, 1), 0,
    };
    memcpy(push, w, sizeof(w));
    return sizeof(w) / 4;
}

static uint64_t div_up(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

// d's depth target as zcull region 0 without storage: what it learns is on the chip and lost at the next region, so
// LOAD_ZCULL, which kills the context on storage never written, isn't needed. Subregions as nil_zcull_new sets them
// out, from a heuristic for what the proprietary driver does. Returns the words written.
static unsigned zcull_region(uint32_t *push, const struct render_draw *d, const struct gsp_zcull *z, float depth)
{
    static const struct { uint32_t format, factor; } formats[] = {
        { 8, 8 },  // Z_4X8_2X2
        { 5, 4 },  // Z_8X8_4X2
        { 2, 2 },  // Z_16X16_4X2
        { 0, 1 },  // Z_16X16X2_4X4
    };
    uint64_t width = div_up(d->width, z->subregion_width_align) * z->subregion_width_align;
    uint64_t height = div_up(d->height, z->subregion_height_align) * z->subregion_height_align;
    uint64_t each = div_up(div_up(width * height, z->pixel_squares_by_aliquots), z->subregion_count);
    uint64_t remaining = z->aliquot_total;
    uint32_t aliquots[GSP_ZCULL_SUBREGIONS] = { 0 }, format[GSP_ZCULL_SUBREGIONS], total = 0, adaptive = 0;
    for (int i = (int)z->subregion_count - 1; i >= 0; i--) {
        format[i] = 0xf;  // none
        for (unsigned f = 0; f < sizeof(formats) / sizeof(formats[0]); f++) {
            uint64_t count = div_up(formats[f].factor * each, 2);
            if (count * (i + 1) <= remaining) {
                aliquots[i] = (uint32_t)count;
                format[i] = formats[f].format;
                remaining -= count;
                break;
            }
        }
        total += aliquots[i];
    }
    for (uint32_t i = 1; i < z->subregion_count; i++)
        adaptive |= format[i] != format[0];

    unsigned n = 0;
    uint32_t w[] = {
        MTHD(0, R_SET_ACTIVE_ZCULL_REGION, 1), 0,
        MTHD(0, R_SET_ZCULL_REGION_LOCATION, 2), total << 16, total,
        MTHD(0, R_SET_ZCULL_STORAGE_A, 4), 0, 0, 0, 0,
        MTHD(0, R_SET_ZCULL_REGION_FORMAT, 1), 0,  // Z_4X4
        MTHD(0, R_SET_ZCULL_REGION_SIZE_A, 4), (uint32_t)width, (uint32_t)height, 1, 0,
        MTHD(0, R_SET_ZCULL_REGION_PIXEL_OFFSET_A, 2), 0, 0,
        MTHD(0, R_SET_ZCULL_SUBREGION, 1), 1 | (uint32_t)(each * z->subregion_count) << 4,
        MTHD(0, R_SET_ZCULL_CRITERION, 1), 0,
        // ZDIR_LESS: the proprietary driver's for every compare function, says NVK
        MTHD(0, R_SET_ZCULL_DIR_FORMAT, 1), (d->zt_format == R_ZT_FORMAT_Z16 ? 1u : 3u) << 16,
    };
    memcpy(push, w, sizeof(w));
    n += sizeof(w) / 4;
    for (uint32_t i = 0; i < z->subregion_count; i++) {
        push[n++] = MTHD(0, R_SET_ZCULL_SUBREGION_ALLOCATION, 1);
        push[n++] = i | aliquots[i] << 8 | format[i] << 24;
    }
    uint32_t end[] = {
        MTHD(0, R_ASSIGN_ZCULL_SUBREGIONS, 1), adaptive,
        MTHD(0, R_SET_ZCULL_SUBREGION_REPORT_TYPE, 1), 1,  // depth test
        MTHD(0, R_SET_ZCULL, 1), 1,
        MTHD(0, R_SET_Z_CLEAR_VALUE, 1), fbits(depth),
        MTHD(0, R_CLEAR_ZCULL_REGION, 1), 1 | 1 << 3,  // Z, make conservative
    };
    memcpy(push + n, end, sizeof(end));
    return n + sizeof(end) / 4;
}

void render_zcull_off(struct chan *ch)
{
    uint32_t push[] = { MTHD(0, R_SET_ACTIVE_ZCULL_REGION, 1), 0x3f };
    chan_push(ch, push, 2);
}

int render_clear_zs(struct chan *ch, const struct render_draw *d, const struct gsp_zcull *z, uint32_t mask, float depth,
                    uint32_t stencil, volatile uint32_t *sem, uint64_t sem_va, uint32_t payload)
{
    uint32_t push[96];
    unsigned n = aa_bind(push, d);
    n += zt_bind(push + n, d);
    if (z) {
        chan_push(ch, push, n);
        n = zcull_region(push, d, z, depth);
        chan_push(ch, push, n);
        n = 0;
    }
    uint32_t w[] = {
        MTHD(0, R_SET_SURFACE_CLIP_HORIZONTAL, 2), d->width << 16, d->height << 16,
        MTHD(0, R_SET_CLEAR_SURFACE_CONTROL, 1), 1u << 4,  // USE_CLEAR_RECT
        MTHD(0, R_SET_CLEAR_RECT_HORIZONTAL, 2), d->width << 16, d->height << 16,
        MTHD(0, R_SET_DEPTH_WRITE, 1), 1,
        // test off, ops keep, func always; the clear writes through the stencil mask
        MTHD(0, R_SET_STENCIL_TEST, 8), 0, 0x1e00, 0x1e00, 0x1e00, 0x207, 0, 0xff, 0xff,
        MTHD(0, R_SET_Z_CLEAR_VALUE, 1), fbits(depth),
        MTHD(0, R_SET_STENCIL_CLEAR_VALUE, 1), stencil,
        MTHD(0, R_CLEAR_SURFACE, 1), mask & 3,  // Z, S; no colour
    };
    memcpy(push + n, w, sizeof(w));
    n += sizeof(w) / 4;
    for (uint32_t l = 1; l < d->layers; l++) {  // a layered target's other layers
        if (n + 2 > sizeof(push) / 4) {
            chan_push(ch, push, n);
            n = 0;
        }
        push[n++] = MTHD(0, R_CLEAR_SURFACE, 1);
        push[n++] = (mask & 3) | l << 10;
    }
    return release(ch, push, n, sem, sem_va, payload, 1);
}

int render_zt_probe(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                    uint32_t *payload)
{
    // colour target first, as a draw has it
    uint32_t ct[] = {
        MTHD(0, R_SET_SURFACE_CLIP_HORIZONTAL, 2), d->width << 16, d->height << 16,
        MTHD(0, R_SET_COLOR_TARGET_A(0), 9), (uint32_t)(d->target >> 32), (uint32_t)d->target, d->pitch,
        d->height, d->format, CT_MEMORY_LAYOUT_PITCH, 1, 0, 0,
        MTHD(0, R_SET_CT_SELECT, 1), 1,
    };
    if (release(ch, ct, sizeof(ct) / 4, sem, sem_va, ++*payload, 1))
        return 0;
    const uint32_t steps[][2] = {
        { R_SET_ZT_A, (uint32_t)(d->zt >> 32) },
        { R_SET_ZT_A + 4, (uint32_t)d->zt },
        { R_SET_ZT_A + 8, d->zt_format },
        { R_SET_ZT_A + 12, 0 },
        { R_SET_ZT_A + 16, 0 },
        { R_SET_ZT_SIZE_A, d->zt_width },
        { R_SET_ZT_SIZE_A + 4, d->zt_height },
        { R_SET_ZT_SIZE_A + 8, 1 | 1 << 16 },
        { R_SET_ZT_LAYER, 0 },
        { R_SET_ZCULL, 0 },
        { R_SET_Z_COMPRESSION, 0 },
        { R_SET_STENCIL_COMPRESSION, 0 },
        { R_SET_ZT_SPARSE, 0 },
        { R_SET_ZT_SELECT, 1 },
        { R_SET_DEPTH_TEST, 1 },
        { R_SET_DEPTH_WRITE, 1 },
        { R_SET_DEPTH_FUNC, 0x201 },
        { R_SET_CLEAR_SURFACE_CONTROL, 1u << 4 },
        { R_SET_CLEAR_RECT_HORIZONTAL, d->width << 16 },
        { R_SET_CLEAR_RECT_HORIZONTAL + 4, d->height << 16 },
        { R_SET_Z_CLEAR_VALUE, fbits(1.0f) },
        { R_CLEAR_SURFACE, 1 },
    };
    for (unsigned i = 0; i < sizeof(steps) / sizeof(steps[0]); i++) {
        uint32_t m[] = { MTHD(0, steps[i][0], 1), steps[i][1] };
        if (release(ch, m, 2, sem, sem_va, ++*payload, 1))
            return (int)i + 1;
    }
    return -1;
}

#define DRAW_WORDS 768  // a draw with every target, stream and stage takes some 630

// viewport j (x, y, width, height, near, far) and scissor j (x, y, width, height) into push; returns the words written
static unsigned viewport_bind(uint32_t *push, unsigned j, const float *vp, const uint32_t *sc)
{
    // NDC -1..1 onto the viewport, +y up as Metal has it; NDC depth is 0..1 already, mapped onto near..far
    float hw = vp[2] / 2.0f, hh = vp[3] / 2.0f;
    float zmin = vp[4] < vp[5] ? vp[4] : vp[5], zmax = vp[4] < vp[5] ? vp[5] : vp[4];
    // clipped to the pixels the viewport covers: a negative height flips it (OpenGL's renderer draws so), and an
    // origin off the target's top or left stops at 0
    uint32_t clip[2][2];
    for (int a = 0; a < 2; a++) {
        float lo = vp[a] < vp[a] + vp[a + 2] ? vp[a] : vp[a] + vp[a + 2], hi = vp[a] + (vp[a + 2] < 0 ? 0 : vp[a + 2]);
        clip[a][0] = lo > 0 ? (uint32_t)lo : 0;
        clip[a][1] = hi > clip[a][0] ? (uint32_t)(hi + 0.999f) - clip[a][0] : 0;
    }
    uint32_t w[] = {
        MTHD(0, R_SET_VIEWPORT_SCALE_X(j), 6), fbits(hw), fbits(-hh), fbits(vp[5] - vp[4]), fbits(vp[0] + hw),
        fbits(vp[1] + hh), fbits(vp[4]),
        MTHD(0, R_SET_VIEWPORT_CLIP_HORIZONTAL(j), 4), clip[0][0] | clip[0][1] << 16, clip[1][0] | clip[1][1] << 16,
        fbits(zmin), fbits(zmax),
        MTHD(0, R_SET_SCISSOR_ENABLE(j), 3), 1, sc[0] | (sc[0] + sc[2]) << 16, sc[1] | (sc[1] + sc[3]) << 16,
    };
    memcpy(push, w, sizeof(w));
    return sizeof(w) / 4;
}

// the methods of a draw into push; returns the words written
static unsigned build_draw(uint32_t *push, const struct render_draw *d)
{
    unsigned n = 0;
#define P(...)                                                   \
    do {                                                         \
        uint32_t _w[] = { __VA_ARGS__ };                         \
        memcpy(push + n, _w, sizeof(_w));                        \
        n += sizeof(_w) / 4;                                     \
    } while (0)
    if (d->wait_idle)
        P(MTHD(0, R_WAIT_FOR_IDLE, 1), 0);
    P(MTHD(0, R_SET_SURFACE_CLIP_HORIZONTAL, 2), d->width << 16, d->height << 16);
    // points: 1 pixel unless the vertex shader writes [[point_size]], square sprites with the origin top left (as
    // NVK has them)
    P(MTHD(0, R_SET_POINT_SIZE, 1), fbits(d->point_size ? d->point_size : 1.0f));
    float lw = d->line_width ? d->line_width : 1.0f;
    P(MTHD(0, R_SET_LINE_WIDTH_FLOAT, 2), fbits(lw), fbits(lw));
    // lines (and triangles drawn as their edges) aliased, unstippled, at the aliased width; polygon modes by the
    // rasterizer itself, as NVK sets them
    P(MTHD(0, R_SET_ALIASED_LINE_WIDTH_ENABLE, 1), 1);
    P(MTHD(0, R_SET_FILL_VIA_TRIANGLE, 1), 0);
    P(MTHD(0, R_SET_LINE_MULTISAMPLE_OVERRIDE, 1), 0);
    P(MTHD(0, R_SET_ANTI_ALIASED_LINE, 1), 0);
    P(MTHD(0, R_SET_LINE_STIPPLE, 1), 0);
    P(MTHD(0, R_SET_PROVOKING_VERTEX, 1), d->provoking_last);
    P(MTHD(0, R_SET_ATTRIBUTE_POINT_SIZE, 1), 1);
    P(MTHD(0, R_SET_POINT_SPRITE, 1), 1);
    P(MTHD(0, R_SET_POINT_SPRITE_SELECT, 1), 1 << 2);
    n += ct_bind(push + n, d);
    n += zt_bind(push + n, d);
    P(MTHD(0, R_SET_DEPTH_TEST, 1), d->zt && d->depth_test);
    P(MTHD(0, R_SET_DEPTH_WRITE, 1), d->zt && d->depth_write);
    P(MTHD(0, R_SET_DEPTH_FUNC, 1), d->depth_func ? d->depth_func : 0x207);  // OGL_ALWAYS
    // an unset compare function (0) is not a valid one: always pass
    const uint32_t *f = d->stencil[0], *b = d->stencil[1];
    P(MTHD(0, R_SET_STENCIL_TEST, 8), d->zt && d->stencil_test, f[0], f[1], f[2], f[3] ? f[3] : 0x207,
      d->stencil_ref & 0xff, d->stencil_read_mask[0], d->stencil_write_mask[0]);
    P(MTHD(0, R_SET_TWO_SIDED_STENCIL_TEST, 5), 1, b[0], b[1], b[2], b[3] ? b[3] : 0x207);
    P(MTHD(0, R_SET_BACK_STENCIL_FUNC_REF, 3), d->stencil_ref >> 8 & 0xff, d->stencil_write_mask[1],
      d->stencil_read_mask[1]);
    // Metal judges winding with +y up, as GL does, so its clockwise is GL's (an Apple GPU says so: the y flip of our
    // viewport doesn't swap them)
    P(MTHD(0, R_OGL_SET_CULL, 3), d->cull_face != 0, d->front_ccw ? 0x901 : 0x900, d->cull_face ? d->cull_face : 0x405);
    P(MTHD(0, R_SET_WINDOW_ORIGIN, 1), 0);  // upper left

    n += viewport_bind(push + n, 0, d->viewport, d->scissor);
    P(MTHD(0, R_SET_VIEWPORT_SCALE_OFFSET, 1), !d->window_coords);
    // the layer the vertex stage picks, or the first
    P(MTHD(0, R_SET_RT_LAYER, 1), d->shader_layer ? 1u << 16 : 0);
    // depth 0..1, fragment depth clamped to it; geometry clipped to it by the geometry clipper with a tight z
    // guardband (before depth bias applies, as NVK has it), or for depth clamp, clipped in x and y only and
    // clamped to near..far
    P(MTHD(0, R_SET_VIEWPORT_CLIP_CONTROL, 1),
      d->depth_clamp ? 1 << 16 | 1 << 3 | 1 << 4 | 2 << 11 | 1 << 1 : 1 | 1 << 3 | 1 << 4 | 3 << 11 | 2 << 1);
    P(MTHD(0, R_SET_VIEWPORT_Z_CLIP, 1), 1);  // clip-space z from 0 to w, not GL's -w: Metal's NDC depth is 0..1
    uint32_t mode = d->fill_lines ? 0x1b01 : 0x1b02;
    P(MTHD(0, R_SET_FRONT_POLYGON_MODE, 2), mode, mode);
    P(MTHD(0, R_SET_EDGE_FLAG, 1), 1);  // every edge drawn in line mode (it starts out false: none are)
    const float *bias = d->depth_bias;
    uint32_t biased = bias[0] != 0 || bias[1] != 0;
    P(MTHD(0, R_SET_POLY_OFFSET_POINT, 3), biased, biased, biased);
    P(MTHD(0, R_SET_DEPTH_BIAS, 1), fbits(bias[0]));
    P(MTHD(0, R_SET_SLOPE_SCALE_DEPTH_BIAS, 1), fbits(bias[1]));
    P(MTHD(0, R_SET_DEPTH_BIAS_CLAMP, 1), fbits(bias[2]));
    const float *bc = d->blend_const;
    P(MTHD(0, R_SET_BLEND_CONST_RED, 4), fbits(bc[0]), fbits(bc[1]), fbits(bc[2]), fbits(bc[3]));

    // the vertex and pixel stages run, and for patches the tessellation init and tessellation stages
    for (uint32_t j = 0; j <= PIPE_PIXEL; j++)
        P(MTHD(0, R_SET_PIPELINE_SHADER(j), 1),
          (j == PIPE_VERTEX || j == PIPE_PIXEL || (d->tess && (j == PIPE_TESS_INIT || j == PIPE_TESS))) | j << 4);
    P(MTHD(0, R_SET_PIPELINE_REGISTER_COUNT(PIPE_VERTEX), 4), d->vs_regs, GROUP_VERTEX, (uint32_t)(d->vs >> 32),
      (uint32_t)d->vs);
    P(MTHD(0, R_SET_PIPELINE_REGISTER_COUNT(PIPE_PIXEL), 4), d->ps_regs, GROUP_PIXEL, (uint32_t)(d->ps >> 32),
      (uint32_t)d->ps);
    if (d->tess) {  // a patch a vertex, tessellated as the domain, spacing and output primitives say
        P(MTHD(0, R_SET_PIPELINE_REGISTER_COUNT(PIPE_TESS_INIT), 4), d->ti_regs, GROUP_TESS_INIT,
          (uint32_t)(d->ti >> 32), (uint32_t)d->ti);
        P(MTHD(0, R_SET_PIPELINE_REGISTER_COUNT(PIPE_TESS), 4), d->ts_regs, GROUP_TESS, (uint32_t)(d->ts >> 32),
          (uint32_t)d->ts);
        P(MTHD(0, R_SET_TESSELLATION_PARAMETERS, 1), d->tess);
        P(MTHD(0, R_SET_PATCH, 1), 1);
    }
    P(MTHD(0, R_SET_MESH_CONTROL, 1), 0);

    // each stage's const bank 0 (its arguments, as compute kernels have them); the tessellation stage reads the
    // vertex stage's
    const uint64_t cbuf[4] = { d->vs_cbuf, d->ps_cbuf, d->tess ? d->ti_cbuf : 0, d->tess ? d->vs_cbuf : 0 };
    const uint32_t group[4] = { GROUP_VERTEX, GROUP_PIXEL, GROUP_TESS_INIT, GROUP_TESS };
    for (int s = 0; s < 4; s++) {
        if (!cbuf[s])
            continue;
        P(MTHD(0, R_SET_CONSTANT_BUFFER_SELECTOR_A, 3), CBUF_BYTES, (uint32_t)(cbuf[s] >> 32), (uint32_t)cbuf[s]);
        P(MTHD(0, R_BIND_GROUP_CONSTANT_BUFFER(group[s]), 1), 1);  // valid, shader slot 0
    }
    if (d->tex_headers) {
        P(MTHD(0, R_SET_TEX_HEADER_POOL_A, 3), (uint32_t)(d->tex_headers >> 32), (uint32_t)d->tex_headers,
          d->tex_count - 1);
        P(MTHD(0, R_SET_TEX_SAMPLER_POOL_A, 3), (uint32_t)(d->tex_samplers >> 32), (uint32_t)d->tex_samplers,
          (d->sampler_count ? d->sampler_count : d->tex_count) - 1);
    }
    // instruction, data, constant; without the wait the plain method makes the pipeline idle first, which a new
    // pass has had above and a draw within one doesn't need (measured: 3 us a draw)
    if (d->invalidate)
        P(MTHD(0, R_INVALIDATE_SHADER_CACHES_NO_WFI, 1), 1 | 1 << 4 | 1 << 12);
    // the pools are rewritten for every launch, and a texture may have been a render target just before
    P(MTHD(0, R_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, 1), 0);
    P(MTHD(0, R_INVALIDATE_SAMPLER_CACHE_NO_WFI, 1), 0);
    P(MTHD(0, R_INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, 1), 0);

    // an unused attribute still needs a valid format
    push[n++] = MTHD(0, R_SET_VERTEX_ATTRIBUTE_A(0), RENDER_ATTRIBUTES);
    for (uint32_t i = 0; i < RENDER_ATTRIBUTES; i++)
        push[n++] = d->attributes[i] ? d->attributes[i] : ATTR_INACTIVE | ATTR_R32G32B32A32_FLOAT;
    for (uint32_t s = 0; s < RENDER_STREAMS; s++) {
        const struct render_stream *v = &d->streams[s];
        if (!v->addr) {
            P(MTHD(0, R_SET_VERTEX_STREAM_A_FORMAT(s), 1), 0);
            continue;
        }
        P(MTHD(0, R_SET_VERTEX_STREAM_A_FORMAT(s), 4), v->stride | 1 << 12, (uint32_t)(v->addr >> 32),
          (uint32_t)v->addr, v->divisor);
        P(MTHD(0, R_SET_VERTEX_STREAM_SIZE_A(s), 2), 0, v->size);
        P(MTHD(0, R_SET_VERTEX_STREAM_INSTANCE_A(s), 1), v->divisor != 0);
    }

    P(MTHD(0, R_SET_DA_OUTPUT, 1), 1 << 12);  // vertex ids count from the first vertex drawn, as Metal's do
    P(MTHD(0, R_SET_GLOBAL_BASE_VERTEX_INDEX, 2), (uint32_t)d->base_vertex, d->base_instance);
    P(MTHD(0, R_SET_VERTEX_ID_BASE, 1), (uint32_t)d->base_vertex);
    // topology, instance ids from 0 and counting up across the instances
    P(MTHD(0, R_SET_DRAW_CONTROL_A, 2), d->primitive | 1 << 9, d->instances);
    if (d->visibility && d->visibility_reset)
        P(MTHD(0, R_CLEAR_REPORT_VALUE, 1), 1);  // ZPASS_PIXEL_CNT
    P(MTHD(0, R_SET_ZPASS_PIXEL_COUNT, 1), d->visibility != 0);
    // Metal always restarts a strip at the largest index its type holds
    static const uint32_t restart[] = { 0xff, 0xffff, 0xffffffff };
    P(MTHD(0, R_SET_DA_PRIMITIVE_RESTART, 2), d->index && d->restart != 1,
      d->restart == 2 ? d->restart_index : restart[d->index_format]);
    if (d->index) {
        P(MTHD(0, R_SET_INDEX_BUFFER_A, 2), (uint32_t)(d->index >> 32), (uint32_t)d->index);
        P(MTHD(0, R_SET_INDEX_BUFFER_SIZE_A, 2), 0, d->index_size);
        P(MTHD(0, R_SET_INDEX_BUFFER_E, 1), d->index_format);
        P(MTHD(0, R_DRAW_INDEX_BUFFER_BEGIN_END_A, 2), d->first, d->count);
    } else {
        P(MTHD(0, R_DRAW_VERTEX_ARRAY_BEGIN_END_A, 2), d->first, d->count);
    }
    // the count so far, once the draw is through the pipeline: ZPASS_PIXEL_CNT64 reported a word at a time, which
    // fills the 8 bytes Metal gives a result and no more (the four-word structure would add a timestamp)
    for (uint32_t w = 0; d->visibility && w < 2; w++)
        P(MTHD(0, R_SET_REPORT_SEMAPHORE_A, 4), (uint32_t)((d->visibility + 4 * w) >> 32),
          (uint32_t)(d->visibility + 4 * w), 0, 2 | 1 << 4 | 0xfu << 12 | 0x15u << 23 | w << 21 | 1u << 28);
#undef P
    return n;
}

// the viewports and scissors beside the first, for a vertex stage that picks one; apart from the draw's methods,
// which have no room for them
static void push_viewports(struct chan *ch, const struct render_draw *d)
{
    uint32_t w[15 * 16];
    unsigned n = 0;
    for (uint32_t j = 1; j < d->nviewports && j < 16; j++)
        n += viewport_bind(w + n, j, d->viewports[j - 1], d->scissors[j - 1]);
    if (n)
        chan_push(ch, w, n);
}

int render_draw(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                uint32_t payload)
{
    push_viewports(ch, d);
    uint32_t push[DRAW_WORDS];
    return release(ch, push, build_draw(push, d), sem, sem_va, payload, 0);
}

int render_draw_probe(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                      uint32_t *payload)
{
    uint32_t push[DRAW_WORDS];
    unsigned n = build_draw(push, d);
    for (unsigned i = 0; i < n;) {
        unsigned len = 1 + ((push[i] >> 16) & 0x1fff);  // the header and its data words
        if (release(ch, push + i, len, sem, sem_va, ++*payload, 1))
            return (int)((push[i] & 0x1fff) << 2);
        i += len;
    }
    return -1;
}
