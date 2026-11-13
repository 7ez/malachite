// The 3D class behind MTLRenderCommandEncoder, on the compute bridge's graphics channel; methods per cl9097.h.
#include "chip.h"
#include "chan.h"
#include "gsp.h"
#include "render.h"
#include "drawbuild.h"

// the release already waits for earlier writes; idle also holds back what comes next
static void release_queued(struct chan *ch, uint32_t *push, unsigned n, uint64_t sem_va, uint32_t payload, int idle)
{
    uint32_t tail[] = {
        MTHD(0, R_WAIT_FOR_IDLE, 1), 0,
        // one word, from the end of the pipeline
        MTHD(0, R_SET_REPORT_SEMAPHORE_A, 4), (uint32_t)(sem_va >> 32), (uint32_t)sem_va, payload,
        (1u << 4) | (0xfu << 12) | (1u << 28),
    };
    if (n)
        chan_push(ch, push, n);
    chan_push(ch, tail + (idle ? 0 : 2), sizeof(tail) / 4 - (idle ? 0 : 2));
}

static int release(struct chan *ch, uint32_t *push, unsigned n, volatile uint32_t *sem, uint64_t sem_va,
                   uint32_t payload, int idle)
{
    release_queued(ch, push, n, sem_va, payload, idle);
    chan_kick(ch);
    return sem ? wait_word(sem, payload, 2000) : 0;
}

void render_release(struct chan *ch, uint64_t sem_va, uint32_t payload)
{
    release(ch, NULL, 0, NULL, sem_va, payload, 1);
}

void render_mark(struct chan *ch, uint64_t sem_va, uint32_t payload)
{
    release(ch, NULL, 0, NULL, sem_va, payload, 0);
}

void render_stamp(struct chan *ch, uint64_t va, uint32_t payload)
{
    // four-word report, the timestamp at +8
    uint32_t w[] = {
        MTHD(0, R_SET_REPORT_SEMAPHORE_A, 4), (uint32_t)(va >> 32), (uint32_t)va, payload, (1u << 4) | (0xfu << 12),
    };
    chan_push(ch, w, sizeof(w) / 4);
}

int render_wait(volatile uint32_t *sem, uint32_t payload, int ms)
{
    uint64_t end = time_us() + (uint64_t)ms * 1000;
    while ((int32_t)(*sem - payload) < 0) {  // a later release may already have overwritten ours
        if (time_us() > end)
            return -1;
        delay_us(10);
    }
    return 0;
}

// Mesa nvc0's Turing+ MME macro for the GR firmware to set a priv register; args are register, value, mask.
static const uint32_t set_priv_reg[] = {
    0x8d147403, 0x18c00011, 0x31c00300, 0x00600003, 0x18c00000, 0x318c0300, 0x1a60346b, 0x19058231,
    0x300c0300, 0x00000003, 0x18c00040, 0x01804300, 0x0000000b, 0x18c02001, 0x340c7700, 0x01380103,
    0x18c00340, 0x31905b00, 0x00000007, 0x18c017ff, 0xf40c7701, 0x00000003, 0x18c02000, 0xf18c6300,
    0x0000002b, 0x18c00000, 0xb1ac5f00, 0x00d40003, 0x18c00010, 0x318c0300, 0x00000003, 0x18c00000,
    0x318c0301, 0x00000003, 0x18c00000, 0x318c0300,
};

// Volta to Ada helper threads skip loads and read zeros unless bit 3 is clear, breaking derivatives at edges.
#define SM_DISP_CTRL 0x419ba4
#define SM_DISP_CTRL_HELPER_LOADS_OFF 8

int render_setup(struct gsp *s, const struct rm_client *c, struct chan *ch)
{
    if (rm_alloc(s, c->client, ch->handle, 0xc7970000, gsp_gpu(s)->chip->eng3d, NULL, 0))
        return -1;
    uint32_t push[] = {
        MTHD(0, R_SET_OBJECT, 1), gsp_gpu(s)->chip->eng3d,
        // as NVK, snorm8/unorm16/snorm16 blend only on request and 0 * inf or NaN is 0
        MTHD(0, R_SET_BLEND_PER_FORMAT_ENABLE, 1), 1 << 4,
        MTHD(0, R_SET_BLEND_FLOAT_OPTION, 1), 1,
        MTHD(0, R_SET_COLOR_COMPRESSION(0), 8), 1, 1, 1, 1, 1, 1, 1, 1,  // the PTE kind decides
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
        MTHD(0, R_SET_SAMPLER_BINDING, 1), 0,  // independent, as handles name header and sampler apart
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

void render_set_vertex_substitute(struct chan *ch, uint64_t va)
{
    uint32_t push[] = { MTHD(0, R_SET_VERTEX_STREAM_SUBSTITUTE_A, 2), (uint32_t)(va >> 32), (uint32_t)va };
    chan_submit(ch, push, sizeof(push) / 4);
}

void render_set_local_memory(struct chan *ch, uint64_t va, uint64_t size, uint32_t per_warp)
{
    uint32_t push[] = {
        MTHD(0, R_WAIT_FOR_IDLE, 1), 0,
        MTHD(0, R_SET_SHADER_LOCAL_MEMORY_WINDOW, 1), 0xff000000,  // same as compute
        MTHD(0, R_SET_SHADER_LOCAL_MEMORY_A, 5), (uint32_t)(va >> 32), (uint32_t)va, (uint32_t)(size >> 32),
        (uint32_t)size, per_warp,
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

int render_clear(struct chan *ch, const struct render_draw *t, unsigned target, const float rgba[4],
                 volatile uint32_t *sem, uint64_t sem_va, uint32_t payload)
{
    uint32_t c[4], w = t->width, h = t->height, push[256];
    memcpy(c, rgba, sizeof(c));
    // unlike a draw, a clear doesn't wait for the previous pass to stop sampling the target
    push[0] = MTHD(0, R_WAIT_FOR_IDLE, 1);
    push[1] = 0;
    unsigned n = 2 + draw_target_words(push + 2, t);
    uint32_t clear[] = {
        MTHD(0, R_SET_SURFACE_CLIP_HORIZONTAL, 2), w << 16, h << 16,
        MTHD(0, R_SET_CLEAR_SURFACE_CONTROL, 1), 1u << 4,  // USE_CLEAR_RECT
        MTHD(0, R_SET_CLEAR_RECT_HORIZONTAL, 2), w << 16, h << 16,
        MTHD(0, R_SET_COLOR_CLEAR_VALUE(0), 4), c[0], c[1], c[2], c[3],
        MTHD(0, R_SET_CT_WRITE(target), 1), 0x1111,  // a clear ignores the draw's write mask
        MTHD(0, R_CLEAR_SURFACE, 1), 0xf << 2 | target << 6,  // R, G, B, A of MRT `target`
    };
    put(push + n, clear, sizeof(clear) / 4);
    n += sizeof(clear) / 4;
    for (uint32_t l = 1; l < t->layers; l++) {
        if (n + 2 > sizeof(push) / 4) {
            chan_push(ch, push, n);
            n = 0;
        }
        push[n++] = MTHD(0, R_CLEAR_SURFACE, 1);
        push[n++] = 0xf << 2 | target << 6 | l << 10;
    }
    return release(ch, push, n, sem, sem_va, payload, 1);
}

static unsigned zt_bind(uint32_t *push, const struct render_draw *d)
{
    if (!d->zt) {
        push[0] = MTHD(0, R_SET_ZT_SELECT, 1);
        push[1] = 0;
        return 2;
    }
    uint32_t w[] = {
        MTHD(0, R_SET_ZT_A, 5), (uint32_t)(d->zt >> 32), (uint32_t)d->zt, d->zt_format, 0,  // one-GOB blocks
        d->zt_layer_pitch >> 2,
        MTHD(0, R_SET_ZT_SELECT, 1), 1,
        // Z/S has no tile width, so width is the row stride in pixels
        MTHD(0, R_SET_ZT_SIZE_A, 3), d->zt_width, d->zt_height, d->layers > 1 ? draw_target_layers(d, 8) : (1 | 1 << 16),
        MTHD(0, R_SET_ZT_LAYER, 1), 0,
        MTHD(0, R_SET_ZCULL, 1), d->zt_zcull,
        // shader-written depth isn't the plane zcull knows
        MTHD(0, R_SET_ZCULL_BOUNDS, 1), d->ps_depth ? 0x11 : 0,
        MTHD(0, R_SET_Z_COMPRESSION, 1), d->zt_compressed,
        MTHD(0, R_SET_STENCIL_COMPRESSION, 1), d->zt_compressed,
        MTHD(0, R_SET_ZT_SPARSE, 1), 0,
    };
    put(push, w, sizeof(w) / 4);
    return sizeof(w) / 4;
}

static uint64_t div_up(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

// Region 0 without storage, so no LOAD_ZCULL, which kills the context on unwritten storage; subregions as
// nil_zcull_new.
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
        // ZDIR_LESS for every compare function, as the proprietary driver does per NVK
        MTHD(0, R_SET_ZCULL_DIR_FORMAT, 1), (d->zt_format == R_ZT_FORMAT_Z16 ? 1u : 3u) << 16,
    };
    put(push, w, sizeof(w) / 4);
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
    put(push + n, end, sizeof(end) / 4);
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
    push[0] = MTHD(0, R_WAIT_FOR_IDLE, 1);
    push[1] = 0;
    unsigned n = 2 + draw_aa_words(push + 2, d);
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
        // the clear writes through the stencil mask
        MTHD(0, R_SET_STENCIL_TEST, 8), 0, 0x1e00, 0x1e00, 0x1e00, 0x207, 0, 0xff, 0xff,
        MTHD(0, R_SET_Z_CLEAR_VALUE, 1), fbits(depth),
        MTHD(0, R_SET_STENCIL_CLEAR_VALUE, 1), stencil,
        MTHD(0, R_CLEAR_SURFACE, 1), mask & 3,  // Z, S; no colour
    };
    put(push + n, w, sizeof(w) / 4);
    n += sizeof(w) / 4;
    for (uint32_t l = 1; l < draw_target_layers(d, 8); l++) {
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

static unsigned build_draw(uint32_t *push, const struct render_draw *d)
{
    unsigned n = 0;
// stores, not memcpy, which is a real call under -fno-builtin, some 90 a draw
#define P(...)                                                   \
    do {                                                         \
        const uint32_t _w[] = { __VA_ARGS__ };                   \
        for (unsigned _i = 0; _i < sizeof(_w) / 4; _i++)         \
            push[n++] = _w[_i];                                  \
    } while (0)
    if (d->wait_idle)
        P(MTHD(0, R_WAIT_FOR_IDLE, 1), 0);
    P(MTHD(0, R_SET_SURFACE_CLIP_HORIZONTAL, 2), d->width << 16, d->height << 16);
    // point sprites with the origin top left, as NVK
    P(MTHD(0, R_SET_POINT_SIZE, 1), fbits(d->point_size ? d->point_size : 1.0f));
    float lw = d->line_width ? d->line_width : 1.0f;
    P(MTHD(0, R_SET_LINE_WIDTH_FLOAT, 2), fbits(lw), fbits(lw));
    // aliased unstippled lines, polygon modes in the rasteriser, as NVK
    P(MTHD(0, R_SET_ALIASED_LINE_WIDTH_ENABLE, 1), 1);
    P(MTHD(0, R_SET_FILL_VIA_TRIANGLE, 1), 0);
    P(MTHD(0, R_SET_LINE_MULTISAMPLE_OVERRIDE, 1), 0);
    P(MTHD(0, R_SET_ANTI_ALIASED_LINE, 1), 0);
    P(MTHD(0, R_SET_LINE_STIPPLE, 1), 0);
    P(MTHD(0, R_SET_PROVOKING_VERTEX, 1), d->provoking_last);
    P(MTHD(0, R_SET_ATTRIBUTE_POINT_SIZE, 1), 1);
    P(MTHD(0, R_SET_POINT_SPRITE, 1), 1);
    P(MTHD(0, R_SET_POINT_SPRITE_SELECT, 1), 1 << 2);
    n += draw_target_words(push + n, d);
    n += zt_bind(push + n, d);
    n += draw_depth_stencil_words(push + n, d);
    n += draw_cull_words(push + n, d);
    P(MTHD(0, R_SET_WINDOW_ORIGIN, 1), 0);  // upper left

    n += draw_viewport_words(push + n, d);
    uint32_t mode = d->fill_lines ? 0x1b01 : 0x1b02;
    P(MTHD(0, R_SET_FRONT_POLYGON_MODE, 2), mode, mode);
    P(MTHD(0, R_SET_EDGE_FLAG, 1), 1);  // starts false, which draws no edges in line mode
    n += draw_bias_words(push + n, d);
    n += draw_blend_const_words(push + n, d);

    n += draw_shader_words(push + n, d);
    P(MTHD(0, R_SET_MESH_CONTROL, 1), 0);

    // const bank 0 holds the arguments; the tessellation stage reads the vertex stage's
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
    // instruction, data, constant; the plain method idles first, which cost 3 us a draw
    if (d->invalidate)
        P(MTHD(0, R_INVALIDATE_SHADER_CACHES_NO_WFI, 1), 1 | 1 << 4 | 1 << 12);
    // pools are rewritten every launch, and a texture may just have been a render target
    P(MTHD(0, R_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, 1), 0);
    P(MTHD(0, R_INVALIDATE_SAMPLER_CACHE_NO_WFI, 1), 0);
    P(MTHD(0, R_INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, 1), 0);

    n += draw_attribute_words(push + n, d);
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

    P(MTHD(0, R_SET_DA_OUTPUT, 1), 1 << 12);  // vertex ids from the first vertex drawn, as Metal
    P(MTHD(0, R_SET_GLOBAL_BASE_VERTEX_INDEX, 2), (uint32_t)d->base_vertex, d->base_instance);
    P(MTHD(0, R_SET_VERTEX_ID_BASE, 1), (uint32_t)d->base_vertex);
    P(MTHD(0, R_SET_DRAW_CONTROL_A, 2), d->primitive | 1 << 9, d->instances);
    if (d->visibility && d->visibility_reset)
        P(MTHD(0, R_CLEAR_REPORT_VALUE, 1), 1);  // ZPASS_PIXEL_CNT
    P(MTHD(0, R_SET_ZPASS_PIXEL_COUNT, 1), d->visibility != 0);
    P(MTHD(0, R_SET_DA_PRIMITIVE_RESTART, 2), draw_restarts(d->primitive, d->restart, d->index != 0),
      draw_restart_index(d->restart, d->restart_index, d->index_format));
    if (d->index) {
        P(MTHD(0, R_SET_INDEX_BUFFER_A, 2), (uint32_t)(d->index >> 32), (uint32_t)d->index);
        P(MTHD(0, R_SET_INDEX_BUFFER_SIZE_A, 2), 0, d->index_size);
        P(MTHD(0, R_SET_INDEX_BUFFER_E, 1), d->index_format);
        P(MTHD(0, R_DRAW_INDEX_BUFFER_BEGIN_END_A, 2), d->first, d->count);
    } else {
        P(MTHD(0, R_DRAW_VERTEX_ARRAY_BEGIN_END_A, 2), d->first, d->count);
    }
    // ZPASS_PIXEL_CNT64 a word at a time, as the four-word report would overrun Metal's 8-byte result
    for (uint32_t w = 0; d->visibility && w < 2; w++)
        P(MTHD(0, R_SET_REPORT_SEMAPHORE_A, 4), (uint32_t)((d->visibility + 4 * w) >> 32),
          (uint32_t)(d->visibility + 4 * w), 0, 2 | 1 << 4 | 0xfu << 12 | 0x15u << 23 | w << 21 | 1u << 28);
#undef P
    return n;
}

// viewports past the first, pushed apart as the draw's methods have no room for them
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

void render_draw_queued(struct chan *ch, const struct render_draw *d, uint64_t sem_va, uint32_t payload)
{
    push_viewports(ch, d);
    uint32_t *push = chan_reserve(ch, DRAW_WORDS + 5);
    unsigned n = build_draw(push, d);
    const uint32_t tail[] = {
        MTHD(0, R_SET_REPORT_SEMAPHORE_A, 4), (uint32_t)(sem_va >> 32), (uint32_t)sem_va, payload,
        (1u << 4) | (0xfu << 12) | (1u << 28),
    };
    put(push + n, tail, sizeof(tail) / 4);
    chan_commit(ch, n + sizeof(tail) / 4);
}

int render_draw_probe(struct chan *ch, const struct render_draw *d, volatile uint32_t *sem, uint64_t sem_va,
                      uint32_t *payload)
{
    uint32_t push[DRAW_WORDS];
    unsigned n = build_draw(push, d);
    for (unsigned i = 0; i < n;) {
        unsigned len = 1 + ((push[i] >> 16) & 0x1fff);  // header and data
        if (release(ch, push + i, len, sem, sem_va, ++*payload, 1))
            return (int)((push[i] & 0x1fff) << 2);
        i += len;
    }
    return -1;
}
