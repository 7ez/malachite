#include "drawbuild.h"

#define P(...)                                                   \
    do {                                                         \
        const uint32_t _w[] = { __VA_ARGS__ };                   \
        for (unsigned _i = 0; _i < sizeof(_w) / 4; _i++)         \
            push[n++] = _w[_i];                                  \
    } while (0)

// the sample grid of a pixel, and SET_ANTI_ALIAS's mode for it: 2X1, 2X2, 4X2_D3D (Metal's positions are D3D's)
static uint32_t aa_mode(uint32_t samples, uint32_t *sx, uint32_t *sy)
{
    *sx = samples >= 8 ? 4 : samples >= 2 ? 2 : 1;
    *sy = samples >= 4 ? 2 : 1;
    return samples >= 8 ? 4 : samples >= 4 ? 2 : samples >= 2 ? 1 : 0;
}

// MSAA state: the fragment shader runs per pixel, or with per_sample once per sample (a pass each). Sample positions
// are D3D/Vulkan/Metal's (16ths, same across a 2x2 quad); the hardware default puts them all at the centre.
// Alpha to coverage dithers over 2x2 pixels with multiple samples, as Apple GPUs do; with one it's alpha >= 0.5.
unsigned draw_aa_words(uint32_t *push, const struct render_draw *d)
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
        // post-shader coverage starts from the rasterizer's; a [[sample_mask]] output narrows it
        MTHD(0, R_SET_POST_PS_INITIAL_COVERAGE, 1), 1,
        MTHD(0, R_SET_ANTI_ALIAS_SAMPLE_POSITIONS, 4), 0, 0, 0, 0,
    };
    for (unsigned n = 0; n < 16; n++) {  // sample n % per of the quad's pixel n / per
        const uint8_t *at = standard[set][n % per];
        w[sizeof(w) / 4 - 4 + n / 4] |= (uint32_t)(at[0] | at[1] << 4) << (n % 4 * 8);
    }
    put(push, w, sizeof(w) / 4);
    return sizeof(w) / 4;
}

// each colour target with its write mask and blend (an absent one gets format 0), then CT_SELECT over them all
unsigned draw_target_words(uint32_t *push, const struct render_draw *d)
{
    uint32_t sx, sy;
    aa_mode(d->samples, &sx, &sy);
    unsigned n = draw_aa_words(push, d), count = 1;
    for (unsigned j = 1; j < 8; j++)
        if (d->more[j - 1].addr)
            count = j + 1;
    uint32_t select = count;
    for (unsigned j = 0; j < count; j++) {
        struct render_ct ct = { d->target, d->pitch, d->format, d->target_blocklinear, d->write_mask };
        memcpy(ct.blend, d->blend, sizeof(ct.blend));
        ct.width = d->target_width;
        ct.height = d->target_height;
        if (j)
            ct = d->more[j - 1];
        // its own size sets the block-linear layout; the pass's surface clip bounds the draw
        uint32_t width = ct.width ? ct.width : d->width, height = ct.height ? ct.height : d->height;
        // block-linear (one-GOB blocks) takes the width in samples, pitch-linear the row pitch in bytes
        uint32_t w[] = {
            MTHD(0, R_SET_COLOR_TARGET_A(j), 9), (uint32_t)(ct.addr >> 32), (uint32_t)ct.addr,
            ct.blocklinear ? width * sx : ct.pitch, height * sy, ct.addr ? ct.format : 0,
            ct.blocklinear ? 0 : CT_MEMORY_LAYOUT_PITCH, draw_target_layers(d, j), d->layer_pitch[j] >> 2, 0,
            MTHD(0, R_SET_CT_WRITE(j), 1), ct.write_mask,
            MTHD(0, R_SET_BLEND(j), 1), ct.blend[0],
        };
        put(push + n, w, sizeof(w) / 4);
        n += sizeof(w) / 4;
        if (ct.blend[0]) {  // the ops and factors only mean something (and are only valid) with blending on
            uint32_t b[] = { MTHD(0, R_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(j), 7), 1, ct.blend[1], ct.blend[2],
                             ct.blend[3], ct.blend[4], ct.blend[5], ct.blend[6] };
            put(push + n, b, sizeof(b) / 4);
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

unsigned draw_depth_stencil_words(uint32_t *push, const struct render_draw *d)
{
    unsigned n = 0;
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
    return n;
}

unsigned draw_cull_words(uint32_t *push, const struct render_draw *d)
{
    unsigned n = 0;
    // Metal judges winding with +y up like GL (checked on an Apple GPU): our viewport's y flip doesn't swap it
    P(MTHD(0, R_OGL_SET_CULL, 3), d->cull_face != 0, d->front_ccw ? 0x901 : 0x900, d->cull_face ? d->cull_face : 0x405);
    return n;
}

unsigned viewport_bind(uint32_t *push, unsigned j, const float *vp, const uint32_t *sc)
{
    // NDC -1..1 onto the viewport, +y up as Metal has it; NDC depth is 0..1 already, mapped onto near..far
    float hw = vp[2] / 2.0f, hh = vp[3] / 2.0f;
    float zmin = vp[4] < vp[5] ? vp[4] : vp[5], zmax = vp[4] < vp[5] ? vp[5] : vp[4];
    // clip to the viewport's pixels: a negative height (GL renderer) flips it, an origin off the top/left clamps to 0
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
    put(push, w, sizeof(w) / 4);
    return sizeof(w) / 4;
}

unsigned draw_viewport_words(uint32_t *push, const struct render_draw *d)
{
    unsigned n = viewport_bind(push, 0, d->viewport, d->scissor);
    P(MTHD(0, R_SET_VIEWPORT_SCALE_OFFSET, 1), !d->window_coords);
    // the layer the vertex stage picks, or the first
    P(MTHD(0, R_SET_RT_LAYER, 1), d->shader_layer ? 1u << 16 : 0);
    // depth 0..1, fragment depth clamped to it; z clipped by the geometry clipper with a tight guardband (before
    // depth bias), or with depth_clamp clipped in x/y only and clamped to near..far
    P(MTHD(0, R_SET_VIEWPORT_CLIP_CONTROL, 1),
      d->depth_clamp ? 1 << 16 | 1 << 3 | 1 << 4 | 2 << 11 | 1 << 1 : 1 | 1 << 3 | 1 << 4 | 3 << 11 | 2 << 1);
    P(MTHD(0, R_SET_VIEWPORT_Z_CLIP, 1), 1);  // clip-space z from 0 to w, not GL's -w: Metal's NDC depth is 0..1
    return n;
}

unsigned draw_bias_words(uint32_t *push, const struct render_draw *d)
{
    unsigned n = 0;
    const float *bias = d->depth_bias;
    uint32_t biased = bias[0] != 0 || bias[1] != 0;
    P(MTHD(0, R_SET_POLY_OFFSET_POINT, 3), biased, biased, biased);
    P(MTHD(0, R_SET_DEPTH_BIAS, 1), fbits(bias[0]));
    P(MTHD(0, R_SET_SLOPE_SCALE_DEPTH_BIAS, 1), fbits(bias[1]));
    P(MTHD(0, R_SET_DEPTH_BIAS_CLAMP, 1), fbits(bias[2]));
    return n;
}

unsigned draw_blend_const_words(uint32_t *push, const struct render_draw *d)
{
    unsigned n = 0;
    const float *bc = d->blend_const;
    P(MTHD(0, R_SET_BLEND_CONST_RED, 4), fbits(bc[0]), fbits(bc[1]), fbits(bc[2]), fbits(bc[3]));
    return n;
}

// The stages' pipeline state, tessellation's parameters included: they sit between its register counts.
unsigned draw_shader_words(uint32_t *push, const struct render_draw *d)
{
    unsigned n = 0;
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
    return n;
}

unsigned draw_attribute_words(uint32_t *push, const struct render_draw *d)
{
    unsigned n = 0;
    // an unused attribute still needs a valid format
    push[n++] = MTHD(0, R_SET_VERTEX_ATTRIBUTE_A(0), RENDER_ATTRIBUTES);
    for (uint32_t i = 0; i < RENDER_ATTRIBUTES; i++)
        push[n++] = d->attributes[i] ? d->attributes[i] : ATTR_INACTIVE | ATTR_R32G32B32A32_FLOAT;
    return n;
}
