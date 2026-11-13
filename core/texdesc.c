#include "texdesc.h"
#include <string.h>

// set bits [lo,hi] of a 256-bit little-endian descriptor
static void setf(uint32_t out[8], unsigned lo, unsigned hi, uint32_t val)
{
    for (unsigned b = lo; b <= hi; b++)
        if (val & (1u << (b - lo)))
            out[b / 32] |= 1u << (b % 32);
}

enum { SNORM = 1, UNORM = 2, SINT = 3, UINT = 4, FLOAT = 7 };  // TEXHEAD component data types
enum { I0 = 0, IR = 2, IG = 3, IB = 4, IA = 5, I1_INT = 6, I1 = 7 };  // swizzle sources: zero, a component, one

// TEXHEAD COMPONENTS_SIZES, R's data type and G/B/A's when different, X/Y/Z/W sources, sRGB, bytes per pixel or
// block, block size, colour target format. From the sdk's clb097tex.h / cl9097.h and NIL's format table; BGRA is
// the RGBA layout with R and B swapped.
struct format {
    uint8_t comps, type, type_gba, swizzle[4], srgb, bytes, block, target;
};

#define F(comps, type, x, y, z, w, srgb, bytes, target) { comps, type, 0, { x, y, z, w }, srgb, bytes, 1, target }
#define BC(comps, type, x, y, z, w, srgb, bytes) { comps, type, 0, { x, y, z, w }, srgb, bytes, 4, 0 }

static const struct format formats[TEX_FORMATS] = {
    [TEX_RGBA8_UNORM] = F(0x08, UNORM, IR, IG, IB, IA, 0, 4, 0xd5),
    [TEX_BGRA8_UNORM] = F(0x08, UNORM, IB, IG, IR, IA, 0, 4, 0xcf),
    [TEX_R32_FLOAT] = F(0x0f, FLOAT, IR, I0, I0, I1, 0, 4, 0xe5),
    [TEX_RGBA16_FLOAT] = F(0x03, FLOAT, IR, IG, IB, IA, 0, 8, 0xca),
    [TEX_RGBA32_FLOAT] = F(0x01, FLOAT, IR, IG, IB, IA, 0, 16, 0xc0),
    [TEX_R8_UNORM] = F(0x1d, UNORM, IR, I0, I0, I1, 0, 1, 0xf3),
    [TEX_RG8_UNORM] = F(0x18, UNORM, IR, IG, I0, I1, 0, 2, 0xea),
    [TEX_A8_UNORM] = F(0x1d, UNORM, I0, I0, I0, IR, 0, 1, 0xf7),
    [TEX_RGBA8_SRGB] = F(0x08, UNORM, IR, IG, IB, IA, 1, 4, 0xd6),
    [TEX_BGRA8_SRGB] = F(0x08, UNORM, IB, IG, IR, IA, 1, 4, 0xd0),
    [TEX_RGB10A2_UNORM] = F(0x09, UNORM, IR, IG, IB, IA, 0, 4, 0xd1),
    [TEX_BGR10A2_UNORM] = F(0x09, UNORM, IB, IG, IR, IA, 0, 4, 0xdf),
    [TEX_R16_UNORM] = F(0x1b, UNORM, IR, I0, I0, I1, 0, 2, 0xee),
    [TEX_RGBA16_UNORM] = F(0x03, UNORM, IR, IG, IB, IA, 0, 8, 0xc6),
    [TEX_R16_FLOAT] = F(0x1b, FLOAT, IR, I0, I0, I1, 0, 2, 0xf2),
    [TEX_RG16_FLOAT] = F(0x0c, FLOAT, IR, IG, I0, I1, 0, 4, 0xde),
    [TEX_RG32_FLOAT] = F(0x04, FLOAT, IR, IG, I0, I1, 0, 8, 0xcb),
    [TEX_RG16_UNORM] = F(0x0c, UNORM, IR, IG, I0, I1, 0, 4, 0xda),
    [TEX_R8_SNORM] = F(0x1d, SNORM, IR, I0, I0, I1, 0, 1, 0xf4),
    [TEX_R8_UINT] = F(0x1d, UINT, IR, I0, I0, I1, 0, 1, 0xf6),
    [TEX_R8_SINT] = F(0x1d, SINT, IR, I0, I0, I1, 0, 1, 0xf5),
    [TEX_RG8_SNORM] = F(0x18, SNORM, IR, IG, I0, I1, 0, 2, 0xeb),
    [TEX_RG8_UINT] = F(0x18, UINT, IR, IG, I0, I1, 0, 2, 0xed),
    [TEX_RG8_SINT] = F(0x18, SINT, IR, IG, I0, I1, 0, 2, 0xec),
    [TEX_RGBA8_SNORM] = F(0x08, SNORM, IR, IG, IB, IA, 0, 4, 0xd7),
    [TEX_RGBA8_UINT] = F(0x08, UINT, IR, IG, IB, IA, 0, 4, 0xd9),
    [TEX_RGBA8_SINT] = F(0x08, SINT, IR, IG, IB, IA, 0, 4, 0xd8),
    [TEX_R16_SNORM] = F(0x1b, SNORM, IR, I0, I0, I1, 0, 2, 0xef),
    [TEX_R16_UINT] = F(0x1b, UINT, IR, I0, I0, I1, 0, 2, 0xf1),
    [TEX_R16_SINT] = F(0x1b, SINT, IR, I0, I0, I1, 0, 2, 0xf0),
    [TEX_RG16_SNORM] = F(0x0c, SNORM, IR, IG, I0, I1, 0, 4, 0xdb),
    [TEX_RG16_UINT] = F(0x0c, UINT, IR, IG, I0, I1, 0, 4, 0xdd),
    [TEX_RG16_SINT] = F(0x0c, SINT, IR, IG, I0, I1, 0, 4, 0xdc),
    [TEX_RGBA16_SNORM] = F(0x03, SNORM, IR, IG, IB, IA, 0, 8, 0xc7),
    [TEX_RGBA16_UINT] = F(0x03, UINT, IR, IG, IB, IA, 0, 8, 0xc9),
    [TEX_RGBA16_SINT] = F(0x03, SINT, IR, IG, IB, IA, 0, 8, 0xc8),
    [TEX_R32_UINT] = F(0x0f, UINT, IR, I0, I0, I1, 0, 4, 0xe4),
    [TEX_R32_SINT] = F(0x0f, SINT, IR, I0, I0, I1, 0, 4, 0xe3),
    [TEX_RG32_UINT] = F(0x04, UINT, IR, IG, I0, I1, 0, 8, 0xcd),
    [TEX_RG32_SINT] = F(0x04, SINT, IR, IG, I0, I1, 0, 8, 0xcc),
    [TEX_RGBA32_UINT] = F(0x01, UINT, IR, IG, IB, IA, 0, 16, 0xc2),
    [TEX_RGBA32_SINT] = F(0x01, SINT, IR, IG, IB, IA, 0, 16, 0xc1),
    [TEX_RGB10A2_UINT] = F(0x09, UINT, IR, IG, IB, IA, 0, 4, 0xd2),
    [TEX_RG11B10_FLOAT] = F(0x21, FLOAT, IR, IG, IB, I1, 0, 4, 0xe0),
    [TEX_RGB9E5_FLOAT] = F(0x20, FLOAT, IR, IG, IB, I1, 0, 4, 0),
    [TEX_BC1] = BC(0x24, UNORM, IR, IG, IB, IA, 0, 8),
    [TEX_BC1_SRGB] = BC(0x24, UNORM, IR, IG, IB, IA, 1, 8),
    [TEX_BC2] = BC(0x25, UNORM, IR, IG, IB, IA, 0, 16),
    [TEX_BC2_SRGB] = BC(0x25, UNORM, IR, IG, IB, IA, 1, 16),
    [TEX_BC3] = BC(0x26, UNORM, IR, IG, IB, IA, 0, 16),
    [TEX_BC3_SRGB] = BC(0x26, UNORM, IR, IG, IB, IA, 1, 16),
    [TEX_BC4_UNORM] = BC(0x27, UNORM, IR, I0, I0, I1, 0, 8),
    [TEX_BC4_SNORM] = BC(0x27, SNORM, IR, I0, I0, I1, 0, 8),
    [TEX_BC5_UNORM] = BC(0x28, UNORM, IR, IG, I0, I1, 0, 16),
    [TEX_BC5_SNORM] = BC(0x28, SNORM, IR, IG, I0, I1, 0, 16),
    [TEX_BC6H_FLOAT] = BC(0x10, FLOAT, IR, IG, IB, I1, 0, 16),
    [TEX_BC6H_UFLOAT] = BC(0x11, FLOAT, IR, IG, IB, I1, 0, 16),
    [TEX_BC7] = BC(0x17, UNORM, IR, IG, IB, IA, 0, 16),
    [TEX_BC7_SRGB] = BC(0x17, UNORM, IR, IG, IB, IA, 1, 16),
    [TEX_Z16] = F(0x3a, UNORM, IR, I0, I0, I1, 0, 2, 0),
    [TEX_Z32_FLOAT] = F(0x2f, FLOAT, IR, I0, I0, I1, 0, 4, 0),
    [TEX_Z32_FLOAT_S8] = { 0x30, FLOAT, UINT, { IR, I0, I0, I1 }, 0, 8, 1, 0 },
    [TEX_S8_OF_Z32] = F(0x05, UINT, IG, I0, I0, I1, 0, 8, 0),  // X32_S8X24: the stencil is G
};

static int is_integer(const struct format *f)
{
    return f->type == SINT || f->type == UINT;
}

struct tex_format_info tex_format_info(enum tex_format f)
{
    const struct format *x = &formats[f];
    return (struct tex_format_info){ x->bytes, x->block, x->target, (uint8_t)is_integer(x) };
}

void tex_fill_header(uint32_t out[8], const struct tex_desc *t)
{
    const struct format *f = &formats[t->format];
    memset(out, 0, 32);
    setf(out, 0, 6, f->comps);
    for (unsigned c = 0; c < 4; c++) {
        uint8_t type = c && f->type_gba ? f->type_gba : f->type;
        uint8_t want = t->swizzled ? t->swizzle[c] : (uint8_t)(2 + c);
        uint8_t from = want == 0 ? I0 : want == 1 ? I1 : f->swizzle[(want - 2) & 3];
        uint8_t src = from == I1 && is_integer(f) ? I1_INT : from;
        setf(out, 7 + 3 * c, 9 + 3 * c, type);  // R, G, B, A data type
        setf(out, 19 + 3 * c, 21 + 3 * c, src);  // X, Y, Z, W source
    }
    setf(out, 150, 150, f->srgb);  // S_R_G_B_CONVERSION
    if (t->blocklinear) {  // TEXHEAD_BL: blocks one GOB wide, high and deep (the fields' 0)
        static const uint8_t types[] = {
            [TEX_TYPE_2D] = 1, [TEX_TYPE_3D] = 2, [TEX_TYPE_CUBE] = 3, [TEX_TYPE_2D_ARRAY] = 5,
            [TEX_TYPE_CUBE_ARRAY] = 8,
        };
        uint32_t max = t->levels ? t->levels - 1 : 0, depth = t->depth ? t->depth - 1 : 0;
        setf(out, 41, 63, (uint32_t)(t->addr >> 9));     // ADDRESS_BITS31TO9
        setf(out, 85, 87, 3);                            // HEADER_VERSION = SELECT_BLOCKLINEAR
        setf(out, 112, 114, 7);                          // LOD_ANISO_QUALITY2, LOD_(AN)ISO_QUALITY high, as NVK
        setf(out, 124, 127, max);                        // MAX_MIP_LEVEL
        setf(out, 145, 145, depth >> 14);                // DEPTH_MINUS_ONE_BIT14
        setf(out, 151, 154, types[t->type]);             // TEXTURE_TYPE
        setf(out, 176, 189, depth);                      // DEPTH_MINUS_ONE
        setf(out, 215, 216, 2);                          // ANISO_FINE_SPREAD_FUNC = TWO
        setf(out, 217, 218, 1);                          // ANISO_COARSE_SPREAD_FUNC = ONE
        // MULTI_SAMPLE_COUNT: MODE_2X1, 2X2, 4X2_D3D, as the 3D class rendered them
        setf(out, 232, 235, t->samples >= 8 ? 4 : t->samples >= 4 ? 2 : t->samples >= 2 ? 1 : 0);
        setf(out, 224, 227, t->view_min);                // RES_VIEW_MIN_MIP_LEVEL
        setf(out, 228, 231, t->view_levels ? t->view_min + t->view_levels - 1 : max);  // RES_VIEW_MAX_MIP_LEVEL
    } else {
        setf(out, 37, 63, (uint32_t)(t->addr >> 5));     // ADDRESS_BITS31TO5
        setf(out, 85, 87, 2);                            // HEADER_VERSION = SELECT_PITCH
        setf(out, 96, 111, t->pitch >> 5);               // PITCH_BITS20TO5
        setf(out, 151, 154, 7);                          // TEXTURE_TYPE = TWO_D_NO_MIPMAP
    }
    setf(out, 64, 79, (uint32_t)(t->addr >> 32));        // ADDRESS_BITS47TO32
    setf(out, 128, 143, t->width - 1);                   // WIDTH_MINUS_ONE
    setf(out, 146, 146, (t->height - 1) >> 16);          // HEIGHT_MINUS_ONE_BIT16
    setf(out, 157, 159, 7);                              // BORDER_SIZE = BORDER_SAMPLER_COLOR
    setf(out, 160, 175, t->height - 1);                  // HEIGHT_MINUS_ONE
    setf(out, 191, 191, 1);                              // NORMALIZED_COORDS
}

static uint32_t getf(const uint32_t in[8], unsigned lo, unsigned hi)
{
    uint32_t v = 0;
    for (unsigned b = lo; b <= hi; b++)
        v |= (in[b / 32] >> (b % 32) & 1) << (b - lo);
    return v;
}

void tex_header_faces_as_layers(uint32_t hdr[8])
{
    uint32_t type = getf(hdr, 151, 154);
    if (getf(hdr, 85, 87) != 3 || (type != 3 && type != 8))  // a block-linear CUBEMAP or CUBE_ARRAY
        return;
    uint32_t layers = ((getf(hdr, 145, 145) << 14 | getf(hdr, 176, 189)) + 1) * 6 - 1;
    hdr[4] &= ~(0xfu << 23);  // TEXTURE_TYPE (151..154)
    hdr[4] &= ~(1u << 17);    // DEPTH_MINUS_ONE_BIT14 (145)
    hdr[5] &= ~(0x3fffu << 16);  // DEPTH_MINUS_ONE (176..189)
    setf(hdr, 151, 154, 5);   // TWO_D_ARRAY
    setf(hdr, 145, 145, layers >> 14);
    setf(hdr, 176, 189, layers);
}

void tex_fill_sampler(uint32_t out[8], const struct samp_desc *s)
{
    // MAX_ANISOTROPY for 1..16 samples, rounded down to the ratios it has (1, 2, 4, 6, 8, 10, 12, 16)
    static const uint8_t aniso[17] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7 };
    unsigned a = s->max_anisotropy > 16 ? 16 : s->max_anisotropy;
    memset(out, 0, 32);
    for (unsigned c = 0; c < 3; c++)
        setf(out, 3 * c, 3 * c + 2, s->address[c]);  // ADDRESS_U, _V, _P: the enum follows TEXSAMP's
    if (s->compare) {
        setf(out, 9, 9, 1);                          // DEPTH_COMPARE
        setf(out, 10, 12, (uint32_t)s->compare - 1);        // DEPTH_COMPARE_FUNC, in MTLCompareFunction's order
    }
    setf(out, 13, 13, 1);                            // S_R_G_B_CONVERSION, as NVK sets it
    setf(out, 14, 16, 1);                            // FONT_FILTER_WIDTH = SIZE_2
    setf(out, 17, 19, 1);                            // FONT_FILTER_HEIGHT = SIZE_2
    if (a > 1)
        setf(out, 20, 22, aniso[a]);                 // MAX_ANISOTROPY
    setf(out, 32, 34, s->mag_linear ? 2 : 1);        // TEXSAMP1 MAG_FILTER: LINEAR, POINT
    setf(out, 36, 37, !s->min_linear ? 1 : a > 1 ? 3 : 2);  // MIN_FILTER: POINT, ANISO, LINEAR
    setf(out, 38, 39, s->mip + 1);                   // MIP_FILTER: NONE, POINT, LINEAR
    setf(out, 40, 41, 2);                            // CUBEMAP_INTERFACE_FILTERING = AUTO_SPAN_SEAM (seamless)
    // Metal rounds a halfway LOD to the lower level, the hardware to the upper: MIP_LOD_BIAS -1/256 (signed 5.8)
    if (s->mip == TEX_MIP_NEAREST)
        setf(out, 44, 56, 0x1fff);
    if (!s->normalized)
        setf(out, 57, 57, 1);                        // FLOAT_COORD_NORMALIZATION = FORCE_UNNORMALIZED
    setf(out, 64, 75, s->lod_min > 0xfff ? 0xfff : s->lod_min);  // TEXSAMP2 MIN_LOD_CLAMP, 4.8
    setf(out, 76, 87, s->lod_max > 0xfff ? 0xfff : s->lod_max);  // MAX_LOD_CLAMP
    if (s->border != TEX_BORDER_TRANSPARENT_BLACK) {
        uint32_t rgb = s->border == TEX_BORDER_OPAQUE_WHITE ? 0x3f800000 : 0, srgb = rgb ? 0xff : 0;
        out[4] = out[5] = out[6] = rgb;              // BORDER_COLOR_R, G, B
        out[7] = 0x3f800000;                         // BORDER_COLOR_A
        setf(out, 88, 95, srgb);                     // S_R_G_B_BORDER_COLOR_R, G, B
        setf(out, 108, 115, srgb);
        setf(out, 116, 123, srgb);
    }
}
