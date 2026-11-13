// Texture headers (TIC) and samplers (TSC) for the Metal plugin, from MTLTextureDescriptor/MTLSamplerDescriptor.
// Pitch-linear 2D, or block-linear (one-GOB blocks) 2D/3D/cube/array with mips. Maxwell TEXHEAD/TEXSAMP layouts,
// used through Ampere.
#pragma once
#include <stdint.h>

enum tex_format {
    TEX_RGBA8_UNORM,
    TEX_BGRA8_UNORM,
    TEX_R32_FLOAT,
    TEX_RGBA16_FLOAT,
    TEX_RGBA32_FLOAT,
    TEX_R8_UNORM,
    TEX_RG8_UNORM,
    TEX_A8_UNORM,
    TEX_RGBA8_SRGB,
    TEX_BGRA8_SRGB,
    TEX_RGB10A2_UNORM,
    TEX_BGR10A2_UNORM,
    TEX_R16_UNORM,
    TEX_RGBA16_UNORM,
    TEX_R16_FLOAT,
    TEX_RG16_FLOAT,
    TEX_RG32_FLOAT,
    TEX_RG16_UNORM,
    TEX_R8_SNORM,
    TEX_R8_UINT,
    TEX_R8_SINT,
    TEX_RG8_SNORM,
    TEX_RG8_UINT,
    TEX_RG8_SINT,
    TEX_RGBA8_SNORM,
    TEX_RGBA8_UINT,
    TEX_RGBA8_SINT,
    TEX_R16_SNORM,
    TEX_R16_UINT,
    TEX_R16_SINT,
    TEX_RG16_SNORM,
    TEX_RG16_UINT,
    TEX_RG16_SINT,
    TEX_RGBA16_SNORM,
    TEX_RGBA16_UINT,
    TEX_RGBA16_SINT,
    TEX_R32_UINT,
    TEX_R32_SINT,
    TEX_RG32_UINT,
    TEX_RG32_SINT,
    TEX_RGBA32_UINT,
    TEX_RGBA32_SINT,
    TEX_RGB10A2_UINT,
    TEX_RG11B10_FLOAT,
    TEX_RGB9E5_FLOAT,
    TEX_BC1,
    TEX_BC1_SRGB,
    TEX_BC2,
    TEX_BC2_SRGB,
    TEX_BC3,
    TEX_BC3_SRGB,
    TEX_BC4_UNORM,
    TEX_BC4_SNORM,
    TEX_BC5_UNORM,
    TEX_BC5_SNORM,
    TEX_BC6H_FLOAT,
    TEX_BC6H_UFLOAT,
    TEX_BC7,
    TEX_BC7_SRGB,
    // depth and stencil textures sampled: the depth of a Z16, ZF32 or ZF32_X24S8 target, the stencil of the last
    TEX_Z16,
    TEX_Z32_FLOAT,
    TEX_Z32_FLOAT_S8,
    TEX_S8_OF_Z32,
    TEX_FORMATS
};

// bytes per pixel (per 4x4 block for a compressed format), the texels a block is wide and high (1, or 4), and the
// SET_COLOR_TARGET_FORMAT value to render to the format (0 for one the 3D class can't render to)
struct tex_format_info {
    uint8_t bytes;
    uint8_t block;
    uint8_t target;
    uint8_t integer;  // read and written as integers (blending doesn't apply)
};

enum tex_address {
    TEX_WRAP,  // repeat
    TEX_MIRROR,
    TEX_CLAMP_TO_EDGE,
    TEX_BORDER,  // the sampler's border colour outside
    TEX_CLAMP_OGL,
    TEX_MIRROR_ONCE_CLAMP_TO_EDGE,
};

enum tex_type { TEX_TYPE_2D, TEX_TYPE_3D, TEX_TYPE_CUBE, TEX_TYPE_2D_ARRAY, TEX_TYPE_CUBE_ARRAY };

struct tex_desc {
    uint64_t addr;        // GPU VA of the pixels, 32-byte aligned (512 for block-linear)
    uint32_t width, height;
    uint32_t pitch;       // row stride in bytes, 32-byte aligned; unused for block-linear
    enum tex_format format;
    int blocklinear;      // one-GOB blocks
    // block-linear only: a 3D texture's slices, an array's layers (a cube array's cubes), 0 for one
    uint32_t depth;
    enum tex_type type;
    uint32_t levels;      // mip levels, 0 for one
    uint32_t view_min, view_levels;  // the levels a view sees: the first, how many (0 for all from it)
    uint32_t samples;     // per pixel (block-linear only), 0 or 1 for one; the memory holds a grid of them a pixel
    // with swizzled: MTLTextureSwizzle values (zero, one, the format's red..alpha) the shader reads as x, y, z, w
    int swizzled;
    uint8_t swizzle[4];
};

// as MTLSamplerMipFilter and MTLSamplerBorderColor
enum tex_mip { TEX_MIP_NONE, TEX_MIP_NEAREST, TEX_MIP_LINEAR };
enum tex_border { TEX_BORDER_TRANSPARENT_BLACK, TEX_BORDER_OPAQUE_BLACK, TEX_BORDER_OPAQUE_WHITE };

struct samp_desc {
    int mag_linear, min_linear;  // nearest when 0
    enum tex_mip mip;
    enum tex_address address[3];  // u, v, p
    int normalized;               // Metal's normalized coordinates
    unsigned max_anisotropy;      // 1 (or 0) for none
    uint16_t lod_min, lod_max;    // LOD clamps, in 1/256 levels
    int compare;                  // a depth compare's MTLCompareFunction + 1, 0 for none
    enum tex_border border;
};

#ifdef __cplusplus
extern "C" {
#endif

struct tex_format_info tex_format_info(enum tex_format f);

void tex_fill_header(uint32_t out[8], const struct tex_desc *t);
void tex_fill_sampler(uint32_t out[8], const struct samp_desc *s);
// turns a cube (array) header into the 2D array of its faces, as texel reads/writes address them; others unchanged
void tex_header_faces_as_layers(uint32_t hdr[8]);

#ifdef __cplusplus
}
#endif
