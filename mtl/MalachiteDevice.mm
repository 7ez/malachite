// Metal.framework loads this bundle by the accelerator's MetalPluginName and requires its principal class to subclass
// the private MTLIOAccelDevice, which the loader binds from the shared cache.
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#import <dirent.h>
#import <dlfcn.h>
#import <fcntl.h>
#import <IOKit/IOKitLib.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <objc/objc-exception.h>
#import <os/lock.h>
#import <os/log.h>
#import <sys/stat.h>
#import <sys/sysctl.h>
#import <unistd.h>
#import <mach/vm_page_size.h>
#import <mach-o/dyld.h>
#import <mach/mach.h>
#import <mach-o/loader.h>
#import <mach-o/nlist.h>
#import <string.h>
#include <map>
#include <vector>
#import "../accel/commands.h"
#import "../compiler/airc.h"
extern "C" {
#import "../compiler/llvm_lite.h"
}
#import "../core/drawbuild.h"
#import "../core/texdesc.h"
#import "../gpu_bridge.h"

// Declared on NSObject for compiling only (-undefined dynamic_lookup); the non-fragile runtime lays our class out
// past the real one's size.
@interface MTLIOAccelDevice : NSObject
- (instancetype)initWithAcceleratorPort:(unsigned int)port;
@end

// its init opens the kernel's command queue user client (type 8) and wires up submission
@interface MTLIOAccelCommandQueue : NSObject
- (instancetype)initWithDevice:(id)device descriptor:(id)descriptor;
- (NSUInteger)maxCommandBufferCount;
@end

@interface MTLIOAccelCommandBuffer : NSObject
- (instancetype)initWithQueue:(id)queue retainedReferences:(bool)retained;
// bump allocator over the kernel command buffer; writes no header, the {opcode, len} is ours
- (void *)_reserveKernelCommandBufferSpace:(uint64_t)nbytes;
// The kernel walks segments, not the cursor, so a record outside beginSegment:/endCurrentSegment is never run.
- (void)beginSegment:(void *)ptr;
- (void)endCurrentSegment;
- (void)commit;
- (void)commitAndWaitUntilSubmitted;
- (void)presentDrawable:(id<MTLDrawable>)drawable;
- (void)presentDrawable:(id<MTLDrawable>)drawable atTime:(CFTimeInterval)time;
- (void)presentDrawable:(id<MTLDrawable>)drawable afterMinimumDuration:(CFTimeInterval)duration;
- (void)getCurrentKernelCommandBufferPointer:(void **)cur end:(void **)end;
- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value;
- (void)encodeWaitForEvent:(id<MTLEvent>)event value:(uint64_t)value;
@end

@interface MalachiteCommandBuffer : MTLIOAccelCommandBuffer <MTLCommandBuffer> {
  @public
    bool _in_segment;  // draws keep adding to the open segment
    // draws are written here, not by the kernel: methods in one block, their constants in another
    uint32_t *_run_cpu;
    uint64_t _run_va;
    uint32_t _run_start, _run_used, _run_cap, _run_draws;  // in words but the last
    uint8_t *_bank_cpu;
    uint64_t _bank_va;
    uint32_t _bank_used, _bank_cap;
    bool _bank_new;  // reused memory, the constant cache may hold it
    uint64_t _records;     // tells whether a draw's state is still the channel's
    const void *_state_of;  // the encoder whose last draw the channel holds
    // members, not associated objects: those retain and autorelease on every look, several times a draw
    NSMutableArray *_kept;
    const void *_kept_recent[4];  // so draws of the same few resources don't grow the array
    unsigned _kept_next;
    id _uploads, _heap_blocks;
    NSMutableSet *_writeback;  // mirrored buffers OpenCL kernels may write, copied back to system memory at commit
    bool _writes_capture;  // to a surface WindowServer hands to another process (see commitAndWaitUntilSubmitted)
}
@end

// The designated init fills the args block's IOAccelNewResourceData [0x00..0x67] from its params and creates the
// kernel resource.
@interface MTLIOAccelBuffer : NSObject
- (instancetype)initWithDevice:(id)device
                       pointer:(void *)ptr
                        length:(NSUInteger)length
                       options:(NSUInteger)options
                    sysMemSize:(NSUInteger)sysMemSize
                    vidMemSize:(NSUInteger)vidMemSize
                          args:(void *)args
                      argsSize:(unsigned int)argsSize
                   deallocator:(id)dealloc;
- (uint64_t)gpuAddress;  // MTLIOAccelResource's _res+0x48
- (void *)contents;
- (void)didModifyRange:(NSRange)range;
@end

// Every encode method of the base is a doesNotRecognize stub, so never call super for one.
@interface MTLIOAccelBlitCommandEncoder : NSObject
- (instancetype)initWithCommandBuffer:(id)cb;
@end

// The base only makes pitch-linear 2D textures over a buffer.
@interface MTLIOAccelTexture : NSObject
- (instancetype)initWithBuffer:(id)buffer descriptor:(id)descriptor offset:(NSUInteger)offset
                   bytesPerRow:(NSUInteger)bytesPerRow;
- (id)buffer;
- (NSUInteger)bufferOffset;
- (NSUInteger)bufferBytesPerRow;
- (NSUInteger)allocatedSize;
- (NSUInteger)height;
- (NSUInteger)width;
- (MTLPixelFormat)pixelFormat;
@end

// A 1D texture is a 2D one a row high, a 1D array a row per layer; the getters answer as _type.
// Anything else pitch-linear can't be is an image: block-linear in one-GOB blocks, levels in a row per layer, layers
// _layer_stride apart, as NIL lays them out.
@class MalachiteHeapBuffer;

@interface MalachiteTexture : MTLIOAccelTexture <MTLTexture> {
@public
    __unsafe_unretained id _backing;
    uint64_t _offset;
    MTLTextureUsage _usage;  // the base reports none, and CoreImage then won't write to it
    uint32_t _pitch;  // an image's: level 0's row of GOBs
    id _owned;  // the backing buffer, when the device made it for the texture
    MalachiteHeapBuffer *_placed;  // keeps the MTLHeap's memory
    MTLTextureType _type;
    IOSurfaceRef _surface;
    bool _capture;  // the surface is a screen or window capture
    NSUInteger _plane;
    uint32_t _heap_entry;  // made once asked for its gpuResourceID
    // when the base was made a row high (an IOSurface's rows the base won't take)
    NSUInteger _linear_height;
    // rows not on 32 bytes, which the texture unit can't read, are sampled from a copy made before each command
    id<MTLBuffer> _shadow;
    uint32_t _zt_format, _zt_bpp;  // 0 for a colour texture
    uint8_t _zt_kind;              // the PTE kind its memory needs
    bool _swizzled;
    uint8_t _swizzle[4];
    // a view sees _levels from _view_min, and _offset is its first layer's
    bool _image;
    MTLPixelFormat _format;
    NSUInteger _width, _height;
    uint32_t _depth;
    uint32_t _samples;  // in a grid per pixel (see malachite_grid); 0 when not multisampled
    uint32_t _layers;
    uint32_t _levels, _image_levels, _view_min;
    uint64_t _layer_stride;
    // when the device put the texture somewhere else (see malachite_storage)
    MTLStorageMode _asked;
    bool _moved;
}
- (void)copyRegion:(MTLRegion)region level:(NSUInteger)level slice:(NSUInteger)slice bytes:(uint8_t *)bytes
            stride:(NSUInteger)stride image:(NSUInteger)image in:(bool)in;
@end

// MALACHITE_TRACE, or /var/tmp/malachite-trace naming the process for apps launched for you
static bool malachite_trace;
static bool malachite_dump;  // "name +dump": AIR saved without the trace's logging
static bool malachite_snap;  // attachment snapshots and buffer dumps, which copy on the GPU

// Another user could plant a control file to make WindowServer write where they like or compile their bitcode.
// own_ok lets the replay tools write overrides for themselves.
static bool malachite_trusted_file(const char *path, bool own_ok)
{
    struct stat st;
    return path && !lstat(path, &st) && S_ISREG(st.st_mode) && !(st.st_mode & (S_IWGRP | S_IWOTH)) &&
           (st.st_uid == 0 || (own_ok && st.st_uid == getuid()));
}
static NSArray<NSString *> *malachite_snap_only;  // "+snap=A|B": only draws whose label has A or B in it
static bool malachite_snap_gate;  // "+snapgate": only while /var/tmp/malachite-snap exists, checked once a pass
// "+snaprect=x,y,w,h": only the rect, of every draw whose scissor reaches into it
static uint32_t malachite_snap_rect[4];
// "+record=A|B": matching draws recorded whole into /tmp/malachite-rec-<pid>-<n>/ for tools/mtl/mtlplay, and the
// buffers of each matching kernel's first dispatch ("+recskip=N": the one after its first N)
static NSArray<NSString *> *malachite_rec_only;
static int malachite_rec_skip;
static bool malachite_rec_bytes(id cb, NSString *path, uint64_t src, uint32_t size);
static bool malachite_rec_shut;
static bool malachite_rec_full;    // the process has written the recorder's budget
static bool malachite_snap_gated;  // the gate opens the snapshots too, not only the recorder
static char malachite_desc_key;    // the descriptor a pipeline, sampler or depth/stencil state was made from
static char malachite_air_key;     // the AIR a function was compiled from, with its linked functions and their names

static int malachite_format(MTLPixelFormat f)
{
    switch (f) {
    case MTLPixelFormatRGBA8Unorm: return TEX_RGBA8_UNORM;
    case MTLPixelFormatBGRA8Unorm: return TEX_BGRA8_UNORM;
    case MTLPixelFormatRGBA8Unorm_sRGB: return TEX_RGBA8_SRGB;
    case MTLPixelFormatBGRA8Unorm_sRGB: return TEX_BGRA8_SRGB;
    case MTLPixelFormatR8Unorm: return TEX_R8_UNORM;
    case MTLPixelFormatRG8Unorm: return TEX_RG8_UNORM;
    case MTLPixelFormatA8Unorm: return TEX_A8_UNORM;
    case MTLPixelFormatRGB10A2Unorm: return TEX_RGB10A2_UNORM;
    case MTLPixelFormatBGR10A2Unorm: return TEX_BGR10A2_UNORM;
    case MTLPixelFormatR16Unorm: return TEX_R16_UNORM;
    case MTLPixelFormatRG16Unorm: return TEX_RG16_UNORM;
    case MTLPixelFormatRGBA16Unorm: return TEX_RGBA16_UNORM;
    case MTLPixelFormatR16Float: return TEX_R16_FLOAT;
    case MTLPixelFormatRG16Float: return TEX_RG16_FLOAT;
    case MTLPixelFormatRGBA16Float: return TEX_RGBA16_FLOAT;
    // CoreAnimation's extended-range video and wide-colour layers, in the nearest layouts the GPU has (their values
    // come out approximate, where an unsupported format was a nil texture and an abort)
    case MTLPixelFormatBGRA10_XR: case MTLPixelFormatBGRA10_XR_sRGB: return TEX_RGBA16_FLOAT;
    case MTLPixelFormatBGR10_XR: case MTLPixelFormatBGR10_XR_sRGB: return TEX_BGR10A2_UNORM;
    case MTLPixelFormatR8Unorm_sRGB: return TEX_R8_UNORM;
    case MTLPixelFormatRG8Unorm_sRGB: return TEX_RG8_UNORM;
    case MTLPixelFormatR32Float: return TEX_R32_FLOAT;
    case MTLPixelFormatRG32Float: return TEX_RG32_FLOAT;
    case MTLPixelFormatRGBA32Float: return TEX_RGBA32_FLOAT;
    case MTLPixelFormatR8Snorm: return TEX_R8_SNORM;
    case MTLPixelFormatR8Uint: return TEX_R8_UINT;
    case MTLPixelFormatR8Sint: return TEX_R8_SINT;
    case MTLPixelFormatRG8Snorm: return TEX_RG8_SNORM;
    case MTLPixelFormatRG8Uint: return TEX_RG8_UINT;
    case MTLPixelFormatRG8Sint: return TEX_RG8_SINT;
    case MTLPixelFormatRGBA8Snorm: return TEX_RGBA8_SNORM;
    case MTLPixelFormatRGBA8Uint: return TEX_RGBA8_UINT;
    case MTLPixelFormatRGBA8Sint: return TEX_RGBA8_SINT;
    case MTLPixelFormatR16Snorm: return TEX_R16_SNORM;
    case MTLPixelFormatR16Uint: return TEX_R16_UINT;
    case MTLPixelFormatR16Sint: return TEX_R16_SINT;
    case MTLPixelFormatRG16Snorm: return TEX_RG16_SNORM;
    case MTLPixelFormatRG16Uint: return TEX_RG16_UINT;
    case MTLPixelFormatRG16Sint: return TEX_RG16_SINT;
    case MTLPixelFormatRGBA16Snorm: return TEX_RGBA16_SNORM;
    case MTLPixelFormatRGBA16Uint: return TEX_RGBA16_UINT;
    case MTLPixelFormatRGBA16Sint: return TEX_RGBA16_SINT;
    case MTLPixelFormatR32Uint: return TEX_R32_UINT;
    case MTLPixelFormatR32Sint: return TEX_R32_SINT;
    case MTLPixelFormatRG32Uint: return TEX_RG32_UINT;
    case MTLPixelFormatRG32Sint: return TEX_RG32_SINT;
    case MTLPixelFormatRGBA32Uint: return TEX_RGBA32_UINT;
    case MTLPixelFormatRGBA32Sint: return TEX_RGBA32_SINT;
    case MTLPixelFormatRGB10A2Uint: return TEX_RGB10A2_UINT;
    case MTLPixelFormatRG11B10Float: return TEX_RG11B10_FLOAT;
    case MTLPixelFormatRGB9E5Float: return TEX_RGB9E5_FLOAT;
    case MTLPixelFormatBC1_RGBA: return TEX_BC1;
    case MTLPixelFormatBC1_RGBA_sRGB: return TEX_BC1_SRGB;
    case MTLPixelFormatBC2_RGBA: return TEX_BC2;
    case MTLPixelFormatBC2_RGBA_sRGB: return TEX_BC2_SRGB;
    case MTLPixelFormatBC3_RGBA: return TEX_BC3;
    case MTLPixelFormatBC3_RGBA_sRGB: return TEX_BC3_SRGB;
    case MTLPixelFormatBC4_RUnorm: return TEX_BC4_UNORM;
    case MTLPixelFormatBC4_RSnorm: return TEX_BC4_SNORM;
    case MTLPixelFormatBC5_RGUnorm: return TEX_BC5_UNORM;
    case MTLPixelFormatBC5_RGSnorm: return TEX_BC5_SNORM;
    case MTLPixelFormatBC6H_RGBFloat: return TEX_BC6H_FLOAT;
    case MTLPixelFormatBC6H_RGBUfloat: return TEX_BC6H_UFLOAT;
    case MTLPixelFormatBC7_RGBAUnorm: return TEX_BC7;
    case MTLPixelFormatBC7_RGBAUnorm_sRGB: return TEX_BC7_SRGB;
    // sampled depth/stencil targets (layouts in malachite_zeta)
    case MTLPixelFormatDepth16Unorm: return TEX_Z16;
    case MTLPixelFormatDepth32Float: return TEX_Z32_FLOAT;
    case MTLPixelFormatDepth24Unorm_Stencil8:
    case MTLPixelFormatDepth32Float_Stencil8: return TEX_Z32_FLOAT_S8;
    case MTLPixelFormatStencil8:
    case MTLPixelFormatX24_Stencil8:
    case MTLPixelFormatX32_Stencil8: return TEX_S8_OF_Z32;
    default: return -1;
    }
}

// Turing's PTE kinds; stencil-only and 24-bit depth take the ZF32_X24S8 layout.
static uint32_t malachite_zeta(MTLPixelFormat f, uint32_t *bpp, uint8_t *kind)
{
    enum { ZF32 = 0x0a, Z16 = 0x13, ZF32_X24S8 = 0x19 };
    enum { KIND_Z16 = 0x01, KIND_ZF32_X24S8 = 0x04, KIND_GENERIC = 0x06 };
    switch (f) {
    case MTLPixelFormatDepth16Unorm: *bpp = 2, *kind = KIND_Z16; return Z16;
    case MTLPixelFormatDepth32Float: *bpp = 4, *kind = KIND_GENERIC; return ZF32;
    case MTLPixelFormatStencil8:
    case MTLPixelFormatDepth24Unorm_Stencil8:
    case MTLPixelFormatDepth32Float_Stencil8: *bpp = 8, *kind = KIND_ZF32_X24S8; return ZF32_X24S8;
    default: return 0;
    }
}

// per 4x4 block for a compressed format
static NSUInteger malachite_bytes_per_pixel(MTLPixelFormat f)
{
    int t = malachite_format(f);
    return t < 0 ? 0 : tex_format_info((enum tex_format)t).bytes;
}

static uint32_t malachite_block(MTLPixelFormat f)
{
    int t = malachite_format(f);
    return t < 0 ? 1 : tex_format_info((enum tex_format)t).block;
}

// sizes in blocks; a level past the end of the chain gives the layer's size
struct malachite_lvl {
    uint32_t width, height, depth, pitch, rows;
    uint64_t offset, slice;
};

static struct malachite_lvl malachite_image_level(MTLPixelFormat f, NSUInteger w, NSUInteger h, uint32_t d, uint32_t l)
{
    uint32_t bpp = (uint32_t)malachite_bytes_per_pixel(f), b = malachite_block(f);
    struct malachite_lvl v = {};
    for (uint32_t i = 0; i <= l; i++) {
        v.offset += v.slice * v.depth;
        v.width = ((uint32_t)MAX(w >> i, 1u) + b - 1) / b;
        v.height = ((uint32_t)MAX(h >> i, 1u) + b - 1) / b;
        v.depth = MAX(d >> i, 1u);
        v.pitch = (v.width * bpp + 63) & ~63u;
        v.rows = (v.height + 7) & ~7u;
        v.slice = (uint64_t)v.pitch * v.rows;
    }
    return v;
}

// SET_COLOR_CLEAR_VALUE takes an integer target's colour as the integers' bits, clamped by the 3D class.
static void malachite_clear_colour(enum tex_format f, MTLClearColor c, float out[4])
{
    const double v[4] = { c.red, c.green, c.blue, c.alpha };
    bool sint = f == TEX_R8_SINT || f == TEX_RG8_SINT || f == TEX_RGBA8_SINT || f == TEX_R16_SINT || f == TEX_RG16_SINT ||
                f == TEX_RGBA16_SINT || f == TEX_R32_SINT || f == TEX_RG32_SINT || f == TEX_RGBA32_SINT;
    for (int i = 0; i < 4; i++) {
        if (!tex_format_info(f).integer) {
            out[i] = (float)v[i];
            continue;
        }
        uint32_t bits = sint ? (uint32_t)(int32_t)fmin(fmax(v[i], INT32_MIN), INT32_MAX)
                             : (uint32_t)fmin(fmax(v[i], 0), UINT32_MAX);
        memcpy(&out[i], &bits, 4);
    }
}

// as the 3D class lays samples out in memory
static void malachite_grid(uint32_t samples, uint32_t *sx, uint32_t *sy)
{
    *sx = samples >= 8 ? 4 : samples >= 2 ? 2 : 1;
    *sy = samples >= 4 ? 2 : 1;
}

// one level of one slice at one z, sized in blocks
struct malachite_sub {
    uint64_t offset;
    uint32_t pitch;
    uint32_t width, height;
    uint32_t rows;  // 0 for pitch-linear
    uint32_t bpp, block;
};

static bool malachite_sub(MalachiteTexture *t, NSUInteger level, NSUInteger slice, NSUInteger z, struct malachite_sub *s)
{
    if (t->_image) {
        if (level >= t->_levels || slice >= t->_layers)
            return false;
        uint32_t sx, sy;
        malachite_grid(t->_samples, &sx, &sy);
        struct malachite_lvl v = malachite_image_level(t->_format, t->_width * sx, t->_height * sy, t->_depth,
                                               t->_view_min + (uint32_t)level);
        if (z >= v.depth)
            return false;
        *s = (struct malachite_sub){ t->_offset + slice * t->_layer_stride + v.offset + z * v.slice, v.pitch, v.width,
                                 v.height, v.rows, (uint32_t)malachite_bytes_per_pixel(t->_format), malachite_block(t->_format) };
        return true;
    }
    if (level || z)
        return false;
    bool rows = t->_type == MTLTextureType1DArray;
    if (slice && !rows)
        return false;
    id<MTLTexture> mt = (id<MTLTexture>)t;
    *s = (struct malachite_sub){ t->_offset + slice * t->_pitch, t->_pitch, (uint32_t)mt.width, (uint32_t)mt.height, 0,
                             (uint32_t)malachite_bytes_per_pixel(mt.pixelFormat), 1 };
    return true;
}

// a GOB is 64 bytes by 8 rows of 16-byte sectors
static size_t malachite_gob_offset(uint32_t pitch, size_t x, size_t y)
{
    size_t gob = y / 8 * (pitch / 64) + x / 64;
    return gob * 512 + (x % 64) / 32 * 256 + (y % 8) / 4 * 128 + (x % 32) / 16 * 64 + (y % 4) * 16 + x % 16;
}

// block-linear surfaces a sector's 16 bytes at a time
static void malachite_copy_sub(uint8_t *pixels, const struct malachite_sub *s, size_t x0, size_t y0, size_t w, size_t h,
                           uint8_t *bytes, NSUInteger stride, bool in)
{
    uint8_t *base = pixels + s->offset;
    for (size_t y = 0; y < h; y++) {
        uint8_t *row = bytes + y * stride;
        if (!s->rows) {
            uint8_t *texel = base + (y0 + y) * s->pitch + x0 * s->bpp;
            memcpy(in ? texel : row, in ? row : texel, w * s->bpp);
            continue;
        }
        for (size_t x = 0; x < w * s->bpp;) {
            size_t from = x0 * s->bpp + x, n = MIN(16 - from % 16, w * s->bpp - x);
            uint8_t *texel = base + malachite_gob_offset(s->pitch, from, y0 + y);
            memcpy(in ? texel : row + x, in ? row + x : texel, n);
            x += n;
        }
    }
}

// A Managed buffer has a VRAM copy the GPU uses, as a discrete card's driver keeps it: reading over PCIe was a fifth
// of a scene's frame time. The next commit copies what the CPU changed; synchronizeResource: copies back.
@interface MalachiteBuffer : MTLIOAccelBuffer <MTLBuffer> {
@public
    MalachiteBuffer *_mirror;
    NSUInteger _dirty_lo, _dirty_hi;  // changed since the copy to _mirror was queued
    bool _pending;                    // on malachite_pending
    bool _one_image;                  // nothing but a single image lies in it
    bool _compress_asked;
    // CL_MEM_USE_HOST_PTR memory the app reads without synchronizeResource (AMD kept it in system memory)
    bool _writeback;
    // the base maps from the page's start
    NSUInteger _page_off;
}
- (uint64_t)systemAddress;
- (instancetype)initWithDevice:(id)device
                    withLength:(NSUInteger)length
                   withOptions:(MTLResourceOptions)options
                   withPointer:(void *)ptr
               withDeallocator:(void (^)(void))dealloc;
@end

// A heap must be one real block: MPS marks temporaries aliasable expecting reuse, and given VRAM each, a graph's
// temporaries outgrow the GPU. Placement heaps keep no track, the app lays out the aliasing.
@class MalachiteHeap;

@interface MalachiteHeapBuffer : NSObject <MTLBuffer> {
@public
    MalachiteHeap *_heap;  // keeps the block's memory alive
    NSUInteger _offset, _length, _size;
    MTLResourceOptions _options;
    bool _aliasable;
}
@property (copy) NSString *label;
@end

@interface MalachiteHeap : NSObject <MTLHeap> {
@public
    __weak id _device;
    MTLHeapDescriptor *_desc;
    id<MTLBuffer> _block;
    os_unfair_lock _lock;
    std::map<NSUInteger, NSUInteger> _holes;  // offset to length
    NSUInteger _free;
}
@property (copy) NSString *label;
- (MalachiteHeapBuffer *)place:(NSUInteger)length
                          size:(MTLSizeAndAlign)range
                       options:(MTLResourceOptions)options
                            at:(NSUInteger)offset;
- (void)giveBack:(NSUInteger)offset size:(NSUInteger)size;
@end

static MalachiteTexture *malachite_swizzled(MalachiteTexture *t, MTLTextureSwizzleChannels swizzle);

static bool opengl_renderer;  // Apple's OpenGL and OpenCL renderer is loaded in this process

// each commit takes the lot
static NSMutableArray<MalachiteBuffer *> *malachite_pending;
static uint64_t malachite_allocated;  // for currentAllocatedSize

static void malachite_mark_dirty(MalachiteBuffer *b, NSUInteger lo, NSUInteger hi)
{
    if (!b->_mirror || lo >= hi)
        return;
    @synchronized([MalachiteBuffer class]) {
        if (b->_dirty_lo >= b->_dirty_hi) {
            b->_dirty_lo = lo;
            b->_dirty_hi = hi;
        } else {
            b->_dirty_lo = MIN(b->_dirty_lo, lo);
            b->_dirty_hi = MAX(b->_dirty_hi, hi);
        }
        if (!b->_pending) {
            if (!malachite_pending)
                malachite_pending = [NSMutableArray array];
            [malachite_pending addObject:b];
            b->_pending = true;
        }
    }
}

static uint64_t malachite_vram(void);

// OpenCL's runtime tags its own allocations
static bool malachite_opencl_memory(void *ptr)
{
    vm_address_t at = (vm_address_t)ptr;
    vm_size_t size = 0;
    natural_t depth = 0;
    vm_region_submap_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    return vm_region_recurse_64(mach_task_self(), &at, &size, &depth, (vm_region_recurse_info_t)&info, &count) ==
               KERN_SUCCESS &&
           at <= (vm_address_t)ptr && info.user_tag == VM_MEMORY_OPENCL;
}

@implementation MalachiteBuffer
- (instancetype)initWithDevice:(id)device
                    withLength:(NSUInteger)length
                   withOptions:(MTLResourceOptions)options
                   withPointer:(void *)ptr
               withDeallocator:(void (^)(void))dealloc
{
    size_t page = vm_page_size;
    uintptr_t off = (uintptr_t)ptr & (page - 1);
    void *aligned = (void *)((uintptr_t)ptr & ~(uintptr_t)(page - 1));
    NSUInteger sysMem = (off + length + page - 1) & ~(NSUInteger)(page - 1);

    // zeroed, or the base init reads garbage; [0x68..0x7f] is our tail: kind, length, non-shared flag
    uint8_t args[0x80];
    memset(args, 0, sizeof(args));
    *(uint32_t *)(args + 0x68) = 1;
    *(uint64_t *)(args + 0x70) = (uint64_t)length;
    *(uint32_t *)(args + 0x78) = (options & 0xF0) ? 1 : 0;
    // the base wants system memory behind every buffer to page VRAM out to; private storage needs none
    bool vid = !ptr && (options & MTLResourceStorageModeMask) == MTLResourceStorageModePrivate;

    self = [super initWithDevice:device
                         pointer:aligned
                          length:length
                         options:options
                      sysMemSize:sysMem
                      vidMemSize:vid ? sysMem : 0
                            args:args
                        argsSize:0x80
                     deallocator:dealloc];
    if (!self)
        return nil;
    _page_off = off;
    __atomic_add_fetch(&malachite_allocated, length, __ATOMIC_RELAXED);
    // Even small ones: GPU writes and atomics over the bus cost far more than copying, and OpenCL kernels ran 10-100x
    // slower. No-copy ones only for OpenCL: WindowServer never calls didModifyRange: on layer memory, so text went stale.
    if ((options & MTLResourceStorageModeMask) == MTLResourceStorageModeManaged && length &&
        (!ptr || opengl_renderer) && length <= malachite_vram() / 4) {
        _mirror = [[MalachiteBuffer alloc] initWithDevice:device
                                               withLength:length
                                              withOptions:MTLResourceStorageModePrivate
                                              withPointer:NULL
                                          withDeallocator:nil];
        _writeback = ptr && !malachite_opencl_memory(ptr);
        malachite_mark_dirty(self, 0, length);
    }
    return self;
}

- (void)dealloc
{
    __atomic_sub_fetch(&malachite_allocated, self.length, __ATOMIC_RELAXED);
}

- (uint64_t)gpuAddress
{
    return _mirror ? [_mirror gpuAddress] : [super gpuAddress] + _page_off;
}

- (uint64_t)systemAddress
{
    return [super gpuAddress] + _page_off;
}

// OpenCL copies data in when contents isn't its own pointer, and from the page's start that overran the caller's stack
- (void *)contents
{
    uint8_t *c = (uint8_t *)[super contents];
    return c ? c + _page_off : NULL;
}

- (void)didModifyRange:(NSRange)range
{
    malachite_mark_dirty(self, range.location, NSMaxRange(range));
    [super didModifyRange:range];
}

- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor offset:(NSUInteger)offset
                   bytesPerRow:(NSUInteger)bytesPerRow
{
    MalachiteTexture *t =
        [[MalachiteTexture alloc] initWithBuffer:self descriptor:descriptor offset:offset bytesPerRow:bytesPerRow];
    if (t) {
        t->_backing = self;
        t->_owned = self;  // as Metal's do, the app may drop the buffer first
        t->_offset = offset;
        t->_pitch = (uint32_t)bytesPerRow;
        t->_type = descriptor.textureType;
        t->_usage = descriptor.usage;
        t->_width = descriptor.width;  // the base keeps 16 bits, a texture buffer's can be millions
    }
    return malachite_swizzled(t, descriptor.swizzle);  // CoreAnimation reads its one-channel surfaces through one
}

// WindowServer's tiled backings; linear is still one layout every reader and writer agrees on
- (id)newTiledTextureWithDescriptor:(MTLTextureDescriptor *)descriptor offset:(NSUInteger)offset
                        bytesPerRow:(NSUInteger)bytesPerRow
{
    return [self newTextureWithDescriptor:descriptor offset:offset bytesPerRow:bytesPerRow];
}
@end

// Apple's GL renderer's NVIDIA table denies last-vertex provoking, which the hardware has, so every draw got a compute
// pass rewriting its indices at ten times the draw's cost. Bit 3 of the table's first byte fixes that, per process.
static void malachite_gl_provoking(const struct mach_header_64 *mh, intptr_t slide)
{
    const struct symtab_command *tab = NULL;
    const struct segment_command_64 *linkedit = NULL;
    const struct load_command *lc = (const struct load_command *)(mh + 1);
    for (uint32_t i = 0; i < mh->ncmds; i++, lc = (const struct load_command *)((const char *)lc + lc->cmdsize)) {
        if (lc->cmd == LC_SYMTAB)
            tab = (const struct symtab_command *)lc;
        else if (lc->cmd == LC_SEGMENT_64 && !strcmp(((const struct segment_command_64 *)lc)->segname, SEG_LINKEDIT))
            linkedit = (const struct segment_command_64 *)lc;
    }
    if (!tab || !linkedit)
        return;
    const char *base = (const char *)(linkedit->vmaddr + slide - linkedit->fileoff);
    const struct nlist_64 *syms = (const struct nlist_64 *)(base + tab->symoff);
    const char *names = base + tab->stroff;
    for (uint32_t i = 0; i < tab->nsyms; i++) {
        if (strcmp(names + syms[i].n_un.n_strx, "__ZZN12GLDDeviceRec19initWithDisplayMaskEjE6nvidia"))
            continue;
        uint8_t *caps = (uint8_t *)(syms[i].n_value + slide);
        vm_address_t page = (vm_address_t)caps & ~(vm_address_t)vm_page_mask;
        // may be refused in some apps, which then draw as before
        if (vm_protect(mach_task_self(), page, vm_page_size, FALSE, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY) ==
            KERN_SUCCESS) {
            *caps |= 8;
            vm_protect(mach_task_self(), page, vm_page_size, FALSE, VM_PROT_READ);
        }
        return;
    }
    os_log(OS_LOG_DEFAULT, "Malachite: no capability table in Apple's GL renderer");
}

// With no WindowServer timestamps, AppKit sleeps to 60 Hz vblanks on every screen, 90 swaps a second at 240 Hz.
// So a swap waits only for the last frame to be taken, and not at all with vsync off, which AppKit ignores.
// The time limit is for a window nothing composites.
static long (*gl_unconsumed)(void *, void *, void *);
static void *(*gl_context)(void);
static int (*gl_get)(void *, int, int *);
static ptrdiff_t gl_queue_offset;

static void gl_synchronize(id self, SEL)
{
    enum { kCGLCPSwapInterval = 222 };  // not linked against OpenGL: WindowServer loads this plugin too
    int interval = 1;
    gl_get(gl_context(), kCGLCPSwapInterval, &interval);
    if (!interval)
        return;
    void *queue = *(void **)((char *)(__bridge void *)self + gl_queue_offset);
    const uint64_t limit = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + 16666667;
    while (gl_unconsumed(queue, NULL, NULL) && clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < limit)
        usleep(100);
}

static void malachite_gl_swaps(void)
{
    Class surface = objc_getClass("NSCGLSurface");
    Ivar queue = class_getInstanceVariable(surface, "_imageQueue");
    Method wait = class_getInstanceMethod(surface, sel_registerName("synchronize"));
    gl_unconsumed = (long (*)(void *, void *, void *))dlsym(RTLD_DEFAULT, "CAImageQueueGetUnconsumedImageCount");
    gl_context = (void *(*)(void))dlsym(RTLD_DEFAULT, "CGLGetCurrentContext");
    gl_get = (int (*)(void *, int, int *))dlsym(RTLD_DEFAULT, "CGLGetParameter");
    if (!queue || !wait || !gl_unconsumed || !gl_context || !gl_get)
        return;
    gl_queue_offset = ivar_getOffset(queue);
    method_setImplementation(wait, (IMP)gl_synchronize);
}

static void malachite_hook_function_reflection(void);

static void note_image(const struct mach_header *mh, intptr_t slide)
{
    Dl_info info;
    if (dladdr(mh, &info) && strstr(info.dli_fname, "AppleMetalOpenGLRenderer.bundle/Contents/MacOS")) {
        opengl_renderer = true;
        malachite_gl_provoking((const struct mach_header_64 *)mh, slide);
        malachite_hook_function_reflection();  // OpenCL runs on this renderer too
    }
}

__attribute__((constructor)) static void watch_images(void)
{
    _dyld_register_func_for_add_image(note_image);
}

static void malachite_end_segment(MalachiteCommandBuffer *cb)
{
    if (cb->_in_segment) {
        cb->_in_segment = false;
        [cb endCurrentSegment];
    }
}

static void malachite_run_flush(MalachiteCommandBuffer *cb);

static void malachite_record(id commandBuffer, uint32_t opcode, const void *payload, uint32_t size)
{
    MalachiteCommandBuffer *cb = (MalachiteCommandBuffer *)commandBuffer;
    if (cb->_run_draws && opcode != MALACHITE_CMD_PUSH)  // keeps encoding order
        malachite_run_flush(cb);
    if (opcode != MALACHITE_CMD_PUSH)
        cb->_records++;
    const uint32_t total = (uint32_t)(sizeof(struct malachite_cmd_hdr) + size);
    void *cur = nullptr, *end = nullptr;
    [cb getCurrentKernelCommandBufferPointer:&cur end:&end];
    if (opcode != MALACHITE_CMD_DRAW || (char *)end - (char *)cur < total)
        malachite_end_segment(cb);
    if (!cb->_in_segment) {
        [cb getCurrentKernelCommandBufferPointer:&cur end:&end];
        [cb beginSegment:cur];
        cb->_in_segment = true;
    }
    uint8_t *rec = (uint8_t *)[cb _reserveKernelCommandBufferSpace:total];
    if (rec) {
        struct malachite_cmd_hdr *h = (struct malachite_cmd_hdr *)rec;
        h->opcode = opcode | (opengl_renderer ? MALACHITE_CMD_OPENGL : 0);
        h->len = total;
        memcpy(rec + sizeof(*h), payload, size);
    }
    if (opcode != MALACHITE_CMD_DRAW)  // the base class has work to do per segment, so draws share one
        malachite_end_segment(cb);
}

static void malachite_run_flush(MalachiteCommandBuffer *cb)
{
    if (!cb->_run_draws)
        return;
    struct malachite_push p = { cb->_run_va + cb->_run_start * 4, cb->_run_used - cb->_run_start, cb->_run_draws };
    cb->_run_start = cb->_run_used;
    cb->_run_draws = 0;
    malachite_record(cb, MALACHITE_CMD_PUSH, &p, sizeof(p));
}

// Commands carry raw GPU addresses, and a resource freed after encoding (an MPS temporary) is unmapped under the GPU.
static void malachite_keep(__unsafe_unretained id commandBuffer, __unsafe_unretained id resource)
{
    if (!resource)
        return;
    __unsafe_unretained MalachiteCommandBuffer *cb = commandBuffer;
    const void *p = (__bridge const void *)resource;
    for (const void *recent : cb->_kept_recent)
        if (recent == p)
            return;
    cb->_kept_recent[cb->_kept_next++ % 4] = p;
    if (!cb->_kept)
        cb->_kept = [NSMutableArray array];
    [cb->_kept addObject:resource];
}

// OpenCL reads some back without a synchronize, so those are copied back at commit
static void malachite_written(__unsafe_unretained id commandBuffer, __unsafe_unretained id buffer)
{
    MalachiteCommandBuffer *cb = commandBuffer;
    MalachiteBuffer *b = buffer;
    if (![b isKindOfClass:[MalachiteBuffer class]] || !b->_writeback)
        return;
    if (!cb->_writeback)
        cb->_writeback = [NSMutableSet set];
    [cb->_writeback addObject:b];
}

static void malachite_texture_written(__unsafe_unretained id commandBuffer, __unsafe_unretained id texture)
{
    MalachiteTexture *t = texture;
    if ([t isKindOfClass:[MalachiteTexture class]] && t->_capture)
        ((MalachiteCommandBuffer *)commandBuffer)->_writes_capture = true;
}

struct malachite_side {
    uint64_t va;
    uint32_t pitch, rows;  // rows 0 for pitch-linear
    uint32_t x, y;  // x in bytes
};

static struct malachite_side malachite_side_of(MalachiteTexture *t, const struct malachite_sub *s, NSUInteger x, NSUInteger y)
{
    return (struct malachite_side){ [(MTLIOAccelBuffer *)t->_backing gpuAddress] + s->offset, s->pitch, s->rows,
                                (uint32_t)(x / s->block * s->bpp), (uint32_t)(y / s->block) };
}

// the copy engine's SET_REMAP_COMPONENTS
enum { REMAP_X = 0, REMAP_Y = 1, REMAP_ZERO = 4, REMAP_NONE = 6 };  // SRC_X, SRC_Y, CONST_A (0), NO_WRITE

static uint32_t malachite_remap(uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t size, uint32_t nsrc,
                            uint32_t ndst)
{
    return x | y << 4 | z << 8 | w << 12 | (size - 1) << 16 | (nsrc - 1) << 20 | (ndst - 1) << 24;
}

// stencil-only formats have packed texels too
enum { MALACHITE_ASPECT_ALL, MALACHITE_ASPECT_DEPTH, MALACHITE_ASPECT_STENCIL };

static int malachite_aspect(id<MTLTexture> t, MTLBlitOption options)
{
    switch (t.pixelFormat) {
    case MTLPixelFormatStencil8:
    case MTLPixelFormatX24_Stencil8:
    case MTLPixelFormatX32_Stencil8:
        return MALACHITE_ASPECT_STENCIL;
    case MTLPixelFormatDepth24Unorm_Stencil8:
    case MTLPixelFormatDepth32Float_Stencil8:
        return options & MTLBlitOptionDepthFromDepthStencil     ? MALACHITE_ASPECT_DEPTH
               : options & MTLBlitOptionStencilFromDepthStencil ? MALACHITE_ASPECT_STENCIL
                                                                : MALACHITE_ASPECT_ALL;
    default:
        return MALACHITE_ASPECT_ALL;
    }
}

static void malachite_depth_kind(id cb, MalachiteTexture *t);
static id<MTLBuffer> malachite_upload(id commandBuffer, const void *bytes, NSUInteger length, uint64_t *va);
@class MalachiteComputePipelineState;
static void malachite_dispatch(id cb, MalachiteComputePipelineState *pipeline, const uint64_t buffers[31],
                           __unsafe_unretained const id *textures, __unsafe_unretained const id *samplers,
                           MTLSize groups, MTLSize threads, uint64_t indirect = 0,
                           const uint32_t *threadgroup = NULL, const uint32_t *part = NULL, uint64_t region = 0);
static id<MTLComputePipelineState> malachite_mip_pipeline(id<MTLDevice> device, bool volume);

// Electron apps carry ANGLE in a framework's Libraries, helpers included.
static bool malachite_has_chromium(void)
{
    char exe[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size))
        return false;
    char *end = strstr(exe, ".app/");
    if (!end)
        return false;
    end[4] = 0;
    char frameworks[PATH_MAX];
    snprintf(frameworks, sizeof frameworks, "%s/Contents/Frameworks", exe);
    DIR *dir = opendir(frameworks);
    bool found = false;
    while (dir && !found) {
        struct dirent *e = readdir(dir);
        if (!e)
            break;
        char lib[PATH_MAX];
        snprintf(lib, sizeof lib, "%s/%s/Libraries/libGLESv2.dylib", frameworks, e->d_name);
        found = !access(lib, F_OK);
    }
    if (dir)
        closedir(dir);
    return found;
}

// ANGLE turns Metal off on NVIDIA, and its GL backend faults the GPU through Apple's renderer. Set at load so helpers
// inherit it, and wherever ANGLE loads; MALACHITE_ANGLE_METAL=0 leaves it, =1 forces it.
static void malachite_angle_override(void)
{
    const char *want = getenv("MALACHITE_ANGLE_METAL"), *key = "ANGLE_FEATURE_OVERRIDES_DISABLED", *cur = getenv(key);
    if ((want && !strcmp(want, "0")) || (cur && strstr(cur, "disableMetalOnNvidia")))
        return;
    char rules[1024];
    snprintf(rules, sizeof rules, "%s%sdisableMetalOnNvidia", cur ? cur : "", cur && *cur ? ":" : "");
    setenv(key, rules, 1);
}

__attribute__((constructor)) static void malachite_angle_metal(void)
{
    const char *want = getenv("MALACHITE_ANGLE_METAL"), *me = getprogname();
    static const char *const chromium[] = { "Google Chrome", "Brave Browser", "Microsoft Edge", "Chromium", "Discord" };
    bool ours = want && !strcmp(want, "1");
    for (size_t i = 0; !ours && me && i < sizeof chromium / sizeof chromium[0]; i++)
        ours = !strncmp(me, chromium[i], strlen(chromium[i]));
    if (ours || malachite_has_chromium())
        malachite_angle_override();
}

// VideoToolbox never offers NVVideo.bundle from /Library/Video/Plug-Ins; registered by hand it is the first choice.
static void malachite_register_video(void)
{
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        void *lib = dlopen("/Library/Video/Plug-Ins/NVVideo.bundle/Contents/MacOS/NVVideo", RTLD_NOW | RTLD_LOCAL);
        auto reg = (OSStatus (*)(FourCharCode, CFDictionaryRef, void *))dlsym(RTLD_DEFAULT,
                                                                             "VTRegisterVideoDecoderWithInfo");
        static const struct {
            FourCharCode codec;
            const char *factory;
            NSString *impl, *name, *label;
        } decoders[] = {
            { 'avc1', "NVDecH264_CreateInstance", @"pl.aochi.videodecoder.avc.nvdec", @"NVIDIA H.264 (NVDEC)", @"H.264" },
            { 'hvc1', "NVDecHEVC_CreateInstance", @"pl.aochi.videodecoder.hevc.nvdec", @"NVIDIA H.265 (NVDEC)", @"H.265" },
            { 'av01', "NVDecAV1_CreateInstance", @"pl.aochi.videodecoder.av1.nvdec", @"NVIDIA AV1 (NVDEC)", @"AV1" },
            { 'vp09', "NVDecVP9_CreateInstance", @"pl.aochi.videodecoder.vp9.nvdec", @"NVIDIA VP9 (NVDEC)", @"VP9" },
            // VP8's ISO media type; CoreMedia has none, VideoToolbox takes any
            { 'vp08', "NVDecVP8_CreateInstance", @"pl.aochi.videodecoder.vp8.nvdec", @"NVIDIA VP8 (NVDEC)", @"VP8" },
        };
        for (const auto &decoder : decoders) {
            void *factory = lib ? dlsym(lib, decoder.factory) : NULL;
            if (!factory || !reg)
                continue;
            // VideoToolbox appends "." and the codec type to the id
            NSDictionary *info = @{
                @"CMClassImplementationID" : decoder.impl,
                @"CMClassImplementationName" : decoder.name,
                @"VTRating" : @1000,
                @"VTIsHardwareAccelerated" : @YES,
            };
            OSStatus st = reg(decoder.codec, (__bridge CFDictionaryRef)info, factory);
            os_log(OS_LOG_DEFAULT, "Malachite: %{public}@ decoder registered with VideoToolbox: %d", decoder.label,
                   (int)st);
        }
    });
}

static const char *malachite_dylib_name(const struct mach_header *mh)
{
    const struct load_command *lc = (const struct load_command *)((const struct mach_header_64 *)mh + 1);
    for (uint32_t i = 0; i < mh->ncmds; i++) {
        if (lc->cmd == LC_ID_DYLIB)
            return (const char *)lc + ((const struct dylib_command *)lc)->dylib.name.offset;
        lc = (const struct load_command *)((const char *)lc + lc->cmdsize);
    }
    return NULL;
}

static bool malachite_ends_with(const char *s, const char *tail)
{
    size_t n = strlen(s), m = strlen(tail);
    return n >= m && !strcmp(s + n - m, tail);
}

// runs inside dyld, so registering, which loads NVVideo.bundle, waits until after
static void malachite_image_added(const struct mach_header *mh, intptr_t slide)
{
    const char *name = malachite_dylib_name(mh);
    if (!name)
        return;
    if (strstr(name, "/VideoToolbox.framework/"))
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            malachite_register_video();
        });
    // libANGLE-shared is the system web engine's, in its GPU process
    if (malachite_ends_with(name, "/libGLESv2.dylib") || malachite_ends_with(name, "/libEGL.dylib") ||
        malachite_ends_with(name, "/libANGLE-shared.dylib"))
        malachite_angle_override();
}

__attribute__((constructor)) static void malachite_video_decoder(void)
{
    const char *me = getprogname();
    if (!me || strcmp(me, "WindowServer"))
        _dyld_register_func_for_add_image(malachite_image_added);
}

// CoreAnimation fills only level 0 of small mipmapped badges and samples them minified, showing black boxes, so the
// levels are made on the CPU until anything writes one of its own.
static char malachite_mips_key;

static void malachite_mips_written(id texture, NSUInteger level)
{
    MalachiteTexture *t = texture;
    if ([t isKindOfClass:[MalachiteTexture class]] && t->_backing && level + t->_view_min > 0)
        objc_setAssociatedObject(t->_backing, &malachite_mips_key, @YES, OBJC_ASSOCIATION_RETAIN);
}

static void malachite_mip_fill(MalachiteTexture *t, MTLRegion region, NSUInteger level, NSUInteger slice, const uint8_t *bytes,
                           NSUInteger stride)
{
    NSUInteger w = t.width, h = t.height, levels = t.mipmapLevelCount, bpp;
    switch (t.pixelFormat) {
    case MTLPixelFormatA8Unorm: case MTLPixelFormatR8Unorm: bpp = 1; break;
    case MTLPixelFormatRG8Unorm: bpp = 2; break;
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB: bpp = 4; break;
    default: return;
    }
    if (!t->_image || levels < 2 || level || slice || t->_view_min || t->_layers > 1 || t.textureType != MTLTextureType2D ||
        w > 2048 || h > 2048 || region.origin.x || region.origin.y || region.size.width != w || region.size.height != h ||
        !bytes || objc_getAssociatedObject(t->_backing, &malachite_mips_key))
        return;
    uint8_t *cur = (uint8_t *)malloc((w + 1) / 2 * ((h + 1) / 2) * bpp), *next = (uint8_t *)malloc((w + 1) / 2 * ((h + 1) / 2) * bpp);
    const uint8_t *src = bytes;
    NSUInteger srow = stride;
    for (NSUInteger l = 1; cur && next && l < levels; l++) {
        NSUInteger dw = w > 1 ? w / 2 : 1, dh = h > 1 ? h / 2 : 1;
        for (NSUInteger y = 0; y < dh; y++)
            for (NSUInteger x = 0; x < dw; x++) {
                NSUInteger x1 = MIN(2 * x + 1, w - 1), y1 = MIN(2 * y + 1, h - 1);
                const uint8_t *r0 = src + 2 * y * srow, *r1 = src + y1 * srow;
                for (NSUInteger c = 0; c < bpp; c++)
                    next[(y * dw + x) * bpp + c] =
                        (uint8_t)((r0[2 * x * bpp + c] + r0[x1 * bpp + c] + r1[2 * x * bpp + c] + r1[x1 * bpp + c] + 2) / 4);
            }
        [t copyRegion:MTLRegionMake2D(0, 0, dw, dh) level:l slice:0 bytes:next stride:dw * bpp image:0 in:true];
        uint8_t *swap = cur;
        cur = next;
        next = swap;
        src = cur;
        srow = dw * bpp;
        w = dw;
        h = dh;
    }
    // the copies marked them as an app's own
    objc_setAssociatedObject(t->_backing, &malachite_mips_key, nil, OBJC_ASSOCIATION_RETAIN);
    free(cur);
    free(next);
}

// The kernel runs these on the copy engine (0xC7B5), which lays out block-linear surfaces itself.
@interface MalachiteBlitCommandEncoder : MTLIOAccelBlitCommandEncoder <MTLBlitCommandEncoder> {
    __unsafe_unretained id _cb;  // outlives the encoder; retaining it would be a cycle
}
- (instancetype)initWithCommandBuffer:(id)cb;
@end

@implementation MalachiteBlitCommandEncoder
- (instancetype)initWithCommandBuffer:(id)cb
{
    self = [super initWithCommandBuffer:cb];
    if (self) _cb = cb;
    return self;
}

// commands run to completion in encoding order, so there is nothing to fence
- (void)updateFence:(id)fence
{
}

- (void)waitForFence:(id)fence
{
}

- (void)barrierAfterQueueStages:(MTLStages)after beforeStages:(MTLStages)before
{
}

- (void)copyFromBuffer:(id)src sourceOffset:(NSUInteger)srcOff
              toBuffer:(id)dst destinationOffset:(NSUInteger)dstOff
                  size:(NSUInteger)size
{
    struct malachite_copy_buf2buf p = {};
    p.srcAddr = [(MTLIOAccelBuffer *)src gpuAddress] + srcOff;
    p.dstAddr = [(MTLIOAccelBuffer *)dst gpuAddress] + dstOff;
    p.size = size;
    malachite_keep(_cb, src);
    malachite_keep(_cb, dst);
    malachite_written(_cb, dst);
    malachite_record(_cb, MALACHITE_CMD_COPY_BUF2BUF, &p, sizeof(p));
}

- (void)fillBuffer:(id)buffer range:(NSRange)range value:(uint8_t)value
{
    struct malachite_fill_buffer p = {};
    p.dstAddr = [(MTLIOAccelBuffer *)buffer gpuAddress] + range.location;
    p.size = range.length;
    p.pattern = (uint32_t)value * 0x01010101u;
    malachite_keep(_cb, buffer);
    malachite_written(_cb, buffer);
    malachite_record(_cb, MALACHITE_CMD_FILL_BUFFER, &p, sizeof(p));
}

// whole GOBs go into a block-linear side's address, as the origin fields are only 16 bits
- (void)copy:(struct malachite_side)a
          to:(struct malachite_side)b
       bytes:(NSUInteger)bytes
       lines:(NSUInteger)lines
       remap:(uint32_t)remap
{
    if (!bytes || !lines)
        return;
    struct malachite_side *side[2] = { &a, &b };
    uint32_t origin[2] = {};
    for (int i = 0; i < 2; i++) {
        struct malachite_side *v = side[i];
        if (!v->rows) {
            v->va += (uint64_t)v->y * v->pitch + v->x;
            continue;
        }
        v->va += (uint64_t)(v->y / 8 * (v->pitch / 64) + v->x / 64) * 512;
        v->rows -= v->y / 8 * 8;
        origin[i] = v->x % 64 | (v->y % 8) << 16;
    }
    struct nvgpu_copy2d c = { a.va, b.va, a.pitch, b.pitch, (uint32_t)bytes, (uint32_t)lines, a.rows != 0, b.rows != 0,
                              a.rows, b.rows, origin[0], origin[1], remap };
    malachite_record(_cb, MALACHITE_CMD_COPY_2D, &c, sizeof(c));
}

// Texels are the depth word, then a word with the stencil in its low byte; the stencil goes through a temporary word
// per texel, as a copy-engine component is at most 4 bytes.
- (void)copy:(struct malachite_side)a
            to:(struct malachite_side)b
       texels:(NSUInteger)w
        lines:(NSUInteger)h
       aspect:(int)aspect
    toTexture:(bool)in
{
    const uint32_t x = REMAP_X, y = REMAP_Y, zero = REMAP_ZERO, none = REMAP_NONE;
    if (aspect == MALACHITE_ASPECT_DEPTH) {
        [self copy:a to:b bytes:w * (in ? 4 : 8) lines:h
             remap:in ? malachite_remap(x, none, none, none, 4, 1, 2) : malachite_remap(x, none, none, none, 4, 2, 1)];
        return;
    }
    id<MTLBuffer> words = [((id<MTLCommandBuffer>)_cb).device newBufferWithLength:w * h * 4
                                                                          options:MTLResourceStorageModePrivate];
    malachite_keep(_cb, words);
    struct malachite_side tmp = { [(MTLIOAccelBuffer *)words gpuAddress], (uint32_t)w * 4 };
    if (in) {
        [self copy:a to:tmp bytes:w lines:h remap:malachite_remap(x, zero, zero, zero, 1, 1, 4)];
        [self copy:tmp to:b bytes:w * 4 lines:h remap:malachite_remap(none, x, none, none, 4, 1, 2)];
    } else {
        [self copy:a to:tmp bytes:w * 8 lines:h remap:malachite_remap(y, none, none, none, 4, 2, 1)];
        [self copy:tmp to:b bytes:w * 4 lines:h remap:malachite_remap(x, none, none, none, 1, 4, 1)];
    }
}

- (void)copyFromTexture:(id)src
             sourceSlice:(NSUInteger)srcSlice
             sourceLevel:(NSUInteger)srcLevel
            sourceOrigin:(MTLOrigin)srcOrigin
              sourceSize:(MTLSize)size
               toTexture:(id)dst
        destinationSlice:(NSUInteger)dstSlice
        destinationLevel:(NSUInteger)dstLevel
       destinationOrigin:(MTLOrigin)dstOrigin
{
    malachite_mips_written(dst, dstLevel);
    malachite_texture_written(_cb, dst);
    malachite_keep(_cb, src);
    malachite_keep(_cb, dst);
    malachite_depth_kind(_cb, src);
    malachite_depth_kind(_cb, dst);
    for (NSUInteger z = 0; z < size.depth; z++) {
        struct malachite_sub a, b;
        if (!malachite_sub(src, srcLevel, srcSlice, srcOrigin.z + z, &a) ||
            !malachite_sub(dst, dstLevel, dstSlice, dstOrigin.z + z, &b))
            return;
        NSUInteger w = (size.width + a.block - 1) / a.block, h = (size.height + a.block - 1) / a.block;
        [self copy:malachite_side_of(src, &a, srcOrigin.x, srcOrigin.y)
                to:malachite_side_of(dst, &b, dstOrigin.x, dstOrigin.y)
             bytes:w * a.bpp
             lines:h
             remap:0];
    }
}

- (void)copyFromBuffer:(id)src
           sourceOffset:(NSUInteger)srcOff
      sourceBytesPerRow:(NSUInteger)srcPitch
    sourceBytesPerImage:(NSUInteger)srcImage
             sourceSize:(MTLSize)size
              toTexture:(id)dst
       destinationSlice:(NSUInteger)dstSlice
       destinationLevel:(NSUInteger)dstLevel
      destinationOrigin:(MTLOrigin)dstOrigin
{
    [self copyFromBuffer:src sourceOffset:srcOff sourceBytesPerRow:srcPitch sourceBytesPerImage:srcImage
              sourceSize:size toTexture:dst destinationSlice:dstSlice destinationLevel:dstLevel
       destinationOrigin:dstOrigin options:MTLBlitOptionNone];
}

- (void)copyFromTexture:(id)src
                 sourceSlice:(NSUInteger)srcSlice
                 sourceLevel:(NSUInteger)srcLevel
                sourceOrigin:(MTLOrigin)srcOrigin
                  sourceSize:(MTLSize)size
                    toBuffer:(id)dst
           destinationOffset:(NSUInteger)dstOff
      destinationBytesPerRow:(NSUInteger)dstPitch
    destinationBytesPerImage:(NSUInteger)dstImage
{
    [self copyFromTexture:src sourceSlice:srcSlice sourceLevel:srcLevel sourceOrigin:srcOrigin sourceSize:size
                 toBuffer:dst destinationOffset:dstOff destinationBytesPerRow:dstPitch
        destinationBytesPerImage:dstImage options:MTLBlitOptionNone];
}

- (void)copyFromTexture:(id)src
            sourceSlice:(NSUInteger)srcSlice
            sourceLevel:(NSUInteger)srcLevel
              toTexture:(id)dst
       destinationSlice:(NSUInteger)dstSlice
       destinationLevel:(NSUInteger)dstLevel
             sliceCount:(NSUInteger)sliceCount
             levelCount:(NSUInteger)levelCount
{
    id<MTLTexture> a = src;
    for (NSUInteger l = 0; l < levelCount; l++) {
        NSUInteger at = srcLevel + l;
        MTLSize size = MTLSizeMake(MAX(a.width >> at, 1u), MAX(a.height >> at, 1u), MAX(a.depth >> at, 1u));
        for (NSUInteger s = 0; s < sliceCount; s++)
            [self copyFromTexture:src sourceSlice:srcSlice + s sourceLevel:at sourceOrigin:(MTLOrigin){ 0, 0, 0 }
                       sourceSize:size toTexture:dst destinationSlice:dstSlice + s destinationLevel:dstLevel + l
                destinationOrigin:(MTLOrigin){ 0, 0, 0 }];
    }
}

- (void)copyFromTexture:(id)src toTexture:(id)dst
{
    id<MTLTexture> a = src, b = dst;
    NSUInteger levels = MIN(a.mipmapLevelCount, b.mipmapLevelCount);
    NSUInteger slices = MIN(((MalachiteTexture *)src)->_image ? ((MalachiteTexture *)src)->_layers : a.arrayLength,
                            ((MalachiteTexture *)dst)->_image ? ((MalachiteTexture *)dst)->_layers : b.arrayLength);
    [self copyFromTexture:src sourceSlice:0 sourceLevel:0 toTexture:dst destinationSlice:0 destinationLevel:0
               sliceCount:slices levelCount:levels];
}

- (void)copyFromBuffer:(id)src
           sourceOffset:(NSUInteger)srcOff
      sourceBytesPerRow:(NSUInteger)srcPitch
    sourceBytesPerImage:(NSUInteger)srcImage
             sourceSize:(MTLSize)size
              toTexture:(id)dst
       destinationSlice:(NSUInteger)dstSlice
       destinationLevel:(NSUInteger)dstLevel
      destinationOrigin:(MTLOrigin)dstOrigin
                options:(MTLBlitOption)options
{
    int aspect = malachite_aspect(dst, options);
    malachite_mips_written(dst, dstLevel);
    malachite_texture_written(_cb, dst);
    malachite_keep(_cb, src);
    malachite_keep(_cb, dst);
    malachite_depth_kind(_cb, dst);
    for (NSUInteger z = 0; z < size.depth; z++) {
        struct malachite_sub b;
        if (!malachite_sub(dst, dstLevel, dstSlice, dstOrigin.z + z, &b))
            return;
        NSUInteger w = (size.width + b.block - 1) / b.block, h = (size.height + b.block - 1) / b.block;
        struct malachite_side a = { [(MTLIOAccelBuffer *)src gpuAddress] + srcOff + z * srcImage, (uint32_t)srcPitch };
        struct malachite_side to = malachite_side_of(dst, &b, dstOrigin.x, dstOrigin.y);
        if (aspect == MALACHITE_ASPECT_ALL)
            [self copy:a to:to bytes:w * b.bpp lines:h remap:0];
        else
            [self copy:a to:to texels:w lines:h aspect:aspect toTexture:true];
    }
}

- (void)copyFromTexture:(id)src
                 sourceSlice:(NSUInteger)srcSlice
                 sourceLevel:(NSUInteger)srcLevel
                sourceOrigin:(MTLOrigin)srcOrigin
                  sourceSize:(MTLSize)size
                    toBuffer:(id)dst
           destinationOffset:(NSUInteger)dstOff
      destinationBytesPerRow:(NSUInteger)dstPitch
    destinationBytesPerImage:(NSUInteger)dstImage
                     options:(MTLBlitOption)options
{
    int aspect = malachite_aspect(src, options);
    malachite_keep(_cb, src);
    malachite_keep(_cb, dst);
    malachite_written(_cb, dst);
    malachite_depth_kind(_cb, src);
    for (NSUInteger z = 0; z < size.depth; z++) {
        struct malachite_sub a;
        if (!malachite_sub(src, srcLevel, srcSlice, srcOrigin.z + z, &a))
            return;
        NSUInteger w = (size.width + a.block - 1) / a.block, h = (size.height + a.block - 1) / a.block;
        struct malachite_side from = malachite_side_of(src, &a, srcOrigin.x, srcOrigin.y);
        struct malachite_side b = { [(MTLIOAccelBuffer *)dst gpuAddress] + dstOff + z * dstImage, (uint32_t)dstPitch };
        if (aspect == MALACHITE_ASPECT_ALL)
            [self copy:from to:b bytes:w * a.bpp lines:h remap:0];
        else
            [self copy:from to:b texels:w lines:h aspect:aspect toTexture:false];
    }
}

// only Managed resources have a VRAM copy, the rest is shared memory
- (void)synchronizeResource:(id)resource
{
    MalachiteBuffer *b = [resource isKindOfClass:[MalachiteTexture class]] ? ((MalachiteTexture *)resource)->_backing
                         : [resource isKindOfClass:[MalachiteHeapBuffer class]]
                             ? ((MalachiteHeapBuffer *)resource)->_heap->_block
                             : resource;
    if (![b isKindOfClass:[MalachiteBuffer class]] || !b->_mirror)
        return;
    struct malachite_copy_buf2buf p = {};
    p.srcAddr = [b->_mirror gpuAddress];
    p.dstAddr = [b systemAddress];
    p.size = b.length;
    malachite_keep(_cb, b);
    malachite_keep(_cb, b->_mirror);
    malachite_record(_cb, MALACHITE_CMD_COPY_BUF2BUF, &p, sizeof(p));
}

- (void)synchronizeTexture:(id)texture slice:(NSUInteger)slice level:(NSUInteger)level
{
    [self synchronizeResource:texture];
}

// An sRGB texture is written through a plain view with the kernel encoding, as stores don't.
- (void)generateMipmapsForTexture:(id)texture
{
    MalachiteTexture *t = texture;
    malachite_mips_written(t, 1);
    if (!t->_image || t->_levels < 2)
        return;
    MTLPixelFormat f = t->_format;
    switch (f) {
    case MTLPixelFormatRGBA8Unorm_sRGB: f = MTLPixelFormatRGBA8Unorm; break;
    case MTLPixelFormatBGRA8Unorm_sRGB: f = MTLPixelFormatBGRA8Unorm; break;
    default: break;
    }
    int tf = malachite_format(f);
    if (t->_zt_format || tf < 0 || tex_format_info((enum tex_format)tf).block > 1 ||
        tex_format_info((enum tex_format)tf).integer) {
        os_log(OS_LOG_DEFAULT, "Malachite: mipmaps of texture type %lu format %lu not generated", (unsigned long)t->_type,
               (unsigned long)t->_format);
        return;
    }
    id<MTLDevice> device = ((id<MTLCommandBuffer>)_cb).device;
    id<MTLComputePipelineState> ps = malachite_mip_pipeline(device, t->_type == MTLTextureType3D);
    static id<MTLSamplerState> linear;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        linear = [device newSamplerStateWithDescriptor:sd];
    });
    if (!ps)
        return;
    malachite_keep(_cb, texture);
    uint64_t buffers[31] = {};
    uint32_t srgb = f != t->_format;
    malachite_upload(_cb, &srgb, sizeof(srgb), &buffers[0]);
    __unsafe_unretained id views[NVGPU_MAX_TEXTURES] = {}, samplers[NVGPU_MAX_TEXTURES] = { linear };
    for (uint32_t layer = 0; layer < t->_layers; layer++)
        for (uint32_t l = 1; l < t->_levels; l++) {
            for (int i = 0; i < 2; i++) {
                id view = [texture newTextureViewWithPixelFormat:i ? f : t->_format
                                                     textureType:t->_type == MTLTextureType3D ? MTLTextureType3D
                                                                                              : MTLTextureType2D
                                                          levels:NSMakeRange(l - 1 + i, 1)
                                                          slices:NSMakeRange(layer, 1)];
                malachite_keep(_cb, view);
                views[i] = view;
            }
            NSUInteger w = MAX(t.width >> l, 1u), h = MAX(t.height >> l, 1u), d = MAX(t.depth >> l, 1u);
            malachite_dispatch(_cb, (MalachiteComputePipelineState *)ps, buffers, views, samplers,
                           MTLSizeMake((w + 7) / 8, (h + 7) / 8, t->_type == MTLTextureType3D ? d : 1),
                           MTLSizeMake(8, 8, 1));
        }
}

- (void)optimizeContentsForGPUAccess:(id)texture
{
}
@end

// the base does the bookkeeping; the driver supplies the limits
@interface _MTLComputePipelineState : NSObject
- (instancetype)initWithDevice:(id)device pipelineStateDescriptor:(id)descriptor;
@end

@interface MalachiteComputePipelineState : _MTLComputePipelineState {
@public
    struct nvk_header _kernel;
    id<MTLBuffer> _code;
    NSString *_label;
    NSUInteger _asked;    // the descriptor's maxTotalThreadsPerThreadgroup, 0 for none
    bool _cl;             // arguments bound from MALACHITE_CL_FIRST
}
@end

enum { MALACHITE_CL_FIRST = 3 };  // where OpenCL's renderer binds a kernel's first argument

@implementation MalachiteComputePipelineState
// Past what the SM's four 16K-register partitions hold, at 8-register units a thread, the launch faults.
- (NSUInteger)maxTotalThreadsPerThreadgroup
{
    NSUInteger regs = (MAX(_kernel.reg_count, 1u) + 7) & ~7u, n = MIN(1024u, 16384 / (regs * 32) * 4 * 32);
    // MPS sizes its threadgroup arrays to the kernel's own limit
    if (_kernel.max_threads)
        n = MIN(n, _kernel.max_threads);
    return _asked ? MIN(n, _asked) : n;
}
- (NSUInteger)threadExecutionWidth { return 32; }
// rounded to setThreadgroupMemoryLength's 16 bytes, as Apple's is
- (NSUInteger)staticThreadgroupMemoryLength { return (_kernel.shared_size + 15) & ~15u; }
@end

@interface _MTLRenderPipelineState : NSObject
- (instancetype)initWithDevice:(id)device pipelineStateDescriptor:(id)descriptor;
@end

#define MALACHITE_DYNAMIC_STRIDE UINT32_MAX

@interface MalachiteRenderPipelineState : _MTLRenderPipelineState {
@public
    struct nvk_header _vertex, _fragment;
    id<MTLBuffer> _vcode, _fcode;  // SPH then SASS
    uint32_t _write_mask[8], _blend[8][7];  // as nvgpu_draw carries them
    uint32_t _alpha_control;
    uint32_t _attributes[32];
    uint32_t _strides[16];            // 0, or MALACHITE_DYNAMIC_STRIDE for the one it's bound with
    uint32_t _divisors[16];           // 0 per vertex
    uint8_t _stream_buffer[16];
    int _zero_stream;                 // 1 + the stream that reads zeros for attributes it can't fetch, 0 for none
    NSString *_label;
    // post-tessellation vertex functions only; init stages by control point count
    uint32_t _tess;
    bool _tess_scaled;
    struct nvk_header _tv;
    id<MTLBuffer> _tvcode;
    struct airc_tess_desc _td;
    id<MTLBuffer> _ticode[33];
    uint32_t _ti_regs[33];
}
@end

static id<MTLBuffer> malachite_tess_stage(id<MTLDevice> device, const struct airc_tess_desc *d, struct nvk_header *h,
                                      NSString **why);

@implementation MalachiteRenderPipelineState
// generated on first use; the function's own control point count wins
- (id<MTLBuffer>)tessInit:(NSUInteger)points
                   device:(id<MTLDevice>)device
                     regs:(uint32_t *)regs
                      why:(NSString **)why
{
    NSUInteger n = _td.points ? _td.points : points;
    if (!n || n > 32) {
        *why = [NSString stringWithFormat:@"patches of %lu control points", (unsigned long)n];
        return nil;
    }
    @synchronized(self) {
        if (!_ticode[n]) {
            struct airc_tess_desc d = _td;
            d.points = (uint32_t)n;
            struct nvk_header h;
            _ticode[n] = malachite_tess_stage(device, &d, &h, why);
            _ti_regs[n] = h.reg_count;
        }
        *regs = _ti_regs[n];
        return _ticode[n];
    }
}
@end

// 64 KB rows of RGBA32Float over the whole buffer, which is sized to a whole number of them
static MalachiteTexture *malachite_image_over(MalachiteBuffer *buf, MTLTextureUsage usage)
{
    NSUInteger length = ((id<MTLBuffer>)buf).length, row = MIN(length, (NSUInteger)65536);
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                 width:row / 16
                                                                                height:length / row
                                                                             mipmapped:NO];
    d.usage = usage;
    MalachiteTexture *t = [buf newTextureWithDescriptor:d offset:0 bytesPerRow:row];
    if (t)
        t->_image = true;
    return t;
}

static bool malachite_is_cube(MTLTextureType type)
{
    return type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray;
}

static void malachite_heap_init(id<MTLDevice> device);
static uint32_t malachite_heap_keep(void);
static void malachite_heap_release(uint32_t e);
static uint32_t *malachite_heap_header(uint32_t e);
static uint32_t malachite_heap_sampler(const uint32_t samp[8]);
static int malachite_fill_texture_header(uint32_t out[8], id texture, id cb);

@implementation MalachiteTexture
- (MTLTextureUsage)usage
{
    return _usage;
}

- (BOOL)canGenerateMipmapLevels
{
    return NO;
}

- (MTLTextureType)textureType
{
    return _type;
}

- (id)newTextureViewWithPixelFormat:(MTLPixelFormat)format
{
    return [self newTextureViewWithPixelFormat:format textureType:_type
                                        levels:NSMakeRange(0, ((id<MTLTexture>)self).mipmapLevelCount)
                                        slices:NSMakeRange(0, _image ? _layers : 1)];
}

static MalachiteTexture *malachite_swizzled(MalachiteTexture *t, MTLTextureSwizzleChannels swizzle)
{
    if (!t || (swizzle.red == MTLTextureSwizzleRed && swizzle.green == MTLTextureSwizzleGreen &&
               swizzle.blue == MTLTextureSwizzleBlue && swizzle.alpha == MTLTextureSwizzleAlpha))
        return t;
    t->_swizzled = true;
    t->_swizzle[0] = swizzle.red;
    t->_swizzle[1] = swizzle.green;
    t->_swizzle[2] = swizzle.blue;
    t->_swizzle[3] = swizzle.alpha;
    return t;
}

// formats must match in size per pixel or block
- (id)newTextureViewWithPixelFormat:(MTLPixelFormat)format
                        textureType:(MTLTextureType)type
                             levels:(NSRange)levels
                             slices:(NSRange)slices
{
    MTLPixelFormat mine = ((id<MTLTexture>)self).pixelFormat;
    bool fits = _image ? levels.length && slices.length && levels.location + levels.length <= _levels &&
                             slices.location + slices.length <= _layers
                       : !levels.location && !slices.location;
    if (!_backing || !fits || malachite_bytes_per_pixel(format) != malachite_bytes_per_pixel(mine) ||
        malachite_block(format) != malachite_block(mine)) {
        os_log(OS_LOG_DEFAULT, "MalachiteTexture: unsupported view (format %lu of %lu, levels %lu+%lu, slices %lu+%lu)",
               (unsigned long)format, (unsigned long)mine, (unsigned long)levels.location,
               (unsigned long)levels.length, (unsigned long)slices.location, (unsigned long)slices.length);
        return nil;
    }
    MalachiteTexture *t;
    if (_image) {
        t = malachite_image_over((MalachiteBuffer *)_backing, self.usage);
        if (t) {
            t->_offset = _offset + slices.location * _layer_stride;
            t->_pitch = _pitch;
            t->_format = format;
            t->_width = _width;
            t->_height = _height;
            t->_depth = _depth;
            t->_samples = _samples;
            t->_layers = (uint32_t)slices.length;
            t->_image_levels = _image_levels;
            t->_view_min = _view_min + (uint32_t)levels.location;
            t->_levels = (uint32_t)levels.length;
            t->_layer_stride = _layer_stride;
            t->_zt_format = _zt_format;
            t->_zt_bpp = _zt_bpp;
            t->_zt_kind = _zt_kind;
        }
    } else {
        // as IOSurface textures are, in case the base won't take these rows
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                     width:[self width]
                                                                                    height:1
                                                                                 mipmapped:NO];
        d.usage = self.usage;
        t = [(MalachiteBuffer *)_backing newTextureWithDescriptor:d offset:_offset bytesPerRow:(_pitch + 63) & ~63u];
        if (t) {
            t->_pitch = _pitch;
            t->_linear_height = _linear_height ? _linear_height : [super height];
        }
    }
    if (t) {
        t->_owned = self;
        t->_type = type;
        // a moved texture's view answering Private let malachite_depth_kind compress over the CPU's data
        t->_asked = _asked;
        t->_moved = _moved;
        t->_swizzled = _swizzled;
        memcpy(t->_swizzle, _swizzle, sizeof(_swizzle));
    }
    return t;
}

- (id)newTextureViewWithPixelFormat:(MTLPixelFormat)format
                        textureType:(MTLTextureType)type
                             levels:(NSRange)levels
                             slices:(NSRange)slices
                            swizzle:(MTLTextureSwizzleChannels)swizzle
{
    return malachite_swizzled([self newTextureViewWithPixelFormat:format textureType:type levels:levels slices:slices],
                          swizzle);
}

- (id)newTextureViewWithDescriptor:(MTLTextureViewDescriptor *)descriptor
{
    return [self newTextureViewWithPixelFormat:descriptor.pixelFormat textureType:descriptor.textureType
                                        levels:descriptor.levelRange slices:descriptor.sliceRange
                                       swizzle:descriptor.swizzle];
}

- (NSUInteger)width
{
    return _image ? MAX(_width >> _view_min, 1u) : _width ? _width : [super width];
}

- (NSUInteger)height
{
    if (_image)
        return _type == MTLTextureType1D ? 1 : MAX(_height >> _view_min, 1u);
    return _type != MTLTextureType2D ? 1 : _linear_height ? _linear_height : [super height];
}

- (NSUInteger)depth
{
    return _image && _type == MTLTextureType3D ? MAX(_depth >> _view_min, 1u) : 1;
}

- (MTLPixelFormat)pixelFormat
{
    return _image ? _format : [super pixelFormat];
}

- (NSUInteger)mipmapLevelCount
{
    return _image ? _levels : 1;
}

- (NSUInteger)sampleCount
{
    return _samples ? _samples : 1;
}

- (NSUInteger)arrayLength
{
    if (_image)
        return malachite_is_cube(_type) ? _layers / 6
               : _type == MTLTextureType2DArray || _type == MTLTextureType2DMultisampleArray ? _layers
                                                                                              : 1;
    return _type != MTLTextureType1DArray ? 1 : _linear_height ? _linear_height : [super height];
}

- (IOSurfaceRef)iosurface
{
    return _surface;
}

- (MTLStorageMode)storageMode
{
    if (_moved)
        return _asked;
    return _backing ? ((id<MTLBuffer>)_backing).storageMode : MTLStorageModeShared;
}

- (NSUInteger)iosurfacePlane
{
    return _plane;
}

- (id<MTLHeap>)heap
{
    return _placed.heap;
}

- (NSUInteger)heapOffset
{
    return _placed.heapOffset;
}

- (NSUInteger)allocatedSize
{
    return _placed ? _placed.allocatedSize : [super allocatedSize];
}

- (void)makeAliasable
{
    [_placed makeAliasable];
}

- (BOOL)isAliasable
{
    return _placed.isAliasable;
}

- (void)dealloc
{
    if (_surface)
        CFRelease(_surface);
    if (_heap_entry)
        malachite_heap_release(_heap_entry);
}

// the descriptor heap entry argument buffers hold
- (MTLResourceID)gpuResourceID
{
    if (!_heap_entry) {
        malachite_heap_init(self.device);
        uint32_t e = malachite_heap_keep();
        if (e && !malachite_fill_texture_header(malachite_heap_header(e), self, nil))
            _heap_entry = e;
        else if (e)
            malachite_heap_release(e);
    }
    return (MTLResourceID){ _heap_entry };
}

- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
            withBytes:(const void *)bytes
          bytesPerRow:(NSUInteger)stride
{
    [self replaceRegion:region mipmapLevel:level slice:0 withBytes:bytes bytesPerRow:stride bytesPerImage:0];
}

- (void)copyRegion:(MTLRegion)region level:(NSUInteger)level slice:(NSUInteger)slice bytes:(uint8_t *)bytes
            stride:(NSUInteger)stride image:(NSUInteger)image in:(bool)in
{
    id<MTLBuffer> backing = _backing;
    uint8_t *pixels = (uint8_t *)backing.contents;
    struct malachite_sub s;
    if (!malachite_sub(self, level, slice, region.origin.z, &s))
        return;
    NSUInteger w = (region.size.width + s.block - 1) / s.block, h = (region.size.height + s.block - 1) / s.block;
    if (pixels && backing.storageMode != MTLStorageModePrivate) {
        for (NSUInteger z = 0; z < region.size.depth && malachite_sub(self, level, slice, region.origin.z + z, &s); z++)
            malachite_copy_sub(pixels, &s, region.origin.x / s.block, region.origin.y / s.block, w, h,
                               bytes + z * image, stride, in);
        return;
    }
    // A texture in VRAM: the region alone, through a buffer the GPU copies it to or from. A private buffer can still
    // have pages the CPU maps, which the GPU never reads (GL's textures written there were sampled as zeros). Taking
    // the whole surface and putting it back undid the tiles other threads uploaded meanwhile. It also wrote a render
    // target before its first use gave the memory a compressed kind, which then read those bytes as garbage; the blit
    // asks for the kind before it copies.
    NSUInteger row = w * (malachite_aspect(self, MTLBlitOptionNone) == MALACHITE_ASPECT_STENCIL ? 1 : s.bpp);
    NSUInteger size = row * h;
    id<MTLBuffer> staging = [backing.device newBufferWithLength:MAX(size * region.size.depth, 1u)
                                                        options:MTLResourceStorageModeShared];
    if (!staging)
        return;
    uint8_t *rows = (uint8_t *)staging.contents;
    for (NSUInteger z = 0; in && z < region.size.depth; z++)
        for (NSUInteger y = 0; y < h; y++)
            memcpy(rows + z * size + y * row, bytes + z * image + y * stride, row);
    static id<MTLCommandQueue> queue;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        queue = [backing.device newCommandQueue];
    });
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (in)
        [blit copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:row sourceBytesPerImage:size
                  sourceSize:region.size toTexture:self destinationSlice:slice destinationLevel:level
           destinationOrigin:region.origin];
    else
        [blit copyFromTexture:self sourceSlice:slice sourceLevel:level sourceOrigin:region.origin
                     sourceSize:region.size toBuffer:staging destinationOffset:0 destinationBytesPerRow:row
        destinationBytesPerImage:size];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    for (NSUInteger z = 0; !in && z < region.size.depth; z++)
        for (NSUInteger y = 0; y < h; y++)
            memcpy(bytes + z * image + y * stride, rows + z * size + y * row, row);
}

- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
                slice:(NSUInteger)slice
            withBytes:(const void *)bytes
          bytesPerRow:(NSUInteger)stride
        bytesPerImage:(NSUInteger)imageStride
{
    [self copyRegion:region level:level slice:slice bytes:(uint8_t *)bytes stride:stride image:imageStride in:true];
    malachite_mips_written(self, level);
    malachite_mip_fill(self, region, level, slice, (const uint8_t *)bytes, stride);
    MalachiteBuffer *backing = _backing;
    if ([backing isKindOfClass:[MalachiteBuffer class]] && backing->_mirror) {
        // an image's rows aren't in order and a 1D array's slice is a row, so those go whole
        NSUInteger lo = 0, hi = backing.length;
        if (!_image && !slice) {
            lo = MIN(_offset + region.origin.y * _pitch, hi);
            hi = MIN(_offset + (region.origin.y + region.size.height) * _pitch, hi);
        }
        malachite_mark_dirty(backing, lo, hi);
    }
}

- (void)getBytes:(void *)bytes bytesPerRow:(NSUInteger)stride fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level
{
    [self getBytes:bytes bytesPerRow:stride bytesPerImage:0 fromRegion:region mipmapLevel:level slice:0];
}

- (void)getBytes:(void *)bytes
      bytesPerRow:(NSUInteger)stride
    bytesPerImage:(NSUInteger)imageStride
       fromRegion:(MTLRegion)region
      mipmapLevel:(NSUInteger)level
            slice:(NSUInteger)slice
{
    [self copyRegion:region level:level slice:slice bytes:(uint8_t *)bytes stride:stride image:imageStride in:false];
}
@end

@interface _MTLIndirectArgumentBufferLayout : NSObject
@end

@interface MTLStructMember (Malachite)
- (void)setOffset:(NSUInteger)offset;
@end

// An argument buffer's layout, the way airc reads the struct: members in index order at their natural alignment,
// textures and samplers (their gpuResourceIDs) and buffers (their addresses) 8 bytes each, arrays element after
// element. Every index an encoder sets goes to one of these.
struct malachite_arg {
    NSUInteger index, count;  // the first index, how many (an array's elements)
    NSUInteger offset, stride;
    MTLDataType type;  // of an element
};

@interface MalachiteArgumentLayout : _MTLIndirectArgumentBufferLayout {
@public
    struct malachite_arg *_args;
    NSUInteger _nargs, _length, _alignment;
}
@end

// a member's size and alignment in the struct
static NSUInteger malachite_arg_size(MTLDataType t, NSUInteger *align)
{
    switch (t) {
    case MTLDataTypeBool: case MTLDataTypeChar: case MTLDataTypeUChar: *align = 1; return 1;
    case MTLDataTypeHalf: case MTLDataTypeShort: case MTLDataTypeUShort: *align = 2; return 2;
    case MTLDataTypeHalf2: case MTLDataTypeShort2: case MTLDataTypeUShort2: *align = 4; return 4;
    case MTLDataTypeFloat: case MTLDataTypeInt: case MTLDataTypeUInt: *align = 4; return 4;
    case MTLDataTypeHalf3: case MTLDataTypeHalf4: case MTLDataTypeShort4: case MTLDataTypeUShort4: *align = 8; return 8;
    case MTLDataTypeFloat2: case MTLDataTypeInt2: case MTLDataTypeUInt2: *align = 8; return 8;
    case MTLDataTypeFloat3: case MTLDataTypeFloat4: case MTLDataTypeInt3: case MTLDataTypeInt4: case MTLDataTypeUInt3:
    case MTLDataTypeUInt4: *align = 16; return 16;
    case MTLDataTypeFloat4x4: *align = 16; return 64;
    default: *align = 8; return 8;  // a texture, sampler or buffer: an ID or address
    }
}

@implementation MalachiteArgumentLayout
- (void)setStructType:(MTLStructType *)type withDevice:(id)device
{
    NSArray<MTLStructMember *> *members = [type.members sortedArrayUsingComparator:^(MTLStructMember *a,
                                                                                     MTLStructMember *b) {
        return a.argumentIndex < b.argumentIndex ? NSOrderedAscending : NSOrderedDescending;
    }];
    free(_args);
    _args = (struct malachite_arg *)calloc(members.count ? members.count : 1, sizeof(*_args));
    _nargs = 0;
    _length = 0;
    _alignment = 8;
    for (MTLStructMember *m in members) {
        struct malachite_arg *a = &_args[_nargs++];
        MTLArrayType *array = m.dataType == MTLDataTypeArray ? m.arrayType : nil;
        NSUInteger align;
        a->index = m.argumentIndex;
        a->count = array ? array.arrayLength : 1;
        a->type = array ? array.elementType : m.dataType;
        a->stride = malachite_arg_size(a->type, &align);
        a->offset = (_length + align - 1) & ~(align - 1);
        _length = a->offset + a->stride * a->count;
        _alignment = MAX(_alignment, align);
        [m setOffset:a->offset];
    }
    _length = (_length + _alignment - 1) & ~(_alignment - 1);
}

- (NSUInteger)encodedLength
{
    return _length;
}

- (NSUInteger)alignment
{
    return _alignment;
}

- (void)dealloc
{
    free(_args);
}
@end

// Writes resources into an argument buffer as its layout says: a texture's or sampler's gpuResourceID, a buffer's
// address, constants in place.
@interface MalachiteArgumentEncoder : NSObject <MTLArgumentEncoder> {
    MalachiteArgumentLayout *_layout;
    uint8_t *_at;  // the bound argument buffer's bytes at its offset
}
@property (readonly) id<MTLDevice> device;
@property (copy, atomic) NSString *label;
- (instancetype)initWithLayout:(MalachiteArgumentLayout *)layout device:(id<MTLDevice>)device;
@end

@implementation MalachiteArgumentEncoder
- (instancetype)initWithLayout:(MalachiteArgumentLayout *)layout device:(id<MTLDevice>)device
{
    self = [super init];
    if (self) {
        _layout = layout;
        _device = device;
    }
    return self;
}

- (NSUInteger)encodedLength
{
    return _layout->_length;
}

- (NSUInteger)alignment
{
    return _layout->_alignment;
}

- (void)setArgumentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset
{
    _at = buffer ? (uint8_t *)buffer.contents + offset : NULL;
}

- (void)setArgumentBuffer:(id<MTLBuffer>)buffer startOffset:(NSUInteger)offset arrayElement:(NSUInteger)element
{
    [self setArgumentBuffer:buffer offset:offset + element * _layout->_length];
}

// where argument `index` goes, NULL for one the layout lacks
- (uint8_t *)slot:(NSUInteger)index
{
    for (NSUInteger i = 0; i < _layout->_nargs && _at; i++) {
        const struct malachite_arg *a = &_layout->_args[i];
        if (index >= a->index && index < a->index + a->count)
            return _at + a->offset + (index - a->index) * a->stride;
    }
    return NULL;
}

- (void)put:(uint64_t)value at:(NSUInteger)index
{
    uint8_t *p = [self slot:index];
    if (p)
        memcpy(p, &value, 8);
}

- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
    [self put:texture ? texture.gpuResourceID._impl : 0 at:index];
}

- (void)setTextures:(const id<MTLTexture> __unsafe_unretained *)textures withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setTexture:textures[i] atIndex:range.location + i];
}

- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
    [self put:sampler ? sampler.gpuResourceID._impl : 0 at:index];
}

- (void)setSamplerStates:(const id<MTLSamplerState> __unsafe_unretained *)samplers withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setSamplerState:samplers[i] atIndex:range.location + i];
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    [self put:buffer ? buffer.gpuAddress + offset : 0 at:index];
}

- (void)setBuffers:(const id<MTLBuffer> __unsafe_unretained *)buffers
           offsets:(const NSUInteger *)offsets
         withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setBuffer:buffers[i] offset:offsets[i] atIndex:range.location + i];
}

- (void *)constantDataAtIndex:(NSUInteger)index
{
    return [self slot:index];
}

- (id<MTLArgumentEncoder>)newArgumentEncoderForBufferAtIndex:(NSUInteger)index
{
    os_log(OS_LOG_DEFAULT, "Malachite: nested argument buffer encoders not supported");
    return nil;
}

// what an argument buffer can hold that we don't make: left as they are
- (void)setRenderPipelineState:(id)pipeline atIndex:(NSUInteger)index
{
}

- (void)setRenderPipelineStates:(const id __unsafe_unretained *)pipelines withRange:(NSRange)range
{
}

- (void)setComputePipelineState:(id)pipeline atIndex:(NSUInteger)index
{
}

- (void)setComputePipelineStates:(const id __unsafe_unretained *)pipelines withRange:(NSRange)range
{
}

- (void)setIndirectCommandBuffer:(id)buffer atIndex:(NSUInteger)index
{
}

- (void)setIndirectCommandBuffers:(const id __unsafe_unretained *)buffers withRange:(NSRange)range
{
}

- (void)setAccelerationStructure:(id)structure atIndex:(NSUInteger)index
{
}

- (void)setVisibleFunctionTable:(id)table atIndex:(NSUInteger)index
{
}

- (void)setVisibleFunctionTables:(const id __unsafe_unretained *)tables withRange:(NSRange)range
{
}

- (void)setIntersectionFunctionTable:(id)table atIndex:(NSUInteger)index
{
}

- (void)setIntersectionFunctionTables:(const id __unsafe_unretained *)tables withRange:(NSRange)range
{
}

- (void)setDepthStencilState:(id)state atIndex:(NSUInteger)index
{
}

- (void)setDepthStencilStates:(const id __unsafe_unretained *)states withRange:(NSRange)range
{
}
@end

// Sampler base (private). initWithDevice:samplerDescriptor: is real; we keep the descriptor's filtering.
@interface _MTLSamplerState : NSObject
- (instancetype)initWithDevice:(id)device samplerDescriptor:(id)descriptor;
@end

@interface MalachiteSamplerState : _MTLSamplerState {
@public
    struct samp_desc _desc;
}
@end
@implementation MalachiteSamplerState
// what an argument buffer holds for the sampler: its descriptor heap entry, as a handle's sampler bits
- (MTLResourceID)gpuResourceID
{
    uint32_t samp[8];
    malachite_heap_init(((id<MTLSamplerState>)self).device);
    tex_fill_sampler(samp, &_desc);
    return (MTLResourceID){ (uint64_t)malachite_heap_sampler(samp) << 20 };
}
@end

// Depth/stencil state base (private); we keep the descriptor in the 3D class's values.
@interface _MTLDepthStencilState : NSObject
- (instancetype)initWithDevice:(id)device depthStencilDescriptor:(id)descriptor;
@end

@interface MalachiteDepthStencilState : _MTLDepthStencilState {
@public
    uint32_t _depth_test, _depth_write, _depth_func;
    uint32_t _stencil_test;
    uint32_t _stencil[2][4];  // front, back: op fail, op zfail, op zpass, func
    uint32_t _read_mask[2], _write_mask[2];
}
@end
@implementation MalachiteDepthStencilState
// the object's address, as the base sampler state answers
- (uint64_t)uniqueIdentifier
{
    return (uint64_t)(__bridge void *)self;
}

- (MTLResourceID)gpuResourceID
{
    return (MTLResourceID){ [self uniqueIdentifier] };
}
@end

// Build a TIC from a bound texture's buffer backing and its descriptor. A 1D texture is a 2D one a row high, a 1D
// array one row per layer (airc samples them that way). An image's view of one level of one layer (or of a 3D
// texture) is that surface on its own, as surface stores don't look at the view's levels; any other is the whole
// image with the levels it sees.
static int malachite_fill_texture_header(uint32_t out[8], id texture, id cb)
{
    MalachiteTexture *t = (MalachiteTexture *)texture;
    id<MTLTexture> mt = (id<MTLTexture>)texture;
    int f = malachite_format(mt.pixelFormat);
    if (f < 0)
        return -1;
    struct tex_desc td = {};
    td.addr = [(MTLIOAccelBuffer *)t->_backing gpuAddress] + t->_offset;
    td.format = (enum tex_format)f;
    td.swizzled = t->_swizzled;
    memcpy(td.swizzle, t->_swizzle, sizeof(td.swizzle));
    td.width = (uint32_t)mt.width;
    td.height = mt.textureType == MTLTextureType1DArray ? (uint32_t)mt.arrayLength : (uint32_t)mt.height;
    td.pitch = t->_pitch;
    if (!t->_image) {
        if (((td.addr | td.pitch) & 31) && cb) {
            uint32_t bytes = td.width * tex_format_info(td.format).bytes, pitch = (bytes + 255) & ~255u;
            if (!t->_shadow)
                t->_shadow = [mt.device newBufferWithLength:(NSUInteger)pitch * td.height
                                                    options:MTLResourceStorageModePrivate];
            struct nvgpu_copy2d c = { td.addr, [(MTLIOAccelBuffer *)t->_shadow gpuAddress], td.pitch, pitch, bytes,
                                      td.height };
            malachite_record(cb, MALACHITE_CMD_COPY_2D, &c, sizeof(c));
            malachite_keep(cb, t->_shadow);
            td.addr = c.dst;
            td.pitch = pitch;
        }
        tex_fill_header(out, &td);
        return 0;
    }
    td.blocklinear = 1;
    bool cube = malachite_is_cube(mt.textureType);
    td.type = mt.textureType == MTLTextureType3D ? TEX_TYPE_3D
              : cube ? (t->_layers > 6 ? TEX_TYPE_CUBE_ARRAY : TEX_TYPE_CUBE)
              : t->_layers > 1 || mt.textureType == MTLTextureType2DArray ||
                        mt.textureType == MTLTextureType2DMultisampleArray
                    ? TEX_TYPE_2D_ARRAY
                    : TEX_TYPE_2D;
    td.samples = t->_samples;
    td.depth = td.type == TEX_TYPE_3D ? (uint32_t)mt.depth : cube ? t->_layers / 6 : t->_layers;
    if (t->_levels == 1 && (t->_layers == 1 || td.type == TEX_TYPE_3D)) {
        struct malachite_sub s;
        malachite_sub(t, 0, 0, 0, &s);
        td.addr = [(MTLIOAccelBuffer *)t->_backing gpuAddress] + s.offset;
        tex_fill_header(out, &td);
        return 0;
    }
    td.width = (uint32_t)t->_width;
    td.height = (uint32_t)t->_height;
    if (td.type == TEX_TYPE_3D)
        td.depth = t->_depth;
    td.levels = t->_image_levels;
    td.view_min = t->_view_min;
    td.view_levels = t->_levels;
    tex_fill_header(out, &td);
    return 0;
}

// A function's buffer arguments go to const bank 0 at the offsets airc gave them: params + (cbuf_off - 0x160), and
// so does the address of its constant data, after its code at code.
// The size can come out past capacity bytes (a draw's 20 buffers), which aren't written: the caller refuses it.
static uint32_t malachite_fill_params(const struct nvk_header *h, const uint64_t buffers[31], uint64_t code,
                                  uint8_t *params, uint32_t capacity)
{
    uint32_t size = 0;
    uint64_t data = code + h->sass_size;
    for (uint32_t i = 0; i < h->nargs; i++) {
        const struct nvk_arg *a = &h->args[i];
        if (a->kind != NVK_ARG_BUFFER && a->kind != NVK_ARG_STAGE_IN_BUFFER && a->kind != NVK_ARG_DATA)
            continue;
        uint32_t at = a->cbuf_off - 0x160;
        if (at + 8 <= capacity)
            memcpy(params + at, a->kind == NVK_ARG_DATA ? &data : &buffers[a->index], 8);
        if (at + 8 > size)
            size = at + 8;
    }
    return size;
}

// setBytes: the bytes go after the command buffer's earlier ones in a block of its own, and the block goes back for
// another command buffer once this one is done. A buffer per call would cost a kernel resource, a GPU mapping and an
// unmap (which waits for every draw in flight).
@interface MalachiteUploads : NSObject {
@public
    NSMutableArray<id<MTLBuffer>> *_blocks;
    NSUInteger _used;  // in the last block
}
@end

@implementation MalachiteUploads
@end

static const NSUInteger MALACHITE_UPLOAD_BLOCK = 256 << 10;
static NSMutableArray<id<MTLBuffer>> *malachite_upload_free;

static id<MTLBuffer> malachite_reserve(__unsafe_unretained id commandBuffer, NSUInteger length, uint64_t *va, void **cpu)
{
    __unsafe_unretained MalachiteUploads *u = ((MalachiteCommandBuffer *)commandBuffer)->_uploads;
    if (!u) {
        MalachiteUploads *made = [MalachiteUploads new];
        made->_blocks = [NSMutableArray array];
        ((MalachiteCommandBuffer *)commandBuffer)->_uploads = made;
        u = made;
        NSMutableArray<id<MTLBuffer>> *blocks = u->_blocks;
        [(id<MTLCommandBuffer>)commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            @synchronized([MalachiteUploads class]) {
                for (id<MTLBuffer> b in blocks)
                    if (b.length == MALACHITE_UPLOAD_BLOCK)
                        [malachite_upload_free addObject:b];
            }
        }];
    }
    NSUInteger at = (u->_used + 255) & ~(NSUInteger)255;
    id<MTLBuffer> buf = u->_blocks.lastObject;
    if (!buf || at + length > buf.length) {
        buf = nil;
        if (length <= MALACHITE_UPLOAD_BLOCK)
            @synchronized([MalachiteUploads class]) {
                if (!malachite_upload_free)
                    malachite_upload_free = [NSMutableArray array];
                buf = malachite_upload_free.lastObject;
                if (buf)
                    [malachite_upload_free removeLastObject];
            }
        if (!buf)
            buf = [((id<MTLCommandBuffer>)commandBuffer).device newBufferWithLength:MAX(length, MALACHITE_UPLOAD_BLOCK)
                                                                          options:MTLResourceStorageModeShared];
        [u->_blocks addObject:buf];
        at = 0;
    }
    *cpu = (uint8_t *)buf.contents + at;
    u->_used = at + length;
    *va = [(MTLIOAccelBuffer *)buf gpuAddress] + at;
    return buf;
}

static id<MTLBuffer> malachite_upload(id commandBuffer, const void *bytes, NSUInteger length, uint64_t *va)
{
    void *cpu;
    id<MTLBuffer> buf = malachite_reserve(commandBuffer, length, va, &cpu);
    memcpy(cpu, bytes, length);
    return buf;
}

// what a slot with no sampler bound reads with: nearest, clamped to the edge, pixel coordinates
static struct samp_desc malachite_default_sampler(void)
{
    return (struct samp_desc){ .address = { TEX_CLAMP_TO_EDGE, TEX_CLAMP_TO_EDGE, TEX_CLAMP_TO_EDGE } };
}

static float malachite_half(uint16_t bits)
{
    __fp16 h;
    memcpy(&h, &bits, 2);
    return h;
}

// an LOD clamp as the sampler's 4.8 fixed point
static uint16_t malachite_lod(float lod)
{
    return lod <= 0 ? 0 : lod >= 15.99f ? 0xfff : (uint16_t)(lod * 256);
}

// The sampler with an encoder's LOD clamps in place of its own: one made per pair of clamps, kept with the sampler.
static char malachite_clamped_key;

static id malachite_clamped(MalachiteSamplerState *sampler, float lodMin, float lodMax)
{
    uint16_t lo = malachite_lod(lodMin), hi = malachite_lod(lodMax);
    if (!sampler || (sampler->_desc.lod_min == lo && sampler->_desc.lod_max == hi))
        return sampler;
    @synchronized(sampler) {
        NSMutableDictionary *made = objc_getAssociatedObject(sampler, &malachite_clamped_key);
        if (!made) {
            made = [NSMutableDictionary dictionary];
            objc_setAssociatedObject(sampler, &malachite_clamped_key, made, OBJC_ASSOCIATION_RETAIN);
        }
        NSNumber *key = @(lo | hi << 16);
        MalachiteSamplerState *s = made[key];
        if (!s) {
            MTLSamplerDescriptor *d = [objc_getAssociatedObject(sampler, &malachite_desc_key) copy];
            if (!d)
                d = [MTLSamplerDescriptor new];
            d.lodMinClamp = lodMin;
            d.lodMaxClamp = lodMax;
            s = [[MalachiteSamplerState alloc] initWithDevice:((id<MTLSamplerState>)sampler).device samplerDescriptor:d];
            s->_desc = sampler->_desc;
            s->_desc.lod_min = lo;
            s->_desc.lod_max = hi;
            if (malachite_rec_only)
                objc_setAssociatedObject(s, &malachite_desc_key, d, OBJC_ASSOCIATION_RETAIN);
            made[key] = s;
        }
        return s;
    }
}

// A constexpr sampler's packed state: the s, t and r address modes in bits 0-2, 3-5 and 6-8 (0 clamp to the border
// colour, 1 clamp to edge, 2 repeat, 3 mirrored repeat, 4 mirror clamp to edge); mag and min linear in bits 9 and
// 11, the mip filter in 13-14 (MTLSamplerMipFilter), pixel coordinates in 15, the compare function in 16-19
// (compare_func's order, 8 for none), max anisotropy - 1 in 20-23, the LOD clamps as halves in 24-39 and 40-55, the
// border colour in 56-57 (MTLSamplerBorderColor).
static struct samp_desc malachite_const_sampler(uint64_t bits)
{
    static const enum tex_address address[] = { TEX_BORDER, TEX_CLAMP_TO_EDGE, TEX_WRAP, TEX_MIRROR,
                                                TEX_MIRROR_ONCE_CLAMP_TO_EDGE, TEX_BORDER, TEX_BORDER, TEX_BORDER };
    struct samp_desc s = {};
    for (int i = 0; i < 3; i++)
        s.address[i] = address[bits >> 3 * i & 7];
    s.mag_linear = bits >> 9 & 1;
    s.min_linear = bits >> 11 & 1;
    s.mip = (enum tex_mip)(bits >> 13 & 3);
    s.normalized = !(bits >> 15 & 1);
    // never, less, less_equal, greater, greater_equal, equal, not_equal, always, as MTLCompareFunction
    static const uint8_t compare_fn[8] = { 0, 1, 3, 4, 6, 2, 5, 7 };
    unsigned compare = bits >> 16 & 15;
    s.compare = compare < 8 ? compare_fn[compare] + 1 : 0;
    s.max_anisotropy = (bits >> 20 & 15) + 1;
    s.lod_min = malachite_lod(malachite_half(bits >> 24 & 0xffff));
    s.lod_max = malachite_lod(malachite_half(bits >> 40 & 0xffff));
    s.border = (enum tex_border)(bits >> 56 & 3);
    return s;
}

// A sampler as the function samples with it: a depth compare's has DEPTH_COMPARE set, with never as its function
// when it has none (Metal's default, under which no compare passes), and a plain sample's has it clear, as the
// texture unit wants each; a slot both kinds use keeps the compare.
static struct samp_desc malachite_compare_as_used(const struct nvk_header *h, unsigned slot, struct samp_desc s)
{
    if (h->compare_samplers >> slot & 1) {
        if (!s.compare)
            s.compare = MTLCompareFunctionNever + 1;
    } else if (h->plain_samplers >> slot & 1) {
        s.compare = 0;
    }
    if (h->nearest_samplers >> slot & 1) {
        s.mag_linear = s.min_linear = 0;
        s.max_anisotropy = 1;
        if (s.mip == TEX_MIP_LINEAR)
            s.mip = TEX_MIP_NEAREST;
    }
    return s;
}

// The bound textures and samplers as pool descriptors, slot for slot, plus the function's constexpr samplers in
// the slots airc gave them. -1 on a pixel format we can't describe.
// What a shader's unbound texture slot reads: Metal leaves it undefined, but a stale or zero header points the
// texture unit at memory that's no longer mapped, which takes the channel down. One 1x1 texture, kept for good.
static id malachite_unbound_texture(id<MTLDevice> device)
{
    static id texture;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:1
                                                                                    height:1
                                                                                 mipmapped:NO];
        texture = [device newTextureWithDescriptor:d];
    });
    return texture;
}

// What the attributes a pipeline can't fetch read: a stream of zeros
static id<MTLBuffer> malachite_zero_buffer(id<MTLDevice> device)
{
    static id<MTLBuffer> buffer;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        buffer = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    });
    return buffer;
}

// A depth texture's memory takes its PTE kind (the generic one included: buffers are made pitch) when it's first a
// depth target: until then the texture unit faults reading it as that format (UNSUPPORTED_KIND, or an MMU fault at
// 0x200000 for a ZF32 one uploaded by a blit). So the first time such a texture is read or written otherwise, a
// pass that draws nothing makes it a depth target first, once for its memory.
static char malachite_kind_key;  // on a depth texture's memory once it has its kind

static void malachite_compress_colour(id cb, MalachiteBuffer *b)
{
    if (b->_compress_asked)
        return;
    b->_compress_asked = true;
    struct malachite_compress c = { [(MTLIOAccelBuffer *)b gpuAddress] };
    malachite_record(cb, MALACHITE_CMD_COMPRESS, &c, sizeof(c));
}

static void malachite_depth_kind(id cb, MalachiteTexture *t)
{
    struct malachite_sub s;
    if ([t isKindOfClass:[MalachiteTexture class]] && !t->_zt_format && t->_image && t->_backing &&
        ((MalachiteBuffer *)t->_backing)->_one_image && !((MalachiteBuffer *)t->_backing)->_compress_asked &&
        (t.usage & MTLTextureUsageRenderTarget) && t.storageMode == MTLStorageModePrivate)
        malachite_compress_colour(cb, (MalachiteBuffer *)t->_backing);
    if (![t isKindOfClass:[MalachiteTexture class]] || !t->_zt_format || !t->_backing ||
        objc_getAssociatedObject(t->_backing, &malachite_kind_key) || !malachite_sub(t, 0, 0, 0, &s))
        return;
    objc_setAssociatedObject(t->_backing, &malachite_kind_key, @YES, OBJC_ASSOCIATION_RETAIN);
    struct nvgpu_draw d = {};
    d.width = s.width;
    d.height = s.height;
    d.samples = t->_samples;
    d.zt = [(MTLIOAccelBuffer *)t->_backing gpuAddress] + s.offset;
    d.zt_format = t->_zt_format;
    d.zt_width = s.pitch / t->_zt_bpp;
    d.zt_height = s.height;
    d.zt_kind = t->_zt_kind;
    d.zt_compress = ((MalachiteBuffer *)t->_backing)->_one_image;
    malachite_record(cb, MALACHITE_CMD_DRAW, &d, sizeof(d));
}

// A stage's textures and samplers in the pools' 16 slots, from its bindings by Metal index: one bound past them (MPS
// binds its sources at 64) is in the slot the compiler gave it, its cbuf_off. Only an app's encoder's tables go
// past 16.
static void malachite_slots(const struct nvk_header *h, __unsafe_unretained const id *textures,
                            __unsafe_unretained const id *samplers, __unsafe_unretained id *slot_tex,
                            __unsafe_unretained id *slot_samp)
{
    for (int i = 0; i < NVGPU_MAX_TEXTURES; i++) {
        slot_tex[i] = textures[i];
        slot_samp[i] = samplers[i];
    }
    for (uint32_t i = 0; i < h->nargs; i++) {
        const struct nvk_arg *a = &h->args[i];
        if (a->index < NVGPU_MAX_TEXTURES || a->cbuf_off >= NVGPU_MAX_TEXTURES)
            continue;
        if (a->kind == NVK_ARG_TEXTURE)
            slot_tex[a->cbuf_off] = textures[a->index];
        else if (a->kind == NVK_ARG_SAMPLER)
            slot_samp[a->cbuf_off] = samplers[a->index];
    }
}

static int malachite_fill_pools(__unsafe_unretained id cb, const struct nvk_header *h, __unsafe_unretained const id *textures,
                            __unsafe_unretained const id *samplers, unsigned int headers[][8],
                            unsigned int samps[][8], unsigned int *nheaders, unsigned int *nsamplers)
{
    for (uint32_t i = 0; i < h->nargs; i++) {  // every slot the shader reads gets a valid header and sampler
        unsigned slot = h->args[i].index;
        if (slot >= NVGPU_MAX_TEXTURES)
            continue;
        if (h->args[i].kind == NVK_ARG_TEXTURE && !textures[slot]) {
            // left all zero, which no header is: the heap gives it entry 0, the unbound texture's, which
            // is_null_texture() checks for
            memset(headers[slot], 0, 32);
            if (slot + 1 > *nheaders)
                *nheaders = slot + 1;
        } else if (h->args[i].kind == NVK_ARG_SAMPLER && !samplers[slot]) {
            struct samp_desc def = malachite_compare_as_used(h, slot, malachite_default_sampler());
            tex_fill_sampler(samps[slot], &def);
            if (slot + 1 > *nsamplers)
                *nsamplers = slot + 1;
        }
    }
    for (int i = 0; i < NVGPU_MAX_TEXTURES; i++) {
        if (textures[i]) {
            malachite_depth_kind(cb, textures[i]);
            if (malachite_fill_texture_header(headers[i], textures[i], cb)) {
                os_log(OS_LOG_DEFAULT, "Malachite: unsupported texture pixel format");
                return -1;
            }
            if (h->face_textures >> i & 1)
                tex_header_faces_as_layers(headers[i]);
            if (i + 1 > (int)*nheaders)
                *nheaders = i + 1;
        }
        if (samplers[i]) {
            struct samp_desc s = malachite_compare_as_used(h, i, ((MalachiteSamplerState *)samplers[i])->_desc);
            tex_fill_sampler(samps[i], &s);
            if (i + 1 > (int)*nsamplers)
                *nsamplers = i + 1;
        }
    }
    if (*nsamplers == 0 && *nheaders) {  // a read uses sampler slot 0; give the pool a valid default there
        struct samp_desc def = malachite_default_sampler();
        tex_fill_sampler(samps[0], &def);
        *nsamplers = 1;
    }
    for (uint32_t i = 0; i < h->nconst_samplers; i++) {
        unsigned slot = NVK_CONST_SAMPLER_SLOT(i);
        struct samp_desc s = malachite_compare_as_used(h, slot, malachite_const_sampler(h->const_samplers[i]));
        tex_fill_sampler(samps[slot], &s);
        if (slot + 1 > *nsamplers)
            *nsamplers = slot + 1;
    }
    return 0;
}

// The process's descriptor heap: the texture headers and samplers every launch's pools are, which shaders name by
// index. A sampler takes an entry per distinct state, for good. A texture an argument buffer names (by its
// gpuResourceID) has an entry of its own for its life, among the first MALACHITE_HEAP_KEPT; a launch's bound textures
// take entries from blocks of the rest, which a command buffer takes as it needs them and gives back once done.
enum {
    MALACHITE_HEAP_HEADERS = 1 << 17,
    MALACHITE_HEAP_KEPT = 1 << 14,
    MALACHITE_HEAP_BLOCK = 256,
    MALACHITE_HEAP_BLOCKS = (MALACHITE_HEAP_HEADERS - MALACHITE_HEAP_KEPT) / MALACHITE_HEAP_BLOCK,
    MALACHITE_HEAP_SAMPLERS = 4096,
};

static struct {
    os_unfair_lock lock;
    id<MTLBuffer> headers, samplers;
    uint32_t nsamplers;
    uint8_t kept[MALACHITE_HEAP_KEPT / 8];  // which of the kept entries are taken (entry 0 never is)
    uint16_t free_blocks[MALACHITE_HEAP_BLOCKS];
    uint32_t nfree;
} malachite_heap;

static void malachite_heap_init(id<MTLDevice> device)
{
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        malachite_heap.headers = [device newBufferWithLength:MALACHITE_HEAP_HEADERS * 32 options:MTLResourceStorageModeShared];
        malachite_heap.samplers = [device newBufferWithLength:MALACHITE_HEAP_SAMPLERS * 32 options:MTLResourceStorageModeShared];
        malachite_heap.kept[0] = 1;  // entry 0 is the unbound texture, which unbound slots name
        malachite_fill_texture_header(malachite_heap_header(0), malachite_unbound_texture(device), nil);
        for (uint32_t b = 0; b < MALACHITE_HEAP_BLOCKS; b++)
            malachite_heap.free_blocks[malachite_heap.nfree++] = (uint16_t)(MALACHITE_HEAP_BLOCKS - 1 - b);
    });
}

// a kept header entry, 0 when they're all taken
static uint32_t malachite_heap_keep(void)
{
    uint32_t e = 0;
    os_unfair_lock_lock(&malachite_heap.lock);
    for (uint32_t i = 1; i < MALACHITE_HEAP_KEPT && !e; i++)
        if (!(malachite_heap.kept[i / 8] >> i % 8 & 1)) {
            malachite_heap.kept[i / 8] |= 1 << i % 8;
            e = i;
        }
    os_unfair_lock_unlock(&malachite_heap.lock);
    return e;
}

static void malachite_heap_release(uint32_t e)
{
    os_unfair_lock_lock(&malachite_heap.lock);
    malachite_heap.kept[e / 8] &= ~(1 << e % 8);
    os_unfair_lock_unlock(&malachite_heap.lock);
}

static uint32_t *malachite_heap_header(uint32_t e)
{
    return (uint32_t *)malachite_heap.headers.contents + 8 * e;
}

// the sampler entry for a sampler state, the same one for the same state; 0 when the heap is full
static uint32_t malachite_heap_sampler(const uint32_t samp[8])
{
    uint32_t (*s)[8] = (uint32_t (*)[8])malachite_heap.samplers.contents, e = 0;
    os_unfair_lock_lock(&malachite_heap.lock);
    while (e < malachite_heap.nsamplers && memcmp(s[e], samp, 32))
        e++;
    if (e == MALACHITE_HEAP_SAMPLERS) {
        os_log(OS_LOG_DEFAULT, "Malachite: descriptor heap out of samplers");
        e = 0;
    } else if (e == malachite_heap.nsamplers) {
        memcpy(s[e], samp, 32);
        malachite_heap.nsamplers++;
    }
    os_unfair_lock_unlock(&malachite_heap.lock);
    return e;
}

// a command buffer's header blocks, given back when it completes
@interface MalachiteHeapBlocks : NSObject {
@public
    uint16_t _blocks[MALACHITE_HEAP_BLOCKS];
    uint32_t _n, _used;  // blocks, entries of the last used
}
@end

@implementation MalachiteHeapBlocks
@end

// n consecutive header entries for one launch of cb's; 0 when the heap has none left
static uint32_t malachite_heap_take(__unsafe_unretained id cb, uint32_t n)
{
    __unsafe_unretained MalachiteHeapBlocks *b = ((MalachiteCommandBuffer *)cb)->_heap_blocks;
    if (!b) {
        MalachiteHeapBlocks *made = [MalachiteHeapBlocks new];
        ((MalachiteCommandBuffer *)cb)->_heap_blocks = made;
        b = made;
        [(id<MTLCommandBuffer>)cb addCompletedHandler:^(id<MTLCommandBuffer>) {
            os_unfair_lock_lock(&malachite_heap.lock);
            for (uint32_t i = 0; i < made->_n; i++)
                malachite_heap.free_blocks[malachite_heap.nfree++] = made->_blocks[i];
            os_unfair_lock_unlock(&malachite_heap.lock);
        }];
    }
    if (!b->_n || b->_used + n > MALACHITE_HEAP_BLOCK) {
        os_unfair_lock_lock(&malachite_heap.lock);
        bool got = malachite_heap.nfree > 0;
        if (got)
            b->_blocks[b->_n++] = malachite_heap.free_blocks[--malachite_heap.nfree];
        os_unfair_lock_unlock(&malachite_heap.lock);
        if (!got) {
            os_log(OS_LOG_DEFAULT, "Malachite: descriptor heap out of header blocks");
            return 0;
        }
        b->_used = 0;
    }
    uint32_t e = MALACHITE_HEAP_KEPT + b->_blocks[b->_n - 1] * MALACHITE_HEAP_BLOCK + b->_used;
    b->_used += n;
    return e;
}

// A launch's pools as the heap: its slots' headers copied into entries of cb's, its samplers' entries found, and
// the indices where its shaders read them. -1 when the heap is full.
static int malachite_heap_commit(id cb, unsigned int headers[][8], unsigned nheaders, unsigned int samps[][8],
                             unsigned nsamplers, struct nvgpu_heap *h)
{
    malachite_heap_init(((id<MTLCommandBuffer>)cb).device);
    malachite_keep(cb, malachite_heap.headers);
    malachite_keep(cb, malachite_heap.samplers);
    h->headers = [(MTLIOAccelBuffer *)malachite_heap.headers gpuAddress];
    h->samplers = [(MTLIOAccelBuffer *)malachite_heap.samplers gpuAddress];
    h->nheaders = MALACHITE_HEAP_HEADERS;
    h->nsamplers = MALACHITE_HEAP_SAMPLERS;
    uint32_t e = nheaders ? malachite_heap_take(cb, nheaders) : 0;
    if (nheaders && !e)
        return -1;
    static const uint32_t none[8] = {};
    for (unsigned i = 0; i < nheaders; i++) {
        memcpy(malachite_heap_header(e + i), headers[i], 32);
        h->handles[i] = memcmp(headers[i], none, 32) ? e + i : 0;
    }
    for (unsigned i = 0; i < nsamplers; i++)
        h->handles[NVGPU_MAX_TEXTURES + i] = malachite_heap_sampler(samps[i]) << 20;
    return 0;
}

// A texture a kernel writes at levels that may be past the first gets a run of header entries, one a level, each a
// view of that level alone, and its handle names the run's start: the shader adds the level to it, as SUST addresses
// what a header describes and not a level of it. Layered images keep their one header, as a view of a layer's levels
// doesn't stride between layers as the whole does.
static void malachite_level_headers(id cb, const struct nvk_header *h, __unsafe_unretained const id *textures,
                                struct nvgpu_heap *heap)
{
    for (int i = 0; i < NVGPU_MAX_TEXTURES; i++) {
        MalachiteTexture *t = textures[i];
        if (!(h->level_textures >> i & 1) || !t || !t->_image || t->_levels < 2 ||
            (t->_layers > 1 && t->_type != MTLTextureType3D))
            continue;
        uint32_t e = malachite_heap_take(cb, t->_levels);
        if (!e)
            continue;
        for (uint32_t l = 0; l < t->_levels; l++) {
            id view = [t newTextureViewWithPixelFormat:t->_format textureType:t->_type levels:NSMakeRange(l, 1)
                                                slices:NSMakeRange(0, t->_layers)];
            malachite_keep(cb, view);  // which keeps it alive
            malachite_fill_texture_header(malachite_heap_header(e + l), view, cb);
        }
        heap->handles[i] = e;
    }
}

// Compute encoder base (private). Like the blit base, its encode methods are stubs, so never call super for them.
@interface MTLIOAccelComputeCommandEncoder : NSObject
- (instancetype)initWithCommandBuffer:(id)cb;
@end

#define MALACHITE_BINDINGS 128  // the texture and sampler indices a Metal stage binds

@interface MalachiteComputeCommandEncoder : MTLIOAccelComputeCommandEncoder <MTLComputeCommandEncoder> {
    __unsafe_unretained id _cb;
    MalachiteComputePipelineState *_pipeline;
    uint64_t _buffers[31];  // GPU VA (+offset) bound at each [[buffer(n)]] slot
    __unsafe_unretained id _bound[31];  // and the buffer, for setBufferOffset
    // by Metal index, which goes to 127; malachite_dispatch puts them in the pools' 16 slots
    __unsafe_unretained id _textures[MALACHITE_BINDINGS];
    __unsafe_unretained id _samplers[MALACHITE_BINDINGS];
    uint32_t _threadgroup[31];  // each [[threadgroup(n)]] argument's bytes
    uint64_t _region;  // the stage-in region's GPU VA, laid out as MTLStageInRegionIndirectArguments; 0 for none set
}
@end

@implementation MalachiteComputeCommandEncoder
- (instancetype)initWithCommandBuffer:(id)cb
{
    self = [super initWithCommandBuffer:cb];
    if (self)
        _cb = cb;
    return self;
}

// as on the blit encoder: dispatches already run in order, each after the work before it is done
- (void)updateFence:(id)fence
{
}

- (void)waitForFence:(id)fence
{
}

- (void)memoryBarrierWithScope:(MTLBarrierScope)scope
{
}

- (void)memoryBarrierWithResources:(const id<MTLResource> __unsafe_unretained *)resources count:(NSUInteger)count
{
}

- (void)barrierAfterQueueStages:(MTLStages)after beforeStages:(MTLStages)before
{
}

- (void)setComputePipelineState:(id)state
{
    _pipeline = state;
    malachite_keep(_cb, state);
}

- (void)setBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (index >= 31)
        return;
    _buffers[index] = buffer ? [(MTLIOAccelBuffer *)buffer gpuAddress] + offset : 0;
    _bound[index] = buffer;
    malachite_keep(_cb, buffer);
}

- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (index < 31)
        [self setBuffer:_bound[index] offset:offset atIndex:index];
}

- (void)setBuffers:(const id __unsafe_unretained *)buffers
           offsets:(const NSUInteger *)offsets
         withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setBuffer:buffers[i] offset:offsets[i] atIndex:range.location + i];
}

// the strides are compiled into a stage-in kernel, which refuses a layout whose stride is dynamic
- (void)setBuffer:(id)buffer offset:(NSUInteger)offset attributeStride:(NSUInteger)stride atIndex:(NSUInteger)index
{
    [self setBuffer:buffer offset:offset atIndex:index];
}

- (void)setBufferOffset:(NSUInteger)offset attributeStride:(NSUInteger)stride atIndex:(NSUInteger)index
{
    [self setBufferOffset:offset atIndex:index];
}

- (void)setBuffers:(const id __unsafe_unretained *)buffers
           offsets:(const NSUInteger *)offsets
  attributeStrides:(const NSUInteger *)strides
         withRange:(NSRange)range
{
    [self setBuffers:buffers offsets:offsets withRange:range];
}

- (void)setBytes:(const void *)bytes length:(NSUInteger)length attributeStride:(NSUInteger)stride atIndex:(NSUInteger)index
{
    [self setBytes:bytes length:length atIndex:index];
}

- (void)enableNullBufferBinds:(BOOL)enable
{
}

- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
    if (index < 31) {
        _bound[index] = malachite_upload(_cb, bytes, length, &_buffers[index]);
    }
}

- (void)setTextures:(const id __unsafe_unretained *)textures withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setTexture:textures[i] atIndex:range.location + i];
}

- (void)setSamplerStates:(const id __unsafe_unretained *)samplers withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setSamplerState:samplers[i] atIndex:range.location + i];
}

// resources the kernel reaches through pointers in other buffers (argument buffers): kept like bound ones
- (void)useResource:(id)resource usage:(MTLResourceUsage)usage
{
    malachite_keep(_cb, resource);
}

- (void)useResources:(const id __unsafe_unretained *)resources count:(NSUInteger)count usage:(MTLResourceUsage)usage
{
    for (NSUInteger i = 0; i < count; i++)
        malachite_keep(_cb, resources[i]);
}

- (void)useHeap:(id)heap
{
    malachite_keep(_cb, heap);
}

- (void)useHeaps:(const id __unsafe_unretained *)heaps count:(NSUInteger)count
{
    for (NSUInteger i = 0; i < count; i++)
        malachite_keep(_cb, heaps[i]);
}

- (void)setSamplerState:(id)sampler lodMinClamp:(float)lodMin lodMaxClamp:(float)lodMax atIndex:(NSUInteger)index
{
    [self setSamplerState:malachite_clamped(sampler, lodMin, lodMax) atIndex:index];
}

- (void)setSamplerStates:(const id __unsafe_unretained *)samplers
            lodMinClamps:(const float *)lodMin
            lodMaxClamps:(const float *)lodMax
               withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setSamplerState:samplers[i] lodMinClamp:lodMin[i] lodMaxClamp:lodMax[i] atIndex:range.location + i];
}

// [[threadgroup(n)]] memory, laid out at dispatch; Apple-GPU imageblocks have nothing to set
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
    if (index < 31)
        _threadgroup[index] = (uint32_t)length;
}

- (void)setImageblockWidth:(NSUInteger)width height:(NSUInteger)height
{
}

// a kernel's stage input reads its region from memory, so the indirect one is the same to it
- (void)setStageInRegion:(MTLRegion)region
{
    uint32_t r[6] = { (uint32_t)region.origin.x, (uint32_t)region.origin.y, (uint32_t)region.origin.z,
                      (uint32_t)region.size.width, (uint32_t)region.size.height, (uint32_t)region.size.depth };
    malachite_upload(_cb, r, sizeof(r), &_region);
}

- (void)setStageInRegionWithIndirectBuffer:(id)buffer indirectBufferOffset:(NSUInteger)offset
{
    _region = [(MTLIOAccelBuffer *)buffer gpuAddress] + offset;
    malachite_keep(_cb, buffer);
}

// The recorder's dispatch: the first one (or the one after +recskip's count) of each kernel +record names, its
// buffers (from their offsets, 256 MB at most) as they are before it and after it, into
// /tmp/malachite-rec-<pid>-c<n>/ with a manifest.txt of
// kernel=, grid= (threads, or groups when groups=1), threads= (a threadgroup's) and buffer.<n>=, buffer.<n>.after=
// lines. Textures aren't recorded.
- (NSString *)recordBuffers:(NSString *)dir manifest:(NSMutableString *)m after:(bool)after
{
    for (uint32_t i = 0; i < _pipeline->_kernel.nargs; i++) {
        const struct nvk_arg *a = &_pipeline->_kernel.args[i];
        if (a->kind != NVK_ARG_BUFFER || a->index >= 31 || !_bound[a->index])
            continue;
        id<MTLBuffer> b = _bound[a->index];
        uint64_t off = _buffers[a->index] - [(MTLIOAccelBuffer *)b gpuAddress];
        uint32_t size = b.length > off ? (uint32_t)MIN(b.length - off, 256ull << 20) : 0;
        NSString *file = [NSString stringWithFormat:@"buffer%u%s.bin", a->index, after ? "-after" : ""];
        if (malachite_rec_bytes(_cb, [dir stringByAppendingPathComponent:file], _buffers[a->index], size))
            [m appendFormat:@"buffer.%u%s=%@\n", a->index, after ? ".after" : "", file];
        // where the dispatch read it from: the storage asked for, and whether a managed one's VRAM copy
        if (!after)
            [m appendFormat:@"buffer.%u.storage=%lu%s\n", a->index, (unsigned long)b.storageMode,
                            [b isKindOfClass:[MalachiteBuffer class]] && ((MalachiteBuffer *)b)->_mirror ? ",vram" : ""];
    }
    return dir;
}

- (NSString *)recordStart:(NSMutableString *)m grid:(MTLSize)grid threads:(MTLSize)threads groups:(bool)groups
{
    static NSCountedSet<NSString *> *seen;
    bool named = false;
    for (NSString *s in malachite_rec_only)
        named |= [_pipeline->_label containsString:s];
    if (!named || malachite_rec_full)
        return nil;
    if (!seen)
        seen = [NSCountedSet set];
    [seen addObject:_pipeline->_label];
    if ([seen countForObject:_pipeline->_label] != malachite_rec_skip + 1)
        return nil;
    static int recordings;
    NSString *dir = [NSString stringWithFormat:@"/tmp/malachite-rec-%d-c%d", getpid(), recordings++];
    mkdir(dir.fileSystemRepresentation, 0755);
    [m appendFormat:@"version=1\nprocess=%s\npid=%d\nkernel=%@\ngrid=%lu,%lu,%lu\nthreads=%lu,%lu,%lu\ngroups=%d\n",
                    getprogname(), getpid(), _pipeline->_label, (unsigned long)grid.width,
                    (unsigned long)grid.height, (unsigned long)grid.depth, (unsigned long)threads.width,
                    (unsigned long)threads.height, (unsigned long)threads.depth, groups];
    return [self recordBuffers:dir manifest:m after:false];
}

- (void)recordEnd:(NSString *)dir manifest:(NSMutableString *)m
{
    [self recordBuffers:dir manifest:m after:true];
    NSString *path = [dir stringByAppendingPathComponent:@"manifest.txt"];
    [(id<MTLCommandBuffer>)_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        [m writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }];
}

// An OpenCL image's channel order << 16 | channel data type, as cl.h numbers them, from the texture its renderer made
// for it (the orders a one- or four-channel format doesn't say, from its swizzle); 0 for none
static uint32_t malachite_cl_format(__unsafe_unretained id texture)
{
    enum { R = 0x10b0, A, RG, RA, RGB, RGBA, BGRA, ARGB, INTENSITY, LUMINANCE, RGBX = 0x10bc, DEPTH,
           SRGBX = 0x10c0, SRGBA, SBGRA, ABGR };
    enum { SNORM8 = 0x10d0, SNORM16, UNORM8, UNORM16, UNORM565, UNORM555, UNORM101010, SINT8, SINT16, SINT32, UINT8,
           UINT16, UINT32, HALF, FLOAT };
    if (![texture conformsToProtocol:@protocol(MTLTexture)])
        return 0;
    uint32_t type, order;
    switch ([(id<MTLTexture>)texture pixelFormat]) {
    case MTLPixelFormatA8Unorm: return A << 16 | UNORM8;
    case MTLPixelFormatR8Unorm: type = UNORM8, order = R; break;
    case MTLPixelFormatR8Snorm: type = SNORM8, order = R; break;
    case MTLPixelFormatR8Uint: type = UINT8, order = R; break;
    case MTLPixelFormatR8Sint: type = SINT8, order = R; break;
    case MTLPixelFormatR16Unorm: type = UNORM16, order = R; break;
    case MTLPixelFormatR16Snorm: type = SNORM16, order = R; break;
    case MTLPixelFormatR16Uint: type = UINT16, order = R; break;
    case MTLPixelFormatR16Sint: type = SINT16, order = R; break;
    case MTLPixelFormatR16Float: type = HALF, order = R; break;
    case MTLPixelFormatR32Uint: type = UINT32, order = R; break;
    case MTLPixelFormatR32Sint: type = SINT32, order = R; break;
    case MTLPixelFormatR32Float: type = FLOAT, order = R; break;
    case MTLPixelFormatRG8Unorm: type = UNORM8, order = RG; break;
    case MTLPixelFormatRG8Snorm: type = SNORM8, order = RG; break;
    case MTLPixelFormatRG8Uint: type = UINT8, order = RG; break;
    case MTLPixelFormatRG8Sint: type = SINT8, order = RG; break;
    case MTLPixelFormatRG16Unorm: type = UNORM16, order = RG; break;
    case MTLPixelFormatRG16Snorm: type = SNORM16, order = RG; break;
    case MTLPixelFormatRG16Uint: type = UINT16, order = RG; break;
    case MTLPixelFormatRG16Sint: type = SINT16, order = RG; break;
    case MTLPixelFormatRG16Float: type = HALF, order = RG; break;
    case MTLPixelFormatRG32Uint: type = UINT32, order = RG; break;
    case MTLPixelFormatRG32Sint: type = SINT32, order = RG; break;
    case MTLPixelFormatRG32Float: type = FLOAT, order = RG; break;
    case MTLPixelFormatRGBA8Unorm: type = UNORM8, order = RGBA; break;
    case MTLPixelFormatRGBA8Unorm_sRGB: type = UNORM8, order = SRGBA; break;
    case MTLPixelFormatRGBA8Snorm: type = SNORM8, order = RGBA; break;
    case MTLPixelFormatRGBA8Uint: type = UINT8, order = RGBA; break;
    case MTLPixelFormatRGBA8Sint: type = SINT8, order = RGBA; break;
    case MTLPixelFormatBGRA8Unorm: return BGRA << 16 | UNORM8;
    case MTLPixelFormatBGRA8Unorm_sRGB: return SBGRA << 16 | UNORM8;
    case MTLPixelFormatRGBA16Unorm: type = UNORM16, order = RGBA; break;
    case MTLPixelFormatRGBA16Snorm: type = SNORM16, order = RGBA; break;
    case MTLPixelFormatRGBA16Uint: type = UINT16, order = RGBA; break;
    case MTLPixelFormatRGBA16Sint: type = SINT16, order = RGBA; break;
    case MTLPixelFormatRGBA16Float: type = HALF, order = RGBA; break;
    case MTLPixelFormatRGBA32Uint: type = UINT32, order = RGBA; break;
    case MTLPixelFormatRGBA32Sint: type = SINT32, order = RGBA; break;
    case MTLPixelFormatRGBA32Float: type = FLOAT, order = RGBA; break;
    case MTLPixelFormatB5G6R5Unorm: return RGB << 16 | UNORM565;
    case MTLPixelFormatBGR5A1Unorm: return RGB << 16 | UNORM555;
    case MTLPixelFormatBGR10A2Unorm:
    case MTLPixelFormatRGB10A2Unorm: return RGB << 16 | UNORM101010;
    case MTLPixelFormatDepth16Unorm: return DEPTH << 16 | UNORM16;
    case MTLPixelFormatDepth32Float: return DEPTH << 16 | FLOAT;
    default: return 0;
    }
    if (![texture isKindOfClass:[MalachiteTexture class]] || !((MalachiteTexture *)texture)->_swizzled)
        return order << 16 | type;
    const uint8_t *s = ((MalachiteTexture *)texture)->_swizzle;
    enum { Z = MTLTextureSwizzleZero, O = MTLTextureSwizzleOne, SR = MTLTextureSwizzleRed,
           SG = MTLTextureSwizzleGreen, SB = MTLTextureSwizzleBlue, SA = MTLTextureSwizzleAlpha };
#define SWIZZLE(x, y, z, w) (s[0] == (x) && s[1] == (y) && s[2] == (z) && s[3] == (w))
    if (order == R)
        order = SWIZZLE(SR, SR, SR, SR) ? INTENSITY : SWIZZLE(SR, SR, SR, O) ? LUMINANCE : SWIZZLE(Z, Z, Z, SR) ? A : R;
    else if (order == RG && SWIZZLE(SR, Z, Z, SG))
        order = RA;
    else if (order == RGBA)
        order = SWIZZLE(SR, SG, SB, O) ? RGBX : SWIZZLE(SG, SB, SA, SR) ? ARGB : SWIZZLE(SA, SB, SG, SR) ? ABGR
              : SWIZZLE(SB, SG, SR, SA) ? BGRA : RGBA;
    else if (order == SRGBA && SWIZZLE(SR, SG, SB, O))
        order = SRGBX;
#undef SWIZZLE
    return order << 16 | type;
}

// The address each [[buffer(n)]] param gets: the bound buffers', or for an OpenCL kernel the ones 3 further on.
// AppleMetalOpenGLRenderer binds buffer i at 3 + i (GLDQueueRec::executeKernel: setBuffer:offset:atIndex: for a
// buffer, setBytes:length:atIndex: for one passed by value, counting buffers alone), where the AIR names it i.
- (const uint64_t *)argumentAddresses:(uint64_t *)out
{
    if (!_pipeline || !_pipeline->_cl)
        return _buffers;
    memset(out, 0, sizeof(_buffers));
    for (uint32_t i = 0; i < _pipeline->_kernel.nargs; i++) {
        const struct nvk_arg *a = &_pipeline->_kernel.args[i];
        if (a->kind == NVK_ARG_TEXTURE && a->access && a->index < MALACHITE_BINDINGS &&
            [_textures[a->index] isKindOfClass:[MalachiteTexture class]])  // an image over one of OpenCL's buffers
            malachite_written(_cb, ((MalachiteTexture *)_textures[a->index])->_backing);
        // the global work offset and work_dim, which the renderer binds at 0
        if (a->kind == NVK_ARG_BUFFER && a->index == AIRC_CL_OFFSETS) {
            static const uint32_t none[4] = { 0, 0, 0, 1 };
            if (_buffers[0])
                out[a->index] = _buffers[0];
            else
                malachite_upload(_cb, none, sizeof(none), &out[a->index]);
            continue;
        }
        if (a->kind == NVK_ARG_BUFFER && a->index == AIRC_CL_FORMATS) {
            uint32_t formats[MALACHITE_BINDINGS];
            NSUInteger n = 1;
            for (NSUInteger t = 0; t < MALACHITE_BINDINGS; t++)
                if ((formats[t] = malachite_cl_format(_textures[t])))
                    n = t + 1;
            malachite_upload(_cb, formats, 4 * n, &out[a->index]);
            continue;
        }
        if (a->kind != NVK_ARG_BUFFER || a->index + MALACHITE_CL_FIRST >= 31)
            continue;
        out[a->index] = _buffers[a->index + MALACHITE_CL_FIRST];
        if (a->access)
            malachite_written(_cb, _bound[a->index + MALACHITE_CL_FIRST]);
    }
    return out;
}

// the kernel reads the grid when it runs the dispatch, after the work before it, which may write it
- (void)dispatchThreadgroupsWithIndirectBuffer:(id)buffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)threads
{
    malachite_keep(_cb, buffer);
    uint64_t cl[31];
    malachite_dispatch(_cb, _pipeline, [self argumentAddresses:cl], _textures, _samplers, MTLSizeMake(1, 1, 1), threads,
                   [(MTLIOAccelBuffer *)buffer gpuAddress] + offset, _threadgroup, NULL, _region);
}

- (void)setTexture:(id)texture atIndex:(NSUInteger)index
{
    if (index >= MALACHITE_BINDINGS)
        return;
    _textures[index] = texture;
    malachite_keep(_cb, texture);
    malachite_texture_written(_cb, texture);
}

- (void)setSamplerState:(id)sampler atIndex:(NSUInteger)index
{
    if (index < MALACHITE_BINDINGS)
        _samplers[index] = sampler;
}

- (void)dispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads
{
    NSMutableString *m = malachite_rec_only && _pipeline ? [NSMutableString string] : nil;
    NSString *rec = m ? [self recordStart:m grid:groups threads:threads groups:true] : nil;
    uint64_t cl[31];
    malachite_dispatch(_cb, _pipeline, [self argumentAddresses:cl], _textures, _samplers, groups, threads, 0,
                   _threadgroup, NULL, _region);
    if (rec)
        [self recordEnd:rec manifest:m];
}

// A grid that isn't whole threadgroups (Metal's non-uniform threadgroups) runs as up to 8 uniform launches: the
// whole threadgroups, and the partial ones along each edge as launches of their own, smaller, threadgroup size;
// each knows where it lies in the dispatch, so thread positions and counts are the dispatch's.
- (void)dispatchThreads:(MTLSize)grid threadsPerThreadgroup:(MTLSize)threads
{
    if (!grid.width || !grid.height || !grid.depth || !threads.width || !threads.height || !threads.depth)
        return;
    NSUInteger g[3] = { grid.width, grid.height, grid.depth }, t[3] = { threads.width, threads.height, threads.depth };
    NSUInteger whole[3], rest[3];
    uint32_t part[12];
    for (int i = 0; i < 3; i++) {
        whole[i] = g[i] / t[i];
        rest[i] = g[i] % t[i];
        part[6 + i] = (uint32_t)(whole[i] + (rest[i] != 0));  // the dispatch's threadgroups
        part[9 + i] = (uint32_t)g[i];                          // and threads
    }
    NSMutableString *m = malachite_rec_only && _pipeline ? [NSMutableString string] : nil;
    NSString *rec = m ? [self recordStart:m grid:grid threads:threads groups:false] : nil;
    uint64_t cl[31];
    const uint64_t *buffers = [self argumentAddresses:cl];
    for (int k = 0; k < 8; k++) {  // bit i of k: the partial threadgroups along axis i rather than the whole ones
        NSUInteger n[3], b[3];
        bool empty = false;
        for (int i = 0; i < 3; i++) {
            bool edge = k >> i & 1;
            n[i] = edge ? (rest[i] ? 1 : 0) : whole[i];
            b[i] = edge ? rest[i] : t[i];
            part[i] = (uint32_t)(edge ? whole[i] * t[i] : 0);
            part[3 + i] = (uint32_t)(edge ? whole[i] : 0);
            empty |= !n[i];
        }
        if (!empty)
            malachite_dispatch(_cb, _pipeline, buffers, _textures, _samplers, MTLSizeMake(n[0], n[1], n[2]),
                           MTLSizeMake(b[0], b[1], b[2]), 0, _threadgroup, part, _region);
    }
    if (rec)
        [self recordEnd:rec manifest:m];
}
@end

static void malachite_dispatch(id cb, MalachiteComputePipelineState *pipeline, const uint64_t buffers[31],
                           __unsafe_unretained const id *textures, __unsafe_unretained const id *samplers,
                           MTLSize groups, MTLSize threads, uint64_t indirect, const uint32_t *threadgroup,
                           const uint32_t *part, uint64_t region)
{
    // the QMD holds at most 65535 threadgroups along y and z: a larger grid runs in pieces, each knowing where it lies
    if (!indirect && (groups.height > 65535 || groups.depth > 65535)) {
        uint32_t whole[12] = {};
        if (part) {
            memcpy(whole, part, sizeof(whole));
        } else {
            NSUInteger n[3] = { groups.width, groups.height, groups.depth }, t[3] = { threads.width, threads.height, threads.depth };
            for (int i = 0; i < 3; i++) {
                whole[6 + i] = (uint32_t)n[i];
                whole[9 + i] = (uint32_t)(n[i] * t[i]);
            }
        }
        for (NSUInteger z = 0; z < groups.depth; z += 65535)
            for (NSUInteger y = 0; y < groups.height; y += 65535) {
                uint32_t p[12];
                memcpy(p, whole, sizeof(p));
                p[1] += (uint32_t)(y * threads.height);
                p[2] += (uint32_t)(z * threads.depth);
                p[4] += (uint32_t)y;
                p[5] += (uint32_t)z;
                malachite_dispatch(cb, pipeline, buffers, textures, samplers,
                               MTLSizeMake(groups.width, MIN(65535u, groups.height - y), MIN(65535u, groups.depth - z)),
                               threads, 0, threadgroup, p, region);
            }
        return;
    }
    if (!pipeline) {  // a pipeline that failed to build, used anyway (OpenGL's renderer doesn't check)
        os_log(OS_LOG_DEFAULT, "Malachite: dispatch without a compute pipeline dropped");
        return;
    }
    struct nvgpu_dispatch d = {};
    d.indirect = indirect;
    d.prog_va = [(MTLIOAccelBuffer *)pipeline->_code gpuAddress];
    d.grid_x = (uint32_t)groups.width;
    d.grid_y = (uint32_t)groups.height;
    d.grid_z = (uint32_t)groups.depth;
    d.block_x = (uint32_t)threads.width;
    d.block_y = (uint32_t)threads.height;
    d.block_z = (uint32_t)threads.depth;
    if (part) {  // origin, group origin, the dispatch's groups and threads
        memcpy(d.origin, part, sizeof(d.origin));
        memcpy(d.group_origin, part + 3, sizeof(d.group_origin));
        memcpy(d.groups, part + 6, sizeof(d.groups));
        memcpy(d.threads, part + 9, sizeof(d.threads));
    }
    d.reg_count = pipeline->_kernel.reg_count;
    d.shared_size = pipeline->_kernel.shared_size;
    d.local_size = pipeline->_kernel.local_size;
    d.barriers = pipeline->_kernel.barriers;
    d.params_size = malachite_fill_params(&pipeline->_kernel, buffers, d.prog_va, d.params, sizeof(d.params));
    // [[threadgroup(n)]] memory after the kernel's own threadgroup variables, each at the offset its param says. An
    // OpenCL __local argument is aligned to its type, up to 128 bytes for a long16.
    uint32_t align = pipeline->_cl ? 128 : 16, slack = 0;
    for (uint32_t i = 0; i < pipeline->_kernel.nargs; i++) {
        const struct nvk_arg *a = &pipeline->_kernel.args[i];
        if (a->kind != NVK_ARG_THREADGROUP)
            continue;
        uint32_t at = (d.shared_size + align - 1) & ~(align - 1);
        uint32_t len = threadgroup && a->index < 31 ? threadgroup[a->index] : 0;
        memcpy(d.params + a->cbuf_off - 0x160, &at, 4);
        d.params_size = MAX(d.params_size, a->cbuf_off - 0x160 + 8u);
        d.shared_size = at + ((len + 15) & ~15u);
        // and a slack after the last: kernels read a little past their threadgroup buffers, which Apple's GPUs
        // allow and an SM takes as an out-of-range address (one framework's temporal scaler reads over 256 bytes past)
        slack = 1024;
    }
    if (d.shared_size + slack <= [((id<MTLCommandBuffer>)cb).device maxThreadgroupMemoryLength])
        d.shared_size += slack;
    for (uint32_t i = 0; i < pipeline->_kernel.nargs; i++) {
        const struct nvk_arg *a = &pipeline->_kernel.args[i];
        if (a->kind != NVK_ARG_STAGE_IN_REGION)
            continue;
        // a region never set reads as empty rather than through a null pointer, which faults the channel
        if (!region) {
            static const uint32_t empty[6] = {};
            malachite_upload(cb, empty, sizeof(empty), &region);
        }
        memcpy(d.params + a->cbuf_off - 0x160, &region, 8);
        d.params_size = MAX(d.params_size, a->cbuf_off - 0x160 + 8u);
    }
    __unsafe_unretained id slot_tex[NVGPU_MAX_TEXTURES], slot_samp[NVGPU_MAX_TEXTURES];
    malachite_slots(&pipeline->_kernel, textures, samplers, slot_tex, slot_samp);
    if (malachite_fill_pools(cb, &pipeline->_kernel, slot_tex, slot_samp, d.headers,
                         d.samplers, &d.nheaders, &d.nsamplers) ||
        malachite_heap_commit(cb, d.headers, d.nheaders, d.samplers, d.nsamplers, &d.heap))
        return;
    malachite_level_headers(cb, &pipeline->_kernel, slot_tex, &d.heap);
    if (malachite_trace) {
        os_log(OS_LOG_DEFAULT, "Malachite trace: dispatch %{public}@: %ux%ux%u groups of %lux%lux%lu", pipeline->_label,
               d.grid_x, d.grid_y, d.grid_z, (unsigned long)threads.width, (unsigned long)threads.height,
               (unsigned long)threads.depth);
        for (uint32_t i = 0; i < pipeline->_kernel.nargs; i++) {
            const struct nvk_arg *a = &pipeline->_kernel.args[i];
            if (a->kind == NVK_ARG_BUFFER && a->index < 31) {
                os_log(OS_LOG_DEFAULT, "Malachite trace:   buffer %u: %#llx", a->index, buffers[a->index]);
            } else if (a->kind == NVK_ARG_THREADGROUP) {
                uint32_t at;
                memcpy(&at, d.params + a->cbuf_off - 0x160, 4);
                os_log(OS_LOG_DEFAULT, "Malachite trace:   threadgroup %u: at %u, %u bytes of %u", a->index, at,
                       threadgroup && a->index < 31 ? threadgroup[a->index] : 0, d.shared_size);
            } else if (a->kind == NVK_ARG_TEXTURE && a->index < NVGPU_MAX_TEXTURES && textures[a->index]) {
                MalachiteTexture *t = (MalachiteTexture *)textures[a->index];
                id<MTLTexture> mt = (id<MTLTexture>)textures[a->index];
                os_log(OS_LOG_DEFAULT, "Malachite trace:   texture %u: %#llx %lux%lu format %lu plane %lu%s", a->index,
                       [(MTLIOAccelBuffer *)t->_backing gpuAddress] + t->_offset, (unsigned long)mt.width,
                       (unsigned long)mt.height, (unsigned long)mt.pixelFormat, (unsigned long)mt.iosurfacePlane,
                       t->_surface ? " iosurface" : "");
            }
        }
    }
    malachite_record(cb, MALACHITE_CMD_DISPATCH, &d, sizeof(d));
}

// generateMipmaps' kernel, compiled once
static id<MTLComputePipelineState> malachite_mip_pipeline(id<MTLDevice> device, bool volume)
{
    static id<MTLComputePipelineState> ps[2];
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSString *src = @"#include <metal_stdlib>\n"
                         "using namespace metal;\n"
                         "static float encode(float c)\n"
                         "{\n"
                         "    c = saturate(c);\n"
                         "    return c <= 0.0031308f ? 12.92f * c : 1.055f * powr(c, 1.0f / 2.4f) - 0.055f;\n"
                         "}\n"
                         "kernel void malachite_mip(texture2d<float> src [[texture(0)]],\n"
                         "                      texture2d<float, access::write> dst [[texture(1)]],\n"
                         "                      sampler s [[sampler(0)]], constant uint &srgb [[buffer(0)]],\n"
                         "                      uint2 p [[thread_position_in_grid]])\n"
                         "{\n"
                         "    uint2 size = uint2(dst.get_width(), dst.get_height());\n"
                         "    if (p.x >= size.x || p.y >= size.y)\n"
                         "        return;\n"
                         "    float4 c = src.sample(s, (float2(p) + 0.5f) / float2(size), level(0));\n"
                         "    if (srgb)\n"
                         "        c = float4(encode(c.r), encode(c.g), encode(c.b), c.a);\n"
                         "    dst.write(c, p);\n"
                         "}\n"
                         "kernel void malachite_mip3d(texture3d<float> src [[texture(0)]],\n"
                         "                        texture3d<float, access::write> dst [[texture(1)]],\n"
                         "                        sampler s [[sampler(0)]], constant uint &srgb [[buffer(0)]],\n"
                         "                        uint3 p [[thread_position_in_grid]])\n"
                         "{\n"
                         "    uint3 size = uint3(dst.get_width(), dst.get_height(), dst.get_depth());\n"
                         "    if (any(p >= size))\n"
                         "        return;\n"
                         "    float4 c = src.sample(s, (float3(p) + 0.5f) / float3(size), level(0));\n"
                         "    if (srgb)\n"
                         "        c = float4(encode(c.r), encode(c.g), encode(c.b), c.a);\n"
                         "    dst.write(c, p);\n"
                         "}\n";
        NSError *err;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        for (int i = 0; i < 2; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:i ? @"malachite_mip3d" : @"malachite_mip"];
            ps[i] = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
            if (!ps[i])
                os_log(OS_LOG_DEFAULT, "Malachite: mipmap kernel: %{public}@", err.localizedDescription);
        }
    });
    return ps[volume];
}

// A multisampled attachment's resolve: a compute kernel averages each pixel's samples into the resolve texture
static void malachite_resolve(id cb, id ms, id to)
{
    static id<MTLComputePipelineState> ps;
    static dispatch_once_t once;
    id<MTLDevice> device = ((id<MTLCommandBuffer>)cb).device;
    dispatch_once(&once, ^{
        NSString *src = @"#include <metal_stdlib>\n"
                         "using namespace metal;\n"
                         "kernel void malachite_resolve(texture2d_ms<float> src [[texture(0)]],\n"
                         "                          texture2d<float, access::write> dst [[texture(1)]],\n"
                         "                          constant uint &samples [[buffer(0)]],\n"
                         "                          uint2 p [[thread_position_in_grid]])\n"
                         "{\n"
                         "    if (p.x >= dst.get_width() || p.y >= dst.get_height())\n"
                         "        return;\n"
                         "    float4 c = 0;\n"
                         "    for (uint s = 0; s < samples; s++)\n"
                         "        c += src.read(p, s);\n"
                         "    dst.write(c / float(samples), p);\n"
                         "}\n";
        NSError *err;
        id<MTLFunction> fn = [[device newLibraryWithSource:src options:nil error:&err] newFunctionWithName:@"malachite_resolve"];
        ps = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!ps)
            os_log(OS_LOG_DEFAULT, "Malachite: resolve kernel: %{public}@", err.localizedDescription);
    });
    if (!ps)
        return;
    uint32_t samples = ((MalachiteTexture *)ms)->_samples;
    uint64_t buffers[31] = {};
    malachite_upload(cb, &samples, sizeof(samples), &buffers[0]);
    __unsafe_unretained id textures[NVGPU_MAX_TEXTURES] = { ms, to }, samplers[NVGPU_MAX_TEXTURES] = {};
    NSUInteger w = ((id<MTLTexture>)to).width, h = ((id<MTLTexture>)to).height;
    malachite_dispatch(cb, (MalachiteComputePipelineState *)ps, buffers, textures, samplers,
                   MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), MTLSizeMake(8, 8, 1));
}

// A multisampled depth/stencil attachment's resolve, by a compute kernel writing a colour view of the resolve
// texture of the same size per pixel: depth as R32Float or R16Unorm, depth and stencil together as RG32Uint (the
// float depth word, then the word holding the stencil byte), keeping the half that doesn't resolve. Depth takes
// sample 0, the nearest or the farthest; stencil sample 0 or the one the depth came from.
struct malachite_zresolve {
    id to;               // the resolve texture's level and slice, or nil
    uint32_t filter;     // MTLMultisampleDepthResolveFilter
    uint32_t depth, stencil;  // what resolves into it
    uint32_t sfilter;    // MTLMultisampleStencilResolveFilter
};

static id<MTLComputePipelineState> malachite_zresolve_pipeline(id<MTLDevice> device, bool packed)
{
    static id<MTLComputePipelineState> ps[2];
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSString *src = @"#include <metal_stdlib>\n"
                         "using namespace metal;\n"
                         "static float pick(texture2d_ms<float> z, uint2 q, uint n, uint filter, thread uint &at)\n"
                         "{\n"
                         "    float d = z.read(q, 0).x;\n"
                         "    at = 0;\n"
                         "    for (uint i = 1; i < n; i++) {\n"
                         "        float v = z.read(q, i).x;\n"
                         "        if ((filter == 1 && v < d) || (filter == 2 && v > d)) {\n"
                         "            d = v;\n"
                         "            at = i;\n"
                         "        }\n"
                         "    }\n"
                         "    return d;\n"
                         "}\n"
                         "kernel void malachite_zresolve(texture2d_ms<float> z [[texture(0)]],\n"
                         "                           texture2d<float, access::write> dst [[texture(1)]],\n"
                         "                           constant uint4 &p [[buffer(0)]], uint2 q [[thread_position_in_grid]])\n"
                         "{\n"
                         "    if (q.x >= dst.get_width() || q.y >= dst.get_height())\n"
                         "        return;\n"
                         "    uint at;\n"
                         "    dst.write(float4(pick(z, q, p.x, p.y, at)), q);\n"
                         "}\n"
                         "kernel void malachite_zsresolve(texture2d_ms<float> z [[texture(0)]],\n"
                         "                            texture2d_ms<uint> s [[texture(1)]],\n"
                         "                            texture2d<uint> old [[texture(2)]],\n"
                         "                            texture2d<uint, access::write> dst [[texture(3)]],\n"
                         "                            constant uint4 &p [[buffer(0)]], uint2 q [[thread_position_in_grid]])\n"
                         "{\n"
                         "    if (q.x >= dst.get_width() || q.y >= dst.get_height())\n"
                         "        return;\n"
                         "    uint at;\n"
                         "    float d = pick(z, q, p.x, p.y, at);\n"
                         "    uint2 o = old.read(q).xy;\n"
                         "    if (p.z & 1)\n"
                         "        o.x = as_type<uint>(d);\n"
                         "    if (p.z & 2)\n"
                         "        o.y = s.read(q, p.w ? at : 0).x;\n"
                         "    dst.write(uint4(o, 0, 0), q);\n"
                         "}\n";
        NSError *err;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        for (int i = 0; i < 2; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:i ? @"malachite_zsresolve" : @"malachite_zresolve"];
            ps[i] = fn ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
            if (!ps[i])
                os_log(OS_LOG_DEFAULT, "Malachite: depth resolve kernel: %{public}@", err.localizedDescription);
        }
    });
    return ps[packed];
}

static void malachite_zresolve(id cb, MalachiteTexture *ms, const struct malachite_zresolve *r)
{
    id<MTLDevice> device = ((id<MTLCommandBuffer>)cb).device;
    id<MTLTexture> to = r->to;
    bool packed = malachite_bytes_per_pixel(to.pixelFormat) == 8;
    MTLPixelFormat view = packed ? MTLPixelFormatRG32Uint
                          : to.pixelFormat == MTLPixelFormatDepth16Unorm ? MTLPixelFormatR16Unorm
                                                                         : MTLPixelFormatR32Float;
    id<MTLComputePipelineState> ps = malachite_zresolve_pipeline(device, packed);
    id dst = [to newTextureViewWithPixelFormat:view];
    id stencil = packed ? [ms newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8] : nil;
    if (!ps || !dst || (packed && !stencil))
        return;
    malachite_keep(cb, dst);
    malachite_keep(cb, stencil);
    uint32_t p[4] = { ms->_samples, r->depth ? r->filter : 0, r->depth | r->stencil << 1,
                      r->sfilter == MTLMultisampleStencilResolveFilterDepthResolvedSample };
    uint64_t buffers[31] = {};
    malachite_upload(cb, p, sizeof(p), &buffers[0]);
    __unsafe_unretained id textures[NVGPU_MAX_TEXTURES] = {}, samplers[NVGPU_MAX_TEXTURES] = {};
    textures[0] = ms;
    if (packed) {
        textures[1] = stencil;
        textures[2] = dst;
        textures[3] = dst;
    } else {
        textures[1] = dst;
    }
    NSUInteger w = to.width, h = to.height;
    malachite_dispatch(cb, (MalachiteComputePipelineState *)ps, buffers, textures, samplers,
                   MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), MTLSizeMake(8, 8, 1));
}

// a level and slice of a resolve texture, as a 2D texture of its own
static id malachite_resolve_target(id<MTLTexture> t, NSUInteger level, NSUInteger slice)
{
    return t && ((MalachiteTexture *)t)->_image ? [t newTextureViewWithPixelFormat:t.pixelFormat
                                                                    textureType:MTLTextureType2D
                                                                         levels:NSMakeRange(level, 1)
                                                                         slices:NSMakeRange(slice, 1)]
                                            : t;
}

// The block-linear stand-in for attachment n of a pass the 3D class can't render to as it is, size bytes. A queue's
// passes reuse one: they run in the order they were recorded, so the store of one is done before the next fills the
// memory, and a buffer made per pass cost a few milliseconds of allocation. In VRAM, which is where the render target
// itself lives.
static id<MTLBuffer> malachite_shadow(id<MTLCommandQueue> queue, unsigned n, NSUInteger size)
{
    static const char shadows_key = 0, order_key = 0;
    NSNumber *key = @(size * 8 + n);
    @synchronized(queue) {
        NSMutableDictionary<NSNumber *, id<MTLBuffer>> *shadows = objc_getAssociatedObject(queue, &shadows_key);
        NSMutableArray<NSNumber *> *order = objc_getAssociatedObject(queue, &order_key);  // the oldest used first
        if (!shadows) {
            shadows = [NSMutableDictionary dictionary];
            order = [NSMutableArray array];
            objc_setAssociatedObject(queue, &shadows_key, shadows, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            objc_setAssociatedObject(queue, &order_key, order, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        }
        id<MTLBuffer> buf = shadows[key];
        if (![order.lastObject isEqual:key]) {
            [order removeObject:key];
            [order addObject:key];
        }
        if (!buf) {
            // Sizes come and go with window and view resizes, while a game goes round its targets every frame:
            // what makes room is the one unused longest, and only for as much memory as they hold
            NSUInteger total = size;
            for (NSNumber *k in order)
                total += shadows[k].length;
            while (order.count > 1 && total > 256 << 20) {
                total -= shadows[order[0]].length;
                [shadows removeObjectForKey:order[0]];
                [order removeObjectAtIndex:0];
            }
            buf = [queue.device newBufferWithLength:(size + 0xffff) & ~(NSUInteger)0xffff
                                            options:MTLResourceStorageModePrivate];
            if (buf)
                ((MalachiteBuffer *)buf)->_one_image = true;
            else
                buf = [queue.device newBufferWithLength:size options:MTLResourceStorageModeShared];
            shadows[key] = buf;
        }
        return buf;
    }
}

// Render encoder base (private): initWithCommandBuffer:descriptor: does the pass bookkeeping.
@interface MTLIOAccelRenderCommandEncoder : NSObject
- (instancetype)initWithCommandBuffer:(id)cb descriptor:(id)descriptor;
- (void)endEncoding;
@end

@interface MalachiteRenderCommandEncoder : MTLIOAccelRenderCommandEncoder <MTLRenderCommandEncoder> {
    __unsafe_unretained id _cb;
    MalachiteRenderPipelineState *_pipeline;
    uint64_t _vbuffers[31], _fbuffers[31];
    uint32_t _vsizes[31];  // bytes from each vertex buffer's offset to its end
    uint32_t _fsizes[31];  // and fragment buffer's, for the recorder
    uint32_t _vstrides[31];  // each vertex buffer's stride as bound, for a layout whose stride is dynamic
    __unsafe_unretained id _vbound[31], _fbound[31];  // the buffers, for set*BufferOffset (kept by the command buffer)
    MTLRenderPassDescriptor *_rpd;  // the recorder's: the pass as asked for
    MTLRenderPassDescriptor *_deferred;  // the pass, when a resolve waits on a store action given at the end
    __unsafe_unretained id _textures[MALACHITE_BINDINGS];  // the fragment stage's, by Metal index
    __unsafe_unretained id _samplers[MALACHITE_BINDINGS];
    __unsafe_unretained id _vtextures[MALACHITE_BINDINGS];  // the vertex stage's
    __unsafe_unretained id _vsamplers[MALACHITE_BINDINGS];
    int _formats[8];  // each colour attachment's enum tex_format, -1 for none; [[color(n)]] inputs read attachment n
    __unsafe_unretained MalachiteTexture *_attached[8];  // the attachments themselves, for the trace's dumps
    uint32_t _attached_level[8];                     // the image level each draws into
    __unsafe_unretained MalachiteTexture *_zattached;
    struct nvgpu_draw _pass;  // the targets and their clears from the pass descriptor, viewport and scissor
    uint64_t _indirect;       // the arguments of the indirect draw being encoded
    uint64_t _factors;        // the tessellation factor buffer, 0 for none
    uint32_t _factor_stride;  // its instance stride
    float _factor_scale;
    bool _patching;           // a patch draw is being encoded
    NSUInteger _patch_points; // its control points a patch
    uint64_t _patch_index;    // and its patch and control point index buffers, 0 for none
    uint64_t _point_index;
    uint64_t _visibility;     // the pass's visibility result buffer, 0 for none
    uint64_t _counted;        // the result the last draw counting samples counted into
    id _resolve[8];           // where each multisampled attachment resolves to, nil for none
    bool _barrier;            // a texture or memory barrier since the last draw
    bool _fast;               // later draws of the pass are written here as what changed, as a run
    __unsafe_unretained id _pools_pipeline;  // what the last draw's heap entries are of, nil for nothing to reuse
    const void *_pools_bound[4][MALACHITE_BINDINGS];
    struct nvgpu_heap _pools_heap;
    unsigned int _pools_nheaders, _pools_nsamplers, _pools_vs_textured;
    unsigned int _pools_vs_handles[NVGPU_HANDLES];
    uint64_t _last_records;   // the command buffer's count just after _last went out, 0 for none
    uint64_t _seen_code[16][2];  // shader pairs run in this pass
    unsigned _seen_next;
    struct nvgpu_draw _last;  // what the channel holds, but for what a draw always sends
    struct malachite_zresolve _zresolve[2];  // and the depth and stencil attachment, apart or together
    MalachiteDepthStencilState *_ds;
    bool _drawn;
    // a sampled attachment's copy: whether this pass made one, and what its draws have covered since (x0 y0 x1 y1)
    bool _copied[8];
    uint32_t _written[8][4];
    // what the 3D class can't draw to (see the init) is drawn to a block-linear shadow: loaded from the attachment
    // before the first draw, stored back at the end
    struct nvgpu_copy2d _load[8], _store[8];  // linear -> shadow, shadow -> linear; bytes 0 when not wanted
    struct nvgpu_copy2d _zload, _zstore;  // the same for a depth texture apart from its stencil texture
}
@end

@implementation MalachiteRenderCommandEncoder
- (instancetype)initWithCommandBuffer:(id)cb descriptor:(MTLRenderPassDescriptor *)descriptor
{
    self = [super initWithCommandBuffer:cb descriptor:descriptor];
    if (!self)
        return nil;
    _cb = cb;
    if (opengl_renderer) {
        static dispatch_once_t once;
        dispatch_once(&once, ^{
            malachite_gl_swaps();
        });
    }
    _fast = !malachite_trace && !malachite_snap && !malachite_rec_only;
    _factor_scale = 1.0f;
    for (unsigned i = 0; i < 8; i++)
        _formats[i] = -1;
    if (malachite_rec_only)
        _rpd = [descriptor copy];
    malachite_keep(cb, descriptor.depthAttachment.texture);
    malachite_keep(cb, descriptor.visibilityResultBuffer);
    _visibility = descriptor.visibilityResultBuffer ? [(MTLIOAccelBuffer *)descriptor.visibilityResultBuffer gpuAddress] : 0;
    malachite_keep(cb, descriptor.stencilAttachment.texture);

    // depth and stencil share one target (a combined format); either attachment names it. A Depth32Float texture
    // beside a Stencil8 one (as Apple's GL makes them) draws to the stencil texture, which is kept in the combined
    // format: the depth is copied into it before the first draw and back at the end
    MTLRenderPassDepthAttachmentDescriptor *da = descriptor.depthAttachment;
    MTLRenderPassStencilAttachmentDescriptor *sa = descriptor.stencilAttachment;
    MalachiteTexture *dt = (MalachiteTexture *)da.texture, *st = (MalachiteTexture *)sa.texture;
    struct malachite_sub ds;
    bool apart = dt && st && dt != st && da.texture.pixelFormat == MTLPixelFormatDepth32Float &&
                 sa.texture.pixelFormat == MTLPixelFormatStencil8 && [dt isKindOfClass:[MalachiteTexture class]] &&
                 [st isKindOfClass:[MalachiteTexture class]] && !dt->_samples && !st->_samples &&
                 malachite_sub(dt, da.level, da.slice, 0, &ds);
    if (apart) {
        malachite_depth_kind(cb, dt);
        malachite_depth_kind(cb, st);
    }
    MTLRenderPassAttachmentDescriptor *za = apart || !da.texture ? (MTLRenderPassAttachmentDescriptor *)sa : da;
    MalachiteTexture *z = (MalachiteTexture *)za.texture;
    struct malachite_sub zs;
    if (z && [z isKindOfClass:[MalachiteTexture class]] && z->_zt_format && malachite_sub(z, za.level, za.slice, 0, &zs)) {
        _zattached = z;
        _pass.zt = [(MTLIOAccelBuffer *)z->_backing gpuAddress] + zs.offset;
        _pass.zt_format = z->_zt_format;
        _pass.zt_width = zs.pitch / z->_zt_bpp;
        _pass.zt_height = zs.height;
        _pass.samples = z->_samples;
        _pass.zt_kind = z->_zt_kind;
        _pass.zt_compress = ((MalachiteBuffer *)z->_backing)->_one_image;
        _pass.zt_layer_pitch = z->_image ? (uint32_t)z->_layer_stride : 0;
        _pass.zt_layers = z->_image ? z->_layers - (uint32_t)za.slice : 1;
        objc_setAssociatedObject(z->_backing, &malachite_kind_key, @YES, OBJC_ASSOCIATION_RETAIN);
        _pass.clear_zs = (da.texture && da.loadAction == MTLLoadActionClear) |
                         (sa.texture && sa.loadAction == MTLLoadActionClear) << 1;
        _pass.clear_depth = (float)da.clearDepth;
        _pass.clear_stencil = sa.clearStencil;
        if (z->_samples) {
            bool zr = da.texture && da.resolveTexture && (da.storeAction == MTLStoreActionMultisampleResolve ||
                                                          da.storeAction == MTLStoreActionStoreAndMultisampleResolve);
            bool sr = sa.texture && sa.resolveTexture && (sa.storeAction == MTLStoreActionMultisampleResolve ||
                                                          sa.storeAction == MTLStoreActionStoreAndMultisampleResolve);
            if (zr)
                [self setDepthResolveTexture:da.resolveTexture slice:da.resolveSlice depthPlane:0
                                       level:da.resolveLevel];
            _zresolve[0].filter = (uint32_t)da.depthResolveFilter;
            if (sr)
                [self setStencilResolveTexture:sa.resolveTexture slice:sa.resolveSlice depthPlane:0
                                         level:sa.resolveLevel];
            _zresolve[0].sfilter = _zresolve[1].sfilter = (uint32_t)sa.stencilResolveFilter;
        }
    } else if (z) {
        os_log(OS_LOG_DEFAULT, "MalachiteRender: depth/stencil attachment format %lu not supported",
               (unsigned long)z.pixelFormat);
    }

    unsigned ncolour = 0;
    // the pass draws in the area every attachment has
    MTLRenderPassAttachmentDescriptor *sized[10] = {};
    for (unsigned i = 0; i < 8; i++)
        sized[i] = descriptor.colorAttachments[i];
    if (_pass.zt) {
        sized[8] = da;
        sized[9] = sa;
    }
    for (MTLRenderPassAttachmentDescriptor *a : sized) {
        MalachiteTexture *t = (MalachiteTexture *)a.texture;
        struct malachite_sub s;
        if (![t isKindOfClass:[MalachiteTexture class]] || !t->_backing ||
            !malachite_sub(t, a.level, a.slice, a.depthPlane, &s))
            continue;
        uint32_t sx, sy;
        malachite_grid(t->_samples, &sx, &sy);
        _pass.width = _pass.width ? MIN(_pass.width, s.width / sx) : s.width / sx;
        _pass.height = _pass.height ? MIN(_pass.height, s.height / sy) : s.height / sy;
    }
    if (apart && _pass.zt) {  // each depth word to the first of each 8-byte texel, the stencil word left as it is
        uint64_t depth = [(MTLIOAccelBuffer *)dt->_backing gpuAddress] + ds.offset;
        const uint32_t x = REMAP_X, none = REMAP_NONE;
        if (da.loadAction == MTLLoadActionLoad)
            _zload = (struct nvgpu_copy2d){ depth, _pass.zt, ds.pitch, zs.pitch, _pass.width * 4, _pass.height, 1, 1,
                                            ds.rows, zs.rows, 0, 0, malachite_remap(x, none, none, none, 4, 1, 2) };
        if (da.storeAction != MTLStoreActionDontCare)
            _zstore = (struct nvgpu_copy2d){ _pass.zt, depth, zs.pitch, ds.pitch, _pass.width * 8, _pass.height, 1, 1,
                                             zs.rows, ds.rows, 0, 0, malachite_remap(x, none, none, none, 4, 2, 1) };
    }
    for (unsigned i = 0; i < 8; i++)
        ncolour += descriptor.colorAttachments[i].texture != nil;
    bool first = true;
    for (unsigned i = 0; i < 8; i++) {
        MTLRenderPassColorAttachmentDescriptor *ca = descriptor.colorAttachments[i];
        MalachiteTexture *t = (MalachiteTexture *)ca.texture;
        if (!t)
            continue;
        malachite_keep(cb, t);
        int f = [t isKindOfClass:[MalachiteTexture class]] && t->_backing ? malachite_format(t.pixelFormat) : -1;
        if (f < 0 || !tex_format_info((enum tex_format)f).target) {
            os_log(OS_LOG_DEFAULT, "MalachiteRender: colour attachment %u (format %lu) not supported", i,
                   (unsigned long)t.pixelFormat);
            continue;
        }
        struct malachite_sub s;
        malachite_depth_kind(cb, t);
        malachite_mips_written(t, ca.level);
        malachite_texture_written(cb, t);
        if (!malachite_sub(t, ca.level, ca.slice, ca.depthPlane, &s)) {
            os_log(OS_LOG_DEFAULT, "MalachiteRender: colour attachment %u at level %lu slice %lu, which it lacks", i,
                   (unsigned long)ca.level, (unsigned long)ca.slice);
            continue;
        }
        uint32_t sx, sy;
        malachite_grid(t->_samples, &sx, &sy);
        if (first) {
            _pass.samples = t->_samples;
            first = false;
        }
        // a multisampled attachment resolved into another texture at the end (a level and slice of it)
        id<MTLTexture> rt = ca.resolveTexture;
        if (t->_samples && rt && (ca.storeAction == MTLStoreActionMultisampleResolve ||
                                  ca.storeAction == MTLStoreActionStoreAndMultisampleResolve)) {
            _resolve[i] = ((MalachiteTexture *)rt)->_image
                              ? [rt newTextureViewWithPixelFormat:rt.pixelFormat textureType:MTLTextureType2D
                                                           levels:NSMakeRange(ca.resolveLevel, 1)
                                                           slices:NSMakeRange(ca.resolveSlice, 1)]
                              : rt;
            malachite_keep(cb, _resolve[i]);
        }
        // an image's surface is block-linear already
        struct nvgpu_target ct = { [(MTLIOAccelBuffer *)t->_backing gpuAddress] + s.offset, s.rows ? 0 : s.pitch,
                                   tex_format_info((enum tex_format)f).target, s.rows != 0,
                                   ca.loadAction == MTLLoadActionClear, 0, {}, {} };
        malachite_clear_colour((enum tex_format)f, ca.clearColor, ct.clear_rgba);
        // a block-linear shadow, copied in and back by the copy engine, for what the 3D class can't render to: a
        // pitch-linear target beside a depth buffer or another colour target (a pass with one draws nothing to any
        // of them), or rows or a start not on 128 bytes (an IOSurface 357 pixels wide)
        if (!ct.blocklinear && (_pass.zt || ncolour > 1 || (ct.pitch & 127) || (ct.addr & 127))) {
            uint32_t bytes = _pass.width * tex_format_info((enum tex_format)f).bytes;
            uint32_t width = (bytes + 63) & ~63u, rows = (_pass.height + 7) & ~7u;  // one-GOB blocks: 64 B x 8 rows
            id<MTLBuffer> shadow = malachite_shadow(((id<MTLCommandBuffer>)cb).commandQueue, i, (NSUInteger)width * rows);
            [(id<MTLCommandBuffer>)cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                (void)shadow;
            }];
            uint64_t sva = [(MTLIOAccelBuffer *)shadow gpuAddress];
            if (((MalachiteBuffer *)shadow)->_one_image)
                malachite_compress_colour(cb, (MalachiteBuffer *)shadow);
            // DontCare too: the shadow is stored back whole, and what the pass doesn't draw has to stay as it was
            if (ca.loadAction != MTLLoadActionClear)
                _load[i] = (struct nvgpu_copy2d){ ct.addr, sva, ct.pitch, width, bytes, _pass.height, 0, 1, 0, rows };
            if (ca.storeAction != MTLStoreActionDontCare || malachite_trace)  // the trace dumps every attachment
                _store[i] = (struct nvgpu_copy2d){ sva, ct.addr, width, ct.pitch, bytes, _pass.height, 1, 0, rows, 0 };
            ct.addr = sva;
            ct.pitch = 0;  // a block-linear target has none
            ct.blocklinear = 1;
        } else {
            ct.width = s.width / sx;
            ct.height = s.height / sy;
        }
        _formats[i] = f;
        _attached[i] = t;
        _attached_level[i] = t->_view_min + (uint32_t)ca.level;
        // for a layered pass: the array layers from the slice on, or a 3D level's depth planes from the attached one
        if (t->_image && t->_type == MTLTextureType3D) {
            struct malachite_lvl v = malachite_image_level(t->_format, t->_width, t->_height, t->_depth,
                                                           t->_view_min + (uint32_t)ca.level);
            _pass.layer_pitch[i] = (uint32_t)v.slice;
            _pass.target_layers[i] = v.depth - (uint32_t)ca.depthPlane;
        } else if (t->_image) {
            _pass.layer_pitch[i] = (uint32_t)t->_layer_stride;
            _pass.target_layers[i] = t->_layers - (uint32_t)ca.slice;
        } else {
            _pass.layer_pitch[i] = 0;
            _pass.target_layers[i] = 1;
        }
        if (i) {
            _pass.more[i - 1] = ct;
            continue;
        }
        _pass.target = ct.addr;
        _pass.pitch = ct.pitch;
        _pass.format = ct.format;
        _pass.target_blocklinear = ct.blocklinear;
        _pass.target_width = ct.width;
        _pass.target_height = ct.height;
        _pass.clear = ct.clear;
        memcpy(_pass.clear_rgba, ct.clear_rgba, sizeof(ct.clear_rgba));
    }
    [self setViewport:(MTLViewport){ 0, 0, (double)_pass.width, (double)_pass.height, 0, 1 }];
    [self setScissorRect:(MTLScissorRect){ 0, 0, _pass.width, _pass.height }];
    for (unsigned j = 0; j < 15; j++) {  // the other viewports' scissors cover the target until set
        _pass.scissors[j][2] = _pass.width;
        _pass.scissors[j][3] = _pass.height;
    }
    _pass.layers = (uint32_t)descriptor.renderTargetArrayLength;
    bool later = (da.resolveTexture && da.storeAction == MTLStoreActionUnknown) ||
                 (sa.resolveTexture && sa.storeAction == MTLStoreActionUnknown);
    for (unsigned i = 0; i < 8; i++)
        later |= descriptor.colorAttachments[i].resolveTexture &&
                 descriptor.colorAttachments[i].storeAction == MTLStoreActionUnknown;
    if (later)
        _deferred = [descriptor copy];
    if (malachite_trace) {
        MTLRenderPassColorAttachmentDescriptor *ca = descriptor.colorAttachments[0];
        for (int i = 0; i < 4; i++) {
            MTLRenderPassColorAttachmentDescriptor *a = descriptor.colorAttachments[i];
            if (a.texture || a.resolveTexture)
                os_log(OS_LOG_DEFAULT, "Malachite trace:   attachment %d: %{public}@ %lux%lu level %lu slice %lu store %lu "
                       "resolve %{public}@",
                       i, NSStringFromClass([a.texture class]), (unsigned long)a.texture.width,
                       (unsigned long)a.texture.height, (unsigned long)a.level, (unsigned long)a.slice,
                       (unsigned long)a.storeAction, a.resolveTexture ? NSStringFromClass([a.resolveTexture class]) : @"none");
        }
        os_log(OS_LOG_DEFAULT,
               "Malachite trace: pass %ux%u target %#llx colour %lu %lu %lu %lu load %lu store %lu (1: load %lu), depth %lu "
               "load %lu clear %g stencil clear %u%{public}s",
               _pass.width, _pass.height, _pass.target, (unsigned long)ca.texture.pixelFormat,
               (unsigned long)descriptor.colorAttachments[1].texture.pixelFormat,
               (unsigned long)descriptor.colorAttachments[2].texture.pixelFormat,
               (unsigned long)descriptor.colorAttachments[3].texture.pixelFormat, (unsigned long)ca.loadAction,
               (unsigned long)ca.storeAction, (unsigned long)descriptor.colorAttachments[1].loadAction,
               (unsigned long)z.pixelFormat, (unsigned long)da.loadAction, da.clearDepth, (unsigned)sa.clearStencil,
               _pass.target_blocklinear ? ", shadowed" : "");
    }
    return self;
}

- (void)setDepthStencilState:(id)state
{
    _ds = state;
}

- (void)setStencilReferenceValue:(uint32_t)value
{
    _pass.stencil_ref = (value & 0xff) * 0x101;
}

- (void)setStencilFrontReferenceValue:(uint32_t)front backReferenceValue:(uint32_t)back
{
    _pass.stencil_ref = (front & 0xff) | (back & 0xff) << 8;
}

- (void)setRenderPipelineState:(id)state
{
    _pipeline = state;
    malachite_keep(_cb, state);
}

- (void)setVertexBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (index >= 31)
        return;
    if (malachite_trace)
        os_log(OS_LOG_DEFAULT, "Malachite trace: vertex buffer %lu: %p offset %lu", (unsigned long)index, buffer,
               (unsigned long)offset);
    _vbuffers[index] = buffer ? [(MTLIOAccelBuffer *)buffer gpuAddress] + offset : 0;
    _vsizes[index] = buffer ? (uint32_t)(((id<MTLBuffer>)buffer).length - offset) : 0;
    _vbound[index] = buffer;
    malachite_keep(_cb, buffer);
}

- (void)setFragmentBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (index >= 31)
        return;
    _fbuffers[index] = buffer ? [(MTLIOAccelBuffer *)buffer gpuAddress] + offset : 0;
    _fsizes[index] = buffer ? (uint32_t)(((id<MTLBuffer>)buffer).length - offset) : 0;
    _fbound[index] = buffer;
    malachite_keep(_cb, buffer);
}

- (void)setVertexBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (index < 31)
        [self setVertexBuffer:_vbound[index] offset:offset atIndex:index];
}

- (void)setFragmentBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
    if (index < 31)
        [self setFragmentBuffer:_fbound[index] offset:offset atIndex:index];
}

- (void)setVertexBuffers:(const id __unsafe_unretained *)buffers
                 offsets:(const NSUInteger *)offsets
               withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setVertexBuffer:buffers[i] offset:offsets[i] atIndex:range.location + i];
}

- (void)setVertexBuffer:(id)buffer offset:(NSUInteger)offset attributeStride:(NSUInteger)stride atIndex:(NSUInteger)index
{
    if (index < 31)
        _vstrides[index] = (uint32_t)stride;
    [self setVertexBuffer:buffer offset:offset atIndex:index];
}

- (void)setVertexBufferOffset:(NSUInteger)offset attributeStride:(NSUInteger)stride atIndex:(NSUInteger)index
{
    if (index < 31)
        [self setVertexBuffer:_vbound[index] offset:offset attributeStride:stride atIndex:index];
}

- (void)setVertexBuffers:(const id __unsafe_unretained *)buffers
                 offsets:(const NSUInteger *)offsets
        attributeStrides:(const NSUInteger *)strides
               withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setVertexBuffer:buffers[i] offset:offsets[i] attributeStride:strides[i] atIndex:range.location + i];
}

- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)length attributeStride:(NSUInteger)stride
               atIndex:(NSUInteger)index
{
    if (index < 31)
        _vstrides[index] = (uint32_t)stride;
    [self setVertexBytes:bytes length:length atIndex:index];
}

- (void)setFragmentBuffers:(const id __unsafe_unretained *)buffers
                   offsets:(const NSUInteger *)offsets
                 withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setFragmentBuffer:buffers[i] offset:offsets[i] atIndex:range.location + i];
}

- (void)setFragmentTextures:(const id __unsafe_unretained *)textures withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setFragmentTexture:textures[i] atIndex:range.location + i];
}

- (void)setFragmentSamplerStates:(const id __unsafe_unretained *)samplers withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setFragmentSamplerState:samplers[i] atIndex:range.location + i];
}

- (void)setVertexTexture:(id)texture atIndex:(NSUInteger)index
{
    if (index >= MALACHITE_BINDINGS)
        return;
    _vtextures[index] = texture;
    malachite_keep(_cb, texture);
}

- (void)setVertexSamplerState:(id)sampler atIndex:(NSUInteger)index
{
    if (index < MALACHITE_BINDINGS)
        _vsamplers[index] = sampler;
}

- (void)setCullMode:(MTLCullMode)mode
{
    _pass.cull_face = mode == MTLCullModeFront ? 0x404 : mode == MTLCullModeBack ? 0x405 : 0;
}

- (void)setFrontFacingWinding:(MTLWinding)winding
{
    _pass.front_ccw = winding == MTLWindingCounterClockwise;
}

// OpenGL's state, which Apple's GL renderer sets through the encoder

- (void)setPointSize:(float)size
{
    _pass.point_size = size;
}

- (void)setLineWidth:(float)width
{
    _pass.line_width = width;
}

// 0 the first vertex (Metal's), 1 the last (OpenGL's default)
- (void)setProvokingVertexMode:(NSUInteger)mode
{
    _pass.provoking_last = mode != 0;
}

- (void)setPrimitiveRestartEnabled:(BOOL)enabled index:(NSUInteger)index
{
    _pass.restart = enabled ? 2 : 1;
    _pass.restart_index = (unsigned int)index;
}

- (void)setViewportTransformEnabled:(BOOL)enabled
{
    _pass.window_coords = !enabled;
}

// a nil buffer bound is already a null address
- (void)enableNullBufferBinds:(BOOL)enable
{
}

- (void)setCommandDataCorruptModeSPI:(NSUInteger)mode
{
}

// a multisampled attachment's resolve target set after the pass began, as the descriptor's would be
- (void)setColorResolveTexture:(id)texture slice:(NSUInteger)slice depthPlane:(NSUInteger)plane
                         level:(NSUInteger)level yInvert:(BOOL)invert atIndex:(NSUInteger)index
{
    if (index >= 8)
        return;
    if (invert)
        os_log(OS_LOG_DEFAULT, "MalachiteRender: y-inverted resolve not supported, resolving upright");
    MalachiteTexture *t = texture;
    _resolve[index] = t && t->_image ? [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D
                                                                 levels:NSMakeRange(level, 1)
                                                                 slices:NSMakeRange(slice, 1)]
                                     : t;
    if (_resolve[index])
        malachite_keep(_cb, _resolve[index]);
}

- (void)setColorResolveTexture:(id)texture slice:(NSUInteger)slice depthPlane:(NSUInteger)plane
                         level:(NSUInteger)level atIndex:(NSUInteger)index
{
    [self setColorResolveTexture:texture slice:slice depthPlane:plane level:level yInvert:NO atIndex:index];
}

- (void)setDepthResolveTexture:(id)texture slice:(NSUInteger)slice depthPlane:(NSUInteger)plane
                         level:(NSUInteger)level yInvert:(BOOL)invert
{
    _zresolve[0].to = malachite_resolve_target(texture, level, slice);
    _zresolve[0].depth = _zresolve[0].to != nil;
}

- (void)setDepthResolveTexture:(id)texture slice:(NSUInteger)slice depthPlane:(NSUInteger)plane
                         level:(NSUInteger)level
{
    [self setDepthResolveTexture:texture slice:slice depthPlane:plane level:level yInvert:NO];
}

- (void)setStencilResolveTexture:(id)texture slice:(NSUInteger)slice depthPlane:(NSUInteger)plane
                           level:(NSUInteger)level yInvert:(BOOL)invert
{
    // a pass of its own even into the depth's resolve texture, as each keeps what the other wrote
    _zresolve[1].to = malachite_resolve_target(texture, level, slice);
    _zresolve[1].stencil = _zresolve[1].to != nil;
}

- (void)setStencilResolveTexture:(id)texture slice:(NSUInteger)slice depthPlane:(NSUInteger)plane
                           level:(NSUInteger)level
{
    [self setStencilResolveTexture:texture slice:slice depthPlane:plane level:level yInvert:NO];
}

- (void)setTriangleFillMode:(MTLTriangleFillMode)mode
{
    _pass.fill_lines = mode == MTLTriangleFillModeLines;
}

- (void)setDepthBias:(float)bias slopeScale:(float)scale clamp:(float)clamp
{
    _pass.depth_bias[0] = bias;
    _pass.depth_bias[1] = scale;
    _pass.depth_bias[2] = clamp;
}

- (void)setDepthClipMode:(MTLDepthClipMode)mode
{
    _pass.depth_clamp = mode == MTLDepthClipModeClamp;
}

- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha
{
    _pass.blend_const[0] = red;
    _pass.blend_const[1] = green;
    _pass.blend_const[2] = blue;
    _pass.blend_const[3] = alpha;
}

// draws count their passing samples into the result at offset (boolean mode reads that as passed or not), each
// going on from the last draw's count while the offset stays the same
- (void)setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:(NSUInteger)offset
{
    _pass.visibility = mode != MTLVisibilityResultModeDisabled && _visibility ? _visibility + offset : 0;
}

// every command runs to completion in order, and resources stay resident: no ordering or residency to add
// resources the shaders reach through pointers in other buffers (argument buffers): kept like bound ones
- (void)useResource:(id)resource usage:(MTLResourceUsage)usage
{
    malachite_keep(_cb, resource);
}

- (void)useResource:(id)resource usage:(MTLResourceUsage)usage stages:(MTLRenderStages)stages
{
    malachite_keep(_cb, resource);
}

- (void)useResources:(const id __unsafe_unretained *)resources count:(NSUInteger)count usage:(MTLResourceUsage)usage
{
    for (NSUInteger i = 0; i < count; i++)
        malachite_keep(_cb, resources[i]);
}

- (void)useResources:(const id __unsafe_unretained *)resources
               count:(NSUInteger)count
               usage:(MTLResourceUsage)usage
              stages:(MTLRenderStages)stages
{
    for (NSUInteger i = 0; i < count; i++)
        malachite_keep(_cb, resources[i]);
}

- (void)useHeap:(id)heap
{
    malachite_keep(_cb, heap);
}

- (void)useHeap:(id)heap stages:(MTLRenderStages)stages
{
    malachite_keep(_cb, heap);
}

- (void)useHeaps:(const id __unsafe_unretained *)heaps count:(NSUInteger)count
{
    for (NSUInteger i = 0; i < count; i++)
        malachite_keep(_cb, heaps[i]);
}

- (void)useHeaps:(const id __unsafe_unretained *)heaps count:(NSUInteger)count stages:(MTLRenderStages)stages
{
    for (NSUInteger i = 0; i < count; i++)
        malachite_keep(_cb, heaps[i]);
}

- (void)textureBarrier
{
    _barrier = true;
}

- (void)memoryBarrierWithScope:(MTLBarrierScope)scope afterStages:(MTLRenderStages)after beforeStages:(MTLRenderStages)before
{
    _barrier = true;
}

- (void)memoryBarrierWithResources:(const id<MTLResource> __unsafe_unretained *)resources
                             count:(NSUInteger)count
                       afterStages:(MTLRenderStages)after
                      beforeStages:(MTLRenderStages)before
{
    _barrier = true;
}

- (void)barrierAfterQueueStages:(MTLStages)after beforeStages:(MTLStages)before
{
    _barrier = true;
}

- (void)updateFence:(id)fence afterStages:(MTLRenderStages)stages
{
}

- (void)waitForFence:(id)fence beforeStages:(MTLRenderStages)stages
{
}

static bool malachite_resolves(MTLStoreAction action)
{
    return action == MTLStoreActionMultisampleResolve || action == MTLStoreActionStoreAndMultisampleResolve;
}

// what the pass left unknown: a resolve into the descriptor's target is only asked for here
- (void)setColorStoreAction:(MTLStoreAction)action atIndex:(NSUInteger)index
{
    if (index >= 8)
        return;
    if (action == MTLStoreActionDontCare)
        _store[index].bytes = 0;
    MTLRenderPassColorAttachmentDescriptor *ca = _deferred.colorAttachments[index];
    if (ca.resolveTexture && ca.storeAction == MTLStoreActionUnknown && malachite_resolves(action) &&
        _attached[index] && _attached[index]->_samples)
        [self setColorResolveTexture:ca.resolveTexture slice:ca.resolveSlice depthPlane:ca.resolveDepthPlane
                               level:ca.resolveLevel atIndex:index];
}

- (void)setDepthStoreAction:(MTLStoreAction)action
{
    MTLRenderPassDepthAttachmentDescriptor *da = _deferred.depthAttachment;
    if (da.resolveTexture && da.storeAction == MTLStoreActionUnknown && malachite_resolves(action) && _zattached &&
        _zattached->_samples)
        [self setDepthResolveTexture:da.resolveTexture slice:da.resolveSlice depthPlane:0 level:da.resolveLevel];
}

- (void)setStencilStoreAction:(MTLStoreAction)action
{
    MTLRenderPassStencilAttachmentDescriptor *sa = _deferred.stencilAttachment;
    if (sa.resolveTexture && sa.storeAction == MTLStoreActionUnknown && malachite_resolves(action) && _zattached &&
        _zattached->_samples)
        [self setStencilResolveTexture:sa.resolveTexture slice:sa.resolveSlice depthPlane:0 level:sa.resolveLevel];
}

- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
    if (index >= 31)
        return;
    if (malachite_trace)
        os_log(OS_LOG_DEFAULT, "Malachite trace: vertex bytes %lu: %lu bytes", (unsigned long)index, (unsigned long)length);
    _vbound[index] = malachite_upload(_cb, bytes, length, &_vbuffers[index]);
    _vsizes[index] = (uint32_t)length;
}

- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
    if (index < 31) {
        _fbound[index] = malachite_upload(_cb, bytes, length, &_fbuffers[index]);
        _fsizes[index] = (uint32_t)length;
    }
}

- (void)setFragmentTexture:(id)texture atIndex:(NSUInteger)index
{
    if (index >= MALACHITE_BINDINGS)
        return;
    _textures[index] = texture;
    malachite_keep(_cb, texture);
    if (malachite_trace) {
        MalachiteTexture *t = texture;
        bool ours = [t isKindOfClass:[MalachiteTexture class]] && t->_backing;
        os_log(OS_LOG_DEFAULT, "Malachite trace: fragment texture %lu: %#llx %lux%lu format %lu %{public}@",
               (unsigned long)index, ours ? [(MTLIOAccelBuffer *)t->_backing gpuAddress] + t->_offset : 0ull,
               (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.pixelFormat,
               ours ? @"" : NSStringFromClass([t class]));
    }
}

- (void)setFragmentSamplerState:(id)sampler atIndex:(NSUInteger)index
{
    if (index < MALACHITE_BINDINGS)
        _samplers[index] = sampler;
}

- (void)setViewport:(MTLViewport)v
{
    _pass.viewport[0] = (float)v.originX;
    _pass.viewport[1] = (float)v.originY;
    _pass.viewport[2] = (float)v.width;
    _pass.viewport[3] = (float)v.height;
    _pass.viewport[4] = (float)v.znear;
    _pass.viewport[5] = (float)v.zfar;
    _pass.nviewports = 1;
}

- (void)setScissorRect:(MTLScissorRect)r
{
    _pass.scissor[0] = (uint32_t)r.x;
    _pass.scissor[1] = (uint32_t)r.y;
    _pass.scissor[2] = (uint32_t)r.width;
    _pass.scissor[3] = (uint32_t)r.height;
}

// several, for a vertex function choosing one with [[viewport_array_index]]
- (void)setViewports:(const MTLViewport *)viewports count:(NSUInteger)count
{
    if (!count)
        return;
    [self setViewport:viewports[0]];
    _pass.nviewports = (uint32_t)MIN(count, 16);
    for (uint32_t j = 1; j < _pass.nviewports; j++) {
        const MTLViewport &v = viewports[j];
        float *w = _pass.viewports[j - 1];
        w[0] = (float)v.originX;
        w[1] = (float)v.originY;
        w[2] = (float)v.width;
        w[3] = (float)v.height;
        w[4] = (float)v.znear;
        w[5] = (float)v.zfar;
    }
}

- (void)setScissorRects:(const MTLScissorRect *)rects count:(NSUInteger)count
{
    if (!count)
        return;
    [self setScissorRect:rects[0]];
    for (NSUInteger j = 1; j < MIN(count, 16); j++) {
        unsigned int *s = _pass.scissors[j - 1];
        s[0] = (uint32_t)rects[j].x;
        s[1] = (uint32_t)rects[j].y;
        s[2] = (uint32_t)rects[j].width;
        s[3] = (uint32_t)rects[j].height;
    }
}

- (void)setVertexTextures:(const id __unsafe_unretained *)textures withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setVertexTexture:textures[i] atIndex:range.location + i];
}

- (void)setVertexSamplerStates:(const id __unsafe_unretained *)samplers withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setVertexSamplerState:samplers[i] atIndex:range.location + i];
}

- (void)setVertexSamplerState:(id)sampler lodMinClamp:(float)lodMin lodMaxClamp:(float)lodMax atIndex:(NSUInteger)index
{
    [self setVertexSamplerState:malachite_clamped(sampler, lodMin, lodMax) atIndex:index];
}

- (void)setVertexSamplerStates:(const id __unsafe_unretained *)samplers
                  lodMinClamps:(const float *)lodMin
                  lodMaxClamps:(const float *)lodMax
                     withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setVertexSamplerState:samplers[i] lodMinClamp:lodMin[i] lodMaxClamp:lodMax[i] atIndex:range.location + i];
}

- (void)setFragmentSamplerState:(id)sampler
                    lodMinClamp:(float)lodMin
                    lodMaxClamp:(float)lodMax
                        atIndex:(NSUInteger)index
{
    [self setFragmentSamplerState:malachite_clamped(sampler, lodMin, lodMax) atIndex:index];
}

- (void)setFragmentSamplerStates:(const id __unsafe_unretained *)samplers
                    lodMinClamps:(const float *)lodMin
                    lodMaxClamps:(const float *)lodMax
                       withRange:(NSRange)range
{
    for (NSUInteger i = 0; i < range.length; i++)
        [self setFragmentSamplerState:samplers[i] lodMinClamp:lodMin[i] lodMaxClamp:lodMax[i]
                              atIndex:range.location + i];
}

- (void)setDepthTestMinBound:(float)minBound maxBound:(float)maxBound
{
}

// tile shading is Apple-GPU state, not compiled
- (NSUInteger)tileWidth
{
    return 16;
}

- (NSUInteger)tileHeight
{
    return 16;
}

- (void)setThreadgroupMemoryLength:(NSUInteger)length offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
}

- (void)setTileBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
}

- (void)setTileBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
}

- (void)setTileBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
}

- (void)setTileBuffers:(const id __unsafe_unretained *)buffers
               offsets:(const NSUInteger *)offsets
             withRange:(NSRange)range
{
}

- (void)setTileTexture:(id)texture atIndex:(NSUInteger)index
{
}

- (void)setTileTextures:(const id __unsafe_unretained *)textures withRange:(NSRange)range
{
}

- (void)setTileSamplerState:(id)sampler atIndex:(NSUInteger)index
{
}

- (void)setTileSamplerStates:(const id __unsafe_unretained *)samplers withRange:(NSRange)range
{
}

- (void)setTileSamplerState:(id)sampler lodMinClamp:(float)lodMin lodMaxClamp:(float)lodMax atIndex:(NSUInteger)index
{
}

- (void)setTileSamplerStates:(const id __unsafe_unretained *)samplers
                lodMinClamps:(const float *)lodMin
                lodMaxClamps:(const float *)lodMax
                   withRange:(NSRange)range
{
}

- (void)dispatchThreadsPerTile:(MTLSize)threads
{
    os_log(OS_LOG_DEFAULT, "MalachiteRender: tile shading not supported");
}

- (void)setTessellationFactorBuffer:(id)buffer offset:(NSUInteger)offset instanceStride:(NSUInteger)stride
{
    _factors = buffer ? [(MTLIOAccelBuffer *)buffer gpuAddress] + offset : 0;
    _factor_stride = (uint32_t)stride;
    malachite_keep(_cb, buffer);
}

- (void)setTessellationFactorScale:(float)scale
{
    _factor_scale = scale;
}

// A patch draw is a draw of a vertex a patch: the tessellation init stage reads the index buffers. The control point
// count is the pipeline's, or the draw's when the function doesn't give one.
- (void)patches:(NSUInteger)points
          start:(NSUInteger)start
          count:(NSUInteger)count
     patchIndex:(id)patchIndex
         offset:(NSUInteger)patchOffset
     pointIndex:(id)pointIndex
         offset:(NSUInteger)pointOffset
      instances:(NSUInteger)instances
   baseInstance:(NSUInteger)baseInstance
{
    malachite_keep(_cb, patchIndex);
    malachite_keep(_cb, pointIndex);
    _patch_index = patchIndex ? [(MTLIOAccelBuffer *)patchIndex gpuAddress] + patchOffset : 0;
    _point_index = pointIndex ? [(MTLIOAccelBuffer *)pointIndex gpuAddress] + pointOffset : 0;
    _patch_points = points;
    _patching = true;
    [self draw:MTLPrimitiveTypePoint first:start count:count instances:instances baseVertex:0
        baseInstance:baseInstance index:nil offset:0 type:MTLIndexTypeUInt16];
    _patching = false;
}

- (void)drawPatches:(NSUInteger)controlPoints
           patchStart:(NSUInteger)start
           patchCount:(NSUInteger)count
     patchIndexBuffer:(id)patchIndex
    patchIndexBufferOffset:(NSUInteger)patchIndexOffset
        instanceCount:(NSUInteger)instances
         baseInstance:(NSUInteger)baseInstance
{
    [self patches:controlPoints start:start count:count patchIndex:patchIndex offset:patchIndexOffset pointIndex:nil
           offset:0 instances:instances baseInstance:baseInstance];
}

// MTLDrawPatchIndirectArguments are laid out as a draw of vertices' are: the kernel reads them as it does those
- (void)drawPatches:(NSUInteger)controlPoints
          patchIndexBuffer:(id)patchIndex
    patchIndexBufferOffset:(NSUInteger)patchIndexOffset
            indirectBuffer:(id)indirect
      indirectBufferOffset:(NSUInteger)indirectOffset
{
    malachite_keep(_cb, indirect);
    _indirect = [(MTLIOAccelBuffer *)indirect gpuAddress] + indirectOffset;
    [self patches:controlPoints start:0 count:1 patchIndex:patchIndex offset:patchIndexOffset pointIndex:nil offset:0
        instances:1 baseInstance:0];
    _indirect = 0;
}

- (void)drawIndexedPatches:(NSUInteger)controlPoints
                     patchStart:(NSUInteger)start
                     patchCount:(NSUInteger)count
               patchIndexBuffer:(id)patchIndex
         patchIndexBufferOffset:(NSUInteger)patchIndexOffset
        controlPointIndexBuffer:(id)controlPointIndex
    controlPointIndexBufferOffset:(NSUInteger)controlPointIndexOffset
                  instanceCount:(NSUInteger)instances
                   baseInstance:(NSUInteger)baseInstance
{
    [self patches:controlPoints start:start count:count patchIndex:patchIndex offset:patchIndexOffset
       pointIndex:controlPointIndex offset:controlPointIndexOffset instances:instances baseInstance:baseInstance];
}

- (void)drawIndexedPatches:(NSUInteger)controlPoints
                 patchIndexBuffer:(id)patchIndex
           patchIndexBufferOffset:(NSUInteger)patchIndexOffset
          controlPointIndexBuffer:(id)controlPointIndex
    controlPointIndexBufferOffset:(NSUInteger)controlPointIndexOffset
                   indirectBuffer:(id)indirect
             indirectBufferOffset:(NSUInteger)indirectOffset
{
    malachite_keep(_cb, indirect);
    _indirect = [(MTLIOAccelBuffer *)indirect gpuAddress] + indirectOffset;
    [self patches:controlPoints start:0 count:1 patchIndex:patchIndex offset:patchIndexOffset
       pointIndex:controlPointIndex offset:controlPointIndexOffset instances:1 baseInstance:0];
    _indirect = 0;
}

// The kernel reads the arguments when it runs the draw, after the work before it, which may write them; the
// draw is encoded with stand-ins
- (void)drawPrimitives:(MTLPrimitiveType)type indirectBuffer:(id)buffer indirectBufferOffset:(NSUInteger)offset
{
    malachite_keep(_cb, buffer);
    _indirect = [(MTLIOAccelBuffer *)buffer gpuAddress] + offset;
    [self drawPrimitives:type vertexStart:0 vertexCount:1 instanceCount:1 baseInstance:0];
    _indirect = 0;
}

- (void)drawIndexedPrimitives:(MTLPrimitiveType)type
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id)index
            indexBufferOffset:(NSUInteger)offset
               indirectBuffer:(id)buffer
         indirectBufferOffset:(NSUInteger)indirectOffset
{
    malachite_keep(_cb, buffer);
    _indirect = [(MTLIOAccelBuffer *)buffer gpuAddress] + indirectOffset;
    [self drawIndexedPrimitives:type indexCount:1 indexType:indexType indexBuffer:index indexBufferOffset:offset
                  instanceCount:1 baseVertex:0 baseInstance:0];
    _indirect = 0;
}

// Every draw call comes here. An indexed one passes the index buffer (from its offset) and its size; first
// and count are then indices.
- (void)draw:(MTLPrimitiveType)type
         first:(NSUInteger)first
         count:(NSUInteger)count
     instances:(NSUInteger)instances
    baseVertex:(NSInteger)baseVertex
  baseInstance:(NSUInteger)baseInstance
         index:(id)index
        offset:(NSUInteger)offset
          type:(MTLIndexType)indexType
{
    static const uint32_t topology[] = { 0, 1, 3, 4, 5 };  // point, line, line strip, triangle, triangle strip
    // a post-tessellation pipeline draws patches only, with a factor buffer
    bool tess = _pipeline && _pipeline->_tess;
    id<MTLBuffer> ti = nil;
    uint32_t ti_regs = 0;
    if (tess && _patching) {
        NSString *why;
        ti = [_pipeline tessInit:_patch_points device:((id<MTLCommandBuffer>)_cb).device regs:&ti_regs why:&why];
        if (!ti)
            os_log(OS_LOG_DEFAULT, "Malachite: %{public}@: %{public}@", _pipeline->_label, why);
    }
    if (!_pass.width || !_pipeline || (NSUInteger)type >= sizeof topology / sizeof topology[0] || !instances ||
        tess != _patching || (tess && (!_factors || !ti))) {
        if (malachite_trace)
            os_log(OS_LOG_DEFAULT,
                   "Malachite trace: draw dropped: pass %ux%u, pipeline %{public}@, type %lu, %lu instances, patches %d, "
                   "factors %#llx",
                   _pass.width, _pass.height, _pipeline ? _pipeline->_label : @"none", (unsigned long)type,
                   (unsigned long)instances, _patching, _factors);
        return;
    }
    struct nvgpu_draw d = _pass;
    d.pass_first = !_drawn;
    d.opengl = opengl_renderer;
    if (_drawn) {  // the load actions clear once, before the first draw
        d.clear = 0;
        d.clear_zs = 0;
        for (struct nvgpu_target &t : d.more)
            t.clear = 0;
    }
    if (_ds) {
        d.depth_test = _ds->_depth_test;
        d.depth_write = _ds->_depth_write;
        d.depth_func = _ds->_depth_func;
        d.stencil_test = _ds->_stencil_test;
        memcpy(d.stencil, _ds->_stencil, sizeof(d.stencil));
        memcpy(d.stencil_read_mask, _ds->_read_mask, sizeof(d.stencil_read_mask));
        memcpy(d.stencil_write_mask, _ds->_write_mask, sizeof(d.stencil_write_mask));
    }
    d.vs = [(MTLIOAccelBuffer *)_pipeline->_vcode gpuAddress];
    d.ps = [(MTLIOAccelBuffer *)_pipeline->_fcode gpuAddress];
    d.vs_regs = _pipeline->_vertex.reg_count;
    d.ps_regs = _pipeline->_fragment.reg_count;
    d.vs_params_size = malachite_fill_params(&_pipeline->_vertex, _vbuffers, d.vs, d.vs_params, sizeof(d.vs_params));
    if (malachite_trace)
        for (uint32_t i = 0; i < _pipeline->_vertex.nargs; i++)
            os_log(OS_LOG_DEFAULT, "Malachite trace:   vertex argument %u: kind %u index %u at %#x", i,
                   _pipeline->_vertex.args[i].kind, _pipeline->_vertex.args[i].index,
                   _pipeline->_vertex.args[i].cbuf_off);
    d.ps_params_size = malachite_fill_params(&_pipeline->_fragment, _fbuffers, d.ps, d.ps_params, sizeof(d.ps_params));
    if (_pipeline == _pools_pipeline && !memcmp(_pools_bound[0], _textures, sizeof(_textures)) &&
        !memcmp(_pools_bound[1], _samplers, sizeof(_samplers)) && !memcmp(_pools_bound[2], _vtextures, sizeof(_vtextures)) &&
        !memcmp(_pools_bound[3], _vsamplers, sizeof(_vsamplers))) {
        d.heap = _pools_heap;
        d.nheaders = _pools_nheaders;
        d.nsamplers = _pools_nsamplers;
        d.vs_textured = _pools_vs_textured;
        if (d.vs_textured)
            memcpy(d.vs_handles, _pools_vs_handles, sizeof(d.vs_handles));
    } else if (![self fillPools:d]) {
        return;
    }
    d.write_mask = _pipeline->_write_mask[0];
    memcpy(d.blend, _pipeline->_blend[0], sizeof(d.blend));
    d.alpha_control = _pipeline->_alpha_control;
    d.per_sample = (_pipeline->_fragment.flags & NVK_FLAG_PER_SAMPLE) != 0;
    d.ps_depth = (_pipeline->_fragment.flags & NVK_FLAG_DEPTH) != 0;
    d.shader_layer = (_pipeline->_vertex.flags & NVK_FLAG_LAYER) != 0;
    for (unsigned j = 0; j < 7; j++) {
        d.more[j].write_mask = _pipeline->_write_mask[j + 1];
        memcpy(d.more[j].blend, _pipeline->_blend[j + 1], sizeof(d.more[j].blend));
    }
    memcpy(d.attributes, _pipeline->_attributes, sizeof(d.attributes));
    id<MTLBuffer> zeros = nil;
    uint32_t used = 0;  // the streams active attributes read, a constant step function's (stride 0) among them
    for (int i = 0; i < RENDER_ATTRIBUTES; i++)
        if (d.attributes[i] && !(d.attributes[i] & ATTR_INACTIVE))
            used |= 1u << (d.attributes[i] & 0x1f);
    for (int s = 0; s < 16; s++) {
        if (!_pipeline->_strides[s] && !_pipeline->_divisors[s] && !(used >> s & 1))
            continue;
        unsigned b = _pipeline->_stream_buffer[s];
        d.streams[s].addr = _vbuffers[b];
        d.streams[s].stride = _pipeline->_strides[s] == MALACHITE_DYNAMIC_STRIDE ? _vstrides[b] : _pipeline->_strides[s];
        d.streams[s].size = _vsizes[b];
        d.streams[s].divisor = _pipeline->_divisors[s];
        // a stream whose buffer isn't bound would leave its attributes fetching from address 0, which faults the
        // channel (WindowServer, under load): they read zeros instead
        if (!d.streams[s].addr) {
            if (!zeros) {
                zeros = malachite_zero_buffer(((id<MTLCommandBuffer>)_cb).device);
                malachite_keep(_cb, zeros);
            }
            d.streams[s].addr = [(MTLIOAccelBuffer *)zeros gpuAddress];
            d.streams[s].size = (uint32_t)zeros.length;
            d.streams[s].stride = 0;
        }
    }
    if (_pipeline->_zero_stream) {
        id<MTLBuffer> zeros = malachite_zero_buffer(((id<MTLCommandBuffer>)_cb).device);
        malachite_keep(_cb, zeros);
        unsigned z = _pipeline->_zero_stream - 1;
        d.streams[z].addr = [(MTLIOAccelBuffer *)zeros gpuAddress];
        d.streams[z].size = (uint32_t)zeros.length;
        d.streams[z].divisor = _pipeline->_divisors[z];
    }
    d.indirect = _indirect;
    if (d.visibility) {
        d.visibility_reset = d.visibility != _counted;
        _counted = d.visibility;
    }
    d.primitive = topology[type];
    if (tess) {  // the vertex function runs as the tessellation stage, after the generated vertex and init stages
        d.tess = _pipeline->_tess;
        d.primitive = 0xe;
        d.ts = d.vs;
        d.ts_regs = d.vs_regs;
        d.vs = [(MTLIOAccelBuffer *)_pipeline->_tvcode gpuAddress];
        d.vs_regs = _pipeline->_tv.reg_count;
        d.ti = [(MTLIOAccelBuffer *)ti gpuAddress];
        d.ti_regs = ti_regs;
        const uint64_t buffers[3] = { _factors, _patch_index, _point_index };
        float scale = _pipeline->_tess_scaled ? _factor_scale : 1.0f;
        memcpy(d.ti_params + AIRC_TESS_FACTORS - 0x160, buffers, sizeof(buffers));
        memcpy(d.ti_params + AIRC_TESS_INSTANCE_STRIDE - 0x160, &_factor_stride, 4);
        memcpy(d.ti_params + AIRC_TESS_SCALE - 0x160, &scale, 4);
        memcpy(d.ti_params + AIRC_TESS_BUFFER(0) - 0x160, _vbuffers, AIRC_TESS_BUFFERS * 8);
        d.ti_params_size = NVGPU_PARAM_BYTES;
    }
    d.first = (uint32_t)first;
    d.count = (uint32_t)count;
    d.instances = (uint32_t)instances;
    d.base_vertex = (int)baseVertex;
    d.base_instance = (uint32_t)baseInstance;
    if (index) {
        d.index = [(MTLIOAccelBuffer *)index gpuAddress] + offset;
        malachite_keep(_cb, index);
        d.index_size = (uint32_t)(((id<MTLBuffer>)index).length - offset);
        d.index_format = indexType == MTLIndexTypeUInt32 ? 2 : 1;
    }
    if (!_drawn)
        [self loadAttachments];
    if ((d.target & 127) || (d.pitch & 127) || d.vs_params_size > NVGPU_PARAM_BYTES ||
        d.ps_params_size > NVGPU_PARAM_BYTES || !d.vs || !d.ps || d.nheaders > NVGPU_MAX_TEXTURES ||
        d.nsamplers > NVGPU_MAX_TEXTURES || !d.width || !d.height)
        os_log(OS_LOG_DEFAULT,
               "MalachiteRender: bad draw: target 0x%llx pitch %u %ux%u params %u/%u textures %u/%u code 0x%llx/0x%llx",
               d.target, d.pitch, d.width, d.height, d.vs_params_size, d.ps_params_size, d.nheaders, d.nsamplers,
               d.vs, d.ps);
    if (malachite_trace)
        os_log(OS_LOG_DEFAULT,
               "Malachite trace: draw %{public}@: type %lu first %u count %u instances %u%s, textures %u, viewport %.0f %.0f "
               "%.0f %.0f, scissor %u %u %u %u, depth %u/%u func %#x, stencil %u ops %#x %#x %#x func %#x ref %u "
               "masks %#x/%#x, back ops %#x %#x %#x func %#x, masks %#x %#x, blend1 %u %#x %#x %#x, zt %#llx format %#x "
               "%ux%u kind %#x, target %ux%u",
               _pipeline->_label, (unsigned long)type, d.first, d.count, d.instances, index ? " indexed" : "",
               d.nheaders, d.viewport[0], d.viewport[1], d.viewport[2], d.viewport[3], d.scissor[0], d.scissor[1],
               d.scissor[2], d.scissor[3], d.depth_test, d.depth_write, d.depth_func, d.stencil_test, d.stencil[0][0],
               d.stencil[0][1], d.stencil[0][2], d.stencil[0][3], d.stencil_ref, d.stencil_read_mask[0],
               d.stencil_write_mask[0], d.stencil[1][0], d.stencil[1][1], d.stencil[1][2], d.stencil[1][3],
               d.write_mask, d.more[0].write_mask, d.more[0].blend[0], d.more[0].blend[1], d.more[0].blend[2],
               d.more[0].blend[3], d.zt, d.zt_format, d.zt_width, d.zt_height, d.zt_kind, d.width, d.height);
    if (malachite_trace) {  // the addresses the draw reads, to tell which one an MMU fault was at
        char streams[256] = "";
        for (unsigned s = 0, n = 0; s < 16 && n < sizeof(streams); s++)
            if (d.streams[s].addr)
                n += snprintf(streams + n, sizeof(streams) - n, " %u:%#llx/%u/%u", s, d.streams[s].addr,
                              d.streams[s].stride, d.streams[s].size);
        // and the parameter words that look like our GPU addresses: the buffers bound as arguments
        char params[256] = "";
        const struct {
            const unsigned char *p;
            unsigned size;
            char stage;
        } stages[] = { { d.vs_params, d.vs_params_size, 'v' }, { d.ps_params, d.ps_params_size, 'f' } };
        for (unsigned s = 0, n = 0; s < 2; s++)
            for (unsigned o = 0; o + 8 <= stages[s].size && n < sizeof(params); o += 8) {
                uint64_t w;
                memcpy(&w, stages[s].p + o, 8);
                if (w >= 0x20000000ull && w < 1ull << 40)
                    n += snprintf(params + n, sizeof(params) - n, " %c%u:%#llx", stages[s].stage, o, w);
            }
        char headers[256] = "";
        for (unsigned h = 0, n = 0; h < d.nheaders && h < NVGPU_MAX_TEXTURES && n < sizeof(headers); h++)
            n += snprintf(headers + n, sizeof(headers) - n, " %u:%08x/%08x", h, d.headers[h][1], d.headers[h][2]);
        for (unsigned t = 0, n = strlen(streams); t < 7 && n < sizeof(streams); t++)  // targets 1..7 too
            if (d.more[t].addr)
                n += snprintf(streams + n, sizeof(streams) - n, " target%u:%#llx%s", t + 1, d.more[t].addr,
                              _attached[t + 1] && _attached[t + 1]->_surface ? "(iosurface)" : "");
        os_log(OS_LOG_DEFAULT,
               "Malachite trace: addresses (command buffer %p): target %#llx pitch %u, vs %#llx ps %#llx, index %#llx, "
               "streams%{public}s, params%{public}s, headers%{public}s",
               _cb, d.target, d.pitch, d.vs, d.ps, d.index, streams, params, headers);
    }
    for (unsigned b = 0; malachite_snap && b < 4; b++) {
        const uint32_t *w = _vbound[b] ? (const uint32_t *)((const uint8_t *)((id<MTLBuffer>)_vbound[b]).contents +
                                                            (_vbuffers[b] - [(MTLIOAccelBuffer *)_vbound[b] gpuAddress]))
                                       : NULL;
        if (_vbuffers[b])
            os_log(OS_LOG_DEFAULT, "Malachite trace:   vertex buffer %u: %#llx size %u%s words %#x %#x %#x %#x", b,
                   _vbuffers[b], _vsizes[b], _vbound[b] ? "" : " (bytes)", w ? w[0] : 0, w ? w[1] : 0, w ? w[2] : 0,
                   w ? w[3] : 0);
        char path[96];
        snprintf(path, sizeof(path), "/tmp/malachite-vb-%d-%s-%u.bin", getpid(),
                 [_pipeline->_label stringByReplacingOccurrencesOfString:@"/" withString:@"_"].UTF8String, b);
        if (w && access(path, F_OK) != 0) {  // each vertex function's first draw, for replaying it
            FILE *o = fopen(path, "wb");
            if (o) {
                fwrite(w, 1, _vsizes[b] < 16384 ? _vsizes[b] : 16384, o);
                fclose(o);
            }
            // and the same bytes once the GPU is done with them, to see what changed after the draw was encoded
            size_t len = _vsizes[b] < 16384 ? _vsizes[b] : 16384;
            id keep = _vbound[b];
            NSString *done = [NSString stringWithFormat:@"%s.done", path];
            [(id<MTLCommandBuffer>)_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                (void)keep;
                [[NSData dataWithBytes:w length:len] writeToFile:done atomically:NO];
            }];
        }
    }
    // a filtered snapshot also has the targets before the draw and the textures it reads
    bool only = false;
    for (NSString *s in malachite_snap_only)
        only |= malachite_snap && [_pipeline->_label containsString:s];  // malachite_snap is off while the gate's shut
    const uint32_t *r = malachite_snap_rect, *sc = d.scissor;
    bool in_rect = r[2] && d.width >= r[0] + r[2] && d.height >= r[1] + r[3] && sc[0] < r[0] + r[2] &&
                   r[0] < sc[0] + sc[2] && sc[1] < r[1] + r[3] && r[1] < sc[1] + sc[3];
    if (malachite_snap && in_rect)
        only = true;
    if (only) {
        [self snapshot:d suffix:"pre"];
        if (!malachite_snap_rect[2])
            [self snapshotTextures:d];
    }
    bool named = false;
    for (NSString *s in malachite_rec_only)
        named |= [_pipeline->_label containsString:s];
    NSString *rec = nil;
    NSMutableString *manifest = nil;
    if (named && !malachite_rec_shut && !malachite_rec_full && !tess && d.samples <= 1 && (!r[2] || in_rect)) {
        static int recordings;
        rec = [NSString stringWithFormat:@"/tmp/malachite-rec-%d-%d", getpid(), __sync_fetch_and_add(&recordings, 1)];
        manifest = [NSMutableString string];
        [self record:rec manifest:manifest draw:d type:type first:first count:count instances:instances
            baseVertex:baseVertex baseInstance:baseInstance index:index offset:offset indexType:indexType];
    }
    d.serialize |= _barrier;
    _barrier = false;
    d.local_size = MAX(_pipeline->_vertex.local_size, _pipeline->_fragment.local_size);
    if (![self fastDraw:d]) {
        malachite_record(_cb, MALACHITE_CMD_DRAW, &d, sizeof(d));
        _last_records = 0;
        if (_fast && d.count && d.instances) {  // the kernel sends no state for a draw of nothing
            MalachiteCommandBuffer *cb = (MalachiteCommandBuffer *)_cb;
            _last = d;
            _last_records = cb->_records;
            cb->_state_of = (__bridge const void *)self;
            _seen_code[_seen_next][0] = d.vs;
            _seen_code[_seen_next][1] = d.ps;
            _seen_next = (_seen_next + 1) % 16;
        }
    }
    _drawn = true;
    if (rec)
        [self record:rec manifest:manifest after:d];
    for (unsigned n = 0; n < 8; n++) {
        uint32_t *w = _written[n];
        if (!_copied[n] || !d.scissor[2] || !d.scissor[3])
            continue;
        bool empty = w[2] <= w[0] || w[3] <= w[1];
        w[0] = empty ? d.scissor[0] : MIN(w[0], d.scissor[0]);
        w[1] = empty ? d.scissor[1] : MIN(w[1], d.scissor[1]);
        w[2] = empty ? d.scissor[0] + d.scissor[2] : MAX(w[2], d.scissor[0] + d.scissor[2]);
        w[3] = empty ? d.scissor[1] + d.scissor[3] : MAX(w[3], d.scissor[1] + d.scissor[3]);
    }
    if (malachite_snap && ((!malachite_snap_only && !malachite_snap_rect[2]) || only))
        [self snapshot:d suffix:""];
}

// the trace: the textures a draw reads, level 0 of each, saved like the snapshots
- (void)snapshotTextures:(const struct nvgpu_draw &)d
{
    static int draws;
    int n = __sync_fetch_and_add(&draws, 1);
    // what else the draw reads: its samplers' filters, and the start of each fragment buffer (shared ones)
    for (unsigned slot = 0; slot < NVGPU_MAX_TEXTURES; slot++)
        if ([_samplers[slot] isKindOfClass:[MalachiteSamplerState class]]) {
            const struct samp_desc *sd = &((MalachiteSamplerState *)_samplers[slot])->_desc;
            os_log(OS_LOG_DEFAULT, "Malachite trace: snapshot %d sampler %u: mag %d min %d mip %d normalized %d", n, slot,
                   sd->mag_linear, sd->min_linear, sd->mip, sd->normalized);
        }
    for (unsigned b = 0; b < 8; b++) {
        id<MTLBuffer> buf = _fbound[b];
        if (!_fbuffers[b] || ![buf respondsToSelector:@selector(contents)] || !buf.contents)
            continue;
        const float *f = (const float *)((const uint8_t *)buf.contents + (_fbuffers[b] - [(MTLIOAccelBuffer *)buf gpuAddress]));
        for (unsigned k = 0; k < 48; k += 12)
            os_log(OS_LOG_DEFAULT, "Malachite trace: snapshot %d fragment buffer %u [%u]: %g %g %g %g %g %g %g %g %g %g %g %g", n, b,
                   k, f[k], f[k + 1], f[k + 2], f[k + 3], f[k + 4], f[k + 5], f[k + 6], f[k + 7], f[k + 8], f[k + 9],
                   f[k + 10], f[k + 11]);
    }
    for (unsigned slot = 0; slot < NVGPU_MAX_TEXTURES; slot++) {
        MalachiteTexture *t = (MalachiteTexture *)_textures[slot];
        if (![t isKindOfClass:[MalachiteTexture class]] || !t->_backing)
            continue;
        // an image's levels as the GOBs they are (one-GOB blocks, rows of pitch bytes 8 lines high), level after
        // level: "W H pitch format label image levels" then each level's "w h pitch rows" line and bytes
        for (uint32_t level = 0; t->_image && level < t->_levels && level < 6; level++) {
            struct malachite_sub s;
            if (!malachite_sub(t, level, 0, 0, &s) || !s.rows || (uint64_t)s.pitch * s.rows > 64u << 20)
                break;
            uint32_t size = s.pitch * s.rows;
            id<MTLBuffer> snap = [((id<MTLCommandBuffer>)_cb).device newBufferWithLength:size
                                                                                options:MTLResourceStorageModeShared];
            struct nvgpu_copy2d c = { [(MTLIOAccelBuffer *)t->_backing gpuAddress] + s.offset,
                                      [(MTLIOAccelBuffer *)snap gpuAddress], s.pitch, s.pitch, s.pitch, s.rows };
            malachite_record(_cb, MALACHITE_CMD_COPY_2D, &c, sizeof(c));
            MTLPixelFormat f = t.pixelFormat;
            NSString *label = _pipeline->_label;
            uint32_t w = s.width, h = s.height, pitch = s.pitch, rows = s.rows;
            [(id<MTLCommandBuffer>)_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                char path[128];
                snprintf(path, sizeof(path), "/tmp/malachite-img-%d-%04d-%u-%u.raw", getpid(), n, slot, level);
                FILE *o = fopen(path, "wb");
                if (!o)
                    return;
                fprintf(o, "%u %u %u %u %lu %s\n", w, h, pitch, rows, (unsigned long)f, label.UTF8String);
                fwrite(snap.contents, 1, size, o);
                fclose(o);
            }];
        }
        if (t->_image)
            continue;
        uint32_t w = (uint32_t)t.width, h = (uint32_t)t.height, bytes = w * (uint32_t)malachite_bytes_per_pixel(t.pixelFormat);
        uint32_t pitch = (bytes + 255) & ~255u;
        if (!bytes || pitch * h > 64u << 20)
            continue;
        id<MTLBuffer> snap = [((id<MTLCommandBuffer>)_cb).device newBufferWithLength:(NSUInteger)pitch * h
                                                                            options:MTLResourceStorageModeShared];
        struct nvgpu_copy2d c = { [(MTLIOAccelBuffer *)t->_backing gpuAddress] + t->_offset,
                                  [(MTLIOAccelBuffer *)snap gpuAddress], t->_pitch, pitch, bytes, h };
        malachite_record(_cb, MALACHITE_CMD_COPY_2D, &c, sizeof(c));
        MTLPixelFormat f = t.pixelFormat;
        NSString *label = _pipeline->_label;
        [(id<MTLCommandBuffer>)_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
            char path[128];
            snprintf(path, sizeof(path), "/tmp/malachite-tex-%d-%04d-%u.raw", getpid(), n, slot);
            FILE *o = fopen(path, "wb");
            if (!o)
                return;
            fprintf(o, "%u %u %u %lu %s\n", w, h, pitch, (unsigned long)f, label.UTF8String);
            fwrite(snap.contents, 1, (size_t)pitch * h, o);
            fclose(o);
        }];
    }
}

// colour target n of d with its size, which may be more than the pass's
static struct nvgpu_target malachite_target(const struct nvgpu_draw &d, unsigned n)
{
    struct nvgpu_target ct = { d.target, d.pitch, 0, d.target_blocklinear };
    ct.width = d.target_width;
    ct.height = d.target_height;
    if (n)
        ct = d.more[n - 1];
    if (!ct.width) {
        ct.width = d.width;
        ct.height = d.height;
    }
    return ct;
}

// attachment n's texture header, at the memory the pass draws it to
- (void)attachment:(unsigned)n header:(unsigned int *)header of:(const struct nvgpu_draw &)d
{
    struct nvgpu_target ct = malachite_target(d, n);
    struct tex_desc td = { ct.addr, ct.width, ct.height, ct.pitch, (enum tex_format)_formats[n], (int)ct.blocklinear };
    tex_fill_header(header, &td);
}

// scratch memory for attachment n's copy, size bytes, one pass after another reusing it: the draws and copies
// after the one that fills it wait for the ones before. In VRAM, as the attachments are: a copy into memory the GPU
// reaches over PCIe was most of the copy's time
static uint64_t malachite_scratch(id<MTLDevice> device, unsigned n, NSUInteger size)
{
    static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *scratch;
    NSNumber *key = @(size * 8 + n);
    id<MTLBuffer> buf;
    @synchronized([MalachiteRenderCommandEncoder class]) {
        if (!scratch)
            scratch = [NSMutableDictionary dictionary];
        buf = scratch[key];
        if (!buf) {
            buf = [device newBufferWithLength:size options:MTLResourceStorageModePrivate];
            if (!buf)
                buf = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
            scratch[key] = buf;
        }
    }
    return [(MTLIOAccelBuffer *)buf gpuAddress];
}

// A copy of a texel size bytes wide drawn as one raw integer of that size: a full-screen triangle whose pixels
// read the texel and write it out as it is, so any format copies exactly and the copy stays in the 3D channel,
// whose own draws it waits for, rather than going to the copy engine and back
static const struct {
    uint32_t bytes;
    MTLPixelFormat pixel;
    enum tex_format format;
} malachite_raw[] = {
    { 1, MTLPixelFormatR8Uint, TEX_R8_UINT },       { 2, MTLPixelFormatR16Uint, TEX_R16_UINT },
    { 4, MTLPixelFormatR32Uint, TEX_R32_UINT },     { 8, MTLPixelFormatRG32Uint, TEX_RG32_UINT },
    { 16, MTLPixelFormatRGBA32Uint, TEX_RGBA32_UINT },
};

static MalachiteRenderPipelineState *malachite_copy_pipeline(id<MTLDevice> device, unsigned raw)
{
    static MalachiteRenderPipelineState *states[sizeof(malachite_raw) / sizeof(malachite_raw[0])];
    static id<MTLLibrary> lib;
    @synchronized([MalachiteRenderCommandEncoder class]) {
        if (!states[raw]) {
            NSError *err;
            if (!lib)
                lib = [device newLibraryWithSource:@"#include <metal_stdlib>\n"
                                                    "using namespace metal;\n"
                                                    "struct V { float4 position [[position]]; };\n"
                                                    "vertex V malachite_copy_vertex(uint v [[vertex_id]])\n"
                                                    "{\n"
                                                    "    float2 p = float2((v << 1) & 2, v & 2);\n"
                                                    "    return { float4(p * 2.0f - 1.0f, 0.0f, 1.0f) };\n"
                                                    "}\n"
                                                    "fragment uint4 malachite_copy_fragment(V in [[stage_in]],\n"
                                                    "                                   texture2d<uint> src [[texture(0)]])\n"
                                                    "{\n"
                                                    "    return src.read(uint2(in.position.xy));\n"
                                                    "}\n"
                                           options:nil
                                             error:&err];
            MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
            desc.vertexFunction = [lib newFunctionWithName:@"malachite_copy_vertex"];
            desc.fragmentFunction = [lib newFunctionWithName:@"malachite_copy_fragment"];
            desc.colorAttachments[0].pixelFormat = malachite_raw[raw].pixel;
            states[raw] = desc.vertexFunction && desc.fragmentFunction
                              ? [device newRenderPipelineStateWithDescriptor:desc error:&err]
                              : nil;
            if (!states[raw])
                os_log(OS_LOG_DEFAULT, "Malachite: copy pipeline: %{public}@", err.localizedDescription);
        }
        return states[raw];
    }
}

// The GPU methods a fast draw uses, from core/render.c
#define FAST_MTHD(m, n) (0x20000000u | (uint32_t)(n) << 16 | (uint32_t)(m) >> 2)
enum {
    FAST_INDEX_SIZE = 0x0238,
    FAST_DRAW_CONTROL = 0x0260,
    FAST_DRAW_INDEXED = 0x0268,
    FAST_DRAW_ARRAYS = 0x0270,
    FAST_STREAM_SIZE = 0x0600,
    FAST_INVALIDATE_SHADER = 0x0da4,
    FAST_VERTEX_ID_BASE = 0x1118,
    FAST_INVALIDATE_TEX_DATA = 0x1288,
    FAST_INVALIDATE_SAMPLER = 0x1424,
    FAST_INVALIDATE_HEADER = 0x1428,
    FAST_BASE_VERTEX = 0x1434,
    FAST_SAMPLER_POOL = 0x155c,
    FAST_HEADER_POOL = 0x1574,
    FAST_RESTART = 0x1644,
    FAST_INDEX_ADDRESS = 0x17c8,
    FAST_STREAM_INSTANCE = 0x1880,
    FAST_STREAM_FORMAT = 0x1c00,
    FAST_CONSTANT_SELECTOR = 0x2380,
    FAST_CONSTANT_BIND = 0x2410,
    FAST_BANK_BYTES = 0x1000,  // as bound; it reads no further than FAST_BANK_USED
    FAST_BANK_USED = 0x280,
    FAST_BANK_STEP = 0x300,
    FAST_RUN_WORDS = 0x4000,
    FAST_DRAW_WORDS = 1024,    // more than one draw can take
    FAST_RUN_DRAWS = 256,      // the GPU gets on with them
};

// Room for one more draw in the run's memory, which lives until the command buffer completes.
static bool malachite_run_room(MalachiteCommandBuffer *cb)
{
    if (!cb->_run_cpu || cb->_run_cap - cb->_run_used < FAST_DRAW_WORDS) {
        malachite_run_flush(cb);
        void *cpu;
        uint64_t va;
        if (!malachite_reserve(cb, FAST_RUN_WORDS * 4, &va, &cpu))
            return false;
        cb->_run_cpu = (uint32_t *)cpu;
        cb->_run_va = va;
        cb->_run_start = cb->_run_used = 0;
        cb->_run_cap = FAST_RUN_WORDS;
    }
    if (!cb->_bank_cpu || cb->_bank_cap - cb->_bank_used < FAST_BANK_STEP * 2 + FAST_BANK_BYTES) {
        void *cpu;
        uint64_t va;
        if (!malachite_reserve(cb, MALACHITE_UPLOAD_BLOCK, &va, &cpu))
            return false;
        cb->_bank_cpu = (uint8_t *)cpu;
        cb->_bank_va = va;
        cb->_bank_used = 0;
        cb->_bank_cap = MALACHITE_UPLOAD_BLOCK;
        cb->_bank_new = true;
    }
    return true;
}

// The textures and samplers of a draw, as heap entries. A draw with the same pipeline and bindings as the last uses the
// same ones, unless it reads an attachment, which takes a copy of its own.
- (bool)fillPools:(struct nvgpu_draw &)d
{
    bool cacheable = true;
    const struct nvk_header *fh = &_pipeline->_fragment;
    __unsafe_unretained id textures[NVGPU_MAX_TEXTURES], samplers[NVGPU_MAX_TEXTURES];
    malachite_slots(fh, _textures, _samplers, textures, samplers);
    if (malachite_fill_pools(_cb, fh, textures, samplers, d.headers, d.samplers, &d.nheaders, &d.nsamplers))
        return false;
    // [[color(n)]] inputs read the attachment where it's drawn: the shadow, if it has one. CoreAnimation's glass
    // needs them (zeros leave its material garbage); a read of an attachment the pass hasn't gets zeros
    for (uint32_t i = 0; i < fh->nargs; i++) {
        unsigned n = fh->args[i].index, slot = NVK_RENDER_TARGET_SLOT(n);
        if (fh->args[i].kind != NVK_ARG_RENDER_TARGET)
            continue;
        if (n >= 8 || _formats[n] < 0) {
            memset(d.headers[slot], 0, sizeof(d.headers[slot]));  // the unbound texture's: reads give 0
        } else {
            [self attachment:n header:d.headers[slot] of:d];
            d.serialize = 1;  // it sees everything drawn before it once that's done
            cacheable = false;
        }
        if (slot + 1 > d.nheaders)
            d.nheaders = slot + 1;
    }
    // an attachment sampled, through its own texture or another over the same memory and only its level: a
    // downsample that draws level n of an image while sampling level n - 1, through a view of either or of all its
    // levels, isn't one (Metal lets a pass read the levels it doesn't draw)
    for (unsigned slot = 0; slot < d.nheaders; slot++) {
        MalachiteTexture *t = textures[slot];
        if (![t isKindOfClass:[MalachiteTexture class]] || !t->_backing)
            continue;
        uint64_t at = [(MTLIOAccelBuffer *)t->_backing gpuAddress] + t->_offset;
        for (unsigned n = 0; n < 8; n++)
            if (_attached[n] && _attached[n]->_backing &&
                at == [(MTLIOAccelBuffer *)_attached[n]->_backing gpuAddress] + _attached[n]->_offset &&
                (!t->_image || (t->_levels == 1 && _attached_level[n] == t->_view_min))) {
                [self attachment:n copy:d.headers[slot] of:d];
                cacheable = false;
            }
    }
    if (!d.nsamplers && d.nheaders) {  // as malachite_fill_pools gives a pool with headers a sampler 0
        struct samp_desc def = malachite_default_sampler();
        tex_fill_sampler(d.samplers[0], &def);
        d.nsamplers = 1;
    }
    if (malachite_heap_commit(_cb, d.headers, d.nheaders, d.samplers, d.nsamplers, &d.heap))
        return false;
    // the vertex stage's textures: headers of their own in the heap, and a handle table of their own
    unsigned int vheaders[NVGPU_MAX_TEXTURES][8] = {}, vsamplers[NVGPU_MAX_TEXTURES][8] = {}, nvh = 0, nvs = 0;
    __unsafe_unretained id vtextures[NVGPU_MAX_TEXTURES], vsamplers_bound[NVGPU_MAX_TEXTURES];
    malachite_slots(&_pipeline->_vertex, _vtextures, _vsamplers, vtextures, vsamplers_bound);
    if (malachite_fill_pools(_cb, &_pipeline->_vertex, vtextures, vsamplers_bound, vheaders, vsamplers, &nvh, &nvs))
        return false;
    if (nvh) {
        struct nvgpu_heap vh;
        if (malachite_heap_commit(_cb, vheaders, nvh, vsamplers, nvs, &vh))
            return false;
        if (!d.heap.headers)
            d.heap = (struct nvgpu_heap){ vh.headers, vh.samplers, vh.nheaders, vh.nsamplers };
        d.vs_textured = 1;
        memcpy(d.vs_handles, vh.handles, sizeof(d.vs_handles));
    }
    _pools_pipeline = cacheable ? _pipeline : nil;
    if (cacheable) {
        memcpy(_pools_bound[0], _textures, sizeof(_textures));
        memcpy(_pools_bound[1], _samplers, sizeof(_samplers));
        memcpy(_pools_bound[2], _vtextures, sizeof(_vtextures));
        memcpy(_pools_bound[3], _vsamplers, sizeof(_vsamplers));
        _pools_heap = d.heap;
        _pools_nheaders = d.nheaders;
        _pools_nsamplers = d.nsamplers;
        _pools_vs_textured = d.vs_textured;
        memcpy(_pools_vs_handles, d.vs_handles, sizeof(d.vs_handles));
    }
    return true;
}

// What a pipeline or the encoder's state changed since the last draw, through the kernel's own code (core/drawbuild.c).
// Re-sending a group is harmless, as the kernel sends them all every draw.
- (uint32_t)fastState:(const struct nvgpu_draw &)d into:(uint32_t *)push
{
    struct nvgpu_draw &l = _last;
#define DIFFERS(f) (memcmp(&d.f, &l.f, sizeof(d.f)) != 0)
    bool shaders = d.vs != l.vs || d.ps != l.ps || d.vs_regs != l.vs_regs || d.ps_regs != l.ps_regs;
    bool depth = d.depth_test != l.depth_test || d.depth_write != l.depth_write || d.depth_func != l.depth_func ||
                 d.stencil_test != l.stencil_test || d.stencil_ref != l.stencil_ref || DIFFERS(stencil) ||
                 DIFFERS(stencil_read_mask) || DIFFERS(stencil_write_mask);
    bool cull = d.cull_face != l.cull_face || d.front_ccw != l.front_ccw;
    bool view = DIFFERS(viewport) || DIFFERS(scissor);
    bool bias = DIFFERS(depth_bias), bconst = DIFFERS(blend_const), attrs = DIFFERS(attributes);
    bool blend = d.write_mask != l.write_mask || DIFFERS(blend);
    for (unsigned j = 0; j < 7; j++)
        blend |= d.more[j].write_mask != l.more[j].write_mask || DIFFERS(more[j].blend);
    if (!(shaders || depth || cull || view || bias || bconst || attrs || blend))
        return 0;

    struct render_draw r = {};
    uint32_t n = 0;
    if (blend) {
        r.target = d.target;
        r.pitch = d.pitch;
        r.width = d.width;
        r.height = d.height;
        r.samples = d.samples;
        r.format = d.format;
        r.target_blocklinear = d.target_blocklinear;
        r.alpha_control = d.alpha_control;
        r.per_sample = d.per_sample;
        r.layers = d.layers;
        memcpy(r.layer_pitch, d.layer_pitch, sizeof(r.layer_pitch));
        r.write_mask = d.write_mask;
        memcpy(r.blend, d.blend, sizeof(r.blend));
        for (int j = 0; j < 7; j++) {
            r.more[j].addr = d.more[j].addr;
            r.more[j].pitch = d.more[j].pitch;
            r.more[j].format = d.more[j].format;
            r.more[j].blocklinear = d.more[j].blocklinear;
            r.more[j].write_mask = d.more[j].write_mask;
            memcpy(r.more[j].blend, d.more[j].blend, sizeof(d.more[j].blend));
            l.more[j].write_mask = d.more[j].write_mask;
            memcpy(l.more[j].blend, d.more[j].blend, sizeof(d.more[j].blend));
        }
        n += draw_target_words(push + n, &r);
        l.write_mask = d.write_mask;
        memcpy(l.blend, d.blend, sizeof(l.blend));
    }
    if (depth) {
        r.zt = d.zt;
        r.depth_test = d.depth_test;
        r.depth_write = d.depth_write;
        r.depth_func = d.depth_func;
        r.stencil_test = d.stencil_test;
        memcpy(r.stencil, d.stencil, sizeof(r.stencil));
        r.stencil_ref = d.stencil_ref;
        memcpy(r.stencil_read_mask, d.stencil_read_mask, sizeof(r.stencil_read_mask));
        memcpy(r.stencil_write_mask, d.stencil_write_mask, sizeof(r.stencil_write_mask));
        n += draw_depth_stencil_words(push + n, &r);
        l.depth_test = d.depth_test;
        l.depth_write = d.depth_write;
        l.depth_func = d.depth_func;
        l.stencil_test = d.stencil_test;
        l.stencil_ref = d.stencil_ref;
        memcpy(l.stencil, d.stencil, sizeof(l.stencil));
        memcpy(l.stencil_read_mask, d.stencil_read_mask, sizeof(l.stencil_read_mask));
        memcpy(l.stencil_write_mask, d.stencil_write_mask, sizeof(l.stencil_write_mask));
    }
    if (cull) {
        r.cull_face = d.cull_face;
        r.front_ccw = d.front_ccw;
        n += draw_cull_words(push + n, &r);
        l.cull_face = d.cull_face;
        l.front_ccw = d.front_ccw;
    }
    if (view) {
        memcpy(r.viewport, d.viewport, sizeof(r.viewport));
        memcpy(r.scissor, d.scissor, sizeof(r.scissor));
        r.window_coords = d.window_coords;
        r.shader_layer = d.shader_layer;
        r.depth_clamp = d.depth_clamp;
        n += draw_viewport_words(push + n, &r);
        memcpy(l.viewport, d.viewport, sizeof(l.viewport));
        memcpy(l.scissor, d.scissor, sizeof(l.scissor));
    }
    if (bias) {
        memcpy(r.depth_bias, d.depth_bias, sizeof(r.depth_bias));
        n += draw_bias_words(push + n, &r);
        memcpy(l.depth_bias, d.depth_bias, sizeof(l.depth_bias));
    }
    if (bconst) {
        memcpy(r.blend_const, d.blend_const, sizeof(r.blend_const));
        n += draw_blend_const_words(push + n, &r);
        memcpy(l.blend_const, d.blend_const, sizeof(l.blend_const));
    }
    if (shaders) {
        r.vs = d.vs;
        r.ps = d.ps;
        r.vs_regs = d.vs_regs;
        r.ps_regs = d.ps_regs;
        n += draw_shader_words(push + n, &r);
        // code this pass has run needs no flush, and the flush is most of what a change of pipeline costs the GPU
        bool seen = false;
        for (unsigned i = 0; i < 16; i++)
            seen |= _seen_code[i][0] == d.vs && _seen_code[i][1] == d.ps;
        if (!seen) {
            _seen_code[_seen_next][0] = d.vs;
            _seen_code[_seen_next][1] = d.ps;
            _seen_next = (_seen_next + 1) % 16;
            push[n++] = FAST_MTHD(FAST_INVALIDATE_SHADER, 1);
            push[n++] = 1 | 1 << 4 | 1 << 12;
        }
        l.vs = d.vs;
        l.ps = d.ps;
        l.vs_regs = d.vs_regs;
        l.ps_regs = d.ps_regs;
    }
    if (attrs) {
        memcpy(r.attributes, d.attributes, sizeof(r.attributes));
        n += draw_attribute_words(push + n, &r);
        memcpy(l.attributes, d.attributes, sizeof(l.attributes));
    }
#undef DIFFERS
    return n;
}

// A later draw of a pass written here as what differs from the one before, with no record for the kernel: a run of
// them is one record. The targets and what the kernel decides from them must match; anything rarer goes the kernel's
// way, which sends the lot.
- (bool)fastDraw:(const struct nvgpu_draw &)d
{
    MalachiteCommandBuffer *cb = (MalachiteCommandBuffer *)_cb;
    // anything sent since, such as a backdrop copy, or another encoder of a parallel pass, left other state behind
    if (!_fast || !_last_records || _last_records != cb->_records || cb->_state_of != (__bridge const void *)self)
        return false;
    const struct nvgpu_draw &l = _last;
    if (d.clear || d.clear_zs || d.serialize || d.indirect || l.indirect || d.tess || l.tess || d.visibility || l.visibility ||
        d.nviewports > 1 || d.vs_params_size > NVGPU_PARAM_BYTES || d.ps_params_size > NVGPU_PARAM_BYTES ||
        (d.heap.headers && !d.heap.nheaders) || !d.vs || !d.ps ||
        d.layers > 1 || !d.count || !d.instances || (d.nheaders && !d.heap.headers) || !d.index != !l.index ||
        d.index_format != l.index_format)
        return false;
    static const struct {
        size_t from, to;
    } same[] = {
        { 0, offsetof(struct nvgpu_draw, clear) },
        { offsetof(struct nvgpu_draw, zt), offsetof(struct nvgpu_draw, pass_first) },
        { offsetof(struct nvgpu_draw, zt_format), offsetof(struct nvgpu_draw, clear_zs) },
        { offsetof(struct nvgpu_draw, point_size), offsetof(struct nvgpu_draw, depth_bias) },
        { offsetof(struct nvgpu_draw, local_size), offsetof(struct nvgpu_draw, indirect) },
        { offsetof(struct nvgpu_draw, alpha_control), offsetof(struct nvgpu_draw, viewports) },
    };
    for (const auto &s : same)
        if (memcmp((const char *)&d + s.from, (const char *)&l + s.from, s.to - s.from))
            return false;
    for (unsigned j = 0; j < 7; j++)
        if (d.more[j].clear || d.more[j].addr != l.more[j].addr || d.more[j].pitch != l.more[j].pitch ||
            d.more[j].format != l.more[j].format || d.more[j].blocklinear != l.more[j].blocklinear)
            return false;
    for (unsigned s = 0; s < 16; s++)
        if (d.streams[s].stride != l.streams[s].stride || d.streams[s].divisor != l.streams[s].divisor)
            return false;
    if (!malachite_run_room(cb))
        return false;

    uint8_t *vb = cb->_bank_cpu + cb->_bank_used, *pb = vb + FAST_BANK_STEP;
    uint64_t vva = cb->_bank_va + cb->_bank_used, pva = vva + FAST_BANK_STEP;
    cb->_bank_used += FAST_BANK_STEP * 2;
    memset(vb, 0, FAST_BANK_USED);
    memset(pb, 0, FAST_BANK_USED);
    memcpy(vb + 0x160, d.vs_params, d.vs_params_size);
    int shader_base_vertex = d.index ? d.base_vertex : (int)d.first;
    memcpy(vb + 0x100, &shader_base_vertex, 4);
    memcpy(vb + 0x104, &d.base_instance, 4);
    memcpy(pb + 0x160, d.ps_params, d.ps_params_size);
    if (d.heap.headers) {
        memcpy(vb + 0x200, d.vs_textured ? d.vs_handles : d.heap.handles, sizeof(d.heap.handles));
        memcpy(pb + 0x200, d.heap.handles, sizeof(d.heap.handles));
    }

    uint32_t *w = cb->_run_cpu + cb->_run_used;
    const uint32_t *start = w;
    w += [self fastState:d into:w];
    *w++ = FAST_MTHD(FAST_CONSTANT_SELECTOR, 3);
    *w++ = FAST_BANK_BYTES;
    *w++ = (uint32_t)(vva >> 32);
    *w++ = (uint32_t)vva;
    *w++ = FAST_MTHD(FAST_CONSTANT_BIND, 1);
    *w++ = 1;
    *w++ = FAST_MTHD(FAST_CONSTANT_SELECTOR, 3);
    *w++ = FAST_BANK_BYTES;
    *w++ = (uint32_t)(pva >> 32);
    *w++ = (uint32_t)pva;
    *w++ = FAST_MTHD(FAST_CONSTANT_BIND + 4 * 32, 1);
    *w++ = 1;
    if (d.heap.headers && (d.heap.headers != l.heap.headers || d.heap.samplers != l.heap.samplers ||
                           d.heap.nheaders != l.heap.nheaders || d.heap.nsamplers != l.heap.nsamplers)) {
        *w++ = FAST_MTHD(FAST_HEADER_POOL, 3);
        *w++ = (uint32_t)(d.heap.headers >> 32);
        *w++ = (uint32_t)d.heap.headers;
        *w++ = d.heap.nheaders - 1;
        *w++ = FAST_MTHD(FAST_SAMPLER_POOL, 3);
        *w++ = (uint32_t)(d.heap.samplers >> 32);
        *w++ = (uint32_t)d.heap.samplers;
        *w++ = (d.heap.nsamplers ? d.heap.nsamplers : d.heap.nheaders) - 1;
    }
    if (cb->_bank_new) {
        cb->_bank_new = false;
        *w++ = FAST_MTHD(FAST_INVALIDATE_SHADER, 1);
        *w++ = 1 | 1 << 4 | 1 << 12;
    }
    // new heap entries were written for this draw, and need the caches to forget the slots. No texture data flush: a
    // fast draw follows nothing that wrote a texture, or it would have gone the kernel's way
    if (d.heap.headers && memcmp(&d.heap, &l.heap, sizeof(d.heap))) {
        *w++ = FAST_MTHD(FAST_INVALIDATE_HEADER, 1);
        *w++ = 0;
        *w++ = FAST_MTHD(FAST_INVALIDATE_SAMPLER, 1);
        *w++ = 0;
    }
    for (unsigned s = 0; s < 16; s++) {
        const auto &v = d.streams[s];
        if (v.addr == l.streams[s].addr && v.size == l.streams[s].size)
            continue;
        if (!v.addr) {
            *w++ = FAST_MTHD(FAST_STREAM_FORMAT + s * 16, 1);
            *w++ = 0;
            continue;
        }
        *w++ = FAST_MTHD(FAST_STREAM_FORMAT + s * 16, 4);
        *w++ = v.stride | 1 << 12;
        *w++ = (uint32_t)(v.addr >> 32);
        *w++ = (uint32_t)v.addr;
        *w++ = v.divisor;
        *w++ = FAST_MTHD(FAST_STREAM_SIZE + s * 8, 2);
        *w++ = 0;
        *w++ = v.size;
        *w++ = FAST_MTHD(FAST_STREAM_INSTANCE + s * 4, 1);
        *w++ = v.divisor != 0;
    }
    if (d.base_vertex != l.base_vertex || d.base_instance != l.base_instance) {
        *w++ = FAST_MTHD(FAST_BASE_VERTEX, 2);
        *w++ = (uint32_t)d.base_vertex;
        *w++ = d.base_instance;
        *w++ = FAST_MTHD(FAST_VERTEX_ID_BASE, 1);
        *w++ = (uint32_t)d.base_vertex;
    }
    uint32_t restarts = draw_restarts(d.primitive, d.restart, d.index != 0);
    if (restarts != draw_restarts(l.primitive, l.restart, l.index != 0)) {
        *w++ = FAST_MTHD(FAST_RESTART, 1);
        *w++ = restarts;
    }
    *w++ = FAST_MTHD(FAST_DRAW_CONTROL, 2);
    *w++ = d.primitive | 1 << 9;
    *w++ = d.instances;
    if (d.index) {
        if (d.index != l.index || d.index_size != l.index_size) {
            *w++ = FAST_MTHD(FAST_INDEX_ADDRESS, 2);
            *w++ = (uint32_t)(d.index >> 32);
            *w++ = (uint32_t)d.index;
            *w++ = FAST_MTHD(FAST_INDEX_SIZE, 2);
            *w++ = 0;
            *w++ = d.index_size;
        }
        *w++ = FAST_MTHD(FAST_DRAW_INDEXED, 2);
    } else {
        *w++ = FAST_MTHD(FAST_DRAW_ARRAYS, 2);
    }
    *w++ = d.first;
    *w++ = d.count;
    cb->_run_used += (uint32_t)(w - start);
    if (++cb->_run_draws == FAST_RUN_DRAWS)
        malachite_run_flush(cb);

    memcpy(_last.streams, d.streams, sizeof(d.streams));
    _last.heap = d.heap;
    _last.primitive = d.primitive;
    _last.index = d.index;
    _last.index_size = d.index_size;
    _last.base_vertex = d.base_vertex;
    _last.base_instance = d.base_instance;
    return true;
}

// Attachment n sampled as a texture by a draw that draws to it (CoreAnimation reads the backdrop so): the draw
// samples a copy of it as it was before the draw. Otherwise a pixel reads neighbours the same draw has already
// covered, and a quad's second triangle samples what its first drew, a streak along the diagonal. A refraction
// samples any distance away, so the pass's first such draw copies all of the attachment and later ones what the
// draws since have covered (their scissors).
- (void)attachment:(unsigned)n copy:(unsigned int *)header of:(const struct nvgpu_draw &)d
{
    struct nvgpu_target ct = malachite_target(d, n);
    uint32_t bpp = tex_format_info((enum tex_format)_formats[n]).bytes;
    uint32_t pitch = ct.blocklinear ? (ct.width * bpp + 63) & ~63u : ct.pitch;
    uint32_t rows = ct.blocklinear ? (ct.height + 7) & ~7u : ct.height;
    id<MTLDevice> device = ((id<MTLCommandBuffer>)_cb).device;
    uint64_t to = malachite_scratch(device, n, (NSUInteger)pitch * rows);
    struct tex_desc td = { to, ct.width, ct.height, pitch, (enum tex_format)_formats[n], (int)ct.blocklinear };
    tex_fill_header(header, &td);
    uint32_t x0 = 0, y0 = 0, x1 = ct.width, y1 = ct.height;
    if (!ct.blocklinear && _copied[n]) {
        const uint32_t *w = _written[n];
        x0 = w[0];
        y0 = w[1];
        x1 = MIN(w[2], ct.width);
        y1 = MIN(w[3], ct.height);
    }
    _copied[n] = true;
    memset(_written[n], 0, sizeof(_written[n]));
    if (x1 <= x0 || y1 <= y0)
        return;
    unsigned raw = 0;
    while (raw < sizeof(malachite_raw) / sizeof(malachite_raw[0]) && malachite_raw[raw].bytes != bpp)
        raw++;
    MalachiteRenderPipelineState *copy = d.samples <= 1 && raw < sizeof(malachite_raw) / sizeof(malachite_raw[0])
                                         ? malachite_copy_pipeline(device, raw)
                                         : nil;
    if (copy) {
        struct nvgpu_draw c = {};
        c.target = to;
        c.pitch = ct.blocklinear ? 0 : pitch;  // a block-linear target has none, and the scratch's 64-byte rows aren't 128
        c.width = ct.width;
        c.height = ct.height;
        c.target_blocklinear = ct.blocklinear;
        c.format = tex_format_info(malachite_raw[raw].format).target;
        c.vs = [(MTLIOAccelBuffer *)copy->_vcode gpuAddress];
        c.ps = [(MTLIOAccelBuffer *)copy->_fcode gpuAddress];
        c.vs_regs = copy->_vertex.reg_count;
        c.ps_regs = copy->_fragment.reg_count;
        struct tex_desc src = { ct.addr, ct.width, ct.height, ct.pitch, malachite_raw[raw].format, (int)ct.blocklinear };
        tex_fill_header(c.headers[0], &src);
        struct samp_desc def = malachite_default_sampler();
        tex_fill_sampler(c.samplers[0], &def);
        c.nheaders = c.nsamplers = 1;
        if (malachite_heap_commit(_cb, c.headers, c.nheaders, c.samplers, c.nsamplers, &c.heap))
            return;
        c.write_mask = copy->_write_mask[0];
        memcpy(c.blend, copy->_blend[0], sizeof(c.blend));
        c.primitive = 4;
        c.count = 3;
        c.instances = 1;
        c.viewport[2] = ct.width;
        c.viewport[3] = ct.height;
        c.viewport[5] = 1;
        c.scissor[0] = x0;
        c.scissor[1] = y0;
        c.scissor[2] = x1 - x0;
        c.scissor[3] = y1 - y0;
        malachite_record(_cb, MALACHITE_CMD_DRAW, &c, sizeof(c));
        return;
    }
    struct nvgpu_copy2d c = { ct.addr, to, pitch, pitch, ct.width * bpp, ct.height, ct.blocklinear, ct.blocklinear,
                              ct.blocklinear ? rows : 0, ct.blocklinear ? rows : 0 };
    if (x0 || y0 || x1 != ct.width || y1 != ct.height) {
        uint64_t at = (uint64_t)y0 * pitch + x0 * bpp;
        c = (struct nvgpu_copy2d){ ct.addr + at, to + at, pitch, pitch, (x1 - x0) * bpp, y1 - y0 };
    }
    malachite_record(_cb, MALACHITE_CMD_COPY_2D, &c, sizeof(c));
}

// the trace: every colour attachment as this draw left it, copied out of where it's drawn and saved once the GPU
// is done, raw after a one-line header
- (void)snapshot:(const struct nvgpu_draw &)d suffix:(const char *)suffix
{
    static int draws;
    int n = __sync_fetch_and_add(&draws, 1);
    for (unsigned i = 0; i < 8; i++) {
        if (_formats[i] < 0)
            continue;
        struct nvgpu_target ct = malachite_target(d, i);
        uint32_t bpp = tex_format_info((enum tex_format)_formats[i]).bytes, bytes = ct.width * bpp;
        uint32_t h = ct.height, w = ct.width;
        // only the rect copied when there is one: a frame of whole-target copies runs the machine out of memory
        const uint32_t *r = malachite_snap_rect;
        bool cut = r[2] && ct.width >= r[0] + r[2] && ct.height >= r[1] + r[3];
        uint32_t keep = cut ? r[2] * bpp : bytes, rows = cut ? r[3] : h, pitch = (keep + 255) & ~255u;
        id<MTLBuffer> snap = [((id<MTLCommandBuffer>)_cb).device newBufferWithLength:(NSUInteger)pitch * rows
                                                                            options:MTLResourceStorageModeShared];
        struct nvgpu_copy2d c = { ct.addr, [(MTLIOAccelBuffer *)snap gpuAddress],
                                  ct.blocklinear ? (bytes + 63) & ~63u : ct.pitch, pitch, keep, rows,
                                  ct.blocklinear, 0, ct.blocklinear ? (h + 7) & ~7u : 0, 0 };
        if (cut && ct.blocklinear)
            c.src_origin = r[0] * bpp | r[1] << 16;
        else if (cut)
            c.src += (uint64_t)r[1] * ct.pitch + r[0] * bpp;
        malachite_record(_cb, MALACHITE_CMD_COPY_2D, &c, sizeof(c));
        MTLPixelFormat f = _attached[i].pixelFormat;
        NSString *label = _pipeline->_label;
        NSString *where = [NSString stringWithFormat:@"%#llx %u,%u,%u,%u", (unsigned long long)ct.addr, d.scissor[0],
                                                     d.scissor[1], d.scissor[2], d.scissor[3]];
        [(id<MTLCommandBuffer>)_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
            char path[128];
            snprintf(path, sizeof(path), "/tmp/malachite-draw-%d-%04d-%u%s.raw", getpid(), n, i, suffix);
            FILE *o = fopen(path, "wb");
            if (!o)
                return;
            fprintf(o, "%u %u %u %lu %s %s\n", cut ? keep / bpp : w, rows, keep, (unsigned long)f, label.UTF8String,
                    where.UTF8String);
            for (uint32_t y = 0; y < rows; y++)
                fwrite((const uint8_t *)snap.contents + (size_t)y * pitch, 1, keep, o);
            fclose(o);
        }];
    }
}

// The recorder. A recording is a directory of raw files and a manifest.txt of key=value lines naming them, written
// last; tools/mtl/mtlplay.m describes the keys. A buffer is recorded from its offset to its end (16 MB at most),
// and a process stops recording once it has written 2 GB (a dispatch's buffers are whole, 256 MB each at most).
enum { MALACHITE_REC_BUFFER = 16 << 20 };

static bool malachite_rec_reserve(uint64_t bytes)
{
    static uint64_t written;
    if (__sync_add_and_fetch(&written, bytes) <= 2ull << 30)
        return true;
    malachite_rec_full = true;
    return false;
}

// lines rows of bytes, pitch apart from src or block-linear there (one-GOB blocks, rows high) when rows isn't 0,
// copied on the GPU where the command buffer has got to and written packed to path once it's done
static bool malachite_rec_copy(id cb, NSString *path, uint64_t src, uint32_t pitch, uint32_t rows, uint32_t bytes,
                           uint32_t lines)
{
    uint32_t to = (bytes + 255) & ~255u;
    if (!bytes || !lines || !malachite_rec_reserve((uint64_t)bytes * lines))
        return false;
    id<MTLBuffer> buf = [((id<MTLCommandBuffer>)cb).device newBufferWithLength:(NSUInteger)to * lines
                                                                       options:MTLResourceStorageModeShared];
    if (!buf)
        return false;
    struct nvgpu_copy2d c = { src, [(MTLIOAccelBuffer *)buf gpuAddress], pitch, to, bytes, lines, rows != 0, 0, rows, 0 };
    malachite_record(cb, MALACHITE_CMD_COPY_2D, &c, sizeof(c));
    [(id<MTLCommandBuffer>)cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        FILE *o = fopen(path.fileSystemRepresentation, "wb");
        if (!o)
            return;
        for (uint32_t y = 0; y < lines; y++)
            fwrite((const uint8_t *)buf.contents + (size_t)y * to, 1, bytes, o);
        fclose(o);
    }];
    return true;
}

// size bytes of memory at src, likewise
static bool malachite_rec_bytes(id cb, NSString *path, uint64_t src, uint32_t size)
{
    if (!size || !malachite_rec_reserve(size))
        return false;
    id<MTLBuffer> buf = [((id<MTLCommandBuffer>)cb).device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buf)
        return false;
    struct malachite_copy_buf2buf p = {};
    p.srcAddr = src;
    p.dstAddr = [(MTLIOAccelBuffer *)buf gpuAddress];
    p.size = size;
    malachite_record(cb, MALACHITE_CMD_COPY_BUF2BUF, &p, sizeof(p));
    [(id<MTLCommandBuffer>)cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        [[NSData dataWithBytesNoCopy:buf.contents length:size freeWhenDone:NO] writeToFile:path atomically:NO];
    }];
    return true;
}

// a stage's function, with the AIR it was compiled from and its linked functions' ([air, [linked], [names]])
static void malachite_rec_function(id cb, NSString *dir, NSMutableString *m, NSString *stage, id<MTLFunction> fn,
                               NSArray *air)
{
    if (!fn)
        return;
    [m appendFormat:@"%@.name=%@\n", stage, fn.name];
    // Metal hands the driver a specialised function's AIR with the values in, so these are only what it declares
    for (MTLFunctionConstant *c in fn.functionConstantsDictionary.allValues)
        [m appendFormat:@"%@.constant.%lu=%@,%lu,%d\n", stage, (unsigned long)c.index, c.name, (unsigned long)c.type,
                        c.required];
    if (!air)
        return;
    NSArray<NSData *> *linked = air[1];
    NSArray<NSString *> *names = air[2];
    NSMutableDictionary<NSString *, NSData *> *files = [NSMutableDictionary dictionary];
    files[[stage stringByAppendingString:@".air"]] = air[0];
    [m appendFormat:@"%@.air=%@.air\n%@.linked=%lu\n", stage, stage, stage, (unsigned long)linked.count];
    for (NSUInteger i = 0; i < linked.count; i++) {
        NSString *file = [NSString stringWithFormat:@"%@.linked%lu.air", stage, (unsigned long)i];
        files[file] = linked[i];
        [m appendFormat:@"%@.linked.%lu.name=%@\n%@.linked.%lu.air=%@\n", stage, (unsigned long)i, names[i], stage,
                        (unsigned long)i, file];
    }
    [(id<MTLCommandBuffer>)cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        for (NSString *file in files)
            [files[file] writeToFile:[dir stringByAppendingPathComponent:file] atomically:NO];
    }];
}

// whether a stage's shader reads the [[buffer(n)]], [[texture(n)]] or [[sampler(n)]] slot
static bool malachite_reads(const struct nvk_header *h, uint8_t kind, unsigned slot)
{
    for (uint32_t i = 0; i < h->nargs; i++)
        if (h->args[i].kind == kind && h->args[i].index == slot)
            return true;
    return false;
}

// the format whose layout the GPU keeps f's pixels in (see malachite_format), which a replay has to use
static MTLPixelFormat malachite_stored_format(MTLPixelFormat f)
{
    switch (f) {
    case MTLPixelFormatBGRA10_XR: case MTLPixelFormatBGRA10_XR_sRGB: return MTLPixelFormatRGBA16Float;
    case MTLPixelFormatBGR10_XR: case MTLPixelFormatBGR10_XR_sRGB: return MTLPixelFormatBGR10A2Unorm;
    case MTLPixelFormatR8Unorm_sRGB: return MTLPixelFormatR8Unorm;
    case MTLPixelFormatRG8Unorm_sRGB: return MTLPixelFormatRG8Unorm;
    default: return f;
    }
}

// colour attachment i, all of it, where the pass draws it (its shadow, if it has one)
- (bool)record:(NSString *)path attachment:(unsigned)i of:(const struct nvgpu_draw &)d
{
    struct nvgpu_target ct = malachite_target(d, i);
    uint32_t bytes = ct.width * tex_format_info((enum tex_format)_formats[i]).bytes;
    return malachite_rec_copy(_cb, path, ct.addr, ct.blocklinear ? (bytes + 63) & ~63u : ct.pitch,
                          ct.blocklinear ? (ct.height + 7) & ~7u : 0, bytes, ct.height);
}

- (bool)record:(NSString *)path depthOf:(const struct nvgpu_draw &)d
{
    uint32_t bpp = _zattached->_zt_bpp;
    return malachite_rec_copy(_cb, path, d.zt, d.zt_width * bpp, (d.zt_height + 7) & ~7u, d.width * bpp, d.height);
}

- (void)record:(NSString *)dir manifest:(NSMutableString *)m texture:(MalachiteTexture *)t key:(NSString *)key
{
    if (![t isKindOfClass:[MalachiteTexture class]] || !t->_backing) {
        [m appendFormat:@"%@.unsupported=%@\n", key, NSStringFromClass([t class])];
        return;
    }
    id<MTLTexture> mt = (id<MTLTexture>)t;
    NSUInteger slices = t->_image ? t->_layers : mt.arrayLength;
    [m appendFormat:@"%@.type=%lu\n%@.format=%lu\n%@.app_format=%lu\n%@.width=%lu\n%@.height=%lu\n%@.depth=%lu\n"
                     "%@.levels=%lu\n%@.array_length=%lu\n%@.slices=%lu\n%@.samples=%lu\n%@.first_level=%u\n",
                    key, (unsigned long)mt.textureType, key, (unsigned long)malachite_stored_format(mt.pixelFormat), key,
                    (unsigned long)mt.pixelFormat, key, (unsigned long)mt.width, key, (unsigned long)mt.height, key,
                    (unsigned long)mt.depth, key, (unsigned long)mt.mipmapLevelCount, key,
                    (unsigned long)mt.arrayLength, key, (unsigned long)slices, key, (unsigned long)mt.sampleCount, key,
                    t->_view_min];
    if (t->_swizzled)
        [m appendFormat:@"%@.swizzle=%u,%u,%u,%u\n", key, t->_swizzle[0], t->_swizzle[1], t->_swizzle[2],
                        t->_swizzle[3]];
    // an attachment's own memory and level, which the draw reads as it was before it (see attachment:copy:)
    uint64_t at = [(MTLIOAccelBuffer *)t->_backing gpuAddress] + t->_offset;
    for (unsigned n = 0; n < 8; n++)
        if (_attached[n] && _attached[n]->_backing &&
            at == [(MTLIOAccelBuffer *)_attached[n]->_backing gpuAddress] + _attached[n]->_offset &&
            (!t->_image || (t->_levels == 1 && _attached_level[n] == t->_view_min))) {
            [m appendFormat:@"%@.attachment=%u\n", key, n];
            return;
        }
    if (t->_samples) {
        [m appendFormat:@"%@.unsupported=multisampled\n", key];
        return;
    }
    for (NSUInteger level = 0; level < mt.mipmapLevelCount; level++) {
        struct malachite_sub s;
        for (NSUInteger slice = 0; slice < slices; slice++)
            for (NSUInteger z = 0; malachite_sub(t, level, slice, z, &s); z++) {
                NSString *file = [NSString stringWithFormat:@"%@-l%lu-s%lu-z%lu.bin", key, (unsigned long)level,
                                                            (unsigned long)slice, (unsigned long)z];
                if (malachite_rec_copy(_cb, [dir stringByAppendingPathComponent:file],
                                   [(MTLIOAccelBuffer *)t->_backing gpuAddress] + s.offset, s.pitch, s.rows,
                                   s.width * s.bpp, s.height))
                    [m appendFormat:@"%@.level.%lu.slice.%lu.z.%lu=%@\n", key, (unsigned long)level,
                                    (unsigned long)slice, (unsigned long)z, file];
            }
        if (malachite_sub(t, level, 0, 0, &s))
            [m appendFormat:@"%@.level.%lu.bytes_per_row=%u\n%@.level.%lu.rows=%u\n", key, (unsigned long)level,
                            s.width * s.bpp, key, (unsigned long)level, s.height];
    }
}

// everything the draw reads as it is where the command buffer has got to, into dir; the attachments as the draw
// leaves them and the manifest follow it (record:manifest:after:)
- (void)record:(NSString *)dir
      manifest:(NSMutableString *)m
          draw:(const struct nvgpu_draw &)d
          type:(MTLPrimitiveType)type
         first:(NSUInteger)first
         count:(NSUInteger)count
     instances:(NSUInteger)instances
    baseVertex:(NSInteger)baseVertex
  baseInstance:(NSUInteger)baseInstance
         index:(id)index
        offset:(NSUInteger)offset
     indexType:(MTLIndexType)indexType
{
    MTLRenderPipelineDescriptor *pd = objc_getAssociatedObject(_pipeline, &malachite_desc_key);
    NSDictionary *air = objc_getAssociatedObject(_pipeline, &malachite_air_key);
    mkdir(dir.fileSystemRepresentation, 0755);
    [m appendFormat:@"version=1\nprocess=%s\npid=%d\nlabel=%@\n", getprogname(), getpid(), _pipeline->_label];
    malachite_rec_function(_cb, dir, m, @"vertex", pd.vertexFunction, air[@"vertex"]);
    malachite_rec_function(_cb, dir, m, @"fragment", pd.fragmentFunction, air[@"fragment"]);

    [m appendFormat:@"pipeline.samples=%lu\npipeline.alpha_to_coverage=%d\npipeline.alpha_to_one=%d\n"
                     "pipeline.rasterization=%d\npipeline.topology_class=%lu\npipeline.depth_format=%lu\n"
                     "pipeline.stencil_format=%lu\n",
                    (unsigned long)pd.rasterSampleCount, pd.alphaToCoverageEnabled, pd.alphaToOneEnabled,
                    pd.rasterizationEnabled, (unsigned long)pd.inputPrimitiveTopology,
                    (unsigned long)pd.depthAttachmentPixelFormat, (unsigned long)pd.stencilAttachmentPixelFormat];
    for (unsigned i = 0; i < 8; i++) {
        MTLRenderPipelineColorAttachmentDescriptor *a = pd.colorAttachments[i];
        if (a.pixelFormat == MTLPixelFormatInvalid)
            continue;
        [m appendFormat:@"pipeline.colour.%u.format=%lu\npipeline.colour.%u.blend=%d\npipeline.colour.%u.rgb_op=%lu\n"
                         "pipeline.colour.%u.src_rgb=%lu\npipeline.colour.%u.dst_rgb=%lu\n"
                         "pipeline.colour.%u.alpha_op=%lu\npipeline.colour.%u.src_alpha=%lu\n"
                         "pipeline.colour.%u.dst_alpha=%lu\npipeline.colour.%u.write_mask=%lu\n",
                        i, (unsigned long)a.pixelFormat, i, a.blendingEnabled, i, (unsigned long)a.rgbBlendOperation,
                        i, (unsigned long)a.sourceRGBBlendFactor, i, (unsigned long)a.destinationRGBBlendFactor, i,
                        (unsigned long)a.alphaBlendOperation, i, (unsigned long)a.sourceAlphaBlendFactor, i,
                        (unsigned long)a.destinationAlphaBlendFactor, i, (unsigned long)a.writeMask];
    }
    MTLVertexDescriptor *vd = pd.vertexDescriptor;
    for (unsigned i = 0; vd && i < 31; i++) {
        MTLVertexAttributeDescriptor *a = vd.attributes[i];
        if (a.format != MTLVertexFormatInvalid)
            [m appendFormat:@"vertex_descriptor.attribute.%u.format=%lu\nvertex_descriptor.attribute.%u.offset=%lu\n"
                             "vertex_descriptor.attribute.%u.buffer=%lu\n",
                            i, (unsigned long)a.format, i, (unsigned long)a.offset, i, (unsigned long)a.bufferIndex];
        MTLVertexBufferLayoutDescriptor *l = vd.layouts[i];
        if (l.stride || l.stepFunction != MTLVertexStepFunctionPerVertex)
            [m appendFormat:@"vertex_descriptor.layout.%u.stride=%lu\nvertex_descriptor.layout.%u.step_function=%lu\n"
                             "vertex_descriptor.layout.%u.step_rate=%lu\n",
                            i, (unsigned long)l.stride, i, (unsigned long)l.stepFunction, i,
                            (unsigned long)l.stepRate];
    }

    [m appendFormat:@"pass.width=%u\npass.height=%u\n", d.width, d.height];
    for (unsigned i = 0; i < 8; i++) {
        if (_formats[i] < 0)
            continue;
        MTLRenderPassColorAttachmentDescriptor *ca = _rpd.colorAttachments[i];
        MTLPixelFormat f = _attached[i].pixelFormat;
        [m appendFormat:@"attachment.%u.format=%lu\nattachment.%u.app_format=%lu\nattachment.%u.bpp=%u\n"
                         "attachment.%u.load=%lu\nattachment.%u.store=%lu\nattachment.%u.level=%lu\n"
                         "attachment.%u.slice=%lu\n",
                        i, (unsigned long)malachite_stored_format(f), i, (unsigned long)f, i,
                        tex_format_info((enum tex_format)_formats[i]).bytes, i, (unsigned long)ca.loadAction, i,
                        (unsigned long)ca.storeAction, i, (unsigned long)ca.level, i, (unsigned long)ca.slice];
        // the pass's clear, which this draw does first: the targets before it are as the pass found them
        MTLClearColor c = ca.clearColor;
        if (i ? d.more[i - 1].clear : d.clear)
            [m appendFormat:@"attachment.%u.clear=%.9g,%.9g,%.9g,%.9g\n", i, c.red, c.green, c.blue, c.alpha];
        NSString *file = [NSString stringWithFormat:@"attachment%u-before.bin", i];
        if ([self record:[dir stringByAppendingPathComponent:file] attachment:i of:d])
            [m appendFormat:@"attachment.%u.before=%@\n", i, file];
    }
    if (_zattached) {
        MTLRenderPassDepthAttachmentDescriptor *da = _rpd.depthAttachment;
        MTLRenderPassStencilAttachmentDescriptor *sa = _rpd.stencilAttachment;
        [m appendFormat:@"depth.format=%lu\ndepth.bpp=%u\ndepth.load=%lu\ndepth.store=%lu\nstencil.load=%lu\n"
                         "stencil.store=%lu\n",
                        (unsigned long)_zattached.pixelFormat, _zattached->_zt_bpp, (unsigned long)da.loadAction,
                        (unsigned long)da.storeAction, (unsigned long)sa.loadAction, (unsigned long)sa.storeAction];
        if (d.clear_zs & 1)
            [m appendFormat:@"depth.clear=%.9g\n", d.clear_depth];
        if (d.clear_zs & 2)
            [m appendFormat:@"stencil.clear=%u\n", d.clear_stencil];
        if ([self record:[dir stringByAppendingPathComponent:@"depth-before.bin"] depthOf:d])
            [m appendString:@"depth.before=depth-before.bin\n"];
    }
    MTLDepthStencilDescriptor *dsd = _ds ? objc_getAssociatedObject(_ds, &malachite_desc_key) : nil;
    if (dsd) {
        [m appendFormat:@"depth_stencil.depth_compare=%lu\ndepth_stencil.depth_write=%d\n",
                        (unsigned long)dsd.depthCompareFunction, dsd.depthWriteEnabled];
        MTLStencilDescriptor *faces[2] = { dsd.frontFaceStencil, dsd.backFaceStencil };
        const char *names[2] = { "front", "back" };
        for (int k = 0; k < 2; k++) {
            MTLStencilDescriptor *f = faces[k];
            const char *n = names[k];
            if (f)
                [m appendFormat:@"depth_stencil.%s.compare=%lu\ndepth_stencil.%s.fail=%lu\n"
                                 "depth_stencil.%s.depth_fail=%lu\ndepth_stencil.%s.pass=%lu\n"
                                 "depth_stencil.%s.read_mask=%u\ndepth_stencil.%s.write_mask=%u\n",
                                n, (unsigned long)f.stencilCompareFunction, n, (unsigned long)f.stencilFailureOperation,
                                n, (unsigned long)f.depthFailureOperation, n,
                                (unsigned long)f.depthStencilPassOperation, n, f.readMask, n, f.writeMask];
        }
    }
    MTLCullMode cull = d.cull_face == 0x404 ? MTLCullModeFront : d.cull_face == 0x405 ? MTLCullModeBack : MTLCullModeNone;
    [m appendFormat:@"state.viewport=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\nstate.scissor=%u,%u,%u,%u\nstate.cull_mode=%lu\n"
                     "state.winding=%u\nstate.fill_mode=%u\nstate.depth_clip_mode=%u\n"
                     "state.blend_colour=%.9g,%.9g,%.9g,%.9g\nstate.stencil_ref=%u\nstate.depth_bias=%.9g,%.9g,%.9g\n",
                    d.viewport[0], d.viewport[1], d.viewport[2], d.viewport[3], d.viewport[4], d.viewport[5],
                    d.scissor[0], d.scissor[1], d.scissor[2], d.scissor[3], (unsigned long)cull, d.front_ccw,
                    d.fill_lines, d.depth_clamp, d.blend_const[0], d.blend_const[1], d.blend_const[2],
                    d.blend_const[3], d.stencil_ref, d.depth_bias[0], d.depth_bias[1], d.depth_bias[2]];

    // what the shaders read: the vertex buffers the descriptor streams from too
    for (unsigned b = 0; b < 31; b++) {
        bool stream = false;
        for (unsigned s = 0; s < 16; s++)
            stream |= (_pipeline->_strides[s] || _pipeline->_divisors[s]) && _pipeline->_stream_buffer[s] == b;
        const struct {
            const char *stage;
            uint64_t va;
            uint32_t size;
            __unsafe_unretained id buffer;
            bool read;
        } bound[2] = {
            { "vertex", _vbuffers[b], _vsizes[b], _vbound[b],
              stream || malachite_reads(&_pipeline->_vertex, NVK_ARG_BUFFER, b) },
            { "fragment", _fbuffers[b], _fsizes[b], _fbound[b], malachite_reads(&_pipeline->_fragment, NVK_ARG_BUFFER, b) },
        };
        for (const auto &v : bound) {
            if (!v.va || !v.read)
                continue;
            NSString *file = [NSString stringWithFormat:@"%s-buffer%u.bin", v.stage, b];
            uint32_t size = MIN(v.size, (uint32_t)MALACHITE_REC_BUFFER);
            if (malachite_rec_bytes(_cb, [dir stringByAppendingPathComponent:file], v.va, size))
                [m appendFormat:@"%s.buffer.%u=%@\n%s.buffer.%u.offset=%llu\n%s.buffer.%u.length=%u\n", v.stage, b,
                                file, v.stage, b, v.va - [(MTLIOAccelBuffer *)v.buffer gpuAddress], v.stage, b, v.size];
        }
    }
    const struct {
        const char *stage;
        __unsafe_unretained const id *textures, *samplers;
        const struct nvk_header *h;
    } stages[2] = { { "vertex", _vtextures, _vsamplers, &_pipeline->_vertex },
                    { "fragment", _textures, _samplers, &_pipeline->_fragment } };
    for (const auto &s : stages)
        for (unsigned slot = 0; slot < NVGPU_MAX_TEXTURES; slot++) {
            if (s.textures[slot] && malachite_reads(s.h, NVK_ARG_TEXTURE, slot))
                [self record:dir manifest:m texture:s.textures[slot]
                         key:[NSString stringWithFormat:@"%s.texture.%u", s.stage, slot]];
            MTLSamplerDescriptor *sd = s.samplers[slot] ? objc_getAssociatedObject(s.samplers[slot], &malachite_desc_key) : nil;
            if (!sd || !malachite_reads(s.h, NVK_ARG_SAMPLER, slot))
                continue;
            NSString *k = [NSString stringWithFormat:@"%s.sampler.%u", s.stage, slot];
            [m appendFormat:@"%@.min_filter=%lu\n%@.mag_filter=%lu\n%@.mip_filter=%lu\n%@.s_address=%lu\n"
                             "%@.t_address=%lu\n%@.r_address=%lu\n%@.normalized=%d\n%@.lod_min=%.9g\n%@.lod_max=%.9g\n"
                             "%@.max_anisotropy=%lu\n%@.compare=%lu\n%@.border_colour=%lu\n",
                            k, (unsigned long)sd.minFilter, k, (unsigned long)sd.magFilter, k,
                            (unsigned long)sd.mipFilter, k, (unsigned long)sd.sAddressMode, k,
                            (unsigned long)sd.tAddressMode, k, (unsigned long)sd.rAddressMode, k,
                            sd.normalizedCoordinates, k, sd.lodMinClamp, k, sd.lodMaxClamp, k,
                            (unsigned long)sd.maxAnisotropy, k, (unsigned long)sd.compareFunction, k,
                            (unsigned long)sd.borderColor];
        }

    [m appendFormat:@"draw.primitive=%lu\ndraw.vertex_start=%lu\ndraw.vertex_count=%lu\ndraw.instance_count=%lu\n"
                     "draw.base_vertex=%ld\ndraw.base_instance=%lu\n",
                    (unsigned long)type, (unsigned long)first, (unsigned long)count, (unsigned long)instances,
                    (long)baseVertex, (unsigned long)baseInstance];
    if (index) {
        [m appendFormat:@"draw.index_type=%lu\n", (unsigned long)indexType];
        if (malachite_rec_bytes(_cb, [dir stringByAppendingPathComponent:@"index.bin"], d.index,
                            MIN(d.index_size, (uint32_t)MALACHITE_REC_BUFFER)))
            [m appendString:@"draw.index=index.bin\n"];
    }
    // MTLDrawPrimitivesIndirectArguments or MTLDrawIndexedPrimitivesIndirectArguments
    if (_indirect && malachite_rec_bytes(_cb, [dir stringByAppendingPathComponent:@"indirect.bin"], _indirect,
                                     index ? 20 : 16))
        [m appendString:@"draw.indirect=indirect.bin\n"];
}

- (void)record:(NSString *)dir manifest:(NSMutableString *)m after:(const struct nvgpu_draw &)d
{
    for (unsigned i = 0; i < 8; i++) {
        NSString *file = [NSString stringWithFormat:@"attachment%u-after.bin", i];
        if (_formats[i] >= 0 && [self record:[dir stringByAppendingPathComponent:file] attachment:i of:d])
            [m appendFormat:@"attachment.%u.after=%@\n", i, file];
    }
    if (_zattached && [self record:[dir stringByAppendingPathComponent:@"depth-after.bin"] depthOf:d])
        [m appendString:@"depth.after=depth-after.bin\n"];
    if (malachite_rec_full)
        [m appendString:@"incomplete=1\n"];
    NSString *path = [dir stringByAppendingPathComponent:@"manifest.txt"];
    [(id<MTLCommandBuffer>)_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        [m writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }];
}

- (void)drawPrimitives:(MTLPrimitiveType)type vertexStart:(NSUInteger)start vertexCount:(NSUInteger)count
{
    [self draw:type first:start count:count instances:1 baseVertex:0 baseInstance:0 index:nil offset:0
          type:MTLIndexTypeUInt16];
}

- (void)drawPrimitives:(MTLPrimitiveType)type
           vertexStart:(NSUInteger)start
           vertexCount:(NSUInteger)count
         instanceCount:(NSUInteger)instances
{
    [self draw:type first:start count:count instances:instances baseVertex:0 baseInstance:0 index:nil offset:0
          type:MTLIndexTypeUInt16];
}

- (void)drawPrimitives:(MTLPrimitiveType)type
           vertexStart:(NSUInteger)start
           vertexCount:(NSUInteger)count
         instanceCount:(NSUInteger)instances
          baseInstance:(NSUInteger)baseInstance
{
    [self draw:type first:start count:count instances:instances baseVertex:0 baseInstance:baseInstance index:nil
         offset:0 type:MTLIndexTypeUInt16];
}

- (void)drawIndexedPrimitives:(MTLPrimitiveType)type
                   indexCount:(NSUInteger)count
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id)index
            indexBufferOffset:(NSUInteger)offset
{
    [self draw:type first:0 count:count instances:1 baseVertex:0 baseInstance:0 index:index offset:offset
          type:indexType];
}

- (void)drawIndexedPrimitives:(MTLPrimitiveType)type
                   indexCount:(NSUInteger)count
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id)index
            indexBufferOffset:(NSUInteger)offset
                instanceCount:(NSUInteger)instances
{
    [self draw:type first:0 count:count instances:instances baseVertex:0 baseInstance:0 index:index
        offset:offset type:indexType];
}

- (void)drawIndexedPrimitives:(MTLPrimitiveType)type
                   indexCount:(NSUInteger)count
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id)index
            indexBufferOffset:(NSUInteger)offset
                instanceCount:(NSUInteger)instances
                   baseVertex:(NSInteger)baseVertex
                 baseInstance:(NSUInteger)baseInstance
{
    [self draw:type first:0 count:count instances:instances baseVertex:baseVertex baseInstance:baseInstance
         index:index offset:offset type:indexType];
}

// what the pass's first draw, or its clears when it has none, starts from: the shadows' contents, and a depth
// texture's apart from its stencil texture
- (void)loadAttachments
{
    for (unsigned i = 0; i < 8; i++)
        if (_load[i].bytes)
            malachite_record(_cb, MALACHITE_CMD_COPY_2D, &_load[i], sizeof(_load[i]));
    if (_zload.bytes)
        malachite_record(_cb, MALACHITE_CMD_COPY_2D, &_zload, sizeof(_zload));
}

// a pass with a clear load action and no draws still clears
- (void)endEncoding
{
    malachite_run_flush((MalachiteCommandBuffer *)_cb);
    bool clears = _pass.clear || _pass.clear_zs;
    for (const struct nvgpu_target &t : _pass.more)
        clears |= t.addr && t.clear;
    bool cleared = _pass.width && clears && !_drawn;
    if (cleared) {
        [self loadAttachments];
        struct nvgpu_draw d = _pass;
        d.pass_first = 1;
        malachite_record(_cb, MALACHITE_CMD_DRAW, &d, sizeof(d));
    }
    for (unsigned i = 0; i < 8 && (_drawn || cleared); i++)
        if (_store[i].bytes)
            malachite_record(_cb, MALACHITE_CMD_COPY_2D, &_store[i], sizeof(_store[i]));
    if (_zstore.bytes && (_drawn || cleared))
        malachite_record(_cb, MALACHITE_CMD_COPY_2D, &_zstore, sizeof(_zstore));
    for (unsigned i = 0; i < 8; i++)
        if (_resolve[i] && (_drawn || cleared))
            malachite_resolve(_cb, _attached[i], _resolve[i]);
    for (unsigned i = 0; i < 2; i++)
        if (_zresolve[i].to && _zattached && (_drawn || cleared))
            malachite_zresolve(_cb, _zattached, &_zresolve[i]);
    // the trace: each attachment as the pass left it, once the GPU is done, raw after a one-line header (depth and
    // stencil as attachment 9: block-linear, one-GOB blocks, rows padded to 64 bytes)
    static int passes;
    int n = passes;
    if (malachite_trace)
        __sync_fetch_and_add(&passes, 1);
    if (malachite_snap && !malachite_snap_only && !malachite_snap_rect[2] && _zattached) {
        MalachiteTexture *z = _zattached;
        uint32_t w = _pass.width, h = _pass.height, pitch = z->_pitch, rows = (h + 7) & ~7u, fmt = z->_zt_format;
        [(id<MTLCommandBuffer>)_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
            char path[96];
            snprintf(path, sizeof(path), "/tmp/malachite-pass-%d-%03d-9.raw", getpid(), n);
            FILE *o = fopen(path, "wb");
            if (!o)
                return;
            fprintf(o, "%u %u %u %u\n", w, h, pitch, 1000 + fmt);
            const uint8_t *src = (const uint8_t *)((id<MTLBuffer>)z->_backing).contents;  // nil for private memory
            if (src)
                fwrite(src + z->_offset, 1, (size_t)pitch * rows, o);
            fclose(o);
        }];
    }
    for (unsigned i = 0; malachite_snap && !malachite_snap_only && !malachite_snap_rect[2] && i < 8; i++) {
        MalachiteTexture *t = _attached[i];
        if (!t || t->_image)
            continue;
        uint32_t w = _pass.width, h = _pass.height, pitch = t->_pitch;
        MTLPixelFormat f = t.pixelFormat;
        [(id<MTLCommandBuffer>)_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
            char path[96];
            snprintf(path, sizeof(path), "/tmp/malachite-pass-%d-%03d-%u.raw", getpid(), n, i);
            FILE *o = fopen(path, "wb");
            if (!o)
                return;
            fprintf(o, "%u %u %u %lu\n", w, h, pitch, (unsigned long)f);
            const uint8_t *src = (const uint8_t *)((id<MTLBuffer>)t->_backing).contents;  // nil for private memory
            if (src)
                fwrite(src + t->_offset, 1, (size_t)pitch * h, o);
            fclose(o);
        }];
    }
    [super endEncoding];
}
@end

// An MTLVertexFormat as SET_VERTEX_ATTRIBUTE_A's component widths (26:21), numerical type (29:27) and R/B swap
// (31) fields; 0 for one we don't handle. as_float: read by a shader as floats, which integers convert to by value
// (Apple's GL renderer passes GL's attributes that aren't normalised so).
static uint32_t malachite_vertex_format(MTLVertexFormat f, bool as_float)
{
    enum { SNORM = 1, UNORM = 2, SINT = 3, UINT = 4, USCALED = 5, SSCALED = 6, FLOAT = 7 };
    enum { R32x4 = 0x01, R32x3 = 0x02, R16x4 = 0x03, R32x2 = 0x04, R16x3 = 0x05, R8x4 = 0x0a, R16x2 = 0x0f,
           R32 = 0x12, R8x3 = 0x13, R8x2 = 0x18, R16 = 0x1b, R8 = 0x1d, A2B10G10R10 = 0x30, B10G11R11 = 0x31 };
    uint32_t w, t;
    switch (f) {
    case MTLVertexFormatFloat: w = R32, t = FLOAT; break;
    case MTLVertexFormatFloat2: w = R32x2, t = FLOAT; break;
    case MTLVertexFormatFloat3: w = R32x3, t = FLOAT; break;
    case MTLVertexFormatFloat4: w = R32x4, t = FLOAT; break;
    case MTLVertexFormatHalf: w = R16, t = FLOAT; break;
    case MTLVertexFormatHalf2: w = R16x2, t = FLOAT; break;
    case MTLVertexFormatHalf3: w = R16x3, t = FLOAT; break;
    case MTLVertexFormatHalf4: w = R16x4, t = FLOAT; break;
    case MTLVertexFormatInt: w = R32, t = SINT; break;
    case MTLVertexFormatInt2: w = R32x2, t = SINT; break;
    case MTLVertexFormatInt3: w = R32x3, t = SINT; break;
    case MTLVertexFormatInt4: w = R32x4, t = SINT; break;
    case MTLVertexFormatUInt: w = R32, t = UINT; break;
    case MTLVertexFormatUInt2: w = R32x2, t = UINT; break;
    case MTLVertexFormatUInt3: w = R32x3, t = UINT; break;
    case MTLVertexFormatUInt4: w = R32x4, t = UINT; break;
    case MTLVertexFormatUChar: w = R8, t = UINT; break;
    case MTLVertexFormatUChar2: w = R8x2, t = UINT; break;
    case MTLVertexFormatUChar3: w = R8x3, t = UINT; break;
    case MTLVertexFormatUChar4: w = R8x4, t = UINT; break;
    case MTLVertexFormatChar: w = R8, t = SINT; break;
    case MTLVertexFormatChar2: w = R8x2, t = SINT; break;
    case MTLVertexFormatChar3: w = R8x3, t = SINT; break;
    case MTLVertexFormatChar4: w = R8x4, t = SINT; break;
    case MTLVertexFormatUCharNormalized: w = R8, t = UNORM; break;
    case MTLVertexFormatUChar2Normalized: w = R8x2, t = UNORM; break;
    case MTLVertexFormatUChar3Normalized: w = R8x3, t = UNORM; break;
    case MTLVertexFormatUChar4Normalized: w = R8x4, t = UNORM; break;
    case MTLVertexFormatUChar4Normalized_BGRA: return R8x4 << 21 | UNORM << 27 | 1u << 31;
    case MTLVertexFormatCharNormalized: w = R8, t = SNORM; break;
    case MTLVertexFormatChar2Normalized: w = R8x2, t = SNORM; break;
    case MTLVertexFormatChar3Normalized: w = R8x3, t = SNORM; break;
    case MTLVertexFormatChar4Normalized: w = R8x4, t = SNORM; break;
    case MTLVertexFormatUShort: w = R16, t = UINT; break;
    case MTLVertexFormatUShort2: w = R16x2, t = UINT; break;
    case MTLVertexFormatUShort3: w = R16x3, t = UINT; break;
    case MTLVertexFormatUShort4: w = R16x4, t = UINT; break;
    case MTLVertexFormatShort: w = R16, t = SINT; break;
    case MTLVertexFormatShort2: w = R16x2, t = SINT; break;
    case MTLVertexFormatShort3: w = R16x3, t = SINT; break;
    case MTLVertexFormatShort4: w = R16x4, t = SINT; break;
    case MTLVertexFormatUShortNormalized: w = R16, t = UNORM; break;
    case MTLVertexFormatUShort2Normalized: w = R16x2, t = UNORM; break;
    case MTLVertexFormatUShort3Normalized: w = R16x3, t = UNORM; break;
    case MTLVertexFormatUShort4Normalized: w = R16x4, t = UNORM; break;
    case MTLVertexFormatShortNormalized: w = R16, t = SNORM; break;
    case MTLVertexFormatShort2Normalized: w = R16x2, t = SNORM; break;
    case MTLVertexFormatShort3Normalized: w = R16x3, t = SNORM; break;
    case MTLVertexFormatShort4Normalized: w = R16x4, t = SNORM; break;
    case MTLVertexFormatUInt1010102Normalized: w = A2B10G10R10, t = UNORM; break;
    case MTLVertexFormatInt1010102Normalized: w = A2B10G10R10, t = SNORM; break;
    case MTLVertexFormatFloatRG11B10: w = B10G11R11, t = FLOAT; break;
    default: return 0;  // RGB9E5: the fetch unit has no such format
    }
    if (as_float && (t == SINT || t == UINT))
        t = t == SINT ? SSCALED : USCALED;
    return w << 21 | t << 27;
}

// A parallel render pass as its sub-encoders one after another, in the order they were made (the order Metal runs
// them in): the first loads as the pass says, the rest load what the one before stored, and all but the last store.
@interface MalachiteParallelRenderCommandEncoder : NSObject <MTLParallelRenderCommandEncoder>
@property (copy, atomic) NSString *label;
@end

static MTLStoreAction malachite_stored(MTLStoreAction action)
{
    return malachite_resolves(action) ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore;
}

@implementation MalachiteParallelRenderCommandEncoder {
    __unsafe_unretained id<MTLCommandBuffer> _cb;
    MTLRenderPassDescriptor *_pass;
    NSMutableArray<id<MTLRenderCommandEncoder>> *_subs;
}

- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)cb descriptor:(MTLRenderPassDescriptor *)descriptor
{
    self = [super init];
    if (self) {
        _cb = cb;
        _pass = [descriptor copy];
        _subs = [NSMutableArray array];
    }
    return self;
}

- (id<MTLDevice>)device
{
    return _cb.device;
}

- (id<MTLRenderCommandEncoder>)renderCommandEncoder
{
    MTLRenderPassDescriptor *d = [_pass copy];
    if (_subs.count) {
        for (NSUInteger i = 0; i < 8; i++)
            d.colorAttachments[i].loadAction = MTLLoadActionLoad;
        d.depthAttachment.loadAction = MTLLoadActionLoad;
        d.stencilAttachment.loadAction = MTLLoadActionLoad;
    }
    // each resolves what it has so far, the last one the whole pass
    for (NSUInteger i = 0; i < 8; i++)
        d.colorAttachments[i].storeAction = malachite_stored(_pass.colorAttachments[i].storeAction);
    d.depthAttachment.storeAction = malachite_stored(_pass.depthAttachment.storeAction);
    d.stencilAttachment.storeAction = malachite_stored(_pass.stencilAttachment.storeAction);
    id<MTLRenderCommandEncoder> e = [_cb renderCommandEncoderWithDescriptor:d];
    if (e)
        [_subs addObject:e];
    return e;
}

// for the sub-encoders made from now on
- (void)setColorStoreAction:(MTLStoreAction)action atIndex:(NSUInteger)index
{
    if (index < 8)
        _pass.colorAttachments[index].storeAction = action;
}

- (void)setDepthStoreAction:(MTLStoreAction)action
{
    _pass.depthAttachment.storeAction = action;
}

- (void)setStencilStoreAction:(MTLStoreAction)action
{
    _pass.stencilAttachment.storeAction = action;
}

- (void)setColorStoreActionOptions:(MTLStoreActionOptions)options atIndex:(NSUInteger)index
{
}

- (void)setDepthStoreActionOptions:(MTLStoreActionOptions)options
{
}

- (void)setStencilStoreActionOptions:(MTLStoreActionOptions)options
{
}

- (void)endEncoding
{
}

- (void)insertDebugSignpost:(NSString *)string
{
}

- (void)pushDebugGroup:(NSString *)string
{
}

- (void)popDebugGroup
{
}

- (void)barrierAfterQueueStages:(MTLStages)after beforeStages:(MTLStages)before
{
}
@end


// What the CPU changed in Managed buffers goes to their VRAM copies in a command buffer committed just before cb on
// its queue: WindowServer's vertices are Managed, so this is on the way of every frame and doesn't wait. That can't
// work for a cb enqueued already, which runs first, or on a queue short of command buffers, where asking for one more
// blocks if the app holds them all; those go through a queue of ours, waited for. The caller holds a lock across
// this, so the copy is in its queue before another thread's commit, and a commit on another queue lets it finish: it
// has to see the data, and a copy arriving later would overwrite what the GPU has written there since.
//
// The wait is for the copy alone, which the last copy of each batch reports by writing its number into a word here. A
// command buffer's completion can't say that: it comes in order, behind whatever draws were queued before it.
static uint32_t malachite_upload_sent;
static id<MTLBuffer> malachite_upload_ack;

static void malachite_upload_wait(void)
{
    volatile uint32_t *done = (volatile uint32_t *)malachite_upload_ack.contents;
    // bounded: a command buffer ahead of the copy can be waiting for an event this commit is to signal
    for (int i = 0; i < 5000 && (int32_t)(*done - malachite_upload_sent) < 0; i++)
        usleep(20);
}

// Whether the queue has a command buffer to spare now. Asking for one waits on the queue's semaphore with no timeout,
// and WindowServer can hold every one of its own until the one being committed is through: its main thread hung there
// for good. Taken and given straight back, so another thread can still take it first, which only delays this one.
static bool malachite_queue_has_room(id<MTLCommandQueue> queue)
{
    static Ivar ivar;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        ivar = class_getInstanceVariable(objc_getClass("_MTLCommandQueue"), "_commandBufferSemaphore");
    });
    dispatch_semaphore_t sem = ivar ? object_getIvar(queue, ivar) : nil;
    if (!sem || dispatch_semaphore_wait(sem, DISPATCH_TIME_NOW))
        return false;
    dispatch_semaphore_signal(sem);
    return true;
}

static void malachite_upload_managed(MalachiteCommandBuffer *cb)
{
    static id<MTLCommandQueue> last;
    id<MTLCommandQueue> queue = ((id<MTLCommandBuffer>)cb).commandQueue;
    if (last && last != queue) {
        malachite_upload_wait();
        last = nil;
    }
    NSArray<MalachiteBuffer *> *work;
    NSMutableArray<NSNumber *> *ranges = [NSMutableArray array];
    @synchronized([MalachiteBuffer class]) {
        if (!malachite_pending.count)
            return;
        work = [malachite_pending copy];
        [malachite_pending removeAllObjects];
        for (MalachiteBuffer *b in work) {
            [ranges addObject:@(b->_dirty_lo)];
            [ranges addObject:@(b->_dirty_hi)];
            b->_dirty_lo = b->_dirty_hi = 0;
            b->_pending = false;
        }
    }
    static id<MTLCommandQueue> own;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        own = [((id<MTLCommandBuffer>)cb).device newCommandQueue];
        malachite_upload_ack = [((id<MTLCommandBuffer>)cb).device newBufferWithLength:64
                                                                              options:MTLResourceStorageModeShared];
    });
    bool direct = ((id<MTLCommandBuffer>)cb).status < MTLCommandBufferStatusEnqueued &&
                  [(MTLIOAccelCommandQueue *)queue maxCommandBufferCount] >= 32 && malachite_queue_has_room(queue);
    id<MTLCommandBuffer> u = [direct ? queue : own commandBuffer];
    for (NSUInteger i = 0; i < work.count; i++) {
        MalachiteBuffer *b = work[i];
        NSUInteger lo = ranges[i * 2].unsignedIntegerValue, hi = MIN(ranges[i * 2 + 1].unsignedIntegerValue, b.length);
        if (lo >= hi)
            continue;
        struct malachite_copy_buf2buf p = {};
        p.srcAddr = [b systemAddress] + lo;
        p.dstAddr = [b->_mirror gpuAddress] + lo;
        p.size = hi - lo;
        malachite_keep(u, b);
        malachite_keep(u, b->_mirror);
        malachite_record(u, MALACHITE_CMD_UPLOAD, &p, sizeof(p));
    }
    ((uint32_t *)malachite_upload_ack.contents)[1] = ++malachite_upload_sent;
    struct malachite_copy_buf2buf ack = {};
    ack.srcAddr = malachite_upload_ack.gpuAddress + 4;
    ack.dstAddr = malachite_upload_ack.gpuAddress;
    ack.size = 4;
    malachite_keep(u, malachite_upload_ack);
    malachite_record(u, MALACHITE_CMD_UPLOAD, &ack, sizeof(ack));
    [u commit];
    if (direct)
        last = queue;
    else
        malachite_upload_wait();
}

// the copies back of what the command buffer's OpenCL kernels wrote (MalachiteBuffer _writeback), after them
static void malachite_write_back(MalachiteCommandBuffer *cb)
{
    for (MalachiteBuffer *b in cb->_writeback) {
        struct malachite_copy_buf2buf p = {};
        p.srcAddr = [b->_mirror gpuAddress];
        p.dstAddr = [b systemAddress];
        p.size = b.length;
        malachite_keep(cb, b);
        malachite_keep(cb, b->_mirror);
        malachite_record(cb, MALACHITE_CMD_COPY_BUF2BUF, &p, sizeof(p));
    }
    cb->_writeback = nil;
}

@implementation MalachiteCommandBuffer
// the trace: a command buffer whose completion comes late, from commit to its completed handlers
- (void)commit
{
    malachite_write_back(self);
    if (malachite_trace) {
        uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        [self addCompletedHandler:^(id<MTLCommandBuffer>) {
            uint64_t ms = (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start) / 1000000;
            if (ms >= 10)
                os_log(OS_LOG_DEFAULT, "Malachite trace: command buffer completed %llu ms after commit", ms);
        }];
    }
    malachite_run_flush(self);
    malachite_end_segment(self);
    @synchronized([MalachiteCommandBuffer class]) {
        malachite_upload_managed(self);
    }
    [super commit];
}

// The base writes its own commands for these and ends the storage's segment, which left ours to end a segment that
// was gone (Chromium's GPU process crashed at commit): the draws before go first, in a segment closed here.
- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value
{
    malachite_run_flush(self);
    malachite_end_segment(self);
    [super encodeSignalEvent:event value:value];
}

- (void)encodeWaitForEvent:(id<MTLEvent>)event value:(uint64_t)value
{
    malachite_run_flush(self);
    malachite_end_segment(self);
    [super encodeWaitForEvent:event value:value];
}

// WindowServer's every frame. The base enqueues before it commits, which would send the upload to the queue that's
// waited for: until the GPU has finished what other processes queued, a game's next frames among them.
- (void)commitAndWaitUntilSubmitted
{
    malachite_write_back(self);
    malachite_run_flush(self);
    malachite_end_segment(self);
    @synchronized([MalachiteCommandBuffer class]) {
        malachite_upload_managed(self);
    }
    [super commitAndWaitUntilSubmitted];
    // WindowServer hands a capture to its client once this returns, and the client's CPU lock doesn't wait for our
    // GPU (only for Apple's): it read the surface's previous frame, or one half drawn
    if (_writes_capture)
        [self waitUntilCompleted];
}

// A drawable goes to WindowServer once its draws are done, not once they're scheduled. WindowServer reads it with no
// wait of its own, so one sent early would show half drawn, and the frame after it would replace it on screen.
- (void)presentDrawable:(id<MTLDrawable>)drawable
{
    if (opengl_renderer) {
        [super presentDrawable:drawable];
        return;
    }
    [self addCompletedHandler:^(id<MTLCommandBuffer>) {
        [drawable present];
    }];
}

- (void)presentDrawable:(id<MTLDrawable>)drawable atTime:(CFTimeInterval)time
{
    if (opengl_renderer) {
        [super presentDrawable:drawable atTime:time];
        return;
    }
    [self addCompletedHandler:^(id<MTLCommandBuffer>) {
        [drawable presentAtTime:time];
    }];
}

- (void)presentDrawable:(id<MTLDrawable>)drawable afterMinimumDuration:(CFTimeInterval)duration
{
    if (opengl_renderer) {
        [super presentDrawable:drawable afterMinimumDuration:duration];
        return;
    }
    [self addCompletedHandler:^(id<MTLCommandBuffer>) {
        [drawable presentAfterMinimumDuration:duration];
    }];
}

- (id)blitCommandEncoder
{
    return [[MalachiteBlitCommandEncoder alloc] initWithCommandBuffer:self];
}

- (id)computeCommandEncoder
{
    return [[MalachiteComputeCommandEncoder alloc] initWithCommandBuffer:self];
}

- (id)computeCommandEncoderWithDispatchType:(MTLDispatchType)type
{
    return [self computeCommandEncoder];
}

// The base's would make its own encoders, whose segments in the command storage ours don't expect: Chromium's GPU
// process crashed at its first commit. Counter sample attachments are dropped: the device samples no counters.
- (id)computeCommandEncoderWithDescriptor:(MTLComputePassDescriptor *)descriptor
{
    return [self computeCommandEncoder];
}

- (id)blitCommandEncoderWithDescriptor:(MTLBlitPassDescriptor *)descriptor
{
    return [self blitCommandEncoder];
}

- (id)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    if (malachite_snap_gate) {
        bool open = malachite_trusted_file("/var/tmp/malachite-snap", false);
        malachite_rec_shut = !open;
        if (malachite_snap_gated)
            malachite_snap = open;
    }
    return [[MalachiteRenderCommandEncoder alloc] initWithCommandBuffer:self descriptor:descriptor];
}

- (id)parallelRenderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)descriptor
{
    return [[MalachiteParallelRenderCommandEncoder alloc] initWithCommandBuffer:self descriptor:descriptor];
}

// sparse textures aren't supported, so there's no resource state to encode
- (id)resourceStateCommandEncoder
{
    return nil;
}
@end

@interface MalachiteCommandQueue : MTLIOAccelCommandQueue <MTLCommandQueue>
@end

@implementation MalachiteCommandQueue
// the base queue doesn't vend a command buffer
- (id)commandBuffer
{
    return [[MalachiteCommandBuffer alloc] initWithQueue:self retainedReferences:true];
}
- (id)commandBufferWithUnretainedReferences
{
    return [[MalachiteCommandBuffer alloc] initWithQueue:self retainedReferences:false];
}
@end

// MTLAddDevice asserts the class conformsToProtocol: both MTLDevice (public) and MTLDeviceSPI (private). The real
// superclass implements the methods at runtime; the formal conformance is what the assert checks (the runtime matches
// protocols by name). MTLDeviceSPI isn't in the public SDK, so declare it inheriting MTLDevice.
@protocol MTLDeviceSPI <MTLDevice>
@end

@interface MalachiteDevice : MTLIOAccelDevice <MTLDeviceSPI>
- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor heap:(MalachiteHeap *)heap offset:(NSUInteger)offset;
@end

// a resource's range of a heap: whole pages. An image needs its GOBs on 512 bytes, as its layers and levels are
// already, and nothing in a heap's memory takes a PTE kind (see newImage:layout:placed:).
static MTLSizeAndAlign malachite_range_in_heap(uint64_t bytes)
{
    NSUInteger page = vm_page_size;
    return (MTLSizeAndAlign){ (MAX(bytes, 1ull) + page - 1) & ~(NSUInteger)(page - 1), page };
}

@implementation MalachiteHeapBuffer
- (id)device
{
    return _heap->_device;
}

- (NSUInteger)length
{
    return _length;
}

- (NSUInteger)allocatedSize
{
    return _size;
}

- (void *)contents
{
    uint8_t *base = (uint8_t *)_heap->_block.contents;
    return base ? base + _offset : NULL;
}

- (uint64_t)gpuAddress
{
    return [(MTLIOAccelBuffer *)_heap->_block gpuAddress] + _offset;
}

- (id<MTLHeap>)heap
{
    return _heap;
}

- (NSUInteger)heapOffset
{
    return _offset;
}

- (MTLStorageMode)storageMode
{
    return (MTLStorageMode)((_options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift);
}

- (MTLCPUCacheMode)cpuCacheMode
{
    return (MTLCPUCacheMode)(_options & MTLResourceCPUCacheModeMask);
}

- (MTLHazardTrackingMode)hazardTrackingMode
{
    return _heap->_desc.hazardTrackingMode;
}

- (MTLResourceOptions)resourceOptions
{
    return _options;
}

- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
    return MTLPurgeableStateNonVolatile;
}

- (void)makeAliasable
{
    if (_aliasable)
        return;
    _aliasable = true;
    [_heap giveBack:_offset size:_size];
}

- (BOOL)isAliasable
{
    return _aliasable;
}

- (void)didModifyRange:(NSRange)range
{
    [_heap->_block didModifyRange:NSMakeRange(_offset + range.location, range.length)];
}

- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor offset:(NSUInteger)offset bytesPerRow:(NSUInteger)bytesPerRow
{
    return [_heap->_block newTextureWithDescriptor:descriptor offset:_offset + offset bytesPerRow:bytesPerRow];
}

- (void)addDebugMarker:(NSString *)marker range:(NSRange)range
{
}

- (void)removeAllDebugMarkers
{
}

// anything else the block answers for, as the IOAccel resource underneath this
- (id)forwardingTargetForSelector:(SEL)selector
{
    return _heap->_block;
}

- (void)dealloc
{
    if (!_aliasable)
        [_heap giveBack:_offset size:_size];
}
@end

@implementation MalachiteHeap
- (id)device
{
    return _device;
}

- (MTLStorageMode)storageMode
{
    return _desc.storageMode;
}

- (MTLCPUCacheMode)cpuCacheMode
{
    return _desc.cpuCacheMode;
}

- (MTLHazardTrackingMode)hazardTrackingMode
{
    return _desc.hazardTrackingMode;
}

- (MTLResourceOptions)resourceOptions
{
    return _desc.resourceOptions;
}

- (MTLHeapType)type
{
    return _desc.type;
}

- (NSUInteger)protectionOptions
{
    return 0;
}

- (NSUInteger)size
{
    return _desc.size;
}

- (NSUInteger)usedSize
{
    os_unfair_lock_lock(&_lock);
    NSUInteger used = _desc.size - _free;
    os_unfair_lock_unlock(&_lock);
    return used;
}

- (NSUInteger)currentAllocatedSize
{
    return _desc.size;
}

- (NSUInteger)allocatedSize
{
    return _desc.size;
}

- (NSUInteger)maxAvailableSizeWithAlignment:(NSUInteger)alignment
{
    NSUInteger best = 0, align = MAX(alignment, vm_page_size);
    os_unfair_lock_lock(&_lock);
    for (auto &h : _holes) {
        NSUInteger at = (h.first + align - 1) & ~(align - 1);
        if (at < h.first + h.second)
            best = MAX(best, h.first + h.second - at);
    }
    os_unfair_lock_unlock(&_lock);
    return best;
}

- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
    return MTLPurgeableStateNonVolatile;
}

- (void)giveBack:(NSUInteger)offset size:(NSUInteger)size
{
    if (_desc.type == MTLHeapTypePlacement)  // nothing was taken
        return;
    os_unfair_lock_lock(&_lock);
    _free += size;
    auto next = _holes.lower_bound(offset);
    if (next != _holes.end() && offset + size == next->first) {
        size += next->second;
        next = _holes.erase(next);
    }
    auto prev = next != _holes.begin() ? std::prev(next) : _holes.end();
    if (prev != _holes.end() && prev->first + prev->second == offset)
        prev->second += size;
    else
        _holes[offset] = size;
    os_unfair_lock_unlock(&_lock);
}

// the free range [offset, offset + size) taken out of whichever hole holds it; false if no hole does
- (bool)take:(NSUInteger)offset size:(NSUInteger)size
{
    auto h = _holes.upper_bound(offset);
    if (h == _holes.begin())
        return false;
    h = std::prev(h);
    NSUInteger start = h->first, len = h->second;
    if (offset + size > start + len)
        return false;
    _holes.erase(h);
    if (offset > start)
        _holes[start] = offset - start;
    if (offset + size < start + len)
        _holes[offset + size] = start + len - offset - size;
    _free -= size;
    return true;
}

// A resource's range of the block: in a placement heap at offset, which must be aligned as the range is; in an
// automatic one at offset if that's free, else in the first hole it fits aligned. nil when it doesn't fit.
- (MalachiteHeapBuffer *)place:(NSUInteger)length
                          size:(MTLSizeAndAlign)range
                       options:(MTLResourceOptions)options
                            at:(NSUInteger)offset
{
    if (_desc.type == MTLHeapTypePlacement) {
        if (offset == NSNotFound || offset % range.align || offset > _desc.size || range.size > _desc.size - offset)
            return nil;
    } else {
        os_unfair_lock_lock(&_lock);
        if (offset == NSNotFound)
            for (auto &h : _holes) {
                NSUInteger at = (h.first + range.align - 1) & ~(range.align - 1);
                if (at + range.size <= h.first + h.second) {
                    offset = at;
                    break;
                }
            }
        bool found = offset != NSNotFound && [self take:offset size:range.size];
        os_unfair_lock_unlock(&_lock);
        if (!found)
            return nil;
    }
    MalachiteHeapBuffer *b = [MalachiteHeapBuffer new];
    b->_heap = self;
    b->_offset = offset;
    b->_length = length;
    b->_size = range.size;
    b->_options = options;
    return b;
}

- (id)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
    return [self place:length size:malachite_range_in_heap(length) options:options at:NSNotFound];
}

- (id)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options offset:(NSUInteger)offset
{
    return [self place:length size:malachite_range_in_heap(length) options:options at:offset];
}

- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
    return [(MalachiteDevice *)_device newTextureWithDescriptor:descriptor heap:self offset:NSNotFound];
}

- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor offset:(NSUInteger)offset
{
    return [(MalachiteDevice *)_device newTextureWithDescriptor:descriptor heap:self offset:offset];
}
@end

// Reflection, as much as OpenGL's renderer reads: each function's arguments (it finds a GL program's uniform buffer,
// textures and samplers by name), and the GL converter's return data. Both are kept on the function when it's
// compiled.
static const char malachite_return_key = 0, malachite_arguments_key = 0;
static const char malachite_cl_program_key = 0;  // an OpenCL kernel's arguments as its program counts them

@interface MalachiteArgument : NSObject
@property (copy) NSString *name;
@property MTLArgumentType type;
@property NSUInteger index;
@property MTLTextureType textureType;
@property MTLDataType textureDataType;
@property BOOL isDepthTexture;
@property (strong) id dataTypeDescription;  // an OpenCL buffer's: whether it's a __constant one
@property NSUInteger bufferDataSize;        // an OpenCL argument passed by value: its bytes
@property BOOL inactive;
@property MTLBindingAccess access;  // OpenCL syncs a Managed buffer back after a kernel that may write it
@end

@interface MalachitePointerType : NSObject
@property BOOL isConstantBuffer;
@end

@implementation MalachitePointerType
@end

@implementation MalachiteArgument
- (BOOL)isActive
{
    return !self.inactive;
}

- (NSUInteger)arrayLength
{
    return 1;
}

- (MTLDataType)bufferDataType
{
    return MTLDataTypeNone;
}

- (id)bufferStructType
{
    return nil;
}
@end

// what the shader cache keeps of a compile, all callers read of its airc_kernel; an entry has the SASS after it
struct malachite_compiled {
    struct nvk_header h;
    struct airc_arg_name arg_names[NVK_MAX_ARGS];
    struct airc_varyings outputs;
};

// the arguments of a compiled function that resources bind to
// the compiler's hidden arguments for an OpenCL kernel's global work offset and image formats, which no one binds
// by their index
static bool malachite_hidden_arg(const struct nvk_arg *a, const char *name)
{
    return a->kind == NVK_ARG_BUFFER && (a->index == AIRC_CL_OFFSETS || a->index == AIRC_CL_FORMATS) && !name[0];
}

static NSArray<MalachiteArgument *> *malachite_arguments(const struct malachite_compiled *k)
{
    NSMutableArray *args = [NSMutableArray array];
    for (uint32_t i = 0; i < k->h.nargs; i++) {
        unsigned kind = k->h.args[i].kind;
        // threadgroup memory too: OpenCL takes an argument's index from its place in this list, and a __local one
        // left out moved every one after it
        if ((kind != NVK_ARG_BUFFER && kind != NVK_ARG_TEXTURE && kind != NVK_ARG_SAMPLER && kind != NVK_ARG_THREADGROUP) ||
            malachite_hidden_arg(&k->h.args[i], k->arg_names[i].name))
            continue;
        MalachiteArgument *a = [MalachiteArgument new];
        a.name = @(k->arg_names[i].name);
        a.index = k->h.args[i].index;
        a.access = (MTLBindingAccess)k->h.args[i].access;
        a.type = kind == NVK_ARG_BUFFER    ? MTLArgumentTypeBuffer
                 : kind == NVK_ARG_TEXTURE ? MTLArgumentTypeTexture
                 : kind == NVK_ARG_SAMPLER ? MTLArgumentTypeSampler
                                           : MTLArgumentTypeThreadgroupMemory;
        if (kind == NVK_ARG_TEXTURE) {  // from the type's name: depth2d<float, sample>, texture2d_array<half>...
            const char *t = k->arg_names[i].type;
            a.isDepthTexture = !strncmp(t, "depth", 5);
            a.textureType = strstr(t, "cube_array") ? MTLTextureTypeCubeArray
                            : strstr(t, "cube")     ? MTLTextureTypeCube
                            : strstr(t, "3d")       ? MTLTextureType3D
                            : strstr(t, "2d_ms")    ? MTLTextureType2DMultisample
                            : strstr(t, "2d_array") ? MTLTextureType2DArray
                            : strstr(t, "1d_array") ? MTLTextureType1DArray
                            : strstr(t, "1d")       ? MTLTextureType1D
                                                    : MTLTextureType2D;
            a.textureDataType = strstr(t, "<half")   ? MTLDataTypeHalf
                                : strstr(t, "<uint") ? MTLDataTypeUInt
                                : strstr(t, "<int")  ? MTLDataTypeInt
                                                     : MTLDataTypeFloat;
        }
        [args addObject:a];
    }
    return args;
}

// An OpenCL kernel's pipeline reflection: the renderer's own buffers 0-2 first (GLDKernelRec::initWithConfig asks
// whether 1 and 2 are active, and our kernels read neither), then its arguments
static NSArray<MalachiteArgument *> *malachite_cl_pipeline_arguments(const struct malachite_compiled *k)
{
    NSMutableArray *args = [NSMutableArray array];
    for (NSUInteger i = 0; i < MALACHITE_CL_FIRST; i++) {
        MalachiteArgument *a = [MalachiteArgument new];
        a.name = [NSString stringWithFormat:@"__implicit%lu", (unsigned long)i];
        a.index = i;
        a.type = MTLArgumentTypeBuffer;
        a.inactive = YES;
        [args addObject:a];
    }
    [args addObjectsFromArray:malachite_arguments(k)];
    return args;
}

// An OpenCL kernel's arguments as building its program counts them (GLDComputeProgramRec::buildComputeProgram,
// from the function's reflectionWithOptions:): each in its own place, a buffer a pointer, one passed by value
// argument type 0x16 with its bytes in bufferDataSize
static NSArray<MalachiteArgument *> *malachite_cl_program_arguments(const struct malachite_compiled *k)
{
    NSMutableArray *args = [NSMutableArray array];
    for (uint32_t i = 0; i < k->h.nargs; i++) {
        const struct nvk_arg *a = &k->h.args[i];
        if ((a->kind != NVK_ARG_BUFFER && a->kind != NVK_ARG_THREADGROUP && a->kind != NVK_ARG_TEXTURE &&
             a->kind != NVK_ARG_SAMPLER) ||
            malachite_hidden_arg(a, k->arg_names[i].name))
            continue;
        MalachiteArgument *arg = [MalachiteArgument new];
        arg.name = @(k->arg_names[i].name);
        arg.index = a->index;
        arg.access = (MTLBindingAccess)a->access;
        arg.type = a->kind == NVK_ARG_THREADGROUP ? MTLArgumentTypeThreadgroupMemory
                   : a->kind == NVK_ARG_TEXTURE   ? MTLArgumentTypeTexture
                   : a->kind == NVK_ARG_SAMPLER   ? MTLArgumentTypeSampler
                   : a->size                      ? (MTLArgumentType)0x16
                                                  : MTLArgumentTypeBuffer;
        arg.bufferDataSize = a->size;
        arg.dataTypeDescription = [MalachitePointerType new];
        [args addObject:arg];
    }
    return args;
}

@interface MalachiteRenderReflection : NSObject
@property (strong) NSArray *vertexArguments, *fragmentArguments;
@property (strong) NSData *vertexPluginReturnData, *fragmentPluginReturnData;
@end

@implementation MalachiteRenderReflection
- (NSUInteger)usageFlags
{
    return 0;
}

- (NSDictionary *)performanceStatistics
{
    return nil;
}
@end

@interface MalachiteComputeReflection : NSObject
@property (strong) NSArray *arguments;
@property (strong) NSData *pluginReturnData;
@end

@implementation MalachiteComputeReflection
- (NSDictionary *)performanceStatistics
{
    return nil;
}
@end

// Building an OpenCL program counts each kernel's arguments from the function's reflection (reflectionWithOptions:,
// in GLDComputeProgramRec::buildComputeProgram), which Metal's compiler service makes from the AIR without the
// vendor's word on them: zero, and every clSetKernelArg failed. A kernel of ours answers with its arguments, compiled
// first if it hasn't been.
static IMP malachite_reflection_orig;

static id malachite_function_reflection(id self, SEL cmd, NSUInteger options)
{
    id<MTLFunction> function = self;
    if ([function.name hasPrefix:@"__entry_"] && [function.device isKindOfClass:[MalachiteDevice class]]) {
        if (!objc_getAssociatedObject(function, &malachite_cl_program_key))
            [(MalachiteDevice *)function.device newComputePipelineStateWithFunction:function error:nil];
        NSArray *args = objc_getAssociatedObject(function, &malachite_cl_program_key);
        if (args) {
            MalachiteComputeReflection *r = [MalachiteComputeReflection new];
            r.arguments = args;
            return r;
        }
    }
    return ((id(*)(id, SEL, NSUInteger))malachite_reflection_orig)(self, cmd, options);
}

static void malachite_hook_function_reflection(void)
{
    Method m = class_getInstanceMethod(objc_getClass("_MTLFunction"), sel_registerName("reflectionWithOptions:"));
    if (m && !malachite_reflection_orig)
        malachite_reflection_orig = method_setImplementation(m, (IMP)malachite_function_reflection);
}

static std::vector<LLVMValueRef> malachite_md(LLVMValueRef node)
{
    std::vector<LLVMValueRef> ops(LLVMGetMDNodeNumOperands(node));
    LLVMGetMDNodeOperands(node, ops.data());
    return ops;
}

static bool malachite_md_is(LLVMValueRef v, const char *s)
{
    unsigned len;
    const char *m = LLVMGetMDString(v, &len);
    return m && len == strlen(s) && !memcmp(m, s, len);
}

// the integers after key in a node, as {..., key, i32 a, i32 b, ...}; false for none
static bool malachite_md_ints(const std::vector<LLVMValueRef> &ops, const char *key, uint64_t *a, uint64_t *b)
{
    for (size_t i = 0; i + 1 < ops.size(); i++) {
        if (!malachite_md_is(ops[i], key) || !LLVMIsAConstantInt(ops[i + 1]))
            continue;
        *a = LLVMConstIntGetZExtValue(ops[i + 1]);
        if (b)
            *b = i + 2 < ops.size() && LLVMIsAConstantInt(ops[i + 2]) ? LLVMConstIntGetZExtValue(ops[i + 2]) : 1;
        return true;
    }
    return false;
}

// A struct_type_info node's resources, at base bytes and ids into the argument buffer; returns how many ids it spans.
// Each member is {offset, size (of an element), array length or 0, type name, name}, then for a resource an
// indirect_argument node with its [[id(n)]]. A struct member (an array of them as metal::array) has its own
// struct_type_info just before it and its first id as an integer after.
static uint64_t malachite_air_members(LLVMValueRef node, uint64_t base, uint64_t first,
                                      std::vector<struct malachite_arg> &out)
{
    std::vector<LLVMValueRef> s = malachite_md(node);
    LLVMValueRef nested = NULL;
    uint64_t span = 0;
    for (size_t j = 0; j < s.size();) {
        if (malachite_md_is(s[j], "air.struct_type_info") && j + 1 < s.size()) {
            nested = s[j + 1];
            j += 2;
            continue;
        }
        if (!LLVMIsAConstantInt(s[j]) || j + 4 >= s.size()) {
            j++;
            continue;
        }
        uint64_t offset = base + LLVMConstIntGetZExtValue(s[j]), size = LLVMConstIntGetZExtValue(s[j + 1]);
        uint64_t n = MAX(LLVMConstIntGetZExtValue(s[j + 2]), 1ull);
        for (j += 5; j + 1 < s.size() && !LLVMIsAConstantInt(s[j]); j++) {
            if (malachite_md_is(s[j], "air.struct_type_info")) {
                nested = s[++j];
                continue;
            }
            if (!malachite_md_is(s[j], "air.indirect_argument"))
                continue;
            uint64_t id, count;
            if (LLVMIsAConstantInt(s[j + 1])) {
                id = LLVMConstIntGetZExtValue(s[j + 1]);
                uint64_t step = 0;
                for (uint64_t e = 0; nested && e < n; e++)
                    step = malachite_air_members(nested, offset + e * size, first + id + e * step, out);
                span = MAX(span, id + n * step);
                nested = NULL;
            } else if (malachite_md_ints(malachite_md(s[j + 1]), "air.location_index", &id, &count)) {
                out.push_back({ first + id, n, offset, size, MTLDataTypeNone });
                span = MAX(span, id + n);
            }
            j++;
        }
    }
    return span;
}

// The layout of the argument buffer at [[buffer(index)]] as the function's AIR has it. nil unless the argument is an
// argument buffer.
static MalachiteArgumentLayout *malachite_air_layout(NSData *air, NSUInteger index)
{
    const uint8_t *bc = (const uint8_t *)air.bytes;
    size_t len = air.length;
    if (len > 20 && bc[0] == 0xde && bc[1] == 0xc0 && bc[2] == 0x17 && bc[3] == 0x0b) {  // Apple's wrapper
        uint32_t off, size;
        memcpy(&off, bc + 8, 4);
        memcpy(&size, bc + 12, 4);
        if ((size_t)off + size > len)
            return nil;
        bc += off;
        len = size;
    }
    LLVMMemoryBufferRef mb = LLVMCreateMemoryBufferWithMemoryRange((const char *)bc, len, "air", 0);
    LLVMModuleRef mod;
    if (LLVMParseBitcode2(mb, &mod)) {
        LLVMDisposeMemoryBuffer(mb);
        return nil;
    }
    MalachiteArgumentLayout *layout = nil;
    // the module is the function's own: its one entry
    std::vector<LLVMValueRef> entries;
    for (const char *stage : { "air.kernel", "air.vertex", "air.fragment" }) {
        std::vector<LLVMValueRef> e(LLVMGetNamedMetadataNumOperands(mod, stage));
        LLVMGetNamedMetadataOperands(mod, stage, e.data());
        entries.insert(entries.end(), e.begin(), e.end());
    }
    for (LLVMValueRef node : entries) {
        std::vector<LLVMValueRef> entry = malachite_md(node);
        for (LLVMValueRef arg : entry.size() > 2 ? malachite_md(entry[2]) : std::vector<LLVMValueRef>()) {
            std::vector<LLVMValueRef> a = malachite_md(arg);
            uint64_t at, length, align = 8;
            if (a.size() < 2 || !malachite_md_is(a[1], "air.indirect_buffer") ||
                !malachite_md_ints(a, "air.location_index", &at, NULL) || at != index ||
                !malachite_md_ints(a, "air.arg_type_size", &length, NULL))
                continue;
            malachite_md_ints(a, "air.arg_type_align_size", &align, NULL);
            std::vector<struct malachite_arg> members;
            for (size_t i = 0; i + 1 < a.size(); i++)
                if (malachite_md_is(a[i], "air.struct_type_info"))
                    malachite_air_members(a[i + 1], 0, 0, members);
            layout = [MalachiteArgumentLayout new];
            layout->_nargs = members.size();
            layout->_args = (struct malachite_arg *)calloc(members.size() ?: 1, sizeof(struct malachite_arg));
            std::copy(members.begin(), members.end(), layout->_args);
            layout->_length = length;
            layout->_alignment = MAX(align, 8u);
            break;
        }
        if (layout)
            break;
    }
    LLVMDisposeModule(mod);
    LLVMDisposeMemoryBuffer(mb);
    return layout;
}

// An encoder for a function's argument buffer. Metal makes it from the function's reflection, which for our
// functions has no argument buffers in it (bufferIndex "does not identify an argument buffer", and the process
// aborts): it comes from the AIR here.
typedef id (*malachite_encoder_fn)(id, SEL, NSUInteger, __autoreleasing id *, id) __attribute__((ns_returns_retained));
static malachite_encoder_fn malachite_encoder_orig;

__attribute__((ns_returns_retained)) static id malachite_function_encoder(id self, SEL cmd, NSUInteger index,
                                                                         __autoreleasing id *reflection, id fr)
{
    id<MTLFunction> function = self;
    if ([function.device isKindOfClass:objc_getClass("MalachiteDevice")]) {
        NSData *air = ((id(*)(id, SEL))objc_msgSend)(self, NSSelectorFromString(@"bitcodeData"));
        MalachiteArgumentLayout *layout = air ? malachite_air_layout(air, index) : nil;
        if (layout) {
            if (reflection)
                *reflection = nil;
            return [[MalachiteArgumentEncoder alloc] initWithLayout:layout device:function.device];
        }
    }
    return malachite_encoder_orig(self, cmd, index, reflection, fr);
}

static void malachite_hook_function_encoder(void)
{
    Method m = class_getInstanceMethod(objc_getClass("_MTLFunction"),
                                       sel_registerName("newArgumentEncoderWithBufferIndex:reflection:functionReflection:"));
    if (!m) {
        os_log(OS_LOG_DEFAULT, "Malachite: no _MTLFunction argument encoder to answer for");
        return;
    }
    malachite_encoder_orig = (malachite_encoder_fn)method_setImplementation(m, (IMP)malachite_function_encoder);
}

@implementation MalachiteDevice

// A Metal method our classes lack shows up as an unrecognised selector, usually caught and rethrown somewhere that
// crashes without saying what was missing: say it, with who called
static objc_exception_preprocessor malachite_prev_pre;
static id malachite_exc_pre(id e)
{
    NSException *x = e;
    if ([x.name isEqualToString:NSInvalidArgumentException] && [x.reason containsString:@"-[Malachite"]) {
        NSArray<NSString *> *stack = [NSThread callStackSymbols];
        os_log(OS_LOG_DEFAULT, "Malachite: %{public}@\n%{public}@", x.reason,
               [[stack subarrayWithRange:NSMakeRange(0, MIN(stack.count, 12u))] componentsJoinedByString:@"\n"]);
    }
    return malachite_prev_pre ? malachite_prev_pre(e) : e;
}

static id malachite_none(id self, SEL _cmd)
{
    return nil;
}

// each of names on the class answering nil, 0 or NO, or doing nothing
static void malachite_nothing(const char *cls, const char *protocol, NSArray<NSString *> *names)
{
    Class c = objc_getClass(cls);
    Protocol *p = objc_getProtocol(protocol);
    for (NSString *name in names) {
        SEL s = NSSelectorFromString(name);
        class_addMethod(c, s, (IMP)malachite_none, protocol_getMethodDescription(p, s, YES, YES).types);
    }
}

+ (void)load
{
    os_log(OS_LOG_DEFAULT, "MalachiteDriver: bundle loaded, MalachiteDevice registered");
    malachite_prev_pre = objc_setExceptionPreprocessor(malachite_exc_pre);
    malachite_hook_function_encoder();

    // What the device doesn't do and says it doesn't: ray tracing, function pointers and their tables, mesh and
    // object stages, indirect command buffers, counters, tensors, more than one view, Metal 4's attachment maps,
    // OpenCL's doubles and halves, Metal's assertions. The bases leave these to the driver, throwing (or calling
    // themselves) if it doesn't answer.
    malachite_nothing("MalachiteDevice", "MTLDeviceSPI", @[
        @"newIntersectionFunctionTableWithDescriptor:", @"newVisibleFunctionTableWithDescriptor:",
        @"functionHandleWithFunction:", @"functionHandleWithFunction:resourceIndex:",
        @"newIndirectCommandBufferWithDescriptor:maxCount:options:", @"newIndirectComputeCommandEncoderWithBuffer:",
        @"newIndirectRenderCommandEncoderWithBuffer:", @"supportsPlacementSparse", @"doubleFPConfig", @"halfFPConfig",
        @"metalAssertionsEnabled", @"setMetalAssertionsEnabled:"
    ]);
    malachite_nothing("MalachiteDepthStencilState", "MTLDepthStencilStateSPI", @[ @"resourceIndex" ]);
    malachite_nothing("MalachiteHeap", "MTLHeap", @[
        @"newAccelerationStructureWithSize:", @"newAccelerationStructureWithSize:offset:",
        @"newAccelerationStructureWithDescriptor:", @"newAccelerationStructureWithDescriptor:offset:"
    ]);
    malachite_nothing("MalachiteBuffer", "MTLBuffer", @[ @"newTensorWithDescriptor:offset:error:", @"sparseBufferTier" ]);
    malachite_nothing("MalachiteTexture", "MTLTexture", @[ @"sparseTextureTier" ]);
    malachite_nothing("MalachiteComputePipelineState", "MTLComputePipelineState", @[
        @"functionHandleWithFunction:", @"functionHandleWithBinaryFunction:", @"functionHandleWithName:",
        @"newComputePipelineStateWithAdditionalBinaryFunctions:error:", @"newComputePipelineStateWithBinaryFunctions:error:"
    ]);
    malachite_nothing("MalachiteRenderPipelineState", "MTLRenderPipelineState", @[
        @"functionHandleWithFunction:stage:", @"functionHandleWithBinaryFunction:stage:", @"functionHandleWithName:stage:",
        @"newVisibleFunctionTableWithDescriptor:stage:", @"newIntersectionFunctionTableWithDescriptor:stage:",
        @"newRenderPipelineStateWithAdditionalBinaryFunctions:error:", @"newRenderPipelineStateWithBinaryFunctions:error:",
        @"newRenderPipelineDescriptorForSpecialization"
    ]);
    malachite_nothing("MalachiteBlitCommandEncoder", "MTLBlitCommandEncoder", @[
        @"sampleCountersInBuffer:atSampleIndex:withBarrier:", @"resolveCounters:inRange:destinationBuffer:destinationOffset:",
        @"copyFromTensor:sourceOrigin:sourceDimensions:toTensor:destinationOrigin:destinationDimensions:"
    ]);
    malachite_nothing("MalachiteComputeCommandEncoder", "MTLComputeCommandEncoder", @[
        @"sampleCountersInBuffer:atSampleIndex:withBarrier:", @"setAccelerationStructure:atBufferIndex:",
        @"setIntersectionFunctionTable:atBufferIndex:", @"setIntersectionFunctionTables:withBufferRange:",
        @"setVisibleFunctionTable:atBufferIndex:", @"setVisibleFunctionTables:withBufferRange:"
    ]);
    malachite_nothing("MalachiteRenderCommandEncoder", "MTLRenderCommandEncoder", @[
        @"sampleCountersInBuffer:atSampleIndex:withBarrier:", @"executeCommandsInBuffer:withRange:",
        @"executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:", @"setVertexAmplificationCount:viewMappings:",
        @"setColorAttachmentMap:",
        @"setVertexAccelerationStructure:atBufferIndex:", @"setFragmentAccelerationStructure:atBufferIndex:",
        @"setTileAccelerationStructure:atBufferIndex:", @"setVertexIntersectionFunctionTable:atBufferIndex:",
        @"setFragmentIntersectionFunctionTable:atBufferIndex:", @"setTileIntersectionFunctionTable:atBufferIndex:",
        @"setVertexIntersectionFunctionTables:withBufferRange:", @"setFragmentIntersectionFunctionTables:withBufferRange:",
        @"setTileIntersectionFunctionTables:withBufferRange:", @"setVertexVisibleFunctionTable:atBufferIndex:",
        @"setFragmentVisibleFunctionTable:atBufferIndex:", @"setTileVisibleFunctionTable:atBufferIndex:",
        @"setVertexVisibleFunctionTables:withBufferRange:", @"setFragmentVisibleFunctionTables:withBufferRange:",
        @"setTileVisibleFunctionTables:withBufferRange:",
        @"setObjectBytes:length:atIndex:", @"setObjectBuffer:offset:atIndex:", @"setObjectBufferOffset:atIndex:",
        @"setObjectBuffers:offsets:withRange:", @"setObjectTexture:atIndex:", @"setObjectTextures:withRange:",
        @"setObjectSamplerState:atIndex:", @"setObjectSamplerStates:withRange:",
        @"setObjectSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
        @"setObjectSamplerStates:lodMinClamps:lodMaxClamps:withRange:", @"setObjectThreadgroupMemoryLength:atIndex:",
        @"setMeshBytes:length:atIndex:", @"setMeshBuffer:offset:atIndex:", @"setMeshBufferOffset:atIndex:",
        @"setMeshBuffers:offsets:withRange:", @"setMeshTexture:atIndex:", @"setMeshTextures:withRange:",
        @"setMeshSamplerState:atIndex:", @"setMeshSamplerStates:withRange:",
        @"setMeshSamplerState:lodMinClamp:lodMaxClamp:atIndex:", @"setMeshSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        @"drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
        @"drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
        @"drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:"
    ]);
    // every allocation is resident from when it's made, so a residency set has nothing to add
    malachite_nothing("MalachiteCommandBuffer", "MTLCommandBuffer", @[ @"useResidencySet:", @"useResidencySets:count:" ]);
    malachite_nothing("MalachiteCommandQueue", "MTLCommandQueue", @[
        @"addResidencySet:", @"addResidencySets:count:", @"removeResidencySet:", @"removeResidencySets:count:"
    ]);
}

// MTLDeviceSPI methods the base MTLIOAccelDevice leaves abstract; _MTLDevice initLimits queries them.
- (unsigned long long)featureProfile { return 0; }  // base profile

// a card in a PCIe slot, which is what a discrete GPU's driver says; the base answers built-in
- (MTLDeviceLocation)location
{
    return MTLDeviceLocationSlot;
}

- (NSUInteger)currentAllocatedSize
{
    return __atomic_load_n(&malachite_allocated, __ATOMIC_RELAXED);
}

// What a discrete GPU of Turing's or Ampere's class does on macOS. The base profile's tables support no feature set
// or family, and CoreAnimation then takes a path that leaves the backdrop a rounded or masked layer redraws over
// empty (an avatar's corners come out black); a profile with them would also bring its limits, which refuse
// IOSurface rows not on 256 bytes.
// what single precision does, as OpenCL's cl_device_fp_config bits: denormals, inf and NaN, every rounding
// mode, fused multiply-add
- (NSUInteger)singleFPConfig
{
    return 0x3f;
}

// Metal 3 too, as apps check for it before anything else (mesh and object functions it implies aren't drawn yet)
- (BOOL)supportsFamily:(MTLGPUFamily)family
{
    return family == MTLGPUFamilyMac2 || family == (MTLGPUFamily)2001 /* Mac1 */ || family == MTLGPUFamilyCommon1 ||
           family == MTLGPUFamilyCommon2 || family == MTLGPUFamilyCommon3 || family == MTLGPUFamilyMetal3;
}

// the GPU's memory, as the PCI device says: what Metal's clients size their work and textures from (the base profile's
// tables answer for a small integrated GPU)
static uint64_t malachite_vram(void)
{
    static uint64_t bytes;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        io_service_t rm = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("MalachiteGpu"));
        io_registry_entry_t pci = 0;
        if (rm && IORegistryEntryGetParentEntry(rm, kIOServicePlane, &pci) == KERN_SUCCESS) {
            CFTypeRef n = IORegistryEntryCreateCFProperty(pci, CFSTR("VRAM,totalMB"), kCFAllocatorDefault, 0);
            if (n && CFGetTypeID(n) == CFNumberGetTypeID())
                bytes = (uint64_t)[(__bridge NSNumber *)n unsignedLongLongValue] << 20;
            if (n)
                CFRelease(n);
            IOObjectRelease(pci);
        }
        if (rm)
            IOObjectRelease(rm);
        if (!bytes)
            bytes = 4ull << 30;
    });
    return bytes;
}

// the standard (D3D) positions, which the rasterizer is set to (OpenGL's renderer asks before drawing anything)
- (void)getDefaultSamplePositions:(MTLSamplePosition *)positions count:(NSUInteger)count
{
    static const float p1[] = { 8, 8 }, p2[] = { 12, 12, 4, 4 }, p4[] = { 6, 2, 14, 6, 2, 10, 10, 14 },
                       p8[] = { 9, 5, 7, 11, 13, 9, 5, 3, 3, 13, 1, 7, 11, 15, 15, 1 };
    const float *p = count == 8 ? p8 : count == 4 ? p4 : count == 2 ? p2 : p1;
    for (NSUInteger i = 0; i < count; i++)
        positions[i] = count == 1 || count == 2 || count == 4 || count == 8
                           ? MTLSamplePositionMake(p[2 * i] / 16, p[2 * i + 1] / 16)
                           : MTLSamplePositionMake(0.5f, 0.5f);
}

- (BOOL)hasUnifiedMemory { return NO; }
- (uint64_t)recommendedMaxWorkingSetSize { return malachite_vram(); }
- (NSUInteger)maxBufferLength { return (NSUInteger)(malachite_vram() / 4 * 3); }
- (MTLReadWriteTextureTier)readWriteTextureSupport { return MTLReadWriteTextureTier2; }
- (BOOL)supportsBCTextureCompression { return YES; }

- (BOOL)supportsFeatureSet:(MTLFeatureSet)set
{
    return set >= MTLFeatureSet_macOS_GPUFamily1_v1 && set <= MTLFeatureSet_macOS_GPUFamily2_v1;
}

// the base device's own query, which CoreAnimation asks
- (BOOL)deviceSupportsFeatureSet:(MTLFeatureSet)set
{
    return [self supportsFeatureSet:set];
}

// the base newCommandQueue recurses unless the concrete device creates the queue
- (id)newCommandQueueWithDescriptor:(id)descriptor
{
    return [[MalachiteCommandQueue alloc] initWithDevice:self descriptor:descriptor];
}

// MTLIOAccelDevice leaves this to the concrete driver (an unrecognized-selector crash otherwise)
- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options
{
    return [[MalachiteBuffer alloc] initWithDevice:self
                                    withLength:length
                                   withOptions:(MTLResourceOptions)options
                                   withPointer:NULL
                               withDeallocator:nil];
}

- (id)newBufferWithBytesNoCopy:(void *)pointer
                        length:(NSUInteger)length
                       options:(NSUInteger)options
                   deallocator:(void (^)(void *, NSUInteger))deallocator
{
    // OpenCL frees a fill's pattern (128 bytes at most) before the GPU runs the fill, so a buffer over it filled with
    // what malloc wrote there: patterns off a page get a copy. Not more: reads and writes into app memory come this
    // way too, and a copy per call loses the GPU's writes and starved the whole system.
    if (((uintptr_t)pointer & (vm_page_size - 1)) && length <= 128 &&
        (options & MTLResourceStorageModeMask) != MTLResourceStorageModeManaged) {
        id b = [self newBufferWithBytes:pointer length:length options:options];
        if (deallocator)
            deallocator(pointer, length);
        return b;
    }
    void (^done)(void) = nil;
    if (deallocator)
        done = ^{ deallocator(pointer, length); };
    return [[MalachiteBuffer alloc] initWithDevice:self
                                    withLength:length
                                   withOptions:(MTLResourceOptions)options
                                   withPointer:pointer
                               withDeallocator:done];
}

// A texture over an IOSurface plane: a no-copy buffer over the surface's memory as this process maps it, which
// the kernel wires and maps for the GPU like any client memory, read as a linear texture at the plane's pitch.
- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor iosurface:(IOSurfaceRef)surface plane:(NSUInteger)plane
{
    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(surface);
    uint8_t *planeBase = IOSurfaceGetPlaneCount(surface) ? (uint8_t *)IOSurfaceGetBaseAddressOfPlane(surface, plane) : base;
    size_t pitch = IOSurfaceGetPlaneCount(surface) ? IOSurfaceGetBytesPerRowOfPlane(surface, plane)
                                                   : IOSurfaceGetBytesPerRow(surface);
    size_t length = (IOSurfaceGetAllocSize(surface) + vm_page_size - 1) & ~(size_t)(vm_page_size - 1);
    if (!base || descriptor.textureType != MTLTextureType2D || !malachite_bytes_per_pixel(descriptor.pixelFormat)) {
        os_log(OS_LOG_DEFAULT, "MalachiteDevice: unsupported IOSurface texture (type %lu, format %lu)",
               (unsigned long)descriptor.textureType, (unsigned long)descriptor.pixelFormat);
        return nil;
    }
    if (planeBase < base || (size_t)(planeBase - base) >= length) {
        os_log(OS_LOG_DEFAULT, "MalachiteDevice: IOSurface %u plane %lu at %p, base %p, %zu bytes, %zu planes, pitch %zu",
               IOSurfaceGetID(surface), (unsigned long)plane, planeBase, base, IOSurfaceGetAllocSize(surface),
               IOSurfaceGetPlaneCount(surface), pitch);
        return nil;
    }
    MalachiteBuffer *buf = [self newBufferWithBytesNoCopy:base length:length options:MTLResourceStorageModeShared
                                          deallocator:nil];
    // the base wants rows 64-byte aligned, which CoreGraphics' surfaces needn't be: it gets a row that is, and
    // the texture answers with the surface's own size and pitch
    MTLTextureDescriptor *d = [descriptor copy];
    d.height = 1;
    // nor that a plane start on 64 bytes, which video surfaces' don't always (32 in): it makes the texture at the
    // surface's start, and the texture reads from the plane (the texture unit takes 32-byte steps, shadows the rest)
    size_t offset = (size_t)(planeBase - base);
    MalachiteTexture *t = [buf newTextureWithDescriptor:d offset:offset & ~(size_t)63 bytesPerRow:(pitch + 63) & ~(size_t)63];
    if (t) {
        t->_offset = offset;
        t->_pitch = (uint32_t)pitch;
        t->_linear_height = descriptor.height;
        t->_owned = buf;
        t->_surface = (IOSurfaceRef)CFRetain(surface);
        t->_plane = plane;
        NSString *name = CFBridgingRelease(IOSurfaceCopyValue(surface, kIOSurfaceName));
        t->_capture = [name containsString:@"DisplayStream"] || [name containsString:@"CaptureSurface"];
    }
    return t;
}

- (id)newIndirectArgumentBufferLayoutWithStructType:(MTLStructType *)type
{
    MalachiteArgumentLayout *l = [MalachiteArgumentLayout new];
    [l setStructType:type withDevice:self];
    return l;
}

- (id)newIndirectArgumentEncoderWithLayout:(id)layout
{
    return [[MalachiteArgumentEncoder alloc] initWithLayout:layout device:self];
}

- (id)newArgumentEncoderWithLayout:(id)layout
{
    return [self newIndirectArgumentEncoderWithLayout:layout];
}

// A sparse heap maps its pages as resources ask for them, which nothing here does: no heap.
- (id)newHeapWithDescriptor:(MTLHeapDescriptor *)descriptor
{
    if (descriptor.type == MTLHeapTypeSparse)
        return nil;
    NSUInteger page = vm_page_size, size = (descriptor.size + page - 1) & ~(page - 1);
    MalachiteHeap *h = [MalachiteHeap new];
    h->_device = self;
    h->_desc = [descriptor copy];
    h->_desc.size = size;
    h->_lock = OS_UNFAIR_LOCK_INIT;
    h->_block = size ? [self newBufferWithLength:size options:descriptor.resourceOptions] : nil;
    // out of VRAM the block gets no address the GPU can reach: no heap, as Metal answers when it can't make one
    if (!h->_block || [(MTLIOAccelBuffer *)h->_block gpuAddress] >> 50)
        return nil;
    h->_holes[0] = size;
    h->_free = size;
    return h;
}

- (MTLSizeAndAlign)heapBufferSizeAndAlignWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
    return malachite_range_in_heap(length);
}

- (id)newBufferWithBytes:(const void *)bytes length:(NSUInteger)length options:(NSUInteger)options
{
    MalachiteBuffer *b = [self newBufferWithLength:length options:options];
    memcpy(b.contents, bytes, length);
    malachite_mark_dirty(b, 0, length);  // again: a commit on another thread may have taken the empty buffer
    return b;
}

// The storage a texture's buffer takes. One the GPU renders to or writes is in VRAM whatever storage it asked for,
// and the CPU reaches its pixels through a copy (copyRegion:): iOS code makes its render targets shared, which costs
// nothing on Apple's GPUs, and in system memory here every pass of Genshin's went across PCIe, 7.5 billion pixels in
// 10 s, at 5 fps. One the GPU only reads stays where it asked to be (a managed one with its VRAM copy): a blit may
// still write it, and a shared one is read back after that with nothing to say so.
static MTLResourceOptions malachite_storage(MTLTextureDescriptor *descriptor)
{
    bool gpu_writes = descriptor.usage & (MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite);
    if (descriptor.storageMode == MTLStorageModePrivate || descriptor.storageMode == MTLStorageModeMemoryless ||
        gpu_writes)
        return MTLResourceStorageModePrivate;
    return descriptor.storageMode == MTLStorageModeManaged ? MTLResourceStorageModeManaged : MTLResourceStorageModeShared;
}

// a texture the device made, told the storage it asked for where that isn't where it is
static void malachite_storage_asked(MalachiteTexture *t, MTLTextureDescriptor *descriptor)
{
    if (!t)
        return;
    t->_asked = descriptor.storageMode;
    t->_moved = true;
}

// How the device lays a texture out. A plain 2D one (a 1D one, a 1D array) is linear rows, the pitch 256-byte
// aligned so it can be a render target as well as sampled; 1D textures are a row, 1D arrays a row per layer. Anything
// else is an image, its layers stride apart. bytes is 0 for a texture the device can't make.
struct malachite_layout {
    bool image;
    NSUInteger pitch, height;  // an image's: level 0's row of GOBs, and its height
    uint32_t depth, layers, levels, samples;
    uint64_t stride, bytes;
};

static struct malachite_layout malachite_layout_of(MTLTextureDescriptor *descriptor)
{
    struct malachite_layout l = {};
    MTLTextureType type = descriptor.textureType;
    int f = malachite_format(descriptor.pixelFormat);
    uint32_t zbpp;
    uint8_t kind;
    bool zeta = malachite_zeta(descriptor.pixelFormat, &zbpp, &kind) != 0;
    bool plain = type == MTLTextureType2D || type == MTLTextureType1D || type == MTLTextureType1DArray;
    if (f < 0 || type == MTLTextureTypeTextureBuffer || descriptor.mipmapLevelCount > 16 ||
        (type == MTLTextureType1DArray && descriptor.mipmapLevelCount > 1)) {
        os_log(OS_LOG_DEFAULT, "MalachiteDevice: unsupported texture (type %lu, %lu levels, format %lu)",
               (unsigned long)type, (unsigned long)descriptor.mipmapLevelCount, (unsigned long)descriptor.pixelFormat);
        return l;
    }
    // a 2D render target is an image too, block-linear: the texture unit reads pitch-linear rows with little reuse
    // of what it cached, and passes sampling Genshin's render targets took 1-2 ms each
    bool target = type == MTLTextureType2D && (descriptor.usage & MTLTextureUsageRenderTarget) &&
                  malachite_storage(descriptor) == MTLResourceStorageModePrivate;
    if (plain && !zeta && !target && descriptor.mipmapLevelCount == 1 && tex_format_info((enum tex_format)f).block == 1) {
        l.pitch = (descriptor.width * tex_format_info((enum tex_format)f).bytes + 255) & ~(NSUInteger)255;
        l.height = type == MTLTextureType1DArray ? descriptor.arrayLength : descriptor.height;
        l.bytes = l.pitch * l.height;
        return l;
    }
    l.image = true;
    l.depth = type == MTLTextureType3D ? (uint32_t)descriptor.depth : 1;
    l.height = type == MTLTextureType1D ? 1 : descriptor.height;
    l.layers = (uint32_t)descriptor.arrayLength * (malachite_is_cube(type) ? 6 : 1);
    l.levels = (uint32_t)descriptor.mipmapLevelCount;
    bool ms = type == MTLTextureType2DMultisample || type == MTLTextureType2DMultisampleArray;
    l.samples = ms && descriptor.sampleCount > 1 ? (uint32_t)descriptor.sampleCount : 0;
    uint32_t sx, sy;
    malachite_grid(l.samples, &sx, &sy);
    l.stride = malachite_image_level(descriptor.pixelFormat, descriptor.width * sx, l.height * sy, l.depth, l.levels).offset;
    l.pitch = malachite_image_level(descriptor.pixelFormat, descriptor.width * sx, l.height * sy, l.depth, 0).pitch;
    l.bytes = l.stride * l.layers;
    return l;
}

- (MTLSizeAndAlign)heapTextureSizeAndAlignWithDescriptor:(MTLTextureDescriptor *)descriptor
{
    return malachite_range_in_heap(malachite_layout_of(descriptor).bytes);
}

- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
    return [self newTextureWithDescriptor:descriptor heap:nil offset:0];
}

// A texture of its own, or with heap, in a range of the heap's memory: at offset, or for NSNotFound wherever the heap
// has room.
- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor heap:(MalachiteHeap *)heap offset:(NSUInteger)offset
{
    struct malachite_layout l = malachite_layout_of(descriptor);
    if (!l.bytes)
        return nil;
    MalachiteHeapBuffer *placed =
        heap ? [heap place:l.bytes size:malachite_range_in_heap(l.bytes) options:heap.resourceOptions at:offset] : nil;
    if (heap && !placed)
        return nil;
    if (l.image)
        return malachite_swizzled([self newImage:descriptor layout:l placed:placed], descriptor.swizzle);
    MTLTextureDescriptor *d2 = [descriptor copy];
    d2.textureType = MTLTextureType2D;
    d2.height = l.height;
    d2.arrayLength = 1;
    MalachiteBuffer *buf = placed ? (MalachiteBuffer *)heap->_block
                                  : [self newBufferWithLength:l.bytes options:malachite_storage(descriptor)];
    MalachiteTexture *t = [buf newTextureWithDescriptor:d2 offset:placed.heapOffset bytesPerRow:l.pitch];
    if (t) {
        t->_owned = buf;
        t->_placed = placed;
        t->_type = descriptor.textureType;
    }
    if (!placed)  // one in a heap is in the heap's storage
        malachite_storage_asked(t, descriptor);
    return malachite_swizzled(t, descriptor.swizzle);
}

// An image: its layer's levels one after another, the layers after each other, in a buffer made for it or its range
// of a heap. A depth/stencil one is also a target the Z/S hardware renders to, which only renders block-linear, and
// its memory takes the target's PTE kind. The kernel gives a kind to a whole buffer's mapping (set_kind), which would
// have the heap's other resources read through a Z kind, so one made in a heap still has a buffer of its own, its
// range of the heap held for it unused.
- (id)newImage:(MTLTextureDescriptor *)descriptor layout:(struct malachite_layout)l placed:(MalachiteHeapBuffer *)placed
{
    uint32_t zbpp;
    uint8_t kind;
    bool own = !placed || malachite_zeta(descriptor.pixelFormat, &zbpp, &kind);
    MalachiteBuffer *buf = own ? [self newBufferWithLength:(l.bytes + 0xffff) & ~(uint64_t)0xffff
                                                   options:malachite_storage(descriptor)]
                               : (MalachiteBuffer *)placed->_heap->_block;
    MalachiteTexture *t = malachite_image_over(buf, descriptor.usage);
    if (!t)
        return nil;
    if (own)
        buf->_one_image = true;
    t->_owned = buf;
    t->_placed = placed;
    t->_offset = own ? 0 : placed->_offset;
    t->_type = descriptor.textureType;
    t->_pitch = (uint32_t)l.pitch;
    t->_samples = l.samples;
    t->_format = descriptor.pixelFormat;
    t->_width = descriptor.width;
    t->_height = l.height;
    t->_depth = l.depth;
    t->_layers = l.layers;
    t->_levels = t->_image_levels = l.levels;
    t->_layer_stride = l.stride;
    t->_zt_format = malachite_zeta(descriptor.pixelFormat, &t->_zt_bpp, &t->_zt_kind);
    if (own)
        malachite_storage_asked(t, descriptor);
    return t;
}

- (id)newDepthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
{
    MalachiteDepthStencilState *s = [[MalachiteDepthStencilState alloc] initWithDevice:self depthStencilDescriptor:descriptor];
    // MTLCompareFunction never..always and MTLStencilOperation in the 3D class's OpenGL-style values
    static const uint32_t op[] = { 0x1e00, 0, 0x1e01, 0x1e02, 0x1e03, 0x150a, 0x8507, 0x8508 };
    s->_depth_func = 0x200 + (uint32_t)descriptor.depthCompareFunction;
    s->_depth_write = descriptor.depthWriteEnabled;
    s->_depth_test = descriptor.depthCompareFunction != MTLCompareFunctionAlways || descriptor.depthWriteEnabled;
    MTLStencilDescriptor *faces[2] = { descriptor.frontFaceStencil, descriptor.backFaceStencil };
    for (int i = 0; i < 2; i++) {
        MTLStencilDescriptor *f = faces[i];
        s->_stencil[i][0] = op[f.stencilFailureOperation];
        s->_stencil[i][1] = op[f.depthFailureOperation];
        s->_stencil[i][2] = op[f.depthStencilPassOperation];
        s->_stencil[i][3] = 0x200 + (uint32_t)f.stencilCompareFunction;
        s->_read_mask[i] = f.readMask & 0xff;  // the 3D class's fields are 8 bits, Metal's default all ones
        s->_write_mask[i] = f.writeMask & 0xff;
        s->_stencil_test |= f.stencilCompareFunction != MTLCompareFunctionAlways ||
                            f.stencilFailureOperation != MTLStencilOperationKeep ||
                            f.depthFailureOperation != MTLStencilOperationKeep ||
                            f.depthStencilPassOperation != MTLStencilOperationKeep;
    }
    if (malachite_rec_only)
        objc_setAssociatedObject(s, &malachite_desc_key, [descriptor copy], OBJC_ASSOCIATION_RETAIN);
    return s;
}

- (id)newSamplerStateWithDescriptor:(MTLSamplerDescriptor *)descriptor
{
    MalachiteSamplerState *s = [[MalachiteSamplerState alloc] initWithDevice:self samplerDescriptor:descriptor];
    if (malachite_trace || malachite_dump)
        os_log(OS_LOG_DEFAULT, "Malachite trace: sampler %{public}@: min %lu mag %lu mip %lu, address %lu %lu, normalized %d, "
               "lod %g..%g, aniso %lu", descriptor.label, (unsigned long)descriptor.minFilter,
               (unsigned long)descriptor.magFilter, (unsigned long)descriptor.mipFilter,
               (unsigned long)descriptor.sAddressMode, (unsigned long)descriptor.tAddressMode,
               descriptor.normalizedCoordinates, descriptor.lodMinClamp, descriptor.lodMaxClamp,
               (unsigned long)descriptor.maxAnisotropy);
    // MTLSamplerAddressMode: clamp to edge, mirror clamp to edge, repeat, mirror repeat, clamp to zero, border
    static const enum tex_address address[] = { TEX_CLAMP_TO_EDGE, TEX_MIRROR_ONCE_CLAMP_TO_EDGE, TEX_WRAP,
                                                TEX_MIRROR,        TEX_BORDER,                    TEX_BORDER };
    struct samp_desc *d = &s->_desc;
    d->mag_linear = descriptor.magFilter == MTLSamplerMinMagFilterLinear;
    d->min_linear = descriptor.minFilter == MTLSamplerMinMagFilterLinear;
    d->mip = (enum tex_mip)descriptor.mipFilter;
    d->address[0] = address[descriptor.sAddressMode];
    d->address[1] = address[descriptor.tAddressMode];
    d->address[2] = address[descriptor.rAddressMode];
    d->normalized = descriptor.normalizedCoordinates;
    d->max_anisotropy = (unsigned)descriptor.maxAnisotropy;
    d->lod_min = malachite_lod(descriptor.lodMinClamp);
    d->lod_max = malachite_lod(descriptor.lodMaxClamp);
    d->compare = descriptor.compareFunction == MTLCompareFunctionNever ? 0 : (int)descriptor.compareFunction + 1;
    // clamp to zero is transparent black outside
    d->border = descriptor.sAddressMode == MTLSamplerAddressModeClampToZero ? TEX_BORDER_TRANSPARENT_BLACK
                                                                            : (enum tex_border)descriptor.borderColor;
    if (malachite_rec_only)
        objc_setAssociatedObject(s, &malachite_desc_key, [descriptor copy], OBJC_ASSOCIATION_RETAIN);
    return s;
}

// OpenGL's shaders, which Apple's GL renderer hands over as GL core IR (bitcodeType 1) with its pipeline key as the
// plugin data, made AIR by the plugin Apple's compiler service uses for them, from the shared cache, with the LLVM
// it's built against. The plugin's return data (where the program's buffers, textures and samplers went) comes
// back in *ret. nil if the plugin isn't there or fails.
static NSData *malachite_glir_to_air(NSData *glir, NSData *key, NSData **ret)
{
    typedef void *(*gen_fn)(void *, const void *, size_t, const void *, size_t, void **, size_t *);
    static gen_fn gen;
    static void *(*context_create)(void), (*context_dispose)(void *), (*module_dispose)(void *),
        *(*write_bitcode)(void *), (*buffer_dispose)(void *);
    static const char *(*buffer_start)(void *);
    static size_t (*buffer_size)(void *);
    static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;  // the plugin is one global instance
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        void *llvm = dlopen("/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/"
                            "libLLVM.dylib", RTLD_NOW | RTLD_LOCAL);
        void *gld = dlopen("/System/Library/Extensions/AppleMetalOpenGLRenderer.bundle/Contents/PlugIns/"
                           "libGLDPlugin32023.dylib", RTLD_NOW | RTLD_LOCAL);
        void (*init)(void) = gld ? (void (*)(void))dlsym(gld, "oglCodeGenServiceInitialize") : NULL;
        if (!llvm || !init) {
            os_log(OS_LOG_DEFAULT, "Malachite: no GL core IR converter: %{public}s", dlerror());
            return;
        }
        context_create = (void *(*)(void))dlsym(llvm, "LLVMContextCreate");
        context_dispose = (void (*)(void *))dlsym(llvm, "LLVMContextDispose");
        module_dispose = (void (*)(void *))dlsym(llvm, "LLVMDisposeModule");
        write_bitcode = (void *(*)(void *))dlsym(llvm, "LLVMWriteBitcodeToMemoryBuffer");
        buffer_dispose = (void (*)(void *))dlsym(llvm, "LLVMDisposeMemoryBuffer");
        buffer_start = (const char *(*)(void *))dlsym(llvm, "LLVMGetBufferStart");
        buffer_size = (size_t(*)(void *))dlsym(llvm, "LLVMGetBufferSize");
        init();
        gen = (gen_fn)dlsym(gld, "oglCodeGenServiceGenerateIRWithPluginDataAndPluginReturnData");
    });
    if (!gen || !key.length)  // without a key the plugin crashes
        return nil;
    os_unfair_lock_lock(&lock);
    void *ctx = context_create(), *data = NULL;
    size_t len = 0;
    void *module = gen(ctx, glir.bytes, glir.length, key.bytes, key.length, &data, &len);
    NSData *air = nil;
    if (module) {
        void *mb = write_bitcode(module);
        air = [NSData dataWithBytes:buffer_start(mb) length:buffer_size(mb)];
        buffer_dispose(mb);
        module_dispose(module);
    }
    context_dispose(ctx);
    os_unfair_lock_unlock(&lock);
    if (ret)
        *ret = data ? [NSData dataWithBytesNoCopy:data length:len freeWhenDone:YES] : nil;
    else
        free(data);
    return air;
}

// Compiled shaders, kept in memory and in the user's cache directory, so a pipeline made again (in this process, or
// in the next: WindowServer's at every login) isn't compiled again. An entry's key is a SHA-256 of everything the
// compile reads, this build of the plugin (its LC_UUID) among it, so an update leaves the old entries unused.
// MALACHITE_NO_CACHE turns it off.
enum {
    MALACHITE_CACHE_MAGIC = 0x434d564e,  // "NVMC"
    MALACHITE_CACHE_VERSION = 1,
    MALACHITE_CACHE_MEMORY = 16 << 20,
    MALACHITE_CACHE_DISK = 64 << 20,
};

struct malachite_cache_file {
    uint32_t magic, version;
    uint8_t identity[16];
    uint8_t key[CC_SHA256_DIGEST_LENGTH];
    uint64_t size;  // of the entry after this header
    uint8_t sum[CC_SHA256_DIGEST_LENGTH];  // the entry's SHA-256
};

static struct {
    bool on;
    uint8_t identity[16];
    NSCache<NSData *, NSData *> *memory;
    int dir;  // the directory, or -1 (a sandbox that won't have it) to keep entries in memory only
    os_unfair_lock disk_lock;
    long long disk_bytes;  // what the directory holds as far as this process knows, -1 before it's looked
    unsigned memory_hits, disk_hits, misses;
} malachite_cache;

static void malachite_cache_log(void)
{
    os_log(OS_LOG_DEFAULT, "Malachite: shader cache: %u hits in memory, %u on disk, %u compiled%{public}s",
           malachite_cache.memory_hits, malachite_cache.disk_hits, malachite_cache.misses,
           malachite_cache.dir < 0 ? " (no cache directory)" : "");
}

static bool malachite_cache_init(void)
{
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        malachite_cache.dir = -1;
        malachite_cache.disk_bytes = -1;
        Dl_info info;
        const struct mach_header_64 *mh =
            dladdr((void *)malachite_cache_init, &info) ? (const struct mach_header_64 *)info.dli_fbase : NULL;
        const struct load_command *lc = mh ? (const struct load_command *)(mh + 1) : NULL;
        for (uint32_t i = 0; lc && i < mh->ncmds; i++, lc = (const struct load_command *)((const char *)lc + lc->cmdsize))
            if (lc->cmd == LC_UUID) {
                memcpy(malachite_cache.identity, ((const struct uuid_command *)lc)->uuid, sizeof(malachite_cache.identity));
                malachite_cache.on = !getenv("MALACHITE_NO_CACHE");
            }
        if (!malachite_cache.on)
            return;
        malachite_cache.memory = [NSCache new];
        malachite_cache.memory.totalCostLimit = MALACHITE_CACHE_MEMORY;
        // confstr's is per user and there for daemons (WindowServer's _windowserver) and sandboxed apps as well
        char base[PATH_MAX];
        size_t n = confstr(_CS_DARWIN_USER_CACHE_DIR, base, sizeof(base));
        NSString *path = n && n <= sizeof(base) ? @(base)
                                                : [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches"];
        const char *dir = [path stringByAppendingPathComponent:@"pl.aochi.Malachite"].fileSystemRepresentation;
        mkdir(dir, 0700);
        int fd = open(dir, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        struct stat st;
        if (fd >= 0 && (fstat(fd, &st) || st.st_uid != geteuid() || (st.st_mode & (S_IWGRP | S_IWOTH)))) {
            close(fd);
            fd = -1;
        }
        malachite_cache.dir = fd;
        if (malachite_trace)
            atexit(malachite_cache_log);
    });
    return malachite_cache.on;
}

// lengths go in ahead of what's variable, so two different sets of inputs can't run together into the same bytes
static void malachite_cache_hash(CC_SHA256_CTX *c, const void *p, size_t n)
{
    uint64_t len = n;
    CC_SHA256_Update(c, &len, sizeof(len));
    CC_SHA256_Update(c, p, (CC_LONG)n);
}

static void malachite_cache_key(uint8_t key[CC_SHA256_DIGEST_LENGTH], NSData *air, NSArray<NSData *> *bitcode,
                            NSArray<NSString *> *names, const struct airc_varyings *vertex,
                            const struct airc_stage_in *stage_in, uint32_t samples)
{
    // a fragment function's inputs pair with the vertex outputs by name, or in order without any
    uint32_t fixed[5] = { getenv("AIRC_SERIAL") != NULL, (uint32_t)bitcode.count, vertex ? vertex->n : UINT32_MAX,
                          stage_in != NULL, samples };
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    CC_SHA256_Update(&c, malachite_cache.identity, sizeof(malachite_cache.identity));
    CC_SHA256_Update(&c, fixed, sizeof(fixed));
    malachite_cache_hash(&c, air.bytes, air.length);
    for (NSUInteger i = 0; i < bitcode.count; i++) {
        malachite_cache_hash(&c, bitcode[i].bytes, bitcode[i].length);
        malachite_cache_hash(&c, names[i].UTF8String, strlen(names[i].UTF8String));
    }
    for (uint32_t i = 0; vertex && i < vertex->n; i++)
        malachite_cache_hash(&c, vertex->names[i], strnlen(vertex->names[i], AIRC_NAME));
    if (stage_in)
        malachite_cache_hash(&c, stage_in, sizeof(*stage_in));
    CC_SHA256_Final(key, &c);
}

static void malachite_cache_name(char name[2 * CC_SHA256_DIGEST_LENGTH + 1], const uint8_t key[CC_SHA256_DIGEST_LENGTH])
{
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
        snprintf(name + 2 * i, 3, "%02x", key[i]);
}

// for the kernel's log of a fault (gpu_bridge.h); it fits in the page of slack after the data
static void malachite_tag_program(id<MTLBuffer> code, const struct nvk_header *h, NSString *name)
{
    struct nvgpu_program_tag t = { NVGPU_PROGRAM_TAG };
    strlcpy(t.name, name.UTF8String ?: "", sizeof(t.name));
    for (uint32_t i = 0; i < h->nargs && t.nbuffers < NVGPU_PROGRAM_TAG_BUFFERS; i++) {
        const struct nvk_arg *a = &h->args[i];
        if (a->kind == NVK_ARG_BUFFER || a->kind == NVK_ARG_STAGE_IN_BUFFER) {
            t.buffers[t.nbuffers].index = a->index;
            t.buffers[t.nbuffers].cbuf_off = a->cbuf_off;
            t.nbuffers++;
        }
    }
    memcpy((uint8_t *)code.contents + ((h->sass_size + h->data_size + 255) & ~255u), &t, sizeof(t));
}

// a compile as the cache keeps it
static NSData *malachite_cache_entry(const struct airc_kernel *k)
{
    size_t bytes = (size_t)k->h.sass_size + k->h.data_size;
    NSMutableData *entry = [NSMutableData dataWithLength:sizeof(struct malachite_compiled) + bytes];
    struct malachite_compiled *c = (struct malachite_compiled *)entry.mutableBytes;
    c->h = k->h;
    memcpy(c->arg_names, k->arg_names, sizeof(c->arg_names));
    c->outputs = k->outputs;
    memcpy(c + 1, k->sass, bytes);
    return entry;
}

// an entry read back, checked as far as anything indexes or reads strings with what's in it
static bool malachite_cache_entry_ok(NSData *entry)
{
    const struct malachite_compiled *c = (const struct malachite_compiled *)entry.bytes;
    if (entry.length < sizeof(*c) || c->h.magic != NVK_MAGIC ||
        entry.length - sizeof(*c) != (size_t)c->h.sass_size + c->h.data_size ||
        c->h.nargs > NVK_MAX_ARGS || c->h.nconst_samplers > NVK_MAX_CONST_SAMPLERS || c->h.stage > NVK_TESS_INIT ||
        c->outputs.n > sizeof(c->outputs.names) / sizeof(c->outputs.names[0]))
        return false;
    for (unsigned i = 0; i < NVK_MAX_ARGS; i++)
        if (!memchr(c->arg_names[i].name, 0, AIRC_NAME) || !memchr(c->arg_names[i].type, 0, AIRC_NAME))
            return false;
    for (unsigned i = 0; i < c->outputs.n; i++)
        if (!memchr(c->outputs.names[i], 0, AIRC_NAME))
            return false;
    return true;
}

// an entry from the directory: anything that isn't exactly what this build wrote under that key is a miss
static NSData *malachite_cache_read(const uint8_t key[CC_SHA256_DIGEST_LENGTH])
{
    char name[2 * CC_SHA256_DIGEST_LENGTH + 1];
    malachite_cache_name(name, key);
    int fd = malachite_cache.dir < 0 ? -1 : openat(malachite_cache.dir, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0)
        return nil;
    struct stat st;
    NSMutableData *file = nil;
    if (!fstat(fd, &st) && S_ISREG(st.st_mode) && st.st_size > (off_t)sizeof(struct malachite_cache_file) &&
        st.st_size <= MALACHITE_CACHE_DISK) {
        file = [NSMutableData dataWithLength:(NSUInteger)st.st_size];
        if (read(fd, file.mutableBytes, file.length) != (ssize_t)file.length)
            file = nil;
    }
    NSData *entry = nil;
    if (file) {
        const struct malachite_cache_file *f = (const struct malachite_cache_file *)file.bytes;
        entry = [file subdataWithRange:NSMakeRange(sizeof(*f), file.length - sizeof(*f))];
        uint8_t sum[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(entry.bytes, (CC_LONG)entry.length, sum);
        if (f->magic != MALACHITE_CACHE_MAGIC || f->version != MALACHITE_CACHE_VERSION ||
            memcmp(f->identity, malachite_cache.identity, sizeof(f->identity)) || memcmp(f->key, key, sizeof(f->key)) ||
            f->size != entry.length || memcmp(f->sum, sum, sizeof(sum)) || !malachite_cache_entry_ok(entry))
            entry = nil;
    }
    if (entry)
        futimens(fd, NULL);  // eviction goes by last use
    close(fd);
    return entry;
}

struct malachite_cache_old {
    struct timespec used;
    off_t size;
    char name[2 * CC_SHA256_DIGEST_LENGTH + 32];
};

static int malachite_cache_older(const void *a, const void *b)
{
    const struct timespec *x = &((const struct malachite_cache_old *)a)->used;
    const struct timespec *y = &((const struct malachite_cache_old *)b)->used;
    if (x->tv_sec != y->tv_sec)
        return x->tv_sec < y->tv_sec ? -1 : 1;
    return (x->tv_nsec > y->tv_nsec) - (x->tv_nsec < y->tv_nsec);
}

// the directory past its cap loses its least recently used entries, down to 3/4 of the cap so the next store doesn't
// do it again; the bytes left in it. Its own fd: readdir moves an offset malachite_cache.dir's other users share.
static long long malachite_cache_evict(void)
{
    int fd = openat(malachite_cache.dir, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    DIR *d = fd >= 0 ? fdopendir(fd) : NULL;
    if (!d) {
        if (fd >= 0)
            close(fd);
        return 0;
    }
    struct malachite_cache_old *old = NULL;
    size_t n = 0, cap = 0;
    long long total = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        struct stat st;
        if (strlen(de->d_name) >= sizeof(old->name) || fstatat(fd, de->d_name, &st, AT_SYMLINK_NOFOLLOW) ||
            !S_ISREG(st.st_mode))
            continue;
        if (n == cap) {
            struct malachite_cache_old *more =
                (struct malachite_cache_old *)realloc(old, (cap = cap ? 2 * cap : 256) * sizeof(*old));
            if (!more)
                break;
            old = more;
        }
        old[n].used = st.st_mtimespec;
        old[n].size = st.st_size;
        strlcpy(old[n].name, de->d_name, sizeof(old[n].name));
        n++;
        total += st.st_size;
    }
    if (total > MALACHITE_CACHE_DISK) {
        qsort(old, n, sizeof(*old), malachite_cache_older);
        for (size_t i = 0; i < n && total > MALACHITE_CACHE_DISK / 4 * 3; i++)
            if (!unlinkat(fd, old[i].name, 0))
                total -= old[i].size;
    }
    free(old);
    closedir(d);
    return total;
}

// written whole under a name of its own first, so a reader never sees half an entry
static void malachite_cache_write(const uint8_t key[CC_SHA256_DIGEST_LENGTH], NSData *entry)
{
    if (malachite_cache.dir < 0)
        return;
    struct malachite_cache_file f = { MALACHITE_CACHE_MAGIC, MALACHITE_CACHE_VERSION };
    memcpy(f.identity, malachite_cache.identity, sizeof(f.identity));
    memcpy(f.key, key, sizeof(f.key));
    f.size = entry.length;
    CC_SHA256(entry.bytes, (CC_LONG)entry.length, f.sum);
    NSMutableData *file = [NSMutableData dataWithBytes:&f length:sizeof(f)];
    [file appendData:entry];
    char name[2 * CC_SHA256_DIGEST_LENGTH + 1], tmp[sizeof(name) + 32];
    malachite_cache_name(name, key);
    snprintf(tmp, sizeof(tmp), ".%s.%d.%08x", name, getpid(), arc4random());
    int fd = openat(malachite_cache.dir, tmp, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0)
        return;
    bool ok = write(fd, file.bytes, file.length) == (ssize_t)file.length;
    if (close(fd) || !ok || renameat(malachite_cache.dir, tmp, malachite_cache.dir, name)) {
        unlinkat(malachite_cache.dir, tmp, 0);
        return;
    }
    os_unfair_lock_lock(&malachite_cache.disk_lock);
    if (malachite_cache.disk_bytes < 0 || (malachite_cache.disk_bytes += file.length) > MALACHITE_CACHE_DISK)
        malachite_cache.disk_bytes = malachite_cache_evict();
    os_unfair_lock_unlock(&malachite_cache.disk_lock);
}

static NSData *malachite_cache_load(const uint8_t key[CC_SHA256_DIGEST_LENGTH])
{
    NSData *k = [NSData dataWithBytes:key length:CC_SHA256_DIGEST_LENGTH];
    NSData *entry = [malachite_cache.memory objectForKey:k];
    if (entry) {
        __sync_fetch_and_add(&malachite_cache.memory_hits, 1);
        return entry;
    }
    entry = malachite_cache_read(key);
    if (entry) {
        [malachite_cache.memory setObject:entry forKey:k cost:entry.length];
        __sync_fetch_and_add(&malachite_cache.disk_hits, 1);
    } else {
        __sync_fetch_and_add(&malachite_cache.misses, 1);
    }
    return entry;
}

static void malachite_cache_store(const uint8_t key[CC_SHA256_DIGEST_LENGTH], NSData *entry)
{
    [malachite_cache.memory setObject:entry forKey:[NSData dataWithBytes:key length:CC_SHA256_DIGEST_LENGTH]
                             cost:entry.length];
    malachite_cache_write(key, entry);
}

static os_unfair_lock malachite_airc_lock = OS_UNFAIR_LOCK_INIT;  // airc has global state; one compile at a time

- (id<MTLBuffer>)compile:(id<MTLFunction>)function header:(struct nvk_header *)h error:(NSError **)error
{
    return [self compile:function linked:nil vertex:NULL stageIn:NULL samples:1 outputs:NULL header:h error:error];
}

// Compile a function's AIR to SASS with airc, in process, into a GPU-mapped buffer. nil (and *error) on failure.
// The pipeline's linked functions are what the function's visible-function calls resolve to; a fragment function's
// inputs are paired with the vertex function's outputs (vertex) by name, and a vertex function's are given back;
// a kernel's stage input is fetched as stageIn lays it out. samples is the pipeline's raster sample count.
- (id<MTLBuffer>)compile:(id<MTLFunction>)function
                  linked:(MTLLinkedFunctions *)linked
                  vertex:(const struct airc_varyings *)vertex
                 stageIn:(const struct airc_stage_in *)stageIn
                 samples:(NSUInteger)samples
                 outputs:(struct airc_varyings *)outputs
                  header:(struct nvk_header *)h
                   error:(NSError **)error
{
    static struct airc_kernel k;
    NSData *air = ((id(*)(id, SEL))objc_msgSend)(function, NSSelectorFromString(@"bitcodeData"));
    BOOL gl = NO;
    if ([(id)function respondsToSelector:NSSelectorFromString(@"bitcodeType")] &&
        ((NSUInteger(*)(id, SEL))objc_msgSend)(function, NSSelectorFromString(@"bitcodeType")) == 1) {
        gl = YES;
        NSData *key = ((id(*)(id, SEL))objc_msgSend)(function, NSSelectorFromString(@"pluginData")), *ret = nil;
        NSData *converted = malachite_glir_to_air(air, key, &ret);
        if (converted)
            air = converted;
        // a vertex function with a uniform block comes back with no return data, and is as much a GL one: without
        // its arguments the renderer finds no uniform buffer to bind
        objc_setAssociatedObject(function, &malachite_return_key, ret, OBJC_ASSOCIATION_RETAIN);
        if (malachite_trace)
            os_log(OS_LOG_DEFAULT, "Malachite trace: GL function: %lu bytes of IR, key %lu bytes, converted %d, return data %lu",
                   (unsigned long)air.length, (unsigned long)key.length, converted != nil, (unsigned long)ret.length);
    }
    NSMutableArray<NSData *> *bitcode = [NSMutableArray array];
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    NSMutableArray<id<MTLFunction>> *fns = [NSMutableArray arrayWithArray:linked.functions ?: @[]];
    [fns addObjectsFromArray:linked.privateFunctions ?: @[]];
    for (id<MTLFunction> fn in fns) {
        NSData *b = ((id(*)(id, SEL))objc_msgSend)(fn, NSSelectorFromString(@"bitcodeData"));
        if (b) {
            [bitcode addObject:b];
            [names addObject:fn.name ?: @""];
        }
    }
    // with the trace on, /tmp/malachite-override-<name>.air stands in for the function's own bitcode: another app's
    // shader replayed under a stand-in function of its name (tools/mtl/mtlreplay), and <name>.linked<i>.air for
    // its i-th linked function's, as the dump names them
    NSString *over_path = [NSString stringWithFormat:@"/tmp/malachite-override-%@.air", function.name];
    NSData *over = malachite_trace && malachite_trusted_file(over_path.fileSystemRepresentation, true)
                       ? [NSData dataWithContentsOfFile:over_path]
                       : nil;
    if (over) {
        os_log(OS_LOG_DEFAULT, "Malachite: %{public}@: compiling the override", function.name);
        air = over;
        for (NSUInteger i = 0; i < bitcode.count; i++) {
            NSString *lp = [NSString stringWithFormat:@"/tmp/malachite-override-%@.linked%lu.air", function.name,
                                                      (unsigned long)i];
            NSData *l = malachite_trusted_file(lp.fileSystemRepresentation, true) ? [NSData dataWithContentsOfFile:lp] : nil;
            if (l)
                bitcode[i] = l;
        }
    }
    if (malachite_rec_only && air)
        objc_setAssociatedObject(function, &malachite_air_key, @[ air, [bitcode copy], [names copy] ],
                                 OBJC_ASSOCIATION_RETAIN);
    // an override is always compiled: it's there because the compiler is being worked on
    bool cached = !over && malachite_cache_init();
    uint8_t key[CC_SHA256_DIGEST_LENGTH];
    if (cached)
        malachite_cache_key(key, air, bitcode, names, vertex, stageIn, (uint32_t)samples);
    NSData *entry = cached ? malachite_cache_load(key) : nil;
    NSString *why = nil;
    if (!entry) {
        std::vector<struct airc_source> sources;
        for (NSUInteger i = 0; i < bitcode.count; i++)
            sources.push_back({ bitcode[i].bytes, bitcode[i].length, names[i].UTF8String });
        os_unfair_lock_lock(&malachite_airc_lock);
        if (airc_compile_linked(air.bytes, air.length, sources.data(), (unsigned)sources.size(), vertex, stageIn,
                                (unsigned)samples, NULL, 0, 0, &k))
            why = @(k.error);
        else
            entry = malachite_cache_entry(&k);
        free(k.sass);
        os_unfair_lock_unlock(&malachite_airc_lock);
        if (entry && cached)
            malachite_cache_store(key, entry);
    }
    bool failed = !entry;
    const struct malachite_compiled *c = (const struct malachite_compiled *)entry.bytes;
    if (!failed && outputs)
        *outputs = c->outputs;
    // reflection for the ones that read it: OpenGL's renderer, and OpenCL, whose kernels come as __entry_<name> and
    // which takes its arguments' indices from it (without, clSetKernelArg says there's no such argument), and counts
    // them from the function's own reflection (malachite_cl_reflection)
    bool cl = !gl && [function.name hasPrefix:@"__entry_"];
    if (!failed && (gl || cl))
        objc_setAssociatedObject(function, &malachite_arguments_key,
                                 cl ? malachite_cl_pipeline_arguments(c) : malachite_arguments(c),
                                 OBJC_ASSOCIATION_RETAIN);
    if (!failed && cl)
        objc_setAssociatedObject(function, &malachite_cl_program_key, malachite_cl_program_arguments(c),
                                 OBJC_ASSOCIATION_RETAIN);
    *h = failed ? (struct nvk_header){} : c->h;
    // the SMs prefetch instructions past a program's end: a page of slack keeps that inside the buffer (with none it
    // reads the next mapping, and faults once that one is freed). The constant data goes between the two.
    id<MTLBuffer> code = failed ? nil : [self newBufferWithLength:h->sass_size + h->data_size + 4096 options:0];
    if (code) {
        memcpy(code.contents, c + 1, h->sass_size + h->data_size);
        malachite_tag_program(code, h, function.name);
    }

    // the AIR, for build/airc to reproduce a failure with; with the trace on, every shader's (not an override)
    // (in /tmp, or for a sandboxed process that can't write there, its own temporary directory)
    // (GL programs are all __main: numbered)
    // (GL programs are all __main, and Unity's translated shaders all xlatMtlMain: numbered, and the code each one
    // compiled to logged, so kern.malachite_prof's addresses lead to their AIR)
    NSString *dump = nil, *file = function.name;
    bool numbered = gl || malachite_trace || malachite_dump;
    if (numbered) {
        static int programs;
        file = [NSString stringWithFormat:@"%@-%d", function.name, __sync_fetch_and_add(&programs, 1)];
    }
    if (failed || ((malachite_trace || malachite_dump) && !over)) {
        // not atomically: where the sandbox refuses the file, an atomic write leaves its temporary folder behind,
        // one for each failed compile
        dump = [NSString stringWithFormat:@"/tmp/malachite-%@.air", file];
        if (![air writeToFile:dump atomically:NO]) {
            dump = [NSTemporaryDirectory() stringByAppendingFormat:@"malachite-%@.air", file];
            [air writeToFile:dump atomically:NO];
        }
        for (NSUInteger i = 0; i < bitcode.count; i++)
            [bitcode[i] writeToFile:[dump stringByReplacingOccurrencesOfString:@".air"
                                                                    withString:[NSString stringWithFormat:@".linked%lu.air", i]]
                         atomically:NO];
        if (code)
            os_log(OS_LOG_DEFAULT, "Malachite dump: code %#llx from %{public}@", [(MTLIOAccelBuffer *)code gpuAddress], dump);
    }
    if (failed) {
        os_log(OS_LOG_DEFAULT, "Malachite: %{public}@: %{public}@ (AIR in %{public}@)", function.name, why, dump);
        if (error) {
            NSString *msg = [NSString stringWithFormat:@"%@: %@", function.name, why];
            *error = [NSError errorWithDomain:@"Malachite" code:1 userInfo:@{NSLocalizedDescriptionKey : msg}];
        }
    }
    return code;
}

// a tessellated draw's vertex stage (d NULL) or the tessellation init stage of d, generated into a GPU-mapped buffer as
// compile: does a function; nil, and *why, on failure
static id<MTLBuffer> malachite_tess_stage(id<MTLDevice> device, const struct airc_tess_desc *d, struct nvk_header *h,
                                      NSString **why)
{
    static struct airc_kernel k;
    os_unfair_lock_lock(&malachite_airc_lock);
    int failed = d ? airc_tess_init(d, &k) : airc_tess_vertex(&k);
    *h = k.h;
    id<MTLBuffer> code = failed ? nil : [device newBufferWithLength:h->sass_size + 4096 options:0];
    if (code)
        memcpy(code.contents, k.sass, h->sass_size);
    free(k.sass);
    if (failed)
        *why = @(k.error);
    os_unfair_lock_unlock(&malachite_airc_lock);
    return code;
}

// A compute pipeline's stage input as airc takes it, in out: NULL for one that describes no attribute, and NULL with
// *why for one airc can't fetch. MTLAttributeFormat has MTLVertexFormat's values.
static const struct airc_stage_in *malachite_stage_in(MTLStageInputOutputDescriptor *sd, struct airc_stage_in *out,
                                                  NSString **why)
{
    *out = (struct airc_stage_in){};
    bool any = false;
    for (unsigned loc = 0; sd && loc < 31; loc++) {
        MTLAttributeDescriptor *a = sd.attributes[loc];
        if (a.format == MTLAttributeFormatInvalid)
            continue;
        MTLBufferLayoutDescriptor *l = sd.layouts[a.bufferIndex];
        uint32_t f = malachite_vertex_format((MTLVertexFormat)a.format, false);
        if (!f) {
            *why = [NSString stringWithFormat:@"stage input attribute %u: format %lu", loc, (unsigned long)a.format];
            return NULL;
        }
        if (l.stride == MTLBufferLayoutStrideDynamic) {
            *why = [NSString stringWithFormat:@"stage input buffer %lu: dynamic stride", (unsigned long)a.bufferIndex];
            return NULL;
        }
        out->attrs[loc] = (struct airc_tess_attr){ f, (uint32_t)a.bufferIndex, (uint32_t)a.offset, (uint32_t)l.stride,
                                                   (uint32_t)l.stepFunction, (uint32_t)l.stepRate };
        any = true;
    }
    out->index_buffer = (uint32_t)sd.indexBufferIndex;
    out->index_size = sd.indexType == MTLIndexTypeUInt32 ? 4 : 2;
    return any ? out : NULL;
}

// a kernel with the functions its visible-function calls resolve to (MPS's stitched ones), and its stage input
- (id)computePipeline:(id<MTLFunction>)function
               linked:(MTLLinkedFunctions *)linked
              stageIn:(MTLStageInputOutputDescriptor *)stageIn
                error:(NSError **)error
{
    struct airc_stage_in si;
    NSString *why = nil;
    const struct airc_stage_in *layout = malachite_stage_in(stageIn, &si, &why);
    if (why) {
        NSString *msg = [NSString stringWithFormat:@"%@: %@", function.name, why];
        os_log(OS_LOG_DEFAULT, "Malachite: %{public}@", msg);
        if (error)
            *error = [NSError errorWithDomain:@"Malachite" code:1 userInfo:@{NSLocalizedDescriptionKey : msg}];
        return nil;
    }
    struct nvk_header h;
    id<MTLBuffer> code = [self compile:function linked:linked vertex:NULL stageIn:layout samples:1 outputs:NULL
                                header:&h
                                 error:error];
    if (!code)
        return nil;
    MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
    desc.computeFunction = function;
    MalachiteComputePipelineState *state = [[MalachiteComputePipelineState alloc] initWithDevice:self pipelineStateDescriptor:desc];
    state->_kernel = h;
    state->_code = code;
    state->_label = function.name;
    state->_cl = objc_getAssociatedObject(function, &malachite_cl_program_key) != nil;
    if (malachite_trace)
        os_log(OS_LOG_DEFAULT, "Malachite trace: kernel %{public}@ at %#llx, %u regs, %u bytes shared, %u args",
               function.name, [(MTLIOAccelBuffer *)code gpuAddress], h.reg_count, h.shared_size, h.nargs);
    return state;
}

- (id)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError **)error
{
    return [self computePipeline:function linked:nil stageIn:nil error:error];
}

// The other ways in to pipeline creation all come to the ones above. Reflection is only what OpenGL's renderer reads.
- (id)computePipeline:(id<MTLFunction>)function
               linked:(MTLLinkedFunctions *)linked
              stageIn:(MTLStageInputOutputDescriptor *)stageIn
           reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                error:(NSError **)error
{
    id state = [self computePipeline:function linked:linked stageIn:stageIn error:error];
    if (reflection)
        *reflection = nil;
    if (reflection && objc_getAssociatedObject(function, &malachite_arguments_key)) {  // a GL program's
        MalachiteComputeReflection *r = [MalachiteComputeReflection new];
        r.arguments = objc_getAssociatedObject(function, &malachite_arguments_key) ?: @[];
        r.pluginReturnData = objc_getAssociatedObject(function, &malachite_return_key);
        *reflection = (MTLComputePipelineReflection *)r;
    }
    return state;
}

- (id)newComputePipelineStateWithFunction:(id<MTLFunction>)function
                                  options:(MTLPipelineOption)options
                               reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                    error:(NSError **)error
{
    return [self computePipeline:function linked:nil stageIn:nil reflection:reflection error:error];
}

// a kernel's declared threadgroup size and the descriptor's must agree, as Apple's drivers have it
static id malachite_threads_asked(MalachiteComputePipelineState *state, MTLComputePipelineDescriptor *desc, NSError **error)
{
    NSUInteger ask = desc.maxTotalThreadsPerThreadgroup;
    if (!state || !ask)
        return state;
    if (state->_kernel.max_threads && state->_kernel.max_threads != ask) {
        if (error)
            *error = [NSError errorWithDomain:@"Malachite" code:1 userInfo:@{
                NSLocalizedDescriptionKey : [NSString stringWithFormat:@"Kernel specified max total threads per threadgroup (%u) "
                                                                       @"must match compute pipeline specified max total "
                                                                       @"threads per threadgroup (%lu)",
                                                                       state->_kernel.max_threads, (unsigned long)ask]
            }];
        return nil;
    }
    state->_asked = ask;
    return state;
}

- (id)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)desc error:(NSError **)error
{
    return malachite_threads_asked([self computePipeline:desc.computeFunction
                                                  linked:desc.linkedFunctions
                                                 stageIn:desc.stageInputDescriptor
                                                   error:error],
                               desc, error);
}

- (id)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)desc
                                    options:(MTLPipelineOption)options
                                 reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                      error:(NSError **)error
{
    return malachite_threads_asked([self computePipeline:desc.computeFunction
                                                  linked:desc.linkedFunctions
                                                 stageIn:desc.stageInputDescriptor
                                              reflection:reflection
                                                   error:error],
                               desc, error);
}

- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)function
                          completionHandler:(MTLNewComputePipelineStateCompletionHandler)handler
{
    NSError *err = nil;
    id ps = [self newComputePipelineStateWithFunction:function error:&err];
    handler(ps, err);
}

- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)desc
                                      options:(MTLPipelineOption)options
                            completionHandler:(MTLNewComputePipelineStateWithReflectionCompletionHandler)handler
{
    NSError *err = nil;
    MTLAutoreleasedComputePipelineReflection r = nil;
    id ps = [self newComputePipelineStateWithDescriptor:desc options:options reflection:&r error:&err];
    handler(ps, r, err);
}

- (id)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)desc
                                   options:(MTLPipelineOption)options
                                reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection
                                     error:(NSError **)error
{
    id state = [self newRenderPipelineStateWithDescriptor:desc error:error];
    if (reflection)
        *reflection = nil;
    if (reflection && (objc_getAssociatedObject(desc.vertexFunction, &malachite_arguments_key) ||
                       objc_getAssociatedObject(desc.fragmentFunction, &malachite_arguments_key))) {  // a GL program's
        MalachiteRenderReflection *r = [MalachiteRenderReflection new];
        r.vertexArguments = objc_getAssociatedObject(desc.vertexFunction, &malachite_arguments_key) ?: @[];
        r.fragmentArguments = objc_getAssociatedObject(desc.fragmentFunction, &malachite_arguments_key) ?: @[];
        r.vertexPluginReturnData = objc_getAssociatedObject(desc.vertexFunction, &malachite_return_key);
        r.fragmentPluginReturnData = objc_getAssociatedObject(desc.fragmentFunction, &malachite_return_key);
        *reflection = (MTLRenderPipelineReflection *)r;
    }
    return state;
}

- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)desc
                           completionHandler:(MTLNewRenderPipelineStateCompletionHandler)handler
{
    NSError *err = nil;
    id ps = [self newRenderPipelineStateWithDescriptor:desc error:&err];
    handler(ps, err);
}

- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)desc
                                     options:(MTLPipelineOption)options
                           completionHandler:(MTLNewRenderPipelineStateWithReflectionCompletionHandler)handler
{
    NSError *err = nil;
    MTLAutoreleasedRenderPipelineReflection r = nil;
    id ps = [self newRenderPipelineStateWithDescriptor:desc options:options reflection:&r error:&err];
    handler(ps, r, err);
}

// what runs for a pipeline without a fragment function (depth only, or no rasterization): nothing
static id<MTLFunction> malachite_no_fragment(id<MTLDevice> device)
{
    static id<MTLFunction> fn;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSError *err;
        fn = [[device newLibraryWithSource:@"fragment void malachite_no_fragment() {}" options:nil error:&err]
            newFunctionWithName:@"malachite_no_fragment"];
        if (!fn)
            os_log(OS_LOG_DEFAULT, "Malachite: empty fragment function: %{public}@", err.localizedDescription);
    });
    return fn;
}

- (id)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)desc error:(NSError **)error
{
    struct nvk_header vh, fh;
    struct airc_varyings varyings;
    id<MTLBuffer> vcode = [self compile:desc.vertexFunction
                                 linked:desc.vertexLinkedFunctions
                                 vertex:NULL
                                stageIn:NULL
                                samples:1
                                outputs:&varyings
                                 header:&vh
                                  error:error];
    id<MTLBuffer> fcode = vcode ? [self compile:desc.fragmentFunction ?: malachite_no_fragment(self)
                                         linked:desc.fragmentLinkedFunctions
                                         vertex:&varyings
                                        stageIn:NULL
                                        samples:desc.rasterSampleCount
                                        outputs:NULL
                                         header:&fh
                                          error:error]
                                : nil;
    if (!fcode)
        return nil;
    MalachiteRenderPipelineState *state = [[MalachiteRenderPipelineState alloc] initWithDevice:self pipelineStateDescriptor:desc];
    state->_vertex = vh;
    state->_fragment = fh;
    state->_vcode = vcode;
    state->_fcode = fcode;
    // for naming the shaders kern.malachite_prof lists by address
    os_log_info(OS_LOG_DEFAULT, "Malachite pipeline: vs %{public}@ %#llx ps %{public}@ %#llx", desc.vertexFunction.name,
                [(MTLIOAccelBuffer *)vcode gpuAddress], desc.fragmentFunction.name,
                [(MTLIOAccelBuffer *)fcode gpuAddress]);
    // the AIR as compiled for this pipeline, its functions' linked ones being the pipeline's
    if (malachite_rec_only) {
        objc_setAssociatedObject(state, &malachite_desc_key, [desc copy], OBJC_ASSOCIATION_RETAIN);
        id vair = objc_getAssociatedObject(desc.vertexFunction, &malachite_air_key);
        id fair = desc.fragmentFunction ? objc_getAssociatedObject(desc.fragmentFunction, &malachite_air_key) : nil;
        NSMutableDictionary *air = [NSMutableDictionary dictionary];
        if (vair)
            air[@"vertex"] = vair;
        if (fair)
            air[@"fragment"] = fair;
        objc_setAssociatedObject(state, &malachite_air_key, air, OBJC_ASSOCIATION_RETAIN);
    }

    // MTLBlendOperation and MTLBlendFactor in the 3D class's OpenGL-style encoding
    static const uint32_t op[] = { 0x8006, 0x800a, 0x800b, 0x8007, 0x8008 };
    static const uint32_t factor[] = { 0x4000, 0x4001, 0x4300, 0x4301, 0x4302, 0x4303, 0x4306, 0x4307,
                                       0x4304, 0x4305, 0x4308, 0xc001, 0xc002, 0xc003, 0xc004, 0xc900,
                                       0xc901, 0xc902, 0xc903 };
    for (unsigned i = 0; i < 8; i++) {
        MTLRenderPipelineColorAttachmentDescriptor *a = desc.colorAttachments[i];
        if (a.pixelFormat == MTLPixelFormatInvalid)
            continue;
        MTLColorWriteMask m = a.writeMask;
        uint32_t *b = state->_blend[i];
        state->_write_mask[i] = (m & MTLColorWriteMaskRed ? 1 : 0) | (m & MTLColorWriteMaskGreen ? 1 << 4 : 0) |
                                (m & MTLColorWriteMaskBlue ? 1 << 8 : 0) | (m & MTLColorWriteMaskAlpha ? 1 << 12 : 0);
        b[0] = a.blendingEnabled;
        b[1] = op[a.rgbBlendOperation];
        b[2] = factor[a.sourceRGBBlendFactor];
        b[3] = factor[a.destinationRGBBlendFactor];
        b[4] = op[a.alphaBlendOperation];
        b[5] = factor[a.sourceAlphaBlendFactor];
        b[6] = factor[a.destinationAlphaBlendFactor];
    }
    state->_alpha_control = (desc.alphaToCoverageEnabled ? 1u : 0) | (desc.alphaToOneEnabled ? 1u << 4 : 0);
    MTLRenderPipelineColorAttachmentDescriptor *ca = desc.colorAttachments[0];
    MTLColorWriteMask m = ca.writeMask;
    state->_label = [NSString stringWithFormat:@"%@/%@", desc.vertexFunction.name, desc.fragmentFunction.name];
    if (malachite_trace)
        os_log(OS_LOG_DEFAULT,
               "Malachite trace: pipeline %{public}@: formats %lu %lu %lu %lu depth %lu stencil %lu, samples %lu, "
               "blend %d (%lu %lu %lu / %lu %lu %lu), mask %#lx, alpha to coverage %d, vertex descriptor %d, "
               "topology class %lu",
               state->_label, (unsigned long)ca.pixelFormat, (unsigned long)desc.colorAttachments[1].pixelFormat,
               (unsigned long)desc.colorAttachments[2].pixelFormat, (unsigned long)desc.colorAttachments[3].pixelFormat,
               (unsigned long)desc.depthAttachmentPixelFormat, (unsigned long)desc.stencilAttachmentPixelFormat,
               (unsigned long)desc.rasterSampleCount, ca.blendingEnabled, (unsigned long)ca.rgbBlendOperation,
               (unsigned long)ca.sourceRGBBlendFactor, (unsigned long)ca.destinationRGBBlendFactor,
               (unsigned long)ca.alphaBlendOperation, (unsigned long)ca.sourceAlphaBlendFactor,
               (unsigned long)ca.destinationAlphaBlendFactor, (unsigned long)m, desc.alphaToCoverageEnabled,
               desc.vertexDescriptor != nil, (unsigned long)desc.inputPrimitiveTopology);

    if (vh.stage == NVK_TESS_EVAL)
        return [self tessellation:state descriptor:desc error:error];

    // [[stage_in]] attributes: attribute i reads a stream at its offset. Metal numbers vertex buffers to 30 and the
    // GPU has 16 streams, so a buffer past them (Roblox binds its vertices from 16) takes a stream no lower one needs
    MTLVertexDescriptor *vd = desc.vertexDescriptor;
    uint32_t unread = 0, taken = 0;
    int stream_of[31];
    memset(stream_of, -1, sizeof(stream_of));
    for (NSUInteger i = 0; vd && i < 31; i++) {
        MTLVertexAttributeDescriptor *a = vd.attributes[i];
        if (a.format != MTLVertexFormatInvalid && a.bufferIndex < 16) {
            stream_of[a.bufferIndex] = (int)a.bufferIndex;
            taken |= 1u << a.bufferIndex;
        }
    }
    for (NSUInteger i = 0; vd && i < 31; i++) {
        MTLVertexAttributeDescriptor *a = vd.attributes[i];
        if (a.format == MTLVertexFormatInvalid)
            continue;
        uint32_t f = malachite_vertex_format(a.format, vh.float_inputs >> i & 1);
        if (f && stream_of[a.bufferIndex] < 0 && taken != 0xffff) {
            stream_of[a.bufferIndex] = __builtin_ctz(~taken);
            taken |= 1u << stream_of[a.bufferIndex];
        }
        if (!f || stream_of[a.bufferIndex] < 0) {
            os_log(OS_LOG_DEFAULT, "MalachiteRender: vertex attribute %lu: format %lu, buffer %lu not supported",
                   (unsigned long)i, (unsigned long)a.format, (unsigned long)a.bufferIndex);
            unread |= 1u << i;
            continue;
        }
        int stream = stream_of[a.bufferIndex];
        state->_attributes[i] = (uint32_t)stream | (uint32_t)a.offset << 7 | f;
        MTLVertexBufferLayoutDescriptor *l = vd.layouts[a.bufferIndex];
        state->_stream_buffer[stream] = (uint8_t)a.bufferIndex;
        state->_strides[stream] =
            l.stride == MTLBufferLayoutStrideDynamic ? MALACHITE_DYNAMIC_STRIDE : (uint32_t)l.stride;
        // a constant buffer is a per-instance one that never steps
        state->_divisors[stream] = l.stepFunction == MTLVertexStepFunctionPerInstance ? (uint32_t)l.stepRate
                                   : l.stepFunction == MTLVertexStepFunctionConstant ? UINT32_MAX
                                                                                     : 0;
    }
    // an attribute left inactive faults the vertex fetch of a shader that reads it, so one we can't fetch reads zeros
    // from a stream of its own instead
    int s = 15;
    while (s >= 0 && (state->_strides[s] || state->_divisors[s]))
        s--;
    if (unread && s >= 0) {
        state->_zero_stream = s + 1;
        state->_divisors[s] = UINT32_MAX;
        for (unsigned i = 0; i < 31; i++)
            if (unread >> i & 1)
                state->_attributes[i] = (uint32_t)s | (1u << 21) | (7u << 27);
    }
    return state;
}

// A post-tessellation vertex function's pipeline: the tessellator as the descriptor sets it up, and the stages
// generated to run before the function, the tessellation init stage fetching the control points and per-patch data
// the vertex descriptor lays out. state, or nil (and *error).
- (id)tessellation:(MalachiteRenderPipelineState *)state
        descriptor:(MTLRenderPipelineDescriptor *)desc
             error:(NSError **)error
{
    const struct nvk_header *vh = &state->_vertex;
    struct airc_tess_desc td = {};
    td.domain = vh->patch_domain;
    td.point_inputs = vh->point_inputs;
    td.patch_inputs = vh->patch_inputs;
    // without control points read, a thread a patch does; without a count, the draw gives it
    td.points = vh->patch_points ? vh->patch_points : !vh->point_inputs;
    MTLTessellationControlPointIndexType it = desc.tessellationControlPointIndexType;
    td.index_size = it == MTLTessellationControlPointIndexTypeUInt32   ? 4
                    : it == MTLTessellationControlPointIndexTypeUInt16 ? 2
                                                                       : 0;
    td.factor_step = (uint32_t)desc.tessellationFactorStepFunction;
    td.pow2 = desc.tessellationPartitionMode == MTLTessellationPartitionModePow2;
    // fractional odd factors go up to the odd number below the maximum, which Metal has even; the tessellator would
    // otherwise split the segments between it and the next odd one up
    td.max_factor = (float)desc.maxTessellationFactor;
    if (desc.tessellationPartitionMode == MTLTessellationPartitionModeFractionalOdd)
        td.max_factor -= 1;
    NSString *why = nil;
    MTLVertexDescriptor *vd = desc.vertexDescriptor;
    for (unsigned loc = 0; loc < 32 && !why; loc++) {
        if (!((td.point_inputs | td.patch_inputs) >> loc & 1))
            continue;
        if (loc == 31) {  // a vertex descriptor has 31 attributes, and asking for another aborts
            why = @"attribute 31";
            break;
        }
        MTLVertexAttributeDescriptor *a = vd.attributes[loc];
        MTLVertexBufferLayoutDescriptor *l = vd.layouts[a.bufferIndex];
        uint32_t f = malachite_vertex_format(a.format, false);
        if (!f || a.bufferIndex >= AIRC_TESS_BUFFERS || l.stepFunction == MTLVertexStepFunctionPerVertex)
            why = [NSString stringWithFormat:@"attribute %u: format %lu, buffer %lu, step function %lu", loc,
                                             (unsigned long)a.format, (unsigned long)a.bufferIndex,
                                             (unsigned long)l.stepFunction];
        td.attrs[loc] = (struct airc_tess_attr){ f, (uint32_t)a.bufferIndex, (uint32_t)a.offset, (uint32_t)l.stride,
                                                 (uint32_t)l.stepFunction, (uint32_t)l.stepRate };
    }
    state->_td = td;
    if (!why)
        state->_tvcode = malachite_tess_stage(self, NULL, &state->_tv, &why);
    // a function without a count is checked with its domain's usual one, which its draws will most likely give
    uint32_t regs;
    if (!why)
        [state tessInit:(vh->patch_domain ? 4 : 3) device:self regs:&regs why:&why];
    if (why) {
        NSString *msg = [NSString stringWithFormat:@"%@: %@", desc.vertexFunction.name, why];
        os_log(OS_LOG_DEFAULT, "Malachite: %{public}@", msg);
        if (error)
            *error = [NSError errorWithDomain:@"Malachite" code:1 userInfo:@{NSLocalizedDescriptionKey : msg}];
        return nil;
    }
    // MTLTessellationPartitionMode as SPACING: pow2 is integer spacing with the factors rounded up by the init stage.
    // The viewport flips y, so Metal's clockwise output is the class's TRIANGLES_CCW.
    static const uint32_t spacing[] = { 0, 0, 1, 2 };
    state->_tess = (vh->patch_domain ? 2 : 1) | spacing[desc.tessellationPartitionMode & 3] << 4 |
                   (desc.tessellationOutputWindingOrder == MTLWindingClockwise ? 3u : 2u) << 8;
    state->_tess_scaled = desc.tessellationFactorScaleEnabled;
    return state;
}

// Metal's designated initializer for an IOAccel device. super sets up the ~54 KB of device state from the port.
- (instancetype)initWithAcceleratorPort:(unsigned int)port
{
    os_log(OS_LOG_DEFAULT, "MalachiteDevice: initWithAcceleratorPort:0x%x", port);
    self = [super initWithAcceleratorPort:port];
    os_log(OS_LOG_DEFAULT, "MalachiteDevice: super init -> %p", (__bridge void *)self);
    // the file names the process to trace, or is empty for all of them
    NSString *who = malachite_trusted_file("/var/tmp/malachite-trace", false)
                        ? [NSString stringWithContentsOfFile:@"/var/tmp/malachite-trace" encoding:NSUTF8StringEncoding error:nil]
                        : nil;
    who = [who stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    // "name +snap" (or MALACHITE_TRACE=snap) adds the snapshots
    NSArray<NSString *> *words = [who componentsSeparatedByString:@" "];
    NSString *name = words.count ? words[0] : @"";
    malachite_trace = getenv("MALACHITE_TRACE") || (who && (!name.length || [name isEqualToString:@(getprogname())] ||
                                                  [name isEqualToString:@"+snap"]));
    if (malachite_trace && [words containsObject:@"+dump"] && !getenv("MALACHITE_TRACE")) {
        malachite_trace = false;
        malachite_dump = true;
    }
    for (NSString *w in words)
        if (malachite_trace && [w hasPrefix:@"+record="])
            malachite_rec_only = [[w substringFromIndex:8] componentsSeparatedByString:@"|"];
        else if (malachite_trace && [w hasPrefix:@"+recskip="])
            malachite_rec_skip = [w substringFromIndex:9].intValue;
    malachite_snap = malachite_trace && ((getenv("MALACHITE_TRACE") && !strcmp(getenv("MALACHITE_TRACE"), "snap")) ||
                                 [words containsObject:@"+snap"]);
    for (NSString *w in words)
        if (malachite_trace && [w hasPrefix:@"+snap="]) {
            malachite_snap = true;
            malachite_snap_only = [[w substringFromIndex:6] componentsSeparatedByString:@"|"];
        }
    malachite_snap_gate = malachite_trace && [words containsObject:@"+snapgate"];
    for (NSString *w in words)
        if (malachite_trace && [w hasPrefix:@"+snaprect="]) {
            NSArray<NSString *> *v = [[w substringFromIndex:10] componentsSeparatedByString:@","];
            for (NSUInteger k = 0; k < 4 && k < v.count; k++)
                malachite_snap_rect[k] = (uint32_t)v[k].intValue;
            malachite_snap |= !malachite_rec_only;  // with the recorder the rect only picks the draws it records
        }
    malachite_snap_gated = malachite_snap_gate && (malachite_snap || !malachite_rec_only);
    return self;
}

// the name GSP-RM gave the GPU, which Gpu publishes
// what Apple's frameworks key vendor behaviour on (the base says Apple): OpenGL's Metal renderer refuses a
// device it can't place
- (NSString *)vendorName
{
    return @"NVIDIA";
}

// What Metal's loader picks a binary archive's slice by (RenderBox loads precompiled shader archives for icons and
// effects): with none it aborts. A GPU type no archive has a slice for, so every pipeline is compiled from its AIR.
- (id)targetDeviceArchitecture
{
    static id arch;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        arch = [NSClassFromString(@"MTLTargetDeviceArchitecture") new];
        ((void (*)(id, SEL, uint32_t))objc_msgSend)(arch, NSSelectorFromString(@"setCpuType:"), 0x0100004e);  // 'N'
        ((void (*)(id, SEL, uint32_t))objc_msgSend)(arch, NSSelectorFromString(@"setSubType:"), 1);
        ((void (*)(id, SEL, uint32_t))objc_msgSend)(arch, NSSelectorFromString(@"setVersion:"), 1);
        ((void (*)(id, SEL, uint64_t))objc_msgSend)(arch, NSSelectorFromString(@"setDriverVersion:"), 1);
    });
    return arch;
}

- (NSString *)name
{
    static NSString *name;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        io_service_t rm = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("MalachiteGpu"));
        CFTypeRef n = rm ? IORegistryEntryCreateCFProperty(rm, CFSTR("MalachiteGPUName"), kCFAllocatorDefault, 0) : NULL;
        name = n && CFGetTypeID(n) == CFStringGetTypeID() ? (NSString *)CFBridgingRelease(n) : @"NVIDIA GPU";
        if (n && CFGetTypeID(n) != CFStringGetTypeID())
            CFRelease(n);
        if (rm)
            IOObjectRelease(rm);
    });
    return name;
}

// OpenCL's compute units (AppleMetalOpenGLRenderer asks for this; without, 8)
- (NSUInteger)gpuCoreCount
{
    static NSUInteger count;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        io_service_t rm = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("MalachiteGpu"));
        CFTypeRef n = rm ? IORegistryEntryCreateCFProperty(rm, CFSTR("MalachiteSMCount"), kCFAllocatorDefault, 0) : NULL;
        if (n && CFGetTypeID(n) == CFNumberGetTypeID())
            CFNumberGetValue((CFNumberRef)n, kCFNumberNSIntegerType, &count);
        if (n)
            CFRelease(n);
        if (rm)
            IOObjectRelease(rm);
    });
    return count ?: 8;
}

- (BOOL)supportsTextureSampleCount:(NSUInteger)count
{
    return count == 1 || count == 2 || count == 4 || count == 8;
}

- (BOOL)supportsSampleCount:(NSUInteger)count
{
    return [self supportsTextureSampleCount:count];
}

// The GPU drives displays even before Framebuffer's head is lit. WindowServer asks once, at startup (a minute before the
// head comes up), and composites in software for good if every device says it's headless.
- (BOOL)isHeadless
{
    return NO;
}

// Depth24Unorm_Stencil8 is laid out as 32-bit float depth here, so apps are told to take Depth32Float_Stencil8
- (BOOL)isDepth24Stencil8PixelFormatSupported
{
    return NO;
}

// argument buffers hold textures and samplers as descriptor heap indices (their gpuResourceID), which shaders use
// as handles: any texture of the heap's kept entries, any sampler of its
- (MTLArgumentBuffersTier)argumentBuffersSupport
{
    return MTLArgumentBuffersTier2;
}

- (NSUInteger)maxArgumentBufferSamplerCount
{
    return MALACHITE_HEAP_SAMPLERS / 2;
}

- (NSUInteger)maxThreadgroupMemoryLength
{
    return 48 * 1024;  // shared memory a CTA gets without asking for the larger carve-out
}

// the base profile's 16 KB otherwise, which OpenCL reports as the local memory size (OpenCL 1.2 wants 32 KB)
- (NSUInteger)maxComputeThreadgroupMemory
{
    return [self maxThreadgroupMemoryLength];
}

- (MTLSize)maxThreadsPerThreadgroup
{
    return MTLSizeMake(1024, 1024, 64);
}

- (NSUInteger)minimumTextureBufferAlignmentForPixelFormat:(MTLPixelFormat)format
{
    return 256;
}

// no sparse textures
- (MTLSize)sparseTileSizeWithTextureType:(MTLTextureType)type
                             pixelFormat:(MTLPixelFormat)format
                             sampleCount:(NSUInteger)samples
{
    return MTLSizeMake(0, 0, 0);
}

- (MTLSize)sparseTileSizeWithTextureType:(MTLTextureType)type
                             pixelFormat:(MTLPixelFormat)format
                             sampleCount:(NSUInteger)samples
                          sparsePageSize:(MTLSparsePageSize)pageSize
{
    return MTLSizeMake(0, 0, 0);
}

// the base answers yes at draw, dispatch and blit boundaries, but has no counter set to sample
- (BOOL)supportsCounterSampling:(MTLCounterSamplingPoint)point
{
    return NO;
}

- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                      iosurface:(IOSurfaceRef)surface
                          plane:(NSUInteger)plane
                          slice:(NSUInteger)slice
{
    return slice ? nil : [self newTextureWithDescriptor:descriptor iosurface:surface plane:plane];
}

@end
