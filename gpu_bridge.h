// callPlatformFunction contract between Accelerator (IOAccelerator kext) and Gpu (owns the GPU/GSP-RM). Accelerator
// finds Gpu by IOServiceMatching("MalachiteGpu") and calls by name (plain strings); Gpu runs them on its copy channel
// + vmm. Scalar arguments are passed as (void*)(uintptr_t).
#pragma once

// Each task has its own GPU context (address space + channels). Calls taking a context take its handle (from
// NVGPU_FN_CTX_OPEN) as the last argument, 0 for the kernel's. VAs are unique across contexts.
// p1 = uintptr_t* out new context handle.
#define NVGPU_FN_CTX_OPEN "NVGpuCtxOpen"
// p1 = context handle: waits for its work, then frees everything it holds.
#define NVGPU_FN_CTX_CLOSE "NVGpuCtxClose"

// p1 = IOMemoryDescriptor*, p2 = uint64_t* out GPU VA, p3 = context. Maps its pages; kIOReturnSuccess on success.
#define NVGPU_FN_MAP "NVGpuMap"

// p1 = GPU VA, p2 = size. Unmaps a prior NVGpuMap or NVGpuMapVRAM.
#define NVGPU_FN_UNMAP "NVGpuUnmap"

// VRAM only the GPU touches (Metal private storage). p1 = size, p2 = uint64_t* out VRAM address (0 if none).
#define NVGPU_FN_VRAM_ALLOC "NVGpuVRAMAlloc"

// p1 = VRAM address, p2 = size, from NVGpuVRAMAlloc
#define NVGPU_FN_VRAM_FREE "NVGpuVRAMFree"

// p1 = VRAM address, p2 = size, p3 = uint64_t* out GPU VA, p4 = context
#define NVGPU_FN_MAP_VRAM "NVGpuMapVRAM"

// p1 = src GPU VA, p2 = dst GPU VA, p3 = size, p4 = context. Linear copy on the copy engine; blocks until done.
#define NVGPU_FN_COPY "NVGpuCopy"

// As NVGpuCopy, for CPU writes into a Managed buffer. Doesn't wait for draws in flight (Metal says they don't read a
// changed range); that wait would be the app's whole frame time.
#define NVGPU_FN_UPLOAD "NVGpuUpload"

// rm::queue_copy2d: a rectangle of rows between pitched surfaces on the copy engine, blocking until done.

struct nvgpu_copy2d {
    unsigned long long src, dst;  // GPU VAs of the first byte of the first row
    unsigned int src_pitch, dst_pitch;
    unsigned int bytes, lines;  // per row, rows
    // block-linear side (one-GOB blocks): pitch is its width in bytes, rows its height; the copy starts at origin
    // (x bytes | y rows << 16)
    unsigned int src_blocklinear, dst_blocklinear, src_rows, dst_rows;
    unsigned int src_origin, dst_origin;
    // SET_REMAP_COMPONENTS to move components rather than bytes, 0 for none; bytes and origins stay in bytes and must
    // be whole elements
    unsigned int remap;
};

// p1 = struct nvgpu_present*. Starts copying a BGRA frame onto the scanout surface, clipped to the mode, and sets
// fence; waits for the previous frame's copy first.
#define NVGPU_FN_PRESENT "NVGpuPresent"

struct nvgpu_present {
    void *memory;              // IOMemoryDescriptor* holding the frame
    unsigned long long offset;  // of the first row in it
    unsigned int width, height, pitch;
    unsigned int screen;  // the framebuffer's (its nub's SURFACE_SCREEN)
    unsigned int fence;   // out: hand it to NVGPU_FN_FENCE
};

// p1 = void (*)(void *) or null, p2 = its argument, p3 = screen. Called on each of the screen's vblanks from Gpu's
// interrupt handling until registered as null.
#define NVGPU_FN_PIPE_VBL "NVGpuPipeVBL"

// p1 = a present's fence. kIOReturnSuccess once its copy has landed (frame memory free again), kIOReturnBusy before.
#define NVGPU_FN_FENCE "NVGpuFence"

// p1 = const unsigned short*: NVGPU_GAMMA_ENTRIES RGB triples, entry i for input i / (NVGPU_GAMMA_ENTRIES - 1),
// 0x10000 = 1.0 (capped at 0xffff); p2 = screen. Becomes the screen's output LUT from the next frame.
#define NVGPU_FN_GAMMA "NVGpuGamma"
#define NVGPU_GAMMA_ENTRIES 1025

// p1 = dst GPU VA, p2 = size, p3 = byte value, p4 = context. Fills on the copy engine (4-byte pattern broadcast).
#define NVGPU_FN_FILL "NVGpuFill"

// rm::queue_dispatch: one compute grid on the context's channel. prog_va must already be NVGpuMap'd. params go to
// const bank 0 at 0x160, where the kernel reads its arguments at the offsets airc assigned.

#define NVGPU_PARAM_BYTES 160  // 0x160..0x200
// a kernel's: 0x160..0x400, with handles at 0x200..0x280 left 0 to fill later (Metal binds 31 buffers, 20 fit below
// 0x200)
#define NVGPU_COMPUTE_PARAM_BYTES 0x2a0
#define NVGPU_MAX_TEXTURES 16
#define NVGPU_HANDLES 32  // at 0x200: texture slots' header indices, then sampler slots' sampler indices << 20

// With a descriptor heap (the process's headers and samplers, named by index from argument buffers), the launch uses
// it as its pools and shaders read each bound slot's index from handles in their const bank. Without, slots 0..n-1.
struct nvgpu_heap {
    unsigned long long headers, samplers;  // GPU VAs, 0 for none
    unsigned int nheaders, nsamplers;
    unsigned int handles[NVGPU_HANDLES];
};

// Headers and samplers go into pool slots 0..n-1; a kernel names a texture by header index | sampler index << 20,
// passed as a param.
struct nvgpu_dispatch {
    unsigned int headers[NVGPU_MAX_TEXTURES][8];   // TIC entries (Maxwell+ TEXHEAD, 32 bytes)
    unsigned int samplers[NVGPU_MAX_TEXTURES][8];  // TSC entries (TEXSAMP, 32 bytes)
    unsigned int nheaders, nsamplers;
    unsigned long long prog_va;
    unsigned char params[NVGPU_COMPUTE_PARAM_BYTES];
    unsigned int params_size;
    unsigned int grid_x, grid_y, grid_z;     // CTAs in the grid
    unsigned int block_x, block_y, block_z;  // threads per CTA
    // part of a larger (non-uniform) dispatch: its first thread and threadgroup, the whole dispatch's threadgroups
    // and threads; all 0 for a standalone dispatch
    unsigned int origin[3], group_origin[3], groups[3], threads[3];
    unsigned int reg_count;  // registers per thread (QMD REGISTER_COUNT)
    unsigned int shared_size;  // threadgroup memory per CTA, bytes
    unsigned int barriers;     // named barriers the kernel uses (QMD BARRIER_COUNT)
    unsigned int local_size;   // local memory per thread, bytes
    struct nvgpu_heap heap;
    // GPU VA of MTLDispatchThreadgroupsIndirectArguments, read for the grid once prior work is done; 0 for none
    unsigned long long indirect;
};

// Left by the plugin 256-aligned after a program's code and constant data, so a fault log can name the function and
// its bound buffers
#define NVGPU_PROGRAM_TAG 0x7f3a5c9e2b6d41c8ull
#define NVGPU_PROGRAM_TAG_BUFFERS 45
struct nvgpu_program_tag {
    unsigned long long magic;
    char name[64];
    unsigned int nbuffers;
    struct {
        unsigned char index, pad;
        unsigned short cbuf_off;
    } buffers[NVGPU_PROGRAM_TAG_BUFFERS];
};

// rm::queue_draw: optional colour clear, then the draw on the context's channel. Each stage's code (SPH first) must
// be NVGpuMap'd; its params go to its own const bank 0 at 0x160.

// p1 = context: waits for its draws so far. Draws don't wait for each other; a command buffer's end is where its
// memory is read back.
#define NVGPU_FN_SYNC "NVGpuSync"
// p1 = context: one of its command buffers starts
#define NVGPU_FN_CB_BEGIN "NVGpuCommandBufferBegin"

// p1 = unsigned int* out, p2 = context: the fence of its last draw not yet waited for, 0 if all done. For
// NVGPU_FN_DRAW_DONE.
#define NVGPU_FN_DRAW_FENCE "NVGpuDrawFence"
// p1 = draw fence, p2 = its context. kIOReturnSuccess once the draws up to it are done, kIOReturnBusy before.
#define NVGPU_FN_DRAW_DONE "NVGpuDrawDone"
// p1 = context, *(bool *)p2 = the task is an app, not WindowServer. kIOReturnSuccess when its command buffers complete
// as their draws do instead of waiting in submit (which holds the next frame up a refresh or more). OpenGL queues
// still wait (MALACHITE_CMD_OPENGL).
#define NVGPU_FN_ASYNC "NVGpuAsync"

// a colour target beside nvgpu_draw's target 0; address 0 for none, as target 0's can be
struct nvgpu_target {
    unsigned long long addr;
    unsigned int pitch, format, blocklinear, clear, write_mask;
    unsigned int blend[7];
    float clear_rgba[4];
    unsigned int width, height;  // as target_width and target_height
};

struct nvgpu_draw {
    unsigned long long target;  // colour target: address and pitch 128-byte aligned
    unsigned int pitch, width, height;  // width and height: the pass's, which every target has
    unsigned int target_blocklinear;  // block-linear in one-GOB blocks (required with depth)
    // the target's own size, which may exceed the pass's (attachments of different sizes); 0 for the pass's
    unsigned int target_width, target_height;
    unsigned int format;  // SET_COLOR_TARGET_FORMAT value (0xd5 RGBA8, 0xcf BGRA8)
    unsigned int clear;   // clear the target to clear_rgba first
    float clear_rgba[4];
    unsigned long long vs, ps;
    unsigned int vs_regs, ps_regs;
    unsigned char vs_params[NVGPU_PARAM_BYTES], ps_params[NVGPU_PARAM_BYTES];
    unsigned int vs_params_size, ps_params_size;
    unsigned int primitive;  // topology: 0 points, 1 lines, 3 line strip, 4 triangles, 5 triangle strip
    unsigned int first, count;  // vertices, or indices when index is set
    unsigned int instances;
    int base_vertex;
    unsigned int base_instance;
    unsigned long long index;  // index buffer address, 0 for none
    unsigned int index_size, index_format;  // bytes from index; 0 u8, 1 u16, 2 u32
    float viewport[6];    // x, y, width, height, near, far
    unsigned int scissor[4];  // x, y, width, height
    // depth/stencil target: block-linear, one GOB per block, mapped with PTE kind zt_kind; 0 for none
    unsigned long long zt;
    unsigned int zt_compress;  // its buffer holds nothing else, so may be mapped compressed
    unsigned int pass_first;  // first draw of a render pass
    unsigned int opengl;      // from Apple's GL renderer
    unsigned int zt_format, zt_width, zt_height, zt_size, zt_kind;  // SET_ZT_FORMAT, row stride in pixels, rows
    unsigned int samples;  // per pixel (0 or 1 for one); the sizes above are then in samples
    unsigned int clear_zs;  // clear depth (bit 0) to clear_depth, stencil (bit 1) to clear_stencil, before drawing
    float clear_depth;
    unsigned int clear_stencil;
    unsigned int depth_test, depth_write, depth_func;  // SET_DEPTH_FUNC value
    unsigned int stencil_test;
    unsigned int stencil[2][4];  // front, back: SET_STENCIL_OP_FAIL, _OP_ZFAIL, _OP_ZPASS, _FUNC values
    unsigned int stencil_ref;  // front's in bits 0-7, back's in bits 8-15
    unsigned int stencil_read_mask[2], stencil_write_mask[2];
    unsigned int headers[NVGPU_MAX_TEXTURES][8];  // fragment stage's textures and samplers, as for a dispatch
    unsigned int samplers[NVGPU_MAX_TEXTURES][8];
    unsigned int nheaders, nsamplers;
    unsigned int write_mask;  // SET_CT_WRITE value: R bit 0, G bit 4, B bit 8, A bit 12
    unsigned int blend[7];    // enable, colour op/src/dst, alpha op/src/dst (SET_BLEND_PER_TARGET_* values)
    struct nvgpu_target more[7];  // colour targets 1..7 (multiple render targets), the size of target 0
    // [[stage_in]] attributes as SET_VERTEX_ATTRIBUTE_A values (0 unused), read from streams (address 0 unused)
    unsigned int attributes[32];
    struct nvgpu_stream {
        unsigned long long addr;
        unsigned int stride, size;
        unsigned int divisor;  // 0 per vertex, n to step every n instances
    } streams[16];
    unsigned int cull_face;  // OGL_SET_CULL_FACE value, 0 to cull nothing
    unsigned int front_ccw;  // counter-clockwise triangles face the front, not Metal's default clockwise ones
    // OpenGL-only state (set by Apple's GL renderer): 0 in each is Metal's behaviour
    float point_size;  // of points whose vertex shader writes none (0 for 1)
    float line_width;  // 0 for 1
    unsigned int provoking_last;  // flat attributes from a primitive's last vertex rather than its first
    unsigned int restart;  // indexed strips restart: 0 at the largest index (Metal), 1 never, 2 at restart_index
    unsigned int restart_index;
    unsigned int window_coords;  // positions are already in pixels: no viewport transform
    unsigned int depth_clamp;  // depth outside near..far clamped rather than clipped
    unsigned int fill_lines;   // triangles drawn as their edges
    float depth_bias[3];       // constant, slope scale, clamp; all 0 for none
    float blend_const[4];      // the blend colour
    struct nvgpu_heap heap;    // its handles are the fragment stage's
    unsigned int vs_textured;  // the vertex stage has its own textures, with these handles into the heap
    unsigned int vs_handles[NVGPU_HANDLES];
    // 64-bit visibility result word counting samples that pass depth and stencil, 0 for none; accumulates from the
    // previous such draw unless visibility_reset
    unsigned long long visibility;
    unsigned int visibility_reset;
    unsigned int serialize;  // earlier draws finish first: it reads its own targets, or follows a barrier
    unsigned int local_size;  // max local memory bytes per thread of its shaders
    // GPU VA of MTLDrawPrimitivesIndirectArguments (MTLDrawIndexedPrimitivesIndirectArguments if indexed), read for
    // first, count, instances and base_vertex once prior work is done; 0 for none
    unsigned long long indirect;
    // tessellated draw (primitive 0xe, a vertex per patch) when tess, its SET_TESSELLATION_PARAMETERS value, is set:
    // ti is the tessellation init stage (params in const bank 0 at 0x160), ts the tessellation stage, reading vs_params
    unsigned int tess;
    unsigned long long ti, ts;
    unsigned int ti_regs, ts_regs;
    unsigned char ti_params[NVGPU_PARAM_BYTES];
    unsigned int ti_params_size;
    // SET_ANTI_ALIAS_ALPHA_CONTROL: alpha to coverage (bit 0), alpha to one (bit 4)
    unsigned int alpha_control;
    unsigned int per_sample;  // the fragment stage runs per sample, not per pixel
    unsigned int ps_depth;  // the fragment stage writes depth
    // layered targets, layers deep (0 or 1: not layered; 512 max): colour target j's layers are layer_pitch[j] bytes
    // apart, depth/stencil's zt_layer_pitch; with shader_layer the vertex stage picks the layer
    unsigned int layers, layer_pitch[8], zt_layer_pitch;
    // layers each target has from the attached one on, capped at layers (a layered pass's attachments may have
    // fewer); 0 for layers
    unsigned int target_layers[8], zt_layers;
    unsigned int shader_layer;
    // viewports and scissors 1..nviewports - 1, beside viewport and scissor 0, for a vertex stage picking one
    unsigned int nviewports;
    float viewports[15][6];
    unsigned int scissors[15][4];
};

// p1 = struct nvgpu_stats*: the GPU's performance statistics from what Gpu already knows, without asking GSP-RM.
// kIOReturnBusy when a launch holds Gpu's lock: skip that tick.
#define NVGPU_FN_STATS "NVGpuStats"

struct nvgpu_stats {
    unsigned int temp_c;  // core temperature, 0 when the sensor gives no reading
    unsigned int reserved;
    unsigned long long vram_total, vram_used;
    unsigned long long sys_mapped;  // system memory mapped into GPU address spaces
    unsigned long long busy_us;     // time any task's graphics work was outstanding (overlap counted once)
    unsigned long long now_us;      // the clock busy_us is measured against
    unsigned long long recoveries;  // channels rebuilt after a fault
    unsigned long long dma_bytes;   // DMA memory the driver holds: climbing with every fault means a leak
};

// p1 = context handle, p2 = uint64_t* out: microseconds its graphics work kept the GPU busy since the last call.
// Accelerator charges them to the submitting thread's task (ml_gpu_stat_update), which feeds Activity Monitor's GPU
// Time and % GPU. Lockless, so it never waits on a launch.
#define NVGPU_FN_CTX_BUSY "NVGpuCtxBusy"
