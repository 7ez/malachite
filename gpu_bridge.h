// callPlatformFunction contract between Accelerator (IOAccelerator kext) and Gpu (owns the GPU/GSP-RM).
// Accelerator finds Gpu via IOServiceMatching("MalachiteGpu") and calls these by name; Gpu compares with isEqualTo and runs
// the op on its persistent copy channel + vmm. Names are plain strings (callPlatformFunction's const char* form).
#pragma once

// Every task has a GPU context of its own: an address space, and channels its work runs in. The calls below
// taking a context take its handle (from NVGPU_FN_CTX_OPEN) in their last argument, 0 for the kernel's. Each
// VA names one mapping, whichever context it's in.
// p1 = uintptr_t* that receives a new context's handle.
#define NVGPU_FN_CTX_OPEN "NVGpuCtxOpen"
// p1 = (void*)(uintptr_t) a context's handle: its work waited for and everything it holds freed.
#define NVGPU_FN_CTX_CLOSE "NVGpuCtxClose"

// p1 = IOMemoryDescriptor* (a buffer's backing pages), p2 = uint64_t* out GPU virtual address, p3 = context.
// Maps the descriptor's pages into the context's address space and returns the VA. kIOReturnSuccess on success.
#define NVGPU_FN_MAP "NVGpuMap"

// p1 = (void*)(uintptr_t) GPU VA, p2 = (void*)(uintptr_t) size. Unmaps a prior NVGpuMap or NVGpuMapVRAM.
#define NVGPU_FN_UNMAP "NVGpuUnmap"

// VRAM for memory only the GPU touches (Metal's private storage). p1 = (void*)(uintptr_t) size, p2 = uint64_t* that
// receives the VRAM address (0 when there is none to give).
#define NVGPU_FN_VRAM_ALLOC "NVGpuVRAMAlloc"

// p1 = (void*)(uintptr_t) VRAM address, p2 = (void*)(uintptr_t) size, from NVGpuVRAMAlloc
#define NVGPU_FN_VRAM_FREE "NVGpuVRAMFree"

// p1 = (void*)(uintptr_t) VRAM address, p2 = (void*)(uintptr_t) size, p3 = uint64_t* that receives the GPU VA it
// is mapped at, p4 = context
#define NVGPU_FN_MAP_VRAM "NVGpuMapVRAM"

// p1 = (void*)(uintptr_t) src GPU VA, p2 = (void*)(uintptr_t) dst GPU VA, p3 = (void*)(uintptr_t) size, p4 = context.
// Runs one linear buffer->buffer copy on the copy engine and blocks until it completes.
#define NVGPU_FN_COPY "NVGpuCopy"

// As NVGpuCopy, for what the CPU wrote into a Managed buffer: draws in flight don't read a range the app has changed,
// which is Metal's rule, so it doesn't wait for them. Behind a game's frames that wait is the game's frame time.
#define NVGPU_FN_UPLOAD "NVGpuUpload"

// rm::queue_copy2d copies a rectangle of rows between pitched surfaces on the copy engine, blocking until it completes.

struct nvgpu_copy2d {
    unsigned long long src, dst;  // GPU VAs of the first byte of the first row
    unsigned int src_pitch, dst_pitch;
    unsigned int bytes, lines;  // per row, rows
    // a block-linear side (one-GOB blocks) instead: its pitch is its width in bytes, rows its height, and the copy
    // starts at its origin (x bytes | y rows << 16) from the address
    unsigned int src_blocklinear, dst_blocklinear, src_rows, dst_rows;
    unsigned int src_origin, dst_origin;
    // SET_REMAP_COMPONENTS for a copy that moves components of each element rather than bytes, 0 for none: bytes
    // and origins stay in bytes, and must be whole elements
    unsigned int remap;
};

// p1 = struct nvgpu_present*. Starts copying a BGRA frame from client memory onto the surface the display scans
// out, clipped to the mode, and sets fence; the previous frame's copy is waited for first.
#define NVGPU_FN_PRESENT "NVGpuPresent"

struct nvgpu_present {
    void *memory;              // IOMemoryDescriptor* holding the frame
    unsigned long long offset;  // of the first row in it
    unsigned int width, height, pitch;
    unsigned int screen;  // the framebuffer's (its nub's SURFACE_SCREEN)
    unsigned int fence;   // out: hand it to NVGPU_FN_FENCE
};

// p1 = void (*)(void *) or null, p2 = its argument, p3 = (void*)(uintptr_t) screen. The function is called on each
// of the screen's vblanks, from Gpu's interrupt handling, until it is registered as null.
#define NVGPU_FN_PIPE_VBL "NVGpuPipeVBL"

// p1 = (void*)(uintptr_t) a present's fence. kIOReturnSuccess once its copy has landed (the frame's memory is free
// again), kIOReturnBusy before.
#define NVGPU_FN_FENCE "NVGpuFence"

// p1 = const unsigned short* holding NVGPU_GAMMA_ENTRIES red, green, blue triples, entry i for input
// i / (NVGPU_GAMMA_ENTRIES - 1), 0x10000 standing for 1.0 (capped at 0xffff); p2 = (void*)(uintptr_t) the screen.
// The screen's output LUT from the next frame on.
#define NVGPU_FN_GAMMA "NVGpuGamma"
#define NVGPU_GAMMA_ENTRIES 1025

// p1 = (void*)(uintptr_t) dst GPU VA, p2 = (void*)(uintptr_t) size, p3 = (void*)(uintptr_t) byte value,
// p4 = context.
// Fills dst with the repeating byte on the copy engine (broadcast copy of a 4-byte pattern).
#define NVGPU_FN_FILL "NVGpuFill"

// rm::queue_dispatch launches one compute grid on the context's channel. The kernel's SASS must already be NVGpuMap'd
// (prog_va). params is copied to const bank 0 at 0x160, where the kernel reads its arguments (buffer GPU VAs, inline
// scalars) at the offsets airc assigned.

#define NVGPU_PARAM_BYTES 160  // 0x160..0x200
// a kernel's: 0x160..0x400, the handles' 0x200..0x280 in among them, left 0 there to be written after (Metal binds
// 31 buffers, 20 fit below 0x200)
#define NVGPU_COMPUTE_PARAM_BYTES 0x2a0
#define NVGPU_MAX_TEXTURES 16
#define NVGPU_HANDLES 32  // at 0x200: texture slots' header indices, then sampler slots' sampler indices << 20

// A launch with a descriptor heap (the process's texture headers and samplers, which argument buffers name by
// index) has the heap for its pools instead of the slots' own, and its shaders read each bound slot's index from
// handles in their const bank. Without one, the headers and samplers go into pool slots 0..n-1.
struct nvgpu_heap {
    unsigned long long headers, samplers;  // GPU VAs, 0 for none
    unsigned int nheaders, nsamplers;
    unsigned int handles[NVGPU_HANDLES];
};

// Texture headers and samplers go into pool slots 0..n-1 for the launch; a kernel names a texture by the
// handle header index | sampler index << 20, passed like any other param.
struct nvgpu_dispatch {
    unsigned int headers[NVGPU_MAX_TEXTURES][8];   // TIC entries (Maxwell+ TEXHEAD, 32 bytes)
    unsigned int samplers[NVGPU_MAX_TEXTURES][8];  // TSC entries (TEXSAMP, 32 bytes)
    unsigned int nheaders, nsamplers;
    unsigned long long prog_va;
    unsigned char params[NVGPU_COMPUTE_PARAM_BYTES];
    unsigned int params_size;
    unsigned int grid_x, grid_y, grid_z;     // CTAs in the grid
    unsigned int block_x, block_y, block_z;  // threads per CTA
    // a part of a larger (non-uniform) dispatch: its first thread and threadgroup in it, and the dispatch's
    // threadgroups and threads; all 0 for a dispatch of its own
    unsigned int origin[3], group_origin[3], groups[3], threads[3];
    unsigned int reg_count;  // registers per thread (QMD REGISTER_COUNT)
    unsigned int shared_size;  // threadgroup memory per CTA, bytes
    unsigned int barriers;     // named barriers the kernel uses (QMD BARRIER_COUNT)
    unsigned int local_size;   // local memory per thread, bytes
    struct nvgpu_heap heap;
    // when set, the GPU VA of MTLDispatchThreadgroupsIndirectArguments, read for the grid once the work before
    // the dispatch is done
    unsigned long long indirect;
};

// rm::queue_draw clears the colour target if asked, then draws on the context's channel. Each stage's code (SPH first)
// must be NVGpuMap'd; its params go to its own const bank 0 at 0x160, as a compute kernel's do.

// p1 = context: wait for its draws so far. Draws go to the GPU without waiting for each other, and a command
// buffer's end is where its memory is read back.
#define NVGPU_FN_SYNC "NVGpuSync"
// p1 = context: a command buffer of its starts
#define NVGPU_FN_CB_BEGIN "NVGpuCommandBufferBegin"

// p1 = unsigned int*, p2 = context: set to the fence of its last draw not yet waited for, 0 when there's none
// (everything queued so far is done). Hand it to NVGPU_FN_DRAW_DONE.
#define NVGPU_FN_DRAW_FENCE "NVGpuDrawFence"
// p1 = (void*)(uintptr_t) a draw fence, p2 = its context. kIOReturnSuccess once the draws up to it are done,
// kIOReturnBusy before.
#define NVGPU_FN_DRAW_DONE "NVGpuDrawDone"
// p1 = context. kIOReturnSuccess when the task's command buffers complete as their draws do, rather than waiting for
// them in submit, where a frame would hold up the next by a refresh or more. OpenGL's queues wait all the same
// (MALACHITE_CMD_OPENGL). *(bool *)p2 = whether the task is an app, not WindowServer.
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
    unsigned int target_blocklinear;  // the target is block-linear in one-GOB blocks (with depth it has to be)
    // the target's own size in pixels, which may be more than the pass's (attachments of different sizes); 0 for the
    // pass's
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
    unsigned int pass_first;  // the first thing of a render pass
    unsigned int opengl;      // from Apple's GL renderer
    unsigned int zt_format, zt_width, zt_height, zt_size, zt_kind;  // SET_ZT_FORMAT, row stride in pixels, rows
    unsigned int samples;  // per pixel of the targets (0 or 1 for one); the sizes above are then in samples
    unsigned int clear_zs;  // clear depth (bit 0) to clear_depth, stencil (bit 1) to clear_stencil, before drawing
    float clear_depth;
    unsigned int clear_stencil;
    unsigned int depth_test, depth_write, depth_func;  // SET_DEPTH_FUNC value
    unsigned int stencil_test;
    unsigned int stencil[2][4];  // front, back: SET_STENCIL_OP_FAIL, _OP_ZFAIL, _OP_ZPASS, _FUNC values
    unsigned int stencil_ref;  // the front's in bits 0-7, the back's in bits 8-15
    unsigned int stencil_read_mask[2], stencil_write_mask[2];
    unsigned int headers[NVGPU_MAX_TEXTURES][8];  // the fragment stage's textures and samplers, as for a dispatch
    unsigned int samplers[NVGPU_MAX_TEXTURES][8];
    unsigned int nheaders, nsamplers;
    unsigned int write_mask;  // SET_CT_WRITE value: R bit 0, G bit 4, B bit 8, A bit 12
    unsigned int blend[7];    // enable, colour op/src/dst, alpha op/src/dst (SET_BLEND_PER_TARGET_* values)
    struct nvgpu_target more[7];  // colour targets 1..7 (multiple render targets), the size of target 0
    // [[stage_in]] vertex attributes as SET_VERTEX_ATTRIBUTE_A values (0 unused), read from the vertex buffers
    // in streams (address 0 unused)
    unsigned int attributes[32];
    struct nvgpu_stream {
        unsigned long long addr;
        unsigned int stride, size;
        unsigned int divisor;  // 0 per vertex, n to step every n instances
    } streams[16];
    unsigned int cull_face;  // OGL_SET_CULL_FACE value, 0 to cull nothing
    unsigned int front_ccw;  // counter-clockwise triangles face the front, not Metal's default clockwise ones
    // OpenGL's state Metal has no say in (Apple's GL renderer sets it): 0 in each is Metal's behaviour
    float point_size;  // of points whose vertex shader writes none (0 for 1)
    float line_width;  // 0 for 1
    unsigned int provoking_last;  // flat attributes from a primitive's last vertex rather than its first
    unsigned int restart;  // indexed strips restart: 0 at the largest index (Metal), 1 never, 2 at restart_index
    unsigned int restart_index;
    unsigned int window_coords;  // positions are in pixels already: no viewport transform
    unsigned int depth_clamp;  // depth outside near..far clamped rather than clipped
    unsigned int fill_lines;   // triangles drawn as their edges
    float depth_bias[3];       // constant, slope scale, clamp; all 0 for none
    float blend_const[4];      // the blend colour
    struct nvgpu_heap heap;    // its handles are the fragment stage's
    unsigned int vs_textured;  // the vertex stage has textures of its own, with these handles into the heap
    unsigned int vs_handles[NVGPU_HANDLES];
    // the 64-bit word the samples passing the depth and stencil tests are counted into (a visibility result),
    // 0 for none; the count goes on from the previous such draw's unless visibility_reset
    unsigned long long visibility;
    unsigned int visibility_reset;
    unsigned int serialize;  // the draws before it finish first: it reads its own targets, or follows a barrier
    unsigned int local_size;  // bytes of local memory a thread of its shaders uses at most
    // when set, the GPU VA of MTLDrawPrimitivesIndirectArguments (MTLDrawIndexedPrimitivesIndirectArguments for an
    // indexed draw), read for first, count, instances and base_vertex once the work before the draw is done
    unsigned long long indirect;
    // a tessellated draw (primitive 0xe, a vertex a patch) when tess, its SET_TESSELLATION_PARAMETERS value, is set:
    // the tessellation init stage's code and params (const bank 0 at 0x160, as for the others), and the
    // tessellation stage's code, which reads vs_params
    unsigned int tess;
    unsigned long long ti, ts;
    unsigned int ti_regs, ts_regs;
    unsigned char ti_params[NVGPU_PARAM_BYTES];
    unsigned int ti_params_size;
    // SET_ANTI_ALIAS_ALPHA_CONTROL: alpha to coverage (bit 0), alpha to one (bit 4)
    unsigned int alpha_control;
    unsigned int per_sample;  // the fragment stage runs once a sample, not once a pixel
    unsigned int ps_depth;  // the fragment stage writes depth
    // layered targets, layers deep (0 or 1: not layered; 512 at most): colour target j's layers layer_pitch[j] bytes
    // apart, the depth/stencil target's zt_layer_pitch; the vertex stage picks the layer when shader_layer is set
    unsigned int layers, layer_pitch[8], zt_layer_pitch;
    // the layers each target has from the attached one on, of which it takes layers at most (attachments of a
    // layered pass may have fewer); 0 for layers
    unsigned int target_layers[8], zt_layers;
    unsigned int shader_layer;
    // viewports and scissors 1..nviewports - 1, beside viewport and scissor 0, for a vertex stage picking one
    unsigned int nviewports;
    float viewports[15][6];
    unsigned int scissors[15][4];
};

// p1 = struct nvgpu_stats*. What Accelerator publishes as the GPU's performance statistics, from what Gpu already
// knows and without asking GSP-RM. kIOReturnBusy when a launch holds Gpu's lock: skip that tick.
#define NVGPU_FN_STATS "NVGpuStats"

struct nvgpu_stats {
    unsigned int temp_c;  // core temperature, 0 when the sensor gives no reading
    unsigned int reserved;
    unsigned long long vram_total, vram_used;
    unsigned long long sys_mapped;  // system memory mapped into GPU address spaces
    unsigned long long busy_us;     // time any task's graphics work was outstanding (overlapping work counts once)
    unsigned long long now_us;      // the clock busy_us is measured against
    unsigned long long recoveries;  // channels rebuilt after a fault
    unsigned long long dma_bytes;   // DMA memory the driver holds: a figure that climbs with every fault is a leak
};

// p1 = a task's context handle, p2 = uint64_t*: out, the microseconds its graphics work has kept the GPU busy since
// the last call. Accelerator charges them to the submitting thread's task (ml_gpu_stat_update), which is where Activity
// Monitor's GPU Time and % GPU columns come from. No lock, so it never waits on a launch.
#define NVGPU_FN_CTX_BUSY "NVGpuCtxBusy"
