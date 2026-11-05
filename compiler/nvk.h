// Compiled-kernel bundle airc writes and the Metal plugin loads: this header, then sass_size bytes of SASS.
// The plugin finds a function's bundle by its AIR bitCodeHash.
#pragma once
#include <stdint.h>

#define NVK_MAGIC 0x314b564e  // "NVK1"
#define NVK_MAX_ARGS 64
#define NVK_MAX_CONST_SAMPLERS 4
#define NVK_CONST_SAMPLER_SLOT(i) (15 - (i))
#define NVK_RENDER_TARGET_SLOT(i) (15 - (i))
// where a shader reads the descriptor heap index of what's bound at texture slot i (and sampler slot i, as
// index << 20): a table at 0x200 of its const bank 0
#define NVK_TEXTURE_HANDLE(i) (0x200 + 4 * (i))
// the draw's base vertex and base instance, in the vertex stage's const bank: the hardware's instance id leaves
// the base out, and Metal's [[instance_id]] counts from it
// a compute launch's place in the dispatch it's part of (a non-uniform one is launched as up to 8 uniform parts),
// in const bank 0 after ntid and nctaid: the part's first thread and threadgroup, the dispatch's threadgroups and
// threads, three words each
#define NVK_CB_ORIGIN 0x18
#define NVK_CB_GROUP_ORIGIN 0x24
#define NVK_CB_GROUPS 0x30
#define NVK_CB_GRID 0x3c
#define NVK_BASE_VERTEX 0x100
#define NVK_BASE_INSTANCE 0x104
#define NVK_SAMPLER_HANDLE(i) (0x240 + 4 * (i))  // texture pool slot the fragment stage reads colour attachment i from

// Only buffers take a const-bank slot the dispatcher fills; the rest are builtins the kernel reads itself.
enum nvk_arg_kind {
    NVK_ARG_BUFFER,
    NVK_ARG_TID,      // thread_position_in_grid
    NVK_ARG_TG_POS,   // threadgroup_position_in_grid
    NVK_ARG_TG_TID,   // thread_position_in_threadgroup
    NVK_ARG_TG_SIZE,  // threads_per_threadgroup
    NVK_ARG_GROUPS,   // threadgroups_per_grid
    NVK_ARG_TEXTURE,  // a texture: index is its TIC pool slot
    NVK_ARG_SAMPLER,  // a sampler: index is its TSC pool slot
    NVK_ARG_VERTEX_ID,
    NVK_ARG_INSTANCE_ID,
    NVK_ARG_POSITION,  // a fragment's window position
    NVK_ARG_VARYING,   // a fragment input: index is the vertex output it comes from
    NVK_ARG_VERTEX_INPUT,  // a [[stage_in]] attribute: index is its [[attribute(n)]] location
    NVK_ARG_RENDER_TARGET,  // a [[color(n)]] input, the pixel's value in colour attachment index
    NVK_ARG_POINT_COORD,    // [[point_coord]]: where in the point sprite the fragment is
    NVK_ARG_FRONT_FACING,   // [[front_facing]]
    NVK_ARG_GRID_SIZE,      // threads_per_grid
    NVK_ARG_TG_INDEX,       // thread_index_in_threadgroup
    NVK_ARG_SIMD_LANE,      // thread_index_in_simdgroup
    NVK_ARG_SIMD_GROUP,     // simdgroup_index_in_threadgroup
    NVK_ARG_SIMD_SIZE,      // threads_per_simdgroup
    NVK_ARG_SIMD_GROUPS,    // simdgroups_per_threadgroup
    NVK_ARG_THREADGROUP,    // [[threadgroup(n)]] memory sized at dispatch: its shared offset, a param at cbuf_off
    NVK_ARG_BASE_VERTEX,    // [[base_vertex]]
    NVK_ARG_BASE_INSTANCE,  // [[base_instance]]
    NVK_ARG_CONTROL_POINTS,     // a post-tessellation vertex function's patch_control_point<T>
    NVK_ARG_PATCH_INPUT,        // its per-patch [[stage_in]] member: index is its [[attribute(n)]] location
    NVK_ARG_POSITION_IN_PATCH,  // [[position_in_patch]]
    NVK_ARG_PATCH_ID,           // [[patch_id]]
    NVK_ARG_SAMPLE_ID,          // [[sample_id]]
    NVK_ARG_SAMPLE_MASK,        // a fragment's [[sample_mask]] input: the samples of the pixel it covers
    NVK_ARG_LAYER,              // a fragment's [[render_target_array_index]]
    NVK_ARG_VIEWPORT,           // a fragment's [[viewport_array_index]]
    NVK_ARG_DISPATCH_TG_SIZE,   // dispatch_threads_per_threadgroup
    NVK_ARG_QUAD_GROUP,         // quadgroup_index_in_threadgroup
    NVK_ARG_QUAD_LANE,          // thread_index_in_quadgroup
    NVK_ARG_QUAD_GROUPS,        // quadgroups_per_threadgroup
    NVK_ARG_STAGE_IN_ORIGIN,    // a kernel's stage_in_grid_origin: the stage-in region's origin
    NVK_ARG_STAGE_IN_SIZE,      // stage_in_grid_size: and its size
    // after the function's own arguments, a kernel with a stage input has a param at cbuf_off the dispatcher points
    // at its stage-in region (as MTLStageInRegionIndirectArguments lays one out), and one for each buffer the stage
    // input reads ([[buffer(index)]]), filled like a buffer argument's
    NVK_ARG_STAGE_IN_REGION,
    NVK_ARG_STAGE_IN_BUFFER,
};

// a post-tessellation vertex function is the tessellation stage; the tessellation init stage before it is generated
enum nvk_stage { NVK_COMPUTE, NVK_VERTEX, NVK_FRAGMENT, NVK_TESS_EVAL, NVK_TESS_INIT };

// what the tessellation init stage leaves the tessellation stage in the patch's attributes: the patch id and
// instance id (from the base instance), then the per-patch [[attribute(n)]]s in location order
#define NVK_PATCH_IDS 0x20
#define NVK_PATCH_INPUT(inputs, n) (0x30 + 16 * __builtin_popcount((inputs) & ((1u << (n)) - 1)))

struct nvk_arg {
    uint8_t kind;
    uint8_t index;      // Metal [[buffer(n)]] slot (air.location_index)
    uint16_t cbuf_off;  // where the kernel reads it in const bank 0
    uint16_t size;      // an OpenCL argument passed by value (air.constant): its bytes; 0 for the rest
    uint8_t access;     // MTLBindingAccess: read (0), read_write (1) or write (2), from air.read_write and air.write
};

struct nvk_header {
    uint32_t magic;
    uint32_t reg_count;
    uint32_t nargs;
    uint32_t sass_size;
    uint32_t shared_size;  // bytes of threadgroup memory per threadgroup
    uint32_t local_size;   // bytes of local memory per thread
    uint32_t max_threads;  // a kernel's [[max_total_threads_per_threadgroup]], 0 for none
    uint32_t barriers;     // hardware barriers the kernel uses (BAR.SYNC 0 needs one)
    struct nvk_arg args[NVK_MAX_ARGS];
    uint32_t stage;  // enum nvk_stage; a vertex or fragment function's SASS starts with its 128-byte SPH
    // samplers the source declares constexpr, in Metal's packed encoding: sampler pool slots 15, 14, ...
    uint32_t nconst_samplers;
    uint64_t const_samplers[NVK_MAX_CONST_SAMPLERS];
    uint32_t targets;  // fragment: the colour targets it writes, a bit each
    // a post-tessellation vertex function's patch: triangle (0) or quad (1), its control points (0 when the source
    // leaves the count to the draw), and the [[attribute(n)]] locations it reads per control point and per patch
    uint32_t patch_domain, patch_points;
    uint32_t point_inputs, patch_inputs;
    // the sampler pool slots its samples with a depth compare use, and those its other samples and gathers use
    uint32_t compare_samplers, plain_samplers;
    uint32_t nearest_samplers;  // those an integer texture is sampled with: Apple's GPUs never filter those
    uint32_t face_textures;  // the texture slots of cubes it reads or writes texels of, as layers of their faces
    uint32_t level_textures;  // those it writes texels of at a level that may not be the first (handle + level)
    uint32_t flags;  // NVK_FLAG_*
};

#define NVK_FLAG_PER_SAMPLE 1  // fragment: runs once a sample, reading [[sample_id]] or an input at the sample
#define NVK_FLAG_LAYER 2       // vertex: writes [[render_target_array_index]]
#define NVK_FLAG_VIEWPORT 4    // and [[viewport_array_index]]
#define NVK_FLAG_DEPTH 8       // fragment: writes [[depth]]

