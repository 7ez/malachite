// Bundle layout: this header, sass_size bytes of SASS, data_size bytes of constant data; found by AIR bitCodeHash
#pragma once
#include <stdint.h>

#define NVK_MAGIC 0x324b564e  // "NVK2"
#define NVK_MAX_ARGS 64
#define NVK_MAX_CONST_SAMPLERS 4
#define NVK_CONST_SAMPLER_SLOT(i) (15 - (i))
#define NVK_RENDER_TARGET_SLOT(i) (15 - (i))
// heap index of texture slot i, and of sampler slot i << 20
#define NVK_TEXTURE_HANDLE(i) (0x200 + 4 * (i))
// A non-uniform dispatch launches as up to 8 uniform parts; each part's first thread and threadgroup, then the
// dispatch's threadgroups and threads, three words each.
#define NVK_CB_ORIGIN 0x18
#define NVK_CB_GROUP_ORIGIN 0x24
#define NVK_CB_GROUPS 0x30
#define NVK_CB_GRID 0x3c
// the hardware's instance id leaves the base out, and Metal's [[instance_id]] counts from it
#define NVK_BASE_VERTEX 0x100
#define NVK_BASE_INSTANCE 0x104
#define NVK_SAMPLER_HANDLE(i) (0x240 + 4 * (i))  // texture pool slot for reading colour attachment i

// Only buffers take a const-bank slot the dispatcher fills; the rest are builtins the kernel reads itself.
enum nvk_arg_kind {
    NVK_ARG_BUFFER,
    NVK_ARG_TID,      // thread_position_in_grid
    NVK_ARG_TG_POS,   // threadgroup_position_in_grid
    NVK_ARG_TG_TID,   // thread_position_in_threadgroup
    NVK_ARG_TG_SIZE,  // threads_per_threadgroup
    NVK_ARG_GROUPS,   // threadgroups_per_grid
    NVK_ARG_TEXTURE,  // index is the TIC pool slot
    NVK_ARG_SAMPLER,  // index is the TSC pool slot
    NVK_ARG_VERTEX_ID,
    NVK_ARG_INSTANCE_ID,
    NVK_ARG_POSITION,
    NVK_ARG_VARYING,   // index is the vertex output it comes from
    NVK_ARG_VERTEX_INPUT,  // index is the [[attribute(n)]] location
    NVK_ARG_RENDER_TARGET,  // [[color(n)]] input from colour attachment index
    NVK_ARG_POINT_COORD,
    NVK_ARG_FRONT_FACING,
    NVK_ARG_GRID_SIZE,      // threads_per_grid
    NVK_ARG_TG_INDEX,       // thread_index_in_threadgroup
    NVK_ARG_SIMD_LANE,      // thread_index_in_simdgroup
    NVK_ARG_SIMD_GROUP,     // simdgroup_index_in_threadgroup
    NVK_ARG_SIMD_SIZE,      // threads_per_simdgroup
    NVK_ARG_SIMD_GROUPS,    // simdgroups_per_threadgroup
    NVK_ARG_THREADGROUP,    // memory sized at dispatch; its shared offset is a param at cbuf_off
    NVK_ARG_BASE_VERTEX,
    NVK_ARG_BASE_INSTANCE,
    NVK_ARG_CONTROL_POINTS,     // patch_control_point<T>
    NVK_ARG_PATCH_INPUT,        // index is the per-patch [[attribute(n)]] location
    NVK_ARG_POSITION_IN_PATCH,
    NVK_ARG_PATCH_ID,
    NVK_ARG_SAMPLE_ID,
    NVK_ARG_SAMPLE_MASK,        // the input, the samples the fragment covers
    NVK_ARG_LAYER,              // [[render_target_array_index]]
    NVK_ARG_VIEWPORT,           // [[viewport_array_index]]
    NVK_ARG_DISPATCH_TG_SIZE,   // dispatch_threads_per_threadgroup
    NVK_ARG_QUAD_GROUP,         // quadgroup_index_in_threadgroup
    NVK_ARG_QUAD_LANE,          // thread_index_in_quadgroup
    NVK_ARG_QUAD_GROUPS,        // quadgroups_per_threadgroup
    NVK_ARG_STAGE_IN_ORIGIN,    // stage_in_grid_origin
    NVK_ARG_STAGE_IN_SIZE,      // stage_in_grid_size
    // After the function's own arguments: the stage-in region as MTLStageInRegionIndirectArguments lays it out,
    // then each buffer the stage input reads, filled like a buffer argument.
    NVK_ARG_STAGE_IN_REGION,
    NVK_ARG_STAGE_IN_BUFFER,
    // the loader puts constant data sass_size bytes (a multiple of 16) into the code's buffer
    NVK_ARG_DATA,
};

// the tessellation init stage before a post-tessellation vertex function is generated
enum nvk_stage { NVK_COMPUTE, NVK_VERTEX, NVK_FRAGMENT, NVK_TESS_EVAL, NVK_TESS_INIT };

// Per-patch attributes in location order, then patch id and instance id. A patch has 128 words and the levels
// take 8, so with 30 attributes the ids don't fit.
#define NVK_PATCH_INPUT(inputs, n) (0x20 + 16 * __builtin_popcount((inputs) & ((1u << (n)) - 1)))
#define NVK_PATCH_IDS(inputs) (0x20 + 16 * __builtin_popcount(inputs))
#define NVK_MAX_PATCH_INPUTS 30

struct nvk_arg {
    uint8_t kind;
    uint8_t index;      // air.location_index
    uint16_t cbuf_off;  // in const bank 0
    uint16_t size;      // bytes of an OpenCL by-value argument (air.constant), else 0
    uint8_t access;     // MTLBindingAccess, from air.read_write and air.write
};

struct nvk_header {
    uint32_t magic;
    uint32_t reg_count;
    uint32_t nargs;
    uint32_t sass_size;
    uint32_t shared_size;  // per threadgroup
    uint32_t local_size;   // per thread
    uint32_t max_threads;  // [[max_total_threads_per_threadgroup]], 0 for none
    uint32_t barriers;     // BAR.SYNC 0 needs one
    struct nvk_arg args[NVK_MAX_ARGS];
    uint32_t stage;  // enum nvk_stage; vertex and fragment SASS starts with a 128-byte SPH
    // in Metal's packed encoding, at sampler pool slots 15, 14, ...
    uint32_t nconst_samplers;
    uint64_t const_samplers[NVK_MAX_CONST_SAMPLERS];
    uint32_t targets;
    // domain triangle 0 or quad 1; points 0 when the draw sets the count
    uint32_t patch_domain, patch_points;
    uint32_t point_inputs, patch_inputs;
    // Metal converts integer formats to float for these
    uint32_t float_inputs;
    // sampler pool slots used with and without a depth compare
    uint32_t compare_samplers, plain_samplers;
    uint32_t nearest_samplers;  // used on integer textures, which Apple's GPUs never filter
    uint32_t face_textures;  // cubes accessed by face, as layers
    uint32_t level_textures;  // written at a level that may not be the first (handle + level)
    uint32_t flags;  // NVK_FLAG_*
    uint32_t data_size;
};

#define NVK_FLAG_PER_SAMPLE 1  // reads [[sample_id]] or an input at the sample
#define NVK_FLAG_LAYER 2       // vertex writes [[render_target_array_index]]
#define NVK_FLAG_VIEWPORT 4    // and [[viewport_array_index]]
#define NVK_FLAG_DEPTH 8       // fragment writes [[depth]]

