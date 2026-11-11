// AIR -> GA10x SASS compiler as a library: the airc tool and the Metal plugin both call airc_compile.
#pragma once
#include <stddef.h>
#include "nvk.h"

// a vertex function's user outputs by name, in output order: Metal pairs a fragment function's inputs with them by
// name, not position
#define AIRC_NAME 64
struct airc_varyings {
    unsigned n;
    char names[32][AIRC_NAME];
};

// an argument's name and type as the source has them, for reflection
struct airc_arg_name {
    char name[AIRC_NAME];
    char type[AIRC_NAME];
};

struct airc_kernel {
    struct nvk_header h;
    struct airc_arg_name arg_names[NVK_MAX_ARGS];  // beside h.args
    unsigned char *sass;  // a graphics stage's SPH, then its code, then its constant data; malloc'd, the caller frees it
    char name[64];
    char error[160];
    struct airc_varyings outputs;  // a vertex function's
};

// the index of an airc_constant that gives every function constant the list doesn't name its value
#define AIRC_FC_DEFAULT 0xffffffffu

struct airc_constant;

// The tessellation init stage of a pipeline whose vertex function is a post-tessellation one: the vertex descriptor
// (the formats as SET_VERTEX_ATTRIBUTE_A has them, component widths 26:21, type 29:27, R/B swap 31) of the
// locations the function reads, and the pipeline's tessellation state
struct airc_tess_attr {
    uint32_t format, buffer, offset, stride;
    uint32_t step, rate;  // MTLVertexStepFunction, and a per-instance one's instances a step
};
struct airc_tess_desc {
    uint32_t domain, points;  // as the function's header has them
    uint32_t point_inputs, patch_inputs;
    uint32_t index_size;   // bytes of a control point index, 0 for none
    uint32_t factor_step;  // MTLTessellationFactorStepFunction
    uint32_t pow2;         // the factors rounded up to powers of two
    float max_factor;
    struct airc_tess_attr attrs[32];  // by location
};

// A kernel's stage input (MTLStageInputOutputDescriptor): its attributes by location, format 0 for one it doesn't
// describe, step as MTLStepFunction (ThreadPositionInGridX 5, Y 6, XIndexed 7, YIndexed 8, or Constant), buffer
// a [[buffer(n)]] index; and where the indexed step functions read their indices, and an index's bytes
struct airc_stage_in {
    struct airc_tess_attr attrs[32];
    uint32_t index_buffer, index_size;
};

// the tessellation init stage's params, from 0x160 of its const bank: the factor buffer, the patch index buffer and
// the control point index buffer (0 for none), the factors' instance stride and scale, then vertex buffers 0..15
#define AIRC_TESS_FACTORS 0x160
#define AIRC_TESS_PATCH_INDEX 0x168
#define AIRC_TESS_POINT_INDEX 0x170
#define AIRC_TESS_INSTANCE_STRIDE 0x178
#define AIRC_TESS_SCALE 0x17c
#define AIRC_TESS_BUFFER(i) (0x180 + 8 * (i))
#define AIRC_TESS_BUFFERS 16

// another module whose functions the shader may call: a pipeline's linked (visible) functions
#define AIRC_MAX_LINKED 16
struct airc_source {
    const void *air;
    size_t len;
    // the name callers know it by when that's not the one it's defined under (MTLFunctionDescriptor's
    // specializedName: RenderBox links glassHighlight_v1 and the like as custom_fn), or NULL
    const char *name;
};

#ifdef __cplusplus
extern "C" {
#endif

// Compile the function of an AIR module (LLVM bitcode, with or without Apple's wrapper header), its function
// constants specialised to fc (nfc values). Returns 0, or -1 with k->error saying why. print_ir dumps the IR
// to stdout. Uses global state: not reentrant.
int airc_compile(const void *air, size_t len, const struct airc_constant *fc, unsigned nfc, int print_ir,
                 struct airc_kernel *k);
// the same, with a call to a function the module only declares (Metal's NAME.MTL_VISIBLE_FN_REF) inlined from
// the linked modules' NAME, and for a fragment function, its inputs read from the slots of the vertex outputs of
// their names (vertex: the vertex function's outputs; NULL takes them in order) and get_num_samples() the
// pipeline's raster sample count (samples), and for a kernel, its stage input fetched from the buffers stage_in
// describes (NULL: none described)
int airc_compile_linked(const void *air, size_t len, const struct airc_source *linked, unsigned nlinked,
                        const struct airc_varyings *vertex, const struct airc_stage_in *stage_in, unsigned samples,
                        const struct airc_constant *fc, unsigned nfc, int print_ir, struct airc_kernel *k);
// A tessellated draw's stages before the post-tessellation vertex function, generated (each a vertex a patch):
// the vertex stage, the same for every pipeline, and the tessellation init stage of d. 0, or -1 with k->error
// saying why; not reentrant either.
int airc_tess_vertex(struct airc_kernel *k);
int airc_tess_init(const struct airc_tess_desc *d, struct airc_kernel *k);

#ifdef __cplusplus
}
#endif
