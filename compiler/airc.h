// AIR to GA10x SASS, shared by the airc tool and the Metal plugin
#pragma once
#include <stddef.h>
#include "nvk.h"

// Metal pairs fragment inputs with vertex outputs by name, not position
#define AIRC_NAME 64
struct airc_varyings {
    unsigned n;
    char names[32][AIRC_NAME];
};

// for reflection
struct airc_arg_name {
    char name[AIRC_NAME];
    char type[AIRC_NAME];
};

struct airc_kernel {
    struct nvk_header h;
    struct airc_arg_name arg_names[NVK_MAX_ARGS];  // beside h.args
    unsigned char *sass;  // a graphics stage's SPH, code, then constant data; the caller frees it
    char name[64];
    char error[160];
    struct airc_varyings outputs;  // a vertex function's
};

// an airc_constant at this index gives the value of every function constant the list doesn't name
#define AIRC_FC_DEFAULT 0xffffffffu

struct airc_constant;

// Formats as SET_VERTEX_ATTRIBUTE_A has them: component widths 26:21, type 29:27, R/B swap 31
struct airc_tess_attr {
    uint32_t format, buffer, offset, stride;
    uint32_t step, rate;  // MTLVertexStepFunction, and instances a step
};
struct airc_tess_desc {
    uint32_t domain, points;  // as the function's header has them
    uint32_t point_inputs, patch_inputs;
    uint32_t index_size;   // 0 for no control point index buffer
    uint32_t factor_step;  // MTLTessellationFactorStepFunction
    uint32_t pow2;         // the factors rounded up to powers of two
    float max_factor;
    struct airc_tess_attr attrs[32];  // by location
};

// format 0 for an attribute the descriptor leaves out; step is MTLStepFunction (ThreadPositionInGridX 5, Y 6,
// XIndexed 7, YIndexed 8)
struct airc_stage_in {
    struct airc_tess_attr attrs[32];
    uint32_t index_buffer, index_size;
};

// the tessellation init stage's const bank params; index buffers are 0 for none
#define AIRC_TESS_FACTORS 0x160
#define AIRC_TESS_PATCH_INDEX 0x168
#define AIRC_TESS_POINT_INDEX 0x170
#define AIRC_TESS_INSTANCE_STRIDE 0x178
#define AIRC_TESS_SCALE 0x17c
#define AIRC_TESS_BUFFER(i) (0x180 + 8 * (i))
#define AIRC_TESS_BUFFERS 16

// OpenCL's hidden global work offset and work_dim buffer; its renderer binds them at 0 and the kernel's buffers from 3
#define AIRC_CL_OFFSETS 30
// images' formats, a word per texture slot of channel order << 16 | channel data type
#define AIRC_CL_FORMATS 29

#define AIRC_MAX_LINKED 256
struct airc_source {
    const void *air;
    size_t len;
    // MTLFunctionDescriptor's specializedName, when callers use a name other than the defined one, or NULL
    const char *name;
};

#ifdef __cplusplus
extern "C" {
#endif

// air may have Apple's wrapper header or not. Uses global state, so not reentrant.
int airc_compile(const void *air, size_t len, const struct airc_constant *fc, unsigned nfc, int print_ir,
                 struct airc_kernel *k);
// Inlines NAME.MTL_VISIBLE_FN_REF calls from linked. A NULL vertex takes fragment inputs in order; a NULL stage_in
// describes none.
int airc_compile_linked(const void *air, size_t len, const struct airc_source *linked, unsigned nlinked,
                        const struct airc_varyings *vertex, const struct airc_stage_in *stage_in, unsigned samples,
                        const struct airc_constant *fc, unsigned nfc, int print_ir, struct airc_kernel *k);
// the generated stages before a post-tessellation vertex function, each a vertex a patch; not reentrant either
int airc_tess_vertex(struct airc_kernel *k);
int airc_tess_init(const struct airc_tess_desc *d, struct airc_kernel *k);

#ifdef __cplusplus
}
#endif
