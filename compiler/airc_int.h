#pragma once
#include "airc.h"
#include "ir.h"
#include "nvk.h"

struct airc_info {
    enum nvk_stage stage;
    struct nvk_arg args[NVK_MAX_ARGS];
    struct airc_arg_name arg_names[NVK_MAX_ARGS];
    unsigned nargs;
    uint64_t readonly_args;  // buffers the AIR marks read-only and non-aliasing
    unsigned array_count[NVK_MAX_ARGS];  // for arrays of textures or samplers
    unsigned ngeneric;      // vertex outputs after the position
    struct airc_varyings outputs;
    uint8_t interp[32];     // enum sph_interp
    uint32_t centroid;
    uint32_t at_sample;
    unsigned flags;         // NVK_FLAG_*
    int denorms;            // OpenCL keeps denormals, Metal flushes them
    unsigned nvary;
    unsigned sysvals;  // SPH_*
    unsigned ps_sysvals;  // SPH_PS_*
    unsigned targets;
    uint32_t vertex_inputs;  // per control point in a post-tessellation function
    uint32_t patch_inputs;
    uint32_t float_inputs;
    unsigned patch_domain, patch_points;  // patch_points is a tessellation init stage's threads
    uint32_t point_outputs;  // a tessellation init stage's generic outputs per control point
    unsigned patch_outputs;
    uint64_t const_samplers[NVK_MAX_CONST_SAMPLERS];
    unsigned nconst_samplers;
    uint32_t compare_samplers, plain_samplers;
    uint32_t nearest_samplers;  // those an integer texture is sampled with
    uint32_t face_textures;
    uint32_t level_textures;  // written at a level that may not be the first
    unsigned shared_size;
    unsigned local_size;   // per thread
    unsigned max_threads;  // 0 when the kernel doesn't declare it
    unsigned barriers;
    const unsigned char *data;  // constant tables read from memory
    unsigned data_size;
};

struct airc_constant {
    uint32_t index;
    uint32_t bits;
};

_Noreturn void airc_fail(const char *fmt, ...);
// also set when a compile is retried without optional passes
int airc_pass_off(const char *name);

// with calls, noinline functions become subroutines
void airc_front(void *module, void **linked, const char *const *linked_names, unsigned nlinked,
                const struct airc_varyings *vertex, const struct airc_stage_in *stage_in,
                const struct airc_constant *fc, unsigned nfc, int calls, struct ir_func *f, struct airc_info *info);
// components as the vertex fetch unit gives them, missing ones 0, 0, 0, 1
void airc_fetch(struct ir_block *b, const struct airc_tess_attr *a, unsigned loc, struct ir_val *buffer,
                struct ir_val *idx, struct ir_val *v[4]);
struct ir_val *airc_udiv(struct ir_block *b, struct ir_val *n, uint32_t d);
void airc_opt(struct ir_func *f);
extern int airc_no_unroll, airc_unrolled;
extern int airc_no_texvec, airc_texvec_shared;
// fetches a block may start before their results are read, 0 for none
extern int airc_batch, airc_batched;
extern unsigned airc_samples;
// reads local arrays that only copy read-only memory from that memory; 1 if f changed
int airc_const_locals(struct ir_func *f, struct airc_info *info);
// remakes cheap values where they're read rather than keeping them live
void airc_remat(struct ir_func *f);
// merges aligned adjacent threadgroup memory accesses into 64- or 128-bit ones
void airc_widen_shared(struct ir_func *f, const struct airc_info *info);
int airc_generate(void (*gen)(const void *arg, struct ir_func *f, struct airc_info *info), const void *arg,
                  struct airc_kernel *k);
// A compile out of registers retries with the values live then kept in memory; airc_spill_more returns 0 if none
// are new.
int airc_spill_more(void);
void airc_spill_reset(void);
unsigned char *airc_back(struct ir_func *f, const struct airc_info *info, unsigned *size, unsigned *regs);
