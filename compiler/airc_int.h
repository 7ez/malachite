// What airc's passes share: the shader's interface, as the front end finds it and the back end and the
// driver need it.
#pragma once
#include "airc.h"
#include "ir.h"
#include "nvk.h"

struct airc_info {
    enum nvk_stage stage;
    struct nvk_arg args[NVK_MAX_ARGS];
    struct airc_arg_name arg_names[NVK_MAX_ARGS];
    unsigned nargs;
    uint64_t readonly_args;  // buffer arguments the AIR says are only read and alias no other, a bit each
    unsigned array_count[NVK_MAX_ARGS];  // elements of an argument that's an array of textures or samplers
    unsigned ngeneric;      // vertex outputs after the position
    struct airc_varyings outputs;  // their names
    uint8_t interp[32];     // fragment inputs: enum sph_interp each
    uint32_t centroid;      // and those interpolated at the centroid, and at each sample, a bit each
    uint32_t at_sample;
    unsigned flags;         // NVK_FLAG_*
    unsigned nvary;
    unsigned sysvals;  // SPH_* system values a vertex function reads
    unsigned ps_sysvals;  // SPH_PS_* system values a fragment function reads
    unsigned targets;     // colour targets a fragment function writes, a bit each
    uint32_t vertex_inputs;  // [[attribute(n)]] locations read (a post-tessellation function's per control point)
    uint32_t patch_inputs;   // and per patch
    uint32_t float_inputs;   // a vertex function's read as floats
    unsigned patch_domain, patch_points;  // a post-tessellation function's patch, a tessellation init stage's threads
    uint32_t point_outputs;  // a tessellation init stage's: its control point's generic outputs, a bit each
    unsigned patch_outputs;  // and its patch's
    uint64_t const_samplers[NVK_MAX_CONST_SAMPLERS];
    unsigned nconst_samplers;
    uint32_t compare_samplers, plain_samplers;  // sampler pool slots sampled with a depth compare, and without
    uint32_t nearest_samplers;  // and those an integer texture is sampled with
    uint32_t face_textures;  // texture slots of cubes read or written by face
    uint32_t level_textures;  // texture slots written at a level that may not be the first
    unsigned shared_size;  // threadgroup memory, bytes
    unsigned local_size;   // local memory a thread uses, bytes
    unsigned max_threads;  // threads a threadgroup may have at most, as the kernel declares (0: not said)
    unsigned barriers;
    const unsigned char *data;  // the constant tables read from memory, data_size bytes
    unsigned data_size;
};

// a function constant's specialised value, by its [[function_constant(index)]]
struct airc_constant {
    uint32_t index;
    uint32_t bits;
};

_Noreturn void airc_fail(const char *fmt, ...);
// an optional pass's switch, named by its environment variable: also set when a compile is run again without them
int airc_pass_off(const char *name);

// LLVM module -> IR in f, the shader's interface in info; calls: noinline functions become subroutines
void airc_front(void *module, void **linked, const char *const *linked_names, unsigned nlinked,
                const struct airc_varyings *vertex, const struct airc_stage_in *stage_in,
                const struct airc_constant *fc, unsigned nfc, int calls, struct ir_func *f, struct airc_info *info);
// attribute a of element idx of the buffer at address buffer, in b: its components as the vertex fetch unit gives
// them (floats for the float and normalised formats, integers otherwise; those the format lacks 0, 0, 0, 1)
void airc_fetch(struct ir_block *b, const struct airc_tess_attr *a, unsigned loc, struct ir_val *buffer,
                struct ir_val *idx, struct ir_val *v[4]);
// n / d in b, for a constant d
struct ir_val *airc_udiv(struct ir_block *b, struct ir_val *n, uint32_t d);
void airc_opt(struct ir_func *f);
// airc_opt unrolls small loops unless airc_no_unroll is set, and sets airc_unrolled when it does
extern int airc_no_unroll, airc_unrolled;
// the raster sample count of the pipeline a fragment function is compiled for
extern unsigned airc_samples;
// a local array that is only a copy of read-only memory, read from that memory instead; 1 if it changed f (which
// airc_opt then cleans up)
int airc_const_locals(struct ir_func *f, struct airc_info *info);
// values cheap to make again, made in the blocks that read them rather than held in registers from far before
void airc_remat(struct ir_func *f);
// adjacent words of threadgroup memory read or written one at a time, as one 64- or 128-bit access where aligned
void airc_widen_shared(struct ir_func *f, const struct airc_info *info);
// compile the stage gen builds the IR of from arg, rather than from AIR, into k; 0, or -1 with k->error saying why
int airc_generate(void (*gen)(const void *arg, struct ir_func *f, struct airc_info *info), const void *arg,
                  struct airc_kernel *k);
// IR -> SASS (malloc'd, *size bytes), with the register count the launch needs in *regs
// A compile that ran out of registers can try again with the values that were live then kept in memory:
// airc_spill_more takes them (0 if none are new), airc_spill_reset starts a compile with none.
int airc_spill_more(void);
void airc_spill_reset(void);
unsigned char *airc_back(struct ir_func *f, const struct airc_info *info, unsigned *size, unsigned *regs);
