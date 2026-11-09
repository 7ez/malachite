// airc: AIR -> IR -> SASS.
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "airc_int.h"
#include "airc.h"
#ifdef AIRC_LIBLLVM
#include <llvm-c/BitReader.h>
#include <llvm-c/Core.h>
#else
#include "llvm_lite.h"
#endif

#define MAX_SPILL_ROUNDS 16

// the optimised IR of a stage before it is compiled, for irsim
void (*airc_ir_hook)(struct ir_func *f, const struct airc_info *info);

static jmp_buf g_bail;
static int g_passes_off;
static char *g_error;
static unsigned g_error_size;

_Noreturn void airc_fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, g_error_size, fmt, ap);
    va_end(ap);
    longjmp(g_bail, 1);
}

int airc_pass_off(const char *name)
{
    return g_passes_off || getenv(name);
}

// the bitcode inside Apple's wrapper, if it has one, as a module; 0, or -1 with the reason in g_error
static int parse(const void *air, size_t len, LLVMMemoryBufferRef *mb, LLVMModuleRef *mod)
{
    const unsigned char *bc = air;
    if (len > 20 && bc[0] == 0xde && bc[1] == 0xc0 && bc[2] == 0x17 && bc[3] == 0x0b) {  // Apple's wrapper
        uint32_t off, size;
        memcpy(&off, bc + 8, 4);
        memcpy(&size, bc + 12, 4);
        if ((size_t)off + size > len) {
            snprintf(g_error, g_error_size, "wrapper header points past the file");
            return -1;
        }
        bc += off;
        len = size;
    }
    *mb = LLVMCreateMemoryBufferWithMemoryRange((const char *)bc, len, "air", 0);
    if (LLVMParseBitcode2(*mb, mod)) {
        LLVMDisposeMemoryBuffer(*mb);
        snprintf(g_error, g_error_size, "not LLVM bitcode");
        return -1;
    }
    return 0;
}

// a stage's IR, optimised and compiled, into k; with remat, values far from their uses made again there
static void finish(struct ir_func *f, struct airc_info *info, int print_ir, int remat, struct airc_kernel *k)
{
    airc_opt(f);
    if (airc_const_locals(f, info))
        airc_opt(f);
    if (print_ir)
        ir_print(f);
    if (airc_ir_hook)
        airc_ir_hook(f, info);
    if (remat)
        airc_remat(f);
    unsigned regs, size;
    k->sass = airc_back(f, info, &size, &regs);
    if (info->data_size) {
        k->sass = realloc(k->sass, size + info->data_size);
        memcpy(k->sass + size, info->data, info->data_size);
    }
    k->h.data_size = info->data_size;
    k->h.magic = NVK_MAGIC;
    k->h.reg_count = regs;
    k->h.nargs = info->nargs;
    k->h.sass_size = size;
    k->h.shared_size = info->shared_size;
    k->h.local_size = (info->local_size + 15) & ~15u;
    k->h.max_threads = info->max_threads;
    k->h.barriers = info->barriers;
    k->h.stage = info->stage;
    k->h.targets = info->targets;
    k->h.patch_domain = info->patch_domain;
    k->h.patch_points = info->patch_points;
    k->h.point_inputs = info->stage == NVK_TESS_EVAL ? info->vertex_inputs : 0;
    k->h.patch_inputs = info->patch_inputs;
    k->h.flags = info->flags;
    k->outputs = info->outputs;
    k->h.nconst_samplers = info->nconst_samplers;
    k->h.compare_samplers = info->compare_samplers;
    k->h.plain_samplers = info->plain_samplers;
    k->h.nearest_samplers = info->nearest_samplers;
    k->h.face_textures = info->face_textures;
    k->h.level_textures = info->level_textures;
    memcpy(k->h.args, info->args, sizeof(k->h.args));
    memcpy(k->arg_names, info->arg_names, sizeof(k->arg_names));
    memcpy(k->h.const_samplers, info->const_samplers, sizeof(k->h.const_samplers));
    snprintf(k->name, sizeof(k->name), "shader");
}

// One compile of the module's stage into k. A shader its subroutines don't compile for (a resource passed in a way
// they can't take, too many registers live across a call) compiles again with every call inlined, and one that
// runs out of registers with some of its values kept in memory.
static int compile_once(LLVMModuleRef mod, LLVMModuleRef *lmod, const struct airc_source *linked, unsigned nlinked,
                         const struct airc_varyings *vertex, const struct airc_stage_in *stage_in,
                         const struct airc_constant *fc, unsigned nfc, int print_ir, int remat, struct airc_kernel *k)
{
    int ret = -1, calls = 1, rounds = 0;
    airc_spill_reset();
    while (ret) {
        struct ir_func f = { 0 };
        struct airc_info info = { 0 };
        if (!setjmp(g_bail)) {
            const char *names[AIRC_MAX_LINKED];
            for (unsigned i = 0; i < nlinked; i++)
                names[i] = linked[i].name;
            airc_front(mod, (void **)lmod, names, nlinked, vertex, stage_in, fc, nfc, calls, &f, &info);
            finish(&f, &info, print_ir, remat, k);
            k->error[0] = 0;
            ret = 0;
        }
        ir_reset();
        if (ret && calls)
            calls = 0;
        else if (ret && rounds++ < MAX_SPILL_ROUNDS && !strcmp(k->error, "out of registers") && airc_spill_more())
            continue;
        else if (ret)
            break;
    }
    return ret;
}

// A shader the optional passes can't get through compiles without them: a pipeline that fails takes WindowServer down.
// AIRC_NOFALLBACK=1 leaves it failed, so a test sees a pass that breaks it.
static int compile_stage(LLVMModuleRef mod, LLVMModuleRef *lmod, const struct airc_source *linked, unsigned nlinked,
                         const struct airc_varyings *vertex, const struct airc_stage_in *stage_in,
                         const struct airc_constant *fc, unsigned nfc, int print_ir, int remat, struct airc_kernel *k)
{
    int ret = compile_once(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, print_ir, remat, k);
    if (ret && !getenv("AIRC_NOFALLBACK")) {
        g_passes_off = 1;
        ret = compile_once(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, print_ir, remat, k);
        g_passes_off = 0;
    }
    return ret;
}

// warps an SM runs of a kernel with this many registers a thread (GA10x: 4 partitions of 16384, 48 warps at most)
static unsigned warps(unsigned regs)
{
    unsigned w = 4 * (16384 / (regs * 32));
    return w < 48 ? w : 48;
}

int airc_compile(const void *air, size_t len, const struct airc_constant *fc, unsigned nfc, int print_ir,
                 struct airc_kernel *k)
{
    return airc_compile_linked(air, len, NULL, 0, NULL, NULL, fc, nfc, print_ir, k);
}

int airc_compile_linked(const void *air, size_t len, const struct airc_source *linked, unsigned nlinked,
                        const struct airc_varyings *vertex, const struct airc_stage_in *stage_in,
                        const struct airc_constant *fc, unsigned nfc, int print_ir, struct airc_kernel *k)
{
    memset(k, 0, sizeof(*k));
    g_error = k->error;
    g_error_size = sizeof(k->error);
    LLVMMemoryBufferRef mb, lmb[AIRC_MAX_LINKED];
    LLVMModuleRef mod, lmod[AIRC_MAX_LINKED];
    if (nlinked > AIRC_MAX_LINKED) {
        snprintf(k->error, sizeof(k->error), "more than %d linked functions", AIRC_MAX_LINKED);
        return -1;
    }
    if (parse(air, len, &mb, &mod))
        return -1;
    for (unsigned i = 0; i < nlinked; i++)
        if (parse(linked[i].air, linked[i].len, &lmb[i], &lmod[i])) {
            while (i--) {
                LLVMDisposeModule(lmod[i]);
                LLVMDisposeMemoryBuffer(lmb[i]);
            }
            LLVMDisposeModule(mod);
            LLVMDisposeMemoryBuffer(mb);
            return -1;
        }
    int ret = compile_stage(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, print_ir, 0, k);
    // a shader that needs many registers runs few warps at a time, which leaves memory waits unhidden: compiled
    // again with values made where they are read, it is kept if that runs more warps (or as many in less code)
    if (ret || k->h.reg_count > 40) {
        struct airc_kernel *r = calloc(1, sizeof(*r));
        char *error = g_error;
        g_error = r->error;
        g_error_size = sizeof(r->error);
        int again = compile_stage(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, 0, 1, r);
        g_error = error;
        g_error_size = sizeof(k->error);
        if (!again && (ret || warps(r->h.reg_count) > warps(k->h.reg_count) ||
                       (warps(r->h.reg_count) == warps(k->h.reg_count) && r->h.sass_size < k->h.sass_size))) {
            free(k->sass);
            *k = *r;
            ret = 0;
        } else {
            free(r->sass);
        }
        free(r);
    }
    for (unsigned i = 0; i < nlinked; i++) {
        LLVMDisposeModule(lmod[i]);
        LLVMDisposeMemoryBuffer(lmb[i]);
    }
    LLVMDisposeModule(mod);
    LLVMDisposeMemoryBuffer(mb);
    return ret;
}

int airc_generate(void (*gen)(const void *arg, struct ir_func *f, struct airc_info *info), const void *arg,
                  struct airc_kernel *k)
{
    memset(k, 0, sizeof(*k));
    g_error = k->error;
    g_error_size = sizeof(k->error);
    int ret = 0;
    struct ir_func f = { 0 };
    struct airc_info info = { 0 };
    if (setjmp(g_bail)) {
        ret = -1;
    } else {
        gen(arg, &f, &info);
        finish(&f, &info, 0, 0, k);
    }
    ir_reset();
    return ret;
}
