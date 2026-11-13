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

// for irsim
void (*airc_ir_hook)(struct ir_func *f, const struct airc_info *info);

unsigned airc_samples = 1;

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
    k->h.float_inputs = info->float_inputs;
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

// Subroutines can fail on a resource passed in a way they can't take or too many registers live across a call,
// so that retries with every call inlined.
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

// A failed pipeline takes WindowServer down, so retry without optional passes; AIRC_NOFALLBACK lets tests see it fail.
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

// GA10x has 4 partitions of 16384 registers and 48 warps at most
static unsigned warps(unsigned regs)
{
    unsigned w = 4 * (16384 / (regs * 32));
    return w < 48 ? w : 48;
}

// retries without the passes that can cost warps and keeps whichever runs the most
static int compile_best(LLVMModuleRef mod, LLVMModuleRef *lmod, const struct airc_source *linked, unsigned nlinked,
                        const struct airc_varyings *vertex, const struct airc_stage_in *stage_in,
                        const struct airc_constant *fc, unsigned nfc, int print_ir, struct airc_kernel *k)
{
    g_error = k->error;
    g_error_size = sizeof(k->error);
    airc_no_unroll = airc_unrolled = 0;
    airc_no_texvec = airc_texvec_shared = 0;
    int ret = compile_stage(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, print_ir, 0, k);
    int batched = airc_batched;
    // Unrolling holds more values at once but keeps more loads in flight, which small dispatches gain from, so the
    // rolled version only wins at 8/7 the warps.
    if (!ret && airc_unrolled) {
        struct airc_kernel *r = calloc(1, sizeof(*r));
        char *error = g_error;
        g_error = r->error;
        g_error_size = sizeof(r->error);
        airc_no_unroll = 1;
        int again = compile_stage(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, 0, 0, r);
        g_error = error;
        g_error_size = sizeof(k->error);
        if (!again && 8 * warps(k->h.reg_count) < 7 * warps(r->h.reg_count)) {
            free(k->sass);
            *k = *r;
            batched = airc_batched;
        } else {
            free(r->sass);
            airc_no_unroll = 0;
        }
        free(r);
    }
    // few warps leave memory waits unhidden, so try remat
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
            batched = airc_batched;
        } else {
            free(r->sass);
        }
        free(r);
    }
    // shared texture operands live longer and can cost a register granule; 40 registers or fewer run all 48 warps
    for (int again_remat = 0; again_remat < 2 && !ret && airc_texvec_shared && k->h.reg_count > 40; again_remat++) {
        struct airc_kernel *r = calloc(1, sizeof(*r));
        char *error = g_error;
        g_error = r->error;
        g_error_size = sizeof(r->error);
        airc_no_texvec = 1;
        int again = compile_stage(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, 0, again_remat, r);
        airc_no_texvec = 0;
        g_error = error;
        g_error_size = sizeof(k->error);
        if (!again && warps(r->h.reg_count) > warps(k->h.reg_count)) {
            free(k->sass);
            *k = *r;
            batched = airc_batched;
        } else {
            free(r->sass);
        }
        free(r);
    }
    airc_no_unroll = airc_no_texvec = 0;
    airc_batched = batched;
    return ret;
}

int airc_compile(const void *air, size_t len, const struct airc_constant *fc, unsigned nfc, int print_ir,
                 struct airc_kernel *k)
{
    return airc_compile_linked(air, len, NULL, 0, NULL, NULL, 1, fc, nfc, print_ir, k);
}

int airc_compile_linked(const void *air, size_t len, const struct airc_source *linked, unsigned nlinked,
                        const struct airc_varyings *vertex, const struct airc_stage_in *stage_in, unsigned samples,
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
    airc_samples = samples;
    int ret = compile_best(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, print_ir, k);
    // batched fetches hold their results longer and can cost a register granule too
    if (!ret && airc_batched && k->h.reg_count > 40) {
        struct airc_kernel *r = calloc(1, sizeof(*r));
        airc_batch = 0;
        if (!compile_best(mod, lmod, linked, nlinked, vertex, stage_in, fc, nfc, 0, r) &&
            warps(r->h.reg_count) > warps(k->h.reg_count)) {
            free(k->sass);
            *k = *r;
        } else {
            free(r->sass);
        }
        free(r);
        airc_batch = 4;
        g_error = k->error;
        g_error_size = sizeof(k->error);
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
