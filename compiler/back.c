// airc v2 back end: IR -> machine instructions on virtual registers -> linear-scan register allocation ->
// SASS. Predicates live as 0/1 integers in GPRs (a single scratch predicate P0 is used inside a compare,
// select or branch and never stays live), so there is only one register bank to allocate. A fragment shader that
// discards also keeps P6 for the whole program: set once the thread has demoted to a helper.
#include <stdlib.h>
#include <string.h>
#include "airc_int.h"
#include "../core/sph.h"
#include "sass86.h"

enum mkind {
    MK_MOVIMM,   // def = imm
    MK_MOV,      // def = src0
    MK_MOVCBUF,  // def = c[0][imm]
    MK_LDC,      // def = c[0][src0 + imm]
    MK_ULDC,     // UR4 = c[0][imm]
    MK_S2R,      // def = SR[imm]
    MK_ALD,      // def = a[imm] of vertex src0 if there is one (aux IR_ATTR_*)
    MK_IPA,      // def = a[imm], interpolated (aux: flat)
    MK_ISBERD,   // def = the input vertex handle at map offset src0
    MK_IADD3,    // def = src0 + src1
    MK_ISUB,     // def = src0 - src1
    MK_IMAD,     // def = src0 * src1
    MK_IMADHI,   // def = the high word of src0 * src1 (aux: signed)
    MK_LOP3,     // def = src0 lut src1 (aux = lut)
    MK_SHL,      // def = src0 << src1
    MK_SHR,      // def = src0 >> src1 (aux: arithmetic)
    MK_IMNMX,    // def = min/max(src0, src1) (aux bit0 max, bit1 signed)
    MK_FADD,     // aux bit0: subtract; aux bit1: half
    MK_FMUL,     // aux bit1: half
    MK_FFMA,     // aux bit1: half; bits 2:3 SASS_RND_*; bit 4: imm is the multiplier (src a, c), bit 5: the addend
    // (src a, b); FADD and FMUL with bit 4: imm is the second operand (src a only)
    MK_FMNMX,    // aux bit0: max
    MK_MUFU,     // aux = MUFU_*
    MK_I2F,      // aux: signed source
    MK_F2I,      // aux: signed result
    MK_F2H,
    MK_H2F,
    MK_FRND,     // aux = SASS_RND_*
    MK_POPC,
    MK_BREV,
    MK_FLO,
    MK_VOTE,     // def = the lanes where P0 is set (aux: all active lanes)
    MK_SHFL,     // def = src0 of another lane, aux = SASS_SHFL_* mode | c << 4; lane imm, or src1 if there is one
    MK_FSWZADD,  // def = quad swizzle add of src0 (neighbour) and src1 (pixel), aux = pattern
    MK_LOP3_IMM,  // def = src0 lut imm (aux = lut) -- for fneg/fabs sign masks
    MK_ISETP,     // P0 = src0 cond src1 (aux = enum ir_cond | IR_SIGNED)
    MK_FSETP,     // P0 = src0 fcond src1 (aux = FSETP_*)
    MK_SEL,       // def = P0 ? src0 : src1
    MK_LDG,       // def = [src0 pair + imm], aux bytes (8 or 16: def is a pair or quad)
    MK_ATOM,      // def = the old value at [src0 + imm] (a pair global, RZ or a word shared), src1 data, src2 swapped
                  // in (CAS), aux IR_ATOMA_*
    MK_STG,       // [src0 pair + imm] = src1, aux bytes (8 or 16: src1 is a pair or quad)
    MK_LDS,       // def = shared[src0 + imm] (src0 optional), aux bytes
    MK_STS,       // shared[src0 + imm] = src1, aux bytes
    MK_LDL,       // def = local[src0 + imm] (src0 optional), aux bytes (8 or 16: def is a pair or quad)
    MK_STL,       // local[src0 + imm] = src1 (a pair or quad when wide), aux bytes
    MK_BAR,       // aux: a memory fence instead
    MK_IMADWIDE,  // def pair = src0 * src1 + src2 pair (aux: signed index)
    MK_TEX,       // def quad = fetch(src0, src1) handle imm (aux: sample)
    MK_TXQ,       // def = a texture query (aux as IR_TXQ's)
    MK_SUST,      // texel (src0, src1) of texture imm = src2 quad
    MK_AST,       // a[imm] = src0.. (aux & 15 values) of vertex src1 if there is one (aux IR_ATTR_PATCH)
    MK_BRA,       // to block `target`; aux 1 = @P0
    MK_OUTPUT,    // fragment: scratch[imm + k] = src k, the colour outputs on their way to R0..
    MK_EXIT,      // fragment: the imm colour registers scratch[0..] into R0.. first
    MK_DEMOTE,    // P6 |= src0 (aux: its inverse), or P6 = true without one: the exit drops the fragment
    MK_BSSY,      // convergence barrier imm, the threads meeting again at block `target`
    MK_BSYNC,     // wait at barrier imm for the threads it was set for
    MK_CALL,      // subroutine imm (entry block `target`): xs into the window, xd out of it; aux the barrier the
                  // caller's threads meet again at after it, -1 for none
    MK_ENTRY,     // a subroutine's start: xd out of the window
    MK_RET,       // a subroutine's return: xs into the window
};

// Subroutines take their arguments and return address in a window of registers from R2: the address in R2 (R2:R3
// for the RET), the arguments and then the results from R4, each in the slot the IR gives it. A value live where a
// call, an entry or a return moves them stays out of the window, so the moves clobber nothing; a callee keeps out
// of every register live across a call to it as well, and the convergence barriers its callers hold.
#define WIN_RA 2
#define WIN_ARGS 4

// a value moved through the window at slot, or one of these
#define SLOT_RA (-1)    // the return address
#define SLOT_KEEP (-2)  // not moved: a value of the caller's the callee reads, which the call keeps live
struct xfer {
    int vreg, slot;
};

struct mop {
    enum mkind k;
    int aux;
    int64_t imm;
    int def;  // vreg, or -1
    int src[6];
    int nsrc;
    int immb;    // src1 is the immediate imm instead (LOP3, IADD3, IMNMX, FMNMX, SHL, SHR, ISETP, FSETP, SEL)
    int target;  // branch target block id, or -1
    int label;   // branch target mop index (a stub), or -1
    struct xfer *xs, *xd;  // a call's, entry's or return's sources and definitions beyond src and def
    int nxs, nxd;
    int mods;  // float sources' modifiers, MOD_NEG and MOD_ABS two bits a source (FADD, FMUL, FFMA, FSETP)
};

#define MAX_MOPS (1 << 17)
static struct mop g_mops[MAX_MOPS];
static int g_nmops;
static int g_block_mop[1 << 16];  // first mop of each block

// Volta and later don't bring a warp's threads back together after a branch on their own: a derivative (a quad
// shuffle) after an if/else would read lanes still on the other side. A conditional branch whose paths meet again
// without a loop between sets a convergence barrier (BSSY) to that block, and the block waits at it (BSYNC); so
// does the branch into a loop, to where the lanes meet after it.
static int g_bssy[1 << 16];     // the barrier a block's branch sets, -1 for none
static int g_bssy_to[1 << 16];  // and the block the paths meet at
static int g_bsync[1 << 16][4];  // the barriers a block waits at first, innermost first
static int g_nbsync[1 << 16];

// is any block reachable from b without passing m at or before b in the block order (a loop)?
static int loops_before(struct ir_block *b, struct ir_block *m, const int *order, char *seen, int start)
{
    for (unsigned i = 0; i < b->nsucc; i++) {
        struct ir_block *s = b->succ[i];
        if (s == m)
            continue;
        if (order[s->id] <= start)
            return 1;
        if (seen[s->id])
            continue;
        seen[s->id] = 1;
        if (loops_before(s, m, order, seen, start))
            return 1;
    }
    return 0;
}

// marks the blocks of loop h that b is in: those reaching b back up without passing h
static void loop_blocks(struct ir_block *b, struct ir_block *h, char *in)
{
    if (in[b->id])
        return;
    in[b->id] = 1;
    if (b == h)
        return;
    for (unsigned i = 0; i < b->npreds; i++)
        loop_blocks(b->preds[i], h, in);
}

// the barriers of one function (the shader's, or subroutine region - 1's); how many it holds at most at once
static int plan_convergence(struct ir_func *f, int region)
{
    int nb = f->nblocks, words = (nb + 63) / 64, used = 0;
    int *order = calloc(nb, sizeof(*order));
    struct ir_block **byid = calloc(nb, sizeof(*byid)), **rpo = calloc(nb, sizeof(*rpo));
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        if (!b->dead && b->func == region) {
            order[b->id] = n;
            rpo[n++] = b;
            byid[b->id] = b;
            g_bssy[b->id] = -1;
            g_nbsync[b->id] = 0;
        }
    // post-dominators: each block's set, iterated to a fixed point (the blocks are in reverse postorder)
    uint64_t *pdom = malloc((size_t)nb * words * 8);
    for (int i = 0; i < nb; i++)
        memset(pdom + (size_t)i * words, 0xff, words * 8);
    for (int changed = 1; changed;) {
        changed = 0;
        for (int k = n - 1; k >= 0; k--) {
            struct ir_block *b = rpo[k];
            uint64_t set[words];
            if (!b->nsucc)
                memset(set, 0, sizeof(set));
            else
                memcpy(set, pdom + (size_t)b->succ[0]->id * words, sizeof(set));
            for (unsigned s = 1; s < b->nsucc; s++)
                for (int w = 0; w < words; w++)
                    set[w] &= pdom[(size_t)b->succ[s]->id * words + w];
            set[b->id / 64] |= 1ull << (b->id % 64);
            if (memcmp(set, pdom + (size_t)b->id * words, sizeof(set))) {
                memcpy(pdom + (size_t)b->id * words, set, sizeof(set));
                changed = 1;
            }
        }
    }
    // a branch's paths meet at its nearest strict post-dominator: the one post-dominated by all the others
    char *seen = calloc(nb, 1), *in_loop = calloc(nb, 1);
    int stack[16], depth = 0, ok = 1;
    for (int k = 0; k < n && ok; k++) {
        struct ir_block *b = rpo[k];
        while (depth && g_bssy_to[stack[depth - 1]] == b->id) {
            if (g_nbsync[b->id] == 4) {
                ok = 0;
                break;
            }
            g_bsync[b->id][g_nbsync[b->id]++] = g_bssy[stack[--depth]];
        }
        // a branch, or the way into a loop: lanes leave a loop when they're done with it, and a barrier or a
        // shuffle after it must not run before the rest have
        struct ir_block *from = b;
        int loop = 0;
        if (b->last && b->last->op == IR_BR) {
            from = b->succ[0];
            if (order[from->id] <= k)
                continue;
            memset(in_loop, 0, nb);
            for (unsigned p = 0; p < from->npreds; p++)
                if (byid[from->preds[p]->id] && order[from->preds[p]->id] >= order[from->id])
                    loop_blocks(from->preds[p], from, in_loop);
            if (!in_loop[from->id])
                continue;
            loop = 1;
        } else if (!b->last || b->last->op != IR_CBR) {
            continue;
        }
        // a loop's lanes meet where it's left for good, past every block of it
        struct ir_block *m = NULL;
        int best = -1;
        for (int i = 0; i < nb; i++) {
            if (i == from->id || !byid[i] || !(pdom[(size_t)from->id * words + i / 64] >> (i % 64) & 1) ||
                (loop && in_loop[i]))
                continue;
            int size = 0;
            for (int w = 0; w < words; w++)
                size += __builtin_popcountll(pdom[(size_t)i * words + w]);
            if (size > best) {
                best = size;
                m = byid[i];
            }
        }
        if (!m || order[m->id] <= k)
            continue;
        int inside = 0;  // and so the regions within it close before its own
        for (int i = 0; loop && i < nb; i++)
            inside |= in_loop[i] && order[i] > order[m->id];
        if (inside)
            continue;
        memset(seen, 0, nb);
        if (loops_before(from, m, order, seen, k))
            continue;
        // a switch's chain of compares, an else-if: the open region's barrier brings these paths together too
        if (depth && g_bssy_to[stack[depth - 1]] == m->id)
            continue;
        if (depth && order[m->id] > order[g_bssy_to[stack[depth - 1]]]) {
            ok = 0;  // not inside the open region: give up on barriers for the function
            break;
        }
        if (depth == 15)
            continue;
        g_bssy[b->id] = depth;
        g_bssy_to[b->id] = m->id;
        stack[depth++] = b->id;
        if (depth > used)
            used = depth;
    }
    if (!ok || depth) {
        for (int k = 0; k < n; k++) {
            g_bssy[rpo[k]->id] = -1;
            g_nbsync[rpo[k]->id] = 0;
        }
        used = 0;
    }
    free(order);
    free(byid);
    free(rpo);
    free(pdom);
    free(seen);
    free(in_loop);
    return used;
}

static struct ir_func *g_fn;
static int g_nregions;                   // the shader and its subroutines
static uint32_t g_callees[IR_MAX_SUBS + 1];  // the regions each one calls, a bit each
static int g_topo[IR_MAX_SUBS + 1];      // callers before callees
static int g_bar_base[IR_MAX_SUBS + 1];  // barriers below it are its callers'
static int g_wrap_bar[IR_MAX_SUBS + 1];  // the barrier its calls reconverge at, -1 for none

static void topo_visit(int r, char *seen, int *n)
{
    seen[r] = 1;
    for (int s = 0; s < g_nregions; s++)
        if ((g_callees[r] >> s & 1) && !seen[s])
            topo_visit(s, seen, n);
    g_topo[--*n] = r;
}

static void plan_regions(struct ir_func *f)
{
    g_nregions = f->nsubs + 1;
    memset(g_callees, 0, sizeof(g_callees));
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v && !b->dead; v = v->next)
            if (v->op == IR_CALL)
                g_callees[b->func] |= 1u << (v->imm + 1);
    char seen[IR_MAX_SUBS + 1] = { 0 };
    int n = g_nregions;
    for (int r = 0; r < g_nregions; r++)  // those no call reaches any more have no blocks: they go anywhere
        if (!seen[r])
            topo_visit(r, seen, &n);
    int used[IR_MAX_SUBS + 1];
    for (int r = 0; r < g_nregions; r++) {
        used[r] = plan_convergence(f, r);
        g_bar_base[r] = 0;
    }
    for (int k = 0; k < g_nregions; k++) {
        int r = g_topo[k], calls = g_callees[r] != 0;
        if (g_bar_base[r] + used[r] + calls > 16) {  // too deep a chain: this one goes without its own
            for (struct ir_block *b = f->entry; b; b = b->next)
                if (b->func == r) {
                    g_bssy[b->id] = -1;
                    g_nbsync[b->id] = 0;
                }
            used[r] = 0;
        }
        g_wrap_bar[r] = calls && g_bar_base[r] + used[r] < 16 ? g_bar_base[r] + used[r] : -1;
        for (int s = 0; s < g_nregions; s++)
            if ((g_callees[r] >> s & 1) && g_bar_base[s] < g_bar_base[r] + used[r] + 1)
                g_bar_base[s] = g_bar_base[r] + used[r] + 1;
    }
}

static struct mop *emit_mop(enum mkind k)
{
    if (g_nmops == MAX_MOPS)
        airc_fail("shader too long");
    struct mop *m = &g_mops[g_nmops++];
    memset(m, 0, sizeof(*m));
    m->k = k;
    m->def = -1;
    m->target = -1;
    m->label = -1;
    return m;
}

struct vreg {
    int size, align;  // in hardware registers
    int next;         // next member of a consecutive cluster, or -1
    int hw;           // assigned hardware register, -1 before
    int start, end;   // live range in mop order: first to last mop it is live at, holes and all
    int seg;          // the first of its live segments (g_segs), -1 for none
    int ir;           // the IR value this is the register of, for the spiller; -1 for none
};

#define MAX_VREGS 65536
#define VREG_RZ (-2)  // a source that reads RZ
static struct vreg g_vregs[MAX_VREGS];
static int g_nvregs;

static int new_vreg(int size, int align)
{
    if (g_nvregs == MAX_VREGS)
        airc_fail("too many values");
    struct vreg *v = &g_vregs[g_nvregs];
    v->size = size;
    v->align = align;
    v->next = -1;
    v->hw = -1;
    v->start = -1;
    v->end = -1;
    v->seg = -1;
    v->ir = -1;
    return g_nvregs++;
}

// a cluster of n consecutive registers, aligned; returns the base vreg, members base..base+n-1 as vregs
static int new_cluster(int n, int align, int *members)
{
    int base = -1, prev = -1;
    for (int i = 0; i < n; i++) {
        int id = new_vreg(1, i == 0 ? align : 1);
        if (i == 0)
            base = id;
        else
            g_vregs[prev].next = id;
        members[i] = id;
        prev = id;
    }
    g_vregs[base].size = n;
    return base;
}

static struct airc_info *g_info;
static int g_one;  // a vreg holding the integer 1, for materialising booleans
static int g_block;  // the block being lowered
static int g_scratch;  // registers above the allocation: temporaries, and a fragment's colours at its exit
static int g_exit_regs;  // the most colour registers an exit hands over
static int g_exit_tmp;   // the register its moves park a value in
static int g_demote;     // the shader discards: P6 holds the demoted flag

static int mov_imm(int64_t imm)
{
    int r = new_vreg(1, 1);
    struct mop *m = emit_mop(MK_MOVIMM);
    m->def = r;
    m->imm = imm;
    return r;
}

// Constants all sit in the entry block; each block makes its own copy of a value, so none holds a register
// across the whole shader. An entry left by another block counts as free.
static struct {
    int64_t imm;
    int block, vreg;
} g_consts[1024];

static int const_vreg(int64_t imm)
{
    unsigned h = (unsigned)(imm ^ imm >> 29) * 2654435761u >> 22;
    for (unsigned i = 0; i < 1024; i++) {
        unsigned k = (h + i) & 1023;
        if (g_consts[k].block == g_block && g_consts[k].imm == imm)
            return g_consts[k].vreg;
        if (g_consts[k].block != g_block) {
            g_consts[k].imm = imm;
            g_consts[k].block = g_block;
            return g_consts[k].vreg = mov_imm(imm);
        }
    }
    return mov_imm(imm);
}

// the vreg of an IR value (its lane). An I64 value is a pair cluster; its vreg is the lo member.
static int vreg_of(struct ir_val *v)
{
    if (v->op == IR_CONST || v->op == IR_UNDEF)
        return const_vreg(v->op == IR_CONST ? v->imm : 0);
    if (v->vreg < 0)
        airc_fail("value %%%d used before it is produced (op %d)", v->id, v->op);
    return v->vreg;
}

// the pair of a 64-bit value, a constant's made here
static int pair_of(struct ir_val *v)
{
    if (v->op != IR_CONST && v->op != IR_UNDEF)
        return vreg_of(v);
    int m[2], r = new_cluster(2, 2, m);
    for (int k = 0; k < 2; k++) {
        struct mop *mov = emit_mop(MK_MOVIMM);
        mov->def = m[k];
        mov->imm = v->op == IR_CONST ? (uint32_t)((uint64_t)v->imm >> 32 * k) : 0;
    }
    return r;
}

// How many times each IR value is read, and whether a value that is read once and was made a few mops back may be
// made straight into the register it is being copied to (an operand of a texture op, a wide store, a vertex
// output, which sit in registers of their own in a row). AIRC_NODEFINTO=1 keeps the copy.
static int *g_use_count;
static int g_no_define_into;

static void count_uses(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_use_count);
    g_use_count = calloc(n, sizeof(*g_use_count));
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            for (unsigned i = 0; i < v->nsrc; i++)
                if (v->src[i])
                    g_use_count[v->src[i]->id]++;
    g_no_define_into = airc_pass_off("AIRC_NODEFINTO");
}

#define DEFINE_INTO_WINDOW 12

static void mov_into(int dst, struct ir_val *v)
{
    if (!g_no_define_into && v->op >= IR_ADD && v->op <= IR_FRND && g_use_count[v->id] == 1 && v->vreg >= 0 &&
        g_vregs[v->vreg].size == 1 && g_vregs[v->vreg].next == -1) {
        for (int i = g_nmops - 1; i >= 0 && i >= g_nmops - DEFINE_INTO_WINDOW && i >= g_block_mop[g_block]; i--) {
            struct mop *d = &g_mops[i];
            if (d->def == v->vreg && !d->nxd) {
                d->def = dst;
                v->vreg = dst;
                return;
            }
        }
    }
    struct mop *m = emit_mop(MK_MOV);
    m->def = dst;
    m->src[0] = vreg_of(v);
    m->nsrc = 1;
}

static void lower_block(struct ir_block *b);

static enum mkind binop_kind(struct ir_val *v)
{
    switch (v->op) {
    case IR_ADD:
        return MK_IADD3;
    case IR_SUB:
        return MK_ISUB;
    case IR_MUL:
        return MK_IMAD;
    case IR_MULHI:
        return MK_IMADHI;
    case IR_AND:
    case IR_OR:
    case IR_XOR:
        return MK_LOP3;
    case IR_SHL:
        return MK_SHL;
    case IR_SHR:
    case IR_SAR:
        return MK_SHR;
    case IR_MIN:
    case IR_MAX:
        return MK_IMNMX;
    case IR_FADD:
    case IR_FSUB:
        return MK_FADD;
    case IR_FMUL:
        return MK_FMUL;
    case IR_FMIN:
    case IR_FMAX:
        return MK_FMNMX;
    default:
        airc_fail("not a binary op (%d)", v->op);
    }
}

#define LOP3_AND 0xc0
#define LOP3_OR 0xfc
#define LOP3_XOR 0x3c

// compares used only as selects' conditions, or as the condition of a branch in their own block: each makes P0 from
// the compare's operands itself, where a 0/1 register would cost a SEL and an ISETP more (and their latencies, the
// predicate written twice before the branch can read it). AIRC_NOFUSEBR=1 leaves the branches out.
static unsigned char *g_cmp_fused;

static void find_fused_compares(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_cmp_fused);
    g_cmp_fused = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            g_cmp_fused[v->id] = v->op == IR_CMP;
    int branches = !airc_pass_off("AIRC_NOFUSEBR");
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            for (unsigned i = 0; i < v->nsrc; i++) {
                struct ir_val *c = v->src[i];
                if (!c || c->op != IR_CMP || (v->op == IR_SEL && i == 0))
                    continue;
                if (!(branches && v->op == IR_CBR && c->block == b))
                    g_cmp_fused[c->id] = 0;
            }
}

// A float source that is a negation or an absolute value goes into an instruction that has source modifiers as the
// modifier (FADD, FMUL, FFMA and FSETP do; FMNMX is left alone, its denormal handling with a modifier isn't known),
// and the LOP3 that would make it is left out when nothing else reads it. A modifier flips or clears the sign bit
// as the LOP3 did, so a NaN and a signed zero come through the same. AIRC_NOMODS=1 turns it off.
#define MOD_NEG 1
#define MOD_ABS 2
static unsigned char *g_mod_needed;  // by IR id: a negation or absolute value something still reads as a register
static int g_no_mods;

// v without its negations and absolute values, their effect in *mods; v itself when there is nothing to take off
// (or what is under them is a constant, which stays a constant for the immediate forms)
static struct ir_val *strip_mods(struct ir_val *v, int *mods)
{
    int inner = 0;
    *mods = 0;
    if (g_no_mods || !v || (v->op != IR_FNEG && v->op != IR_FABS) || v->type != IR_F32)
        return v;
    struct ir_val *b = strip_mods(v->src[0], &inner);
    if (b->op == IR_CONST || b->op == IR_UNDEF)
        return v;
    *mods = v->op == IR_FNEG ? inner ^ MOD_NEG : MOD_ABS;
    return b;
}

static int takes_mods(const struct ir_val *u)
{
    if (u->type == IR_F32)
        return u->op == IR_FADD || u->op == IR_FSUB || u->op == IR_FMUL || u->op == IR_FFMA;
    return u->op == IR_CMP && (u->aux & IR_FCMP) && u->src[0]->type == IR_F32;
}

static void find_needed_mods(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_mod_needed);
    g_mod_needed = calloc(n, 1);
    for (int changed = 1; changed;) {
        changed = 0;
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *u = b->first; u; u = u->next)
                for (unsigned i = 0; i < u->nsrc; i++) {
                    struct ir_val *s = u->src[i];
                    int m;
                    if (!s || (s->op != IR_FNEG && s->op != IR_FABS) || g_mod_needed[s->id])
                        continue;
                    int absorbed = takes_mods(u) && strip_mods(s, &m) != s;
                    int chained = (u->op == IR_FNEG || u->op == IR_FABS) && !g_mod_needed[u->id];
                    if (!absorbed && !chained) {
                        g_mod_needed[s->id] = 1;
                        changed = 1;
                    }
                }
    }
}

// A float rounded to half and straight back to float (a half texture sample used as a float) is two instructions a
// value, F2FP and HADD2.F32. Two such roundings that sit near each other make one F2FP.PACK_AB between them, the
// halves read back from the low and the high half: the roundings are the same conversions. Only a half that
// nothing but a conversion back reads is paired, so nothing sees the other half. AIRC_NOPACKH=1 turns it off.
static struct ir_val **g_pair;  // by IR id: the F2H a second of a pair is packed with
static unsigned char *g_pair_hi, *g_first;  // the second of the two, and the first

static void find_half_pairs(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_pair);
    free(g_pair_hi);
    free(g_first);
    g_pair = calloc(n, sizeof(*g_pair));
    g_pair_hi = calloc(n, 1);
    g_first = calloc(n, 1);
    unsigned char *bad = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *u = b->first; u; u = u->next)
            for (unsigned i = 0; i < u->nsrc; i++)
                if (u->src[i] && u->src[i]->op == IR_F2H && u->op != IR_H2F)
                    bad[u->src[i]->id] = 1;
    if (!airc_pass_off("AIRC_NOPACKH"))
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *a = b->first; a; a = a->next) {
                if (a->op != IR_F2H || bad[a->id] || g_pair[a->id] || g_pair_hi[a->id] || g_first[a->id])
                    continue;
                int seen = 0;
                struct ir_val *c = a->next;
                for (; c && seen < 64; c = c->next, seen++) {
                    int uses_a = 0;
                    for (unsigned i = 0; i < c->nsrc; i++)
                        uses_a |= c->src[i] == a;
                    if (uses_a)
                        break;
                    if (c->op == IR_F2H && !bad[c->id] && !g_pair[c->id] && !g_pair_hi[c->id]) {
                        // nothing reads a before c, so both are made at c
                        g_pair[c->id] = a;
                        g_pair_hi[c->id] = 1;
                        g_first[a->id] = 1;
                        break;
                    }
                }
            }
    free(bad);
}

// a constant an instruction takes as its immediate rather than in a register
static int is_imm(const struct ir_val *v)
{
    return v->op == IR_CONST && v->type != IR_I64;
}

// P0 = the compare c, the constant side as an immediate (so on the right: its operands swapped if need be)
static void emit_setp(const struct ir_val *c)
{
    struct ir_val *a = c->src[0], *b = c->src[1];
    int aux = c->aux;
    if (is_imm(a) && !is_imm(b)) {
        struct ir_val *t = a;
        a = b;
        b = t;
        if (aux & IR_FCMP)
            aux = (aux & ~6) | (aux & 2) << 1 | (aux & 4) >> 1;
        else
            aux = (aux & ~7) | ((int[]){ IR_EQ, IR_NE, IR_GT, IR_GE, IR_LT, IR_LE }[aux & 7]);
    }
    struct mop *s = emit_mop(aux & IR_FCMP ? MK_FSETP : MK_ISETP);
    if (takes_mods(c)) {
        int ma, mb;
        a = strip_mods(a, &ma);
        b = strip_mods(b, &mb);
        s->mods = ma | mb << 2;
    }
    s->src[0] = vreg_of(a);
    s->nsrc = 1;
    if (is_imm(b)) {
        s->immb = 1;
        s->imm = (uint32_t)b->imm;
    } else {
        s->src[s->nsrc++] = vreg_of(b);
    }
    s->aux = aux;
}

static void lower_value(struct ir_val *v)
{
    if (v->vreg >= 0)
        return;  // a prologue value, already produced
    int half = v->type == IR_F16;
    switch (v->op) {
    case IR_CONST:
    case IR_UNDEF:
        break;  // made where used (vreg_of)
    case IR_CBUF:
        if (v->type == IR_I64) {
            int m[2];
            v->vreg = new_cluster(2, 2, m);
            struct mop *lo = emit_mop(MK_MOVCBUF);
            lo->def = m[0];
            lo->imm = v->imm;
            struct mop *hi = emit_mop(MK_MOVCBUF);
            hi->def = m[1];
            hi->imm = v->imm + 4;
        } else {
            v->vreg = new_vreg(1, 1);
            struct mop *m = emit_mop(v->nsrc ? MK_LDC : MK_MOVCBUF);
            m->def = v->vreg;
            m->imm = v->imm;
            if (v->nsrc)
                m->src[m->nsrc++] = vreg_of(v->src[0]);
        }
        break;
    case IR_S2R: {
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_S2R);
        m->def = v->vreg;
        m->imm = v->imm;
        break;
    }
    case IR_ALD:
    case IR_IPA: {
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(v->op == IR_ALD ? MK_ALD : MK_IPA);
        m->def = v->vreg;
        m->imm = v->imm;
        m->aux = v->aux;
        if (v->nsrc) {
            m->src[0] = vreg_of(v->src[0]);
            m->nsrc = 1;
        }
        break;
    }
    case IR_ISBERD: {
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_ISBERD);
        m->def = v->vreg;
        m->src[0] = vreg_of(v->src[0]);
        m->nsrc = 1;
        break;
    }
    case IR_ADD:
    case IR_SUB:
    case IR_MUL:
    case IR_MULHI:
    case IR_AND:
    case IR_OR:
    case IR_XOR:
    case IR_SHL:
    case IR_SHR:
    case IR_SAR:
    case IR_MIN:
    case IR_MAX:
    case IR_FADD:
    case IR_FSUB:
    case IR_FMUL:
    case IR_FMIN:
    case IR_FMAX: {
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(binop_kind(v));
        m->def = v->vreg;
        struct ir_val *fa = v->src[0], *fb = v->src[1];
        // a float constant as the second operand goes in the instruction rather than a register
        int imm = !half && (v->op == IR_FADD || v->op == IR_FSUB || v->op == IR_FMUL);
        if (imm && fa->op == IR_CONST && fb->op != IR_CONST && v->op != IR_FSUB) {
            fa = v->src[1];
            fb = v->src[0];
        }
        // c - x is -x + c in the instruction: the first source negated
        int neg_a = 0;
        if (imm && v->op == IR_FSUB && fa->op == IR_CONST && fb->op != IR_CONST && !airc_pass_off("AIRC_NONEGA")) {
            fa = v->src[1];
            fb = v->src[0];
            neg_a = 1;
        }
        imm = imm && fb->op == IR_CONST && fa->op != IR_CONST;
        // an integer or logic operand as the immediate: the first one if the operation doesn't care which
        int immb = 0;
        if (!imm && !(v->op == IR_FADD || v->op == IR_FSUB || v->op == IR_FMUL) &&
            (m->k == MK_LOP3 || m->k == MK_IADD3 || m->k == MK_IMNMX || m->k == MK_FMNMX || m->k == MK_ISUB ||
             m->k == MK_SHL || m->k == MK_SHR) &&
            v->type != IR_I64) {
            if (is_imm(fa) && !is_imm(fb) && m->k != MK_ISUB && m->k != MK_SHL && m->k != MK_SHR) {
                fa = v->src[1];
                fb = v->src[0];
            }
            immb = is_imm(fb) && !is_imm(fa);
        }
        if (!half && (m->k == MK_FADD || m->k == MK_FMUL)) {
            int ma, mb;
            fa = strip_mods(fa, &ma);
            fb = strip_mods(fb, &mb);
            m->mods = ma | mb << 2;
        }
        m->src[0] = vreg_of(fa);
        m->nsrc = 1;
        if (immb) {
            m->immb = 1;
            m->imm = (uint32_t)fb->imm;
            if (m->k == MK_ISUB) {  // x - c is x + -c
                m->k = MK_IADD3;
                m->imm = (uint32_t)-(int32_t)fb->imm;
            }
        } else if (imm) {
            m->imm = v->op == IR_FSUB && !neg_a ? (uint32_t)fb->imm ^ 0x80000000u : (uint32_t)fb->imm;
        } else {
            m->src[m->nsrc++] = vreg_of(fb);
        }
        if (m->k == MK_LOP3)
            m->aux = v->op == IR_AND ? LOP3_AND : v->op == IR_OR ? LOP3_OR : LOP3_XOR;
        else if (m->k == MK_SHR)
            m->aux = v->op == IR_SAR;
        else if (m->k == MK_IMNMX)
            m->aux = (v->op == IR_MAX) | (v->aux ? 2 : 0);
        else if (m->k == MK_FADD) {
            m->aux = (v->op == IR_FSUB && !imm) | (half ? 2 : 0) | (imm ? 16 : 0) | (neg_a ? 128 : 0);
            // the operand made constant can reach here as a register: .SAT works on either form
            if (v->op == IR_FADD && (v->aux & IR_FADD_SAT))
                m->aux |= 64;
        } else if (m->k == MK_FMUL)
            m->aux = (half ? 2 : 0) | (imm ? 16 : 0);
        else if (m->k == MK_FMNMX)
            m->aux = v->op == IR_FMAX;
        else if (m->k == MK_IMADHI)
            m->aux = v->aux;
        break;
    }
    case IR_FFMA: {
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_FFMA);
        m->def = v->vreg;
        struct ir_val *x[3] = { v->src[0], v->src[1], v->src[2] };
        int flag = 0;  // a lone float constant goes in the instruction: as the addend, or as the multiplier
        if (!half) {
            if (x[0]->op == IR_CONST && x[1]->op != IR_CONST) {
                x[0] = v->src[1];
                x[1] = v->src[0];
            }
            if (x[0]->op != IR_CONST && x[1]->op != IR_CONST && x[2]->op == IR_CONST) {
                flag = 32;
            } else if (x[0]->op != IR_CONST && x[1]->op == IR_CONST && x[2]->op != IR_CONST) {
                flag = 16;
                x[1] = x[2];
                x[2] = v->src[0]->op == IR_CONST ? v->src[0] : v->src[1];
            } else if (!airc_pass_off("AIRC_NOFFMA2C") && x[0]->op != IR_CONST && x[1]->op == IR_CONST &&
                       x[2]->op == IR_CONST) {  // two constants: one in the instruction, one in a register
                flag = 32;
            } else if (!airc_pass_off("AIRC_NOFFMA2C") && x[0]->op == IR_CONST && x[1]->op == IR_CONST &&
                       x[2]->op != IR_CONST) {
                struct ir_val *c = x[1];
                flag = 16;
                x[1] = x[2];
                x[2] = c;
            }
        }
        if (!half)
            for (int k = 0; k < 3; k++) {
                int mk;
                x[k] = strip_mods(x[k], &mk);
                m->mods |= mk << 2 * k;
            }
        m->src[0] = vreg_of(x[0]);
        m->src[1] = vreg_of(x[1]);
        m->nsrc = 2;
        if (flag == 32) {
            m->imm = (uint32_t)x[2]->imm;
        } else if (flag == 16) {
            m->imm = (uint32_t)x[2]->imm;
        } else {
            m->src[m->nsrc++] = vreg_of(x[2]);
        }
        m->aux = (half ? 2 : 0) | v->aux << 2 | flag;
        break;
    }
    case IR_FNEG:
    case IR_FABS: {
        if (v->type == IR_F32 && !g_mod_needed[v->id])  // every reader takes it as a modifier
            break;
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_LOP3_IMM);
        m->def = v->vreg;
        m->src[0] = vreg_of(v->src[0]);
        m->nsrc = 1;
        if (v->op == IR_FNEG) {
            m->aux = LOP3_XOR;
            m->imm = half ? 0x8000 : 0x80000000u;
        } else {
            m->aux = LOP3_AND;
            m->imm = half ? 0x7fff : 0x7fffffffu;
        }
        break;
    }
    case IR_MUFU:
    case IR_I2F:
    case IR_F2I:
    case IR_F2H:
    case IR_H2F:
    case IR_FRND:
    case IR_POPC:
    case IR_BREV:
    case IR_FLO: {
        if (v->op == IR_F2H && g_first[v->id])  // made with the one it is paired with, later
            break;
        if (v->op == IR_F2H && g_pair[v->id]) {  // the second of a pair: both halves of one register
            struct ir_val *lo = g_pair[v->id];
            v->vreg = lo->vreg = new_vreg(1, 1);
            struct mop *m = emit_mop(MK_F2H);
            m->def = v->vreg;
            m->src[0] = vreg_of(lo->src[0]);
            m->src[1] = vreg_of(v->src[0]);
            m->nsrc = 2;
            break;
        }
        v->vreg = new_vreg(1, 1);
        enum mkind k = v->op == IR_MUFU ? MK_MUFU
                       : v->op == IR_I2F ? MK_I2F
                       : v->op == IR_F2I ? MK_F2I
                       : v->op == IR_F2H ? MK_F2H
                       : v->op == IR_H2F ? MK_H2F
                       : v->op == IR_POPC ? MK_POPC
                       : v->op == IR_BREV ? MK_BREV
                       : v->op == IR_FLO ? MK_FLO
                                         : MK_FRND;
        struct mop *m = emit_mop(k);
        m->def = v->vreg;
        m->src[0] = vreg_of(v->src[0]);
        m->nsrc = 1;
        m->aux = v->aux;
        if (v->op == IR_H2F && v->src[0]->op == IR_F2H && g_pair_hi[v->src[0]->id])
            m->aux |= 1;
        break;
    }
    case IR_SHFL: {  // the whole warp or a quad: clamped at its first lane shifting up, its last otherwise
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_SHFL);
        m->def = v->vreg;
        m->src[0] = vreg_of(v->src[0]);
        m->nsrc = 1;
        if (v->src[1]->op == IR_CONST)
            m->imm = v->src[1]->imm & 31;
        else
            m->src[m->nsrc++] = vreg_of(v->src[1]);
        m->aux = (v->aux & 3) | ((v->aux & IR_SHFL_QUAD ? 0x1c00 : 0) | ((v->aux & 3) == SASS_SHFL_UP ? 0 : 0x1f)) << 4;
        break;
    }
    case IR_DERIV: {  // the quad neighbour across x (lane ^ 1) or y (lane ^ 2), then a swizzled subtract
        int n = new_vreg(1, 1);
        struct mop *sh = emit_mop(MK_SHFL);
        sh->def = n;
        sh->src[0] = vreg_of(v->src[0]);
        sh->nsrc = 1;
        sh->imm = v->aux ? 2 : 1;
        sh->aux = SASS_SHFL_BFLY | 0x1c03 << 4;  // quads of 4 lanes
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_FSWZADD);
        m->def = v->vreg;
        m->src[0] = n;
        m->src[1] = vreg_of(v->src[0]);
        m->nsrc = 2;
        m->aux = v->aux ? FSWZADD_DDY : FSWZADD_DDX;
        break;
    }
    case IR_VOTE: {  // through P0, or of every active lane for a constant true
        int all = v->src[0]->op == IR_CONST && v->src[0]->imm;
        if (!all) {
            struct mop *s = emit_mop(MK_ISETP);
            s->src[0] = vreg_of(v->src[0]);
            s->src[1] = VREG_RZ;
            s->nsrc = 2;
            s->aux = IR_NE;
        }
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_VOTE);
        m->def = v->vreg;
        m->aux = all;
        break;
    }
    case IR_CMP: {
        if (g_cmp_fused[v->id])
            break;
        v->vreg = new_vreg(1, 1);
        emit_setp(v);
        struct mop *sel = emit_mop(MK_SEL);
        sel->def = v->vreg;
        sel->src[0] = g_one;
        sel->src[1] = VREG_RZ;
        sel->nsrc = 2;
        break;
    }
    case IR_SEL: {
        int m[2], wide = v->type == IR_I64;
        v->vreg = wide ? new_cluster(2, 2, m) : new_vreg(1, 1);
        struct ir_val *c = v->src[0];
        if (c->op == IR_CMP && g_cmp_fused[c->id]) {  // P0 = the compare itself
            emit_setp(c);
        } else {
            struct mop *s = emit_mop(MK_ISETP);
            s->src[0] = vreg_of(c);
            s->src[1] = VREG_RZ;
            s->nsrc = 2;
            s->aux = IR_NE;
        }
        if (!wide && (is_imm(v->src[1]) || is_imm(v->src[2]))) {  // the constant in the instruction, under !P0 if it is
            int swap = !is_imm(v->src[2]);  // the true value
            struct mop *sel = emit_mop(MK_SEL);
            sel->def = v->vreg;
            sel->src[0] = vreg_of(v->src[swap ? 2 : 1]);
            sel->nsrc = 1;
            sel->immb = 1;
            sel->imm = (uint32_t)v->src[swap ? 1 : 2]->imm;
            sel->aux = swap;
            break;
        }
        // a 64-bit value is a pair: each half selected
        for (int d = v->vreg, a = wide ? pair_of(v->src[1]) : vreg_of(v->src[1]),
                 b = wide ? pair_of(v->src[2]) : vreg_of(v->src[2]), k = 0;
             k < (wide ? 2 : 1); k++) {
            struct mop *sel = emit_mop(MK_SEL);
            sel->def = d;
            sel->src[0] = a;
            sel->src[1] = b;
            sel->nsrc = 2;
            d = g_vregs[d].next;
            a = g_vregs[a].next;
            b = g_vregs[b].next;
        }
        break;
    }
    case IR_PNOT: {
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_LOP3_IMM);
        m->def = v->vreg;
        m->src[0] = vreg_of(v->src[0]);
        m->nsrc = 1;
        m->aux = LOP3_XOR;
        m->imm = 1;
        break;
    }
    case IR_PAND:
    case IR_POR: {
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_LOP3);
        m->def = v->vreg;
        m->src[0] = vreg_of(v->src[0]);
        m->src[1] = vreg_of(v->src[1]);
        m->nsrc = 2;
        m->aux = v->op == IR_PAND ? LOP3_AND : LOP3_OR;
        break;
    }
    case IR_ADDR: {
        int m[2];
        v->vreg = new_cluster(2, 2, m);
        int scale = mov_imm(v->imm);
        struct mop *w = emit_mop(MK_IMADWIDE);
        w->def = v->vreg;
        w->src[0] = vreg_of(v->src[1]);
        w->src[1] = scale;
        w->src[2] = vreg_of(v->src[0]);
        w->nsrc = 3;
        w->aux = v->aux;
        break;
    }
    case IR_ATOM: {
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_ATOM);
        m->def = v->vreg;
        m->src[0] = v->src[0] ? vreg_of(v->src[0]) : VREG_RZ;
        m->src[1] = vreg_of(v->src[1]);
        m->src[2] = v->nsrc > 2 ? vreg_of(v->src[2]) : VREG_RZ;
        m->nsrc = 3;
        m->imm = v->imm;
        m->aux = v->aux;
        break;
    }
    case IR_LDG:
    case IR_LDS:
    case IR_LDL: {
        int q[4];
        // 8 or 16 bytes land in 2 or 4 registers aligned as many, the lanes read through IR_PROJ
        v->vreg = v->aux > 4 ? new_cluster(v->aux / 4, v->aux / 4, q) : new_vreg(1, 1);
        struct mop *m = emit_mop(v->op == IR_LDG ? MK_LDG : v->op == IR_LDS ? MK_LDS : MK_LDL);
        m->def = v->vreg;
        m->nsrc = v->nsrc;
        for (unsigned i = 0; i < v->nsrc; i++)
            m->src[i] = vreg_of(v->src[i]);
        m->imm = v->imm;
        m->aux = v->aux;
        break;
    }
    case IR_TEX: {
        // two source vectors, each aligned as a vector of its size: the handle goes first in TXD's first one,
        // first in the second one otherwise
        int txd = IR_TEXA_OP(v->aux) == SASS_TXD, n0 = IR_TEXA_N0(v->aux), nsrc = (int)v->nsrc - 1;
        int k[2] = { n0 + txd, nsrc - n0 + !txd }, vec[2], c[2][4], q[4];
        if (k[0] > 4 || k[1] > 4)
            airc_fail("a texture op with too many operands");
        for (int j = 0; j < 2; j++)
            vec[j] = new_cluster(k[j], k[j] > 2 ? 4 : k[j], c[j]);
        mov_into(c[!txd][0], v->src[nsrc]);
        for (int i = 0; i < nsrc; i++) {
            int j = i >= n0;
            mov_into(c[j][i - (j ? n0 : 0) + (j != txd)], v->src[i]);
        }
        v->vreg = new_cluster(4, 2, q);  // R,G in the first pair, B,A in the second, an even base
        struct mop *m = emit_mop(MK_TEX);
        m->def = v->vreg;
        m->src[0] = vec[0];
        m->src[1] = vec[1];
        m->nsrc = 2;
        m->aux = v->aux;
        m->imm = k[0] | k[1] << 4;
        break;
    }
    case IR_PROJ: {
        int base = vreg_of(v->src[0]);
        for (int i = 0; i < (int)v->imm; i++)
            base = g_vregs[base].next;
        v->vreg = base;
        break;
    }
    case IR_TXQ: {
        int p[2];
        int handle = new_cluster(2, 2, p);
        mov_into(p[0], v->src[1]);
        mov_into(p[1], v->src[0]);
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_TXQ);
        m->def = v->vreg;
        m->src[0] = handle;
        m->nsrc = 1;
        m->aux = v->aux;
        break;
    }
    default:
        airc_fail("cannot lower value op %d", v->op);
    }
}

static void lower_effect(struct ir_val *v)
{
    switch (v->op) {
    case IR_STG:
    case IR_STS:
    case IR_STL: {
        int data;
        if (v->aux > 4) {  // the 2 or 4 words of a wide store, gathered into registers aligned as many
            int c[4];
            data = new_cluster(v->aux / 4, v->aux / 4, c);
            for (int k = 0; k < v->aux / 4; k++)
                mov_into(c[k], v->src[1 + k]);
        } else {
            data = vreg_of(v->src[1]);
        }
        struct mop *m = emit_mop(v->op == IR_STG ? MK_STG : v->op == IR_STS ? MK_STS : MK_STL);
        m->src[0] = v->src[0] ? vreg_of(v->src[0]) : VREG_RZ;  // a shared or local store may have no base register
        m->src[1] = data;
        m->nsrc = 2;
        m->imm = v->imm;
        m->aux = v->aux;
        break;
    }
    case IR_BAR:
        emit_mop(MK_BAR)->aux = v->aux;
        break;
    case IR_KILL: {
        // Metal's discard is a demote: the thread runs on as a helper so its quad's derivatives and implicit-LOD
        // samples still see it; only its exit kills it, and its memory writes are off
        struct mop *m = emit_mop(MK_DEMOTE);
        if (v->nsrc) {
            m->src[0] = vreg_of(v->src[0]);
            m->nsrc = 1;
        }
        m->aux = v->aux;
        g_demote = 1;
        break;
    }
    case IR_SUST: {
        int c[4], q[4], n = v->aux;
        int coord = new_cluster(n, n > 2 ? 4 : n, c);  // the coordinates as a vector
        for (int i = 0; i < n; i++)
            mov_into(c[i], v->src[i]);
        int color = new_cluster(4, 4, q);
        for (int i = 0; i < 4; i++)
            mov_into(q[i], v->src[n + i]);
        struct mop *m = emit_mop(MK_SUST);
        m->src[0] = coord;
        m->src[1] = color;
        m->src[2] = vreg_of(v->src[n + 4]);
        m->nsrc = 3;
        m->aux = n;
        m->imm = v->imm;
        break;
    }
    case IR_AST: {
        int n = v->aux & 15, mm[4];
        int grp = new_cluster(n, n > 2 ? 4 : n, mm);  // AST.64 wants an even pair, .96/.128 a 4-aligned quad
        for (int i = 0; i < n; i++)
            mov_into(mm[i], v->src[i]);
        struct mop *m = emit_mop(MK_AST);
        m->src[0] = grp;
        m->nsrc = 1;
        if ((int)v->nsrc > n)
            m->src[m->nsrc++] = vreg_of(v->src[n]);
        m->aux = v->aux;
        m->imm = v->imm;
        break;
    }
    default:
        airc_fail("cannot lower effect op %d", v->op);
    }
}

// For each edge pred -> b and each phi of b, copy the phi's input from that edge into the phi's vreg. All the
// reads go to temporaries first, so a phi feeding another phi on the same edge reads the old value.
// A loop of one block carries a value round in a phi: v, made late in the block, is copied to the phi's register at
// its end. When nothing reads the phi after v is made, v can be made in the phi's register and the copy never
// happens. The readers of v after it, in the block and past the loop, read that register all the same: it holds v
// from then on. The block's own mops have all been made by now, so that is checked on them; a read in another
// block, by a phi, or by something that takes its operand later (a compare fused into a branch, a modifier, a
// paired half) rules it out beforehand (phi_outside). AIRC_NOPHIDEF=1 turns it off.
static unsigned char *g_phi_outside, *g_phi_moved;

static void phi_outside(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_phi_outside);
    free(g_phi_moved);
    g_phi_outside = calloc(n, 1);
    g_phi_moved = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *u = b->first; u; u = u->next)
            for (unsigned i = 0; i < u->nsrc; i++) {
                struct ir_val *s = u->src[i];
                if (s && s->op == IR_PHI &&
                    (u->block != s->block || u->op == IR_PHI || u->op == IR_FNEG || u->op == IR_FABS ||
                     u->op == IR_CMP || u->op == IR_F2H))
                    g_phi_outside[s->id] = 1;
            }
}

static void phi_into_def(struct ir_block *b, struct ir_val *p, struct ir_val *v)
{
    if (airc_pass_off("AIRC_NOPHIDEF") || p->type == IR_I64 || v->type != p->type || v->op < IR_ADD || v->op > IR_FRND ||
        v->block != b || p->vreg < 0 || v->vreg < 0 || g_phi_moved[v->id] || g_phi_moved[p->id] ||
        g_phi_outside[p->id] || g_vregs[v->vreg].size != 1 || g_vregs[v->vreg].next != -1 ||
        g_vregs[p->vreg].size != 1 || g_vregs[p->vreg].next != -1)
        return;
    int def = -1;
    for (int k = g_nmops - 1; k >= g_block_mop[b->id] && k >= g_nmops - 256 && def < 0; k--)
        if (g_mops[k].def == v->vreg && !g_mops[k].nxd)
            def = k;
    if (def < 0)
        return;
    for (int k = def + 1; k < g_nmops; k++) {  // the old value must not be read once v is made
        const struct mop *m = &g_mops[k];
        for (int j = 0; j < m->nsrc; j++)
            if (m->src[j] == p->vreg)
                return;
        for (int j = 0; j < m->nxs; j++)
            if (m->xs[j].vreg == p->vreg)
                return;
    }
    int old = v->vreg;
    g_mops[def].def = p->vreg;
    for (int k = def + 1; k < g_nmops; k++) {
        struct mop *m = &g_mops[k];
        for (int j = 0; j < m->nsrc; j++)
            if (m->src[j] == old)
                m->src[j] = p->vreg;
        for (int j = 0; j < m->nxs; j++)
            if (m->xs[j].vreg == old)
                m->xs[j].vreg = p->vreg;
    }
    v->vreg = p->vreg;
    g_phi_moved[v->id] = g_phi_moved[p->id] = 1;
}

static void phi_copies(struct ir_block *pred, struct ir_block *b)
{
    if (pred == b)
        for (struct ir_val *p = b->first; p && p->op == IR_PHI; p = p->next)
            for (unsigned k = 0; k < p->nsrc; k++)
                if (p->phi_blocks[k] == pred && p->src[k] && p->src[k]->op != IR_CONST && p->src[k]->op != IR_UNDEF)
                    phi_into_def(b, p, p->src[k]);
    static int mark[MAX_VREGS], gen;
    enum { MAX_COPIES = 1024 };
    int dst[MAX_COPIES], src[MAX_COPIES], tmp[MAX_COPIES], n = 0;  // src -1: a constant, its imm in imm[]
    int64_t imm[MAX_COPIES];
    for (struct ir_val *p = b->first; p && p->op == IR_PHI; p = p->next) {
        struct ir_val *in = NULL;
        for (unsigned k = 0; k < p->nsrc; k++)
            if (p->phi_blocks[k] == pred)
                in = p->src[k];
        if (!in)
            airc_fail("phi edge without a value");
        if (n + 2 > MAX_COPIES)
            airc_fail("too many phis in a block");
        int is_const = in->op == IR_CONST || in->op == IR_UNDEF;
        int64_t value = in->op == IR_CONST ? in->imm : 0;
        int s = is_const ? -1 : vreg_of(in);
        // a 64-bit value is a pair: each half copied
        for (int d = p->vreg, k = 0; k < (p->type == IR_I64 ? 2 : 1); k++) {
            dst[n] = d;
            src[n] = s;
            imm[n] = is_const && p->type == IR_I64 ? (int64_t)(uint32_t)((uint64_t)value >> (32 * k)) : value;
            n++;
            if (s >= 0)
                s = g_vregs[s].next;
            d = g_vregs[d].next;
        }
    }
    // Every copy reads the old values, so one whose source another copy overwrites takes it into a temporary first;
    // the others go straight in (a temporary for each of a hundred phis would leave no register for the rest).
    gen++;
    for (int i = 0; i < n; i++)
        mark[dst[i]] = gen;
    for (int i = 0; i < n; i++) {
        tmp[i] = -1;
        if (src[i] >= 0 && src[i] != dst[i] && mark[src[i]] == gen) {
            tmp[i] = new_vreg(1, 1);
            struct mop *m = emit_mop(MK_MOV);
            m->def = tmp[i];
            m->src[0] = src[i];
            m->nsrc = 1;
        }
    }
    for (int i = 0; i < n; i++) {
        struct mop *m;
        if (src[i] < 0) {
            m = emit_mop(MK_MOVIMM);
            m->imm = imm[i];
        } else {
            if (tmp[i] < 0 && src[i] == dst[i])
                continue;
            m = emit_mop(MK_MOV);
            m->src[0] = tmp[i] >= 0 ? tmp[i] : src[i];
            m->nsrc = 1;
        }
        m->def = dst[i];
    }
}

static void set_convergence(struct ir_block *b)
{
    if (g_bssy[b->id] < 0)
        return;
    struct mop *m = emit_mop(MK_BSSY);
    m->imm = g_bar_base[b->func] + g_bssy[b->id];
    m->target = g_bssy_to[b->id];
}

static int g_ra;       // the vreg of the return address in the subroutine being lowered
static int g_win_end;  // the register past the window's last slot

static void window_slots(int end)
{
    if (WIN_ARGS + end > g_win_end)
        g_win_end = WIN_ARGS + end;
}

// sources into the window's slots one after another, a 64-bit one taking two; extra more entries kept after them
static struct xfer *xfer_sources(struct ir_val *v, int extra)
{
    struct xfer *xs = ir_alloc((v->nsrc + extra) * sizeof(*xs));
    int slot = 0;
    for (unsigned i = 0; i < v->nsrc; i++) {
        int wide = v->src[i]->type == IR_I64;
        xs[i].vreg = wide ? pair_of(v->src[i]) : vreg_of(v->src[i]);
        xs[i].slot = slot;
        slot += wide ? 2 : 1;
    }
    window_slots(slot);
    return xs;
}

static void lower_entry(struct ir_block *b)
{
    int n = 0;
    for (struct ir_val *v = b->first; v; v = v->next)
        n += v->op == IR_PARAM;
    struct xfer *xd = ir_alloc((n + 1) * sizeof(*xd));
    n = 0;
    for (struct ir_val *v = b->first; v; v = v->next)
        if (v->op == IR_PARAM) {
            int m[2], wide = v->type == IR_I64;
            v->vreg = wide ? new_cluster(2, 2, m) : new_vreg(1, 1);
            xd[n].vreg = v->vreg;
            xd[n++].slot = v->aux;
            window_slots(v->aux + 1 + wide);
        }
    g_ra = new_vreg(1, 1);
    xd[n].vreg = g_ra;
    xd[n++].slot = SLOT_RA;
    struct mop *m = emit_mop(MK_ENTRY);
    m->xd = xd;
    m->nxd = n;
}

// results go into their IR_PROJs' registers straight from the window, IR_PROJ imm the slot
static void lower_call(struct ir_val *v)
{
    struct xfer *xs = xfer_sources(v, 0);
    int n = 0;
    for (struct ir_val *p = v->block->first; p; p = p->next)
        n += p->op == IR_PROJ && p->src[0] == v;
    struct xfer *xd = ir_alloc((n + 1) * sizeof(*xd));
    n = 0;
    for (struct ir_val *p = v->block->first; p; p = p->next)
        if (p->op == IR_PROJ && p->src[0] == v) {
            p->vreg = new_vreg(1, 1);
            xd[n].vreg = p->vreg;
            xd[n++].slot = (int)p->imm;
            window_slots((int)p->imm + 1);
        }
    struct mop *m = emit_mop(MK_CALL);
    m->imm = v->imm;
    m->target = g_fn->subs[v->imm]->id;
    m->aux = g_wrap_bar[v->block->func];
    m->xs = xs;
    m->nxs = (int)v->nsrc;
    m->xd = xd;
    m->nxd = n;
}

static void lower_ret(struct ir_val *v)
{
    struct xfer *xs = xfer_sources(v, 1);
    xs[v->nsrc].vreg = g_ra;
    xs[v->nsrc].slot = SLOT_RA;
    struct mop *m = emit_mop(MK_RET);
    m->xs = xs;
    m->nxs = (int)v->nsrc + 1;
}

static void lower_block(struct ir_block *b)
{
    g_block = b->id;
    g_block_mop[b->id] = g_nmops;
    if (b->func && g_fn->subs[b->func - 1] == b)
        lower_entry(b);
    for (int i = 0; i < g_nbsync[b->id]; i++)
        emit_mop(MK_BSYNC)->imm = g_bar_base[b->func] + g_bsync[b->id][i];
    for (struct ir_val *v = b->first; v; v = v->next) {
        // constant operands made ahead: lowering may emit its instruction before asking for their registers
        if (v->op != IR_PHI)
            for (unsigned i = 0; i < v->nsrc; i++)
                if (v->src[i] && (v->src[i]->op == IR_CONST || v->src[i]->op == IR_UNDEF))
                    vreg_of(v->src[i]);
        switch (v->op) {
        case IR_PHI:
        case IR_PARAM:
            break;
        case IR_CALL:
            lower_call(v);
            break;
        case IR_RET:
            lower_ret(v);
            break;
        case IR_STG:
        case IR_STS:
        case IR_STL:
        case IR_BAR:
        case IR_KILL:
        case IR_SUST:
        case IR_AST:
            lower_effect(v);
            break;
        case IR_BR:
            set_convergence(b);
            phi_copies(b, b->succ[0]);
            {
                struct mop *m = emit_mop(MK_BRA);
                m->target = b->succ[0]->id;
            }
            break;
        case IR_CBR: {
            set_convergence(b);
            if (v->src[0]->op == IR_CMP && g_cmp_fused[v->src[0]->id]) {
                emit_setp(v->src[0]);
            } else {
                struct mop *s = emit_mop(MK_ISETP);
                s->src[0] = vreg_of(v->src[0]);
                s->src[1] = VREG_RZ;
                s->nsrc = 2;
                s->aux = IR_NE;
            }
            // the true edge's phi copies may only run when it is taken: they get a stub after the false path
            int stub = b->succ[0]->first && b->succ[0]->first->op == IR_PHI;
            int bt = g_nmops;
            struct mop *m = emit_mop(MK_BRA);
            m->aux = 1;
            if (!stub)
                m->target = b->succ[0]->id;
            phi_copies(b, b->succ[1]);
            emit_mop(MK_BRA)->target = b->succ[1]->id;
            if (stub) {
                g_mops[bt].label = g_nmops;
                phi_copies(b, b->succ[0]);
                emit_mop(MK_BRA)->target = b->succ[0]->id;
            }
            break;
        }
        case IR_EXIT: {  // a fragment's colours four at a time into the scratch registers; EXIT moves them to R0..
            for (unsigned i = 0; i < v->nsrc; i += 4) {
                struct mop *o = emit_mop(MK_OUTPUT);
                o->nsrc = v->nsrc - i < 4 ? (int)(v->nsrc - i) : 4;
                o->imm = i;
                for (int k = 0; k < o->nsrc; k++)
                    o->src[k] = vreg_of(v->src[i + k]);
            }
            if ((int)v->nsrc > g_exit_regs)
                g_exit_regs = (int)v->nsrc;
            emit_mop(MK_EXIT)->imm = v->nsrc;
            break;
        }
        default:
            lower_value(v);
        }
    }
}

// each function's mops, the shader's first: [start, end)
static int g_region_start[IR_MAX_SUBS + 1], g_region_end[IR_MAX_SUBS + 1];

// a cluster's members for its base (the vreg alone otherwise), as a mop naming it reads or writes them all
#define MEMBERS(w, v) for (int w = (v); w >= 0; w = g_vregs[v].size > 1 ? g_vregs[w].next : -1)

static void touch(int v, int k)
{
    struct vreg *r = &g_vregs[v];
    if (r->hw >= 0)  // a caller's, allocated with it
        return;
    if (r->start < 0 || k < r->start)
        r->start = k;
    if (k > r->end)
        r->end = k;
}

// The mops a value is live at, as runs in mop order: a value used in one arm of an if is not live in the other, so
// the registers there are free for others, and the members of a cluster are each only live where they are read.
// The range above, from first to last, is these with the gaps filled.
struct seg {
    int start, end, next;  // inclusive
};
static struct seg *g_segs;
static int g_nsegs, g_cap_segs;

static int add_seg(int v, int tail, int start, int end)
{
    if (tail >= 0 && start <= g_segs[tail].end + 1) {  // the block before's run goes on
        if (end > g_segs[tail].end)
            g_segs[tail].end = end;
        return tail;
    }
    if (g_nsegs == g_cap_segs) {
        g_cap_segs = g_cap_segs ? 2 * g_cap_segs : 1 << 16;
        g_segs = realloc(g_segs, g_cap_segs * sizeof(*g_segs));
    }
    g_segs[g_nsegs] = (struct seg){ start, end, -1 };
    if (tail >= 0)
        g_segs[tail].next = g_nsegs;
    else
        g_vregs[v].seg = g_nsegs;
    return g_nsegs++;
}

// Live ranges from liveness dataflow over the blocks of one function, so a value read again around a loop's back
// edge stays live to the end of the loop. A range is one interval [start, end] in mop order, covering every block
// the value is live through. Values of the shader a subroutine reads keep the ranges they have in the shader.
static void compute_ranges(struct ir_func *f, int region)
{
    int nb = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        if (!b->dead && b->func == region)
            nb++;
    struct ir_block **blocks = malloc(nb * sizeof(*blocks));
    int *bstart = malloc(nb * sizeof(*bstart)), *bend = malloc(nb * sizeof(*bend));
    int *index = malloc(f->nblocks * sizeof(*index));
    nb = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        if (!b->dead && b->func == region) {
            index[b->id] = nb;
            blocks[nb++] = b;
        }
    for (int i = 0; i < nb; i++) {
        bstart[i] = g_block_mop[blocks[i]->id];
        bend[i] = i + 1 < nb ? g_block_mop[blocks[i + 1]->id] : g_region_end[region];
    }
    // the leading MOVIMM of the integer 1 sits before the first block: fold it into block 0
    if (nb)
        bstart[0] = g_region_start[region];

    int words = (g_nvregs + 63) / 64;
    uint64_t *use = calloc((size_t)nb * words, 8), *def = calloc((size_t)nb * words, 8);
    uint64_t *in = calloc((size_t)nb * words, 8), *out = calloc((size_t)nb * words, 8);
#define BIT(set, i, v) ((set)[(size_t)(i) * words + (v) / 64] >> ((v) % 64) & 1)
#define SETBIT(set, i, v) ((set)[(size_t)(i) * words + (v) / 64] |= 1ull << ((v) % 64))
    for (int i = 0; i < nb; i++)
        for (int k = bstart[i]; k < bend[i]; k++) {
            struct mop *m = &g_mops[k];
            for (int j = 0; j < m->nsrc + m->nxs; j++) {
                int v = j < m->nsrc ? m->src[j] : m->xs[j - m->nsrc].vreg;
                MEMBERS(w, v)
                    if (!BIT(def, i, w))
                        SETBIT(use, i, w);
            }
            for (int j = -1; j < m->nxd; j++) {
                int v = j < 0 ? m->def : m->xd[j].vreg;
                MEMBERS(w, v)
                    SETBIT(def, i, w);
            }
        }
    // iterate in reverse layout order until nothing changes
    for (int changed = 1; changed;) {
        changed = 0;
        for (int i = nb - 1; i >= 0; i--) {
            uint64_t *o = &out[(size_t)i * words], *n = &in[(size_t)i * words];
            for (unsigned s = 0; s < blocks[i]->nsucc; s++) {
                uint64_t *si = &in[(size_t)index[blocks[i]->succ[s]->id] * words];
                for (int w = 0; w < words; w++)
                    o[w] |= si[w];
            }
            for (int w = 0; w < words; w++) {
                uint64_t v = use[(size_t)i * words + w] | (o[w] & ~def[(size_t)i * words + w]);
                if (v != n[w]) {
                    n[w] = v;
                    changed = 1;
                }
            }
        }
    }

    for (int v = 0; v < g_nvregs; v++)
        if (g_vregs[v].hw < 0) {
            g_vregs[v].start = -1;
            g_vregs[v].end = -1;
        }
    // defs and uses
    for (int k = g_region_start[region]; k < g_region_end[region]; k++) {
        struct mop *m = &g_mops[k];
        if (m->def >= 0)
            touch(m->def, k);
        for (int j = 0; j < m->nsrc; j++)
            if (m->src[j] >= 0)
                touch(m->src[j], k);
        for (int j = 0; j < m->nxs; j++)
            touch(m->xs[j].vreg, k);
        for (int j = 0; j < m->nxd; j++)
            touch(m->xd[j].vreg, k);
    }
    // stretch over every block a value is live into or out of
    for (int i = 0; i < nb; i++)
        for (int v = 0; v < g_nvregs; v++) {
            int li = BIT(in, i, v), lo = BIT(out, i, v);
            if ((!li && !lo) || g_vregs[v].hw >= 0)
                continue;
            struct vreg *r = &g_vregs[v];
            int s = li ? bstart[i] : r->start, e = lo ? bend[i] - 1 : r->end;
            if (r->start < 0 || s < r->start)
                r->start = s;
            if (e > r->end)
                r->end = e;
        }
    // the runs: in a block, from its start if the value is live in (else from where it is first named) to its
    // end if live out (else to where it is last named)
    g_nsegs = 0;
    int *tail = malloc((unsigned)g_nvregs * sizeof(*tail)), *stamp = calloc((unsigned)g_nvregs, sizeof(*stamp));
    int *first = malloc((unsigned)g_nvregs * sizeof(*first)), *last = malloc((unsigned)g_nvregs * sizeof(*last));
    int *named = malloc((unsigned)g_nvregs * sizeof(*named));
    for (int v = 0; v < g_nvregs; v++) {
        tail[v] = -1;
        if (g_vregs[v].hw < 0)
            g_vregs[v].seg = -1;
    }
    for (int i = 0; i < nb; i++) {
        int nnamed = 0;
        for (int k = bstart[i]; k < bend[i]; k++) {
            struct mop *m = &g_mops[k];
            for (int j = -1; j < m->nsrc + m->nxs + m->nxd; j++) {
                int v = j < 0 ? m->def : j < m->nsrc ? m->src[j] : j < m->nsrc + m->nxs ? m->xs[j - m->nsrc].vreg
                                                                                        : m->xd[j - m->nsrc - m->nxs].vreg;
                MEMBERS(w, v) {
                    if (stamp[w] != i + 1) {
                        stamp[w] = i + 1;
                        first[w] = k;
                        named[nnamed++] = w;
                    }
                    last[w] = k;
                }
            }
        }
        for (int n = 0; n < nnamed; n++) {
            int w = named[n];
            if (g_vregs[w].hw >= 0)
                continue;
            int s = BIT(in, i, w) ? bstart[i] : first[w], e = BIT(out, i, w) ? bend[i] - 1 : last[w];
            tail[w] = add_seg(w, tail[w], s, e);
        }
        for (int w = 0; w < words; w++)  // live through without being named
            for (uint64_t bits = in[(size_t)i * words + w] & out[(size_t)i * words + w]; bits; bits &= bits - 1) {
                int v = w * 64 + __builtin_ctzll(bits);
                if (stamp[v] != i + 1 && g_vregs[v].hw < 0)
                    tail[v] = add_seg(v, tail[v], bstart[i], bend[i] - 1);
            }
    }
    free(tail);
    free(stamp);
    free(first);
    free(last);
    free(named);
#undef BIT
#undef SETBIT
    // a cluster shares one range across its members
    for (int v = 0; v < g_nvregs; v++)
        if (g_vregs[v].size > 1 && g_vregs[v].hw < 0) {
            int s = g_vregs[v].start, e = g_vregs[v].end;
            for (int j = g_vregs[v].next; j >= 0; j = g_vregs[j].next) {
                if (g_vregs[j].start >= 0 && (s < 0 || g_vregs[j].start < s))
                    s = g_vregs[j].start;
                if (g_vregs[j].end > e)
                    e = g_vregs[j].end;
            }
            for (int j = v; j >= 0; j = g_vregs[j].next) {
                g_vregs[j].start = s;
                g_vregs[j].end = e;
            }
        }
    free(blocks);
    free(bstart);
    free(bend);
    free(index);
    free(use);
    free(def);
    free(in);
    free(out);
}
// A shader that runs out of registers is compiled again with some of the values that were live at the failure kept
// in local memory: stored right after they're made and loaded again before each use, so that what stays in a
// register is a short range. The values are the IR ones' ids, which a compile of the same shader makes the same.

#define SPILL_ROUND 64   // values added each time
#define SPILL_MAX 2048
static int g_spill_ids[SPILL_MAX];
static unsigned g_nspill_ids;
static int g_cand[SPILL_ROUND];
static unsigned g_ncand;
static int g_spill_limit;  // ids from here on were made by spilling, and aren't spilled themselves

void airc_spill_reset(void)
{
    g_nspill_ids = 0;
    g_ncand = 0;
    g_spill_limit = 0;
}

// The live values at the failure that have an IR value of their own and the longest ranges, longest first.
static void note_spill_candidates(const int *active, int na)
{
    g_ncand = 0;
    for (int i = 0; i < na; i++) {
        const struct vreg *v = &g_vregs[active[i]];
        if (v->ir < 0 || (g_spill_limit && v->ir >= g_spill_limit))
            continue;
        unsigned k = 0;
        while (k < g_nspill_ids && g_spill_ids[k] != v->ir)
            k++;
        if (k < g_nspill_ids)  // already out in memory, and still short of registers
            continue;
        int len = v->end - v->start, at = (int)g_ncand;
        while (at > 0 && g_vregs[g_cand[at - 1]].end - g_vregs[g_cand[at - 1]].start < len)
            at--;
        if (at >= SPILL_ROUND)
            continue;
        if (g_ncand < SPILL_ROUND)
            g_ncand++;
        for (int j = (int)g_ncand - 1; j > at; j--)
            g_cand[j] = g_cand[j - 1];
        g_cand[at] = active[i];
    }
    for (unsigned i = 0; i < g_ncand; i++)
        g_cand[i] = g_vregs[g_cand[i]].ir;
}

// the next round's spills: the failure's candidates; 0 when there are none new
int airc_spill_more(void)
{
    int added = 0;
    for (unsigned i = 0; i < g_ncand && g_nspill_ids < SPILL_MAX; i++) {
        unsigned k = 0;
        while (k < g_nspill_ids && g_spill_ids[k] != g_cand[i])
            k++;
        if (k == g_nspill_ids) {
            g_spill_ids[g_nspill_ids++] = g_cand[i];
            added = 1;
        }
    }
    g_ncand = 0;
    return added;
}

static void link_before(struct ir_val *x, struct ir_val *at)
{
    ir_remove(x);
    x->block = at->block;
    x->next = at;
    x->prev = at->prev;
    if (at->prev)
        at->prev->next = x;
    else
        at->block->first = x;
    at->prev = x;
}

static void spill_values(struct ir_func *f)
{
    if (!g_nspill_ids)
        return;
    int maxid = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id > maxid)
                maxid = v->id;
    if (!g_spill_limit)
        g_spill_limit = maxid + 1;
    int *slot = calloc((size_t)maxid + 1, sizeof(*slot)), nslots = 0;
    unsigned base = (g_info->local_size + 3) & ~3u;
    // a wide load's or a texture fetch's registers go in one aligned block, kept until its last lane is: so it's
    // its lanes that are spilled, each stored as soon as it's read out
    struct ir_val **byid = calloc((size_t)maxid + 1, sizeof(*byid));
    char *wide = calloc((size_t)maxid + 1, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            byid[v->id] = v;
    for (unsigned i = 0; i < g_nspill_ids; i++) {
        struct ir_val *w = byid[g_spill_ids[i]];
        if (w && w->type == IR_VOID)
            wide[w->id] = 1;
        else if (w)
            slot[w->id] = ++nslots;
    }
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->op == IR_PROJ && wide[v->src[0]->id])
                slot[v->id] = ++nslots;
    free(byid);
    free(wide);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first, *next; v; v = next) {
            next = v->next;
            if (v->op == IR_PHI && slot[v->id])  // goes to memory whole, below
                continue;
            for (unsigned i = 0; i < v->nsrc; i++) {
                struct ir_val *s = v->src[i];
                if (!s || s->id > maxid || !slot[s->id])
                    continue;
                struct ir_val *ld = ir_add(v->op == IR_PHI ? v->phi_blocks[i] : b, IR_LDL, s->type, 0);
                ld->imm = base + 4u * (unsigned)(slot[s->id] - 1);
                ld->aux = 4;
                if (v->op != IR_PHI && v != b->last)  // a phi's goes at its predecessor's end, the rest before the use
                    link_before(ld, v);
                v->src[i] = ld;
            }
        }
    // A phi that's spilled lives in its slot: each edge stores what it brings, in the predecessor, and the phi goes.
    // Kept as a register it would still hold one all through a loop it carries a value round, as the copy in at the
    // loop's end and the use at its start make its range the whole loop. The values the edges bring are read before
    // any is written, as a phi reads the old values of the others.
    for (int pass = 0; pass < 2; pass++)
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *v = b->first; v && v->op == IR_PHI; v = v->next) {
                if (v->id > maxid || !slot[v->id])
                    continue;
                for (unsigned i = 0; i < v->nsrc; i++) {
                    struct ir_val *s = v->src[i];
                    int in_memory = s && s->id <= maxid && slot[s->id];
                    if (pass == 0 && in_memory && s->op == IR_PHI) {  // only another phi's slot is about to change
                        struct ir_val *ld = ir_add(v->phi_blocks[i], IR_LDL, s->type, 0);
                        ld->imm = base + 4u * (unsigned)(slot[s->id] - 1);
                        ld->aux = 4;
                        v->src[i] = ld;
                    } else if (pass == 1) {
                        if (in_memory && s->op != IR_PHI) {
                            struct ir_val *ld = ir_add(v->phi_blocks[i], IR_LDL, s->type, 0);
                            ld->imm = base + 4u * (unsigned)(slot[s->id] - 1);
                            ld->aux = 4;
                            s = ld;
                        }
                        struct ir_val *st = ir_add(v->phi_blocks[i], IR_STL, IR_VOID, 2);
                        st->src[1] = s;
                        st->imm = base + 4u * (unsigned)(slot[v->id] - 1);
                        st->aux = 4;
                    }
                }
            }
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first, *next; v; v = next) {
            next = v->next;
            if (v->id > maxid || !slot[v->id])
                continue;
            if (v->op == IR_PHI) {
                ir_remove(v);
                continue;
            }
            struct ir_val *st = ir_add(b, IR_STL, IR_VOID, 2);
            st->src[1] = v;
            st->imm = base + 4u * (unsigned)(slot[v->id] - 1);
            st->aux = 4;
            struct ir_val *after = v;  // right after the value; a phi's after the block's last phi
            while (after->op == IR_PHI && after->next && after->next->op == IR_PHI)
                after = after->next;
            ir_remove(st);
            st->block = b;
            st->prev = after;
            st->next = after->next;
            if (after->next)
                after->next->prev = st;
            else
                b->last = st;
            after->next = st;
        }
    g_info->local_size = base + 4u * (unsigned)nslots;
    free(slot);
}

#define NREGS 240
static int g_regcount;

static char g_reserved[IR_MAX_SUBS + 1][NREGS];  // registers a function keeps out of: R1, the stack pointer, and
                                                  // a subroutine's callers' live across calls to it

// A subroutine reads the shader's values where it uses them (its arguments, the integer 1 and the like): each call
// to it, and to whatever calls it, reads them too, so they stay live across it.
static void add_externals(void)
{
    int words = (g_nvregs + 63) / 64;
    int *home = malloc(g_nvregs * sizeof(*home));
    memset(home, 0xff, g_nvregs * sizeof(*home));
    uint64_t *ext = calloc((size_t)g_nregions * words, 8);
    for (int r = 0; r < g_nregions; r++)
        for (int k = g_region_start[r]; k < g_region_end[r]; k++) {
            struct mop *m = &g_mops[k];
            for (int j = -1; j < m->nxd; j++)
                MEMBERS(w, j < 0 ? m->def : m->xd[j].vreg)
                    home[w] = r;
        }
    for (int r = 1; r < g_nregions; r++)
        for (int k = g_region_start[r]; k < g_region_end[r]; k++) {
            struct mop *m = &g_mops[k];
            for (int j = 0; j < m->nsrc + m->nxs; j++) {
                int v = j < m->nsrc ? m->src[j] : m->xs[j - m->nsrc].vreg;
                MEMBERS(w, v)
                    if (home[w] != r)
                        ext[(size_t)r * words + w / 64] |= 1ull << (w % 64);
            }
        }
    // through calls: a subroutine's callers read what it does
    for (int changed = 1; changed;) {
        changed = 0;
        for (int r = 1; r < g_nregions; r++)
            for (int s = 1; s < g_nregions; s++)
                if (g_callees[r] >> s & 1)
                    for (int w = 0; w < words; w++) {
                        uint64_t add = ext[(size_t)s * words + w] & ~ext[(size_t)r * words + w];
                        for (; add; add &= add - 1) {
                            int v = w * 64 + __builtin_ctzll(add);
                            if (home[v] != r) {
                                ext[(size_t)r * words + w] |= 1ull << (v % 64);
                                changed = 1;
                            }
                        }
                    }
    }
    for (int k = 0; k < g_nmops; k++) {
        struct mop *m = &g_mops[k];
        if (m->k != MK_CALL)
            continue;
        const uint64_t *e = ext + (size_t)(m->imm + 1) * words;
        int n = m->nxs;
        for (int w = 0; w < words; w++)
            n += __builtin_popcountll(e[w]);
        struct xfer *xs = ir_alloc(n * sizeof(*xs));
        memcpy(xs, m->xs, m->nxs * sizeof(*xs));
        for (int w = 0; w < words; w++)
            for (uint64_t bits = e[w]; bits; bits &= bits - 1) {
                xs[m->nxs].vreg = w * 64 + __builtin_ctzll(bits);
                xs[m->nxs++].slot = SLOT_KEEP;
            }
        m->xs = xs;
    }
    free(home);
    free(ext);
}

static int is_transfer(const struct mop *m)
{
    return m->k == MK_CALL || m->k == MK_ENTRY || m->k == MK_RET;
}

// are all the mops of a run list free in a register's row, and take them
static int seg_free(const uint64_t *row, int seg, int base)
{
    for (; seg >= 0; seg = g_segs[seg].next)
        for (int lo = g_segs[seg].start - base, hi = g_segs[seg].end - base; lo <= hi;) {
            int w = lo / 64, from = lo % 64, to = hi / 64 == w ? hi % 64 : 63;
            uint64_t mask = (to == 63 ? ~0ull : (1ull << (to + 1)) - 1) & ~((1ull << from) - 1);
            if (row[w] & mask)
                return 0;
            lo = (w + 1) * 64;
        }
    return 1;
}

static void seg_take(uint64_t *row, int seg, int base)
{
    for (; seg >= 0; seg = g_segs[seg].next)
        for (int lo = g_segs[seg].start - base, hi = g_segs[seg].end - base; lo <= hi;) {
            int w = lo / 64, from = lo % 64, to = hi / 64 == w ? hi % 64 : 63;
            row[w] |= (to == 63 ? ~0ull : (1ull << (to + 1)) - 1) & ~((1ull << from) - 1);
            lo = (w + 1) * 64;
        }
}

static int g_clusters_first;

static int place_before(int a, int b)
{
    int ca = g_vregs[a].size > 1, cb = g_vregs[b].size > 1;
    if (g_clusters_first && ca != cb)
        return ca;
    return g_vregs[a].start < g_vregs[b].start;
}

static void regalloc(struct ir_func *f, int region)
{
    compute_ranges(f, region);
    // the mops moving values through the subroutine window, in order
    int *xfer = malloc((g_region_end[region] - g_region_start[region] + 1) * sizeof(*xfer)), nxfer = 0;
    for (int k = g_region_start[region]; k < g_region_end[region]; k++)
        if (is_transfer(&g_mops[k]))
            xfer[nxfer++] = k;
    // heads of clusters (size >= 1, not a non-head member)
    int *order = malloc(g_nvregs * sizeof(*order));
    int n = 0;
    char ishead[1] = { 0 };
    (void)ishead;
    char *member = calloc(g_nvregs, 1);
    for (int i = 0; i < g_nvregs; i++)
        for (int j = g_vregs[i].next; j >= 0; j = g_vregs[j].next)
            member[j] = 1;
    for (int i = 0; i < g_nvregs; i++)
        if (!member[i] && g_vregs[i].start >= 0 && g_vregs[i].hw < 0)
            order[n++] = i;
    int base = g_region_start[region], words = (g_region_end[region] - base + 64) / 64, regcount = g_regcount;
    uint64_t *occ = malloc((size_t)NREGS * words * 8);
    char *placed = malloc(g_nvregs);
    int *active = malloc(g_nvregs * sizeof(*active));
    // Placed twice, in start order and with the clusters first (they need aligned runs, which singles would leave no
    // room for), and the one that takes fewer registers kept: neither always does, and one may not fit at all.
    int *best = malloc(g_nvregs * sizeof(*best)), best_count = 0, fitted = 0;
    for (int attempt = 0; attempt < 2; attempt++) {
        g_clusters_first = attempt;
        for (int i = 1; i < n; i++)
            for (int j = i; j > 0 && place_before(order[j], order[j - 1]); j--) {
                int t = order[j];
                order[j] = order[j - 1];
                order[j - 1] = t;
            }
        memset(occ, 0, (size_t)NREGS * words * 8);
        memset(placed, 0, g_nvregs);
        int failed = -1;
        for (int i = 0, x = 0; i < n && failed < 0; i++) {
            int h = order[i], start = g_vregs[h].start;
            if (i && start < g_vregs[order[i - 1]].start)
                x = 0;
            while (x < nxfer && xfer[x] < start)
                x++;
            int from = x < nxfer && xfer[x] <= g_vregs[h].end ? g_win_end : 2;  // live at a transfer: past the window
            // a copy whose source dies at it takes the source's register, and the copy goes away
            const struct mop *dm = &g_mops[start];
            int src = dm->k == MK_MOV && dm->def == h ? dm->src[0] : -1;
            if (src >= 0 && src != VREG_RZ && g_vregs[h].size == 1 && g_vregs[src].size == 1 && !member[src] &&
                placed[src] && g_vregs[src].end == start && g_vregs[src].hw >= from) {
                // the source holds its register at this mop, which the copy is welcome to
                uint64_t *row = occ + (size_t)g_vregs[src].hw * words;
                row[(start - base) / 64] &= ~(1ull << ((start - base) % 64));
                if (seg_free(row, g_vregs[h].seg, base)) {
                    g_vregs[h].hw = g_vregs[src].hw;
                    seg_take(row, g_vregs[h].seg, base);
                    placed[h] = 1;
                    continue;
                }
                row[(start - base) / 64] |= 1ull << ((start - base) % 64);
            }
            int r = -1, size = g_vregs[h].size, align = g_vregs[h].align;
            for (int c = (from + align - 1) & ~(align - 1); c + size <= NREGS && r < 0; c += align) {
                int ok = 1, j = h;
                for (int k = 0; k < size && ok; k++, j = g_vregs[j].next)
                    ok = !g_reserved[region][c + k] && seg_free(occ + (size_t)(c + k) * words, g_vregs[j].seg, base);
                if (ok)
                    r = c;
            }
            if (r < 0) {
                failed = i;
                break;
            }
            int k = 0;
            for (int j = h; j >= 0; j = g_vregs[j].next, k++) {
                g_vregs[j].hw = r + k;
                seg_take(occ + (size_t)(r + k) * words, g_vregs[j].seg, base);
                if (r + k + 1 > g_regcount)
                    g_regcount = r + k + 1;
            }
            placed[h] = 1;
        }
        if (failed < 0 && (!fitted || g_regcount < best_count)) {
            fitted = 1;
            best_count = g_regcount;
            for (int i = 0; i < n; i++)
                for (int j = order[i]; j >= 0; j = g_vregs[j].next)
                    best[j] = g_vregs[j].hw;
        }
        if (failed >= 0 && attempt && !fitted) {  // what holds registers: the placed values whose range spans this one's start
            int na = 0, start = g_vregs[order[failed]].start;
            for (int a = 0; a < failed; a++)
                if (g_vregs[order[a]].start <= start && g_vregs[order[a]].end >= start)
                    active[na++] = order[a];
            note_spill_candidates(active, na);
            airc_fail("out of registers");
        }
        for (int i = 0; i < (failed < 0 ? n : failed); i++)
            for (int j = order[i]; j >= 0; j = g_vregs[j].next)
                g_vregs[j].hw = -1;
        g_regcount = regcount;
    }
    for (int i = 0; i < n; i++)
        for (int j = order[i]; j >= 0; j = g_vregs[j].next)
            g_vregs[j].hw = best[j];
    g_regcount = best_count;
    free(best);
    free(occ);
    free(placed);
    free(active);
    // what each callee keeps out of: what this one does, and its values live across the call or read in the callee
    for (int x = 0; x < nxfer; x++) {
        const struct mop *m = &g_mops[xfer[x]];
        if (m->k != MK_CALL)
            continue;
        char *keep = g_reserved[m->imm + 1];
        int k = xfer[x];
        for (int r = 0; r < NREGS; r++)
            keep[r] |= g_reserved[region][r];
        for (int v = 0; v < g_nvregs; v++)
            if (g_vregs[v].hw >= 0 && g_vregs[v].start < k && g_vregs[v].end > k)
                keep[g_vregs[v].hw] = 1;
        for (int j = 0; j < m->nxs; j++)
            if (m->xs[j].slot == SLOT_KEEP)
                MEMBERS(w, m->xs[j].vreg)
                    keep[g_vregs[w].hw] = 1;
    }
    free(xfer);
    free(order);
    free(member);
}

static int hw(int vreg)
{
    if (vreg == VREG_RZ)
        return SASS_RZ;
    return g_vregs[vreg].hw;
}

// What an instruction leaves behind in its destination, for sched_push: nothing yet (a variable-latency result
// is waited for on a scoreboard), a fixed-latency result, a predicate, or the uniform register.
#define FIXED_LAT 1
#define PRED_LAT 2
#define UNIFORM_LAT 3
#define BAR_LAT 2
#define UNIFORM_CYCLES 7  // the uniform datapath is slower: a load reading UR4 right after its ULDC reads garbage
#define SLOW_CYCLES 13    // I2F: no table entry, kept well above the ALU's
#define UR4_PSEUDO 254
#define P0_PSEUDO 250
#define P6_PSEUDO 249

// The pipe a fixed-latency result comes from, and the one that reads it, for the cycles from the writer's issue to
// the reader's (NVIDIA's sm_80 table, as Mesa's NAK has it). IMAD.WIDE writes its low and high words at different
// times and reads its operands at different stages.
enum { WC_NONE, WC_ALU, WC_FMA, WC_WIDE_LO, WC_WIDE_HI, WC_FP16, WC_PRED, WC_UNIFORM, WC_SLOW };
enum { RC_ALU, RC_FMA, RC_WIDE_AB, RC_WIDE_CH, RC_FP16, RC_DECOUPLED, RC_AGU, RC_VOTE };

static int writer_class(const struct sass_insn *in)
{
    switch (in->lo & 0x1ff) {
    case 0x025:
        return WC_WIDE_LO;
    case 0x020 ... 0x024:
    case 0x027:
        return WC_FMA;
    case 0x030 ... 0x032:
        return WC_FP16;
    case 0x045:
        return WC_SLOW;
    default:
        return WC_ALU;
    }
}

static int reader_class(const struct sass_insn *in, int i)
{
    switch (in->lo & 0x1ff) {
    case 0x002: case 0x007: case 0x009: case 0x00b: case 0x00c: case 0x010: case 0x012: case 0x017: case 0x019:
    case 0x03e:
        return RC_ALU;
    case 0x006:
        return RC_VOTE;
    case 0x020 ... 0x024:
    case 0x027:
        return RC_FMA;
    case 0x025:  // A and B, then C's low and high words
        return i < 2 ? RC_WIDE_AB : i == 2 ? RC_FMA : RC_WIDE_CH;
    case 0x030 ... 0x032:
        return RC_FP16;
    case 0x045: case 0x100: case 0x101: case 0x105: case 0x107: case 0x108: case 0x109: case 0x119: case 0x150:
    case 0x161: case 0x164: case 0x167: case 0x16d: case 0x170: case 0x182:
        return RC_DECOUPLED;
    default:
        return RC_AGU;
    }
}

static unsigned raw_latency(int writer, int reader)
{
    // [reader][ALU, FMA, WIDE_LO, WIDE_HI, FP16 writer]
    static const uint8_t reg[][5] = {
        [RC_ALU] = { 4, 5, 3, 5, 5 },       [RC_FMA] = { 5, 4, 2, 4, 5 },       [RC_WIDE_AB] = { 5, 4, 4, 6, 5 },
        [RC_WIDE_CH] = { 3, 2, 2, 2, 3 },   [RC_FP16] = { 5, 5, 3, 5, 5 },      [RC_DECOUPLED] = { 4, 4, 4, 4, 4 },
        [RC_AGU] = { 5, 5, 5, 5, 5 },
    };
    if (writer == WC_UNIFORM)
        return UNIFORM_CYCLES;
    if (writer == WC_SLOW)
        return SLOW_CYCLES;
    if (writer == WC_PRED)  // only an ALU or FMA instruction takes a predicate quickly
        return reader == RC_ALU ? 4 : reader == RC_FMA ? 5 : 13;
    // a cycle more for readers outside the ALU and FMA pipes: at the table's figures the compute thread ID,
    // threadgroup memory and SIMD probes read stale operands on the 3060
    unsigned slack = reader != RC_ALU && reader != RC_FMA;
    return reg[reader == RC_VOTE ? RC_ALU : reader][writer - WC_ALU] + slack;
}

// the longest any reader waits on a result
static unsigned worst_latency(int writer)
{
    unsigned worst = 0;
    for (int r = RC_ALU; r <= RC_VOTE; r++)
        if (raw_latency(writer, r) > worst)
            worst = raw_latency(writer, r);
    return worst;
}

static struct sass_insn *g_prog;
static unsigned g_np, g_prog_cap;
static int g_barrier_of[256];
static unsigned g_wcycle[256];  // when a fixed-latency write to each register issued
static uint8_t g_wclass[256];    // and from which pipe, WC_NONE when nothing is pending
static unsigned g_bar_set[6];
static unsigned g_cycle;

static void prog_emit(struct sass_insn in)
{
    if (g_np == g_prog_cap) {
        g_prog_cap = g_prog_cap ? g_prog_cap * 2 : 4096;
        g_prog = realloc(g_prog, g_prog_cap * sizeof(*g_prog));
    }
    g_prog[g_np++] = in;
}

static unsigned get_stall(const struct sass_insn *in)
{
    return (in->hi >> 41) & 0xf;
}

// nvcc sets the yield bit only on short stalls, and nvdisasm rejects it beside the long ones
static void set_stall(struct sass_insn *in, unsigned s)
{
    in->hi = (in->hi & ~(0xfull << 41 | 1ull << 45)) | ((uint64_t)s << 41) | (uint64_t)(s <= 2) << 45;
}

// lengthen the stalls before the next instruction until it issues at cycle need
static void stall_until(unsigned need)
{
    while (need > g_cycle && g_np > 0) {
        unsigned room = 15 - get_stall(&g_prog[g_np - 1]);
        if (!room) {
            prog_emit(sass_nop(sass_ctrl_word((struct sass_ctrl){ .wr = SASS_NO_BARRIER, .rd = SASS_NO_BARRIER })));
            continue;
        }
        unsigned add = need - g_cycle < room ? need - g_cycle : room;
        set_stall(&g_prog[g_np - 1], get_stall(&g_prog[g_np - 1]) + add);
        g_cycle += add;
    }
}

// What a mop reads and writes, found by running its emitter with g_probe set: sched and sched_push then record
// their operands instead of emitting. The list scheduler orders mops by it.
struct fx {
    uint8_t rd[16], rclass[16], wr[8], wclass[8];
    int nrd, nwr;
    int async;     // the result comes back on a scoreboard
    int late;      // the sources are read after issue
    int overflow;  // more operands than fit here: leave the mop where it is
};
static int g_probe;
static struct fx g_fx;

static void probe_read(uint8_t reg, int rclass)
{
    if (g_fx.nrd == 16)
        g_fx.overflow = 1;
    else {
        g_fx.rd[g_fx.nrd] = reg;
        g_fx.rclass[g_fx.nrd++] = (uint8_t)rclass;
    }
}

static void probe_write(int reg)
{
    if (g_fx.nwr == 8)
        g_fx.overflow = 1;
    else
        g_fx.wr[g_fx.nwr++] = (uint8_t)reg;
}

static void sched_push(struct sass_insn in, const uint8_t *srcs, int nsrc, int dst, int wide, unsigned lat)
{
    if (g_probe) {
        for (int i = 0; i < nsrc; i++)
            if (srcs[i] != SASS_RZ)
                probe_read(srcs[i], reader_class(&in, i));
        for (int k = 0; dst >= 0 && lat && k < g_fx.nwr; k++)
            if (g_fx.wr[k] == dst || (wide && g_fx.wr[k] == dst + 1))
                g_fx.wclass[k] = g_fx.wr[k] != dst  ? WC_WIDE_HI
                                 : lat == PRED_LAT ? WC_PRED
                                 : lat == UNIFORM_LAT ? WC_UNIFORM
                                                      : writer_class(&in);
        return;
    }
    unsigned need = g_cycle;
    for (int i = 0; i < nsrc; i++) {
        if (srcs[i] == SASS_RZ || g_wclass[srcs[i]] == WC_NONE)
            continue;
        unsigned t = g_wcycle[srcs[i]] + raw_latency(g_wclass[srcs[i]], reader_class(&in, i));
        if (t > need)
            need = t;
    }
    unsigned wait = (in.hi >> 52) & 0x3f;
    for (int b = 0; b < 6; b++)
        if ((wait >> b & 1) && g_bar_set[b] + BAR_LAT > need)
            need = g_bar_set[b] + BAR_LAT;
    stall_until(need);
    unsigned wr = (in.hi >> 46) & 7, rd = (in.hi >> 49) & 7;
    prog_emit(in);
    if (wr < 6)
        g_bar_set[wr] = g_cycle;
    if (rd < 6)
        g_bar_set[rd] = g_cycle;
    if (dst >= 0 && lat) {
        g_wcycle[dst] = g_cycle;
        g_wclass[dst] = lat == PRED_LAT ? WC_PRED : lat == UNIFORM_LAT ? WC_UNIFORM : writer_class(&in);
        if (wide) {
            g_wcycle[dst + 1] = g_cycle;
            g_wclass[dst + 1] = WC_WIDE_HI;
        }
    }
    g_cycle += get_stall(&in);
}

// g_rbar_of[r]: the read barrier a still-pending instruction holds on r (it reads r after issue), or -1
static int g_rbar_of[256];

// The scoreboards held by atomics and loads of shared memory, which a threadgroup barrier has to wait for: one
// still in flight when the barrier releases the other warps lets them read or update the word before it lands.
static unsigned g_mem_bars;
static int g_mem_op;

// A scoreboard nothing waits on any more, the one set longest ago first; with none free the one set longest ago,
// which is the nearest to done.
static int take_bar(void)
{
    char used[6] = { 0 };
    for (int r = 0; r < 256; r++) {
        if (g_barrier_of[r] >= 0)
            used[g_barrier_of[r]] = 1;
        if (g_rbar_of[r] >= 0)
            used[g_rbar_of[r]] = 1;
    }
    int best = 0;
    for (int b = 1; b < 6; b++)
        if (used[b] < used[best] || (used[b] == used[best] && g_bar_set[b] < g_bar_set[best]))
            best = b;
    if (!used[best])
        g_mem_bars &= ~(1u << best);
    return best;
}

// Build a control word. Waits: on async results the sources need (RAW), on async results still to land in the
// destination (WAW), and on pending late readers of the destination (WAR). Allocates a read barrier on the
// sources when the instruction reads them late (memory, texture, attribute, MUFU), and a write barrier on dst
// when its result is async. dstw: registers dst writes.
static uint32_t sched(const uint8_t *srcs, int nsrc, int var_dst, int dst, int dstw, int late)
{
    if (g_probe) {
        for (int k = 0; dst >= 0 && k < dstw; k++)
            probe_write(dst + k);
        g_fx.async |= var_dst;
        g_fx.late |= late;
        return 0;
    }
    struct sass_ctrl c = { .stall = 1, .yield = 1, .wr = SASS_NO_BARRIER, .rd = SASS_NO_BARRIER };
    for (int i = 0; i < nsrc; i++)
        if (srcs[i] != SASS_RZ && g_barrier_of[srcs[i]] >= 0) {
            c.wait |= 1u << g_barrier_of[srcs[i]];
            g_barrier_of[srcs[i]] = -1;
        }
    for (int k = 0; dst >= 0 && k < dstw; k++) {
        int r = dst + k;
        if (g_barrier_of[r] >= 0) {
            c.wait |= 1u << g_barrier_of[r];
            g_barrier_of[r] = -1;
        }
        if (g_rbar_of[r] >= 0) {
            int b = g_rbar_of[r];
            c.wait |= 1u << b;
            for (int j = 0; j < 256; j++)  // that barrier covered every register its reader held
                if (g_rbar_of[j] == b)
                    g_rbar_of[j] = -1;
        }
    }
    if (late) {
        c.rd = take_bar();
        if (g_mem_op)
            g_mem_bars |= 1u << c.rd;
        for (int i = 0; i < nsrc; i++)
            if (srcs[i] != SASS_RZ && srcs[i] < 240)
                g_rbar_of[srcs[i]] = c.rd;
    }
    if (var_dst) {
        c.wr = take_bar();
        if (g_mem_op)
            g_mem_bars |= 1u << c.wr;
        for (int k = 0; k < dstw; k++)
            g_barrier_of[dst + k] = c.wr;
    }
    return sass_ctrl_word(c);
}

// Before control leaves a block: wait until every outstanding result and late read is done, so the block
// reached starts from nothing pending (it may be entered from elsewhere).
static void drain(void)
{
    unsigned need = g_cycle;
    struct sass_ctrl c = { .stall = 1, .wr = SASS_NO_BARRIER, .rd = SASS_NO_BARRIER };
    for (int r = 0; r < 256; r++) {
        if (g_wclass[r] != WC_NONE) {
            unsigned t = g_wcycle[r] + worst_latency(g_wclass[r]);
            if (t > need)
                need = t;
            g_wclass[r] = WC_NONE;
        }
        if (g_barrier_of[r] >= 0) {
            c.wait |= 1u << g_barrier_of[r];
            g_barrier_of[r] = -1;
        }
        if (g_rbar_of[r] >= 0) {
            c.wait |= 1u << g_rbar_of[r];
            g_rbar_of[r] = -1;
        }
    }
    stall_until(need);
    if (c.wait)  // the NOP only carries the waits
        sched_push(sass_nop(sass_ctrl_word(c)), NULL, 0, -1, 0, 0);
}

// A demoted thread's memory writes don't happen: they go under @!P6, which joins their sources so they wait for
// its last write.
static int demote_src(uint8_t *srcs, int n)
{
    if (g_demote)
        srcs[n++] = P6_PSEUDO;
    return n;
}

static struct sass_insn demote_guard(struct sass_insn in)
{
    return g_demote ? sass_pred(in, 6, 1) : in;
}

static unsigned *g_mop_np;    // SASS index where each mop's instructions begin
static int *g_block_np;       // SASS index of each block's first instruction
static struct {
    unsigned at;   // the BRA's SASS index
    int block;     // target block id, or -1
    int mop;       // target mop index (a stub), or -1
} g_fix[8192];
static int g_nfix;

static uint8_t mem_size(int bytes)
{
    return bytes == 1 ? SASS_U8 : bytes == 2 ? SASS_U16 : bytes == 8 ? SASS_B64 : bytes == 16 ? SASS_B128 : SASS_B32;
}

// map an IR integer condition (cond | IR_SIGNED) to an ISETP op + sign
static void isetp_op(int aux, uint8_t *op, int *sgn)
{
    static const uint8_t m[] = { ISETP_EQ, ISETP_NE, ISETP_LT, ISETP_LE, ISETP_GT, ISETP_GE };
    *op = m[aux & 7];
    *sgn = (aux & IR_SIGNED) != 0;
}

static uint8_t fsetp_op(int pred)
{
    static const uint8_t m[16] = {
        FSETP_F,   ISETP_EQ,  ISETP_GT,  ISETP_GE,  ISETP_LT,  ISETP_LE,  ISETP_NE,  FSETP_NUM,
        FSETP_NAN, FSETP_EQU, FSETP_GTU, FSETP_GEU, FSETP_LTU, FSETP_LEU, FSETP_NEU, FSETP_T,
    };
    return m[pred & 15];
}

// the source modifiers of a mop's mods on the instruction's operands: slot[i] the operand of source i, -1 for none
static struct sass_insn with_mods(struct sass_insn in, int mods, int s0, int s1, int s2)
{
    const int slot[3] = { s0, s1, s2 };
    for (int i = 0; i < 3; i++)
        if (slot[i] >= 0)
            in = sass_src_mod(in, slot[i], mods >> 2 * i & MOD_NEG, (mods >> 2 * i & MOD_ABS) != 0);
    return in;
}

static void emit_mov(int d, int s)
{
    if (d == s)
        return;
    uint8_t src[1] = { (uint8_t)s };
    uint32_t c = sched(src, 1, 0, d, 1, 0);
    sched_push(sass_mov_rr(d, s, c), src, 1, d, 0, FIXED_LAT);
}

// a branch's, BSSY's, CALL's or RET's offset, as BRA has it
static void set_rel(struct sass_insn *in, int32_t rel)
{
    in->lo = (in->lo & 0xffffffffull) | ((uint64_t)(uint32_t)rel << 32);
    in->hi = (in->hi & ~0x3ffffull) | (rel < 0 ? 0x3ffff : 0);
}

// each value of xs (into the window) or xd (out of it) moved, the return address too
static void window_moves(const struct xfer *x, int n, int into)
{
    for (int j = 0; j < n; j++) {
        if (x[j].slot == SLOT_KEEP)
            continue;
        MEMBERS(w, x[j].vreg) {
            int win = x[j].slot == SLOT_RA ? WIN_RA : WIN_ARGS + x[j].slot + (w != x[j].vreg);
            if (into)
                emit_mov(win, hw(w));
            else
                emit_mov(hw(w), win);
        }
    }
}

static void emit_call(const struct mop *m)
{
    window_moves(m->xs, m->nxs, 1);
    uint32_t c = sched(NULL, 0, 0, WIN_RA, 1, 0);
    sched_push(sass_imadmov_imm(WIN_RA, 0, c), NULL, 0, WIN_RA, 0, FIXED_LAT);
    unsigned ra = g_np - 1, bssy = 0;
    drain();
    if (m->aux >= 0) {  // where the call's threads meet again: Volta's subroutines may return each on its own
        sched_push(sass_bssy((uint8_t)m->aux, 0, sched(NULL, 0, 0, -1, 0, 0)), NULL, 0, -1, 0, 0);
        bssy = g_np - 1;
    }
    sched_push(sass_call_rel(0, sched(NULL, 0, 0, -1, 0, 0)), NULL, 0, -1, 0, 0);
    unsigned at = g_np - 1;
    g_fix[g_nfix].at = at;
    g_fix[g_nfix].block = m->target;
    g_fix[g_nfix++].mop = -1;
    // RET.REL goes to the address counted from the program's first instruction
    g_prog[ra].lo = (g_prog[ra].lo & 0xffffffffull) | (uint64_t)((at + 1) * 16) << 32;
    if (m->aux >= 0) {
        set_rel(&g_prog[bssy], (int32_t)(at - bssy) * 16);
        sched_push(sass_bsync((uint8_t)m->aux, sched(NULL, 0, 0, -1, 0, 0)), NULL, 0, -1, 0, 0);
    }
    window_moves(m->xd, m->nxd, 0);
}

static void emit_ret(const struct mop *m)
{
    window_moves(m->xs, m->nxs, 1);
    uint32_t c = sched(NULL, 0, 0, WIN_RA + 1, 1, 0);
    sched_push(sass_imadmov_imm(WIN_RA + 1, 0, c), NULL, 0, WIN_RA + 1, 0, FIXED_LAT);
    drain();
    sched_push(sass_ret_rel(WIN_RA, 0, sched(NULL, 0, 0, -1, 0, 0)), NULL, 0, -1, 0, 0);
    set_rel(&g_prog[g_np - 1], -(int32_t)g_np * 16);
}

// A fragment's colours into R0.. at its exit, from the registers MK_OUTPUT found them in. Some of those may be in
// R0.. already, so a move waits until nothing still to be made reads its target; a cycle of them (R0 and R1
// swapped) is broken through the scratch register. AIRC_STAGEEXIT=1 stages every colour through the scratch
// registers as before.
static uint8_t g_out_reg[64];
static int g_stage_exit;

static void exit_moves(int n)
{
    uint8_t from[64];
    int pending[64], np = 0, tmp = g_exit_tmp;
    memcpy(from, g_out_reg, n);
    for (int k = 0; k < n; k++)
        if (from[k] != k)
            pending[np++] = k;
    while (np) {
        int pick = -1;
        for (int i = 0; i < np && pick < 0; i++) {
            pick = i;
            for (int j = 0; j < np; j++)
                if (from[pending[j]] == pending[i])
                    pick = -1;
        }
        if (pick < 0) {
            // every target is still wanted: park the source of the first in the scratch register
            uint8_t park = from[pending[0]];
            uint8_t src[1] = { park };
            sched_push(sass_mov_rr(tmp, park, sched(src, 1, 0, tmp, 1, 0)), src, 1, tmp, 0, FIXED_LAT);
            for (int j = 0; j < np; j++)
                if (from[pending[j]] == park)
                    from[pending[j]] = (uint8_t)tmp;
            continue;
        }
        int k = pending[pick];
        uint8_t src[1] = { from[k] };
        sched_push(sass_mov_rr(k, from[k], sched(src, 1, 0, k, 1, 0)), src, 1, k, 0, FIXED_LAT);
        pending[pick] = pending[--np];
    }
}

static void emit_mops_in(const int *list, int n, enum nvk_stage stage)
{
    for (int k = 0; k < n; k++) {
        int i = list[k];
        struct mop *m = &g_mops[i];
        if (!g_probe)
            g_mop_np[i] = g_np;
        int d = m->def >= 0 ? hw(m->def) : -1;
        uint8_t s[6];
        for (int j = 0; j < m->nsrc; j++)
            s[j] = (uint8_t)hw(m->src[j]);
        uint32_t c;
        switch (m->k) {
        case MK_MOVIMM:
            if (g_vregs[m->def].end == i && g_vregs[m->def].start == i)
                break;  // a constant made for an operand that took it as an immediate after all
            c = sched(NULL, 0, 0, d, 1, 0);
            sched_push(sass_imadmov_imm(d, (uint32_t)m->imm, c), NULL, 0, d, 0, FIXED_LAT);
            break;
        case MK_MOV:
            if (d == s[0])  // coalesced
                break;
            c = sched(s, 1, 0, d, 1, 0);
            sched_push(sass_mov_rr(d, s[0], c), s, 1, d, 0, FIXED_LAT);
            break;
        case MK_MOVCBUF:
            c = sched(NULL, 0, 0, d, 1, 0);
            sched_push(sass_mov_cbuf(d, 0, (uint16_t)m->imm, c), NULL, 0, d, 0, FIXED_LAT);
            break;
        case MK_LDC:  // variable latency
            c = sched(s, 1, 1, d, 1, 1);
            sched_push(sass_ldc(d, s[0], 0, (uint16_t)m->imm, c), s, 1, d, 0, 0);
            break;
        case MK_S2R:
            c = sched(NULL, 0, 1, d, 1, 0);
            sched_push(m->imm & 0x100 ? sass_pixld(d, m->imm & 0xff, c) : sass_s2r(d, (uint8_t)m->imm, c), NULL, 0,
                       d, 0, 0);
            break;
        case MK_ALD:  // a vertex handle is read late, as a load's address is
            c = sched(s, m->nsrc, 1, d, 1, m->nsrc);
            sched_push(sass_ald(d, m->nsrc ? s[0] : SASS_RZ, (uint16_t)m->imm, 1, (m->aux & IR_ATTR_PATCH) != 0,
                                (m->aux & IR_ATTR_OUTPUT) != 0, c),
                       s, m->nsrc, d, 0, 0);
            break;
        case MK_ISBERD:
            c = sched(s, 1, 1, d, 1, 1);
            sched_push(sass_isberd(d, s[0], c), s, 1, d, 0, 0);
            break;
        case MK_IPA:
            c = sched(NULL, 0, 1, d, 1, 0);
            sched_push(sass_ipa(d, (uint16_t)m->imm, m->aux, c), NULL, 0, d, 0, 0);
            break;
        case MK_IADD3:
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(sass_iadd3_rr(d, s[0], 0, c), (uint32_t)m->imm)
                               : sass_iadd3_rr(d, s[0], s[1], c),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        case MK_ISUB:
            c = sched(s, 2, 0, d, 1, 0);
            sched_push(sass_isub_rr(d, s[0], s[1], c), s, 2, d, 0, FIXED_LAT);
            break;
        case MK_IMAD:
            c = sched(s, 2, 0, d, 1, 0);
            sched_push(sass_imad_rr(d, s[0], s[1], c), s, 2, d, 0, FIXED_LAT);
            break;
        case MK_IMADHI:
            c = sched(s, 2, 0, d, 1, 0);
            sched_push(sass_imadhi_rr(d, s[0], s[1], m->aux, c), s, 2, d, 0, FIXED_LAT);
            break;
        case MK_LOP3:
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(sass_lop3(d, s[0], 0, (uint8_t)m->aux, c), (uint32_t)m->imm)
                               : sass_lop3(d, s[0], s[1], (uint8_t)m->aux, c),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        case MK_LOP3_IMM:
            c = sched(s, 1, 0, d, 1, 0);
            sched_push(sass_imm(sass_lop3(d, s[0], 0, (uint8_t)m->aux, c), (uint32_t)m->imm), s, 1, d, 0, FIXED_LAT);
            break;
        case MK_SHL:
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(sass_shf_l(d, s[0], 0, c), (uint32_t)m->imm)
                               : sass_shf_l(d, s[0], s[1], c),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        case MK_SHR: {
            struct sass_insn (*shf)(uint8_t, uint8_t, uint8_t, uint32_t) = m->aux ? sass_shf_r_s : sass_shf_r;
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(shf(d, s[0], 0, c), (uint32_t)m->imm) : shf(d, s[0], s[1], c), s, m->nsrc,
                       d, 0, FIXED_LAT);
            break;
        }
        case MK_IMNMX:
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(sass_imnmx(d, s[0], 0, m->aux & 1, (m->aux & 2) != 0, c), (uint32_t)m->imm)
                               : sass_imnmx(d, s[0], s[1], m->aux & 1, (m->aux & 2) != 0, c),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        // float arithmetic and compares flush denormals as Apple's GPUs do (denormal * inf is NaN there,
        // step(denormal, 0) is 1); fmin/fmax and moves keep them
        case MK_FADD: {
            c = sched(s, m->nsrc, 0, d, 1, 0);
            struct sass_insn add =
                m->aux & 2    ? sass_hadd2(d, s[0], s[1], m->aux & 1, c)
                : m->aux & 16 ? with_mods(sass_ftz(sass_fadd_imm(d, s[0], (uint32_t)m->imm, c)), m->mods, SASS_MOD_A,
                                          -1, -1)
                              : with_mods(sass_ftz(sass_fadd(d, s[0], s[1], m->aux & 1, c)), m->mods, SASS_MOD_A,
                                          SASS_MOD_B, -1);
            if (m->aux & 64)
                add = sass_sat(add);
            if (m->aux & 128)
                add = sass_neg_a(add);
            sched_push(add, s, m->nsrc, d, 0, FIXED_LAT);
            break;
        }
        case MK_FMUL:
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->aux & 2    ? sass_hmul2(d, s[0], s[1], c)
                       : m->aux & 16 ? with_mods(sass_ftz(sass_fmul_imm(d, s[0], (uint32_t)m->imm, c)), m->mods,
                                                 SASS_MOD_A, -1, -1)
                                     : with_mods(sass_ftz(sass_fmul(d, s[0], s[1], c)), m->mods, SASS_MOD_A,
                                                 SASS_MOD_B, -1),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        case MK_FFMA: {
            int mode = m->aux >> 2 & 3;
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->aux & 2    ? sass_hfma2(d, s[0], s[1], s[2], c)
                       : m->aux & 16 ? with_mods(sass_ftz(sass_ffma_imm_b(d, s[0], (uint32_t)m->imm, s[1], mode, c)),
                                                 m->mods, SASS_MOD_A, SASS_MOD_C, -1)
                       : m->aux & 32 ? with_mods(sass_ftz(sass_ffma_imm_c(d, s[0], s[1], (uint32_t)m->imm, mode, c)),
                                                 m->mods, SASS_MOD_A, SASS_MOD_C, -1)
                                     : with_mods(sass_ftz(sass_ffma(d, s[0], s[1], s[2], mode, c)), m->mods,
                                                 SASS_MOD_A, SASS_MOD_B, SASS_MOD_C),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        }
        case MK_FMNMX:
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(sass_fmnmx(d, s[0], 0, m->aux, c), (uint32_t)m->imm)
                               : sass_fmnmx(d, s[0], s[1], m->aux, c),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        case MK_MUFU:
            c = sched(s, 1, 1, d, 1, 1);
            sched_push(sass_mufu(d, s[0], (uint8_t)m->aux, c), s, 1, d, 0, 0);
            break;
        case MK_POPC:
        case MK_BREV:
        case MK_FLO:  // variable latency, like MUFU
            c = sched(s, 1, 1, d, 1, 1);
            sched_push(m->k == MK_POPC   ? sass_popc(d, s[0], c)
                       : m->k == MK_BREV ? sass_brev(d, s[0], c)
                                         : sass_flo(d, s[0], c),
                       s, 1, d, 0, 0);
            break;
        case MK_VOTE: {
            uint8_t ps[1] = { P0_PSEUDO };
            int n = !m->aux;
            c = sched(ps, n, 0, d, 1, 0);
            sched_push(sass_vote_any(d, m->aux ? 7 : 0, c), ps, n, d, 0, FIXED_LAT);
            break;
        }
        case MK_I2F:
            c = sched(s, 1, 0, d, 1, 0);
            sched_push(sass_i2f(d, s[0], m->aux, c), s, 1, d, 0, FIXED_LAT);
            break;
        case MK_F2I:
            c = sched(s, 1, 1, d, 1, 1);
            sched_push(sass_f2i(d, s[0], m->aux, c), s, 1, d, 0, 0);
            break;
        case MK_F2H:
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->nsrc > 1 ? sass_f2h2(d, s[0], s[1], c) : sass_f2h(d, s[0], c), s, m->nsrc, d, 0, FIXED_LAT);
            break;
        case MK_SHFL:  // variable latency
            c = sched(s, m->nsrc, 1, d, 1, 1);
            sched_push(m->nsrc > 1 ? sass_shfl_r(m->aux & 3, d, s[0], s[1], (uint16_t)(m->aux >> 4), c)
                                   : sass_shfl(m->aux & 3, d, s[0], (uint8_t)m->imm, (uint16_t)(m->aux >> 4), c),
                       s, m->nsrc, d, 0, 0);
            break;
        case MK_FSWZADD:
            c = sched(s, 2, 0, d, 1, 0);
            sched_push(sass_fswzadd(d, s[0], s[1], (uint8_t)m->aux, c), s, 2, d, 0, FIXED_LAT);
            break;
        case MK_FRND:  // variable latency, like F2I
            c = sched(s, 1, 1, d, 1, 1);
            sched_push(sass_frnd(d, s[0], m->aux, c), s, 1, d, 0, 0);
            break;
        case MK_H2F:
            c = sched(s, 1, 0, d, 1, 0);
            sched_push(m->aux & 1 ? sass_h2f_hi(sass_h2f(d, s[0], c)) : sass_h2f(d, s[0], c), s, 1, d, 0, FIXED_LAT);
            break;
        case MK_ISETP: {
            uint8_t op;
            int sgn;
            isetp_op(m->aux, &op, &sgn);
            c = sched(s, m->nsrc, 0, P0_PSEUDO, 1, 0);
            sched_push(m->immb ? sass_imm(sass_isetp(0, s[0], 0, op, sgn, c), (uint32_t)m->imm)
                               : sass_isetp(0, s[0], s[1], op, sgn, c),
                       s, m->nsrc, P0_PSEUDO, 0, PRED_LAT);
            break;
        }
        case MK_FSETP:
            c = sched(s, m->nsrc, 0, P0_PSEUDO, 1, 0);
            sched_push(m->immb ? with_mods(sass_ftz(sass_imm(sass_fsetp(0, s[0], 0, fsetp_op(m->aux & 15), c),
                                                             (uint32_t)m->imm)),
                                           m->mods, SASS_MOD_A, -1, -1)
                               : with_mods(sass_ftz(sass_fsetp(0, s[0], s[1], fsetp_op(m->aux & 15), c)), m->mods,
                                           SASS_MOD_A, SASS_MOD_B, -1),
                       s, m->nsrc, P0_PSEUDO, 0, PRED_LAT);
            break;
        case MK_SEL: {
            uint8_t ss[3] = { s[0], s[1], P0_PSEUDO };
            if (m->immb)
                ss[1] = P0_PSEUDO;
            c = sched(ss, m->immb ? 1 : 2, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(sass_sel(d, s[0], 0, m->aux ? 8 : 0, c), (uint32_t)m->imm)
                               : sass_sel(d, s[0], s[1], 0, c),
                       ss, m->immb ? 2 : 3, d, 0, FIXED_LAT);
            break;
        }
        case MK_ATOM: {  // variable latency, like a load
            int shared = IR_ATOMA_SHARED(m->aux);
            uint8_t ss[5] = { s[0], s[1], s[2], (uint8_t)(s[0] + 1) };
            int n = demote_src(ss, shared ? 3 : 4);
            g_mem_op = 1;
            c = sched(ss, n, 1, d, 1, 1);
            g_mem_op = 0;
            sched_push(demote_guard(sass_atom(IR_ATOMA_OP(m->aux), shared, IR_ATOMA_TYPE(m->aux), d, s[0],
                                              (uint32_t)m->imm, s[1], s[2], c)),
                       ss, n, d, 0, 0);
            break;
        }
        case MK_LDG: {
            uint8_t ss[3] = { s[0], (uint8_t)(s[0] + 1), UR4_PSEUDO };
            c = sched(ss, 2, 1, d, m->aux > 4 ? m->aux / 4 : 1, 1);
            sched_push(sass_ldg(d, s[0], (uint32_t)m->imm, mem_size(m->aux), c), ss, 3, d, 0, 0);
            break;
        }
        case MK_STG: {
            int words = m->aux > 4 ? m->aux / 4 : 1;
            uint8_t ss[7] = { s[0], (uint8_t)(s[0] + 1) };
            for (int k = 0; k < words; k++)
                ss[2 + k] = (uint8_t)(s[1] + k);
            int n = demote_src(ss, 2 + words);
            c = sched(ss, n, 0, -1, 0, 1);
            sched_push(demote_guard(sass_stg(s[0], (uint32_t)m->imm, s[1], mem_size(m->aux), c)), ss, n, -1, 0, 0);
            break;
        }
        case MK_LDS: {
            uint8_t a = m->nsrc ? s[0] : SASS_RZ;
            uint8_t ss[1] = { a };
            g_mem_op = 1;
            c = sched(ss, m->nsrc, 1, d, 1, 1);
            g_mem_op = 0;
            sched_push(sass_lds(d, a, (uint32_t)m->imm, mem_size(m->aux), c), ss, m->nsrc, d, 0, 0);
            break;
        }
        case MK_STS: {
            uint8_t ss[2] = { s[1], s[0] };
            c = sched(ss, 2, 0, -1, 0, 1);
            sched_push(sass_sts(s[0], (uint32_t)m->imm, s[1], mem_size(m->aux), c), ss, 2, -1, 0, 0);
            break;
        }
        case MK_LDL: {  // as LDG: every register of a wide load waits on it
            uint8_t a = m->nsrc ? s[0] : SASS_RZ;
            uint8_t ss[1] = { a };
            c = sched(ss, m->nsrc, 1, d, m->aux > 4 ? m->aux / 4 : 1, 1);
            sched_push(sass_ldl(d, a, (uint32_t)m->imm, mem_size(m->aux), c), ss, m->nsrc, d, 0, 0);
            break;
        }
        case MK_STL: {
            int words = m->aux > 4 ? m->aux / 4 : 1, n = 0;
            uint8_t ss[5];
            if (m->src[0] != VREG_RZ)
                ss[n++] = s[0];
            for (int k = 0; k < words; k++)
                ss[n++] = (uint8_t)(s[1] + k);
            c = sched(ss, n, 0, -1, 0, 1);
            sched_push(sass_stl(m->src[0] != VREG_RZ ? s[0] : SASS_RZ, (uint32_t)m->imm, s[1], mem_size(m->aux), c), ss,
                       n, -1, 0, 0);
            break;
        }
        case MK_BAR:
            if (m->aux) {  // a memory fence, with nvcc's control words
                sched_push(sass_membar_gpu(0x000fec00), NULL, 0, -1, 0, 0);
                sched_push(sass_errbar(0x000fc000), NULL, 0, -1, 0, 0);
                sched_push(sass_cctl_ivall(0x000fca00), NULL, 0, -1, 0, 0);
                break;
            }
            c = sched(NULL, 0, 0, -1, 0, 0);
            for (int r = 0; r < 256; r++) {
                if (g_barrier_of[r] >= 0 && (g_mem_bars >> g_barrier_of[r] & 1)) {
                    c |= (uint32_t)1 << (20 + g_barrier_of[r]);
                    g_barrier_of[r] = -1;
                }
                if (g_rbar_of[r] >= 0 && (g_mem_bars >> g_rbar_of[r] & 1)) {
                    c |= (uint32_t)1 << (20 + g_rbar_of[r]);
                    g_rbar_of[r] = -1;
                }
            }
            sched_push(sass_bar_sync(c), NULL, 0, -1, 0, 0);
            break;
        case MK_BSSY:
            drain();
            c = sched(NULL, 0, 0, -1, 0, 0);
            sched_push(sass_bssy((uint8_t)m->imm, 0, c), NULL, 0, -1, 0, 0);
            g_fix[g_nfix].at = g_np - 1;
            g_fix[g_nfix].block = m->target;
            g_fix[g_nfix++].mop = 0;
            break;
        case MK_BSYNC:
            drain();
            c = sched(NULL, 0, 0, -1, 0, 0);
            sched_push(sass_bsync((uint8_t)m->imm, c), NULL, 0, -1, 0, 0);
            break;
        case MK_DEMOTE:
            if (m->nsrc) {
                uint8_t ps[2] = { s[0], P6_PSEUDO };
                c = sched(ps, 2, 0, P6_PSEUDO, 1, 0);
                sched_push(sass_isetp_or(6, s[0], SASS_RZ, m->aux ? ISETP_EQ : ISETP_NE, 6, c), ps, 2, P6_PSEUDO, 0,
                           PRED_LAT);
            } else {
                c = sched(NULL, 0, 0, P6_PSEUDO, 1, 0);
                sched_push(sass_isetp(6, SASS_RZ, SASS_RZ, ISETP_EQ, 0, c), NULL, 0, P6_PSEUDO, 0, PRED_LAT);
            }
            break;
        case MK_IMADWIDE: {
            uint8_t ss[4] = { s[0], s[1], s[2], (uint8_t)(s[2] + 1) };
            c = sched(ss, 4, 0, d, 2, 0);
            sched_push(sass_imadwide_rr(d, s[0], s[1], s[2], m->aux, c), ss, 4, d, 1, FIXED_LAT);
            break;
        }
        case MK_TEX: {
            uint8_t ss[8];
            int n = 0;
            for (int j = 0; j < 2; j++)
                for (int r = 0; r < (int)(m->imm >> 4 * j & 15); r++)
                    ss[n++] = (uint8_t)(s[j] + r);
            c = sched(ss, n, 1, d, 4, 1);
            sched_push(sass_texop(IR_TEXA_OP(m->aux), d, s[0], s[1], m->aux & IR_TEXA_SCALAR ? 1 : 0xf,
                                  IR_TEXA_DIM(m->aux), IR_TEXA_FLAGS(m->aux), c),
                       ss, n, d, 0, 0);
            break;
        }
        case MK_TXQ: {
            uint8_t ss[2] = { s[0], (uint8_t)(s[0] + 1) };
            c = sched(ss, 2, 1, d, 1, 1);
            sched_push(sass_txq(d, s[0], m->aux & 15, m->aux >> 4, c), ss, 2, d, 0, 0);
            break;
        }
        case MK_SUST: {
            uint8_t coord = s[0], color = s[1], handle = s[2], ss[10];
            int n = 0;
            for (int r = 0; r < m->aux; r++)
                ss[n++] = (uint8_t)(coord + r);
            for (int r = 0; r < 4; r++)
                ss[n++] = (uint8_t)(color + r);
            ss[n++] = handle;
            n = demote_src(ss, n);
            c = sched(ss, n, 0, -1, 0, 1);
            sched_push(demote_guard(sass_sust(coord, color, handle, 0xf, (int)m->imm, c)), ss, n, -1, 0, 0);
            break;
        }
        case MK_AST: {
            int n = m->aux & 15, vtx = m->nsrc > 1;
            uint8_t ss[5];
            for (int k = 0; k < n; k++)
                ss[k] = (uint8_t)(s[0] + k);
            if (vtx)
                ss[n] = s[1];
            c = sched(ss, n + vtx, 0, -1, 0, 1);
            sched_push(sass_ast((uint16_t)m->imm, s[0], vtx ? s[1] : SASS_RZ, n, (m->aux & IR_ATTR_PATCH) != 0, c), ss,
                       n + vtx, -1, 0, 0);
            break;
        }
        case MK_BRA: {
            drain();
            uint8_t ps[1] = { P0_PSEUDO };
            int cond = m->aux;
            if (!cond && m->target >= 0 && g_block_mop[m->target] == i + 1)
                break;  // the block it goes to follows
            c = sched(cond ? ps : NULL, cond, 0, -1, 0, 0);
            sched_push(sass_bra(0, cond ? 0 : 7, 0, c), cond ? ps : NULL, cond, -1, 0, 0);
            g_fix[g_nfix].at = g_np - 1;
            g_fix[g_nfix].block = m->target;
            g_fix[g_nfix++].mop = m->label;
            break;
        }
        case MK_OUTPUT:
            // Where the colours live is all the exit needs. Probing still stages them, which is what the scheduler
            // sees: the sources are read here, and nothing is allowed to write them before this.
            for (int k = 0; k < m->nsrc; k++) {
                if (!g_probe && !g_stage_exit) {
                    g_out_reg[m->imm + k] = s[k];
                    continue;
                }
                int r = g_scratch + (int)m->imm + k;
                uint8_t src[1] = { s[k] };
                c = sched(src, 1, 0, r, 1, 0);
                sched_push(sass_mov_rr(r, s[k], c), src, 1, r, 0, FIXED_LAT);
            }
            break;
        case MK_EXIT:
            if (stage == NVK_FRAGMENT && g_stage_exit) {
                for (int k = 0; k < (int)m->imm; k++) {
                    uint8_t src[1] = { (uint8_t)(g_scratch + k) };
                    c = sched(src, 1, 0, k, 1, 0);
                    sched_push(sass_mov_rr(k, g_scratch + k, c), src, 1, k, 0, FIXED_LAT);
                }
            } else if (stage == NVK_FRAGMENT) {
                exit_moves((int)m->imm);
            }
            drain();
            if (g_demote) {
                uint8_t ps[1] = { P6_PSEUDO };
                c = sched(ps, 1, 0, -1, 0, 0);
                sched_push(sass_pred(sass_kill(7, c), 6, 0), ps, 1, -1, 0, 0);
            }
            c = sched(NULL, 0, 0, -1, 0, 0);
            sched_push(sass_exit(c), NULL, 0, -1, 0, 0);
            break;
        case MK_CALL:
            emit_call(m);
            break;
        case MK_ENTRY:
            window_moves(m->xd, m->nxd, 0);
            break;
        case MK_RET:
            emit_ret(m);
            break;
        default:
            airc_fail("cannot emit mop %d", m->k);
        }
    }
}

// The registers are allocated by now, so the order of a block's mops may change as long as every register and
// every memory space sees its reads and writes in the same relative order. A mop that starts or ends a block or
// moves control, a threadgroup barrier or a demote stays where it is, and nothing crosses it.

static int is_fence(const struct mop *m)
{
    switch (m->k) {
    case MK_BAR:
    case MK_BSSY:
    case MK_BSYNC:
    case MK_BRA:
    case MK_EXIT:
    case MK_DEMOTE:
    case MK_CALL:
    case MK_ENTRY:
    case MK_RET:
        return 1;
    default:
        return 0;
    }
}

enum { SP_NONE, SP_GLOBAL, SP_SHARED, SP_LOCAL, SP_ATTR, SP_COUNT };

// the memory space a mop touches and whether it writes there: stores and atomics keep their order against every
// other access of the space, loads move past each other
static int mem_space(const struct mop *m, int *store)
{
    *store = 0;
    switch (m->k) {
    case MK_LDG:
    case MK_TEX:
        return SP_GLOBAL;
    case MK_STG:
    case MK_SUST:
        *store = 1;
        return SP_GLOBAL;
    case MK_ATOM:
        *store = 1;
        return IR_ATOMA_SHARED(m->aux) ? SP_SHARED : SP_GLOBAL;
    case MK_LDS:
        return SP_SHARED;
    case MK_STS:
        *store = 1;
        return SP_SHARED;
    case MK_LDL:
        return SP_LOCAL;
    case MK_STL:
        *store = 1;
        return SP_LOCAL;
    case MK_AST:
        *store = 1;
        return SP_ATTR;
    case MK_ALD:
        return m->aux & IR_ATTR_OUTPUT ? SP_ATTR : SP_NONE;
    default:
        return SP_NONE;
    }
}

// roughly the cycles to a variable-latency result, for deciding what to start early
static unsigned result_latency(enum mkind k)
{
    switch (k) {
    case MK_TEX:
        return 60;
    case MK_LDG:
    case MK_ATOM:
        return 40;
    case MK_LDL:
    case MK_TXQ:
        return 30;
    case MK_LDS:
    case MK_ALD:
    case MK_ISBERD:
    case MK_SHFL:
        return 24;
    case MK_S2R:
    case MK_IPA:
    case MK_LDC:
        return 20;
    default:
        return 14;
    }
}

struct edge {
    int to;
    unsigned lat;
};

struct snode {
    int mop;
    struct fx fx;
    unsigned lat;      // to the result, when it is async
    struct edge *out;
    int nout, capout;
    int *producers;    // the async nodes whose results it reads
    int nprod, capprod;
    int npred;         // edges still to be satisfied
    unsigned avail;    // the cycle its operands are ready
    unsigned height;   // the longest chain of latencies from it to the end of the block
    int consumed;      // an async node: something has read its result
};

struct vec {
    int *v;
    int n, cap;
};

static void vec_push(struct vec *a, int x)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 8;
        a->v = realloc(a->v, a->cap * sizeof(*a->v));
    }
    a->v[a->n++] = x;
}

static void add_edge(struct snode *nodes, int from, int to, unsigned lat)
{
    struct snode *f = &nodes[from];
    if (f->nout == f->capout) {
        f->capout = f->capout ? f->capout * 2 : 4;
        f->out = realloc(f->out, f->capout * sizeof(*f->out));
    }
    f->out[f->nout++] = (struct edge){ to, lat };
    nodes[to].npred++;
}

static unsigned raw_edge(const struct snode *w, int reg, int rclass)
{
    if (w->fx.async)
        return w->lat;
    for (int k = 0; k < w->fx.nwr; k++)
        if (w->fx.wr[k] == reg)
            return w->fx.wclass[k] == WC_NONE ? 6 : raw_latency(w->fx.wclass[k], rclass);
    return 1;
}

// A segment's order, in order[lo..hi): the mops are all movable. sink is a register the fence after the segment
// reads (the predicate of a branch), whose last write should start early.
static void schedule_segment(int *order, int lo, int hi, int sink, enum nvk_stage stage)
{
    int n = hi - lo;
    struct snode *nodes = calloc(n, sizeof(*nodes));
    struct vec readers[256] = { 0 };
    struct vec mem_readers[SP_COUNT] = { 0 };
    int last_write[256], last_store[SP_COUNT];
    memset(last_write, 0xff, sizeof(last_write));
    memset(last_store, 0xff, sizeof(last_store));

    int ok = 1;
    for (int j = 0; j < n && ok; j++) {
        struct snode *nd = &nodes[j];
        nd->mop = lo + j;
        g_probe = 1;
        memset(&g_fx, 0, sizeof(g_fx));
        emit_mops_in(&nd->mop, 1, stage);
        g_probe = 0;
        nd->fx = g_fx;
        nd->lat = result_latency(g_mops[nd->mop].k);
        ok = !g_fx.overflow;
    }
    for (int j = 0; j < n && ok; j++) {
        struct snode *nd = &nodes[j];
        for (int a = 0; a < nd->fx.nrd; a++) {
            int reg = nd->fx.rd[a], w = last_write[reg];
            if (w >= 0) {
                add_edge(nodes, w, j, raw_edge(&nodes[w], reg, nd->fx.rclass[a]));
                if (nodes[w].fx.async) {
                    if (nd->nprod == nd->capprod) {
                        nd->capprod = nd->capprod ? nd->capprod * 2 : 4;
                        nd->producers = realloc(nd->producers, nd->capprod * sizeof(*nd->producers));
                    }
                    nd->producers[nd->nprod++] = w;
                }
            }
            vec_push(&readers[reg], j);
        }
        for (int k = 0; k < nd->fx.nwr; k++) {
            int reg = nd->fx.wr[k], w = last_write[reg];
            if (w >= 0 && w != j)
                add_edge(nodes, w, j, nodes[w].fx.async ? nodes[w].lat : 1);
            for (int q = 0; q < readers[reg].n; q++)
                if (readers[reg].v[q] != j)
                    add_edge(nodes, readers[reg].v[q], j, nodes[readers[reg].v[q]].fx.late ? 4 : 1);
            readers[reg].n = 0;
            last_write[reg] = j;
        }
        int store;
        int sp = mem_space(&g_mops[nd->mop], &store);
        if (sp != SP_NONE) {
            if (last_store[sp] >= 0)
                add_edge(nodes, last_store[sp], j, 1);
            if (store) {
                for (int q = 0; q < mem_readers[sp].n; q++)
                    add_edge(nodes, mem_readers[sp].v[q], j, 1);
                mem_readers[sp].n = 0;
                last_store[sp] = j;
            } else
                vec_push(&mem_readers[sp], j);
        }
    }

    for (int j = n - 1; j >= 0 && ok; j--) {
        struct snode *nd = &nodes[j];
        nd->height = nd->fx.async ? nd->lat : 1;
        if (sink >= 0 && last_write[sink] == j && raw_latency(WC_PRED, RC_DECOUPLED) > nd->height)
            nd->height = raw_latency(WC_PRED, RC_DECOUPLED);
        for (int e = 0; e < nd->nout; e++)
            if (nd->out[e].lat + nodes[nd->out[e].to].height > nd->height)
                nd->height = nd->out[e].lat + nodes[nd->out[e].to].height;
    }

    if (ok) {
        // each scoreboard held by a result nobody has read yet makes the next load share one
        enum { MAX_LIVE = 5 };
        int *ready = malloc(n * sizeof(*ready));
        int nready = 0, placed = 0, live = 0;
        unsigned now = 0;
        for (int j = 0; j < n; j++)
            if (!nodes[j].npred)
                ready[nready++] = j;
        while (placed < n) {
            int pick = -1;
            unsigned soonest = ~0u;
            int quiet = 0;  // a candidate that holds no scoreboard
            for (int r = 0; r < nready; r++) {
                struct snode *c = &nodes[ready[r]];
                if (c->avail < soonest)
                    soonest = c->avail;
                if (c->avail <= now && !c->fx.async)
                    quiet = 1;
            }
            for (int r = 0; r < nready; r++) {
                struct snode *c = &nodes[ready[r]];
                if (c->avail > now || (c->fx.async && live >= MAX_LIVE && quiet))
                    continue;
                if (pick < 0 || c->height > nodes[ready[pick]].height ||
                    (c->height == nodes[ready[pick]].height && ready[r] < ready[pick]))
                    pick = r;
            }
            if (pick < 0) {
                now = soonest;
                continue;
            }
            int j = ready[pick];
            ready[pick] = ready[--nready];
            struct snode *nd = &nodes[j];
            order[lo + placed++] = nd->mop;
            for (int p = 0; p < nd->nprod; p++)
                if (!nodes[nd->producers[p]].consumed) {
                    nodes[nd->producers[p]].consumed = 1;
                    live--;
                }
            live += nd->fx.async;
            for (int e = 0; e < nd->nout; e++) {
                struct snode *s = &nodes[nd->out[e].to];
                if (now + nd->out[e].lat > s->avail)
                    s->avail = now + nd->out[e].lat;
                if (!--s->npred)
                    ready[nready++] = nd->out[e].to;
            }
            now++;
        }
        free(ready);
    }
    for (int j = 0; j < n; j++) {
        free(nodes[j].out);
        free(nodes[j].producers);
    }
    for (int r = 0; r < 256; r++)
        free(readers[r].v);
    for (int s = 0; s < SP_COUNT; s++)
        free(mem_readers[s].v);
    free(nodes);
}

static void emit_mops(enum nvk_stage stage)
{
    int *order = malloc((g_nmops + 1) * sizeof(*order));
    char *start = calloc(g_nmops + 1, 1);
    for (int i = 0; i < g_nmops; i++) {
        order[i] = i;
        if (g_mops[i].label >= 0)
            start[g_mops[i].label] = 1;
    }
    for (struct ir_block *b = g_fn->entry; b; b = b->next)
        if (!b->dead)
            start[g_block_mop[b->id]] = 1;
    // AIRC_NOSCHED=1: keep the program order, for telling a reordering bug from other wrong output
    int plain = getenv("AIRC_NOSCHED") != NULL;
    for (int lo = 0; lo < g_nmops;) {
        int hi = lo + 1;
        if (!is_fence(&g_mops[lo]))
            while (hi < g_nmops && !start[hi] && !is_fence(&g_mops[hi]))
                hi++;
        if (!plain && hi - lo > 1) {
            int sink = hi < g_nmops && g_mops[hi].k == MK_BRA && g_mops[hi].aux ? P0_PSEUDO : -1;
            schedule_segment(order, lo, hi, sink, stage);
        }
        unsigned at = g_np;
        emit_mops_in(order + lo, hi - lo, stage);
        g_mop_np[lo] = at;
        lo = hi;
    }
    free(order);
    free(start);
}

unsigned char *airc_back(struct ir_func *f, const struct airc_info *info, unsigned *size, unsigned *regs)
{
    g_info = (struct airc_info *)info;
    spill_values(f);
    g_nmops = 0;
    g_nvregs = 0;
    g_nfix = 0;
    g_exit_regs = 0;
    g_demote = 0;
    g_np = 0;
    g_mem_bars = 0;
    g_cycle = 0;
    for (int i = 0; i < 256; i++)
        g_barrier_of[i] = g_rbar_of[i] = -1;
    memset(g_wclass, WC_NONE, sizeof(g_wclass));
    memset(g_bar_set, 0, sizeof(g_bar_set));

    memset(g_consts, 0xff, sizeof(g_consts));  // block -1: all free
    g_one = new_vreg(1, 1);
    struct mop *one = emit_mop(MK_MOVIMM);
    one->def = g_one;
    one->imm = 1;
    // every phi has its vreg before any block is lowered: a predecessor's copies may come first
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v && v->op == IR_PHI; v = v->next) {
            int m[2];
            v->vreg = v->type == IR_I64 ? new_cluster(2, 2, m) : new_vreg(1, 1);
        }
    g_fn = f;
    g_win_end = 0;
    plan_regions(f);
    find_fused_compares(f);
    g_no_mods = airc_pass_off("AIRC_NOMODS");
    find_needed_mods(f);
    find_half_pairs(f);
    count_uses(f);
    phi_outside(f);
    memset(g_region_start, 0, sizeof(g_region_start));
    memset(g_region_end, 0, sizeof(g_region_end));
    for (struct ir_block *b = f->entry, *prev = NULL; b; b = b->next)
        if (!b->dead) {
            if (b->func && (!prev || prev->func != b->func))
                g_region_start[b->func] = g_nmops;
            lower_block(b);
            g_region_end[b->func] = g_nmops;
            prev = b;
        }
    add_externals();
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->vreg >= 0 && (v->type == IR_I32 || v->type == IR_F32) && v->op != IR_PROJ && v->op != IR_PARAM &&
                v->op != IR_CALL && g_vregs[v->vreg].size == 1)
                g_vregs[v->vreg].ir = v->id;
            else if (v->vreg >= 0 && v->type == IR_VOID && g_vregs[v->vreg].size > 1 &&
                     (v->op == IR_LDG || v->op == IR_LDL || v->op == IR_TEX))
                g_vregs[v->vreg].ir = v->id;

    memset(g_reserved, 0, sizeof(g_reserved));
    g_regcount = 2;
    for (int r = 0; r < g_nregions; r++)
        g_reserved[r][1] = 1;
    for (int k = 0; k < g_nregions; k++)
        regalloc(f, g_topo[k]);
    if (g_win_end > g_regcount)
        g_regcount = g_win_end;
    g_scratch = g_regcount;
    g_stage_exit = airc_pass_off("AIRC_STAGEEXIT");
    if (g_stage_exit) {
        g_regcount += g_exit_regs > 4 ? g_exit_regs : 4;
        g_exit_tmp = g_regcount - 1;
    } else {
        // the colours land in R0.. and the one temporary a cycle of moves needs is past them
        g_exit_tmp = g_scratch > g_exit_regs ? g_scratch : g_exit_regs;
        g_regcount = g_exit_tmp + 1;
    }

    g_mop_np = calloc(g_nmops + 1, sizeof(*g_mop_np));
    g_block_np = calloc(f->nblocks, sizeof(*g_block_np));

    // prologue: R1 = stack pointer, UR4 = the global memory base (LDG reads it implicitly)
    sched_push(sass_mov_cbuf(1, 0, 0x28, sched(NULL, 0, 0, 1, 1, 0)), NULL, 0, 1, 0, FIXED_LAT);
    sched_push(sass_uldc64_cbuf(4, 0, 0x118, sched(NULL, 0, 0, -1, 0, 0)), NULL, 0, UR4_PSEUDO, 0, UNIFORM_LAT);
    if (g_demote)
        sched_push(sass_isetp(6, SASS_RZ, SASS_RZ, ISETP_NE, 0, sched(NULL, 0, 0, P6_PSEUDO, 1, 0)), NULL, 0, P6_PSEUDO,
                   0, PRED_LAT);

    emit_mops(info->stage);
    for (struct ir_block *b = f->entry; b; b = b->next)
        if (!b->dead)
            g_block_np[b->id] = g_mop_np[g_block_mop[b->id]];

    prog_emit(sass_bra_self(CTRL_STALL1));
    while (g_np % 8 || g_np < 16)
        prog_emit(sass_nop(CTRL_STALL1));

    for (int i = 0; i < g_nfix; i++) {
        int target = g_fix[i].block >= 0 ? g_block_np[g_fix[i].block] : (int)g_mop_np[g_fix[i].mop];
        set_rel(&g_prog[g_fix[i].at], target * 16 - ((int)g_fix[i].at * 16 + 0x10));
    }

    unsigned hdr = info->stage == NVK_COMPUTE ? 0 : SPH_BYTES;
    *size = hdr + g_np * 16;
    unsigned char *out = malloc(*size);
    if (info->stage == NVK_VERTEX)
        sph_vertex((uint32_t *)out, info->vertex_inputs, info->ngeneric, info->sysvals);
    else if (info->stage == NVK_FRAGMENT)
        sph_pixel((uint32_t *)out, info->interp, info->nvary, info->ps_sysvals, info->targets);
    else if (info->stage == NVK_TESS_INIT)
        sph_tess_init((uint32_t *)out, info->vertex_inputs, info->point_outputs, info->patch_outputs,
                      info->patch_points);
    else if (info->stage == NVK_TESS_EVAL)
        sph_tess_eval((uint32_t *)out, info->vertex_inputs, info->ngeneric, info->sysvals);
    if (hdr)  // SHADER_LOCAL_MEMORY_LOW_SIZE
        ((uint32_t *)out)[1] |= ((info->local_size + 15) & ~15u) & 0xffffff;
    // AIRC_SERIAL=1: every instruction waits on all six scoreboards and stalls 15 cycles, for telling a scheduling
    // hazard (a result read before it lands) from other wrong output
    if (getenv("AIRC_SERIAL"))
        for (unsigned i = 0; i < g_np; i++)
            g_prog[i].hi |= 0x3full << 52 | 0xfull << 41;
    sass_emit(g_prog, g_np, out + hdr);

    unsigned r = (g_regcount + 2 + 7) & ~7u;  // + 2 for the program counter, rounded to the 8-register granule
    *regs = r < 8 ? 8 : r;

    free(g_mop_np);
    free(g_block_np);
    free(g_prog);
    g_prog = NULL;
    g_prog_cap = 0;
    return out;
}
