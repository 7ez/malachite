// Predicates live as 0/1 GPRs and P0 never stays live, so there is one register bank to allocate; a discarding
// fragment shader also holds P6, the demoted flag, throughout.
#include <stdlib.h>
#include <string.h>
#include "airc_int.h"
#include "../core/sph.h"
#include "sass86.h"

enum mkind {
    MK_MOVIMM,   // def = imm
    MK_MOV,      // def = src0 (aux: a phi copy)
    MK_MOVCBUF,  // def = c[0][imm]
    MK_LDC,      // def = c[0][src0 + imm]
    MK_ULDC,     // UR4 = c[0][imm]
    MK_S2R,      // def = SR[imm]
    MK_ALD,      // def = a[imm] of vertex src0 if there is one (aux IR_ATTR_*)
    MK_IPA,      // def = a[imm], interpolated (aux: flat)
    MK_ISBERD,   // def = the input vertex handle at map offset src0
    MK_IADD3,    // def = src0 + src1
    MK_ISUB,     // def = src0 - src1
    MK_IMAD,     // def = src0 * src1, + src2 if there is one (src1 then VREG_RZ with immb)
    MK_IMADHI,   // def = the high word of src0 * src1 (aux: signed)
    MK_LOP3,     // def = src0 lut src1 (aux = lut)
    MK_PRMT,     // def = the bytes of src0 (and zero) selector imm picks
    MK_SHL,      // def = src0 << src1
    MK_SHR,      // def = src0 >> src1 (aux: arithmetic)
    MK_IMNMX,    // def = min/max(src0, src1) (aux bit0 max, bit1 signed)
    MK_FADD,     // aux bit0: subtract; aux bit1: half
    MK_FMUL,     // aux bit1: half
    MK_FFMA,     // aux bit1 half, bits 2:3 SASS_RND_*, bit 4 imm multiplier (bit 4 on FADD/FMUL: imm src b),
    // bit 5 imm addend, bit 6 .SAT on all three
    MK_FMNMX,    // aux bit0: max
    MK_MUFU,     // aux = MUFU_*
    MK_I2F,      // aux: signed source
    MK_F2I,      // aux: signed result
    MK_F2IP,     // def = src0 to an unsigned byte, saturated
    MK_F2H,
    MK_H2F,
    MK_FRND,     // aux = SASS_RND_*
    MK_POPC,
    MK_BREV,
    MK_FLO,
    MK_IABS,
    MK_VOTE,     // def = the lanes where P0 is set (aux: all active lanes)
    MK_SHFL,     // def = src0 of another lane, aux = SASS_SHFL_* mode | c << 4; lane imm, or src1 if there is one
    MK_FSWZADD,  // def = quad swizzle add of src0 (neighbour) and src1 (pixel), aux = pattern
    MK_LOP3_IMM,  // def = src0 lut imm (aux = lut) -- for fneg/fabs sign masks
    MK_ISETP,     // P0 = src0 cond src1 (aux = enum ir_cond | IR_SIGNED | SETP_AND or SETP_OR)
    MK_FSETP,     // P0 = src0 fcond src1 (aux = FSETP_* | SETP_AND or SETP_OR)
    MK_SEL,       // def = P0 ? src0 : src1
    MK_LDG,       // def = [src0 pair + imm], aux bytes (8 or 16: def is a pair or quad)
    MK_ATOM,      // def = old [src0 + imm], src1 data, src2 CAS value, aux IR_ATOMA_*; no def: RED or ATOMS to RZ
    MK_STG,       // [src0 pair + imm] = src1, aux bytes (8 or 16: src1 is a pair or quad)
    MK_LDS,       // def = shared[src0 + imm] (src0 optional), aux bytes
    MK_STS,       // shared[src0 + imm] = src1, aux bytes
    MK_LDL,       // def = local[src0 + imm] (src0 optional), aux bytes (8 or 16: def is a pair or quad)
    MK_STL,       // local[src0 + imm] = src1 (a pair or quad when wide), aux bytes
    MK_BAR,       // aux: a memory fence instead
    MK_IMADWIDE,  // def pair = src0 * src1 (or imm, immb) + src2 pair (aux: signed index)
    MK_TEX,       // def quad = fetch(src0, src1) handle imm (aux: sample)
    MK_TXQ,       // def = a texture query (aux as IR_TXQ's)
    MK_SUST,      // texel (src0, src1) of texture imm = src2 quad
    MK_AST,       // a[imm] = src0.. (aux & 15 values) of vertex src1 if there is one (aux IR_ATTR_PATCH)
    MK_BRA,       // to block `target`; aux 1 = @P0, 2 = @!P0
    MK_OUTPUT,    // fragment: scratch[imm + k] = src k, the colour outputs on their way to R0..
    MK_EXIT,      // fragment: the imm colour registers scratch[0..] into R0.. first
    MK_DEMOTE,    // P6 |= src0 (aux: its inverse), or P6 = true without one: the exit drops the fragment
    MK_BSSY,      // convergence barrier imm, the threads meeting again at block `target`
    MK_BSYNC,     // wait at barrier imm for the threads it was set for
    MK_CALL,      // subroutine imm at block `target`, xs in, xd out; aux the reconvergence barrier or -1
    MK_ENTRY,     // a subroutine's start: xd out of the window
    MK_RET,       // a subroutine's return: xs into the window
};

// The subroutine window: return address in R2 (R2:R3 for RET), arguments then results from R4. Values live across
// the moves stay out of it, and a callee avoids its callers' live registers and barriers.
#define WIN_RA 2
#define WIN_ARGS 4

#define SLOT_RA (-1)
#define SLOT_KEEP (-2)  // a caller value the callee reads, kept live rather than moved
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
    int immb;    // src1 is the immediate imm instead (LOP3, IADD3, IMAD, IMNMX, FMNMX, SHL, SHR, ISETP, FSETP, SEL)
    int target;  // branch target block id, or -1
    int label;   // branch target mop index (a stub), or -1
    struct xfer *xs, *xd;  // window moves of a call, entry or return
    int nxs, nxd;
    int mods;  // float sources' modifiers, MOD_NEG and MOD_ABS two bits a source (FADD, FMUL, FFMA, FSETP)
    int ur;    // LDG, STG: uniform base pair with src0 the offset (word when narrow, or RZ); 0 if src0 is the address
    int narrow;
    int readonly;  // LDG through the read-only cache
    int guarded;   // LDG under P0
};

// ISETP/FSETP aux: combine with the previous P0
#define SETP_AND 64
#define SETP_OR 128

#define MAX_MOPS (1 << 17)
static struct mop g_mops[MAX_MOPS];
static int g_nmops;
static int g_block_mop[1 << 16];

// Volta+ doesn't reconverge after a branch by itself, so a quad shuffle after an if/else would read lanes still on
// the other side: branches and loop entries set a BSSY to where the paths meet, which BSYNCs there.
static int g_bssy[1 << 16];     // -1 for none
static int g_bssy_to[1 << 16];
static int g_bsync[1 << 16][4];  // innermost first
static int g_nbsync[1 << 16];

// a loop: something at or before b in block order reached from b without passing m
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

// the blocks of loop h reaching b without passing h
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

// returns the most barriers the function holds at once
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
    // post-dominators to a fixed point; blocks are in reverse postorder
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
    // paths meet at the nearest strict post-dominator
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
        // lanes leave a loop one by one, and a barrier or shuffle after it must wait for the rest
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
        // past every block of the loop
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
        int inside = 0;  // inner regions close before it
        for (int i = 0; loop && i < nb; i++)
            inside |= in_loop[i] && order[i] > order[m->id];
        if (inside)
            continue;
        memset(seen, 0, nb);
        if (loops_before(from, m, order, seen, k))
            continue;
        // a switch's compare chain or an else-if shares the open region's barrier
        if (depth && g_bssy_to[stack[depth - 1]] == m->id)
            continue;
        if (depth && order[m->id] > order[g_bssy_to[stack[depth - 1]]]) {
            ok = 0;  // not nested, so no barriers for this function
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
static uint32_t g_callees[IR_MAX_SUBS + 1];
static int g_topo[IR_MAX_SUBS + 1];      // callers before callees
static int g_bar_base[IR_MAX_SUBS + 1];  // barriers below it are its callers'
static int g_wrap_bar[IR_MAX_SUBS + 1];  // where its calls reconverge, -1 for none

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
    for (int r = 0; r < g_nregions; r++)  // unreached regions have no blocks
        if (!seen[r])
            topo_visit(r, seen, &n);
    int used[IR_MAX_SUBS + 1];
    for (int r = 0; r < g_nregions; r++) {
        used[r] = plan_convergence(f, r);
        g_bar_base[r] = 0;
    }
    for (int k = 0; k < g_nregions; k++) {
        int r = g_topo[k], calls = g_callees[r] != 0;
        if (g_bar_base[r] + used[r] + calls > 16) {  // out of the 16 barriers: this one goes without
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
    int next;         // next cluster member, or -1
    int hw;
    int start, end;   // in mop order, holes and all
    int seg;          // first of g_segs, -1 for none
    int ir;           // IR value id for the spiller, -1 for none
};

#define MAX_VREGS 65536
#define VREG_RZ (-2)
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
static int g_one;  // the integer 1, for materialising booleans
static int g_block;
static int g_scratch;  // above the allocation: temporaries and exit colours
static int g_exit_regs;
static int g_exit_tmp;   // where exit moves park a value
static int g_demote;     // P6 holds the demoted flag

static int mov_imm(int64_t imm)
{
    int r = new_vreg(1, 1);
    struct mop *m = emit_mop(MK_MOVIMM);
    m->def = r;
    m->imm = imm;
    return r;
}

// per block, so no constant holds a register across the whole shader; another block's entry counts as free
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

// an I64 is a pair cluster named by its low member
static int vreg_of(struct ir_val *v)
{
    if (v->op == IR_CONST || v->op == IR_UNDEF)
        return const_vreg(v->op == IR_CONST ? v->imm : 0);
    if (v->vreg < 0)
        airc_fail("value %%%d used before it is produced (op %d)", v->id, v->op);
    return v->vreg;
}

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

// A value read once, made a few mops back, goes straight into the cluster register it'd be copied to (texture
// operands, wide stores, vertex outputs). AIRC_NODEFINTO=1 keeps the copy.
static int *g_use_count;
static int g_no_define_into;
// bitcasts (or 0) sharing their source's register; phi_into_def skips them so the two don't split
static unsigned char *g_alias;

static void count_uses(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_use_count);
    free(g_alias);
    g_use_count = calloc(n, sizeof(*g_use_count));
    g_alias = calloc(n, 1);
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

// Compares read only as a select or same-block branch condition go straight to P0, saving a SEL, an ISETP and their
// latency; predicate and/or chains do too via SETP.AND/.OR. AIRC_NOFUSEBR=1 leaves branches out.
static unsigned char *g_cmp_fused;

static int is_pand_or(const struct ir_val *v)
{
    return v->op == IR_PAND || v->op == IR_POR;
}

static int is_condition(const struct ir_val *v, unsigned i)
{
    return (v->op == IR_SEL && i == 0) || (v->op == IR_LDG && i == 1);
}

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
            g_cmp_fused[v->id] = v->op == IR_CMP || is_pand_or(v);
    int branches = !airc_pass_off("AIRC_NOFUSEBR");
    // a chain of SETP.AND/.OR needs a compare or a register on one side of each link
    for (int again = 1; again;) {
        again = 0;
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *v = b->first; v; v = v->next) {
                for (unsigned i = 0; i < v->nsrc; i++) {
                    struct ir_val *c = v->src[i];
                    if (c && is_pand_or(c) && g_cmp_fused[c->id] &&
                        !(c->block == b && (is_condition(v, i) || (branches && v->op == IR_CBR) ||
                                            (is_pand_or(v) && g_cmp_fused[v->id])))) {
                        g_cmp_fused[c->id] = 0;
                        again = 1;
                    }
                }
                if (is_pand_or(v) && g_cmp_fused[v->id] && is_pand_or(v->src[0]) && g_cmp_fused[v->src[0]->id] &&
                    is_pand_or(v->src[1]) && g_cmp_fused[v->src[1]->id]) {
                    g_cmp_fused[v->id] = 0;
                    again = 1;
                }
            }
    }
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            for (unsigned i = 0; i < v->nsrc; i++) {
                struct ir_val *c = v->src[i];
                if (!c || c->op != IR_CMP || is_condition(v, i))
                    continue;
                if (is_pand_or(v) && g_cmp_fused[v->id] && c->block == b)
                    continue;
                if (!(branches && v->op == IR_CBR && c->block == b))
                    g_cmp_fused[c->id] = 0;
            }
}

// fneg/fabs become source modifiers on FADD, FMUL, FFMA and FSETP; FMNMX's denormal handling with one is unknown.
// They only touch the sign bit like the LOP3, so NaNs and signed zeros match. AIRC_NOMODS=1 turns it off.
#define MOD_NEG 1
#define MOD_ABS 2
static unsigned char *g_mod_needed;  // still read as a register
static int g_no_mods;

// a constant underneath is left alone for the immediate forms
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

// Two nearby float->half->float roundings share one F2FP.PACK_AB, read back from each half. Only halves nothing but
// the conversion back reads are paired. AIRC_NOPACKH=1 turns it off.
static struct ir_val **g_pair;  // the second's partner
static unsigned char *g_pair_hi, *g_first;

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

static int is_imm(const struct ir_val *v)
{
    return v->op == IR_CONST && v->type != IR_I64;
}

// a multiply read only by a same-block add becomes IMAD (y * width + x); a constant addend stays IADD3's immediate
static unsigned char *g_mul_fused, *g_add_fused;
static unsigned char *g_late_def;  // made with the phi copies (lower_late_def)

static void find_fused_muls(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_mul_fused);
    g_mul_fused = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next) {
            if (v->op != IR_ADD || v->type != IR_I32)
                continue;
            for (int i = 0; i < 2; i++) {
                struct ir_val *m = v->src[i];
                if (m->op == IR_MUL && m->type == IR_I32 && m->block == b && g_use_count[m->id] == 1 &&
                    !is_imm(v->src[!i]) && !(is_imm(m->src[0]) && is_imm(m->src[1]))) {
                    g_mul_fused[m->id] = 1;
                    break;
                }
            }
        }
    // a register sum read only by another same-block sum becomes its IADD3's third source
    free(g_add_fused);
    g_add_fused = calloc(n, 1);
    unsigned char *three = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next) {
            if (v->op != IR_ADD || v->type != IR_I32 || is_imm(v->src[0]) || is_imm(v->src[1]) ||
                g_mul_fused[v->src[0]->id] || g_mul_fused[v->src[1]->id])
                continue;
            for (int i = 0; i < 2; i++) {
                struct ir_val *u = v->src[i];
                if (u->op == IR_ADD && u->type == IR_I32 && u->block == b && g_use_count[u->id] == 1 && !three[u->id] &&
                    !is_imm(u->src[0]) && !is_imm(u->src[1]) && !g_mul_fused[u->src[0]->id] &&
                    !g_mul_fused[u->src[1]->id]) {
                    g_add_fused[u->id] = three[v->id] = 1;
                    break;
                }
            }
        }
    free(three);
}

static int lower_add3(struct ir_val *v)
{
    int i = v->src[0]->op == IR_ADD && g_add_fused[v->src[0]->id] ? 0 : 1;
    struct ir_val *u = v->src[i];
    if (u->op != IR_ADD || !g_add_fused[u->id] || u->vreg >= 0)
        return 0;
    v->vreg = new_vreg(1, 1);
    struct mop *m = emit_mop(MK_IADD3);
    m->def = v->vreg;
    m->src[0] = vreg_of(u->src[0]);
    m->src[1] = vreg_of(u->src[1]);
    m->src[2] = vreg_of(v->src[!i]);
    m->nsrc = 3;
    return 1;
}

static int lower_mul_add(struct ir_val *v)
{
    int i = v->src[0]->op == IR_MUL && g_mul_fused[v->src[0]->id] ? 0 : 1;
    struct ir_val *mul = v->src[i], *a = mul->src[0], *b = mul->src[1];
    if (mul->op != IR_MUL || !g_mul_fused[mul->id] || mul->vreg >= 0)
        return 0;
    if (is_imm(a)) {
        a = mul->src[1];
        b = mul->src[0];
    }
    int ra = vreg_of(a), rb = is_imm(b) ? VREG_RZ : vreg_of(b), rc = vreg_of(v->src[!i]);
    v->vreg = new_vreg(1, 1);
    struct mop *m = emit_mop(MK_IMAD);
    m->def = v->vreg;
    m->src[0] = ra;
    m->src[1] = rb;
    m->src[2] = rc;
    m->nsrc = 3;
    if (is_imm(b)) {
        m->immb = 1;
        m->imm = (uint32_t)b->imm;
    }
    return 1;
}

// Single-reader values folded into their reader: byte extracts and sign extensions into PRMT, a [0, 255] clamped
// conversion into F2IP.U8, and an FMUL/FFMA under FADD.SAT x, +0 into .SAT, which rounds and flushes alike.
static unsigned char *g_folded, *g_sat;

// -1 if v isn't a PRMT of *word
static int byte_select(const struct ir_val *v, struct ir_val **word)
{
    if (v->type != IR_I32 || v->nsrc != 2)
        return -1;
    struct ir_val *x = v->src[0], *k = v->src[1];
    if (v->op == IR_AND && x->op == IR_CONST) {
        x = v->src[1];
        k = v->src[0];
    }
    if (k->op != IR_CONST || x->type != IR_I32 || (x->op != IR_SHR && x->op != IR_SAR && x->op != IR_SHL) ||
        x->src[1]->op != IR_CONST)
        return -1;
    int64_t s = x->src[1]->imm;
    if (v->op == IR_AND && (x->op == IR_SHR || x->op == IR_SAR) && (k->imm == 0xff || k->imm == 0xffff) &&
        (s == 8 || s == 16 || (s == 24 && k->imm == 0xff))) {
        *word = x;
        return k->imm == 0xff ? 0x4440 | (int)s / 8 : 0x4400 | ((int)s / 8 + 1) << 4 | (int)s / 8;
    }
    if (v->op == IR_SAR && x->op == IR_SHL && k->imm == s && (s == 16 || s == 24)) {
        *word = x;
        return s == 24 ? 0x8880 : 0x9910;
    }
    return -1;
}

static int is_float(const struct ir_val *v, uint32_t bits)
{
    return v->op == IR_CONST && (uint32_t)v->imm == bits;
}

// F2IP saturates a NaN to 0, matching a clamp after F2I or fmin(fmax(x, +0), 255.0) before it, or a saturate
// times under 256; *inner must fold too.
static struct ir_val *byte_convert(const struct ir_val *v, struct ir_val **inner)
{
    struct ir_val *d;
    if (v->op == IR_MIN && !v->aux && v->type == IR_I32)
        for (int i = 0; i < 2; i++) {
            struct ir_val *c = v->src[i], *k = v->src[!i];
            if (c->op == IR_F2I && !c->aux && c->src[0]->type == IR_F32 && k->op == IR_CONST && k->imm == 255 &&
                !byte_convert(c, &d)) {
                *inner = c;
                return c->src[0];
            }
        }
    if (v->op != IR_F2I || v->aux || v->src[0]->type != IR_F32)
        return NULL;
    struct ir_val *x = v->src[0];
    if (x->op == IR_FMIN && is_float(x->src[1], 0x437f0000) && x->src[0]->op == IR_FMAX &&
        is_float(x->src[0]->src[1], 0)) {
        *inner = x;
        return x->src[0]->src[0];
    }
    for (int i = 0; x->op == IR_FMUL && i < 2; i++) {
        struct ir_val *s = x->src[i], *k = x->src[!i];
        // positive float bits order as integers: 0 < k < 256.0
        if (s->op == IR_FADD && (s->aux & IR_FADD_SAT) && k->op == IR_CONST && (uint32_t)k->imm > 0 &&
            (uint32_t)k->imm < 0x43800000) {
            *inner = NULL;
            return x;
        }
    }
    return NULL;
}

static struct ir_val *sat_source(const struct ir_val *v)
{
    if (v->op != IR_FADD || !(v->aux & IR_FADD_SAT) || v->type != IR_F32 || v->src[1]->op != IR_CONST ||
        v->src[1]->imm != 0)
        return NULL;
    struct ir_val *x = v->src[0];
    return (x->op == IR_FMUL || x->op == IR_FFMA) && x->type == IR_F32 ? x : NULL;
}

// the front end's abs is max(x, 0 - x)
static struct ir_val *abs_negation(const struct ir_val *v)
{
    if (v->op != IR_MAX || !v->aux || v->type != IR_I32)
        return NULL;
    for (int i = 0; i < 2; i++) {
        struct ir_val *n = v->src[i];
        if (n->op == IR_SUB && n->src[0]->op == IR_CONST && !n->src[0]->imm && n->src[1] == v->src[!i])
            return n;
    }
    return NULL;
}

static void find_folds(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_folded);
    free(g_sat);
    g_folded = calloc(n, 1);
    g_sat = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next) {
            struct ir_val *in;
            if (byte_select(v, &in) >= 0) {
                g_folded[in->id] = g_use_count[in->id] == 1 && in->block == b;
            } else if (byte_convert(v, &in)) {
                if (in) {
                    struct ir_val *max = in->op == IR_FMIN ? in->src[0] : in;
                    g_folded[in->id] = g_folded[max->id] = g_use_count[in->id] == 1 && in->block == b &&
                                                           g_use_count[max->id] == 1 && max->block == b;
                }
            } else if ((in = sat_source(v))) {
                g_sat[in->id] = g_use_count[in->id] == 1 && in->block == b;
            } else if ((in = abs_negation(v))) {
                g_folded[in->id] = g_use_count[in->id] == 1 && in->block == b;
            }
        }
}

// A tree of single-use same-block logic ops becomes one LOP3 if it has at most three inputs (an immediate counts as
// b; 0 and ~0 are free), covering copysign, sign masks and bit packing.
#define LOP3_A 0xf0
#define LOP3_B 0xcc
#define LOP3_C 0xaa

struct lop3 {
    struct ir_val *reg[3];  // read as a, c, b
    int nreg, has_imm;
    uint32_t imm;  // b
    uint8_t root, lut;
};
static struct lop3 *g_lop3;
static unsigned char *g_lop3_in;  // inside the root being built

static int is_logic(const struct ir_val *v)
{
    return (v->op == IR_AND || v->op == IR_OR || v->op == IR_XOR) && (v->type == IR_I32 || v->type == IR_F32);
}

// -1 when u needs too many inputs
static int lop3_eval(struct ir_val *u, struct ir_val *root, struct lop3 *l)
{
    if (u == root || g_lop3_in[u->id]) {
        int x = lop3_eval(u->src[0], root, l), y = x < 0 ? -1 : lop3_eval(u->src[1], root, l);
        if (y < 0)
            return -1;
        return u->op == IR_AND ? x & y : u->op == IR_OR ? x | y : x ^ y;
    }
    if (u->op == IR_CONST) {
        uint32_t k = (uint32_t)u->imm;
        if (!k || k == ~0u)
            return k ? 0xff : 0;
        if (!l->has_imm && l->nreg < 3) {
            l->has_imm = 1;
            l->imm = k;
        }
        return l->has_imm && k == l->imm ? LOP3_B : l->has_imm && k == ~l->imm ? (~LOP3_B & 0xff) : -1;
    }
    static const int pattern[3] = { LOP3_A, LOP3_C, LOP3_B };
    for (int i = 0; i < l->nreg; i++)
        if (l->reg[i] == u)
            return pattern[i];
    if (l->nreg == (l->has_imm ? 2 : 3))
        return -1;
    l->reg[l->nreg] = u;
    return pattern[l->nreg++];
}

// only nearby operands fold, so theirs aren't kept live much longer
#define LOP3_NEAR 8

static int lop3_inner(const struct ir_val *u, const struct ir_val *root)
{
    struct ir_val *in;
    if (!is_logic(u) || u->block != root->block || g_use_count[u->id] != 1 || g_folded[u->id] || g_lop3_in[u->id] ||
        byte_select(u, &in) >= 0)
        return 0;
    int k = 0;
    for (const struct ir_val *w = root->prev; w && w != u && k <= LOP3_NEAR; w = w->prev)
        k++;
    return k <= LOP3_NEAR;
}

static void find_lop3(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_lop3);
    free(g_lop3_in);
    g_lop3 = calloc(n, sizeof(*g_lop3));
    g_lop3_in = calloc(n, 1);
    unsigned char *taken = calloc(n, 1);
    struct ir_val *in;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->last; v; v = v->prev) {
            if (!is_logic(v) || taken[v->id] || g_folded[v->id] || g_late_def[v->id] || byte_select(v, &in) >= 0 ||
                byte_convert(v, &in))
                continue;
            struct ir_val *open[16], *inner[16];
            int nopen = 0, ninner = 0;
            open[nopen++] = v->src[0];
            open[nopen++] = v->src[1];
            while (nopen && ninner < 16) {
                struct ir_val *u = open[--nopen];
                if (!lop3_inner(u, v) || taken[u->id])
                    continue;
                g_lop3_in[u->id] = 1;
                struct lop3 l = { 0 };
                if (lop3_eval(v, v, &l) < 0) {
                    g_lop3_in[u->id] = 0;
                    continue;
                }
                inner[ninner++] = u;
                if (nopen < 15) {
                    open[nopen++] = u->src[0];
                    open[nopen++] = u->src[1];
                }
            }
            if (!ninner)
                continue;
            struct lop3 l = { 0 };
            l.lut = (uint8_t)lop3_eval(v, v, &l);
            l.root = 1;
            g_lop3[v->id] = l;
            for (int i = 0; i < ninner; i++) {
                taken[inner[i]->id] = g_folded[inner[i]->id] = 1;
                g_lop3_in[inner[i]->id] = 0;
            }
        }
    free(taken);
}

static int lower_lop3(struct ir_val *v)
{
    struct lop3 *l = &g_lop3[v->id];
    if (!l->root)
        return 0;
    v->vreg = new_vreg(1, 1);
    struct mop *m = emit_mop(MK_LOP3);
    m->def = v->vreg;
    m->aux = l->lut;
    m->src[0] = l->nreg > 0 ? vreg_of(l->reg[0]) : VREG_RZ;
    int c = l->nreg > 1 ? vreg_of(l->reg[1]) : VREG_RZ;
    if (l->has_imm) {
        m->immb = 1;
        m->imm = l->imm;
        m->src[1] = c;
        m->nsrc = 2;
    } else {
        m->src[1] = l->nreg > 2 ? vreg_of(l->reg[2]) : VREG_RZ;
        m->src[2] = c;
        m->nsrc = 3;
    }
    return 1;
}

static int lower_byte_convert(struct ir_val *v)
{
    struct ir_val *in, *x = byte_convert(v, &in);
    if (!x || (in && !g_folded[in->id]))
        return 0;
    v->vreg = new_vreg(1, 1);
    struct mop *m = emit_mop(MK_F2IP);
    m->def = v->vreg;
    m->src[0] = vreg_of(x);
    m->nsrc = 1;
    return 1;
}

// Unread atomics become RED or ATOMS to RZ, saving a register and write barrier, unless a barrier, fence or call may
// follow: barriers wait on atomics' write barriers (g_mem_bars), and ptxas also keeps ATOMG before a fence.
static unsigned char *g_reduce;

static void find_reductions(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_reduce);
    g_reduce = calloc(n, 1);
    unsigned char *reach = calloc(f->nblocks, 1);  // reaches a barrier, fence or call
    int anywhere = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->op == IR_BAR || v->op == IR_CALL)
                reach[b->id] = anywhere = 1;
    for (int changed = 1; changed;) {
        changed = 0;
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (unsigned k = 0; k < b->nsucc && !reach[b->id]; k++)
                if (reach[b->succ[k]->id])
                    reach[b->id] = changed = 1;
    }
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next) {
            int op = IR_ATOMA_OP(v->aux);
            if (v->op != IR_ATOM || g_use_count[v->id] || op == SASS_ATOM_CAS)
                continue;
            // a subroutine's caller may reach one
            int later = (op == SASS_ATOM_EXCH && !IR_ATOMA_SHARED(v->aux)) || (b->func && anywhere);
            for (struct ir_val *u = v->next; u && !later; u = u->next)
                later = u->op == IR_BAR || u->op == IR_CALL;
            for (unsigned k = 0; k < b->nsucc && !later; k++)
                later = reach[b->succ[k]->id];
            g_reduce[v->id] = !later;
        }
    free(reach);
}

// Buffer bases are uniform, so they load once into UR pairs and accesses use [R.64+UR] as NAK does, saving a pair
// per base and sharing offsets. Offsets stay 64-bit for signed indices and buffers past 4 GB, except unsigned 32-bit
// ones ([R.U32+UR]). An address used other than by a load or store is built whole, with its base.
#define UR_FIRST 6  // UR4:UR5 hold the memory descriptor
#define UR_PAIRS 28
static int *g_ur;  // 0 for none
static unsigned char *g_whole;  // a base also read as a whole address
static unsigned char *g_narrow;  // a zero-extended 32-bit offset
static struct {
    int64_t off;  // in const bank 0
    int used;
} g_ubase[UR_PAIRS];
static int g_nubase;

// this block's offsets, by index and scale
static struct offset {
    int block;
    struct ir_val *index;
    int64_t scale;
    int aux;
    int vreg;
} g_offsets[32];
static int g_noffsets;

// texture operand vectors this block gathered, reused by later ops with the same values
static struct texvec {
    int block, n;
    struct ir_val *val[4];
    int vreg;
} g_texvecs[8];
static int g_ntexvecs;
int airc_no_texvec, airc_texvec_shared;

static int same_value(const struct ir_val *a, const struct ir_val *b)
{
    return a == b || (a->op == IR_CONST && b->op == IR_CONST && a->imm == b->imm && a->type == b->type);
}

static int address_use(const struct ir_val *u, unsigned i)
{
    return i == 0 && (u->op == IR_LDG || u->op == IR_STG || (u->op == IR_ADDR && g_ur[u->id]));
}

static void find_uniform_bases(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_ur);
    free(g_whole);
    free(g_narrow);
    g_ur = calloc(n, sizeof(*g_ur));
    g_whole = calloc(n, 1);
    g_narrow = calloc(n, 1);
    g_nubase = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->op == IR_CBUF && v->type == IR_I64 && !v->nsrc && !(v->imm & 7)) {
                int k = 0;
                while (k < g_nubase && g_ubase[k].off != v->imm)
                    k++;
                if (k == UR_PAIRS)
                    continue;
                if (k == g_nubase) {
                    g_ubase[k].off = v->imm;
                    g_ubase[k].used = 0;
                    g_nubase++;
                }
                g_ur[v->id] = UR_FIRST + 2 * k;
            } else if (v->op == IR_ADDR && g_ur[v->src[0]->id]) {
                g_ur[v->id] = g_ur[v->src[0]->id];
            }
    for (int changed = 1; changed;) {
        changed = 0;
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *u = b->first; u; u = u->next) {
                if (u->op == IR_ADDR && g_ur[u->id] && !g_ur[u->src[0]->id]) {
                    g_ur[u->id] = 0;
                    changed = 1;
                }
                for (unsigned i = 0; i < u->nsrc; i++) {
                    struct ir_val *s = u->src[i];
                    if (s && s->op == IR_ADDR && g_ur[s->id] && !address_use(u, i)) {
                        g_ur[s->id] = 0;
                        changed = 1;
                    }
                }
            }
    }
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            g_narrow[v->id] = v->op == IR_ADDR && g_ur[v->id] && v->src[0]->op == IR_CBUF && v->imm == 1 && !v->aux;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *u = b->first; u; u = u->next)
            for (unsigned i = 0; i < u->nsrc; i++) {
                struct ir_val *s = u->src[i];
                if (s && u->op == IR_ADDR)  // extending it needs the pair
                    g_narrow[s->id] = 0;
                if (!s || s->op != IR_CBUF || !g_ur[s->id])
                    continue;
                if (address_use(u, i))
                    g_ubase[(g_ur[s->id] - UR_FIRST) / 2].used = 1;
                else
                    g_whole[s->id] = 1;
            }
}

// LDG.CONSTANT for loads from a buffer the AIR marks read-only and unaliased, with no store or untraced write in the
// kernel. A written texture may view a buffer, so that disables it. Kernels only, as each launch empties the cache
// and draws in a pass may not; the shader's own constant data qualifies in every stage.
#define ROOT_NONE (-1)  // a null pointer
#define ROOT_ANY (-2)   // from memory, a parameter, or several buffers
static unsigned char *g_readonly_load;

static int meet_root(int a, int b)
{
    return a == ROOT_NONE ? b : b == ROOT_NONE || a == b ? a : ROOT_ANY;
}

static int buffer_arg_at(int off)
{
    for (unsigned i = 0; i < g_info->nargs; i++)
        if (g_info->args[i].kind == NVK_ARG_BUFFER && g_info->args[i].cbuf_off == off)
            return (int)i;
    return -1;
}

static void find_readonly_loads(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_readonly_load);
    g_readonly_load = calloc(n, 1);
    int data = -1, buffers = g_info->stage == NVK_COMPUTE && g_info->readonly_args;
    for (unsigned i = 0; i < g_info->nargs; i++)
        if (g_info->args[i].kind == NVK_ARG_DATA)
            data = g_info->args[i].cbuf_off;
    if (!buffers && data < 0)
        return;
    int *root = malloc(n * sizeof(*root));
    for (int i = 0; i < n; i++)
        root[i] = ROOT_NONE;
    for (int changed = 1; changed;) {
        changed = 0;
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *v = b->first; v; v = v->next) {
                if (v->type != IR_I64)
                    continue;
                int r = ROOT_ANY;
                if (v->op == IR_CBUF && !v->nsrc)
                    r = (int)v->imm;
                else if (v->op == IR_CONST || v->op == IR_UNDEF)
                    r = ROOT_NONE;
                else if (v->op == IR_ADDR)
                    r = root[v->src[0]->id];
                else if (v->op == IR_SEL)
                    r = meet_root(root[v->src[1]->id], root[v->src[2]->id]);
                else if (v->op == IR_PHI) {
                    r = ROOT_NONE;
                    for (unsigned i = 0; i < v->nsrc; i++)
                        r = meet_root(r, root[v->src[i]->id]);
                }
                if (r != root[v->id]) {
                    root[v->id] = r;
                    changed = 1;
                }
            }
    }
    uint64_t written = 0;
    int anywhere = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next) {
            if (v->op == IR_SUST)
                anywhere = 1;
            if (v->op != IR_STG && (v->op != IR_ATOM || IR_ATOMA_SHARED(v->aux)))
                continue;
            int a = root[v->src[0]->id] >= 0 ? buffer_arg_at(root[v->src[0]->id]) : -1;
            if (a < 0)
                anywhere = 1;
            else
                written |= 1ull << a;
        }
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next) {
            int r = v->op == IR_LDG ? root[v->src[0]->id] : ROOT_ANY;
            int a = r >= 0 && buffers && !anywhere ? buffer_arg_at(r) : -1;
            g_readonly_load[v->id] = (r >= 0 && r == data) || (a >= 0 && (g_info->readonly_args & ~written) >> a & 1);
        }
    free(root);
}

// base + scale * (index + k) + off
struct access {
    const struct ir_val *base, *index;
    uint32_t k;
    int64_t scale, off;
    int ext;
};

static struct access access_of(const struct ir_val *v)
{
    struct access a = { .off = v->imm };
    const struct ir_val *p = v->src[0];
    for (; p->op == IR_ADDI; p = p->src[0])
        a.off += p->imm;
    if (p->op == IR_ADDR) {
        a.scale = p->imm;
        a.ext = p->aux;
        a.index = p->src[1];
        while ((a.index->op == IR_ADD || a.index->op == IR_SUB) && a.index->src[1]->op == IR_CONST) {
            uint32_t c = (uint32_t)a.index->src[1]->imm;
            a.k += a.index->op == IR_ADD ? c : -c;
            a.index = a.index->src[0];
        }
        p = p->src[0];
    }
    a.base = p;
    return a;
}

// same base and index a few elements apart; an index wrapping between them puts them 2^32 apart, still disjoint
static int apart(const struct ir_val *x, const struct ir_val *y)
{
    struct access a = access_of(x), b = access_of(y);
    int32_t dk = (int32_t)(b.k - a.k);
    if (a.base != b.base || a.index != b.index || a.scale != b.scale || a.ext != b.ext || dk < -65536 || dk > 65536)
        return 0;
    int64_t d = dk * a.scale + b.off - a.off;
    return d >= x->aux || -d >= y->aux;
}

// Some kernels accumulate a sum in their output buffer, a store per term; all but the last go if nothing between
// can read or publish the word.
static void drop_overwritten_stores(struct ir_func *f)
{
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first, *next; v; v = next) {
            next = v->next;
            if (v->op != IR_STG)
                continue;
            for (struct ir_val *w = v->next; w; w = w->next) {
                if (w->op == IR_STG && w->src[0] == v->src[0] && w->imm == v->imm && w->aux == v->aux &&
                    w->nsrc == v->nsrc) {
                    ir_remove(v);
                    break;
                }
                if ((w->op == IR_LDG && !g_readonly_load[w->id] && !apart(w, v)) || w->op == IR_TEX ||
                    w->op == IR_ATOM || w->op == IR_BAR || w->op == IR_CALL || w->op == IR_KILL)
                    break;
            }
        }
}

// off a uniform base this is the offset, with the base in m->ur
static int vreg_of_address(struct ir_val *v, unsigned i, struct mop *m)
{
    struct ir_val *a = v->src[i];
    if (i || (v->op != IR_LDG && v->op != IR_STG) || !g_ur[a->id])
        return vreg_of(a);
    m->ur = g_ur[a->id];
    m->narrow = g_narrow[a->id];
    return a->op == IR_CBUF ? VREG_RZ : vreg_of(a);
}

// an immediate must be on the right, so the compare may be swapped
static struct mop *emit_setp(const struct ir_val *c)
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
    return s;
}

static struct mop *emit_test(struct ir_val *c)
{
    if (c->op == IR_CMP && g_cmp_fused[c->id])
        return emit_setp(c);
    int r = vreg_of(c);  // before the ISETP, as this may emit a constant
    struct mop *s = emit_mop(MK_ISETP);
    s->src[0] = r;
    s->src[1] = VREG_RZ;
    s->nsrc = 2;
    s->aux = IR_NE;
    return s;
}

static void emit_cond(struct ir_val *c)
{
    if (is_pand_or(c) && g_cmp_fused[c->id]) {
        int chain = is_pand_or(c->src[1]) && g_cmp_fused[c->src[1]->id];
        emit_cond(c->src[chain]);
        emit_test(c->src[!chain])->aux |= c->op == IR_PAND ? SETP_AND : SETP_OR;
    } else {
        emit_test(c);
    }
}

// lets a vector's lanes or a flattened if's selects reuse P0 while nothing has set it since g_p0_at
static struct ir_val *g_p0_cond;
static int g_p0_at;

static void emit_sel_cond(struct ir_val *c)
{
    int holds = c == g_p0_cond;
    for (int k = g_p0_at; k < g_nmops && holds; k++)
        holds = g_mops[k].k != MK_ISETP && g_mops[k].k != MK_FSETP && g_mops[k].k != MK_CALL;
    if (!holds)
        emit_cond(c);
    g_p0_cond = c;
    g_p0_at = g_nmops;
}

static void lower_value(struct ir_val *v)
{
    if (v->vreg >= 0)
        return;  // a prologue value
    int half = v->type == IR_F16;
    switch (v->op) {
    case IR_CONST:
    case IR_UNDEF:
        break;  // made where used (vreg_of)
    case IR_CBUF:
        if (v->type == IR_I64 && g_ur[v->id] && !g_whole[v->id])
            break;  // only a uniform base
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
        if ((v->op == IR_MUL && g_mul_fused[v->id]) || (v->op == IR_ADD && g_add_fused[v->id]) ||
            g_late_def[v->id] || g_folded[v->id])
            break;  // made by its reader or at its loop's end
        if (v->op == IR_ADD && (lower_mul_add(v) || lower_add3(v)))
            break;
        struct ir_val *neg = abs_negation(v);
        if (neg) {
            v->vreg = new_vreg(1, 1);
            struct mop *m = emit_mop(MK_IABS);
            m->def = v->vreg;
            m->src[0] = vreg_of(neg->src[1]);
            m->nsrc = 1;
            break;
        }
        if (is_logic(v) && lower_lop3(v))
            break;
        if (v->op == IR_OR && v->src[1]->op == IR_CONST && !v->src[1]->imm && v->src[0]->op != IR_CONST &&
            v->type != IR_I64 && v->src[0]->type != IR_I64 && v->src[0]->type != IR_PRED) {
            v->vreg = vreg_of(v->src[0]);
            g_alias[v->id] = g_alias[v->src[0]->id] = 1;
            break;
        }
        struct ir_val *in;
        int sel = byte_select(v, &in);
        if (sel >= 0 && g_folded[in->id]) {
            v->vreg = new_vreg(1, 1);
            struct mop *m = emit_mop(MK_PRMT);
            m->def = v->vreg;
            m->src[0] = vreg_of(in->src[0]);
            m->nsrc = 1;
            m->imm = sel;
            break;
        }
        if (lower_byte_convert(v))
            break;
        if ((in = sat_source(v)) && g_sat[in->id]) {
            v->vreg = in->vreg;
            break;
        }
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(binop_kind(v));
        m->def = v->vreg;
        struct ir_val *fa = v->src[0], *fb = v->src[1];
        int imm = !half && (v->op == IR_FADD || v->op == IR_FSUB || v->op == IR_FMUL);
        if (imm && fa->op == IR_CONST && fb->op != IR_CONST && v->op != IR_FSUB) {
            fa = v->src[1];
            fb = v->src[0];
        }
        // c - x as -x + c
        int neg_a = 0;
        if (imm && v->op == IR_FSUB && fa->op == IR_CONST && fb->op != IR_CONST && !airc_pass_off("AIRC_NONEGA")) {
            fa = v->src[1];
            fb = v->src[0];
            neg_a = 1;
        }
        imm = imm && fb->op == IR_CONST && fa->op != IR_CONST;
        int immb = 0;
        if (!imm && !(v->op == IR_FADD || v->op == IR_FSUB || v->op == IR_FMUL) &&
            (m->k == MK_LOP3 || m->k == MK_IADD3 || m->k == MK_IMAD || m->k == MK_IMNMX || m->k == MK_FMNMX ||
             m->k == MK_ISUB || m->k == MK_SHL || m->k == MK_SHR) &&
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
        int rz = v->type == IR_I32 && (m->k == MK_IADD3 || m->k == MK_ISUB || m->k == MK_LOP3 || m->k == MK_IMNMX);
        m->src[0] = rz && fa->op == IR_CONST && !fa->imm ? VREG_RZ : vreg_of(fa);
        m->nsrc = 1;
        if (immb) {
            m->immb = 1;
            m->imm = (uint32_t)fb->imm;
            if (m->k == MK_ISUB) {
                m->k = MK_IADD3;
                m->imm = (uint32_t)-(int32_t)fb->imm;
            }
        } else if (imm) {
            m->imm = v->op == IR_FSUB && !neg_a ? (uint32_t)fb->imm ^ 0x80000000u : (uint32_t)fb->imm;
        } else {
            m->src[m->nsrc++] = rz && fb->op == IR_CONST && !fb->imm ? VREG_RZ : vreg_of(fb);
        }
        if (m->k == MK_LOP3)
            m->aux = v->op == IR_AND ? LOP3_AND : v->op == IR_OR ? LOP3_OR : LOP3_XOR;
        else if (m->k == MK_SHR)
            m->aux = v->op == IR_SAR;
        else if (m->k == MK_IMNMX)
            m->aux = (v->op == IR_MAX) | (v->aux ? 2 : 0);
        else if (m->k == MK_FADD) {
            m->aux = (v->op == IR_FSUB && !imm) | (half ? 2 : 0) | (imm ? 16 : 0) | (neg_a ? 128 : 0);
            // the constant may have become a register, and .SAT works on either form
            if (v->op == IR_FADD && (v->aux & IR_FADD_SAT))
                m->aux |= 64;
        } else if (m->k == MK_FMUL)
            m->aux = (half ? 2 : 0) | (imm ? 16 : 0) | (g_sat[v->id] ? 64 : 0);
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
        int flag = 0;
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
                       x[2]->op == IR_CONST) {
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
        m->aux = (half ? 2 : 0) | v->aux << 2 | flag | (g_sat[v->id] ? 64 : 0);
        break;
    }
    case IR_FNEG:
    case IR_FABS: {
        if (v->type == IR_F32 && !g_mod_needed[v->id])
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
        if ((v->op == IR_F2H && g_first[v->id]) || g_folded[v->id] || lower_byte_convert(v))
            break;
        if (v->op == IR_F2H && g_pair[v->id]) {
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
    case IR_SHFL: {  // clamped at the first lane shifting up, the last otherwise
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
    case IR_DERIV: {  // the neighbour at lane ^ 1 or ^ 2, then a swizzled subtract
        int n = new_vreg(1, 1);
        struct mop *sh = emit_mop(MK_SHFL);
        sh->def = n;
        sh->src[0] = vreg_of(v->src[0]);
        sh->nsrc = 1;
        sh->imm = v->aux ? 2 : 1;
        sh->aux = SASS_SHFL_BFLY | 0x1c03 << 4;  // within a quad
        v->vreg = new_vreg(1, 1);
        struct mop *m = emit_mop(MK_FSWZADD);
        m->def = v->vreg;
        m->src[0] = n;
        m->src[1] = vreg_of(v->src[0]);
        m->nsrc = 2;
        m->aux = v->aux ? FSWZADD_DDY : FSWZADD_DDX;
        break;
    }
    case IR_VOTE: {
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
        emit_sel_cond(v->src[0]);
        if (!wide && (is_imm(v->src[1]) || is_imm(v->src[2]))) {
            int swap = !is_imm(v->src[2]);  // the true value is the constant, so select on !P0
            struct mop *sel = emit_mop(MK_SEL);
            sel->def = v->vreg;
            sel->src[0] = vreg_of(v->src[swap ? 2 : 1]);
            sel->nsrc = 1;
            sel->immb = 1;
            sel->imm = (uint32_t)v->src[swap ? 1 : 2]->imm;
            sel->aux = swap;
            break;
        }
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
        if (g_cmp_fused[v->id])
            break;
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
        int m[2], from_base = g_ur[v->id] && v->src[0]->op == IR_CBUF;
        if (g_narrow[v->id]) {
            v->vreg = vreg_of(v->src[1]);
            break;
        }
        if (from_base) {
            if (g_noffsets && g_offsets[0].block != g_block)
                g_noffsets = 0;
            for (int k = 0; k < g_noffsets && v->vreg < 0; k++)
                if (g_offsets[k].index == v->src[1] && g_offsets[k].scale == v->imm && g_offsets[k].aux == v->aux)
                    v->vreg = g_offsets[k].vreg;
            if (v->vreg >= 0)
                break;
        }
        v->vreg = new_cluster(2, 2, m);
        if (from_base && g_noffsets < (int)(sizeof g_offsets / sizeof g_offsets[0]))
            g_offsets[g_noffsets++] = (struct offset){ g_block, v->src[1], v->imm, v->aux, v->vreg };
        struct mop *w = emit_mop(MK_IMADWIDE);
        w->def = v->vreg;
        w->src[0] = vreg_of(v->src[1]);
        w->src[1] = VREG_RZ;
        w->src[2] = from_base ? VREG_RZ : vreg_of(v->src[0]);
        w->nsrc = 3;
        w->immb = 1;
        w->imm = (uint32_t)v->imm;
        w->aux = v->aux;
        break;
    }
    case IR_ATOM: {
        struct mop *m = emit_mop(MK_ATOM);
        if (!g_reduce[v->id]) {
            v->vreg = new_vreg(1, 1);
            m->def = v->vreg;
        }
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
        v->vreg = v->aux > 4 ? new_cluster(v->aux / 4, v->aux / 4, q) : new_vreg(1, 1);
        int guarded = v->op == IR_LDG && v->nsrc > 1;
        if (guarded)
            emit_sel_cond(v->src[1]);
        struct mop *m = emit_mop(v->op == IR_LDG ? MK_LDG : v->op == IR_LDS ? MK_LDS : MK_LDL);
        m->def = v->vreg;
        m->nsrc = guarded ? 1 : v->nsrc;
        m->guarded = guarded;
        for (int i = 0; i < m->nsrc; i++)
            m->src[i] = vreg_of_address(v, i, m);
        m->imm = v->imm;
        m->aux = v->aux;
        m->readonly = v->op == IR_LDG && g_readonly_load[v->id];
        break;
    }
    case IR_TEX: {
        // the handle leads TXD's first vector and the second vector of the rest
        int txd = IR_TEXA_OP(v->aux) == SASS_TXD, n0 = IR_TEXA_N0(v->aux), nsrc = (int)v->nsrc - 1;
        int k[2] = { n0 + txd, nsrc - n0 + !txd }, vec[2], c[2][4], q[4];
        if (k[0] > 4 || k[1] > 4)
            airc_fail("a texture op with too many operands");
        struct ir_val *val[2][4];
        val[!txd][0] = v->src[nsrc];
        for (int i = 0; i < nsrc; i++) {
            int j = i >= n0;
            val[j][i - (j ? n0 : 0) + (j != txd)] = v->src[i];
        }
        if (g_ntexvecs && g_texvecs[0].block != g_block)
            g_ntexvecs = 0;
        for (int j = 0; j < 2; j++) {
            vec[j] = -1;
            for (int t = 0; t < g_ntexvecs && vec[j] < 0; t++) {
                int same = !airc_no_texvec && g_texvecs[t].n == k[j];
                for (int i = 0; i < k[j] && same; i++)
                    same = same_value(g_texvecs[t].val[i], val[j][i]);
                if (same)
                    vec[j] = g_texvecs[t].vreg;
            }
            airc_texvec_shared |= vec[j] >= 0;
            if (vec[j] >= 0)
                continue;
            vec[j] = new_cluster(k[j], k[j] > 2 ? 4 : k[j], c[j]);
            for (int i = 0; i < k[j]; i++)
                mov_into(c[j][i], val[j][i]);
            struct texvec *t = &g_texvecs[g_ntexvecs < 8 ? g_ntexvecs++ : 7];
            *t = (struct texvec){ g_block, k[j], { 0 }, vec[j] };
            memcpy(t->val, val[j], k[j] * sizeof(*val[j]));
        }
        v->vreg = new_cluster(4, 2, q);  // two pairs, R,G and B,A
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
        if (v->aux > 4) {
            int c[4];
            data = new_cluster(v->aux / 4, v->aux / 4, c);
            for (int k = 0; k < v->aux / 4; k++)
                mov_into(c[k], v->src[1 + k]);
        } else {
            data = vreg_of(v->src[1]);
        }
        struct mop *m = emit_mop(v->op == IR_STG ? MK_STG : v->op == IR_STS ? MK_STS : MK_STL);
        m->src[0] = v->src[0] ? vreg_of_address(v, 0, m) : VREG_RZ;
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
        // Metal's discard demotes to a helper so the quad's derivatives still see it; memory writes stop
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
        int coord = new_cluster(n, n > 2 ? 4 : n, c);
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

// In a one-block loop, the value v carried round in a phi is made straight in the phi's register when nothing
// reads the phi after v, saving the copy. Reads elsewhere or deferred (fused compares, modifiers, paired halves)
// rule it out up front (phi_outside). AIRC_NOPHIDEF=1 turns it off.
// g_phi_escapes marks phis read after the loop, whose copies can't run on the way out.
static unsigned char *g_phi_outside, *g_phi_moved, *g_phi_escapes;

static void phi_outside(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    free(g_phi_outside);
    free(g_phi_moved);
    free(g_phi_escapes);
    g_phi_outside = calloc(n, 1);
    g_phi_moved = calloc(n, 1);
    g_phi_escapes = calloc(n, 1);
    unsigned char *late = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *u = b->first; u; u = u->next)
            for (unsigned i = 0; i < u->nsrc; i++) {
                struct ir_val *s = u->src[i];
                if (s && (u->block != s->block || u->op == IR_PHI || u->op == IR_FNEG || u->op == IR_FABS))
                    late[s->id] = 1;
            }
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *u = b->first; u; u = u->next)
            for (unsigned i = 0; i < u->nsrc; i++) {
                struct ir_val *s = u->src[i];
                if (!s || s->op != IR_PHI)
                    continue;
                int deferred = u->op == IR_FNEG || u->op == IR_FABS || u->op == IR_CMP || u->op == IR_F2H;
                if (u->block != s->block || u->op == IR_PHI || deferred)
                    g_phi_outside[s->id] = 1;
                if (u->block != s->block || u->op == IR_PHI || (deferred && late[u->id]))
                    g_phi_escapes[s->id] = 1;
            }
    free(late);
    free(g_late_def);
    g_late_def = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next) {
        if (!b->last || b->last->op != IR_CBR || b->succ[0] == b->succ[1] || (b->succ[0] != b && b->succ[1] != b))
            continue;
        for (struct ir_val *p = b->first; p && p->op == IR_PHI; p = p->next)
            for (unsigned k = 0; k < p->nsrc; k++) {
                struct ir_val *v = p->src[k];
                if (p->phi_blocks[k] != b || v->block != b || v->type != IR_I32 || p->type != IR_I32 ||
                    g_use_count[v->id] != 1 || v->op < IR_ADD || v->op > IR_SAR || v->op == IR_MUL || v->op == IR_MULHI)
                    continue;
                for (int i = 0; i < 2; i++) {
                    struct ir_val *x = v->src[!i];
                    if (v->src[i] == p && x != p && (x->op == IR_CONST || x->block != b))
                        g_late_def[v->id] = 1;
                }
            }
    }
}

// made last in the loop, after p's final reads, so it can write p's register without a copy
static void lower_late_def(struct ir_val *p, struct ir_val *v)
{
    for (unsigned i = 0; i < v->nsrc; i++)
        if (v->src[i]->op == IR_CONST)
            vreg_of(v->src[i]);
    g_late_def[v->id] = 0;
    lower_value(v);
    for (int k = g_nmops - 1; k >= 0; k--)
        if (g_mops[k].def == v->vreg) {
            g_mops[k].def = p->vreg;
            break;
        }
    v->vreg = p->vreg;
}

static void phi_into_def(struct ir_block *b, struct ir_val *p, struct ir_val *v)
{
    if (airc_pass_off("AIRC_NOPHIDEF") || p->type == IR_I64 || v->type != p->type || g_alias[v->id] || g_alias[p->id] ||
        ((v->op < IR_ADD || v->op > IR_FRND) && v->op != IR_SEL) ||
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
    for (int k = def + 1; k < g_nmops; k++) {  // nothing may read the old value after v
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
    int dst[MAX_COPIES], src[MAX_COPIES], tmp[MAX_COPIES], n = 0;  // src -1 for a constant in imm[]
    int64_t imm[MAX_COPIES];
    for (struct ir_val *p = b->first; p && p->op == IR_PHI; p = p->next) {
        struct ir_val *in = NULL;
        for (unsigned k = 0; k < p->nsrc; k++)
            if (p->phi_blocks[k] == pred)
                in = p->src[k];
        if (!in)
            airc_fail("phi edge without a value");
        if (pred == b && g_late_def[in->id])
            continue;
        if (n + 2 > MAX_COPIES)
            airc_fail("too many phis in a block");
        int is_const = in->op == IR_CONST || in->op == IR_UNDEF;
        int64_t value = in->op == IR_CONST ? in->imm : 0;
        int s = is_const ? -1 : vreg_of(in);
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
    // only sources another copy overwrites go via a temporary; one per phi would run out with a hundred phis
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
            m->aux = 1;
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
            m->aux = 1;
        }
        m->def = dst[i];
    }
    // after the copies, which read the old values
    for (struct ir_val *p = b->first; pred == b && p && p->op == IR_PHI; p = p->next)
        for (unsigned k = 0; k < p->nsrc; k++)
            if (p->phi_blocks[k] == b && g_late_def[p->src[k]->id])
                lower_late_def(p, p->src[k]);
}

static void set_convergence(struct ir_block *b)
{
    if (g_bssy[b->id] < 0)
        return;
    struct mop *m = emit_mop(MK_BSSY);
    m->imm = g_bar_base[b->func] + g_bssy[b->id];
    m->target = g_bssy_to[b->id];
}

static int g_ra;
static int g_win_end;  // past the window's last slot

static void window_slots(int end)
{
    if (WIN_ARGS + end > g_win_end)
        g_win_end = WIN_ARGS + end;
}

// room for extra entries after the sources
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

// an IR_PROJ's imm is its result's window slot
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
    g_p0_cond = NULL;
    if (b->func && g_fn->subs[b->func - 1] == b)
        lower_entry(b);
    for (int i = 0; i < g_nbsync[b->id]; i++)
        emit_mop(MK_BSYNC)->imm = g_bar_base[b->func] + g_bsync[b->id][i];
    for (struct ir_val *v = b->first; v; v = v->next) {
        // constants first, as lowering may emit the instruction before asking for them
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
            emit_cond(v->src[0]);
            // a one-block loop whose phis don't escape copies before the branch, so the back edge needs no stub
            int self = b->succ[0] == b->succ[1] ? -1 : b->succ[0] == b ? 0 : b->succ[1] == b ? 1 : -1;
            for (struct ir_val *p = b->first; self >= 0 && p && p->op == IR_PHI; p = p->next)
                if (g_phi_escapes[p->id])
                    self = -1;
            if (self >= 0)
                phi_copies(b, b);
            // after the compare so it starts early, as the branch waits 13 cycles for P0
            set_convergence(b);
            // the true edge's copies go in a stub after the false path
            int stub = self != 0 && b->succ[0]->first && b->succ[0]->first->op == IR_PHI;
            int bt = g_nmops;
            struct mop *m = emit_mop(MK_BRA);
            m->aux = 1;
            if (!stub)
                m->target = b->succ[0]->id;
            if (self != 1)
                phi_copies(b, b->succ[1]);
            emit_mop(MK_BRA)->target = b->succ[1]->id;
            if (stub) {
                g_mops[bt].label = g_nmops;
                phi_copies(b, b->succ[0]);
                emit_mop(MK_BRA)->target = b->succ[0]->id;
            }
            break;
        }
        case IR_EXIT: {  // colours to scratch registers, then EXIT moves them to R0..
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

// [start, end), the shader first
static int g_region_start[IR_MAX_SUBS + 1], g_region_end[IR_MAX_SUBS + 1];

// a mop naming a cluster base touches every member
#define MEMBERS(w, v) for (int w = (v); w >= 0; w = g_vregs[v].size > 1 ? g_vregs[w].next : -1)

static void touch(int v, int k)
{
    struct vreg *r = &g_vregs[v];
    if (r->hw >= 0)  // allocated with a caller
        return;
    if (r->start < 0 || k < r->start)
        r->start = k;
    if (k > r->end)
        r->end = k;
}

// live runs, so a value used in one arm of an if leaves its register free in the other, and cluster members are
// live only where read
struct seg {
    int start, end, next;  // inclusive
};
static struct seg *g_segs;
static int g_nsegs, g_cap_segs;

static int add_seg(int v, int tail, int start, int end)
{
    if (tail >= 0 && start <= g_segs[tail].end + 1) {
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

// Dataflow, so values read round a back edge live to the loop's end. A conditional branch's block splits into the
// branch, the false edge's copies and the true edge's stub, so one edge's copies may reuse a value only the other
// reads. Shader values a subroutine reads keep their shader ranges.
static void compute_ranges(struct ir_func *f, int region)
{
    int nb = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        if (!b->dead && b->func == region)
            nb++;
    struct ir_block **blocks = malloc(nb * sizeof(*blocks));
    int *bstart = malloc(3 * nb * sizeof(*bstart)), *bend = malloc(3 * nb * sizeof(*bend));
    int(*succ)[2] = malloc(3 * nb * sizeof(*succ)), *nsucc = malloc(3 * nb * sizeof(*nsucc));
    int *index = malloc(f->nblocks * sizeof(*index)), *cbr = malloc(nb * sizeof(*cbr));
    nb = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        if (!b->dead && b->func == region)
            blocks[nb++] = b;
    int np = 0;
    for (int i = 0; i < nb; i++) {
        int k = g_block_mop[blocks[i]->id], e = i + 1 < nb ? g_block_mop[blocks[i + 1]->id] : g_region_end[region];
        while (k < e && !(g_mops[k].k == MK_BRA && g_mops[k].aux))
            k++;
        cbr[i] = k < e ? k : -1;
        index[blocks[i]->id] = np;
        np += k == e ? 1 : g_mops[k].label < 0 ? 2 : 3;
    }
    np = 0;
    for (int i = 0; i < nb; i++) {
        struct ir_block *b = blocks[i];
        int s = g_block_mop[b->id], e = i + 1 < nb ? g_block_mop[blocks[i + 1]->id] : g_region_end[region];
        int k = cbr[i];
        if (k < 0) {
            bstart[np] = s;
            bend[np] = e;
            nsucc[np] = (int)b->nsucc;
            for (unsigned j = 0; j < b->nsucc; j++)
                succ[np][j] = index[b->succ[j]->id];
            np++;
            continue;
        }
        int stub = g_mops[k].label;
        bstart[np] = s;
        bend[np] = k + 1;
        nsucc[np] = 2;
        succ[np][0] = np + 1;
        succ[np][1] = stub < 0 ? index[b->succ[0]->id] : np + 2;
        np++;
        bstart[np] = k + 1;
        bend[np] = stub < 0 ? e : stub;
        nsucc[np] = 1;
        succ[np][0] = index[b->succ[1]->id];
        np++;
        if (stub >= 0) {
            bstart[np] = stub;
            bend[np] = e;
            nsucc[np] = 1;
            succ[np][0] = index[b->succ[0]->id];
            np++;
        }
    }
    // the MOVIMM of the integer 1 sits before the first block
    if (np)
        bstart[0] = g_region_start[region];

    int words = (g_nvregs + 63) / 64;
    uint64_t *use = calloc((size_t)np * words, 8), *def = calloc((size_t)np * words, 8);
    uint64_t *in = calloc((size_t)np * words, 8), *out = calloc((size_t)np * words, 8);
#define BIT(set, i, v) ((set)[(size_t)(i) * words + (v) / 64] >> ((v) % 64) & 1)
#define SETBIT(set, i, v) ((set)[(size_t)(i) * words + (v) / 64] |= 1ull << ((v) % 64))
    for (int i = 0; i < np; i++)
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
    for (int changed = 1; changed;) {
        changed = 0;
        for (int i = np - 1; i >= 0; i--) {
            uint64_t *o = &out[(size_t)i * words], *n = &in[(size_t)i * words];
            for (int s = 0; s < nsucc[i]; s++) {
                uint64_t *si = &in[(size_t)succ[i][s] * words];
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
    for (int i = 0; i < np; i++)
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
    g_nsegs = 0;
    int *tail = malloc((unsigned)g_nvregs * sizeof(*tail)), *stamp = calloc((unsigned)g_nvregs, sizeof(*stamp));
    int *first = malloc((unsigned)g_nvregs * sizeof(*first)), *last = malloc((unsigned)g_nvregs * sizeof(*last));
    int *named = malloc((unsigned)g_nvregs * sizeof(*named));
    for (int v = 0; v < g_nvregs; v++) {
        tail[v] = -1;
        if (g_vregs[v].hw < 0)
            g_vregs[v].seg = -1;
    }
    for (int i = 0; i < np; i++) {
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
        for (int w = 0; w < words; w++)  // live through but unnamed
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
    free(succ);
    free(nsucc);
    free(index);
    free(cbr);
    free(use);
    free(def);
    free(in);
    free(out);
}
// Out of registers, the shader recompiles with values live at the failure spilled to local memory. They're named
// by IR id, which is stable across compiles of one shader.

#define SPILL_ROUND 64   // values added each time
#define SPILL_MAX 2048
static int g_spill_ids[SPILL_MAX];
static unsigned g_nspill_ids;
static int g_cand[SPILL_ROUND];
static unsigned g_ncand;
static int g_spill_limit;  // ids from here on came from spilling

void airc_spill_reset(void)
{
    g_nspill_ids = 0;
    g_ncand = 0;
    g_spill_limit = 0;
}

// longest ranges first
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
        if (k < g_nspill_ids)
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

// 0 when there are no new candidates
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

// 1 global, 2 shared, 3 local, 0 none
static int load_space(const struct ir_val *v)
{
    switch (v->op) {
    case IR_LDG:
    case IR_TEX:
        return 1;
    case IR_LDS:
        return 2;
    case IR_LDL:
        return 3;
    default:
        return 0;
    }
}

static int store_space(const struct ir_val *v)
{
    switch (v->op) {
    case IR_STG:
    case IR_SUST:
        return 1;
    case IR_ATOM:
        return IR_ATOMA_SHARED(v->aux) ? 2 : 1;
    case IR_STS:
        return 2;
    case IR_STL:
        return 3;
    default:
        return 0;
    }
}

static int pure(const struct ir_val *v)
{
    return v->op == IR_CONST || v->op == IR_CBUF || v->op == IR_PROJ || (v->op >= IR_ADD && v->op <= IR_FRND) ||
           (v->op >= IR_CMP && v->op <= IR_ADDI);
}

// Unrolled copies would reuse each other's registers and stop the scheduler starting the next copy's loads early,
// so the loads and their address maths go first, never past a write that may alias.
static void loads_first(struct ir_func *f)
{
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    unsigned char *ok = calloc(n, 1), *need = calloc(n, 1);
    const struct ir_val **writes = malloc(n * sizeof(*writes));
    for (struct ir_block *b = f->entry; b; b = b->next) {
        if (b->unrolled != 2)
            continue;
        int fenced = 0;
        unsigned nw = 0;
        for (struct ir_val *v = b->first; v; v = v->next) {
            if (v->op == IR_PHI)
                continue;
            int sp = load_space(v), in = pure(v) || (sp && !fenced);
            if (sp && !(v->op == IR_LDG && g_readonly_load[v->id]))
                for (unsigned i = 0; i < nw && in; i++)
                    in = store_space(writes[i]) != sp || (v->op == IR_LDG && writes[i]->op == IR_STG &&
                                                          apart(v, writes[i]));
            for (unsigned i = 0; i < v->nsrc && in; i++)
                if (v->src[i] && v->src[i]->block == b && v->src[i]->op != IR_PHI)
                    in = ok[v->src[i]->id];
            ok[v->id] = in;
            if (store_space(v))
                writes[nw++] = v;
            fenced |= v->op == IR_BAR || v->op == IR_CALL || v->op == IR_KILL;
        }
        for (struct ir_val *v = b->last; v; v = v->prev) {
            if (!ok[v->id] || (!load_space(v) && !need[v->id]))
                continue;
            need[v->id] = 1;
            for (unsigned i = 0; i < v->nsrc; i++)
                if (v->src[i] && v->src[i]->block == b)
                    need[v->src[i]->id] = 1;
        }
        struct ir_val *at = b->first;
        while (at->op == IR_PHI || need[at->id])
            at = at->next;
        for (struct ir_val *v = at, *next; v; v = next) {
            next = v->next;
            if (need[v->id])
                link_before(v, at);
        }
    }
    free(writes);
    free(need);
    free(ok);
}

// Fetches left at their use reuse the previous one's registers and wait one at a time, so each moves up behind the
// one before, up to airc_batch in flight, without raising the block's register peak or passing an aliasing write.
int airc_batch = 4, airc_batched;

static int batchable(const struct ir_val *v)
{
    return v->op == IR_TEX || v->op == IR_LDG;
}

static int movable(const struct ir_val *v)
{
    return v->op != IR_PHI && (pure(v) || v->op == IR_IPA || (v->op == IR_ALD && !(v->aux & IR_ATTR_OUTPUT)));
}

static struct ir_val *load_read(struct ir_val *s)
{
    s = s && s->op == IR_PROJ ? s->src[0] : s;
    return s && batchable(s) ? s : NULL;
}

// reads: fetches a value is first to read; pend: fetches in flight before it. Fetches read only in other blocks
// hold registers regardless and aren't counted.
static void count_pending(struct ir_block *b, int *reads, int *pend, char *seen)
{
    for (struct ir_val *v = b->first; v; v = v->next)
        seen[v->id] = 0;
    for (struct ir_val *v = b->first; v; v = v->next) {
        reads[v->id] = 0;
        if (v->op != IR_PROJ)
            for (unsigned i = 0; i < v->nsrc; i++) {
                struct ir_val *l = load_read(v->src[i]);
                if (l && l->block == b && !seen[l->id]) {
                    seen[l->id] = 1;
                    reads[v->id]++;
                }
            }
    }
    int n = 0;
    for (struct ir_val *v = b->first; v; v = v->next) {
        pend[v->id] = n;
        n += (batchable(v) && seen[v->id]) - reads[v->id];
    }
}

// roughly; wide loads and texture ops count through their IR_PROJs
static int regs_of(const struct ir_val *v)
{
    if (v->type == IR_VOID || v->op == IR_CONST || v->op == IR_UNDEF || (v->op == IR_CBUF && !v->nsrc))
        return 0;
    return v->type == IR_I64 ? 2 : 1;
}

// pr gets the live registers before each value, out marks values live to the end; returns the peak
static int block_pressure(struct ir_block *b, const char *out, int *pr, int *pos, int *last, int *stamp, int *diff,
                          struct ir_val **before, int mark)
{
    int n = 0, nbefore = 0;
    for (struct ir_val *v = b->first; v; v = v->next) {
        pos[v->id] = n++;
        stamp[v->id] = mark;
        last[v->id] = -1;
    }
    for (struct ir_val *v = b->first; v; v = v->next)
        for (unsigned i = 0; i < v->nsrc; i++) {
            struct ir_val *s = v->src[i];
            if (!s)
                continue;
            if (stamp[s->id] != mark) {
                stamp[s->id] = mark;
                before[nbefore++] = s;
            }
            last[s->id] = pos[v->id];
        }
    memset(diff, 0, (n + 2) * sizeof(*diff));
    for (struct ir_val *v = b->first; v; v = v->next) {
        int end = out[v->id] ? n : last[v->id];
        if (end > pos[v->id]) {
            diff[pos[v->id] + 1] += regs_of(v);
            diff[end + 1] -= regs_of(v);
        }
    }
    for (int i = 0; i < nbefore; i++) {
        diff[0] += regs_of(before[i]);
        diff[last[before[i]->id] + 1] -= regs_of(before[i]);
    }
    int live = 0, most = 0;
    struct ir_val *v = b->first;
    for (int k = 0; k < n; k++, v = v->next) {
        live += diff[k];
        pr[v->id] = live;
        most = live > most ? live : most;
    }
    return most;
}

static void batch_loads(struct ir_func *f)
{
    enum { MAX_MOVED = 32 };
    airc_batched = 0;
    if (!airc_batch || airc_pass_off("AIRC_NOBATCH"))
        return;
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            if (v->id >= n)
                n = v->id + 1;
    int *reads = malloc(n * sizeof(*reads)), *pend = malloc(n * sizeof(*pend)), *pr = malloc(n * sizeof(*pr));
    int *pos = malloc(n * sizeof(*pos)), *last = malloc(n * sizeof(*last)), *stamp = calloc(n, sizeof(*stamp));
    int *diff = malloc((n + 2) * sizeof(*diff)), marks = 0;
    struct ir_val **before = malloc(n * sizeof(*before));
    char *seen = malloc(n), *need = calloc(n, 1), *out = calloc(n, 1);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            for (unsigned i = 0; i < v->nsrc; i++)
                if (v->src[i] && (v->src[i]->block != b || v->op == IR_PHI))
                    out[v->src[i]->id] = 1;
    for (struct ir_block *b = f->entry; b; b = b->next) {
        if (b->dead || b->unrolled == 2)
            continue;
        count_pending(b, reads, pend, seen);
        int budget = block_pressure(b, out, pr, pos, last, stamp, diff, before, ++marks);
        struct ir_val *prev = NULL;
        for (struct ir_val *v = b->first, *next; v; v = next) {
            next = v->next;
            if (!batchable(v))
                continue;
            int extra = regs_of(v);
            while (next && next->op == IR_PROJ && next->src[0] == v) {
                extra += regs_of(next);
                next = next->next;
            }
            struct ir_val *moved[MAX_MOVED];
            int nmoved = 0, waiting = pend[v->id];
            for (unsigned i = 0; i < v->nsrc; i++)
                if (v->src[i])
                    need[v->src[i]->id] = 1;
            struct ir_val *stop = v->prev;
            for (; stop && stop != prev; stop = stop->prev) {
                waiting += reads[stop->id];
                if (waiting >= airc_batch || stop->op == IR_PHI || stop->op == IR_PARAM || stop->op == IR_BAR ||
                    stop->op == IR_CALL || stop->op == IR_KILL)
                    break;
                int sp = store_space(stop);
                if (sp && sp == load_space(v) && !(v->op == IR_LDG && g_readonly_load[v->id]) &&
                    !(v->op == IR_LDG && stop->op == IR_STG && apart(v, stop)))
                    break;
                if (!need[stop->id]) {
                    if (pr[stop->id] + extra > budget)
                        break;
                    continue;
                }
                if (!movable(stop) || nmoved == MAX_MOVED)
                    break;
                moved[nmoved++] = stop;
                extra += regs_of(stop);
                for (unsigned i = 0; i < stop->nsrc; i++)
                    if (stop->src[i])
                        need[stop->src[i]->id] = 1;
            }
            for (unsigned i = 0; i < v->nsrc; i++)
                if (v->src[i])
                    need[v->src[i]->id] = 0;
            for (int i = 0; i < nmoved; i++) {
                need[moved[i]->id] = 0;
                for (unsigned j = 0; j < moved[i]->nsrc; j++)
                    if (moved[i]->src[j])
                        need[moved[i]->src[j]->id] = 0;
            }
            prev = v;
            struct ir_val *at = stop ? stop->next : b->first;
            while (nmoved && at == moved[nmoved - 1]) {
                at = at->next;
                nmoved--;
            }
            if (at == v)
                continue;
            for (int i = nmoved; i--;)
                link_before(moved[i], at);
            for (struct ir_val *x = v, *after; x != next; x = after) {
                after = x->next;
                link_before(x, at);
            }
            airc_batched = 1;
            count_pending(b, reads, pend, seen);
            block_pressure(b, out, pr, pos, last, stamp, diff, before, ++marks);
        }
    }
    free(reads);
    free(pend);
    free(pr);
    free(pos);
    free(last);
    free(stamp);
    free(diff);
    free(before);
    free(seen);
    free(need);
    free(out);
}

static void to_entry_start(struct ir_func *f, struct ir_val *v)
{
    ir_remove(v);
    v->block = f->entry;
    v->prev = NULL;
    v->next = f->entry->first;
    f->entry->first->prev = v;
    f->entry->first = v;
}

// A helper lane's extrapolated inputs on a sliver triangle can index a buffer at INT_MIN and fault the channel, so
// loads addressed from interpolants are guarded by coverage when their values reach nothing cross-lane.
static void guard_helper_loads(struct ir_func *f)
{
    if (g_info->stage != NVK_FRAGMENT || f->nsubs)
        return;
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            n = v->id >= n ? v->id + 1 : n;
    char *reach = calloc(n, 1), *varies = calloc(n, 1);
    int sinks = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next) {
            int lod = IR_TEXA_FLAGS(v->aux);
            if (v->op == IR_DERIV || v->op == IR_SHFL || v->op == IR_VOTE ||
                (v->op == IR_TEX && IR_TEXA_OP(v->aux) == SASS_TEX && SASS_TEXF_LOD(lod) != SASS_LOD_ZERO &&
                 SASS_TEXF_LOD(lod) != SASS_LOD_LOD)) {
                reach[v->id] = 1;
                sinks = 1;
            }
        }
    for (struct ir_block *b = f->entry; sinks && b; b = b->next)
        if (b->last && b->last->op == IR_CBR)  // a helper on another path changes what its quad sees
            reach[b->last->id] = 1;
    // local memory counts as one value
    for (int changed = 1; changed;) {
        changed = 0;
        int local = 0;
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *v = b->first; v; v = v->next)
                local |= v->op == IR_LDL && reach[v->id];
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *v = b->first; v; v = v->next) {
                if (!reach[v->id] && !(local && v->op == IR_STL))
                    continue;
                if (!reach[v->id])
                    changed = reach[v->id] = 1;
                for (unsigned i = 0; i < v->nsrc; i++)
                    if (v->src[i] && !reach[v->src[i]->id])
                        changed = reach[v->src[i]->id] = 1;
            }
    }
    // a uniform address is the same in every lane, so only interpolants matter
    for (int changed = 1; changed;) {
        changed = 0;
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *v = b->first; v; v = v->next) {
                int any = v->op == IR_IPA;
                for (unsigned i = 0; i < v->nsrc && !any; i++)
                    any = v->src[i] && varies[v->src[i]->id];
                if (any && !varies[v->id])
                    changed = varies[v->id] = 1;
            }
    }
    struct ir_val *covered = NULL;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next) {
            if (v->op != IR_LDG || reach[v->id] || v->nsrc != 1 || !varies[v->src[0]->id])
                continue;
            if (!covered) {
                struct ir_val *mask = ir_add(f->entry, IR_S2R, IR_I32, 0);
                mask->imm = SR_PIXLD(SASS_PIXLD_COVMASK);
                covered = ir_op2(f->entry, IR_CMP, IR_PRED, mask, ir_const(f->entry, IR_I32, 0));
                covered->aux = IR_NE;
                to_entry_start(f, covered);
                to_entry_start(f, covered->src[1]);
                to_entry_start(f, mask);
            }
            struct ir_val **src = ir_alloc(2 * sizeof(*src));
            src[0] = v->src[0];
            src[1] = covered;
            v->src = src;
            v->nsrc = 2;
        }
    free(varies);
    free(reach);
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
    // a wide result's block lives until its last lane does, so spill the lanes
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
            if (v->op == IR_PHI && slot[v->id])  // below
                continue;
            for (unsigned i = 0; i < v->nsrc; i++) {
                struct ir_val *s = v->src[i];
                if (!s || s->id > maxid || !slot[s->id])
                    continue;
                struct ir_val *ld = ir_add(v->op == IR_PHI ? v->phi_blocks[i] : b, IR_LDL, s->type, 0);
                ld->imm = base + 4u * (unsigned)(slot[s->id] - 1);
                ld->aux = 4;
                if (v->op != IR_PHI && v != b->last)  // a phi's at its predecessor's end
                    link_before(ld, v);
                v->src[i] = ld;
            }
        }
    // A spilled phi lives only in its slot, as a register would span its whole loop; each edge stores into it, after
    // loading every edge value first since phis read each other's old values.
    for (int pass = 0; pass < 2; pass++)
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *v = b->first; v && v->op == IR_PHI; v = v->next) {
                if (v->id > maxid || !slot[v->id])
                    continue;
                for (unsigned i = 0; i < v->nsrc; i++) {
                    struct ir_val *s = v->src[i];
                    int in_memory = s && s->id <= maxid && slot[s->id];
                    if (pass == 0 && in_memory && s->op == IR_PHI) {  // only phi slots are about to change
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
            struct ir_val *after = v;  // a phi's after the last phi
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

static char g_reserved[IR_MAX_SUBS + 1][NREGS];  // R1 (stack pointer) and callers' registers live across calls

// shader values a subroutine reads (the integer 1 and the like) stay live across every call reaching it
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

// Phi copies coalesce by Chaitin's test on the live runs, applied to whole groups. Fixed-latency ops read before
// writing, so their result may reuse a dying source's register; nothing else may. Clusters are never coalesced.
static int g_group[MAX_VREGS], g_gnext[MAX_VREGS];

static int group_of(int v)
{
    while (g_group[v] != v)
        v = g_group[v] = g_group[g_group[v]];
    return v;
}

static int in_place(enum mkind k)
{
    switch (k) {
    case MK_MOV:
    case MK_IADD3:
    case MK_ISUB:
    case MK_IMAD:
    case MK_IMADHI:
    case MK_LOP3:
    case MK_LOP3_IMM:
    case MK_PRMT:
    case MK_SHL:
    case MK_SHR:
    case MK_IMNMX:
    case MK_IABS:
    case MK_FADD:
    case MK_FMUL:
    case MK_FFMA:
    case MK_FMNMX:
    case MK_I2F:
    case MK_F2IP:
    case MK_F2H:
    case MK_H2F:
    case MK_FSWZADD:
    case MK_SEL:
        return 1;
    default:
        return 0;
    }
}

static int group_live(int g, int k)
{
    for (int v = g; v >= 0; v = g_gnext[v])
        for (int s = g_vregs[v].seg; s >= 0 && g_segs[s].start <= k; s = g_segs[s].next)
            if (g_segs[s].end >= k)
                return 1;
    return 0;
}

// a write never ends a part, so k + 1 follows k on every path
static int live_past(int g, int k, const char *part_start)
{
    const struct mop *m = &g_mops[k + 1];
    for (int v = g; v >= 0; v = g_gnext[v])
        for (int s = g_vregs[v].seg; s >= 0 && g_segs[s].start <= k + 1; s = g_segs[s].next) {
            if (g_segs[s].end <= k)
                continue;
            if (g_segs[s].start <= k || part_start[k + 1])
                return 1;
            for (int j = 0; j < m->nsrc + m->nxs; j++)
                if ((j < m->nsrc ? m->src[j] : m->xs[j - m->nsrc].vreg) == v)
                    return 1;
        }
    return 0;
}

// the mops defining v are dmop[dstart[v]..]
static int clobbers(int a, int b, const int *dstart, const int *dmop, const char *part_start)
{
    for (int v = a; v >= 0; v = g_gnext[v])
        for (int d = dstart[v]; d < dstart[v + 1]; d++) {
            const struct mop *m = &g_mops[dmop[d]];
            if (m->k == MK_MOV && m->src[0] >= 0 && group_of(m->src[0]) == b)
                continue;
            if (live_past(b, dmop[d], part_start) || (!in_place(m->k) && group_live(b, dmop[d])))
                return 1;
        }
    return 0;
}

// kept so the renames can be undone and redone
static struct {
    int *at, from, to;
} *g_renames;
static int g_nrenames, g_cap_renames;

static void rename_to(int *at, int to)
{
    if (*at == to)
        return;
    if (g_nrenames == g_cap_renames) {
        g_cap_renames = g_cap_renames ? 2 * g_cap_renames : 1024;
        g_renames = realloc(g_renames, g_cap_renames * sizeof(*g_renames));
    }
    g_renames[g_nrenames].at = at;
    g_renames[g_nrenames].from = *at;
    g_renames[g_nrenames++].to = to;
    *at = to;
}

static void rename_vregs(int coalesced)
{
    for (int i = 0; i < g_nrenames; i++)
        *g_renames[i].at = coalesced ? g_renames[i].to : g_renames[i].from;
}

static int coalesce_phis(int region)
{
    int lo = g_region_start[region], hi = g_region_end[region];
    int *dstart = calloc(g_nvregs + 1, sizeof(*dstart));
    char *single = calloc(g_nvregs, 1);
    for (int v = 0; v < g_nvregs; v++) {
        g_group[v] = v;
        g_gnext[v] = -1;
        single[v] = g_vregs[v].hw < 0 && g_vregs[v].size == 1;
    }
    for (int v = 0; v < g_nvregs; v++)
        for (int j = g_vregs[v].next; j >= 0; j = g_vregs[j].next)
            single[j] = single[v] = 0;
    for (int k = lo; k < hi; k++)
        for (int j = -1; j < g_mops[k].nxd; j++)
            MEMBERS(w, j < 0 ? g_mops[k].def : g_mops[k].xd[j].vreg)
                dstart[w + 1]++;
    for (int v = 0; v < g_nvregs; v++)
        dstart[v + 1] += dstart[v];
    int *dmop = malloc((dstart[g_nvregs] + 1) * sizeof(*dmop)), *fill = malloc(g_nvregs * sizeof(*fill));
    memcpy(fill, dstart, g_nvregs * sizeof(*fill));
    for (int k = lo; k < hi; k++)
        for (int j = -1; j < g_mops[k].nxd; j++)
            MEMBERS(w, j < 0 ? g_mops[k].def : g_mops[k].xd[j].vreg)
                dmop[fill[w]++] = k;
    char *part_start = calloc(hi + 1, 1);
    for (struct ir_block *b = g_fn->entry; b; b = b->next)
        if (!b->dead && b->func == region)
            part_start[g_block_mop[b->id]] = 1;
    for (int k = lo; k < hi; k++)
        if (g_mops[k].k == MK_BRA && g_mops[k].aux) {
            part_start[k + 1] = 1;
            if (g_mops[k].label >= 0)
                part_start[g_mops[k].label] = 1;
        }
    part_start[hi] = 1;
    int merged = 0;
    for (int k = lo; k < hi; k++) {
        const struct mop *m = &g_mops[k];
        if (m->k != MK_MOV || !m->aux || !single[m->def] || !single[m->src[0]])
            continue;
        int a = group_of(m->def), b = group_of(m->src[0]);
        if (a == b || clobbers(a, b, dstart, dmop, part_start) || clobbers(b, a, dstart, dmop, part_start))
            continue;
        int t = a;
        while (g_gnext[t] >= 0)
            t = g_gnext[t];
        g_gnext[t] = b;
        g_group[b] = a;
        merged = 1;
    }
    // every mop, as subroutines read shader values too
    g_nrenames = 0;
    for (int k = 0; merged && k < g_nmops; k++) {
        struct mop *m = &g_mops[k];
        if (m->def >= 0)
            rename_to(&m->def, group_of(m->def));
        for (int j = 0; j < m->nsrc; j++)
            if (m->src[j] >= 0)
                rename_to(&m->src[j], group_of(m->src[j]));
        for (int j = 0; j < m->nxs; j++)
            rename_to(&m->xs[j].vreg, group_of(m->xs[j].vreg));
        for (int j = 0; j < m->nxd; j++)
            rename_to(&m->xd[j].vreg, group_of(m->xd[j].vreg));
    }
    free(dstart);
    free(single);
    free(dmop);
    free(fill);
    free(part_start);
    return merged;
}

static int g_clusters_first;
static int g_second_pass;

// GA10x has four partitions of 16K registers, allocated in units of 8, and at most 48 warps
static int warps_at(int regs)
{
    regs = (regs + 7) & ~7;
    int w = 16384 / (regs * 32) * 4;
    return w < 48 ? w : 48;
}

static int last_taken(const uint64_t *row, const int *row_end, int c, int k)
{
    if (row_end[c] < k)
        return row_end[c];
    for (int w = k / 64; w >= 0; w--) {
        uint64_t bits = row[w];
        if (w == k / 64)
            bits &= (1ull << (k % 64)) - 1;
        if (bits)
            return w * 64 + 63 - __builtin_clzll(bits);
    }
    return -1;
}

static void took(int *row_end, int v, int base)
{
    for (int s = g_vregs[v].seg; s >= 0; s = g_segs[s].next)
        if (g_segs[s].end - base > row_end[g_vregs[v].hw])
            row_end[g_vregs[v].hw] = g_segs[s].end - base;
}

static int place_before(int a, int b)
{
    int ca = g_vregs[a].size > 1, cb = g_vregs[b].size > 1;
    if (g_clusters_first && ca != cb)
        return ca;
    return g_vregs[a].start < g_vregs[b].start;
}

// returns the register count, or -1 with spill candidates noted
static int place(int region)
{
    int *xfer = malloc((g_region_end[region] - g_region_start[region] + 1) * sizeof(*xfer)), nxfer = 0;
    for (int k = g_region_start[region]; k < g_region_end[region]; k++)
        if (is_transfer(&g_mops[k]))
            xfer[nxfer++] = k;
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
    // Packed in start order and clusters-first, keeping the smaller, then spread within the same occupancy into the
    // longest-idle register, since reusing a just-freed one blocks the scheduler. If spreading leaves a cluster no
    // room, the other order is tried, and in the second pass only singles spread (doing that in the first pass lost
    // six corpus kernels a register granule). Local memory shaders spread down to 40 warps: a BVH walk was 5% faster
    // there than packed at 48, while a shader without local memory was 10% slower.
    int *best = malloc(g_nvregs * sizeof(*best)), best_count = 0, fitted = 0, best_first = 0, budget = NREGS;
    int *row_end = malloc(NREGS * sizeof(*row_end));
    int spread_fitted = 0;
    for (int attempt = 0; attempt < 4 + g_second_pass && !spread_fitted; attempt++) {
        int spread = attempt >= 2;
        if (attempt == 2) {
            if (!fitted)
                break;
            // + 1 for the exit's temporary and + 2 for the program counter
            budget = best_count;
            int target = warps_at(best_count + 3);
            if (g_info->local_size && target > 40)
                target = 40;
            while (budget < NREGS && warps_at(budget + 1 + 3) >= target)
                budget++;
            if (budget == best_count)
                break;
        }
        g_clusters_first = attempt == 4 || (spread ? best_first ^ (attempt == 3) : attempt);
        for (int c = 0; c < NREGS; c++)
            row_end[c] = -1;
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
            // a copy whose source dies there takes its register and vanishes
            const struct mop *dm = &g_mops[start];
            int src = dm->k == MK_MOV && dm->def == h ? dm->src[0] : -1;
            if (src >= 0 && src != VREG_RZ && g_vregs[h].size == 1 && g_vregs[src].size == 1 && !member[src] &&
                placed[src] && g_vregs[src].end == start && g_vregs[src].hw >= from) {
                uint64_t *row = occ + (size_t)g_vregs[src].hw * words;
                row[(start - base) / 64] &= ~(1ull << ((start - base) % 64));
                if (seg_free(row, g_vregs[h].seg, base)) {
                    g_vregs[h].hw = g_vregs[src].hw;
                    seg_take(row, g_vregs[h].seg, base);
                    took(row_end, h, base);
                    placed[h] = 1;
                    continue;
                }
                row[(start - base) / 64] |= 1ull << ((start - base) % 64);
            }
            int r = -1, size = g_vregs[h].size, align = g_vregs[h].align, age = 0;
            int scatter = spread && (attempt < 4 || size == 1);
            for (int c = (from + align - 1) & ~(align - 1); c + size <= budget && (r < 0 || scatter); c += align) {
                int ok = 1, j = h, last = -1;
                for (int k = 0; k < size && ok; k++, j = g_vregs[j].next) {
                    ok = !g_reserved[region][c + k] && seg_free(occ + (size_t)(c + k) * words, g_vregs[j].seg, base);
                    int t = ok && scatter ? last_taken(occ + (size_t)(c + k) * words, row_end, c + k, start - base)
                                          : -1;
                    if (t > last)
                        last = t;
                }
                if (ok && (r < 0 || last < age)) {
                    r = c;
                    age = last;
                }
            }
            if (r < 0) {
                failed = i;
                break;
            }
            int k = 0;
            for (int j = h; j >= 0; j = g_vregs[j].next, k++) {
                g_vregs[j].hw = r + k;
                seg_take(occ + (size_t)(r + k) * words, g_vregs[j].seg, base);
                took(row_end, j, base);
                if (r + k + 1 > g_regcount)
                    g_regcount = r + k + 1;
            }
            placed[h] = 1;
        }
        spread_fitted = spread && failed < 0;
        if (failed < 0 && (!fitted || g_regcount < best_count || spread)) {
            fitted = 1;
            best_count = g_regcount;
            best_first = g_clusters_first;
            for (int i = 0; i < n; i++)
                for (int j = order[i]; j >= 0; j = g_vregs[j].next)
                    best[j] = g_vregs[j].hw;
        }
        if (failed >= 0 && attempt && !fitted) {
            int na = 0, start = g_vregs[order[failed]].start;
            for (int a = 0; a < failed; a++)
                if (g_vregs[order[a]].start <= start && g_vregs[order[a]].end >= start)
                    active[na++] = order[a];
            note_spill_candidates(active, na);
        }
        for (int i = 0; i < (failed < 0 ? n : failed); i++)
            for (int j = order[i]; j >= 0; j = g_vregs[j].next)
                g_vregs[j].hw = -1;
        g_regcount = regcount;
    }
    for (int i = 0; i < n && fitted; i++)
        for (int j = order[i]; j >= 0; j = g_vregs[j].next)
            g_vregs[j].hw = best[j];
    g_regcount = fitted ? best_count : regcount;
    free(xfer);
    free(order);
    free(member);
    free(best);
    free(row_end);
    free(occ);
    free(placed);
    free(active);
    return fitted ? best_count : -1;
}

static int regalloc(struct ir_func *f, int region)
{
    // coalescing can leave clusters no aligned room, so it is kept only if it costs no extra 8-register granule
    int *hw0 = malloc(g_nvregs * sizeof(*hw0)), *hw1 = malloc(g_nvregs * sizeof(*hw1)), regcount = g_regcount;
    for (int v = 0; v < g_nvregs; v++)
        hw0[v] = g_vregs[v].hw;
    compute_ranges(f, region);
    int coalesced = coalesce_phis(region), count = -1, plain;
    if (coalesced) {
        compute_ranges(f, region);
        count = place(region);
        for (int v = 0; v < g_nvregs; v++) {
            hw1[v] = g_vregs[v].hw;
            g_vregs[v].hw = hw0[v];
        }
        int used = g_regcount;
        g_regcount = regcount;
        g_ncand = 0;
        rename_vregs(0);
        compute_ranges(f, region);
        plain = place(region);
        if (count >= 0 && (plain < 0 || (count + 9) / 8 <= (plain + 9) / 8)) {
            for (int v = 0; v < g_nvregs; v++)
                g_vregs[v].hw = hw0[v];
            rename_vregs(1);
            compute_ranges(f, region);
            for (int v = 0; v < g_nvregs; v++)
                g_vregs[v].hw = hw1[v];
            g_regcount = used;
            plain = count;
        }
    } else {
        plain = place(region);
    }
    free(hw0);
    free(hw1);
    if (plain < 0)
        return -1;
    for (int k = g_region_start[region]; k < g_region_end[region]; k++) {
        const struct mop *m = &g_mops[k];
        if (m->k != MK_CALL)
            continue;
        char *keep = g_reserved[m->imm + 1];
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
    return 0;
}

static int hw(int vreg)
{
    if (vreg == VREG_RZ)
        return SASS_RZ;
    return g_vregs[vreg].hw;
}

// sched_push's destination kinds; 0 is a variable-latency result, waited for on a scoreboard
#define FIXED_LAT 1
#define PRED_LAT 2
#define UNIFORM_LAT 3
#define BAR_LAT 2
#define UNIFORM_CYCLES 7  // a load reading UR4 right after its ULDC reads garbage
#define UR_PSEUDO 254     // all loaded in the prologue
#define P0_PSEUDO 250
#define P6_PSEUDO 249

// Pipe classes for NVIDIA's sm_80 latency table as NAK has it; IMAD.WIDE's halves and operands differ in timing.
enum { WC_NONE, WC_ALU, WC_FMA, WC_WIDE_LO, WC_WIDE_HI, WC_FP16, WC_PRED, WC_UNIFORM, WC_I2FP };
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
    case 0x043:  // F2IP, as nvcc spaces its readers
    case 0x045:
        return WC_I2FP;
    default:
        return WC_ALU;
    }
}

static int reader_class(const struct sass_insn *in, int i)
{
    switch (in->lo & 0x1ff) {
    case 0x002: case 0x007: case 0x009: case 0x00b: case 0x00c: case 0x010: case 0x012: case 0x016: case 0x017:
    case 0x019: case 0x03e:
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
    case 0x043: case 0x045: case 0x100: case 0x101: case 0x105: case 0x107: case 0x108: case 0x109: case 0x119:
    case 0x150: case 0x161: case 0x164: case 0x167: case 0x16d: case 0x170: case 0x182:
        return RC_DECOUPLED;
    default:
        return RC_AGU;
    }
}

static unsigned raw_latency(int writer, int reader)
{
    // [reader][writer ALU, FMA, WIDE_LO, WIDE_HI, FP16]
    static const uint8_t reg[][5] = {
        [RC_ALU] = { 4, 5, 3, 5, 5 },       [RC_FMA] = { 5, 4, 2, 4, 5 },       [RC_WIDE_AB] = { 5, 4, 4, 6, 5 },
        [RC_WIDE_CH] = { 3, 2, 2, 2, 3 },   [RC_FP16] = { 5, 5, 3, 5, 5 },      [RC_DECOUPLED] = { 4, 4, 4, 4, 4 },
        [RC_AGU] = { 5, 5, 5, 5, 5 },
    };
    if (writer == WC_UNIFORM)
        return UNIFORM_CYCLES;
    if (writer == WC_I2FP)  // not in the table; nvcc's sm_86 distances
        return reader == RC_ALU ? 4 : reader == RC_FMA ? 5 : 6;
    if (writer == WC_PRED)  // only ALU and FMA take a predicate quickly
        return reader == RC_ALU ? 4 : reader == RC_FMA ? 5 : 13;
    // at the table's figures, thread ID, threadgroup memory and SIMD probes read stale operands on GA106
    unsigned slack = reader != RC_ALU && reader != RC_FMA;
    return reg[reader == RC_VOTE ? RC_ALU : reader][writer - WC_ALU] + slack;
}

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
static unsigned g_wcycle[256];  // issue cycle of each register's fixed-latency write
static uint8_t g_wclass[256];
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

// filled by running a mop's emitter with g_probe set, for the list scheduler
struct fx {
    uint8_t rd[16], rclass[16], wr[8], wclass[8];
    int nrd, nwr;
    int async;     // result on a scoreboard
    int late;      // sources read after issue
    int overflow;  // too many operands, so the mop stays put
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

// a drain's waits for the next instruction, harmless if it's also reached from elsewhere
static unsigned g_carried_waits;

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
    in.hi |= (uint64_t)g_carried_waits << 52;
    g_carried_waits = 0;
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

// Pending scoreboards per register, for results landing (g_wbar_of) and late source reads (g_rbar_of). Several can
// be pending where paths meet, and a write must wait for all of them.
static unsigned g_wbar_of[256], g_rbar_of[256];

// Scoreboards of atomics and shared loads; a threadgroup barrier must wait for them, or other warps see the word
// before it lands.
static unsigned g_mem_bars;
static int g_mem_op;

// The mop's late-read registers aren't rewritten in its run, so its write barrier covers them and saves one
// (schedule_segment).
static unsigned char *g_rd_free_of;
static int g_rd_free;

static void settle(unsigned waits)
{
    if (!waits)
        return;
    for (int r = 0; r < 256; r++) {
        g_wbar_of[r] &= ~waits;
        g_rbar_of[r] &= ~waits;
    }
    g_mem_bars &= ~waits;
}

static unsigned taken(void)
{
    unsigned used = 0;
    for (int r = 0; r < 256; r++)
        used |= g_wbar_of[r] | g_rbar_of[r];
    return used;
}

// Prefers a free scoreboard, then a stale one (waited for at no cost), then the oldest set for the same kind of op,
// as a short op's readers can't afford a memory wait. Never the instruction's own write barrier in avoid.
#define STALE_BAR 150
#define STALE_SHORT 30
static unsigned g_bar_long;  // set for memory or texture ops
static int g_long_op;
static int take_bar(unsigned *wait, unsigned avoid, int lng)
{
    unsigned used = taken();
    int best = -1, cost_best = 4;
    for (int b = 0; b < 6; b++) {
        if (avoid >> b & 1)
            continue;
        int bl = g_bar_long >> b & 1;
        int cost = !(used >> b & 1) ? 0 : g_bar_set[b] + (bl ? STALE_BAR : STALE_SHORT) <= g_cycle ? 1 : 2 + (bl != lng);
        if (cost < cost_best || (cost == cost_best && g_bar_set[b] < g_bar_set[best])) {
            best = b;
            cost_best = cost;
        }
    }
    if (cost_best == 0)
        g_mem_bars &= ~(1u << best);
    else if (cost_best == 1) {
        *wait |= 1u << best;
        settle(1u << best);
    }
    if (cost_best < 2)
        g_bar_long &= ~(1u << best);
    g_bar_long |= (unsigned)lng << best;
    return best;
}

// The control word waits for RAW, WAW and WAR hazards on scoreboards, and sets a read barrier for late sources
// and a write barrier for an async result. dstw is how many registers dst writes.
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
        if (srcs[i] != SASS_RZ)
            c.wait |= g_wbar_of[srcs[i]];
    for (int k = 0; dst >= 0 && k < dstw; k++)
        c.wait |= g_wbar_of[dst + k] | g_rbar_of[dst + k];
    settle(c.wait);
    unsigned wait = c.wait;
    if (var_dst) {
        c.wr = take_bar(&wait, 0, g_long_op);
        if (g_mem_op)
            g_mem_bars |= 1u << c.wr;
        for (int k = 0; k < dstw; k++)
            g_wbar_of[dst + k] = 1u << c.wr;
    }
    // a result lands after its sources are read, so with all scoreboards taken its barrier covers the reads too
    if (late) {
        if (!var_dst || !g_rd_free || taken() != 0x3f)
            c.rd = take_bar(&wait, var_dst ? 1u << c.wr : 0, 0);
        int b = c.rd >= 0 ? c.rd : c.wr;
        if (g_mem_op)
            g_mem_bars |= 1u << b;
        for (int i = 0; i < nsrc; i++)
            if (srcs[i] != SASS_RZ && srcs[i] < 240)
                g_rbar_of[srcs[i]] |= 1u << b;
    }
    c.wait = wait;
    return sass_ctrl_word(c);
}

// The branch target doesn't know fixed-latency timing, and a scoreboard set just before mustn't be waited on
// straight after the jump.
static void settle_fixed(void)
{
    unsigned need = g_cycle;
    for (int r = 0; r < 256; r++)
        if (g_wclass[r] != WC_NONE) {
            unsigned t = g_wcycle[r] + worst_latency(g_wclass[r]);
            if (t > need)
                need = t;
            g_wclass[r] = WC_NONE;
        }
    unsigned pending = taken() | g_mem_bars;
    for (int b = 0; b < 6; b++)
        if ((pending >> b & 1) && g_bar_set[b] + BAR_LAT > need)
            need = g_bar_set[b] + BAR_LAT;
    stall_until(need);
}

// before a back branch, call or return, so the target starts with nothing pending
static void drain(void)
{
    settle_fixed();
    unsigned wait = taken();
    settle(wait);
    g_carried_waits |= wait;
}

// What forward branches leave pending at a mop, merged over every way in; back branches drain, so a loop head only
// sees the ways in from before it.
struct pending {
    unsigned wbar[256], rbar[256], mem;
};
static struct pending **g_into;
static int g_reached;  // falls through from the previous instruction

static void branch_to(int mop)
{
    if (!g_into[mop])
        g_into[mop] = calloc(1, sizeof(**g_into));
    struct pending *p = g_into[mop];
    for (int r = 0; r < 256; r++) {
        p->wbar[r] |= g_wbar_of[r];
        p->rbar[r] |= g_rbar_of[r];
    }
    p->mem |= g_mem_bars;
}

static void arrive(int mop)
{
    struct pending *p = g_into[mop];
    if (!g_reached) {
        memset(g_wbar_of, 0, sizeof(g_wbar_of));
        memset(g_rbar_of, 0, sizeof(g_rbar_of));
        g_mem_bars = 0;
    }
    g_reached = 1;
    if (!p)
        return;
    for (int r = 0; r < 256; r++) {
        g_wbar_of[r] |= p->wbar[r];
        g_rbar_of[r] |= p->rbar[r];
    }
    g_mem_bars |= p->mem;
    free(p);
    g_into[mop] = NULL;
}

// a demoted thread's writes go under @!P6, so P6 joins their sources
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

static unsigned *g_mop_np;
static int *g_block_np;
static struct {
    unsigned at;
    int block;     // or -1
    int mop;       // a stub, or -1
} g_fix[8192];
static int g_nfix;

static uint8_t mem_size(int bytes)
{
    return bytes == 1 ? SASS_U8 : bytes == 2 ? SASS_U16 : bytes == 8 ? SASS_B64 : bytes == 16 ? SASS_B128 : SASS_B32;
}

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

// s0..s2 are the operand slots of sources 0..2, -1 for none
static struct sass_insn with_mods(struct sass_insn in, int mods, int s0, int s1, int s2)
{
    const int slot[3] = { s0, s1, s2 };
    for (int i = 0; i < 3; i++)
        if (slot[i] >= 0)
            in = sass_src_mod(in, slot[i], mods >> 2 * i & MOD_NEG, (mods >> 2 * i & MOD_ABS) != 0);
    return in;
}

// Apple's GPUs flush denormals (denormal * inf is NaN, step(denormal, 0) is 1); OpenCL keeps them, as that's free
static struct sass_insn ftz(struct sass_insn in)
{
    return g_info->denorms ? in : sass_ftz(in);
}

static void emit_mov(int d, int s)
{
    if (d == s)
        return;
    uint8_t src[1] = { (uint8_t)s };
    uint32_t c = sched(src, 1, 0, d, 1, 0);
    sched_push(sass_mov_rr(d, s, c), src, 1, d, 0, FIXED_LAT);
}

// BRA, BSSY, CALL and RET share the offset field
static void set_rel(struct sass_insn *in, int32_t rel)
{
    in->lo = (in->lo & 0xffffffffull) | ((uint64_t)(uint32_t)rel << 32);
    in->hi = (in->hi & ~0x3ffffull) | (rel < 0 ? 0x3ffff : 0);
}

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
    if (m->aux >= 0) {  // Volta+ threads may return from a subroutine separately
        sched_push(sass_bssy((uint8_t)m->aux, 0, sched(NULL, 0, 0, -1, 0, 0)), NULL, 0, -1, 0, 0);
        bssy = g_np - 1;
    }
    sched_push(sass_call_rel(0, sched(NULL, 0, 0, -1, 0, 0)), NULL, 0, -1, 0, 0);
    unsigned at = g_np - 1;
    g_fix[g_nfix].at = at;
    g_fix[g_nfix].block = m->target;
    g_fix[g_nfix++].mop = -1;
    // RET.REL's target is relative to the program start
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
    g_reached = 0;
}

// Exit colours move to R0.. as a parallel copy, breaking cycles through the scratch register. AIRC_STAGEEXIT=1
// stages every colour through scratch instead.
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

// An edge whose copies all coalesced needs no stub, so a loop branches once a trip; the conditional branch targets
// the block directly, or is negated when the true block follows.
static int coalesced(const struct mop *m)
{
    return m->k == MK_MOV && hw(m->def) == hw(m->src[0]);
}

// past nothing but coalesced copies and dropped branches
static int falls_into(int i, int b)
{
    int k = i + 1;
    while (k < g_block_mop[b] &&
           (coalesced(&g_mops[k]) || (g_mops[k].k == MK_BRA && g_mops[k].target < 0 && g_mops[k].label < 0)))
        k++;
    return k == g_block_mop[b];
}

static void skip_empty_edges(void)
{
    for (int i = 0; i < g_nmops; i++) {
        struct mop *m = &g_mops[i];
        if (m->k != MK_BRA || !m->aux)
            continue;
        if (m->label >= 0) {
            int k = m->label;
            while (coalesced(&g_mops[k]))
                k++;
            if (g_mops[k].k != MK_BRA || g_mops[k].aux)
                continue;
            m->target = g_mops[k].target;
            m->label = -1;
            g_mops[k].target = -1;
        }
        int k = i + 1;
        while (coalesced(&g_mops[k]))
            k++;
        if (g_mops[k].k == MK_BRA && !g_mops[k].aux && g_mops[k].target >= 0 && falls_into(k, m->target)) {
            m->aux = 2;
            m->target = g_mops[k].target;
            g_mops[k].target = -1;
        }
    }
}

static void emit_mops_in(const int *list, int n, enum nvk_stage stage)
{
    for (int k = 0; k < n; k++) {
        int i = list[k];
        struct mop *m = &g_mops[i];
        if (!g_probe) {
            g_mop_np[i] = g_np;
            g_rd_free = g_rd_free_of[i];
            g_long_op = m->k == MK_TEX || m->k == MK_LDG || m->k == MK_ATOM;
        }
        int d = m->def >= 0 ? hw(m->def) : -1;
        uint8_t s[6];
        for (int j = 0; j < m->nsrc; j++)
            s[j] = (uint8_t)hw(m->src[j]);
        uint32_t c;
        switch (m->k) {
        case MK_MOVIMM:
            if (g_vregs[m->def].end == i && g_vregs[m->def].start == i)
                break;  // its reader took it as an immediate after all
            c = sched(NULL, 0, 0, d, 1, 0);
            sched_push(sass_imadmov_imm(d, (uint32_t)m->imm, c), NULL, 0, d, 0, FIXED_LAT);
            break;
        case MK_MOV:
            // the probe still sees a coalesced copy, as the second allocation may not coalesce it
            if (d == s[0] && !g_probe)
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
        case MK_ALD:  // the vertex handle is read late
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
            sched_push(m->immb      ? sass_imm(sass_iadd3_rr(d, s[0], 0, c), (uint32_t)m->imm)
                       : m->nsrc > 2 ? sass_iadd3_rrr(d, s[0], s[1], s[2], c)
                                     : sass_iadd3_rr(d, s[0], s[1], c),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        case MK_ISUB:
            c = sched(s, 2, 0, d, 1, 0);
            sched_push(sass_isub_rr(d, s[0], s[1], c), s, 2, d, 0, FIXED_LAT);
            break;
        case MK_IMAD: {
            uint8_t rc = m->nsrc > 2 ? s[2] : SASS_RZ, ss[3] = { s[0] };
            int n = 1;
            if (!m->immb)
                ss[n++] = s[1];
            if (m->nsrc > 2)
                ss[n++] = rc;
            c = sched(ss, n, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(sass_imad_rr(d, s[0], 0, rc, c), (uint32_t)m->imm)
                               : sass_imad_rr(d, s[0], s[1], rc, c),
                       ss, n, d, 0, FIXED_LAT);
            break;
        }
        case MK_IMADHI:
            c = sched(s, 2, 0, d, 1, 0);
            sched_push(sass_imadhi_rr(d, s[0], s[1], m->aux, c), s, 2, d, 0, FIXED_LAT);
            break;
        case MK_LOP3: {
            c = sched(s, m->nsrc, 0, d, 1, 0);
            struct sass_insn in = m->immb ? sass_imm(sass_lop3(d, s[0], 0, (uint8_t)m->aux, c), (uint32_t)m->imm)
                                          : sass_lop3(d, s[0], s[1], (uint8_t)m->aux, c);
            int rc = m->immb ? (m->nsrc > 1 ? 1 : 0) : (m->nsrc > 2 ? 2 : 0);  // source c
            if (rc > 0)
                in.hi = (in.hi & ~0xffull) | s[rc];
            sched_push(in, s, m->nsrc, d, 0, FIXED_LAT);
            break;
        }
        case MK_PRMT:
            c = sched(s, 1, 0, d, 1, 0);
            sched_push(sass_prmt(d, s[0], (uint16_t)m->imm, SASS_RZ, c), s, 1, d, 0, FIXED_LAT);
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
        case MK_IABS:
            c = sched(s, 1, 0, d, 1, 0);
            sched_push(sass_iabs(d, s[0], c), s, 1, d, 0, FIXED_LAT);
            break;
        case MK_IMNMX:
            c = sched(s, m->nsrc, 0, d, 1, 0);
            sched_push(m->immb ? sass_imm(sass_imnmx(d, s[0], 0, m->aux & 1, (m->aux & 2) != 0, c), (uint32_t)m->imm)
                               : sass_imnmx(d, s[0], s[1], m->aux & 1, (m->aux & 2) != 0, c),
                       s, m->nsrc, d, 0, FIXED_LAT);
            break;
        case MK_FADD: {
            c = sched(s, m->nsrc, 0, d, 1, 0);
            struct sass_insn add =
                m->aux & 2    ? sass_hadd2(d, s[0], s[1], m->aux & 1, c)
                : m->aux & 16 ? with_mods(ftz(sass_fadd_imm(d, s[0], (uint32_t)m->imm, c)), m->mods, SASS_MOD_A,
                                          -1, -1)
                              : with_mods(ftz(sass_fadd(d, s[0], s[1], m->aux & 1, c)), m->mods, SASS_MOD_A,
                                          SASS_MOD_B, -1);
            if (m->aux & 64)
                add = sass_sat(add);
            if (m->aux & 128)
                add = sass_neg_a(add);
            sched_push(add, s, m->nsrc, d, 0, FIXED_LAT);
            break;
        }
        case MK_FMUL: {
            c = sched(s, m->nsrc, 0, d, 1, 0);
            struct sass_insn mul = m->aux & 2    ? sass_hmul2(d, s[0], s[1], c)
                                   : m->aux & 16 ? with_mods(ftz(sass_fmul_imm(d, s[0], (uint32_t)m->imm, c)),
                                                             m->mods, SASS_MOD_A, -1, -1)
                                                 : with_mods(ftz(sass_fmul(d, s[0], s[1], c)), m->mods,
                                                             SASS_MOD_A, SASS_MOD_B, -1);
            sched_push(m->aux & 64 ? sass_sat(mul) : mul, s, m->nsrc, d, 0, FIXED_LAT);
            break;
        }
        case MK_FFMA: {
            int mode = m->aux >> 2 & 3;
            c = sched(s, m->nsrc, 0, d, 1, 0);
            struct sass_insn fma =
                m->aux & 2    ? sass_hfma2(d, s[0], s[1], s[2], c)
                : m->aux & 16 ? with_mods(ftz(sass_ffma_imm_b(d, s[0], (uint32_t)m->imm, s[1], mode, c)), m->mods,
                                          SASS_MOD_A, SASS_MOD_C, -1)
                : m->aux & 32 ? with_mods(ftz(sass_ffma_imm_c(d, s[0], s[1], (uint32_t)m->imm, mode, c)),
                                          m->mods, SASS_MOD_A, SASS_MOD_C, -1)
                              : with_mods(ftz(sass_ffma(d, s[0], s[1], s[2], mode, c)), m->mods, SASS_MOD_A,
                                          SASS_MOD_B, SASS_MOD_C);
            sched_push(m->aux & 64 ? sass_sat(fma) : fma, s, m->nsrc, d, 0, FIXED_LAT);
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
        case MK_F2IP:
            c = sched(s, 1, 0, d, 1, 0);
            sched_push(sass_f2ip_u8(d, s[0], c), s, 1, d, 0, FIXED_LAT);
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
            sched_push(ftz(sass_frnd(d, s[0], m->aux, c)), s, 1, d, 0, 0);  // flushed, floor(-denormal) is -0
            break;
        case MK_H2F:
            c = sched(s, 1, 0, d, 1, 0);
            sched_push(m->aux & 1 ? sass_h2f_hi(sass_h2f(d, s[0], c)) : sass_h2f(d, s[0], c), s, 1, d, 0, FIXED_LAT);
            break;
        case MK_ISETP:
        case MK_FSETP: {
            int n = m->nsrc;
            if (m->aux & (SETP_AND | SETP_OR))
                s[n++] = P0_PSEUDO;
            c = sched(s, n, 0, P0_PSEUDO, 1, 0);
            struct sass_insn in;
            if (m->k == MK_ISETP) {
                uint8_t op;
                int sgn;
                isetp_op(m->aux, &op, &sgn);
                in = m->immb ? sass_imm(sass_isetp(0, s[0], 0, op, sgn, c), (uint32_t)m->imm)
                             : sass_isetp(0, s[0], s[1], op, sgn, c);
            } else {
                in = m->immb ? with_mods(ftz(sass_imm(sass_fsetp(0, s[0], 0, fsetp_op(m->aux & 15), c),
                                                           (uint32_t)m->imm)),
                                         m->mods, SASS_MOD_A, -1, -1)
                             : with_mods(ftz(sass_fsetp(0, s[0], s[1], fsetp_op(m->aux & 15), c)), m->mods,
                                         SASS_MOD_A, SASS_MOD_B, -1);
            }
            if (m->aux & (SETP_AND | SETP_OR))
                in = sass_setp_bop(in, m->aux & SETP_OR ? 1 : 0, 0);
            sched_push(in, s, n, P0_PSEUDO, 0, PRED_LAT);
            break;
        }
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
        case MK_ATOM: {  // with no result, only a read barrier on its sources
            int shared = IR_ATOMA_SHARED(m->aux), ret = m->def >= 0;
            uint8_t ss[6] = { s[0], s[1], s[2], (uint8_t)(s[0] + 1) };
            int n = demote_src(ss, shared ? 3 : 4);
            g_mem_op = 1;
            c = sched(ss, n, ret, d, ret, 1);
            g_mem_op = 0;
            struct sass_insn in;
            if (ret || shared) {
                in = sass_atom(IR_ATOMA_OP(m->aux), shared, IR_ATOMA_TYPE(m->aux), ret ? d : SASS_RZ, s[0],
                               (uint32_t)m->imm, s[1], s[2], c);
            } else {
                in = sass_red(IR_ATOMA_OP(m->aux), IR_ATOMA_TYPE(m->aux), s[0], (uint32_t)m->imm, s[1], c);
                ss[n++] = UR_PSEUDO;  // .E reads the descriptor
            }
            sched_push(demote_guard(in), ss, n, d, 0, 0);
            break;
        }
        case MK_LDG: {
            int wide = m->src[0] != VREG_RZ && !m->narrow, n = 0;
            uint8_t ss[3];
            if (m->src[0] != VREG_RZ)
                ss[n++] = s[0];
            if (wide)
                ss[n++] = (uint8_t)(s[0] + 1);
            c = sched(ss, n, 1, d, m->aux > 4 ? m->aux / 4 : 1, n > 0);
            ss[n++] = UR_PSEUDO;
            struct sass_insn in = m->ur ? sass_ldg_ur(d, s[0], wide, (uint8_t)m->ur, (uint32_t)m->imm, mem_size(m->aux), c)
                                        : sass_ldg(d, s[0], (uint32_t)m->imm, mem_size(m->aux), c);
            if (m->readonly)
                in = sass_ldg_constant(in);
            if (m->guarded) {
                in = sass_pred(in, 0, 0);
                ss[n++] = P0_PSEUDO;
            }
            sched_push(in, ss, n, d, 0, 0);
            break;
        }
        case MK_STG: {
            int words = m->aux > 4 ? m->aux / 4 : 1, wide = m->src[0] != VREG_RZ && !m->narrow, n = 0;
            uint8_t ss[8];
            if (m->src[0] != VREG_RZ)
                ss[n++] = s[0];
            if (wide)
                ss[n++] = (uint8_t)(s[0] + 1);
            for (int k = 0; k < words; k++)
                ss[n++] = (uint8_t)(s[1] + k);
            n = demote_src(ss, n);
            c = sched(ss, n, 0, -1, 0, 1);
            ss[n++] = UR_PSEUDO;
            sched_push(demote_guard(m->ur ? sass_stg_ur(s[0], wide, (uint8_t)m->ur, (uint32_t)m->imm, s[1],
                                                        mem_size(m->aux), c)
                                          : sass_stg(s[0], (uint32_t)m->imm, s[1], mem_size(m->aux), c)),
                       ss, n, -1, 0, 0);
            break;
        }
        case MK_LDS: {
            uint8_t a = m->nsrc ? s[0] : SASS_RZ;
            uint8_t ss[1] = { a };
            g_mem_op = 1;
            c = sched(ss, m->nsrc, 1, d, m->aux > 4 ? m->aux / 4 : 1, 1);
            g_mem_op = 0;
            sched_push(sass_lds(d, a, (uint32_t)m->imm, mem_size(m->aux), c), ss, m->nsrc, d, 0, 0);
            break;
        }
        case MK_STS: {  // all of a wide store's data is read late
            int words = m->aux > 4 ? m->aux / 4 : 1;
            uint8_t ss[5] = { s[0] };
            for (int k = 0; k < words; k++)
                ss[1 + k] = (uint8_t)(s[1] + k);
            c = sched(ss, 1 + words, 0, -1, 0, 1);
            sched_push(sass_sts(s[0], (uint32_t)m->imm, s[1], mem_size(m->aux), c), ss, 1 + words, -1, 0, 0);
            break;
        }
        case MK_LDL: {
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
            if (m->aux) {  // a fence, with nvcc's control words
                sched_push(sass_membar_gpu(0x000fec00), NULL, 0, -1, 0, 0);
                sched_push(sass_errbar(0x000fc000), NULL, 0, -1, 0, 0);
                sched_push(sass_cctl_ivall(0x000fca00), NULL, 0, -1, 0, 0);
                break;
            }
            c = sched(NULL, 0, 0, -1, 0, 0);
            unsigned held = 0;
            for (int r = 0; r < 256; r++)
                held |= (g_wbar_of[r] | g_rbar_of[r]) & g_mem_bars;
            settle(held);
            // six cycles as nvcc and NAK use; a shared load 1-3 cycles after BAR missed other warps' stores
            c = (c & ~(0xfu << 9)) | 6u << 9;
            sched_push(sass_bar_sync(c | held << 20), NULL, 0, -1, 0, 0);
            break;
        case MK_BSSY:
            c = sched(NULL, 0, 0, -1, 0, 0);
            sched_push(sass_bssy((uint8_t)m->imm, 0, c), NULL, 0, -1, 0, 0);
            g_fix[g_nfix].at = g_np - 1;
            g_fix[g_nfix].block = m->target;
            g_fix[g_nfix++].mop = 0;
            break;
        case MK_BSYNC:
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
            uint8_t ss[4] = { s[0], s[2], (uint8_t)(s[2] + 1), s[1] };
            int n = m->immb ? 3 : 4;
            c = sched(ss, n, 0, d, 2, 0);
            struct sass_insn in = sass_imadwide_rr(d, s[0], m->immb ? 0 : s[1], s[2], m->aux, c);
            sched_push(m->immb ? sass_imm(in, m->imm) : in, ss, n, d, 1, FIXED_LAT);
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
            uint8_t ps[1] = { P0_PSEUDO };
            int cond = m->aux != 0;
            if (m->target < 0 && m->label < 0)
                break;
            if (!cond && falls_into(i, m->target))
                break;
            settle_fixed();
            int to = m->target >= 0 ? g_block_mop[m->target] : m->label;
            if (to <= i)
                drain();
            else
                branch_to(to);
            c = sched(cond ? ps : NULL, cond, 0, -1, 0, 0);
            sched_push(sass_bra(0, cond ? 0 : 7, m->aux == 2, c), cond ? ps : NULL, cond, -1, 0, 0);
            g_reached = cond;
            g_fix[g_nfix].at = g_np - 1;
            g_fix[g_nfix].block = m->target;
            g_fix[g_nfix++].mop = m->label;
            break;
        }
        case MK_OUTPUT:
            // the exit only needs where the colours live; probing stages them so nothing overwrites them first
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
            g_reached = 0;
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

// After allocation, mops reorder within a block so long as each register and memory space keeps its access order;
// nothing crosses a fence.

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
    // moved up, the second allocation would end the colours' ranges early and reuse their registers
    case MK_OUTPUT:
        return 1;
    default:
        return 0;
    }
}

enum { SP_NONE, SP_GLOBAL, SP_SHARED, SP_LOCAL, SP_ATTR, SP_COUNT };

// loads pass each other, and read-only loads pass stores too
static int mem_space(const struct mop *m, int *store)
{
    *store = 0;
    switch (m->k) {
    case MK_LDG:
        return m->readonly ? SP_NONE : SP_GLOBAL;
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

// rough guesses, only for deciding what to start early
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
    unsigned lat;      // when async
    struct edge *out;
    int nout, capout;
    int *producers;    // async nodes it reads
    int nprod, capprod;
    int npred;
    unsigned avail;
    unsigned height;   // longest latency chain to the block's end
    int consumed;      // async result already read
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

// sink is a register the following fence reads, such as a branch predicate, whose last write should start early
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
        // past MAX_LIVE unread async results another load would have to share a scoreboard
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
            int quiet = 0;  // a ready candidate needing no scoreboard
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
        uint64_t later[4] = { 0 };  // written later in the chosen order
        for (int p = n - 1; p >= 0; p--) {
            struct snode *nd = &nodes[order[lo + p] - lo];
            int free_rd = nd->fx.late && nd->fx.async;
            for (int a = 0; a < nd->fx.nrd && free_rd; a++) {
                int r = nd->fx.rd[a];
                free_rd = !(later[r / 64] >> r % 64 & 1);
                for (int k = 0; k < nd->fx.nwr && free_rd; k++)
                    free_rd = nd->fx.wr[k] != r;
            }
            g_rd_free_of[nd->mop] = (unsigned char)free_rd;
            for (int k = 0; k < nd->fx.nwr; k++)
                later[nd->fx.wr[k] / 64] |= 1ull << nd->fx.wr[k] % 64;
        }
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

// leaves each straight run's chosen order in order[]
static void emit_mops(enum nvk_stage stage, int *order)
{
    char *start = calloc(g_nmops + 1, 1);
    for (int i = 0; i < g_nmops; i++) {
        order[i] = i;
        if (g_mops[i].label >= 0)
            start[g_mops[i].label] = 1;
    }
    for (struct ir_block *b = g_fn->entry; b; b = b->next)
        if (!b->dead)
            start[g_block_mop[b->id]] = 1;
    // AIRC_NOSCHED=1 keeps program order, to tell a reordering bug from other wrong output
    int plain = getenv("AIRC_NOSCHED") != NULL;
    for (int lo = 0; lo < g_nmops;) {
        int hi = lo + 1;
        if (!is_fence(&g_mops[lo]))
            while (hi < g_nmops && !start[hi] && !is_fence(&g_mops[hi]))
                hi++;
        if (!plain && hi - lo > 1) {
            int f = hi < g_nmops && g_mops[hi].k == MK_BSSY ? hi + 1 : hi;
            int sink = f < g_nmops && g_mops[f].k == MK_BRA && g_mops[f].aux ? P0_PSEUDO : -1;
            schedule_segment(order, lo, hi, sink, stage);
        }
        unsigned at = g_np;
        arrive(lo);
        emit_mops_in(order + lo, hi - lo, stage);
        g_mop_np[lo] = at;
        lo = hi;
    }
    free(start);
}

// -1 when the values don't fit
static int allocate_and_emit(struct ir_func *f, const struct airc_info *info, int *order)
{
    memset(g_reserved, 0, sizeof(g_reserved));
    g_regcount = 2;
    for (int r = 0; r < g_nregions; r++)
        g_reserved[r][1] = 1;
    for (int k = 0; k < g_nregions; k++)
        if (regalloc(f, g_topo[k]) < 0)
            return -1;
    skip_empty_edges();
    if (g_win_end > g_regcount)
        g_regcount = g_win_end;
    g_scratch = g_regcount;
    g_stage_exit = airc_pass_off("AIRC_STAGEEXIT");
    if (g_stage_exit) {
        g_regcount += g_exit_regs > 4 ? g_exit_regs : 4;
        g_exit_tmp = g_regcount - 1;
    } else {
        // the cycle-breaking temporary sits past the colours in R0..
        g_exit_tmp = g_scratch > g_exit_regs ? g_scratch : g_exit_regs;
        g_regcount = g_exit_tmp + 1;
    }

    g_np = 0;
    g_nfix = 0;
    g_mem_bars = 0;
    g_cycle = 0;
    g_carried_waits = 0;
    g_reached = 1;
    memset(g_wbar_of, 0, sizeof(g_wbar_of));
    memset(g_rbar_of, 0, sizeof(g_rbar_of));
    memset(g_wclass, WC_NONE, sizeof(g_wclass));
    memset(g_bar_set, 0, sizeof(g_bar_set));
    g_bar_long = 0;
    g_long_op = 0;
    g_mop_np = calloc(g_nmops + 1, sizeof(*g_mop_np));
    g_rd_free_of = calloc(g_nmops + 1, 1);
    g_rd_free = 0;
    g_into = calloc(g_nmops + 1, sizeof(*g_into));
    g_block_np = calloc(f->nblocks, sizeof(*g_block_np));

    // R1 is the stack pointer and UR4 the memory descriptor LDG reads implicitly
    sched_push(sass_mov_cbuf(1, 0, 0x28, sched(NULL, 0, 0, 1, 1, 0)), NULL, 0, 1, 0, FIXED_LAT);
    sched_push(sass_uldc64_cbuf(4, 0, 0x118, sched(NULL, 0, 0, -1, 0, 0)), NULL, 0, UR_PSEUDO, 0, UNIFORM_LAT);
    for (int k = 0; k < g_nubase; k++)
        if (g_ubase[k].used)
            sched_push(sass_uldc64_cbuf(UR_FIRST + 2 * k, 0, (uint16_t)g_ubase[k].off, sched(NULL, 0, 0, -1, 0, 0)),
                       NULL, 0, UR_PSEUDO, 0, UNIFORM_LAT);
    if (g_demote)
        sched_push(sass_isetp(6, SASS_RZ, SASS_RZ, ISETP_NE, 0, sched(NULL, 0, 0, P6_PSEUDO, 1, 0)), NULL, 0, P6_PSEUDO,
                   0, PRED_LAT);

    emit_mops(info->stage, order);
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
    free(g_mop_np);
    free(g_rd_free_of);
    free(g_block_np);
    for (int i = 0; i <= g_nmops; i++)  // a dropped stub's may remain
        free(g_into[i]);
    free(g_into);
    return 0;
}

// whether every mop stayed within its straight run
static int in_runs(const int *order)
{
    char *start = calloc(g_nmops + 1, 1);
    for (int i = 0; i < g_nmops; i++)
        if (g_mops[i].label >= 0)
            start[g_mops[i].label] = 1;
    for (struct ir_block *b = g_fn->entry; b; b = b->next)
        if (!b->dead)
            start[g_block_mop[b->id]] = 1;
    int ok = 1;
    for (int lo = 0; lo < g_nmops && ok;) {
        int hi = lo + 1;
        if (!is_fence(&g_mops[lo]))
            while (hi < g_nmops && !start[hi] && !is_fence(&g_mops[hi]))
                hi++;
        for (int i = lo; i < hi && ok; i++)
            ok = order[i] >= lo && order[i] < hi;
        lo = hi;
    }
    free(start);
    return ok;
}

unsigned char *airc_back(struct ir_func *f, const struct airc_info *info, unsigned *size, unsigned *regs)
{
    g_info = (struct airc_info *)info;
    airc_widen_shared(f, info);
    guard_helper_loads(f);
    spill_values(f);
    g_nmops = 0;
    g_nvregs = 0;
    g_exit_regs = 0;
    g_demote = 0;

    memset(g_consts, 0xff, sizeof(g_consts));  // block -1, all free
    g_one = new_vreg(1, 1);
    struct mop *one = emit_mop(MK_MOVIMM);
    one->def = g_one;
    one->imm = 1;
    // a predecessor's copies may be lowered before the phi's block
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v && v->op == IR_PHI; v = v->next) {
            int m[2];
            v->vreg = v->type == IR_I64 ? new_cluster(2, 2, m) : new_vreg(1, 1);
        }
    g_fn = f;
    g_win_end = 0;
    find_readonly_loads(f);
    drop_overwritten_stores(f);
    loads_first(f);
    batch_loads(f);
    plan_regions(f);
    find_fused_compares(f);
    g_no_mods = airc_pass_off("AIRC_NOMODS");
    find_needed_mods(f);
    find_half_pairs(f);
    count_uses(f);
    find_fused_muls(f);
    find_folds(f);
    find_reductions(f);
    phi_outside(f);
    find_lop3(f);
    find_uniform_bases(f);
    g_noffsets = 0;
    g_ntexvecs = 0;
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
                     (v->op == IR_LDG || v->op == IR_LDS || v->op == IR_LDL || v->op == IR_TEX))
                g_vregs[v->vreg].ir = v->id;

    // A second allocation in the first schedule's order lets registers follow the early loads instead of blocking
    // them; kept unless it costs occupancy or doesn't fit.
    struct mop *mops = malloc(g_nmops * sizeof(*mops));
    memcpy(mops, g_mops, g_nmops * sizeof(*mops));
    struct vreg *vregs = malloc(g_nvregs * sizeof(*vregs));
    memcpy(vregs, g_vregs, g_nvregs * sizeof(*vregs));
    size_t nxfers = 0;
    for (int i = 0; i < g_nmops; i++)
        nxfers += g_mops[i].nxs + g_mops[i].nxd;
    struct xfer *xfers = malloc((nxfers + 1) * sizeof(*xfers));
    nxfers = 0;
    for (int i = 0; i < g_nmops; i++) {
        memcpy(xfers + nxfers, g_mops[i].xs, g_mops[i].nxs * sizeof(*xfers));
        nxfers += g_mops[i].nxs;
        memcpy(xfers + nxfers, g_mops[i].xd, g_mops[i].nxd * sizeof(*xfers));
        nxfers += g_mops[i].nxd;
    }
    int *order = malloc((g_nmops + 1) * sizeof(*order));
    g_second_pass = 0;
    if (allocate_and_emit(f, info, order) < 0) {
        free(mops);
        free(vregs);
        free(xfers);
        free(order);
        airc_fail("out of registers");
    }
    struct sass_insn *prog = g_prog;
    unsigned np = g_np, count = g_regcount;
    g_prog = NULL;
    g_prog_cap = 0;

    memcpy(g_mops, mops, g_nmops * sizeof(*mops));
    memcpy(g_vregs, vregs, g_nvregs * sizeof(*vregs));
    nxfers = 0;
    for (int i = 0; i < g_nmops; i++) {
        memcpy(g_mops[i].xs, xfers + nxfers, g_mops[i].nxs * sizeof(*xfers));
        nxfers += g_mops[i].nxs;
        memcpy(g_mops[i].xd, xfers + nxfers, g_mops[i].nxd * sizeof(*xfers));
        nxfers += g_mops[i].nxd;
    }
    int again = in_runs(order);
    for (int i = 0; again && i < g_nmops; i++)
        g_mops[i] = mops[order[i]];
    g_second_pass = 1;
    if (again && allocate_and_emit(f, info, order) == 0 && warps_at(g_regcount + 2) >= warps_at(count + 2)) {
        free(prog);
    } else {
        free(g_prog);
        g_prog = prog;
        g_np = np;
        g_regcount = count;
    }
    free(mops);
    free(vregs);
    free(xfers);
    free(order);

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
    // AIRC_SERIAL=1 waits on every scoreboard and stalls 15 cycles, to tell a scheduling hazard from other bugs
    if (getenv("AIRC_SERIAL"))
        for (unsigned i = 0; i < g_np; i++)
            g_prog[i].hi |= 0x3full << 52 | 0xfull << 41;
    sass_emit(g_prog, g_np, out + hdr);

    unsigned r = (g_regcount + 2 + 7) & ~7u;  // + 2 for the program counter, in 8-register granules
    *regs = r < 8 ? 8 : r;

    free(g_prog);
    g_prog = NULL;
    g_prog_cap = 0;
    return out;
}
