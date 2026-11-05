// IR clean-up before instruction selection: constant folding, constant branches, dead block and dead code
// removal, trivial phi elimination, run to a fixed point. Specialising a shader on its function constants
// turns most of its branches constant, so folding them away is what makes the big SkyLight shaders tractable.
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "airc_int.h"
#include "sass86.h"

// Replacing a value points it at its stand-in; its users are repointed in one sweep (resolve), not one sweep
// each, which is what dominates compile time on a big shader.
static unsigned g_pending;
static int g_no_range;  // AIRC_NORANGE=1: no folding that rests on a value's known upper bound
static int g_no_sat;    // AIRC_NOSAT=1: a clamp to [0, 1] stays a min, a max and an add
static int g_no_local_forward;

static int is_float(const struct ir_val *v, uint32_t bits)
{
    return v && v->op == IR_CONST && (uint32_t)v->imm == bits;
}
static uint64_t umax(const struct ir_val *v, int depth);
static struct ir_val *insert_before(struct ir_val *at, enum ir_op op, enum ir_type type, unsigned nsrc);

static struct ir_val *live(struct ir_val *v)
{
    while (v && v->fwd)
        v = v->fwd;
    return v;
}

static void replace_all(struct ir_func *f, struct ir_val *v, struct ir_val *with)
{
    (void)f;
    v->fwd = with;
    g_pending++;
}

static void resolve(struct ir_func *f)
{
    if (!g_pending)
        return;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *u = b->first; u; u = u->next)
            for (unsigned i = 0; i < u->nsrc; i++)
                u->src[i] = live(u->src[i]);
    g_pending = 0;
}

static float as_float(int64_t bits)
{
    float f;
    uint32_t u = (uint32_t)bits;
    memcpy(&f, &u, 4);
    return f;
}

static int64_t from_float(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// LLVM's real predicates: bit 0 equal, bit 1 greater, bit 2 less, bit 3 also when unordered
static int float_cmp(int pred, float a, float b)
{
    if (a != a || b != b)
        return (pred & 8) != 0;
    return ((pred & 1) && a == b) || ((pred & 2) && a > b) || ((pred & 4) && a < b);
}

static int int_cmp(int aux, int32_t a, int32_t b)
{
    int s = aux & IR_SIGNED;
    int lt = s ? a < b : (uint32_t)a < (uint32_t)b;
    switch (aux & 7) {
    case IR_EQ:
        return a == b;
    case IR_NE:
        return a != b;
    case IR_LT:
        return lt;
    case IR_LE:
        return lt || a == b;
    case IR_GT:
        return !lt && a != b;
    default:
        return !lt;
    }
}

static void make_const(struct ir_val *v, int64_t imm)
{
    v->op = IR_CONST;
    v->imm = imm;
    v->nsrc = 0;
}

static struct ir_val *new_const(struct ir_val *at, int64_t imm)
{
    struct ir_val *k = insert_before(at, IR_CONST, IR_I32, 0);
    k->imm = imm;
    return k;
}

// Integer sums and products gather their constants in the last term, where a shared-memory access takes it as its
// offset: (x + c1) * c2 is x * c2 + c1 * c2, a + (b + c) is (a + b) + c, and a constant in the first place moves
// to the second.
static int reassociate(struct ir_val *v)
{
    if ((v->op != IR_ADD && v->op != IR_MUL) || v->type != IR_I32)
        return 0;
    struct ir_val *x = v->src[0], *y = v->src[1];
    if (x->op == IR_CONST && y->op != IR_CONST) {
        v->src[0] = y;
        v->src[1] = x;
        return 1;
    }
    if (y->op != IR_CONST && v->op == IR_ADD && y->op == IR_ADD && y->type == IR_I32 && y->src[1]->op == IR_CONST) {
        struct ir_val *s = insert_before(v, IR_ADD, IR_I32, 2);
        s->src[0] = x;
        s->src[1] = y->src[0];
        v->src[0] = s;
        v->src[1] = y->src[1];
        return 1;
    }
    if (y->op != IR_CONST || (x->op != v->op && x->op != IR_ADD) || x->type != IR_I32 || x->src[1]->op != IR_CONST)
        return 0;
    uint32_t c1 = (uint32_t)x->src[1]->imm, c2 = (uint32_t)y->imm;
    if (x->op == v->op) {  // (a + c1) + c2, (a * c1) * c2
        v->src[0] = x->src[0];
        v->src[1] = new_const(v, v->op == IR_ADD ? c1 + c2 : c1 * c2);
    } else {  // (a + c1) * c2
        struct ir_val *m = insert_before(v, IR_MUL, IR_I32, 2);
        m->src[0] = x->src[0];
        m->src[1] = y;
        v->op = IR_ADD;
        v->src[0] = m;
        v->src[1] = new_const(v, c1 * c2);
    }
    return 1;
}

// fold v if its operands allow; 1 when something changed
static int fold(struct ir_func *f, struct ir_val *v)
{
    for (unsigned i = 0; i < v->nsrc; i++)
        v->src[i] = live(v->src[i]);
    struct ir_val *x = v->nsrc > 0 ? v->src[0] : NULL, *y = v->nsrc > 1 ? v->src[1] : NULL;
    if ((v->nsrc > 0 && !x) || (v->nsrc > 1 && !y))  // a shared or local access at a fixed offset
        return 0;
    if (reassociate(v))
        return 1;
    if ((v->op == IR_LDS || v->op == IR_STS) && x && x->op == IR_ADD && x->src[1]->op == IR_CONST) {
        int64_t imm = v->imm + (int32_t)x->src[1]->imm;
        if (imm >= 0 && imm < 1 << 20) {
            v->src[0] = x->src[0];
            v->imm = imm;
            return 1;
        }
    }
    if (v->op == IR_SEL && x->op == IR_CONST) {
        replace_all(f, v, x->imm ? v->src[1] : v->src[2]);
        return 1;
    }
    // LLVM's and and or that keep a poison operand from mattering: select c, x, false and select c, true, x
    if (v->op == IR_SEL && v->type == IR_PRED) {
        struct ir_val *z = v->src[2];
        if (z->op == IR_CONST && !z->imm) {
            v->op = IR_PAND;
            v->nsrc = 2;
            return 1;
        }
        if (y->op == IR_CONST && y->imm) {
            v->op = IR_POR;
            v->src[1] = z;
            v->nsrc = 2;
            return 1;
        }
    }
    if (v->op == IR_PNOT && x->op == IR_CONST) {
        make_const(v, !x->imm);
        return 1;
    }
    if ((v->op == IR_PAND || v->op == IR_POR) && x->op == IR_CONST && y->op == IR_CONST) {
        make_const(v, v->op == IR_PAND ? x->imm && y->imm : x->imm || y->imm);
        return 1;
    }
    if ((v->op == IR_POPC || v->op == IR_BREV || v->op == IR_FLO) && x->op == IR_CONST) {
        uint32_t a = (uint32_t)x->imm, r = 0;
        for (int i = 0; i < 32; i++)
            if (a >> i & 1)
                r = v->op == IR_POPC ? r + 1 : v->op == IR_BREV ? r | 1u << (31 - i) : (uint32_t)i;
        make_const(v, v->op == IR_FLO && !a ? 0xffffffffu : r);
        return 1;
    }
    if (v->op == IR_FFMA && v->type == IR_F32 && !v->aux && x->op == IR_CONST && y->op == IR_CONST &&
        v->src[2]->op == IR_CONST) {
        make_const(v, from_float(as_float(x->imm) * as_float(y->imm) + as_float(v->src[2]->imm)));
        return 1;
    }
    if (v->nsrc == 2 && x->op == IR_CONST && y->op == IR_CONST && !(v->op == IR_FADD && v->aux)) {
        int32_t a = (int32_t)x->imm, b = (int32_t)y->imm;
        float fa = as_float(x->imm), fb = as_float(y->imm);
        switch (v->op) {
        case IR_ADD:
            make_const(v, (uint32_t)(a + b));
            return 1;
        case IR_SUB:
            make_const(v, (uint32_t)(a - b));
            return 1;
        case IR_MUL:
            make_const(v, (uint32_t)a * (uint32_t)b);
            return 1;
        case IR_MULHI:
            make_const(v, v->aux ? (uint32_t)((int64_t)a * b >> 32)
                                 : (uint32_t)((uint64_t)(uint32_t)a * (uint32_t)b >> 32));
            return 1;
        case IR_AND:
            make_const(v, (uint32_t)(a & b));
            return 1;
        case IR_OR:
            if (v->type != x->type)  // a reinterpreting bitcast: keep it, it marks the change of type
                return 0;
            make_const(v, (uint32_t)(a | b));
            return 1;
        case IR_XOR:
            make_const(v, (uint32_t)(a ^ b));
            return 1;
        case IR_SHL:
            make_const(v, (uint32_t)a << (b & 31));
            return 1;
        case IR_SHR:
            make_const(v, (uint32_t)a >> (b & 31));
            return 1;
        case IR_SAR:
            make_const(v, (uint32_t)(a >> (b & 31)));
            return 1;
        case IR_MIN:
        case IR_MAX: {
            int lt = v->aux ? a < b : (uint32_t)a < (uint32_t)b;
            make_const(v, (uint32_t)((v->op == IR_MIN) == lt ? a : b));
            return 1;
        }
        case IR_FADD:
            if (v->aux & IR_FADD_SAT) {
                // as FADD.FTZ.SAT: a NaN or a denormal gives 0
                float sum = fa + fb;
                make_const(v, from_float(sum >= FLT_MIN ? fminf(sum, 1.0f) : 0.0f));
            } else {
                make_const(v, from_float(fa + fb));
            }
            return 1;
        case IR_FSUB:
            make_const(v, from_float(fa - fb));
            return 1;
        case IR_FMUL:
            make_const(v, from_float(fa * fb));
            return 1;
        case IR_FMIN:
            make_const(v, from_float(fminf(fa, fb)));
            return 1;
        case IR_FMAX:
            make_const(v, from_float(fmaxf(fa, fb)));
            return 1;
        case IR_CMP:
            make_const(v, v->aux & IR_FCMP ? float_cmp(v->aux & 15, fa, fb) : int_cmp(v->aux, a, b));
            return 1;
        default:
            break;
        }
    }
    // saturate(z) is min(max(z, 0), 1) + 0 (the sum flushes a denormal): one FADD.SAT
    if (!g_no_sat && v->op == IR_FADD && !v->aux && v->type == IR_F32 && is_float(y, 0) && x->op == IR_FMIN &&
        is_float(live(x->src[1]), 0x3f800000) && live(x->src[0])->op == IR_FMAX &&
        is_float(live(live(x->src[0])->src[1]), 0) && live(live(x->src[0])->src[0])->op != IR_CONST) {
        v->src[0] = live(live(x->src[0])->src[0]);
        v->aux = IR_FADD_SAT;
        return 1;
    }
    // an unsigned min that can't bite: the clamp on a local array's index after the index was clamped already
    if (!g_no_range && v->op == IR_MIN && !v->aux && v->type == IR_I32 && y->op == IR_CONST &&
        umax(x, 0) <= (uint32_t)y->imm) {
        replace_all(f, v, x);
        return 1;
    }
    // a mask of low bits that clears none the value can have: a conversion saturated to 255 packed as a byte
    if (!g_no_range && v->op == IR_AND && v->type == IR_I32 && y->op == IR_CONST &&
        !((uint32_t)y->imm & ((uint32_t)y->imm + 1)) && umax(x, 0) <= (uint32_t)y->imm) {
        replace_all(f, v, x);
        return 1;
    }
    // x op identity
    if (v->nsrc == 2 && y->op == IR_CONST && v->type == x->type) {
        int64_t c = y->imm;
        int zero_id = v->op == IR_ADD || v->op == IR_SUB || v->op == IR_OR || v->op == IR_XOR || v->op == IR_SHL ||
                      v->op == IR_SHR || v->op == IR_SAR;
        if ((zero_id && c == 0) || (v->op == IR_MUL && c == 1)) {
            replace_all(f, v, x);
            return 1;
        }
    }
    return 0;
}

static void visit(struct ir_block *b, struct ir_block **post, int *n, char *seen)
{
    if (seen[b->id])
        return;
    seen[b->id] = 1;
    for (unsigned i = 0; i < b->nsucc; i++)
        visit(b->succ[i], post, n, seen);
    post[(*n)++] = b;
}

static void add_pred(struct ir_block *to, struct ir_block *from)
{
    if (to->npreds == to->cappreds) {
        unsigned cap = to->cappreds ? to->cappreds * 2 : 4;
        struct ir_block **p = ir_alloc(cap * sizeof(*p));
        if (to->npreds)
            memcpy(p, to->preds, to->npreds * sizeof(*p));
        to->preds = p;
        to->cappreds = cap;
    }
    to->preds[to->npreds++] = from;
}

// Blocks in reverse postorder from the entry, then those of each subroutine a call reaches, each in reverse
// postorder from its own entry; unreachable ones dropped, predecessor lists rebuilt from the edges, phis matched
// to them again.
static void reorder(struct ir_func *f)
{
    struct ir_block **post = ir_alloc(f->nblocks * sizeof(*post));
    char *seen = ir_alloc(f->nblocks), queued[IR_MAX_SUBS] = { 0 };
    int n = 0, end[IR_MAX_SUBS + 1], order[IR_MAX_SUBS], norder = 0;
    for (int k = -1; k < norder; k++) {
        int from = n;
        visit(k < 0 ? f->entry : f->subs[order[k]], post, &n, seen);
        for (int i = from; i < n; i++)
            for (struct ir_val *v = post[i]->first; v; v = v->next)
                if (v->op == IR_CALL && !queued[v->imm]) {
                    queued[v->imm] = 1;
                    order[norder++] = (int)v->imm;
                }
        end[k + 1] = n;
    }
    for (struct ir_block *b = f->entry; b; b = b->next)
        b->dead = !seen[b->id];
    f->entry = f->last = NULL;
    for (int k = 0; k <= norder; k++)
        for (int i = end[k] - 1; i >= (k ? end[k - 1] : 0); i--) {
            struct ir_block *b = post[i];
            b->next = NULL;
            b->npreds = 0;
            b->func = k ? order[k - 1] + 1 : 0;
            if (f->last)
                f->last->next = b;
            else
                f->entry = b;
            f->last = b;
        }
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (unsigned i = 0; i < b->nsucc; i++)
            add_pred(b->succ[i], b);
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v && v->op == IR_PHI; v = v->next) {
            struct ir_val **src = ir_alloc(b->npreds * sizeof(*src));
            struct ir_block **from = ir_alloc(b->npreds * sizeof(*from));
            for (unsigned k = 0; k < b->npreds; k++) {
                unsigned j = 0;
                while (j < v->nsrc && v->phi_blocks[j] != b->preds[k])
                    j++;
                if (j == v->nsrc)
                    airc_fail("phi lost the input of a predecessor");
                src[k] = v->src[j];
                from[k] = b->preds[k];
            }
            v->src = src;
            v->phi_blocks = from;
            v->nsrc = b->npreds;
        }
}

// a conditional branch on a constant becomes an unconditional one; reorder() then rebuilds the edges
static int fold_branch(struct ir_block *b)
{
    struct ir_val *t = b->last;
    if (!t || t->op != IR_CBR || t->src[0]->op != IR_CONST)
        return 0;
    b->succ[0] = b->succ[t->src[0]->imm ? 0 : 1];
    b->nsucc = 1;
    t->op = IR_BR;
    t->nsrc = 0;
    return 1;
}

// `if (c) discard_fragment();`: a branch to a block that only discards and rejoins becomes a kill on c in line.
// The branch would split the warp with nothing to bring it back together, and a derivative after it would read
// quad neighbours that aren't there; a killed thread lives on as a helper, so the quad stays whole.
static int inline_kill(struct ir_block *b)
{
    struct ir_val *t = b->last;
    if (!t || t->op != IR_CBR)
        return 0;
    for (int k = 0; k < 2; k++) {
        struct ir_block *s = b->succ[k], *other = b->succ[!k];
        struct ir_val *v = s->first;
        if (s->npreds != 1 || !v || v->op != IR_KILL || v->nsrc || !v->next || v->next->op != IR_BR ||
            s->succ[0] != other)
            continue;
        struct ir_val *kill = ir_add(b, IR_KILL, IR_VOID, 1);
        kill->src[0] = t->src[0];
        kill->aux = k;  // the false edge kills when the condition is clear
        b->succ[0] = other;
        b->nsucc = 1;
        t->op = IR_BR;
        t->nsrc = 0;
        return 1;
    }
    return 0;
}

static int speculatable(const struct ir_val *v)
{
    switch (v->op) {
    case IR_CBUF:  // at a computed offset it may be out of the bank
        return v->nsrc == 0;
    case IR_CONST:
    case IR_UNDEF:
    case IR_S2R:
    case IR_IPA:
    case IR_DERIV:
    case IR_POPC:
    case IR_BREV:
    case IR_FLO:
    case IR_CMP:
    case IR_SEL:
    case IR_PNOT:
    case IR_PAND:
    case IR_POR:
    case IR_ADDR:
    case IR_ADDI:
    case IR_TEX:
    case IR_PROJ:
    case IR_TXQ:
    case IR_BR:
        return 1;
    default:
        return v->op >= IR_ADD && v->op <= IR_FRND;
    }
}

static int takes_derivatives(const struct ir_val *v)
{
    int lod = SASS_TEXF_LOD(IR_TEXA_FLAGS(v->aux));
    return v->op == IR_DERIV ||
           (v->op == IR_TEX && IR_TEXA_OP(v->aux) == SASS_TEX && lod != SASS_LOD_ZERO && lod != SASS_LOD_LOD);
}

// a block a branch of b's alone leads to, of nothing with an effect, going on to m; *deriv set if it takes
// derivatives
static int side_of(struct ir_block *s, struct ir_block *b, struct ir_block *m, int *deriv)
{
    if (s->npreds != 1 || s->preds[0] != b || s->nsucc != 1 || s->succ[0] != m || s == m)
        return 0;
    for (struct ir_val *v = s->first; v; v = v->next) {
        if (!speculatable(v))
            return 0;
        *deriv |= takes_derivatives(v);
    }
    return 1;
}

static void move_to(struct ir_block *s, struct ir_block *b)
{
    for (struct ir_val *v = s->first, *next; v; v = next) {
        next = v->next;
        if (v->op == IR_BR)
            continue;
        ir_remove(v);
        struct ir_val *at = b->last;
        v->block = b;
        v->next = at;
        v->prev = at->prev;
        if (at->prev)
            at->prev->next = v;
        else
            b->first = v;
        at->prev = v;
    }
    s->first = s->last = NULL;
    s->nsucc = 0;
}

// m, b's lone successor, goes on the end of b
static void splice(struct ir_block *b, struct ir_block *m)
{
    ir_remove(b->last);
    for (struct ir_val *v = m->first; v; v = v->next)
        v->block = b;
    m->first->prev = b->last;
    if (b->last)
        b->last->next = m->first;
    else
        b->first = m->first;
    b->last = m->last;
    b->nsucc = m->nsucc;
    for (unsigned i = 0; i < m->nsucc; i++) {
        b->succ[i] = m->succ[i];
        for (struct ir_val *p = m->succ[i]->first; p && p->op == IR_PHI; p = p->next)
            for (unsigned j = 0; j < p->nsrc; j++)
                if (p->phi_blocks[j] == m)
                    p->phi_blocks[j] = b;
    }
    m->first = m->last = NULL;
    m->nsucc = 0;
}

// A derivative (or an implicit-LOD sample) needs its whole quad, and a branch that splits the quad leaves it a
// neighbour short: an Apple GPU has its value anyway. So an if, or an if/else, around one, of side-effect-free
// code only, runs on both paths and selects the result. Where the paths met goes on the end, so an enclosing if
// sees one block.
static int flatten(struct ir_func *f, struct ir_block *b)
{
    struct ir_val *t = b->last;
    if (!t || t->op != IR_CBR || b->succ[0] == b->succ[1])
        return 0;
    struct ir_block *s[2] = { b->succ[0], b->succ[1] }, *m = NULL;
    int deriv = 0, side[2] = { 0, 0 };
    for (int k = 0; k < 2 && !m; k++) {  // if
        deriv = 0;
        if (side_of(s[k], b, s[!k], &deriv)) {
            m = s[!k];
            side[k] = 1;
        }
    }
    if (!m) {  // if/else
        deriv = 0;
        m = s[0]->nsucc == 1 ? s[0]->succ[0] : NULL;
        if (!m || !side_of(s[0], b, m, &deriv) || !side_of(s[1], b, m, &deriv))
            return 0;
        side[0] = side[1] = 1;
    }
    if (!deriv)
        return 0;
    for (int k = 0; k < 2; k++)
        if (side[k])
            move_to(s[k], b);
    int joined = m->npreds == 2 && m != f->entry;  // b alone reaches m now
    // each phi's inputs from the two paths become one from b
    for (struct ir_val *p = m->first, *next; p && p->op == IR_PHI; p = next) {
        next = p->next;
        struct ir_val *in[2] = { NULL, NULL };
        unsigned keep = 0;
        for (unsigned i = 0; i < p->nsrc; i++) {
            struct ir_block *from = p->phi_blocks[i];
            int k = from == b ? (side[0] ? 1 : 0) : from == s[0] && side[0] ? 0 : from == s[1] && side[1] ? 1 : -1;
            if (k < 0) {
                p->src[keep] = p->src[i];
                p->phi_blocks[keep++] = from;
            } else {
                in[k] = p->src[i];
            }
        }
        struct ir_val *sel = ir_add(b, IR_SEL, p->type, 3);
        sel->src[0] = t->src[0];
        sel->src[1] = in[0];
        sel->src[2] = in[1];
        if (joined) {
            resolve(f);
            for (struct ir_block *rb = f->entry; rb; rb = rb->next)
                for (struct ir_val *u = rb->first; u; u = u->next)
                    for (unsigned i = 0; i < u->nsrc; i++)
                        if (u->src[i] == p)
                            u->src[i] = sel;
            ir_remove(p);
            continue;
        }
        p->src[keep] = sel;
        p->phi_blocks[keep++] = b;
        p->nsrc = keep;
    }
    b->succ[0] = m;
    b->nsucc = 1;
    t->op = IR_BR;
    t->nsrc = 0;
    if (joined)
        splice(b, m);
    return 1;
}

static int has_effect(const struct ir_val *v)
{
    switch (v->op) {
    case IR_STG:
    case IR_STS:
    case IR_STL:
    case IR_SUST:
    case IR_AST:
    case IR_BAR:
    case IR_ATOM:
    case IR_KILL:
    case IR_BR:
    case IR_CBR:
    case IR_EXIT:
    case IR_CALL:
    case IR_RET:
        return 1;
    default:
        return 0;
    }
}

static int dce(struct ir_func *f)
{
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            v->uses = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v; v = v->next)
            for (unsigned i = 0; i < v->nsrc; i++)
                if (v->src[i])
                    v->src[i]->uses++;
    int changed = 0;
    // each block last to first, so a chain of dead values in it goes in one sweep
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->last; v;) {
            struct ir_val *prev = v->prev;
            if (!has_effect(v) && !v->uses) {
                for (unsigned i = 0; i < v->nsrc; i++)
                    if (v->src[i])
                        v->src[i]->uses--;
                ir_remove(v);
                changed = 1;
            }
            v = prev;
        }
    return changed;
}

// a phi all of whose inputs are one value (or itself) is that value
static int simplify_phis(struct ir_func *f)
{
    int changed = 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v && v->op == IR_PHI;) {
            struct ir_val *next = v->next, *same = NULL;
            int unique = 1;
            for (unsigned i = 0; i < v->nsrc && unique; i++) {
                struct ir_val *s = live(v->src[i]);
                if (s == v || s == same)
                    continue;
                if (same)
                    unique = 0;
                else
                    same = s;
            }
            if (unique && same) {
                replace_all(f, v, same);
                changed = 1;
            }
            v = next;
        }
    resolve(f);
    return changed;
}

// Common subexpressions: a value computed again where an equal one is already available (made in a block that
// dominates this one) is that one. Only values that depend on nothing but their sources: no texture sample, no
// derivative or lane exchange (which read other lanes, so the same sources give different values under another
// active mask), and a load only when nothing in the shader could write memory it reads. AIRC_NOCSE=1 turns it off.
static int cse_pure(const struct ir_val *v, int loads_pure)
{
    switch (v->op) {
    case IR_CONST:
    case IR_CBUF:
    case IR_S2R:
    case IR_IPA:
    case IR_POPC:
    case IR_BREV:
    case IR_FLO:
    case IR_CMP:
    case IR_SEL:
    case IR_PNOT:
    case IR_PAND:
    case IR_POR:
    case IR_ADDR:
    case IR_ADDI:
    case IR_PROJ:
        return 1;
    case IR_LDG:
        return loads_pure;
    default:
        return v->op >= IR_ADD && v->op <= IR_FRND;
    }
}

static unsigned cse_hash(const struct ir_val *v)
{
    uint64_t h = (uint64_t)v->op * 0x9e3779b97f4a7c15ull ^ (uint64_t)v->type << 8 ^ (uint64_t)(unsigned)v->aux << 16 ^
                 (uint64_t)v->imm * 0xff51afd7ed558ccdull;
    for (unsigned i = 0; i < v->nsrc; i++)
        h = (h ^ (uint64_t)(v->src[i] ? v->src[i]->id : -1)) * 0xc4ceb9fe1a85ec53ull;
    return (unsigned)(h >> 32);
}

static int cse_equal(const struct ir_val *a, const struct ir_val *b)
{
    if (a->op != b->op || a->type != b->type || a->aux != b->aux || a->imm != b->imm || a->nsrc != b->nsrc)
        return 0;
    for (unsigned i = 0; i < a->nsrc; i++)
        if (a->src[i] != b->src[i])
            return 0;
    return 1;
}

static int cse(struct ir_func *f)
{
    if (airc_pass_off("AIRC_NOCSE"))
        return 0;
    int nb = 0, maxid = 0, loads_pure = 1;
    for (struct ir_block *b = f->entry; b; b = b->next) {
        nb++;
        for (struct ir_val *v = b->first; v; v = v->next) {
            maxid = v->id > maxid ? v->id : maxid;
            if (v->op == IR_STG || v->op == IR_ATOM || v->op == IR_SUST || v->op == IR_BAR || v->op == IR_CALL)
                loads_pure = 0;
        }
    }
    // dominators (Cooper, Harvey, Kennedy) over the blocks in the order reorder() left them: reverse postorder
    // from each entry. idom and the preorder numbers are by position in that order.
    struct ir_block **blocks = malloc(nb * sizeof(*blocks));
    int *pos = calloc(f->nblocks, sizeof(*pos)), *idom = malloc(nb * sizeof(*idom)), *tin = malloc(nb * sizeof(*tin)),
        *tout = malloc(nb * sizeof(*tout)), *child = malloc(nb * sizeof(*child)), *sibling = malloc(nb * sizeof(*sibling)),
        *stack = malloc(nb * sizeof(*stack));
    int n = 0;
    for (struct ir_block *b = f->entry; b; b = b->next) {
        pos[b->id] = n;
        blocks[n++] = b;
    }
    for (int i = 0; i < nb; i++)
        idom[i] = -1;
    for (int changed = 1; changed;) {
        changed = 0;
        for (int i = 0; i < nb; i++) {
            struct ir_block *b = blocks[i];
            if (!b->npreds) {
                idom[i] = i;
                continue;
            }
            int cand = -1;
            for (unsigned p = 0; p < b->npreds; p++) {
                int q = pos[b->preds[p]->id];
                if (idom[q] < 0)
                    continue;
                if (cand < 0) {
                    cand = q;
                    continue;
                }
                while (q != cand) {
                    while (q > cand)
                        q = idom[q];
                    while (cand > q)
                        cand = idom[cand];
                }
            }
            if (cand != idom[i]) {
                idom[i] = cand;
                changed = 1;
            }
        }
    }
    for (int i = 0; i < nb; i++)
        child[i] = sibling[i] = -1;
    for (int i = nb - 1; i >= 0; i--)
        if (idom[i] != i) {
            sibling[i] = child[idom[i]];
            child[idom[i]] = i;
        }
    int clock = 0;
    for (int r = 0; r < nb; r++) {
        if (idom[r] != r)
            continue;
        int sp = 0;
        stack[sp++] = r;
        tin[r] = clock++;
        while (sp) {
            int u = stack[sp - 1];
            if (child[u] >= 0) {
                int c = child[u];
                child[u] = sibling[c];
                tin[c] = clock++;
                stack[sp++] = c;
            } else {
                tout[u] = clock++;
                sp--;
            }
        }
    }
    unsigned size = 64;
    while (size < 2u * (maxid + 1))
        size <<= 1;
    int *head = malloc(size * sizeof(*head)), *next = malloc((maxid + 1) * sizeof(*next));
    for (unsigned i = 0; i < size; i++)
        head[i] = -1;
    struct ir_val **by_id = calloc(maxid + 1, sizeof(*by_id));
    int changed = 0;
    for (int bi = 0; bi < nb; bi++)
        for (struct ir_val *v = blocks[bi]->first; v; v = v->next) {
            for (unsigned i = 0; i < v->nsrc; i++)
                v->src[i] = live(v->src[i]);
            if (!cse_pure(v, loads_pure))
                continue;
            unsigned h = cse_hash(v) & (size - 1);
            struct ir_val *same = NULL;
            for (int c = head[h]; c >= 0; c = next[c]) {
                struct ir_val *w = by_id[c];
                int wb = pos[w->block->id];
                if (cse_equal(v, w) && tin[wb] <= tin[bi] && tout[bi] <= tout[wb]) {
                    same = w;
                    break;
                }
            }
            if (same) {
                replace_all(f, v, same);
                changed = 1;
            } else {
                next[v->id] = head[h];
                head[h] = v->id;
                by_id[v->id] = v;
            }
        }
    resolve(f);
    free(by_id);
    free(next);
    free(head);
    free(stack);
    free(sibling);
    free(child);
    free(tout);
    free(tin);
    free(idom);
    free(pos);
    free(blocks);
    return changed;
}

// The most a float value can be, INFINITY when nothing is known: a clamp's top, scaled. A value with a bound is
// never a NaN, which the min's needs (a min with a NaN is its other side). Rounding is monotonic, so x <= a gives
// x * k <= a * k for k > 0; not for k = 0, as -inf * 0 is a NaN.
static float fmax_of(const struct ir_val *v, int depth)
{
    if (!v || v->type != IR_F32 || depth > 8)
        return INFINITY;
    float a, b;
    switch (v->op) {
    case IR_CONST:
        a = as_float(v->imm);
        return a == a ? a : INFINITY;
    case IR_FMIN:
        a = fmax_of(v->src[0], depth + 1);
        b = fmax_of(v->src[1], depth + 1);
        return a < b ? a : b;
    case IR_FADD:
        return v->aux & IR_FADD_SAT ? 1.0f : INFINITY;
    case IR_FMUL:
        for (int i = 0; i < 2; i++) {
            const struct ir_val *k = v->src[i];
            if (k->op == IR_CONST && as_float(k->imm) > 0 && as_float(k->imm) < INFINITY)
                return fmax_of(v->src[!i], depth + 1) * as_float(k->imm);
        }
        return INFINITY;
    default:
        return INFINITY;
    }
}

// The most a 32-bit integer value can be, unsigned; UINT32_MAX when nothing is known.
static uint64_t umax(const struct ir_val *v, int depth)
{
    if (!v || v->type != IR_I32 || depth > 8)
        return UINT32_MAX;
    uint64_t a, b;
    switch (v->op) {
    case IR_CONST:
        return (uint32_t)v->imm;
    case IR_MIN:
        a = umax(v->src[0], depth + 1);
        b = umax(v->src[1], depth + 1);
        return v->aux ? UINT32_MAX : a < b ? a : b;
    case IR_AND:
        a = umax(v->src[0], depth + 1);
        b = umax(v->src[1], depth + 1);
        return a < b ? a : b;
    case IR_MUL:
        a = umax(v->src[0], depth + 1) * umax(v->src[1], depth + 1);
        return a > UINT32_MAX ? UINT32_MAX : a;
    case IR_ADD:
        a = umax(v->src[0], depth + 1) + umax(v->src[1], depth + 1);
        return a > UINT32_MAX ? UINT32_MAX : a;
    case IR_SHR:
        return v->src[1]->op == IR_CONST ? umax(v->src[0], depth + 1) >> (v->src[1]->imm & 31) : UINT32_MAX;
    case IR_LDG:
    case IR_LDS:
    case IR_LDL:  // a byte or halfword is loaded zero-extended (LDG.U8, .U16)
        return v->aux == 1 ? 0xff : v->aux == 2 ? 0xffff : UINT32_MAX;
    case IR_F2I: {  // F2I.U32.TRUNC: a NaN or anything below 0 is 0
        float f = fmax_of(v->src[0], depth + 1);
        if (v->aux || !(f < 4294967296.0f))
            return UINT32_MAX;
        return f < 0 ? 0 : (uint64_t)f;
    }
    default:
        return UINT32_MAX;
    }
}

// v made in b in front of at
static struct ir_val *insert_before(struct ir_val *at, enum ir_op op, enum ir_type type, unsigned nsrc)
{
    struct ir_val *v = ir_add(at->block, op, type, nsrc);
    ir_remove(v);
    v->next = at;
    v->prev = at->prev;
    if (at->prev)
        at->prev->next = v;
    else
        at->block->first = v;
    at->prev = v;
    return v;
}

// A local array a shader fills from read-only memory (a struct passed by value, copied to the stack and then
// indexed) costs a load, a store and the thread's local memory for each word, where reading the memory itself costs
// a load a use. This applies when the shader's first blocks (a run of them with no branch out or in) copy words
// into local memory and nothing else ever writes it, nothing in the shader writes memory a load could see, and
// every local load lands on a word the copy
// filled (an index the shader clamps is bounded by what the clamp says). The loads then read the source, and the
// copy and the local memory go. AIRC_NOCONSTLOCAL=1 turns it off.
int airc_const_locals(struct ir_func *f, struct airc_info *info)
{
    if (airc_pass_off("AIRC_NOCONSTLOCAL") || !info->local_size)
        return 0;
    struct ir_val *base = NULL;
    int64_t delta = 0, first = 1;
    unsigned nstl = 0;
    uint8_t *filled = calloc(info->local_size / 4 + 1, 1), *straight = calloc(f->nblocks, 1);
    int ok = filled && straight;
    int seen_ldl = 0;
    // every other block comes after the last of these, and so after the copy
    for (struct ir_block *b = f->entry; ok && b;) {
        straight[b->id] = 1;
        b = b->nsucc == 1 && b->succ[0]->npreds == 1 ? b->succ[0] : NULL;
    }
    for (struct ir_block *b = f->entry; b && ok; b = b->next)
        for (struct ir_val *v = b->first; v && ok; v = v->next) {
            switch (v->op) {
            case IR_STG:
            case IR_ATOM:
            case IR_SUST:
            case IR_BAR:
            case IR_CALL:
            case IR_STS:
                ok = 0;
                break;
            case IR_STL: {
                struct ir_val *src = v->src[1];
                if (!straight[b->id] || seen_ldl || v->src[0] || v->aux != 4 || !src || src->op != IR_LDG ||
                    src->aux != 4 || (v->imm & 3) || v->imm < 0 || v->imm / 4 >= info->local_size / 4 + 1 ||
                    filled[v->imm / 4]) {
                    ok = 0;
                    break;
                }
                int64_t d = src->imm - v->imm;
                if (first) {
                    base = src->src[0];
                    delta = d;
                    first = 0;
                } else if (src->src[0] != base || d != delta) {
                    ok = 0;
                }
                filled[v->imm / 4] = 1;
                nstl++;
                break;
            }
            case IR_LDL: {
                struct ir_val *idx = v->nsrc ? v->src[0] : NULL;
                seen_ldl = 1;
                if (v->aux != 4 || (v->imm & 3) || v->imm < 0) {
                    ok = 0;
                    break;
                }
                for (int64_t at = v->imm; at < v->imm + 4 + (idx ? (int64_t)umax(idx, 0) : 0) && ok; at += 4)
                    if (at / 4 >= info->local_size / 4 + 1 || !filled[at / 4])
                        ok = 0;
                break;
            }
            default:
                break;
            }
        }
    free(filled);
    free(straight);
    if (!ok || !nstl || !base)
        return 0;
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v;) {
            struct ir_val *next = v->next;
            if (v->op == IR_STL) {
                ir_remove(v);
            } else if (v->op == IR_LDL) {
                struct ir_val *idx = v->nsrc ? v->src[0] : NULL;
                struct ir_val *addr = base;
                if (idx) {
                    addr = insert_before(v, IR_ADDR, IR_I64, 2);
                    addr->src[0] = base;
                    addr->src[1] = idx;
                    addr->imm = 1;
                }
                v->op = IR_LDG;
                v->nsrc = 1;
                v->src = ir_alloc(sizeof(*v->src));
                v->src[0] = addr;
                v->imm += delta;
            }
            v = next;
        }
    info->local_size = 0;
    return 1;
}

// Local memory at fixed addresses, within a block: a load of a word stored there earlier is the value stored, and a
// store overwritten before anything could read it goes. Code built without optimisation (Apple's GL compiler's)
// changes a vector in memory a lane at a time, loading all of it and storing all of it back each time. A store of
// an undefined value goes too. AIRC_NOLOCALFWD=1 turns it off.
struct local_word {
    int64_t at;
    int bytes;
    struct ir_val *val;    // what the bytes hold, as far as this block knows
    struct ir_val *store;  // the store that put it there, while nothing could have read it since
};

static int local_forward(struct ir_func *f)
{
    if (g_no_local_forward)
        return 0;
    int changed = 0;
    struct local_word *w = NULL;
    unsigned cap = 0;
    for (struct ir_block *b = f->entry; b; b = b->next) {
        unsigned n = 0;
        for (struct ir_val *v = b->first, *next; v; v = next) {
            next = v->next;
            int fixed_store = v->op == IR_STL && v->nsrc == 2 && !v->src[0];
            int fixed_load = v->op == IR_LDL && !v->nsrc;
            if (v->op == IR_CALL || (v->op == IR_LDL && !fixed_load)) {
                for (unsigned i = 0; i < n; i++)
                    w[i].store = NULL;
                if (v->op == IR_CALL)
                    n = 0;
                continue;
            }
            if (v->op == IR_STL && !fixed_store) {
                for (unsigned i = 0; i < n; i++)
                    w[i].val = NULL;
                continue;
            }
            if (!fixed_store && !fixed_load)
                continue;
            unsigned same = n;
            for (unsigned i = 0; i < n; i++) {
                if (w[i].at == v->imm && w[i].bytes == v->aux) {
                    same = i;
                } else if (w[i].at < v->imm + v->aux && v->imm < w[i].at + w[i].bytes) {
                    w[i].store = NULL;
                    if (fixed_store)
                        w[i].val = NULL;
                }
            }
            if (same == n) {
                if (n == cap) {
                    cap = cap ? 2 * cap : 64;
                    w = realloc(w, cap * sizeof(*w));
                }
                w[n++] = (struct local_word){ v->imm, v->aux, fixed_load ? v : NULL, NULL };
                if (fixed_load)
                    continue;
            }
            if (fixed_load) {
                struct ir_val *val = live(w[same].val);
                w[same].store = NULL;
                if (val && v->aux == 4 && val->type == v->type) {
                    replace_all(f, v, val);
                    changed = 1;
                } else if (!val) {
                    w[same].val = v;
                }
                continue;
            }
            if (w[same].val && live(w[same].val) == live(v->src[1])) {  // what's there already
                ir_remove(v);
                changed = 1;
                continue;
            }
            if (w[same].store) {
                ir_remove(w[same].store);
                changed = 1;
            }
            w[same].val = v->src[1];
            w[same].store = v;
            if (v->src[1]->op == IR_UNDEF) {
                ir_remove(v);
                w[same].store = NULL;
                changed = 1;
            }
        }
    }
    free(w);
    resolve(f);  // before dce: a load another is now read as has no readers of its own yet
    return changed;
}

// What a 32-bit value is known to be a multiple of, as a power of two up to 16 (4). The kernel's threadgroup
// variables are laid out 16-byte aligned from 0, and the dispatcher puts each [[threadgroup(n)]] buffer after them
// aligned as much.
static int known_align(const struct ir_val *v, const struct airc_info *info, int depth)
{
    if (!v)  // a shared access at a fixed address
        return 4;
    if (depth > 8 || v->type != IR_I32)
        return 0;
    int a, b;
    switch (v->op) {
    case IR_CONST:
        return v->imm & 15 ? __builtin_ctz((uint32_t)v->imm) : 4;
    case IR_CBUF:
        if (v->nsrc)
            return 0;
        for (unsigned i = 0; i < info->nargs; i++)
            if (info->args[i].kind == NVK_ARG_THREADGROUP && info->args[i].cbuf_off == v->imm)
                return 4;
        return 0;
    case IR_ADD:
    case IR_SUB:
    case IR_OR:
    case IR_XOR:
        a = known_align(v->src[0], info, depth + 1);
        b = known_align(v->src[1], info, depth + 1);
        return a < b ? a : b;
    case IR_AND:
        a = known_align(v->src[0], info, depth + 1);
        b = known_align(v->src[1], info, depth + 1);
        return a > b ? a : b;
    case IR_MUL:
        a = known_align(v->src[0], info, depth + 1) + known_align(v->src[1], info, depth + 1);
        return a < 4 ? a : 4;
    case IR_SHL:
        if (v->src[1]->op != IR_CONST)
            return 0;
        a = known_align(v->src[0], info, depth + 1) + (int)(v->src[1]->imm & 31);
        return a < 4 ? a : 4;
    case IR_SEL:
        a = known_align(v->src[1], info, depth + 1);
        b = known_align(v->src[2], info, depth + 1);
        return a < b ? a : b;
    default:
        return 0;
    }
}

static struct ir_val *shared_base(const struct ir_val *v)
{
    return v->op == IR_LDS && !v->nsrc ? NULL : v->src[0];
}

static int shared_word(const struct ir_val *v, enum ir_op op)
{
    if (v->op != op || v->aux != 4)
        return 0;
    const struct ir_val *w = op == IR_LDS ? v : v->src[1];
    return w->type == IR_I32 || w->type == IR_F32;
}

// what the lanes of a shared load (op IR_LDS) or store may not be moved across
static int fence(const struct ir_val *x, enum ir_op op)
{
    if (op == IR_STS)
        return x->op == IR_LDS || (has_effect(x) && x->op != IR_STS);
    return has_effect(x);
}

// v and the words after it, n of them in all, as one access if they're there: 1 if they were
static int widen(struct ir_val *v, int n, const struct airc_info *info)
{
    enum ir_op op = v->op;
    struct ir_val *base = shared_base(v), *lane[4] = { NULL }, *first = NULL, *last = NULL, *start = v;
    int at = v->imm & 15 ? __builtin_ctz((uint32_t)v->imm) : 4, align = known_align(base, info, 0);
    if ((at < align ? at : align) < (n == 4 ? 4 : 3))
        return 0;
    while (start->prev && !fence(start->prev, op))
        start = start->prev;
    for (struct ir_val *x = start; x && !fence(x, op); x = x->next) {
        int64_t k = x->imm - v->imm;
        if (!shared_word(x, op) || shared_base(x) != base || k < 0 || k >= 4 * n || (k & 3) || lane[k / 4])
            continue;
        lane[k / 4] = x;
        if (!first)
            first = x;
        last = x;
    }
    for (int k = 0; k < n; k++)
        if (!lane[k])
            return 0;
    // The lanes meet at the first load or the last store, passing other accesses of the kind only where those are
    // to other words off the same base: shared atomics are plain LDS and STS, and two on one word keep their order.
    for (struct ir_val *x = first; x != last; x = x->next) {
        int member = 0;
        for (int k = 0; k < n; k++)
            member |= x == lane[k];
        if (x->op == op && !member &&
            (shared_base(x) != base || (x->imm + x->aux > v->imm && x->imm < v->imm + 4 * n)))
            return 0;
    }
    if (op == IR_LDS) {
        struct ir_val *w = insert_before(first, IR_LDS, IR_VOID, base ? 1 : 0);
        if (base)
            w->src[0] = base;
        w->imm = v->imm;
        w->aux = 4 * n;
        for (int k = 0; k < n; k++) {
            struct ir_val *x = lane[k];
            if (!x->nsrc)
                x->src = ir_alloc(sizeof(*x->src));
            x->op = IR_PROJ;
            x->nsrc = 1;
            x->src[0] = w;
            x->imm = k;
            x->aux = 0;
        }
        return 1;
    }
    struct ir_val *w = insert_before(last, IR_STS, IR_VOID, 1 + n);
    w->src[0] = base;
    for (int k = 0; k < n; k++)
        w->src[1 + k] = lane[k]->src[1];
    w->imm = v->imm;
    w->aux = 4 * n;
    for (int k = 0; k < n; k++)
        ir_remove(lane[k]);
    return 1;
}

// Words of threadgroup memory read or written one at a time side by side become one LDS or STS .64 or .128 (MPS's
// convolutions read their filters a float4 at a time as four floats). Sound because the lanes are one base value
// plus constant offsets a word apart, so they're the very words the wide access moves; its address is proven a
// multiple of its size, as a misaligned one faults; and the lanes move past no other access of their kind but to
// other words off the same base, nothing else with an effect (a store, atomic, barrier, fence, call or discard)
// and, for stores, no shared load, so gathering them changes nothing another access sees. Quads go first, so a
// pair doesn't take half of one.
// Device memory stays as it is: a buffer is only bound aligned to its element type, so nothing proves an address
// that a scalar access uses aligned to more.
void airc_widen_shared(struct ir_func *f, const struct airc_info *info)
{
    for (int n = 4; n >= 2; n -= 2)
        for (struct ir_block *b = f->entry; b; b = b->next)
            for (struct ir_val *v = b->first; v;)
                v = (shared_word(v, IR_LDS) || shared_word(v, IR_STS)) && widen(v, n, info) ? b->first : v->next;
}

void airc_opt(struct ir_func *f)
{
    g_no_range = airc_pass_off("AIRC_NORANGE");
    g_no_sat = airc_pass_off("AIRC_NOSAT");
    g_no_local_forward = airc_pass_off("AIRC_NOLOCALFWD");
    // right after the front end each phi's inputs follow its block's predecessor order
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (struct ir_val *v = b->first; v && v->op == IR_PHI; v = v->next) {
            v->phi_blocks = ir_alloc(b->npreds * sizeof(*v->phi_blocks));
            memcpy(v->phi_blocks, b->preds, b->npreds * sizeof(*v->phi_blocks));
        }
    for (;;) {
        int changed = 0;
        reorder(f);
        int reshaped = 0;  // the predecessor lists are stale until the next reorder
        for (struct ir_block *b = f->entry; b; b = b->next) {
            for (struct ir_val *v = b->first; v; v = v->next)
                changed |= fold(f, v);
            if (!reshaped)
                reshaped = fold_branch(b) || inline_kill(b) || flatten(f, b);
        }
        changed |= reshaped;
        resolve(f);
        changed |= simplify_phis(f);
        changed |= cse(f);
        changed |= local_forward(f);
        changed |= dce(f);
        if (!changed)
            break;
    }
}

// A value made early and read a long way on holds its register all that way, and in a big shader (CoreAnimation's
// uber shader: its configuration read up front, each field used a thousand values later) those registers are most
// of what it needs. Values that are cheap to make again are made again in the blocks that read them instead.

#define REMAT_FAR 64      // values between a definition and a read in another block before it is made again
#define REMAT_SIZE 4      // values a copy makes at most
#define REMAT_VALUES 45000  // what a shader and its copies may come to: past it the back end has too many values

static int g_loads_pure;  // nothing in the function writes global memory, so a load gives the same each time

// the values making v again takes, with its inputs that are cheap too; -1 if v can't be made again. An input that
// can't stays as it is and live (an anchor), counted in *anchors.
static int remat_size(const struct ir_val *v, int depth, int *anchors)
{
    int size = 1;
    switch (v->op) {
    case IR_CBUF:
        return v->nsrc ? -1 : 1;
    case IR_IPA:
        return 1;
    case IR_LDG:
        break;
    case IR_PROJ:
        if (v->src[0]->op != IR_LDG || !g_loads_pure)
            return -1;
        return 0;  // the load it reads is made once a block, whichever lanes are read
    default:
        if (v->op < IR_ADD || v->op > IR_FRND || v->op == IR_MUFU)
            return -1;
        break;
    }
    if (v->op == IR_LDG && !g_loads_pure)
        return -1;
    if (!depth)
        return -1;
    for (unsigned i = 0; i < v->nsrc; i++) {
        const struct ir_val *s = v->src[i];
        if (s->op == IR_CONST || s->op == IR_UNDEF)
            continue;
        int n = remat_size(s, depth - 1, anchors);
        if (n < 0)
            (*anchors)++;
        else
            size += n;
    }
    return size;
}

struct remat_copy {
    int val, block;
    struct ir_val *copy;
};

static struct remat_copy *g_copies;
static unsigned g_ncopies, g_cap_copies;

static struct ir_val *find_copy(const struct ir_val *v, const struct ir_block *b)
{
    for (unsigned i = g_ncopies; i--;)
        if (g_copies[i].val == v->id && g_copies[i].block == b->id)
            return g_copies[i].copy;
    return NULL;
}

// v made again in b, in front of at (the value that reads it, or b's last) together with what it takes
static struct ir_val *copy_into(struct ir_val *v, struct ir_block *b, struct ir_val *at)
{
    struct ir_val *c = find_copy(v, b);
    if (c)
        return c;
    struct ir_val *src[3];
    for (unsigned i = 0; i < v->nsrc; i++) {
        int anchors = 0;
        src[i] = v->src[i];
        if (src[i]->op != IR_CONST && src[i]->op != IR_UNDEF && remat_size(src[i], REMAT_SIZE, &anchors) >= 0)
            src[i] = copy_into(src[i], b, at);
    }
    c = ir_add(b, v->op, v->type, v->nsrc);
    c->aux = v->aux;
    c->imm = v->imm;
    memcpy(c->src, src, v->nsrc * sizeof(*src));
    // ir_add appends (before the terminator); move it in front of at
    ir_remove(c);
    c->next = at;
    c->prev = at->prev;
    if (at->prev)
        at->prev->next = c;
    else
        b->first = c;
    at->prev = c;
    if (g_ncopies == g_cap_copies) {
        g_cap_copies = g_cap_copies ? 2 * g_cap_copies : 1024;
        g_copies = realloc(g_copies, g_cap_copies * sizeof(*g_copies));
    }
    g_copies[g_ncopies++] = (struct remat_copy){ v->id, b->id, c };
    return c;
}

void airc_remat(struct ir_func *f)
{
    int maxid = 0, nb = 0;
    g_loads_pure = 1;
    for (struct ir_block *b = f->entry; b; b = b->next) {
        nb = b->id >= nb ? b->id + 1 : nb;
        for (struct ir_val *v = b->first; v; v = v->next) {
            maxid = v->id > maxid ? v->id : maxid;
            if (v->op == IR_STG || v->op == IR_ATOM || v->op == IR_SUST)
                g_loads_pure = 0;
        }
    }
    int *pos = calloc(maxid + 1, sizeof(*pos)), *index = calloc(nb, sizeof(*index)), *depth = calloc(nb, sizeof(*depth));
    int n = 0, nblocks = 0;
    for (struct ir_block *b = f->entry; b; b = b->next) {
        if (b->dead)
            continue;
        index[b->id] = nblocks++;
        for (struct ir_val *v = b->first; v; v = v->next)
            pos[v->id] = n++;
    }
    // a loop is the blocks from its header to its last back edge in layout order
    for (struct ir_block *b = f->entry; b; b = b->next)
        for (unsigned i = 0; i < b->nsucc && !b->dead; i++)
            if (b->succ[i]->func == b->func && index[b->succ[i]->id] <= index[b->id])
                for (struct ir_block *m = f->entry; m; m = m->next)
                    if (!m->dead && index[m->id] >= index[b->succ[i]->id] && index[m->id] <= index[b->id])
                        depth[m->id]++;
    g_ncopies = 0;
    unsigned budget = n / 8;
    if (n + budget > REMAT_VALUES)
        budget = n < REMAT_VALUES ? REMAT_VALUES - n : 0;
    for (struct ir_block *b = f->entry; b; b = b->next) {
        if (b->dead)
            continue;
        for (struct ir_val *u = b->first; u; u = u->next) {
            if (u->id > maxid || g_ncopies >= budget)  // one of the copies, or all that may be made
                continue;
            for (unsigned i = 0; i < u->nsrc; i++) {
                struct ir_val *s = u->src[i];
                struct ir_block *at_block = u->op == IR_PHI ? u->phi_blocks[i] : b;
                if (!s || s->block == at_block || s->id > maxid || pos[u->id] - pos[s->id] < REMAT_FAR ||
                    depth[at_block->id] > depth[s->block->id])
                    continue;
                int anchors = 0, size = remat_size(s, REMAT_SIZE, &anchors);
                if (size < 1 || size > REMAT_SIZE || anchors > 1)
                    continue;
                u->src[i] = copy_into(s, at_block, u->op == IR_PHI ? at_block->last : u);
            }
        }
    }
    free(pos);
    free(index);
    free(depth);
    dce(f);
}
