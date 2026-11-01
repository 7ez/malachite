#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ir.h"

// chunks of a bump allocator, freed together by ir_reset
static struct chunk {
    struct chunk *next;
    unsigned used;
    unsigned char data[1 << 20];
} *g_chunks;

static int g_next_id;

void ir_reset(void)
{
    g_next_id = 0;
    while (g_chunks) {
        struct chunk *c = g_chunks;
        g_chunks = c->next;
        free(c);
    }
}

void *ir_alloc(unsigned bytes)
{
    bytes = (bytes + 15) & ~15u;
    if (!g_chunks || g_chunks->used + bytes > sizeof(g_chunks->data)) {
        struct chunk *c = malloc(sizeof(*c));
        if (!c || bytes > sizeof(c->data))
            abort();
        c->next = g_chunks;
        c->used = 0;
        g_chunks = c;
    }
    void *p = g_chunks->data + g_chunks->used;
    g_chunks->used += bytes;
    memset(p, 0, bytes);
    return p;
}

struct ir_block *ir_block(struct ir_func *f)
{
    struct ir_block *b = ir_alloc(sizeof(*b));
    b->id = f->nblocks++;
    if (f->last)
        f->last->next = b;
    else
        f->entry = b;
    f->last = b;
    return b;
}

static int is_terminator(enum ir_op op)
{
    return op == IR_BR || op == IR_CBR || op == IR_EXIT || op == IR_RET;
}

struct ir_val *ir_add(struct ir_block *b, enum ir_op op, enum ir_type type, unsigned nsrc)
{
    struct ir_val *v = ir_alloc(sizeof(*v));
    v->op = op;
    v->type = type;
    v->nsrc = nsrc;
    v->src = nsrc ? ir_alloc(nsrc * sizeof(*v->src)) : NULL;
    v->block = b;
    v->id = g_next_id++;
    v->vreg = -1;
    struct ir_val *at = b->last && is_terminator(b->last->op) && !is_terminator(op) ? b->last : NULL;
    if (op == IR_PHI)  // phis lead the block, in no order that matters: the front, which costs nothing
        at = b->first;
    if (at) {
        v->next = at;
        v->prev = at->prev;
        if (at->prev)
            at->prev->next = v;
        else
            b->first = v;
        at->prev = v;
    } else {
        v->prev = b->last;
        if (b->last)
            b->last->next = v;
        else
            b->first = v;
        b->last = v;
    }
    return v;
}

struct ir_val *ir_op2(struct ir_block *b, enum ir_op op, enum ir_type type, struct ir_val *x, struct ir_val *y)
{
    struct ir_val *v = ir_add(b, op, type, 2);
    v->src[0] = x;
    v->src[1] = y;
    return v;
}

struct ir_val *ir_const(struct ir_block *b, enum ir_type type, int64_t imm)
{
    struct ir_val *v = ir_add(b, IR_CONST, type, 0);
    v->imm = imm;
    return v;
}

void ir_edge(struct ir_block *from, struct ir_block *to)
{
    from->succ[from->nsucc++] = to;
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

void ir_remove(struct ir_val *v)
{
    struct ir_block *b = v->block;
    if (v->prev)
        v->prev->next = v->next;
    else
        b->first = v->next;
    if (v->next)
        v->next->prev = v->prev;
    else
        b->last = v->prev;
}

static const char *const g_names[] = {
    "const", "undef", "cbuf", "s2r",  "ald",  "ipa",  "isberd", "add",  "sub",  "mul",  "mulhi", "and",  "or",   "xor",
    "shl",   "shr",   "sar",  "min",  "max",  "fadd", "fsub", "fmul", "ffma", "fmin", "fmax", "fneg",
    "fabs",  "mufu",  "i2f",  "f2i",  "f2h",  "h2f",  "frnd", "deriv", "shfl", "popc", "brev", "flo", "vote", "cmp",  "sel",  "pnot", "pand", "por",  "addr",
    "addi",  "ldg",   "stg",  "lds",  "sts",  "ldl",  "stl",  "bar",  "atom", "tex",  "proj", "txq",  "sust", "ast",  "phi",  "kill",
    "br",    "cbr",   "exit", "param", "call", "ret",
};

void ir_print(const struct ir_func *f)
{
    static const char *const types[] = { "", "p", "i32", "f32", "f16", "i64" };
    for (struct ir_block *b = f->entry; b; b = b->next) {
        if (b->dead)
            continue;
        if (b->func && f->subs[b->func - 1] == b)
            printf("sub%d:\n", b->func - 1);
        printf("b%d:", b->id);
        for (unsigned i = 0; i < b->npreds; i++)
            printf(" %sb%d", i ? "" : "<- ", b->preds[i]->id);
        printf("\n");
        for (struct ir_val *v = b->first; v; v = v->next) {
            printf("  ");
            if (v->type != IR_VOID)
                printf("%%%d:%s = ", v->id, types[v->type]);
            printf("%s", g_names[v->op]);
            for (unsigned i = 0; i < v->nsrc; i++)
                printf(" %%%d", v->src[i] ? v->src[i]->id : -1);
            if (v->imm || v->op == IR_CONST)
                printf(" #%lld", (long long)v->imm);
            if (v->aux)
                printf(" aux %d", v->aux);
            if (v->op == IR_BR || v->op == IR_CBR)
                for (unsigned i = 0; i < b->nsucc; i++)
                    printf(" b%d", b->succ[i]->id);
            printf("\n");
        }
    }
}
