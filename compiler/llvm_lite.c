// Bitcode reader for what Metal emits as AIR, behind llvm_lite.h's LLVM-C subset. Value and metadata numbering
// follow BitcodeReader.cpp so operands resolve as libLLVM resolves them.
#include "llvm_lite.h"
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct chunk {
    struct chunk *next;
    size_t used, size;
    _Alignas(16) unsigned char data[];
};

enum tkind { T_VOID, T_INT, T_FP, T_PTR, T_FUNC, T_LABEL, T_MD, T_ARRAY, T_VECTOR, T_STRUCT, T_OTHER };

struct ll_type {
    struct ll_module *m;
    enum tkind kind;
    unsigned bits;
    int elem;       // -1 for an opaque pointer
    unsigned as;
    int ret;
    unsigned nparams;
    int *params;
    int vararg;
    int *members;
    unsigned nmembers;
};

enum vkind { V_PLACEHOLDER, V_FUNCTION, V_ARG, V_INST, V_CINT, V_CFP, V_CONST, V_GLOBAL, V_MDSTRING, V_MDNODE, V_MDVALUE };

struct ll_value {
    struct ll_module *m;  // to resolve type ids
    enum vkind kind;
    unsigned char undef;  // 1 undef, 2 poison
    unsigned char zero;   // an all-zero vector or aggregate
    unsigned char constant;
    int type;
    int elty;  // a GEP's source element type, a global's value type; -1 otherwise
    LLVMOpcode op;
    int pred;
    unsigned nops;
    struct ll_value **ops;  // a call keeps its callee last, like LLVM
    unsigned nblocks;
    struct ll_block **blocks;
    struct ll_value *next;
    const char *name;
    uint64_t ival;
    double fval;
    const char *str;
    unsigned len;
    unsigned strtab_off, strtab_len;
    int fntype;
    unsigned nargs;
    struct ll_value **args;
    struct ll_block *body;
    unsigned indices[4], nindices;  // outermost first
    unsigned align;            // bytes, 0 for none given
    unsigned init;             // value id + 1, 0 for none
    const struct attr_list *attrs;
    unsigned char contract;    // contract or fast
    unsigned char wrap;        // 1 nuw, 2 nsw
};

// index ~0 is the function, 0 its result, i + 1 parameter i
struct attr_list {
    unsigned n;
    struct {
        uint32_t index;
        unsigned bits;
    } at[];
};

struct attr_group {
    uint64_t id;
    uint32_t index;
    unsigned bits;
};

struct ll_block {
    struct ll_value *first, *last;
    struct ll_block *next;
};

struct named_md {
    const char *name;
    unsigned n;
    uint64_t *ids;
    struct named_md *next;
};

struct abbrev_op {
    unsigned char kind;
    uint64_t val;
};

struct abbrev {
    unsigned n;
    struct abbrev_op ops[];
};

struct abbrevs {
    struct abbrev **v;
    unsigned n, cap;
};

struct ll_module {
    struct chunk *arena;
    struct ll_type *types;
    unsigned ntypes, captypes;
    struct ll_value **vals;
    unsigned nvals, capvals;
    struct ll_value **mds;
    unsigned nmds, capmds;
    struct named_md *named;
    struct ll_value **globals;  // named from the STRTAB once it is read
    unsigned nglobals, capglobals;
    struct ll_value **bodies;
    unsigned nbodies, capbodies, next_body;
    struct abbrevs blockinfo[32];
    struct attr_group *groups;  // only those with an attribute airc asks about
    unsigned ngroups, capgroups;
    const struct attr_list **lists;
    unsigned nlists, caplists;
    const char *strtab;
    size_t strtab_len;
    unsigned version;
    int i1;
    jmp_buf fail;
    char msg[160];
};

struct ll_membuf {
    const unsigned char *data;
    size_t len;
};

struct bits {
    const unsigned char *p;
    size_t nbits, pos;
};

struct cursor {
    struct bits *b;
    unsigned width;
    struct abbrevs ab;
    struct abbrevs *define_into;
};

struct record {
    unsigned code;
    uint64_t *ops;
    unsigned n, cap;
    const unsigned char *blob;
    size_t bloblen;
};

enum { A_LIT, A_FIXED, A_VBR, A_ARRAY, A_CHAR6, A_BLOB };
enum { E_END, E_BLOCK, E_RECORD };

static _Noreturn void fail(struct ll_module *m, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(m->msg, sizeof(m->msg), fmt, ap);
    va_end(ap);
    longjmp(m->fail, 1);
}

static void *zalloc(struct ll_module *m, size_t n)
{
    n = (n + 15) & ~(size_t)15;
    if (!m->arena || m->arena->used + n > m->arena->size) {
        size_t size = n > 65536 ? n : 65536;
        struct chunk *c = malloc(sizeof(*c) + size);
        if (!c)
            fail(m, "out of memory");
        c->next = m->arena;
        c->used = 0;
        c->size = size;
        m->arena = c;
    }
    void *p = m->arena->data + m->arena->used;
    m->arena->used += n;
    return memset(p, 0, n);
}

static void grow(struct ll_module *m, void **v, unsigned *cap, unsigned need, size_t elem)
{
    if (need <= *cap)
        return;
    unsigned ncap = *cap ? *cap : 64;
    while (ncap < need)
        ncap *= 2;
    void *nv = realloc(*v, (size_t)ncap * elem);
    if (!nv)
        fail(m, "out of memory");
    memset((char *)nv + (size_t)*cap * elem, 0, (size_t)(ncap - *cap) * elem);
    *v = nv;
    *cap = ncap;
}

static uint64_t rd(struct ll_module *m, struct bits *b, unsigned n)
{
    if (b->pos + n > b->nbits)
        fail(m, "read past the end of the bitcode");
    uint64_t v = 0;
    for (unsigned got = 0; got < n;) {
        unsigned off = b->pos & 7, take = 8 - off;
        if (take > n - got)
            take = n - got;
        v |= (uint64_t)((b->p[b->pos >> 3] >> off) & ((1u << take) - 1)) << got;
        got += take;
        b->pos += take;
    }
    return v;
}

static uint64_t vbr(struct ll_module *m, struct bits *b, unsigned n)
{
    uint64_t v = 0;
    for (unsigned shift = 0; shift < 64; shift += n - 1) {
        uint64_t c = rd(m, b, n);
        v |= (c & ((1ull << (n - 1)) - 1)) << shift;
        if (!(c >> (n - 1)))
            return v;
    }
    fail(m, "vbr overflow");
}

static void align32(struct bits *b)
{
    b->pos = (b->pos + 31) & ~(size_t)31;
}

static int64_t decode_signed(uint64_t v)
{
    if (v == 1)
        return INT64_MIN;
    return (v & 1) ? -(int64_t)(v >> 1) : (int64_t)(v >> 1);
}

static void push_abbrev(struct ll_module *m, struct abbrevs *ab, struct abbrev *a)
{
    grow(m, (void **)&ab->v, &ab->cap, ab->n + 1, sizeof(*ab->v));
    ab->v[ab->n++] = a;
}

static void define_abbrev(struct ll_module *m, struct bits *b, struct abbrevs *into)
{
    unsigned n = (unsigned)vbr(m, b, 5);
    struct abbrev *a = zalloc(m, sizeof(*a) + n * sizeof(a->ops[0]));
    for (unsigned i = 0; i < n; i++) {
        struct abbrev_op *o = &a->ops[a->n++];
        if (rd(m, b, 1)) {
            o->kind = A_LIT;
            o->val = vbr(m, b, 8);
            continue;
        }
        unsigned enc = (unsigned)rd(m, b, 3);
        if (enc == 1 || enc == 2) {
            o->val = vbr(m, b, 5);
            if (o->val > 64)
                fail(m, "abbrev operand wider than 64 bits");
            o->kind = !o->val ? A_LIT : enc == 1 ? A_FIXED : A_VBR;  // Fixed(0)/VBR(0) read as literal 0
        } else if (enc == 3) {
            o->kind = A_ARRAY;
        } else if (enc == 4) {
            o->kind = A_CHAR6;
        } else if (enc == 5) {
            o->kind = A_BLOB;
        } else {
            fail(m, "bad abbrev encoding %u", enc);
        }
    }
    push_abbrev(m, into, a);
}

static void push_op(struct ll_module *m, struct record *r, uint64_t v)
{
    grow(m, (void **)&r->ops, &r->cap, r->n + 1, sizeof(*r->ops));
    r->ops[r->n++] = v;
}

static uint64_t read_scalar(struct ll_module *m, struct bits *b, const struct abbrev_op *o)
{
    static const char char6[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._";
    switch (o->kind) {
    case A_LIT:
        return o->val;
    case A_FIXED:
        return rd(m, b, (unsigned)o->val);
    case A_VBR:
        return vbr(m, b, (unsigned)o->val);
    case A_CHAR6:
        return (unsigned char)char6[rd(m, b, 6)];
    }
    fail(m, "array or blob inside an array");
}

static void read_record(struct ll_module *m, struct cursor *c, unsigned id, struct record *r)
{
    struct bits *b = c->b;
    r->n = 0;
    r->blob = NULL;
    r->bloblen = 0;
    if (id == 3) {
        r->code = (unsigned)vbr(m, b, 6);
        unsigned n = (unsigned)vbr(m, b, 6);
        for (unsigned i = 0; i < n; i++)
            push_op(m, r, vbr(m, b, 6));
        return;
    }
    if (id - 4 >= c->ab.n)
        fail(m, "undefined abbrev %u", id);
    const struct abbrev *a = c->ab.v[id - 4];
    for (unsigned i = 0; i < a->n; i++) {
        const struct abbrev_op *o = &a->ops[i];
        if (o->kind == A_ARRAY) {
            if (i + 1 >= a->n)
                fail(m, "array abbrev without element type");
            unsigned n = (unsigned)vbr(m, b, 6);
            const struct abbrev_op *e = &a->ops[++i];
            for (unsigned k = 0; k < n; k++)
                push_op(m, r, read_scalar(m, b, e));
        } else if (o->kind == A_BLOB) {
            size_t n = (size_t)vbr(m, b, 6);
            align32(b);
            if (b->pos + n * 8 > b->nbits)
                fail(m, "blob past the end of the bitcode");
            r->blob = b->p + (b->pos >> 3);
            r->bloblen = n;
            b->pos += n * 8;
            align32(b);
        } else {
            push_op(m, r, read_scalar(m, b, o));
        }
    }
    if (!r->n)
        fail(m, "record without a code");
    r->code = (unsigned)r->ops[0];
    memmove(r->ops, r->ops + 1, (r->n - 1) * sizeof(*r->ops));
    r->n--;
}

static int next(struct ll_module *m, struct cursor *c, struct record *r, unsigned *blockid)
{
    for (;;) {
        unsigned id = (unsigned)rd(m, c->b, c->width);
        if (id == 0) {
            align32(c->b);
            return E_END;
        }
        if (id == 1) {
            *blockid = (unsigned)vbr(m, c->b, 8);
            return E_BLOCK;
        }
        if (id == 2) {
            define_abbrev(m, c->b, c->define_into);
            continue;
        }
        read_record(m, c, id, r);
        return E_RECORD;
    }
}

static void enter(struct ll_module *m, struct bits *b, unsigned blockid, struct cursor *c)
{
    memset(c, 0, sizeof(*c));
    c->b = b;
    c->width = (unsigned)vbr(m, b, 4);
    align32(b);
    rd(m, b, 32);  // block length in words
    c->define_into = &c->ab;
    if (blockid < 32)
        for (unsigned i = 0; i < m->blockinfo[blockid].n; i++)
            push_abbrev(m, &c->ab, m->blockinfo[blockid].v[i]);
}

static void leave(struct cursor *c, struct record *r)
{
    free(c->ab.v);
    free(r->ops);
}

static void skip(struct ll_module *m, struct bits *b)
{
    vbr(m, b, 4);
    align32(b);
    size_t words = (size_t)rd(m, b, 32);
    if (b->pos + words * 32 > b->nbits)
        fail(m, "block past the end of the bitcode");
    b->pos += words * 32;
}

static void parse_blockinfo(struct ll_module *m, struct bits *b)
{
    struct cursor c;
    struct record r = { 0 };
    struct abbrevs discard = { 0 };
    unsigned id;
    enter(m, b, 0, &c);
    c.define_into = &discard;  // abbrevs before the first SETBID belong to no block
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            skip(m, b);
            continue;
        }
        if (r.code == 1) {  // SETBID
            if (!r.n || r.ops[0] >= 32)
                fail(m, "bad SETBID");
            c.define_into = &m->blockinfo[r.ops[0]];
        }
    }
    free(discard.v);
    leave(&c, &r);
}

// Attr kinds are 0 enum, 1 integer, 3 and 4 strings, 5 and 6 types. Others (constant ranges) sort after these but
// before the strings, so a group is left at the first of them.
enum { ATTR_NOINLINE = 1, ATTR_READONLY = 2, ATTR_NOALIAS = 4, ATTR_BUFFER_NO_ALIAS = 8 };

static void parse_attr_groups(struct ll_module *m, struct bits *b)
{
    static const char no_alias[] = "air-buffer-no-alias";
    struct cursor c;
    struct record r = { 0 };
    unsigned id;
    enter(m, b, 10, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            skip(m, b);
            continue;
        }
        if (r.code != 3 || r.n < 2)
            continue;
        unsigned bits = 0;
        for (unsigned i = 2; i < r.n;) {
            uint64_t kind = r.ops[i];
            if (kind == 0 && i + 1 < r.n)
                bits |= r.ops[i + 1] == 14 ? ATTR_NOINLINE : r.ops[i + 1] == 21 ? ATTR_READONLY
                        : r.ops[i + 1] == 9                                    ? ATTR_NOALIAS
                                                                               : 0;
            if (kind == 0 || kind == 6)
                i += 2;
            else if (kind == 1 || kind == 5)
                i += 3;
            else if (kind == 3 || kind == 4) {
                unsigned k = 0;
                while (i + 1 + k < r.n && r.ops[i + 1 + k] && k < sizeof no_alias - 1 &&
                       r.ops[i + 1 + k] == (unsigned char)no_alias[k])
                    k++;
                if (k == sizeof no_alias - 1 && i + 1 + k < r.n && !r.ops[i + 1 + k])
                    bits |= ATTR_BUFFER_NO_ALIAS;
                for (int strings = kind == 3 ? 1 : 2; strings && ++i < r.n;)
                    strings -= !r.ops[i];
                i++;
            } else
                break;
        }
        if (!bits)
            continue;
        grow(m, (void **)&m->groups, &m->capgroups, m->ngroups + 1, sizeof(*m->groups));
        m->groups[m->ngroups++] = (struct attr_group){ r.ops[0], (uint32_t)r.ops[1], bits };
    }
    leave(&c, &r);
}

static void parse_attr_lists(struct ll_module *m, struct bits *b)
{
    struct cursor c;
    struct record r = { 0 };
    unsigned id;
    enter(m, b, 9, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            skip(m, b);
            continue;
        }
        if (r.code != 2)
            continue;
        struct attr_list *l = zalloc(m, sizeof(*l) + r.n * sizeof(l->at[0]));
        for (unsigned i = 0; i < r.n; i++)
            for (unsigned k = 0; k < m->ngroups; k++)
                if (r.ops[i] == m->groups[k].id) {
                    l->at[l->n].index = m->groups[k].index;
                    l->at[l->n++].bits = m->groups[k].bits;
                }
        grow(m, (void **)&m->lists, &m->caplists, m->nlists + 1, sizeof(*m->lists));
        m->lists[m->nlists++] = l;
    }
    leave(&c, &r);
}

static struct ll_type *type(struct ll_module *m, int id)
{
    if (id < 0 || (unsigned)id >= m->ntypes)
        fail(m, "bad type id %d", id);
    return &m->types[id];
}

static int is_void(struct ll_module *m, int id)
{
    return id >= 0 && type(m, id)->kind == T_VOID;
}

static int is_fp(struct ll_module *m, int id)
{
    if (id < 0)
        return 0;
    struct ll_type *t = type(m, id);
    if (t->kind == T_VECTOR)
        t = type(m, t->elem);
    return t->kind == T_FP;
}

static int add_type(struct ll_module *m, enum tkind kind, unsigned bits)
{
    grow(m, (void **)&m->types, &m->captypes, m->ntypes + 1, sizeof(*m->types));
    struct ll_type *t = &m->types[m->ntypes];
    t->m = m;
    t->kind = kind;
    t->bits = bits;
    t->elem = -1;
    t->ret = -1;
    return (int)m->ntypes++;
}

static int vec_type(struct ll_module *m, int elem, unsigned n)
{
    for (unsigned i = 0; i < m->ntypes; i++)
        if (m->types[i].kind == T_VECTOR && m->types[i].elem == elem && m->types[i].bits == n)
            return (int)i;
    int t = add_type(m, T_VECTOR, n);
    m->types[t].elem = elem;
    return t;
}

static int ptr_to(struct ll_module *m, int elem, unsigned as)
{
    for (unsigned i = 0; i < m->ntypes; i++)
        if (m->types[i].kind == T_PTR && m->types[i].as == as && m->types[i].elem == elem)
            return (int)i;
    int t = add_type(m, T_PTR, 64);
    m->types[t].as = as;
    m->types[t].elem = elem;
    return t;
}

static int ptr_type(struct ll_module *m, unsigned as)
{
    return ptr_to(m, -1, as);
}

static int i1_type(struct ll_module *m)
{
    if (m->i1 < 0)
        m->i1 = add_type(m, T_INT, 1);
    return m->i1;
}

static struct ll_value *val_slot(struct ll_module *m, unsigned id, int ty)
{
    if (id > m->nvals + (1u << 20))
        fail(m, "value id %u out of range", id);
    grow(m, (void **)&m->vals, &m->capvals, id + 1, sizeof(*m->vals));
    if (!m->vals[id]) {
        m->vals[id] = zalloc(m, sizeof(struct ll_value));
        m->vals[id]->m = m;
        m->vals[id]->type = ty;
        m->vals[id]->elty = -1;
    } else if (m->vals[id]->type < 0) {
        m->vals[id]->type = ty;
    }
    return m->vals[id];
}

// reuses the placeholder a forward reference may already have made
static struct ll_value *define_value(struct ll_module *m, enum vkind kind, int ty)
{
    struct ll_value *v = val_slot(m, m->nvals++, ty);
    v->kind = kind;
    v->type = ty;
    return v;
}

static struct ll_value *md_slot(struct ll_module *m, unsigned id)
{
    grow(m, (void **)&m->mds, &m->capmds, id + 1, sizeof(*m->mds));
    if (!m->mds[id])
        m->mds[id] = zalloc(m, sizeof(struct ll_value));
    return m->mds[id];
}

static struct ll_value *define_md(struct ll_module *m, enum vkind kind)
{
    struct ll_value *v = md_slot(m, m->nmds++);
    v->kind = kind;
    v->type = -1;
    return v;
}

static void add_global(struct ll_module *m, struct ll_value *v, const struct record *r)
{
    if (m->version >= 2) {
        v->strtab_off = (unsigned)r->ops[0];
        v->strtab_len = (unsigned)r->ops[1];
    }
    grow(m, (void **)&m->globals, &m->capglobals, m->nglobals + 1, sizeof(*m->globals));
    m->globals[m->nglobals++] = v;
}

static void parse_types(struct ll_module *m, struct bits *b)
{
    struct cursor c;
    struct record r = { 0 };
    unsigned id;
    enter(m, b, 17, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            skip(m, b);
            continue;
        }
        int t;
        switch (r.code) {
        case 1:   // NUMENTRY
        case 19:  // STRUCT_NAME
            continue;
        case 2:
            add_type(m, T_VOID, 0);
            break;
        case 3:
            add_type(m, T_FP, 32);
            break;
        case 4:
            add_type(m, T_FP, 64);
            break;
        case 5:
            add_type(m, T_LABEL, 0);
            break;
        case 7:
            add_type(m, T_INT, r.n ? (unsigned)r.ops[0] : 0);
            break;
        case 8:  // POINTER [pointee, addrspace]
            t = add_type(m, T_PTR, 64);
            m->types[t].elem = r.n ? (int)r.ops[0] : -1;
            m->types[t].as = r.n > 1 ? (unsigned)r.ops[1] : 0;
            break;
        case 25:  // OPAQUE_POINTER [addrspace]
            t = add_type(m, T_PTR, 64);
            m->types[t].as = r.n ? (unsigned)r.ops[0] : 0;
            break;
        case 10:  // HALF
            add_type(m, T_FP, 16);
            break;
        case 23:  // BFLOAT, width 17 to tell it from half
            add_type(m, T_FP, 17);
            break;
        case 13:
            add_type(m, T_FP, 80);
            break;
        case 14:
        case 15:
            add_type(m, T_FP, 128);
            break;
        case 11:  // ARRAY [n, elem]
        case 12:  // VECTOR [n, elem]
            if (r.n < 2)
                fail(m, "short array/vector type");
            t = add_type(m, r.code == 11 ? T_ARRAY : T_VECTOR, (unsigned)r.ops[0]);
            m->types[t].elem = (int)r.ops[1];
            break;
        case 16:
            add_type(m, T_MD, 0);
            break;
        case 9:   // FUNCTION_OLD [vararg, attrs, ret, params...]
        case 21:  // FUNCTION [vararg, ret, params...]
        {
            unsigned first = r.code == 9 ? 2 : 1;
            if (r.n < first + 1)
                fail(m, "short function type");
            t = add_type(m, T_FUNC, 0);
            struct ll_type *ft = &m->types[t];
            ft->vararg = (int)r.ops[0];
            ft->ret = (int)r.ops[first];
            ft->nparams = r.n - first - 1;
            ft->params = zalloc(m, (ft->nparams + 1) * sizeof(int));
            for (unsigned i = 0; i < ft->nparams; i++)
                ft->params[i] = (int)r.ops[first + 1 + i];
            break;
        }
        case 6:  // OPAQUE (named struct body)
            add_type(m, T_STRUCT, 0);
            break;
        case 18:    // STRUCT_ANON [ispacked, members...]
        case 20: {  // STRUCT_NAMED [ispacked, members...]
            t = add_type(m, T_STRUCT, 0);
            struct ll_type *st = &m->types[t];
            if (r.n > 1) {
                st->nmembers = r.n - 1;
                st->members = zalloc(m, (st->nmembers + 1) * sizeof(int));
                for (unsigned i = 0; i < st->nmembers; i++)
                    st->members[i] = (int)r.ops[1 + i];
            }
            break;
        }
        case 17:
        case 22:
        case 24:
        case 26:  // X86_MMX, TOKEN, X86_AMX, TARGET_TYPE
            add_type(m, T_OTHER, 0);
            break;
        default:
            fail(m, "unknown type record %u", r.code);
        }
    }
    leave(&c, &r);
}

static float half_to_float(unsigned h)
{
    unsigned sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    float f;
    if (exp == 0)
        f = (float)man / 16777216.0f;  // man * 2^-24
    else if (exp == 31)
        f = man ? __builtin_nanf("") : __builtin_inff();
    else
        f = (1.0f + (float)man / 1024.0f) * (float)(1u << exp) / 32768.0f;  // 2^(exp-15)
    return sign ? -f : f;
}

static double fp_from_bits(const struct ll_type *t, uint64_t bits)
{
    unsigned width = t->bits;
    if (width == 16)
        return half_to_float((unsigned)bits);
    if (width == 17) {
        width = 32;
        bits = (uint32_t)bits << 16;
    }
    if (width == 32) {
        uint32_t u = (uint32_t)bits;
        float f;
        memcpy(&f, &u, sizeof(f));
        return f;
    }
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

static void parse_constants(struct ll_module *m, struct bits *b)
{
    struct cursor c;
    struct record r = { 0 };
    unsigned id;
    int cur = -1;
    enter(m, b, 11, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            skip(m, b);
            continue;
        }
        if (r.code == 1) {  // SETTYPE
            if (!r.n)
                fail(m, "short SETTYPE");
            cur = (int)r.ops[0];
            continue;
        }
        struct ll_type *t = type(m, cur);
        // a splat is an integer or float record of a vector type
        struct ll_type *et = t->kind == T_VECTOR ? type(m, t->elem) : t;
        struct ll_value *v;
        if ((r.code == 4 || r.code == 5) && r.n && et->kind == T_INT) {  // INTEGER, WIDE_INTEGER (low word)
            v = define_value(m, V_CINT, cur);
            v->ival = (uint64_t)decode_signed(r.ops[0]);
            if (et->bits < 64)
                v->ival &= (1ull << et->bits) - 1;
        } else if (r.code == 6 && r.n && et->kind == T_FP) {  // FLOAT [bits]
            v = define_value(m, V_CFP, cur);
            v->fval = fp_from_bits(et, r.ops[0]);
        } else if (r.code == 2 && t->kind == T_INT) {  // NULL
            define_value(m, V_CINT, cur);
        } else if (r.code == 2 && t->kind == T_FP) {
            define_value(m, V_CFP, cur);
        } else if (r.code == 2) {
            define_value(m, V_CONST, cur)->zero = 1;
        } else if (r.code == 3 || r.code == 26) {  // UNDEF, POISON
            define_value(m, V_CONST, cur)->undef = r.code == 3 ? 1 : 2;
        } else if (r.code == 7) {  // AGGREGATE [element value ids]
            v = define_value(m, V_CONST, cur);
            v->nops = r.n;
            v->ops = zalloc(m, (r.n + 1) * sizeof(*v->ops));
            for (unsigned i = 0; i < r.n; i++)
                v->ops[i] = val_slot(m, (unsigned)r.ops[i], t->elem);
        } else if (r.code == 12 || r.code == 20 || r.code == 24 || r.code == 32) {
            // the CE_GEP family [source type?, flags?, (type, absolute value id)...]
            unsigned i = 0;
            int elty = -1;
            if (r.code == 24 || r.code == 32 || r.n % 2)
                elty = (int)r.ops[i++];
            if (r.code == 24 || r.code == 32)
                i++;
            v = define_value(m, V_CONST, cur);
            v->op = LLVMGetElementPtr;
            v->elty = elty;
            v->nops = (r.n - i) / 2;
            v->ops = zalloc(m, (v->nops + 1) * sizeof(*v->ops));
            for (unsigned k = 0; k < v->nops; k++, i += 2)
                v->ops[k] = val_slot(m, (unsigned)r.ops[i + 1], (int)r.ops[i]);
        } else if (r.code == 11 && r.n >= 3 && (r.ops[0] == 11 || r.ops[0] == 12)) {
            // CE_CAST [opcode, operand type, operand]; libLLVM folds no-op ones away, airc's strip() looks through them
            v = define_value(m, V_CONST, cur);
            v->op = r.ops[0] == 11 ? LLVMBitCast : LLVMAddrSpaceCast;
            v->nops = 1;
            v->ops = zalloc(m, 2 * sizeof(*v->ops));
            v->ops[0] = val_slot(m, (unsigned)r.ops[2], (int)r.ops[1]);
        } else if ((r.code == 22 || r.code == 8 || r.code == 9) && (t->kind == T_VECTOR || t->kind == T_ARRAY)) {
            // DATA, STRING, CSTRING; CSTRING's missing trailing 0 reads as the missing element
            struct ll_type *et = type(m, t->elem);
            v = define_value(m, V_CONST, cur);
            v->nops = r.n;
            v->ops = zalloc(m, (r.n + 1) * sizeof(*v->ops));
            for (unsigned i = 0; i < r.n; i++) {
                struct ll_value *e = zalloc(m, sizeof(*e));
                e->m = m;
                e->type = t->elem;
                if (et->kind == T_INT) {
                    e->kind = V_CINT;
                    e->ival = et->bits < 64 ? r.ops[i] & ((1ull << et->bits) - 1) : r.ops[i];
                } else {
                    e->kind = V_CFP;
                    e->fval = fp_from_bits(et, r.ops[i]);
                }
                v->ops[i] = e;
            }
        } else {
            define_value(m, V_CONST, cur);  // every other constant record defines exactly one value
        }
    }
    leave(&c, &r);
}

static void parse_metadata(struct ll_module *m, struct bits *b)
{
    struct cursor c;
    struct record r = { 0 };
    unsigned id;
    const char *pending_name = NULL;
    enter(m, b, 15, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            skip(m, b);
            continue;
        }
        switch (r.code) {
        case 35: {  // STRINGS [count, offset] + blob: vbr6 lengths, then the characters
            if (r.n < 2 || !r.blob || r.ops[1] > r.bloblen)
                fail(m, "bad METADATA_STRINGS");
            struct bits lens = { r.blob, (size_t)r.ops[1] * 8, 0 };
            const char *chars = (const char *)r.blob + r.ops[1];
            size_t avail = r.bloblen - r.ops[1];
            for (uint64_t i = 0; i < r.ops[0]; i++) {
                unsigned len = (unsigned)vbr(m, &lens, 6);
                if (len > avail)
                    fail(m, "metadata string past its blob");
                struct ll_value *s = define_md(m, V_MDSTRING);
                s->str = chars;
                s->len = len;
                chars += len;
                avail -= len;
            }
            break;
        }
        case 1: {  // STRING_OLD [chars]
            struct ll_value *s = define_md(m, V_MDSTRING);
            char *str = zalloc(m, r.n + 1);
            for (unsigned i = 0; i < r.n; i++)
                str[i] = (char)r.ops[i];
            s->str = str;
            s->len = r.n;
            break;
        }
        case 2: {  // VALUE [type, value id]
            if (r.n < 2)
                fail(m, "short METADATA_VALUE");
            struct ll_value *v = define_md(m, V_MDVALUE);
            v->nops = 1;
            v->ops = zalloc(m, sizeof(*v->ops));
            v->ops[0] = val_slot(m, (unsigned)r.ops[1], (int)r.ops[0]);
            break;
        }
        case 3:    // NODE [id+1...]
        case 5: {  // DISTINCT_NODE
            struct ll_value *v = define_md(m, V_MDNODE);
            v->nops = r.n;
            v->ops = zalloc(m, (r.n + 1) * sizeof(*v->ops));
            for (unsigned i = 0; i < r.n; i++)
                v->ops[i] = r.ops[i] ? md_slot(m, (unsigned)r.ops[i] - 1) : NULL;
            break;
        }
        case 8:    // OLD_NODE [(type, metadata or value id)...]
        case 9: {  // OLD_FN_NODE
            struct ll_value *v = define_md(m, V_MDNODE);
            v->nops = r.n / 2;
            v->ops = zalloc(m, (v->nops + 1) * sizeof(*v->ops));
            for (unsigned i = 0; i < v->nops; i++) {
                struct ll_type *t = type(m, (int)r.ops[2 * i]);
                unsigned id = (unsigned)r.ops[2 * i + 1];
                if (t->kind == T_MD) {
                    v->ops[i] = md_slot(m, id);
                } else if (t->kind != T_VOID) {
                    struct ll_value *w = zalloc(m, sizeof(*w));
                    w->m = m;
                    w->kind = V_MDVALUE;
                    w->type = -1;
                    w->nops = 1;
                    w->ops = zalloc(m, sizeof(*w->ops));
                    w->ops[0] = val_slot(m, id, (int)r.ops[2 * i]);
                    v->ops[i] = w;
                }
            }
            break;
        }
        case 4: {  // NAME [chars], followed by NAMED_NODE
            char *name = zalloc(m, r.n + 1);
            for (unsigned i = 0; i < r.n; i++)
                name[i] = (char)r.ops[i];
            pending_name = name;
            break;
        }
        case 10: {  // NAMED_NODE [ids]
            if (!pending_name)
                fail(m, "NAMED_NODE without NAME");
            struct named_md *nm = zalloc(m, sizeof(*nm));
            nm->name = pending_name;
            nm->n = r.n;
            nm->ids = zalloc(m, (r.n + 1) * sizeof(*nm->ids));
            memcpy(nm->ids, r.ops, r.n * sizeof(*nm->ids));
            nm->next = m->named;
            m->named = nm;
            pending_name = NULL;
            break;
        }
        case 6:   // KIND
        case 11:  // ATTACHMENT
        case 36:  // GLOBAL_DECL_ATTACHMENT
        case 38:  // INDEX_OFFSET
        case 39:  // INDEX
            break;
        default:
            define_md(m, V_MDNODE);  // locations, DI nodes, ...: one id each, contents unused
        }
    }
    leave(&c, &r);
}

struct ops {
    struct ll_module *m;
    const struct record *r;
    unsigned i;
};

static uint64_t op(struct ops *o)
{
    if (o->i >= o->r->n)
        fail(o->m, "short instruction record %u", o->r->code);
    return o->r->ops[o->i++];
}

static int more(const struct ops *o)
{
    return o->i < o->r->n;
}

static unsigned value_id(struct ops *o, uint64_t enc)
{
    return o->m->version ? o->m->nvals - (uint32_t)enc : (unsigned)enc;
}

static struct ll_value *val_rel(struct ops *o, int ty)
{
    return val_slot(o->m, value_id(o, op(o)), ty);
}

// a value, preceded by its type when it is a forward reference
static struct ll_value *val_pair(struct ops *o)
{
    unsigned id = value_id(o, op(o));
    int ty = id >= o->m->nvals ? (int)op(o) : -1;
    return val_slot(o->m, id, ty);
}

static struct ll_value *val_signed(struct ops *o, int ty)
{
    int64_t rel = decode_signed(op(o));
    return val_slot(o->m, (unsigned)((int64_t)o->m->nvals - rel), ty);
}

struct body {
    struct ll_block **blocks;
    unsigned nblocks, cur;
};

static struct ll_block *block_ref(struct ll_module *m, struct body *f, uint64_t i)
{
    if (i >= f->nblocks)
        fail(m, "bad basic block %llu", (unsigned long long)i);
    return f->blocks[i];
}

static int is_terminator(LLVMOpcode op)
{
    return op == LLVMRet || op == LLVMUncondBr || op == LLVMCondBr || op == LLVMSwitch || op == LLVMUnreachable;
}

static struct ll_value *inst(struct ll_module *m, struct body *f, LLVMOpcode opc, int ty, int has_value,
                             unsigned nops)
{
    if (f->cur >= f->nblocks)
        fail(m, "instruction after the last basic block");
    struct ll_value *v = has_value ? define_value(m, V_INST, ty) : zalloc(m, sizeof(*v));
    v->m = m;
    v->elty = -1;
    v->kind = V_INST;
    v->type = ty;
    v->op = opc;
    v->nops = nops;
    v->ops = zalloc(m, (nops + 1) * sizeof(*v->ops));
    struct ll_block *bb = f->blocks[f->cur];
    if (bb->last)
        bb->last->next = v;
    else
        bb->first = v;
    bb->last = v;
    if (is_terminator(opc))
        f->cur++;
    return v;
}

static const LLVMOpcode binops[2][13] = {
    { LLVMAdd, LLVMSub, LLVMMul, LLVMUDiv, LLVMSDiv, LLVMURem, LLVMSRem, LLVMShl, LLVMLShr, LLVMAShr, LLVMAnd, LLVMOr,
      LLVMXor },
    { LLVMFAdd, LLVMFSub, LLVMFMul, LLVMFDiv, LLVMFDiv, LLVMFRem, LLVMFRem },
};

static const LLVMOpcode casts[13] = {
    LLVMTrunc,  LLVMZExt,    LLVMSExt,     LLVMFPToUI,   LLVMFPToSI,  LLVMUIToFP,       LLVMSIToFP,
    LLVMFPTrunc, LLVMFPExt, LLVMPtrToInt, LLVMIntToPtr, LLVMBitCast, LLVMAddrSpaceCast,
};

static void parse_instruction(struct ll_module *m, struct body *f, const struct record *r)
{
    struct ops o = { m, r, 0 };
    struct ll_value *v, *a, *b2;
    switch (r->code) {
    case 1: {  // DECLAREBLOCKS [n]
        f->nblocks = (unsigned)op(&o);
        f->blocks = zalloc(m, (f->nblocks + 1) * sizeof(*f->blocks));
        for (unsigned i = 0; i < f->nblocks; i++)
            f->blocks[i] = zalloc(m, sizeof(struct ll_block));
        for (unsigned i = 0; i + 1 < f->nblocks; i++)
            f->blocks[i]->next = f->blocks[i + 1];
        return;
    }
    case 2: {  // BINOP [lhs pair, rhs, opcode, flags?]
        a = val_pair(&o);
        b2 = val_rel(&o, a->type);
        unsigned code = (unsigned)op(&o);
        int fp = is_fp(m, a->type);
        if (code >= 13 || (fp && code >= 7))
            fail(m, "bad binop %u", code);
        v = inst(m, f, binops[fp][code], a->type, 1, 2);
        v->ops[0] = a;
        v->ops[1] = b2;
        if (fp && more(&o))
            v->contract = (op(&o) & 0x21) != 0;  // unsafe algebra (fast) or contract
        else if (more(&o))
            v->wrap = op(&o) & 3;
        return;
    }
    case 3: {  // CAST [op pair, dest type, opcode, flags?]
        a = val_pair(&o);
        int ty = (int)op(&o);
        unsigned code = (unsigned)op(&o);
        if (code >= 13)
            fail(m, "bad cast %u", code);
        v = inst(m, f, casts[code], ty, 1, 1);
        v->ops[0] = a;
        return;
    }
    case 4:    // GEP_OLD [operand pairs...]
    case 30:   // INBOUNDS_GEP_OLD
    case 43: {  // GEP [flags, source element type, operand pairs...]
        int elty = -1;
        if (r->code == 43) {
            op(&o);
            elty = (int)op(&o);
        }
        struct ll_value *list[64];
        unsigned n = 0;
        while (more(&o)) {
            if (n == 64)
                fail(m, "GEP with too many indices");
            list[n++] = val_pair(&o);
        }
        if (!n)
            fail(m, "GEP without a base");
        int rty = list[0]->type;
        struct ll_type *bt = rty >= 0 ? type(m, rty) : NULL;
        if (r->code != 43 && bt && bt->kind == T_PTR && bt->elem >= 0) {
            // with typed pointers the result points at what the indices reach
            elty = bt->elem;
            int t = elty;
            for (unsigned i = 2; i < n && t >= 0; i++) {
                struct ll_type *at = type(m, t);
                t = at->kind == T_STRUCT ? (list[i]->kind == V_CINT && list[i]->ival < at->nmembers
                                                ? at->members[list[i]->ival]
                                                : -1)
                    : at->kind == T_ARRAY || at->kind == T_VECTOR ? at->elem
                                                                   : -1;
            }
            rty = ptr_to(m, t, bt->as);
        }
        v = inst(m, f, LLVMGetElementPtr, rty, 1, n);
        v->elty = elty;
        memcpy(v->ops, list, n * sizeof(*list));
        return;
    }
    case 5:    // SELECT [t pair, f, cond]
    case 29: {  // VSELECT [t pair, f, cond pair]
        a = val_pair(&o);
        b2 = val_rel(&o, a->type);
        struct ll_value *cond = r->code == 29 ? val_pair(&o) : val_rel(&o, i1_type(m));
        v = inst(m, f, LLVMSelect, a->type, 1, 3);
        v->ops[0] = cond;
        v->ops[1] = a;
        v->ops[2] = b2;
        return;
    }
    case 6: {  // EXTRACTELT [vec pair, index pair]
        a = val_pair(&o);
        b2 = val_pair(&o);
        int ty = a->type >= 0 && type(m, a->type)->kind == T_VECTOR ? type(m, a->type)->elem : -1;
        v = inst(m, f, LLVMExtractElement, ty, 1, 2);
        v->ops[0] = a;
        v->ops[1] = b2;
        return;
    }
    case 7: {  // INSERTELT [vec pair, elt, index pair]
        a = val_pair(&o);
        int ety = a->type >= 0 && type(m, a->type)->kind == T_VECTOR ? type(m, a->type)->elem : -1;
        b2 = val_rel(&o, ety);
        struct ll_value *idx = val_pair(&o);
        v = inst(m, f, LLVMInsertElement, a->type, 1, 3);
        v->ops[0] = a;
        v->ops[1] = b2;
        v->ops[2] = idx;
        return;
    }
    case 8: {  // SHUFFLEVEC [v1 pair, v2, mask pair]
        a = val_pair(&o);
        b2 = val_rel(&o, a->type);
        struct ll_value *mask = val_pair(&o);
        int ty = a->type;
        if (a->type >= 0 && mask->type >= 0 && type(m, a->type)->kind == T_VECTOR && type(m, mask->type)->kind == T_VECTOR)
            ty = vec_type(m, type(m, a->type)->elem, type(m, mask->type)->bits);
        v = inst(m, f, LLVMShuffleVector, ty, 1, 3);
        v->ops[0] = a;
        v->ops[1] = b2;
        v->ops[2] = mask;
        return;
    }
    case 9:
    case 28: {  // CMP/CMP2 [lhs pair, rhs, predicate, flags?]
        a = val_pair(&o);
        b2 = val_rel(&o, a->type);
        int pred = (int)op(&o);
        struct ll_type *at = type(m, a->type);
        int rt = at->kind == T_VECTOR ? vec_type(m, i1_type(m), at->bits) : i1_type(m);
        v = inst(m, f, is_fp(m, a->type) ? LLVMFCmp : LLVMICmp, rt, 1, 2);
        v->pred = pred;
        v->ops[0] = a;
        v->ops[1] = b2;
        return;
    }
    case 10:  // RET [value pair?]
        if (more(&o)) {
            a = val_pair(&o);
            v = inst(m, f, LLVMRet, -1, 0, 1);
            v->ops[0] = a;
        } else {
            inst(m, f, LLVMRet, -1, 0, 0);
        }
        return;
    case 11: {  // BR [true bb, false bb, cond] or [bb]
        struct ll_block *t = block_ref(m, f, op(&o));
        if (!more(&o)) {
            v = inst(m, f, LLVMUncondBr, -1, 0, 0);
            v->nblocks = 1;
            v->blocks = zalloc(m, sizeof(*v->blocks));
            v->blocks[0] = t;
            return;
        }
        struct ll_block *fb = block_ref(m, f, op(&o));
        a = val_rel(&o, i1_type(m));
        v = inst(m, f, LLVMCondBr, -1, 0, 1);
        v->ops[0] = a;
        v->nblocks = 2;
        v->blocks = zalloc(m, 2 * sizeof(*v->blocks));
        v->blocks[0] = t;
        v->blocks[1] = fb;
        return;
    }
    case 12: {  // SWITCH [type, cond, default bb, (case value id, bb)...]
        if (r->n && (r->ops[0] >> 16) == 0x4b5)
            fail(m, "wide-integer switch not supported");
        int ty = (int)op(&o);
        a = val_rel(&o, ty);
        struct ll_block *def = block_ref(m, f, op(&o));
        unsigned ncases = (r->n - 3) / 2;
        v = inst(m, f, LLVMSwitch, -1, 0, 1 + ncases);
        v->ops[0] = a;
        v->nblocks = 1 + ncases;
        v->blocks = zalloc(m, (1 + ncases) * sizeof(*v->blocks));
        v->blocks[0] = def;
        for (unsigned i = 0; i < ncases; i++) {
            v->ops[1 + i] = val_slot(m, (unsigned)op(&o), ty);
            v->blocks[1 + i] = block_ref(m, f, op(&o));
        }
        return;
    }
    case 15:  // UNREACHABLE
        inst(m, f, LLVMUnreachable, -1, 0, 0);
        return;
    case 16: {  // PHI [type, (signed value, bb)..., flags?]
        int ty = (int)op(&o);
        unsigned n = (r->n - 1) / 2;
        struct ll_value **vals = zalloc(m, (n + 1) * sizeof(*vals));
        struct ll_block **bbs = zalloc(m, (n + 1) * sizeof(*bbs));
        for (unsigned i = 0; i < n; i++) {
            vals[i] = val_signed(&o, ty);
            bbs[i] = block_ref(m, f, op(&o));
        }
        v = inst(m, f, LLVMPHI, ty, 1, 0);
        v->nops = n;
        v->ops = vals;
        v->nblocks = n;
        v->blocks = bbs;
        return;
    }
    case 19: {  // ALLOCA [inst type, size type, size id, align]
        int ety = r->n ? (int)op(&o) : -1;
        // without the explicit-type flag the record names the pointer, not the allocated type
        int legacy = r->n < 4 || !(r->ops[3] & 1u << 6);
        if (legacy && ety >= 0 && type(m, ety)->kind == T_PTR)
            ety = type(m, ety)->elem;
        v = inst(m, f, LLVMAlloca, legacy ? ptr_to(m, ety, 0) : ptr_type(m, 0), 1, 0);
        v->elty = ety;
        return;
    }
    case 20:    // LOAD [ptr pair, type, align, vol]
    case 41: {  // LOADATOMIC [ptr pair, type, align, vol, ordering, scope]
        a = val_pair(&o);
        int ty = -1;
        if (r->n - o.i == (r->code == 20 ? 3u : 5u))
            ty = (int)op(&o);
        else if (a->type >= 0 && type(m, a->type)->kind == T_PTR)
            ty = type(m, a->type)->elem;
        v = inst(m, f, LLVMLoad, ty, 1, 1);
        v->ops[0] = a;
        if (more(&o)) {
            uint64_t log2p1 = op(&o);
            v->align = log2p1 ? 1u << (log2p1 - 1) : 0;
        }
        return;
    }
    case 24:    // STORE_OLD [ptr pair, value, align, vol], the value typed by the pointee
    case 44:    // STORE [ptr pair, value pair, align, vol]
    case 45: {  // STOREATOMIC
        a = val_pair(&o);
        if (r->code == 24) {
            int pty = a->type >= 0 ? type(m, a->type)->elem : -1;
            b2 = val_rel(&o, pty);
        } else {
            b2 = val_pair(&o);
        }
        v = inst(m, f, LLVMStore, -1, 0, 2);
        v->ops[0] = b2;
        v->ops[1] = a;
        if (more(&o)) {
            uint64_t log2p1 = op(&o);
            v->align = log2p1 ? 1u << (log2p1 - 1) : 0;
        }
        return;
    }
    case 46: {  // CMPXCHG [ptr pair, cmp pair, new, ...]
        a = val_pair(&o);
        b2 = val_pair(&o);
        struct ll_value *nv = val_rel(&o, b2->type);
        v = inst(m, f, LLVMAtomicCmpXchg, -1, 1, 3);
        v->ops[0] = a;
        v->ops[1] = b2;
        v->ops[2] = nv;
        return;
    }
    case 59: {  // ATOMICRMW [ptr pair, value pair, op, vol, ordering, scope, align]
        a = val_pair(&o);
        b2 = val_pair(&o);
        v = inst(m, f, LLVMAtomicRMW, b2->type, 1, 2);
        v->ops[0] = a;
        v->ops[1] = b2;
        return;
    }
    case 36:  // FENCE
        inst(m, f, LLVMFence, -1, 0, 0);
        return;
    case 34: {  // CALL [attrs, cc, fmf?, fn type?, callee pair, args...]
        op(&o);
        uint64_t cc = op(&o);
        if (cc & (1u << 17))
            op(&o);
        int fty = (cc & (1u << 15)) ? (int)op(&o) : -1;
        struct ll_value *callee = val_pair(&o);
        if (fty < 0) {
            if (callee->kind != V_FUNCTION)
                fail(m, "indirect call without an explicit type");
            fty = callee->fntype;
        }
        struct ll_type *ft = type(m, fty);
        if (ft->kind != T_FUNC)
            fail(m, "call through a non-function type");
        struct ll_value *list[256];
        unsigned n = 0;
        for (unsigned i = 0; i < ft->nparams; i++) {
            if (n == 255)
                fail(m, "call with too many arguments");
            enum tkind k = type(m, ft->params[i])->kind;
            if (k == T_LABEL) {
                op(&o);
            } else if (k == T_MD) {
                op(&o);  // indexes the function's metadata, not its values
                list[n++] = NULL;
            } else {
                list[n++] = val_rel(&o, ft->params[i]);
            }
        }
        while (ft->vararg && more(&o) && n < 255)
            list[n++] = val_pair(&o);
        v = inst(m, f, LLVMCall, ft->ret, !is_void(m, ft->ret), n + 1);
        memcpy(v->ops, list, n * sizeof(*list));
        v->ops[n] = callee;
        return;
    }
    case 56: {  // UNOP [op pair, opcode]
        a = val_pair(&o);
        if (op(&o) != 0)
            fail(m, "unknown unary op");
        v = inst(m, f, LLVMFNeg, a->type, 1, 1);
        v->ops[0] = a;
        return;
    }
    case 58:  // FREEZE [op pair]
        a = val_pair(&o);
        v = inst(m, f, LLVMFreeze, a->type, 1, 1);
        v->ops[0] = a;
        return;
    case 26: {  // EXTRACTVAL [agg pair, indices...]
        a = val_pair(&o);
        unsigned idx[4], n = 0;
        while (more(&o) && n < 4)
            idx[n++] = (unsigned)op(&o);
        int ty = a->type;
        for (unsigned i = 0; i < n && ty >= 0; i++) {
            struct ll_type *at = type(m, ty);
            ty = at->kind == T_STRUCT ? (idx[i] < at->nmembers ? at->members[idx[i]] : -1)
                 : at->kind == T_ARRAY ? at->elem
                                       : -1;
        }
        v = inst(m, f, LLVMExtractValue, ty, 1, 1);
        v->ops[0] = a;
        memcpy(v->indices, idx, sizeof(idx));
        v->nindices = n;
        return;
    }
    case 27:  // INSERTVAL [agg pair, value pair, indices...]
        a = val_pair(&o);
        b2 = val_pair(&o);
        v = inst(m, f, LLVMInsertValue, a->type, 1, 2);
        v->ops[0] = a;
        v->ops[1] = b2;
        while (more(&o) && v->nindices < 4)
            v->indices[v->nindices++] = (unsigned)op(&o);
        return;
    case 33:  // DEBUG_LOC_AGAIN
    case 35:  // DEBUG_LOC
    case 55:  // OPERAND_BUNDLE
    case 60:  // BLOCKADDR_USERS
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:  // DEBUG_RECORD_*
        return;
    default:
        fail(m, "unsupported instruction record %u", r->code);
    }
}

static void parse_function(struct ll_module *m, struct bits *b)
{
    if (m->next_body >= m->nbodies)
        fail(m, "more function bodies than defined functions");
    struct ll_value *fn = m->bodies[m->next_body++];
    unsigned base = m->nvals;
    struct ll_type *ft = type(m, fn->fntype);
    fn->nargs = ft->nparams;
    fn->args = zalloc(m, (ft->nparams + 1) * sizeof(*fn->args));
    for (unsigned i = 0; i < ft->nparams; i++)
        fn->args[i] = define_value(m, V_ARG, ft->params[i]);

    struct cursor c;
    struct record r = { 0 };
    struct body f = { 0 };
    unsigned id;
    enter(m, b, 12, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            if (id == 11)
                parse_constants(m, b);
            else
                skip(m, b);
            continue;
        }
        parse_instruction(m, &f, &r);
    }
    leave(&c, &r);
    fn->body = f.nblocks ? f.blocks[0] : NULL;

    // function-local value ids are reused by the next body
    memset(m->vals + base, 0, (m->capvals - base) * sizeof(*m->vals));
    m->nvals = base;
}

static void parse_vst(struct ll_module *m, struct bits *b);

static void parse_module(struct ll_module *m, struct bits *b)
{
    struct cursor c;
    struct record r = { 0 };
    unsigned id;
    enter(m, b, 8, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            switch (id) {
            case 0:
                parse_blockinfo(m, b);
                break;
            case 9:
                parse_attr_lists(m, b);
                break;
            case 10:
                parse_attr_groups(m, b);
                break;
            case 11:
                parse_constants(m, b);
                break;
            case 12:
                parse_function(m, b);
                break;
            case 15:
                parse_metadata(m, b);
                break;
            case 14:
                if (m->version < 2)
                    parse_vst(m, b);
                else
                    skip(m, b);
                break;
            case 17:
                parse_types(m, b);
                break;
            default:
                skip(m, b);
            }
            continue;
        }
        struct ll_value *v;
        unsigned s = m->version >= 2 ? 2 : 0;  // records lead with a STRTAB offset/size from version 2
        switch (r.code) {
        case 1:  // VERSION
            m->version = r.n ? (unsigned)r.ops[0] : 0;
            break;
        case 8: {  // FUNCTION [strtab?, type, cc, isproto, linkage, attribute list + 1, ...]
            if (r.n < s + 3)
                fail(m, "short FUNCTION record");
            int fty = (int)r.ops[s];
            if (type(m, fty)->kind == T_PTR)
                fty = type(m, fty)->elem;
            v = define_value(m, V_FUNCTION, -1);
            v->fntype = fty;
            uint64_t list = r.n > s + 4 ? r.ops[s + 4] : 0;
            v->attrs = list && list <= m->nlists ? m->lists[list - 1] : NULL;
            add_global(m, v, &r);
            if (!r.ops[s + 2]) {
                grow(m, (void **)&m->bodies, &m->capbodies, m->nbodies + 1, sizeof(*m->bodies));
                m->bodies[m->nbodies++] = v;
            }
            break;
        }
        case 7: {  // GLOBALVAR [strtab?, type, isconst | explicit_type << 1 | addrspace << 2, ...]
            if (r.n < s + 2)
                fail(m, "short GLOBALVAR record");
            int ty = (int)r.ops[s];
            unsigned flags = (unsigned)r.ops[s + 1];
            unsigned as = flags & 2 ? flags >> 2 : type(m, ty)->as;
            int elty = flags & 2 ? ty : type(m, ty)->elem;
            v = define_value(m, V_GLOBAL, flags & 2 ? ptr_type(m, as) : ptr_to(m, elty, as));
            v->elty = elty;
            v->init = r.n > s + 2 ? (unsigned)r.ops[s + 2] : 0;
            v->constant = flags & 1;
            add_global(m, v, &r);
            break;
        }
        case 9:   // ALIAS_OLD
        case 14:  // ALIAS
        case 15:  // IFUNC
            v = define_value(m, V_GLOBAL, -1);
            add_global(m, v, &r);
            break;
        }
    }
    leave(&c, &r);
}

// names the globals before bitcode version 2 moved them to the STRTAB
static void parse_vst(struct ll_module *m, struct bits *b)
{
    struct cursor c;
    struct record r = { 0 };
    unsigned id;
    enter(m, b, 14, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK) {
            skip(m, b);
            continue;
        }
        if (r.code != 1 || r.n < 1)  // ENTRY [value id, chars...]
            continue;
        char *name = zalloc(m, r.n);
        for (unsigned i = 1; i < r.n; i++)
            name[i - 1] = (char)r.ops[i];
        val_slot(m, (unsigned)r.ops[0], -1)->name = name;
    }
    leave(&c, &r);
}

static void parse_strtab(struct ll_module *m, struct bits *b)
{
    struct cursor c;
    struct record r = { 0 };
    unsigned id;
    enter(m, b, 23, &c);
    for (;;) {
        int e = next(m, &c, &r, &id);
        if (e == E_END)
            break;
        if (e == E_BLOCK)
            skip(m, b);
        else if (r.code == 1 && r.blob) {
            m->strtab = (const char *)r.blob;
            m->strtab_len = r.bloblen;
        }
    }
    leave(&c, &r);
}

LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRange(const char *data, size_t len, const char *name,
                                                          LLVMBool requires_null)
{
    (void)name;
    (void)requires_null;
    struct ll_membuf *mb = malloc(sizeof(*mb));
    if (mb) {
        mb->data = (const unsigned char *)data;
        mb->len = len;
    }
    return mb;
}

void LLVMDisposeMemoryBuffer(LLVMMemoryBufferRef buf)
{
    free(buf);
}

void LLVMDisposeModule(LLVMModuleRef m)
{
    if (!m)
        return;
    while (m->arena) {
        struct chunk *n = m->arena->next;
        free(m->arena);
        m->arena = n;
    }
    for (int i = 0; i < 32; i++)
        free(m->blockinfo[i].v);
    free(m->types);
    free(m->vals);
    free(m->mds);
    free(m->globals);
    free(m->bodies);
    free(m->groups);
    free(m->lists);
    free(m);
}

LLVMBool LLVMParseBitcode2(LLVMMemoryBufferRef buf, LLVMModuleRef *out)
{
    *out = NULL;
    struct ll_module *m = calloc(1, sizeof(*m));
    if (!m)
        return 1;
    m->i1 = -1;
    if (setjmp(m->fail)) {
        fprintf(stderr, "bitcode: %s\n", m->msg);
        LLVMDisposeModule(m);
        return 1;
    }
    const unsigned char *p = buf->data;
    size_t len = buf->len;
    if (len >= 20 && p[0] == 0xde && p[1] == 0xc0 && p[2] == 0x17 && p[3] == 0x0b) {  // wrapper header
        uint32_t off, size;
        memcpy(&off, p + 8, 4);
        memcpy(&size, p + 12, 4);
        if ((size_t)off + size > len)
            fail(m, "wrapper points past the file");
        p += off;
        len = size;
    }
    if (len < 4 || p[0] != 'B' || p[1] != 'C' || p[2] != 0xc0 || p[3] != 0xde)
        fail(m, "not LLVM bitcode");
    struct bits b = { p, len * 8, 32 };
    while (b.pos + 32 <= b.nbits) {
        if (rd(m, &b, 2) != 1)  // anything but a block at top level is end padding
            break;
        unsigned id = (unsigned)vbr(m, &b, 8);
        if (id == 8)
            parse_module(m, &b);
        else if (id == 23)
            parse_strtab(m, &b);
        else
            skip(m, &b);
    }
    for (unsigned i = 0; i < m->nglobals; i++) {
        struct ll_value *g = m->globals[i];
        if (!m->strtab || (size_t)g->strtab_off + g->strtab_len > m->strtab_len)
            continue;
        char *name = zalloc(m, g->strtab_len + 1);
        memcpy(name, m->strtab + g->strtab_off, g->strtab_len);
        g->name = name;
    }
    *out = m;
    return 0;
}

static struct named_md *find_named(LLVMModuleRef m, const char *name)
{
    for (struct named_md *nm = m->named; nm; nm = nm->next)
        if (!strcmp(nm->name, name))
            return nm;
    return NULL;
}

unsigned LLVMGetNamedMetadataNumOperands(LLVMModuleRef m, const char *name)
{
    struct named_md *nm = find_named(m, name);
    return nm ? nm->n : 0;
}

void LLVMGetNamedMetadataOperands(LLVMModuleRef m, const char *name, LLVMValueRef *out)
{
    struct named_md *nm = find_named(m, name);
    for (unsigned i = 0; nm && i < nm->n; i++)
        out[i] = nm->ids[i] < m->nmds ? m->mds[nm->ids[i]] : NULL;
}

unsigned LLVMGetMDNodeNumOperands(LLVMValueRef v)
{
    return v && v->kind == V_MDNODE ? v->nops : 0;
}

void LLVMGetMDNodeOperands(LLVMValueRef v, LLVMValueRef *out)
{
    for (unsigned i = 0; i < LLVMGetMDNodeNumOperands(v); i++) {
        struct ll_value *o = v->ops[i];
        out[i] = o && o->kind == V_MDVALUE ? o->ops[0] : o;  // LLVM-C unwraps ValueAsMetadata
    }
}

const char *LLVMGetMDString(LLVMValueRef v, unsigned *len)
{
    if (!v || v->kind != V_MDSTRING) {
        *len = 0;
        return NULL;
    }
    *len = v->len;
    return v->str;
}

LLVMTypeRef LLVMTypeOf(LLVMValueRef v)
{
    return v && v->m && v->type >= 0 && (unsigned)v->type < v->m->ntypes ? &v->m->types[v->type] : NULL;
}

LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef t)
{
    switch (t->kind) {
    case T_VOID:
        return LLVMVoidTypeKind;
    case T_INT:
        return LLVMIntegerTypeKind;
    case T_FP:
        return t->bits == 16   ? LLVMHalfTypeKind
               : t->bits == 17 ? LLVMBFloatTypeKind
               : t->bits == 32 ? LLVMFloatTypeKind
               : t->bits == 64 ? LLVMDoubleTypeKind
                               : LLVMFP128TypeKind;
    case T_PTR:
        return LLVMPointerTypeKind;
    case T_FUNC:
        return LLVMFunctionTypeKind;
    case T_LABEL:
        return LLVMLabelTypeKind;
    case T_MD:
        return LLVMMetadataTypeKind;
    case T_ARRAY:
        return LLVMArrayTypeKind;
    case T_VECTOR:
        return LLVMVectorTypeKind;
    case T_STRUCT:
        return LLVMStructTypeKind;
    case T_OTHER:
        break;
    }
    return LLVMTokenTypeKind;
}

unsigned LLVMGetIntTypeWidth(LLVMTypeRef t)
{
    return t->bits;
}

LLVMTypeRef LLVMGetAllocatedType(LLVMValueRef alloca)
{
    return LLVMGetGEPSourceElementType(alloca);
}

LLVMTypeRef LLVMInt32Type(void)
{
    static struct ll_type i32 = { .kind = T_INT, .bits = 32, .elem = -1 };
    return &i32;
}

LLVMTypeRef LLVMInt8Type(void)
{
    static struct ll_type i8 = { .kind = T_INT, .bits = 8, .elem = -1 };
    return &i8;
}

LLVMTypeRef LLVMHalfType(void)
{
    static struct ll_type half = { .kind = T_FP, .bits = 16, .elem = -1 };
    return &half;
}

LLVMTypeRef LLVMGetGEPSourceElementType(LLVMValueRef gep)
{
    return gep->elty >= 0 && (unsigned)gep->elty < gep->m->ntypes ? &gep->m->types[gep->elty] : NULL;
}

unsigned LLVMGetAlignment(LLVMValueRef v)
{
    return v->align;
}

unsigned LLVMGetVectorSize(LLVMTypeRef t)
{
    return t->kind == T_VECTOR ? t->bits : 0;
}

LLVMTypeRef LLVMGetElementType(LLVMTypeRef t)
{
    return t->elem >= 0 && (unsigned)t->elem < t->m->ntypes ? &t->m->types[t->elem] : NULL;
}

unsigned LLVMGetPointerAddressSpace(LLVMTypeRef t)
{
    return t->as;
}

uint64_t LLVMGetArrayLength2(LLVMTypeRef t)
{
    return t->kind == T_ARRAY ? t->bits : 0;
}

unsigned LLVMGetNumIndices(LLVMValueRef v)
{
    return v->nindices;
}

const unsigned *LLVMGetIndices(LLVMValueRef v)
{
    return v->indices;
}

unsigned LLVMCountStructElementTypes(LLVMTypeRef t)
{
    return t->nmembers;
}

LLVMTypeRef LLVMStructGetTypeAtIndex(LLVMTypeRef t, unsigned i)
{
    return i < t->nmembers ? &t->m->types[t->members[i]] : NULL;
}

LLVMValueRef LLVMIsAGlobalVariable(LLVMValueRef v)
{
    return v && v->kind == V_GLOBAL && v->elty >= 0 ? v : NULL;
}

LLVMTypeRef LLVMGlobalGetValueType(LLVMValueRef g)
{
    return g->elty >= 0 && (unsigned)g->elty < g->m->ntypes ? &g->m->types[g->elty] : NULL;
}

LLVMValueRef LLVMIsAConstantExpr(LLVMValueRef v)
{
    return v && v->kind == V_CONST && v->op != LLVMUnknownOp ? v : NULL;
}

LLVMOpcode LLVMGetConstOpcode(LLVMValueRef v)
{
    return v->op;
}

LLVMBool LLVMIsUndef(LLVMValueRef v)
{
    return v->kind == V_CONST && v->undef;
}

LLVMBool LLVMIsPoison(LLVMValueRef v)
{
    return v->kind == V_CONST && v->undef == 2;
}

LLVMValueRef LLVMGetInitializer(LLVMValueRef g)
{
    return g->kind == V_GLOBAL && g->init && g->init <= g->m->nvals ? g->m->vals[g->init - 1] : NULL;
}

LLVMBool LLVMIsGlobalConstant(LLVMValueRef g)
{
    return g->kind == V_GLOBAL && g->constant;
}

LLVMValueRef LLVMGetAggregateElement(LLVMValueRef c, unsigned i)
{
    if (c->kind != V_CONST)
        return NULL;
    if (i < c->nops)
        return c->ops[i];
    if (!c->undef && !c->zero)
        return NULL;
    struct ll_type *t = LLVMTypeOf(c);
    if (!t || (t->kind != T_VECTOR && t->kind != T_ARRAY) || i >= t->bits)
        return NULL;
    struct ll_value *e = zalloc(c->m, sizeof(*e));
    e->m = c->m;
    e->type = t->elem;
    e->undef = c->undef;
    enum tkind k = type(c->m, t->elem)->kind;
    e->zero = !c->undef && (k == T_ARRAY || k == T_VECTOR || k == T_STRUCT);
    e->kind = c->undef || e->zero ? V_CONST : k == T_FP ? V_CFP : V_CINT;
    return e;
}

unsigned LLVMGetNumMaskElements(LLVMValueRef shuffle)
{
    struct ll_type *t = LLVMTypeOf(shuffle->ops[2]);
    return t ? LLVMGetVectorSize(t) : 0;
}

int LLVMGetUndefMaskElem(void)
{
    return -1;
}

int LLVMGetMaskValue(LLVMValueRef shuffle, unsigned i)
{
    struct ll_value *e = LLVMGetAggregateElement(shuffle->ops[2], i);
    return e && e->kind == V_CINT ? (int)e->ival : LLVMGetUndefMaskElem();
}

const char *LLVMGetValueName(LLVMValueRef v)
{
    return v && v->name ? v->name : "";
}

LLVMValueRef LLVMIsAFunction(LLVMValueRef v)
{
    return v && v->kind == V_FUNCTION ? v : NULL;
}

LLVMValueRef LLVMIsAInstruction(LLVMValueRef v)
{
    return v && v->kind == V_INST ? v : NULL;
}

LLVMValueRef LLVMIsAICmpInst(LLVMValueRef v)
{
    return v && v->kind == V_INST && v->op == LLVMICmp ? v : NULL;
}

LLVMValueRef LLVMIsAConstantInt(LLVMValueRef v)
{
    return v && v->kind == V_CINT ? v : NULL;
}

LLVMValueRef LLVMIsAConstantFP(LLVMValueRef v)
{
    return v && v->kind == V_CFP ? v : NULL;
}

unsigned long long LLVMConstIntGetZExtValue(LLVMValueRef v)
{
    return v->ival;
}

double LLVMConstRealGetDouble(LLVMValueRef v, LLVMBool *loses_info)
{
    *loses_info = 0;
    return v->fval;
}

unsigned LLVMCountParams(LLVMValueRef fn)
{
    return fn->nargs;
}

LLVMValueRef LLVMGetParam(LLVMValueRef fn, unsigned i)
{
    return i < fn->nargs ? fn->args[i] : NULL;
}

LLVMBasicBlockRef LLVMGetFirstBasicBlock(LLVMValueRef fn)
{
    return fn->body;
}

static LLVMValueRef function_from(struct ll_module *m, unsigned i)
{
    for (; i < m->nglobals; i++)
        if (m->globals[i]->kind == V_FUNCTION)
            return m->globals[i];
    return NULL;
}

LLVMValueRef LLVMGetFirstFunction(LLVMModuleRef m)
{
    return function_from(m, 0);
}

LLVMValueRef LLVMGetNextFunction(LLVMValueRef fn)
{
    struct ll_module *m = fn->m;
    for (unsigned i = 0; i < m->nglobals; i++)
        if (m->globals[i] == fn)
            return function_from(m, i + 1);
    return NULL;
}

LLVMBasicBlockRef LLVMGetNextBasicBlock(LLVMBasicBlockRef bb)
{
    return bb->next;
}

LLVMValueRef LLVMGetFirstInstruction(LLVMBasicBlockRef bb)
{
    return bb->first;
}

LLVMValueRef LLVMGetNextInstruction(LLVMValueRef v)
{
    return v->next;
}

LLVMValueRef LLVMGetBasicBlockTerminator(LLVMBasicBlockRef bb)
{
    return bb->last && is_terminator(bb->last->op) ? bb->last : NULL;
}

LLVMBool LLVMCanContract(LLVMValueRef v)
{
    return v->kind == V_INST && v->contract;
}

LLVMBool LLVMGetNUW(LLVMValueRef v)
{
    return v->wrap & 1;
}

LLVMBool LLVMGetNSW(LLVMValueRef v)
{
    return v->wrap >> 1 & 1;
}

LLVMOpcode LLVMGetInstructionOpcode(LLVMValueRef v)
{
    return v && v->kind == V_INST ? v->op : LLVMUnknownOp;
}

int LLVMGetNumOperands(LLVMValueRef v)
{
    return (int)v->nops;
}

LLVMValueRef LLVMGetOperand(LLVMValueRef v, unsigned i)
{
    return i < v->nops ? v->ops[i] : NULL;
}

LLVMIntPredicate LLVMGetICmpPredicate(LLVMValueRef v)
{
    return (LLVMIntPredicate)v->pred;
}

LLVMRealPredicate LLVMGetFCmpPredicate(LLVMValueRef v)
{
    return (LLVMRealPredicate)v->pred;
}

LLVMValueRef LLVMGetCalledValue(LLVMValueRef call)
{
    LLVMValueRef v = call->nops ? call->ops[call->nops - 1] : NULL;
    // typed pointers call a function through a cast of it when the types differ
    while (v && v->kind == V_CONST && (v->op == LLVMBitCast || v->op == LLVMAddrSpaceCast) && v->nops)
        v = v->ops[0];
    return v;
}

static unsigned attr_bits(LLVMValueRef fn, LLVMAttributeIndex idx)
{
    unsigned bits = 0;
    for (unsigned i = 0; fn->attrs && i < fn->attrs->n; i++)
        if (fn->attrs->at[i].index == idx)
            bits |= fn->attrs->at[i].bits;
    return bits;
}

// the kinds are this reader's ATTR_* bits, not LLVM's numbers
unsigned LLVMGetEnumAttributeKindForName(const char *name, size_t len)
{
    return len == 8 && !memcmp(name, "noinline", 8)   ? ATTR_NOINLINE
           : len == 8 && !memcmp(name, "readonly", 8) ? ATTR_READONLY
           : len == 7 && !memcmp(name, "noalias", 7)  ? ATTR_NOALIAS
                                                      : 0;
}

LLVMAttributeRef LLVMGetEnumAttributeAtIndex(LLVMValueRef fn, LLVMAttributeIndex idx, unsigned kind)
{
    return attr_bits(fn, idx) & kind ? fn : NULL;
}

LLVMAttributeRef LLVMGetStringAttributeAtIndex(LLVMValueRef fn, LLVMAttributeIndex idx, const char *key, unsigned len)
{
    return len == 19 && !memcmp(key, "air-buffer-no-alias", 19) && attr_bits(fn, idx) & ATTR_BUFFER_NO_ALIAS ? fn
                                                                                                            : NULL;
}

LLVMValueRef LLVMGetCondition(LLVMValueRef br)
{
    return br->op == LLVMCondBr ? br->ops[0] : NULL;
}

unsigned LLVMGetNumSuccessors(LLVMValueRef term)
{
    return is_terminator(term->op) ? term->nblocks : 0;
}

LLVMBasicBlockRef LLVMGetSuccessor(LLVMValueRef term, unsigned i)
{
    return i < LLVMGetNumSuccessors(term) ? term->blocks[i] : NULL;
}

LLVMValueRef LLVMGetSwitchCaseValue(LLVMValueRef sw, unsigned i)
{
    return sw->ops[i];
}

unsigned LLVMCountIncoming(LLVMValueRef phi)
{
    return phi->nops;
}

LLVMValueRef LLVMGetIncomingValue(LLVMValueRef phi, unsigned i)
{
    return i < phi->nops ? phi->ops[i] : NULL;
}

LLVMBasicBlockRef LLVMGetIncomingBlock(LLVMValueRef phi, unsigned i)
{
    return i < phi->nblocks ? phi->blocks[i] : NULL;
}
