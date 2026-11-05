// The stages of a tessellated draw before the post-tessellation vertex function, built as IR rather than compiled
// from AIR. A draw has a vertex a patch: the vertex stage hands the patch's draw position and instance on, and the
// tessellation init stage, a thread a control point, fetches the control points and the per-patch data from the
// vertex buffers as the vertex descriptor lays them out (converting each format as the vertex fetch unit would),
// and the patch's tessellation factors from the factor buffer. A kernel's stage input is fetched the same way.
#include <string.h>
#include "airc_int.h"
#include "../core/sph.h"
#include "sass86.h"

// MTLVertexStepFunction's values
enum { STEP_CONSTANT, STEP_PER_VERTEX, STEP_PER_INSTANCE, STEP_PER_PATCH, STEP_PER_PATCH_CONTROL_POINT };

static struct ir_block *g_b;

static struct ir_val *k32(uint32_t imm)
{
    return ir_const(g_b, IR_I32, imm);
}

static struct ir_val *kf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return ir_const(g_b, IR_F32, bits);
}

static struct ir_val *op(enum ir_op o, enum ir_type t, struct ir_val *x, struct ir_val *y)
{
    return ir_op2(g_b, o, t, x, y);
}

static struct ir_val *op1(enum ir_op o, enum ir_type t, struct ir_val *x, int aux)
{
    struct ir_val *v = ir_add(g_b, o, t, 1);
    v->src[0] = x;
    v->aux = aux;
    return v;
}

static struct ir_val *cmp(int cond, struct ir_val *x, struct ir_val *y)
{
    struct ir_val *c = op(IR_CMP, IR_PRED, x, y);
    c->aux = cond;
    return c;
}

static struct ir_val *sel(struct ir_val *c, struct ir_val *x, struct ir_val *y)
{
    struct ir_val *v = ir_add(g_b, IR_SEL, x->type, 3);
    v->src[0] = c;
    v->src[1] = x;
    v->src[2] = y;
    return v;
}

static struct ir_val *s2r(unsigned sr)
{
    struct ir_val *v = ir_add(g_b, IR_S2R, IR_I32, 0);
    v->imm = sr;
    return v;
}

static struct ir_val *cbuf(enum ir_type t, unsigned off)
{
    struct ir_val *v = ir_add(g_b, IR_CBUF, t, 0);
    v->imm = off;
    return v;
}

static struct ir_val *addr(struct ir_val *base, struct ir_val *index, int64_t scale)
{
    struct ir_val *v = op(IR_ADDR, IR_I64, base, index);
    v->imm = scale;
    return v;
}

static struct ir_val *ldg(struct ir_val *at, int64_t off, int bytes)
{
    struct ir_val *v = ir_add(g_b, IR_LDG, IR_I32, 1);
    v->src[0] = at;
    v->imm = off;
    v->aux = bytes;
    return v;
}

// an attribute of the vertex whose handle is vtx, or of the shader's own
static struct ir_val *ald(struct ir_val *vtx, unsigned at)
{
    struct ir_val *v = ir_add(g_b, IR_ALD, IR_I32, vtx ? 1 : 0);
    if (vtx)
        v->src[0] = vtx;
    v->imm = at;
    return v;
}

// n values to the outputs at `at`: of the control point vtx, or the shader's own (flags IR_ATTR_PATCH: the patch's)
static void ast(struct ir_val *vtx, unsigned at, int flags, struct ir_val **v, unsigned n)
{
    struct ir_val *s = ir_add(g_b, IR_AST, IR_VOID, n + (vtx != NULL));
    memcpy(s->src, v, n * sizeof(*v));
    if (vtx)
        s->src[n] = vtx;
    s->imm = at;
    s->aux = (int)n | flags;
}

// n / d for a constant d, exactly, by a multiply (Granlund and Montgomery's round-up method)
struct ir_val *airc_udiv(struct ir_block *b, struct ir_val *n, uint32_t d)
{
    g_b = b;
    if (d <= 1)
        return n;
    unsigned l = 32 - __builtin_clz(d - 1);  // ceil(log2 d)
    uint32_t m = (uint32_t)((1ull << 32) * ((1ull << l) - d) / d + 1);
    struct ir_val *t = op(IR_MULHI, IR_I32, n, k32(m));
    struct ir_val *q = op(IR_ADD, IR_I32, t, op(IR_SHR, IR_I32, op(IR_SUB, IR_I32, n, t), k32(1)));
    return op(IR_SHR, IR_I32, q, k32(l - 1));
}

// the element of a's buffer a control point reads: by the control point's index, the patch id, or the instance (its
// base plus a step every rate instances)
static struct ir_val *element(const struct airc_tess_attr *a, struct ir_val *cp, struct ir_val *pid,
                              struct ir_val *instance, struct ir_val *base)
{
    switch (a->step) {
    case STEP_PER_PATCH_CONTROL_POINT:
        return cp;
    case STEP_PER_PATCH:
        return pid;
    case STEP_PER_INSTANCE:
        return op(IR_ADD, IR_I32, airc_udiv(g_b, instance, a->rate), base);
    default:
        return k32(0);
    }
}

void airc_fetch(struct ir_block *b, const struct airc_tess_attr *a, unsigned loc, struct ir_val *buffer,
                struct ir_val *idx, struct ir_val *v[4])
{
    g_b = b;
    static const struct {
        uint8_t code, n, bits;
    } widths[] = {
        { 0x01, 4, 32 }, { 0x02, 3, 32 }, { 0x04, 2, 32 }, { 0x12, 1, 32 }, { 0x03, 4, 16 }, { 0x05, 3, 16 },
        { 0x0f, 2, 16 }, { 0x1b, 1, 16 }, { 0x0a, 4, 8 },  { 0x13, 3, 8 },  { 0x18, 2, 8 },  { 0x1d, 1, 8 },
        { 0x30, 4, 10 },  // A2B10G10R10, its last component 2 bits
        { 0x31, 3, 11 },  // B10G11R11, its last 10
    };
    enum { SNORM = 1, UNORM = 2, SINT = 3, UINT = 4, FLOAT = 7 };
    unsigned code = a->format >> 21 & 0x3f, type = a->format >> 27 & 7, n = 0, bits = 0;
    for (unsigned i = 0; i < sizeof widths / sizeof widths[0]; i++)
        if (widths[i].code == code) {
            n = widths[i].n;
            bits = widths[i].bits;
        }
    if (!n || !type)
        airc_fail("attribute %u: format %#x", loc, a->format);
    struct ir_val *at = addr(buffer, idx, a->stride);
    struct ir_val *word[4];
    unsigned total = code >= 0x30 ? 32 : n * bits;
    for (unsigned w = 0; w * 32 < total; w++)  // Metal keeps attributes on 4 bytes
        word[w] = ldg(at, a->offset + 4 * w, 4);
    int is_signed = type == SINT || type == SNORM;
    for (unsigned c = 0; c < 4; c++) {
        if (c >= n) {
            v[c] = c < 3 ? k32(0) : type == SINT || type == UINT ? k32(1) : kf(1.0f);
            continue;
        }
        unsigned first = code == 0x30 ? 10 * c : bits * c;
        unsigned size = (code == 0x30 && c == 3) ? 2 : (code == 0x31 && c == 2) ? 10 : bits, sh = first % 32;
        struct ir_val *x = word[first / 32];
        if (size < 32 && is_signed) {  // the field to the top, then back down arithmetically
            x = op(IR_SAR, IR_I32, op(IR_SHL, IR_I32, x, k32(32 - sh - size)), k32(32 - size));
        } else if (size < 32) {
            if (sh)
                x = op(IR_SHR, IR_I32, x, k32(sh));
            if (sh + size < 32)
                x = op(IR_AND, IR_I32, x, k32((1u << size) - 1));
        }
        if (type == FLOAT && size < 32)  // 11- and 10-bit floats are halves with fewer mantissa bits
            x = op1(IR_H2F, IR_F32, size < 16 ? op(IR_SHL, IR_I32, x, k32(15 - size)) : x, 0);
        else if (type == UNORM)
            x = op(IR_FMUL, IR_F32, op1(IR_I2F, IR_F32, x, 0), kf(1.0f / (float)((1ull << size) - 1)));
        else if (type == SNORM)
            x = op(IR_FMAX, IR_F32,
                   op(IR_FMUL, IR_F32, op1(IR_I2F, IR_F32, x, 1), kf(1.0f / (float)((1u << (size - 1)) - 1))),
                   kf(-1.0f));
        v[c] = x;
    }
    if (a->format >> 31) {  // BGRA
        struct ir_val *r = v[0];
        v[0] = v[2];
        v[2] = r;
    }
}

static void fetch(const struct airc_tess_attr *a, unsigned loc, struct ir_val *idx, struct ir_val *v[4])
{
    if (a->buffer >= AIRC_TESS_BUFFERS)
        airc_fail("attribute %u: buffer %u", loc, a->buffer);
    airc_fetch(g_b, a, loc, cbuf(IR_I64, AIRC_TESS_BUFFER(a->buffer)), idx, v);
}

// generic 0: the patch's draw position (vertex ids count from the first vertex drawn), its instance from 0 and the
// base instance
static void vertex(const void *arg, struct ir_func *f, struct airc_info *info)
{
    (void)arg;
    g_b = ir_block(f);
    info->stage = NVK_VERTEX;
    info->ngeneric = 1;
    info->sysvals = SPH_VERTEX_ID | SPH_INSTANCE_ID;
    struct ir_val *zero = kf(0.0f), *position[4] = { zero, zero, zero, zero };
    struct ir_val *ids[4] = { ald(NULL, 0x2fc), ald(NULL, 0x2f8), cbuf(IR_I32, NVK_BASE_INSTANCE), k32(0) };
    ast(NULL, 0x70, 0, position, 4);  // as its header says it does
    ast(NULL, 0x80, 0, ids, 4);
    ir_add(g_b, IR_EXIT, IR_VOID, 0);
}

static void tess_init(const void *arg, struct ir_func *f, struct airc_info *info)
{
    const struct airc_tess_desc *d = arg;
    if (!d->points || d->points > 32)
        airc_fail("patches of %u control points", d->points);
    g_b = ir_block(f);
    info->stage = NVK_TESS_INIT;
    info->vertex_inputs = 1;
    info->point_outputs = d->point_inputs;
    info->patch_outputs = 1 + __builtin_popcount(d->patch_inputs);
    info->patch_points = d->points;

    // the patch's vertex, where the patch's vertices start in the input map: bytes 0 and 2 of INVOCATION_INFO
    // multiplied
    struct ir_val *map = s2r(SR_INVOCATION_INFO);
    struct ir_val *start = op(IR_MUL, IR_I32, op(IR_AND, IR_I32, map, k32(0xff)),
                              op(IR_AND, IR_I32, op(IR_SHR, IR_I32, map, k32(16)), k32(0xff)));
    struct ir_val *vtx = op1(IR_ISBERD, IR_I32, start, 0);
    struct ir_val *pos = ald(vtx, 0x80), *instance = ald(vtx, 0x84), *base = ald(vtx, 0x88);
    struct ir_val *iid = op(IR_ADD, IR_I32, instance, base), *point = s2r(SR_INVOCATION_ID);

    // the patch id: the patch index buffer's at the draw position, or the position (the load then reads the factor
    // buffer, which a draw always has)
    struct ir_val *factors = cbuf(IR_I64, AIRC_TESS_FACTORS);
    struct ir_val *indexed = cmp(IR_NE, op(IR_OR, IR_I32, cbuf(IR_I32, AIRC_TESS_PATCH_INDEX),
                                          cbuf(IR_I32, AIRC_TESS_PATCH_INDEX + 4)), k32(0));
    struct ir_val *at = sel(indexed, addr(cbuf(IR_I64, AIRC_TESS_PATCH_INDEX), pos, 4), factors);
    struct ir_val *pid = sel(indexed, ldg(at, 0, 4), pos);
    struct ir_val *cp = op(IR_ADD, IR_I32, op(IR_MUL, IR_I32, pid, k32(d->points)), point);
    if (d->index_size)
        cp = ldg(addr(cbuf(IR_I64, AIRC_TESS_POINT_INDEX), cp, d->index_size), 0, (int)d->index_size);

    struct ir_val *v[4];
    for (unsigned loc = 0; loc < 32; loc++)
        if (d->point_inputs >> loc & 1) {
            fetch(&d->attrs[loc], loc, element(&d->attrs[loc], cp, pid, instance, base), v);
            ast(point, 0x80 + 16 * loc, 0, v, 4);
        }
    struct ir_val *ids[2] = { pid, iid };
    ast(NULL, NVK_PATCH_IDS, IR_ATTR_PATCH, ids, 2);
    for (unsigned loc = 0; loc < 32; loc++)
        if (d->patch_inputs >> loc & 1) {
            fetch(&d->attrs[loc], loc, element(&d->attrs[loc], cp, pid, instance, base), v);
            ast(NULL, NVK_PATCH_INPUT(d->patch_inputs, loc), IR_ATTR_PATCH, v, 4);
        }

    // the factors, halves: the edges' then the inside's, 8 bytes a patch for triangles and 12 for quads; scaled,
    // clamped to the maximum (a NaN, which culls the patch, kept) and rounded up to a power of two for pow2
    unsigned edges = d->domain ? 4 : 3, insides = d->domain ? 2 : 1;
    struct ir_val *fa = factors, *level[6], *word = NULL;
    if (d->factor_step & 1)
        fa = addr(fa, pos, d->domain ? 12 : 8);
    if (d->factor_step & 2)
        fa = addr(fa, op(IR_MUL, IR_I32, iid, cbuf(IR_I32, AIRC_TESS_INSTANCE_STRIDE)), 1);
    for (unsigned i = 0; i < edges + insides; i++) {
        if (!(i & 1))
            word = ldg(fa, 2 * i, 4);
        struct ir_val *x = op1(IR_H2F, IR_F32, i & 1 ? op(IR_SHR, IR_I32, word, k32(16)) : word, 0);
        x = op(IR_FMUL, IR_F32, x, cbuf(IR_F32, AIRC_TESS_SCALE));
        struct ir_val *max = kf(d->max_factor), *over = cmp(IR_FCMP | 2, x, max);  // ordered >
        x = sel(over, max, x);
        if (d->pow2) {
            struct ir_val *up = op1(IR_FRND, IR_F32, op1(IR_MUFU, IR_F32, x, MUFU_LG2), SASS_RND_CEIL);
            x = op1(IR_MUFU, IR_F32, up, MUFU_EX2);
        }
        level[i] = x;
    }
    struct ir_val *outer[4] = { level[0], level[1], level[2], d->domain ? level[3] : kf(0.0f) };
    struct ir_val *inner[2] = { level[edges], d->domain ? level[edges + 1] : kf(0.0f) };
    ast(NULL, 0x00, IR_ATTR_PATCH, outer, 4);
    ast(NULL, 0x10, IR_ATTR_PATCH, inner, 2);
    ir_add(g_b, IR_EXIT, IR_VOID, 0);
}

int airc_tess_vertex(struct airc_kernel *k)
{
    return airc_generate(vertex, NULL, k);
}

int airc_tess_init(const struct airc_tess_desc *d, struct airc_kernel *k)
{
    return airc_generate(tess_init, d, k);
}
