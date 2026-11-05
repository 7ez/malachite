// airc's front end: an AIR module (LLVM bitcode) -> scalar SSA IR. Vectors, structs and arrays held in values
// become one IR value per lane; calls to the module's own functions are inlined; the function constants the
// static initialiser copies around are followed to their specialised values; pointers stay symbolic (a base
// value and a constant offset) until a load or store uses them.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "airc_int.h"
#include "../core/sph.h"
#include "sass86.h"
#include "sass86.h"
#ifdef AIRC_LIBLLVM
#include <llvm-c/Core.h>
static int can_contract(LLVMValueRef v)
{
    return LLVMCanValueUseFastMathFlags(v) && (LLVMGetFastMathFlags(v) & LLVMFastMathAllowContract);
}
#else
#include "llvm_lite.h"
#define can_contract LLVMCanContract
#endif

#define PARAM_BASE 0x160  // sm_86 driver ABI: kernel params start here in const bank 0
#define MAX_INLINE_DEPTH 16

enum space { SP_GLOBAL, SP_SHARED, SP_TEXTURE, SP_SAMPLER, SP_PRIVATE, SP_LOCAL };

struct ptr {
    enum space space;
    struct ir_val *base;  // IR_I64 for global, IR_I32 (or NULL for 0) for shared and local
    int64_t off;
    unsigned slot;  // texture / sampler pool slot; for private memory, which alloca; for local, where it ends
    // private memory indexed by a variable: base is the index into count elements stride bytes apart
    unsigned stride, count;
    unsigned bytes;  // private memory: the size of the alloca or constant table it's in, 0 when not known
    struct sub_param *param;  // a subroutine's texture or sampler parameter: base is its handle
};

// value, pointer and block maps, keyed by LLVM object and the inlined instance (scope) it belongs to
struct entry {
    const void *key;
    unsigned lane, scope;
    struct entry *next;
    union {
        struct ir_val *v;
        struct ptr p;
        struct ir_block *blocks[2];  // head (where the LLVM block starts), tail (where it ends)
        LLVMBasicBlockRef bb;        // for an IR block that branches on to an LLVM block's successors: that block
    };
};

#define HASH_BITS 18
static struct entry *g_vals[1 << HASH_BITS], *g_ptrs[1 << HASH_BITS], *g_blocks[1 << HASH_BITS],
    *g_origins[1 << HASH_BITS];
// the high words of 64-bit integers where they're known (a value itself keeps the low one): loads, extensions,
// and shifts and bitwise operations of those
static struct entry *g_highs[1 << HASH_BITS];

static unsigned hash(const void *key, unsigned lane, unsigned scope)
{
    uintptr_t k = (uintptr_t)key;
    return (unsigned)((k >> 4) * 2654435761u + lane * 40503u + scope * 9973u) & ((1 << HASH_BITS) - 1);
}

static struct entry *find(struct entry **table, const void *key, unsigned lane, unsigned scope)
{
    for (struct entry *e = table[hash(key, lane, scope)]; e; e = e->next)
        if (e->key == key && e->lane == lane && e->scope == scope)
            return e;
    return NULL;
}

static struct entry *insert(struct entry **table, const void *key, unsigned lane, unsigned scope)
{
    struct entry *e = find(table, key, lane, scope);
    if (e)
        return e;
    e = ir_alloc(sizeof(*e));
    e->key = key;
    e->lane = lane;
    e->scope = scope;
    unsigned h = hash(key, lane, scope);
    e->next = table[h];
    table[h] = e;
    return e;
}

static struct ir_func *g_f;
static struct airc_info *g_info;
static LLVMModuleRef g_module;
static LLVMValueRef g_entry_fn;
static LLVMModuleRef *g_linked;  // the pipeline's linked functions' modules
static const struct airc_varyings *g_vertex;  // the vertex function's outputs a fragment function's inputs match
static unsigned g_nlinked;
static const char *const *g_linked_names;  // the names callers know each by, NULL where it's its own
static struct ir_block *g_prologue;  // the entry block: arguments, builtins and constants are made here
static struct ir_block *g_cur;
static unsigned g_scope, g_next_scope, g_depth;
static const struct airc_constant *g_fc;
static unsigned g_nfc;
static struct ir_val *g_rcp_w[2];  // 1/w at the pixel's centre, and at its centroid (in a pass of a sample, the sample)
static unsigned g_nallocas;  // private memory: allocas seen (each a slot)
// a post-tessellation vertex function's control points: the function it gets one from, the location of each member
// of what that returns, and where the patch's vertices start in the input map
static LLVMValueRef g_point_fn;
static uint8_t g_point_loc[16];
static unsigned g_npoint_loc;
static struct ir_val *g_point_map;
// a kernel's stage input: how its attributes are laid out, the params its region's address and each buffer it
// reads are at (0 for none), and the region's words as read so far
static const struct airc_stage_in *g_stage_in;
static unsigned g_region_off, g_stage_in_buffer[31];
static struct ir_val *g_region[6];

// threadgroup variables, laid out one after another (16-byte aligned) from shared address 0
static struct {
    LLVMValueRef g;
    unsigned off;
} g_shared[256];
static unsigned g_nshared;

static int is_ptr(LLVMTypeRef t)
{
    return t && LLVMGetTypeKind(t) == LLVMPointerTypeKind;
}

static unsigned nlanes(LLVMTypeRef t)
{
    if (!t)
        return 1;
    switch (LLVMGetTypeKind(t)) {
    case LLVMVectorTypeKind:
        return LLVMGetVectorSize(t);
    case LLVMArrayTypeKind:
        return (unsigned)LLVMGetArrayLength2(t) * nlanes(LLVMGetElementType(t));
    case LLVMStructTypeKind: {
        unsigned n = 0;
        for (unsigned i = 0; i < LLVMCountStructElementTypes(t); i++)
            n += nlanes(LLVMStructGetTypeAtIndex(t, i));
        return n;
    }
    case LLVMVoidTypeKind:
        return 0;
    default:
        return 1;
    }
}

static LLVMTypeRef lane_type(LLVMTypeRef t, unsigned lane)
{
    for (;;) {
        switch (LLVMGetTypeKind(t)) {
        case LLVMVectorTypeKind:
            return LLVMGetElementType(t);
        case LLVMArrayTypeKind: {
            LLVMTypeRef e = LLVMGetElementType(t);
            lane %= nlanes(e);
            t = e;
            continue;
        }
        case LLVMStructTypeKind:
            for (unsigned i = 0;; i++) {
                LLVMTypeRef m = LLVMStructGetTypeAtIndex(t, i);
                if (lane < nlanes(m)) {
                    t = m;
                    break;
                }
                lane -= nlanes(m);
            }
            continue;
        default:
            return t;
        }
    }
}

static enum ir_type ir_type_of(LLVMTypeRef t)
{
    switch (LLVMGetTypeKind(t)) {
    case LLVMFloatTypeKind:
        return IR_F32;
    case LLVMHalfTypeKind:
        return IR_F16;
    case LLVMBFloatTypeKind:  // its bits, as a 16-bit integer's: only conversions and moves are done on it
        return IR_I32;
    case LLVMIntegerTypeKind:
        // wider integers are index arithmetic in practice; they keep their low 32 bits
        return LLVMGetIntTypeWidth(t) == 1 ? IR_PRED : IR_I32;
    default:
        airc_fail("unsupported value type %d", LLVMGetTypeKind(t));
    }
}

static unsigned scalar_bytes(LLVMTypeRef t)
{
    switch (LLVMGetTypeKind(t)) {
    case LLVMFloatTypeKind:
        return 4;
    case LLVMHalfTypeKind:
    case LLVMBFloatTypeKind:
        return 2;
    case LLVMIntegerTypeKind:
        return (LLVMGetIntTypeWidth(t) + 7) / 8;
    case LLVMPointerTypeKind:
        return 8;
    default:
        return 0;
    }
}

// Size and alignment of a type in memory, laid out the way Metal lays out its vectors, structs and arrays
// (a 3-lane vector takes a 4-lane slot).
static unsigned layout(LLVMTypeRef t, unsigned *align)
{
    unsigned a = 1, size;
    switch (LLVMGetTypeKind(t)) {
    case LLVMStructTypeKind:
        size = 0;
        for (unsigned i = 0; i < LLVMCountStructElementTypes(t); i++) {
            unsigned ma, ms = layout(LLVMStructGetTypeAtIndex(t, i), &ma);
            size = (size + ma - 1) & ~(ma - 1);
            size += ms;
            if (ma > a)
                a = ma;
        }
        size = (size + a - 1) & ~(a - 1);
        break;
    case LLVMArrayTypeKind:
        size = (unsigned)LLVMGetArrayLength2(t) * layout(LLVMGetElementType(t), &a);
        break;
    case LLVMVectorTypeKind: {
        unsigned n = LLVMGetVectorSize(t);
        size = a = (n == 3 ? 4 : n) * scalar_bytes(LLVMGetElementType(t));
        break;
    }
    default:
        size = a = scalar_bytes(t);
        if (!size)
            airc_fail("unsupported type in memory");
    }
    *align = a;
    return size;
}

// byte offset of field k of struct type t
static unsigned field_offset(LLVMTypeRef t, unsigned k)
{
    unsigned a, off = 0;
    for (unsigned f = 0; f <= k; f++) {
        unsigned fs = layout(LLVMStructGetTypeAtIndex(t, f), &a);
        off = (off + a - 1) & ~(a - 1);
        if (f < k)
            off += fs;
    }
    return off;
}

// byte offset of flat lane `lane` within a value of type t in memory
static unsigned lane_offset(LLVMTypeRef t, unsigned lane)
{
    unsigned a;
    switch (LLVMGetTypeKind(t)) {
    case LLVMVectorTypeKind:
        return lane * scalar_bytes(LLVMGetElementType(t));
    case LLVMArrayTypeKind: {
        LLVMTypeRef e = LLVMGetElementType(t);
        unsigned n = nlanes(e);
        return lane / n * layout(e, &a) + lane_offset(e, lane % n);
    }
    case LLVMStructTypeKind: {
        unsigned off = 0;
        for (unsigned i = 0;; i++) {
            LLVMTypeRef m = LLVMStructGetTypeAtIndex(t, i);
            unsigned ms = layout(m, &a);
            off = (off + a - 1) & ~(a - 1);
            if (lane < nlanes(m))
                return off + lane_offset(m, lane);
            lane -= nlanes(m);
            off += ms;
        }
    }
    default:
        return 0;
    }
}

// IEEE half bits of f, rounded to nearest even
static uint32_t half_bits(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, man = x & 0x7fffff;
    if (e == 0xff)
        return sign | 0x7c00 | (man ? 0x200 : 0);
    int he = (int)e - 127 + 15;
    if (he >= 31)
        return sign | 0x7c00;
    if (he <= 0) {
        if (he < -10)
            return sign;
        man |= 0x800000;
        unsigned shift = (unsigned)(14 - he);
        uint32_t h = man >> shift, rest = man & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rest > halfway || (rest == halfway && (h & 1)))
            h++;
        return sign | h;
    }
    uint32_t h = sign | ((uint32_t)he << 10) | (man >> 13), rest = man & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (h & 1)))
        h++;
    return h;
}

static struct ir_val *konst(enum ir_type t, int64_t imm)
{
    return ir_const(g_prologue, t, imm);
}

static struct ir_val *fconst(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return konst(IR_F32, bits);
}

static struct ir_val *op1(enum ir_op op, enum ir_type t, struct ir_val *x)
{
    struct ir_val *v = ir_add(g_cur, op, t, 1);
    v->src[0] = x;
    return v;
}

static struct ir_val *op2(enum ir_op op, enum ir_type t, struct ir_val *x, struct ir_val *y)
{
    return ir_op2(g_cur, op, t, x, y);
}

static struct ir_val *op3(enum ir_op op, enum ir_type t, struct ir_val *x, struct ir_val *y, struct ir_val *z)
{
    struct ir_val *v = ir_add(g_cur, op, t, 3);
    v->src[0] = x;
    v->src[1] = y;
    v->src[2] = z;
    return v;
}

static void set(LLVMValueRef key, unsigned lane, struct ir_val *v)
{
    insert(g_vals, key, lane, g_scope)->v = v;
}

// lane `lane` of a constant (vector, struct or array constants hold their lanes flattened)
static struct ir_val *const_lane(LLVMValueRef c, unsigned lane)
{
    LLVMTypeRef t = LLVMTypeOf(c);
    // a splat (LLVM's vector-typed ConstantInt/ConstantFP) has the same value in every lane
    if ((LLVMIsAConstantInt(c) || LLVMIsAConstantFP(c)) && LLVMGetTypeKind(t) == LLVMVectorTypeKind)
        t = LLVMGetElementType(t);
    if (LLVMIsAConstantInt(c)) {
        enum ir_type it = ir_type_of(t);
        return konst(it, it == IR_PRED ? (int64_t)(LLVMConstIntGetZExtValue(c) & 1)
                                       : (int64_t)(uint32_t)LLVMConstIntGetZExtValue(c));
    }
    if (LLVMIsAConstantFP(c)) {
        LLVMBool lost;
        float f = (float)LLVMConstRealGetDouble(c, &lost);
        uint32_t bits;
        memcpy(&bits, &f, 4);
        return konst(ir_type_of(t), LLVMGetTypeKind(t) == LLVMHalfTypeKind     ? half_bits(f)
                                    : LLVMGetTypeKind(t) == LLVMBFloatTypeKind ? bits >> 16
                                                                               : bits);
    }
    LLVMTypeKind k = LLVMGetTypeKind(t);
    if (k == LLVMVectorTypeKind || k == LLVMArrayTypeKind || k == LLVMStructTypeKind) {
        unsigned n = k == LLVMVectorTypeKind ? LLVMGetVectorSize(t)
                     : k == LLVMArrayTypeKind ? (unsigned)LLVMGetArrayLength2(t)
                                              : LLVMCountStructElementTypes(t);
        for (unsigned i = 0; i < n; i++) {
            LLVMTypeRef et = k == LLVMStructTypeKind ? LLVMStructGetTypeAtIndex(t, i) : LLVMGetElementType(t);
            unsigned m = nlanes(et);
            if (lane < m) {
                LLVMValueRef e = LLVMIsUndef(c) ? NULL : LLVMGetAggregateElement(c, i);
                return e ? const_lane(e, lane) : konst(ir_type_of(lane_type(et, lane)), 0);
            }
            lane -= m;
        }
        airc_fail("constant lane out of range");
    }
    // undef, poison, zeroinitializer of a scalar
    return konst(ir_type_of(t), 0);
}

static struct ir_val *get(LLVMValueRef v, unsigned lane)
{
    if (nlanes(LLVMTypeOf(v)) == 1)
        lane = 0;
    struct entry *e = find(g_vals, v, lane, g_scope);
    if (e)
        return e->v;
    // parameters are mapped up front, so anything else that isn't an instruction is a constant
    if (!LLVMIsAInstruction(v) && !LLVMIsAFunction(v)) {
        struct ir_val *c = const_lane(v, lane);
        set(v, lane, c);
        return c;
    }
    airc_fail("a value is used before it is defined (lane %u)", lane);
}

static int is_i64(LLVMTypeRef t)
{
    return LLVMGetTypeKind(t) == LLVMIntegerTypeKind && LLVMGetIntTypeWidth(t) == 64;
}

// the high word of lane `lane` of a constant 64-bit integer (a vector's lanes are flattened like const_lane's)
static struct ir_val *const_high(LLVMValueRef c, unsigned lane)
{
    LLVMTypeRef t = LLVMTypeOf(c);
    if (LLVMIsAConstantInt(c))
        return konst(IR_I32, (int64_t)(uint32_t)(LLVMConstIntGetZExtValue(c) >> 32));
    if (LLVMIsUndef(c) || LLVMIsPoison(c))
        return konst(IR_I32, 0);
    if (LLVMGetTypeKind(t) == LLVMVectorTypeKind) {
        LLVMValueRef e = LLVMGetAggregateElement(c, lane);
        if (e && LLVMIsAConstantInt(e))
            return konst(IR_I32, (int64_t)(uint32_t)(LLVMConstIntGetZExtValue(e) >> 32));
    }
    return NULL;
}

// lane `lane` of a 64-bit integer value's high word, or NULL when it isn't known
static struct ir_val *high_word(LLVMValueRef v, unsigned lane)
{
    if (nlanes(LLVMTypeOf(v)) == 1)
        lane = 0;
    struct entry *e = find(g_highs, v, lane, g_scope);
    if (e)
        return e->v;
    return !LLVMIsAInstruction(v) && !LLVMIsAFunction(v) ? const_high(v, lane) : NULL;
}

static void set_high(LLVMValueRef v, unsigned lane, struct ir_val *hi)
{
    insert(g_highs, v, lane, g_scope)->v = hi;
}

// high_word where a missing one reads as 0, for the places that only move words about
static struct ir_val *high_or_zero(LLVMValueRef v, unsigned lane)
{
    struct ir_val *hi = high_word(v, lane);
    return hi ? hi : konst(IR_I32, 0);
}


static unsigned shared_offset(LLVMValueRef g)
{
    for (unsigned i = 0; i < g_nshared; i++)
        if (g_shared[i].g == g)
            return g_shared[i].off;
    unsigned a, bytes = layout(LLVMGlobalGetValueType(g), &a);
    if (g_nshared == sizeof g_shared / sizeof g_shared[0])
        airc_fail("too many threadgroup variables");
    unsigned off = (g_info->shared_size + 15) & ~15u;
    g_shared[g_nshared].g = g;
    g_shared[g_nshared++].off = off;
    g_info->shared_size = off + bytes;
    return off;
}

// a constexpr sampler: a constant global holding Metal's packed sampler state, alone or as a struct's first
// member; slots from the top of the pool
static unsigned const_sampler(LLVMValueRef g)
{
    LLVMValueRef init = LLVMGetInitializer(g);
    LLVMValueRef word = init && LLVMIsAConstantInt(init) ? init : init ? LLVMGetAggregateElement(init, 0) : NULL;
    if (!word || !LLVMIsAConstantInt(word))
        airc_fail("unsupported constexpr sampler");
    uint64_t bits = LLVMConstIntGetZExtValue(word);
    unsigned i = 0;
    while (i < g_info->nconst_samplers && g_info->const_samplers[i] != bits)
        i++;
    if (i == NVK_MAX_CONST_SAMPLERS)
        airc_fail("more than %d constexpr samplers", NVK_MAX_CONST_SAMPLERS);
    g_info->const_samplers[i] = bits;
    if (i == g_info->nconst_samplers)
        g_info->nconst_samplers++;
    return NVK_CONST_SAMPLER_SLOT(i);
}

static struct ptr ptr_of(LLVMValueRef v);
static struct ir_val *int_extend(struct ir_val *v, unsigned bits, int is_signed);

// signed when the index was sign-extended to the GEP's width
static struct ir_val *index_value(LLVMValueRef idx, int *is_signed)
{
    *is_signed = 1;
    if (LLVMIsAInstruction(idx)) {
        LLVMOpcode op = LLVMGetInstructionOpcode(idx);
        if (op == LLVMZExt || op == LLVMSExt) {
            // a narrower integer's bits above its own are junk (a truncation keeps them)
            LLVMValueRef x = LLVMGetOperand(idx, 0);
            *is_signed = op == LLVMSExt;
            return int_extend(get(x, 0), 8 * scalar_bytes(LLVMTypeOf(x)), *is_signed);
        }
    }
    return get(idx, 0);
}

// count: how many elements the index picks from, 0 if unknown
static struct ptr add_index(struct ptr p, LLVMValueRef idx, unsigned scale, unsigned count)
{
    if (LLVMIsAConstantInt(idx)) {
        p.off += (int64_t)LLVMConstIntGetZExtValue(idx) * scale;  // wraps right for negative i64 indices
        return p;
    }
    int is_signed;
    struct ir_val *i = index_value(idx, &is_signed);
    if (p.space == SP_GLOBAL) {
        struct ir_val *a = op2(IR_ADDR, IR_I64, p.base, i);
        a->imm = scale;
        a->aux = is_signed;
        p.base = a;
    } else if (p.space == SP_SHARED || p.space == SP_LOCAL) {
        if (p.space == SP_LOCAL && count) {  // past the array's end faults, where Metal leaves it undefined: the last
            i = op2(IR_MIN, IR_I32, i, konst(IR_I32, count - 1));
            i->aux = 0;  // unsigned: a negative index is past the end too
        }
        struct ir_val *s = scale == 1 ? i : op2(IR_MUL, IR_I32, i, konst(IR_I32, scale));
        p.base = p.base ? op2(IR_ADD, IR_I32, p.base, s) : s;
    } else if (p.space == SP_PRIVATE) {
        if (!count && p.bytes > p.off && scale)  // a pointer stepping through it: the elements from here on
            count = (unsigned)(p.bytes - p.off) / scale;
        if (p.base && count && p.stride == count * scale && p.count * count <= 128) {
            // an array of arrays picked twice: one index into all the elements, row by row
            p.base = op2(IR_ADD, IR_I32, op2(IR_MUL, IR_I32, p.base, konst(IR_I32, count)), i);
            p.stride = scale;
            p.count *= count;
            return p;
        }
        if (p.base || !count || count > 64)
            airc_fail("a private array indexed by a variable");
        p.base = i;
        p.stride = scale;
        p.count = count;
    } else if (!p.base && !p.param && scale == 8) {  // an array of textures or samplers: the element
        unsigned n = count ? count : p.count;
        p.base = i;
        if (n) {  // an index past the end is undefined in Metal; here it stays on the last
            p.base = op2(IR_MIN, IR_I32, i, konst(IR_I32, n - 1));
            p.base->aux = 0;
        }
    } else {
        airc_fail("indexing into a texture or sampler");
    }
    return p;
}

static struct ptr gep(LLVMValueRef g)
{
    struct ptr p = ptr_of(LLVMGetOperand(g, 0));
    LLVMTypeRef t = LLVMGetGEPSourceElementType(g);
    unsigned a;
    p = add_index(p, LLVMGetOperand(g, 1), layout(t, &a), 0);
    for (int i = 2; i < LLVMGetNumOperands(g); i++) {
        LLVMValueRef idx = LLVMGetOperand(g, i);
        if (LLVMGetTypeKind(t) == LLVMStructTypeKind) {
            if (!LLVMIsAConstantInt(idx))
                airc_fail("struct field index is not a constant");
            unsigned k = (unsigned)LLVMConstIntGetZExtValue(idx);
            p.off += field_offset(t, k);
            t = LLVMStructGetTypeAtIndex(t, k);
        } else {
            unsigned count = LLVMGetTypeKind(t) == LLVMArrayTypeKind ? (unsigned)LLVMGetArrayLength2(t) : 0;
            t = LLVMGetElementType(t);
            p = add_index(p, idx, layout(t, &a), count);
        }
    }
    return p;
}

// Private memory goes to the thread's local memory (LDL/STL) rather than registers when it's indexed by a variable
// (a SEL over every element at each access otherwise) or read in other pieces than it was written in (MPS moves its
// reduction state through it as bytes, then reads it as words): through calls too, as they're inlined. Fields of
// different sizes side by side are fine (MPS's array structs hold a pointer, a 64-bit count and a 32-bit rank).
static LLVMValueRef ptr_root(LLVMValueRef v)
{
    while (LLVMIsAInstruction(v) && (LLVMGetInstructionOpcode(v) == LLVMGetElementPtr ||
                                     LLVMGetInstructionOpcode(v) == LLVMBitCast ||
                                     LLVMGetInstructionOpcode(v) == LLVMAddrSpaceCast))
        v = LLVMGetOperand(v, 0);
    return v;
}

// v's byte offset from ptr_root(v); one with a variable index counts as 0, as scan_uses takes it as dynamic anyway
static int64_t root_offset(LLVMValueRef v)
{
    int64_t off = 0;
    for (; v != ptr_root(v); v = LLVMGetOperand(v, 0)) {
        if (LLVMGetInstructionOpcode(v) != LLVMGetElementPtr)
            continue;
        LLVMTypeRef t = LLVMGetGEPSourceElementType(v);
        unsigned a;
        for (int i = 1; i < LLVMGetNumOperands(v); i++) {
            LLVMValueRef idx = LLVMGetOperand(v, i);
            if (!LLVMIsAConstantInt(idx))
                return 0;
            unsigned shift = 64 - LLVMGetIntTypeWidth(LLVMTypeOf(idx));
            int64_t k = (int64_t)(LLVMConstIntGetZExtValue(idx) << shift) >> shift;
            if (i == 1) {
                off += k * layout(t, &a);
            } else if (LLVMGetTypeKind(t) == LLVMStructTypeKind) {
                off += field_offset(t, (unsigned)k);
                t = LLVMStructGetTypeAtIndex(t, (unsigned)k);
            } else {
                t = LLVMGetElementType(t);
                off += k * layout(t, &a);
            }
        }
    }
    return off;
}

// The pieces of private memory accessed, each as the variable it is: a lane, or a word of a 64-bit integer, which
// values keep as two
struct piece {
    int64_t off;
    unsigned bytes;
};

struct pieces {
    struct piece *p;
    unsigned n, cap;
    unsigned sizes;  // of the lanes, a bit each
    int dynamic;     // indexed by a variable, or moved as bytes by a memcpy
    int escapes;     // its address goes where the scan doesn't follow it
};

static void add_pieces(struct pieces *s, LLVMTypeRef t, int64_t off)
{
    for (unsigned l = 0; l < nlanes(t); l++) {
        LLVMTypeRef st = lane_type(t, l);
        unsigned bytes = scalar_bytes(st);
        s->sizes |= bytes >= 8 ? 8 : bytes;
        unsigned step = LLVMGetTypeKind(st) == LLVMIntegerTypeKind && bytes > 4 ? 4 : bytes;
        for (unsigned w = 0; w < bytes; w += step) {
            struct piece q = { off + lane_offset(t, l) + w, step };
            unsigned i = 0;
            while (i < s->n && (s->p[i].off != q.off || s->p[i].bytes != q.bytes))
                i++;
            if (i < s->n)
                continue;
            if (s->n == s->cap) {
                s->cap = s->cap ? 2 * s->cap : 32;
                s->p = realloc(s->p, s->cap * sizeof(*s->p));
            }
            s->p[s->n++] = q;
        }
    }
}

// two pieces that overlap without being the same variable
static int pieces_clash(const struct pieces *s)
{
    for (unsigned i = 0; i < s->n; i++)
        for (unsigned j = i + 1; j < s->n; j++)
            if (s->p[i].off < s->p[j].off + s->p[j].bytes && s->p[j].off < s->p[i].off + s->p[i].bytes)
                return 1;
    return 0;
}

static LLVMValueRef linked_function(const char *name);

static int rooted_at(LLVMValueRef v, LLVMValueRef root)
{
    return is_ptr(LLVMTypeOf(v)) && ptr_root(v) == root;
}

// whether alloca b is only loaded from and stored to in fn, at constant offsets: every load of what it holds is in
// sight then
static int stays_put(LLVMValueRef fn, LLVMValueRef b)
{
    for (LLVMBasicBlockRef bb = LLVMGetFirstBasicBlock(fn); bb; bb = LLVMGetNextBasicBlock(bb))
        for (LLVMValueRef in = LLVMGetFirstInstruction(bb); in; in = LLVMGetNextInstruction(in)) {
            LLVMOpcode op = LLVMGetInstructionOpcode(in);
            LLVMValueRef callee = op == LLVMCall ? LLVMGetCalledValue(in) : NULL;
            const char *name = callee ? LLVMGetValueName(callee) : NULL;
            for (int i = 0; i < LLVMGetNumOperands(in); i++) {
                if (!rooted_at(LLVMGetOperand(in, i), b) || op == LLVMLoad || op == LLVMBitCast ||
                    op == LLVMAddrSpaceCast || (op == LLVMStore && i == 1) ||
                    (name && !strncmp(name, "llvm.lifetime.", 14)))
                    continue;
                if (op != LLVMGetElementPtr)
                    return 0;
                for (int k = 1; k < LLVMGetNumOperands(in); k++)
                    if (!LLVMIsAConstantInt(LLVMGetOperand(in, k)))
                        return 0;
            }
        }
    return 1;
}

static void scan_uses(LLVMValueRef fn, LLVMValueRef root, int64_t base, struct pieces *s, unsigned depth);

// A pointer at bytes into the alloca, stored into fn's memory at y (MPS's visible functions spill the struct pointer
// they're given to the stack): what's done through each load of it back. 0 when not every load can be seen.
static int follow_spill(LLVMValueRef fn, LLVMValueRef y, int64_t at, struct pieces *s, unsigned depth)
{
    LLVMValueRef b = ptr_root(y);
    int64_t off = root_offset(y);
    if (!LLVMIsAInstruction(b) || LLVMGetInstructionOpcode(b) != LLVMAlloca || !stays_put(fn, b))
        return 0;
    for (LLVMBasicBlockRef bb = LLVMGetFirstBasicBlock(fn); bb; bb = LLVMGetNextBasicBlock(bb))
        for (LLVMValueRef in = LLVMGetFirstInstruction(bb); in; in = LLVMGetNextInstruction(in)) {
            if (LLVMGetInstructionOpcode(in) != LLVMLoad || ptr_root(LLVMGetOperand(in, 0)) != b)
                continue;
            unsigned a;
            int64_t o = root_offset(LLVMGetOperand(in, 0));
            if (o == off && is_ptr(LLVMTypeOf(in)))
                scan_uses(fn, in, at, s, depth + 1);
            else if (o < off + 8 && off < o + layout(LLVMTypeOf(in), &a))
                return 0;  // the pointer read as something else
        }
    return 1;
}

// what fn does through pointers rooted at root, which is base bytes into the alloca: the pieces it loads and stores,
// whether it indexes them by a variable, and whether the address goes somewhere else (stored where its loads can't
// all be seen, returned, passed to an intrinsic or a call too deep)
static void scan_uses(LLVMValueRef fn, LLVMValueRef root, int64_t base, struct pieces *s, unsigned depth)
{
    if (depth > 8) {
        s->escapes = 1;
        return;
    }
    for (LLVMBasicBlockRef bb = LLVMGetFirstBasicBlock(fn); bb; bb = LLVMGetNextBasicBlock(bb))
        for (LLVMValueRef in = LLVMGetFirstInstruction(bb); in; in = LLVMGetNextInstruction(in)) {
            LLVMOpcode op = LLVMGetInstructionOpcode(in);
            LLVMValueRef val = op == LLVMStore ? LLVMGetOperand(in, 0) : NULL;
            if (val && rooted_at(val, root) &&
                !follow_spill(fn, LLVMGetOperand(in, 1), base + root_offset(val), s, depth))
                s->escapes = 1;
            if (op == LLVMLoad && ptr_root(LLVMGetOperand(in, 0)) == root) {
                add_pieces(s, LLVMTypeOf(in), base + root_offset(LLVMGetOperand(in, 0)));
            } else if (op == LLVMStore && ptr_root(LLVMGetOperand(in, 1)) == root) {
                add_pieces(s, LLVMTypeOf(LLVMGetOperand(in, 0)), base + root_offset(LLVMGetOperand(in, 1)));
            } else if (op == LLVMGetElementPtr && ptr_root(in) == root) {
                for (int i = 1; i < LLVMGetNumOperands(in); i++)
                    if (!LLVMIsAConstantInt(LLVMGetOperand(in, i)))
                        s->dynamic = 1;
            } else if (op == LLVMPHI && is_ptr(LLVMTypeOf(in))) {  // a pointer walking it, picked when it runs
                for (unsigned i = 0; i < LLVMCountIncoming(in); i++)
                    if (ptr_root(LLVMGetIncomingValue(in, i)) == root)
                        s->dynamic = 1;
            } else if (op == LLVMSelect && is_ptr(LLVMTypeOf(in))) {
                if (ptr_root(LLVMGetOperand(in, 1)) == root || ptr_root(LLVMGetOperand(in, 2)) == root)
                    s->dynamic = 1;
            } else if (op == LLVMCall) {
                LLVMValueRef callee = LLVMGetCalledValue(in);
                const char *name = callee ? LLVMGetValueName(callee) : NULL;
                if (callee && !LLVMGetFirstBasicBlock(callee) && name && strncmp(name, "air.", 4) &&
                    strncmp(name, "llvm.", 5))
                    callee = linked_function(name);
                // a memcpy or memset moves bytes whatever the lanes around it are, which only memory keeps
                if (name && !strncmp(name, "llvm.mem", 8) &&
                    (ptr_root(LLVMGetOperand(in, 0)) == root || ptr_root(LLVMGetOperand(in, 1)) == root))
                    s->dynamic = 1;
                int lifetime = name && !strncmp(name, "llvm.lifetime.", 14);
                for (int k = 0; k < LLVMGetNumOperands(in) - 1; k++) {
                    LLVMValueRef arg = LLVMGetOperand(in, k);
                    if (!rooted_at(arg, root) || lifetime)
                        continue;
                    if (callee && LLVMGetFirstBasicBlock(callee) && (unsigned)k < LLVMCountParams(callee))
                        scan_uses(callee, LLVMGetParam(callee, k), base + root_offset(arg), s, depth + 1);
                    else
                        s->escapes = 1;
                }
            } else if (op != LLVMStore && op != LLVMBitCast && op != LLVMAddrSpaceCast) {
                for (int i = 0; i < LLVMGetNumOperands(in); i++)
                    if (rooted_at(LLVMGetOperand(in, i), root))
                        s->escapes = 1;
            }
        }
}

static struct local_choice {
    LLVMValueRef alloca;
    int local;
    struct local_choice *next;
} *g_local_choices;

static int alloca_is_local(LLVMValueRef alloca, LLVMValueRef fn)
{
    struct local_choice *c = g_local_choices;
    while (c && c->alloca != alloca)
        c = c->next;
    if (!c) {
        struct pieces s = { 0 };
        scan_uses(fn, alloca, 0, &s, 0);
        c = ir_alloc(sizeof(*c));
        c->alloca = alloca;
        // what's done through an escaped address isn't seen: different sizes anywhere might overlap there
        c->local = s.dynamic || pieces_clash(&s) || (s.escapes && (s.sizes & (s.sizes - 1)));
        c->next = g_local_choices;
        g_local_choices = c;
        free(s.p);
    }
    return c->local;
}

static LLVMValueRef g_body_fn;  // the function whose body is being converted (an inlined one's while it is)
static void const_table(struct ptr p, LLVMValueRef init);

static struct ptr ptr_of(LLVMValueRef v)
{
    struct entry *e = find(g_ptrs, v, 0, g_scope);
    if (e)
        return e->p;
    struct ptr p = { 0 };
    if (LLVMIsAGlobalVariable(v)) {
        unsigned as = LLVMGetPointerAddressSpace(LLVMTypeOf(v));
        const char *name = LLVMGetValueName(v);
        if (as == 3) {
            p.space = SP_SHARED;
            p.off = shared_offset(v);
        } else if (as == 2 && name && !strncmp(name, "__air_sampler_state", 19)) {
            p.space = SP_SAMPLER;
            p.slot = const_sampler(v);
        } else if (as == 2 && LLVMGetInitializer(v)) {  // a constant table: variables that hold it from the start
            unsigned a;
            p.space = SP_PRIVATE;
            p.slot = ++g_nallocas;
            p.bytes = layout(LLVMGlobalGetValueType(v), &a);
            const_table(p, LLVMGetInitializer(v));
        } else {
            airc_fail("unsupported global %s", name ? name : "?");
        }
    } else if (LLVMIsAConstantExpr(v)) {
        LLVMOpcode op = LLVMGetConstOpcode(v);
        if (op == LLVMBitCast || op == LLVMAddrSpaceCast)
            p = ptr_of(LLVMGetOperand(v, 0));
        else if (op == LLVMGetElementPtr)
            p = gep(v);
        else
            airc_fail("unsupported constant pointer expression %d", op);
    } else if (LLVMIsAInstruction(v) && LLVMGetInstructionOpcode(v) == LLVMAlloca && LLVMGetAllocatedType(v) &&
               alloca_is_local(v, g_body_fn)) {
        unsigned a, bytes = layout(LLVMGetAllocatedType(v), &a);  // a fresh area for each inlined instance
        a = a < 4 ? 4 : a;
        p.space = SP_LOCAL;
        p.off = (g_info->local_size + a - 1) & ~(a - 1);
        g_info->local_size = (unsigned)p.off + bytes;
        p.slot = g_info->local_size;
    } else if (LLVMIsAInstruction(v) && LLVMGetInstructionOpcode(v) == LLVMAlloca) {
        unsigned a;
        p.space = SP_PRIVATE;
        p.slot = ++g_nallocas;  // a fresh one for each inlined instance (pointers are keyed by scope)
        p.bytes = LLVMGetAllocatedType(v) ? layout(LLVMGetAllocatedType(v), &a) : 0;
    } else if (LLVMIsAInstruction(v) && LLVMGetInstructionOpcode(v) == LLVMGetElementPtr) {
        p = gep(v);
    } else if (LLVMIsAInstruction(v) && (LLVMGetInstructionOpcode(v) == LLVMBitCast ||
                                         LLVMGetInstructionOpcode(v) == LLVMAddrSpaceCast)) {
        p = ptr_of(LLVMGetOperand(v, 0));
    } else if (!LLVMIsAInstruction(v) && !LLVMIsAFunction(v)) {  // null (or undefined): address 0, handle 0
        p.space = SP_GLOBAL;
        p.base = konst(IR_I64, 0);
    } else {
        airc_fail("unsupported pointer (opcode %d)", LLVMIsAInstruction(v) ? LLVMGetInstructionOpcode(v) : -1);
    }
    insert(g_ptrs, v, 0, g_scope)->p = p;
    return p;
}

// A pointer held in an aggregate (CoreImage passes its samplers around as structs of a texture, a sampler and a
// constant pointer): lane `lane` of v, kept beside the value lanes
static int lane_is_ptr(LLVMTypeRef t, unsigned lane)
{
    return LLVMGetTypeKind(lane_type(t, lane)) == LLVMPointerTypeKind;
}

static struct ptr ptr_lane(LLVMValueRef v, unsigned lane)
{
    if (LLVMGetTypeKind(LLVMTypeOf(v)) == LLVMPointerTypeKind)
        return ptr_of(v);
    struct entry *e = find(g_ptrs, v, lane, g_scope);
    if (e)
        return e->p;
    if (LLVMIsUndef(v) || LLVMIsPoison(v))
        return (struct ptr){ 0 };
    airc_fail("a pointer in an aggregate that isn't built in the function");
}

// lane dl of dst is lane sl of src, a value or a pointer
static void copy_lane(LLVMValueRef dst, unsigned dl, LLVMValueRef src, unsigned sl)
{
    if (lane_is_ptr(LLVMTypeOf(src), sl))
        insert(g_ptrs, dst, dl, g_scope)->p = ptr_lane(src, sl);
    else {
        set(dst, dl, get(src, sl));
        if (is_i64(lane_type(LLVMTypeOf(src), sl)) && high_word(src, sl))
            set_high(dst, dl, high_word(src, sl));
    }
}

// A local access's variable part (NULL for none), kept inside its alloca for an access ending end bytes past p.off:
// past the thread's local size faults the warp (Out Of Range Address), where a register array read garbage, and
// code the compiler runs on both sides of a branch may compute any index. The size covers whatever the access can
// reach, as one reading past its alloca's end would (undefined, but no fault).
static struct ir_val *local_base(struct ptr p, unsigned end)
{
    int64_t room = (int64_t)p.slot - p.off - end, reach = p.off + end + (p.base && room > 0 ? room : 0);
    if (reach > g_info->local_size)
        g_info->local_size = (unsigned)reach;
    if (!p.base)
        return NULL;
    struct ir_val *m = op2(IR_MIN, IR_I32, p.base, konst(IR_I32, room > 0 ? room : 0));
    m->aux = 0;  // unsigned: below the start is past the end too
    return m;
}

// base + off for a global pointer, as an address: whole megabytes of the offset through IMAD.WIDE's 32-bit index,
// so any offset fits
static struct ir_val *add_offset(struct ir_val *base, int64_t off)
{
    int big = off < INT32_MIN || off > INT32_MAX;
    struct ir_val *a = op2(IR_ADDR, IR_I64, base, konst(IR_I32, big ? off >> 20 : off));
    a->imm = big ? 1 << 20 : 1;
    a->aux = 1;  // signed
    return big && (off & 0xfffff) ? add_offset(a, off & 0xfffff) : a;
}

// a global pointer whose offset a memory op's signed 24-bit immediate holds, with room for the lanes after it;
// the rest goes into the base
static struct ptr fit_offset(struct ptr p)
{
    if (p.space != SP_GLOBAL || (p.off >= -(1 << 23) && p.off < (1 << 23) - 256))
        return p;
    p.base = add_offset(p.base, p.off);
    p.off = 0;
    return p;
}

static struct ir_val *handle_of_ptr(struct ptr p, enum space space);

// a pointer as one value, for a phi or select of pointers: a global one's 64-bit address, a shared one's offset, a
// texture's or sampler's handle
static struct ir_val *ptr_value(struct ptr p)
{
    if (p.space == SP_GLOBAL)
        return p.off ? add_offset(p.base, p.off) : p.base;
    if (p.space == SP_SHARED || p.space == SP_LOCAL) {
        struct ir_val *off = konst(IR_I32, p.off);
        return !p.base ? off : p.off ? op2(IR_ADD, IR_I32, p.base, off) : p.base;
    }
    if (p.space == SP_TEXTURE || p.space == SP_SAMPLER)
        return handle_of_ptr(p, p.space);
    airc_fail("a phi or select of private pointers");
}

static enum space ptr_space(LLVMTypeRef t)
{
    return LLVMGetPointerAddressSpace(t) == 3 ? SP_SHARED : SP_GLOBAL;
}

// A phi of pointers, before the values it picks from are all made (a loop's step comes after it): in local memory
// when one it starts from is an alloca there, and kept inside the furthest of those. An alloca's address space is
// the one device memory has, so that alone can't tell.
static struct ptr phi_ptr(LLVMValueRef phi)
{
    struct ptr q = { .space = ptr_space(LLVMTypeOf(phi)) };
    for (unsigned i = 0; i < LLVMCountIncoming(phi); i++) {
        LLVMValueRef r = ptr_root(LLVMGetIncomingValue(phi, i));
        if (!(LLVMIsAInstruction(r) && LLVMGetInstructionOpcode(r) == LLVMAlloca) && !find(g_ptrs, r, 0, g_scope))
            continue;
        struct ptr p = ptr_of(r);
        if (p.space == SP_LOCAL) {
            q.space = SP_LOCAL;
            q.slot = p.slot > q.slot ? p.slot : q.slot;
        }
        if (p.space == SP_TEXTURE || p.space == SP_SAMPLER)
            q.space = p.space;
    }
    return q;
}

// A global the static initialiser writes (from a function constant's specialised value) and the shader then
// only reads: its lanes, once written.
static struct ir_val *global_state(LLVMValueRef g, unsigned lane)
{
    struct entry *e = find(g_vals, g, lane, 0);
    return e ? e->v : NULL;
}

// the specialised value of the function constant behind an initialiser global (named
// <name>.MTL_FC_INIT_<n>_<type>), 1 if it has one; else its index and name for the error
static int fc_bits(LLVMValueRef g, uint32_t *bits, int *index, const char **name, unsigned *len)
{
    unsigned n = LLVMGetNamedMetadataNumOperands(g_module, "air.function_constants");
    LLVMValueRef *ops = calloc(n, sizeof(*ops));
    LLVMGetNamedMetadataOperands(g_module, "air.function_constants", ops);
    *index = -1;
    *name = "?";
    *len = 1;
    for (unsigned i = 0; i < n && *index < 0; i++) {
        int on = LLVMGetMDNodeNumOperands(ops[i]);
        LLVMValueRef o[8];
        if (on < 4 || on > 8)
            continue;
        LLVMGetMDNodeOperands(ops[i], o);
        if (o[0] == g && LLVMIsAConstantInt(o[3])) {
            *name = LLVMGetMDString(o[2], len);
            *index = (int)LLVMConstIntGetZExtValue(o[3]);
        }
    }
    free(ops);
    for (unsigned i = 0; i < g_nfc; i++)
        if ((int)g_fc[i].index == *index) {
            *bits = g_fc[i].bits;
            return 1;
        }
    for (unsigned i = 0; i < g_nfc; i++)
        if (g_fc[i].index == AIRC_FC_DEFAULT) {
            *bits = g_fc[i].bits;
            return 1;
        }
    return 0;
}

static struct ir_val *function_constant(LLVMValueRef g, LLVMTypeRef t)
{
    uint32_t bits;
    int index;
    const char *name;
    unsigned len;
    if (fc_bits(g, &bits, &index, &name, &len))
        return konst(ir_type_of(t), bits);
    airc_fail("function constant %d (%.*s) has no value", index, (int)len, name);
}

static LLVMValueRef static_init(LLVMValueRef after);

// The value of a predicate a [[function_constant(x)]] output names: what the static initialiser stores into its
// global, worked out at compile time, since an output that exists only when it's true has to be settled before
// the code is made.
static unsigned long long fc_eval(LLVMValueRef v)
{
    uint32_t bits;
    int index;
    const char *fc_name;
    unsigned len;
    if (LLVMIsAConstantInt(v))
        return LLVMConstIntGetZExtValue(v);
    if (LLVMIsAGlobalVariable(v)) {
        const char *name = LLVMGetValueName(v);
        if (name && strstr(name, ".MTL_FC_INIT_")) {
            if (!fc_bits(v, &bits, &index, &fc_name, &len))
                airc_fail("function constant %d (%.*s) has no value", index, (int)len, fc_name);
            return bits;
        }
        for (LLVMValueRef init = static_init(NULL); init; init = static_init(init))
            for (LLVMBasicBlockRef bb = LLVMGetFirstBasicBlock(init); bb; bb = LLVMGetNextBasicBlock(bb))
                for (LLVMValueRef in = LLVMGetFirstInstruction(bb); in; in = LLVMGetNextInstruction(in))
                    if (LLVMGetInstructionOpcode(in) == LLVMStore && LLVMGetOperand(in, 1) == v)
                        return fc_eval(LLVMGetOperand(in, 0));
        airc_fail("function constant predicate %s is never set", name ? name : "?");
    }
    if (!LLVMIsAInstruction(v))
        airc_fail("unsupported function constant expression");
    LLVMValueRef a = LLVMGetOperand(v, 0);
    unsigned long long mask = LLVMGetTypeKind(LLVMTypeOf(v)) == LLVMIntegerTypeKind
                                  ? ~0ull >> (64 - LLVMGetIntTypeWidth(LLVMTypeOf(v)))
                                  : ~0ull;
    switch (LLVMGetInstructionOpcode(v)) {
    case LLVMLoad:
    case LLVMZExt:
        return fc_eval(a);
    case LLVMTrunc:
        return fc_eval(a) & mask;
    case LLVMSelect:
        return fc_eval(a) & 1 ? fc_eval(LLVMGetOperand(v, 1)) : fc_eval(LLVMGetOperand(v, 2));
    case LLVMAnd:
        return fc_eval(a) & fc_eval(LLVMGetOperand(v, 1));
    case LLVMOr:
        return fc_eval(a) | fc_eval(LLVMGetOperand(v, 1));
    case LLVMXor:
        return (fc_eval(a) ^ fc_eval(LLVMGetOperand(v, 1))) & mask;
    case LLVMICmp: {
        unsigned bits = LLVMGetIntTypeWidth(LLVMTypeOf(a));
        unsigned long long x = fc_eval(a), y = fc_eval(LLVMGetOperand(v, 1));
        long long sx = (long long)(x << (64 - bits)) >> (64 - bits), sy = (long long)(y << (64 - bits)) >> (64 - bits);
        switch (LLVMGetICmpPredicate(v)) {
        case LLVMIntEQ:
            return x == y;
        case LLVMIntNE:
            return x != y;
        case LLVMIntUGT:
            return x > y;
        case LLVMIntUGE:
            return x >= y;
        case LLVMIntULT:
            return x < y;
        case LLVMIntULE:
            return x <= y;
        case LLVMIntSGT:
            return sx > sy;
        case LLVMIntSGE:
            return sx >= sy;
        case LLVMIntSLT:
            return sx < sy;
        default:
            return sx <= sy;
        }
    }
    case LLVMSExt: {
        unsigned bits = LLVMGetIntTypeWidth(LLVMTypeOf(a));
        return (unsigned long long)((long long)(fc_eval(a) << (64 - bits)) >> (64 - bits)) & mask;
    }
    case LLVMAdd:
        return (fc_eval(a) + fc_eval(LLVMGetOperand(v, 1))) & mask;
    case LLVMSub:
        return (fc_eval(a) - fc_eval(LLVMGetOperand(v, 1))) & mask;
    case LLVMMul:
        return (fc_eval(a) * fc_eval(LLVMGetOperand(v, 1))) & mask;
    case LLVMShl:
        return (fc_eval(a) << (fc_eval(LLVMGetOperand(v, 1)) & 63)) & mask;
    case LLVMLShr:
        return fc_eval(a) >> (fc_eval(LLVMGetOperand(v, 1)) & 63);
    case LLVMCall: {
        const char *name = LLVMGetValueName(LLVMGetCalledValue(v));
        if (name && !strncmp(name, "air.normalize_function_constant_predicate", 41))
            return fc_eval(a) != 0;
        if (name && !strcmp(name, "air.is_function_constant_defined"))
            return fc_bits(a, &bits, &index, &fc_name, &len);
        airc_fail("unsupported call %s in a function constant", name ? name : "?");
    }
    default:
        airc_fail("unsupported function constant instruction %d", LLVMGetInstructionOpcode(v));
    }
}

static struct ir_val *s2r(unsigned sr)
{
    struct ir_val *v = ir_add(g_prologue, IR_S2R, IR_I32, 0);
    v->imm = sr;
    return v;
}

static struct ir_val *cbuf(enum ir_type t, unsigned off)
{
    struct ir_val *v = ir_add(g_prologue, IR_CBUF, t, 0);
    v->imm = off;
    return v;
}

static struct ir_val *attr(enum ir_op op, unsigned addr)
{
    struct ir_val *v = ir_add(g_prologue, op, IR_F32, 0);
    v->imm = addr;
    return v;
}

// a tessellation stage's attribute at addr in b: of the vertex whose handle is vtx, or (IR_ATTR_* flags) the patch's
// or the stage's own output
static struct ir_val *tess_attr(struct ir_block *b, struct ir_val *vtx, unsigned addr, int flags)
{
    struct ir_val *v = ir_add(b, IR_ALD, IR_F32, vtx ? 1 : 0);
    if (vtx)
        v->src[0] = vtx;
    v->imm = addr;
    v->aux = flags;
    return v;
}

static struct ir_val *flat_tid(void)
{
    struct ir_val *zy = ir_op2(g_prologue, IR_ADD, IR_I32,
                               ir_op2(g_prologue, IR_MUL, IR_I32, s2r(SR_TID_Z), cbuf(IR_I32, 4)), s2r(SR_TID_Y));
    return ir_op2(g_prologue, IR_ADD, IR_I32, ir_op2(g_prologue, IR_MUL, IR_I32, zy, cbuf(IR_I32, 0)), s2r(SR_TID_X));
}

// a thread's place in the dispatch along axis l, for the builtin kinds that are one: where it is in the grid
// (origin + ctaid * ntid + tid, ntid from c[0x0][4 * l]) or its threadgroup, and the sizes
static struct ir_val *thread_value(enum nvk_arg_kind kind, unsigned l)
{
    switch (kind) {
    case NVK_ARG_TID: {
        struct ir_val *v = ir_op2(g_prologue, IR_ADD, IR_I32,
                                  ir_op2(g_prologue, IR_MUL, IR_I32, s2r(SR_CTAID_X + l), cbuf(IR_I32, 4 * l)),
                                  s2r(SR_TID_X + l));
        return ir_op2(g_prologue, IR_ADD, IR_I32, v, cbuf(IR_I32, NVK_CB_ORIGIN + 4 * l));
    }
    case NVK_ARG_TG_POS:
        return ir_op2(g_prologue, IR_ADD, IR_I32, s2r(SR_CTAID_X + l), cbuf(IR_I32, NVK_CB_GROUP_ORIGIN + 4 * l));
    case NVK_ARG_TG_TID:
        return s2r(SR_TID_X + l);
    case NVK_ARG_TG_SIZE:
        return cbuf(IR_I32, 4 * l);
    case NVK_ARG_GROUPS:
        return cbuf(IR_I32, NVK_CB_GROUPS + 4 * l);
    case NVK_ARG_GRID_SIZE:
        return cbuf(IR_I32, NVK_CB_GRID + 4 * l);
    default:
        airc_fail("thread value of argument kind %d", kind);
    }
}

static struct ir_val *udivmod(struct ir_val *n, struct ir_val *d, int rem);

// MTLStepFunction's values for a kernel's stage input
enum { STEP_CONSTANT = 0, STEP_X = 5, STEP_Y, STEP_X_INDEXED, STEP_Y_INDEXED };

// word i of the stage-in region: its origin's x, y and z, then its size's
static struct ir_val *stage_in_region(unsigned i)
{
    if (!g_region[i]) {
        struct ir_val *at = cbuf(IR_I64, g_region_off);
        g_region[i] = ir_add(g_prologue, IR_LDG, IR_I32, 1);
        g_region[i]->src[0] = at;
        g_region[i]->imm = 4 * i;
        g_region[i]->aux = 4;
    }
    return g_region[i];
}

// The element of a's buffer the stage input reads at stage-in grid position (x, y), a step every rate positions:
// the position along the step function's axis past the region's origin, or the index buffer's at that position
// past the origin (which makes the origin a base vertex, as Metal's tessellation kernels have it)
static struct ir_val *stage_in_element(const struct airc_tess_attr *a, unsigned loc, struct ir_val *x,
                                       struct ir_val *y)
{
    if (a->step == STEP_CONSTANT)
        return konst(IR_I32, 0);
    if (a->step < STEP_X || a->step > STEP_Y_INDEXED)
        airc_fail("stage input attribute %u: step function %u", loc, a->step);
    int axis = a->step == STEP_Y || a->step == STEP_Y_INDEXED;
    struct ir_val *pos = airc_udiv(g_cur, axis ? y : x, a->rate);
    if (a->step >= STEP_X_INDEXED) {
        unsigned bytes = g_stage_in->index_size;
        if (bytes != 2 && bytes != 4)
            airc_fail("stage input indices of %u bytes", bytes);
        struct ir_val *at = op2(IR_ADDR, IR_I64, cbuf(IR_I64, g_stage_in_buffer[g_stage_in->index_buffer]), pos);
        at->imm = bytes;
        pos = ir_add(g_cur, IR_LDG, IR_I32, 1);
        pos->src[0] = at;
        pos->aux = (int)bytes;
    }
    return op2(IR_ADD, IR_I32, pos, stage_in_region(axis));
}

// attribute loc of the stage input at stage-in grid position (x, y); one the descriptor leaves out reads zeros, as
// a vertex function's does
static void stage_in_attr(unsigned loc, struct ir_val *x, struct ir_val *y, struct ir_val *v[4])
{
    const struct airc_tess_attr *a = g_stage_in ? &g_stage_in->attrs[loc] : NULL;
    if (!a || !a->format) {
        v[0] = v[1] = v[2] = v[3] = konst(IR_I32, 0);
        return;
    }
    airc_fetch(g_cur, a, loc, cbuf(IR_I64, g_stage_in_buffer[a->buffer]), stage_in_element(a, loc, x, y), v);
}

// a fetched component as a lane of type t holds it, a half still as a float: its bits as a float's or an integer's
static struct ir_val *stage_in_lane(struct ir_val *v, LLVMTypeRef t)
{
    enum ir_type want = ir_type_of(t) == IR_I32 ? IR_I32 : IR_F32;
    return v->type == want ? v : op2(IR_OR, want, v, konst(want, 0));
}

// The entry function's arguments: builtins become values made in the prologue, buffers pointers whose base
// the dispatcher puts in const bank 0, textures and samplers pool slots.
static void entry_args(void)
{
    for (unsigned i = 0; i < LLVMCountParams(g_entry_fn); i++) {  // the stage input's hidden ones follow
        const struct nvk_arg *a = &g_info->args[i];
        LLVMValueRef p = LLVMGetParam(g_entry_fn, i);
        unsigned n = nlanes(LLVMTypeOf(p));
        struct ptr ptr = { 0 };
        struct ir_val *texel = NULL, *uv[2] = { NULL, NULL }, *fetched[4];
        if (a->kind == NVK_ARG_RENDER_TARGET) {  // fetch the attachment at the pixel's integer position
            struct ir_val *xy[2];
            for (int c = 0; c < 2; c++) {
                struct ir_val *pos = attr(IR_IPA, 0x70 + 4 * c);
                xy[c] = ir_add(g_prologue, IR_F2I, IR_I32, 1);
                xy[c]->src[0] = pos;
            }
            g_info->ps_sysvals |= SPH_PS_POSITION;
            struct ir_val *handle = cbuf(IR_I32, NVK_TEXTURE_HANDLE(NVK_RENDER_TARGET_SLOT(a->index)));
            texel = ir_add(g_prologue, IR_TEX, IR_VOID, 3);
            texel->src[0] = xy[0];
            texel->src[1] = xy[1];
            texel->src[2] = handle;
            texel->aux = IR_TEXA(SASS_TLD, SASS_DIM_2D, SASS_LOD_ZERO, 2);
        }
        for (unsigned l = 0; l < n; l++) {
            struct ir_val *v = NULL;
            switch (a->kind) {
            case NVK_ARG_BUFFER:
                ptr.space = SP_GLOBAL;
                ptr.base = cbuf(IR_I64, a->cbuf_off);
                break;
            case NVK_ARG_THREADGROUP:  // after the static threadgroup variables, where the dispatcher put it
                ptr.space = SP_SHARED;
                ptr.base = cbuf(IR_I32, a->cbuf_off);
                break;
            case NVK_ARG_TEXTURE:
            case NVK_ARG_SAMPLER:
                ptr.space = a->kind == NVK_ARG_TEXTURE ? SP_TEXTURE : SP_SAMPLER;
                ptr.slot = a->index < 16 ? a->index : a->cbuf_off;
                ptr.count = g_info->array_count[i];
                break;
            case NVK_ARG_TID:
            case NVK_ARG_TG_POS:
            case NVK_ARG_TG_TID:
            case NVK_ARG_TG_SIZE:
            case NVK_ARG_GROUPS:
            case NVK_ARG_GRID_SIZE:
                v = thread_value((enum nvk_arg_kind)a->kind, l);
                break;
            case NVK_ARG_TG_INDEX:
                v = flat_tid();
                break;
            case NVK_ARG_SIMD_LANE:
                v = s2r(SR_LANEID);
                break;
            case NVK_ARG_SIMD_GROUP:  // warps are 32 consecutive flat thread indices
                v = ir_op2(g_prologue, IR_SHR, IR_I32, flat_tid(), konst(IR_I32, 5));
                break;
            case NVK_ARG_SIMD_SIZE:
                v = konst(IR_I32, 32);
                break;
            case NVK_ARG_SIMD_GROUPS:  // ntid.x * ntid.y * ntid.z in whole SIMD groups of 32, or quads
            case NVK_ARG_QUAD_GROUPS: {
                unsigned shift = a->kind == NVK_ARG_SIMD_GROUPS ? 5 : 2;
                struct ir_val *n = ir_op2(g_prologue, IR_MUL, IR_I32,
                                          ir_op2(g_prologue, IR_MUL, IR_I32, cbuf(IR_I32, 0), cbuf(IR_I32, 4)),
                                          cbuf(IR_I32, 8));
                n = ir_op2(g_prologue, IR_ADD, IR_I32, n, konst(IR_I32, (1 << shift) - 1));
                v = ir_op2(g_prologue, IR_SHR, IR_I32, n, konst(IR_I32, shift));
                break;
            }
            case NVK_ARG_QUAD_GROUP:
                v = ir_op2(g_prologue, IR_SHR, IR_I32, flat_tid(), konst(IR_I32, 2));
                break;
            case NVK_ARG_QUAD_LANE:
                v = ir_op2(g_prologue, IR_AND, IR_I32, flat_tid(), konst(IR_I32, 3));
                break;
            case NVK_ARG_DISPATCH_TG_SIZE: {
                // ntid is short in the part of a non-uniform dispatch with the edge's threadgroups, which starts
                // past whole ones: their size is its origin over its threadgroup origin. A grid smaller than one
                // threadgroup leaves no trace of the size asked for, and gets ntid.
                struct ir_val *groups = cbuf(IR_I32, NVK_CB_GROUP_ORIGIN + 4 * l);
                struct ir_val *edge = ir_op2(g_prologue, IR_CMP, IR_PRED, groups, konst(IR_I32, 0));
                edge->aux = IR_NE;
                v = op3(IR_SEL, IR_I32, edge, udivmod(cbuf(IR_I32, NVK_CB_ORIGIN + 4 * l), groups, 0),
                        cbuf(IR_I32, 4 * l));
                break;
            }
            case NVK_ARG_VERTEX_ID:
            case NVK_ARG_INSTANCE_ID:
                if (a->kind == NVK_ARG_INSTANCE_ID && g_info->stage == NVK_TESS_EVAL) {  // from the base instance
                    v = tess_attr(g_prologue, NULL, NVK_PATCH_IDS + 4, IR_ATTR_PATCH);
                    v->type = IR_I32;
                    break;
                }
                v = attr(IR_ALD, a->kind == NVK_ARG_VERTEX_ID ? 0x2fc : 0x2f8);
                v->type = IR_I32;
                g_info->sysvals |= a->kind == NVK_ARG_VERTEX_ID ? SPH_VERTEX_ID : SPH_INSTANCE_ID;
                if (a->kind == NVK_ARG_INSTANCE_ID)
                    v = ir_op2(g_prologue, IR_ADD, IR_I32, v, cbuf(IR_I32, NVK_BASE_INSTANCE));
                break;
            case NVK_ARG_BASE_VERTEX:
                v = cbuf(IR_I32, NVK_BASE_VERTEX);
                break;
            case NVK_ARG_BASE_INSTANCE:
                v = cbuf(IR_I32, NVK_BASE_INSTANCE);
                break;
            case NVK_ARG_VERTEX_INPUT:
                if (g_info->stage == NVK_COMPUTE) {  // a kernel's [[stage_in]], fetched at the thread's position
                    if (!l)
                        stage_in_attr(a->index, thread_value(NVK_ARG_TID, 0), thread_value(NVK_ARG_TID, 1), fetched);
                    v = stage_in_lane(fetched[l], lane_type(LLVMTypeOf(p), l));
                    break;
                }
                v = attr(IR_ALD, 0x80 + 16 * a->index + 4 * l);
                v->type = ir_type_of(lane_type(LLVMTypeOf(p), l)) == IR_I32 ? IR_I32 : IR_F32;
                break;
            case NVK_ARG_STAGE_IN_ORIGIN:
            case NVK_ARG_STAGE_IN_SIZE:
                v = stage_in_region(3 * (a->kind == NVK_ARG_STAGE_IN_SIZE) + l);
                break;
            case NVK_ARG_CONTROL_POINTS: {  // bytes 0 and 2 of INVOCATION_INFO multiplied
                if (g_info->stage == NVK_COMPUTE)  // a kernel's are its stage input, fetched where they're read
                    break;
                struct ir_val *info = s2r(SR_INVOCATION_INFO);
                struct ir_val *lo = ir_op2(g_prologue, IR_AND, IR_I32, info, konst(IR_I32, 0xff));
                struct ir_val *hi = ir_op2(g_prologue, IR_SHR, IR_I32, info, konst(IR_I32, 16));
                hi = ir_op2(g_prologue, IR_AND, IR_I32, hi, konst(IR_I32, 0xff));
                g_point_map = ir_op2(g_prologue, IR_MUL, IR_I32, lo, hi);
                break;
            }
            case NVK_ARG_PATCH_INPUT:
                v = tess_attr(g_prologue, NULL, NVK_PATCH_INPUT(g_info->patch_inputs, a->index) + 4 * l, IR_ATTR_PATCH);
                v->type = ir_type_of(lane_type(LLVMTypeOf(p), l)) == IR_I32 ? IR_I32 : IR_F32;
                break;
            case NVK_ARG_PATCH_ID:
                v = tess_attr(g_prologue, NULL, NVK_PATCH_IDS, IR_ATTR_PATCH);
                v->type = IR_I32;
                break;
            case NVK_ARG_POSITION_IN_PATCH:  // u and v, outputs of the stage indexed by lane; a triangle's w 1 - u - v
                if (l < 2)
                    v = uv[l] = tess_attr(g_prologue, s2r(SR_LANEID), 0x2f0 + 4 * l, IR_ATTR_OUTPUT);
                else
                    v = ir_op2(g_prologue, IR_FSUB, IR_F32, ir_op2(g_prologue, IR_FSUB, IR_F32, fconst(1.0f), uv[0]),
                               uv[1]);
                break;
            case NVK_ARG_POSITION:
                v = attr(IR_IPA, 0x70 + 4 * l);
                g_info->ps_sysvals |= l == 2 ? SPH_PS_POSITION_Z : SPH_PS_POSITION;
                break;
            case NVK_ARG_POINT_COORD:
                v = attr(IR_IPA, 0x2e0 + 4 * l);
                g_info->ps_sysvals |= SPH_PS_POINT_COORD;
                break;
            case NVK_ARG_FRONT_FACING: {  // a[0x3fc] read flat (NAK's NAK_ATTR_FRONT_FACE): non-zero for the front
                struct ir_val *f = attr(IR_IPA, 0x3fc);
                f->type = IR_I32;
                f->aux = 1;
                v = ir_op2(g_prologue, IR_CMP, IR_PRED, f, konst(IR_I32, 0));
                v->aux = IR_NE;
                break;
            }
            case NVK_ARG_RENDER_TARGET:
                v = ir_add(g_prologue, IR_PROJ, IR_F32, 1);
                v->src[0] = texel;
                v->imm = l;
                break;
            case NVK_ARG_VARYING: {
                v = attr(IR_IPA, 0x80 + 16 * a->index + 4 * l);
                // read as is: IPA.PASS evaluates the plane as floats, which flushes an integer's small values,
                // denormal as floats, to 0
                int at = (g_info->centroid | g_info->at_sample) >> a->index & 1;
                v->aux = g_info->interp[a->index] == SPH_FLAT ? SASS_IPA_FLAT : at ? SASS_IPA_CENTROID : 0;
                if (g_info->interp[a->index] == SPH_PERSPECTIVE) {  // SM70+ leaves the divide by w to the shader
                    if (!g_rcp_w[at]) {
                        struct ir_val *w = attr(IR_IPA, 0x7c);
                        w->aux = at ? SASS_IPA_CENTROID : 0;
                        g_rcp_w[at] = ir_add(g_prologue, IR_MUFU, IR_F32, 1);
                        g_rcp_w[at]->aux = MUFU_RCP;
                        g_rcp_w[at]->src[0] = w;
                    }
                    v = ir_op2(g_prologue, IR_FMUL, IR_F32, v, g_rcp_w[at]);
                }
                break;
            }
            case NVK_ARG_SAMPLE_ID:
                v = s2r(SR_PIXLD(SASS_PIXLD_MY_INDEX));
                break;
            case NVK_ARG_LAYER:  // what the vertex function wrote, read flat
            case NVK_ARG_VIEWPORT:
                v = attr(IR_IPA, a->kind == NVK_ARG_LAYER ? 0x64 : 0x68);
                v->type = IR_I32;
                v->aux = SASS_IPA_FLAT;
                g_info->ps_sysvals |= a->kind == NVK_ARG_LAYER ? SPH_PS_LAYER : SPH_PS_VIEWPORT;
                break;
            case NVK_ARG_SAMPLE_MASK:  // the pixel's coverage; run once a sample, only the sample's
                v = s2r(SR_PIXLD(SASS_PIXLD_COVMASK));
                if (g_info->flags & NVK_FLAG_PER_SAMPLE)
                    v = ir_op2(g_prologue, IR_AND, IR_I32, v,
                               ir_op2(g_prologue, IR_SHL, IR_I32, konst(IR_I32, 1),
                                      s2r(SR_PIXLD(SASS_PIXLD_MY_INDEX))));
                break;
            }
            if (v) {
                if (ir_type_of(lane_type(LLVMTypeOf(p), l)) == IR_F16) {
                    struct ir_val *h = ir_add(g_prologue, IR_F2H, IR_F16, 1);
                    h->src[0] = v;
                    v = h;
                }
                set(p, l, v);
            }
        }
        if (is_ptr(LLVMTypeOf(p)))
            insert(g_ptrs, p, 0, g_scope)->p = ptr;
    }
}

// intrinsic names look like air.<base>.<type>, air.fast_<base>.<type> or llvm.<base>.<type>
static int is_fn(const char *name, const char *base)
{
    static const char *const prefixes[] = { "air.", "air.fast_", "air.native_", "llvm." };
    size_t n = strlen(base);
    for (unsigned i = 0; i < sizeof prefixes / sizeof prefixes[0]; i++) {
        size_t p = strlen(prefixes[i]);
        if (!strncmp(name, prefixes[i], p) && !strncmp(name + p, base, n) && (name[p + n] == '.' || !name[p + n]))
            return 1;
    }
    return 0;
}

static struct ir_val *mufu(int op, struct ir_val *x)
{
    struct ir_val *v = op1(IR_MUFU, IR_F32, x);
    v->aux = op;
    return v;
}

// float math on halves runs in float: widen, compute, narrow
static struct ir_val *widen(struct ir_val *x)
{
    return x->type == IR_F16 ? op1(IR_H2F, IR_F32, x) : x;
}

static struct ir_val *narrow(struct ir_val *x, enum ir_type t)
{
    return t == IR_F16 ? op1(IR_F2H, IR_F16, x) : x;
}

static struct ir_val *sel(struct ir_val *c, struct ir_val *a, struct ir_val *b)
{
    return op3(IR_SEL, a->type, c, a, b);
}

static struct ir_val *fcmp(int pred, struct ir_val *a, struct ir_val *b)
{
    struct ir_val *v = op2(IR_CMP, IR_PRED, a, b);
    v->aux = IR_FCMP | pred;
    return v;
}

enum { FCMP_OEQ = 1, FCMP_OGT = 2, FCMP_OGE = 3, FCMP_OLT = 4, FCMP_OLE = 5, FCMP_UNO = 8, FCMP_UNE = 14 };

static struct ir_val *clampf(struct ir_val *x, struct ir_val *lo, struct ir_val *hi)
{
    return op2(IR_FMIN, IR_F32, op2(IR_FMAX, IR_F32, x, lo), hi);
}

// Apple's results are the reference. precise:: (and the default with fast maths off) is within an ulp or two with
// C's special cases, and flushes denormals like the arithmetic does except where a tiny x gives x back. fast::, the
// default and so nearly every shader, takes cheap MUFU forms instead (cheap_math), ulps out next to Apple's.

#define PIO2_HI 1.57079637f
#define PIO2_LO -4.37113883e-8f
#define PI_HI 3.14159274f
#define PI_LO -8.74227766e-8f
#define LOG2E_HI 1.44269502f
#define LOG2E_LO 1.92596303e-8f
#define LN2 0.693147182f
#define FLT_NAN 0x7fc00000u

static struct ir_val *icmp(int cond, struct ir_val *a, struct ir_val *b);
static struct ir_val *iminmax(enum ir_op op, struct ir_val *a, struct ir_val *b, int is_signed);

static struct ir_val *ik(uint32_t v)
{
    return konst(IR_I32, v);
}

static struct ir_val *ibin(enum ir_op op, struct ir_val *a, struct ir_val *b)
{
    return op2(op, IR_I32, a, b);
}

static struct ir_val *addf(struct ir_val *a, struct ir_val *b)
{
    return op2(IR_FADD, IR_F32, a, b);
}

static struct ir_val *subf(struct ir_val *a, struct ir_val *b)
{
    return op2(IR_FSUB, IR_F32, a, b);
}

static struct ir_val *mulf(struct ir_val *a, struct ir_val *b)
{
    return op2(IR_FMUL, IR_F32, a, b);
}

static struct ir_val *madf(struct ir_val *a, struct ir_val *b, struct ir_val *c)
{
    return op3(IR_FFMA, IR_F32, a, b, c);
}

static struct ir_val *absf(struct ir_val *x)
{
    return op1(IR_FABS, IR_F32, x);
}

static struct ir_val *negf(struct ir_val *x)
{
    return op1(IR_FNEG, IR_F32, x);
}

static struct ir_val *fbits(enum ir_op op, struct ir_val *x, uint32_t mask)
{
    return op2(op, IR_F32, x, konst(IR_F32, mask));
}

static struct ir_val *xorsign(struct ir_val *x, struct ir_val *s)
{
    return op2(IR_XOR, IR_F32, x, fbits(IR_AND, s, 0x80000000u));
}

static struct ir_val *isnan_ir(struct ir_val *x)
{
    return fcmp(FCMP_UNO, x, x);
}

static struct ir_val *finf(int negative)
{
    return konst(IR_F32, negative ? 0xff800000u : 0x7f800000u);
}

static struct ir_val *poly(struct ir_val *x, const float *k, unsigned n)
{
    struct ir_val *p = fconst(k[0]);
    for (unsigned i = 1; i < n; i++)
        p = madf(p, x, fconst(k[i]));
    return p;
}

// a / b: the reciprocal's product, then a Newton step, which only a zero or infinite b turns into NaN
static struct ir_val *divf(struct ir_val *a, struct ir_val *b)
{
    struct ir_val *r = mufu(MUFU_RCP, b), *q = mulf(a, r);
    struct ir_val *e = madf(madf(negf(b), q, a), r, q);
    return sel(isnan_ir(e), q, e);
}

// 1 / b correctly rounded but for the odd tie: MUFU.RCP's reciprocal, then a Newton step, which only a zero or
// infinite b turns into NaN
static struct ir_val *rcp_exact(struct ir_val *b)
{
    struct ir_val *r = mufu(MUFU_RCP, b);
    struct ir_val *e = madf(madf(negf(b), r, fconst(1)), r, r);
    return sel(isnan_ir(e), r, e);
}

// precise: a Newton step on the exact remainder, where there's one (not at 0 or inf)
static struct ir_val *sqrtf_ir(struct ir_val *x, int precise)
{
    struct ir_val *s = mufu(MUFU_SQRT, x);
    if (!precise)
        return s;
    struct ir_val *e = madf(madf(negf(s), s, x), mulf(mufu(MUFU_RSQ, x), fconst(0.5f)), s);
    return sel(isnan_ir(e), s, e);
}

// 2^(hi + lo), lo small next to hi: 2^n exactly times 2^f for f = hi - n + lo, n clamped to the normal range so
// that what's past it overflows or underflows MUFU.EX2 instead, and 2^f from a polynomial on [-1/2, 1/2] (an ulp,
// where MUFU.EX2 is off by two)
static struct ir_val *exp2_split(struct ir_val *hi, struct ir_val *lo)
{
    static const float k[] = { 0.000154035122f, 0.00133907353f, 0.00961823761f, 0.0555035733f, 0.240226492f,
                               0.693147182f };
    struct ir_val *n = op1(IR_FRND, IR_F32, hi);
    n->aux = SASS_RND_NEAREST;
    n = clampf(n, fconst(-126), fconst(127));
    struct ir_val *i = op1(IR_F2I, IR_I32, n);
    i->aux = 1;
    struct ir_val *scale = ibin(IR_SHL, ibin(IR_ADD, i, ik(127)), ik(23));
    lo = sel(fcmp(FCMP_OLT, absf(hi), fconst(256)), lo, fconst(0));  // NaN beside an infinite hi
    struct ir_val *f = addf(subf(hi, n), lo), *p = mufu(MUFU_EX2, f);
    p = sel(fcmp(FCMP_OLE, absf(f), fconst(0.5f)), madf(f, poly(f, k, 6), fconst(1)), p);
    return mulf(p, scale);
}

// e^x, x log2(e) carried in two parts: its rounding alone would cost tens of ulps near the ends of the range
static struct ir_val *exp_precise(struct ir_val *x, float k_hi, float k_lo, float bias)
{
    struct ir_val *hi = mulf(x, fconst(k_hi));
    struct ir_val *lo = madf(x, fconst(k_lo), madf(x, fconst(k_hi), negf(hi)));
    return exp2_split(bias ? subf(hi, fconst(bias)) : hi, lo);
}

// log(x) times k_near near 1, times k_far / ln 2 elsewhere: MUFU.LG2's error is absolute, so within 1/16 of 1 it's
// log(1 + u) = u - u^2/2 + u^3 P(u) instead
static struct ir_val *log_ir(struct ir_val *x, float k_near, float k_far)
{
    static const float k[] = { -0.167180747f, 0.200546488f, -0.249999687f, 0.333333105f };
    struct ir_val *u = subf(x, fconst(1));
    struct ir_val *near = madf(mulf(u, u), madf(u, poly(u, k, 4), fconst(-0.5f)), u);
    struct ir_val *far = mufu(MUFU_LG2, x);
    if (k_near != 1)
        near = mulf(near, fconst(k_near));
    if (k_far != 1)
        far = mulf(far, fconst(k_far));
    return sel(fcmp(FCMP_OLT, absf(u), fconst(0.0625f)), near, far);
}

// x = r + q pi/2, r in [-pi/4, pi/4]; only q mod 4 matters
struct quadrant {
    struct ir_val *r, *q;
};

// |x| = m 2^(e - 150): x 2/pi mod 4 is m times the 96 bits of 2/pi from bit e - 151 on, in 32-bit limbs (Payne and
// Hanek); two bits of quadrant, then a 64-bit fraction, rounded to the nearest quadrant, times pi/2
static struct quadrant payne_hanek(struct ir_val *x)
{
    static const uint32_t k_2_pi[] = {
        0, 0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0, 0xdb629599, 0x3c439041, 0xfe5163ab,
    };
    struct ir_val *bits = fbits(IR_AND, x, 0x7fffffff);
    struct ir_val *m = ibin(IR_OR, ibin(IR_AND, bits, ik(0x7fffff)), ik(0x800000));
    struct ir_val *at = ibin(IR_SUB, ibin(IR_SHR, bits, ik(23)), ik(120));  // the bit in k_2_pi, its 0 word first
    struct ir_val *w = ibin(IR_SHR, at, ik(5)), *sh = ibin(IR_AND, at, ik(31)), *back = ibin(IR_SUB, ik(31), sh);
    struct ir_val *t[4], *win[3];
    for (unsigned i = 0; i < 4; i++) {
        t[i] = ik(k_2_pi[i]);
        for (unsigned k = 1; k <= 4; k++)
            t[i] = sel(icmp(IR_EQ, w, ik(k)), ik(k_2_pi[k + i]), t[i]);
    }
    for (unsigned i = 0; i < 3; i++)  // never a shift by 32
        win[i] = ibin(IR_OR, ibin(IR_SHL, t[i], sh), ibin(IR_SHR, ibin(IR_SHR, t[i + 1], ik(1)), back));
    // m * win: bits 30 to 95 matter, the quadrant at 94 and 95
    struct ir_val *r0 = ibin(IR_MUL, m, win[2]), *l1 = ibin(IR_MUL, m, win[1]), *l0 = ibin(IR_MUL, m, win[0]);
    struct ir_val *r1 = ibin(IR_ADD, ibin(IR_MULHI, m, win[2]), l1);
    struct ir_val *r2 = ibin(IR_ADD, ibin(IR_ADD, ibin(IR_MULHI, m, win[1]), l0), sel(icmp(IR_LT, r1, l1), ik(1), ik(0)));
    struct ir_val *q = ibin(IR_SHR, r2, ik(30));
    struct ir_val *hi = ibin(IR_OR, ibin(IR_SHL, r2, ik(2)), ibin(IR_SHR, r1, ik(30)));
    struct ir_val *lo = ibin(IR_OR, ibin(IR_SHL, r1, ik(2)), ibin(IR_SHR, r0, ik(30)));
    // half a quadrant or more rounds up to the next one and leaves the fraction negative: its magnitude by one's
    // complement, 2^-64 out
    struct ir_val *neg = ibin(IR_SAR, hi, ik(31));
    q = ibin(IR_SUB, q, neg);
    hi = ibin(IR_XOR, hi, neg);
    lo = ibin(IR_XOR, lo, neg);
    struct ir_val *lz = iminmax(IR_MIN, ibin(IR_SUB, ik(31), op1(IR_FLO, IR_I32, hi)), ik(31), 0);
    struct ir_val *top = ibin(IR_OR, ibin(IR_SHL, hi, lz), ibin(IR_SHR, ibin(IR_SHR, lo, ik(1)), ibin(IR_SUB, ik(31), lz)));
    struct ir_val *f = op1(IR_I2F, IR_F32, top);
    struct ir_val *r = mulf(mulf(f, ibin(IR_SHL, ibin(IR_SUB, ik(95), lz), ik(23))), fconst(PIO2_HI));
    r = op2(IR_XOR, IR_F32, r, ibin(IR_AND, neg, ik(0x80000000u)));
    struct quadrant k = { xorsign(r, x), sel(fcmp(FCMP_OLT, x, fconst(0)), ibin(IR_SUB, ik(0), q), q) };
    return k;
}

// how far a sin, cos or tan is exact: precise ones everywhere; Apple's fast ones to about 2^20 revolutions (sin and
// cos, which give 0 past it, M4: right at 6.16e6, 0 at 7.39e6) and tan to 2^24 (past it, mostly NaN) with the
// three-part reduction alone; half's inputs never need more
enum { TRIG_PRECISE, TRIG_HALF, TRIG_FAST };

// x - q pi/2 with pi/2 in three parts, exact while q pi/2's first part is (|x| to 2^25); precise, |x| past 105615,
// where the three parts' error starts to show, goes to Payne-Hanek in the lanes that need it. x 2/pi rounds, so
// past 2^17 or so q can be a quadrant out and r outside [-pi/4, pi/4], where the polynomials aren't good: fast
// takes a second, short reduction of r, which is cheaper than a more exact q
static struct quadrant reduce_pio2(struct ir_val *x, int mode)
{
    struct ir_val *qf = op1(IR_FRND, IR_F32, mulf(x, fconst(0.636619747f)));
    qf->aux = SASS_RND_NEAREST;
    struct ir_val *r = madf(qf, fconst(-1.57079637f), x);
    r = madf(qf, fconst(4.37113883e-8f), r);
    r = madf(qf, fconst(1.71512451e-15f), r);
    struct quadrant k = { r, op1(IR_F2I, IR_I32, qf) };
    k.q->aux = 1;
    if (mode == TRIG_FAST) {
        struct ir_val *adj = op1(IR_FRND, IR_F32, mulf(r, fconst(0.636619747f)));
        adj->aux = SASS_RND_NEAREST;
        k.r = madf(adj, fconst(4.37113883e-8f), madf(adj, fconst(-1.57079637f), r));
        struct ir_val *dq = op1(IR_F2I, IR_I32, adj);
        dq->aux = 1;
        k.q = ibin(IR_ADD, k.q, dq);
    }
    if (mode != TRIG_PRECISE)
        return k;
    struct ir_val *ax = absf(x);
    struct ir_val *beyond = op2(IR_PAND, IR_PRED, fcmp(FCMP_OGT, ax, fconst(105615)), fcmp(FCMP_OLT, ax, finf(0)));
    struct ir_block *from = g_cur, *slow = ir_block(g_f), *join = ir_block(g_f);
    ir_add(from, IR_CBR, IR_VOID, 1)->src[0] = beyond;
    ir_edge(from, slow);
    ir_edge(from, join);
    g_cur = slow;
    struct quadrant ph = payne_hanek(x);
    ir_add(g_cur, IR_BR, IR_VOID, 0);
    ir_edge(g_cur, join);
    g_cur = join;
    struct ir_val *pr = ir_add(join, IR_PHI, IR_F32, 2), *pq = ir_add(join, IR_PHI, IR_I32, 2);
    pr->src[0] = k.r;
    pr->src[1] = ph.r;
    pq->src[0] = k.q;
    pq->src[1] = ph.q;
    k.r = pr;
    k.q = pq;
    return k;
}

static struct ir_val *is_odd(struct ir_val *q)
{
    return icmp(IR_NE, ibin(IR_AND, q, ik(1)), ik(0));
}

// the sign bit where (q + add) mod 4 is 2 or 3
static struct ir_val *quadrant_sign(struct ir_val *q, unsigned add)
{
    return ibin(IR_SHL, ibin(IR_AND, add ? ibin(IR_ADD, q, ik(add)) : q, ik(2)), ik(30));
}

// sin x and cos x from Cephes' polynomials on [-pi/4, pi/4]
static void sincos_ir(struct ir_val *x, struct ir_val **sin_out, struct ir_val **cos_out, int mode)
{
    static const float k_sin[] = { -1.9515295891e-4f, 8.3321608736e-3f, -1.6666654611e-1f };
    static const float k_cos[] = { 2.443315711809948e-5f, -1.388731625493765e-3f, 4.166664568298827e-2f };
    struct quadrant k = reduce_pio2(x, mode);
    struct ir_val *r2 = mulf(k.r, k.r);
    struct ir_val *sr = madf(mulf(k.r, r2), poly(r2, k_sin, 3), k.r);
    struct ir_val *cr = madf(r2, madf(r2, poly(r2, k_cos, 3), fconst(-0.5f)), fconst(1));
    struct ir_val *in_range = mode == TRIG_FAST ? fcmp(FCMP_OLT, absf(x), fconst(6588397.0f)) : NULL;  // not NaN
    if (sin_out) {
        struct ir_val *s = op2(IR_XOR, IR_F32, sel(is_odd(k.q), cr, sr), quadrant_sign(k.q, 0));
        *sin_out = sel(fcmp(FCMP_OLT, absf(x), fconst(0x1p-12f)), x, s);  // keeps denormals and -0
        if (mode == TRIG_FAST)
            *sin_out = sel(in_range, *sin_out, fconst(0));
    }
    if (cos_out) {
        *cos_out = op2(IR_XOR, IR_F32, sel(is_odd(k.q), sr, cr), quadrant_sign(k.q, 1));
        if (mode == TRIG_FAST)
            *cos_out = sel(in_range, *cos_out, fconst(0));
    }
}

static struct ir_val *tan_ir(struct ir_val *x, int mode)
{
    static const float k_tan[] = { 9.38540185543e-3f, 3.11992232697e-3f, 2.44301354525e-2f,
                                   5.34112807005e-2f, 1.33387994085e-1f, 3.33331568548e-1f };
    struct quadrant k = reduce_pio2(x, mode);
    struct ir_val *r2 = mulf(k.r, k.r);
    struct ir_val *t = madf(mulf(k.r, r2), poly(r2, k_tan, 6), k.r);
    t = sel(is_odd(k.q), divf(fconst(-1), t), t);
    if (mode == TRIG_FAST)
        t = sel(fcmp(FCMP_OLT, absf(x), fconst(0x1p24f)), t, konst(IR_F32, FLT_NAN));
    return sel(fcmp(FCMP_OLT, absf(x), fconst(0x1p-12f)), x, t);
}

// sin(pi x) and cos(pi x): x - q/2 is exact, so there's no reduction error; past 2^24 x is an even integer. cospi
// is never -0, as Apple's isn't (so tanpi(0.5) is +inf).
static void sincospi_ir(struct ir_val *x, struct ir_val **sin_out, struct ir_val **cos_out)
{
    static const float k_sin[] = { -0.588681519f, 2.54985952f, -5.16771269f };
    static const float k_cos[] = { 0.258082092f, -1.33748591f, 4.05876064f, -4.93480206f };
    struct ir_val *qf = op1(IR_FRND, IR_F32, addf(x, x));
    qf->aux = SASS_RND_NEAREST;
    struct ir_val *big = fcmp(FCMP_OGE, absf(x), fconst(0x1p24f));
    struct ir_val *r = sel(big, mulf(x, fconst(0)), madf(qf, fconst(-0.5f), x));  // x * 0: NaN for inf
    struct ir_val *q = op1(IR_F2I, IR_I32, qf);
    q->aux = 1;
    q = sel(big, ik(0), q);
    struct ir_val *r2 = mulf(r, r);
    struct ir_val *sr = madf(mulf(r, r2), poly(r2, k_sin, 3), mulf(r, fconst(PI_HI)));
    struct ir_val *cr = madf(r2, poly(r2, k_cos, 4), fconst(1));
    if (sin_out)
        *sin_out = op2(IR_XOR, IR_F32, sel(is_odd(q), cr, sr), quadrant_sign(q, 0));
    if (cos_out)
        *cos_out = addf(op2(IR_XOR, IR_F32, sel(is_odd(q), sr, cr), quadrant_sign(q, 1)), fconst(0));
}

// for |x| over a half (pi/2 - asin |x|) / 2, otherwise asin |x|: asin t = t + t z P(z), z = t^2, with
// t = sqrt((1 - |x|) / 2) for the first (Cephes' asinf)
static struct ir_val *asin_core(struct ir_val *ax, struct ir_val *big)
{
    static const float k[] = { 4.2163199048e-2f, 2.4181311049e-2f, 4.5470025998e-2f, 7.4953002686e-2f,
                               1.6666752422e-1f };
    struct ir_val *z = sel(big, mulf(subf(fconst(1), ax), fconst(0.5f)), mulf(ax, ax));
    struct ir_val *t = sel(big, sqrtf_ir(z, 1), ax);
    return madf(mulf(t, z), poly(z, k, 5), t);
}

// atan z for z in [0, 1]
static struct ir_val *atan01(struct ir_val *z)
{
    static const float k[] = { 0.00392521592f, -0.0202347394f, 0.0492059328f, -0.080300495f,
                               0.108731121f,   -0.142583236f, 0.199987665f,  -0.333333254f };
    struct ir_val *z2 = mulf(z, z);
    return madf(mulf(z, z2), poly(z2, k, 8), z);
}

static struct ir_val *atan_ir(struct ir_val *x)
{
    struct ir_val *ax = absf(x), *inv = fcmp(FCMP_OGT, ax, fconst(1));
    struct ir_val *a = atan01(sel(inv, divf(fconst(1), ax), ax));
    return xorsign(sel(inv, subf(fconst(PIO2_HI), subf(a, fconst(PIO2_LO))), a), x);
}

static struct ir_val *atan2_ir(struct ir_val *y, struct ir_val *x)
{
    // atan of the smaller magnitude over the larger, put in its octant; with that ratio 0 for two zeros and 1 for
    // two infinities, the moves give C's atan2(+-0, +-0) and atan2(+-inf, +-inf) as well
    struct ir_val *ax = absf(x), *ay = absf(y);
    struct ir_val *hi = op2(IR_FMAX, IR_F32, ax, ay), *lo = op2(IR_FMIN, IR_F32, ax, ay);
    struct ir_val *s = sel(fcmp(FCMP_OGT, hi, fconst(0x1p64f)), fconst(0x1p-64f), fconst(1));  // or 1 / hi flushes
    struct ir_val *z = divf(mulf(lo, s), mulf(hi, s));
    z = sel(fcmp(FCMP_OEQ, hi, fconst(0)), fconst(0), sel(fcmp(FCMP_OEQ, lo, finf(0)), fconst(1), z));
    struct ir_val *a = atan01(z);
    a = sel(fcmp(FCMP_OGT, ay, ax), subf(fconst(PIO2_HI), subf(a, fconst(PIO2_LO))), a);
    a = sel(icmp(IR_NE, fbits(IR_AND, x, 0x80000000u), ik(0)), subf(fconst(PI_HI), subf(a, fconst(PI_LO))), a);
    return sel(fcmp(FCMP_UNO, x, y), addf(x, y), xorsign(a, y));
}

// log2 x in two parts, x normal and positive: x = m 2^e with m in [sqrt(1/2), sqrt(2)), and
// log m = u + u^3/12 + u^5/80 + ... for u = 2f / (2 + f), f = m - 1, with u's remainder (exact) as its low part
static void log2_split(struct ir_val *xb, struct ir_val **hi, struct ir_val **lo)
{
    static const float k[] = { 1.0f / 53248, 1.0f / 11264, 1.0f / 2304, 1.0f / 448, 1.0f / 80, 1.0f / 12 };
    struct ir_val *ke = ibin(IR_AND, ibin(IR_SUB, xb, ik(0x3f3504f3)), ik(0xff800000u));
    struct ir_val *e = op1(IR_I2F, IR_F32, ibin(IR_SAR, ke, ik(23)));
    e->aux = 1;
    struct ir_val *f = subf(ibin(IR_SUB, xb, ke), fconst(1));
    struct ir_val *r = mufu(MUFU_RCP, addf(f, fconst(2)));
    struct ir_val *u = mulf(addf(f, f), r);
    // 2f - u (2 + f) = 2 (f - u) - u f, and f - u, about f^2 / 2, is exact
    struct ir_val *ul = mulf(madf(negf(u), f, mulf(subf(f, u), fconst(2))), r);
    struct ir_val *u2 = mulf(u, u);
    struct ir_val *tail = madf(mulf(u, u2), poly(u2, k, 6), ul);
    struct ir_val *p = mulf(u, fconst(LOG2E_HI));
    struct ir_val *pl = madf(u, fconst(LOG2E_LO), madf(tail, fconst(LOG2E_HI), madf(u, fconst(LOG2E_HI), negf(p))));
    *hi = addf(e, p);
    *lo = addf(addf(subf(e, *hi), p), pl);
}

// pow (C's special cases) or powr (x < 0 is NaN, as are 0^0, inf^0 and 1^inf, which exp2(y log2 x) gives anyway).
// precise carries log2 x and y log2 x in two parts; half maths, with its 2^-11 ulp, does without (and gets the
// special cases all the same). 0, inf, NaN, x < 0 and denormals (MUFU flushes them) take MUFU.LG2's log2.
static struct ir_val *pow_ir(struct ir_val *x, struct ir_val *y, int powr, int precise)
{
    struct ir_val *xb = powr ? x : fbits(IR_AND, x, 0x7fffffff);
    struct ir_val *l = mufu(MUFU_LG2, xb), *r;
    if (precise) {
        struct ir_val *normal = icmp(IR_LT, ibin(IR_SUB, xb, ik(0x00800000)), ik(0x7f000000));
        struct ir_val *hi, *lo;
        log2_split(xb, &hi, &lo);
        hi = sel(normal, hi, l);
        struct ir_val *th = mulf(y, hi);
        r = exp2_split(th, madf(y, sel(normal, lo, fconst(0)), madf(y, hi, negf(th))));
    } else {
        r = mufu(MUFU_EX2, mulf(y, l));
    }
    if (powr)
        return r;
    struct ir_val *ay = absf(y);
    struct ir_val *is_int = fcmp(FCMP_OEQ, op1(IR_FRND, IR_F32, y), y);
    is_int->src[0]->aux = SASS_RND_TRUNC;
    struct ir_val *yi = op1(IR_F2I, IR_I32, y);
    yi->aux = 1;
    struct ir_val *odd = op2(IR_PAND, IR_PRED, op2(IR_PAND, IR_PRED, is_int, fcmp(FCMP_OLT, ay, fconst(0x1p24f))),
                             icmp(IR_NE, ibin(IR_AND, yi, ik(1)), ik(0)));
    r = op2(IR_XOR, IR_F32, r, sel(odd, fbits(IR_AND, x, 0x80000000u), ik(0)));
    struct ir_val *neg = op2(IR_PAND, IR_PRED, fcmp(FCMP_OLT, x, fconst(0)), fcmp(FCMP_OGT, x, finf(1)));
    r = sel(op2(IR_PAND, IR_PRED, neg, op1(IR_PNOT, IR_PRED, is_int)), konst(IR_F32, FLT_NAN), r);
    if (!precise)  // Apple's gives x itself for a y of 1, where exp2(log2) would be a few ulps out
        r = sel(fcmp(FCMP_OEQ, y, fconst(1)), x, r);
    struct ir_val *one = op2(IR_POR, IR_PRED, fcmp(FCMP_OLT, ay, fconst(0x1p-126f)), fcmp(FCMP_OEQ, x, fconst(1)));
    one = op2(IR_POR, IR_PRED, one,
              op2(IR_PAND, IR_PRED, fcmp(FCMP_OEQ, absf(x), fconst(1)), fcmp(FCMP_OEQ, ay, finf(0))));
    return sel(one, fconst(1), r);
}

// r - q d for q = trunc(r / d), r / d below 2^23: exact, with q put right where the quotient rounded across an
// integer
static struct ir_val *rem_step(struct ir_val *r, struct ir_val *d)
{
    struct ir_val *q = op1(IR_FRND, IR_F32, divf(r, d));
    q->aux = SASS_RND_TRUNC;
    struct ir_val *v = madf(negf(q), d, r);
    v = sel(fcmp(FCMP_OLT, v, fconst(0)), addf(v, d), v);
    return sel(fcmp(FCMP_OGE, v, d), subf(v, d), v);
}

// fmod exactly, as precise maths wants it: while |x| is past 2^22 |y|, |x| mod |y| 2^k takes q's top bits (in a
// loop: there can be 11 such steps), then |y| itself
static struct ir_val *fmod_precise(struct ir_val *x, struct ir_val *y)
{
    struct ir_val *ax = absf(x), *ay = absf(y);
    struct ir_val *ok = op2(IR_PAND, IR_PRED, fcmp(FCMP_OGE, ay, fconst(0x1p-126f)), fcmp(FCMP_OLT, ax, finf(0)));
    struct ir_block *from = g_cur, *head = ir_block(g_f), *body = ir_block(g_f), *done = ir_block(g_f);
    ir_add(from, IR_BR, IR_VOID, 0);
    ir_edge(from, head);
    g_cur = head;
    struct ir_val *r = ir_add(head, IR_PHI, IR_F32, 2);
    // the way out first: block order then puts the body ahead of it, so the loop gets a convergence region
    struct ir_val *far = op2(IR_PAND, IR_PRED, ok, fcmp(FCMP_OGT, r, mulf(ay, fconst(0x1p22f))));
    ir_add(head, IR_CBR, IR_VOID, 1)->src[0] = op1(IR_PNOT, IR_PRED, far);
    ir_edge(head, done);
    ir_edge(head, body);
    g_cur = body;
    // |y| 2^k, k the exponent difference less 22, by adding to its exponent field
    struct ir_val *k = ibin(IR_SUB, ibin(IR_AND, r, ik(0x7f800000)), ibin(IR_AND, ay, ik(0x7f800000)));
    struct ir_val *step = rem_step(r, ibin(IR_ADD, ay, ibin(IR_SUB, k, ik(22 << 23))));
    ir_add(g_cur, IR_BR, IR_VOID, 0);
    ir_edge(g_cur, head);
    r->src[0] = ax;
    r->src[1] = step;
    g_cur = done;
    struct ir_val *v = sel(fcmp(FCMP_OEQ, ay, finf(0)), x, xorsign(rem_step(r, ay), x));
    return sel(ok, v, konst(IR_F32, FLT_NAN));  // y 0 or NaN, x inf or NaN
}

// ldexp in two steps, so that x 2^n can be normal when 2^n isn't
static struct ir_val *ldexp_ir(struct ir_val *x, struct ir_val *n)
{
    n = iminmax(IR_MAX, iminmax(IR_MIN, n, ik(300), 1), ik((uint32_t)-300), 1);
    struct ir_val *h = ibin(IR_SAR, n, ik(1));
    struct ir_val *p[2] = { h, ibin(IR_SUB, n, h) };
    for (unsigned i = 0; i < 2; i++) {
        struct ir_val *c = iminmax(IR_MAX, iminmax(IR_MIN, p[i], ik(127), 1), ik((uint32_t)-126), 1);
        x = mulf(x, ibin(IR_SHL, ibin(IR_ADD, c, ik(127)), ik(23)));
    }
    return x;
}

// the next float (or half) after x towards y, on the bits: Apple's steps through denormals
static struct ir_val *nextafter_ir(struct ir_val *x, struct ir_val *y, enum ir_type t)
{
    uint32_t sign = t == IR_F16 ? 0x8000 : 0x80000000u, inf = t == IR_F16 ? 0x7c00 : 0x7f800000;
    struct ir_val *v[2] = { x, y }, *o[2], *nan = NULL;
    for (unsigned i = 0; i < 2; i++) {  // to integers in the floats' order, both zeros 0
        struct ir_val *mag = ibin(IR_AND, v[i], ik(sign - 1));
        o[i] = sel(icmp(IR_NE, ibin(IR_AND, v[i], ik(sign)), ik(0)), ibin(IR_SUB, ik(0), mag), mag);
        struct ir_val *is_nan = icmp(IR_GT, mag, ik(inf));
        nan = nan ? op2(IR_POR, IR_PRED, nan, is_nan) : is_nan;
    }
    struct ir_val *step = sel(icmp(IR_GT | IR_SIGNED, o[1], o[0]), ik(1), ik((uint32_t)-1));
    struct ir_val *s = ibin(IR_ADD, o[0], step);
    s = sel(icmp(IR_LT | IR_SIGNED, s, ik(0)), ibin(IR_OR, ibin(IR_SUB, ik(0), s), ik(sign)), s);
    s = sel(icmp(IR_EQ, o[0], o[1]), y, s);
    s = sel(nan, ik(t == IR_F16 ? 0x7e00 : FLT_NAN), s);
    return op2(IR_OR, t, s, konst(t, 0));
}

// sinh, cosh, tanh: Cephes' polynomials for a small |x|, x itself for a tiny one, and e^|x| / 2 so that sinh and
// cosh reach 89.4
static struct ir_val *hyperbolic(const char *name, struct ir_val *x)
{
    static const float k_sinh[] = { 2.03721912945e-4f, 8.33028376239e-3f, 1.66667160211e-1f };
    static const float k_tanh[] = { -5.70498872745e-3f, 2.06390887954e-2f, -5.37397155531e-2f, 1.33314422036e-1f,
                                    -3.33332819422e-1f };
    struct ir_val *ax = absf(x), *x2 = mulf(x, x), *r;
    if (is_fn(name, "tanh")) {
        struct ir_val *e = exp_precise(addf(ax, ax), LOG2E_HI, LOG2E_LO, 0);
        r = madf(divf(fconst(1), addf(e, fconst(1))), fconst(-2), fconst(1));
        r = sel(fcmp(FCMP_OLT, ax, fconst(0.625f)), madf(mulf(ax, x2), poly(x2, k_tanh, 5), ax), r);
    } else {
        struct ir_val *h = exp_precise(ax, LOG2E_HI, LOG2E_LO, 1);
        r = madf(mufu(MUFU_RCP, h), fconst(is_fn(name, "cosh") ? 0.25f : -0.25f), h);
        if (is_fn(name, "cosh"))
            return r;
        r = sel(fcmp(FCMP_OLT, ax, fconst(1)), madf(mulf(ax, x2), poly(x2, k_sinh, 3), ax), r);
    }
    r = xorsign(r, x);
    return sel(fcmp(FCMP_OLT, ax, fconst(0x1p-12f)), x, r);
}

// asinh, acosh, atanh: Cephes' polynomials for a small |x|, log(2|x|) for a big one
static struct ir_val *inverse_hyperbolic(const char *name, struct ir_val *x)
{
    static const float k_asinh[] = { 2.0122003309e-2f, -4.2699340972e-2f, 7.4847586088e-2f, -1.6666288134e-1f };
    static const float k_acosh[] = { 1.7596881071e-3f, -7.5272886713e-3f, 2.6454905019e-2f, -1.1784741703e-1f,
                                     1.4142135263f };
    static const float k_atanh[] = { 1.81740078349e-1f, 8.24370301058e-2f, 1.46691431730e-1f, 1.99782164500e-1f,
                                     3.33337300303e-1f };
    struct ir_val *ax = absf(x), *x2 = mulf(x, x), *r;
    if (is_fn(name, "atanh")) {
        r = mulf(log_ir(divf(addf(fconst(1), ax), subf(fconst(1), ax)), 1, LN2), fconst(0.5f));
        r = sel(fcmp(FCMP_OLT, ax, fconst(0.5f)), madf(mulf(ax, x2), poly(x2, k_atanh, 5), ax), r);
        r = sel(fcmp(FCMP_OGT, ax, fconst(1)), konst(IR_F32, FLT_NAN), r);  // FLT_MAX's ratio rounds to -1
    } else {
        int acosh = is_fn(name, "acosh");
        struct ir_val *s = madf(x, x, fconst(acosh ? -1 : 1));
        struct ir_val *v = acosh ? x : ax, *big = fcmp(FCMP_OGT, v, fconst(0x1p12f));
        r = log_ir(sel(big, v, addf(v, sqrtf_ir(s, 1))), 1, LN2);
        r = sel(big, addf(r, fconst(LN2)), r);
        if (acosh) {
            struct ir_val *z = subf(x, fconst(1));
            return sel(fcmp(FCMP_OLT, x, fconst(1.5f)), mulf(poly(z, k_acosh, 5), sqrtf_ir(z, 1)), r);
        }
        r = sel(fcmp(FCMP_OLT, ax, fconst(0.5f)), madf(mulf(ax, x2), poly(x2, k_asinh, 4), ax), r);
    }
    r = xorsign(r, x);
    return sel(fcmp(FCMP_OLT, ax, fconst(0x1p-12f)), x, r);
}

// MUFU.SIN's error is absolute, tens of ulps next to Apple's answers (which are as exact as the precise ones, to
// where its fast sin and cos give 0, as they do for inf and NaN), so these reduce and take polynomials
static void cheap_sincos(struct ir_val *x, int fast, struct ir_val **sin_out, struct ir_val **cos_out)
{
    sincos_ir(x, sin_out, cos_out, fast ? TRIG_FAST : TRIG_HALF);
}

// atan with a short polynomial on [0, 1] (a few ulps)
static struct ir_val *cheap_atan(struct ir_val *x)
{
    static const float k[] = { -0.0134809297f, 0.0574785769f, -0.121240333f, 0.195636481f, -0.332994699f,
                               0.999995649f };
    struct ir_val *ax = absf(x), *inv = fcmp(FCMP_OGT, ax, fconst(1));
    struct ir_val *z = sel(inv, mufu(MUFU_RCP, ax), ax);
    struct ir_val *a = mulf(z, poly(mulf(z, z), k, 6));
    return xorsign(sel(inv, subf(fconst(PIO2_HI), a), a), x);
}

// fast:: (the default, so nearly every shader) and half maths, whose ulp is 2^-11: MUFU and short polynomials,
// a few instructions each, ulps out next to Apple's, keeping the edges a compare and select buy; NULL for the rest
static struct ir_val *cheap_math(const char *name, struct ir_val *a, struct ir_val *b, int fast)
{
    // acos x = sqrt(1 - x) P(x) on [0, 1] (1e-6 out), P(0) pi/2 so that asin 0 is 0
    static const float k_acos[] = { -0.00506330794f, 0.0209922828f, -0.0462528504f, 0.0882927552f, -0.214560702f,
                                    PIO2_HI };
    struct ir_val *s, *c;
    if (is_fn(name, "sin") || is_fn(name, "cos")) {
        int sin = is_fn(name, "sin");
        cheap_sincos(a, fast, sin ? &s : NULL, sin ? NULL : &c);
        return sin ? s : c;
    }
    if (is_fn(name, "tan"))
        return tan_ir(a, fast ? TRIG_FAST : TRIG_HALF);
    if (is_fn(name, "asin") || is_fn(name, "acos")) {  // NaN past 1 from the sqrt
        if (fast)  // Apple's fast ones are as exact as the precise ones
            return NULL;
        struct ir_val *ax = absf(a);
        struct ir_val *v = mulf(mufu(MUFU_SQRT, subf(fconst(1), ax)), poly(ax, k_acos, 6));
        if (is_fn(name, "asin"))
            return xorsign(subf(fconst(PIO2_HI), v), a);
        return sel(fcmp(FCMP_OLT, a, fconst(0)), subf(fconst(PI_HI), v), v);
    }
    if (is_fn(name, "atan"))  // Apple's fast atan is as exact as the precise one
        return fast ? NULL : cheap_atan(a);
    if (is_fn(name, "atan2")) {
        if (!fast)
            return NULL;
        // atan(y / x), turned by pi into the left half-plane, as Apple's: fast atan2(0, 0) is NaN
        struct ir_val *t = atan_ir(mulf(a, mufu(MUFU_RCP, b)));
        struct ir_val *turn = sel(fcmp(FCMP_OGE, a, fconst(0)), fconst(PI_HI), fconst(-PI_HI));
        return sel(fcmp(FCMP_OLT, b, fconst(0)), addf(t, turn), t);
    }
    if (is_fn(name, "exp"))
        return mufu(MUFU_EX2, mulf(a, fconst(LOG2E_HI)));
    if (is_fn(name, "exp10"))
        return mufu(MUFU_EX2, mulf(a, fconst(3.32192802f)));
    // MUFU.LG2's error is absolute (tens of thousands of ulps just below 1), Apple's fast logs are the precise ones
    if (is_fn(name, "log2"))
        return fast ? NULL : mufu(MUFU_LG2, a);
    if (is_fn(name, "log"))
        return fast ? NULL : mulf(mufu(MUFU_LG2, a), fconst(LN2));
    if (is_fn(name, "log10"))
        return fast ? NULL : mulf(mufu(MUFU_LG2, a), fconst(0.30103001f));
    if (is_fn(name, "sqrt"))
        return mufu(MUFU_SQRT, a);
    if (is_fn(name, "pow") || is_fn(name, "powr")) {
        int powr = is_fn(name, "powr");
        return pow_ir(a, b, powr, 0);  // fast has C's special cases as well
    }
    if (is_fn(name, "sinh") || is_fn(name, "cosh")) {  // (e^x -+ e^-x) / 2: inf from e^-x past -87, Apple's isn't
        int cosh = is_fn(name, "cosh");
        if (fast) {  // of e^|x| and its reciprocal, so that the sign is only put back: sinh(-88.5) is finite
            struct ir_val *e = mufu(MUFU_EX2, mulf(absf(a), fconst(LOG2E_HI))), *r = mufu(MUFU_RCP, e);
            r = mulf(cosh ? addf(e, r) : subf(e, r), fconst(0.5f));
            return cosh ? r : xorsign(r, a);
        }
        struct ir_val *e = mufu(MUFU_EX2, mulf(a, fconst(LOG2E_HI)));
        return madf(mufu(MUFU_RCP, e), fconst(cosh ? 0.5f : -0.5f), mulf(e, fconst(0.5f)));
    }
    if (is_fn(name, "tanh")) {  // fast: (e^2x - 1) / (e^2x + 1), inf / inf = NaN for a big x as Apple's; half +-1
        struct ir_val *e = mufu(MUFU_EX2, mulf(a, fconst(2 * LOG2E_HI)));
        struct ir_val *r = mufu(MUFU_RCP, addf(e, fconst(1)));
        return fast ? mulf(subf(e, fconst(1)), r) : madf(r, fconst(-2), fconst(1));
    }
    if (is_fn(name, "asinh") || is_fn(name, "acosh")) {
        int acosh = is_fn(name, "acosh");
        s = madf(a, a, fconst(acosh ? -1 : 1));
        if (!fast) {  // log(|x| + sqrt(x^2 +- 1)), asinh's signed as x
            struct ir_val *v = acosh ? a : absf(a);
            struct ir_val *r = mulf(mufu(MUFU_LG2, addf(v, mufu(MUFU_SQRT, s))), fconst(LN2));
            if (acosh)  // x + sqrt(x^2 - 1) cancels to 0 for a big negative x, where Apple's has NaN
                return sel(fcmp(FCMP_OLT, a, fconst(1)), konst(IR_F32, FLT_NAN), r);
            return xorsign(r, a);
        }
        // as Apple's: x^2 +- 1 times its rsqrt fused into the sum (wrong for a big negative x there too), the
        // rsqrt's argument held below inf so that an infinite square gives inf
        struct ir_val *v = madf(s, mufu(MUFU_RSQ, op2(IR_FMIN, IR_F32, s, fconst(0x1.fffffep127f))), a);
        if (acosh)
            v = sel(fcmp(FCMP_OEQ, s, fconst(0)), a, v);
        return mulf(mufu(MUFU_LG2, v), fconst(LN2));
    }
    if (is_fn(name, "atanh")) {
        // fast: log((1 + x) / (1 - x)) / 2 as Apple's, whose quotient is (1 + x) times the exactly rounded
        // reciprocal; for a small x that rounding is most of the result, so the log has to be exact near 1
        if (fast)
            return log_ir(mulf(addf(fconst(1), a), rcp_exact(subf(fconst(1), a))), 0.5f, 0.5f * LN2);
        struct ir_val *q = mulf(addf(fconst(1), a), mufu(MUFU_RCP, subf(fconst(1), a)));
        return mulf(mufu(MUFU_LG2, q), fconst(0.5f * LN2));
    }
    return NULL;
}

// One lane of a math intrinsic on float/half lanes; NULL when name isn't one.
static struct ir_val *math(const char *name, LLVMValueRef call, unsigned lane, enum ir_type t)
{
    if (is_fn(name, "nextafter"))
        return nextafter_ir(get(LLVMGetOperand(call, 0), lane), get(LLVMGetOperand(call, 1), lane), t);
    int precise = strncmp(name, "air.fast_", 9) && strncmp(name, "air.native_", 11);  // OpenCL's native_ ones as fast
    struct ir_val *a = LLVMGetNumOperands(call) > 1 ? widen(get(LLVMGetOperand(call, 0), lane)) : NULL;
    struct ir_val *b = LLVMGetNumOperands(call) > 2 ? widen(get(LLVMGetOperand(call, 1), lane)) : NULL;
    struct ir_val *c = LLVMGetNumOperands(call) > 3 ? widen(get(LLVMGetOperand(call, 2), lane)) : NULL;
    struct ir_val *r;
    if ((!precise || t == IR_F16) && (r = cheap_math(name, a, b, !precise)))
        return narrow(r, t);
    if (is_fn(name, "fma") || is_fn(name, "fmuladd"))
        r = op3(IR_FFMA, IR_F32, a, b, c);
    else if (is_fn(name, "fmax3"))
        r = op2(IR_FMAX, IR_F32, op2(IR_FMAX, IR_F32, a, b), c);
    else if (is_fn(name, "fmin3"))
        r = op2(IR_FMIN, IR_F32, op2(IR_FMIN, IR_F32, a, b), c);
    else if (is_fn(name, "fmedian3")) {
        // min(max(a, b), max(min(a, b), c)) as Apple's: half leaves every NaN out; float too for a NaN a or b, but
        // one that reaches the outer compares (a NaN c, or both a and b) is read as a select takes it
        struct ir_val *lo = op2(IR_FMIN, IR_F32, a, b), *hi = op2(IR_FMAX, IR_F32, a, b);
        if (t == IR_F16) {
            r = op2(IR_FMIN, IR_F32, hi, op2(IR_FMAX, IR_F32, lo, c));
        } else {
            struct ir_val *inner = sel(fcmp(FCMP_OGT, c, lo), c, lo);
            r = sel(fcmp(FCMP_OLT, inner, hi), inner, hi);
        }
    }
    else if (is_fn(name, "atan2"))
        r = atan2_ir(a, b);
    else if (is_fn(name, "atan")) {  // Apple's flushes a denormal x even in precise maths, and fast's FLT_MIN too
        r = atan_ir(a);
        if (!precise)
            r = sel(fcmp(FCMP_OLE, absf(a), fconst(0x1p-126f)), fconst(0), r);
    }
    else if (is_fn(name, "asin")) {
        struct ir_val *ax = absf(a), *big = fcmp(FCMP_OGT, ax, fconst(0.5f));
        struct ir_val *p = asin_core(ax, big);
        r = xorsign(sel(big, addf(madf(p, fconst(-2), fconst(PIO2_HI)), fconst(PIO2_LO)), p), a);
        r = sel(fcmp(FCMP_OLT, ax, fconst(0x1p-12f)), a, r);
        if (!precise)  // and fast's flushes a denormal
            r = sel(fcmp(FCMP_OLT, ax, fconst(0x1p-126f)), fconst(0), r);
    } else if (is_fn(name, "acos")) {
        struct ir_val *ax = absf(a), *big = fcmp(FCMP_OGT, ax, fconst(0.5f));
        struct ir_val *p = asin_core(ax, big);
        struct ir_val *side = sel(fcmp(FCMP_OLT, a, fconst(0)), addf(madf(p, fconst(-2), fconst(PI_HI)), fconst(PI_LO)),
                                  mulf(p, fconst(2)));
        r = sel(big, side, subf(fconst(PIO2_HI), subf(xorsign(p, a), fconst(PIO2_LO))));
    } else if (is_fn(name, "tan"))
        r = tan_ir(a, TRIG_PRECISE);
    else if (is_fn(name, "fmax") || is_fn(name, "maxnum"))
        r = op2(IR_FMAX, IR_F32, a, b);
    else if (is_fn(name, "fmin") || is_fn(name, "minnum"))
        r = op2(IR_FMIN, IR_F32, a, b);
    else if (is_fn(name, "fabs") || is_fn(name, "abs"))
        r = op1(IR_FABS, IR_F32, a);
    else if (is_fn(name, "saturate"))  // + 0 flushes a denormal, as Apple's does
        r = addf(clampf(a, fconst(0), fconst(1)), fconst(0));
    else if (is_fn(name, "clamp") && t != IR_I32) {
        r = clampf(a, b, c);
        // fast maths flushes a denormal, so a clamp to [0, 1] is a saturate: one FADD.SAT
        if (!precise && b->op == IR_CONST && b->imm == 0 && c->op == IR_CONST && c->imm == 0x3f800000)
            r = addf(r, fconst(0));
    }
    else if (is_fn(name, "sin"))
        sincos_ir(a, &r, NULL, TRIG_PRECISE);
    else if (is_fn(name, "cos"))
        sincos_ir(a, NULL, &r, TRIG_PRECISE);
    else if (is_fn(name, "exp2"))
        r = mufu(MUFU_EX2, a);
    else if (is_fn(name, "exp"))
        r = exp_precise(a, LOG2E_HI, LOG2E_LO, 0);
    else if (is_fn(name, "log2"))
        r = log_ir(a, LOG2E_HI, 1);
    else if (is_fn(name, "log"))
        r = log_ir(a, 1, LN2);
    else if (is_fn(name, "rsqrt"))
        r = mufu(MUFU_RSQ, a);
    else if (is_fn(name, "sqrt"))
        r = sqrtf_ir(a, 1);
    else if (is_fn(name, "pow") || is_fn(name, "powr"))
        r = pow_ir(a, b, is_fn(name, "powr"), 1);
    else if (is_fn(name, "mix"))  // t b + (a - t a), as Apple's: mix(0, 1, inf) is NaN, mix(x, x, t) not always x
        r = madf(c, b, madf(negf(c), a, a));
    else if (is_fn(name, "step"))  // x >= edge ? 1 : 0, so 0 with a NaN as Apple's
        r = sel(fcmp(FCMP_OGE, b, a), fconst(1), fconst(0));
    else if (is_fn(name, "sign"))
        r = sel(fcmp(FCMP_OGT, a, fconst(0)), fconst(1), sel(fcmp(FCMP_OLT, a, fconst(0)), fconst(-1), fconst(0)));
    else if (is_fn(name, "floor") || is_fn(name, "ceil") || is_fn(name, "trunc") || is_fn(name, "rint")) {
        r = op1(IR_FRND, IR_F32, a);
        r->aux = is_fn(name, "floor") ? SASS_RND_FLOOR
                 : is_fn(name, "ceil") ? SASS_RND_CEIL
                 : is_fn(name, "trunc") ? SASS_RND_TRUNC
                                        : SASS_RND_NEAREST;
    } else if (is_fn(name, "round")) {
        // halves away from zero: trunc(x), one further out when x is at least half past it (exact, unlike
        // trunc(x + 0.5), which rounds 0.49999997 + 0.5 up to 1)
        struct ir_val *t = op1(IR_FRND, IR_F32, a);
        t->aux = SASS_RND_TRUNC;
        struct ir_val *one = op2(IR_OR, IR_F32, op2(IR_AND, IR_F32, a, konst(IR_F32, 0x80000000u)),
                                 konst(IR_F32, 0x3f800000));
        struct ir_val *up = fcmp(FCMP_OGE, op1(IR_FABS, IR_F32, op2(IR_FSUB, IR_F32, a, t)), fconst(0.5f));
        r = sel(up, op2(IR_FADD, IR_F32, t, one), t);
    } else if (is_fn(name, "dfdx") || is_fn(name, "dfdy")) {
        r = op1(IR_DERIV, IR_F32, a);
        r->aux = is_fn(name, "dfdy");
    } else if (is_fn(name, "fwidth")) {  // |d/dx| + |d/dy|
        struct ir_val *dx = op1(IR_DERIV, IR_F32, a), *dy = op1(IR_DERIV, IR_F32, a);
        dy->aux = 1;
        r = op2(IR_FADD, IR_F32, op1(IR_FABS, IR_F32, dx), op1(IR_FABS, IR_F32, dy));
    } else if (is_fn(name, "fract")) {  // x - floor(x); precise: below 1 (for tiny negative x) and 0 for inf
        struct ir_val *fl = op1(IR_FRND, IR_F32, a);
        fl->aux = SASS_RND_FLOOR;
        r = op2(IR_FSUB, IR_F32, a, fl);
        if (t == IR_F16)  // Apple's half fract is min(x - floor(x), 0x1.ffcp-1): even NaN and inf give that
            r = op2(IR_FMIN, IR_F32, r, fconst(0x1.ffcp-1f));
        else if (precise) {
            r = sel(fcmp(FCMP_OGE, r, fconst(1)), fconst(0x1.fffffep-1f), r);
            r = sel(fcmp(FCMP_OEQ, absf(a), finf(0)), fconst(0), r);
        }
    } else if (is_fn(name, "copysign"))
        r = op2(IR_OR, IR_F32, op2(IR_AND, IR_F32, a, konst(IR_F32, 0x7fffffff)),
                op2(IR_AND, IR_F32, b, konst(IR_F32, 0x80000000u)));
    else if (is_fn(name, "roundeven") || is_fn(name, "nearbyint")) {
        r = op1(IR_FRND, IR_F32, a);
        r->aux = SASS_RND_NEAREST;
    } else if (is_fn(name, "minimum") || is_fn(name, "maximum"))
        r = op2(is_fn(name, "minimum") ? IR_FMIN : IR_FMAX, IR_F32, a, b);
    else if (is_fn(name, "mad"))
        r = op3(IR_FFMA, IR_F32, a, b, c);
    else if (is_fn(name, "fmod") && precise)
        r = fmod_precise(a, b);
    else if (is_fn(name, "fmod")) {  // x - y * trunc(x / y), as Apple's fast fmod (garbage for a big x / y)
        struct ir_val *q = op1(IR_FRND, IR_F32, op2(IR_FMUL, IR_F32, a, mufu(MUFU_RCP, b)));
        q->aux = SASS_RND_TRUNC;
        r = op3(IR_FFMA, IR_F32, op1(IR_FNEG, IR_F32, q), b, a);
    } else if (is_fn(name, "fdim"))
        r = sel(fcmp(FCMP_UNO, a, b), addf(a, b), sel(fcmp(FCMP_OGT, a, b), subf(a, b), fconst(0)));
    else if (is_fn(name, "exp10"))
        r = exp_precise(a, 3.32192802f, 7.05953696e-8f, 0);
    else if (is_fn(name, "log10"))
        r = log_ir(a, 0.434294492f, 0.30103001f);
    else if (is_fn(name, "ldexp"))
        r = ldexp_ir(a, get(LLVMGetOperand(call, 1), lane));
    else if (is_fn(name, "sinpi"))
        sincospi_ir(a, &r, NULL);
    else if (is_fn(name, "cospi"))
        sincospi_ir(a, NULL, &r);
    else if (is_fn(name, "tanpi")) {
        struct ir_val *s, *c;
        sincospi_ir(a, &s, &c);
        r = divf(s, c);
    } else if (is_fn(name, "sinh") || is_fn(name, "cosh") || is_fn(name, "tanh"))
        r = hyperbolic(name, a);
    else if (is_fn(name, "asinh") || is_fn(name, "acosh") || is_fn(name, "atanh"))
        r = inverse_hyperbolic(name, a);
    else
        return NULL;
    return narrow(r, t);
}

// one lane of an integer intrinsic; NULL when name isn't one
// a narrower integer's register as a whole 32-bit value: its bits above the type's are junk (a truncation keeps the
// low bits as they were), so they're cleared, or for a signed one copies of its sign
static struct ir_val *int_extend(struct ir_val *v, unsigned bits, int is_signed)
{
    if (bits >= 32 || v->type != IR_I32)
        return v;
    if (!is_signed)
        return op2(IR_AND, IR_I32, v, konst(IR_I32, (1u << bits) - 1));
    struct ir_val *shift = konst(IR_I32, 32 - bits);
    return op2(IR_SAR, IR_I32, op2(IR_SHL, IR_I32, v, shift), shift);
}

static struct ir_val *icmp(int cond, struct ir_val *a, struct ir_val *b)
{
    struct ir_val *v = op2(IR_CMP, IR_PRED, a, b);
    v->aux = cond;
    return v;
}

static struct ir_val *iminmax(enum ir_op op, struct ir_val *a, struct ir_val *b, int is_signed)
{
    struct ir_val *v = op2(op, IR_I32, a, b);
    v->aux = is_signed;
    return v;
}

// A saturating sum or product r: of lanes narrower than 32 bits (exact in 32), clamped to the type's range (an
// unsigned one's from above only); of 32-bit ones, the limit where overflow says it didn't fit, signed on the side
// of sign's sign, unsigned sign itself.
static struct ir_val *saturate_int(struct ir_val *r, struct ir_val *overflow, struct ir_val *sign, unsigned bits,
                                   int is_signed)
{
    if (bits < 32 && is_signed)
        return iminmax(IR_MIN, iminmax(IR_MAX, r, konst(IR_I32, -(1u << (bits - 1))), 1),
                       konst(IR_I32, (1u << (bits - 1)) - 1), 1);
    if (bits < 32)
        return iminmax(IR_MIN, r, konst(IR_I32, (1u << bits) - 1), 0);
    struct ir_val *limit = is_signed ? op2(IR_XOR, IR_I32, op2(IR_SAR, IR_I32, sign, konst(IR_I32, 31)),
                                           konst(IR_I32, 0x7fffffff))
                                     : sign;
    return sel(overflow, limit, r);
}

// The rest of the integer intrinsics, on a and b as imath extends them; NULL when name isn't one. Lanes of more
// than 32 bits keep their low word, so only the counts of their bits, which take the high one as 0, are made.
static struct ir_val *ibits(const char *name, LLVMValueRef call, unsigned lane, struct ir_val *a, struct ir_val *b,
                            unsigned bits, int is_signed)
{
    enum ir_op shr = is_signed ? IR_SAR : IR_SHR;
    if (is_fn(name, "popcount") || is_fn(name, "ctpop"))
        return op1(IR_POPC, IR_I32, a);
    if (is_fn(name, "clz") || is_fn(name, "ctlz"))
        return op2(IR_SUB, IR_I32, konst(IR_I32, bits - 1), op1(IR_FLO, IR_I32, a));
    if (is_fn(name, "ctz") || is_fn(name, "cttz"))
        return sel(icmp(IR_EQ, a, konst(IR_I32, 0)), konst(IR_I32, bits),
                   op2(IR_SUB, IR_I32, konst(IR_I32, 31), op1(IR_FLO, IR_I32, op1(IR_BREV, IR_I32, a))));
    if (bits > 32)
        return NULL;
    struct ir_val *c = LLVMGetNumOperands(call) > 3 ? int_extend(get(LLVMGetOperand(call, 2), lane), bits, is_signed)
                                                    : NULL;
    if (is_fn(name, "reverse_bits") || is_fn(name, "bitreverse"))
        return op2(IR_SHR, IR_I32, op1(IR_BREV, IR_I32, a), konst(IR_I32, 32 - bits));
    if (is_fn(name, "bswap")) {
        struct ir_val *r = NULL;
        for (unsigned k = 0; k < bits / 8; k++) {
            struct ir_val *byte = op2(IR_SHR, IR_I32, a, konst(IR_I32, 8 * k));
            byte = op2(IR_SHL, IR_I32, op2(IR_AND, IR_I32, byte, konst(IR_I32, 0xff)), konst(IR_I32, bits - 8 - 8 * k));
            r = r ? op2(IR_OR, IR_I32, r, byte) : byte;
        }
        return r;
    }
    if (is_fn(name, "rotate")) {  // a << s | a >> (bits - s), where s = 0 shifts by 0 both ways
        struct ir_val *s = op2(IR_AND, IR_I32, b, konst(IR_I32, bits - 1));
        struct ir_val *back = op2(IR_AND, IR_I32, op2(IR_SUB, IR_I32, konst(IR_I32, bits), s), konst(IR_I32, bits - 1));
        return op2(IR_OR, IR_I32, op2(IR_SHL, IR_I32, a, s), op2(IR_SHR, IR_I32, a, back));
    }
    if (is_fn(name, "fshl") || is_fn(name, "fshr")) {  // the shift into the other operand split in two: never by 32
        struct ir_val *s = op2(IR_AND, IR_I32, c, konst(IR_I32, bits - 1));
        struct ir_val *rest = op2(IR_SUB, IR_I32, konst(IR_I32, bits - 1), s);
        if (is_fn(name, "fshl"))
            return op2(IR_OR, IR_I32, op2(IR_SHL, IR_I32, a, s),
                       op2(IR_SHR, IR_I32, op2(IR_SHR, IR_I32, b, konst(IR_I32, 1)), rest));
        return op2(IR_OR, IR_I32, op2(IR_SHR, IR_I32, b, s),
                   op2(IR_SHL, IR_I32, op2(IR_SHL, IR_I32, a, konst(IR_I32, 1)), rest));
    }
    if (!strncmp(name, "air.extract_bits.", 17) || !strncmp(name, "air.insert_bits.", 16)) {
        int ins = name[4] == 'i';
        struct ir_val *off = get(LLVMGetOperand(call, 1 + ins), lane), *n = get(LLVMGetOperand(call, 2 + ins), lane);
        struct ir_val *none = icmp(IR_EQ, n, konst(IR_I32, 0));
        if (!ins) {  // the field to the top, then down
            struct ir_val *up = op2(IR_SHL, IR_I32, get(LLVMGetOperand(call, 0), lane),
                                    op2(IR_SUB, IR_I32, op2(IR_SUB, IR_I32, konst(IR_I32, 32), off), n));
            return sel(none, konst(IR_I32, 0), op2(shr, IR_I32, up, op2(IR_SUB, IR_I32, konst(IR_I32, 32), n)));
        }
        struct ir_val *ones = konst(IR_I32, 0xffffffffu);
        struct ir_val *mask = op2(IR_SHR, IR_I32, ones, op2(IR_SUB, IR_I32, konst(IR_I32, 32), n));
        mask = sel(none, konst(IR_I32, 0), op2(IR_SHL, IR_I32, mask, off));
        return op2(IR_OR, IR_I32, op2(IR_AND, IR_I32, a, op2(IR_XOR, IR_I32, mask, ones)),
                   op2(IR_AND, IR_I32, op2(IR_SHL, IR_I32, b, off), mask));
    }
    if (!strncmp(name, "air.abs_diff.", 13))
        return op2(IR_SUB, IR_I32, iminmax(IR_MAX, a, b, is_signed), iminmax(IR_MIN, a, b, is_signed));
    if (!strncmp(name, "air.max3.", 9) || !strncmp(name, "air.min3.", 9)) {
        enum ir_op op = name[5] == 'a' ? IR_MAX : IR_MIN;
        return iminmax(op, iminmax(op, a, b, is_signed), c, is_signed);
    }
    if (!strncmp(name, "air.median3.", 12))
        return iminmax(IR_MAX, iminmax(IR_MIN, a, b, is_signed),
                       iminmax(IR_MIN, iminmax(IR_MAX, a, b, is_signed), c, is_signed), is_signed);
    if (!strncmp(name, "air.hadd.", 9) || !strncmp(name, "air.rhadd.", 10)) {
        // (a + b) >> 1 without the carry out: a & b + (a ^ b) >> 1; rounded up, a | b - (a ^ b) >> 1
        struct ir_val *half = op2(shr, IR_I32, op2(IR_XOR, IR_I32, a, b), konst(IR_I32, 1));
        if (name[4] == 'h')
            return op2(IR_ADD, IR_I32, op2(IR_AND, IR_I32, a, b), half);
        return op2(IR_SUB, IR_I32, op2(IR_OR, IR_I32, a, b), half);
    }
    if (!strncmp(name, "air.mul24.", 10) || !strncmp(name, "air.mad24.", 10)) {
        struct ir_val *x = int_extend(a, 24, is_signed), *y = int_extend(b, 24, is_signed);
        struct ir_val *p = op2(IR_MUL, IR_I32, x, y);
        return name[5] == 'a' ? op2(IR_ADD, IR_I32, p, c) : p;
    }
    if (!strncmp(name, "air.mul_hi.", 11) || !strncmp(name, "air.mad_hi.", 11)) {
        struct ir_val *hi;
        if (bits < 32) {
            hi = op2(shr, IR_I32, op2(IR_MUL, IR_I32, a, b), konst(IR_I32, bits));
        } else {
            hi = op2(IR_MULHI, IR_I32, a, b);
            hi->aux = is_signed;
        }
        return name[5] == 'a' ? op2(IR_ADD, IR_I32, hi, c) : hi;
    }
    int add = !strncmp(name, "air.add_sat.", 12) || is_fn(name, "sadd.sat") || is_fn(name, "uadd.sat");
    if (add || !strncmp(name, "air.sub_sat.", 12) || is_fn(name, "ssub.sat") || is_fn(name, "usub.sat")) {
        struct ir_val *r = op2(add ? IR_ADD : IR_SUB, IR_I32, a, b), *overflow;
        if (is_signed)  // the result's sign wrong: a and b alike (a sum) or unlike (a difference) and r not as a
            overflow = op2(IR_AND, IR_I32, op2(IR_XOR, IR_I32, a, add ? r : b), op2(IR_XOR, IR_I32, add ? b : a, r));
        overflow = is_signed ? icmp(IR_LT | IR_SIGNED, overflow, konst(IR_I32, 0))
                             : icmp(IR_LT, add ? r : a, add ? a : b);
        if (bits < 32 && !is_signed && !add)
            r = iminmax(IR_MAX, r, konst(IR_I32, 0), 1);
        return saturate_int(r, overflow, is_signed ? a : konst(IR_I32, add ? 0xffffffffu : 0), bits, is_signed);
    }
    if (!strncmp(name, "air.mad_sat.", 12)) {
        struct ir_val *lo = op2(IR_MUL, IR_I32, a, b), *r = op2(IR_ADD, IR_I32, lo, c);
        if (bits < 32)
            return saturate_int(r, NULL, NULL, bits, is_signed);
        struct ir_val *hi = op2(IR_MULHI, IR_I32, a, b);
        hi->aux = is_signed;
        struct ir_val *carry = icmp(IR_LT, r, lo);
        if (!is_signed)
            return saturate_int(r, op2(IR_POR, IR_PRED, icmp(IR_NE, hi, konst(IR_I32, 0)), carry),
                                konst(IR_I32, 0xffffffffu), bits, 0);
        // the 64-bit sum's high word, against the sign the low one would extend to
        hi = op2(IR_ADD, IR_I32, op2(IR_ADD, IR_I32, hi, op2(IR_SAR, IR_I32, c, konst(IR_I32, 31))),
                 sel(carry, konst(IR_I32, 1), konst(IR_I32, 0)));
        return saturate_int(r, icmp(IR_NE, hi, op2(IR_SAR, IR_I32, r, konst(IR_I32, 31))), hi, bits, 1);
    }
    return NULL;
}

static struct ir_val *imath(const char *name, LLVMValueRef call, unsigned lane)
{
    int is_signed = strstr(name, ".s.") || !strncmp(name, "llvm.s", 6) || !strncmp(name, "air.abs.", 8) ||
                    is_fn(name, "abs");
    unsigned bits = 8 * scalar_bytes(lane_type(LLVMTypeOf(call), lane));
    struct ir_val *a = int_extend(get(LLVMGetOperand(call, 0), lane), bits, is_signed);
    struct ir_val *b = LLVMGetNumOperands(call) > 2 ? int_extend(get(LLVMGetOperand(call, 1), lane), bits, is_signed)
                                                    : NULL;
    struct ir_val *r;
    if (!strncmp(name, "air.max.", 8) || is_fn(name, "smax") || is_fn(name, "umax"))
        r = op2(IR_MAX, IR_I32, a, b);
    else if (!strncmp(name, "air.min.", 8) || is_fn(name, "smin") || is_fn(name, "umin"))
        r = op2(IR_MIN, IR_I32, a, b);
    else if (!strncmp(name, "air.abs.", 8) || is_fn(name, "abs"))  // max(x, -x)
        r = op2(IR_MAX, IR_I32, a, op2(IR_SUB, IR_I32, konst(IR_I32, 0), a));
    else if (!strncmp(name, "air.clamp.", 10)) {  // min(max(x, lo), hi)
        struct ir_val *hi = int_extend(get(LLVMGetOperand(call, 2), lane), bits, is_signed);
        struct ir_val *lo = op2(IR_MAX, IR_I32, a, b);
        lo->aux = is_signed;
        r = op2(IR_MIN, IR_I32, lo, hi);
    } else
        return ibits(name, call, lane, a, b, bits, is_signed);
    r->aux = is_signed;
    return r;
}

struct pair64 {
    struct ir_val *lo, *hi;
};

static struct ir_val *word(uint32_t c)
{
    return konst(IR_I32, c);
}

static struct ir_val *flag(struct ir_val *c)
{
    return sel(c, word(1), word(0));
}

static struct pair64 pair_zero(void)
{
    return (struct pair64){ word(0), word(0) };
}

// lane `lane` of v as a pair; 0 where its high word isn't known
static int pair_at(LLVMValueRef v, unsigned lane, struct pair64 *p)
{
    p->hi = high_word(v, lane);
    p->lo = get(v, lane);
    return p->hi != NULL;
}

static void set_pair(LLVMValueRef v, unsigned lane, struct pair64 p)
{
    set(v, lane, p.lo);
    set_high(v, lane, p.hi);
}

static struct ir_val *less64(struct pair64 x, struct pair64 y, int is_signed)
{
    struct ir_val *hi = icmp(IR_LT, x.hi, y.hi);
    if (is_signed)
        hi->aux |= IR_SIGNED;
    return op2(IR_POR, IR_PRED, hi, op2(IR_PAND, IR_PRED, icmp(IR_EQ, x.hi, y.hi), icmp(IR_LT, x.lo, y.lo)));
}

static struct ir_val *is_zero64(struct pair64 x)
{
    return op2(IR_PAND, IR_PRED, icmp(IR_EQ, x.hi, word(0)), icmp(IR_EQ, x.lo, word(0)));
}

static struct pair64 pick64(struct ir_val *c, struct pair64 a, struct pair64 b)
{
    return (struct pair64){ sel(c, a.lo, b.lo), sel(c, a.hi, b.hi) };
}

static struct pair64 add64(struct pair64 x, struct pair64 y)
{
    struct ir_val *lo = op2(IR_ADD, IR_I32, x.lo, y.lo);
    struct ir_val *hi = op2(IR_ADD, IR_I32, x.hi, y.hi);
    return (struct pair64){ lo, op2(IR_ADD, IR_I32, hi, flag(icmp(IR_LT, lo, x.lo))) };
}

static struct pair64 sub64(struct pair64 x, struct pair64 y)
{
    struct ir_val *hi = op2(IR_SUB, IR_I32, x.hi, y.hi);
    return (struct pair64){ op2(IR_SUB, IR_I32, x.lo, y.lo),
                            op2(IR_SUB, IR_I32, hi, flag(icmp(IR_LT, x.lo, y.lo))) };
}

static struct pair64 neg64(struct pair64 x)
{
    return sub64(pair_zero(), x);
}

static struct ir_val *mulhi_u32(struct ir_val *a, struct ir_val *b)
{
    struct ir_val *t = op2(IR_MULHI, IR_I32, a, b);
    t->aux = 0;
    return t;
}

static struct pair64 mul64(struct pair64 x, struct pair64 y)
{
    struct ir_val *top = op2(IR_ADD, IR_I32, mulhi_u32(x.lo, y.lo), op2(IR_MUL, IR_I32, x.lo, y.hi));
    return (struct pair64){ op2(IR_MUL, IR_I32, x.lo, y.lo),
                            op2(IR_ADD, IR_I32, top, op2(IR_MUL, IR_I32, x.hi, y.lo)) };
}

static struct ir_val *add_carry(struct ir_val *a, struct ir_val *b, struct ir_val **carry)
{
    struct ir_val *s = op2(IR_ADD, IR_I32, a, b), *c = flag(icmp(IR_LT, s, a));
    *carry = *carry ? op2(IR_ADD, IR_I32, *carry, c) : c;
    return s;
}

// the 128-bit product of two unsigned 64-bit integers: four words, lowest first
static void mul128(struct pair64 x, struct pair64 y, struct ir_val *w[4])
{
    struct ir_val *c1 = NULL, *c2 = NULL;
    struct ir_val *p01l = op2(IR_MUL, IR_I32, x.lo, y.hi), *p01h = mulhi_u32(x.lo, y.hi);
    struct ir_val *p10l = op2(IR_MUL, IR_I32, x.hi, y.lo), *p10h = mulhi_u32(x.hi, y.lo);
    struct ir_val *t = add_carry(mulhi_u32(x.lo, y.lo), p01l, &c1);
    w[0] = op2(IR_MUL, IR_I32, x.lo, y.lo);
    w[1] = add_carry(t, p10l, &c1);
    t = add_carry(c1, p01h, &c2);
    t = add_carry(t, p10h, &c2);
    w[2] = add_carry(t, op2(IR_MUL, IR_I32, x.hi, y.hi), &c2);
    w[3] = op2(IR_ADD, IR_I32, mulhi_u32(x.hi, y.hi), c2);
}

// the high 64 bits of the 128-bit product: a signed one is the unsigned product less each operand that was
// negative, shifted up 64 bits
static struct pair64 mulhi64(struct pair64 x, struct pair64 y, int is_signed)
{
    struct ir_val *w[4];
    mul128(x, y, w);
    struct pair64 hi = { w[2], w[3] };
    if (is_signed) {
        struct ir_val *xn = icmp(IR_LT, x.hi, word(0)), *yn = icmp(IR_LT, y.hi, word(0));
        xn->aux |= IR_SIGNED;
        yn->aux |= IR_SIGNED;
        hi = sub64(hi, pick64(xn, y, pair_zero()));
        hi = sub64(hi, pick64(yn, x, pair_zero()));
    }
    return hi;
}

// x << n, x >> n or its arithmetic shift, by the low 6 bits of n: 32 and over move a word, the rest split
static struct pair64 shift64(LLVMOpcode op, struct pair64 x, struct ir_val *amount)
{
    struct ir_val *n = op2(IR_AND, IR_I32, amount, word(63));
    struct ir_val *big = icmp(IR_GE, n, word(32)), *k = op2(IR_AND, IR_I32, n, word(31));
    struct ir_val *rest = op2(IR_SUB, IR_I32, word(31), k);  // 31 - k, so a split never shifts by 32
    if (op == LLVMShl) {
        struct ir_val *carry = op2(IR_SHR, IR_I32, op2(IR_SHR, IR_I32, x.lo, word(1)), rest);
        return (struct pair64){ sel(big, word(0), op2(IR_SHL, IR_I32, x.lo, k)),
                                sel(big, op2(IR_SHL, IR_I32, x.lo, k),
                                    op2(IR_OR, IR_I32, op2(IR_SHL, IR_I32, x.hi, k), carry)) };
    }
    enum ir_op sh = op == LLVMLShr ? IR_SHR : IR_SAR;
    struct ir_val *sign = op == LLVMLShr ? word(0) : op2(IR_SAR, IR_I32, x.hi, word(31));
    struct ir_val *carry = op2(IR_SHL, IR_I32, op2(IR_SHL, IR_I32, x.hi, word(1)), rest);
    return (struct pair64){ sel(big, op2(sh, IR_I32, x.hi, k), op2(IR_OR, IR_I32, op2(IR_SHR, IR_I32, x.lo, k), carry)),
                            sel(big, sign, op2(sh, IR_I32, x.hi, k)) };
}

static struct pair64 shift64_const(LLVMOpcode op, struct pair64 x, unsigned c)
{
    c &= 63;
    if (!c)
        return x;
    if (op == LLVMShl) {
        if (c >= 32)
            return (struct pair64){ word(0), op2(IR_SHL, IR_I32, x.lo, word(c - 32)) };
        return (struct pair64){ op2(IR_SHL, IR_I32, x.lo, word(c)),
                                op2(IR_OR, IR_I32, op2(IR_SHL, IR_I32, x.hi, word(c)),
                                    op2(IR_SHR, IR_I32, x.lo, word(32 - c))) };
    }
    enum ir_op sh = op == LLVMLShr ? IR_SHR : IR_SAR;
    struct ir_val *sign = op == LLVMLShr ? word(0) : op2(IR_SAR, IR_I32, x.hi, word(31));
    if (c >= 32)
        return (struct pair64){ op2(sh, IR_I32, x.hi, word(c - 32)), sign };
    return (struct pair64){ op2(IR_OR, IR_I32, op2(IR_SHR, IR_I32, x.lo, word(c)),
                                op2(IR_SHL, IR_I32, x.hi, word(32 - c))),
                            op2(sh, IR_I32, x.hi, word(c)) };
}

// n / d and n % d, unsigned, by shift and subtract in a loop of 64 steps: the remainder takes the dividend's bits
// from the top, and the quotient a bit for each time the divisor goes into it. Apple's GPUs give 0 for a division
// by 0 and the dividend for its remainder.
static void udivmod64(struct pair64 n, struct pair64 d, struct pair64 *q, struct pair64 *r)
{
    struct ir_block *pre = g_cur, *head = ir_block(g_f), *done = ir_block(g_f);
    ir_add(pre, IR_BR, IR_VOID, 0);
    ir_edge(pre, head);
    g_cur = head;
    struct ir_val *zero = word(0), *one = word(1);
    // the loop's carried values: the bit being taken, the remainder and the quotient so far
    struct ir_val *i = ir_add(head, IR_PHI, IR_I32, 2);
    struct ir_val *rl = ir_add(head, IR_PHI, IR_I32, 2), *rh = ir_add(head, IR_PHI, IR_I32, 2);
    struct ir_val *ql = ir_add(head, IR_PHI, IR_I32, 2), *qh = ir_add(head, IR_PHI, IR_I32, 2);
    struct pair64 rem = { rl, rh }, quo = { ql, qh };
    struct ir_val *high = icmp(IR_GE, i, word(32));
    struct ir_val *bit = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, sel(high, n.hi, n.lo), op2(IR_AND, IR_I32, i, word(31))), one);
    struct ir_val *out = icmp(IR_NE, op2(IR_SHR, IR_I32, rem.hi, word(31)), zero);  // a bit past 64 bits
    rem = (struct pair64){ op2(IR_OR, IR_I32, op2(IR_SHL, IR_I32, rem.lo, one), bit),
                           op2(IR_OR, IR_I32, op2(IR_SHL, IR_I32, rem.hi, one), op2(IR_SHR, IR_I32, rem.lo, word(31))) };
    struct ir_val *fits = icmp(IR_GT, rem.hi, d.hi);
    fits = op2(IR_POR, IR_PRED, fits, op2(IR_PAND, IR_PRED, icmp(IR_EQ, rem.hi, d.hi), icmp(IR_GE, rem.lo, d.lo)));
    fits = op2(IR_POR, IR_PRED, out, fits);
    rem = pick64(fits, sub64(rem, d), rem);
    quo = (struct pair64){ op2(IR_OR, IR_I32, op2(IR_SHL, IR_I32, quo.lo, one), flag(fits)),
                           op2(IR_OR, IR_I32, op2(IR_SHL, IR_I32, quo.hi, one), op2(IR_SHR, IR_I32, quo.lo, word(31))) };
    struct ir_val *next = op2(IR_SUB, IR_I32, i, one);
    struct ir_val *more = icmp(IR_NE, i, zero);
    struct ir_val *loop = ir_add(head, IR_CBR, IR_VOID, 1);
    loop->src[0] = more;
    ir_edge(head, head);
    ir_edge(head, done);
    struct ir_val *start[5] = { word(63), zero, zero, zero, zero };
    struct ir_val *carried[5] = { i, rl, rh, ql, qh }, *updated[5] = { next, rem.lo, rem.hi, quo.lo, quo.hi };
    for (int k = 0; k < 5; k++) {
        carried[k]->src[0] = start[k];
        carried[k]->src[1] = updated[k];
    }
    g_cur = done;
    *q = pick64(is_zero64(d), pair_zero(), quo);
    *r = rem;
}

// LLVM's udiv, sdiv, urem and srem of 64-bit lanes: signed ones divide the magnitudes
static struct pair64 idiv64(struct pair64 a, struct pair64 b, int is_signed, int rem)
{
    struct ir_val *an = NULL, *bn = NULL;
    if (is_signed) {
        an = icmp(IR_LT, a.hi, word(0));
        bn = icmp(IR_LT, b.hi, word(0));
        an->aux |= IR_SIGNED;
        bn->aux |= IR_SIGNED;
        a = pick64(an, neg64(a), a);
        b = pick64(bn, neg64(b), b);
    }
    struct pair64 q, r;
    udivmod64(a, b, &q, &r);
    if (!is_signed)
        return rem ? r : q;
    if (rem)
        return pick64(an, neg64(r), r);
    return pick64(icmp(IR_NE, flag(an), flag(bn)), neg64(q), q);
}

// the pair a saturated signed result takes: the limit on the side of word's sign
static struct pair64 signed_limit(struct ir_val *sign_word)
{
    struct ir_val *s = op2(IR_SAR, IR_I32, sign_word, word(31));
    return (struct pair64){ op2(IR_XOR, IR_I32, s, word(0xffffffffu)), op2(IR_XOR, IR_I32, s, word(0x7fffffffu)) };
}

// 64-bit min, max, clamp, abs, bit counts and the saturating, high-product, averaging and bit-field functions of
// lane `lane`, where the operands' high words are known; 0 if not
static int imath64(const char *name, LLVMValueRef call, unsigned lane)
{
    int is_signed = strstr(name, ".s.") || !strncmp(name, "air.abs.", 8) || is_fn(name, "abs");
    struct pair64 x, y = { 0 }, z = { 0 }, r;
    unsigned nargs = LLVMGetNumOperands(call) - 1;
    if (!pair_at(LLVMGetOperand(call, 0), lane, &x))
        return 0;
    // the 64-bit operands after the first: min3 and the like take pairs, bit fields a pair and then offsets
    int bits_fn = !strncmp(name, "air.extract_bits.", 17) || !strncmp(name, "air.insert_bits.", 16) ||
                  is_fn(name, "rotate");
    if (!bits_fn && nargs > 1 && !pair_at(LLVMGetOperand(call, 1), lane, &y))
        return 0;
    if (!bits_fn && nargs > 2 && !pair_at(LLVMGetOperand(call, 2), lane, &z))
        return 0;
    struct ir_val *zero = word(0), *ones = word(0xffffffffu);
    if (is_fn(name, "popcount") || is_fn(name, "ctpop")) {
        r = (struct pair64){ op2(IR_ADD, IR_I32, op1(IR_POPC, IR_I32, x.lo), op1(IR_POPC, IR_I32, x.hi)), zero };
    } else if (is_fn(name, "clz") || is_fn(name, "ctlz")) {  // 32 + the low word's, where the high one is 0
        struct ir_val *hi = op2(IR_SUB, IR_I32, word(31), op1(IR_FLO, IR_I32, x.hi));
        struct ir_val *lo = op2(IR_SUB, IR_I32, word(63), op1(IR_FLO, IR_I32, x.lo));
        r = (struct pair64){ sel(icmp(IR_EQ, x.hi, zero), lo, hi), zero };
    } else if (is_fn(name, "ctz") || is_fn(name, "cttz")) {
        struct ir_val *lo = sel(icmp(IR_EQ, x.lo, zero), word(32),
                                op2(IR_SUB, IR_I32, word(31), op1(IR_FLO, IR_I32, op1(IR_BREV, IR_I32, x.lo))));
        struct ir_val *hi = sel(icmp(IR_EQ, x.hi, zero), word(64),
                                op2(IR_SUB, IR_I32, word(63), op1(IR_FLO, IR_I32, op1(IR_BREV, IR_I32, x.hi))));
        r = (struct pair64){ sel(icmp(IR_EQ, x.lo, zero), hi, lo), zero };
    } else if (!strncmp(name, "air.abs.", 8) || is_fn(name, "abs")) {
        struct ir_val *below = icmp(IR_LT, x.hi, zero);
        below->aux |= IR_SIGNED;
        r = is_signed ? pick64(below, neg64(x), x) : x;
    } else if (!strncmp(name, "air.min.", 8) || !strncmp(name, "air.max.", 8) || is_fn(name, "smin") ||
               is_fn(name, "umin") || is_fn(name, "smax") || is_fn(name, "umax")) {
        int is_min = strstr(name, "min") != NULL;
        struct ir_val *lt = less64(x, y, is_signed);
        r = is_min ? pick64(lt, x, y) : pick64(lt, y, x);
    } else if (!strncmp(name, "air.clamp.", 10)) {
        r = pick64(less64(x, y, is_signed), y, x);  // max(x, lo)
        r = pick64(less64(r, z, is_signed), r, z);  // min(that, hi)
    } else if (!strncmp(name, "air.min3.", 9) || !strncmp(name, "air.max3.", 9)) {
        int is_min = name[5] == 'i';
        struct pair64 m = is_min ? pick64(less64(x, y, is_signed), x, y) : pick64(less64(x, y, is_signed), y, x);
        r = is_min ? pick64(less64(m, z, is_signed), m, z) : pick64(less64(m, z, is_signed), z, m);
    } else if (!strncmp(name, "air.median3.", 12)) {
        struct ir_val *xy = less64(x, y, is_signed);
        struct pair64 lo = pick64(xy, x, y), hi = pick64(xy, y, x);  // min and max of the first two
        struct pair64 top = pick64(less64(hi, z, is_signed), hi, z);  // min(hi, z)
        r = pick64(less64(lo, top, is_signed), top, lo);              // max(lo, that)
    } else if (!strncmp(name, "air.add_sat.", 12) || !strncmp(name, "air.sub_sat.", 12)) {
        int add = name[4] == 'a';
        r = add ? add64(x, y) : sub64(x, y);
        if (!is_signed) {
            struct ir_val *over = add ? less64(r, x, 0) : less64(x, y, 0);
            r = pick64(over, add ? (struct pair64){ ones, ones } : pair_zero(), r);
        } else {
            // the sign of the result against the operand's, where the operands could overflow it
            struct ir_val *same = op2(IR_XOR, IR_I32, x.hi, y.hi);
            if (add)
                same = op2(IR_XOR, IR_I32, same, ones);
            struct ir_val *over = icmp(IR_LT, op2(IR_AND, IR_I32, same, op2(IR_XOR, IR_I32, x.hi, r.hi)), zero);
            over->aux |= IR_SIGNED;
            r = pick64(over, signed_limit(x.hi), r);
        }
    } else if (!strncmp(name, "air.mul_hi.", 11)) {
        r = mulhi64(x, y, is_signed);
    } else if (!strncmp(name, "air.mad_hi.", 11)) {
        r = add64(mulhi64(x, y, is_signed), z);
    } else if (!strncmp(name, "air.mad_sat.", 12)) {
        struct ir_val *w[4];
        mul128(x, y, w);
        struct pair64 low = { w[0], w[1] };
        if (!is_signed) {
            struct pair64 sum = add64(low, z);
            struct ir_val *over = op2(IR_POR, IR_PRED, icmp(IR_NE, op2(IR_OR, IR_I32, w[2], w[3]), zero),
                                      less64(sum, low, 0));
            r = pick64(over, (struct pair64){ ones, ones }, sum);
        } else {
            struct pair64 hi = mulhi64(x, y, 1), sum = add64(low, z);
            // the 128-bit sum: z extended with its sign into the high 64 bits, and the carry out of the low ones
            struct ir_val *ext = op2(IR_SAR, IR_I32, z.hi, word(31));
            hi = add64(add64(hi, (struct pair64){ ext, ext }), (struct pair64){ flag(less64(sum, low, 0)), zero });
            struct ir_val *s = op2(IR_SAR, IR_I32, sum.hi, word(31));  // what the high 64 bits are when it fits
            struct ir_val *over = op2(IR_POR, IR_PRED, icmp(IR_NE, hi.lo, s), icmp(IR_NE, hi.hi, s));
            r = pick64(over, signed_limit(hi.hi), sum);
        }
    } else if (!strncmp(name, "air.abs_diff.", 13)) {
        struct ir_val *lt = less64(x, y, is_signed);
        r = sub64(pick64(lt, y, x), pick64(lt, x, y));
    } else if (!strncmp(name, "air.hadd.", 9) || !strncmp(name, "air.rhadd.", 10)) {
        // (a + b) >> 1 without the carry out: a & b + (a ^ b) >> 1; rounded up, a | b - (a ^ b) >> 1
        struct pair64 diff = { op2(IR_XOR, IR_I32, x.lo, y.lo), op2(IR_XOR, IR_I32, x.hi, y.hi) };
        struct pair64 half = shift64_const(is_signed ? LLVMAShr : LLVMLShr, diff, 1);
        if (name[4] == 'h')
            r = add64((struct pair64){ op2(IR_AND, IR_I32, x.lo, y.lo), op2(IR_AND, IR_I32, x.hi, y.hi) }, half);
        else
            r = sub64((struct pair64){ op2(IR_OR, IR_I32, x.lo, y.lo), op2(IR_OR, IR_I32, x.hi, y.hi) }, half);
    } else if (is_fn(name, "reverse_bits") || is_fn(name, "bitreverse")) {
        r = (struct pair64){ op1(IR_BREV, IR_I32, x.hi), op1(IR_BREV, IR_I32, x.lo) };
    } else if (is_fn(name, "rotate")) {  // a << s | a >> (64 - s), where s = 0 shifts by 0 both ways
        struct ir_val *s = op2(IR_AND, IR_I32, get(LLVMGetOperand(call, 1), lane), word(63));
        struct pair64 up = shift64(LLVMShl, x, s);
        struct pair64 down = shift64(LLVMLShr, x, op2(IR_SUB, IR_I32, word(64), s));
        r = (struct pair64){ op2(IR_OR, IR_I32, up.lo, down.lo), op2(IR_OR, IR_I32, up.hi, down.hi) };
    } else if (!strncmp(name, "air.extract_bits.", 17)) {
        struct ir_val *off = get(LLVMGetOperand(call, 1), lane), *n = get(LLVMGetOperand(call, 2), lane);
        // the field to the top, then down
        struct pair64 up = shift64(LLVMShl, x, op2(IR_SUB, IR_I32, op2(IR_SUB, IR_I32, word(64), off), n));
        struct pair64 field = shift64(is_signed ? LLVMAShr : LLVMLShr, up, op2(IR_SUB, IR_I32, word(64), n));
        r = pick64(icmp(IR_EQ, n, zero), pair_zero(), field);
    } else if (!strncmp(name, "air.insert_bits.", 16)) {
        struct pair64 b;
        if (!pair_at(LLVMGetOperand(call, 1), lane, &b))
            return 0;
        struct ir_val *off = get(LLVMGetOperand(call, 2), lane), *n = get(LLVMGetOperand(call, 3), lane);
        struct pair64 all = { ones, ones };
        struct pair64 mask = shift64(LLVMShl, shift64(LLVMLShr, all, op2(IR_SUB, IR_I32, word(64), n)), off);
        mask = pick64(icmp(IR_EQ, n, zero), pair_zero(), mask);
        struct pair64 moved = shift64(LLVMShl, b, off);
        r = (struct pair64){ op2(IR_OR, IR_I32, op2(IR_AND, IR_I32, x.lo, op2(IR_XOR, IR_I32, mask.lo, ones)),
                                 op2(IR_AND, IR_I32, moved.lo, mask.lo)),
                             op2(IR_OR, IR_I32, op2(IR_AND, IR_I32, x.hi, op2(IR_XOR, IR_I32, mask.hi, ones)),
                                 op2(IR_AND, IR_I32, moved.hi, mask.hi)) };
    } else {
        return 0;
    }
    set_pair(call, lane, r);
    return 1;
}

// A float as the 64-bit integer Apple's GPUs make of it: the mantissa shifted by the exponent, whatever the shift
// carries out of the top lost (a signed result wraps, and is 0 from 2^87 on, a NaN and infinity with it), then
// negated if the float was. An unsigned result is 0 for a negative float or a NaN and saturates above 2^64.
static struct pair64 float_to_pair(struct ir_val *f, int is_signed)
{
    struct ir_val *zero = word(0);
    struct ir_val *abs = op2(IR_AND, IR_I32, f, word(0x7fffffff));
    struct ir_val *neg = icmp(IR_NE, op2(IR_AND, IR_I32, f, word(0x80000000u)), zero);
    struct ir_val *m = op2(IR_OR, IR_I32, op2(IR_AND, IR_I32, abs, word(0x7fffff)), word(0x800000));
    struct ir_val *e = op2(IR_SUB, IR_I32, op2(IR_SHR, IR_I32, abs, word(23)), word(150));  // the lowest bit's
    struct ir_val *left = icmp(IR_GE, e, zero), *wide = icmp(IR_GE, e, word(64));
    left->aux |= IR_SIGNED;
    wide->aux |= IR_SIGNED;
    struct pair64 up = shift64(LLVMShl, (struct pair64){ m, zero }, e);
    struct ir_val *back = iminmax(IR_MIN, op2(IR_SUB, IR_I32, zero, e), word(31), 0);
    struct pair64 mag = pick64(left, pick64(wide, pair_zero(), up),
                               (struct pair64){ op2(IR_SHR, IR_I32, m, back), zero });
    if (is_signed)
        return pick64(neg, neg64(mag), mag);
    struct ir_val *nan = icmp(IR_GT, abs, word(0x7f800000));
    struct ir_val *over = icmp(IR_GE, e, word(41));  // 2^64 and up
    over->aux |= IR_SIGNED;
    mag = pick64(over, (struct pair64){ word(0xffffffffu), word(0xffffffffu) }, mag);
    return pick64(op2(IR_POR, IR_PRED, neg, nan), pair_zero(), mag);
}

// A 64-bit integer as a float, rounded once: the value normalised to its top 32 bits and a sticky bit for the
// rest, which a 32-bit conversion then rounds to nearest even. A half is made from the float rounded to odd
// instead, so that its own rounding after it is the only one.
static struct ir_val *pair_to_float(struct pair64 x, int is_signed, int to_half)
{
    struct ir_val *zero = word(0);
    struct ir_val *neg = NULL;
    if (is_signed) {
        neg = icmp(IR_LT, x.hi, zero);
        neg->aux |= IR_SIGNED;
        x = pick64(neg, neg64(x), x);
    }
    struct ir_val *lz = op2(IR_SUB, IR_I32,
                            sel(icmp(IR_EQ, x.hi, zero), op2(IR_SUB, IR_I32, word(63), op1(IR_FLO, IR_I32, x.lo)),
                                op2(IR_SUB, IR_I32, word(31), op1(IR_FLO, IR_I32, x.hi))),
                            zero);
    struct pair64 y = shift64(LLVMShl, x, lz);  // the top bit set, unless x is 0
    struct ir_val *sticky = flag(icmp(IR_NE, y.lo, zero));
    struct ir_val *w = y.hi, *scale;
    if (to_half) {
        struct ir_val *m24 = op2(IR_SHR, IR_I32, w, word(8));
        struct ir_val *inexact = icmp(IR_NE, op2(IR_OR, IR_I32, op2(IR_AND, IR_I32, w, word(0xff)), sticky), zero);
        w = op2(IR_OR, IR_I32, m24, flag(inexact));
        scale = op2(IR_SHL, IR_I32, op2(IR_SUB, IR_I32, word(127 + 40), lz), word(23));
    } else {
        w = op2(IR_OR, IR_I32, w, sticky);
        scale = op2(IR_SHL, IR_I32, op2(IR_SUB, IR_I32, word(127 + 32), lz), word(23));
    }
    struct ir_val *f = op1(IR_I2F, IR_F32, w);
    f->aux = 0;
    f = op2(IR_FMUL, IR_F32, f, op2(IR_OR, IR_F32, scale, konst(IR_F32, 0)));
    f = sel(is_zero64(x), fconst(0.0f), f);
    return neg ? sel(neg, op1(IR_FNEG, IR_F32, f), f) : f;
}

// The handle a texture or sampler operand names: a bound slot's descriptor heap index, from the const bank's table,
// or the resource ID an argument buffer holds (a texture's header index, a sampler's index << 20).
static struct ir_val *handle_of_ptr(struct ptr p, enum space space)
{
    if (p.param) {
        if (p.space != space || p.off)
            airc_fail("a subroutine's texture or sampler used as another kind or at an offset");
        return p.base;
    }
    if (p.space == space) {  // read where it's used: kept in a register from the prologue on, it costs one throughout
        // (an array's element picked at run time p.base slots further on)
        struct ir_val *at = p.base ? op2(IR_SHL, IR_I32, p.base, konst(IR_I32, 2)) : NULL;
        struct ir_val *h = ir_add(g_cur, IR_CBUF, IR_I32, at ? 1 : 0);
        h->imm = space == SP_TEXTURE ? NVK_TEXTURE_HANDLE(p.slot) : NVK_SAMPLER_HANDLE(p.slot);
        if (at)
            h->src[0] = at;
        return h;
    }
    if (p.space == SP_GLOBAL && !p.off)
        return op2(IR_OR, IR_I32, p.base, konst(IR_I32, 0));  // the ID's low word
    airc_fail("texture/sampler operand is not a texture/sampler");
}

static struct ir_val *handle_of(LLVMValueRef v, enum space space)
{
    return handle_of_ptr(ptr_of(v), space);
}

enum { USE_PLAIN, USE_COMPARE, USE_FACES, USE_LEVELS, USE_NEAREST };

// A slot's use the plugin sets the sampler or texture up for: a subroutine's parameter keeps it for the slots its
// calls pass.
static void note_use(struct ptr p, int use);

// A bound sampler's handle, noting whether the slot is sampled with a depth compare or without: a compare wants
// DEPTH_COMPARE in the sampler even for Metal's default compare function, never, and a plain sample wants it clear,
// so the plugin sets or clears it per slot.
static struct ir_val *sampler_handle(LLVMValueRef v, int compare)
{
    struct ptr p = ptr_of(v);
    note_use(p, compare ? USE_COMPARE : USE_PLAIN);
    return handle_of_ptr(p, SP_SAMPLER);
}

// A cube's handle for a texel read or write, which address its faces as layers: the plugin describes a texture
// at such a slot as the 2D array of its faces.
static struct ir_val *faces_handle(LLVMValueRef v)
{
    struct ptr p = ptr_of(v);
    note_use(p, USE_FACES);
    return handle_of_ptr(p, SP_TEXTURE);
}

static int is_konst(const struct ir_val *v, int64_t imm)
{
    return v->op == IR_CONST && v->imm == imm;
}

// A texel write's texture handle at a level past the base: SUST addresses the level its header describes, so the
// plugin gives a texture written at levels a run of headers, one a level, and the handle is the run's start plus
// the level. Only a bound texture has the run; an argument buffer's is written at its base.
static struct ir_val *level_handle(struct ir_val *handle, LLVMValueRef texture, struct ir_val *level)
{
    if (is_konst(level, 0))
        return handle;
    struct ptr p = ptr_of(texture);
    if (!p.param && p.space != SP_TEXTURE)
        return handle;
    note_use(p, USE_LEVELS);
    return op2(IR_ADD, IR_I32, handle, level);
}

// A 2D fetch or sample. 1D textures are 2D ones of height 1, sampled down the middle of their row; a 1D array
// is a 2D texture with one row per layer. Metal samples arrays only with normalized coordinates, so a layer's
// row centre is (layer + 0.5) / layers.
// a texture op's offset, n components of `bits` bits each in one register (AOFFI), as NAK packs it; NULL for none
static struct ir_val *packed_offset(LLVMValueRef flag, LLVMValueRef off, unsigned n, unsigned bits)
{
    struct ir_val *f = get(flag, 0);
    if (is_konst(f, 0))
        return NULL;
    if (f->op != IR_CONST)
        airc_fail("a texture offset that may or may not be there");
    int64_t k = 0;
    int all_konst = 1, all_zero = 1;
    for (unsigned c = 0; c < n; c++) {
        struct ir_val *v = get(off, c);
        all_konst &= v->op == IR_CONST;
        all_zero &= is_konst(v, 0);
        if (v->op == IR_CONST) {
            int32_t o = (int32_t)v->imm, lim = 1 << (bits - 1);
            o = o < -lim ? -lim : o > lim - 1 ? lim - 1 : o;
            k |= (int64_t)((uint32_t)o & ((1u << bits) - 1)) << (c * bits);
        }
    }
    if (all_zero)
        return NULL;
    if (all_konst)
        return konst(IR_I32, k);
    struct ir_val *r = NULL;
    for (unsigned c = 0; c < n; c++) {
        struct ir_val *v = op2(IR_AND, IR_I32, get(off, c), konst(IR_I32, (1 << bits) - 1));
        if (c)
            v = op2(IR_SHL, IR_I32, v, konst(IR_I32, c * bits));
        r = r ? op2(IR_OR, IR_I32, r, v) : v;
    }
    return r;
}

static struct ir_val *shfl(struct ir_val *x, int mode, struct ir_val *lane)
{
    struct ir_val *v = op2(IR_SHFL, x->type, x, lane);
    v->aux = mode;
    return v;
}

static struct ir_val *load_lane(struct ptr p, unsigned off, LLVMTypeRef st);
static void store_value(struct ptr p, LLVMTypeRef t, LLVMValueRef val, unsigned align);
static unsigned private_var(struct ptr p, unsigned off);
static void mem_copy(LLVMValueRef in);
static void mem_set(LLVMValueRef in);
static void private_write(unsigned var, struct ir_val *v);

// A coherent load or store, volatile or not, is seen by the whole device (and, as no other processor shares this
// memory, by the system): an atomic add of 0 returns what's there, an exchange writes without caching. The word's
// bits, so a float's sign of zero stays.
static void coherent_access(LLVMValueRef call, int load)
{
    struct ptr p = fit_offset(ptr_of(LLVMGetOperand(call, load ? 0 : 1)));
    if (p.space != SP_GLOBAL)
        airc_fail("a coherent access outside device memory");
    LLVMTypeRef vt = load ? LLVMTypeOf(call) : LLVMTypeOf(LLVMGetOperand(call, 0));
    if (scalar_bytes(vt) != 4 || nlanes(vt) != 1)
        airc_fail("a coherent access of other than one 32-bit word");
    struct ir_val *data = load ? konst(IR_I32, 0) : get(LLVMGetOperand(call, 0), 0);
    struct ir_val *a = ir_add(g_cur, IR_ATOM, ir_type_of(vt), 2);
    a->src[0] = p.base;
    a->src[1] = data;
    a->imm = p.off;
    a->aux = IR_ATOMA(load ? SASS_ATOM_ADD : SASS_ATOM_EXCH, SASS_ATOM_U32, 0);
    if (load)
        set(call, 0, a);
}

// AIR's atomics, air.atomic.{global,local}.<op>[.weak][.s|.u].<type>: one ATOM each, on the 32-bit word the pointer
// names. A load is an add of 0 and a store an exchange; a compare-exchange takes the expected value from the
// pointer it's given and leaves the old value there, returning it.
static void atomic_op(LLVMValueRef call, const char *name)
{
    static const struct {
        const char *name;
        int op;
    } ops[] = {
        { "add", SASS_ATOM_ADD },   { "sub", SASS_ATOM_ADD },  { "min", SASS_ATOM_MIN },   { "max", SASS_ATOM_MAX },
        { "and", SASS_ATOM_AND },   { "or", SASS_ATOM_OR },    { "xor", SASS_ATOM_XOR },   { "xchg", SASS_ATOM_EXCH },
        { "store", SASS_ATOM_EXCH }, { "load", SASS_ATOM_ADD }, { "cmpxchg", SASS_ATOM_CAS },
    };
    int shared = !strncmp(name, "air.atomic.local.", 17);
    const char *what = name + (shared ? 17 : 18);
    int op = -1;
    for (unsigned i = 0; i < sizeof ops / sizeof ops[0]; i++) {
        size_t n = strlen(ops[i].name);
        if (!strncmp(what, ops[i].name, n) && what[n] == '.')
            op = (int)i;
    }
    struct ptr p = ptr_of(LLVMGetOperand(call, 0));
    if (op < 0 || strstr(name, "64") || p.space != (shared ? SP_SHARED : SP_GLOBAL))
        airc_fail("unsupported atomic (%s)", name);
    // an aligned word of threadgroup memory is read and written atomically by LDS and STS; as ATOMS (an add of 0, an
    // exchange) the load can return a word that atomics of other warps, past the barrier before it, still change
    if (shared && !strcmp(ops[op].name, "load")) {
        set(call, 0, load_lane(fit_offset(p), 0, LLVMTypeOf(call)));
        return;
    }
    if (shared && !strcmp(ops[op].name, "store")) {
        store_value(p, LLVMTypeOf(LLVMGetOperand(call, 1)), LLVMGetOperand(call, 1), 4);
        return;
    }
    int f = strstr(name, ".f32") != NULL;
    int type = f ? SASS_ATOM_F32 : strstr(name, ".s.") ? SASS_ATOM_I32 : SASS_ATOM_U32;
    if (f && ops[op].op != SASS_ATOM_ADD && ops[op].op != SASS_ATOM_EXCH && ops[op].op != SASS_ATOM_CAS)
        airc_fail("unsupported float atomic (%s)", name);
    enum ir_type t = f ? IR_F32 : IR_I32;
    struct ir_val *data, *swap = NULL;
    LLVMValueRef expected = NULL;
    if (!strcmp(ops[op].name, "load")) {
        data = konst(t, 0);
    } else if (ops[op].op == SASS_ATOM_CAS) {
        expected = LLVMGetOperand(call, 1);
        data = load_lane(ptr_of(expected), 0, LLVMTypeOf(LLVMGetOperand(call, 2)));
        swap = get(LLVMGetOperand(call, 2), 0);
    } else {
        data = get(LLVMGetOperand(call, 1), 0);
        if (!strcmp(ops[op].name, "sub"))
            data = f ? op1(IR_FNEG, IR_F32, data) : op2(IR_SUB, IR_I32, konst(IR_I32, 0), data);
    }
    if (f && (ops[op].op == SASS_ATOM_EXCH || ops[op].op == SASS_ATOM_CAS))
        type = SASS_ATOM_U32;  // the same bits
    p = fit_offset(p);
    struct ir_val *a = ir_add(g_cur, IR_ATOM, t, swap ? 3 : 2);
    a->src[0] = p.base;
    a->src[1] = data;
    if (swap)
        a->src[2] = swap;
    a->imm = p.off;
    a->aux = IR_ATOMA(ops[op].op, type, shared);
    if (expected) {
        struct ptr q = ptr_of(expected);
        if (q.space != SP_PRIVATE || q.base)
            airc_fail("a compare-exchange's expected value outside thread memory");
        private_write(private_var(q, 0), a);
    }
    if (LLVMGetTypeKind(LLVMTypeOf(call)) != LLVMVoidTypeKind)
        set(call, 0, a);
}

// The SIMD-group shuffles to a lane the call names, lane by lane. 16-bit values go through 32 bits. False for a
// call that isn't one.
static int simd_op(LLVMValueRef call, const char *name)
{
    static const struct {
        const char *fn;
        int mode;
    } shuffles[] = {
        { "simd_shuffle", SASS_SHFL_IDX },        { "simd_broadcast", SASS_SHFL_IDX },
        { "simd_shuffle_xor", SASS_SHFL_BFLY },   { "simd_shuffle_up", SASS_SHFL_UP },
        { "simd_shuffle_down", SASS_SHFL_DOWN },
    };
    int shuffle = -1;
    for (unsigned i = 0; i < sizeof shuffles / sizeof shuffles[0]; i++)
        if (is_fn(name, shuffles[i].fn))
            shuffle = (int)i;
    if (shuffle < 0)
        return 0;
    LLVMValueRef a = LLVMGetOperand(call, 0);
    LLVMTypeRef t = LLVMTypeOf(call);
    for (unsigned l = 0; l < nlanes(t); l++) {
        enum ir_type lt = ir_type_of(lane_type(t, l));
        struct ir_val *x = widen(get(a, l));
        set(call, l, narrow(shfl(x, shuffles[shuffle].mode, widen(get(LLVMGetOperand(call, 1), 0))), lt));
    }
    return 1;
}

static struct ir_val *active_lanes(void)
{
    return op1(IR_VOTE, IR_I32, konst(IR_PRED, 1));
}

static struct ir_val *lowest_bit(struct ir_val *mask)
{
    return op2(IR_SUB, IR_I32, konst(IR_I32, 31), op1(IR_FLO, IR_I32, op1(IR_BREV, IR_I32, mask)));
}

// Where a reduction over the active lanes of each `cluster` (the SIMD group's 32, or a quad's 4) reads at each
// step: at distance o, any active lane of the other o-lane half of its aligned 2o-lane group, as every active lane
// of a half holds that half's whole by then. A half with none adds nothing. Lanes switched off by divergence, an
// early return or a short threadgroup are never read: their registers hold anything.
static unsigned reduce_plan(unsigned cluster, struct ir_val **src, struct ir_val **has)
{
    struct ir_val *lane = s2r(SR_LANEID), *active = active_lanes();
    unsigned n = 0;
    for (unsigned o = 1; o < cluster; o <<= 1, n++) {
        struct ir_val *base = op2(IR_AND, IR_I32, op2(IR_XOR, IR_I32, lane, konst(IR_I32, o)), konst(IR_I32, 32 - o));
        struct ir_val *half = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, active, base), konst(IR_I32, (1u << o) - 1));
        src[n] = op2(IR_ADD, IR_I32, base, lowest_bit(half));
        has[n] = icmp(IR_NE, half, konst(IR_I32, 0));
    }
    return n;
}

// A prefix over the active lanes below this one in its cluster, as Mesa's generic scan: each step takes in what the
// nearest lane below not yet counted has gathered, then goes on to what that lane hadn't. below is those lanes,
// the ones an exclusive prefix reads the inclusive one of the nearest of.
static unsigned scan_plan(unsigned cluster, struct ir_val **src, struct ir_val **has, struct ir_val **below)
{
    struct ir_val *lane = s2r(SR_LANEID);
    struct ir_val *lt = op2(IR_SUB, IR_I32, op2(IR_SHL, IR_I32, konst(IR_I32, 1), lane), konst(IR_I32, 1));
    if (cluster == 4)
        lt = op2(IR_AND, IR_I32, lt,
                 op2(IR_SHL, IR_I32, konst(IR_I32, -1), op2(IR_AND, IR_I32, lane, konst(IR_I32, 28))));
    struct ir_val *rest = *below = op2(IR_AND, IR_I32, active_lanes(), lt);
    unsigned n = 0;
    for (unsigned o = 1; o < cluster; o <<= 1, n++) {
        has[n] = icmp(IR_NE, rest, konst(IR_I32, 0));
        src[n] = op1(IR_FLO, IR_I32, rest);
        rest = sel(has[n], shfl(rest, SASS_SHFL_IDX, src[n]), konst(IR_I32, 0));
    }
    return n;
}

// The rest of the SIMD-group functions, and the quad ones (groups of 4 lanes, which IR_SHFL_QUAD keeps a shuffle
// within): votes, reductions, prefix sums and products, and shuffles to a lane worked out from this one's. False
// for a call that isn't one of them.
static int simd_lanes(LLVMValueRef call, const char *name)
{
    static const char *const prefixes[] = { "simd_prefix_inclusive_", "simd_prefix_exclusive_",
                                            "quad_prefix_inclusive_", "quad_prefix_exclusive_" };
    static const struct {
        const char *fn;
        enum ir_op f, i;
    } reductions[] = {
        { "simd_sum", IR_FADD, IR_ADD }, { "simd_product", IR_FMUL, IR_MUL }, { "simd_max", IR_FMAX, IR_MAX },
        { "simd_min", IR_FMIN, IR_MIN }, { "simd_and", IR_UNDEF, IR_AND },    { "simd_or", IR_UNDEF, IR_OR },
        { "simd_xor", IR_UNDEF, IR_XOR }, { "quad_sum", IR_FADD, IR_ADD },     { "quad_product", IR_FMUL, IR_MUL },
        { "quad_max", IR_FMAX, IR_MAX }, { "quad_min", IR_FMIN, IR_MIN },     { "quad_and", IR_UNDEF, IR_AND },
        { "quad_or", IR_UNDEF, IR_OR },  { "quad_xor", IR_UNDEF, IR_XOR },
    };
    LLVMTypeRef t = LLVMTypeOf(call);
    int is_signed = strstr(name, ".s.") != NULL;
    if (!strcmp(name, "air.simdgroup.barrier"))  // a warp runs its lanes together: nothing to wait for
        return 1;
    if (is_fn(name, "get_simdgroup_size")) {
        set(call, 0, konst(IR_I32, 32));
        return 1;
    }
    if (is_fn(name, "simd_ballot") || is_fn(name, "simd_active_threads_mask")) {
        int all = LLVMGetNumOperands(call) == 1;
        set(call, 0, all ? active_lanes() : op1(IR_VOTE, IR_I32, get(LLVMGetOperand(call, 0), 0)));
        return 1;
    }
    if (!strcmp(name, "air.simd_any") || !strcmp(name, "air.simd_all")) {
        struct ir_val *mask = op1(IR_VOTE, IR_I32, get(LLVMGetOperand(call, 0), 0));
        set(call, 0, name[10] == 'n' ? icmp(IR_NE, mask, konst(IR_I32, 0)) : icmp(IR_EQ, mask, active_lanes()));
        return 1;
    }
    if (!strcmp(name, "air.simd_is_first")) {
        set(call, 0, icmp(IR_EQ, s2r(SR_LANEID), lowest_bit(active_lanes())));
        return 1;
    }
    if (is_fn(name, "quad_active_threads_mask") || is_fn(name, "quad_ballot") || !strcmp(name, "air.quad_ballot") ||
        !strcmp(name, "air.quad_active_threads_mask") || !strcmp(name, "air.quad_any") ||
        !strcmp(name, "air.quad_all") || !strcmp(name, "air.quad_is_first")) {
        struct ir_val *lane = s2r(SR_LANEID), *base = op2(IR_AND, IR_I32, lane, konst(IR_I32, 28));
        struct ir_val *quad = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, active_lanes(), base), konst(IR_I32, 15)), *r;
        if (name[9] == 'a' && name[10] == 'c') {
            r = quad;
        } else if (name[9] == 'i') {
            r = icmp(IR_EQ, op2(IR_AND, IR_I32, lane, konst(IR_I32, 3)), lowest_bit(quad));
        } else {
            struct ir_val *mask = op1(IR_VOTE, IR_I32, get(LLVMGetOperand(call, 0), 0));
            mask = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, mask, base), konst(IR_I32, 15));
            r = name[9] == 'b'    ? mask
                : name[10] == 'n' ? icmp(IR_NE, mask, konst(IR_I32, 0))
                                  : icmp(IR_EQ, mask, quad);
        }
        set(call, 0, r);
        return 1;
    }
    if (is_fn(name, "is_uniform")) {  // every active lane holds the first one's value
        struct ir_val *x = widen(get(LLVMGetOperand(call, 0), 0));
        struct ir_val *same = icmp(IR_EQ, x, shfl(x, SASS_SHFL_IDX, lowest_bit(active_lanes())));
        set(call, 0, icmp(IR_EQ, op1(IR_VOTE, IR_I32, same), active_lanes()));
        return 1;
    }
    int prefix = -1, reduce = -1;
    for (unsigned i = 0; i < sizeof prefixes / sizeof prefixes[0]; i++)
        if (!strncmp(name + 4, prefixes[i], strlen(prefixes[i])))
            prefix = (int)i;
    for (unsigned i = 0; i < sizeof reductions / sizeof reductions[0]; i++)
        if (is_fn(name, reductions[i].fn))
            reduce = (int)i;
    if (prefix >= 0 || reduce >= 0) {
        unsigned cluster = !strncmp(name + 4, "quad_", 5) ? 4 : 32, n;
        int product = prefix >= 0 && strstr(name, "_product.") != NULL;
        struct ir_val *src[5], *has[5], *below = NULL;
        n = reduce >= 0 ? reduce_plan(cluster, src, has) : scan_plan(cluster, src, has, &below);
        for (unsigned l = 0; l < nlanes(t); l++) {
            enum ir_type lt = ir_type_of(lane_type(t, l));
            struct ir_val *x = widen(get(LLVMGetOperand(call, 0), l));
            int f = x->type == IR_F32;
            enum ir_op op = reduce >= 0 ? (f ? reductions[reduce].f : reductions[reduce].i)
                            : product ? (f ? IR_FMUL : IR_MUL)
                                      : (f ? IR_FADD : IR_ADD);
            if (op == IR_UNDEF)
                airc_fail("unsupported SIMD-group function (%s)", name);
            if (op == IR_MIN || op == IR_MAX)  // a narrower integer's high bits are junk
                x = int_extend(x, 8 * scalar_bytes(lane_type(t, l)), is_signed);
            for (unsigned i = 0; i < n; i++) {
                struct ir_val *r = op2(op, x->type, x, shfl(x, SASS_SHFL_IDX, src[i]));
                if (op == IR_MIN || op == IR_MAX)
                    r->aux = is_signed;
                x = sel(has[i], r, x);
            }
            if (prefix == 1 || prefix == 3) {  // the inclusive one of the nearest lane below, the identity for none
                struct ir_val *identity = product ? (f ? fconst(1) : konst(IR_I32, 1)) : konst(x->type, 0);
                x = sel(icmp(IR_NE, below, konst(IR_I32, 0)), shfl(x, SASS_SHFL_IDX, op1(IR_FLO, IR_I32, below)),
                        identity);
            }
            set(call, l, narrow(x, lt));
        }
        return 1;
    }
    // shuffles to a lane computed from this one's: index (or delta) and, for a fill, the data's width
    static const struct {
        const char *fn;
        int mode, rotate, fill;
    } shuffles[] = {
        { "simd_broadcast_first", SASS_SHFL_IDX, 0, 0 },
        { "quad_broadcast_first", SASS_SHFL_IDX | IR_SHFL_QUAD, 0, 0 },
        { "simd_shuffle_rotate_down", SASS_SHFL_IDX, 1, 0 },
        { "simd_shuffle_rotate_up", SASS_SHFL_IDX, -1, 0 },
        { "simd_shuffle_and_fill_down", SASS_SHFL_IDX, 1, 1 },
        { "simd_shuffle_and_fill_up", SASS_SHFL_IDX, -1, 1 },
        { "quad_shuffle", SASS_SHFL_IDX | IR_SHFL_QUAD, 0, 0 },
        { "quad_broadcast", SASS_SHFL_IDX | IR_SHFL_QUAD, 0, 0 },
        { "quad_shuffle_xor", SASS_SHFL_BFLY | IR_SHFL_QUAD, 0, 0 },
        { "quad_shuffle_up", SASS_SHFL_UP | IR_SHFL_QUAD, 0, 0 },
        { "quad_shuffle_down", SASS_SHFL_DOWN | IR_SHFL_QUAD, 0, 0 },
        { "quad_shuffle_rotate_down", SASS_SHFL_IDX | IR_SHFL_QUAD, 1, 0 },
        { "quad_shuffle_rotate_up", SASS_SHFL_IDX | IR_SHFL_QUAD, -1, 0 },
        { "quad_shuffle_and_fill_down", SASS_SHFL_IDX | IR_SHFL_QUAD, 1, 1 },
        { "quad_shuffle_and_fill_up", SASS_SHFL_IDX | IR_SHFL_QUAD, -1, 1 },
    };
    int s = -1;
    for (unsigned i = 0; i < sizeof shuffles / sizeof shuffles[0]; i++)
        if (is_fn(name, shuffles[i].fn))
            s = (int)i;
    if (s < 0)
        return 0;
    LLVMValueRef data = LLVMGetOperand(call, 0);
    int quad = shuffles[s].mode & IR_SHFL_QUAD;
    struct ir_val *src, *in_range = NULL;
    if (s == 0) {
        src = lowest_bit(active_lanes());
    } else if (s == 1) {
        struct ir_val *base = op2(IR_AND, IR_I32, s2r(SR_LANEID), konst(IR_I32, 28));
        src = lowest_bit(op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, active_lanes(), base), konst(IR_I32, 15)));
    } else {
        int fill = shuffles[s].fill;
        struct ir_val *by = op2(IR_AND, IR_I32, get(LLVMGetOperand(call, 1 + fill), 0), konst(IR_I32, 0xffff));
        src = by;
        if (shuffles[s].rotate) {  // a fill's width is a power of 2: the lane moves within its slice that wide
            struct ir_val *lane = s2r(SR_LANEID), *mask = konst(IR_I32, quad ? 3 : 31);
            if (fill && LLVMGetNumOperands(call) > 4)
                mask = op2(IR_SUB, IR_I32, op2(IR_AND, IR_I32, get(LLVMGetOperand(call, 3), 0), konst(IR_I32, 0xffff)),
                           konst(IR_I32, 1));
            struct ir_val *at = op2(IR_AND, IR_I32, lane, mask);
            struct ir_val *moved = op2(shuffles[s].rotate > 0 ? IR_ADD : IR_SUB, IR_I32, at, by);
            src = op2(IR_OR, IR_I32, op2(IR_XOR, IR_I32, lane, at), op2(IR_AND, IR_I32, moved, mask));
            if (fill)  // past the slice's end (or its start), from the fill
                in_range = shuffles[s].rotate > 0 ? icmp(IR_LE, moved, mask)
                                                  : icmp(IR_GE | IR_SIGNED, moved, konst(IR_I32, 0));
        }
    }
    for (unsigned l = 0; l < nlanes(t); l++) {
        enum ir_type lt = ir_type_of(lane_type(t, l));
        struct ir_val *x = shfl(widen(get(data, l)), shuffles[s].mode, src);
        if (in_range)
            x = sel(in_range, x, shfl(widen(get(LLVMGetOperand(call, 1), l)), shuffles[s].mode, src));
        set(call, l, narrow(x, lt));
    }
    return 1;
}

// AIR's texture calls - sample (with its LOD, gradient, depth and compare forms), gather and read - as one IR_TEX
// in NAK's operand order. After the texture and the sampler come: a depth texture's i32, the coordinate, an
// array's layer (a cube's face, to read one), a compare's reference, a gradient's two derivatives and min LOD,
// then (but for cubes) an offset (whether there is one, then it; a read has no flag), then a sample's LOD (level
// or bias, its value, a min-LOD clamp), a gather's component, a read's LOD.
static void texture_op(LLVMValueRef call, const char *name)
{
    int gather = strstr(name, "gather_") != NULL, grad = strstr(name, "_grad.") != NULL;
    int read = strstr(name, "read_") != NULL;
    int depth = strstr(name, "_depth_") != NULL, compare = strstr(name, "_compare_") != NULL;
    int cube = strstr(name, "_cube") != NULL, array = strstr(name, "_array") != NULL;
    int d1 = strstr(name, "_1d") != NULL, d3 = strstr(name, "_3d") != NULL;
    int ms = strstr(name, "_ms") != NULL;
    // Metal 1.x's AIR (Roblox ships shaders built for it) returns a plain vector, with no residency flag; its samples
    // have no min LOD operand, its reads no sampler or offset
    int legacy = LLVMGetTypeKind(LLVMTypeOf(call)) != LLVMStructTypeKind;
    // the OpenGL renderer's reads name no sampler either: texture, coordinate, level, 0 (a residency flag comes back)
    int bare = read && LLVMGetTypeKind(LLVMTypeOf(LLVMGetOperand(call, 1))) != LLVMPointerTypeKind;
    if ((ms && (!read || array)) || (!d1 && !d3 && !cube && !strstr(name, "_2d")) || (grad && (cube || d3)))
        airc_fail("unsupported texture op (%s)", name);
    struct ir_val *texture = cube && read ? faces_handle(LLVMGetOperand(call, 0))
                                          : handle_of(LLVMGetOperand(call, 0), SP_TEXTURE);
    struct ir_val *handle = read ? texture : op2(IR_OR, IR_I32, texture, sampler_handle(LLVMGetOperand(call, 1), compare));
    if (!read && !gather && (strstr(name, ".u.") || strstr(name, ".s.") || strstr(name, ".i.")))
        note_use(ptr_of(LLVMGetOperand(call, 1)), USE_NEAREST);
    unsigned i = (legacy || bare) && read ? 1 : depth ? 3 : 2;  // a legacy read has no sampler
    LLVMValueRef coord = LLVMGetOperand(call, i++);
    LLVMValueRef layer = array || (cube && read) ? LLVMGetOperand(call, i++) : NULL;
    // a cube array's texel is read as the layer of its cube's face in the 2D array of all the faces
    struct ir_val *layer_val = layer ? get(layer, 0) : NULL;
    if (cube && array && read) {
        struct ir_val *cube_index = op2(IR_MUL, IR_I32, get(LLVMGetOperand(call, i++), 0), konst(IR_I32, 6));
        layer_val = op2(IR_ADD, IR_I32, cube_index, layer_val);
    }
    struct ir_val *ref = compare ? get(LLVMGetOperand(call, i++), 0) : NULL;
    LLVMValueRef ddx = NULL, ddy = NULL;
    if (grad) {
        ddx = LLVMGetOperand(call, i++);
        ddy = LLVMGetOperand(call, i);
        i += legacy ? 1 : 2;  // and the min LOD, which goes
    }
    unsigned ncoord = d3 || (cube && !read) ? 3 : d1 ? 1 : 2;
    struct ir_val *offset = NULL;
    if (!cube && !(read && d1) && !ms) {
        if (read && (legacy || bare)) {
            // no offset
        } else if (read) {
            LLVMValueRef off = LLVMGetOperand(call, i++);
            int zero = 1;
            for (unsigned c = 0; c < ncoord; c++)
                zero &= is_konst(get(off, c), 0);
            if (!zero)
                airc_fail("an offset texel read");
        } else {
            offset = packed_offset(LLVMGetOperand(call, i), LLVMGetOperand(call, i + 1), ncoord, gather ? 8 : 4);
            i += 2;
        }
    }

    // the LOD: a read's or a sample's level, a sample's bias (implicit LOD in a fragment function, else LOD 0)
    int lod_mode = SASS_LOD_ZERO;
    struct ir_val *lod = NULL, *min_lod = NULL;
    unsigned comp = 0;
    struct ir_val *sample = ms ? get(LLVMGetOperand(call, i), 0) : NULL;
    if (read && !d1 && !ms) {
        lod = get(LLVMGetOperand(call, i), 0);
    } else if (gather) {
        if (!compare)
            comp = (unsigned)get(LLVMGetOperand(call, i), 0)->imm & 3;
    } else if (!grad && !read) {
        struct ir_val *level = get(LLVMGetOperand(call, i), 0);
        lod = get(LLVMGetOperand(call, i + 1), 0);
        if (level->op != IR_CONST)
            airc_fail("a sample whose LOD may be a level or a bias");
        if (!level->imm && !legacy) {
            min_lod = get(LLVMGetOperand(call, i + 2), 0);
            if (is_konst(min_lod, 0))
                min_lod = NULL;
        }
        if (!level->imm && g_info->stage == NVK_FRAGMENT) {
            lod_mode = is_konst(lod, 0) ? SASS_LOD_AUTO : SASS_LOD_BIAS;
            if (min_lod)
                lod_mode = lod_mode == SASS_LOD_AUTO ? SASS_LOD_CLAMP : SASS_LOD_BIAS_CLAMP;
        } else if (!level->imm) {
            // no derivatives: LOD 0, a bias left out, raised to the clamp
            lod = min_lod;
            min_lod = NULL;
        }
    }
    if (lod && lod_mode == SASS_LOD_ZERO)
        lod_mode = is_konst(lod, 0) ? SASS_LOD_ZERO : SASS_LOD_LOD;
    if (lod_mode == SASS_LOD_ZERO || lod_mode == SASS_LOD_AUTO || lod_mode == SASS_LOD_CLAMP)
        lod = NULL;

    // the first vector: the layer (an integer), then the coordinates (TXD: the coordinates, then the layer, with
    // the offset in its low half); a 1D texture is a 2D one a row high, a 1D array one with a row per layer (the
    // plugin lays them out so)
    struct ir_val *src[12];
    unsigned n = 0;
    int dim = d3 ? SASS_DIM_3D
              : cube ? (read ? SASS_DIM_ARRAY_2D : array ? SASS_DIM_ARRAY_CUBE : SASS_DIM_CUBE)
              : array && !d1 ? SASS_DIM_ARRAY_2D
                             : SASS_DIM_2D;
    struct ir_val *first = layer && !d1 && !grad ? layer_val : NULL;
    if (min_lod) {  // the clamp (8.8, NAK keeps it at 1/16 at least) in the high half of the layer's word
        struct ir_val *ml = op2(IR_FMAX, IR_F32, op2(IR_FMUL, IR_F32, min_lod, fconst(256)), fconst(16));
        ml = op2(IR_SHL, IR_I32, op1(IR_F2I, IR_I32, ml), konst(IR_I32, 16));
        first = first ? op2(IR_OR, IR_I32, op2(IR_AND, IR_I32, first, konst(IR_I32, 0xffff)), ml) : ml;
    }
    if (first)
        src[n++] = first;
    // the hardware takes a cube's direction over a major axis of 1: it picks the face but does not divide by it
    struct ir_val *cube_scale = NULL;
    if (cube && !read) {
        struct ir_val *major = op1(IR_FABS, IR_F32, get(coord, 0));
        for (unsigned c = 1; c < 3; c++)
            major = op2(IR_FMAX, IR_F32, major, op1(IR_FABS, IR_F32, get(coord, c)));
        cube_scale = mufu(MUFU_RCP, major);
    }
    for (unsigned c = 0; c < ncoord; c++)
        src[n++] = cube_scale ? op2(IR_FMUL, IR_F32, get(coord, c), cube_scale) : get(coord, c);
    if (d1 && !array) {
        src[n++] = read ? konst(IR_I32, 0) : fconst(0.5f);
    } else if (d1) {
        struct ir_val *y = get(layer, 0);
        if (!read) {
            struct ir_val *row = op1(IR_I2F, IR_F32, y);
            row->aux = 1;
            struct ir_val *layers = op2(IR_TXQ, IR_I32, konst(IR_I32, 0), texture);
            layers->aux = 2;
            struct ir_val *lf = op1(IR_I2F, IR_F32, layers);
            lf->aux = 1;
            y = op2(IR_FMUL, IR_F32, op2(IR_FADD, IR_F32, row, fconst(0.5f)), mufu(MUFU_RCP, lf));
        }
        src[n++] = y;
    }
    if (grad && (offset || (layer && !d1))) {
        struct ir_val *l = layer && !d1 ? op2(IR_SHL, IR_I32, get(layer, 0), konst(IR_I32, 16)) : konst(IR_I32, 0);
        src[n++] = offset ? op2(IR_OR, IR_I32, l, offset) : l;
    }
    unsigned n0 = n;
    // the second, after the handle: the LOD, the offset, the reference; a gradient's derivatives instead
    if (grad) {
        for (unsigned c = 0; c < ncoord; c++) {
            src[n++] = get(ddx, c);
            src[n++] = get(ddy, c);
        }
    } else {
        if (lod)
            src[n++] = lod;
        if (offset)
            src[n++] = offset;
        if (ref)
            src[n++] = ref;
        if (sample)
            src[n++] = sample;
    }
    LLVMTypeRef rt = LLVMTypeOf(call);
    unsigned results = nlanes(legacy ? rt : LLVMStructGetTypeAtIndex(rt, 0));
    int op = grad ? SASS_TXD : gather ? SASS_TLD4 : read ? SASS_TLD : SASS_TEX;
    unsigned flags = SASS_TEXF_LOD(lod_mode) | (offset ? SASS_TEXF_AOFFI : 0) | (ref ? SASS_TEXF_DC : 0) |
                     SASS_TEXF_COMP(comp) | (sample ? SASS_TEXF_MS : 0);
    src[n++] = handle;
    struct ir_val *t = ir_add(g_cur, IR_TEX, IR_VOID, n);
    memcpy(t->src, src, n * sizeof(src[0]));
    t->aux = IR_TEXA(op, dim, flags, n0) | (results == 1 ? IR_TEXA_SCALAR : 0);

    // the results: a vector or (depth) a scalar, then whether the texel was resident
    for (unsigned l = 0; l < results; l++) {
        enum ir_type lt = ir_type_of(lane_type(rt, l));
        struct ir_val *p = op1(IR_PROJ, lt == IR_F16 ? IR_F32 : lt, t);
        p->imm = l;
        set(call, l, narrow(p, lt));
    }
    if (!legacy)
        set(call, results, konst(IR_I32, 1));
}

static void store_value(struct ptr p, LLVMTypeRef t, LLVMValueRef val, unsigned align);

// A call: an intrinsic in place, or one of the module's functions inlined or (noinline) called.
static void inline_call(LLVMValueRef call, LLVMValueRef fn);
static int call_sub(LLVMValueRef call, LLVMValueRef fn);
static void control_point(LLVMValueRef call);
static void picked_from(struct sub_param *param, struct ptr from);
static void note_picked_uses(void);

static LLVMValueRef first_node(LLVMModuleRef m, const char *name)
{
    LLVMValueRef *ops = calloc(LLVMGetNamedMetadataNumOperands(m, name), sizeof(*ops));
    LLVMGetNamedMetadataOperands(m, name, ops);
    LLVMValueRef node = ops[0];
    free(ops);
    return node;
}

// a function the module only declares, defined in a linked module: Metal calls a visible function NAME through
// a declaration NAME.MTL_VISIBLE_FN_REF
static LLVMValueRef linked_function(const char *name)
{
    size_t n = strlen(name);
    static const char ref[] = ".MTL_VISIBLE_FN_REF";
    if (n > sizeof ref - 1 && !strcmp(name + n - (sizeof ref - 1), ref))
        n -= sizeof ref - 1;
    for (unsigned i = 0; i < g_nlinked; i++)
        for (LLVMValueRef fn = LLVMGetFirstFunction(g_linked[i]); fn; fn = LLVMGetNextFunction(fn)) {
            const char *s = LLVMGetValueName(fn);
            if (s && strlen(s) == n && !strncmp(s, name, n) && LLVMGetFirstBasicBlock(fn))
                return fn;
        }
    // one linked under another name: its module's visible function
    for (unsigned i = 0; g_linked_names && i < g_nlinked; i++) {
        const char *a = g_linked_names[i];
        if (!a || strlen(a) != n || strncmp(a, name, n) || !LLVMGetNamedMetadataNumOperands(g_linked[i], "air.visible"))
            continue;
        LLVMValueRef node = first_node(g_linked[i], "air.visible");
        if (LLVMGetMDNodeNumOperands(node) < 1)
            continue;
        LLVMValueRef ops[8];
        if (LLVMGetMDNodeNumOperands(node) > 8)
            continue;
        LLVMGetMDNodeOperands(node, ops);
        if (LLVMIsAFunction(ops[0]) && LLVMGetFirstBasicBlock(ops[0]))
            return ops[0];
    }
    return NULL;
}

// The normalized integer fields of a 32-bit word an air.pack or air.unpack name packs: <unorm|snorm><n>x<bits>,
// unorm4x8.srgb, unorm.rgb10a2 and unorm.rgb565, first field lowest. How many there are, or 0 for none of these.
static unsigned packed_fields(const char *name, unsigned *bits, int *snorm, int *srgb)
{
    const char *k = strchr(name + 4, '.') + 1;
    int count, b;
    *snorm = !strncmp(k, "snorm", 5);
    if (!*snorm && strncmp(k, "unorm", 5))
        return 0;
    k += 5;
    *srgb = !strncmp(k, "4x8.srgb.", 9);
    if (!strncmp(k, ".rgb10a2.", 9) || !strncmp(k, ".rgb565.", 8)) {
        static const unsigned rgb10a2[] = { 10, 10, 10, 2 }, rgb565[] = { 5, 6, 5 };
        int wide = k[4] == '1';
        memcpy(bits, wide ? rgb10a2 : rgb565, (wide ? 4 : 3) * sizeof(*bits));
        return wide ? 4 : 3;
    }
    if (sscanf(k, "%dx%d", &count, &b) != 2 || count * b != 32)
        return 0;
    for (int l = 0; l < count; l++)
        bits[l] = (unsigned)b;
    return (unsigned)count;
}

// sRGB's transfer function, to it from linear (encode) or back
static struct ir_val *srgb(struct ir_val *c, int encode)
{
    struct ir_val *lin, *curve;
    if (encode) {  // 12.92 c up to 0.0031308, 1.055 c^(1/2.4) - 0.055 above
        lin = op2(IR_FMUL, IR_F32, c, fconst(12.92f));
        curve = mufu(MUFU_EX2, op2(IR_FMUL, IR_F32, mufu(MUFU_LG2, c), fconst(1 / 2.4f)));
        curve = op3(IR_FFMA, IR_F32, curve, fconst(1.055f), fconst(-0.055f));
        return sel(fcmp(FCMP_OGT, c, fconst(0.0031308f)), curve, lin);
    }
    lin = op2(IR_FMUL, IR_F32, c, fconst(1 / 12.92f));  // c / 12.92 up to 0.04045, ((c + 0.055) / 1.055)^2.4 above
    curve = op3(IR_FFMA, IR_F32, c, fconst(1 / 1.055f), fconst(0.055f / 1.055f));
    curve = mufu(MUFU_EX2, op2(IR_FMUL, IR_F32, mufu(MUFU_LG2, curve), fconst(2.4f)));
    return sel(fcmp(FCMP_OGT, c, fconst(0.04045f)), curve, lin);
}

// k / d correctly rounded, k an integral float: the reciprocal's product corrected by its exact remainder (checked
// against every 16-bit k for the divisors packed fields use), where k * (1 / d) is off by an ulp at times
static struct ir_val *div_exact(struct ir_val *k, unsigned d)
{
    struct ir_val *rcp = fconst(1.0f / (float)d);
    struct ir_val *q = op2(IR_FMUL, IR_F32, k, rcp);
    struct ir_val *e = op3(IR_FFMA, IR_F32, op1(IR_FNEG, IR_F32, q), fconst((float)d), k);
    return op3(IR_FFMA, IR_F32, e, rcp, q);
}

// An 8x8 simdgroup matrix is a <64 x T> in AIR that only its first two lanes carry: each thread owns two elements
// of one row, (fm, fn) and (fm, fn + 1) with the coordinates Apple's hardware deals them out by, and the operations
// work across the whole simdgroup. The multiply-accumulate reads each row of A and column of B from the threads that
// hold them, a shuffle for every element.
static void simdgroup_matrix(LLVMValueRef in, int multiply)
{
    LLVMTypeRef rt = LLVMTypeOf(in);
    struct ir_val *lane = s2r(SR_LANEID);
    struct ir_val *bit[5];
    for (unsigned i = 0; i < 5; i++)
        bit[i] = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, lane, konst(IR_I32, i)), konst(IR_I32, 1));
    struct ir_val *fm = op2(IR_ADD, IR_I32, op2(IR_SHL, IR_I32, bit[4], konst(IR_I32, 2)),
                            op2(IR_ADD, IR_I32, op2(IR_SHL, IR_I32, bit[2], konst(IR_I32, 1)), bit[1]));
    struct ir_val *fn = op2(IR_ADD, IR_I32, op2(IR_SHL, IR_I32, bit[3], konst(IR_I32, 2)),
                            op2(IR_SHL, IR_I32, bit[0], konst(IR_I32, 1)));
    enum ir_type et = ir_type_of(lane_type(rt, 0));
    struct ir_val *out[2];
    if (!multiply) {  // the diagonal: element e of this thread is on it when the row is the column
        struct ir_val *x = get(LLVMGetOperand(in, 0), 0);
        for (unsigned e = 0; e < 2; e++)
            out[e] = sel(icmp(IR_EQ, fm, op2(IR_ADD, IR_I32, fn, konst(IR_I32, e))), x, konst(x->type, 0));
    } else {
        struct ir_val *a[2], *b[2], *acc[2];
        for (unsigned e = 0; e < 2; e++) {
            a[e] = widen(get(LLVMGetOperand(in, 0), e));
            b[e] = widen(get(LLVMGetOperand(in, 1), e));
            acc[e] = widen(get(LLVMGetOperand(in, 2), e));
        }
        for (unsigned k = 0; k < 8; k++) {
            // A[fm][k] lives in the thread with this thread's row bits and k's column bits, B[k][fn + e] in the one
            // with k's row bits and this thread's column bits
            struct ir_val *from_a = op2(IR_OR, IR_I32, op2(IR_AND, IR_I32, lane, konst(IR_I32, 0x16)),
                                        konst(IR_I32, (k >> 2 & 1) << 3 | (k >> 1 & 1)));
            struct ir_val *from_b = op2(IR_OR, IR_I32, op2(IR_AND, IR_I32, lane, konst(IR_I32, 0x09)),
                                        konst(IR_I32, (k >> 2 & 1) << 4 | (k >> 1 & 1) << 2 | (k & 1) << 1));
            struct ir_val *ak = shfl(a[k & 1], SASS_SHFL_IDX, from_a);
            for (unsigned e = 0; e < 2; e++)
                acc[e] = op3(IR_FFMA, IR_F32, ak, shfl(b[e], SASS_SHFL_IDX, from_b), acc[e]);
        }
        for (unsigned e = 0; e < 2; e++)
            out[e] = narrow(acc[e], et);
    }
    for (unsigned l = 0; l < nlanes(rt); l++)
        set(in, l, l < 2 ? out[l] : konst(et, 0));
}

static struct ir_val *bfloat_to_float(struct ir_val *x);
static struct ir_val *float_to_bfloat(struct ir_val *x);

// OpenCL's work-item functions (air.get_global_id.i32 and the rest), which Apple's OpenCL compiles to these rather
// than to kernel arguments. The dimension is an argument; from 3 on an id is 0 and a size 1, as OpenCL has it. There
// is no global offset (the dispatch has none), and the work dimension is the last axis longer than 1. 0 for another
// air.get_ name.
static int workitem(LLVMValueRef in, const char *what)
{
    static const struct {
        const char *name;
        enum nvk_arg_kind kind;
        int size;  // 1 for a size, which is 1 past the axes
    } fn[] = {
        { "global_id.", NVK_ARG_TID, 0 },       { "local_id.", NVK_ARG_TG_TID, 0 },
        { "group_id.", NVK_ARG_TG_POS, 0 },     { "global_size.", NVK_ARG_GRID_SIZE, 1 },
        { "local_size.", NVK_ARG_TG_SIZE, 1 },  { "num_groups.", NVK_ARG_GROUPS, 1 },
    };
    struct ir_val *v = NULL;
    if (!strncmp(what, "global_offset.", 14)) {
        v = konst(IR_I32, 0);
    } else if (!strncmp(what, "work_dim", 8)) {
        struct ir_val *y = op2(IR_CMP, IR_PRED, thread_value(NVK_ARG_GRID_SIZE, 1), konst(IR_I32, 1));
        struct ir_val *z = op2(IR_CMP, IR_PRED, thread_value(NVK_ARG_GRID_SIZE, 2), konst(IR_I32, 1));
        y->aux = z->aux = IR_GT;
        v = sel(z, konst(IR_I32, 3), sel(y, konst(IR_I32, 2), konst(IR_I32, 1)));
    } else {
        unsigned i = 0;
        while (i < sizeof(fn) / sizeof(fn[0]) && strncmp(what, fn[i].name, strlen(fn[i].name)))
            i++;
        if (i == sizeof(fn) / sizeof(fn[0]))
            return 0;
        LLVMValueRef dim = LLVMGetOperand(in, 0);
        if (LLVMIsAConstantInt(dim)) {
            uint64_t d = LLVMConstIntGetZExtValue(dim);
            v = d < 3 ? thread_value(fn[i].kind, (unsigned)d) : konst(IR_I32, fn[i].size);
        } else {
            struct ir_val *d = get(dim, 0);
            v = konst(IR_I32, fn[i].size);
            for (int l = 2; l >= 0; l--) {
                struct ir_val *is = op2(IR_CMP, IR_PRED, d, konst(IR_I32, l));
                is->aux = IR_EQ;
                v = sel(is, thread_value(fn[i].kind, (unsigned)l), v);
            }
        }
    }
    set(in, 0, v);
    if (is_i64(LLVMTypeOf(in)))
        set_high(in, 0, konst(IR_I32, 0));
    return 1;
}

static void call(LLVMValueRef in)
{
    LLVMValueRef fn = LLVMGetCalledValue(in);
    const char *name = LLVMGetValueName(fn);
    if (fn && LLVMGetFirstBasicBlock(fn)) {
        if (!call_sub(in, fn))
            inline_call(in, fn);
        return;
    }
    LLVMValueRef def = name && strncmp(name, "air.", 4) && strncmp(name, "llvm.", 5) ? linked_function(name) : NULL;
    if (def) {
        inline_call(in, def);
        return;
    }
    if (fn && fn == g_point_fn) {
        control_point(in);
        return;
    }
    if (!fn || !LLVMIsAFunction(fn))  // a function pointer's value, which has no name to look up
        airc_fail("indirect call");
    if (strncmp(name, "air.", 4) && strncmp(name, "llvm.", 5))
        airc_fail("call to %s, which no linked module defines", name);
    LLVMTypeRef rt = LLVMTypeOf(in);
    unsigned n = nlanes(rt);

    if (!strcmp(name, "air.wg.barrier")) {
        ir_add(g_cur, IR_BAR, IR_VOID, 0);
        g_info->barriers = 1;
        return;
    }
    if (!strncmp(name, "llvm.lifetime.", 14))  // allocas are variables, alive wherever they're used
        return;
    if (!strncmp(name, "llvm.memcpy.", 12) || !strncmp(name, "llvm.memmove.", 13)) {
        mem_copy(in);
        return;
    }
    if (!strncmp(name, "llvm.memset.", 12)) {
        mem_set(in);
        return;
    }
    if (!strcmp(name, "air.discard_fragment")) {
        ir_add(g_cur, IR_KILL, IR_VOID, 0);
        g_info->ps_sysvals |= SPH_PS_KILLS;
        return;
    }
    if (simd_op(in, name) || simd_lanes(in, name))
        return;
    if (!strcmp(name, "air.atomic.fence")) {  // whatever the memory and scope: as for the whole device
        ir_add(g_cur, IR_BAR, IR_VOID, 0)->aux = 1;
        return;
    }
    if (!strncmp(name, "air.load.device_coherent.", 25) || !strncmp(name, "air.load.system_coherent.", 25) ||
        !strncmp(name, "air.store.device_coherent.", 26) || !strncmp(name, "air.store.system_coherent.", 26)) {
        coherent_access(in, name[4] == 'l');
        return;
    }
    if (!strncmp(name, "air.atomic.", 11)) {
        atomic_op(in, name);
        return;
    }
    if (!strncmp(name, "air.sample_", 11) || !strncmp(name, "air.gather_", 11) || !strncmp(name, "air.read_texture", 16) ||
        !strncmp(name, "air.read_depth", 14)) {
        texture_op(in, name);
        return;
    }
    if (!strncmp(name, "air.is_null_texture", 19)) {  // no texture bound: header index 0, the unbound texture's
        struct ir_val *index = op2(IR_AND, IR_I32, handle_of(LLVMGetOperand(in, 0), SP_TEXTURE), konst(IR_I32, 0xfffff));
        struct ir_val *null = op2(IR_CMP, IR_PRED, index, konst(IR_I32, 0));
        null->aux = IR_EQ;
        set(in, 0, null);
        return;
    }
    if (!strncmp(name, "air.get_", 8) && workitem(in, name + 8))
        return;
    if (!strncmp(name, "air.get_null_texture", 20)) {  // the unbound texture's handle, 0, as an argument buffer's ID
        struct ptr p = { .space = SP_GLOBAL, .base = konst(IR_I64, 0) };
        insert(g_ptrs, in, 0, g_scope)->p = p;
        return;
    }
    if (!strcmp(name, "air.get_read_sampler")) {
        struct ptr p = { .space = SP_SAMPLER };
        insert(g_ptrs, in, 0, g_scope)->p = p;
        return;
    }
    if (!strncmp(name, "air.write_texture_", 18)) {
        // after the texture: the coordinate, an array's layer (a cube's face, then a cube array's cube), the texel;
        // 1D textures are 2D ones a row (a layer) high, cubes and cube arrays 2D arrays of their faces
        int d1 = strstr(name, "_1d") != NULL, d3 = strstr(name, "_3d") != NULL;
        int cube = strstr(name, "_cube") != NULL, cube_array = strstr(name, "_cube_array.") != NULL;
        int layered = strstr(name, "_array") != NULL || cube;
        if (!d1 && !d3 && !strstr(name, "_2d") && !cube)
            airc_fail("unsupported texture write (%s)", name);
        unsigned ncoord = d3 ? 3 : d1 ? 1 : 2, n = ncoord + (layered || d1);
        LLVMValueRef coord = LLVMGetOperand(in, 1), texel = LLVMGetOperand(in, cube_array ? 4 : layered ? 3 : 2);
        struct ir_val *src[9];
        for (unsigned c = 0; c < ncoord; c++)
            src[c] = get(coord, c);
        if (n > ncoord)
            src[ncoord] = layered ? get(LLVMGetOperand(in, 2), 0) : konst(IR_I32, 0);
        if (cube_array)
            src[ncoord] = op2(IR_ADD, IR_I32, op2(IR_MUL, IR_I32, get(LLVMGetOperand(in, 3), 0), konst(IR_I32, 6)),
                              src[ncoord]);
        for (unsigned c = 0; c < 4; c++)
            src[n + c] = widen(get(texel, c));  // before the store, which ir_add appends
        LLVMValueRef tex = LLVMGetOperand(in, 0);
        struct ir_val *level = get(LLVMGetOperand(in, cube_array ? 5 : layered ? 4 : 3), 0);
        src[n + 4] = level_handle(cube ? faces_handle(tex) : handle_of(tex, SP_TEXTURE), tex, level);
        struct ir_val *s = ir_add(g_cur, IR_SUST, IR_VOID, n + 5);
        memcpy(s->src, src, (n + 5) * sizeof(src[0]));
        s->imm = d3 ? SASS_IMG_3D : layered && !d1 ? SASS_IMG_ARRAY_2D : SASS_IMG_2D;
        s->aux = (int)n;
        return;
    }
    if (!strncmp(name, "air.get_", 8) && (strstr(name, "_texture_") || strstr(name, "_depth_"))) {
        // TXQ's dimensions are width, height, depth (an array's layers; a 1D array's are its height) and levels
        int levels = !strncmp(name, "air.get_num_mip_levels", 22);
        if (!strncmp(name, "air.get_num_samples", 19)) {  // TEXTURE_TYPE's third word
            struct ir_val *q = op2(IR_TXQ, IR_I32, konst(IR_I32, 0), handle_of(LLVMGetOperand(in, 0), SP_TEXTURE));
            q->aux = 4 | 1 << 4;
            set(in, 0, q);
            return;
        }
        int mask = !strncmp(name, "air.get_width", 13) ? 1
                   : !strncmp(name, "air.get_height", 14) ? 2
                   : levels ? 8
                   : !strncmp(name, "air.get_depth", 13) ? 4
                   : !strncmp(name, "air.get_array_size", 18) ? (strstr(name, "_1d_array") ? 2 : 4)
                   : 0;
        if (!mask)
            airc_fail("unsupported texture query (%s)", name);
        unsigned nops = (unsigned)LLVMGetNumOperands(in);  // the texture, the LOD if any, the callee
        struct ir_val *lod = !levels && nops > 2 ? get(LLVMGetOperand(in, 1), 0) : konst(IR_I32, 0);
        struct ir_val *q = op2(IR_TXQ, IR_I32, lod, handle_of(LLVMGetOperand(in, 0), SP_TEXTURE));
        q->aux = mask;
        set(in, 0, q);
        return;
    }
    if (!strcmp(name, "air.is_function_constant_defined")) {
        uint32_t bits;
        int index;
        const char *fc_name;
        unsigned len;
        LLVMValueRef g = LLVMGetOperand(in, 0);
        while (LLVMIsAConstantExpr(g))
            g = LLVMGetOperand(g, 0);
        set(in, 0, konst(IR_PRED, fc_bits(g, &bits, &index, &fc_name, &len)));
        return;
    }
    if (!strncmp(name, "air.normalize_function_constant_predicate", 41)) {
        struct ir_val *ne = op2(IR_CMP, IR_PRED, get(LLVMGetOperand(in, 0), 0), konst(IR_I32, 0));
        ne->aux = IR_NE;
        set(in, 0, sel(ne, konst(IR_I32, 1), konst(IR_I32, 0)));
        return;
    }
    if (is_fn(name, "sincos")) {  // the sine, and the cosine into the variable the pointer names
        struct ptr p = ptr_of(LLVMGetOperand(in, 1));
        if (p.space != SP_PRIVATE || p.base)
            airc_fail("the cosine of %s goes elsewhere than a variable", name);
        for (unsigned l = 0; l < n; l++) {
            enum ir_type t = ir_type_of(lane_type(rt, l));
            struct ir_val *s, *c, *x = widen(get(LLVMGetOperand(in, 0), l));
            int fast = !strncmp(name, "air.fast_", 9) || !strncmp(name, "air.native_", 11);
            if (fast || t == IR_F16)
                cheap_sincos(x, fast, &s, &c);
            else
                sincos_ir(x, &s, &c, TRIG_PRECISE);
            set(in, l, narrow(s, t));
            private_write(private_var(p, lane_offset(rt, l)), narrow(c, t));
        }
        return;
    }
    if (is_fn(name, "all") || is_fn(name, "any")) {
        LLVMValueRef a = LLVMGetOperand(in, 0);
        struct ir_val *r = get(a, 0);
        for (unsigned l = 1; l < nlanes(LLVMTypeOf(a)); l++)
            r = op2(is_fn(name, "all") ? IR_PAND : IR_POR, IR_PRED, r, get(a, l));
        set(in, 0, r);
        return;
    }
    if (is_fn(name, "dot")) {
        LLVMValueRef a = LLVMGetOperand(in, 0), b = LLVMGetOperand(in, 1);
        struct ir_val *r = NULL;
        enum ir_type t = ir_type_of(rt);
        for (unsigned l = 0; l < nlanes(LLVMTypeOf(a)); l++) {
            // half rounds after every step, as Apple's does
            struct ir_val *x = get(a, l), *y = get(b, l);
            if (t != IR_F16) {
                x = widen(x);
                y = widen(y);
            }
            r = r ? op3(IR_FFMA, t, x, y, r) : op2(IR_FMUL, t, x, y);
        }
        set(in, 0, r);
        return;
    }
    // air.unpack.<fields>.<type> / air.pack...: normalized integer fields of a 32-bit word
    unsigned fields[4], count;
    int snorm, to_srgb;
    if (!strncmp(name, "air.simdgroup_matrix_8x8_multiply_accumulate.", 45) ||
        !strncmp(name, "air.simdgroup_matrix_8x8_init_diag.", 35)) {
        simdgroup_matrix(in, name[25] == 'm');
        return;
    }
    if (!strncmp(name, "air.unpack.unorm.rgb9e5.", 24)) {
        // three 9-bit mantissas sharing a 5-bit exponent: each is m * 2^(e - 15 - 9), the scale built in the
        // exponent field of a float
        struct ir_val *x = get(LLVMGetOperand(in, 0), 0);
        struct ir_val *e = op2(IR_SHR, IR_I32, x, konst(IR_I32, 27));
        struct ir_val *scale = op2(IR_OR, IR_F32, op2(IR_SHL, IR_I32, op2(IR_ADD, IR_I32, e, konst(IR_I32, 103)),
                                                        konst(IR_I32, 23)),
                                   konst(IR_F32, 0));
        for (unsigned l = 0; l < 3; l++) {
            struct ir_val *m = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, x, konst(IR_I32, 9 * l)), konst(IR_I32, 0x1ff));
            set(in, l, op2(IR_FMUL, IR_F32, op1(IR_I2F, IR_F32, m), scale));
        }
        return;
    }
    if (!strncmp(name, "air.unpack.unorm.rg11b10f.", 26)) {
        // unsigned floats of 5 exponent bits and 6 (red, green) or 5 (blue) of mantissa: a half's layout with the
        // sign gone and the mantissa short, so shifted up into a half they widen exactly, denormals and infinity too
        struct ir_val *x = get(LLVMGetOperand(in, 0), 0);
        for (unsigned l = 0, at = 0; l < 3; at += l++ < 2 ? 11 : 10) {
            unsigned lift = l < 2 ? 4 : 5;
            struct ir_val *field = op2(IR_SHR, IR_I32, x, konst(IR_I32, at));
            struct ir_val *half = op2(IR_AND, IR_I32, op2(IR_SHL, IR_I32, field, konst(IR_I32, lift)),
                                      konst(IR_I32, l < 2 ? 0x7ff0 : 0x7fe0));
            set(in, l, widen(op2(IR_OR, IR_F16, half, konst(IR_F16, 0))));
        }
        return;
    }
    if (!strncmp(name, "air.unpack.", 11) && (count = packed_fields(name, fields, &snorm, &to_srgb))) {
        struct ir_val *x = get(LLVMGetOperand(in, 0), 0);
        for (unsigned l = 0, at = 0; l < count && l < n; at += fields[l++]) {
            unsigned bits = fields[l], max = (1u << (bits - snorm)) - 1;
            struct ir_val *f;
            if (snorm) {  // the field to the top, then back down arithmetically: sign-extended
                struct ir_val *top = op2(IR_SHL, IR_I32, x, konst(IR_I32, 32 - at - bits));
                f = op1(IR_I2F, IR_F32, op2(IR_SAR, IR_I32, top, konst(IR_I32, 32 - bits)));
                f->aux = 1;
                f = op2(IR_FMAX, IR_F32, div_exact(f, max), fconst(-1.0f));
            } else if (to_srgb && l < 3) {
                // Apple decodes sRGB to 12 bits: the nearest k / 4095. Byte 80 is 328.50016 / 4095, nearer the
                // midpoint than the curve's accuracy here, so it's set apart.
                struct ir_val *field = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, x, konst(IR_I32, at)),
                                           konst(IR_I32, 0xff));
                f = op2(IR_FMUL, IR_F32, op1(IR_I2F, IR_F32, field), fconst(1.0f / 255));
                f = op1(IR_FRND, IR_F32, op2(IR_FMUL, IR_F32, srgb(f, 0), fconst(4095.0f)));
                f->aux = SASS_RND_NEAREST;
                f = div_exact(sel(icmp(IR_EQ, field, konst(IR_I32, 80)), fconst(329.0f), f), 4095);
            } else {
                struct ir_val *field = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, x, konst(IR_I32, at)),
                                           konst(IR_I32, max));
                f = div_exact(op1(IR_I2F, IR_F32, field), max);
            }
            set(in, l, narrow(f, ir_type_of(lane_type(rt, l))));
        }
        return;
    }
    if (!strncmp(name, "air.pack.", 9) && (count = packed_fields(name, fields, &snorm, &to_srgb))) {
        LLVMValueRef v = LLVMGetOperand(in, 0);
        struct ir_val *word = NULL;
        for (unsigned l = 0, at = 0; l < count; at += fields[l++]) {
            unsigned bits = fields[l], max = (1u << (bits - snorm)) - 1;
            struct ir_val *f = widen(get(v, l));
            if (to_srgb && l < 3)  // the max first: NaN is 0
                f = srgb(op2(IR_FMIN, IR_F32, op2(IR_FMAX, IR_F32, f, fconst(0.0f)), fconst(1.0f)), 1);
            // f * max rounded half away from zero, exactly as Apple does: rounded towards zero, f * max + 0.5 can't
            // reach the next integer when it's just short of it, so truncating it is exact
            struct ir_val *half = snorm ? op2(IR_OR, IR_F32, op2(IR_AND, IR_F32, f, konst(IR_F32, 0x80000000u)),
                                              konst(IR_F32, 0x3f000000))
                                        : fconst(0.5f);
            struct ir_val *t = op3(IR_FFMA, IR_F32, f, fconst((float)max), half);
            t->aux = SASS_RND_TRUNC;
            struct ir_val *i = op1(IR_F2I, IR_I32, t);  // NaN is 0, and below 0 is 0 unsigned
            i->aux = snorm;
            if (snorm)
                i = iminmax(IR_MAX, i, konst(IR_I32, -(int)max), 1);
            i = iminmax(IR_MIN, i, konst(IR_I32, max), snorm);
            struct ir_val *field = op2(IR_SHL, IR_I32, op2(IR_AND, IR_I32, i, konst(IR_I32, (1u << bits) - 1)),
                                       konst(IR_I32, at));
            word = word ? op2(IR_OR, IR_I32, word, field) : field;
        }
        set(in, 0, word);
        return;
    }
    if (!strncmp(name, "air.pack.half2x16.", 18) || !strncmp(name, "air.unpack.half2x16.", 20)) {
        LLVMValueRef v = LLVMGetOperand(in, 0);
        if (name[4] == 'u') {  // each half widened
            for (unsigned l = 0; l < 2; l++) {
                struct ir_val *h = op2(IR_SHR, IR_I32, get(v, 0), konst(IR_I32, 16 * l));
                set(in, l, narrow(op1(IR_H2F, IR_F32, h), ir_type_of(lane_type(rt, l))));
            }
            return;
        }
        struct ir_val *lo = op1(IR_F2H, IR_F16, widen(get(v, 0))), *hi = op1(IR_F2H, IR_F16, widen(get(v, 1)));
        lo = op2(IR_AND, IR_I32, lo, konst(IR_I32, 0xffff));
        set(in, 0, op2(IR_OR, IR_I32, lo, op2(IR_SHL, IR_I32, hi, konst(IR_I32, 16))));
        return;
    }
    // air.convert.<kind>.<type>.<kind>.<type>, lane by lane
    char dk, sk, dbuf[16], sbuf[16];
    if (sscanf(name, "air.convert.%c.%15[^.].%c.%15s", &dk, dbuf, &sk, sbuf) == 4) {
        for (unsigned l = 0; l < n; l++) {
            struct ir_val *s = get(LLVMGetOperand(in, 0), l), *r;
            enum ir_type dt = ir_type_of(lane_type(rt, l));
            if (s->type == IR_PRED)  // a bool is 1 (or as a signed one, -1) or 0
                s = sel(s, konst(IR_I32, sk == 's' ? -1 : 1), konst(IR_I32, 0));
            LLVMValueRef from = LLVMGetOperand(in, 0);
            if (sk == 'f' && dk != 'f' && is_i64(lane_type(rt, l))) {
                set_pair(in, l, float_to_pair(widen(s), dk == 's'));
                continue;
            }
            struct pair64 wide;
            if (dk == 'f' && sk != 'f' && is_i64(lane_type(LLVMTypeOf(from), l)) && pair_at(from, l, &wide)) {
                set(in, l, narrow(pair_to_float(wide, sk == 's', dt == IR_F16), dt));
                continue;
            }
            if (sk != 'f')  // a narrow lane keeps only its low bits
                s = int_extend(s, 8 * scalar_bytes(lane_type(LLVMTypeOf(from), l)), sk == 's');
            if (dk == 'f' && sk == 'f') {
                r = narrow(widen(s), dt);
            } else if (dk == 'f') {  // from an integer
                struct ir_val *i2f = op1(IR_I2F, IR_F32, s);
                i2f->aux = sk == 's';
                r = narrow(i2f, dt);
            } else if (sk == 'f' && dt == IR_PRED) {
                // true unless 0 or a denormal (flushed), NaN included: the exponent isn't 0
                r = op2(IR_AND, IR_I32, widen(s), konst(IR_I32, 0x7f800000));
            } else if (sk == 'f') {
                r = op1(IR_F2I, IR_I32, widen(s));
                r->aux = dk == 's';
                unsigned bits = 8 * scalar_bytes(lane_type(rt, l));
                if (bits < 32) {  // saturated to the narrower type, not its low bits of the 32-bit one
                    int max = dk == 's' ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
                    if (dk == 's')
                        r = iminmax(IR_MAX, r, konst(IR_I32, -max - 1), 1);
                    r = iminmax(IR_MIN, r, konst(IR_I32, max), dk == 's');
                }
            } else {
                r = s;  // integer to integer: 32-bit lanes already
                if (is_i64(lane_type(rt, l))) {
                    struct ir_val *hi = is_i64(lane_type(LLVMTypeOf(from), l)) ? high_word(from, l) : NULL;
                    if (!is_i64(lane_type(LLVMTypeOf(from), l)))
                        hi = sk == 's' ? op2(IR_SAR, IR_I32, r, konst(IR_I32, 31)) : konst(IR_I32, 0);
                    if (hi)
                        set_high(in, l, hi);
                }
            }
            if (dt == IR_PRED) {  // to a bool: anything but 0 is true
                struct ir_val *ne = op2(IR_CMP, IR_PRED, r, konst(r->type, 0));
                ne->aux = IR_NE;
                r = ne;
            }
            set(in, l, r);
        }
        return;
    }
    if (is_i64(lane_type(rt, 0))) {
        unsigned done = 0;
        while (done < n && imath64(name, in, done))
            done++;
        if (done == n)
            return;
        if (n > 1)
            airc_fail("unsupported intrinsic %s", name);
    }
    for (unsigned l = 0; l < n; l++) {
        enum ir_type t = ir_type_of(lane_type(rt, l));
        if (LLVMGetTypeKind(lane_type(rt, l)) == LLVMBFloatTypeKind && (is_fn(name, "fma") || is_fn(name, "fmuladd"))) {
            struct ir_val *x[3];
            for (unsigned k = 0; k < 3; k++)
                x[k] = bfloat_to_float(get(LLVMGetOperand(in, k), l));
            set(in, l, float_to_bfloat(op3(IR_FFMA, IR_F32, x[0], x[1], x[2])));
            continue;
        }
        struct ir_val *r = t == IR_I32 ? imath(name, in, l) : math(name, in, l, t);
        if (!r)
            airc_fail("unsupported intrinsic %s", name);
        set(in, l, r);
    }
}

// An alloca indexed only by constants is a set of variables, one per lane offset. A store sets a variable's value
// in the current block; a load reads the block's own last store, or else the value on entry to the block, which
// waits as an input-less phi at the block's head until the whole function is converted and every block's
// predecessors are known (resolve_private).
static struct entry *g_defs[1 << HASH_BITS], *g_entry_vals[1 << HASH_BITS];
struct private_read {
    struct ir_block *block;
    unsigned var;
    struct ir_val *phi;
    struct private_read *next;
};
static struct private_read *g_private_reads;

// Pointers kept in private memory (MPS packs its buffers into structs on the stack): one per variable, as the one
// stored there, in whichever block. A variable given different pointers in different blocks isn't followed.
struct private_ptr {
    unsigned var;
    struct ptr p;
    struct ir_block *block;
    struct private_ptr *next;
};
static struct private_ptr *g_private_ptrs;

static int same_ptr(struct ptr a, struct ptr b)
{
    return a.space == b.space && a.base == b.base && a.off == b.off && a.slot == b.slot && a.stride == b.stride &&
           a.count == b.count;
}

// the variable a pointer kept in private or local memory goes in: local memory's by its offset
static unsigned pointer_var(struct ptr p)
{
    return p.space == SP_LOCAL ? 0x80000000u | (unsigned)p.off : private_var(p, 0);
}

static struct private_ptr *private_ptr(unsigned var)
{
    struct private_ptr *e = g_private_ptrs;
    while (e && e->var != var)
        e = e->next;
    return e;
}

static unsigned private_var(struct ptr p, unsigned off)
{
    return p.slot << 12 | (unsigned)(p.off + off);
}

static void private_write(unsigned var, struct ir_val *v)
{
    insert(g_defs, g_cur, var, 0)->v = v;
}

// a constant global's lanes as the values its variables hold where the shader starts
static void const_table(struct ptr p, LLVMValueRef init)
{
    unsigned a;
    LLVMTypeRef t = LLVMTypeOf(init);
    if (layout(t, &a) > 0xfff)
        airc_fail("a constant table of more than 4 KB");
    for (unsigned l = 0; l < nlanes(t); l++)
        insert(g_defs, g_prologue, private_var(p, lane_offset(t, l)), 0)->v = const_lane(init, l);
}

static struct ir_val *private_read(unsigned var, enum ir_type t)
{
    struct entry *e = find(g_defs, g_cur, var, 0);
    if (!e)
        e = find(g_entry_vals, g_cur, var, 0);
    if (e)
        return e->v;
    struct ir_val *phi = ir_add(g_cur, IR_PHI, t, 0);
    insert(g_entry_vals, g_cur, var, 0)->v = phi;
    struct private_read *r = ir_alloc(sizeof(*r));
    r->block = g_cur;
    r->var = var;
    r->phi = phi;
    r->next = g_private_reads;
    g_private_reads = r;
    return phi;
}

static struct ir_val *private_entry(struct ir_block *b, unsigned var, enum ir_type t);

// a variable's value where b ends: its last store there, or what it held coming in
static struct ir_val *private_exit(struct ir_block *b, unsigned var, enum ir_type t)
{
    struct entry *e = find(g_defs, b, var, 0);
    return e ? e->v : private_entry(b, var, t);
}

static int is_sub_entry(const struct ir_block *b)
{
    for (int i = 0; i < g_f->nsubs; i++)
        if (g_f->subs[i] == b)
            return 1;
    return 0;
}

// one input per predecessor (a lone one is folded away later); none at all: never stored, undefined, but for a
// constant table read first in a subroutine
static void private_fill(struct ir_block *b, unsigned var, struct ir_val *phi)
{
    if (!b->npreds) {
        struct entry *e = is_sub_entry(b) ? find(g_defs, g_prologue, var, 0) : NULL;
        if (e && e->v->op == IR_CONST) {
            phi->op = IR_CONST;
            phi->imm = e->v->imm;
        } else {
            phi->op = IR_UNDEF;
        }
        return;
    }
    phi->src = ir_alloc(b->npreds * sizeof(*phi->src));
    phi->nsrc = b->npreds;  // before the recursion: a loop comes back to this phi
    for (unsigned k = 0; k < b->npreds; k++)
        phi->src[k] = private_exit(b->preds[k], var, phi->type);
}

static struct ir_val *private_entry(struct ir_block *b, unsigned var, enum ir_type t)
{
    struct entry *e = find(g_entry_vals, b, var, 0);
    struct ir_val *phi = e ? e->v : NULL;
    if (!phi) {
        phi = ir_add(b, IR_PHI, t, 0);
        insert(g_entry_vals, b, var, 0)->v = phi;
    }
    if (phi->op == IR_PHI && !phi->nsrc)
        private_fill(b, var, phi);
    return phi;
}

static void resolve_private(void)
{
    for (struct private_read *r = g_private_reads; r; r = r->next)
        private_entry(r->block, r->var, r->phi->type);
}

// element k's variable at a variable index's i == k
static struct ir_val *index_is(struct ptr p, unsigned k)
{
    struct ir_val *eq = op2(IR_CMP, IR_PRED, p.base, konst(IR_I32, k));
    eq->aux = IR_EQ;
    return eq;
}

static struct ir_val *load_lane(struct ptr p, unsigned off, LLVMTypeRef st)
{
    if (p.space == SP_PRIVATE && p.base) {  // the indexed element out of all of them
        struct ir_val *r = private_read(private_var(p, off), ir_type_of(st));
        for (unsigned k = 1; k < p.count; k++)
            r = sel(index_is(p, k), private_read(private_var(p, off + k * p.stride), ir_type_of(st)), r);
        return r;
    }
    if (p.space == SP_PRIVATE)
        return private_read(private_var(p, off), ir_type_of(st));
    unsigned bytes = scalar_bytes(st);
    if (bytes == 8 && LLVMGetTypeKind(st) == LLVMIntegerTypeKind)  // a 64-bit integer keeps its low word
        bytes = 4;
    if (bytes != 1 && bytes != 2 && bytes != 4)
        airc_fail("unsupported load of %u-byte lanes", bytes);
    struct ir_val *v;
    int flag = ir_type_of(st) == IR_PRED;
    enum ir_type lt = flag ? IR_I32 : ir_type_of(st);
    if (p.space == SP_GLOBAL) {
        v = ir_add(g_cur, IR_LDG, lt, 1);
        v->src[0] = p.base;
    } else if (p.space == SP_SHARED || p.space == SP_LOCAL) {
        struct ir_val *base = p.space == SP_LOCAL ? local_base(p, off + bytes) : p.base;
        v = ir_add(g_cur, p.space == SP_SHARED ? IR_LDS : IR_LDL, lt, base ? 1 : 0);
        if (base)
            v->src[0] = base;
    } else {
        airc_fail("load from a texture or sampler");
    }
    v->imm = p.off + off;
    v->aux = (int)bytes;
    if (flag)  // a bool is a byte in memory, its low bit the value
        return icmp(IR_NE, op2(IR_AND, IR_I32, v, konst(IR_I32, 1)), konst(IR_I32, 0));
    return v;
}

static unsigned low_bit(int64_t x, unsigned cap)
{
    uint64_t b = (uint64_t)x & -(uint64_t)x;
    return !b || b > cap ? cap : (unsigned)b;
}

// What a global pointer's alignment is known to be: a buffer starts 16-byte aligned, and every index or offset on
// the way keeps only what its step does. MPS indexes a float * and stores a float4 through it, declaring 16 bytes
// of alignment Apple's GPUs don't need; a 128-bit STG on an address that isn't faults (Misaligned Address).
static unsigned global_align(struct ptr p)
{
    unsigned a = low_bit(p.off, 16);
    for (struct ir_val *v = p.base; v && a > 1;) {
        if (v->op == IR_ADDR) {
            struct ir_val *i = v->src[1];
            int64_t step = v->imm;
            if (i->op == IR_CONST)
                step *= i->imm;
            else if (i->op == IR_MUL && i->src[1]->op == IR_CONST)
                step *= i->src[1]->imm;
            else if (i->op == IR_SHL && i->src[1]->op == IR_CONST && i->src[1]->imm < 8)
                step <<= i->src[1]->imm;
            if (step)
                a = a < low_bit(step, 16) ? a : low_bit(step, 16);
            v = v->src[0];
        } else if (v->op == IR_ADDI) {
            a = a < low_bit(v->imm, 16) ? a : low_bit(v->imm, 16);
            v = v->src[0];
        } else {
            break;
        }
    }
    return a;
}

// 2 or 4 lanes of 32 bits one after another, which one 64- or 128-bit access moves when aligned to its size
static int wide_lanes(LLVMTypeRef t, unsigned align)
{
    unsigned n = nlanes(t);
    if ((n != 2 && n != 4) || align < 4 * n)
        return 0;
    for (unsigned l = 0; l < n; l++) {
        LLVMTypeRef st = lane_type(t, l);
        if (scalar_bytes(st) != 4 || lane_offset(t, l) != 4 * l || ir_type_of(st) == IR_PRED)
            return 0;
    }
    return 1;
}

// A vector of bytes or halfwords filling 4, 8 or 16 bytes (a uchar4, a ushort2, a half4), which one access of that
// size moves when aligned to it: its size, or 0. A 3-lane one's slot has a lane of padding, which a load may read.
static unsigned packed_bytes(LLVMTypeRef t, unsigned align)
{
    if (LLVMGetTypeKind(t) != LLVMVectorTypeKind)
        return 0;
    LLVMTypeRef et = LLVMGetElementType(t);
    unsigned a, size = layout(t, &a), b = scalar_bytes(et);
    if ((b != 1 && b != 2) || ir_type_of(et) == IR_PRED || (size != 4 && size != 8 && size != 16) || align < size)
        return 0;
    return size;
}

// The 32-bit value lane x is bits at.. of, as a packed load leaves its lanes; NULL if it isn't one. Only x's low
// bits get stored, so a mask on the way doesn't matter.
static struct ir_val *word_of(struct ir_val *x, unsigned at, unsigned bits)
{
    if (x->op == IR_AND && x->src[1]->op == IR_CONST && x->src[1]->imm == (1u << bits) - 1)
        x = x->src[0];
    if (at) {
        if (x->op != IR_SHR || x->src[1]->op != IR_CONST || x->src[1]->imm != at)
            return NULL;
        x = x->src[0];
    }
    return x->type == IR_I32 ? x : NULL;
}

static void load(LLVMValueRef in)
{
    LLVMValueRef addr = LLVMGetOperand(in, 0);
    LLVMTypeRef t = LLVMTypeOf(in);
    LLVMValueRef g = addr;
    while (LLVMIsAConstantExpr(g) && LLVMGetConstOpcode(g) == LLVMBitCast)
        g = LLVMGetOperand(g, 0);
    if (LLVMIsAGlobalVariable(g) && LLVMGetPointerAddressSpace(LLVMTypeOf(g)) == 2) {
        // the static initialiser's globals, and the function constants it copies into them
        const char *name = LLVMGetValueName(g);
        for (unsigned l = 0; l < nlanes(t); l++) {
            struct ir_val *v = global_state(g, l);
            if (!v && name && strstr(name, ".MTL_FC_INIT_"))
                v = function_constant(g, lane_type(t, l));
            if (!v && LLVMGetInitializer(g))  // a constant that's just its initialiser (MPS's implicit attributes)
                v = const_lane(LLVMGetInitializer(g), l);
            if (!v)
                airc_fail("read of constant global %s, which holds nothing known", name ? name : "?");
            set(in, l, v);
        }
        return;
    }
    struct ptr p = fit_offset(ptr_of(addr));
    if (is_ptr(t) && (p.space == SP_PRIVATE || p.space == SP_LOCAL) && !p.base) {
        struct private_ptr *e = private_ptr(pointer_var(p));
        if (!e || !e->block)
            airc_fail("loading a pointer from thread memory that holds no one pointer");
        insert(g_ptrs, in, 0, g_scope)->p = e->p;
        return;
    }
    if (is_ptr(t) && (p.space == SP_TEXTURE || p.space == SP_SAMPLER)) {
        // an element of an array of textures or samplers: the slot that many after the array's first
        if (p.param)
            airc_fail("an array of textures or samplers passed to a subroutine");
        struct ptr q = { .space = p.space, .base = p.base, .slot = p.slot + (unsigned)(p.off / 8) };
        insert(g_ptrs, in, 0, g_scope)->p = q;
        return;
    }
    if (is_ptr(t)) {  // a buffer's address, or a texture's or sampler's resource ID, from an argument buffer
        if (p.space != SP_GLOBAL || ptr_space(t) != SP_GLOBAL)
            airc_fail("loading a pointer from or to threadgroup or thread memory");
        struct ir_val *v = ir_add(g_cur, IR_LDG, IR_I64, 1);
        v->src[0] = p.base;
        v->imm = p.off;
        v->aux = 8;
        struct ptr q = { .space = SP_GLOBAL, .base = v };
        insert(g_ptrs, in, 0, g_scope)->p = q;
        return;
    }
    if (p.space == SP_GLOBAL && wide_lanes(t, LLVMGetAlignment(in)) && wide_lanes(t, global_align(p))) {
        struct ir_val *v = ir_add(g_cur, IR_LDG, IR_VOID, 1);
        v->src[0] = p.base;
        v->imm = p.off;
        v->aux = (int)(4 * nlanes(t));
        for (unsigned l = 0; l < nlanes(t); l++) {
            struct ir_val *lane = op1(IR_PROJ, ir_type_of(lane_type(t, l)), v);
            lane->imm = l;
            set(in, l, lane);
        }
        return;
    }
    unsigned packed = p.space == SP_GLOBAL ? packed_bytes(t, LLVMGetAlignment(in)) : 0;
    if (packed && global_align(p) >= packed) {
        // one load, its lanes shifted out of the words and cleared above their bits, as a U8 or U16 load leaves them
        struct ir_val *v = ir_add(g_cur, IR_LDG, packed > 4 ? IR_VOID : IR_I32, 1), *words[4] = { v };
        v->src[0] = p.base;
        v->imm = p.off;
        v->aux = (int)packed;
        for (unsigned w = 0; packed > 4 && w < packed / 4; w++) {
            words[w] = op1(IR_PROJ, IR_I32, v);
            words[w]->imm = w;
        }
        unsigned bits = 8 * scalar_bytes(lane_type(t, 0));
        enum ir_type lt = ir_type_of(lane_type(t, 0));
        for (unsigned l = 0; l < nlanes(t); l++) {
            unsigned at = l * bits % 32;
            struct ir_val *x = words[l * bits / 32];
            if (at)
                x = op2(IR_SHR, lt, x, konst(IR_I32, at));
            if (at + bits < 32)
                x = op2(IR_AND, lt, x, konst(IR_I32, (1u << bits) - 1));
            set(in, l, x);
        }
        return;
    }
    for (unsigned l = 0; l < nlanes(t); l++) {
        LLVMTypeRef st = lane_type(t, l);
        struct ir_val *v = load_lane(p, lane_offset(t, l), st);
        set(in, l, v);
        if (is_i64(st))
            set_high(in, l, load_lane(p, lane_offset(t, l) + 4, LLVMInt32Type()));
    }
}

static void store_value(struct ptr p, LLVMTypeRef t, LLVMValueRef val, unsigned align)
{
    p = fit_offset(p);
    if (p.space == SP_GLOBAL && wide_lanes(t, align) && wide_lanes(t, global_align(p))) {
        unsigned n = nlanes(t);
        struct ir_val *s = ir_add(g_cur, IR_STG, IR_VOID, 1 + n);
        s->src[0] = p.base;
        for (unsigned l = 0; l < n; l++)
            s->src[1 + l] = get(val, l);
        s->imm = p.off;
        s->aux = (int)(4 * n);
        return;
    }
    unsigned packed = p.space == SP_GLOBAL ? packed_bytes(t, align) : 0;
    unsigned bits = packed ? 8 * scalar_bytes(lane_type(t, 0)) : 0;
    if (packed && nlanes(t) * bits == 8 * packed && global_align(p) >= packed) {
        // the lanes packed into words and stored in one go; a 3-lane vector isn't, as that would write its padding.
        // Masked, the lanes share no bit, so x * 2^at + word packs them as an IMAD each.
        struct ir_val *words[4] = { NULL }, *whole[4] = { NULL };
        for (unsigned l = 0; l < nlanes(t); l++) {
            unsigned at = l * bits % 32;
            struct ir_val *x = get(val, l), **w = &words[l * bits / 32], *from = word_of(x, at, bits);
            whole[l * bits / 32] = !at || whole[l * bits / 32] == from ? from : NULL;
            if (at + bits < 32)
                x = op2(IR_AND, IR_I32, x, konst(IR_I32, (1u << bits) - 1));
            *w = !*w ? x : op2(IR_ADD, IR_I32, op2(IR_MUL, IR_I32, x, konst(IR_I32, 1u << at)), *w);
        }
        for (unsigned w = 0; w < packed / 4; w++)  // the lanes of a word loaded the same way: the word itself
            if (whole[w])
                words[w] = whole[w];
        struct ir_val *s = ir_add(g_cur, IR_STG, IR_VOID, 1 + packed / 4);
        s->src[0] = p.base;
        for (unsigned w = 0; w < packed / 4; w++)
            s->src[1 + w] = words[w];
        s->imm = p.off;
        s->aux = (int)packed;
        return;
    }
    for (unsigned l = 0; l < nlanes(t); l++) {
        LLVMTypeRef st = lane_type(t, l);
        unsigned bytes = scalar_bytes(st);
        if (p.space == SP_PRIVATE && p.base) {  // every element keeps its value but the indexed one
            for (unsigned k = 0; k < p.count; k++) {
                unsigned var = private_var(p, lane_offset(t, l) + k * p.stride);
                struct ir_val *old = private_read(var, ir_type_of(st));
                private_write(var, sel(index_is(p, k), get(val, l), old));
            }
            continue;
        }
        if (bytes == 8 && LLVMGetTypeKind(st) == LLVMIntegerTypeKind) {
            // a 64-bit integer, which values keep the low word of: the high one where it's known, else 0
            struct ir_val *words[2] = { get(val, l), high_or_zero(val, l) };
            for (unsigned w = 0; w < 2; w++) {
                unsigned at = lane_offset(t, l) + 4 * w;
                if (p.space == SP_PRIVATE && !p.base) {
                    private_write(private_var(p, at), words[w]);
                    continue;
                }
                if (p.space == SP_PRIVATE)
                    airc_fail("a 64-bit integer stored into a private element picked at run time");
                struct ir_val *base = p.space == SP_LOCAL ? local_base(p, at + 4) : p.base;
                struct ir_val *s = p.space == SP_GLOBAL ? op2(IR_STG, IR_VOID, p.base, words[w])
                                                        : ir_add(g_cur, p.space == SP_SHARED ? IR_STS : IR_STL, IR_VOID, 2);
                if (p.space != SP_GLOBAL) {
                    s->src[0] = base;
                    s->src[1] = words[w];
                }
                s->imm = p.off + at;
                s->aux = 4;
            }
            continue;
        }
        if (p.space == SP_PRIVATE) {
            private_write(private_var(p, lane_offset(t, l)), get(val, l));
            continue;
        }
        if (bytes != 1 && bytes != 2 && bytes != 4)
            airc_fail("unsupported store of %u-byte lanes", bytes);
        struct ir_val *s, *data = get(val, l);
        if (data->type == IR_PRED)
            data = sel(data, konst(IR_I32, 1), konst(IR_I32, 0));
        if (p.space == SP_GLOBAL) {
            s = op2(IR_STG, IR_VOID, p.base, data);
        } else if (p.space == SP_SHARED || p.space == SP_LOCAL) {
            struct ir_val *base = p.space == SP_LOCAL ? local_base(p, lane_offset(t, l) + bytes) : p.base;
            s = ir_add(g_cur, p.space == SP_SHARED ? IR_STS : IR_STL, IR_VOID, 2);
            s->src[0] = base;  // may be NULL: address 0 plus the offset
            s->src[1] = data;
        } else {
            airc_fail("store to a texture or sampler");
        }
        s->imm = p.off + lane_offset(t, l);
        s->aux = (int)bytes;
    }
}

static void ptr_words(struct ptr p, struct ir_val **lo, struct ir_val **hi);

static void store(LLVMValueRef in)
{
    LLVMValueRef val = LLVMGetOperand(in, 0), addr = LLVMGetOperand(in, 1);
    LLVMTypeRef t = LLVMTypeOf(val);
    LLVMValueRef g = addr;
    while (LLVMIsAConstantExpr(g) && LLVMGetConstOpcode(g) == LLVMBitCast)
        g = LLVMGetOperand(g, 0);
    if (LLVMIsAGlobalVariable(g) && LLVMGetPointerAddressSpace(LLVMTypeOf(g)) == 2) {
        for (unsigned l = 0; l < nlanes(t); l++)  // the static initialiser filling a global
            insert(g_vals, g, l, 0)->v = get(val, l);
        return;
    }
    struct ptr p = ptr_of(addr);
    if (is_ptr(t) && (p.space == SP_PRIVATE || p.space == SP_LOCAL)) {
        if (p.base)  // into an element picked at run time: only a load that never comes gets through
            return;
        unsigned var = pointer_var(p);
        struct ptr q = LLVMIsUndef(val) || LLVMIsPoison(val) ? (struct ptr){ 0 } : ptr_of(val);
        struct private_ptr *e = private_ptr(var);
        if (!e) {
            e = ir_alloc(sizeof(*e));
            e->var = var;
            e->p = q;
            e->block = g_cur;
            e->next = g_private_ptrs;
            g_private_ptrs = e;
        } else if (e->block == g_cur) {
            e->p = q;
        } else if (e->block && !same_ptr(e->p, q)) {
            e->block = NULL;  // a different one on another path: not followed
        }
        return;
    }
    if (is_ptr(t) && p.space == SP_GLOBAL && ptr_space(t) == SP_GLOBAL) {  // an address into device memory, as its words
        struct ir_val *lo, *hi;
        ptr_words(ptr_of(val), &lo, &hi);
        p = fit_offset(p);
        struct ir_val *s = ir_add(g_cur, IR_STG, IR_VOID, 3);
        s->src[0] = p.base;
        s->src[1] = lo;
        s->src[2] = hi;
        s->imm = p.off;
        s->aux = 8;
        return;
    }
    // a wider integer loaded and stored again, as memcpy does it: its words copied, as values only keep the low one
    LLVMTypeRef lane0 = nlanes(t) ? lane_type(t, 0) : t;
    if (LLVMGetTypeKind(lane0) == LLVMIntegerTypeKind && LLVMGetIntTypeWidth(lane0) > 32 && LLVMIsAInstruction(val) &&
        LLVMGetInstructionOpcode(val) == LLVMLoad) {
        struct ptr from = ptr_of(LLVMGetOperand(val, 0));
        for (unsigned w = 0; w < nlanes(t) * LLVMGetIntTypeWidth(lane0) / 32; w++) {
            struct ir_val *word = load_lane(fit_offset(from), 4 * w, LLVMInt32Type());
            if (p.space == SP_PRIVATE && !p.base) {
                private_write(private_var(p, 4 * w), word);
                continue;
            }
            struct ptr q = fit_offset(p);
            struct ir_val *base = q.space == SP_LOCAL ? local_base(q, 4 * w + 4) : q.base;
            struct ir_val *s = q.space == SP_GLOBAL ? op2(IR_STG, IR_VOID, q.base, word)
                                                    : ir_add(g_cur, q.space == SP_SHARED ? IR_STS : IR_STL, IR_VOID, 2);
            if (q.space != SP_GLOBAL) {
                s->src[0] = base;
                s->src[1] = word;
            }
            s->imm = q.off + 4 * w;
            s->aux = 4;
        }
        return;
    }
    store_value(p, t, val, LLVMGetAlignment(in));
}

// the length of a memcpy or memset, which local memory's 4 KB a thread bounds well enough to write out
static unsigned mem_length(LLVMValueRef in)
{
    LLVMValueRef len = LLVMGetOperand(in, 2);
    if (!LLVMIsAConstantInt(len))
        airc_fail("a memcpy or memset of a length known only at run time");
    unsigned long long n = LLVMConstIntGetZExtValue(len);
    if (n > 4096)
        airc_fail("a memcpy or memset of %llu bytes", n);
    return (unsigned)n;
}

static void mem_store(struct ptr p, unsigned off, struct ir_val *data, unsigned bytes)
{
    struct ir_val *s;
    if (p.space == SP_GLOBAL) {
        s = op2(IR_STG, IR_VOID, p.base, data);
    } else if (p.space == SP_SHARED || p.space == SP_LOCAL) {
        struct ir_val *base = p.space == SP_LOCAL ? local_base(p, off + bytes) : p.base;
        s = ir_add(g_cur, p.space == SP_SHARED ? IR_STS : IR_STL, IR_VOID, 2);
        s->src[0] = base;
        s->src[1] = data;
    } else {
        airc_fail("a memcpy or memset into anything but memory");
    }
    s->imm = p.off + off;
    s->aux = (int)bytes;
}

// words where both ends sit on one, read before any is written for a memmove's overlap
static void mem_copy(LLVMValueRef in)
{
    unsigned n = mem_length(in);
    struct ptr to = fit_offset(ptr_of(LLVMGetOperand(in, 0))), from = fit_offset(ptr_of(LLVMGetOperand(in, 1)));
    struct ir_val **v = ir_alloc((n + 1) * sizeof(*v));
    int words = !((to.off | from.off) & 3);
    for (unsigned at = 0, i = 0; at < n; i++) {
        unsigned step = words && n - at >= 4 ? 4 : 1;
        v[i] = load_lane(from, at, step == 4 ? LLVMInt32Type() : LLVMInt8Type());
        at += step;
    }
    for (unsigned at = 0, i = 0; at < n; i++) {
        unsigned step = words && n - at >= 4 ? 4 : 1;
        mem_store(to, at, v[i], step);
        at += step;
    }
}

static void mem_set(LLVMValueRef in)
{
    unsigned n = mem_length(in);
    struct ptr to = fit_offset(ptr_of(LLVMGetOperand(in, 0)));
    struct ir_val *byte = op2(IR_AND, IR_I32, get(LLVMGetOperand(in, 1), 0), konst(IR_I32, 0xff));
    struct ir_val *word = op2(IR_MUL, IR_I32, byte, konst(IR_I32, 0x01010101));
    for (unsigned at = 0, step; at < n; at += step) {
        step = !((to.off + at) & 3) && n - at >= 4 ? 4 : 1;
        mem_store(to, at, step == 4 ? word : byte, step);
    }
}

static const uint8_t g_icond[][2] = {
    // LLVMIntPredicate from LLVMIntEQ: condition, signed
    { IR_EQ, 0 }, { IR_NE, 0 }, { IR_GT, 0 }, { IR_GE, 0 }, { IR_LT, 0 },
    { IR_LE, 0 }, { IR_GT, 1 }, { IR_GE, 1 }, { IR_LT, 1 }, { IR_LE, 1 },
};

// a 64-bit bitwise operation, sum, product or shift of lane `lane`, with both words where the high ones it needs
// are known (MPS takes a struct of two floats apart with lshr i64 32); 0 if not
static int binary64(LLVMValueRef in, LLVMOpcode op, unsigned lane)
{
    LLVMValueRef yv = LLVMGetOperand(in, 1);
    struct pair64 x, y, r;
    int xk = pair_at(LLVMGetOperand(in, 0), lane, &x), yk = pair_at(yv, lane, &y);
    if (op == LLVMAnd || op == LLVMOr || op == LLVMXor) {
        if (!xk || !yk)
            return 0;
        enum ir_op ir = op == LLVMAnd ? IR_AND : op == LLVMOr ? IR_OR : IR_XOR;
        r = (struct pair64){ op2(ir, IR_I32, x.lo, y.lo), op2(ir, IR_I32, x.hi, y.hi) };
    } else if (op == LLVMAdd || op == LLVMSub || op == LLVMMul) {
        if (!xk || !yk)
            return 0;
        r = op == LLVMAdd ? add64(x, y) : op == LLVMSub ? sub64(x, y) : mul64(x, y);
    } else if (op == LLVMShl || op == LLVMLShr || op == LLVMAShr) {
        // by a count in the low word's low 6 bits; a constant one, left past 32, doesn't need the high word
        struct ir_val *amount = y.lo;
        int constant = amount->op == IR_CONST;
        unsigned c = constant ? (unsigned)amount->imm & 63 : 0;
        if (!xk && !(constant && op == LLVMShl && c >= 32))
            return 0;
        if (!xk)
            x.hi = word(0);
        r = constant ? shift64_const(op, x, c) : shift64(op, x, amount);
    } else {
        return 0;
    }
    set_pair(in, lane, r);
    return 1;
}

// a - b or a + b as one fused multiply-add when an operand is a product marked for contraction (the multiply stays
// for any other use; dead code goes otherwise); NULL when neither is
static struct ir_val *fuse_multiply(LLVMValueRef in, int sub, enum ir_type t, struct ir_val *a, struct ir_val *b)
{
    LLVMValueRef x = LLVMGetOperand(in, 0), y = LLVMGetOperand(in, 1);
    int a_mul = a->op == IR_FMUL && LLVMIsAInstruction(x) && LLVMGetInstructionOpcode(x) == LLVMFMul && can_contract(x);
    int b_mul = b->op == IR_FMUL && LLVMIsAInstruction(y) && LLVMGetInstructionOpcode(y) == LLVMFMul && can_contract(y);
    if (a_mul)
        return op3(IR_FFMA, t, a->src[0], a->src[1], sub ? op1(IR_FNEG, t, b) : b);
    if (b_mul)
        return op3(IR_FFMA, t, sub ? op1(IR_FNEG, t, b->src[0]) : b->src[0], b->src[1], a);
    return NULL;
}

static void binary(LLVMValueRef in, LLVMOpcode op, unsigned n)
{
    static const struct {
        LLVMOpcode llvm;
        enum ir_op ir;
    } map[] = {
        { LLVMAdd, IR_ADD },   { LLVMSub, IR_SUB },   { LLVMMul, IR_MUL },   { LLVMAnd, IR_AND },
        { LLVMOr, IR_OR },     { LLVMXor, IR_XOR },   { LLVMShl, IR_SHL },   { LLVMLShr, IR_SHR },
        { LLVMAShr, IR_SAR },  { LLVMFAdd, IR_FADD }, { LLVMFSub, IR_FSUB }, { LLVMFMul, IR_FMUL },
    };
    enum ir_op ir = IR_ADD;
    for (unsigned i = 0; i < sizeof map / sizeof map[0]; i++)
        if (map[i].llvm == op)
            ir = map[i].ir;
    for (unsigned l = 0; l < n; l++) {
        if (is_i64(lane_type(LLVMTypeOf(in), l)) && binary64(in, op, l))
            continue;
        struct ir_val *a = get(LLVMGetOperand(in, 0), l), *b = get(LLVMGetOperand(in, 1), l);
        enum ir_type t = a->type;
        if (ir >= IR_FADD && ir <= IR_FMUL && LLVMGetTypeKind(lane_type(LLVMTypeOf(in), l)) == LLVMBFloatTypeKind) {
            // widened, computed as floats and rounded back to bfloat, as its conversions are
            set(in, l, float_to_bfloat(op2(ir, IR_F32, bfloat_to_float(a), bfloat_to_float(b))));
            continue;
        }
        if (t == IR_PRED) {  // i1 logic
            if (op == LLVMAnd || op == LLVMOr) {
                set(in, l, op2(op == LLVMAnd ? IR_PAND : IR_POR, IR_PRED, a, b));
                continue;
            }
            if (op == LLVMXor) {  // xor with true is not; in general a != b
                struct ir_val *c = op2(IR_CMP, IR_PRED, sel(a, konst(IR_I32, 1), konst(IR_I32, 0)),
                                       sel(b, konst(IR_I32, 1), konst(IR_I32, 0)));
                c->aux = IR_NE;
                set(in, l, c);
                continue;
            }
            airc_fail("unsupported i1 arithmetic %d", op);
        }
        if (op == LLVMAShr || op == LLVMLShr)  // a narrow lane keeps only its low bits; the shift needs the rest
            a = int_extend(a, 8 * scalar_bytes(lane_type(LLVMTypeOf(in), l)), op == LLVMAShr);
        // Metal's fast-math sums of products are fused, as Apple's GPUs do: x * y + c as one rounding
        if ((op == LLVMFAdd || op == LLVMFSub) && (t == IR_F32 || t == IR_F16) && can_contract(in)) {
            struct ir_val *fused = fuse_multiply(in, op == LLVMFSub, t, a, b);
            if (fused) {
                set(in, l, fused);
                continue;
            }
        }
        set(in, l, op2(ir, t, a, b));
    }
}

static void fdiv(LLVMValueRef in, unsigned n)
{
    for (unsigned l = 0; l < n; l++) {
        struct ir_val *a = get(LLVMGetOperand(in, 0), l), *b = get(LLVMGetOperand(in, 1), l);
        enum ir_type t = a->type;
        if (t == IR_F32 && b->op == IR_CONST) {
            // a constant divisor's reciprocal is rounded once, where MUFU.RCP's error can tip x / 62 off an exact
            // half
            float d, r;
            uint32_t bits = (uint32_t)b->imm;
            memcpy(&d, &bits, 4);
            r = 1.0f / d;
            memcpy(&bits, &r, 4);
            if ((bits >> 23 & 0xff) - 1 < 254) {
                set(in, l, op2(IR_FMUL, IR_F32, a, fconst(r)));
                continue;
            }
        }
        // a / fast::sqrt(x): one MUFU.RSQ instead of MUFU.SQRT and MUFU.RCP after it, its error no larger
        if (t == IR_F32 && b->op == IR_MUFU && b->aux == MUFU_SQRT && can_contract(in)) {
            set(in, l, op2(IR_FMUL, IR_F32, a, mufu(MUFU_RSQ, b->src[0])));
            continue;
        }
        set(in, l, narrow(op2(IR_FMUL, IR_F32, widen(a), mufu(MUFU_RCP, widen(b))), t));
    }
}

// n / d (or n % d), unsigned, as NIR lowers it: the divisor's float reciprocal scaled to 32 bits and refined with a
// multiply-high, a quotient estimate from it, then two corrections
static struct ir_val *udivmod(struct ir_val *n, struct ir_val *d, int rem)
{
    struct ir_val *rcp = op1(IR_F2I, IR_I32,
                             op2(IR_FMUL, IR_F32, mufu(MUFU_RCP, op1(IR_I2F, IR_F32, d)), fconst(4294966784.0f)));
    struct ir_val *neg = op2(IR_MUL, IR_I32, rcp, op2(IR_SUB, IR_I32, konst(IR_I32, 0), d));
    rcp = op2(IR_ADD, IR_I32, rcp, op2(IR_MULHI, IR_I32, rcp, neg));
    struct ir_val *q = op2(IR_MULHI, IR_I32, n, rcp);
    struct ir_val *r = op2(IR_SUB, IR_I32, n, op2(IR_MUL, IR_I32, q, d));
    for (int i = 0; i < 2; i++) {
        struct ir_val *over = op2(IR_CMP, IR_PRED, r, d);
        over->aux = IR_GE;
        q = sel(over, op2(IR_ADD, IR_I32, q, konst(IR_I32, 1)), q);
        r = sel(over, op2(IR_SUB, IR_I32, r, d), r);
    }
    if (rem)
        return r;
    // as Apple's GPUs divide by 0: all ones, but 0 for 0 / 0 (the estimate above would make it n + 1)
    struct ir_val *zero = konst(IR_I32, 0);
    struct ir_val *ones = sel(icmp(IR_NE, n, zero), konst(IR_I32, -1), zero);
    return sel(icmp(IR_EQ, d, zero), ones, q);
}

// n / d or n % d for a power of two d over 1 by shifts and masks; a signed one rounds towards zero, d - 1 added to a
// negative n first
static struct ir_val *divmod_pow2(struct ir_val *n, uint32_t d, int is_signed, int rem)
{
    int k = __builtin_ctz(d);
    if (!is_signed)
        return rem ? ibin(IR_AND, n, ik(d - 1)) : ibin(IR_SHR, n, ik(k));
    struct ir_val *biased = ibin(IR_ADD, n, ibin(IR_SHR, ibin(IR_SAR, n, ik(31)), ik(32 - k)));
    return rem ? ibin(IR_SUB, n, ibin(IR_AND, biased, ik(~(d - 1)))) : ibin(IR_SAR, biased, ik(k));
}

// LLVM's udiv, sdiv, urem and srem. Signed ones divide the magnitudes: the quotient is negated when the signs
// differ, the remainder takes the dividend's sign. Narrower integers are widened first (their high bits are junk).
static void idiv(LLVMValueRef in, LLVMOpcode op, unsigned n)
{
    int is_signed = op == LLVMSDiv || op == LLVMSRem, rem = op == LLVMURem || op == LLVMSRem;
    for (unsigned l = 0; l < n; l++) {
        struct ir_val *a = get(LLVMGetOperand(in, 0), l), *b = get(LLVMGetOperand(in, 1), l);
        unsigned bits = 8 * scalar_bytes(lane_type(LLVMTypeOf(in), l));
        if (bits == 64) {
            struct pair64 x, y;
            if (!pair_at(LLVMGetOperand(in, 0), l, &x) || !pair_at(LLVMGetOperand(in, 1), l, &y))
                airc_fail("64-bit integer division of a value whose high word isn't known");
            set_pair(in, l, idiv64(x, y, is_signed, rem));
            continue;
        }
        if (bits > 32)
            airc_fail("%u-bit integer division", bits);
        uint32_t d = (uint32_t)b->imm;
        if (bits == 32 && b->op == IR_CONST && d > 1 && !(d & (d - 1)) && (!is_signed || (int32_t)d > 0)) {
            set(in, l, divmod_pow2(a, d, is_signed, rem));
            continue;
        }
        a = int_extend(a, bits, is_signed);
        b = int_extend(b, bits, is_signed);
        if (!is_signed) {
            set(in, l, udivmod(a, b, rem));
            continue;
        }
        struct ir_val *zero = konst(IR_I32, 0);
        struct ir_val *an = op2(IR_CMP, IR_PRED, a, zero), *bn = op2(IR_CMP, IR_PRED, b, zero);
        an->aux = bn->aux = IR_LT | IR_SIGNED;
        struct ir_val *r = udivmod(sel(an, op2(IR_SUB, IR_I32, zero, a), a), sel(bn, op2(IR_SUB, IR_I32, zero, b), b),
                                   rem);
        struct ir_val *flip = rem ? an : op2(IR_CMP, IR_PRED, op2(IR_XOR, IR_I32, a, b), zero);
        if (!rem)
            flip->aux = IR_LT | IR_SIGNED;
        set(in, l, sel(flip, op2(IR_SUB, IR_I32, zero, r), r));
    }
}

// a bitcast between vectors of different lane counts (<4 x half> to <2 x i32>, <4 x i8> to i32...): the source
// lanes packed into 32-bit words, the result's lanes sliced out of them
static void bitcast_lanes(LLVMValueRef in)
{
    LLVMValueRef src = LLVMGetOperand(in, 0);
    LLVMTypeRef st = LLVMTypeOf(src), dt = LLVMTypeOf(in);
    struct ir_val *words[8] = { 0 };
    unsigned bit = 0;
    for (unsigned l = 0; l < nlanes(st); l++) {
        LLVMTypeRef lt = lane_type(st, l);
        unsigned w = 8 * scalar_bytes(lt);
        int wide = w == 64 && LLVMGetTypeKind(lt) == LLVMIntegerTypeKind;  // a pair of words: the low one, the high one
        if ((w > 32 && !wide) || (bit % 32) + w > 32 + 32 * wide || bit / 32 + wide >= 8 || (wide && bit % 32))
            airc_fail("bitcast of %u-bit lanes", w);
        struct ir_val *v = op2(IR_OR, IR_I32, get(src, l), konst(IR_I32, 0));
        if (wide) {
            words[bit / 32] = v;
            words[bit / 32 + 1] = high_or_zero(src, l);
            bit += 64;
            continue;
        }
        if (w < 32)
            v = op2(IR_SHL, IR_I32, op2(IR_AND, IR_I32, v, konst(IR_I32, (1u << w) - 1)), konst(IR_I32, bit % 32));
        words[bit / 32] = words[bit / 32] ? op2(IR_OR, IR_I32, words[bit / 32], v) : v;
        bit += w;
    }
    bit = 0;
    for (unsigned l = 0; l < nlanes(dt); l++) {
        LLVMTypeRef lt = lane_type(dt, l);
        unsigned w = 8 * scalar_bytes(lt);
        int wide = w == 64 && LLVMGetTypeKind(lt) == LLVMIntegerTypeKind;
        if ((w > 32 && !wide) || !words[bit / 32] || (wide && (bit % 32 || !words[bit / 32 + 1])))
            airc_fail("bitcast to %u-bit lanes", w);
        struct ir_val *v = words[bit / 32];
        if (wide) {
            set(in, l, v);
            set_high(in, l, words[bit / 32 + 1]);
            bit += 64;
            continue;
        }
        if (w < 32)
            v = op2(IR_AND, IR_I32, op2(IR_SHR, IR_I32, v, konst(IR_I32, bit % 32)), konst(IR_I32, (1u << w) - 1));
        enum ir_type t = ir_type_of(lt);
        set(in, l, t == IR_I32 ? v : op2(IR_OR, t, v, konst(t, 0)));
        bit += w;
    }
}

// a bfloat is the top 16 bits of a float
static struct ir_val *bfloat_to_float(struct ir_val *x)
{
    return op2(IR_OR, IR_F32, op2(IR_SHL, IR_I32, x, konst(IR_I32, 16)), konst(IR_F32, 0));
}

// rounded to nearest even, a NaN kept a (quiet) NaN, and a value past the largest bfloat rounding up to inf
static struct ir_val *float_to_bfloat(struct ir_val *x)
{
    struct ir_val *bits = op2(IR_OR, IR_I32, x, konst(IR_I32, 0));
    struct ir_val *high = op2(IR_SHR, IR_I32, bits, konst(IR_I32, 16));
    struct ir_val *odd = op2(IR_AND, IR_I32, high, konst(IR_I32, 1));
    struct ir_val *rounded = op2(IR_SHR, IR_I32, op2(IR_ADD, IR_I32, op2(IR_ADD, IR_I32, bits, konst(IR_I32, 0x7fff)), odd),
                                 konst(IR_I32, 16));
    struct ir_val *nan = icmp(IR_GT, op2(IR_AND, IR_I32, bits, konst(IR_I32, 0x7fffffff)), konst(IR_I32, 0x7f800000));
    return sel(nan, op2(IR_OR, IR_I32, high, konst(IR_I32, 0x40)), rounded);
}

static void cast(LLVMValueRef in, LLVMOpcode op, unsigned n)
{
    LLVMTypeRef dt = LLVMTypeOf(in), st = LLVMTypeOf(LLVMGetOperand(in, 0));
    if ((op == LLVMFPToSI || op == LLVMFPToUI) && is_i64(lane_type(dt, 0))) {
        for (unsigned l = 0; l < n; l++)
            set_pair(in, l, float_to_pair(widen(get(LLVMGetOperand(in, 0), l)), op == LLVMFPToSI));
        return;
    }
    if ((op == LLVMSIToFP || op == LLVMUIToFP) && is_i64(lane_type(st, 0))) {
        int half = LLVMGetTypeKind(lane_type(dt, 0)) == LLVMHalfTypeKind, done = 1;
        for (unsigned l = 0; l < n; l++) {
            struct pair64 x;
            if (!pair_at(LLVMGetOperand(in, 0), l, &x)) {
                done = 0;
                break;
            }
            set(in, l, narrow(pair_to_float(x, op == LLVMSIToFP, half), ir_type_of(lane_type(dt, l))));
        }
        if (done)
            return;
    }
    if (op == LLVMBitCast && !is_ptr(LLVMTypeOf(in)) && nlanes(LLVMTypeOf(LLVMGetOperand(in, 0))) != n) {
        bitcast_lanes(in);
        return;
    }
    for (unsigned l = 0; l < n; l++) {
        struct ir_val *s = get(LLVMGetOperand(in, 0), l), *r;
        enum ir_type t = ir_type_of(lane_type(dt, l));
        switch (op) {
        case LLVMZExt:
        case LLVMSExt:
            if (s->type == IR_PRED)  // true is 1, or -1 sign-extended
                r = sel(s, konst(IR_I32, op == LLVMSExt ? -1 : 1), konst(IR_I32, 0));
            else if (op == LLVMSExt && scalar_bytes(lane_type(LLVMTypeOf(LLVMGetOperand(in, 0)), l)) < 4) {
                int shift = 32 - 8 * (int)scalar_bytes(lane_type(LLVMTypeOf(LLVMGetOperand(in, 0)), l));
                r = op2(IR_SAR, IR_I32, op2(IR_SHL, IR_I32, s, konst(IR_I32, shift)), konst(IR_I32, shift));
            } else if (op == LLVMZExt && scalar_bytes(lane_type(LLVMTypeOf(LLVMGetOperand(in, 0)), l)) < 4) {
                unsigned bits = 8 * scalar_bytes(lane_type(LLVMTypeOf(LLVMGetOperand(in, 0)), l));
                r = op2(IR_AND, IR_I32, s, konst(IR_I32, (1u << bits) - 1));
            } else {
                r = s;
            }
            if (is_i64(lane_type(dt, l)))
                set_high(in, l, op == LLVMSExt ? op2(IR_SAR, IR_I32, r, konst(IR_I32, 31)) : konst(IR_I32, 0));
            break;
        case LLVMTrunc:
            if (t == IR_PRED) {
                r = op2(IR_CMP, IR_PRED, op2(IR_AND, IR_I32, s, konst(IR_I32, 1)), konst(IR_I32, 0));
                r->aux = IR_NE;
            } else {
                r = s;  // narrower integers keep their low bits; users mask what they need
            }
            break;
        case LLVMBitCast:  // the same bits; a change of type keeps a value of its own so widen() sees halves
            r = s->type == t ? s : op2(IR_OR, t, s, konst(t, 0));
            break;
        case LLVMFPExt:
            r = LLVMGetTypeKind(lane_type(st, l)) == LLVMBFloatTypeKind ? bfloat_to_float(s) : widen(s);
            break;
        case LLVMFPTrunc:
            r = LLVMGetTypeKind(lane_type(dt, l)) == LLVMBFloatTypeKind ? float_to_bfloat(s) : narrow(s, t);
            break;
        case LLVMSIToFP:
        case LLVMUIToFP:
            // a narrow lane keeps only its low bits
            s = int_extend(s, 8 * scalar_bytes(lane_type(LLVMTypeOf(LLVMGetOperand(in, 0)), l)), op == LLVMSIToFP);
            r = op1(IR_I2F, IR_F32, s->type == IR_PRED ? sel(s, konst(IR_I32, 1), konst(IR_I32, 0)) : s);
            r->aux = op == LLVMSIToFP;
            r = narrow(r, t);
            break;
        case LLVMFPToSI:
        case LLVMFPToUI:
            r = op1(IR_F2I, IR_I32, widen(s));
            r->aux = op == LLVMFPToSI;
            break;
        default:
            airc_fail("unsupported cast %d", op);
        }
        set(in, l, r);
    }
}

// the words of a pointer's value (ptr_value): a global one's two, a shared or local one's offset and 0
static void ptr_words(struct ptr p, struct ir_val **lo, struct ir_val **hi)
{
    struct ir_val *v = ptr_value(p);
    if (v->type != IR_I64) {
        *lo = v;
        *hi = konst(IR_I32, 0);
    } else if (v->op == IR_CONST) {
        *lo = konst(IR_I32, (uint32_t)v->imm);
        *hi = konst(IR_I32, (uint32_t)((uint64_t)v->imm >> 32));
    } else {
        *lo = op1(IR_PROJ, IR_I32, v);
        (*lo)->imm = 0;
        *hi = op1(IR_PROJ, IR_I32, v);
        (*hi)->imm = 1;
    }
}

// two pointers compared as the unsigned addresses they are
static struct ir_val *compare_pointers(LLVMValueRef in)
{
    struct ptr a = ptr_of(LLVMGetOperand(in, 0)), b = ptr_of(LLVMGetOperand(in, 1));
    if ((a.space != SP_GLOBAL && a.space != SP_SHARED && a.space != SP_LOCAL) || a.space != b.space)
        airc_fail("a comparison of pointers into texture, sampler, thread or mixed memory");
    struct ir_val *alo, *ahi, *blo, *bhi;
    ptr_words(a, &alo, &ahi);
    ptr_words(b, &blo, &bhi);
    unsigned cond = g_icond[LLVMGetICmpPredicate(in) - LLVMIntEQ][0];
    struct ir_val *lo = icmp(cond, alo, blo), *hi_eq = icmp(IR_EQ, ahi, bhi);
    if (cond == IR_EQ)
        return op2(IR_PAND, IR_PRED, lo, hi_eq);
    if (cond == IR_NE)
        return op2(IR_POR, IR_PRED, lo, op1(IR_PNOT, IR_PRED, hi_eq));
    // the high words decide unless they're equal, where the low ones do
    struct ir_val *hi = icmp(cond == IR_LE ? IR_LT : cond == IR_GE ? IR_GT : cond, ahi, bhi);
    return op2(IR_POR, IR_PRED, hi, op2(IR_PAND, IR_PRED, hi_eq, lo));
}

// a 64-bit integer comparison of lane `lane` where both high words are known: the high words decide unless they're
// equal, and then the low ones do, as unsigned whatever the predicate
static int compare64(LLVMValueRef in, unsigned lane)
{
    struct pair64 x, y;
    if (!pair_at(LLVMGetOperand(in, 0), lane, &x) || !pair_at(LLVMGetOperand(in, 1), lane, &y))
        return 0;
    unsigned p = LLVMGetICmpPredicate(in) - LLVMIntEQ;
    int cond = g_icond[p][0], is_signed = g_icond[p][1];
    struct ir_val *c;
    if (cond == IR_EQ) {
        c = op2(IR_PAND, IR_PRED, icmp(IR_EQ, x.hi, y.hi), icmp(IR_EQ, x.lo, y.lo));
    } else if (cond == IR_NE) {
        c = op2(IR_POR, IR_PRED, icmp(IR_NE, x.hi, y.hi), icmp(IR_NE, x.lo, y.lo));
    } else {
        struct ir_val *hi_eq = icmp(IR_EQ, x.hi, y.hi), *lo = icmp(cond, x.lo, y.lo);
        struct ir_val *hi = icmp(cond == IR_GT || cond == IR_GE ? IR_GT : IR_LT, x.hi, y.hi);
        if (is_signed)
            hi->aux |= IR_SIGNED;
        c = op2(IR_POR, IR_PRED, hi, op2(IR_PAND, IR_PRED, hi_eq, lo));
    }
    set(in, lane, c);
    return 1;
}

static void compare(LLVMValueRef in, unsigned n)
{
    int is_float = LLVMGetInstructionOpcode(in) == LLVMFCmp;
    if (is_ptr(LLVMTypeOf(LLVMGetOperand(in, 0)))) {
        set(in, 0, compare_pointers(in));
        return;
    }
    for (unsigned l = 0; l < n; l++) {
        if (!is_float && is_i64(lane_type(LLVMTypeOf(LLVMGetOperand(in, 0)), l)) && compare64(in, l))
            continue;
        struct ir_val *a = get(LLVMGetOperand(in, 0), l), *b = get(LLVMGetOperand(in, 1), l);
        if (is_float) {
            struct ir_val *c = op2(IR_CMP, IR_PRED, widen(a), widen(b));
            c->aux = IR_FCMP | LLVMGetFCmpPredicate(in);
            set(in, l, c);
            continue;
        }
        unsigned p = LLVMGetICmpPredicate(in) - LLVMIntEQ;
        if (p >= sizeof g_icond / sizeof g_icond[0])
            airc_fail("bad icmp predicate");
        unsigned bits = 8 * scalar_bytes(lane_type(LLVMTypeOf(LLVMGetOperand(in, 0)), l));
        struct ir_val *c = op2(IR_CMP, IR_PRED, int_extend(a, bits, g_icond[p][1]), int_extend(b, bits, g_icond[p][1]));
        c->aux = g_icond[p][0] | (g_icond[p][1] ? IR_SIGNED : 0);
        set(in, l, c);
    }
}

// lanes [first, first + count) of an aggregate value, as LLVM's extractvalue/insertvalue name them
static unsigned member_lane(LLVMTypeRef t, unsigned idx)
{
    unsigned first = 0;
    LLVMTypeKind k = LLVMGetTypeKind(t);
    for (unsigned i = 0; i < idx; i++)
        first += nlanes(k == LLVMStructTypeKind ? LLVMStructGetTypeAtIndex(t, i) : LLVMGetElementType(t));
    return first;
}

static LLVMTypeRef member_type(LLVMTypeRef t, unsigned idx)
{
    return LLVMGetTypeKind(t) == LLVMStructTypeKind ? LLVMStructGetTypeAtIndex(t, idx) : LLVMGetElementType(t);
}

// control point i of a post-tessellation vertex function's patch, (i, patch): each member of the struct it
// returns is the generic attribute of its location, of the patch's input vertex i. A kernel's (OpenGL's
// tessellation control stage) is its stage input at stage-in grid position i, the thread's own y.
static void control_point(LLVMValueRef call)
{
    struct ir_val *i = get(LLVMGetOperand(call, 0), 0), *vtx = NULL;
    if (g_info->stage != NVK_COMPUTE)
        vtx = op1(IR_ISBERD, IR_I32, op2(IR_ADD, IR_I32, g_point_map, i));
    LLVMTypeRef t = LLVMTypeOf(call);
    int whole = LLVMGetTypeKind(t) != LLVMStructTypeKind;
    unsigned members = whole ? 1 : LLVMCountStructElementTypes(t);
    if (members > g_npoint_loc)
        airc_fail("a control point of %u members, %u locations", members, g_npoint_loc);
    for (unsigned m = 0; m < members; m++) {
        unsigned first = whole ? 0 : member_lane(t, m), n = whole ? nlanes(t) : nlanes(member_type(t, m));
        if (n > 4)
            airc_fail("a control point member wider than 4 lanes");
        struct ir_val *fetched[4];
        if (!vtx)
            stage_in_attr(g_point_loc[m], i, thread_value(NVK_ARG_TID, 1), fetched);
        for (unsigned l = 0; l < n; l++) {
            enum ir_type lt = ir_type_of(lane_type(t, first + l));
            struct ir_val *v;
            if (vtx) {
                v = tess_attr(g_cur, vtx, 0x80 + 16 * g_point_loc[m] + 4 * l, 0);
                v->type = lt == IR_I32 ? IR_I32 : IR_F32;
            } else {
                v = stage_in_lane(fetched[l], lane_type(t, first + l));
            }
            set(call, first + l, lt == IR_F16 ? op1(IR_F2H, IR_F16, v) : v);
        }
    }
}

static void instruction(LLVMValueRef in);

struct blockinfo {
    LLVMBasicBlockRef bb;
    struct ir_block *head, *tail;
};

static struct ir_block **blocks_of(LLVMBasicBlockRef bb)
{
    struct entry *e = find(g_blocks, bb, 0, g_scope);
    if (!e)
        airc_fail("branch to an unknown block");
    return e->blocks;
}

// returns of an inlined function: where each comes from, and what it returns
struct ret {
    struct ir_block *from;
    struct ir_val **vals;
    struct ptr *ptrs;  // the pointer lanes'
    struct ir_val **highs;  // the 64-bit lanes' high words
    struct ret *next;
};
static struct ret *g_rets;
static struct ir_block *g_cont;  // where an inlined function's returns go, NULL in a subroutine's own body
static LLVMTypeRef g_ret_type;

static void emit_outputs(LLVMValueRef ret);
static void sub_return(LLVMValueRef ret);

// a phi waiting for its incoming values, which may come from blocks not converted yet
struct pending_phi {
    LLVMValueRef phi;
    int pointer;  // one lane, the pointer's value (ptr_value)
    unsigned scope;
    struct ir_val **lanes;
    struct ir_val **highs;  // the high words of its 64-bit lanes, NULL for the others
    struct pending_phi *next;
};
static struct pending_phi *g_phis;
static LLVMBasicBlockRef g_bb;  // the block being converted

static void postorder(LLVMBasicBlockRef b, LLVMBasicBlockRef *seen, unsigned *nseen, LLVMBasicBlockRef *out,
                      unsigned *nout, unsigned cap)
{
    for (unsigned i = 0; i < *nseen; i++)
        if (seen[i] == b)
            return;
    if (*nseen == cap)
        airc_fail("more than %u blocks in a function", cap);
    seen[(*nseen)++] = b;
    LLVMValueRef term = LLVMGetBasicBlockTerminator(b);
    for (unsigned i = LLVMGetNumSuccessors(term); i-- > 0;)
        postorder(LLVMGetSuccessor(term, i), seen, nseen, out, nout, cap);
    out[(*nout)++] = b;
}

static void terminator(LLVMValueRef in, LLVMOpcode op)
{
    struct ir_block *b = g_cur;
    switch (op) {
    case LLVMRet:
        if (g_depth == 0) {
            emit_outputs(in);
        } else if (!g_cont) {
            sub_return(in);
        } else {
            struct ret *r = ir_alloc(sizeof(*r));
            unsigned n = g_ret_type ? nlanes(g_ret_type) : 0;
            r->from = b;
            r->vals = n ? ir_alloc(n * sizeof(*r->vals)) : NULL;
            r->ptrs = n ? ir_alloc(n * sizeof(*r->ptrs)) : NULL;
            r->highs = n ? ir_alloc(n * sizeof(*r->highs)) : NULL;
            for (unsigned l = 0; l < n; l++) {
                if (lane_is_ptr(g_ret_type, l)) {
                    r->ptrs[l] = ptr_lane(LLVMGetOperand(in, 0), l);
                } else {
                    r->vals[l] = get(LLVMGetOperand(in, 0), l);
                    if (is_i64(lane_type(g_ret_type, l)))
                        r->highs[l] = high_word(LLVMGetOperand(in, 0), l);
                }
            }
            r->next = g_rets;
            g_rets = r;
            ir_add(b, IR_BR, IR_VOID, 0);
            ir_edge(b, g_cont);
        }
        break;
    case LLVMUncondBr: {
        ir_add(b, IR_BR, IR_VOID, 0);
        ir_edge(b, blocks_of(LLVMGetSuccessor(in, 0))[0]);
        break;
    }
    case LLVMCondBr: {
        struct ir_val *c = ir_add(b, IR_CBR, IR_VOID, 1);
        c->src[0] = get(LLVMGetCondition(in), 0);
        ir_edge(b, blocks_of(LLVMGetSuccessor(in, 0))[0]);
        ir_edge(b, blocks_of(LLVMGetSuccessor(in, 1))[0]);
        break;
    }
    case LLVMSwitch: {  // a chain of compares; successor 0 the default
        struct ir_val *v = get(LLVMGetOperand(in, 0), 0);
        unsigned ncases = LLVMGetNumSuccessors(in) - 1;
        for (unsigned i = 0; i < ncases; i++) {
            struct ir_val *eq = ir_op2(g_cur, IR_CMP, IR_PRED, v, get(LLVMGetSwitchCaseValue(in, i + 1), 0));
            eq->aux = IR_EQ;
            struct ir_val *c = ir_add(g_cur, IR_CBR, IR_VOID, 1);
            c->src[0] = eq;
            struct ir_block *next = ir_block(g_f);
            ir_edge(g_cur, blocks_of(LLVMGetSuccessor(in, i + 1))[0]);
            ir_edge(g_cur, next);
            insert(g_origins, g_cur, 0, 0)->bb = g_bb;  // a phi's edge may come from here
            g_cur = next;
        }
        ir_add(g_cur, IR_BR, IR_VOID, 0);
        ir_edge(g_cur, blocks_of(LLVMGetSuccessor(in, 0))[0]);
        break;
    }
    case LLVMUnreachable:
        ir_add(b, IR_EXIT, IR_VOID, 0);
        break;
    default:
        airc_fail("unsupported terminator %d", op);
    }
}

// Convert fn's body in the current scope, into fresh blocks; the caller has made g_cur branch to its entry.
static struct ir_block *body(LLVMValueRef fn)
{
    LLVMValueRef outer_fn = g_body_fn;
    g_body_fn = fn;
    enum { CAP = 4096 };
    LLVMBasicBlockRef *seen = malloc(CAP * sizeof(*seen)), *post = malloc(CAP * sizeof(*post));
    unsigned nseen = 0, npost = 0;
    postorder(LLVMGetFirstBasicBlock(fn), seen, &nseen, post, &npost, CAP);
    for (unsigned i = 0; i < npost; i++) {
        struct entry *e = insert(g_blocks, post[i], 0, g_scope);
        e->blocks[0] = e->blocks[1] = ir_block(g_f);
    }
    struct ir_block *entry = blocks_of(LLVMGetFirstBasicBlock(fn))[0];
    struct pending_phi *outer = g_phis;
    g_phis = NULL;
    while (npost) {  // reverse postorder: a block's dominators come first
        LLVMBasicBlockRef bb = post[--npost];
        struct ir_block **bl = blocks_of(bb);
        g_cur = bl[0];
        for (LLVMValueRef in = LLVMGetFirstInstruction(bb); in; in = LLVMGetNextInstruction(in)) {
            g_bb = bb;  // an inlined call converts other blocks on the way
            instruction(in);
        }
        blocks_of(bb)[1] = g_cur;  // calls inlined on the way may have moved the end elsewhere
        insert(g_origins, g_cur, 0, 0)->bb = bb;
        (void)bl;
    }
    // every phi's incoming values in its block's predecessor order: the IR block that ends each LLVM
    // predecessor is the tail recorded for it
    for (struct pending_phi *p = g_phis; p; p = p->next) {
        unsigned saved = g_scope;
        g_scope = p->scope;
        struct ir_block *b = p->lanes[0]->block;
        unsigned n = p->pointer ? 1 : nlanes(LLVMTypeOf(p->phi));
        for (unsigned k = 0; k < b->npreds; k++) {
            int found = 0;
            struct entry *origin = find(g_origins, b->preds[k], 0, 0);
            for (unsigned i = 0; i < LLVMCountIncoming(p->phi) && !found; i++)
                if (origin && origin->bb == LLVMGetIncomingBlock(p->phi, i)) {
                    struct ir_block *saved_cur = g_cur;
                    g_cur = b->preds[k];  // a constant made for the edge lives in the prologue anyway
                    for (unsigned l = 0; l < n; l++) {
                        if (p->lanes[l]->nsrc != b->npreds) {
                            p->lanes[l]->nsrc = b->npreds;
                            p->lanes[l]->src = ir_alloc(b->npreds * sizeof(*p->lanes[l]->src));
                        }
                        LLVMValueRef in = LLVMGetIncomingValue(p->phi, i);
                        if (p->pointer) {
                            struct ptr from = ptr_of(in), to = find(g_ptrs, p->phi, 0, p->scope)->p;
                            int handle = from.space == SP_TEXTURE || from.space == SP_SAMPLER;
                            if ((from.space == SP_LOCAL) != (to.space == SP_LOCAL))
                                airc_fail("a phi of pointers into local and other memory");
                            if (handle != (to.space == SP_TEXTURE || to.space == SP_SAMPLER))
                                airc_fail("a phi of textures or samplers and other pointers");
                            if (handle)
                                picked_from(to.param, from);
                        }
                        p->lanes[l]->src[k] = p->pointer ? ptr_value(ptr_of(in)) : get(in, l);
                        if (!p->pointer && p->highs[l]) {
                            if (p->highs[l]->nsrc != b->npreds) {
                                p->highs[l]->nsrc = b->npreds;
                                p->highs[l]->src = ir_alloc(b->npreds * sizeof(*p->highs[l]->src));
                            }
                            p->highs[l]->src[k] = high_or_zero(in, l);
                        }
                    }
                    g_cur = saved_cur;
                    found = 1;
                }
            if (!found)
                airc_fail("phi without a value for one of its edges");
        }
        g_scope = saved;
    }
    note_picked_uses();  // a subroutine's calls take its parameters' uses as soon as its body is done
    g_phis = outer;
    free(seen);
    free(post);
    g_body_fn = outer_fn;
    return entry;
}

static void inline_call(LLVMValueRef call, LLVMValueRef fn)
{
    if (g_depth == MAX_INLINE_DEPTH)
        airc_fail("calls nest deeper than %d", MAX_INLINE_DEPTH);
    unsigned caller = g_scope, callee = ++g_next_scope;
    // the callee's parameters are the call's arguments, lanes and pointers alike
    for (unsigned i = 0; i < LLVMCountParams(fn); i++) {
        LLVMValueRef p = LLVMGetParam(fn, i), a = LLVMGetOperand(call, i);
        if (is_ptr(LLVMTypeOf(p))) {
            if (LLVMIsUndef(a) || LLVMIsPoison(a))  // a resource the callee doesn't use; any use fails
                continue;
            struct ptr q = ptr_of(a);
            insert(g_ptrs, p, 0, callee)->p = q;
        } else {
            for (unsigned l = 0; l < nlanes(LLVMTypeOf(p)); l++) {
                if (!lane_is_ptr(LLVMTypeOf(p), l)) {
                    insert(g_vals, p, l, callee)->v = get(a, l);
                    if (is_i64(lane_type(LLVMTypeOf(p), l)) && high_word(a, l))
                        insert(g_highs, p, l, callee)->v = high_word(a, l);
                } else if (!LLVMIsUndef(a) && !LLVMIsPoison(a))
                    insert(g_ptrs, p, l, callee)->p = ptr_lane(a, l);
            }
        }
    }
    struct ir_block *from = g_cur, *cont = ir_block(g_f), *saved_cont = g_cont;
    struct ret *saved_rets = g_rets;
    LLVMTypeRef saved_type = g_ret_type;
    g_cont = cont;
    g_rets = NULL;
    g_ret_type = LLVMTypeOf(call);
    g_scope = callee;
    g_depth++;
    struct ir_block *entry = body(fn);
    g_depth--;
    g_scope = caller;
    ir_add(from, IR_BR, IR_VOID, 0);
    ir_edge(from, entry);
    // the returned lanes: straight through from a single return, else one phi per lane in cont
    unsigned n = nlanes(g_ret_type);
    for (unsigned l = 0; l < n; l++) {
        if (lane_is_ptr(g_ret_type, l)) {  // the same pointer from every return, or a phi of global ones
            struct ptr q = g_rets->ptrs[l];
            for (struct ret *r = g_rets->next; r; r = r->next)
                if (memcmp(&r->ptrs[l], &q, sizeof(q)))
                    airc_fail("a function returns different pointers");
            insert(g_ptrs, call, l, g_scope)->p = q;
            continue;
        }
        int wide = is_i64(lane_type(g_ret_type, l));
        if (cont->npreds == 1) {
            set(call, l, g_rets->vals[l]);
            if (wide && g_rets->highs[l])
                set_high(call, l, g_rets->highs[l]);
            continue;
        }
        struct ir_val *phi = ir_add(cont, IR_PHI, g_rets->vals[l]->type, cont->npreds);
        struct ir_val *hphi = wide ? ir_add(cont, IR_PHI, IR_I32, cont->npreds) : NULL;
        for (unsigned k = 0; k < cont->npreds; k++)
            for (struct ret *r = g_rets; r; r = r->next)
                if (r->from == cont->preds[k]) {
                    phi->src[k] = r->vals[l];
                    if (wide)
                        hphi->src[k] = r->highs[l] ? r->highs[l] : konst(IR_I32, 0);
                }
        set(call, l, phi);
        if (wide)
            set_high(call, l, hphi);
    }
    g_cont = saved_cont;
    g_rets = saved_rets;
    g_ret_type = saved_type;
    g_cur = cont;
}

// A noinline function becomes a subroutine: its body converted once, each call passing the arguments in and the
// results out. A pointer goes as its address (device, constant or threadgroup memory) or a texture's or sampler's
// handle; an alloca's variables go in as values and come back out with the results. A call that can't pass its
// arguments the way the first one did is inlined.

enum pass { PASS_LANES, PASS_ADDRESS, PASS_HANDLE, PASS_PRIVATE, PASS_NONE };

struct sub_param {
    enum pass how;
    enum space space;   // what an address points into, or what a handle is of
    LLVMTypeRef type;   // the alloca whose variables go in and out
    unsigned uses;      // a handle's USE_* in the body, a bit each
};

#define SUB_MAX_PARAMS 16
#define SUB_MAX_SLOTS 32  // registers of arguments, and of results

struct sub {
    LLVMValueRef fn;
    int index;  // in g_f->subs
    int converting;
    struct sub_param params[SUB_MAX_PARAMS];
    struct sub *next;
};
static struct sub *g_subs, *g_sub;  // g_sub: the one whose body is being converted
static int g_calls;                 // noinline functions become subroutines (else they're inlined like the rest)

static void note_use(struct ptr p, int use)
{
    if (p.param)
        p.param->uses |= 1u << use;
    else if (p.space == SP_SAMPLER && !p.base && p.slot < 32 && use <= USE_COMPARE)
        *(use == USE_COMPARE ? &g_info->compare_samplers : &g_info->plain_samplers) |= 1u << p.slot;
    else if (p.space == SP_SAMPLER && !p.base && p.slot < 32 && use == USE_NEAREST)
        g_info->nearest_samplers |= 1u << p.slot;
    else if (p.space == SP_TEXTURE && !p.base && p.slot < 32 && use == USE_FACES)
        g_info->face_textures |= 1u << p.slot;
    else if (p.space == SP_TEXTURE && !p.base && p.slot < 32 && use == USE_LEVELS)
        g_info->level_textures |= 1u << p.slot;
}

// A phi or select of textures or samplers picks a handle like a subroutine's parameter: what its uses need goes to
// each texture or sampler it picks from, once they are all known.
struct picked {
    struct sub_param *param;
    struct ptr from;
    struct picked *next;
};
static struct picked *g_picked;

static struct sub_param *picker(enum space space)
{
    struct sub_param *p = ir_alloc(sizeof(*p));
    *p = (struct sub_param){ .how = PASS_HANDLE, .space = space };
    return p;
}

static void picked_from(struct sub_param *param, struct ptr from)
{
    struct picked *p = ir_alloc(sizeof(*p));
    p->param = param;
    p->from = from;
    p->next = g_picked;
    g_picked = p;
}

// until nothing changes: one picked from another picks the uses up after it
static void note_picked_uses(void)
{
    for (int changed = 1; changed;) {
        changed = 0;
        for (struct picked *p = g_picked; p; p = p->next) {
            unsigned before = p->from.param ? p->from.param->uses : 0;
            for (int use = USE_PLAIN; use <= USE_NEAREST; use++)
                if (p->param->uses >> use & 1)
                    note_use(p->from, use);
            changed |= p->from.param && p->from.param->uses != before;
        }
    }
}

static int wants_sub(LLVMValueRef fn)
{
    const char *name = LLVMGetValueName(fn);
    return g_calls && (LLVMGetEnumAttributeAtIndex(fn, LLVMAttributeFunctionIndex,
                                                   LLVMGetEnumAttributeKindForName("noinline", 8)) ||
                       (name && strstr(name, "noinline")));
}

static int has_ptr_lane(LLVMTypeRef t)
{
    for (unsigned l = 0; l < nlanes(t); l++)
        if (lane_is_ptr(t, l))
            return 1;
    return 0;
}

// a pointer, which private memory doesn't hold as a value, or a 64-bit integer, which it holds as two words: an
// alloca with one can't go in and out of a subroutine a lane at a time
static int has_wide_lane(LLVMTypeRef t)
{
    for (unsigned l = 0; l < nlanes(t); l++)
        if (scalar_bytes(lane_type(t, l)) > 4)
            return 1;
    return 0;
}

// how a call passes argument a for parameter p: 0 when it can't
static int pass_of(LLVMValueRef p, LLVMValueRef a, struct sub_param *how)
{
    memset(how, 0, sizeof(*how));
    if (!is_ptr(LLVMTypeOf(p))) {
        how->how = PASS_LANES;
        return !has_ptr_lane(LLVMTypeOf(p));
    }
    if (LLVMIsUndef(a) || LLVMIsPoison(a)) {
        how->how = PASS_NONE;
        return 1;
    }
    struct ptr q = ptr_of(a);
    LLVMValueRef r = ptr_root(a);
    how->space = q.space;
    switch (q.space) {
    case SP_GLOBAL:
    case SP_SHARED:
        how->how = PASS_ADDRESS;
        return 1;
    case SP_TEXTURE:
    case SP_SAMPLER:
        how->how = PASS_HANDLE;
        return 1;
    case SP_PRIVATE:  // the whole of an alloca, its variables at fixed offsets
        if (q.base || q.off || !LLVMIsAInstruction(r) || LLVMGetInstructionOpcode(r) != LLVMAlloca)
            return 0;
        how->how = PASS_PRIVATE;
        how->type = LLVMGetAllocatedType(r);
        return !has_wide_lane(how->type);
    default:
        return 0;
    }
}

// registers an argument or a result passed this way takes
static unsigned pass_slots(const struct sub_param *how)
{
    switch (how->how) {
    case PASS_ADDRESS:
        return how->space == SP_GLOBAL ? 2 : 1;
    case PASS_HANDLE:
        return 1;
    case PASS_PRIVATE:
        return nlanes(how->type);
    default:
        return 0;
    }
}

static struct ir_val *param_value(enum ir_type t, unsigned *slot)
{
    struct ir_val *v = ir_add(g_cur, IR_PARAM, t, 0);
    v->aux = (int)*slot;
    *slot += t == IR_I64 ? 2 : 1;
    return v;
}

static struct sub *make_sub(LLVMValueRef fn, const struct sub_param *how, LLVMTypeRef ret_type)
{
    struct sub *s = ir_alloc(sizeof(*s));
    s->fn = fn;
    s->index = g_f->nsubs++;
    memcpy(s->params, how, LLVMCountParams(fn) * sizeof(*how));
    s->next = g_subs;
    g_subs = s;
    struct ir_block *entry = ir_block(g_f), *saved_cur = g_cur, *saved_cont = g_cont;
    g_f->subs[s->index] = entry;
    unsigned saved_scope = g_scope, saved_depth = g_depth;
    struct ret *saved_rets = g_rets;
    LLVMTypeRef saved_type = g_ret_type;
    LLVMBasicBlockRef saved_bb = g_bb;
    struct sub *saved_sub = g_sub;
    g_scope = ++g_next_scope;
    g_depth = 1;
    g_cont = NULL;
    g_rets = NULL;
    g_ret_type = ret_type;
    g_sub = s;
    g_cur = entry;
    unsigned slot = 0;
    for (unsigned i = 0; i < LLVMCountParams(fn); i++) {
        LLVMValueRef p = LLVMGetParam(fn, i);
        LLVMTypeRef t = LLVMTypeOf(p);
        struct ptr q = { .space = how[i].space };
        switch (how[i].how) {
        case PASS_LANES:
            for (unsigned l = 0; l < nlanes(t); l++)
                insert(g_vals, p, l, g_scope)->v = param_value(ir_type_of(lane_type(t, l)), &slot);
            break;
        case PASS_ADDRESS:
            q.base = param_value(q.space == SP_GLOBAL ? IR_I64 : IR_I32, &slot);
            insert(g_ptrs, p, 0, g_scope)->p = q;
            break;
        case PASS_HANDLE:
            q.base = param_value(IR_I32, &slot);
            q.param = &s->params[i];
            insert(g_ptrs, p, 0, g_scope)->p = q;
            break;
        case PASS_PRIVATE:
            q.slot = ++g_nallocas;
            insert(g_ptrs, p, 0, g_scope)->p = q;
            for (unsigned l = 0; l < nlanes(how[i].type); l++)
                private_write(private_var(q, lane_offset(how[i].type, l)),
                              param_value(ir_type_of(lane_type(how[i].type, l)), &slot));
            break;
        case PASS_NONE:  // a resource the body doesn't use; any use fails
            break;
        }
    }
    s->converting = 1;
    struct ir_block *start = body(fn);
    s->converting = 0;
    ir_add(entry, IR_BR, IR_VOID, 0);
    ir_edge(entry, start);
    g_cur = saved_cur;
    g_cont = saved_cont;
    g_scope = saved_scope;
    g_depth = saved_depth;
    g_rets = saved_rets;
    g_ret_type = saved_type;
    g_bb = saved_bb;
    g_sub = saved_sub;
    return s;
}

// a subroutine's return: its results, then the variables of the allocas it was given
static void sub_return(LLVMValueRef in)
{
    struct ir_val *src[SUB_MAX_SLOTS];
    unsigned n = 0;
    for (unsigned l = 0; l < nlanes(g_ret_type); l++)
        src[n++] = get(LLVMGetOperand(in, 0), l);
    for (unsigned i = 0; i < LLVMCountParams(g_sub->fn); i++) {
        const struct sub_param *how = &g_sub->params[i];
        if (how->how != PASS_PRIVATE)
            continue;
        struct ptr q = ptr_of(LLVMGetParam(g_sub->fn, i));
        for (unsigned l = 0; l < nlanes(how->type); l++)
            src[n++] = private_read(private_var(q, lane_offset(how->type, l)), ir_type_of(lane_type(how->type, l)));
    }
    struct ir_val *r = ir_add(g_cur, IR_RET, IR_VOID, n);
    memcpy(r->src, src, n * sizeof(*src));
}

// whether a call passing its arguments as how has them fits s: one passing nothing where s takes an address or a
// handle passes 0
static int fits(const struct sub *s, const struct sub_param *how, unsigned np)
{
    for (unsigned i = 0; i < np; i++) {
        const struct sub_param *p = &s->params[i];
        if (how[i].how == PASS_NONE ? p->how != PASS_NONE && p->how != PASS_ADDRESS && p->how != PASS_HANDLE
                                    : how[i].how != p->how || how[i].space != p->space || how[i].type != p->type)
            return 0;
    }
    return 1;
}

static int call_sub(LLVMValueRef call, LLVMValueRef fn)
{
    unsigned np = LLVMCountParams(fn);
    if (!wants_sub(fn) || np > SUB_MAX_PARAMS)
        return 0;
    struct sub_param how[SUB_MAX_PARAMS];
    LLVMTypeRef rt = LLVMTypeOf(call);
    unsigned nargs = 0, nresults = nlanes(rt);
    for (unsigned i = 0; i < np; i++) {
        if (!pass_of(LLVMGetParam(fn, i), LLVMGetOperand(call, i), &how[i]))
            return 0;
        nargs += how[i].how == PASS_LANES ? nlanes(LLVMTypeOf(LLVMGetParam(fn, i))) : pass_slots(&how[i]);
        nresults += how[i].how == PASS_PRIVATE ? pass_slots(&how[i]) : 0;
    }
    if (has_ptr_lane(rt) || nargs > SUB_MAX_SLOTS || nresults > SUB_MAX_SLOTS)
        return 0;
    struct sub *s = g_subs;
    while (s && s->fn != fn)
        s = s->next;
    if (s && s->converting)
        airc_fail("a recursive call");
    if (s && !fits(s, how, np))
        return 0;
    if (!s && g_f->nsubs == IR_MAX_SUBS)
        return 0;
    if (!s)
        s = make_sub(fn, how, rt);

    struct ir_val *args[SUB_MAX_SLOTS];
    unsigned n = 0;
    for (unsigned i = 0; i < np; i++) {
        LLVMValueRef a = LLVMGetOperand(call, i);
        const struct sub_param *sp = &s->params[i];
        if (sp->how == PASS_LANES) {
            for (unsigned l = 0; l < nlanes(LLVMTypeOf(a)); l++)
                args[n++] = get(a, l);
        } else if (how[i].how == PASS_NONE) {
            if (sp->how != PASS_NONE)
                args[n++] = konst(sp->how == PASS_ADDRESS && sp->space == SP_GLOBAL ? IR_I64 : IR_I32, 0);
        } else if (sp->how == PASS_ADDRESS) {
            args[n++] = ptr_value(ptr_of(a));
        } else if (sp->how == PASS_HANDLE) {
            struct ptr q = ptr_of(a);
            for (int use = USE_PLAIN; use <= USE_NEAREST; use++)
                if (sp->uses >> use & 1)
                    note_use(q, use);
            args[n++] = handle_of_ptr(q, sp->space);
        } else if (sp->how == PASS_PRIVATE) {
            struct ptr q = ptr_of(a);
            for (unsigned l = 0; l < nlanes(sp->type); l++)
                args[n++] = private_read(private_var(q, lane_offset(sp->type, l)), ir_type_of(lane_type(sp->type, l)));
        }
    }
    struct ir_val *c = ir_add(g_cur, IR_CALL, IR_VOID, n);
    memcpy(c->src, args, n * sizeof(*args));
    c->imm = s->index;
    c->aux = (int)nresults;
    // the results in the order the return has them, a register each
    unsigned k = 0;
    for (unsigned l = 0; l < nlanes(rt); l++, k++) {
        struct ir_val *r = op1(IR_PROJ, ir_type_of(lane_type(rt, l)), c);
        r->imm = k;
        set(call, l, r);
    }
    for (unsigned i = 0; i < np; i++) {
        const struct sub_param *sp = &s->params[i];
        if (sp->how != PASS_PRIVATE)
            continue;
        struct ptr q = ptr_of(LLVMGetOperand(call, i));
        for (unsigned l = 0; l < nlanes(sp->type); l++, k++) {
            struct ir_val *r = op1(IR_PROJ, ir_type_of(lane_type(sp->type, l)), c);
            r->imm = k;
            private_write(private_var(q, lane_offset(sp->type, l)), r);
        }
    }
    return 1;
}

static void instruction(LLVMValueRef in)
{
    LLVMOpcode op = LLVMGetInstructionOpcode(in);
    LLVMTypeRef t = LLVMTypeOf(in);
    unsigned n = nlanes(t);
    switch (op) {
    case LLVMRet:
    case LLVMUncondBr:
    case LLVMCondBr:
    case LLVMSwitch:
    case LLVMUnreachable:
        terminator(in, op);
        break;
    case LLVMPHI: {
        struct pending_phi *p = ir_alloc(sizeof(*p));
        p->phi = in;
        p->scope = g_scope;
        if (is_ptr(t)) {
            struct ptr q = phi_ptr(in);
            if (q.space == SP_TEXTURE || q.space == SP_SAMPLER)
                q.param = picker(q.space);
            p->pointer = 1;
            p->lanes = ir_alloc(sizeof(*p->lanes));
            p->lanes[0] = q.base = ir_add(g_cur, IR_PHI, q.space == SP_GLOBAL ? IR_I64 : IR_I32, 0);
            insert(g_ptrs, in, 0, g_scope)->p = q;
            p->next = g_phis;
            g_phis = p;
            break;
        }
        p->lanes = ir_alloc(n * sizeof(*p->lanes));
        p->highs = ir_alloc(n * sizeof(*p->highs));
        for (unsigned l = 0; l < n; l++) {
            p->lanes[l] = ir_add(g_cur, IR_PHI, ir_type_of(lane_type(t, l)), 0);
            set(in, l, p->lanes[l]);
            if (is_i64(lane_type(t, l))) {
                p->highs[l] = ir_add(g_cur, IR_PHI, IR_I32, 0);
                set_high(in, l, p->highs[l]);
            }
        }
        p->next = g_phis;
        g_phis = p;
        break;
    }
    case LLVMAdd:
    case LLVMSub:
    case LLVMMul:
    case LLVMAnd:
    case LLVMOr:
    case LLVMXor:
    case LLVMShl:
    case LLVMLShr:
    case LLVMAShr:
    case LLVMFAdd:
    case LLVMFSub:
    case LLVMFMul:
        binary(in, op, n);
        break;
    case LLVMFDiv:
        fdiv(in, n);
        break;
    case LLVMUDiv:
    case LLVMSDiv:
    case LLVMURem:
    case LLVMSRem:
        idiv(in, op, n);
        break;
    case LLVMFNeg:
        for (unsigned l = 0; l < n; l++) {
            struct ir_val *a = get(LLVMGetOperand(in, 0), l);
            set(in, l, op1(IR_FNEG, a->type, a));
        }
        break;
    case LLVMZExt:
    case LLVMSExt:
    case LLVMTrunc:
    case LLVMBitCast:
    case LLVMFPExt:
    case LLVMFPTrunc:
    case LLVMSIToFP:
    case LLVMUIToFP:
    case LLVMFPToSI:
    case LLVMFPToUI:
        if (is_ptr(t))  // a pointer cast: the same pointer
            insert(g_ptrs, in, 0, g_scope)->p = ptr_of(LLVMGetOperand(in, 0));
        else
            cast(in, op, n);
        break;
    case LLVMAddrSpaceCast:
        insert(g_ptrs, in, 0, g_scope)->p = ptr_of(LLVMGetOperand(in, 0));
        break;
    case LLVMPtrToInt: {  // the address's words, or a shared or local pointer's offset
        struct ir_val *lo, *hi;
        ptr_words(ptr_of(LLVMGetOperand(in, 0)), &lo, &hi);
        set(in, 0, lo);
        if (is_i64(t))
            set_high(in, 0, hi);
        break;
    }
    case LLVMICmp:
    case LLVMFCmp:
        compare(in, n);
        break;
    case LLVMSelect: {
        LLVMValueRef c = LLVMGetOperand(in, 0);
        if (is_ptr(t)) {
            struct ptr a = ptr_of(LLVMGetOperand(in, 1)), b = ptr_of(LLVMGetOperand(in, 2));
            if (a.space != b.space)
                airc_fail("a select of pointers into different memory");
            struct ptr q = { .space = a.space, .base = sel(get(c, 0), ptr_value(a), ptr_value(b)),
                             .slot = a.slot > b.slot ? a.slot : b.slot };  // local: the further alloca's end
            if (a.space == SP_TEXTURE || a.space == SP_SAMPLER) {
                q.param = picker(a.space);
                picked_from(q.param, a);
                picked_from(q.param, b);
            }
            insert(g_ptrs, in, 0, g_scope)->p = q;
            break;
        }
        int vector_cond = nlanes(LLVMTypeOf(c)) > 1;
        for (unsigned l = 0; l < n; l++) {
            struct ir_val *cond = get(c, vector_cond ? l : 0);
            set(in, l, sel(cond, get(LLVMGetOperand(in, 1), l), get(LLVMGetOperand(in, 2), l)));
            struct ir_val *ha = is_i64(lane_type(t, l)) ? high_word(LLVMGetOperand(in, 1), l) : NULL;
            struct ir_val *hb = ha ? high_word(LLVMGetOperand(in, 2), l) : NULL;
            if (hb)
                set_high(in, l, sel(cond, ha, hb));
        }
        break;
    }
    case LLVMGetElementPtr:
    case LLVMAlloca:
        ptr_of(in);  // pointers are made where they're used; keep any error close to its cause
        break;
    case LLVMLoad:
        load(in);
        break;
    case LLVMStore:
        store(in);
        break;
    case LLVMCall:
        call(in);
        break;
    case LLVMExtractElement: {  // a variable index picks its lane with a chain of selects
        LLVMValueRef vec = LLVMGetOperand(in, 0), idx = LLVMGetOperand(in, 1);
        int wide = is_i64(t);
        if (LLVMIsAConstantInt(idx)) {
            unsigned k = (unsigned)LLVMConstIntGetZExtValue(idx);
            set(in, 0, get(vec, k));
            if (wide && high_word(vec, k))
                set_high(in, 0, high_word(vec, k));
            break;
        }
        struct ir_val *i = get(idx, 0), *r = get(vec, 0), *rh = wide ? high_or_zero(vec, 0) : NULL;
        for (unsigned l = 1; l < nlanes(LLVMTypeOf(vec)); l++) {
            struct ir_val *eq = op2(IR_CMP, IR_PRED, i, konst(IR_I32, l));
            eq->aux = IR_EQ;
            r = sel(eq, get(vec, l), r);
            if (wide)
                rh = sel(eq, high_or_zero(vec, l), rh);
        }
        set(in, 0, r);
        if (wide)
            set_high(in, 0, rh);
        break;
    }
    case LLVMInsertElement: {
        LLVMValueRef idx = LLVMGetOperand(in, 2);
        LLVMValueRef vec = LLVMGetOperand(in, 0), elem = LLVMGetOperand(in, 1);
        struct ir_val *val = get(elem, 0);
        int wide = is_i64(LLVMTypeOf(elem));
        if (LLVMIsAConstantInt(idx)) {
            unsigned k = (unsigned)LLVMConstIntGetZExtValue(idx);
            for (unsigned l = 0; l < n; l++) {
                set(in, l, l == k ? val : get(vec, l));
                if (wide)
                    set_high(in, l, l == k ? high_or_zero(elem, 0) : high_or_zero(vec, l));
            }
            break;
        }
        struct ir_val *i = get(idx, 0);
        for (unsigned l = 0; l < n; l++) {
            struct ir_val *eq = op2(IR_CMP, IR_PRED, i, konst(IR_I32, l));
            eq->aux = IR_EQ;
            set(in, l, sel(eq, val, get(vec, l)));
            if (wide)
                set_high(in, l, sel(eq, high_or_zero(elem, 0), high_or_zero(vec, l)));
        }
        break;
    }
    case LLVMShuffleVector: {
        LLVMValueRef a = LLVMGetOperand(in, 0), b = LLVMGetOperand(in, 1);
        unsigned na = nlanes(LLVMTypeOf(a));
        for (unsigned l = 0; l < n; l++) {
            int m = LLVMGetMaskValue(in, l);
            set(in, l, m == LLVMGetUndefMaskElem() ? konst(ir_type_of(lane_type(t, l)), 0)
                       : (unsigned)m < na          ? get(a, (unsigned)m)
                                                   : get(b, (unsigned)m - na));
            if (is_i64(lane_type(t, l)) && m != LLVMGetUndefMaskElem())
                set_high(in, l, (unsigned)m < na ? high_or_zero(a, (unsigned)m) : high_or_zero(b, (unsigned)m - na));
        }
        break;
    }
    case LLVMExtractValue: {
        LLVMValueRef agg = LLVMGetOperand(in, 0);
        LLVMTypeRef at = LLVMTypeOf(agg);
        unsigned first = 0;
        for (unsigned i = 0; i < LLVMGetNumIndices(in); i++) {
            first += member_lane(at, LLVMGetIndices(in)[i]);
            at = member_type(at, LLVMGetIndices(in)[i]);
        }
        for (unsigned l = 0; l < n; l++)
            copy_lane(in, l, agg, first + l);
        break;
    }
    case LLVMInsertValue: {
        LLVMValueRef agg = LLVMGetOperand(in, 0), val = LLVMGetOperand(in, 1);
        LLVMTypeRef at = t;
        unsigned first = 0;
        for (unsigned i = 0; i < LLVMGetNumIndices(in); i++) {
            first += member_lane(at, LLVMGetIndices(in)[i]);
            at = member_type(at, LLVMGetIndices(in)[i]);
        }
        unsigned m = nlanes(LLVMTypeOf(val));
        for (unsigned l = 0; l < n; l++) {
            if (l >= first && l < first + m)
                copy_lane(in, l, val, l - first);
            else
                copy_lane(in, l, agg, l);
        }
        break;
    }
    case LLVMFreeze:
        for (unsigned l = 0; l < n; l++)
            copy_lane(in, l, LLVMGetOperand(in, 0), l);
        break;
    default:
        airc_fail("unsupported instruction %d", op);
    }
}

enum { OUT_ABSENT = -9 };  // an output a function constant removes
static int g_out[40];  // vertex: member i of the returned struct is the position (-1), point size (-2), render target
                       // array index (-4), viewport index (-5) or generic output n
                       // fragment: colour target n, or the sample mask (-3)
static unsigned g_nout;
enum { FRAG_DEPTH = 8 };  // the g_out of a fragment function's [[depth]], after its colour targets

static void emit_outputs(LLVMValueRef ret)
{
    if (g_info->stage == NVK_COMPUTE) {
        ir_add(g_cur, IR_EXIT, IR_VOID, 0);
        return;
    }
    if (!LLVMGetNumOperands(ret)) {  // a fragment function returning nothing: depth or stencil only
        ir_add(g_cur, IR_EXIT, IR_VOID, 0);
        return;
    }
    LLVMValueRef val = LLVMGetOperand(ret, 0);
    LLVMTypeRef t = LLVMTypeOf(val);
    if (g_info->stage == NVK_VERTEX || g_info->stage == NVK_TESS_EVAL) {
        int whole = g_nout == 1 && LLVMGetTypeKind(t) != LLVMStructTypeKind;
        for (unsigned i = 0; i < g_nout; i++) {
            if (g_out[i] == OUT_ABSENT)
                continue;
            unsigned first = whole ? 0 : member_lane(t, i), m = whole ? nlanes(t) : nlanes(member_type(t, i));
            if (m > 4)
                airc_fail("a vertex output wider than 4 lanes");
            struct ir_val *src[4];
            for (unsigned l = 0; l < m; l++)  // before the store: a half lane's widening is an op of its own
                src[l] = widen(get(val, first + l));
            struct ir_val *s = ir_add(g_cur, IR_AST, IR_VOID, m);
            static const unsigned builtin[] = { 0x70, 0x6c, 0, 0x64, 0x68 };
            s->imm = g_out[i] < 0 ? builtin[-1 - g_out[i]] : 0x80 + 16 * g_out[i];
            s->aux = (int)m;
            memcpy(s->src, src, m * sizeof(*src));
        }
        ir_add(g_cur, IR_EXIT, IR_VOID, 0);
        return;
    }
    // fragment: a vec4 per colour target it writes, in target order from R0; a missing lane is 0 (alpha 1); then
    // the sample mask, and the depth a register past the sample mask's (as NAK has it)
    struct ir_val *src[34];
    unsigned nsrc = 0;
    int whole = g_nout == 1 && LLVMGetTypeKind(t) != LLVMStructTypeKind;
    int mask = -1, depth = -1;
    for (int target = 0; target < FRAG_DEPTH; target++) {
        for (unsigned i = 0; i < g_nout; i++) {
            if (g_out[i] != target)
                continue;
            unsigned first = whole ? 0 : member_lane(t, i), n = whole ? nlanes(t) : nlanes(member_type(t, i));
            for (unsigned l = 0; l < 4; l++)
                src[nsrc++] = l < n ? widen(get(val, first + l)) : fconst(l == 3 ? 1.0f : 0.0f);
        }
    }
    for (unsigned i = 0; i < g_nout; i++) {
        if (g_out[i] == -3)
            mask = (int)i;
        else if (g_out[i] == FRAG_DEPTH)
            depth = (int)i;
    }
    if (mask >= 0)  // the sample mask after the colours
        src[nsrc++] = get(val, whole ? 0 : member_lane(t, (unsigned)mask));
    else if (depth >= 0)  // its register is skipped all the same
        src[nsrc++] = fconst(0);
    if (depth >= 0)
        src[nsrc++] = widen(get(val, whole ? 0 : member_lane(t, (unsigned)depth)));
    struct ir_val *e = ir_add(g_cur, IR_EXIT, IR_VOID, nsrc);
    memcpy(e->src, src, nsrc * sizeof(*src));
}

static int is_fc_marker(const char *s, unsigned len)
{
    return s && len == 21 && !memcmp(s, "air.function_constant", 21);
}

// the node's kind, past a function constant's marker
static const char *md_kind(LLVMValueRef node, unsigned *len)
{
    int n = LLVMGetMDNodeNumOperands(node);
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(node, o);
    const char *s = NULL;
    for (int i = 0; i < n && !s; i++) {
        s = LLVMGetMDString(o[i], len);
        if (is_fc_marker(s, *len))
            s = NULL;
    }
    free(o);
    return s;
}

// whether an output exists: always, unless [[function_constant(x)]] makes it depend on a predicate
// (!{!"air.function_constant", !{ptr @predicate, ...}, !"kind", ...})
static int md_present(LLVMValueRef node)
{
    int n = LLVMGetMDNodeNumOperands(node);
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(node, o);
    unsigned len = 0;
    const char *s = n > 1 ? LLVMGetMDString(o[0], &len) : NULL;
    int present = 1;
    if (is_fc_marker(s, len)) {
        LLVMValueRef pred[8];
        LLVMGetMDNodeOperands(o[1], pred);
        present = (int)(fc_eval(pred[0]) & 1);
    }
    free(o);
    return present;
}

// the string after the node's kind (a vertex output's or fragment input's name), copied into out
static void md_name(LLVMValueRef node, char *out, size_t size)
{
    int n = LLVMGetMDNodeNumOperands(node), seen = 0;
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(node, o);
    out[0] = 0;
    for (int i = 0; i < n; i++) {
        unsigned len;
        const char *s = LLVMGetMDString(o[i], &len);
        if (s && !is_fc_marker(s, len) && seen++) {
            snprintf(out, size, "%.*s", (int)len, s);
            break;
        }
    }
    free(o);
}

static int md_has(LLVMValueRef node, const char *want)
{
    int n = LLVMGetMDNodeNumOperands(node), found = 0;
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(node, o);
    for (int i = 0; i < n && !found; i++) {
        unsigned len;
        const char *s = LLVMGetMDString(o[i], &len);
        found = s && len == strlen(want) && !memcmp(s, want, len);
    }
    free(o);
    return found;
}

static void vertex_outputs(LLVMValueRef list)
{
    int n = LLVMGetMDNodeNumOperands(list);
    if (n > (int)(sizeof g_out / sizeof g_out[0]))
        airc_fail("more than %zu vertex outputs", sizeof g_out / sizeof g_out[0]);
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(list, o);
    for (int i = 0; i < n; i++) {
        unsigned len;
        const char *k = md_kind(o[i], &len);
        if (!md_present(o[i]))
            g_out[i] = OUT_ABSENT;
        else if (k && len == 12 && !memcmp(k, "air.position", 12))
            g_out[i] = -1;
        else if (k && len == 14 && !memcmp(k, "air.point_size", 14)) {
            g_out[i] = -2;
            g_info->sysvals |= SPH_POINT_SIZE;
        } else if (k && len == 29 && !memcmp(k, "air.render_target_array_index", 29)) {
            g_out[i] = -4;
            g_info->sysvals |= SPH_LAYER;
            g_info->flags |= NVK_FLAG_LAYER;
        } else if (k && len == 24 && !memcmp(k, "air.viewport_array_index", 24)) {
            g_out[i] = -5;
            g_info->sysvals |= SPH_VIEWPORT;
            g_info->flags |= NVK_FLAG_VIEWPORT;
        } else if (k && len == 17 && !memcmp(k, "air.vertex_output", 17)) {
            if (g_info->ngeneric == sizeof g_info->outputs.names / sizeof g_info->outputs.names[0])
                airc_fail("more than %u user-defined vertex outputs", g_info->ngeneric);
            md_name(o[i], g_info->outputs.names[g_info->ngeneric], AIRC_NAME);
            g_out[i] = (int)g_info->ngeneric++;
            g_info->outputs.n = g_info->ngeneric;
        }
        else
            airc_fail("unsupported vertex output %.*s", (int)len, k ? k : "?");
    }
    g_nout = n;
    free(o);
}

// an integer attribute of an argument: a constant, or a global that a function constant sets
// ([[texture(n)]] with n an expression of them)
static int md_int(LLVMValueRef v, unsigned *out)
{
    if (LLVMIsAConstantInt(v))
        *out = (unsigned)LLVMConstIntGetZExtValue(v);
    else if (LLVMIsAGlobalVariable(v))
        *out = (unsigned)fc_eval(v);
    else
        return 0;
    return 1;
}

// fragment: member i of the returned struct (or the whole value) is [[color(g_out[i])]]
static void fragment_outputs(LLVMValueRef list)
{
    int n = LLVMGetMDNodeNumOperands(list);
    if (n > 8)
        airc_fail("more than 8 fragment outputs");
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(list, o);
    for (int i = 0; i < n; i++) {
        int m = LLVMGetMDNodeNumOperands(o[i]);
        LLVMValueRef *all = calloc(m, sizeof(*all)), *f = all;
        LLVMGetMDNodeOperands(o[i], f);
        unsigned len = 0;
        const char *k = m ? LLVMGetMDString(f[0], &len) : NULL;
        if (is_fc_marker(k, len) && m > 3) {  // the function constant's predicate and the kind that follows
            if (!md_present(o[i])) {
                g_out[i] = OUT_ABSENT;
                free(all);
                continue;
            }
            f += 2;
            m -= 2;
            k = LLVMGetMDString(f[0], &len);
        }
        if (k && len == 9 && !memcmp(k, "air.depth", 9)) {
            g_out[i] = FRAG_DEPTH;
            g_info->ps_sysvals |= SPH_PS_DEPTH;
            g_info->flags |= NVK_FLAG_DEPTH;
            free(all);
            continue;
        } else if (k && len == 15 && !memcmp(k, "air.sample_mask", 15)) {
            g_out[i] = -3;
            g_info->ps_sysvals |= SPH_PS_SAMPLE_MASK;
            free(all);
            continue;
        }
        unsigned target, second = 0;
        if (!k || len != 17 || memcmp(k, "air.render_target", 17) || m < 2 || !md_int(f[1], &target))
            airc_fail("unsupported fragment output %.*s", k ? (int)len : 1, k ? k : "?");
        g_out[i] = (int)target;
        // [[color(0), index(1)]], dual-source blending's second colour, goes out as target 1's, where the blender
        // takes it from
        if (m > 2 && md_int(f[2], &second) && second)
            g_out[i]++;
        if (g_out[i] > 7)
            airc_fail("colour target %d", g_out[i]);
        g_info->targets |= 1u << g_out[i];
        free(all);
    }
    g_nout = n;
    free(o);
}

// the string after key in an argument's metadata node, "" if it has none
static void md_value(LLVMValueRef node, const char *key, char *out, size_t size)
{
    int n = LLVMGetMDNodeNumOperands(node);
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(node, o);
    out[0] = 0;
    for (int i = 0; i + 1 < n; i++) {
        unsigned len, vlen;
        const char *s = LLVMGetMDString(o[i], &len);
        const char *v = s && len == strlen(key) && !memcmp(s, key, len) ? LLVMGetMDString(o[i + 1], &vlen) : NULL;
        if (v) {
            snprintf(out, size, "%.*s", (int)vlen, v);
            break;
        }
    }
    free(o);
}

// a post-tessellation vertex function's [[patch(domain, n)]]: {"air.patch", domain, "air.patch_control_point", n},
// without the count when the draw gives it
static void patch_of(LLVMValueRef node)
{
    char domain[16];
    md_value(node, "air.patch", domain, sizeof domain);
    if (strcmp(domain, "triangle") && strcmp(domain, "quad"))
        airc_fail("unsupported patch %s", domain);
    g_info->stage = NVK_TESS_EVAL;
    g_info->patch_domain = !strcmp(domain, "quad");
    int n = LLVMGetMDNodeNumOperands(node);
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(node, o);
    for (int i = 0; i + 1 < n; i++) {
        unsigned len;
        const char *s = LLVMGetMDString(o[i], &len);
        if (s && len == 23 && !memcmp(s, "air.patch_control_point", 23) && LLVMIsAConstantInt(o[i + 1]))
            g_info->patch_points = (unsigned)LLVMConstIntGetZExtValue(o[i + 1]);
    }
    free(o);
}

// the stage's function, from its !air.kernel/vertex/fragment node {fn, outputs, arguments}, a post-tessellation
// vertex function's with its patch after them
static LLVMValueRef stage_function(LLVMValueRef *arglist)
{
    static const struct {
        const char *md;
        enum nvk_stage stage;
    } stages[] = { { "air.kernel", NVK_COMPUTE }, { "air.vertex", NVK_VERTEX }, { "air.fragment", NVK_FRAGMENT } };
    for (unsigned s = 0; s < sizeof stages / sizeof stages[0]; s++) {
        if (!LLVMGetNamedMetadataNumOperands(g_module, stages[s].md))
            continue;
        LLVMValueRef node = first_node(g_module, stages[s].md);
        int cnt = LLVMGetMDNodeNumOperands(node);
        if (cnt < 3)
            airc_fail("malformed %s metadata", stages[s].md);
        LLVMValueRef *ops = calloc(cnt, sizeof(*ops));
        LLVMGetMDNodeOperands(node, ops);
        LLVMValueRef fn = ops[0], outputs = ops[1], patch = cnt > 3 ? ops[3] : NULL;
        *arglist = ops[2];
        for (int i = 3; i < cnt; i++) {  // the function's attributes: !{!"air.max_work_group_size", i32 N}
            LLVMValueRef a[2];
            unsigned len;
            if (LLVMGetMDNodeNumOperands(ops[i]) != 2)
                continue;
            LLVMGetMDNodeOperands(ops[i], a);
            const char *k = LLVMGetMDString(a[0], &len);
            if (k && len == 23 && !memcmp(k, "air.max_work_group_size", 23) && LLVMIsAConstantInt(a[1]))
                g_info->max_threads = (unsigned)LLVMConstIntGetZExtValue(a[1]);
        }
        free(ops);
        if (!LLVMIsAFunction(fn))
            airc_fail("%s does not name a function", stages[s].md);
        g_info->stage = stages[s].stage;
        if (g_info->stage == NVK_VERTEX && patch && md_has(patch, "air.patch"))
            patch_of(patch);
        if (g_info->stage == NVK_VERTEX || g_info->stage == NVK_TESS_EVAL)
            vertex_outputs(outputs);
        else if (g_info->stage == NVK_FRAGMENT)
            fragment_outputs(outputs);
        return fn;
    }
    airc_fail("no air.kernel, air.vertex or air.fragment found");
}

// a patch_control_point<T> argument's node: {"air.patch_control_point_input", {"air.patch_control_point_function",
// fn}, then {"air.location_index", n, ...} for each member of T}
static void control_points(LLVMValueRef node)
{
    int n = LLVMGetMDNodeNumOperands(node);
    LLVMValueRef *o = calloc(n, sizeof(*o));
    LLVMGetMDNodeOperands(node, o);
    for (int i = 0; i < n; i++) {
        unsigned len, m;
        if (LLVMGetMDString(o[i], &len) || LLVMIsAConstantInt(o[i]) || (m = LLVMGetMDNodeNumOperands(o[i])) < 2)
            continue;
        LLVMValueRef *f = calloc(m, sizeof(*f));
        LLVMGetMDNodeOperands(o[i], f);
        const char *s = LLVMGetMDString(f[0], &len);
        if (s && len == 32 && !memcmp(s, "air.patch_control_point_function", 32) && LLVMIsAFunction(f[1])) {
            g_point_fn = f[1];
        } else if (s && len == 18 && !memcmp(s, "air.location_index", 18) && LLVMIsAConstantInt(f[1])) {
            if (g_npoint_loc == sizeof g_point_loc)
                airc_fail("a control point of more than 16 members");
            unsigned loc = (unsigned)LLVMConstIntGetZExtValue(f[1]);
            g_point_loc[g_npoint_loc++] = (uint8_t)loc;
            g_info->vertex_inputs |= 1u << loc;
        }
        free(f);
    }
    free(o);
}

// the next 8-byte param of const bank 0, for a buffer's address or threadgroup memory's offset
static unsigned param_slot(unsigned *off)
{
    if (*off == NVK_TEXTURE_HANDLE(0))  // past the 20th, after the handles (the plugin draws only 20)
        *off = NVK_SAMPLER_HANDLE(16);
    if (*off + 8 > 0x400)
        airc_fail("more than %d buffer and threadgroup arguments", (0x400 - PARAM_BASE - 0x80) / 8);
    *off += 8;
    return *off - 8;
}

static unsigned stage_in_arg(enum nvk_arg_kind kind, unsigned index, unsigned *off)
{
    if (g_info->nargs == NVK_MAX_ARGS)
        airc_fail("more than %d arguments with the stage input's", NVK_MAX_ARGS);
    struct nvk_arg *a = &g_info->args[g_info->nargs++];
    a->kind = (uint8_t)kind;
    a->index = (uint8_t)index;
    a->cbuf_off = (uint16_t)param_slot(off);
    return a->cbuf_off;
}

// A kernel's stage input: its region's param, and one for each buffer its attributes are read from, and their
// indices
static void stage_in_args(unsigned *off)
{
    g_region_off = stage_in_arg(NVK_ARG_STAGE_IN_REGION, 0, off);
    for (unsigned loc = 0; g_stage_in && loc < 32; loc++) {
        const struct airc_tess_attr *a = &g_stage_in->attrs[loc];
        if (!(g_info->vertex_inputs >> loc & 1) || !a->format)
            continue;
        unsigned buffers[2] = { a->buffer, g_stage_in->index_buffer };
        for (unsigned i = 0; i < 1u + (a->step == STEP_X_INDEXED || a->step == STEP_Y_INDEXED); i++) {
            if (buffers[i] >= 31)
                airc_fail("stage input attribute %u: buffer %u", loc, buffers[i]);
            if (!g_stage_in_buffer[buffers[i]])
                g_stage_in_buffer[buffers[i]] = stage_in_arg(NVK_ARG_STAGE_IN_BUFFER, buffers[i], off);
        }
    }
}

static void stage_args(LLVMValueRef fn, LLVMValueRef arglist)
{
    static const struct {
        const char *name;
        enum nvk_arg_kind kind;
    } kinds[] = {
        { "air.buffer", NVK_ARG_BUFFER },
        { "air.indirect_buffer", NVK_ARG_BUFFER },  // an argument buffer: its address like any buffer's
        // an OpenCL kernel's argument passed by value: its bytes in the buffer bound at its index, which the
        // function reads through a constant pointer
        { "air.constant", NVK_ARG_BUFFER },
        { "air.threadgroup", NVK_ARG_THREADGROUP },
        { "air.thread_position_in_grid", NVK_ARG_TID },
        { "air.threadgroup_position_in_grid", NVK_ARG_TG_POS },
        { "air.thread_position_in_threadgroup", NVK_ARG_TG_TID },
        { "air.threads_per_threadgroup", NVK_ARG_TG_SIZE },
        { "air.threadgroups_per_grid", NVK_ARG_GROUPS },
        { "air.texture", NVK_ARG_TEXTURE },
        { "air.sampler", NVK_ARG_SAMPLER },
        { "air.vertex_id", NVK_ARG_VERTEX_ID },
        { "air.instance_id", NVK_ARG_INSTANCE_ID },
        { "air.base_vertex", NVK_ARG_BASE_VERTEX },
        { "air.base_instance", NVK_ARG_BASE_INSTANCE },
        { "air.position", NVK_ARG_POSITION },
        { "air.fragment_input", NVK_ARG_VARYING },
        { "air.vertex_input", NVK_ARG_VERTEX_INPUT },
        { "air.stage_in", NVK_ARG_VERTEX_INPUT },  // a kernel's [[stage_in]] member: its attribute's location
        { "air.render_target", NVK_ARG_RENDER_TARGET },
        { "air.point_coord", NVK_ARG_POINT_COORD },
        { "air.front_facing", NVK_ARG_FRONT_FACING },
        { "air.threads_per_grid", NVK_ARG_GRID_SIZE },
        { "air.thread_index_in_threadgroup", NVK_ARG_TG_INDEX },
        { "air.thread_index_in_simdgroup", NVK_ARG_SIMD_LANE },
        { "air.simdgroup_index_in_threadgroup", NVK_ARG_SIMD_GROUP },
        { "air.threads_per_simdgroup", NVK_ARG_SIMD_SIZE },
        { "air.simdgroups_per_threadgroup", NVK_ARG_SIMD_GROUPS },
        { "air.dispatch_threads_per_threadgroup", NVK_ARG_DISPATCH_TG_SIZE },
        { "air.quadgroup_index_in_threadgroup", NVK_ARG_QUAD_GROUP },
        { "air.thread_index_in_quadgroup", NVK_ARG_QUAD_LANE },
        { "air.quadgroups_per_threadgroup", NVK_ARG_QUAD_GROUPS },
        { "air.patch_control_point_input", NVK_ARG_CONTROL_POINTS },
        { "air.patch_input", NVK_ARG_PATCH_INPUT },
        { "air.position_in_patch", NVK_ARG_POSITION_IN_PATCH },
        { "air.patch_id", NVK_ARG_PATCH_ID },
        { "air.sample_id", NVK_ARG_SAMPLE_ID },
        { "air.sample_mask_in", NVK_ARG_SAMPLE_MASK },
        { "air.render_target_array_index", NVK_ARG_LAYER },
        { "air.viewport_array_index", NVK_ARG_VIEWPORT },
        { "air.stage_in_grid_origin", NVK_ARG_STAGE_IN_ORIGIN },
        { "air.stage_in_grid_size", NVK_ARG_STAGE_IN_SIZE },
    };
    int an = LLVMGetMDNodeNumOperands(arglist);
    if (an > NVK_MAX_ARGS || an != (int)LLVMCountParams(fn))
        airc_fail("%d arguments: more than %d, or not one node each", an, NVK_MAX_ARGS);
    LLVMValueRef *nodes = calloc(an, sizeof(*nodes));
    LLVMGetMDNodeOperands(arglist, nodes);
    unsigned off = PARAM_BASE;
    int stage_in = 0;
    for (int i = 0; i < an; i++) {
        int on = LLVMGetMDNodeNumOperands(nodes[i]);
        LLVMValueRef *o = calloc(on, sizeof(*o));
        LLVMGetMDNodeOperands(nodes[i], o);
        const char *kind = NULL;
        unsigned kind_len = 0, index = 0, space = 0, value, size = 0;
        for (int j = 0; j < on; j++) {
            unsigned len;
            const char *s = LLVMGetMDString(o[j], &len);
            if (!s || (len == 21 && !memcmp(s, "air.function_constant", 21)))
                continue;  // an argument that exists when a function constant says so: bound when it does
            if (!kind) {
                kind = s;
                kind_len = len;
                if (len == 17 && !memcmp(s, "air.render_target", 17) && j + 1 < on && LLVMIsAConstantInt(o[j + 1]))
                    index = (unsigned)LLVMConstIntGetZExtValue(o[j + 1]);
            } else if (len == 18 && !memcmp(s, "air.location_index", 18) && j + 1 < on && md_int(o[j + 1], &value)) {
                index = value;
                if (j + 2 < on && md_int(o[j + 2], &value))  // how many elements an array of them has
                    g_info->array_count[g_info->nargs] = value;
            } else if (len == 17 && !memcmp(s, "air.address_space", 17) && j + 1 < on && LLVMIsAConstantInt(o[j + 1])) {
                space = (unsigned)LLVMConstIntGetZExtValue(o[j + 1]);
            } else if (len == 17 && !memcmp(s, "air.arg_type_size", 17) && j + 1 < on && md_int(o[j + 1], &value)) {
                size = value;
            }
        }
        free(o);
        int k = -1;
        for (unsigned j = 0; kind && j < sizeof kinds / sizeof kinds[0]; j++)
            if (strlen(kinds[j].name) == kind_len && !memcmp(kinds[j].name, kind, kind_len))
                k = kinds[j].kind;
        if (k < 0)
            airc_fail("unsupported argument %.*s", (int)kind_len, kind ? kind : "?");
        if (k == NVK_ARG_BUFFER && space == 3)  // a threadgroup(n) buffer
            k = NVK_ARG_THREADGROUP;
        if (k == NVK_ARG_CONTROL_POINTS)
            control_points(nodes[i]);
        if (k == NVK_ARG_VERTEX_INPUT)
            g_info->vertex_inputs |= 1u << index;
        if (k == NVK_ARG_PATCH_INPUT)
            g_info->patch_inputs |= 1u << index;
        if (k == NVK_ARG_SAMPLE_ID)
            g_info->flags |= NVK_FLAG_PER_SAMPLE;
        if (k == NVK_ARG_VARYING) {  // the slot of the vertex output of its name, or the next one without them
            index = g_info->nvary;
            if (g_vertex) {
                char name[AIRC_NAME];
                md_name(nodes[i], name, sizeof name);
                for (index = 0; index < g_vertex->n && strcmp(g_vertex->names[index], name); index++)
                    ;
                // SDL's solid colour shader takes its vertex input struct and reads only the colour from it, which
                // Metal lets through: the slot past the outputs is one nothing writes
                if (index == g_vertex->n)
                    index = g_info->nvary > g_vertex->n ? g_info->nvary : g_vertex->n;
            }
            if (index >= sizeof g_info->interp)
                airc_fail("more than %zu fragment inputs", sizeof g_info->interp);
            g_info->interp[index] = md_has(nodes[i], "air.flat")             ? SPH_FLAT
                                    : md_has(nodes[i], "air.no_perspective") ? SPH_LINEAR
                                                                             : SPH_PERSPECTIVE;
            if (md_has(nodes[i], "air.centroid"))
                g_info->centroid |= 1u << index;
            if (md_has(nodes[i], "air.sample")) {
                g_info->at_sample |= 1u << index;
                g_info->flags |= NVK_FLAG_PER_SAMPLE;
            }
            if (index + 1 > g_info->nvary)
                g_info->nvary = index + 1;
        }
        md_value(nodes[i], "air.arg_name", g_info->arg_names[g_info->nargs].name, AIRC_NAME);
        md_value(nodes[i], "air.arg_type_name", g_info->arg_names[g_info->nargs].type, AIRC_NAME);
        struct nvk_arg *a = &g_info->args[g_info->nargs++];
        a->kind = (uint8_t)k;
        a->index = (uint8_t)index;
        if (kind_len == 12 && !memcmp(kind, "air.constant", 12))
            a->size = (uint16_t)size;
        a->access = md_has(nodes[i], "air.read_write") ? 1 : md_has(nodes[i], "air.write") ? 2 : 0;
        // Metal lets the same buffer be bound twice, so read-only alone isn't enough: the AIR must also say no
        // other argument reaches it
        if (k == NVK_ARG_BUFFER &&
            (space == 2 || md_has(nodes[i], "air.read") ||
             LLVMGetEnumAttributeAtIndex(fn, i + 1, LLVMGetEnumAttributeKindForName("readonly", 8))) &&
            (LLVMGetEnumAttributeAtIndex(fn, i + 1, LLVMGetEnumAttributeKindForName("noalias", 7)) ||
             LLVMGetStringAttributeAtIndex(fn, i + 1, "air-buffer-no-alias", 19)))
            g_info->readonly_args |= 1ull << (g_info->nargs - 1);
        if (k == NVK_ARG_BUFFER || k == NVK_ARG_THREADGROUP)
            a->cbuf_off = (uint16_t)param_slot(&off);
        if (k == NVK_ARG_STAGE_IN_ORIGIN || k == NVK_ARG_STAGE_IN_SIZE)
            stage_in = 1;
    }
    free(nodes);
    if (g_info->stage == NVK_COMPUTE && (stage_in || g_info->vertex_inputs))
        stage_in_args(&off);
    // Metal binds textures and samplers at indices up to 127, the pools have 16 slots: one past them (MPS binds its
    // sources at 64) takes a free low slot, kept in its cbuf_off for the dispatcher, clear of the constant samplers'
    // and render targets' slots at the top
    for (int kind = NVK_ARG_TEXTURE; kind <= NVK_ARG_SAMPLER; kind++) {
        uint32_t used = 0;
        for (unsigned i = 0; i < g_info->nargs; i++) {
            const struct nvk_arg *a = &g_info->args[i];
            unsigned n = g_info->array_count[i] ? g_info->array_count[i] : 1;
            if (a->kind == kind && a->index < 16)
                used |= (n >= 16 ? 0xffffu : (1u << n) - 1) << a->index;
        }
        for (unsigned i = 0; i < g_info->nargs; i++) {
            struct nvk_arg *a = &g_info->args[i];
            unsigned n = g_info->array_count[i] ? g_info->array_count[i] : 1, slot = 0;
            if (a->kind != kind || a->index < 16)
                continue;
            uint32_t run = (1u << n) - 1;
            while (slot + n <= 16 - NVK_MAX_CONST_SAMPLERS && (used & run << slot))
                slot++;
            if (slot + n > 16 - NVK_MAX_CONST_SAMPLERS)
                airc_fail("%s %u: no pool slot left", kind == NVK_ARG_TEXTURE ? "texture" : "sampler", a->index);
            used |= run << slot;
            a->cbuf_off = (uint16_t)slot;
        }
    }
}

// the static initialisers (C++ global constructors, one per source file a library was built from), which copy
// function constants into globals: the first, or the one after another
static LLVMValueRef static_init(LLVMValueRef after)
{
    for (LLVMValueRef fn = after ? LLVMGetNextFunction(after) : LLVMGetFirstFunction(g_module); fn;
         fn = LLVMGetNextFunction(fn)) {
        const char *name = LLVMGetValueName(fn);
        if (name && !strncmp(name, "_GLOBAL__sub_I", 14) && LLVMGetFirstBasicBlock(fn))
            return fn;
    }
    return NULL;
}

void airc_front(void *module, void **linked, const char *const *linked_names, unsigned nlinked,
                const struct airc_varyings *vertex, const struct airc_stage_in *stage_in,
                const struct airc_constant *fc, unsigned nfc, int calls, struct ir_func *f, struct airc_info *info)
{
    g_calls = calls;
    g_subs = g_sub = NULL;
    g_vertex = vertex;
    g_stage_in = stage_in;
    g_region_off = 0;
    memset(g_stage_in_buffer, 0, sizeof(g_stage_in_buffer));
    memset(g_region, 0, sizeof(g_region));
    g_linked = (LLVMModuleRef *)linked;
    g_linked_names = linked_names;
    g_nlinked = nlinked;
    memset(g_vals, 0, sizeof(g_vals));
    memset(g_ptrs, 0, sizeof(g_ptrs));
    memset(g_blocks, 0, sizeof(g_blocks));
    memset(g_origins, 0, sizeof(g_origins));
    memset(g_highs, 0, sizeof(g_highs));
    g_module = module;
    g_f = f;
    g_info = info;
    g_fc = fc;
    g_nfc = nfc;
    g_scope = g_next_scope = g_depth = 0;
    g_rcp_w[0] = g_rcp_w[1] = NULL;
    g_point_fn = NULL;
    g_npoint_loc = 0;
    g_point_map = NULL;
    g_nshared = 0;
    g_nout = 0;
    g_phis = NULL;
    g_rets = NULL;
    memset(g_defs, 0, sizeof(g_defs));
    memset(g_entry_vals, 0, sizeof(g_entry_vals));
    g_nallocas = 0;
    g_private_reads = NULL;
    g_private_ptrs = NULL;
    g_local_choices = NULL;
    g_picked = NULL;

    LLVMValueRef arglist;
    g_entry_fn = stage_function(&arglist);
    stage_args(g_entry_fn, arglist);
    g_prologue = g_cur = ir_block(f);
    entry_args();

    // the static initialisers run first, inlined like any call (their stores fill the globals' state)
    for (LLVMValueRef init = static_init(NULL); init; init = static_init(init)) {
        g_depth = 1;  // its return goes on into the shader proper
        struct ir_block *saved = g_cont;
        g_cont = ir_block(f);
        g_rets = NULL;
        g_ret_type = NULL;
        unsigned caller = g_scope;
        g_scope = ++g_next_scope;
        struct ir_block *from = g_cur, *entry = body(init);
        g_scope = caller;
        ir_add(from, IR_BR, IR_VOID, 0);
        ir_edge(from, entry);
        g_cur = g_cont;
        g_cont = saved;
        g_depth = 0;
    }

    struct ir_block *from = g_cur, *entry = body(g_entry_fn);
    ir_add(from, IR_BR, IR_VOID, 0);
    ir_edge(from, entry);
    resolve_private();
}
