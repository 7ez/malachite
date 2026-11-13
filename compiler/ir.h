// Scalar SSA: the front end splits vectors into lanes and inlines all but noinline calls
#pragma once
#include <stdint.h>

enum ir_type {
    IR_VOID,
    IR_PRED,
    IR_I32,
    IR_F32,
    IR_F16,  // in the low half of a 32-bit register
    IR_I64,  // in an even register pair
};

enum ir_op {
    IR_CONST,  // imm
    IR_UNDEF,
    IR_CBUF,  // const bank 0 at imm, plus src0 bytes if it has a source
    IR_S2R,   // special register imm
    IR_ALD,   // vertex attribute at imm, of the vertex whose handle is src0 if any; aux IR_ATTR_* flags
    IR_IPA,   // pixel attribute at imm, interpolated as the shader header says
    IR_ISBERD,  // the handle of the input vertex at map offset src0

    IR_ADD,
    IR_SUB,
    IR_MUL,
    IR_MULHI,  // aux: signed
    IR_AND,
    IR_OR,
    IR_XOR,
    IR_SHL,
    IR_SHR,  // logical
    IR_SAR,
    IR_MIN,  // aux: signed
    IR_MAX,

    IR_FADD,
    IR_FSUB,
    IR_FMUL,
    IR_FFMA,  // aux: SASS_RND_* rounding (f32 only)
    IR_FMIN,
    IR_FMAX,
    IR_FNEG,
    IR_FABS,
    IR_MUFU,  // aux: MUFU_* (sin/cos take revolutions)
    IR_I2F,   // aux: signed source
    IR_F2I,   // aux: signed result
    IR_F2H,
    IR_H2F,
    IR_FRND,   // aux SASS_RND_* mode
    IR_DERIV,  // aux 0 for d/dx, 1 for d/dy
    IR_SHFL,   // src0 from lane src1 per SASS_SHFL_* mode aux, within the quad with IR_SHFL_QUAD
    IR_POPC,
    IR_BREV,
    IR_FLO,   // ~0 when no bit is set
    IR_VOTE,  // mask of lanes where predicate src0 is set

    IR_CMP,  // aux enum ir_cond | IR_SIGNED, or IR_FCMP | LLVM's float predicate (0 false .. 15 true)
    IR_SEL,  // src0 ? src1 : src2
    IR_PNOT,
    IR_PAND,
    IR_POR,

    IR_ADDR,  // IR_I64 = src0 + src1 * imm, src1 sign-extended when aux
    IR_ADDI,  // IR_I64 = src0 + imm

    IR_LDG,  // src0 + imm, aux bytes (2 or 4); a src1 predicate leaves clear lanes undefined
    IR_STG,  // src0 + imm <- src1
    IR_LDS,  // src0 (or none) + imm
    IR_STS,
    IR_LDL,  // src0 (or none) + imm, aux bytes (8 or 16 through IR_PROJ)
    IR_STL,  // src0 (or none) + imm <- src1.., aux / 4 words when wide
    IR_BAR,   // aux: a memory fence instead
    IR_ATOM,  // old value of [src0 + imm] (src0 NULL for 0) op src1, CAS swaps src2 in; aux IR_ATOMA

    // Handle in the last source, aux IR_TEXA. First vector: array index then coordinates (TXD: after the handle);
    // second, after the handle: LOD or bias, packed offset, depth reference (TXD: gradients). Results via IR_PROJ.
    IR_TEX,
    IR_PROJ,  // result imm of src0
    IR_TXQ,   // handle src1 at LOD src0, aux mask 1 width, 2 height, 4 depth, 8 levels; bit 4 TEXTURE_TYPE (4 samples)
    IR_SUST,  // aux coordinates from src0 (layer last), then four values, then the handle; imm SASS_IMG_* dimension
    IR_AST,   // output at imm <- aux & 15 values, then the vertex handle if any; IR_ATTR_PATCH in aux
    IR_PHI,   // in the block's predecessor order
    IR_KILL,  // with src0, only if it's set (aux: if it's clear)

    IR_BR,
    IR_CBR,  // src0 ? succ[0] : succ[1]
    IR_EXIT,  // a fragment function's colour in src0..src3

    IR_PARAM,  // argument imm, in the subroutine's entry block
    IR_CALL,   // subroutine imm; aux results, read through IR_PROJ
    IR_RET,
};

enum ir_cond { IR_EQ, IR_NE, IR_LT, IR_LE, IR_GT, IR_GE };

// IR_ATTR_OUTPUT reads the stage's own outputs, ALD only
#define IR_ATTR_PATCH 16
#define IR_ATTR_OUTPUT 32

// FADD.SAT, which also turns a NaN into 0
#define IR_FADD_SAT 1

// LLVM's nuw and nsw on an i32 add or sub
#define IR_NUW 1
#define IR_NSW 2

#define IR_ATOMA(op, type, shared) ((op) | (type) << 4 | (shared) << 8)
#define IR_ATOMA_OP(a) ((a) & 15)
#define IR_ATOMA_TYPE(a) ((a) >> 4 & 15)
#define IR_ATOMA_SHARED(a) ((a) >> 8 & 1)

// op is SASS_TEX/TLD/TLD4/TXD; n0 is the first source vector's length
#define IR_TEXA(op, dim, flags, n0) ((op) | (dim) << 12 | (int)(flags) << 15 | (n0) << 24)
#define IR_TEXA_SCALAR (1 << 27)
#define IR_TEXA_OP(a) ((a) & 0xfff)
#define IR_TEXA_DIM(a) ((a) >> 12 & 7)
#define IR_TEXA_FLAGS(a) ((unsigned)(a) >> 15 & 0x1ff)
#define IR_TEXA_N0(a) ((a) >> 24 & 7)
#define IR_SHFL_QUAD 4
#define IR_SIGNED 8
#define IR_FCMP 32

struct ir_block;

struct ir_val {
    enum ir_op op;
    enum ir_type type;
    int aux;
    int64_t imm;
    unsigned nsrc;
    struct ir_val **src;
    struct ir_block *block;
    struct ir_val *prev, *next;
    unsigned uses;
    int id;
    int vreg;                      // -1 before isel
    struct ir_block **phi_blocks;  // src[i] comes from phi_blocks[i]
    struct ir_val *fwd;            // opt's replacement, until its users are repointed
};

struct ir_block {
    int id;
    struct ir_val *first, *last;
    struct ir_block **preds;
    unsigned npreds, cappreds;
    struct ir_block *succ[2];
    unsigned nsucc;
    struct ir_block *next;  // creation order until opt sorts them in reverse postorder
    int dead;
    int func;  // 0 the shader's own, i + 1 subroutine i's
    int unrolled;  // 1 the loop left after unrolling, 2 one of its copies; never unrolled again
};

// Subroutine blocks may read the shader's values, which then stay live across every call.
#define IR_MAX_SUBS 16

struct ir_func {
    struct ir_block *entry, *last;
    int nblocks, nvals;
    struct ir_block *subs[IR_MAX_SUBS];  // entries; opt puts their blocks after the shader's own
    int nsubs;
};

// every block and value lives until ir_reset
void ir_reset(void);
void *ir_alloc(unsigned bytes);

struct ir_block *ir_block(struct ir_func *f);
// before b's terminator if it has one
struct ir_val *ir_add(struct ir_block *b, enum ir_op op, enum ir_type type, unsigned nsrc);
struct ir_val *ir_op2(struct ir_block *b, enum ir_op op, enum ir_type type, struct ir_val *x, struct ir_val *y);
struct ir_val *ir_const(struct ir_block *b, enum ir_type type, int64_t imm);
void ir_edge(struct ir_block *from, struct ir_block *to);
void ir_remove(struct ir_val *v);
void ir_print(const struct ir_func *f);
