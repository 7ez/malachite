// airc's intermediate representation: scalar SSA in basic blocks. The front end splits every LLVM vector into
// one value per lane and inlines calls (all but noinline functions, which become subroutines), so everything after
// it sees scalar operations.
#pragma once
#include <stdint.h>

enum ir_type {
    IR_VOID,
    IR_PRED,  // a predicate (an LLVM i1)
    IR_I32,
    IR_F32,
    IR_F16,  // in the low half of a 32-bit register
    IR_I64,  // a global address, in an even register pair
};

enum ir_op {
    IR_CONST,  // imm (an i64's constant is imm too)
    IR_UNDEF,
    IR_CBUF,  // const bank 0 at imm (32 bits, or 64 for IR_I64), plus src0 bytes if it has a source
    IR_S2R,   // special register imm
    IR_ALD,   // vertex attribute at imm, of the vertex whose handle is src0 if it has one; aux IR_ATTR_* flags
    IR_IPA,   // pixel attribute at imm, interpolated as the shader header says
    IR_ISBERD,  // the handle of the input vertex at map offset src0

    IR_ADD,
    IR_SUB,
    IR_MUL,
    IR_MULHI,  // the high 32 bits of src0 * src1, aux: signed
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
    IR_FRND,   // round to an integral float, aux = SASS_RND_* mode
    IR_DERIV,  // the screen-space derivative of a fragment value, aux 0 for d/dx, 1 for d/dy
    IR_SHFL,   // src0 of another lane of the SIMD group: src1 says which, as SASS_SHFL_* mode aux has it (with
               // IR_SHFL_QUAD, of its quad)
    IR_POPC,
    IR_BREV,
    IR_FLO,   // the index of src0's top set bit, ~0 when there's none
    IR_VOTE,  // the lanes of the SIMD group where predicate src0 is set, as a mask

    IR_CMP,  // integers: aux = enum ir_cond | IR_SIGNED; floats: IR_FCMP | LLVM's real predicate (0 false .. 15 true)
    IR_SEL,  // src0 ? src1 : src2
    IR_PNOT,
    IR_PAND,
    IR_POR,

    IR_ADDR,  // IR_I64 = src0 + src1 * imm, src1 sign-extended when aux
    IR_ADDI,  // IR_I64 = src0 + imm

    IR_LDG,  // src0 address + imm, aux bytes (2 or 4); with src1, only where that predicate is set (else undefined)
    IR_STG,  // src0 address + imm <- src1
    IR_LDS,  // shared src0 (or none) + imm
    IR_STS,
    IR_LDL,  // the thread's local memory src0 (or none) + imm, aux bytes (8 or 16: through IR_PROJ)
    IR_STL,  // local src0 (or none) + imm <- src1.. (aux / 4 words when wide)
    IR_BAR,   // aux: a memory fence instead
    IR_ATOM,  // an atomic on [src0 + imm] (global; shared with aux IR_ATOMA_SHARED, src0 NULL for 0) of src1 (CAS:
              // compare src1, swap src2), the old value; aux the SASS_ATOM_* op and type

    IR_TEX,   // a texture op on the handle in its last source, aux IR_TEXA_* (below). The first source vector is an
              // array's index then the coordinates (TXD: the handle goes before them); the second, after the
              // handle, is the LOD or bias, the packed offset, the depth reference (TXD: the gradients). Four
              // results through IR_PROJ
    IR_PROJ,  // result imm of src0
    IR_TXQ,   // the size of texture handle src1 at LOD src0, aux the component mask: 1 width, 2 height, 4 depth,
              // 8 levels; with aux bit 4, the TEXTURE_TYPE query (4: the sample count)
    IR_SUST,  // the texel at coordinates src0.. (aux of them, an array's layer last) <- the next four, of the
              // texture whose handle is the last source; imm the SASS_IMG_* dimension
    IR_AST,   // vertex output at imm <- src0.. (aux & 15 values), of the vertex whose handle is the source after
              // them if there is one; IR_ATTR_PATCH in aux
    IR_PHI,   // one source per predecessor, in the block's predecessor order
    IR_KILL,  // discard the fragment; with src0, only if it's set (aux: if it's clear)

    IR_BR,
    IR_CBR,  // src0 ? succ[0] : succ[1]
    IR_EXIT,  // a fragment function's colour in src0..src3

    IR_PARAM,  // a subroutine's argument imm, in its entry block
    IR_CALL,   // subroutine imm on arguments src..; aux results, read through IR_PROJ
    IR_RET,    // a subroutine's return, with its results in src..
};

enum ir_cond { IR_EQ, IR_NE, IR_LT, IR_LE, IR_GT, IR_GE };

// IR_ALD's and IR_AST's flags: the patch's attributes, the stage's own outputs (ALD)
#define IR_ATTR_PATCH 16
#define IR_ATTR_OUTPUT 32

// IR_FADD's aux: the sum clamped to [0, 1], a NaN as 0 (FADD.SAT)
#define IR_FADD_SAT 1

// An i32 IR_ADD's or IR_SUB's aux, from LLVM's flags: the result doesn't wrap round as unsigned, or as signed
#define IR_NUW 1
#define IR_NSW 2

// IR_ATOM's aux: SASS_ATOM_* op, SASS_ATOM_* type << 4, shared memory
#define IR_ATOMA(op, type, shared) ((op) | (type) << 4 | (shared) << 8)
#define IR_ATOMA_OP(a) ((a) & 15)
#define IR_ATOMA_TYPE(a) ((a) >> 4 & 15)
#define IR_ATOMA_SHARED(a) ((a) >> 8 & 1)

// IR_TEX's aux: the SASS op (SASS_TEX/TLD/TLD4/TXD), the SASS_DIM_* dimension, the SASS_TEXF_* flags, how many
// of its sources make the first vector, and the result components wanted
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
    int vreg;                      // isel's virtual register, -1 before
    struct ir_block **phi_blocks;  // for a phi: src[i] comes from predecessor phi_blocks[i]
    struct ir_val *fwd;            // opt: the value that stands in for this one, until its users are repointed
};

struct ir_block {
    int id;
    struct ir_val *first, *last;
    struct ir_block **preds;
    unsigned npreds, cappreds;
    struct ir_block *succ[2];
    unsigned nsucc;
    struct ir_block *next;  // in creation order; opt puts them in reverse postorder
    int dead;
    int func;  // 0 the shader's own, i + 1 subroutine i's (as opt finds them)
    int unrolled;  // opt: a loop left after unrolling it (1) or made of its copies (2), not unrolled again
};

// Subroutines: the functions called rather than inlined, each a body of its own with no edge into it. Their
// blocks may read the shader's values (its arguments and the like), which then stay live across every call.
#define IR_MAX_SUBS 16

struct ir_func {
    struct ir_block *entry, *last;
    int nblocks, nvals;
    struct ir_block *subs[IR_MAX_SUBS];  // each one's entry; opt puts their blocks after the shader's own
    int nsubs;
};

// One arena per compile: every block and value lives until ir_reset.
void ir_reset(void);
void *ir_alloc(unsigned bytes);

struct ir_block *ir_block(struct ir_func *f);
// Append a value to b, before its terminator if it has one.
struct ir_val *ir_add(struct ir_block *b, enum ir_op op, enum ir_type type, unsigned nsrc);
struct ir_val *ir_op2(struct ir_block *b, enum ir_op op, enum ir_type type, struct ir_val *x, struct ir_val *y);
struct ir_val *ir_const(struct ir_block *b, enum ir_type type, int64_t imm);
void ir_edge(struct ir_block *from, struct ir_block *to);
void ir_remove(struct ir_val *v);
void ir_print(const struct ir_func *f);
