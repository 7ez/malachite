// GA10x (SM86) SASS encoder: raw 128-bit instruction encodings, each reversed from nvcc -arch=sm_86 output and
// checked byte-for-byte against it (see sass86_test.c); nvdisasm -b SM86 round-trips the bytes.
//
// Instruction word layout (little-endian, lo = bits 0..63, hi = bits 64..127):
//   [0:11]    opcode
//   [12:15]   guard predicate (7 = PT, always true)
//   [16:23]   Rd    (destination register; 255 = RZ)
//   [24:31]   Ra    (first source register; 255 = RZ)
//   [32:63]   immediate, OR const-bank operand (bank at [32:39], offset>>2 at [40:53])
//   [64:71]   Rb/Rc (further source register; 255 = RZ)
//   [72:95]   per-opcode modifiers
//   [96:127]  scheduling control (see struct sass_ctrl)
#pragma once
#include <stdint.h>
#include <stddef.h>

#define SASS_RZ 255   // the zero register
#define SASS_URZ 63   // the uniform zero register
#define SASS_NO_BARRIER (-1)

// A decoded scheduling control word (bits 96:127 of the instruction):
//   stall [105:108], yield [109], write barrier [110:112], read barrier [113:115], wait mask [116:121].
// Variable-latency instructions (S2R, LDG, ...) allocate a barrier in wr; a consumer waits on it via the
// wait mask. 7 (or -1 here) means "no barrier". sass_ctrl_word() packs this into the raw word the emitters
// take; sched() in back.c builds these from data dependencies.
struct sass_ctrl {
    uint8_t stall;   // 0..15 cycles until the next instruction may issue
    uint8_t yield;   // 0/1
    int8_t wr, rd;   // barrier to allocate, or SASS_NO_BARRIER
    uint8_t wait;    // 6-bit mask of barriers to wait on
};
uint32_t sass_ctrl_word(struct sass_ctrl c);

// Raw control words (bits 96:127) as nvcc emits them for straight-line integer code.
#define CTRL_STALL(n) (0x000fc000u | ((uint32_t)((n) & 0xf) << 9) | 0x2000u)  // stall n, yield
#define CTRL_STALL1 0x000fc000u          // stall 0, no yield (NOP/BRA)
#define CTRL_EXIT 0x000fea00u            // stall 5 (EXIT)
#define CTRL_RD0 0x000e2200u             // sets write barrier 0 (variable-latency result, e.g. S2R)
#define CTRL_WAIT0 0x001fca00u           // waits on barrier 0, stall 5 (consumer of S2R)

struct sass_insn {
    uint64_t lo, hi;
};

// emitters: ctrl is the raw scheduling word, bits 96:127
struct sass_insn sass_mov_imm(uint8_t rd, uint32_t imm, uint32_t ctrl);
struct sass_insn sass_mov_cbuf(uint8_t rd, uint8_t bank, uint16_t off, uint32_t ctrl);
struct sass_insn sass_imadmov_imm(uint8_t rd, uint32_t imm, uint32_t ctrl);   // IMAD.MOV.U32 Rd,RZ,RZ,imm
struct sass_insn sass_imadmov_cbuf(uint8_t rd, uint8_t bank, uint16_t off, uint32_t ctrl);
struct sass_insn sass_imadwide_cbuf(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t bank, uint16_t off,
                                    uint32_t ctrl);  // IMAD.WIDE.U32 Rd,Ra,Rb,c[bank][off]
struct sass_insn sass_iadd3_cbuf(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint32_t ctrl);
struct sass_insn sass_iadd3_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl);   // IADD3 Rd,Ra,Rb,RZ
struct sass_insn sass_isub_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl);    // IADD3 Rd,Ra,-Rb,RZ
struct sass_insn sass_imad_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, uint32_t ctrl);  // IMAD Rd,Ra,Rb,Rc
struct sass_insn sass_imadhi_rr(uint8_t rd, uint8_t ra, uint8_t rb, int is_signed,
                                uint32_t ctrl);  // IMAD.HI[.U32] Rd,Ra,Rb,RZ
struct sass_insn sass_imad_cbuf(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint8_t rc,
                                uint32_t ctrl);  // IMAD Rd,Ra,c[bank][off],Rc
struct sass_insn sass_lop3(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t lut, uint32_t ctrl);  // LOP3.LUT
struct sass_insn sass_prmt(uint8_t rd, uint8_t ra, uint16_t sel, uint8_t rc, uint32_t ctrl);  // PRMT Rd,Ra,sel,Rc
struct sass_insn sass_shf_l(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl);     // SHF.L.U32 Rd,val,sh,RZ
struct sass_insn sass_shf_r(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl);     // SHF.R.U32.HI Rd,RZ,sh,val
struct sass_insn sass_shf_r_s(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl);   // SHF.R.S32.HI (arithmetic)
struct sass_insn sass_imadwide_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, int is_signed, uint32_t ctrl);
struct sass_insn sass_fadd(uint8_t rd, uint8_t ra, uint8_t rb, int neg_b, uint32_t ctrl);
struct sass_insn sass_fmul(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl);
struct sass_insn sass_ffma(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, int mode, uint32_t ctrl);  // SASS_RND_*
struct sass_insn sass_fmnmx(uint8_t rd, uint8_t ra, uint8_t rb, int is_max, uint32_t ctrl);
struct sass_insn sass_imnmx(uint8_t rd, uint8_t ra, uint8_t rb, int is_max, int is_signed,
                            uint32_t ctrl);  // IMNMX[.U32]
struct sass_insn sass_mufu(uint8_t rd, uint8_t ra, uint8_t op, uint32_t ctrl);  // MUFU Rd, Ra (op = MUFU_*)
struct sass_insn sass_popc(uint8_t rd, uint8_t rs, uint32_t ctrl);  // POPC Rd, Rs
struct sass_insn sass_brev(uint8_t rd, uint8_t rs, uint32_t ctrl);  // BREV Rd, Rs
struct sass_insn sass_flo(uint8_t rd, uint8_t rs, uint32_t ctrl);   // FLO.U32 Rd, Rs (the top set bit, ~0 for none)
struct sass_insn sass_vote_any(uint8_t rd, uint8_t pred, uint32_t ctrl);  // VOTE.ANY Rd, PT, Pp
struct sass_insn sass_i2f(uint8_t rd, uint8_t rs, int is_signed, uint32_t ctrl);  // I2FP.F32.{S32,U32}
struct sass_insn sass_f2i(uint8_t rd, uint8_t rs, int is_signed, uint32_t ctrl);  // F2I[.U32].TRUNC.NTZ
// F2IP.U8.F32.TRUNC.NTZ: a float to a byte, saturated (a NaN as 0) and zero-extended; fixed latency, unlike F2I
struct sass_insn sass_f2ip_u8(uint8_t rd, uint8_t rs, uint32_t ctrl);
struct sass_insn sass_frnd(uint8_t rd, uint8_t rs, int mode, uint32_t ctrl);       // FRND.FTZ (mode SASS_RND_*)
struct sass_insn sass_kill(uint8_t pred, uint32_t ctrl);  // @Pp KILL (discard), p 7 = PT
enum { SASS_SHFL_IDX, SASS_SHFL_UP, SASS_SHFL_DOWN, SASS_SHFL_BFLY };
struct sass_insn sass_shfl_bfly(uint8_t rd, uint8_t rs, uint8_t lane, uint16_t c, uint32_t ctrl);
struct sass_insn sass_shfl(int mode, uint8_t rd, uint8_t rs, uint8_t lane, uint16_t c, uint32_t ctrl);
struct sass_insn sass_shfl_r(int mode, uint8_t rd, uint8_t rs, uint8_t rl, uint16_t c, uint32_t ctrl);
struct sass_insn sass_fswzadd(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t subop, uint32_t ctrl);
// FSWZADD patterns for quad derivatives (after a SHFL.BFLY by 1 or 2): Ra is the neighbour, Rb the pixel
#define FSWZADD_DDX 0x66
#define FSWZADD_DDY 0x5a
#define SASS_RND_NEAREST 0
#define SASS_RND_FLOOR 1
#define SASS_RND_CEIL 2
#define SASS_RND_TRUNC 3
struct sass_insn sass_fadd_imm(uint8_t rd, uint8_t ra, uint32_t imm, uint32_t ctrl);   // FADD Rd, Ra, imm
struct sass_insn sass_fmul_imm(uint8_t rd, uint8_t ra, uint32_t imm, uint32_t ctrl);   // FMUL Rd, Ra, imm
// FFMA Rd, Ra, imm, Rc and FFMA Rd, Ra, Rb, imm (mode SASS_RND_*)
struct sass_insn sass_ffma_imm_b(uint8_t rd, uint8_t ra, uint32_t imm, uint8_t rc, int mode, uint32_t ctrl);
struct sass_insn sass_ffma_imm_c(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t imm, int mode, uint32_t ctrl);
struct sass_insn sass_fmul_rz_imm(uint8_t rd, uint8_t ra, uint32_t imm, uint32_t ctrl);  // FMUL.RZ Rd, Ra, imm
// ISETP.<op>[.U32].AND Pd, PT, Ra, Rb, PT  -- integer compare into predicate Pd (op = ISETP_*)
struct sass_insn sass_isetp(uint8_t pd, uint8_t ra, uint8_t rb, uint8_t op, int is_signed, uint32_t ctrl);
// ISETP.<op>.U32.OR Pd, PT, Ra, Rb, Pc  -- the compare OR'd into Pc, e.g. to accumulate a flag
struct sass_insn sass_isetp_or(uint8_t pd, uint8_t ra, uint8_t rb, uint8_t op, uint8_t pc, uint32_t ctrl);
// SEL Rd, Ra, Rb, Pn -- Rd = Pn ? Ra : Rb (a bit select, so floats too)
struct sass_insn sass_sel(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t pred, uint32_t ctrl);
// FSETP.<op>.AND Pd, PT, Ra, Rb, PT  -- float compare; op = ISETP_* for the ordered tests, FSETP_* below
struct sass_insn sass_fsetp(uint8_t pd, uint8_t ra, uint8_t rb, uint8_t op, uint32_t ctrl);
// ISETP against a const-bank operand: P0 = (Ra <op> c[bank][off]) -- avoids a register RAW on the operand
struct sass_insn sass_isetp_cbuf(uint8_t ra, uint8_t bank, uint16_t off, uint8_t op, int is_signed,
                                 uint32_t ctrl);
// The immediate form of an integer, logic, shift, select or compare instruction built in its register form (0x2xx
// opcodes, a register in Rb): its 32-bit immediate takes Rb's place, in opcode 0x8xx
struct sass_insn sass_imm(struct sass_insn in, uint32_t imm);
// Set an instruction's guard predicate (bits 12:15): predicate index 0..6, or negated (@!Pn).
struct sass_insn sass_pred(struct sass_insn in, uint8_t pred, int negate);
// .FTZ (bit 80) on FADD, FMUL, FFMA, FSETP, FRND, FMNMX or F2I: denormal inputs and results become zero
struct sass_insn sass_ftz(struct sass_insn in);
// Source modifiers on FADD, FMUL, FFMA, FMNMX and FSETP, as nvdisasm decodes them: A is bits 72 (neg) and 73 (abs);
// B in the register forms is 63 and 62, and C (or an FFMA's register B beside an immediate C) 75 and 74. Neg
// toggles what is set already; neg with abs is -|x|.
enum { SASS_MOD_A, SASS_MOD_B, SASS_MOD_C };
struct sass_insn sass_src_mod(struct sass_insn in, int slot, int neg, int abs);
// -Ra (bit 72) on FADD: the first source negated
struct sass_insn sass_neg_a(struct sass_insn in);
// .SAT (bit 77) on FADD, FMUL or FFMA: the result clamped to [0, 1], a NaN becoming 0
struct sass_insn sass_sat(struct sass_insn in);
// LUT values for LOP3.LUT with operands (a, b): AND/OR/XOR
#define LOP3_AND 0xc0
#define LOP3_OR 0xfc
#define LOP3_XOR 0x3c
// ISETP comparison operators
#define ISETP_LT 1
#define ISETP_EQ 2
#define ISETP_LE 3
#define ISETP_GT 4
#define ISETP_NE 5
#define ISETP_GE 6
// FSETP-only tests: NUM/NAN (ordered/unordered), and the unordered-or-<op> forms
#define FSETP_F 0
#define FSETP_NUM 7
#define FSETP_NAN 8
#define FSETP_LTU 9
#define FSETP_EQU 10
#define FSETP_LEU 11
#define FSETP_GTU 12
#define FSETP_NEU 13
#define FSETP_GEU 14
#define FSETP_T 15
// MUFU operation selectors
#define MUFU_COS 0
#define MUFU_SIN 1
#define MUFU_EX2 2
#define MUFU_LG2 3
#define MUFU_RCP 4
#define MUFU_RSQ 5
#define MUFU_SQRT 8
struct sass_insn sass_s2r(uint8_t rd, uint8_t sr, uint32_t ctrl);  // S2R Rd, SR<sr>
#define SR_LANEID 0x00
#define SR_INVOCATION_ID 0x11    // a tessellation init thread's control point
#define SR_INVOCATION_INFO 0x1d  // bytes 0 and 2 multiplied: where its patch's vertices start in the input map
#define SR_TID_X 0x21
#define SR_TID_Y 0x22
#define SR_TID_Z 0x23
#define SR_CTAID_X 0x25
#define SR_CTAID_Y 0x26
#define SR_CTAID_Z 0x27
#define SR_PIXLD(v) (0x100 | (v))  // no special register: PIXLD's SASS_PIXLD_* value v, read the way one is
struct sass_insn sass_uldc64_cbuf(uint8_t urd, uint8_t bank, uint16_t off, uint32_t ctrl);
// global memory access sizes (LDG/STG hi bits 73:75)
#define SASS_U8 0
#define SASS_U16 2
#define SASS_B32 4
#define SASS_B64 5
#define SASS_B128 6
struct sass_insn sass_ldg(uint8_t rd, uint8_t ra, uint32_t off, uint8_t size, uint32_t ctrl);  // LDG.E Rd, [Ra.64+off]
struct sass_insn sass_stg(uint8_t ra, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl);  // STG.E [Ra.64+off], Rb
// LDG.E Rd, [Ra.U32+URu+off] and STG.E [Ra.U32+URu+off], Rb: a uniform pair as the base, Ra .64 when wide (RZ for
// none)
struct sass_insn sass_ldg_ur(uint8_t rd, uint8_t ra, int wide, uint8_t ur, uint32_t off, uint8_t size, uint32_t ctrl);
struct sass_insn sass_stg_ur(uint8_t ra, int wide, uint8_t ur, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl);
// .CONSTANT on LDG: through the read-only (texture) cache, as __ldg loads; the memory mustn't change meanwhile
struct sass_insn sass_ldg_constant(struct sass_insn in);
struct sass_insn sass_lds(uint8_t rd, uint8_t ra, uint32_t off, uint8_t size, uint32_t ctrl);  // LDS Rd, [Ra+off]
struct sass_insn sass_sts(uint8_t ra, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl);  // STS [Ra+off], Rb
// the thread's local memory: Ra a byte offset into it (RZ for none), off a signed 24-bit one
struct sass_insn sass_ldl(uint8_t rd, uint8_t ra, uint32_t off, uint8_t size, uint32_t ctrl);  // LDL Rd, [Ra+off]
struct sass_insn sass_stl(uint8_t ra, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl);  // STL [Ra+off], Rb
struct sass_insn sass_ldc(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint32_t ctrl);  // LDC Rd, c[b][Ra+off]
struct sass_insn sass_bar_sync(uint32_t ctrl);  // BAR.SYNC 0x0 (threadgroup barrier)
struct sass_insn sass_membar_gpu(uint32_t ctrl);  // MEMBAR.SC.GPU
struct sass_insn sass_errbar(uint32_t ctrl);
struct sass_insn sass_cctl_ivall(uint32_t ctrl);  // CCTL.IVALL: invalidate the L1
// atomics: SASS_ATOM_* ops (or SASS_ATOM_CAS), on global memory ([Ra.64 + off]) or shared ([Ra + off]), of
// SASS_ATOM_U32/I32/F32 data in Rb (CAS: compare with Rb, swap in Rc); the old value to Rd
enum { SASS_ATOM_ADD, SASS_ATOM_MIN, SASS_ATOM_MAX, SASS_ATOM_INC, SASS_ATOM_DEC, SASS_ATOM_AND, SASS_ATOM_OR,
       SASS_ATOM_XOR, SASS_ATOM_EXCH, SASS_ATOM_CAS };
enum { SASS_ATOM_U32 = 0, SASS_ATOM_I32 = 1, SASS_ATOM_F32 = 3 };
struct sass_insn sass_atom(int op, int shared, int type, uint8_t rd, uint8_t ra, uint32_t off, uint8_t rb, uint8_t rc,
                           uint32_t ctrl);
// RED: a global atomic (not EXCH or CAS) that returns nothing, so no write barrier
struct sass_insn sass_red(int op, int type, uint8_t ra, uint32_t off, uint8_t rb, uint32_t ctrl);
// texture dimensions (TEX/TLD/TLD4/TXD), and image ones (SUST)
enum {
    SASS_DIM_1D = 0, SASS_DIM_2D = 1, SASS_DIM_3D = 2, SASS_DIM_CUBE = 3, SASS_DIM_ARRAY_1D = 4,
    SASS_DIM_ARRAY_2D = 5, SASS_DIM_ARRAY_CUBE = 7,
};
enum { SASS_IMG_1D = 0, SASS_IMG_ARRAY_1D = 2, SASS_IMG_2D = 3, SASS_IMG_ARRAY_2D = 4, SASS_IMG_3D = 5 };
// texture ops, by opcode, and their LOD modes
enum { SASS_TEX = 0x361, SASS_TLD4 = 0x364, SASS_TLD = 0x367, SASS_TXD = 0x36d };
enum { SASS_LOD_AUTO, SASS_LOD_ZERO, SASS_LOD_BIAS, SASS_LOD_LOD, SASS_LOD_CLAMP, SASS_LOD_BIAS_CLAMP };
// a texture op's options: the LOD mode, then AOFFI, depth compare and a gather's component as bits
#define SASS_TEXF_LOD(f) ((f) & 7)
#define SASS_TEXF_AOFFI 8
#define SASS_TEXF_DC 16
#define SASS_TEXF_COMP(c) ((unsigned)(c) << 5)
#define SASS_TEXF_MS 128  // TLD of a sample of a multisampled texture, the sample index last in the second vector
struct sass_insn sass_texop(int op, uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, unsigned flags,
                            uint32_t ctrl);  // bindless TEX/TLD/TLD4/TXD
struct sass_insn sass_tld(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, uint32_t ctrl);  // TLD.LZ
struct sass_insn sass_tex(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, uint32_t ctrl);  // TEX.LZ
struct sass_insn sass_txq(uint8_t rd, uint8_t ra, uint8_t mask, int query, uint32_t ctrl);  // bindless query
struct sass_insn sass_sust(uint8_t ra, uint8_t rb, uint8_t rc, uint8_t mask, int dim, uint32_t ctrl);  // formatted store
// attribute load and store: of the vertex whose handle is in vtx (RZ for the shader's own), or with patch of the
// patch's attributes; output loads what the stage itself wrote
struct sass_insn sass_ald(uint8_t rd, uint8_t vtx, uint16_t addr, unsigned comps, int patch, int output,
                          uint32_t ctrl);
struct sass_insn sass_ast(uint16_t addr, uint8_t rs, uint8_t vtx, unsigned comps, int patch, uint32_t ctrl);
struct sass_insn sass_isberd(uint8_t rd, uint8_t ra, uint32_t ctrl);  // ISBERD Rd, [Ra]: an input vertex's handle
#define SASS_IPA_FLAT 1
#define SASS_IPA_CENTROID 2
struct sass_insn sass_ipa(uint8_t rd, uint16_t addr, int mode, uint32_t ctrl);
#define SASS_PIXLD_COVMASK 1  // the samples of the pixel the primitive covers
#define SASS_PIXLD_MY_INDEX 3  // the sample a pass of one sample shades
struct sass_insn sass_pixld(uint8_t rd, unsigned val, uint32_t ctrl);
// half precision, on the low 16 bits of each register
struct sass_insn sass_hadd2(uint8_t rd, uint8_t ra, uint8_t rb, int neg_b, uint32_t ctrl);
struct sass_insn sass_hmul2(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl);
struct sass_insn sass_hfma2(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, uint32_t ctrl);
struct sass_insn sass_h2f(uint8_t rd, uint8_t rs, uint32_t ctrl);  // HADD2.F32: half -> float
struct sass_insn sass_f2h(uint8_t rd, uint8_t rs, uint32_t ctrl);  // F2FP.PACK_AB: float -> half
// F2FP.PACK_AB Rd, Rhi, Rlo: two floats into the two halves of Rd; and the same HADD2.F32 reading the high half
struct sass_insn sass_f2h2(uint8_t rd, uint8_t lo, uint8_t hi, uint32_t ctrl);
struct sass_insn sass_h2f_hi(struct sass_insn in);
struct sass_insn sass_mov_rr(uint8_t rd, uint8_t rsrc, uint32_t ctrl);  // Rd = Rsrc (via IMAD.MOV.U32)
struct sass_insn sass_exit(uint32_t ctrl);
struct sass_insn sass_bra_self(uint32_t ctrl);   // BRA to itself (traps at end of program)
// BRA with a byte-relative offset (target - (this_insn + 0x10)); guard predicate via pred/negate (PT=7).
struct sass_insn sass_bra(int32_t rel, uint8_t pred, int negate, uint32_t ctrl);
// CALL.REL.NOINC to a byte-relative offset (as for BRA); the callee returns with RET.REL.NODEC Ra, rel: to
// the address in the even pair Ra (high word 0), rel the offset from the next instruction to where that address
// counts from
struct sass_insn sass_call_rel(int32_t rel, uint32_t ctrl);
struct sass_insn sass_ret_rel(uint8_t ra, int32_t rel, uint32_t ctrl);
// BSSY Bbar, <reconverge>: the threads here meet again at the target (offset as for BRA), at its BSYNC Bbar
struct sass_insn sass_bssy(uint8_t bar, int32_t rel, uint32_t ctrl);
// BSYNC Bbar: wait there for the threads the barrier was set for
struct sass_insn sass_bsync(uint8_t bar, uint32_t ctrl);
struct sass_insn sass_nop(uint32_t ctrl);

void sass_emit(const struct sass_insn *insn, size_t n, uint8_t *out);
