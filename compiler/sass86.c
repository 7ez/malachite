// SM86 SASS encoder. Operand layouts reversed from nvcc -arch=sm_86 (see sass86_test.c for the checks).
// Each emitter starts from the opcode/modifier template for its instruction form and ORs in the operands.
#include "sass86.h"

// const-bank operand into bits [32:63]: bank at [32:39], (off>>2) at [40:53]
static uint64_t cbuf_operand(uint8_t bank, uint16_t off)
{
    return ((uint64_t)bank << 32) | ((uint64_t)((off >> 2) & 0x3fff) << 40);
}

static struct sass_insn mk(uint64_t lo, uint32_t hi_lo32, uint32_t ctrl)
{
    struct sass_insn in;
    in.lo = lo;
    in.hi = (uint64_t)hi_lo32 | ((uint64_t)ctrl << 32);
    return in;
}

struct sass_insn sass_mov_imm(uint8_t rd, uint32_t imm, uint32_t ctrl)
{
    return mk(0x00007802ull | ((uint64_t)rd << 16) | ((uint64_t)imm << 32), 0x00000f00, ctrl);
}

struct sass_insn sass_mov_cbuf(uint8_t rd, uint8_t bank, uint16_t off, uint32_t ctrl)
{
    return mk(0x00007a02ull | ((uint64_t)rd << 16) | cbuf_operand(bank, off), 0x00000f00, ctrl);
}

struct sass_insn sass_imadmov_imm(uint8_t rd, uint32_t imm, uint32_t ctrl)
{
    // IMAD.MOV.U32 Rd, RZ, RZ, imm  -- Ra = RZ (0xff), Rb = RZ (0xff in hi)
    return mk(0x0000ff007424ull | ((uint64_t)rd << 16) | ((uint64_t)imm << 32), 0x078e00ff, ctrl);
}

struct sass_insn sass_imadmov_cbuf(uint8_t rd, uint8_t bank, uint16_t off, uint32_t ctrl)
{
    return mk(0x0000ff007624ull | ((uint64_t)rd << 16) | cbuf_operand(bank, off), 0x078e00ff, ctrl);
}

struct sass_insn sass_imadwide_cbuf(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t bank, uint16_t off,
                                    uint32_t ctrl)
{
    // IMAD.WIDE.U32 Rd, Ra, Rb, c[bank][off]  -- Rb goes in hi byte 8
    return mk(0x00007625ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | cbuf_operand(bank, off),
              0x078e0000 | rb, ctrl);
}

struct sass_insn sass_iadd3_cbuf(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint32_t ctrl)
{
    // IADD3 Rd, Ra, c[bank][off], RZ  -- Rc = RZ (0xff in hi byte 8)
    return mk(0x00007a10ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | cbuf_operand(bank, off),
              0x07ffe0ff, ctrl);
}

struct sass_insn sass_iadd3_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl)
{
    // IADD3 Rd, Ra, Rb, RZ  -- Rb at byte 4, Rc = RZ (0xff in hi byte 8)
    return mk(0x00007210ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x07ffe0ff, ctrl);
}

struct sass_insn sass_iadd3_rrr(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, uint32_t ctrl)
{
    return mk(0x00007210ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x07ffe000 | rc, ctrl);
}

struct sass_insn sass_isub_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl)
{
    // IADD3 Rd, Ra, -Rb, RZ  -- negate-Rb flag at bit 63
    return mk(0x00007210ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  (1ull << 63),
              0x07ffe0ff, ctrl);
}

struct sass_insn sass_imad_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, uint32_t ctrl)
{
    // IMAD Rd, Ra, Rb, Rc  -- 32-bit multiply-add; Rc in hi byte 8
    return mk(0x00007224ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x078e0200 | rc, ctrl);
}

struct sass_insn sass_imadhi_rr(uint8_t rd, uint8_t ra, uint8_t rb, int is_signed, uint32_t ctrl)
{
    // IMAD.HI[.U32] Rd, Ra, Rb, RZ  -- signed at bit 73
    return mk(0x00007227ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x078e00ff | (is_signed ? 0x200 : 0), ctrl);
}

struct sass_insn sass_imad_cbuf(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint8_t rc, uint32_t ctrl)
{
    return mk(0x00007a24ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | cbuf_operand(bank, off),
              0x078e0200 | rc, ctrl);
}

// PRMT Rd, Ra, sel, Rc -- byte i of Rd is byte (nibble i of sel) of Rc:Ra (0-3 Ra's, 4-7 Rc's), its sign
// repeated when the nibble's top bit is set
struct sass_insn sass_prmt(uint8_t rd, uint8_t ra, uint16_t sel, uint8_t rc, uint32_t ctrl)
{
    return mk(0x00007816ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)sel << 32), rc, ctrl);
}

struct sass_insn sass_lop3(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t lut, uint32_t ctrl)
{
    // LOP3.LUT Rd, Ra, Rb, RZ, lut, !PT  -- bitwise op selected by the 8-bit LUT (hi byte 9); Rc = RZ
    return mk(0x00007212ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x078e00ff | ((uint32_t)lut << 8), ctrl);
}

struct sass_insn sass_shf_l(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl)
{
    // SHF.L.U32 Rd, val, sh, RZ  -- logical left shift
    return mk(0x00007219ull | ((uint64_t)rd << 16) | ((uint64_t)val << 24) | ((uint64_t)sh << 32),
              0x000006ff, ctrl);
}

struct sass_insn sass_shf_r(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl)
{
    // SHF.R.U32.HI Rd, RZ, sh, val  -- logical right shift; value goes in Rc (hi byte 8)
    return mk(0x00007219ull | ((uint64_t)rd << 16) | ((uint64_t)0xff << 24) | ((uint64_t)sh << 32),
              0x00011600 | val, ctrl);
}

// SHF.R.S32.HI Rd, RZ, sh, val -- arithmetic right shift (sign fill)
struct sass_insn sass_shf_r_s(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl)
{
    return mk(0x00007219ull | ((uint64_t)rd << 16) | ((uint64_t)0xff << 24) | ((uint64_t)sh << 32),
              0x00011400 | val, ctrl);
}

// IMAD.WIDE Rd, Ra, Rb, Rc -- 64-bit Rd pair = Ra * Rb + Rc pair; .S32 sign-extends Ra and Rb
struct sass_insn sass_imadwide_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, int is_signed, uint32_t ctrl)
{
    return mk(0x00007225ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x078e0000 | (is_signed ? 0x200 : 0) | rc, ctrl);
}

struct sass_insn sass_fadd(uint8_t rd, uint8_t ra, uint8_t rb, int neg_b, uint32_t ctrl)
{
    return mk(0x00007221ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  (neg_b ? (1ull << 63) : 0),
              0x00000000, ctrl);
}

struct sass_insn sass_fmul(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl)
{
    return mk(0x00007220ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x00400000, ctrl);
}

struct sass_insn sass_ffma(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, int mode, uint32_t ctrl)
{
    // FFMA[.RM|.RP|.RZ] Rd, Ra, Rb, Rc  -- Rd = Ra*Rb + Rc; Rc in hi byte 8, rounding (SASS_RND_*) at hi bits 14:15
    return mk(0x00007223ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              rc | ((uint32_t)mode << 14), ctrl);
}

struct sass_insn sass_fadd_imm(uint8_t rd, uint8_t ra, uint32_t imm, uint32_t ctrl)
{
    return mk(0x00007421ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)imm << 32), 0x00000000, ctrl);
}

struct sass_insn sass_fmul_imm(uint8_t rd, uint8_t ra, uint32_t imm, uint32_t ctrl)
{
    return mk(0x00007820ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)imm << 32), 0x00400000, ctrl);
}

struct sass_insn sass_ffma_imm_b(uint8_t rd, uint8_t ra, uint32_t imm, uint8_t rc, int mode, uint32_t ctrl)
{
    return mk(0x00007823ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)imm << 32),
              rc | ((uint32_t)mode << 14), ctrl);
}

struct sass_insn sass_ffma_imm_c(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t imm, int mode, uint32_t ctrl)
{
    return mk(0x00007423ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)imm << 32),
              rb | ((uint32_t)mode << 14), ctrl);
}

struct sass_insn sass_fmnmx(uint8_t rd, uint8_t ra, uint8_t rb, int is_max, uint32_t ctrl)
{
    // FMNMX Rd, Ra, Rb, PT/!PT  -- !PT selects max, PT selects min (predicate bit 90)
    return mk(0x00007209ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x03800000 | (is_max ? 0x04000000 : 0), ctrl);
}

struct sass_insn sass_imnmx(uint8_t rd, uint8_t ra, uint8_t rb, int is_max, int is_signed, uint32_t ctrl)
{
    return mk(0x00007217ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x03800000 | (is_max ? 0x04000000 : 0) | (is_signed ? 0x200 : 0), ctrl);
}

struct sass_insn sass_mufu(uint8_t rd, uint8_t ra, uint8_t op, uint32_t ctrl)
{
    // MUFU.<op> Rd, Rs  -- source in byte 4 (like I2F), op selector (COS/SIN/EX2/LG2/RCP/RSQ) at bits 74+
    return mk(0x00007308ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 32), (uint32_t)op << 10, ctrl);
}

// POPC, BREV and FLO.U32 Rd, Rs -- the source in byte 4, like MUFU; FLO's predicate output (bits 81:83) is PT
struct sass_insn sass_popc(uint8_t rd, uint8_t rs, uint32_t ctrl)
{
    return mk(0x00007309ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0, ctrl);
}

struct sass_insn sass_iabs(uint8_t rd, uint8_t rs, uint32_t ctrl)
{
    return mk(0x00007213ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0, ctrl);
}

struct sass_insn sass_brev(uint8_t rd, uint8_t rs, uint32_t ctrl)
{
    return mk(0x00007301ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0, ctrl);
}

struct sass_insn sass_flo(uint8_t rd, uint8_t rs, uint32_t ctrl)
{
    return mk(0x00007300ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x000e0000, ctrl);
}

// VOTE.ANY Rd, PT, Pp -- the lanes where Pp is set, as a mask (p 7 = PT: the active lanes)
struct sass_insn sass_vote_any(uint8_t rd, uint8_t pred, uint32_t ctrl)
{
    return mk(0x00007806ull | ((uint64_t)rd << 16), 0x000e0100 | (uint32_t)pred << 23, ctrl);
}

struct sass_insn sass_i2f(uint8_t rd, uint8_t rs, int is_signed, uint32_t ctrl)
{
    // I2FP.F32.{S32,U32} Rd, Rs  -- source register in byte 4
    return mk(0x00007245ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x00201000 | (is_signed ? 0x400 : 0),
              ctrl);
}

// SHFL.<mode> PT, Rd, Rs, lane, c -- Rs of another lane: lane `lane` (IDX), `lane` below (UP) or above (DOWN), or
// the one whose id differs by xor `lane` (BFLY), within segments and clamped as c says (segment mask << 8 | clamp)
struct sass_insn sass_shfl_bfly(uint8_t rd, uint8_t rs, uint8_t lane, uint16_t c, uint32_t ctrl)
{
    return sass_shfl(SASS_SHFL_BFLY, rd, rs, lane, c, ctrl);
}

struct sass_insn sass_shfl(int mode, uint8_t rd, uint8_t rs, uint8_t lane, uint16_t c, uint32_t ctrl)
{
    return mk(0x0000000000007f89ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 24) | ((uint64_t)c << 40) |
                  ((uint64_t)lane << 53) | ((uint64_t)mode << 58),
              0x000e0000, ctrl);
}

// the same with the lane in register Rl
struct sass_insn sass_shfl_r(int mode, uint8_t rd, uint8_t rs, uint8_t rl, uint16_t c, uint32_t ctrl)
{
    return mk(0x0000000000007589ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 24) | ((uint64_t)rl << 32) |
                  ((uint64_t)c << 40) | ((uint64_t)mode << 58),
              0x000e0000, ctrl);
}

// FSWZADD Rd, Ra, Rb, subop -- per lane of a quad, Ra +/- Rb as the 2-bit fields of subop say (derivatives)
struct sass_insn sass_fswzadd(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t subop, uint32_t ctrl)
{
    return mk(0x0000000000007822ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)subop << 32), rb,
              ctrl);
}

// @Pp KILL -- end the thread's fragment without writing it (discard_fragment); p 7 is PT, always
struct sass_insn sass_kill(uint8_t pred, uint32_t ctrl)
{
    return mk(0x000000000000795bull, (uint32_t)pred << 23, ctrl);
}

// FRND[.FLOOR|.CEIL|.TRUNC] Rd, Rs -- round to an integral float; mode SASS_RND_*
struct sass_insn sass_frnd(uint8_t rd, uint8_t rs, int mode, uint32_t ctrl)
{
    return mk(0x00007307ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x00201000 | ((uint32_t)mode << 14), ctrl);
}

struct sass_insn sass_f2i(uint8_t rd, uint8_t rs, int is_signed, uint32_t ctrl)
{
    // F2I[.U32].TRUNC.NTZ Rd, Rs  -- source register in byte 4
    return mk(0x00007305ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x0020f000 | (is_signed ? 0x100 : 0),
              ctrl);
}

struct sass_insn sass_f2ip_u8(uint8_t rd, uint8_t rs, uint32_t ctrl)
{
    // F2IP.U8.F32.TRUNC.NTZ Rd, RZ, Rs, RZ  -- source register in byte 4
    return mk(0x00007243ull | ((uint64_t)rd << 16) | (0xffull << 24) | ((uint64_t)rs << 32), 0x0000c4ff, ctrl);
}

struct sass_insn sass_fmul_rz_imm(uint8_t rd, uint8_t ra, uint32_t imm, uint32_t ctrl)
{
    return mk(0x00007820ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)imm << 32), 0x0040c000, ctrl);
}

struct sass_insn sass_isetp(uint8_t pd, uint8_t ra, uint8_t rb, uint8_t op, int is_signed, uint32_t ctrl)
{
    // ISETP.<op>.AND Pd, PT, Ra, Rb, PT  -- comparison selector (op<<4 | signed-bit) in hi byte 9, Pd at hi bit 17
    uint32_t sel = ((uint32_t)op << 4) | (is_signed ? 2 : 0);
    return mk(0x0000720cull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x03f00070 | (sel << 8) | ((uint32_t)pd << 17), ctrl);
}

// ISETP.<op>.U32.OR Pd, PT, Ra, Rb, Pc  -- as sass_isetp, the result OR Pc (bop at hi bits 10:11, Pc at 23:25)
struct sass_insn sass_isetp_or(uint8_t pd, uint8_t ra, uint8_t rb, uint8_t op, uint8_t pc, uint32_t ctrl)
{
    return mk(0x0000720cull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x00700470 | ((uint32_t)op << 12) | ((uint32_t)pd << 17) | ((uint32_t)pc << 23), ctrl);
}

struct sass_insn sass_sel(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t pred, uint32_t ctrl)
{
    return mk(0x00007207ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32), (uint32_t)pred << 23,
              ctrl);
}

struct sass_insn sass_fsetp(uint8_t pd, uint8_t ra, uint8_t rb, uint8_t op, uint32_t ctrl)
{
    return mk(0x0000720bull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x03f00000 | ((uint32_t)op << 12) | ((uint32_t)pd << 17), ctrl);
}

struct sass_insn sass_setp_bop(struct sass_insn in, uint8_t bop, uint8_t pc)
{
    in.hi = (in.hi & ~(3ull << 10 | 0xfull << 23)) | (uint64_t)bop << 10 | (uint64_t)pc << 23;
    return in;
}

struct sass_insn sass_isetp_cbuf(uint8_t ra, uint8_t bank, uint16_t off, uint8_t op, int is_signed,
                                 uint32_t ctrl)
{
    uint32_t sel = ((uint32_t)op << 4) | (is_signed ? 2 : 0);
    return mk(0x00007a0cull | ((uint64_t)ra << 24) | cbuf_operand(bank, off), 0x03f00070 | (sel << 8), ctrl);
}

struct sass_insn sass_imm(struct sass_insn in, uint32_t imm)
{
    in.lo = (in.lo & 0xf0ffull) | 0x0800 | (in.lo & 0xffff0000ull) | (uint64_t)imm << 32;
    return in;
}

struct sass_insn sass_pred(struct sass_insn in, uint8_t pred, int negate)
{
    in.lo = (in.lo & ~(0xfull << 12)) | ((uint64_t)((pred & 7) | (negate ? 8 : 0)) << 12);
    return in;
}

struct sass_insn sass_ftz(struct sass_insn in)
{
    in.hi |= 1ull << 16;
    return in;
}

struct sass_insn sass_neg_a(struct sass_insn in)
{
    in.hi ^= 1ull << 8;
    return in;
}

struct sass_insn sass_src_mod(struct sass_insn in, int slot, int neg, int abs)
{
    // abs is applied first, so both bits are -|x|. Negation toggles, to combine with one already set.
    switch (slot) {
    case SASS_MOD_A:
        in.hi |= (uint64_t)abs << 9;
        in.hi ^= (uint64_t)neg << 8;
        break;
    case SASS_MOD_B:
        in.lo |= (uint64_t)abs << 62;
        in.lo ^= (uint64_t)neg << 63;
        break;
    default:  // C, or the register B of an FFMA with an immediate C
        in.hi |= (uint64_t)abs << 10;
        in.hi ^= (uint64_t)neg << 11;
        break;
    }
    return in;
}

struct sass_insn sass_sat(struct sass_insn in)
{
    in.hi |= 1ull << 13;
    return in;
}

struct sass_insn sass_s2r(uint8_t rd, uint8_t sr, uint32_t ctrl)
{
    // S2R Rd, SR<sr>  -- special-register index in hi byte 9. TID.{x,y,z}=0x21..0x23, CTAID.{x,y,z}=0x25..0x27
    return mk(0x00007919ull | ((uint64_t)rd << 16), (uint32_t)sr << 8, ctrl);
}

uint32_t sass_ctrl_word(struct sass_ctrl c)
{
    uint32_t wr = (c.wr < 0 ? 7 : c.wr) & 7;
    uint32_t rd = (c.rd < 0 ? 7 : c.rd) & 7;
    return ((uint32_t)(c.stall & 0xf) << 9) | ((uint32_t)(c.yield ? 1 : 0) << 13) | (wr << 14) |
           (rd << 17) | ((uint32_t)(c.wait & 0x3f) << 20);
}

struct sass_insn sass_uldc64_cbuf(uint8_t urd, uint8_t bank, uint16_t off, uint32_t ctrl)
{
    // ULDC.64 URd, c[bank][off]  -- .64 modifier (0x0a) in hi byte 10
    return mk(0x00007ab9ull | ((uint64_t)urd << 16) | cbuf_operand(bank, off), 0x00000a00, ctrl);
}

// ATOMG/ATOMS Rd, [Ra + off], Rb (ATOMG.CAS/ATOMS.CAS: [Ra + off], Rb compared, Rc swapped in), as NAK's
// sm70_encode has them: hi = the uniform base URZ [64:71] (64-bit [72], a 64-bit Ra [70], [91]), the data type
// [73:75], strong GPU ordering [77:80], no predicate [81:83], eviction normal [84:86], the op [87:90]
struct sass_insn sass_atom(int op, int shared, int type, uint8_t rd, uint8_t ra, uint32_t off, uint8_t rb, uint8_t rc,
                           uint32_t ctrl)
{
    int cas = op == SASS_ATOM_CAS;
    uint64_t opcode = shared ? (cas ? 0x38d : 0x38c) : (cas ? 0x3a9 : 0x9a8);
    uint64_t lo = (0x7000 | opcode) | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  ((uint64_t)(off & 0xffffff) << 40);
    uint32_t hi = (uint32_t)type << 9;
    if (cas)
        hi |= rc;  // the value swapped in [64:71]
    else
        hi |= (uint32_t)op << 23;
    if (!cas && !shared)
        hi |= 63u | 1u << 27;  // URZ, the uniform base bit
    if (!shared) {
        hi |= 7u << 13 | 7u << 17 | 1u << 20;  // strong GPU, no predicate, eviction normal
        hi |= cas ? 1u << 8 : 1u << 6 | 1u << 8;  // 64-bit address (and uniform base)
    }
    return mk(lo, hi, ctrl);
}

// RED.E.<op>.STRONG.GPU [Ra.64 + off], Rb: the global atomic without its old value, in nvcc's form, through the
// memory descriptor in UR4 (hi [64:71] 0x84, 64-bit [72], [90:91]); the type and ordering as ATOMG's, the op in
// [87:89], so no EXCH or CAS
struct sass_insn sass_red(int op, int type, uint8_t ra, uint32_t off, uint8_t rb, uint32_t ctrl)
{
    return mk(0x0000798eull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) | ((uint64_t)(off & 0xffffff) << 40),
              0x0c10e184 | (uint32_t)type << 9 | (uint32_t)op << 23, ctrl);
}

struct sass_insn sass_ldg(uint8_t rd, uint8_t ra, uint32_t off, uint8_t size, uint32_t ctrl)
{
    // LDG.E Rd, [Ra.64 + off]  -- Ra at byte3, UR4 (the ULDC'd global base) at byte4, 24-bit offset at
    // bits[40:63], access size at hi bits [73:75], .E/strong flags in hi
    return mk(0x00007981ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)4 << 32) |
                  ((uint64_t)(off & 0xffffff) << 40),
              0x0c1e1100 | ((uint32_t)size << 9), ctrl);
}

struct sass_insn sass_stg(uint8_t ra, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl)
{
    // STG.E [Ra.64 + off], Rb  -- Ra at byte3, Rb at byte4, offset and size as for LDG
    return mk(0x00007986ull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  ((uint64_t)(off & 0xffffff) << 40),
              0x0c101104 | ((uint32_t)size << 9), ctrl);
}

// The uniform-base forms are LDG's and STG's without the memory descriptor (bit 76): the uniform register, in the
// descriptor's place for LDG and at [64:71] for STG, is a 64-bit base [72] added to Ra, which is 64-bit with bit 90
struct sass_insn sass_ldg_ur(uint8_t rd, uint8_t ra, int wide, uint8_t ur, uint32_t off, uint8_t size, uint32_t ctrl)
{
    return mk(0x00007981ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)ur << 32) |
                  ((uint64_t)(off & 0xffffff) << 40),
              0x081e0100 | ((uint32_t)size << 9) | (uint32_t)wide << 26, ctrl);
}

struct sass_insn sass_stg_ur(uint8_t ra, int wide, uint8_t ur, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl)
{
    return mk(0x00007986ull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) | ((uint64_t)(off & 0xffffff) << 40),
              0x08100100 | ((uint32_t)size << 9) | (uint32_t)wide << 26 | ur, ctrl);
}

struct sass_insn sass_ldg_constant(struct sass_insn in)
{
    in.hi |= 1ull << 15;  // the memory order [77:80] = 4
    return in;
}

// LDS / STS: shared memory at the 32-bit byte address Ra + off (offset and size fields as for LDG/STG)
struct sass_insn sass_lds(uint8_t rd, uint8_t ra, uint32_t off, uint8_t size, uint32_t ctrl)
{
    return mk(0x00007984ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)(off & 0xffffff) << 40),
              (uint32_t)size << 9, ctrl);
}

struct sass_insn sass_sts(uint8_t ra, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl)
{
    return mk(0x00007388ull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) | ((uint64_t)(off & 0xffffff) << 40),
              (uint32_t)size << 9, ctrl);
}

struct sass_insn sass_ldl(uint8_t rd, uint8_t ra, uint32_t off, uint8_t size, uint32_t ctrl)
{
    return mk(0x00007983ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)(off & 0xffffff) << 40),
              (uint32_t)size << 9 | 1u << 20, ctrl);
}

struct sass_insn sass_stl(uint8_t ra, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl)
{
    return mk(0x00007387ull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) | ((uint64_t)(off & 0xffffff) << 40),
              (uint32_t)size << 9 | 1u << 20, ctrl);
}

// TEX/TLD/TLD4/TXD.B Rd, Ra, Rb, mask, dim: a bindless texture op on two source vectors, Ra.. (an array index,
// then the coordinates; TXD: the handle first) and Rb.. (the handle (TIC index | TSC index << 20), then the LOD
// or bias, the packed AOFFI offset and the depth reference, as the flags have them; TXD: the gradients), RZ for
// none. mask selects result components (RGBA = 0xf): R,G land at Rd, B,A at the second dst [64:71] = Rd+2. The
// result is barrier-guarded (variable latency). Layout as NAK's sm70_encode has it.
struct sass_insn sass_texop(int op, uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, unsigned flags,
                            uint32_t ctrl)
{
    uint64_t lo = (uint64_t)(0x7000 | op) | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  (1ull << 59) |          // .B (handle in a register)
                  ((uint64_t)dim << 61);  // SASS_DIM_*
    // hi: second dst [64:71], channel mask [72:75], AOFFI [76], depth compare [78], fault pred PT(7) [81:83],
    //     eviction Normal(1) [84:86], LOD mode (TLD4: the component) [87:89]
    uint32_t hi = (uint32_t)(rd + 2) | ((uint32_t)mask << 8) | (7u << 17) | (1u << 20);
    if (flags & SASS_TEXF_AOFFI)
        hi |= 1u << 12;
    if (flags & (SASS_TEXF_DC | SASS_TEXF_MS))  // the same bit, TEX's depth compare and TLD's multisample
        hi |= 1u << 14;
    if (op == SASS_TLD4)
        hi |= (flags >> 5 & 3) << 23;
    else if (op != SASS_TXD)
        hi |= SASS_TEXF_LOD(flags) << 23;
    return mk(lo, hi, ctrl);
}

// TLD.B.LZ: a texel fetch at LOD 0, the integer coordinates at Ra, the handle in Rb
struct sass_insn sass_tld(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, uint32_t ctrl)
{
    return sass_texop(SASS_TLD, rd, ra, rb, mask, dim, SASS_LOD_ZERO, ctrl);
}

// TEX.B.LZ Rd, Ra, Rb, mask, 2D|3D: bindless sampled fetch at LOD 0. Ra = first of the consecutive float-coord
// regs; Rb = handle; the sampler in the handle filters the result. LOD 0 (no operand) suits a non-mipmapped
// texture, which is all we build, and needs no derivatives (there are none in compute).
struct sass_insn sass_tex(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, uint32_t ctrl)
{
    return sass_texop(SASS_TEX, rd, ra, rb, mask, dim, SASS_LOD_ZERO, ctrl);
}

// TXQ.B RZ, Rd, Ra, query, mask: query the texture whose handle is in Ra. TEX_HEADER_DIMENSION (query 0) at the
// LOD in Ra+1: mask picks width (1), height (2), depth (4), levels (8); TEXTURE_TYPE (1): 4 is the sample count.
// One bit per call keeps the result in Rd alone.
struct sass_insn sass_txq(uint8_t rd, uint8_t ra, uint8_t mask, int query, uint32_t ctrl)
{
    uint64_t lo = 0x0000000000007370ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | (1ull << 59) |
                  ((uint64_t)query << 62);
    return mk(lo, 0xffu | ((uint32_t)mask << 8), ctrl);  // second dst RZ
}

// SUST.P.<dim>.STRONG.SM.IGN [Ra], Rb, Rc: formatted store of the channels in mask, taken from consecutive
// registers at Rb, to the texel whose coordinates (an array's layer last) start at Ra, of the texture whose handle
// is in Rc; the header's format does the conversion. A texel outside the texture is dropped, as Metal has it
// (.TRAP, 2 in [59:60], raises an out-of-range-address exception that takes the channel down).
struct sass_insn sass_sust(uint8_t ra, uint8_t rb, uint8_t rc, uint8_t mask, int dim, uint32_t ctrl)
{
    uint64_t lo = 0x000000000000799cull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  ((uint64_t)dim << 61);  // SASS_IMG_* [61:63], IGN [59:60]
    // hi: handle [64:71], channel mask [72:75], STRONG.SM [76:79], eviction normal [84:86]
    return mk(lo, (uint32_t)rc | ((uint32_t)mask << 8) | 0xa000 | 0x100000, ctrl);
}

// ALD[.P][.O] Rd, a[addr], vtx: load comps (1-4) consecutive 32-bit attributes of a vertex into Rd.. (vertex
// stages); vertex handle [32:39], patch [76], output [79], as NAK's sm70_encode has them
struct sass_insn sass_ald(uint8_t rd, uint8_t vtx, uint16_t addr, unsigned comps, int patch, int output,
                          uint32_t ctrl)
{
    uint64_t lo = 0x0000000000007321ull | ((uint64_t)rd << 16) | (0xffull << 24) | ((uint64_t)vtx << 32) |
                  ((uint64_t)addr << 40);
    return mk(lo, (comps - 1) << 10 | (patch ? 1u << 12 : 0) | (output ? 1u << 15 : 0), ctrl);
}

// AST[.P] a[addr], Rs, vtx: store comps (1-4) consecutive registers from Rs to a vertex's output attributes;
// vertex handle [64:71]
struct sass_insn sass_ast(uint16_t addr, uint8_t rs, uint8_t vtx, unsigned comps, int patch, uint32_t ctrl)
{
    uint64_t lo = 0x0000000000007322ull | (0xffull << 24) | ((uint64_t)rs << 32) | ((uint64_t)addr << 40);
    return mk(lo, vtx | (comps - 1) << 10 | (patch ? 1u << 12 : 0), ctrl);
}

// ISBERD Rd, [Ra]: the byte at Ra of the input map (MAP, U8), the handle ALD takes for an input vertex
struct sass_insn sass_isberd(uint8_t rd, uint8_t ra, uint32_t ctrl)
{
    return mk(0x0000000000007923ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24), 0, ctrl);
}

// IPA.PASS Rd, a[addr]: the pixel's value of an input attribute, interpolated as the pixel shader header's
// imap says (perspective inputs still need a multiply by 1/w on SM70+); IPA.CONSTANT with SASS_IPA_FLAT, for
// system values the header has no imap for (front facing); at the centroid with SASS_IPA_CENTROID, which in a
// pass of one sample is that sample
struct sass_insn sass_ipa(uint8_t rd, uint16_t addr, int mode, uint32_t ctrl)
{
    uint64_t lo = 0x0000000000007326ull | ((uint64_t)rd << 16) | (0xffull << 24) | (0xffull << 32);
    uint32_t hi = (uint32_t)(addr >> 2) | (mode & SASS_IPA_CENTROID ? 1u : 0u) << 12 |
                  (mode & SASS_IPA_FLAT ? 1u : 0u) << 14 | 7u << 17;  // no predicate out (PT)
    return mk(lo, hi, ctrl);
}

// PIXLD Rd, val: a SASS_PIXLD_* value of the pixel being shaded
struct sass_insn sass_pixld(uint8_t rd, unsigned val, uint32_t ctrl)
{
    return mk(0x0000000000007925ull | ((uint64_t)rd << 16), val << 14 | 7u << 17, ctrl);
}

// LDC Rd, c[bank][Ra + off] -- a const bank word at a register's offset: the byte offset at bit 38, the bank at 54
struct sass_insn sass_ldc(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint32_t ctrl)
{
    return mk(0x0000000000007b82ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)off << 38) |
                  ((uint64_t)bank << 54),
              0x00000800, ctrl);
}

// BAR.SYNC.DEFER_BLOCKING 0x0: wait for the whole threadgroup
struct sass_insn sass_bar_sync(uint32_t ctrl)
{
    return mk(0x0000000000007b1dull, 0x00010000, ctrl);
}

// nvcc's __threadfence(): MEMBAR.SC.GPU, ERRBAR, then CCTL.IVALL so later loads miss the L1
struct sass_insn sass_membar_gpu(uint32_t ctrl)
{
    return mk(0x0000000000007992ull, 0x00002000, ctrl);
}

struct sass_insn sass_errbar(uint32_t ctrl)
{
    return mk(0x00000000000079abull, 0, ctrl);
}

struct sass_insn sass_cctl_ivall(uint32_t ctrl)
{
    return mk(0x00000000ff00798full, 0x02000000, ctrl);
}

// Half-precision ops work on the low half of each register (.H0_H0 swizzles: lo bit 61 for Rb, hi 0x800 for
// Ra, hi 0x40000 for Rc); both halves of Rd get the result.
struct sass_insn sass_hadd2(uint8_t rd, uint8_t ra, uint8_t rb, int neg_b, uint32_t ctrl)
{
    return mk(0x2000000000007230ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  ((uint64_t)!!neg_b << 63),
              0x00000800, ctrl);
}

struct sass_insn sass_hmul2(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl)
{
    return mk(0x2000000000007232ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x00000800, ctrl);
}

struct sass_insn sass_hfma2(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, uint32_t ctrl)
{
    return mk(0x2000000000007231ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x00040800 | rc, ctrl);
}

struct sass_insn sass_h2f(uint8_t rd, uint8_t rs, uint32_t ctrl)
{
    // HADD2.F32 Rd, -RZ, Rs.H0_H0
    return mk(0x20000000ff007230ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x00004100, ctrl);
}

struct sass_insn sass_f2h(uint8_t rd, uint8_t rs, uint32_t ctrl)
{
    // F2FP.PACK_AB Rd, RZ, Rs  -- the half lands in the low 16 bits
    return mk(0x00000000ff00723eull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x000000ff, ctrl);
}

struct sass_insn sass_f2h2(uint8_t rd, uint8_t lo, uint8_t hi, uint32_t ctrl)
{
    // F2FP.PACK_AB Rd, Rhi, Rlo
    struct sass_insn in = sass_f2h(rd, lo, ctrl);
    in.lo = (in.lo & ~(0xffull << 24)) | (uint64_t)hi << 24;
    return in;
}

struct sass_insn sass_h2f_hi(struct sass_insn in)
{
    in.lo |= 1ull << 60;
    return in;
}

struct sass_insn sass_mov_rr(uint8_t rd, uint8_t rsrc, uint32_t ctrl)
{
    // IMAD.MOV.U32 Rd, RZ, RZ, Rsrc  -- Rd = RZ*RZ + Rsrc = Rsrc; Rsrc is Rc in hi byte 8
    return mk(0x00007224ull | ((uint64_t)rd << 16) | ((uint64_t)0xff << 24) | ((uint64_t)0xff << 32),
              0x078e0000 | rsrc, ctrl);
}

struct sass_insn sass_exit(uint32_t ctrl)
{
    return mk(0x0000794dull, 0x03800000, ctrl);
}

struct sass_insn sass_bssy(uint8_t bar, int32_t rel, uint32_t ctrl)
{
    // the reconverge offset (target - (this + 0x10)) in lo bytes 4:7, the barrier in [16:19]
    return mk(0x00007945ull | ((uint64_t)bar << 16) | ((uint64_t)(uint32_t)rel << 32),
              0x03800000 | (rel < 0 ? 0x0003ffff : 0), ctrl);
}

struct sass_insn sass_bsync(uint8_t bar, uint32_t ctrl)
{
    return mk(0x00007941ull | ((uint64_t)bar << 16), 0x03800000, ctrl);
}

struct sass_insn sass_bra(int32_t rel, uint8_t pred, int negate, uint32_t ctrl)
{
    // target offset is 50-bit: low 32 in lo[32:63], high 18 in hi[0:17] (sign-extended for backward branches)
    return mk(0x00000947ull | ((uint64_t)((pred & 7) | (negate ? 8 : 0)) << 12) |
                  ((uint64_t)(uint32_t)rel << 32),
              0x03800000 | (rel < 0 ? 0x0003ffff : 0), ctrl);
}

struct sass_insn sass_call_rel(int32_t rel, uint32_t ctrl)
{
    return mk(0x00007944ull | ((uint64_t)(uint32_t)rel << 32), 0x03c00000 | (rel < 0 ? 0x0003ffff : 0), ctrl);
}

struct sass_insn sass_ret_rel(uint8_t ra, int32_t rel, uint32_t ctrl)
{
    return mk(0x00007950ull | ((uint64_t)ra << 24) | ((uint64_t)(uint32_t)rel << 32),
              0x03c00000 | (rel < 0 ? 0x0003ffff : 0), ctrl);
}

struct sass_insn sass_bra_self(uint32_t ctrl)
{
    // BRA to itself: relative offset -0x10, position-independent
    return mk(0xfffffff000007947ull, 0x0383ffff, ctrl);
}

struct sass_insn sass_nop(uint32_t ctrl)
{
    return mk(0x00007918ull, 0x00000000, ctrl);
}

void sass_emit(const struct sass_insn *insn, size_t n, uint8_t *out)
{
    for (size_t i = 0; i < n; i++) {
        uint64_t lo = insn[i].lo, hi = insn[i].hi;
        for (int b = 0; b < 8; b++) {
            out[i * 16 + b] = (lo >> (b * 8)) & 0xff;
            out[i * 16 + 8 + b] = (hi >> (b * 8)) & 0xff;
        }
    }
}
