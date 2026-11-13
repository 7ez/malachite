// SM86 SASS encoder, layouts reversed from nvcc -arch=sm_86 and checked by sass86_test.c
#include "sass86.h"

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
    // IMAD.MOV.U32 Rd, RZ, RZ, imm
    return mk(0x0000ff007424ull | ((uint64_t)rd << 16) | ((uint64_t)imm << 32), 0x078e00ff, ctrl);
}

struct sass_insn sass_imadmov_cbuf(uint8_t rd, uint8_t bank, uint16_t off, uint32_t ctrl)
{
    return mk(0x0000ff007624ull | ((uint64_t)rd << 16) | cbuf_operand(bank, off), 0x078e00ff, ctrl);
}

struct sass_insn sass_imadwide_cbuf(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t bank, uint16_t off,
                                    uint32_t ctrl)
{
    return mk(0x00007625ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | cbuf_operand(bank, off),
              0x078e0000 | rb, ctrl);
}

struct sass_insn sass_iadd3_cbuf(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint32_t ctrl)
{
    return mk(0x00007a10ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | cbuf_operand(bank, off),
              0x07ffe0ff, ctrl);
}

struct sass_insn sass_iadd3_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint32_t ctrl)
{
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
    return mk(0x00007210ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  (1ull << 63),
              0x07ffe0ff, ctrl);
}

struct sass_insn sass_imad_rr(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t rc, uint32_t ctrl)
{
    return mk(0x00007224ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x078e0200 | rc, ctrl);
}

struct sass_insn sass_imadhi_rr(uint8_t rd, uint8_t ra, uint8_t rb, int is_signed, uint32_t ctrl)
{
    return mk(0x00007227ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x078e00ff | (is_signed ? 0x200 : 0), ctrl);
}

struct sass_insn sass_imad_cbuf(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint8_t rc, uint32_t ctrl)
{
    return mk(0x00007a24ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | cbuf_operand(bank, off),
              0x078e0200 | rc, ctrl);
}

// byte i of Rd is byte (nibble i of sel) of Rc:Ra, sign-filled when the nibble's top bit is set
struct sass_insn sass_prmt(uint8_t rd, uint8_t ra, uint16_t sel, uint8_t rc, uint32_t ctrl)
{
    return mk(0x00007816ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)sel << 32), rc, ctrl);
}

struct sass_insn sass_lop3(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t lut, uint32_t ctrl)
{
    return mk(0x00007212ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x078e00ff | ((uint32_t)lut << 8), ctrl);
}

struct sass_insn sass_shf_l(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl)
{
    return mk(0x00007219ull | ((uint64_t)rd << 16) | ((uint64_t)val << 24) | ((uint64_t)sh << 32),
              0x000006ff, ctrl);
}

struct sass_insn sass_shf_r(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl)
{
    // SHF.R.U32.HI Rd, RZ, sh, val
    return mk(0x00007219ull | ((uint64_t)rd << 16) | ((uint64_t)0xff << 24) | ((uint64_t)sh << 32),
              0x00011600 | val, ctrl);
}

struct sass_insn sass_shf_r_s(uint8_t rd, uint8_t val, uint8_t sh, uint32_t ctrl)
{
    return mk(0x00007219ull | ((uint64_t)rd << 16) | ((uint64_t)0xff << 24) | ((uint64_t)sh << 32),
              0x00011400 | val, ctrl);
}

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
    // !PT selects max, PT min
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
    return mk(0x00007308ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 32), (uint32_t)op << 10, ctrl);
}

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

// pred 7 (PT) gives the active lanes
struct sass_insn sass_vote_any(uint8_t rd, uint8_t pred, uint32_t ctrl)
{
    return mk(0x00007806ull | ((uint64_t)rd << 16), 0x000e0100 | (uint32_t)pred << 23, ctrl);
}

struct sass_insn sass_i2f(uint8_t rd, uint8_t rs, int is_signed, uint32_t ctrl)
{
    return mk(0x00007245ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x00201000 | (is_signed ? 0x400 : 0),
              ctrl);
}

// c is segment mask << 8 | clamp
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

struct sass_insn sass_shfl_r(int mode, uint8_t rd, uint8_t rs, uint8_t rl, uint16_t c, uint32_t ctrl)
{
    return mk(0x0000000000007589ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 24) | ((uint64_t)rl << 32) |
                  ((uint64_t)c << 40) | ((uint64_t)mode << 58),
              0x000e0000, ctrl);
}

// per quad lane, Ra +/- Rb as subop's 2-bit fields say; derivatives use it
struct sass_insn sass_fswzadd(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t subop, uint32_t ctrl)
{
    return mk(0x0000000000007822ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)subop << 32), rb,
              ctrl);
}

struct sass_insn sass_kill(uint8_t pred, uint32_t ctrl)
{
    return mk(0x000000000000795bull, (uint32_t)pred << 23, ctrl);
}

struct sass_insn sass_frnd(uint8_t rd, uint8_t rs, int mode, uint32_t ctrl)
{
    return mk(0x00007307ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x00201000 | ((uint32_t)mode << 14), ctrl);
}

struct sass_insn sass_f2i(uint8_t rd, uint8_t rs, int is_signed, uint32_t ctrl)
{
    return mk(0x00007305ull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x0020f000 | (is_signed ? 0x100 : 0),
              ctrl);
}

struct sass_insn sass_f2ip_u8(uint8_t rd, uint8_t rs, uint32_t ctrl)
{
    return mk(0x00007243ull | ((uint64_t)rd << 16) | (0xffull << 24) | ((uint64_t)rs << 32), 0x0000c4ff, ctrl);
}

struct sass_insn sass_fmul_rz_imm(uint8_t rd, uint8_t ra, uint32_t imm, uint32_t ctrl)
{
    return mk(0x00007820ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)imm << 32), 0x0040c000, ctrl);
}

struct sass_insn sass_isetp(uint8_t pd, uint8_t ra, uint8_t rb, uint8_t op, int is_signed, uint32_t ctrl)
{
    uint32_t sel = ((uint32_t)op << 4) | (is_signed ? 2 : 0);
    return mk(0x0000720cull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32),
              0x03f00070 | (sel << 8) | ((uint32_t)pd << 17), ctrl);
}

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
    // both bits give -|x|; neg toggles so it combines with one already set
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
    // TID.xyz are 0x21..0x23, CTAID.xyz 0x25..0x27
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
    return mk(0x00007ab9ull | ((uint64_t)urd << 16) | cbuf_operand(bank, off), 0x00000a00, ctrl);
}

// Layout as NAK's sm70_encode has it; CAS compares Rb and swaps in Rc
struct sass_insn sass_atom(int op, int shared, int type, uint8_t rd, uint8_t ra, uint32_t off, uint8_t rb, uint8_t rc,
                           uint32_t ctrl)
{
    int cas = op == SASS_ATOM_CAS;
    uint64_t opcode = shared ? (cas ? 0x38d : 0x38c) : (cas ? 0x3a9 : 0x9a8);
    uint64_t lo = (0x7000 | opcode) | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  ((uint64_t)(off & 0xffffff) << 40);
    uint32_t hi = (uint32_t)type << 9;
    if (cas)
        hi |= rc;
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

// nvcc's form, through the memory descriptor in UR4; the op field is 3 bits, so no EXCH or CAS
struct sass_insn sass_red(int op, int type, uint8_t ra, uint32_t off, uint8_t rb, uint32_t ctrl)
{
    return mk(0x0000798eull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) | ((uint64_t)(off & 0xffffff) << 40),
              0x0c10e184 | (uint32_t)type << 9 | (uint32_t)op << 23, ctrl);
}

struct sass_insn sass_ldg(uint8_t rd, uint8_t ra, uint32_t off, uint8_t size, uint32_t ctrl)
{
    // UR4 holds the ULDC'd memory descriptor
    return mk(0x00007981ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)4 << 32) |
                  ((uint64_t)(off & 0xffffff) << 40),
              0x0c1e1100 | ((uint32_t)size << 9), ctrl);
}

struct sass_insn sass_stg(uint8_t ra, uint32_t off, uint8_t rb, uint8_t size, uint32_t ctrl)
{
    return mk(0x00007986ull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  ((uint64_t)(off & 0xffffff) << 40),
              0x0c101104 | ((uint32_t)size << 9), ctrl);
}

// no memory descriptor (bit 76); the uniform register is a 64-bit base added to Ra, which is 64-bit with bit 90
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
    in.hi |= 1ull << 15;  // memory order 4
    return in;
}

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

// Layout as NAK's sm70_encode has it. Ra.. is array index then coordinates (TXD: handle first); Rb.. is the handle
// (TIC | TSC << 20), then LOD or bias, AOFFI and depth reference (TXD: gradients). B,A land at Rd+2.
struct sass_insn sass_texop(int op, uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, unsigned flags,
                            uint32_t ctrl)
{
    uint64_t lo = (uint64_t)(0x7000 | op) | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  (1ull << 59) |          // .B, handle in a register
                  ((uint64_t)dim << 61);
    // LOD mode at [87:89] is the component for TLD4
    uint32_t hi = (uint32_t)(rd + 2) | ((uint32_t)mask << 8) | (7u << 17) | (1u << 20);
    if (flags & SASS_TEXF_AOFFI)
        hi |= 1u << 12;
    if (flags & (SASS_TEXF_DC | SASS_TEXF_MS))  // one bit, TEX's depth compare and TLD's multisample
        hi |= 1u << 14;
    if (op == SASS_TLD4)
        hi |= (flags >> 5 & 3) << 23;
    else if (op != SASS_TXD)
        hi |= SASS_TEXF_LOD(flags) << 23;
    return mk(lo, hi, ctrl);
}

struct sass_insn sass_tld(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, uint32_t ctrl)
{
    return sass_texop(SASS_TLD, rd, ra, rb, mask, dim, SASS_LOD_ZERO, ctrl);
}

// LOD 0 needs no derivatives, which compute doesn't have
struct sass_insn sass_tex(uint8_t rd, uint8_t ra, uint8_t rb, uint8_t mask, int dim, uint32_t ctrl)
{
    return sass_texop(SASS_TEX, rd, ra, rb, mask, dim, SASS_LOD_ZERO, ctrl);
}

// Query 0 at the LOD in Ra+1, mask width 1, height 2, depth 4, levels 8; query 1 mask 4 is the sample count.
// One mask bit per call keeps the result in Rd alone.
struct sass_insn sass_txq(uint8_t rd, uint8_t ra, uint8_t mask, int query, uint32_t ctrl)
{
    uint64_t lo = 0x0000000000007370ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | (1ull << 59) |
                  ((uint64_t)query << 62);
    return mk(lo, 0xffu | ((uint32_t)mask << 8), ctrl);  // second dst RZ
}

// .IGN drops out-of-range texels as Metal does; .TRAP raises an exception that takes the channel down
struct sass_insn sass_sust(uint8_t ra, uint8_t rb, uint8_t rc, uint8_t mask, int dim, uint32_t ctrl)
{
    uint64_t lo = 0x000000000000799cull | ((uint64_t)ra << 24) | ((uint64_t)rb << 32) |
                  ((uint64_t)dim << 61);
    return mk(lo, (uint32_t)rc | ((uint32_t)mask << 8) | 0xa000 | 0x100000, ctrl);
}

// layout as NAK's sm70_encode has it
struct sass_insn sass_ald(uint8_t rd, uint8_t vtx, uint16_t addr, unsigned comps, int patch, int output,
                          uint32_t ctrl)
{
    uint64_t lo = 0x0000000000007321ull | ((uint64_t)rd << 16) | (0xffull << 24) | ((uint64_t)vtx << 32) |
                  ((uint64_t)addr << 40);
    return mk(lo, (comps - 1) << 10 | (patch ? 1u << 12 : 0) | (output ? 1u << 15 : 0), ctrl);
}

struct sass_insn sass_ast(uint16_t addr, uint8_t rs, uint8_t vtx, unsigned comps, int patch, uint32_t ctrl)
{
    uint64_t lo = 0x0000000000007322ull | (0xffull << 24) | ((uint64_t)rs << 32) | ((uint64_t)addr << 40);
    return mk(lo, vtx | (comps - 1) << 10 | (patch ? 1u << 12 : 0), ctrl);
}

// reads the input map, giving the vertex handle ALD takes
struct sass_insn sass_isberd(uint8_t rd, uint8_t ra, uint32_t ctrl)
{
    return mk(0x0000000000007923ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24), 0, ctrl);
}

// Perspective inputs still need a multiply by 1/w on SM70+. SASS_IPA_FLAT is for system values the header has
// no imap for, such as front facing.
struct sass_insn sass_ipa(uint8_t rd, uint16_t addr, int mode, uint32_t ctrl)
{
    uint64_t lo = 0x0000000000007326ull | ((uint64_t)rd << 16) | (0xffull << 24) | (0xffull << 32);
    uint32_t hi = (uint32_t)(addr >> 2) | (mode & SASS_IPA_CENTROID ? 1u : 0u) << 12 |
                  (mode & SASS_IPA_FLAT ? 1u : 0u) << 14 | 7u << 17;
    return mk(lo, hi, ctrl);
}

struct sass_insn sass_pixld(uint8_t rd, unsigned val, uint32_t ctrl)
{
    return mk(0x0000000000007925ull | ((uint64_t)rd << 16), val << 14 | 7u << 17, ctrl);
}

// c[bank][Ra + off]; unlike cbuf operands the offset here is in bytes
struct sass_insn sass_ldc(uint8_t rd, uint8_t ra, uint8_t bank, uint16_t off, uint32_t ctrl)
{
    return mk(0x0000000000007b82ull | ((uint64_t)rd << 16) | ((uint64_t)ra << 24) | ((uint64_t)off << 38) |
                  ((uint64_t)bank << 54),
              0x00000800, ctrl);
}

struct sass_insn sass_bar_sync(uint32_t ctrl)
{
    return mk(0x0000000000007b1dull, 0x00010000, ctrl);
}

// nvcc's __threadfence() is MEMBAR.SC.GPU, ERRBAR, then CCTL.IVALL so later loads miss the L1
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

// Half ops read the low halves (.H0_H0) and write both halves of Rd
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
    // F2FP.PACK_AB Rd, RZ, Rs
    return mk(0x00000000ff00723eull | ((uint64_t)rd << 16) | ((uint64_t)rs << 32), 0x000000ff, ctrl);
}

struct sass_insn sass_f2h2(uint8_t rd, uint8_t lo, uint8_t hi, uint32_t ctrl)
{
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
    // IMAD.MOV.U32 Rd, RZ, RZ, Rsrc
    return mk(0x00007224ull | ((uint64_t)rd << 16) | ((uint64_t)0xff << 24) | ((uint64_t)0xff << 32),
              0x078e0000 | rsrc, ctrl);
}

struct sass_insn sass_exit(uint32_t ctrl)
{
    return mk(0x0000794dull, 0x03800000, ctrl);
}

struct sass_insn sass_bssy(uint8_t bar, int32_t rel, uint32_t ctrl)
{
    // rel is target - (this + 0x10)
    return mk(0x00007945ull | ((uint64_t)bar << 16) | ((uint64_t)(uint32_t)rel << 32),
              0x03800000 | (rel < 0 ? 0x0003ffff : 0), ctrl);
}

struct sass_insn sass_bsync(uint8_t bar, uint32_t ctrl)
{
    return mk(0x00007941ull | ((uint64_t)bar << 16), 0x03800000, ctrl);
}

struct sass_insn sass_bra(int32_t rel, uint8_t pred, int negate, uint32_t ctrl)
{
    // the offset is 50-bit, its high 18 bits in hi[0:17]
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
