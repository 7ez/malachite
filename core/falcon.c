// Falcon reset, DMA load and heavy-secure boot on GA102, following nova-core (falcon.rs, falcon/hal/ga102.rs)
#include "falcon.h"

const struct falcon gsp_falcon = { "gsp", 0x110000, 0x111000 };
const struct falcon sec2_falcon = { "sec2", 0x840000, 0x841000 };

#define FALCON_MAILBOX0     0x040
#define FALCON_MAILBOX1     0x044
#define FALCON_RM           0x084
#define FALCON_HWCFG2       0x0f4
#define FALCON_CPUCTL       0x100
#define FALCON_BOOTVEC      0x104
#define FALCON_DMACTL       0x10c
#define FALCON_DMATRFBASE   0x110
#define FALCON_DMATRFMOFFS  0x114
#define FALCON_DMATRFCMD    0x118
#define FALCON_DMATRFFBOFFS 0x11c
#define FALCON_DMATRFBASE1  0x128
#define FALCON_CPUCTL_ALIAS 0x130
#define FALCON_ENGINE       0x3c0
#define FBIF_TRANSCFG0      0x600
#define FBIF_CTL            0x624

#define FALCON2_MOD_SEL        0x180
#define FALCON2_BROM_UCODE_ID  0x198
#define FALCON2_BROM_ENGIDMASK 0x19c
#define FALCON2_BROM_PARAADDR0 0x210
#define RISCV_CPUCTL           0x388
#define RISCV_BCR_CTRL         0x668

#define DMA_SIZE_256B (6 << 8)
#define DMA_IMEM      (1 << 4)
#define DMA_SECURE    (1 << 2)

int poll32(struct gpu *g, uint32_t reg, uint32_t mask, uint32_t want, int timeout_us)
{
    uint64_t end = time_us() + timeout_us;
    for (;;) {
        if ((rd32(g, reg) & mask) == want)
            return 0;
        if (time_us() > end)
            return -1;
    }
}

int fuse_version(struct gpu *g, uint32_t engine_mask, uint32_t ucode_id)
{
    uint32_t reg;
    if (engine_mask & 0x1)
        reg = 0x824140;  // SEC2
    else if (engine_mask & 0x4)
        reg = 0x824100;  // NVDEC
    else if (engine_mask & 0x400)
        reg = 0x8241c0;  // GSP
    else
        return -1;
    uint16_t v = rd32(g, reg + (ucode_id - 1) * 4) & 0xffff;
    return v ? 32 - __builtin_clz(v) : 0;
}

uint32_t flcn_rd(struct gpu *g, const struct falcon *f, uint32_t reg) { return rd32(g, f->base + reg); }
void flcn_wr(struct gpu *g, const struct falcon *f, uint32_t reg, uint32_t v) { wr32(g, f->base + reg, v); }

int falcon_reset_eng(struct gpu *g, const struct falcon *f)
{
    poll32(g, f->base + FALCON_HWCFG2, 1u << 31, 1u << 31, 150);
    flcn_wr(g, f, FALCON_ENGINE, flcn_rd(g, f, FALCON_ENGINE) | 1);
    delay_us(10);
    flcn_wr(g, f, FALCON_ENGINE, flcn_rd(g, f, FALCON_ENGINE) & ~1u);
    if (poll32(g, f->base + FALCON_HWCFG2, 1u << 12, 0, 20000)) {
        log_msg("%s: memory scrubbing did not finish\n", f->name);
        return -1;
    }
    return 0;
}

int falcon_reset(struct gpu *g, const struct falcon *f)
{
    if (falcon_reset_eng(g, f))
        return -1;
    if (rd32(g, f->base2 + RISCV_BCR_CTRL) & (1u << 4)) {
        wr32(g, f->base2 + RISCV_BCR_CTRL, 0);
        if (poll32(g, f->base2 + RISCV_BCR_CTRL, 1, 1, 10000)) {
            log_msg("%s: core select to falcon timed out\n", f->name);
            return -1;
        }
    }
    flcn_wr(g, f, FALCON_RM, rd32(g, 0x0));
    return 0;
}

// 256-byte transfers from DMA memory at base+src into IMEM/DMEM at dst
static int dma_wr(struct gpu *g, const struct falcon *f, uint64_t base, uint32_t src, uint32_t dst,
                  uint32_t len, uint32_t cmd)
{
    flcn_wr(g, f, FALCON_DMATRFBASE, base >> 8);
    flcn_wr(g, f, FALCON_DMATRFBASE1, base >> 40);
    for (uint32_t pos = 0; pos < len; pos += 256) {
        flcn_wr(g, f, FALCON_DMATRFMOFFS, dst + pos);
        flcn_wr(g, f, FALCON_DMATRFFBOFFS, src + pos);
        flcn_wr(g, f, FALCON_DMATRFCMD, cmd);
        if (poll32(g, f->base + FALCON_DMATRFCMD, 2, 2, 2000000)) {
            log_msg("%s: DMA transfer at %#x timed out\n", f->name, pos);
            return -1;
        }
    }
    return 0;
}

int falcon_load(struct gpu *g, const struct falcon *f, const struct falcon_load *l)
{
    falcon_dma_reset(g, f);
    flcn_wr(g, f, FBIF_TRANSCFG0, (flcn_rd(g, f, FBIF_TRANSCFG0) & ~0x10007u) | 1 | 4);  // coherent sysmem, physical

    if (dma_wr(g, f, l->iova, l->imem_src, l->imem_dst, l->imem_len, DMA_SIZE_256B | DMA_IMEM | DMA_SECURE) ||
        dma_wr(g, f, l->iova + l->dmem_src, 0, l->dmem_dst, l->dmem_len, DMA_SIZE_256B))
        return -1;

    wr32(g, f->base2 + FALCON2_BROM_PARAADDR0, l->pkc_offset);
    wr32(g, f->base2 + FALCON2_BROM_ENGIDMASK, l->engine_mask);
    wr32(g, f->base2 + FALCON2_BROM_UCODE_ID, l->ucode_id);
    wr32(g, f->base2 + FALCON2_MOD_SEL, 1);  // RSA-3K
    flcn_wr(g, f, FALCON_BOOTVEC, l->boot_addr);
    return 0;
}

void falcon_dma_reset(struct gpu *g, const struct falcon *f)
{
    flcn_wr(g, f, FBIF_CTL, flcn_rd(g, f, FBIF_CTL) | (1 << 7));
    flcn_wr(g, f, FALCON_DMACTL, 0);
}

void falcon_start(struct gpu *g, const struct falcon *f)
{
    if (flcn_rd(g, f, FALCON_CPUCTL) & (1 << 6))
        flcn_wr(g, f, FALCON_CPUCTL_ALIAS, 2);
    else
        flcn_wr(g, f, FALCON_CPUCTL, 2);
}

int falcon_wait_halt(struct gpu *g, const struct falcon *f)
{
    if (poll32(g, f->base + FALCON_CPUCTL, 1 << 4, 1 << 4, 2000000)) {
        log_msg("%s: did not halt (cpuctl %08x)\n", f->name, flcn_rd(g, f, FALCON_CPUCTL));
        return -1;
    }
    return 0;
}

int falcon_boot(struct gpu *g, const struct falcon *f, uint32_t *mbox0, uint32_t *mbox1)
{
    flcn_wr(g, f, FALCON_MAILBOX0, *mbox0);
    flcn_wr(g, f, FALCON_MAILBOX1, *mbox1);
    falcon_start(g, f);
    if (falcon_wait_halt(g, f))
        return -1;
    *mbox0 = flcn_rd(g, f, FALCON_MAILBOX0);
    *mbox1 = flcn_rd(g, f, FALCON_MAILBOX1);
    return 0;
}

int falcon_riscv_active(struct gpu *g, const struct falcon *f)
{
    return !!(rd32(g, f->base2 + RISCV_CPUCTL) & 0x80);
}
