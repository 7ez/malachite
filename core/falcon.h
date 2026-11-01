#pragma once
#include "plat.h"

struct falcon {
    const char *name;
    uint32_t base, base2;
};

extern const struct falcon gsp_falcon, sec2_falcon;

struct falcon_load {
    uint64_t iova;  // ucode image in DMA memory
    uint32_t imem_src, imem_dst, imem_len;
    uint32_t dmem_src, dmem_dst, dmem_len;
    uint32_t pkc_offset, engine_mask, ucode_id;
    uint32_t boot_addr;
};

int poll32(struct gpu *g, uint32_t reg, uint32_t mask, uint32_t want, int timeout_us);
int fuse_version(struct gpu *g, uint32_t engine_mask, uint32_t ucode_id);

uint32_t flcn_rd(struct gpu *g, const struct falcon *f, uint32_t reg);
void flcn_wr(struct gpu *g, const struct falcon *f, uint32_t reg, uint32_t v);

int falcon_reset_eng(struct gpu *g, const struct falcon *f);
int falcon_reset(struct gpu *g, const struct falcon *f);
int falcon_load(struct gpu *g, const struct falcon *f, const struct falcon_load *l);
void falcon_dma_reset(struct gpu *g, const struct falcon *f);
void falcon_start(struct gpu *g, const struct falcon *f);
int falcon_wait_halt(struct gpu *g, const struct falcon *f);
int falcon_boot(struct gpu *g, const struct falcon *f, uint32_t *mbox0, uint32_t *mbox1);
int falcon_riscv_active(struct gpu *g, const struct falcon *f);
