#pragma once
#include "plat.h"

struct gsp;
struct rm_client;

struct vram_range;

struct vram {
    struct vram_range *free;
    uint64_t total, used;  // bytes in the range, and handed out of it
};

// usable VRAM starts above what GSP-RM reserves for itself at the bottom and ends below what it keeps at the top:
// WPR2 with its heap, FRTS and the VGA workspace, well under 256 MB whatever the size (the heap grows 96 KB a GB)
#define VRAM_BASE 0x10000000ull
#define VRAM_TOP_RESERVED 0x10000000ull

void vram_init(struct vram *v, uint64_t fb_size);  // the usable part of a GPU's fb_size bytes of VRAM
void vram_range(struct vram *v, uint64_t start, uint64_t end);  // or any other range
uint64_t vram_alloc(struct vram *v, uint64_t size, uint64_t align);  // 0 when full
void vram_free(struct vram *v, uint64_t addr, uint64_t size);
void vram_destroy(struct vram *v);

// CPU access to VRAM through the 1 MB PRAMIN window in BAR0
void vram_wr32(struct gpu *g, uint64_t addr, uint32_t v);
uint32_t vram_rd32(struct gpu *g, uint64_t addr);
void vram_zero(struct gpu *g, uint64_t addr, uint64_t size);
void vram_flush(struct gpu *g);  // after PRAMIN writes something else will read

#define VMM_PRIV 1
#define VMM_RO 2
// system memory the GPU's L2 may keep (no VOL): whoever maps it flushes and invalidates L2 around the work using it
#define VMM_CACHED 4
#define VMM_KINDF(k) ((unsigned)(k) << 8)  // PTE kind
#define VMM_KIND(flags) (((flags) >> 8) & 0xff)

struct pt_node;

struct vmm {
    struct gsp *s;
    uint32_t client, handle;
    struct pt_node *root;
    uint64_t next_va;
    struct vmm *va_from;  // another vmm whose addresses this one shares (each VA names one mapping), or NULL
    struct vram_range *va_free;  // addresses below next_va given back
};

// base: where vmm_alloc starts handing out addresses (0 for the default)
int vmm_create(struct gsp *s, const struct rm_client *c, uint32_t handle, uint64_t base, struct vmm *v);
// v's page tables, kept across a power cycle, handed to a new GSP-RM under a VA space object of c's
int vmm_rebind(struct gsp *s, const struct rm_client *c, struct vmm *v);
uint64_t vmm_alloc(struct vmm *v, uint64_t size, uint64_t align);  // 0 when full
void vmm_free(struct vmm *v, uint64_t va, uint64_t size);  // an unmapped range, for vmm_alloc to hand out again
int vmm_map_sys(struct vmm *v, uint64_t va, const struct dma *mem, uint64_t offset, uint64_t size, unsigned flags);
int vmm_map_vram(struct vmm *v, uint64_t va, uint64_t vram, uint64_t size, unsigned flags);
int vmm_map_vram_big(struct vmm *v, uint64_t va, uint64_t vram, uint64_t size, unsigned flags);  // on 64 KB, all three
void vmm_unmap(struct vmm *v, uint64_t va, uint64_t size);
void vmm_flush(struct gpu *g);
void vmm_destroy(struct vmm *v);
