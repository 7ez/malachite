// VRAM allocation and GPU address spaces with client-owned page tables in system memory.
// GP100 layout (nouveau's vmmgp100.c): PD3 2 bits, PD2 9, PD1 9, PD0 8 (dual 16-byte entries), PT 9 of 4 KB pages.
// The VA space is externally owned (SET_PAGE_DIRECTORY): a split layout leaves channels on GSP-RM's root, without ours.
#include "../nvrm/fifo.h"
#include "gsp.h"
#include "falcon.h"
#include "mem.h"

#define PAGE 0x1000ull
#define BIG_PAGE 0x10000ull
#define VMM_BASE (1ull << 32)  // above the shader memory windows, so every address is a 64-bit one

#define APER_VRAM 0
#define APER_SYS 2

struct vram_range {
    uint64_t start, end;
    struct vram_range *next;
};

void vram_init(struct vram *v, uint64_t fb_size)
{
    vram_range(v, VRAM_BASE, fb_size - VRAM_TOP_RESERVED);
}

// the same allocator over another range, such as the BAR1 window
void vram_range(struct vram *v, uint64_t start, uint64_t end)
{
    v->free = mem_alloc(sizeof(*v->free));
    *v->free = (struct vram_range){ start, end, NULL };
    v->total = end - start;
    v->used = 0;
}

// first fit over a sorted free list of ranges; 0 when none is big enough
static uint64_t ranges_take(struct vram_range **head, uint64_t size, uint64_t align)
{
    for (struct vram_range **pp = head; *pp; pp = &(*pp)->next) {
        struct vram_range *r = *pp;
        uint64_t addr = (r->start + align - 1) & ~(align - 1);
        if (addr + size > r->end)
            continue;
        if (addr + size < r->end) {
            struct vram_range *tail = mem_alloc(sizeof(*tail));
            *tail = (struct vram_range){ addr + size, r->end, r->next };
            r->next = tail;
        }
        r->end = addr;
        if (r->start == r->end) {
            *pp = r->next;
            mem_free(r);
        }
        return addr;
    }
    return 0;
}

// a range given back, joined to the ones either side of it
static void ranges_put(struct vram_range **head, uint64_t addr, uint64_t size)
{
    struct vram_range *prev = NULL, **pp = head;
    while (*pp && (*pp)->start < addr) {
        prev = *pp;
        pp = &prev->next;
    }
    struct vram_range *r = *pp;
    if (prev && prev->end == addr) {
        prev->end = addr + size;
        r = prev;
    } else {
        r = mem_alloc(sizeof(*r));
        *r = (struct vram_range){ addr, addr + size, *pp };
        *pp = r;
    }
    if (r->next && r->end == r->next->start) {
        struct vram_range *b = r->next;
        r->end = b->end;
        r->next = b->next;
        mem_free(b);
    }
}

static void ranges_destroy(struct vram_range **head)
{
    while (*head) {
        struct vram_range *r = *head;
        *head = r->next;
        mem_free(r);
    }
}

uint64_t vram_alloc(struct vram *v, uint64_t size, uint64_t align)
{
    size = (size + PAGE - 1) & ~(PAGE - 1);
    if (align < PAGE)
        align = PAGE;
    uint64_t addr = ranges_take(&v->free, size, align);
    if (addr)
        v->used += size;
    else
        log_msg("vram: out of space for %#llx bytes\n", (unsigned long long)size);
    return addr;
}

void vram_free(struct vram *v, uint64_t addr, uint64_t size)
{
    size = (size + PAGE - 1) & ~(PAGE - 1);
    v->used -= size;
    ranges_put(&v->free, addr, size);
}

void vram_destroy(struct vram *v)
{
    ranges_destroy(&v->free);
}

static uint64_t pde_sys(uint64_t addr) { return (addr >> 4) | (2 << 1) | (1 << 3); }  // SYS_COH, VOL

// one node per page directory or table; PD0 holds dual 16-byte entries, so 256 children
struct pt_node {
    struct dma mem;
    struct pt_node *child[512];
};

static const struct { uint8_t shift, bits, entry; } levels[] = {
    { 47, 2, 8 },   // PD3
    { 38, 9, 8 },   // PD2
    { 29, 9, 8 },   // PD1
    { 21, 8, 16 },  // PD0
    { 12, 9, 8 },   // PT
};

static struct pt_node *node_new(struct gsp *s)
{
    struct pt_node *n = mem_alloc(sizeof(*n));
    if (dma_alloc(gsp_gpu(s), &n->mem, PAGE, 1)) {
        mem_free(n);
        return NULL;
    }
    return n;
}

static void node_free(struct gsp *s, struct pt_node *n, int level)
{
    if (!n)
        return;
    for (int i = 0; level < 4 && i < 512; i++)
        node_free(s, n->child[i], level + 1);
    dma_free(gsp_gpu(s), &n->mem);
    mem_free(n);
}

// v's root handed to GSP-RM under a VA space object of c's
static int vmm_register(struct gsp *s, const struct rm_client *c, struct vmm *v)
{
    NV_VASPACE_ALLOCATION_PARAMETERS vas = {
        .index = NV_VASPACE_ALLOCATION_INDEX_GPU_NEW,
        .flags = NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED,
    };
    NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS pd = {
        .physAddress = v->root->mem.addr,
        .numEntries = 4,
        .flags = NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_SYSMEM_COH,
        .hVASpace = v->handle,
    };
    if (rm_alloc(s, c->client, c->device, v->handle, FERMI_VASPACE_A, &vas, sizeof(vas)) ||
        !rm_control(s, c->client, c->device, NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY, &pd, sizeof(pd)))
        return -1;
    return 0;
}

int vmm_create(struct gsp *s, const struct rm_client *c, uint32_t handle, uint64_t base, struct vmm *v)
{
    *v = (struct vmm){ .s = s, .handle = handle, .client = c->client, .next_va = base ? base : VMM_BASE };
    v->root = node_new(s);
    if (!v->root)
        return -1;
    return vmm_register(s, c, v);
}

int vmm_rebind(struct gsp *s, const struct rm_client *c, struct vmm *v)
{
    v->s = s;
    v->client = c->client;
    return vmm_register(s, c, v);
}

uint64_t vmm_alloc(struct vmm *v, uint64_t size, uint64_t align)
{
    if (align < PAGE)
        align = PAGE;
    if (v->va_from)
        v = v->va_from;
    // freed addresses first, so map/unmap churn reuses the page tables already there
    size = (size + PAGE - 1) & ~(PAGE - 1);
    uint64_t va = ranges_take(&v->va_free, size, align);
    if (va)
        return va;
    va = (v->next_va + align - 1) & ~(align - 1);
    // 0xfe000000..0xffffffff is the shared/local memory window (compute.c); access there is Invalid Address Space
    if (va < 1ull << 32 && va + size > 0xfeull << 24)
        va = 1ull << 32;
    if (va + size > 1ull << 49) {
        log_msg("vmm: out of address space for %#llx bytes\n", (unsigned long long)size);
        return 0;
    }
    v->next_va = va + size;
    return va;
}

// an address range vmm_alloc gave out, unmapped, free for the next
void vmm_free(struct vmm *v, uint64_t va, uint64_t size)
{
    if (v->va_from)
        v = v->va_from;
    ranges_put(&v->va_free, va, (size + PAGE - 1) & ~(PAGE - 1));
}

// made on first use; a PD0 entry pairs big- and small-page tables, the big ones in child[256..511]
static struct pt_node *pt_child(struct vmm *v, struct pt_node *n, int l, uint64_t va, int big)
{
    unsigned i = (va >> levels[l].shift) & ((1u << levels[l].bits) - 1);
    unsigned slot = big ? 256 + i : i;
    if (!n->child[slot]) {
        n->child[slot] = node_new(v->s);
        if (!n->child[slot])
            return NULL;
        uint64_t *e = (uint64_t *)((uint8_t *)n->mem.cpu + i * levels[l].entry);
        e[levels[l].entry == 16 && !big ? 1 : 0] = pde_sys(n->child[slot]->mem.addr);
    }
    return n->child[slot];
}

static struct pt_node *pd0_of(struct vmm *v, uint64_t va)
{
    struct pt_node *n = v->root;
    for (int l = 0; n && l < 3; l++)
        n = pt_child(v, n, l, va, 0);
    return n;
}

static uint64_t *pte_slot(struct vmm *v, uint64_t va)
{
    struct pt_node *n = pd0_of(v, va);
    n = n ? pt_child(v, n, 3, va, 0) : NULL;
    return n ? (uint64_t *)n->mem.cpu + ((va >> 12) & 0x1ff) : NULL;
}

static uint64_t *big_pte_slot(struct vmm *v, uint64_t va)
{
    struct pt_node *n = pd0_of(v, va);
    n = n ? pt_child(v, n, 3, va, 1) : NULL;
    return n ? (uint64_t *)n->mem.cpu + ((va >> 16) & 0x1f) : NULL;
}

static uint64_t pte(uint64_t addr, int aper, unsigned flags)
{
    uint64_t e = (addr >> 4) | 1 | (uint64_t)aper << 1 | (uint64_t)VMM_KIND(flags) << 56;
    if (aper == APER_SYS && !(flags & VMM_CACHED))
        e |= 1 << 3;  // VOL
    if (flags & VMM_PRIV)
        e |= 1 << 5;
    if (flags & VMM_RO)
        e |= 1 << 6;
    return e;
}

int vmm_map_sys(struct vmm *v, uint64_t va, const struct dma *mem, uint64_t offset, uint64_t size, unsigned flags)
{
    for (uint64_t off = 0; off < size; off += PAGE) {
        uint64_t *e = pte_slot(v, va + off);
        if (!e)
            return -1;
        *e = pte(dma_addr(mem, offset + off), APER_SYS, flags);
    }
    return 0;
}

int vmm_map_vram(struct vmm *v, uint64_t va, uint64_t vram, uint64_t size, unsigned flags)
{
    for (uint64_t off = 0; off < size; off += PAGE) {
        uint64_t *e = pte_slot(v, va + off);
        if (!e)
            return -1;
        *e = pte(vram + off, APER_VRAM, flags);
    }
    return 0;
}

// compressed kinds need 64 KB or 2 MB VRAM pages; a zero big-page entry defers to the small pages under it
int vmm_map_vram_big(struct vmm *v, uint64_t va, uint64_t vram, uint64_t size, unsigned flags)
{
    vmm_unmap(v, va, size);
    for (uint64_t off = 0; off < size; off += BIG_PAGE) {
        uint64_t *e = big_pte_slot(v, va + off);
        if (!e)
            return -1;
        *e = pte(vram + off, APER_VRAM, flags);
    }
    return 0;
}

void vmm_unmap(struct vmm *v, uint64_t va, uint64_t size)
{
    for (uint64_t off = 0; off < size; off += PAGE) {
        uint64_t *e = pte_slot(v, va + off);
        if (e)
            *e = 0;
        if (!((va + off) & (BIG_PAGE - 1)) && off + BIG_PAGE <= size) {
            struct pt_node *n = pd0_of(v, va + off);
            n = n ? n->child[256 + ((va + off) >> 21 & 0xff)] : NULL;
            if (n)
                ((uint64_t *)n->mem.cpu)[((va + off) >> 16) & 0x1f] = 0;
        }
    }
}

// drops every cached translation on the GPU; needed after changing PTEs of a VA space in use
void vmm_flush(struct gpu *g)
{
    wr32(g, 0xb830a0, 0);
    wr32(g, 0xb830a4, 0);
    wr32(g, 0xb830b0, 0x80000000 | 3);  // trigger, ALL_PDB, PAGE_ALL
    poll32(g, 0xb830b0, 0x80000000, 0, 2000000);
}

// after the VA space object is gone from GSP-RM
void vmm_destroy(struct vmm *v)
{
    node_free(v->s, v->root, 0);
    v->root = NULL;
    ranges_destroy(&v->va_free);
}

void vram_wr32(struct gpu *g, uint64_t addr, uint32_t v)
{
    wr32(g, 0x1700, addr >> 16);
    wr32(g, 0x700000 + (addr & 0xffff), v);
}

uint32_t vram_rd32(struct gpu *g, uint64_t addr)
{
    wr32(g, 0x1700, addr >> 16);
    return rd32(g, 0x700000 + (addr & 0xffff));
}

// writes through PRAMIN can sit in L2; the display engine (among others) reads memory behind it
void vram_flush(struct gpu *g)
{
    wr32(g, 0x070000, 1);  // NV_UFLUSH_FB_FLUSH
    poll32(g, 0x070000, 2, 0, 2000000);
}

void vram_zero(struct gpu *g, uint64_t addr, uint64_t size)
{
    for (uint64_t off = 0; off < size; off += 4)
        vram_wr32(g, addr + off, 0);
}
