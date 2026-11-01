// Memory objects the CPU-side RM owns: VRAM (NV01_MEMORY_LOCAL_USER), system memory (NV01_MEMORY_SYSTEM),
// GPU VA ranges (NV50_MEMORY_VIRTUAL) and the mappings MAP_MEMORY_DMA puts in them
#include "nvmisc.h"
#include "nvos.h"
#include "class/cl003e.h"
#include "class/cl0040.h"
#include "class/cl50a0.h"
#include "class/cl90f1.h"

#include "../gsp.h"
#include "rmfe_priv.h"

#define PAGE 0x1000ull

// GA10x PTE kinds, as RM derives them from a surface's type and attributes
enum { KIND_PITCH = 0x0, KIND_Z16 = 0x1, KIND_S8 = 0x2, KIND_S8Z24 = 0x3, KIND_ZF32_X24S8 = 0x4,
       KIND_Z24S8 = 0x5, KIND_GENERIC = 0x6 };

static uint32_t surface_kind(const NV_MEMORY_ALLOCATION_PARAMS *p)
{
    if (p->format)
        return p->format;
    switch (DRF_VAL(OS32, _ATTR, _ZS_PACKING, p->attr)) {
    case NVOS32_ATTR_ZS_PACKING_Z16:
        return p->type == NVOS32_TYPE_DEPTH ? KIND_Z16 : KIND_GENERIC;
    case NVOS32_ATTR_ZS_PACKING_S8:  // same value as Z24S8; the depth tells them apart
        if (p->type != NVOS32_TYPE_STENCIL)
            return KIND_GENERIC;
        return DRF_VAL(OS32, _ATTR, _DEPTH, p->attr) == NVOS32_ATTR_DEPTH_8 ? KIND_S8 : KIND_Z24S8;
    case NVOS32_ATTR_ZS_PACKING_S8Z24:
        return KIND_S8Z24;
    case NVOS32_ATTR_ZS_PACKING_Z32_X24S8:
        return KIND_ZF32_X24S8;
    default:
        return KIND_GENERIC;
    }
}

static uint64_t page_align(const NV_MEMORY_ALLOCATION_PARAMS *p)
{
    uint64_t align = PAGE;
    switch (DRF_VAL(OS32, _ATTR, _PAGE_SIZE, p->attr)) {
    case NVOS32_ATTR_PAGE_SIZE_BIG:
        align = 0x10000;
        break;
    case NVOS32_ATTR_PAGE_SIZE_HUGE:
        align = 0x200000;
        break;
    }
    if ((p->flags & NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE) && p->alignment > align)
        align = p->alignment;
    return align;
}

static struct obj *find_vaspace(struct client *c, uint32_t handle)
{
    struct obj *o = obj_find(c, handle);
    return o && o->kind == K_VASPACE ? o : NULL;
}

NV_STATUS rmfe_alloc_memory(struct rmfe *fe, struct client *c, uint32_t parent, uint32_t handle, uint32_t cls, void *params)
{
    NV_MEMORY_ALLOCATION_PARAMS *p = params;
    if (!p->size)
        return NV_ERR_INVALID_ARGUMENT;
    uint64_t size = (p->size + PAGE - 1) & ~(PAGE - 1);
    uint64_t align = page_align(p);

    switch (cls) {
    case NV01_MEMORY_LOCAL_USER: {
        // whole 64 KB pages, so a BAR1 window onto it never shows a neighbour
        size = (size + 0xffff) & ~0xffffull;
        if (align < 0x10000)
            align = 0x10000;
        uint64_t addr = vram_alloc(gsp_vram(fe->s), size, align);
        if (!addr)
            return NV_ERR_NO_MEMORY;
        struct obj *o = obj_add(c, handle, parent, cls, K_VRAM);
        o->addr = addr;
        o->size = size;
        o->pte_kind = p->format;
        p->offset = addr;
        break;
    }
    case NV01_MEMORY_SYSTEM: {
        struct obj *o = obj_add(c, handle, parent, cls, K_SYSMEM);
        if (dma_alloc(gsp_gpu(fe->s), &o->dma, size, 0)) {
            o->kind = K_GSP;  // nothing to release
            return NV_ERR_NO_MEMORY;
        }
        o->size = size;
        p->offset = 0;
        break;
    }
    case NV50_MEMORY_VIRTUAL: {
        struct obj *vas = find_vaspace(c, p->hVASpace);
        if (!vas)
            return NV_ERR_INVALID_OBJECT_HANDLE;
        uint64_t va = vmm_alloc(vas->vmm, size, align);
        if (!va)
            return NV_ERR_NO_MEMORY;
        struct obj *o = obj_add(c, handle, parent, cls, K_VIRTUAL);
        o->vmm = vas->vmm;
        o->addr = va;
        o->size = size;
        o->pte_kind = surface_kind(p);
        p->offset = va;
        p->format = o->pte_kind;
        break;
    }
    }
    p->size = size;
    p->limit = p->offset + size - 1;
    p->alignment = align;
    return NV_OK;
}

void rmfe_release_memory(struct rmfe *fe, struct obj *o)
{
    switch (o->kind) {
    case K_VRAM:
        while (o->maps)
            rmfe_unmap_bar1(fe, o);
        vram_free(gsp_vram(fe->s), o->addr, o->size);
        break;
    case K_SYSMEM:
        dma_free(gsp_gpu(fe->s), &o->dma);
        break;
    case K_VIRTUAL:
        while (o->maps) {
            struct gpu_map *m = o->maps;
            o->maps = m->next;
            vmm_unmap(o->vmm, m->va, m->size);
            mem_free(m);
        }
        vmm_flush(gsp_gpu(fe->s));
        break;
    default:
        break;
    }
}

static int bar1_up(struct rmfe *fe)
{
    if (!fe->bar1_up) {
        if (bar1_init(fe->s, &fe->bar1))
            return -1;
        fe->bar1_up = 1;
    }
    return 0;
}

int rmfe_bar1_fb(struct rmfe *fe, int slot, uint64_t vram, uint64_t size)
{
    return bar1_up(fe) ? -1 : bar1_map_fb(&fe->bar1, slot, vram, size);
}

// a window of BAR1 onto VRAM for the CPU; BAR1 is small (256 MB without resizable BAR), so windows come and go
NV_STATUS rmfe_map_bar1(struct rmfe *fe, struct obj *o, uint64_t offset, uint64_t size, uint64_t *bar1)
{
    if (!size || offset + size > o->size)
        return NV_ERR_INVALID_LIMIT;
    if (bar1_up(fe))
        return NV_ERR_NOT_SUPPORTED;
    uint64_t off = bar1_map(&fe->bar1, o->addr + offset, size);
    if (!off)
        return NV_ERR_NO_MEMORY;
    // once per session: a CPU write through the first window has to land in the VRAM behind it
    struct gpu *g = gsp_gpu(fe->s);
    volatile uint8_t *cpu = fe->bar1_checked ? NULL : bar1_cpu(g, off);
    if (cpu) {
        uint32_t old = vram_rd32(g, o->addr + offset);
        *(volatile uint32_t *)cpu = 0x5ca1ab1e;
        __sync_synchronize();
        uint32_t back = vram_rd32(g, o->addr + offset);
        log_msg("bar1: window at %#llx onto vram %#llx %s (read %#x)\n", (unsigned long long)off,
                (unsigned long long)(o->addr + offset), back == 0x5ca1ab1e ? "works" : "BROKEN", back);
        vram_wr32(g, o->addr + offset, old);
        fe->bar1_checked = 1;
    }
    struct gpu_map *m = mem_alloc(sizeof(*m));
    *m = (struct gpu_map){ off, size, o->maps };
    o->maps = m;
    *bar1 = off;
    return NV_OK;
}

// the oldest window: the escape doesn't say which CPU mapping went away
NV_STATUS rmfe_unmap_bar1(struct rmfe *fe, struct obj *o)
{
    struct gpu_map **pp = &o->maps;
    if (!*pp)
        return NV_ERR_INVALID_ARGUMENT;
    while ((*pp)->next)
        pp = &(*pp)->next;
    struct gpu_map *m = *pp;
    bar1_free(&fe->bar1, m->va, m->size);
    *pp = NULL;
    mem_free(m);
    return NV_OK;
}

NV_STATUS rmfe_map_dma(struct rmfe *fe, struct client *c, void *params)
{
    NVOS46_PARAMETERS *p = params;
    struct obj *virt = obj_find(c, p->hDma), *mem = obj_find(c, p->hMemory);
    if (!virt || virt->kind != K_VIRTUAL || !mem)
        return NV_ERR_INVALID_OBJECT_HANDLE;
    uint64_t len = (p->length + PAGE - 1) & ~(PAGE - 1);
    uint64_t mem_size = mem->kind == K_VRAM ? mem->size : mem->kind == K_SYSMEM ? mem->dma.size : 0;
    if (!len || p->offset + len > mem_size || p->dmaOffset + len > virt->size)
        return NV_ERR_INVALID_LIMIT;

    uint64_t va = virt->addr + p->dmaOffset;
    unsigned flags = VMM_KINDF(virt->pte_kind);
    int ret = mem->kind == K_VRAM ? vmm_map_vram(virt->vmm, va, mem->addr + p->offset, len, flags)
                                  : vmm_map_sys(virt->vmm, va, &mem->dma, p->offset, len, flags);
    if (ret)
        return NV_ERR_NO_MEMORY;
    vmm_flush(gsp_gpu(fe->s));

    struct gpu_map *m = mem_alloc(sizeof(*m));
    *m = (struct gpu_map){ va, len, virt->maps };
    virt->maps = m;
    p->dmaOffset = va;
    return NV_OK;
}

NV_STATUS rmfe_unmap_dma(struct rmfe *fe, struct client *c, void *params)
{
    NVOS47_PARAMETERS *p = params;
    struct obj *virt = obj_find(c, p->hDma);
    if (!virt || virt->kind != K_VIRTUAL)
        return NV_ERR_INVALID_OBJECT_HANDLE;
    for (struct gpu_map **pp = &virt->maps; *pp; pp = &(*pp)->next) {
        struct gpu_map *m = *pp;
        if (m->va != p->dmaOffset)
            continue;
        vmm_unmap(virt->vmm, m->va, m->size);
        vmm_flush(gsp_gpu(fe->s));
        *pp = m->next;
        mem_free(m);
        return NV_OK;
    }
    return NV_ERR_INVALID_OFFSET;
}
