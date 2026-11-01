// CPU windows onto VRAM through BAR1. GSP-RM sets BAR1 up with 64 KB pages and page tables in VRAM; we
// keep its directories and add big-page tables under its PD0, written through PRAMIN like everything there.
// GSP-RM maps no VRAM into BAR1 itself, whatever its size: a static mapping of all of it is the client's to make
// (kbusEnableStaticBar1Mapping in CPU-RM), and with a BAR1 that covers VRAM we make one too, with 2 MB pages
#include "bar1.h"
#include "gsp.h"
#include "mem.h"

#define BIG 0x10000ull
#define PD0_SPAN 0x200000ull  // one PD0 entry
#define PD1_SPAN (PD0_SPAN * BAR1_MAX_PTS)
#define PT_ENTRIES (PD0_SPAN / BIG)

static uint64_t rd64(struct gpu *g, uint64_t a)
{
    return (uint64_t)vram_rd32(g, a + 4) << 32 | vram_rd32(g, a);
}

static void wr64(struct gpu *g, uint64_t a, uint64_t v)
{
    vram_wr32(g, a, (uint32_t)v);
    vram_wr32(g, a + 4, (uint32_t)(v >> 32));
}

// a directory entry pointing into VRAM, or 0
static uint64_t vram_pde(uint64_t e)
{
    return ((e >> 1) & 3) == 1 ? (e >> 8) << 12 | (e & 0xf0) << 4 : 0;
}

// a valid PTE or a directory in any aperture; empty and sparse entries are free to take
static int in_use(uint64_t e)
{
    return (e & 7) != 0;
}

// the PD0 entry for BAR1 offset va, 0 when its PD1 entry has no PD0
static uint64_t pd0_entry(struct bar1 *b, uint64_t va)
{
    uint64_t pd0 = va < PD1_SPAN ? b->pd0 : vram_pde(rd64(gsp_gpu(b->s), b->pd1 + va / PD1_SPAN * 8));
    return pd0 ? pd0 + va % PD1_SPAN / PD0_SPAN * 16 : 0;
}

static void unmap_static(struct bar1 *b, uint64_t end)
{
    struct gpu *g = gsp_gpu(b->s);
    for (uint64_t va = VRAM_BASE; va < end; va += PD0_SPAN) {
        if (!b->pd0_owned[va / PD1_SPAN])
            wr64(g, pd0_entry(b, va), 0);
    }
    for (unsigned i = 0; i < BAR1_PD1_ENTRIES; i++) {
        if (b->pd0_owned[i]) {
            uint64_t pd0 = vram_pde(rd64(g, b->pd1 + i * 8));
            wr64(g, b->pd1 + i * 8, 0);
            vram_free(gsp_vram(b->s), pd0, 0x1000);
            b->pd0_owned[i] = 0;
        }
    }
    vmm_flush(g);
}

// usable VRAM up to end at the same BAR1 offsets, in 2 MB pages; only into entries
// nobody else uses (or a mapping of ours an earlier session left), so a failure leaves BAR1 as it was
static int map_static(struct bar1 *b, uint64_t end)
{
    struct gpu *g = gsp_gpu(b->s);
    for (uint64_t va = VRAM_BASE; va < end; va += PD0_SPAN) {
        uint64_t e = pd0_entry(b, va);
        if (!e) {
            if (in_use(rd64(g, b->pd1 + va / PD1_SPAN * 8)))
                return -1;
        } else if (rd64(g, e) != (va >> 4 | 1) && (in_use(rd64(g, e)) || in_use(rd64(g, e + 8)))) {
            return -1;
        }
    }
    for (uint64_t va = VRAM_BASE; va < end; va += PD0_SPAN) {
        if (!pd0_entry(b, va)) {
            uint64_t pd0 = vram_alloc(gsp_vram(b->s), 0x1000, 0x1000);
            if (!pd0) {
                unmap_static(b, va);
                return -1;
            }
            vram_zero(g, pd0, BAR1_MAX_PTS * 16);
            wr64(g, b->pd1 + va / PD1_SPAN * 8, pd0 >> 4 | 1 << 1);  // PD0, VRAM
            b->pd0_owned[va / PD1_SPAN] = 1;
        }
        uint64_t e = pd0_entry(b, va);
        wr64(g, e + 8, 0);
        wr64(g, e, va >> 4 | 1);  // 2 MB page: VRAM, valid
    }
    vmm_flush(g);
    return 0;
}

int bar1_init(struct gsp *s, struct bar1 *b)
{
    struct gpu *g = gsp_gpu(s);
    *b = (struct bar1){ .s = s, .size = bar_size(g, 1) };
    // PD3 -> PD2 -> PD1 -> PD0: BAR1 lives in the first entry of each
    uint64_t t = gsp_get_info(s)->bar1_pdb;
    for (int l = 0; l < 2 && t; l++)
        t = vram_pde(rd64(g, t));
    b->pd1 = t;
    b->pd0 = t ? vram_pde(rd64(g, t)) : 0;
    if (!b->pd0) {
        log_msg("bar1: GSP-RM's BAR1 page directories are not where expected\n");
        return -1;
    }
    uint64_t fb = gsp_get_info(s)->fb_size;
    uint64_t end = (fb - VRAM_TOP_RESERVED) & ~(PD0_SPAN - 1);
    if (b->size >= fb && end <= PD1_SPAN * BAR1_PD1_ENTRIES) {
        if (!map_static(b, end)) {
            b->static_end = end;
            log_msg("bar1: %llu MB, static mapping of VRAM\n", (unsigned long long)(b->size >> 20));
            return 0;
        }
        log_msg("bar1: covers VRAM, but its directories above the framebuffer slots are in use\n");
    }
    log_msg("bar1: %llu MB, page tables\n", (unsigned long long)(b->size >> 20));
    if (b->size > PD0_SPAN * BAR1_MAX_PTS)
        b->size = PD0_SPAN * BAR1_MAX_PTS;
    // GSP-RM keeps the first 2 MB slot, so a window is never at offset 0; the framebuffers' slots follow it
    vram_range(&b->va, BAR1_FB_OFF + BAR1_FB_SLOTS * BAR1_FB_SIZE, b->size);
    return 0;
}

static int pt_for(struct bar1 *b, unsigned i)
{
    struct gpu *g = gsp_gpu(b->s);
    if (b->pts[i])
        return 0;
    uint64_t pde = rd64(g, b->pd0 + i * 16);
    if (vram_pde(pde)) {
        b->pts[i] = vram_pde(pde);  // GSP-RM's own, keep it
        return 0;
    }
    uint64_t pt = vram_alloc(gsp_vram(b->s), 0x1000, 0x1000);
    if (!pt)
        return -1;
    vram_zero(g, pt, PT_ENTRIES * 8);
    wr64(g, b->pd0 + i * 16, pt >> 4 | 1 << 1);  // big-page table, VRAM
    b->pts[i] = pt;
    b->owned[i] = 1;
    return 0;
}

static int map_at(struct bar1 *b, uint64_t off, uint64_t vram, uint64_t size)
{
    struct gpu *g = gsp_gpu(b->s);
    for (uint64_t o = 0; o < size; o += BIG) {
        uint64_t va = off + o;
        if (pt_for(b, (unsigned)(va / PD0_SPAN))) {
            bar1_unmap(b, off, o);
            return -1;
        }
        wr64(g, b->pts[va / PD0_SPAN] + (va % PD0_SPAN) / BIG * 8, (vram + o) >> 4 | 1);  // VRAM, valid
    }
    vmm_flush(g);
    return 0;
}

uint64_t bar1_map(struct bar1 *b, uint64_t vram, uint64_t size)
{
    size = (size + BIG - 1) & ~(BIG - 1);
    if (vram & (BIG - 1))
        return 0;
    if (b->static_end)
        return vram >= VRAM_BASE && vram + size <= b->static_end ? vram : 0;
    uint64_t off = vram_alloc(&b->va, size, BIG);
    if (!off)
        return 0;
    if (map_at(b, off, vram, size)) {
        vram_free(&b->va, off, size);
        return 0;
    }
    return off;
}

int bar1_map_fb(struct bar1 *b, int slot, uint64_t vram, uint64_t size)
{
    uint64_t off = BAR1_FB_OFF + (uint64_t)slot * BAR1_FB_SIZE;
    size = (size + BIG - 1) & ~(BIG - 1);
    if (slot < 0 || slot >= BAR1_FB_SLOTS || (vram & (BIG - 1)) || size > BAR1_FB_SIZE)
        return -1;
    bar1_unmap(b, off, BAR1_FB_SIZE);
    return map_at(b, off, vram, size);
}

void bar1_unmap(struct bar1 *b, uint64_t off, uint64_t size)
{
    struct gpu *g = gsp_gpu(b->s);
    size = (size + BIG - 1) & ~(BIG - 1);
    for (uint64_t o = 0; o < size; o += BIG) {
        uint64_t va = off + o;
        if (b->pts[va / PD0_SPAN])
            wr64(g, b->pts[va / PD0_SPAN] + (va % PD0_SPAN) / BIG * 8, 0);
    }
    vmm_flush(g);
}

void bar1_free(struct bar1 *b, uint64_t off, uint64_t size)
{
    if (b->static_end)
        return;
    bar1_unmap(b, off, size);
    vram_free(&b->va, off, (size + BIG - 1) & ~(BIG - 1));
}

void bar1_fini(struct bar1 *b)
{
    struct gpu *g = gsp_gpu(b->s);
    if (b->static_end)
        unmap_static(b, b->static_end);
    for (unsigned i = 0; i < BAR1_MAX_PTS; i++) {
        if (b->owned[i]) {
            wr64(g, b->pd0 + i * 16, 0);
            vram_free(gsp_vram(b->s), b->pts[i], 0x1000);
        }
    }
    vmm_flush(g);
    vram_destroy(&b->va);
}
