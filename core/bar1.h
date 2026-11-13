#pragma once
#include "mem.h"

#define BAR1_MAX_PTS 256  // 512 MB of BAR1, PD0's reach
#define BAR1_PD1_ENTRIES 512  // PD1's reach, 256 GB of BAR1 in PD0s
// one scanout slot per screen so framebuffer apertures are known before GSP-RM is up; slot i at OFF + i * SIZE
#define BAR1_FB_OFF 0x200000ull
#define BAR1_FB_SIZE 0x2200000ull  // 34 MB: 3840x2160 at 32 bpp
#define BAR1_FB_SLOTS 4  // the most heads Turing to Ada have; 138 MB of a 256 MB BAR1

struct bar1 {
    struct gsp *s;
    uint64_t size, pd1, pd0;  // usable BAR1, GSP-RM's PD1 and PD0 for it
    // resizable BAR: usable VRAM below this is identity mapped once, so windows are free; 0 when they take page tables
    uint64_t static_end;
    struct vram va;      // free BAR1 offsets
    uint64_t pts[BAR1_MAX_PTS];  // big-page table per 2 MB
    uint8_t owned[BAR1_MAX_PTS];
    uint8_t pd0_owned[BAR1_PD1_ENTRIES];  // PD0s the static mapping added
};

int bar1_init(struct gsp *s, struct bar1 *b);
// a window onto VRAM (64 KB aligned); its BAR1 offset, 0 when out of space
uint64_t bar1_map(struct bar1 *b, uint64_t vram, uint64_t size);
void bar1_unmap(struct bar1 *b, uint64_t off, uint64_t size);
int bar1_map_fb(struct bar1 *b, int slot, uint64_t vram, uint64_t size);  // a surface into a framebuffer slot
void bar1_free(struct bar1 *b, uint64_t off, uint64_t size);  // unmap and give the offsets back
void bar1_fini(struct bar1 *b);
