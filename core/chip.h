// Per-family engine classes, limits and GSP-RM signature section, by chipset (PMC_BOOT_0 [28:20]), from nouveau and
// open-gpu-kernel-modules; docs/gpu-support.md has what each family still needs.
#pragma once
#include <stdint.h>

struct chip {
    uint16_t first, last;  // the chipsets it covers
    const char *family;    // for the log
    uint32_t gpfifo, eng3d, compute, copy, usermode;  // channel, 3D, compute, copy engine, usermode classes
    uint32_t nvdec;                                   // video decoder
    uint32_t disp, core, wndw, wimm, curs;  // display root, core channel, window, window immediate, cursor
    uint32_t max_shared;  // bytes of shared memory one CTA may take
    const char *fwsig;    // the GSP-RM ELF's signature section for the chip
    const char *fwdir;    // folder in 570/ with the family's bootloader and booters, NULL for GA10x's beside gsp.bin
    const char *missing;  // what the driver lacks for the family, NULL once it drives it
};

// NULL outside every family
const struct chip *chip_find(uint32_t chipset);
// "GA104", NULL if unknown
const char *chip_name(uint32_t chipset);
// by PCI device ID, NULL if unlisted; the chip comes from PMC_BOOT_0, not this
const char *chip_product(uint16_t device);
