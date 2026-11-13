// What differs between the GPUs GSP-RM runs that we drive: the classes each engine is reached through, limits,
// and the GSP-RM image's signature section. One entry per family of chipsets (PMC_BOOT_0 [28:20]) that take the same
// classes and firmware (nouveau, open-gpu-kernel-modules); docs/gpu-support.md has what each still needs.
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
    const char *missing;  // what the driver lacks for the family, NULL once it drives it
};

// the chipset's entry, NULL for one outside every family
const struct chip *chip_find(uint32_t chipset);
// "GA104", NULL for a chipset without a known name
const char *chip_name(uint32_t chipset);
// the product's name by PCI device ID, NULL when it isn't listed; the chip comes from PMC_BOOT_0, not this
const char *chip_product(uint16_t device);
