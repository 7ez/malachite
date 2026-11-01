// What differs between the GPUs GSP-RM runs that we drive: the classes each engine is reached through, limits,
// and the GSP-RM image's signature section. One entry per architecture, the chipset's high bits (PMC_BOOT_0
// [28:20] >> 4); within one, the chips take the same classes and firmware layout (nouveau, open-gpu-kernel-modules).
#pragma once
#include <stdint.h>

struct chip {
    uint32_t arch;       // 0x17 Ampere (GA10x), 0x19 Ada (AD10x)
    const char *family;  // for the log
    uint32_t gpfifo, eng3d, compute, copy, usermode;  // channel, 3D, compute, copy engine, usermode classes
    uint32_t nvdec;                                   // video decoder
    uint32_t disp, core, wndw, wimm, curs;  // display root, core channel, window, window immediate, cursor
    uint32_t max_shared;  // bytes of shared memory one CTA may take
    const char *fwsig;    // the GSP-RM ELF's signature section for the chip
};

// the chip's entry, NULL for one we can't drive (Turing's boot path, QMD and memory ordering aren't done yet)
const struct chip *chip_find(uint32_t chipset);
