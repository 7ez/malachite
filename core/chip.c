#include "chip.h"
#include <stddef.h>

// Ada keeps GA10x's channel, copy, usermode and window classes; its 3D, compute, video decoder, display root and
// core are new. Both give a CTA up to 99 KB of shared memory.
static const struct chip chips[] = {
    { 0x17, "GA10x", 0xc56f, 0xc797, 0xc7c0, 0xc7b5, 0xc561, 0xc7b0, 0xc670, 0xc67d, 0xc67e, 0xc67b, 0xc67a, 99 * 1024,
      ".fwsignature_ga10x" },
    { 0x19, "AD10x", 0xc56f, 0xc997, 0xc9c0, 0xc7b5, 0xc561, 0xc9b0, 0xc770, 0xc77d, 0xc67e, 0xc67b, 0xc67a, 99 * 1024,
      ".fwsignature_ad10x" },
};

const struct chip *chip_find(uint32_t chipset)
{
    if (chipset == 0x170)  // GA100: a compute part, no display
        return NULL;
    for (size_t i = 0; i < sizeof chips / sizeof chips[0]; i++)
        if (chips[i].arch == chipset >> 4)
            return &chips[i];
    return NULL;
}
