#include "chip.h"
#include <stddef.h>

// Ada keeps GA10x's channel, copy, usermode and window classes; its 3D, compute, video decoder, display root and
// core are new. GA100 (0x170, no display) and the Tegra parts fall outside every range.
static const struct chip chips[] = {
    { 0x162, 0x166, "TU10x", 0xc46f, 0xc597, 0xc5c0, 0xc5b5, 0xc461, 0xc4b0, 0xc570, 0xc57d, 0xc57e, 0xc57b, 0xc57a,
      64 * 1024, ".fwsignature_tu10x", "Turing's FWSEC, booter and GSP boot, QMD v2 and sm75 shaders" },
    { 0x167, 0x168, "TU11x", 0xc46f, 0xc597, 0xc5c0, 0xc5b5, 0xc461, 0xc4b0, 0xc570, 0xc57d, 0xc57e, 0xc57b, 0xc57a,
      64 * 1024, ".fwsignature_tu11x", "Turing's FWSEC, booter and GSP boot, QMD v2 and sm75 shaders" },
    { 0x172, 0x177, "GA10x", 0xc56f, 0xc797, 0xc7c0, 0xc7b5, 0xc561, 0xc7b0, 0xc670, 0xc67d, 0xc67e, 0xc67b, 0xc67a,
      99 * 1024, ".fwsignature_ga10x", NULL },
    { 0x192, 0x197, "AD10x", 0xc56f, 0xc997, 0xc9c0, 0xc7b5, 0xc561, 0xc9b0, 0xc770, 0xc77d, 0xc67e, 0xc67b, 0xc67a,
      99 * 1024, ".fwsignature_ad10x", "Ada's own booter and bootloader firmware" },
};

const struct chip *chip_find(uint32_t chipset)
{
    for (size_t i = 0; i < sizeof chips / sizeof chips[0]; i++)
        if (chipset >= chips[i].first && chipset <= chips[i].last)
            return &chips[i];
    return NULL;
}
