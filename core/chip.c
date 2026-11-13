#include "chip.h"
#include <stddef.h>

// Ada reuses GA10x's channel, copy, usermode and window classes. GA100 (0x170, no display) and Tegra are left out.
static const struct chip chips[] = {
    { 0x162, 0x166, "TU10x", 0xc46f, 0xc597, 0xc5c0, 0xc5b5, 0xc461, 0xc4b0, 0xc570, 0xc57d, 0xc57e, 0xc57b, 0xc57a,
      64 * 1024, ".fwsignature_tu10x", NULL, "Turing's FWSEC, booter and GSP boot, QMD v2 and sm75 shaders" },
    { 0x167, 0x168, "TU11x", 0xc46f, 0xc597, 0xc5c0, 0xc5b5, 0xc461, 0xc4b0, 0xc570, 0xc57d, 0xc57e, 0xc57b, 0xc57a,
      64 * 1024, ".fwsignature_tu11x", NULL, "Turing's FWSEC, booter and GSP boot, QMD v2 and sm75 shaders" },
    { 0x172, 0x177, "GA10x", 0xc56f, 0xc797, 0xc7c0, 0xc7b5, 0xc561, 0xc7b0, 0xc670, 0xc67d, 0xc67e, 0xc67b, 0xc67a,
      99 * 1024, ".fwsignature_ga10x", NULL, NULL },
    // boots the way GA10x does, with Ada's own booters and bootloader; untested
    { 0x192, 0x197, "AD10x", 0xc56f, 0xc997, 0xc9c0, 0xc7b5, 0xc561, 0xc9b0, 0xc770, 0xc77d, 0xc67e, 0xc67b, 0xc67a,
      99 * 1024, ".fwsignature_ad10x", "ad10x", NULL },
};

static const struct {
    uint16_t chipset;
    const char *name;
} names[] = {
    { 0x162, "TU102" }, { 0x164, "TU104" }, { 0x166, "TU106" }, { 0x167, "TU117" }, { 0x168, "TU116" },
    { 0x172, "GA102" }, { 0x173, "GA103" }, { 0x174, "GA104" }, { 0x176, "GA106" }, { 0x177, "GA107" },
    { 0x192, "AD102" }, { 0x193, "AD103" }, { 0x194, "AD104" }, { 0x196, "AD106" }, { 0x197, "AD107" },
};

// RTX 30 parts from NVIDIA's open kernel modules, minus OEM variants sharing an ID; the 3050 A laptops are Ada
static const struct {
    uint16_t device;
    const char *name;
} products[] = {
    { 0x2203, "NVIDIA GeForce RTX 3090 Ti" },
    { 0x2204, "NVIDIA GeForce RTX 3090" },
    { 0x2206, "NVIDIA GeForce RTX 3080" },
    { 0x2207, "NVIDIA GeForce RTX 3070 Ti" },
    { 0x2208, "NVIDIA GeForce RTX 3080 Ti" },
    { 0x220a, "NVIDIA GeForce RTX 3080" },
    { 0x2216, "NVIDIA GeForce RTX 3080" },
    { 0x2414, "NVIDIA GeForce RTX 3060 Ti" },
    { 0x2420, "NVIDIA GeForce RTX 3080 Ti Laptop GPU" },
    { 0x2460, "NVIDIA GeForce RTX 3080 Ti Laptop GPU" },
    { 0x2482, "NVIDIA GeForce RTX 3070 Ti" },
    { 0x2484, "NVIDIA GeForce RTX 3070" },
    { 0x2486, "NVIDIA GeForce RTX 3060 Ti" },
    { 0x2487, "NVIDIA GeForce RTX 3060" },
    { 0x2488, "NVIDIA GeForce RTX 3070" },
    { 0x2489, "NVIDIA GeForce RTX 3060 Ti" },
    { 0x249c, "NVIDIA GeForce RTX 3080 Laptop GPU" },
    { 0x249d, "NVIDIA GeForce RTX 3070 Laptop GPU" },
    { 0x24a0, "NVIDIA GeForce RTX 3070 Ti Laptop GPU" },
    { 0x24c7, "NVIDIA GeForce RTX 3060" },
    { 0x24c9, "NVIDIA GeForce RTX 3060 Ti" },
    { 0x24dc, "NVIDIA GeForce RTX 3080 Laptop GPU" },
    { 0x24dd, "NVIDIA GeForce RTX 3070 Laptop GPU" },
    { 0x24e0, "NVIDIA GeForce RTX 3070 Ti Laptop GPU" },
    { 0x2503, "NVIDIA GeForce RTX 3060" },
    { 0x2504, "NVIDIA GeForce RTX 3060" },
    { 0x2507, "NVIDIA GeForce RTX 3050" },
    { 0x2508, "NVIDIA GeForce RTX 3050 OEM" },
    { 0x2520, "NVIDIA GeForce RTX 3060 Laptop GPU" },
    { 0x2521, "NVIDIA GeForce RTX 3060 Laptop GPU" },
    { 0x2523, "NVIDIA GeForce RTX 3050 Ti Laptop GPU" },
    { 0x2544, "NVIDIA GeForce RTX 3060" },
    { 0x2560, "NVIDIA GeForce RTX 3060 Laptop GPU" },
    { 0x2563, "NVIDIA GeForce RTX 3050 Ti Laptop GPU" },
    { 0x2582, "NVIDIA GeForce RTX 3050" },
    { 0x2584, "NVIDIA GeForce RTX 3050" },
    { 0x25a0, "NVIDIA GeForce RTX 3050 Ti Laptop GPU" },
    { 0x25a2, "NVIDIA GeForce RTX 3050 Laptop GPU" },
    { 0x25a5, "NVIDIA GeForce RTX 3050 Laptop GPU" },
    { 0x25ab, "NVIDIA GeForce RTX 3050 4GB Laptop GPU" },
    { 0x25ac, "NVIDIA GeForce RTX 3050 6GB Laptop GPU" },
    { 0x25e0, "NVIDIA GeForce RTX 3050 Ti Laptop GPU" },
    { 0x25e2, "NVIDIA GeForce RTX 3050 Laptop GPU" },
    { 0x25e5, "NVIDIA GeForce RTX 3050 Laptop GPU" },
    { 0x25ec, "NVIDIA GeForce RTX 3050 6GB Laptop GPU" },
    { 0x2822, "NVIDIA GeForce RTX 3050 A Laptop GPU" },
    { 0x28a3, "NVIDIA GeForce RTX 3050 A Laptop GPU" },
    { 0x28e3, "NVIDIA GeForce RTX 3050 A Laptop GPU" },
};

const struct chip *chip_find(uint32_t chipset)
{
    for (size_t i = 0; i < sizeof chips / sizeof chips[0]; i++)
        if (chipset >= chips[i].first && chipset <= chips[i].last)
            return &chips[i];
    return NULL;
}

const char *chip_name(uint32_t chipset)
{
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++)
        if (names[i].chipset == chipset)
            return names[i].name;
    return NULL;
}

const char *chip_product(uint16_t device)
{
    for (size_t i = 0; i < sizeof products / sizeof products[0]; i++)
        if (products[i].device == device)
            return products[i].name;
    return NULL;
}
