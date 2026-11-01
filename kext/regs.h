#pragma once

#define NV_PMC_BOOT_0   0x000000
#define NV_PMC_ENABLE   0x000200

#define NV_CHIPSET(boot0)   (((boot0) >> 20) & 0x1ff)
