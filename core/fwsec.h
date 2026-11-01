#pragma once
#include "plat.h"

#define FWSEC_FRTS 0x15
#define FWSEC_SB 0x19

int fwsec_run(struct gpu *g, uint32_t cmd, uint64_t frts_addr, uint64_t frts_size);
