// The GPU's power state as GSP-RM reports it
#pragma once
#include "gsp.h"

// a status of 0 is a reading, anything else the control's failure (~0: GSP-RM didn't answer)
struct pwr {
    uint32_t pstate_status, pstate;  // pstate is the number, 0 for P0; ~0 when GSP-RM reports none
    uint32_t clk_status;
    uint32_t gpc_khz, dram_khz, nvd_khz;  // ~0 where the chip has no such clock
};

void pwr_read(struct gsp *s, struct pwr *out);
