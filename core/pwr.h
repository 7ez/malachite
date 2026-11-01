// What GSP-RM says about the GPU's power state, and the hints it takes from the driver about when to downclock
#pragma once
#include "gsp.h"

// a status of 0 is a reading, anything else the control's failure (~0: GSP-RM didn't answer)
struct pwr {
    uint32_t pstate_status, pstate;  // pstate is the number, 0 for P0; ~0 when GSP-RM reports none
    uint32_t clk_status;
    uint32_t gpc_khz, dram_khz, nvd_khz;  // ~0 where the chip has no such clock
};

void pwr_read(struct gsp *s, struct pwr *out);
// the GPU idle (capped to its lowest P-state) or not, with the time it spent busy and idle since the last call
int pwr_idle_cap(struct gsp *s, int idle, uint64_t idle_us, uint64_t busy_us);
int pwr_boost_clear(struct gsp *s);
