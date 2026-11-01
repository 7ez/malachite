#pragma once
#include "plat.h"

// runs booter_load/booter_unload on SEC2 with arg in mailbox0/1; mbox0_out is 0 on success
int booter_run(struct gpu *g, const char *name, uint64_t arg, uint32_t *mbox0_out);
