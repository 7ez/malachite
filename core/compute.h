#pragma once
#include "plat.h"

struct gsp;
struct rm_client;
struct vmm;
struct chan;
struct gr_ctx;

#define COMPUTE_QMD_BYTES 256  // a QMDV03_00 is 256 bytes, 256-aligned
#define COMPUTE_CBUF_BYTES 0x280  // the params end at 0x200, the handles at 0x280

// Graphics channel on `vm` with the golden context promoted and the 0xc7c0 object bound on subchannel 1, plus the
// one-time preamble (shared/local windows, cache invalidates). Returns 0 on success.
int compute_setup(struct gsp *s, const struct rm_client *c, struct vmm *vm, uint32_t chid, int privileged,
                  struct chan *ch, struct gr_ctx *ctx);
// Fill a QMDV03_00 for one grid of grid_x CTAs x block_x threads running prog_va with const bank 0 at
// cbuf_va, shared_size bytes of shared memory and `barriers` named barriers per CTA, releasing payload to
// sem_va (sysmembar) on completion.
void compute_build_qmd(uint32_t *qmd, uint64_t prog_va, uint64_t cbuf_va, uint64_t sem_va, uint32_t payload,
                       const uint32_t block[3], const uint32_t grid[3], uint32_t reg_count, uint32_t shared_size,
                       uint32_t barriers);
// Point the channel's texture header (TIC) and sampler (TSC) pools at GPU memory: `headers`/`samplers`
// 32-byte descriptors each. A texture handle in a kernel is header index | sampler index << 20.
void compute_set_tex_pools(struct chan *ch, uint64_t header_va, uint32_t headers, uint64_t sampler_va,
                           uint32_t samplers);
// A kernel's local memory, bytes a thread, in its QMD; and the buffer every thread's comes out of, as much a TPC
// (once the channel is idle).
void compute_qmd_set_local(uint32_t *qmd, uint32_t bytes);
void compute_set_local_memory(struct chan *ch, uint64_t va, uint64_t per_tpc);
// Hold back what comes next on the channel until everything before it is done.
void compute_wait_idle(struct chan *ch);
// Launch the QMD at qmd_va on the channel, without waiting for it.
void compute_launch(struct chan *ch, uint64_t qmd_va);
// compute_launch, then wait up to 2s for its completion semaphore. Returns 0 on done.
int compute_run(struct chan *ch, uint64_t qmd_va, volatile uint32_t *sem, uint32_t payload);

