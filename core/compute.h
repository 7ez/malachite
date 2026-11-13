#pragma once
#include "plat.h"

struct gsp;
struct rm_client;
struct vmm;
struct chan;
struct gr_ctx;

#define COMPUTE_QMD_BYTES 256  // a QMDV03_00 is 256 bytes, 256-aligned
// params at 0x160..0x200, the handles at 0x200..0x280, then the params past the 20th up to 0x400
#define COMPUTE_CBUF_BYTES 0x400

// a graphics channel on vm, context promoted, 0xc7c0 on subchannel 1 and the window/cache preamble; 0 on success
int compute_setup(struct gsp *s, const struct rm_client *c, struct vmm *vm, uint32_t chid, int privileged,
                  struct chan *ch, struct gr_ctx *ctx);
// QMDV03_00 running prog_va over grid CTAs of block threads, const bank 0 at cbuf_va, shared_size bytes and
// `barriers` barriers per CTA; releases payload to sem_va (sysmembar) when done
void compute_build_qmd(uint32_t *qmd, uint64_t prog_va, uint64_t cbuf_va, uint64_t sem_va, uint32_t payload,
                       const uint32_t block[3], const uint32_t grid[3], uint32_t reg_count, uint32_t shared_size,
                       uint32_t barriers);
// TIC/TSC pools of 32-byte descriptors; a kernel's texture handle is header index | sampler index << 20
void compute_set_tex_pools(struct chan *ch, uint64_t header_va, uint32_t headers, uint64_t sampler_va,
                           uint32_t samplers);
// local memory bytes per thread in the QMD; set_local_memory gives the backing buffer, per_tpc bytes a TPC, after a WFI
void compute_qmd_set_local(uint32_t *qmd, uint32_t bytes);
// once the grid is done: payload and the GPU's time in the 16 bytes at va (kern.malachite_prof)
void compute_qmd_set_stamp(uint32_t *qmd, uint64_t va, uint32_t payload);
void compute_set_local_memory(struct chan *ch, uint64_t va, uint64_t per_tpc);
// holds what follows until everything before it is done
void compute_wait_idle(struct chan *ch);
// doesn't wait
void compute_launch(struct chan *ch, uint64_t qmd_va);
// launch, then wait up to 2 s for the semaphore; 0 when done
int compute_run(struct chan *ch, uint64_t qmd_va, volatile uint32_t *sem, uint32_t payload);

