#pragma once
#include "plat.h"

struct gsp;
struct rm_client;
struct vmm;

#define CHAN_COPY 0
#define CHAN_GRAPHICS 1

#define MTHD(subch, mthd, n) (0x20000000u | ((uint32_t)(n) << 16) | ((subch) << 13) | ((mthd) >> 2))

struct chan_params {
    uint64_t gpfifo_va;
    uint32_t entries, engine, chid;
    uint32_t vaspace, ctxshare;  // GSP-RM handles; one of them
    int privileged;
    uint64_t userd;  // physical, 0x200 bytes
    int userd_sys;
};

struct chan_res {
    uint64_t inst;  // instance block and RAMFC, VRAM
    struct dma mthd;
};

struct chan {
    struct gsp *s;
    uint32_t client, handle, chid, runlist;
    uint64_t userd, va;  // va holds the GPFIFO ring, then the push buffer
    struct vmm *vm;
    struct chan_res res;
    struct dma *buf;
    uint32_t gp_put, push_off;
};

uint32_t chan_runlist(struct gsp *s, const struct rm_client *c, uint32_t engine);  // ~0 if no such engine
int chan_alloc(struct gsp *s, const struct rm_client *c, uint32_t parent, uint32_t handle, const struct chan_params *cp,
               struct chan_res *res);
void chan_res_free(struct gsp *s, struct chan_res *res);  // after GSP-RM freed the channel

int chan_find_engine(struct gsp *s, const struct rm_client *c, uint32_t want, uint32_t *engine, uint32_t *runlist);
// Admin channels may run MEM_OP's L2 operations. CHAN_ADMIN is an app's, pushed to by the kernel alone, its rings
// read-only to the app's shaders.
#define CHAN_KERNEL 1
#define CHAN_ADMIN 2
int chan_create(struct gsp *s, const struct rm_client *c, struct vmm *vm, uint32_t engine, uint32_t runlist,
                uint32_t chid, uint32_t handle, int privileged, struct chan *ch);
int chan_schedule(struct chan *ch);
void chan_submit(struct chan *ch, const uint32_t *push, uint32_t n);
// a WindowServer frame is a few hundred KB of methods, which a pushbuffer half takes without waiting
#define GPFIFO_ENTRIES 4096
#define GPFIFO_BYTES (GPFIFO_ENTRIES * 8)
// every submit takes at least 256 bytes, so a lap of the pushbuffer fits in the GPFIFO
#define PUSH_SIZE (GPFIFO_ENTRIES * 0x100)
// chan_save records [length, methods...] from GPFIFO entry gp on, for chan_replay on a new channel after a loss
#define CHAN_SAVE_WORDS (PUSH_SIZE / 4 + GPFIFO_ENTRIES)
uint32_t chan_save(const struct chan *ch, uint32_t gp, uint32_t *out);
int chan_replay(struct chan *ch, const uint32_t *saved, uint32_t n);
// chan_submit without the kick
void chan_push(struct chan *ch, const uint32_t *push, uint32_t n);
// chan_push split in two, for methods written in place
uint32_t *chan_reserve(struct chan *ch, uint32_t n);
void chan_commit(struct chan *ch, uint32_t n);
// methods outside the pushbuffer (va 4-aligned); push_off stays
void chan_entry(struct chan *ch, uint64_t va, uint32_t words);
void chan_kick(struct chan *ch);
uint32_t chan_gp_get(struct chan *ch);
// the last n GPFIFO entries before GP_PUT, each with the first words of its push, for a channel that stopped answering
void chan_log_tail(const struct chan *ch, uint32_t n);

// GP_GET can't say when the GPU is done with methods (it moves on fetch and is written back only now and then), so
// callers wait a pushbuffer half at a time. This is the half `bytes` more of submits would end in.
int chan_half(const struct chan *ch, uint32_t bytes);
// after GSP-RM freed the channel; safe to repeat
void chan_free(struct chan *ch);

int wait_word(volatile uint32_t *word, uint32_t want, int ms);

// GPU VAs in the channel's vmm; payload must differ from the semaphore's word, which is never reset
int chan_copy(struct chan *ch, uint64_t src, uint64_t dst, uint64_t size, volatile uint32_t *sem, uint64_t semva,
              uint32_t payload);
// block-linear sides are one-GOB blocks, pitch bytes wide (a multiple of 64) and rows high
struct copy_surface {
    uint64_t addr;
    uint32_t pitch;
    uint32_t rows;
    int blocklinear;
    uint32_t origin;  // block-linear start, x bytes | y rows << 16
};
// remap is SET_REMAP_COMPONENTS to move components rather than bytes, or 0; bytes and origins stay in bytes
int chan_copy_surface(struct chan *ch, const struct copy_surface *src, const struct copy_surface *dst,
                      uint32_t bytes, uint32_t lines, uint32_t remap, volatile uint32_t *sem, uint64_t semva,
                      uint32_t payload);
// done when the semaphore reaches payload
void chan_copy_surface_queue(struct chan *ch, const struct copy_surface *src, const struct copy_surface *dst,
                             uint32_t bytes, uint32_t lines, uint32_t remap, uint64_t semva, uint32_t payload);
void chan_release_queue(struct chan *ch, uint64_t semva, uint32_t payload);
// admin channels only; L2 writeback then sysmem invalidate, so VMM_CACHED memory the CPU writes next is read afresh
void chan_l2_sysmem(struct chan *ch);
// waits for another channel's release
void chan_acquire_queue(struct chan *ch, uint64_t semva, uint32_t payload);
// gives up the engine while waiting (ACQUIRE_SWITCH_TSG), so a wait on work a fault lost stalls only this channel
void chan_acquire_queue_switch(struct chan *ch, uint64_t semva, uint32_t payload);
// rows src_pitch and dst_pitch apart; otherwise as chan_copy
int chan_copy2d(struct chan *ch, uint64_t src, uint32_t src_pitch, uint64_t dst, uint32_t dst_pitch, uint32_t bytes,
                uint32_t lines, volatile uint32_t *sem, uint64_t semva, uint32_t payload);
// low byte of pattern, from REMAP's constant with no source read
int chan_fill(struct chan *ch, uint64_t dst, uint64_t size, uint32_t pattern, volatile uint32_t *sem, uint64_t semva,
              uint32_t payload);
