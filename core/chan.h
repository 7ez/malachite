#pragma once
#include "plat.h"

struct gsp;
struct rm_client;
struct vmm;

#define CHAN_COPY 0
#define CHAN_GRAPHICS 1

#define MTHD(subch, mthd, n) (0x20000000u | ((uint32_t)(n) << 16) | ((subch) << 13) | ((mthd) >> 2))

// what a channel allocation asks for, whoever the caller is
struct chan_params {
    uint64_t gpfifo_va;
    uint32_t entries, engine, chid;
    uint32_t vaspace, ctxshare;  // handles GSP-RM knows; one of them
    int privileged;
    uint64_t userd;  // physical, 0x200 bytes
    int userd_sys;   // in system memory rather than VRAM
};

// what the CPU side holds for a channel GSP-RM runs
struct chan_res {
    uint64_t inst;  // instance block and RAMFC, VRAM
    struct dma mthd;
};

struct chan {
    struct gsp *s;
    uint32_t client, handle, chid, runlist;
    uint64_t userd, va;  // va: GPFIFO ring, then the push buffer
    struct vmm *vm;      // the address space va is in
    struct chan_res res;
    struct dma *buf;
    uint32_t gp_put, push_off;
};

uint32_t chan_runlist(struct gsp *s, const struct rm_client *c, uint32_t engine);  // ~0 if no such engine
int chan_alloc(struct gsp *s, const struct rm_client *c, uint32_t parent, uint32_t handle, const struct chan_params *cp,
               struct chan_res *res);
void chan_res_free(struct gsp *s, struct chan_res *res);  // after GSP-RM freed the channel

int chan_find_engine(struct gsp *s, const struct rm_client *c, uint32_t want, uint32_t *engine, uint32_t *runlist);
// privileged: an admin channel, CHAN_KERNEL for the kernel's own, CHAN_ADMIN for an app's the kernel alone pushes
// to (its GPFIFO and push buffer read-only to the app's shaders), which may run host methods a user one may not
// (MEM_OP's L2 operations); 0 a user one, which can't touch its own GPFIFO and push buffer, nor the registers an
// admin channel may
#define CHAN_KERNEL 1
#define CHAN_ADMIN 2
int chan_create(struct gsp *s, const struct rm_client *c, struct vmm *vm, uint32_t engine, uint32_t runlist,
                uint32_t chid, uint32_t handle, int privileged, struct chan *ch);
int chan_schedule(struct chan *ch);
void chan_submit(struct chan *ch, const uint32_t *push, uint32_t n);
// a frame of WindowServer's draws is a few hundred KB of methods: the pushbuffer's halves wait for the GPU, so
// they're big enough not to be waited for within one
#define GPFIFO_ENTRIES 4096
#define GPFIFO_BYTES (GPFIFO_ENTRIES * 8)
// every submit takes at least 256 bytes of it, so a whole lap of the pushbuffer fits in the GPFIFO
#define PUSH_SIZE (GPFIFO_ENTRIES * 0x100)
// What a channel was given from GPFIFO entry `gp` up to now, as [length, methods...] records into `out` (room for
// CHAN_SAVE_WORDS); for another channel to run once this one is lost (chan_replay, -1 when it hasn't the room)
#define CHAN_SAVE_WORDS (PUSH_SIZE / 4 + GPFIFO_ENTRIES)
uint32_t chan_save(const struct chan *ch, uint32_t gp, uint32_t *out);
int chan_replay(struct chan *ch, const uint32_t *saved, uint32_t n);
// chan_submit in two: queue methods, then have the GPU go on with everything queued
void chan_push(struct chan *ch, const uint32_t *push, uint32_t n);
// chan_push in two, for methods written in place: room for n words, then the n written
uint32_t *chan_reserve(struct chan *ch, uint32_t n);
void chan_commit(struct chan *ch, uint32_t n);
// one GPFIFO entry for `words` of methods elsewhere than the pushbuffer (va 4-aligned); push_off stays
void chan_entry(struct chan *ch, uint64_t va, uint32_t words);
void chan_kick(struct chan *ch);
uint32_t chan_gp_get(struct chan *ch);

// Methods go in without waiting, so before overwriting older ones the caller waits for the GPU to be done with
// them; GP_GET can't tell (it moves once an entry is fetched, and its copy in memory is written back only now and
// then). The pushbuffer is a ring of two halves, waited for a half at a time: this is the half `bytes` more of
// submits would end in, which the caller has to have waited for when it isn't the one it's in.
int chan_half(const struct chan *ch, uint32_t bytes);
// after the channel object is gone from GSP-RM: its USERD page, ring and address range given back; safe to repeat
void chan_free(struct chan *ch);

int wait_word(volatile uint32_t *word, uint32_t want, int ms);

// one linear buffer->buffer copy on a set-up copy channel; src/dst/semva are GPU VAs in the channel's vmm
// payload = a fence value that differs from the semaphore's current word each call (monotonic); no reset
int chan_copy(struct chan *ch, uint64_t src, uint64_t dst, uint64_t size, volatile uint32_t *sem, uint64_t semva,
              uint32_t payload);
// one side of a copy: pitch-linear rows pitch bytes apart, or block-linear (one-GOB blocks) pitch bytes wide
// (64-byte multiple) and rows high
struct copy_surface {
    uint64_t addr;
    uint32_t pitch;
    uint32_t rows;
    int blocklinear;
    uint32_t origin;  // block-linear: where the copy starts, x bytes | y rows << 16
};
// remap: SET_REMAP_COMPONENTS to move components of each element instead of bytes, 0 for none (bytes and the
// origins stay in bytes)
int chan_copy_surface(struct chan *ch, const struct copy_surface *src, const struct copy_surface *dst,
                      uint32_t bytes, uint32_t lines, uint32_t remap, volatile uint32_t *sem, uint64_t semva,
                      uint32_t payload);
// the same, returning once the copy is queued: it's done when the semaphore reaches payload
void chan_copy_surface_queue(struct chan *ch, const struct copy_surface *src, const struct copy_surface *dst,
                             uint32_t bytes, uint32_t lines, uint32_t remap, uint64_t semva, uint32_t payload);
// a copy channel writing payload to the word at semva once the work queued before it is done
void chan_release_queue(struct chan *ch, uint64_t semva, uint32_t payload);
// queued on an admin channel: L2's dirty lines written back, and with invalidate its lines of system memory dropped
// too, so VMM_CACHED memory the CPU wrote is read afresh
void chan_l2_sysmem(struct chan *ch, int invalidate);
// the channel's later work waits until the word at semva has reached payload (a release by another channel)
void chan_acquire_queue(struct chan *ch, uint64_t semva, uint32_t payload);
// likewise, but giving the engine to other channels while it waits (ACQUIRE_SWITCH_TSG): a wait that may never end
// (on work a fault lost) then holds up only this channel
void chan_acquire_queue_switch(struct chan *ch, uint64_t semva, uint32_t payload);
// lines rows of bytes each, from src to dst, rows src_pitch and dst_pitch apart; otherwise as chan_copy
int chan_copy2d(struct chan *ch, uint64_t src, uint32_t src_pitch, uint64_t dst, uint32_t dst_pitch, uint32_t bytes,
                uint32_t lines, volatile uint32_t *sem, uint64_t semva, uint32_t payload);
// memset dst to the low byte of `pattern` via copy-engine REMAP const (no source read); payload as for chan_copy2d
int chan_fill(struct chan *ch, uint64_t dst, uint64_t size, uint32_t pattern, volatile uint32_t *sem, uint64_t semva,
              uint32_t payload);
