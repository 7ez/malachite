#pragma once
#include "plat.h"

struct gsp;
struct rm_client;
struct vmm;

#define GR_MAX_CTXBUF 12

struct ctxbuf {
    uint32_t id;
    uint64_t size, align, vram, va;
    int global, init, ro;
    int shared;  // the session's global buffer: mapped here, not owned
};

// the graphics context buffers the CPU side allocates for a channel
struct gr_ctx {
    struct vmm *vm;
    struct ctxbuf bufs[GR_MAX_CTXBUF];
    int n;
};

// before the first graphics-class object on a graphics channel
int gr_promote(struct gsp *s, const struct rm_client *c, uint32_t channel, struct vmm *vm, int user, struct gr_ctx *ctx);
void gr_ctx_free(struct gsp *s, struct gr_ctx *ctx);  // after GSP-RM freed the channel
// the golden context's global buffers given up with the GSP-RM that had them; the next graphics channel makes them
// again
void gr_forget(struct gsp *s);
// after GSP-RM resumed with its state: its golden context made again on the global buffers kept
int gr_resume(struct gsp *s);

