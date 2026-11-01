#pragma once
#include "nvstatus.h"

#include "../bar1.h"
#include "../chan.h"
#include "../gr.h"
#include "../mem.h"
#include "rmfe.h"

#define RMFE_GPU_ID 0x100

enum kind { K_GSP, K_VASPACE, K_USERMODE, K_VRAM, K_SYSMEM, K_VIRTUAL, K_CTXDMA, K_CHAN, K_EVENT };

struct gpu_map {
    uint64_t va, size;
    struct gpu_map *next;
};

#define RMFE_MAX_CHID 2048

struct rmfe_chan {
    struct chan_res res;
    struct gr_ctx gr;
    uint32_t chid, runlist, engine;
    uint32_t error_ctxdma, token_index;
    int promoted;
};

struct obj {
    uint32_t handle, parent, cls;
    enum kind kind;
    struct vmm *vmm;     // K_VASPACE, the space a K_VIRTUAL range lives in, a channel group's or channel's space
    uint64_t addr, size;  // VRAM offset (K_VRAM), GPU VA (K_VIRTUAL), offset into the memory (K_CTXDMA)
    uint32_t target;     // K_CTXDMA: the memory object
    struct rmfe_chan *ch;  // K_CHAN
    uint32_t pte_kind;
    struct dma dma;      // K_SYSMEM
    struct gpu_map *maps;  // K_VIRTUAL: what MAP_MEMORY_DMA put in the range; K_VRAM: its BAR1 windows
    struct obj *next;
};

struct client {
    uint32_t handle, next_handle;
    struct rmfe_file *owner;
    struct obj *objs;
    struct client *next;
};

struct rmfe_file {
    struct rmfe *fe;
    void *task;
    struct rmfe_map map;
    int map_fd;
    struct rmfe_file *next;
};

struct rmfe {
    struct gsp *s;
    struct bar1 bar1;     // CPU mappings of VRAM
    int bar1_up, bar1_checked;
    struct client *clients;
    struct rmfe_file *files;
    uint32_t next_client;
    struct rc_event { uint32_t chid, exception; } rc[8];  // recoveries GSP-RM reported, not yet passed on
    uint32_t rc_n;
};

struct client *client_find(struct rmfe *fe, uint32_t handle);
struct obj *obj_find(struct client *c, uint32_t handle);
struct obj *obj_add(struct client *c, uint32_t handle, uint32_t parent, uint32_t cls, enum kind kind);

// controls the CPU-side RM answers itself; returns 0 to pass the control on to GSP-RM
int rmfe_local_control(struct rmfe *fe, struct client *c, uint32_t object, uint32_t cmd, void *params, uint32_t size,
                       NV_STATUS *st);

NV_STATUS rmfe_alloc_memory(struct rmfe *fe, struct client *c, uint32_t parent, uint32_t handle, uint32_t cls, void *params);
NV_STATUS rmfe_map_bar1(struct rmfe *fe, struct obj *o, uint64_t offset, uint64_t size, uint64_t *bar1);
NV_STATUS rmfe_unmap_bar1(struct rmfe *fe, struct obj *o);
NV_STATUS rmfe_map_dma(struct rmfe *fe, struct client *c, void *params);
NV_STATUS rmfe_unmap_dma(struct rmfe *fe, struct client *c, void *params);
void rmfe_release_memory(struct rmfe *fe, struct obj *o);
NV_STATUS rmfe_alloc_chan(struct rmfe *fe, struct client *c, struct obj *parent, uint32_t handle, uint32_t cls, void *params);
NV_STATUS rmfe_alloc_engine(struct rmfe *fe, struct client *c, struct obj *chan, uint32_t handle, uint32_t cls, void *params,
                            uint32_t size);
void rmfe_release_chan(struct rmfe *fe, struct obj *o);
int rmfe_chan_control(struct rmfe *fe, struct client *c, struct obj *o, uint32_t cmd, void *params, uint32_t size,
                      NV_STATUS *st);
int client_rm(struct client *c, struct rm_client *rc);
void rmfe_rc_event(void *ctx, uint32_t engine, uint32_t chid, uint32_t exception);
void rmfe_drain_rc(struct rmfe *fe);
