#pragma once
#include "plat.h"

struct gsp;
struct vram;
struct gr_global;

struct rm_client {
    uint32_t client, device, subdevice;
};

#define GSP_GR_INFO_MAX 64

#define GSP_ZCULL_SUBREGIONS 16

// NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS
struct gsp_zcull {
    uint32_t width_align, height_align, pixel_squares_by_aliquots, aliquot_total;
    uint32_t region_byte_multiplier, region_header_size, subregion_header_size;
    uint32_t subregion_count, subregion_width_align, subregion_height_align;
};

// what GSP-RM told us at start-up, for the controls the CPU-side RM answers itself
struct gsp_info {
    char name[64], short_name[64];
    uint64_t fb_size;
    uint64_t bar1_pdb;  // GSP-RM's BAR1 page directory, VRAM
    uint32_t gr[GSP_GR_INFO_MAX];  // by NV0080_CTRL_GR_INFO_INDEX_*
    struct gsp_zcull zcull;  // subregion_count 0 when GSP-RM didn't say
};

// boots GSP-RM and returns once it has initialised; NULL on failure, with the GPU cleaned up again
struct gsp *gsp_start(struct gpu *g);
// unloads GSP-RM, releases WPR2 and frees everything gsp_start and gsp_dma allocated
void gsp_stop(struct gsp *s);
// A power cycle: GSP-RM unloaded and WPR2 released, what lived in GSP-RM (the golden context) dropped; s's system
// memory, gsp_dma() buffers, VRAM allocations and channel IDs stay. 0 when FWSEC-SB ran and WPR2 came down.
int gsp_unload(struct gsp *s);
// and booted again on them; 0 once it has initialised, else the GPU is cleaned up again and s is left unloaded
int gsp_boot(struct gsp *s);
// A power cycle that keeps GSP-RM's state (its clients, VA spaces, channels and engine objects) in system memory, as
// a suspend does: channel scheduling stopped, GSP-RM unloaded for a power transition and WPR2 saved and released.
// VRAM of ours isn't saved. -1 with GSP-RM still running if the copy couldn't be set up.
int gsp_suspend(struct gsp *s);
// After gsp_suspend() or gsp_unload(): 0 when GSP-RM came back from the copy with everything it had, 1 when it was
// booted afresh with nothing (no copy, or it didn't resume from it), -1 when it isn't up
int gsp_resume(struct gsp *s);
// FWSEC-SB and Booter-Unload for a GPU left with GSP-RM loaded
int gsp_cleanup(struct gpu *g);

struct gpu *gsp_gpu(struct gsp *s);
struct vram *gsp_vram(struct gsp *s);

// called from message processing when GSP-RM recovers a channel; must not send RPCs itself
typedef void (*gsp_rc_fn)(void *ctx, uint32_t engine, uint32_t chid, uint32_t exception, uint64_t fault_va);
// Channel IDs, one space for every client of ours: the core keeps 1-5 (copy, its tests, golden context, the
// kernel's graphics channel), contexts and the frontend's clients take the rest from first on; -1 when all are taken
#define GSP_CHIDS 2048
int gsp_chid_get(struct gsp *s, uint32_t first);
void gsp_chid_put(struct gsp *s, uint32_t chid);

// up to two listeners; -1 when there's no room
int gsp_on_rc(struct gsp *s, gsp_rc_fn fn, void *ctx);
void gsp_off_rc(struct gsp *s, void *ctx);
// called from message processing for an event GSP-RM posts to one of our event objects; same restriction
typedef void (*gsp_event_fn)(void *ctx, uint32_t client, uint32_t event, const void *data, uint32_t size);
void gsp_on_event(struct gsp *s, gsp_event_fn fn, void *ctx);

// handles whatever GSP-RM posted; only while no RPC reply is in use, as it replaces the last one
int gsp_pending(const struct gsp *s);  // whether GSP-RM has messages waiting
void gsp_poll(struct gsp *s);

// interrupts: the platform calls gsp_irq() from its handler; it acks, handles messages and says what came in
#define GSP_IRQ_MSG 1
#define GSP_IRQ_GR 2  // graphics or copy engine nonstall: a semaphore release with an interrupt
#define GSP_IRQ_VBLANK(head) (0x100 << (head))  // a head's vblank, once the display enabled it
void gsp_irq_enable(struct gsp *s, int on);
int gsp_irq(struct gsp *s);
struct gr_global **gsp_gr(struct gsp *s);  // gr.c's, freed with the session
const struct rm_client *gsp_internal(struct gsp *s);
const struct gsp_info *gsp_get_info(struct gsp *s);
struct dma *gsp_dma(struct gsp *s, size_t size, int contiguous);
void gsp_dma_free(struct gsp *s, struct dma *d);

int rm_client(struct gsp *s, struct rm_client *c);
int rm_alloc(struct gsp *s, uint32_t client, uint32_t parent, uint32_t object, uint32_t cls,
             const void *params, uint32_t size);
int rm_free(struct gsp *s, uint32_t client, uint32_t object);
// the status GSP-RM gave the last rm_control(), ~0 when it didn't answer
uint32_t gsp_ctrl_status(const struct gsp *s);
// the reply stays valid only until the next RPC
void *rm_control(struct gsp *s, uint32_t client, uint32_t object, uint32_t cmd, const void *params, uint32_t size);
