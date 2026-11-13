// The NVRM escape ABI, served over GSP-RM
#pragma once
#include "../plat.h"

struct gsp;
struct rmfe;
struct rmfe_file;

enum { RMFE_MAP_NONE, RMFE_MAP_BAR0, RMFE_MAP_BAR1, RMFE_MAP_SYS };

// what the last NV_ESC_RM_MAP_MEMORY on a file asked to be mapped into the process
struct rmfe_map {
    int kind;
    uint64_t offset, size;  // into BAR0 or BAR1, or into sys
    struct dma *sys;
};

struct rmfe *rmfe_create(struct gsp *s);
void rmfe_destroy(struct rmfe *fe);

// one per open device handle; task identifies the process for copy_from_user/copy_to_user
struct rmfe_file *rmfe_open(struct rmfe *fe, void *task);
void rmfe_close(struct rmfe_file *f);  // frees every client created through it

// arg is a kernel copy of the escape's parameter struct, updated in place; -1 for a malformed call
int rmfe_escape(struct rmfe_file *f, uint32_t cmd, void *arg, uint32_t size);

// after NV_ESC_RM_MAP_MEMORY: what the process maps next on the device file numbered fd; -1 if nothing
int rmfe_take_map(struct rmfe_file *f, int *fd, struct rmfe_map *out);

// GSP-RM's events (channel recovery) passed on to clients; stands in for interrupts
void rmfe_poll(struct rmfe *fe);

// a scanout surface into one of BAR1's framebuffer slots (bar1.h), for that framebuffer's aperture
int rmfe_bar1_fb(struct rmfe *fe, int slot, uint64_t vram, uint64_t size);
