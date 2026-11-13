#pragma once
#include <IOKit/IOService.h>
#include "../gpu_bridge.h"
#include "commands.h"

namespace malachite::accel {

extern IOService *gGpu;  // null until Gpu is up
// stamps wait for the GPU, so a command queue needn't
extern bool gStampsFollowGPU;

}  // namespace malachite::accel

namespace malachite::rm {

IOReturn queue_dispatch(IOService *gpu, void *ctx, const struct nvgpu_dispatch *d);
// queue: the record's command queue, as a run only changes what that queue's last draw or run set
IOReturn queue_draw(IOService *gpu, void *ctx, const void *queue, const struct nvgpu_draw *d, const char **why);
IOReturn queue_push(IOService *gpu, void *ctx, const void *queue, const struct malachite_push *p);
IOReturn queue_copy2d(IOService *gpu, void *ctx, const struct nvgpu_copy2d *p);
IOReturn queue_compress(IOService *gpu, void *ctx, uint64_t va);

}  // namespace malachite::rm
