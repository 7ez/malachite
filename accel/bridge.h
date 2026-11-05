// The Gpu service Accelerator talks to for GPU memory mapping + copies, found once in Accelerator::start.
// Shared by the memory map (map/unmap buffer pages) and the command queue (run copies). nullptr => GPU not up yet.
#pragma once
#include <IOKit/IOService.h>
#include "../gpu_bridge.h"
#include "commands.h"

namespace malachite::accel {

extern IOService *gGpu;
// Accelerator's stamps wait for the GPU, so a command queue needn't (see Accelerator::writeStamp)
extern bool gStampsFollowGPU;

}  // namespace malachite::accel

namespace malachite::rm {

IOReturn queue_dispatch(IOService *gpu, void *ctx, const struct nvgpu_dispatch *d);
IOReturn queue_draw(IOService *gpu, void *ctx, const struct nvgpu_draw *d, const char **why);
IOReturn queue_push(IOService *gpu, void *ctx, const struct malachite_push *p);
IOReturn queue_copy2d(IOService *gpu, void *ctx, const struct nvgpu_copy2d *p);
IOReturn queue_compress(IOService *gpu, void *ctx, uint64_t va);

}  // namespace malachite::rm
