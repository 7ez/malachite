// The type-8 user client Metal submits through. The base submit path signals completion synchronously, so
// processCommandBuffer decides when it may: by stamp or by waiting for Gpu.
#pragma once
#include "IOAccelCommandQueue.h"

namespace malachite::accel {

class CommandQueue : public IOAccelCommandQueue {
    OSDeclareDefaultStructors(CommandQueue)

public:
    // Not commandQueueStart: overriding it breaks queue creation (QoS setup). These two aren't exported, so they
    // copy the base's behaviour.
    uint64_t canSubmitCommandBuffer() override { return 1; }
    uint64_t pauseSubmitCommandBuffer() override { return 0; }  // base IOSleep(1)s; only called if !canSubmit

    // ours (commands.h); the base takes the generic types and rejects the rest
    uint64_t processSegmentKernelCommand(IOAccelSegmentResourceList *resList, IOAccelKernelCommand const *cmd,
                                         IOAccelKernelCommand const *end) override;

    // The base's returns NULL, and processCommandBuffer then bails. Records carry raw GPU VAs, so a count-0 list will do.
    uint64_t parseSegmentList(IOAccelShared2 *shared, IOAccelSegmentResourceListHeader const *hdr,
                              IOAccelDeviceShmem *shmem) override;
    // Completion follows a stamp written once the draws are done (gStampsFollowGPU); otherwise waits for them here,
    // before the base signals it.
    uint64_t processCommandBuffer(unsigned int a, unsigned int b) override;

private:
    int async = -1;  // completion follows the GPU: NVGPU_FN_ASYNC at the first command buffer, unless OpenGL
    bool app = false;  // not WindowServer's, per NVGPU_FN_ASYNC
};

}  // namespace malachite::accel
