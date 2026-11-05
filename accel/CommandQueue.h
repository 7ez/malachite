// Our command queue, the type-8 user client Metal opens and submits command buffers through. The base
// IOAccelCommandQueue submit path (s_submit_command_buffers -> submit_command_buffer -> processCommandBuffer
// -> ... -> IOAccelFenceMachine::sendBlockFenceNotification) runs synchronously and fires the userspace
// completion itself; stamps and Gpu syncs (processCommandBuffer below) decide when it is allowed to.
#pragma once
#include "IOAccelCommandQueue.h"

namespace malachite::accel {

class CommandQueue : public IOAccelCommandQueue {
    OSDeclareDefaultStructors(CommandQueue)

public:
    // commandQueueStart stays the base's: overriding it breaks queue creation (it does essential QoS setup).
    // The two virtuals below are not exported by the base, so they are defined here with its behaviour.
    uint64_t canSubmitCommandBuffer() override { return 1; }
    uint64_t pauseSubmitCommandBuffer() override { return 0; }  // base IOSleep(1)s; only called if !canSubmit

    // The base handles only the generic command types (NoOp/resource-ref/timestamp) and marks anything else
    // unsupported; we parse our own command stream (commands.h) and defer the rest to the base.
    uint64_t processSegmentKernelCommand(IOAccelSegmentResourceList *resList, IOAccelKernelCommand const *cmd,
                                         IOAccelKernelCommand const *end) override;

    // The base parseSegmentList is a hard NULL stub (concrete drivers must override it); inheriting it makes
    // processCommandBuffer bail before dispatching any command. A count-0 list is enough: we carry raw GPU VAs,
    // it only has to be non-null and prepare() successfully.
    uint64_t parseSegmentList(IOAccelShared2 *shared, IOAccelSegmentResourceListHeader const *hdr,
                              IOAccelDeviceShmem *shmem) override;
    // A command buffer's draws run on the GPU without waiting for each other. Its completion follows its stamp,
    // which Accelerator writes once they are done (gStampsFollowGPU); without that the queue waits for them
    // here, before the base signals completion (every command is a segment of its own, so not per segment).
    uint64_t processCommandBuffer(unsigned int a, unsigned int b) override;

private:
    int async = -1;  // whether completion follows the GPU: NVGPU_FN_ASYNC, unless a record says OpenGL. Asked at the first command buffer
    bool app = false;  // not WindowServer's, as NVGPU_FN_ASYNC saysr
};

}  // namespace malachite::accel
