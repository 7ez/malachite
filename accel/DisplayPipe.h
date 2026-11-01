// The accelerator's display pipe for Framebuffer, which is how WindowServer shows its frames on a GPU it composites with:
// it renders each frame into an IOSurface and hands it over in a transaction. We start copying the frame onto the
// surface the head scans out (NVGpuPresent), on a thread of ours so WindowServer carries on with its next frame, and
// report the transaction done once the copy has landed. The base pipe completes transactions and tells clients about
// vblanks only when signalled: the head's vblank interrupt does both, and a 60 Hz timer stands in for it while the
// head isn't lit.
#pragma once
#include <IOKit/IOTimerEventSource.h>
#include <IOKit/IOWorkLoop.h>
#include <kern/thread_call.h>

#include "../gpu_bridge.h"

#include "IOAccelDisplayPipe.h"

namespace malachite::accel {

class DisplayPipe : public IOAccelDisplayPipe {
    OSDeclareDefaultStructors(DisplayPipe)

    IOWorkLoop *loop = nullptr;
    IOTimerEventSource *tick = nullptr;
    uint32_t failures = 0;
    uint32_t logged = 0;  // transactions logged
    uint32_t gammas = 0;  // gamma tables logged
    IOAccelDisplayMachine *machine = nullptr;
    IOGraphicsAccelerator2 *accel = nullptr;
    uint32_t inactive = 0;  // ticks spent inactive
    uint32_t screen = 0;    // Malachite's screen the pipe's framebuffer shows: its index (Accelerator::foundFramebuffer)
    IOAccelDisplayPipeTransaction2 *copying = nullptr;  // the transaction whose frame is being copied
    uint32_t fence = 0;                                 // and its copy's
    // the frame the thread is presenting: its surface, kept until it is done, the call it makes, and its answer
    thread_call_t present_call = nullptr;
    OSObject *presenting = nullptr;
    struct nvgpu_present request = {};
    volatile int present_busy = 0;
    IOReturn present_result = kIOReturnSuccess;

    uint64_t last_head_vblank = 0;  // mach time of the head's last vblank, 0 before the first

    static void present_run(thread_call_param_t self, thread_call_param_t);
    static void head_vblank(void *pipe);

    static void vblank(OSObject *owner, IOTimerEventSource *sender);
    void set_gamma(const IOAccelDisplayPipeGammaTable2 *table);

public:
    uint64_t init(IOGraphicsAccelerator2 *accel, IOAccelDisplayMachine *machine, IOFramebuffer *fb,
                  unsigned int index) override;
    void free() override;
    uint64_t initFramebufferResource(unsigned int index, IOAccelResource2 *res) override;
    uint64_t submitTransaction(IOAccelDisplayPipeTransaction2 *t) override;
    uint64_t isTransactionComplete(IOAccelDisplayPipeTransaction2 *t) override;
};

}  // namespace malachite::accel
