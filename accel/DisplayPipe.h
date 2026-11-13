// WindowServer hands each frame over as an IOSurface in a transaction; a thread of ours copies it to the scanout so
// WindowServer can start the next. The base only completes transactions and reports vblanks when signalled: the
// head's vblank does that, or a 60 Hz timer while the head is unlit.
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
    uint32_t screen = 0;    // = pipe index
    IOAccelDisplayPipeTransaction2 *copying = nullptr;
    uint32_t fence = 0;  // copying's copy
    thread_call_t present_call = nullptr;
    OSObject *presenting = nullptr;  // surface held until the copy is done
    struct nvgpu_present request = {};
    volatile int present_busy = 0;
    IOReturn present_result = kIOReturnSuccess;

    uint64_t last_head_vblank = 0;  // mach time, 0 before the first

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
