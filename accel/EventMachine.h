// The accelerator's event machine. IOGraphicsAccelerator2 tracks GPU completion through per-channel "stamps"
// (monotonic counters in a shared page): an IOAccelEvent is done once its channel's stamp has reached the value
// recorded for it, and Metal and IOPresentment test that in user space against the shared page.
//
// Apple's IOAccelEventMachineFast2 does all the event bookkeeping; a driver only writes a stamp's new value once
// the work before it is done. Accelerator::writeStamp holds it back until the GPU has finished the draws queued so
// far, then writes it and tells the event and fence machines.
#pragma once
#include "IOAccelEventMachineFast2.h"

namespace malachite::accel {

class EventMachine : public IOAccelEventMachineFast2 {
    OSDeclareDefaultStructors(EventMachine)

    IOGraphicsAccelerator2 *accel = nullptr;

public:
    uint64_t init(IOGraphicsAccelerator2 *accel, unsigned int ratio, int flags) override;
    uint64_t writeStamp(int stamp, vendevtCommandRec *, unsigned int value) override;
    // no barriers between channels: there is one queue, run in order
    uint64_t prepareBarrier(vendevtBarrierRec *) override { return 0; }
    uint64_t completeBarrier(vendevtBarrierRec *) override { return 0; }
    uint64_t writeBarrierElement(vendevtBarrierRec *, int, unsigned int) override { return 0; }
    uint64_t checkGPUProgress() override { return 1; }
    uint64_t checkChannelProgress(int) override { return 1; }
    uint64_t eventTimeout(int) override { return 0; }
    uint64_t collectAllLiveEventIfNeeded() override { return 0; }  // the base's isn't exported
};

}  // namespace malachite::accel
