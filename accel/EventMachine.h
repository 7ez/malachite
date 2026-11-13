// An IOAccelEvent is done once its channel's stamp, a counter in a page user space tests, reaches its value.
// IOAccelEventMachineFast2 does the bookkeeping; Accelerator::writeStamp holds each stamp back until its draws are done.
#pragma once
#include "IOAccelEventMachineFast2.h"

namespace malachite::accel {

class EventMachine : public IOAccelEventMachineFast2 {
    OSDeclareDefaultStructors(EventMachine)

    IOGraphicsAccelerator2 *accel = nullptr;

public:
    uint64_t init(IOGraphicsAccelerator2 *accel, unsigned int ratio, int flags) override;
    uint64_t writeStamp(int stamp, vendevtCommandRec *, unsigned int value) override;
    // one queue, run in order: no barriers needed
    uint64_t prepareBarrier(vendevtBarrierRec *) override { return 0; }
    uint64_t completeBarrier(vendevtBarrierRec *) override { return 0; }
    uint64_t writeBarrierElement(vendevtBarrierRec *, int, unsigned int) override { return 0; }
    uint64_t checkGPUProgress() override { return 1; }
    uint64_t checkChannelProgress(int) override { return 1; }
    uint64_t eventTimeout(int) override { return 0; }
    uint64_t collectAllLiveEventIfNeeded() override { return 0; }  // the base's isn't exported
};

}  // namespace malachite::accel
