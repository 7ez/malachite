#include "../registry.h"
#include "EventMachine.h"
#include "Accelerator.h"

namespace malachite::accel {

MalachiteDefineClass(EventMachine, IOAccelEventMachineFast2)

uint64_t EventMachine::init(IOGraphicsAccelerator2 *accel, unsigned int ratio, int flags)
{
    this->accel = accel;
    return IOAccelEventMachineFast2::init(accel, ratio, flags);
}

uint64_t EventMachine::writeStamp(int stamp, vendevtCommandRec *, unsigned int value)
{
    static_cast<Accelerator *>(accel)->writeStamp(stamp, value);
    return 0;
}

}  // namespace malachite::accel
