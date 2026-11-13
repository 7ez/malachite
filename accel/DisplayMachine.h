// Mode changes are no-ops: Framebuffer drives scanout on its own.
#pragma once
#include "IOAccelDisplayMachine.h"

namespace malachite::accel {

class DisplayMachine : public IOAccelDisplayMachine {
    OSDeclareDefaultStructors(DisplayMachine)

public:
    // the base runs a synchronous work-loop action from requestProbe() under the accelerator's busy lock and
    // deadlocks the boot
    uint64_t start(IOPCIDevice *) override { return 1; }

    uint64_t displayModeWillChange() override { return 1; }
    uint64_t displayModeDidChange() override { return 1; }
};

}  // namespace malachite::accel
