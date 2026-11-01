// The accelerator's display machine - IOGraphicsAccelerator2's hook for framebuffer/mode-change coordination.
// Base IOAccelDisplayMachine::init() does the real setup; we only supply the two pure virtuals. Our scanout is
// driven by Framebuffer over BAR1, independent of the accelerator, so the mode-change notifications are no-ops here.
#pragma once
#include "IOAccelDisplayMachine.h"

namespace malachite::accel {

class DisplayMachine : public IOAccelDisplayMachine {
    OSDeclareDefaultStructors(DisplayMachine)

public:
    // No-op. Base IOAccelDisplayMachine::start() runs a synchronous hardware-workloop action; it is called from
    // requestProbe() while the accelerator holds its busy lock, which deadlocks the boot on our setup (Framebuffer owns
    // scanout independently, so there is nothing for the display machine to start). Skip it.
    uint64_t start(IOPCIDevice *) override { return 1; }

    uint64_t displayModeWillChange() override { return 1; }
    uint64_t displayModeDidChange() override { return 1; }
};

}  // namespace malachite::accel
