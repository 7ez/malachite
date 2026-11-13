// IOPresentment lists a GPU's framebuffers through its AGDC; without one CoreDisplay composites without the GPU.
// Holding the framebuffer match category keeps IONDRVFramebuffer off the firmware framebuffer, which CoreDisplay
// can't open and aborts over. Values follow Apple's paravirtual GPU where nothing says otherwise.
#pragma once
#include "AppleGraphicsDeviceControl.h"

namespace malachite::accel {

class GraphicsControl : public AppleGraphicsDeviceControl {
    OSDeclareDefaultStructors(GraphicsControl)

    uint32_t unknown = 0;  // unsupported attributes logged
    uint16_t device_id = 0;  // PCI

public:
    bool start(IOService *provider) override;
    using AppleGraphicsDeviceControl::vendor_doDeviceAttribute;
    uint64_t vendor_doDeviceAttribute(unsigned int attr, unsigned long *in, unsigned long inSize, unsigned long *out,
                                      unsigned long *outSize, IOExternalMethodArguments *args) override;
};

}  // namespace malachite::accel
