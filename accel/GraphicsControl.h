// The GPU's graphics device control (AGDC) object. WindowServer finds display pipes through IOPresentment, which
// lists a GPU's framebuffers by asking its AppleGraphicsDeviceControl over a user client (vendor info, then the
// framebuffer list) and pairs them with the IOAccelerator under the same PCI device. Without one, CoreDisplay sees
// no pipe for Framebuffer and composites without the GPU. Like Apple's paravirtual GPU control it holds the PCI device's
// framebuffer match category, which keeps IONDRVFramebuffer off the firmware's framebuffer (CoreDisplay can't open
// that one, and aborts when an AGDC GPU has a framebuffer it can't open). The attributes answered are the ones
// IOPresentment and AppleGPUWrangler ask for, with the paravirtual GPU's values where nothing says otherwise.
#pragma once
#include "AppleGraphicsDeviceControl.h"

namespace malachite::accel {

class GraphicsControl : public AppleGraphicsDeviceControl {
    OSDeclareDefaultStructors(GraphicsControl)

    uint32_t unknown = 0;  // unsupported attributes logged
    uint16_t device_id = 0;  // the GPU's PCI device ID, for the vendor info

public:
    bool start(IOService *provider) override;
    using AppleGraphicsDeviceControl::vendor_doDeviceAttribute;
    uint64_t vendor_doDeviceAttribute(unsigned int attr, unsigned long *in, unsigned long inSize, unsigned long *out,
                                      unsigned long *outSize, IOExternalMethodArguments *args) override;
};

}  // namespace malachite::accel
