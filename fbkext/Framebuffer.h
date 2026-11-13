// Malachite's lit surface, drawn through BAR1. The nub is published long before GSP-RM is up, as WindowServer only
// opens framebuffers that exist when it starts; lighting the head later shows up as a monitor being plugged in.
#pragma once
#include <IOKit/graphics/IOFramebuffer.h>

#include "surface.h"

namespace malachite::fb {

// CoreDisplay reads the vendor from the class name ("NVDA...") and turns off GPU compositing for an unknown one
class Framebuffer : public IOFramebuffer {
    OSDeclareDefaultStructors(Framebuffer)

    struct mode {
        uint32_t id, width, height, refresh;  // refresh in Hz, 16.16
        surface_timing timing;                   // clock 0 if not given
    };
    static const uint32_t kMaxModes = 32;

    IODeviceMemory *aperture = nullptr;
    bool online = false, hw_cursor = false;
    mode modes[kMaxModes] = {};
    uint32_t nmodes = 0, current = 0;  // current: index into modes
    uint8_t edid[512] = {};
    uint32_t edid_size = 0;
    uint16_t image_w = 0, image_h = 0;  // mm, from the EDID; 0 if unknown
    uint16_t gamma[SURFACE_GAMMA_ENTRIES * 3] = {};
    uint32_t *cursor = nullptr;  // converted cursor image, CURS_MAX x CURS_MAX
    struct handler {
        IOFBInterruptProc proc;
        OSObject *target;
        void *ref;
        bool enabled;
    };
    handler hotplug = {};  // connection changed
    handler vbl = {};      // called through vbl_tick

    void read_surface(IOService *nub);
    const mode *find(IODisplayModeID id) const;
    static void vbl_tick(void *ctx);

public:
    bool start(IOService *provider) override;
    void stop(IOService *provider) override;
    void free() override;
    IOReturn message(UInt32 type, IOService *provider, void *arg) override;
    IOReturn registerForInterruptType(IOSelect type, IOFBInterruptProc proc, OSObject *target, void *ref,
                                      void **interruptRef) override;
    IOReturn unregisterInterrupt(void *interruptRef) override;
    IOReturn setInterruptState(void *interruptRef, UInt32 state) override;

    IOReturn enableController() override;
    bool isConsoleDevice() override { return false; }
    IODeviceMemory *getApertureRange(IOPixelAperture aperture) override;
    const char *getPixelFormats() override;
    IOItemCount getDisplayModeCount() override { return nmodes; }
    IOReturn getDisplayModes(IODisplayModeID *ids) override;
    IOReturn getInformationForDisplayMode(IODisplayModeID mode, IODisplayModeInformation *info) override;
    UInt64 getPixelFormatsForDisplayMode(IODisplayModeID, IOIndex) override { return 0; }
    IOReturn getPixelInformation(IODisplayModeID mode, IOIndex depth, IOPixelAperture aperture,
                                 IOPixelInformation *info) override;
    IOReturn getTimingInfoForDisplayMode(IODisplayModeID mode, IOTimingInformation *info) override;
    IOReturn setGammaTable(UInt32 channelCount, UInt32 dataCount, UInt32 dataWidth, void *data) override;
    IOReturn setGammaTable(UInt32 channelCount, UInt32 dataCount, UInt32 dataWidth, void *data, bool) override
    {
        return setGammaTable(channelCount, dataCount, dataWidth, data);  // the head takes a new LUT at vblank anyway
    }
    IOReturn getCurrentDisplayMode(IODisplayModeID *mode, IOIndex *depth) override;
    IOReturn setDisplayMode(IODisplayModeID mode, IOIndex depth) override;
    IOReturn getStartupDisplayMode(IODisplayModeID *mode, IOIndex *depth) override;
    IOReturn getAttribute(IOSelect attribute, uintptr_t *value) override;
    IOReturn setAttribute(IOSelect attribute, uintptr_t value) override;
    IOItemCount getConnectionCount() override { return 1; }
    IOReturn getAttributeForConnection(IOIndex connect, IOSelect attribute, uintptr_t *value) override;
    IOReturn setAttributeForConnection(IOIndex connect, IOSelect attribute, uintptr_t value) override;
    bool hasDDCConnect(IOIndex connect) override { return online && edid_size >= 128; }
    IOReturn getDDCBlock(IOIndex connect, UInt32 block, IOSelect type, IOOptionBits options, UInt8 *data,
                         IOByteCount *length) override;
    IOReturn setCursorImage(void *cursorImage) override;
    IOReturn setCursorState(SInt32 x, SInt32 y, bool visible) override;
};

}  // namespace malachite::fb
