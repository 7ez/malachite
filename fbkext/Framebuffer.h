// The surface Malachite lit, as an IOFramebuffer, drawn into through BAR1. Malachite publishes a MalachiteSurface
// nub at start, long before GSP-RM is up (WindowServer only opens framebuffers that exist when it starts), and
// sends kSurfaceChanged once the head is lit or goes dark; Framebuffer then reports a connection change like a
// plugged-in monitor. Modes, the hardware cursor and vblank go to Malachite over callPlatformFunction (surface.h).
#pragma once
#include <IOKit/graphics/IOFramebuffer.h>

#include "surface.h"

namespace malachite::fb {

// CoreDisplay tells a framebuffer's GPU vendor by its class name ("Intel", "AMD", "NVDA", ...) and turns off
// IOPresentment, and with it WindowServer's GPU compositing, when it finds a framebuffer of no known vendor
class Framebuffer : public IOFramebuffer {
    OSDeclareDefaultStructors(Framebuffer)

    struct mode {
        uint32_t id, width, height, refresh;  // refresh in Hz, 16.16
        surface_timing timing;                   // clock 0 when Malachite gave none
    };
    static const uint32_t kMaxModes = 32;

    IODeviceMemory *aperture = nullptr;
    bool online = false, hw_cursor = false;
    mode modes[kMaxModes] = {};
    uint32_t nmodes = 0, current = 0;  // current: index into modes
    uint8_t edid[512] = {};
    uint32_t edid_size = 0;
    uint16_t image_w = 0, image_h = 0;  // mm, from the EDID; 0 when it doesn't say
    uint16_t gamma[SURFACE_GAMMA_ENTRIES * 3] = {};
    uint32_t *cursor = nullptr;  // converted cursor image, CURS_MAX x CURS_MAX
    struct handler {
        IOFBInterruptProc proc;
        OSObject *target;
        void *ref;
        bool enabled;
    };
    handler hotplug = {};  // IOFramebuffer's connection-changed handler
    handler vbl = {};      // and its vblank one, which Malachite calls through vbl_tick

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
