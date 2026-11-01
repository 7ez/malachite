#include "../registry.h"
#include <mach/kmod.h>

#include "Framebuffer.h"
#include "../core/cursor.h"

namespace malachite::fb {

#define super IOFramebuffer
MalachiteDefineClassNamed(Framebuffer, IOFramebuffer, "NVDAMalachiteFramebuffer")

static uint64_t num(OSObject *o)
{
    OSNumber *n = OSDynamicCast(OSNumber, o);
    return n ? n->unsigned64BitValue() : 0;
}

static uint64_t prop(IOService *s, const char *key)
{
    return num(s->getProperty(key));
}

// Malachite's surfaces are 32 bpp with rows padded to 256 bytes (show and set_mode)
static uint32_t pitch_of(uint32_t width)
{
    return (width * 4 + 255) & ~255u;
}

// the modes, the lit one and the EDID; while dark, the one mode Malachite expects to light (WindowServer has crashed
// on the geometry changing under it at the hotplug), or a placeholder
void Framebuffer::read_surface(IOService *nub)
{
    online = nub->getProperty(SURFACE_ONLINE) == kOSBooleanTrue;
    hw_cursor = nub->getProperty(SURFACE_HW_CURSOR) == kOSBooleanTrue;
    nmodes = current = 0;
    OSArray *list = online ? OSDynamicCast(OSArray, nub->getProperty(SURFACE_MODES)) : nullptr;
    uint64_t lit = prop(nub, SURFACE_MODE);
    for (unsigned i = 0; list && i < list->getCount() && nmodes < kMaxModes; i++) {
        OSDictionary *e = OSDynamicCast(OSDictionary, list->getObject(i));
        if (!e)
            continue;
        mode m = { (uint32_t)num(e->getObject(SURFACE_MODE_ID)), (uint32_t)num(e->getObject(SURFACE_WIDTH)),
                   (uint32_t)num(e->getObject(SURFACE_HEIGHT)), (uint32_t)num(e->getObject(SURFACE_REFRESH)), {} };
        if (!m.id || !m.width || !m.height)
            continue;
        OSData *t = OSDynamicCast(OSData, e->getObject(SURFACE_TIMING));
        if (t && t->getLength() == sizeof(m.timing))
            memcpy(&m.timing, t->getBytesNoCopy(), sizeof(m.timing));
        if (m.id == lit)
            current = nmodes;
        modes[nmodes++] = m;
    }
    if (!nmodes) {
        mode m = { 1, (uint32_t)prop(nub, SURFACE_WIDTH), (uint32_t)prop(nub, SURFACE_HEIGHT),
                   (uint32_t)prop(nub, SURFACE_REFRESH), {} };
        if (!m.width || !m.height)
            m = { 1, 1024, 768, 60 << 16, {} };
        modes[nmodes++] = m;
    }
    edid_size = 0;
    if (OSData *e = OSDynamicCast(OSData, nub->getProperty(SURFACE_EDID))) {
        edid_size = e->getLength() < sizeof(edid) ? e->getLength() : sizeof(edid);
        memcpy(edid, e->getBytesNoCopy(), edid_size);
    }
    // macOS works the DPI out from the image size: the preferred detailed timing's in mm,
    // else the base block's in cm (one byte of which alone is an aspect ratio)
    image_w = image_h = 0;
    if (edid_size >= 128) {
        const uint8_t *d = edid + 54;
        if (d[0] || d[1]) {
            image_w = d[12] | (d[14] & 0xf0) << 4;
            image_h = d[13] | (d[14] & 0x0f) << 8;
        }
        if (!image_w || !image_h) {
            image_w = edid[21] * 10;
            image_h = edid[22] * 10;
        }
        if (!image_w || !image_h)
            image_w = image_h = 0;
    }
}

const Framebuffer::mode *Framebuffer::find(IODisplayModeID id) const
{
    for (uint32_t i = 0; i < nmodes; i++)
        if (modes[i].id == (uint32_t)id)
            return &modes[i];
    return nullptr;
}

bool Framebuffer::start(IOService *provider)
{
    uint64_t base = prop(provider, SURFACE_APERTURE), size = prop(provider, SURFACE_APERTURE_SIZE);
    read_surface(provider);
    if (!base || !size)
        return false;
    aperture = IODeviceMemory::withRange(base, size);
    cursor = (uint32_t *)IOMalloc(CURS_MAX * CURS_MAX * 4);
    if (!aperture || !cursor)
        return false;
    // without a dependent ID IOFramebuffer has no controller, and terminate() dereferences it
    IOService *pci = provider;
    while (pci && !pci->metaCast("IOPCIDevice"))
        pci = pci->getProvider();
    setProperty(kIOFBDependentIDKey, (pci ? pci : provider)->getRegistryEntryID(), 64);
    setProperty(kIOFBDependentIndexKey, prop(provider, SURFACE_SCREEN), 32);  // the GPU's framebuffers, told apart
    // read at open: IOFramebuffer resamples every table to this before setGammaTable, the output LUT's 1024 steps
    setProperty(kIOFBGammaWidthKey, 16ull, 32);
    setProperty(kIOFBGammaCountKey, SURFACE_GAMMA_ENTRIES - 1ull, 32);
    IOLog("fb: aperture %#llx, %s\n", (unsigned long long)base, online ? "lit" : "waiting for the head");
    if (!super::start(provider))
        return false;
    // the GPU's memory, not the aperture's 16 MB, which WindowServer would take for the whole of it
    if (OSNumber *mb = pci ? OSDynamicCast(OSNumber, pci->getProperty("VRAM,totalMB")) : nullptr)
        setProperty(kIOFBMemorySizeKey, mb->unsigned64BitValue() << 20, 64);
    // the power states IONDRVFramebuffer has (initForPM): IOFramebuffer turns changes between them into
    // setAttribute(kIOPowerAttribute); display sleep is doze
    static IOPMPowerState states[] = {
        { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, IOPMPowerOn, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, IOPMDeviceUsable, IOPMPowerOn, IOPMPowerOn, 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    registerPowerDriver(this, states, 3);
    changePowerStateTo(1);  // not below doze until system sleep
    return true;
}

IOReturn Framebuffer::setAttribute(IOSelect attribute, uintptr_t value)
{
    if (attribute != kIOPowerAttribute)
        return super::setAttribute(attribute, value);
    IOReturn r = getProvider()->callPlatformFunction(SURFACE_FN_POWER, true, (void *)value, nullptr, nullptr, nullptr);
    return r == kIOReturnNotReady ? kIOReturnSuccess : r;  // dark: nothing to power
}

IOReturn Framebuffer::message(UInt32 type, IOService *provider, void *arg)
{
    if (type != kSurfaceChanged)
        return super::message(type, provider, arg);
    read_surface(provider);
    IOLog("fb: %s, %u modes, %ux%u\n", online ? "lit" : "dark", nmodes, modes[current].width,
          modes[current].height);
    if (hotplug.proc && hotplug.enabled)
        hotplug.proc(hotplug.target, hotplug.ref);  // as a hot-plugged monitor would
    return kIOReturnSuccess;
}

void Framebuffer::vbl_tick(void *ctx)
{
    Framebuffer *fb = (Framebuffer *)ctx;
    if (fb->vbl.proc)
        fb->vbl.proc(fb->vbl.target, fb->vbl.ref);
}

void Framebuffer::stop(IOService *provider)
{
    provider->callPlatformFunction(SURFACE_FN_SET_VBL, false, nullptr, nullptr, nullptr, nullptr);
    super::stop(provider);
}

void Framebuffer::free()
{
    OSSafeReleaseNULL(aperture);
    if (cursor)
        IOFree(cursor, CURS_MAX * CURS_MAX * 4);
    cursor = nullptr;
    super::free();
}

IOReturn Framebuffer::registerForInterruptType(IOSelect type, IOFBInterruptProc proc, OSObject *target, void *ref,
                                        void **interruptRef)
{
    handler *h = type == kIOFBConnectInterruptType ? &hotplug : type == kIOFBVBLInterruptType ? &vbl : nullptr;
    if (!h)
        return super::registerForInterruptType(type, proc, target, ref, interruptRef);
    *h = { proc, target, ref, true };
    *interruptRef = h;
    if (h == &vbl)
        getProvider()->callPlatformFunction(SURFACE_FN_SET_VBL, false, (void *)&vbl_tick, this, nullptr, nullptr);
    return kIOReturnSuccess;
}

IOReturn Framebuffer::unregisterInterrupt(void *interruptRef)
{
    if (interruptRef != &hotplug && interruptRef != &vbl)
        return super::unregisterInterrupt(interruptRef);
    if (interruptRef == &vbl)
        getProvider()->callPlatformFunction(SURFACE_FN_SET_VBL, false, nullptr, nullptr, nullptr, nullptr);
    *(handler *)interruptRef = {};
    return kIOReturnSuccess;
}

IOReturn Framebuffer::setInterruptState(void *interruptRef, UInt32 state)
{
    if (interruptRef != &hotplug && interruptRef != &vbl)
        return super::setInterruptState(interruptRef, state);
    ((handler *)interruptRef)->enabled = state != 0;
    return kIOReturnSuccess;
}

IOReturn Framebuffer::enableController()
{
    return kIOReturnSuccess;  // Malachite lit the head before publishing the surface
}

IODeviceMemory *Framebuffer::getApertureRange(IOPixelAperture which)
{
    if (which != kIOFBSystemAperture || !aperture)
        return nullptr;
    aperture->retain();  // the caller consumes one reference per call
    return aperture;
}

const char *Framebuffer::getPixelFormats()
{
    return IO32BitDirectPixels "\0\0";
}

IOReturn Framebuffer::getDisplayModes(IODisplayModeID *ids)
{
    for (uint32_t i = 0; i < nmodes; i++)
        ids[i] = modes[i].id;
    return kIOReturnSuccess;
}

IOReturn Framebuffer::getInformationForDisplayMode(IODisplayModeID id, IODisplayModeInformation *info)
{
    const mode *m = find(id);
    if (!m || !info)
        return kIOReturnBadArgument;
    bzero(info, sizeof(*info));
    info->nominalWidth = m->width;
    info->nominalHeight = m->height;
    info->refreshRate = m->refresh;
    info->maxDepthIndex = 0;
    // CoreDisplay gives a mode of at least 1440x900 a HiDPI twin at half its size, backed by the mode itself, only when
    // the mode says ValidForHiRes, unless it takes the display for a Retina one (built in or 4K at over 160 dpi) or
    // the DisplayResolutionEnabled preference is set
    info->flags = kDisplayModeValidFlag | kDisplayModeSafeFlag | (m == &modes[0] ? kDisplayModeDefaultFlag : 0);
    // bigger than the panel is a desktop that size: macOS would only offer it as a twin at half the size
    if (m->width <= modes[0].width)
        info->flags |= kDisplayModeValidForHiResFlag;
    // the panel's own resolution: the size of the preferred mode (modes[0], which Malachite lists first)
    if (m->width == modes[0].width && m->height == modes[0].height)
        info->flags |= kDisplayModeNativeFlag;
    info->imageWidth = image_w;
    info->imageHeight = image_h;
    return kIOReturnSuccess;
}

IOReturn Framebuffer::getTimingInfoForDisplayMode(IODisplayModeID id, IOTimingInformation *info)
{
    const mode *m = find(id);
    if (!m || !info)
        return kIOReturnBadArgument;
    const surface_timing &t = m->timing;
    if (!t.clock)
        return super::getTimingInfoForDisplayMode(id, info);
    bzero(info, sizeof(*info));
    info->appleTimingID = kIOTimingIDInvalid;
    info->flags = kIODetailedTimingValid;
    IODetailedTimingInformationV2 &d = info->detailedInfo.v2;
    d.pixelClock = d.minPixelClock = d.maxPixelClock = (UInt64)t.clock * 1000;
    d.horizontalActive = t.hactive;
    d.horizontalBlanking = t.hblank;
    d.horizontalSyncOffset = t.hsync_off;
    d.horizontalSyncPulseWidth = t.hsync_len;
    d.verticalActive = t.vactive;
    d.verticalBlanking = t.vblank;
    d.verticalSyncOffset = t.vsync_off;
    d.verticalSyncPulseWidth = t.vsync_len;
    d.horizontalSyncConfig = t.hsync_positive ? kIOSyncPositivePolarity : 0;
    d.verticalSyncConfig = t.vsync_positive ? kIOSyncPositivePolarity : 0;
    d.signalConfig = kIODigitalSignal;
    return kIOReturnSuccess;
}

IOReturn Framebuffer::getPixelInformation(IODisplayModeID id, IOIndex depth, IOPixelAperture which, IOPixelInformation *info)
{
    const mode *m = find(id);
    if (!m || depth != 0 || which != kIOFBSystemAperture || !info)
        return kIOReturnUnsupportedMode;
    bzero(info, sizeof(*info));
    info->bytesPerRow = pitch_of(m->width);
    info->activeWidth = m->width;
    info->activeHeight = m->height;
    info->pixelType = kIORGBDirectPixels;
    info->bitsPerPixel = 32;
    info->componentCount = 3;
    info->bitsPerComponent = 8;
    info->componentMasks[0] = 0x00ff0000;  // the window scans A8R8G8B8
    info->componentMasks[1] = 0x0000ff00;
    info->componentMasks[2] = 0x000000ff;
    strlcpy(info->pixelFormat, IO32BitDirectPixels, sizeof(info->pixelFormat));
    return kIOReturnSuccess;
}

IOReturn Framebuffer::getCurrentDisplayMode(IODisplayModeID *id, IOIndex *depth)
{
    if (id)
        *id = modes[current].id;
    if (depth)
        *depth = 0;
    return kIOReturnSuccess;
}

IOReturn Framebuffer::setDisplayMode(IODisplayModeID id, IOIndex depth)
{
    if (!find(id) || depth != 0)
        return kIOReturnUnsupportedMode;
    if ((uint32_t)id == modes[current].id)
        return kIOReturnSuccess;
    IOReturn r = getProvider()->callPlatformFunction(SURFACE_FN_SET_MODE, true, (void *)(uintptr_t)id, nullptr, nullptr,
                                                     nullptr);
    read_surface(getProvider());
    return r;
}

IOReturn Framebuffer::getStartupDisplayMode(IODisplayModeID *id, IOIndex *depth)
{
    return getCurrentDisplayMode(id, depth);
}

IOReturn Framebuffer::getAttribute(IOSelect attribute, uintptr_t *value)
{
    // IOFramebuffer::open() asks this last and, with the connection offline, returns its answer: "unsupported"
    // would fail every open of a dark framebuffer
    if (attribute == kIOHardwareCursorAttribute) {
        if (value)
            *value = hw_cursor ? 1 : 0;  // kIOFBHWCursorSupported
        return kIOReturnSuccess;
    }
    // VRAM doesn't survive sleep: IOFramebuffer then keeps a copy of the screen and puts it back at wake
    if (attribute == kIOVRAMSaveAttribute) {
        if (value)
            *value = 1;
        return kIOReturnSuccess;
    }
    return super::getAttribute(attribute, value);
}

IOReturn Framebuffer::getAttributeForConnection(IOIndex connect, IOSelect attribute, uintptr_t *value)
{
    if (connect != 0)
        return kIOReturnBadArgument;
    switch (attribute) {
    case kConnectionEnable:
    case kConnectionCheckEnable:
        if (value)
            *value = online;
        return kIOReturnSuccess;
    case kConnectionFlags:
        if (value)
            *value = 0;
        return kIOReturnSuccess;
    case kConnectionSupportsHLDDCSense:
        return online && edid_size >= 128 ? kIOReturnSuccess : kIOReturnUnsupported;
    default:
        return super::getAttributeForConnection(connect, attribute, value);
    }
}

IOReturn Framebuffer::setAttributeForConnection(IOIndex connect, IOSelect attribute, uintptr_t value)
{
    if (connect != 0)
        return kIOReturnBadArgument;
    if (attribute == kConnectionProbe)
        return kIOReturnSuccess;
    return super::setAttributeForConnection(connect, attribute, value);
}

IOReturn Framebuffer::getDDCBlock(IOIndex connect, UInt32 block, IOSelect type, IOOptionBits, UInt8 *data,
                           IOByteCount *length)
{
    // blocks count from 1
    if (connect != 0 || !online || type != kIODDCBlockTypeEDID || block < 1 || block * 128 > edid_size || !data ||
        !length)
        return kIOReturnUnsupported;
    IOByteCount n = *length < 128 ? *length : 128;
    memcpy(data, edid + (block - 1) * 128, n);
    *length = n;
    return kIOReturnSuccess;
}

// Night Shift, colour calibration and fades: one table for all three channels or one each, one after another, of
// bytes or 16-bit words; resampled to the head's output LUT
IOReturn Framebuffer::setGammaTable(UInt32 channelCount, UInt32 dataCount, UInt32 dataWidth, void *data)
{
    if ((channelCount != 1 && channelCount != 3) || dataCount < 2 || (dataWidth != 8 && dataWidth != 16) || !data)
        return kIOReturnBadArgument;
    auto entry = [&](UInt32 channel, UInt32 i) -> int {
        UInt32 at = (channelCount == 3 ? channel : 0) * dataCount + i;
        return dataWidth == 8 ? ((const UInt8 *)data)[at] * 257 : ((const UInt16 *)data)[at];
    };
    const UInt32 last = SURFACE_GAMMA_ENTRIES - 1;
    for (UInt32 i = 0; i <= last; i++) {
        UInt32 p = i * (dataCount - 1), k = p / last, frac = p % last;
        for (UInt32 c = 0; c < 3; c++) {
            int a = entry(c, k), b = k + 1 < dataCount ? entry(c, k + 1) : a;
            gamma[i * 3 + c] = (uint16_t)(a + (b - a) * (int)frac / (int)last);
        }
    }
    return getProvider()->callPlatformFunction(SURFACE_FN_GAMMA, true, gamma, nullptr, nullptr, nullptr);
}

// IOFramebuffer also calls this while finding out what cursors the driver takes, so it always converts with the
// one descriptor Malachite's head supports: 32-bit ARGB (straight alpha, which Malachite premultiplies) up to CURS_MAX
// square
IOReturn Framebuffer::setCursorImage(void *cursorImage)
{
    if (!cursor)
        return kIOReturnNoMemory;
    IOHardwareCursorDescriptor desc = {};
    desc.majorVersion = kHardwareCursorDescriptorMajorVersion;
    desc.minorVersion = kHardwareCursorDescriptorMinorVersion;
    desc.width = desc.height = CURS_MAX;
    desc.bitDepth = 32;
    IOHardwareCursorInfo info = {};
    info.majorVersion = kHardwareCursorInfoMajorVersion;
    info.minorVersion = kHardwareCursorInfoMinorVersion;
    info.hardwareCursorData = (UInt8 *)cursor;
    // cleared, so that cursor_unpad can tell rows the converter spaced as the descriptor's width from tight ones
    bzero(cursor, CURS_MAX * CURS_MAX * 4);
    if (!convertCursorImage(cursorImage, &desc, &info))
        return kIOReturnUnsupported;
    if (!hw_cursor)
        return kIOReturnUnsupported;
    if (!info.cursorWidth || !info.cursorHeight || info.cursorWidth > CURS_MAX || info.cursorHeight > CURS_MAX)
        return kIOReturnUnsupported;
    int padded = cursor_unpad(cursor, info.cursorWidth, info.cursorHeight, CURS_MAX);
    return getProvider()->callPlatformFunction(SURFACE_FN_CURSOR_IMAGE, true, cursor, (void *)(uintptr_t)info.cursorWidth,
                                               (void *)(uintptr_t)info.cursorHeight, (void *)(uintptr_t)padded);
}

// x, y: the image's top left (IOFramebuffer has taken the hot spot off)
IOReturn Framebuffer::setCursorState(SInt32 x, SInt32 y, bool visible)
{
    if (!hw_cursor)
        return kIOReturnUnsupported;
    if (!visible)
        x = y = -CURS_OFFSCREEN;
    return getProvider()->callPlatformFunction(SURFACE_FN_CURSOR_MOVE, false, (void *)(intptr_t)x, (void *)(intptr_t)y,
                                               nullptr, nullptr);
}

}  // namespace malachite::fb
