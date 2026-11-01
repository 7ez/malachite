#include "../registry.h"
#include "GraphicsControl.h"
#include <IOKit/IOLib.h>
#include <IOKit/pci/IOPCIDevice.h>
#include <pexpert/pexpert.h>

#include "../fbkext/surface.h"

namespace malachite::accel {

MalachiteDefineClass(GraphicsControl, AppleGraphicsDeviceControl)

// attributes, numbered as AGDC's user client selectors are
enum {
    kAGDCVendorInfo = 1,
    kAGDCInitFlags = 3,
    kAGDCExtendedCapability = 0x711,
    kAGDCLinkConfig = 0x921,
    kAGDCGPUCapability = 0x980,
};

#pragma pack(push, 1)
struct agdc_timing {
    uint32_t h_scaled_inset, v_scaled_inset, scaler_flags, h_scaled, v_scaled, signal_config, signal_levels;
    uint64_t pixel_clock, min_pixel_clock, max_pixel_clock;  // Hz
    uint32_t h_active, h_blanking, h_sync_offset, h_sync_width;
    uint32_t v_active, v_blanking, v_sync_offset, v_sync_width;
    uint32_t h_border_left, h_border_right, v_border_top, v_border_bottom;
    uint32_t h_sync_config, h_sync_level, v_sync_config, v_sync_level;  // config 1: positive
    uint32_t links, v_blanking_extension;
    uint16_t pixel_encoding, bits_per_component, colorimetry, dynamic_range;
    uint16_t dsc_bits_per_pixel, dsc_slice_height, dsc_slice_width, reserved;
    uint8_t tail[8];
};

// a framebuffer's link: its timing, and the stream address (port, stream) it goes out on
struct agdc_link_config {
    int32_t id, group;
    uint64_t flags;
    struct agdc_timing timing;
    uint32_t port, stream, state;
};

// one kind (type, a bit) of a framebuffer's display pipeline capabilities, up to 8 entries
struct agdc_ext_cap {
    uint32_t fb, type, count;
    uint8_t entry[8][0x32c];
};
#pragma pack(pop)
static_assert(sizeof(agdc_link_config) == 0xb0, "AGDCLinkConfig_t");
static_assert(sizeof(agdc_ext_cap) == 0x196c, "AGDCFBGetExtendedCapability_t");

enum { kAGDCCapGamma = 0x2, kAGDCCapPlaneScaler = 0x10 };

bool GraphicsControl::start(IOService *provider)
{
    uint32_t off = 0;
    if (PE_parse_boot_argn("-nonvidia", &off, sizeof(off)) || oldDriverLoaded())
        return false;
    IOLog("GraphicsControl: starting\n");
    if (IOPCIDevice *pci = OSDynamicCast(IOPCIDevice, provider))
        device_id = pci->configRead16(kIOPCIConfigDeviceID);
    return AppleGraphicsDeviceControl::start(provider);
}

static void put32(unsigned long *out, size_t at, uint32_t v)
{
    memcpy((uint8_t *)out + at, &v, 4);
}

static void put64(unsigned long *out, size_t at, uint64_t v)
{
    memcpy((uint8_t *)out + at, &v, 8);
}

static const uint32_t kMaxFramebuffers = 4;

// Framebuffer's framebuffers, by screen (Malachite's nub for each says which): fb[i] is screen i's, retained, or null;
// how many there are in a row from screen 0
static uint32_t framebuffers(IOService *fb[kMaxFramebuffers])
{
    bzero(fb, kMaxFramebuffers * sizeof(*fb));
    OSDictionary *match = IOService::serviceMatching("NVDAMalachiteFramebuffer");
    OSIterator *it = match ? IOService::getMatchingServices(match) : nullptr;
    OSSafeReleaseNULL(match);
    while (IOService *s = it ? OSDynamicCast(IOService, it->getNextObject()) : nullptr) {
        OSNumber *n = s->getProvider() ? OSDynamicCast(OSNumber, s->getProvider()->getProperty(SURFACE_SCREEN)) : nullptr;
        uint32_t i = n ? n->unsigned32BitValue() : 0;
        if (i < kMaxFramebuffers && !fb[i]) {
            s->retain();
            fb[i] = s;
        }
    }
    OSSafeReleaseNULL(it);
    uint32_t count = 0;
    while (count < kMaxFramebuffers && fb[count])
        count++;
    return count;
}

static void release_framebuffers(IOService *fb[kMaxFramebuffers])
{
    for (uint32_t i = 0; i < kMaxFramebuffers; i++)
        OSSafeReleaseNULL(fb[i]);
}

// a screen's lit timing, from Malachite's nub for it; before its head lights (WindowServer asks at startup, a minute
// earlier), 1920x1080 at 60 Hz, the mode Framebuffer guesses too
static void lit_timing(uint32_t screen, struct surface_timing *t)
{
    *t = { 148500, 1920, 280, 88, 44, 1080, 45, 4, 5, 1, 1 };
    OSDictionary *match = IOService::nameMatching(SURFACE_NUB_NAME);
    OSIterator *it = match ? IOService::getMatchingServices(match) : nullptr;
    OSSafeReleaseNULL(match);
    while (IOService *nub = it ? OSDynamicCast(IOService, it->getNextObject()) : nullptr) {
        OSNumber *n = OSDynamicCast(OSNumber, nub->getProperty(SURFACE_SCREEN));
        OSData *d = OSDynamicCast(OSData, nub->getProperty(SURFACE_TIMING));
        if (n && n->unsigned32BitValue() == screen && d && d->getLength() == sizeof(*t) &&
            nub->getProperty(SURFACE_ONLINE) == kOSBooleanTrue)
            memcpy(t, d->getBytesNoCopy(), sizeof(*t));
    }
    OSSafeReleaseNULL(it);
}

// the request comes in the buffer the answer goes out in, or a copy of it
static bool take_request(const unsigned long *in, unsigned long inSize, unsigned long *out, unsigned long *outSize,
                         size_t size)
{
    if (!in || inSize != size || !out || !outSize || *outSize != size)
        return false;
    if (in != out)
        memmove(out, in, size);
    return true;
}

uint64_t GraphicsControl::vendor_doDeviceAttribute(unsigned int attr, unsigned long *in, unsigned long inSize,
                                                     unsigned long *out, unsigned long *outSize,
                                                     IOExternalMethodArguments *args)
{
    switch (attr) {
    case kAGDCVendorInfo:  // version, name, PCI vendor and device, and a type (2) IOPresentment accepts
        if (!out || !outSize || *outSize != 0x2c)
            return kIOReturnBadArgument;
        bzero(out, 0x2c);
        put32(out, 0, 0x10000);
        strlcpy((char *)out + 4, "NVIDIA", 16);
        put32(out, 0x24, 0x10de | (uint32_t)device_id << 16);  // PCI vendor, device
        put32(out, 0x28, 2);
        return kIOReturnSuccess;
    case kAGDCInitFlags:  // what AppleGPUWrangler may switch on: framebuffers and accelerator (both exist already)
        if (out && outSize && *outSize == 4) {
            put32(out, 0, 3);
            return kIOReturnSuccess;
        }
        return inSize == 4 ? kIOReturnSuccess : kIOReturnBadArgument;
    case kAGDCGPUCapability: {
        // the framebuffers, in display pipe order, which is screen order (Accelerator gives screen i's pipe i), each on
        // a port of its own from 1; the base turns the pointers into registry IDs
        if (!out || !outSize || *outSize != 0xdc)
            return kIOReturnBadArgument;
        bzero(out, 0xdc);
        IOService *fb[kMaxFramebuffers];
        uint32_t n = framebuffers(fb);
        for (int i = 0; i < 4; i++)
            put64(out, 8 * i, (uint64_t)(~(~0u << n)) << 1);
        for (int i = 0; i < 5; i++)
            put32(out, 0x20 + 4 * i, n);
        put64(out, 0x34, (uint64_t)getProvider());
        for (uint32_t i = 0; i < n; i++)
            put64(out, 0x3c + 8 * i, (uint64_t)fb[i]);
        release_framebuffers(fb);  // registered, so they outlive this call
        return kIOReturnSuccess;
    }
    case kAGDCLinkConfig: {  // a framebuffer's link: its lit mode on stream 0 of its port
        if (!take_request(in, inSize, out, outSize, sizeof(agdc_link_config)))
            return kIOReturnBadArgument;
        agdc_link_config *l = (agdc_link_config *)out;
        IOService *fb[kMaxFramebuffers];
        uint32_t n = framebuffers(fb);
        release_framebuffers(fb);
        if (l->id < 0 || (uint32_t)l->id >= n)
            return kIOReturnNotFound;
        uint32_t screen = (uint32_t)l->id;
        struct surface_timing t;
        lit_timing(screen, &t);
        bzero(l, sizeof(*l));
        l->group = -1;
        l->flags = 0xb;
        agdc_timing *g = &l->timing;
        g->pixel_clock = g->min_pixel_clock = g->max_pixel_clock = (uint64_t)t.clock * 1000;
        g->h_active = t.hactive;
        g->h_blanking = t.hblank;
        g->h_sync_offset = t.hsync_off;
        g->h_sync_width = t.hsync_len;
        g->v_active = t.vactive;
        g->v_blanking = t.vblank;
        g->v_sync_offset = t.vsync_off;
        g->v_sync_width = t.vsync_len;
        g->h_sync_config = t.hsync_positive;
        g->v_sync_config = t.vsync_positive;
        g->links = 1;
        g->pixel_encoding = 1;      // RGB
        g->bits_per_component = 2;  // 8
        g->colorimetry = 1;
        g->dynamic_range = 1;
        l->port = screen + 1;
        l->state = 1;
        return kIOReturnSuccess;
    }
    case kAGDCExtendedCapability: {
        // only the plane scaler, which IOPresentment can't do without: one plane, scanned out at the mode's size
        // (the entry's layout is known only this far: two scaler records, each a count, the largest and smallest
        // height and width, and a ratio pair)
        if (!take_request(in, inSize, out, outSize, sizeof(agdc_ext_cap)))
            return kIOReturnBadArgument;
        agdc_ext_cap *c = (agdc_ext_cap *)out;
        IOService *fb[kMaxFramebuffers];
        uint32_t n = framebuffers(fb);
        release_framebuffers(fb);
        if (c->fb >= (n ? n : 1))  // framebuffer 0 before Framebuffer is up, as WindowServer may ask then
            return kIOReturnNotFound;
        if (c->type == kAGDCCapGamma) {
            // the head's output LUT, which CoreDisplay otherwise applies in a shader of its own on every frame: one
            // table of up to 1024 entries per channel on plane 0, values 0 to 1 (as AMD's driver describes its own)
            bzero(c->entry, sizeof(c->entry));
            c->count = 1;
            unsigned long *e = (unsigned long *)c->entry[0];
            put32(e, 0x00, 0x88005);
            put64(e, 0x04, 1);           // planes
            put32(e, 0x10, 0x10);        // table
            put32(e, 0x24, 0x3f800000);  // maximum value, 1.0f
            put64(e, 0x28, 1024);        // entries
            return kIOReturnSuccess;
        }
        if (c->type != kAGDCCapPlaneScaler) {
            if (unknown++ < 64)
                IOLog("GraphicsControl: capability %#x of framebuffer %u unsupported\n", c->type, c->fb);
            return kIOReturnUnsupported;
        }
        struct surface_timing t;
        lit_timing(c->fb, &t);
        bzero(c->entry, sizeof(c->entry));
        c->count = 1;
        unsigned long *e = (unsigned long *)c->entry[0];
        put32(e, 0x000, 0x80004);
        static const size_t records[] = { 0x1c, 0x1a4 };
        for (size_t r : records) {
            put32(e, r, 1);
            put32(e, r + 0x08, t.vactive);
            put32(e, r + 0x0c, t.hactive);
            put32(e, r + 0x10, t.vactive);
            put32(e, r + 0x14, t.hactive);
            put32(e, r + 0x18, 0x70007);
        }
        return kIOReturnSuccess;
    }
    default:
        if (unknown++ < 64)
            IOLog("GraphicsControl: attribute %#x unsupported (in %lu, out %lu bytes)\n", attr, inSize,
                  outSize ? *outSize : 0);
        return kIOReturnUnsupported;
    }
}

}  // namespace malachite::accel
