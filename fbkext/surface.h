// The MalachiteSurface nub between Gpu and Framebuffer: only these names are shared
#pragma once
#include <IOKit/IOMessage.h>

#define SURFACE_NUB_NAME "MalachiteSurface"
#define kSurfaceChanged iokit_vendor_specific_msg(0x4e46)  // the properties below changed

// physical address and size of the BAR1 slot the surface is scanned out of; fixed for the nub's life
#define SURFACE_APERTURE "MalachiteAperture"
#define SURFACE_APERTURE_SIZE "MalachiteApertureSize"
// which of Malachite's screens (and heads) the nub is, from 0
#define SURFACE_SCREEN "MalachiteScreen"
// whether the head is lit; the rest only means something while it is
#define SURFACE_ONLINE "MalachiteOnline"
#define SURFACE_WIDTH "MalachiteWidth"
#define SURFACE_HEIGHT "MalachiteHeight"
#define SURFACE_PITCH "MalachitePitch"
#define SURFACE_REFRESH "MalachiteRefresh"  // Hz, 16.16
#define SURFACE_EDID "MalachiteEDID"
// an array of dictionaries with these keys and SURFACE_TIMING, IDs from 1; SURFACE_MODE is the lit one
#define SURFACE_MODES "MalachiteModes"
#define SURFACE_MODE_ID "ID"
#define SURFACE_MODE "MalachiteMode"
// the lit mode's struct surface_timing in an OSData
#define SURFACE_TIMING "MalachiteTiming"
struct surface_timing {
    uint32_t clock;  // kHz
    uint16_t hactive, hblank, hsync_off, hsync_len;
    uint16_t vactive, vblank, vsync_off, vsync_len;
    uint8_t hsync_positive, vsync_positive;
};

// called on the screen's nub
// p1: void (*)(void *ctx) for every vblank of the head, or NULL to stop; p2: ctx
#define SURFACE_FN_SET_VBL "MalachiteSetVBL"
// p1: (uintptr_t) mode ID; returns once the head shows it and the properties are updated
#define SURFACE_FN_SET_MODE "MalachiteSetMode"
// p1: const uint32_t * A8R8G8B8 premultiplied, p2/p3: (uintptr_t) width, height (up to 256)
#define SURFACE_FN_CURSOR_IMAGE "MalachiteCursorImage"
// p1/p2: (intptr_t) x, y of the image's top left; cheap at any rate
#define SURFACE_FN_CURSOR_MOVE "MalachiteCursorMove"
// p1: (uintptr_t) 0 system sleep, 1 doze (display sleep), 2 on; below 2 the monitor goes to standby
#define SURFACE_FN_POWER "MalachitePower"
// property: Malachite has the cursor calls, so Framebuffer may offer a hardware cursor
#define SURFACE_HW_CURSOR "MalachiteHardwareCursor"
#define CURS_OFFSCREEN 512  // a hidden cursor sits this far up and left of the screen
#define CURS_MAX 256        // the largest cursor image, square
// p1: const uint16_t * of SURFACE_GAMMA_ENTRIES RGB triples, 0xffff = 1.0; kept across modesets and power cycles
#define SURFACE_FN_GAMMA "MalachiteGamma"
#define SURFACE_GAMMA_ENTRIES 1025
