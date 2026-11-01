// Resource object behind every MTL allocation. IOAccelResource2 leaves the paging/aperture primitives abstract
// (kchdr shows them =0 because they're unexported). We mirror the paravirt stubs: paging is a no-op, level offsets
// and aperture return empty, the IOSurface cache math is 0 (a plain MTLBuffer has no surface).
// init/initialize/prepare/addToChannel/... stay inherited (exported).
#pragma once
#include "IOAccelResource2.h"
#include <IOKit/IOLib.h>

namespace malachite::accel {

class Resource : public IOAccelResource2 {
    OSDeclareDefaultStructors(Resource)

public:
    // the display pipe's framebuffer resource: its memory stands in for Framebuffer's scanout and is never mapped for the
    // GPU, so the base prepare (which maps it, and has no shared context to map it in) is skipped
    bool framebuffer = false;
    uint64_t prepare() override { return framebuffer ? 1 : IOAccelResource2::prepare(); }
    uint64_t complete() override { return framebuffer ? 0 : IOAccelResource2::complete(); }

    uint64_t pageon(IOAccelEvent *, bool) override { return 0; }
    uint64_t pageon(IOAccelEvent *e, bool b, unsigned long long) override { return pageon(e, b); }
    uint64_t pageoff(IOAccelEvent *, bool, bool *) override { return 0; }
    uint64_t pageoff(IOAccelEvent *e, bool b, bool *bp, unsigned long long) override { return pageoff(e, b, bp); }
    uint64_t rebuildPagingBuffer() override { return 0; }
    uint64_t getMemoryAllocParameter() override { return 0; }
    uint64_t getLevelOffset(unsigned char, unsigned char, int *out) override { if (out) *out = 0; return 0; }
    uint64_t getBackingLevelOffset(unsigned char, unsigned char, int *out) override { if (out) *out = 0; return 0; }
    uint64_t calculateIOSurfaceDeviceCacheVRAMBytes(unsigned long long *p1, unsigned long long *p2) override {
        if (p1) *p1 = 0;
        if (p2) *p2 = 0;
        return 0;
    }
    uint64_t addToAperture() override { return 0; }
    uint64_t removeFromAperture() override { return 0; }
    uint64_t getApertureMemoryDescriptor(unsigned long long *) override { return 0; }
};

}  // namespace malachite::accel
