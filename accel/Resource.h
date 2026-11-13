// The unexported paging and aperture slots are stubs: memory is mapped by Task, not paged.
#pragma once
#include "IOAccelResource2.h"
#include <IOKit/IOLib.h>

namespace malachite::accel {

class Resource : public IOAccelResource2 {
    OSDeclareDefaultStructors(Resource)

public:
    // the display pipe's: never mapped for the GPU, and the base prepare has no shared context to map it in
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
