// Private storage in VRAM. kchdr misnames allocPhysical/deallocPhysical as _RESERVEDIOAccelSysMemory1/2.
#pragma once
#include "IOAccelVidMemory.h"
#include <IOKit/IOLib.h>

namespace malachite::accel {

class VidMemory : public IOAccelVidMemory {
    OSDeclareDefaultStructors(VidMemory)

public:
    uint64_t vram = 0;  // 0 before allocPhysical or when VRAM ran out

    void free() override;

    uint64_t getLength() const override { return *(uint64_t const *)((char const *)this + 0x40); }
    uint64_t setTag(unsigned int t) override { *(uint32_t *)((char *)this + 0xc0) = t; return 0; }
    uint64_t getTag() override { return *(uint32_t const *)((char const *)this + 0xc0); }

    // *length: contiguous bytes from offset
    uint64_t getPhysicalSegment(unsigned long long offset, unsigned long long *length) override;
    uint64_t _RESERVEDIOAccelSysMemory1() override;  // allocPhysical
    uint64_t _RESERVEDIOAccelSysMemory2() override;  // deallocPhysical
};

}  // namespace malachite::accel
