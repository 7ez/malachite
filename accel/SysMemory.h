// Shared storage. Only the unexported getters, as the kernel does them: length at +0x40, tag on the descriptor at +0xd0.
#pragma once
#include "IOAccelSysMemory.h"
#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IOLib.h>

namespace malachite::accel {

class SysMemory : public IOAccelSysMemory {
    OSDeclareDefaultStructors(SysMemory)

    IOMemoryDescriptor *backing() const { return *(IOMemoryDescriptor * const *)((char const *)this + 0xd0); }

public:
    uint64_t getLength() const override { return *(uint64_t const *)((char const *)this + 0x40); }
    uint64_t setTag(unsigned int t) override { if (auto *m = backing()) m->setTag(t); return 0; }
    uint64_t getTag() override { auto *m = backing(); return m ? m->getTag() : 0; }
};

}  // namespace malachite::accel
