// Page-table hooks are no-ops: Task::allocate maps the memory in Gpu.
#pragma once
#include "IOAccelMemoryMap.h"
#include "IOAccelMemory.h"
#include <IOKit/IOLib.h>

namespace malachite::accel {

class MemoryMap : public IOAccelMemoryMap {
    OSDeclareDefaultStructors(MemoryMap)

    IOAccelMemory *mapped() const { return *(IOAccelMemory * const *)((char const *)this + 0x18); }

public:
    uint64_t matchOptionBits(unsigned int) override { return 1; }
    uint64_t getLength() const override { auto *m = mapped(); return m ? m->getLength() : 0; }


    uint64_t commitIntoGPUPageTable() override { return 1; }
    uint64_t releaseFromGPUPageTable() override { return 1; }
    uint64_t updateGPUPageTable() override { return 1; }
};

}  // namespace malachite::accel
