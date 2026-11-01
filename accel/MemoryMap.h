// GPU virtual-address mapping for a resource. IOAccelMemoryMap is abstract at matchOptionBits, getLength, and the
// three page-table hooks. getGPUVirtualAddress is inherited (reads the VA at +0x98). The page-table hooks are
// no-ops: Task::allocate maps the backing memory in Gpu.
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

    // GPU VAs come from the task (Task::allocate), through the base allocGPUVirtualAddress, which also keeps the
    // map on the task's mapping list

    uint64_t commitIntoGPUPageTable() override { return 1; }
    uint64_t releaseFromGPUPageTable() override { return 1; }
    uint64_t updateGPUPageTable() override { return 1; }
};

}  // namespace malachite::accel
