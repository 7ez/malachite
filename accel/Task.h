// A client's (or the kernel's) GPU address space; the real one lives in Gpu.
#pragma once
#include "IOAccelTask.h"

namespace malachite::accel {

class Task : public IOAccelTask {
    OSDeclareDefaultStructors(Task)

public:
    // opened at the first mapping; the kernel task's is Gpu's own, 0
    uintptr_t ctx = 0;
    bool kernel = false;
    uintptr_t context();
    uint64_t allocate(const IOAccelMemoryMap *map) override;
    uint64_t deallocate(const IOAccelMemoryMap *map, unsigned long long va) override;
    void free() override;
};

}  // namespace malachite::accel
