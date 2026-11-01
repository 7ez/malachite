// A GPU task: an IOAccelTask is the accelerator's per-client (or kernel) GPU address space and allocation
// context. IOAccelTask has no pure virtuals and a usable base init(). The address spaces themselves live in Gpu.
#pragma once
#include "IOAccelTask.h"

namespace malachite::accel {

class Task : public IOAccelTask {
    OSDeclareDefaultStructors(Task)

public:
    // the task's GPU context in Gpu (its address space and channels), opened with its first mapping; the kernel
    // task's is Gpu's own, 0
    uintptr_t ctx = 0;
    bool kernel = false;
    uintptr_t context();
    uint64_t allocate(const IOAccelMemoryMap *map) override;
    uint64_t deallocate(const IOAccelMemoryMap *map, unsigned long long va) override;
    void free() override;
};

}  // namespace malachite::accel
