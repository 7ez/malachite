#include "../registry.h"
#include "Task.h"
#include "IOAccelMemory.h"
#include "IOAccelMemoryMap.h"
#include "VidMemory.h"
#include "bridge.h"
#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <libkern/OSAtomic.h>

extern "C" void log_msg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

namespace malachite::accel {

MalachiteDefineClass(Task, IOAccelTask)

// Placeholder VAs for when Gpu isn't up: allocation still succeeds, but GPU work against one moves no bytes
// (nothing is mapped there). They start above the 49 bits Gpu hands out, so one can never name a real mapping,
// which an unmap of it would otherwise drop.
static SInt64 gNextPlaceholderVA = 1ll << 50;

// IOAccelMemoryMap keeps its memory at +0x18
static IOAccelMemory *map_memory(const IOAccelMemoryMap *map)
{
    return *(IOAccelMemory * const *)((const char *)map + 0x18);
}

uintptr_t Task::context()
{
    if (!ctx && !kernel && gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_CTX_OPEN, true, &ctx, nullptr, nullptr, nullptr);
    return ctx;
}

void Task::free()
{
    if (ctx && gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_CTX_CLOSE, true, (void *)ctx, nullptr, nullptr, nullptr);
    IOAccelTask::free();
}

// The base map's allocGPUVirtualAddress asks its task for the VA, then keeps the map on the task's mapping list
// (which preparing the map later walks); we hand out Gpu's: the map's VRAM or pages, mapped in the GPU vmm.
uint64_t Task::allocate(const IOAccelMemoryMap *map)
{
    // with no context of its own a task's maps would land in the kernel's address space
    if (gGpu && !kernel && !context()) {
        log_msg("Malachite: a buffer is refused, its task has no GPU context\n");
        return 0;
    }
    uint64_t va = 0;
    IOReturn r = kIOReturnNotReady;
    IOAccelMemory *mem = map_memory(map);
    if (VidMemory *vid = OSDynamicCast(VidMemory, mem)) {
        vid->_RESERVEDIOAccelSysMemory1();  // allocPhysical
        if (vid->vram && gGpu)
            r = gGpu->callPlatformFunction(NVGPU_FN_MAP_VRAM, true, (void *)(uintptr_t)vid->vram,
                                            (void *)(uintptr_t)vid->getLength(), &va, (void *)context());
    } else if (IOMemoryDescriptor *md = mem ? *(IOMemoryDescriptor * const *)((const char *)mem + 0xd0) : nullptr) {
        if (gGpu)  // IOAccelSysMemory keeps its descriptor at +0xd0
            r = gGpu->callPlatformFunction(NVGPU_FN_MAP, true, md, &va, (void *)context(), nullptr);
    }
    // a mapping Gpu refused (table or address space full) is an error, not a buffer the GPU can't see
    if (!va && r != kIOReturnNotReady && r != kIOReturnSuccess)
        return 0;
    if (!va) {
        uint64_t len = (((IOAccelMemoryMap *)map)->getLength() + 0xfff) & ~0xfffull;
        va = (uint64_t)OSAddAtomic64((SInt64)(len ? len : 0x1000), &gNextPlaceholderVA);
    }
    return va;
}

// Gpu finds the mapping by VA, and does nothing for a placeholder it never mapped
uint64_t Task::deallocate(const IOAccelMemoryMap *map, unsigned long long va)
{
    if (va && gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_UNMAP, true, (void *)(uintptr_t)va, nullptr, (void *)ctx, nullptr);
    return 0;
}

}  // namespace malachite::accel
