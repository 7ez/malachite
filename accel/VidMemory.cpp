#include "../registry.h"
#include "VidMemory.h"
#include "bridge.h"

namespace malachite::accel {

MalachiteDefineClass(VidMemory, IOAccelVidMemory)

// Task allocates at the first mapping, not at page-in as the base expects
void VidMemory::free()
{
    _RESERVEDIOAccelSysMemory2();
    IOAccelVidMemory::free();
}

uint64_t VidMemory::getPhysicalSegment(unsigned long long offset, unsigned long long *length)
{
    if (!vram || offset >= getLength()) {
        if (length)
            *length = 0;
        return 0;
    }
    if (length)
        *length = getLength() - offset;
    return vram + offset;
}

// An IOSurface's is length 0 (no VRAM copy), and every display transaction wires it, so that must succeed.
uint64_t VidMemory::_RESERVEDIOAccelSysMemory1()
{
    if (!getLength())
        return 1;
    if (!vram && gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_VRAM_ALLOC, true, (void *)(uintptr_t)getLength(), &vram, nullptr,
                                    nullptr);
    return vram != 0;
}

uint64_t VidMemory::_RESERVEDIOAccelSysMemory2()
{
    if (vram && gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_VRAM_FREE, true, (void *)(uintptr_t)vram,
                                    (void *)(uintptr_t)getLength(), nullptr, nullptr);
    vram = 0;
    return 0;
}

}  // namespace malachite::accel
