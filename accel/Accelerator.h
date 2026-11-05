// Our IOAcceleratorFamily2 accelerator. Metal.framework walks the IOService plane for a provider carrying
// MetalPluginName/MetalPluginClassName and loads that GPU bundle in user space. The bundle talks to the GPU
// through the Gpu user client, never through this kext directly, so this class mostly publishes those
// properties and is the IOAccelerator the plugin attaches to. It subclasses IOGraphicsAccelerator2
// (IOGraphicsAccelerator2.h) because Metal expects its provider to be one.
#pragma once
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IORangeAllocator.h>
#include <IOKit/IOTimerEventSource.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/pci/IOPCIDevice.h>

#include "IOGraphicsAccelerator2.h"

namespace malachite::accel {

class Accelerator : public IOGraphicsAccelerator2 {
    OSDeclareDefaultStructors(Accelerator)

    IOPCIDevice *pci = nullptr;
    // completion-stamp page. getStampMemory() hands it out as an IOMemoryDescriptor: IOAccelDevice::deviceStart
    // retains it and maps it into the Metal client task, so it must be an object, not raw memory.
    // kIOMemoryKernelUserShared makes it user-mappable.
    IOBufferMemoryDescriptor *stampBuf = nullptr;
    static const uint32_t kStampCount = 4096;
    // Stamps whose command buffer's draws are still on the GPU, oldest first: each goes into the page once the
    // draws before it are done, which a timer of its own looks for (not on the accelerator's work loop, which may
    // be waiting for one of these stamps).
    struct pending_stamp {
        int index;
        uint32_t value, fence;
        uintptr_t ctx;  // the GPU context whose draws the fence is of
    };
    static const unsigned kPendingStamps = 512;
    pending_stamp pending[kPendingStamps];
    unsigned pendingHead = 0, pendingTail = 0;
    IOLock *stampLock = nullptr;
    IOWorkLoop *stampLoop = nullptr;
    IOTimerEventSource *stampTimer = nullptr;
    static void retireStamps(OSObject *owner, IOTimerEventSource *timer);
    unsigned retireStamps();
    void queueStamp(int index, uint32_t value, uintptr_t ctx);  // under stampLock
    static const int kQueueStamp = 0;            // the command queues'
    static const int kAppStamp = 1;              // apps' queues: their frames don't wait for WindowServer's stamps
    // PerformanceStatistics (what Activity Monitor and iStat read), refreshed from Gpu's NVGPU_FN_STATS; on the
    // stamp loop, as it never waits for the GPU
    IOTimerEventSource *statsTimer = nullptr;
    uint64_t lastBusyUs = 0, lastNowUs = 0;
    static void publishStats(OSObject *owner, IOTimerEventSource *timer);
    // the display machine start() builds, and the notifier that hands it Framebuffer once that is published: we give it
    // the framebuffer ourselves because base IOAccelDisplayMachine::start deadlocks on our setup (see
    // DisplayMachine)
    class DisplayMachine *display = nullptr;
    IONotifier *fbNotifier = nullptr;
    IONotifier *gpuNotifier = nullptr;
    // Framebuffer's framebuffers, handed to the display machine in screen order (which numbers their pipes as it gets
    // them): each waits here for those of the screens before it
    static const unsigned kScreens = 4;
    IOService *waitingFB[kScreens] = {};
    unsigned nextScreen = 0;
    IOLock *fbLock = nullptr;
    IORangeAllocator *ranges[32] = {};  // every task's; unused, as GPU VAs come from Gpu (Task::allocate)
    static bool foundGpu(void *target, void *ref, IOService *gpu, IONotifier *notifier);
    static bool foundFramebuffer(void *target, void *ref, IOService *fb, IONotifier *notifier);

public:
    bool start(IOService *provider) override;
    void stop(IOService *provider) override;

    // IOGraphicsAccelerator2's 17 pure virtuals. getStampMemory and populateAccelConfig are the two the base
    // start() path reads first.
    uint64_t getStampMemory(unsigned int *count) override;
    // the event machine: stamp `index` reaches `value` once the work queued so far is done
    void writeStamp(int index, uint32_t value);
    // a command queue's event (its last command buffer's) given a stamp of its own that the page reaches once the
    // GPU has done what the queue has submitted so far: its command buffer completes then
    void stampEvent(IOAccelEvent *event, uintptr_t ctx, bool app);
    uint64_t newEventMachine() override;
    uint64_t createUserGPUTask() override;
    uint64_t createKernelGPUTask() override;
    uint64_t populateAccelConfig(IOAccelConfig *config) override;
    uint64_t configureDevice(IOPCIDevice *device) override;
    uint64_t teardownDevice(IOPCIDevice *device) override { return 0; }
    uint64_t newDisplayMachine() override;
    uint64_t newDisplayPipe() override;
    uint64_t newGLContext() override { return 0; }
    uint64_t newCLContext() override { return 0; }
    uint64_t newSurface() override { return 0; }
    uint64_t new2DContext() override { return 0; }
    uint64_t newVideoContext() override { return 0; }
    // new_resource (shared user client) drives these to build the object graph behind every MTLBuffer/texture.
    // Each allocates the right-sized subclass (returning 0 is not an option); real GPU memory is wired inside
    // the subclasses (VidMemory VRAM hooks, MemoryMap GPU-VA).
    uint64_t newSysMemory() override;
    uint64_t newVidMemory() override;
    uint64_t newResource() override;
    uint64_t newMemoryMap() override;

    // Base newCommandQueue allocates sizeof(IOAccelCommandQueue); our subclass is larger (own metaclass), so we
    // must override to allocate the right size, or the object is under-allocated and corrupts the heap.
    uint64_t newCommandQueue() override;

    // not in this macOS's exported symbols, so defined here
    uint64_t tmpTotalVRAM() override;
    uint64_t totalTextureMemory() override;

    // get_config (device selector 0) feeds these into the device caps Metal validates. The base returns the
    // uninitialised field accel+0x348, a garbage shared-memory size that makes MTLIOAccelDevice init reject the
    // device.
    uint64_t getMaxWiredSysMemorySize() override { return 2ull << 30; }
    uint64_t getMaxSingleSysMemorySize() override { return 2ull << 30; }
};

}  // namespace malachite::accel
