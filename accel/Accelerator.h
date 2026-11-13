// The IOGraphicsAccelerator2 Metal finds by its MetalPluginName; the plugin then talks to Gpu, not to this.
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
    IOBufferMemoryDescriptor *stampBuf = nullptr;  // completion-stamp page, mapped into Metal clients
    static const uint32_t kStampCount = 4096;
    // stamps whose draws are still on the GPU, oldest first; retired by a timer of their own, as the accelerator's
    // work loop may be waiting for one
    struct pending_stamp {
        int index;
        uint32_t value, fence;
        uintptr_t ctx;  // GPU context the fence belongs to
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
    static const int kQueueStamp = 0;
    static const int kAppStamp = 1;  // so apps' frames don't wait for WindowServer's stamps
    // PerformanceStatistics; on the stamp loop, as it never waits for the GPU
    IOTimerEventSource *statsTimer = nullptr;
    uint64_t lastBusyUs = 0, lastNowUs = 0;
    static void publishStats(OSObject *owner, IOTimerEventSource *timer);
    // we hand it framebuffers ourselves, as the base IOAccelDisplayMachine::start deadlocks (see DisplayMachine)
    class DisplayMachine *display = nullptr;
    IONotifier *fbNotifier = nullptr;
    IONotifier *gpuNotifier = nullptr;
    // the display machine numbers pipes in the order it gets framebuffers, so each waits here for lower screens
    static const unsigned kScreens = 4;
    IOService *waitingFB[kScreens] = {};
    unsigned nextScreen = 0;
    IOLock *fbLock = nullptr;
    IORangeAllocator *ranges[32] = {};  // unused: GPU VAs come from Gpu
    static bool foundGpu(void *target, void *ref, IOService *gpu, IONotifier *notifier);
    static bool foundFramebuffer(void *target, void *ref, IOService *fb, IONotifier *notifier);

public:
    bool start(IOService *provider) override;
    void stop(IOService *provider) override;

    uint64_t getStampMemory(unsigned int *count) override;
    // stamp `index` reaches `value` once the work queued so far is done
    void writeStamp(int index, uint32_t value);
    // gives the event a stamp that lands once the GPU has done what ctx has submitted so far
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
    // behind every MTLBuffer and texture; none may return 0
    uint64_t newSysMemory() override;
    uint64_t newVidMemory() override;
    uint64_t newResource() override;
    uint64_t newMemoryMap() override;

    // the base allocates sizeof(IOAccelCommandQueue), too small for ours
    uint64_t newCommandQueue() override;

    // not exported
    uint64_t tmpTotalVRAM() override;
    uint64_t totalTextureMemory() override;

    // the base returns uninitialised accel+0x348, and Metal rejects the device over the garbage size
    uint64_t getMaxWiredSysMemorySize() override { return 2ull << 30; }
    uint64_t getMaxSingleSysMemorySize() override { return 2ull << 30; }
};

}  // namespace malachite::accel
