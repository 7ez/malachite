#include "../registry.h"
#include "Accelerator.h"
#include "CommandQueue.h"
#include "SysMemory.h"
#include "VidMemory.h"
#include "Resource.h"
#include "MemoryMap.h"
#include "DisplayMachine.h"
#include "DisplayPipe.h"
#include "EventMachine.h"
#include "Task.h"

#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IOInterruptEventSource.h>
#include <IOKit/IOLib.h>
#include <mach/kmod.h>
#include <pexpert/pexpert.h>

#include "bridge.h"
#include "../fbkext/surface.h"

namespace malachite::accel {

IOService *gGpu = nullptr;
bool gStampsFollowGPU = false;

static IOAccelEventMachine2 *event_machine(IOGraphicsAccelerator2 *a);

// sampled by readers about once a second
static const unsigned kStatsMs = 250;

MalachiteDefineClass(Accelerator, IOGraphicsAccelerator2)

// MetalPluginName is a bundle name, not a path: libsystem_sandbox's gpu_bundle_find_trusted() searches the
// standard GPU bundle dirs (/Library/GPUBundles, /System/Library/Extensions) for "<name>.bundle".
// MalachiteDevice is the principal class Metal instantiates from it.
#define kMetalPluginPath  "MalachiteMetal"
#define kMetalPluginClass "MalachiteDevice"

bool Accelerator::start(IOService *provider)
{
    uint32_t off = 0;
    if (PE_parse_boot_argn("-nonvidia", &off, sizeof(off)) || oldDriverLoaded())
        return false;

    pci = OSDynamicCast(IOPCIDevice, provider);
    if (!pci) {
        IOLog("accel: provider is not an IOPCIDevice\n");
        return false;
    }

    // Metal.framework reads these off the registered accelerator; set before super::start(), which registers
    // the service on success.
    setProperty("MetalPluginName", kMetalPluginPath);
    setProperty("MetalPluginClassName", kMetalPluginClass);
    // OpenGL's renderer bundle: Apple's, which runs GL on whatever Metal device is ours. Boot-arg nvgl=0 leaves
    // GL apps on Apple's software renderer instead.
    uint32_t gl = 1;
    PE_parse_boot_argn("nvgl", &gl, sizeof(gl));
    if (gl)
        setProperty("IOGLBundleName", "AppleMetalOpenGLRenderer");
    setProperty("IOAccelRevision", 2ull, 32);
    // the display pipe's capabilities (IOAccelDisplayPipe::copyCapabilities reads them from here): WindowServer
    // presents through it only when it says transactions work. Needs the graphics device control too: without
    // it CoreDisplay wants a pipe for every display once Framebuffer is online, takes the main one offline and
    // WindowServer exits, over and over
    OSDictionary *caps = OSDictionary::withCapacity(2);
    if (caps) {
        caps->setObject("DisplayPipeSupported", kOSBooleanTrue);
        caps->setObject("TransactionsSupported", kOSBooleanTrue);
        setProperty("IOAccelDisplayPipeCapabilities", caps);
        caps->release();
    }

    // allocates the accelerator's locks, calls our factory virtuals (createEventMachine -> newEventMachine,
    // newDisplayMachine, createKernelGPUTask, createMemoryMap, ...) and fails if any returns 0. Registering
    // before this panics requestProbe on the null locks.
    if (!IOGraphicsAccelerator2::start(provider)) {
        IOLog("accel: super::start() bailed (accelerator factories not implemented yet)\n");
        return false;
    }

    pci->retain();
    stampLock = IOLockAlloc();
    stampLoop = IOWorkLoop::workLoop();
    stampTimer = stampLoop ? IOTimerEventSource::timerEventSource(this, retireStamps) : nullptr;
    if (!stampLock || !stampTimer || stampLoop->addEventSource(stampTimer) != kIOReturnSuccess) {
        OSSafeReleaseNULL(stampTimer);  // stamps then go straight into the page, as the command queue waits
        IOLog("accel: no stamp timer\n");
    }
    statsTimer = stampLoop ? IOTimerEventSource::timerEventSource(this, publishStats) : nullptr;
    if (statsTimer && stampLoop->addEventSource(statsTimer) == kIOReturnSuccess)
        statsTimer->setTimeoutMS(kStatsMs);
    else
        OSSafeReleaseNULL(statsTimer);
    // base start() does not registerService(); unregistered, Metal's service matching never finds us
    registerService();
    // without this, command-queue submit (acceleratorWaitEnabled) blocks forever
    enableAccelerator();

    // Find the Gpu service so the memory map and command queue can drive the GPU (map buffers into the GPU
    // vmm, run copy-engine blits) over callPlatformFunction. Not waited for: matching of this device is
    // serial and Gpu's own start is queued behind this one. Until it publishes, buffers get placeholder VAs
    // and GPU work no-ops.
    if (!gGpu) {
        OSDictionary *match = IOService::serviceMatching("MalachiteGpu");
        if (match) {
            gpuNotifier = addMatchingNotification(gIOFirstPublishNotification, match, foundGpu, this);
            match->release();
        }
    }

    fbLock = IOLockAlloc();
    OSDictionary *fbMatch = fbLock ? IOService::serviceMatching("NVDAMalachiteFramebuffer") : nullptr;
    if (fbMatch) {
        fbNotifier = addMatchingNotification(gIOFirstPublishNotification, fbMatch, foundFramebuffer, this);
        fbMatch->release();
    }

    IOLog("accel: up on %s, registered + enabled, advertising Metal plugin %s\n", pci->getName(), kMetalPluginClass);
    return true;
}

bool Accelerator::foundGpu(void *, void *, IOService *gpu, IONotifier *)
{
    gpu->retain();
    gGpu = gpu;
    IOLog("accel: Gpu service found\n");
    return true;
}

// Framebuffer is an IOFramebuffer: the display machine builds it a display pipe (createDisplayPipe ->
// newDisplayPipe), which tags it with our accelerator so WindowServer finds the pipe to present through
bool Accelerator::foundFramebuffer(void *target, void *ref, IOService *fb, IONotifier *notifier)
{
    Accelerator *self = (Accelerator *)target;
    OSNumber *n = fb->getProvider() ? OSDynamicCast(OSNumber, fb->getProvider()->getProperty(SURFACE_SCREEN)) : nullptr;
    unsigned screen = n ? n->unsigned32BitValue() : 0;
    if (screen >= kScreens)
        return true;
    IOLockLock(self->fbLock);
    if (!self->waitingFB[screen] && screen >= self->nextScreen) {
        fb->retain();
        self->waitingFB[screen] = fb;
    }
    while (self->nextScreen < kScreens && self->waitingFB[self->nextScreen]) {
        IOService *next = self->waitingFB[self->nextScreen];
        bool ok = self->display && self->display->found_framebuffer((IOFramebuffer *)next);
        IOLog("accel: display pipe %u for %s: %s\n", self->nextScreen, next->getName(), ok ? "made" : "failed");
        next->release();
        self->waitingFB[self->nextScreen++] = nullptr;
    }
    IOLockUnlock(self->fbLock);
    return true;
}

uint64_t Accelerator::newDisplayPipe()
{
    return (uint64_t)OSTypeAlloc(DisplayPipe);
}

void Accelerator::stop(IOService *provider)
{
    if (gpuNotifier) {
        gpuNotifier->remove();
        gpuNotifier = nullptr;
    }
    if (fbNotifier) {
        fbNotifier->remove();
        fbNotifier = nullptr;
    }
    for (IOService *&fb : waitingFB)
        OSSafeReleaseNULL(fb);
    if (fbLock) {
        IOLockFree(fbLock);
        fbLock = nullptr;
    }
    gStampsFollowGPU = false;
    if (statsTimer) {
        statsTimer->cancelTimeout();
        stampLoop->removeEventSource(statsTimer);
        OSSafeReleaseNULL(statsTimer);
    }
    if (stampTimer) {
        stampTimer->cancelTimeout();
        stampLoop->removeEventSource(stampTimer);
        OSSafeReleaseNULL(stampTimer);
    }
    OSSafeReleaseNULL(stampLoop);
    IOGraphicsAccelerator2::stop(provider);  // tears down the accelerator (event machine, lists) first
    if (stampLock) {
        IOLockFree(stampLock);
        stampLock = nullptr;
    }
    if (stampBuf) {
        stampBuf->release();
        stampBuf = nullptr;
    }
    if (pci) {
        pci->release();
        pci = nullptr;
    }
}

uint64_t Accelerator::getStampMemory(unsigned int *count)
{
    // IOAccelDevice::deviceStart retains this and createMappingInTask()s it into the Metal client, so it has to
    // be an IOMemoryDescriptor. The kernel/user-shared buffer holds the stamp page the event machine writes and
    // user space tests events against.
    if (!stampBuf) {
        stampBuf = IOBufferMemoryDescriptor::inTaskWithOptions(
            kernel_task, kIOMemoryKernelUserShared | kIODirectionOutIn,
            kStampCount * sizeof(uint32_t), page_size);
        if (!stampBuf) {
            if (count)
                *count = 0;
            return 0;
        }
        bzero(stampBuf->getBytesNoCopy(), kStampCount * sizeof(uint32_t));
        // the event machine reads each stamp through a table it builds from the page's address, which the base
        // never hands it: without it, an event with a stamp in it faults when tested
        IOAccelEventMachine2 *em = event_machine(this);
        if (em && !*(void **)((char *)em + 0x28))
            em->setStampBaseAddress((volatile unsigned int *)stampBuf->getBytesNoCopy());
        gStampsFollowGPU = stampTimer && em && *(void **)((char *)em + 0x28);
    }
    if (count)
        *count = kStampCount;
    return (uint64_t)stampBuf;
}

// IOGraphicsAccelerator2 keeps its event machine and fence machine here, and the fence machine the interrupt event
// source that sends the completions of command buffers whose events are done
class IOAccelFenceMachine;
static IOAccelEventMachine2 *event_machine(IOGraphicsAccelerator2 *a)
{
    return *(IOAccelEventMachine2 **)((char *)a + 0x380);
}
static IOInterruptEventSource *fence_notifier(IOGraphicsAccelerator2 *a)
{
    char *fm = *(char **)((char *)a + 0x390);
    return fm ? *(IOInterruptEventSource **)(fm + 0x30) : nullptr;
}

void Accelerator::writeStamp(int index, uint32_t value)
{
    if (!stampBuf || index < 0 || (uint32_t)index >= kStampCount)
        return;
    if (!stampLock) {
        ((volatile uint32_t *)stampBuf->getBytesNoCopy())[index] = value;
        return;
    }
    IOLockLock(stampLock);
    queueStamp(index, value, 0);
    IOLockUnlock(stampLock);
}

void Accelerator::stampEvent(IOAccelEvent *event, uintptr_t ctx)
{
    IOAccelEventMachine2 *em = event_machine(this);
    if (!em || !stampLock || !stampBuf)
        return;
    IOLockLock(stampLock);
    em->incrementStamp(kQueueStamp);
    uint32_t value = (uint32_t)em->getStamp(kQueueStamp);
    em->setEventStamp(kQueueStamp, event);
    queueStamp(kQueueStamp, value, ctx);
    IOLockUnlock(stampLock);
}

// in order behind the stamps before it, after the draws ctx has queued so far; into the page now if there's nothing
// to wait for
void Accelerator::queueStamp(int index, uint32_t value, uintptr_t ctx)
{
    volatile uint32_t *page = (volatile uint32_t *)stampBuf->getBytesNoCopy();
    uint32_t fence = 0;
    if (stampTimer && gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_DRAW_FENCE, true, &fence, (void *)ctx, nullptr, nullptr);
    if (pendingHead == pendingTail && !fence) {
        page[index] = value;
        return;
    }
    if (pendingTail - pendingHead == kPendingStamps) {  // full: wait for the GPU here
        IOLockUnlock(stampLock);
        gGpu->callPlatformFunction(NVGPU_FN_SYNC, true, (void *)ctx, nullptr, nullptr, nullptr);
        retireStamps();
        IOLockLock(stampLock);
    }
    bool idle = pendingHead == pendingTail;
    pending[pendingTail++ % kPendingStamps] = { index, value, fence, ctx };
    if (idle)
        stampTimer->setTimeoutUS(250);
}

// the stamps whose draws are done, in order, into the page; their waiters told and the fences looked at again
unsigned Accelerator::retireStamps()
{
    volatile uint32_t *page = (volatile uint32_t *)stampBuf->getBytesNoCopy();
    unsigned n = 0;
    for (;;) {
        IOLockLock(stampLock);
        if (pendingHead == pendingTail) {
            IOLockUnlock(stampLock);
            break;
        }
        pending_stamp s = pending[pendingHead % kPendingStamps];
        if (s.fence && gGpu &&
            gGpu->callPlatformFunction(NVGPU_FN_DRAW_DONE, true, (void *)(uintptr_t)s.fence, (void *)s.ctx,
                                        nullptr, nullptr) == kIOReturnBusy) {
            IOLockUnlock(stampLock);
            break;
        }
        page[s.index] = s.value;
        pendingHead++;
        IOLockUnlock(stampLock);
        if (IOAccelEventMachine2 *em = event_machine(this))
            em->signalStamp(s.index, s.value);
        n++;
    }
    if (n)
        if (IOInterruptEventSource *f = fence_notifier(this))
            f->interruptOccurred(nullptr, nullptr, 0);
    return n;
}

void Accelerator::retireStamps(OSObject *owner, IOTimerEventSource *timer)
{
    Accelerator *self = (Accelerator *)owner;
    self->retireStamps();
    IOLockLock(self->stampLock);
    bool more = self->pendingHead != self->pendingTail;
    IOLockUnlock(self->stampLock);
    if (more)
        timer->setTimeoutUS(250);
}

static void put_number(OSDictionary *d, const char *key, unsigned long long v)
{
    OSNumber *n = OSNumber::withNumber(v, 64);
    if (n) {
        d->setObject(key, n);
        n->release();
    }
}

void Accelerator::publishStats(OSObject *owner, IOTimerEventSource *timer)
{
    Accelerator *self = (Accelerator *)owner;
    timer->setTimeoutMS(kStatsMs);
    nvgpu_stats s;
    if (!gGpu || gGpu->callPlatformFunction(NVGPU_FN_STATS, true, &s, nullptr, nullptr, nullptr) != kIOReturnSuccess)
        return;

    unsigned long long util = 0;
    if (self->lastNowUs && s.now_us > self->lastNowUs)
        util = (s.busy_us - self->lastBusyUs) * 100 / (s.now_us - self->lastNowUs);
    if (util > 100)
        util = 100;
    self->lastBusyUs = s.busy_us;
    self->lastNowUs = s.now_us;

    // a copy of what is in there now: the accelerator base keeps keys of its own in the dictionary
    OSDictionary *cur = OSDynamicCast(OSDictionary, self->getProperty("PerformanceStatistics"));
    OSDictionary *d = cur ? OSDynamicCast(OSDictionary, cur->copyCollection()) : OSDictionary::withCapacity(10);
    if (!d)
        return;
    // an NVIDIA GPU has no tiler, so the three utilisations are the one measure of the graphics engine
    put_number(d, "Device Utilization %", util);
    put_number(d, "Renderer Utilization %", util);
    put_number(d, "Tiler Utilization %", util);
    put_number(d, "In use system memory", s.sys_mapped);
    put_number(d, "Alloc system memory", s.sys_mapped);
    put_number(d, "recoveryCount", s.recoveries);
    put_number(d, "dmaBytes", s.dma_bytes);
    put_number(d, "vramUsedBytes", s.vram_used);
    put_number(d, "vramFreeBytes", s.vram_total - s.vram_used);
    if (s.temp_c)
        put_number(d, "Temperature(C)", s.temp_c);
    self->setProperty("PerformanceStatistics", d);
    d->release();
}

uint64_t Accelerator::newEventMachine()
{
    // init(this, ratio, flags) is called on it by IOGraphicsAccelerator2::createEventMachine()
    return (uint64_t)OSTypeAlloc(EventMachine);
}

uint64_t Accelerator::newDisplayMachine()
{
    // start() calls init(this) on it
    display = OSTypeAlloc(DisplayMachine);
    return (uint64_t)display;
}

// Base IOAccelTask::init(accel, flags, rangeAllocators): tasks get range allocators (flags 1) as a real
// driver's do, though nothing is carved from them: Task::allocate hands out Gpu's VAs
static uint64_t make_task(Accelerator *accel, IORangeAllocator **ranges, bool kernel)
{
    Task *task = OSTypeAlloc(Task);
    if (!task)
        return 0;
    task->kernel = kernel;
    if (!task->init(accel, ranges ? 1 : 0, ranges)) {
        task->release();
        return 0;
    }
    return (uint64_t)task;
}

// Kernel GPU address space
uint64_t Accelerator::createKernelGPUTask()
{
    for (IORangeAllocator *&r : ranges)
        if (!r && (r = IORangeAllocator::withRange(1ull << 40, 4096)))
            r->allocateRange(0, 0x100000);
    return make_task(this, ranges, true);
}

// Per-client GPU address space. IOAccelShared2::init needs this non-null when Metal opens the shared user
// client (type 6) for its command-buffer shmem, or the open fails and lazyInitialize blocks.
uint64_t Accelerator::createUserGPUTask() { return make_task(this, ranges, false); }

// The type-8 command-queue user client Metal submits through
uint64_t Accelerator::newCommandQueue() { return (uint64_t)OSTypeAlloc(CommandQueue); }

uint64_t Accelerator::populateAccelConfig(IOAccelConfig *config)
{
    // The device's get_config external method (selector 0) returns this to Metal; an all-zero config makes
    // Metal reject the device (init returns nil) even though every selector succeeds. Field offsets and values
    // mirror AppleParavirtAccelerator's populateAccelConfig; the config lives in the accelerator at the
    // offsets get_config reads (config+0x8 == accel+0xc90, ...).
    static const char kName[] = "Malachite";
    const char *name = kName;
    uint8_t *c = (uint8_t *)config;
    uint32_t v32;
    uint64_t v64;
    memcpy(c + 0x00, &name, sizeof(name));
    v32 = 0x480000;       memcpy(c + 0x08, &v32, 4);
    v64 = 0x7;            memcpy(c + 0x0c, &v64, 8);
    v64 = 0x2000000000ull; memcpy(c + 0x14, &v64, 8);
    v64 = 0x40000000;     memcpy(c + 0x20, &v64, 8);
    v64 = 0x100000000008ull; memcpy(c + 0x28, &v64, 8);
    v32 = 0x40004000;     memcpy(c + 0x30, &v32, 4);
    c[0x47] = 1;
    v32 = 4;              memcpy(c + 0x5c, &v32, 4);
    v32 = 2;              memcpy(c + 0x64, &v32, 4);
    v64 = 0x10000000aull; memcpy(c + 0x68, &v64, 8);
    v64 = 0;              memcpy(c + 0x88, &v64, 8);
    return 0;
}

uint64_t Accelerator::configureDevice(IOPCIDevice *device)
{
    pci = device;
    return 1;
}

// the VRAM Gpu read from the GPU
uint64_t Accelerator::tmpTotalVRAM()
{
    OSNumber *n = gGpu ? OSDynamicCast(OSNumber, gGpu->getProperty("MalachiteVRAMBytes")) : nullptr;
    return n ? n->unsigned64BitValue() : 0;
}

uint64_t Accelerator::totalTextureMemory() { return tmpTotalVRAM(); }

uint64_t Accelerator::newSysMemory() { return (uint64_t)OSTypeAlloc(SysMemory); }
uint64_t Accelerator::newVidMemory() { return (uint64_t)OSTypeAlloc(VidMemory); }
uint64_t Accelerator::newResource() { return (uint64_t)OSTypeAlloc(Resource); }
uint64_t Accelerator::newMemoryMap() { return (uint64_t)OSTypeAlloc(MemoryMap); }

}  // namespace malachite::accel
