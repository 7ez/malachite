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
#include <mach-o/loader.h>
#include <mach/kmod.h>
#include <pexpert/pexpert.h>
#include <sys/fcntl.h>
#include <sys/vnode.h>

#include "bridge.h"
#include "../fbkext/surface.h"

namespace malachite::accel {

IOService *gGpu = nullptr;
bool gStampsFollowGPU = false;

static IOAccelEventMachine2 *event_machine(IOGraphicsAccelerator2 *a);

// sampled by readers about once a second
static const unsigned kStatsMs = 250;

MalachiteDefineClass(Accelerator, IOGraphicsAccelerator2)

// a bundle name, not a path: gpu_bundle_find_trusted() looks for "<name>.bundle" in the GPU bundle dirs
#define kMetalPluginPath  "MalachiteMetal"
#define kMetalPluginClass "MalachiteDevice"

extern "C" void log_msg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static bool macho_uuid(const char *path, uint8_t uuid[16])
{
    const int size = 16384;
    bool found = false;
    vfs_context_t ctx = vfs_context_create(nullptr);
    uint8_t *buf = (uint8_t *)IOMalloc(size);
    vnode_t vp = nullptr;
    int resid = 0;
    if (ctx && buf && !vnode_open(path, FREAD, 0, 0, &vp, ctx)) {
        if (!vn_rdwr(UIO_READ, vp, (caddr_t)buf, size, 0, UIO_SYSSPACE, 0, vfs_context_ucred(ctx), &resid,
                     vfs_context_proc(ctx))) {
            uint32_t len = (uint32_t)(size - resid);
            auto mh = (const mach_header_64 *)buf;
            uint32_t off = sizeof(*mh);
            for (uint32_t i = 0; len >= off && mh->magic == MH_MAGIC_64 && i < mh->ncmds; i++) {
                auto lc = (const load_command *)(buf + off);
                if (off + sizeof(uuid_command) > len || lc->cmdsize < sizeof(load_command))
                    break;
                if (lc->cmd == LC_UUID) {
                    memcpy(uuid, ((const uuid_command *)lc)->uuid, 16);
                    found = true;
                    break;
                }
                off += lc->cmdsize;
            }
        }
        vnode_close(vp, FREAD, ctx);
    }
    if (buf)
        IOFree(buf, size);
    if (ctx)
        vfs_context_rele(ctx);
    return found;
}

// Sandboxed GPU processes may map code only from the system volume, so the installer's copy there is used when it is
// the same build, named relative to /Library/GPUBundles, which gpu_bundle_find_trusted() searches first.
static const char *metal_plugin_name()
{
    // looking a path up before the root is mounted panics in namei
    vnode_t root = vfs_rootvnode();
    if (!root)
        return kMetalPluginPath;
    vnode_put(root);
    uint8_t ours[16], system[16];
    if (!macho_uuid("/Library/GPUBundles/MalachiteMetal.bundle/Contents/MacOS/MalachiteMetal", ours) ||
        !macho_uuid("/System/Library/Extensions/MalachiteMetal.bundle/Contents/MacOS/MalachiteMetal", system))
        return kMetalPluginPath;
    if (memcmp(ours, system, sizeof(ours))) {
        log_msg("accel: the Metal driver on the system volume is another build, using /Library/GPUBundles's\n");
        return kMetalPluginPath;
    }
    log_msg("accel: using the Metal driver on the system volume\n");
    return "../../System/Library/Extensions/" kMetalPluginPath;
}

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

    // set before super::start(), which registers the service
    setProperty("MetalPluginName", metal_plugin_name());
    setProperty("MetalPluginClassName", kMetalPluginClass);
    // Apple's GL-on-Metal renderer; nvgl=0 leaves GL on the software renderer
    uint32_t gl = 1;
    PE_parse_boot_argn("nvgl", &gl, sizeof(gl));
    if (gl)
        setProperty("IOGLBundleName", "AppleMetalOpenGLRenderer");
    setProperty("IOAccelRevision", 2ull, 32);
    // read by IOAccelDisplayPipe::copyCapabilities: WindowServer presents through the pipe only if transactions
    // work. Needs GraphicsControl too, or WindowServer exits in a loop once Framebuffer is online.
    OSDictionary *caps = OSDictionary::withCapacity(2);
    if (caps) {
        caps->setObject("DisplayPipeSupported", kOSBooleanTrue);
        caps->setObject("TransactionsSupported", kOSBooleanTrue);
        setProperty("IOAccelDisplayPipeCapabilities", caps);
        caps->release();
    }

    // allocates the locks and calls our factories, failing if any returns 0; registering before this panics
    // requestProbe on the null locks
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
    // the base start() doesn't register, and Metal only finds registered services
    registerService();
    // or submit blocks forever in acceleratorWaitEnabled
    enableAccelerator();

    // Not waited for: Gpu's start is queued behind this one. Until it publishes, buffers get placeholder VAs and
    // GPU work no-ops.
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

// in screen order, as the display machine numbers pipes in the order it gets framebuffers
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
    // IOAccelDevice::deviceStart maps this into the Metal client, so it must be an IOMemoryDescriptor
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
        // the base never gives the event machine the page's address, so testing a stamped event faults
        IOAccelEventMachine2 *em = event_machine(this);
        if (em && !*(void **)((char *)em + 0x28))
            em->setStampBaseAddress((volatile unsigned int *)stampBuf->getBytesNoCopy());
        gStampsFollowGPU = stampTimer && em && *(void **)((char *)em + 0x28);
    }
    if (count)
        *count = kStampCount;
    return (uint64_t)stampBuf;
}

// IOGraphicsAccelerator2's event and fence machines, and the fence machine's completion event source
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

void Accelerator::stampEvent(IOAccelEvent *event, uintptr_t ctx, bool app)
{
    IOAccelEventMachine2 *em = event_machine(this);
    if (!em || !stampLock || !stampBuf)
        return;
    const int index = app ? kAppStamp : kQueueStamp;
    IOLockLock(stampLock);
    em->incrementStamp(index);
    uint32_t value = (uint32_t)em->getStamp(index);
    em->setEventStamp(index, event);
    queueStamp(index, value, ctx);
    IOLockUnlock(stampLock);
}

// held behind earlier stamps and ctx's queued draws; written at once if there's nothing to wait for
void Accelerator::queueStamp(int index, uint32_t value, uintptr_t ctx)
{
    volatile uint32_t *page = (volatile uint32_t *)stampBuf->getBytesNoCopy();
    uint32_t fence = 0;
    if (stampTimer && gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_DRAW_FENCE, true, &fence, (void *)ctx, nullptr, nullptr);
    bool behind = false;
    for (unsigned i = pendingHead; i != pendingTail; i++)
        behind |= pending[i % kPendingStamps].index == index;
    if (!behind && !fence) {
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

// writes stamps whose draws are done, in order per timeline; retired ones keep index -1 until the head passes them
unsigned Accelerator::retireStamps()
{
    volatile uint32_t *page = (volatile uint32_t *)stampBuf->getBytesNoCopy();
    unsigned n = 0;
    for (;;) {
        IOLockLock(stampLock);
        bool held[2] = { false, false };
        pending_stamp *next = nullptr;
        for (unsigned i = pendingHead; i != pendingTail && !next; i++) {
            pending_stamp &s = pending[i % kPendingStamps];
            bool app = s.index == kAppStamp;
            if (s.index < 0 || held[app])
                continue;
            if (s.fence && gGpu &&
                gGpu->callPlatformFunction(NVGPU_FN_DRAW_DONE, true, (void *)(uintptr_t)s.fence, (void *)s.ctx,
                                            nullptr, nullptr) == kIOReturnBusy)
                held[app] = true;
            else
                next = &s;
        }
        if (!next) {
            IOLockUnlock(stampLock);
            break;
        }
        const int index = next->index;
        const uint32_t value = next->value;
        page[index] = value;
        next->index = -1;
        while (pendingHead != pendingTail && pending[pendingHead % kPendingStamps].index < 0)
            pendingHead++;
        IOLockUnlock(stampLock);
        if (IOAccelEventMachine2 *em = event_machine(this))
            em->signalStamp(index, value);
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

    // copied, as the base keeps keys of its own in there
    OSDictionary *cur = OSDynamicCast(OSDictionary, self->getProperty("PerformanceStatistics"));
    OSDictionary *d = cur ? OSDynamicCast(OSDictionary, cur->copyCollection()) : OSDictionary::withCapacity(10);
    if (!d)
        return;
    // no tiler: all three are the graphics engine's
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
    return (uint64_t)OSTypeAlloc(EventMachine);
}

uint64_t Accelerator::newDisplayMachine()
{
    display = OSTypeAlloc(DisplayMachine);
    return (uint64_t)display;
}

// range allocators as a real driver's tasks have, though Task::allocate hands out Gpu's VAs instead
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

uint64_t Accelerator::createKernelGPUTask()
{
    for (IORangeAllocator *&r : ranges)
        if (!r && (r = IORangeAllocator::withRange(1ull << 40, 4096)))
            r->allocateRange(0, 0x100000);
    return make_task(this, ranges, true);
}

// without one, opening the shared user client (type 6) fails and Metal's lazyInitialize blocks
uint64_t Accelerator::createUserGPUTask() { return make_task(this, ranges, false); }

uint64_t Accelerator::newCommandQueue() { return (uint64_t)OSTypeAlloc(CommandQueue); }

uint64_t Accelerator::populateAccelConfig(IOAccelConfig *config)
{
    // get_config (selector 0) hands this to Metal, which rejects an all-zero one. Values are Apple's paravirtual
    // accelerator's; config+0x8 is accel+0xc90.
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
