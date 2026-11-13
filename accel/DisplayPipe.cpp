#include "../registry.h"
#include "DisplayPipe.h"
#include "IOAccelDisplayMachine.h"
#include "IOGraphicsAccelerator2.h"
#include "Resource.h"
#include "bridge.h"

#include <kern/clock.h>

// IOSurface has no kernel header
class IOSurface {
public:
    IOMemoryDescriptor *getMemoryDescriptor() const;
    uint32_t getBaseOffset() const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    uint32_t getBytesPerRow() const;
    uint32_t getSurfaceID() const;
};

namespace malachite::accel {

MalachiteDefineClass(DisplayPipe, IOAccelDisplayPipe)

static const uint32_t kFrameUS = 16667;

// WindowServer's frame pacing, for kern.malachite_power
static struct {
    uint64_t submits, gap_us[7], phase_us[5], busy_us, think_us, think_n;
    uint64_t last_submit, last_done;
} pace;

static uint64_t now_us()
{
    uint64_t ns;
    absolutetime_to_nanoseconds(mach_absolute_time(), &ns);
    return ns / 1000;
}

size_t pace_report(char *out, size_t cap)
{
    const char *gaps[] = { "<2.5", "<3.5", "<5", "<7", "<9", "<13", ">=13" };
    size_t n = snprintf(out, cap, "pace: %llu transactions, gap ms:", pace.submits);
    for (int i = 0; i < 7; i++)
        n += snprintf(out + n, cap - n, " %s=%llu", gaps[i], pace.gap_us[i]);
    n += snprintf(out + n, cap - n, "\n  phase after a vblank, ms (<1 <2 <3 <4 more):");
    for (int i = 0; i < 5; i++)
        n += snprintf(out + n, cap - n, " %llu", pace.phase_us[i]);
    n += snprintf(out + n, cap - n, "\n  copy %llu us, think %llu us\n", pace.submits ? pace.busy_us / pace.submits : 0,
                  pace.think_n ? pace.think_us / pace.think_n : 0);
    return n;
}

static void pace_submit(uint64_t last_vblank)
{
    uint64_t ns, vbl_ns;
    absolutetime_to_nanoseconds(mach_absolute_time(), &ns);
    absolutetime_to_nanoseconds(last_vblank, &vbl_ns);
    uint64_t t = ns / 1000;
    pace.submits++;
    if (pace.last_submit) {
        uint64_t g = t - pace.last_submit;
        pace.gap_us[g < 2500 ? 0 : g < 3500 ? 1 : g < 5000 ? 2 : g < 7000 ? 3 : g < 9000 ? 4 : g < 13000 ? 5 : 6]++;
    }
    pace.last_submit = t;
    uint64_t ph = last_vblank && ns > vbl_ns ? (ns - vbl_ns) / 1000 : 0;
    pace.phase_us[ph < 1000 ? 0 : ph < 2000 ? 1 : ph < 3000 ? 2 : ph < 4000 ? 3 : 4]++;
    if (pace.last_done && t > pace.last_done) {
        pace.think_us += t - pace.last_done;
        pace.think_n++;
    }
}

static void pace_done()
{
    pace.last_done = now_us();
    pace.busy_us += pace.last_done - pace.last_submit;
}

uint64_t DisplayPipe::init(IOGraphicsAccelerator2 *accel, IOAccelDisplayMachine *machine, IOFramebuffer *fb,
                             unsigned int index)
{
    if (!IOAccelDisplayPipe::init(accel, machine, fb, index))
        return 0;
    this->machine = machine;
    this->accel = accel;
    screen = index;
    loop = IOWorkLoop::workLoop();
    tick = IOTimerEventSource::timerEventSource(this, vblank);
    present_call = thread_call_allocate(present_run, this);
    if (!loop || !tick || !present_call || loop->addEventSource(tick) != kIOReturnSuccess)
        return 0;
    tick->setTimeoutUS(kFrameUS);
    if (gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_PIPE_VBL, true, (void *)head_vblank, this, (void *)(uintptr_t)screen,
                                    nullptr);
    IOLog("DisplayPipe: pipe %u up, screen %u\n", index, screen);
    return 1;
}

void DisplayPipe::free()
{
    if (gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_PIPE_VBL, true, nullptr, nullptr, (void *)(uintptr_t)screen, nullptr);
    if (present_call) {
        for (int i = 0; __atomic_load_n(&present_busy, __ATOMIC_ACQUIRE) && i < 4000; i++)
            IODelay(50);
        thread_call_cancel(present_call);
        thread_call_free(present_call);
    }
    if (tick) {
        tick->cancelTimeout();
        if (loop)
            loop->removeEventSource(tick);
        tick->release();
    }
    if (loop)
        loop->release();
    IOAccelDisplayPipe::free();
}

// A pipe without framebuffer memory never turns active, and the base gives none. Ours only has to exist: frames reach
// the scanout by the copy in submitTransaction.
uint64_t DisplayPipe::initFramebufferResource(unsigned int index, IOAccelResource2 *res)
{
    if (Resource *r = OSDynamicCast(Resource, res))
        r->framebuffer = true;
    uint64_t size = *(const uint64_t *)((const char *)res + 0xc8);  // the base filled in rows * pitch
    uint64_t mem = accel ? accel->createVidMemory(nullptr, res, size ? size : 1920 * 1080 * 4, nullptr) : 0;
    IOLog("DisplayPipe: framebuffer resource %llu bytes -> memory %#llx\n", size, mem);
    return mem;
}

// IOAccelDisplayPipe::isActive; transactions are aborted while it's clear
static bool pipe_active(const IOAccelDisplayPipe *p)
{
    return *((const volatile uint8_t *)p + 0x298) != 0;
}

// frames flip on the head's vblank, so clients pace on it
void DisplayPipe::head_vblank(void *pipe)
{
    DisplayPipe *self = (DisplayPipe *)pipe;
    self->last_head_vblank = mach_absolute_time();
    self->signalVBLInterrupt(nullptr);
}

void DisplayPipe::vblank(OSObject *owner, IOTimerEventSource *sender)
{
    DisplayPipe *self = (DisplayPipe *)owner;
    // Only a mode change makes a pipe active, and only once the framebuffer has a mode, which Framebuffer gets about a
    // minute after boot without the display machine knowing: repeat it each second until then.
    if (!pipe_active(self) && self->machine && ++self->inactive % 60 == 0) {
        self->machine->display_mode_will_change(self->screen);
        self->machine->display_mode_did_change(self->screen);
        if (pipe_active(self))
            IOLog("DisplayPipe: active after a mode change\n");
    }
    // stands in for an unlit head
    uint64_t ns;
    absolutetime_to_nanoseconds(mach_absolute_time() - self->last_head_vblank, &ns);
    if (!self->last_head_vblank || ns > 100000000)
        self->signalVBLInterrupt(nullptr);
    sender->setTimeoutUS(kFrameUS);
}

// float bits to the LUT's fixed point, 0x10000 = 1.0, clamped to 0..1
static uint32_t lut_value(uint32_t f)
{
    int exp = (int)(f >> 23 & 0xff) - 127;
    if (f >> 31 || exp < -24)
        return 0;
    if (exp >= 0)
        return 0x10000;
    return ((f & 0x7fffff) | 0x800000) >> (7 - exp);
}

// CoreDisplay's table is any size; resampled to the head's
void DisplayPipe::set_gamma(const IOAccelDisplayPipeGammaTable2 *table)
{
    unsigned n = table->getTableCount();
    if (n < 2 || n > 0x10000 || !gGpu)
        return;
    const unsigned last = NVGPU_GAMMA_ENTRIES - 1;
    uint32_t *in = (uint32_t *)IOMallocData(n * 3 * sizeof(uint32_t));
    uint16_t *out = (uint16_t *)IOMallocData(NVGPU_GAMMA_ENTRIES * 3 * sizeof(uint16_t));
    if (in && out) {
        for (unsigned i = 0; i < n; i++) {
            float rgb[3];
            table->getTableEntry(i, &rgb[0], &rgb[1], &rgb[2]);
            memcpy(&in[i * 3], rgb, sizeof(rgb));
            for (int c = 0; c < 3; c++)
                in[i * 3 + c] = lut_value(in[i * 3 + c]);
        }
        for (unsigned i = 0; i <= last; i++) {
            unsigned p = i * (n - 1), k = p / last, frac = p % last;
            for (int c = 0; c < 3; c++) {
                int a = (int)in[k * 3 + c], b = k + 1 < n ? (int)in[(k + 1) * 3 + c] : a;
                int v = a + (b - a) * (int)frac / (int)last;
                out[i * 3 + c] = (uint16_t)(v > 0xffff ? 0xffff : v);
            }
        }
        IOReturn r = gGpu->callPlatformFunction(NVGPU_FN_GAMMA, true, out, (void *)(uintptr_t)screen, nullptr,
                                                 nullptr);
        if (r != kIOReturnSuccess && failures++ < 8)
            IOLog("DisplayPipe: gamma table of %u entries not applied: 0x%x\n", n, r);
        else if (gammas++ < 4)
            IOLog("DisplayPipe: gamma table of %u entries, green %#x %#x %#x\n", n, out[1], out[last / 2 * 3 + 1],
                  out[last * 3 + 1]);
    }
    if (in)
        IOFreeData(in, n * 3 * sizeof(uint32_t));
    if (out)
        IOFreeData(out, NVGPU_GAMMA_ENTRIES * 3 * sizeof(uint16_t));
}

uint64_t DisplayPipe::submitTransaction(IOAccelDisplayPipeTransaction2 *t)
{
    uint32_t dirty = t->getTransactionDirtyBits();
    if (dirty & IOAccelDisplayPipeTransaction2::kDirtyPipePostGamma) {
        if (const IOAccelDisplayPipeGammaTable2 *g = t->getPipePostGammaTable())
            set_gamma(g);
    } else if (dirty & IOAccelDisplayPipeTransaction2::kDirtyPipePreGamma) {
        if (const IOAccelDisplayPipeGammaTable2 *g = t->getPipePreGammaTable())
            set_gamma(g);
    }
    IOSurface *s = t->getPlaneIOSurface(0, 0);
    if (s && logged++ < 4)
        IOLog("DisplayPipe: transaction: surface %#x %ux%u\n", s->getSurfaceID(), s->getWidth(), s->getHeight());
    if (!s || !gGpu)
        return kIOReturnSuccess;
    // normally done already: WindowServer waits for each transaction before the next
    for (int i = 0; __atomic_load_n(&present_busy, __ATOMIC_ACQUIRE) && i < 4000; i++)
        IODelay(50);
    // a stuck one still uses the request and surface, so this frame is dropped
    if (__atomic_load_n(&present_busy, __ATOMIC_ACQUIRE))
        return kIOReturnSuccess;
    pace_submit(last_head_vblank);
    request = { s->getMemoryDescriptor(), s->getBaseOffset(), s->getWidth(), s->getHeight(), s->getBytesPerRow(),
                screen, 0 };
    presenting = reinterpret_cast<OSObject *>(s);
    presenting->retain();
    copying = t;
    __atomic_store_n(&present_busy, 1, __ATOMIC_RELEASE);
    thread_call_enter(present_call);
    return kIOReturnSuccess;
}

void DisplayPipe::present_run(thread_call_param_t p0, thread_call_param_t)
{
    DisplayPipe *self = (DisplayPipe *)p0;
    IOReturn r = gGpu->callPlatformFunction(NVGPU_FN_PRESENT, true, &self->request, nullptr, nullptr, nullptr);
    if (r != kIOReturnSuccess && self->failures++ < 8)
        IOLog("DisplayPipe: present %ux%u pitch %u failed 0x%x\n", self->request.width, self->request.height,
              self->request.pitch, r);
    self->fence = self->request.fence;
    self->present_result = r;
    // the copy takes a fraction of a frame; complete then, not at the next vblank
    for (int i = 0; r == kIOReturnSuccess && i < 250 &&
                    gGpu->callPlatformFunction(NVGPU_FN_FENCE, true, (void *)(uintptr_t)self->fence, nullptr, nullptr,
                                                nullptr) == kIOReturnBusy;
         i++)
        IODelay(20);
    self->presenting->release();
    self->presenting = nullptr;
    pace_done();
    __atomic_store_n(&self->present_busy, 0, __ATOMIC_RELEASE);
    self->signalTransactionInterrupt(nullptr);
}

// polled on every vblank and on the present thread's signal
uint64_t DisplayPipe::isTransactionComplete(IOAccelDisplayPipeTransaction2 *t)
{
    if (t != copying)
        return 1;
    if (__atomic_load_n(&present_busy, __ATOMIC_ACQUIRE))
        return 0;
    if (present_result == kIOReturnSuccess && gGpu &&
        gGpu->callPlatformFunction(NVGPU_FN_FENCE, true, (void *)(uintptr_t)fence, nullptr, nullptr, nullptr) ==
            kIOReturnBusy)
        return 0;
    copying = nullptr;
    return 1;
}

}  // namespace malachite::accel
