// The core's platform layer on IOKit
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IOLib.h>
#include <IOKit/pci/IOPCIDevice.h>
#include <kern/clock.h>
#include <mach/mach_time.h>
#include <pexpert/pexpert.h>
#include <sys/kauth.h>
#include <sys/sysctl.h>
#include <sys/systm.h>

#include "abi.h"
#include "plat.h"

// the GPU sees physical addresses directly: OpenCore disables the I/O mapper
static const IOOptionBits no_mapper = kIOMemoryMapperNone;

extern "C" {

static uint64_t dma_live;

uint64_t dma_live_bytes(void)
{
    return __atomic_load_n(&dma_live, __ATOMIC_RELAXED);
}

int dma_alloc(struct gpu *g, struct dma *d, size_t size, int contiguous)
{
    memset(d, 0, sizeof(*d));
    size = (size + 0xfff) & ~(size_t)0xfff;
    IOOptionBits opts = kIODirectionInOut | (contiguous ? kIOMemoryPhysicallyContiguous : 0);
    IOBufferMemoryDescriptor *md = IOBufferMemoryDescriptor::inTaskWithPhysicalMask(kernel_task, opts, size,
                                                                                    0x0000fffffffff000ull);
    if (!md)
        return -1;
    if (md->prepare() != kIOReturnSuccess) {
        md->release();
        return -1;
    }
    d->cpu = md->getBytesNoCopy();
    d->size = size;
    d->handle = md;
    __atomic_add_fetch(&dma_live, size, __ATOMIC_RELAXED);
    memset(d->cpu, 0, size);
    if (contiguous) {
        d->addr = md->getPhysicalSegment(0, nullptr, no_mapper);
    } else {
        d->pages = (uint64_t *)mem_alloc((size >> 12) * sizeof(uint64_t));
        for (size_t off = 0; off < size; off += 0x1000)
            d->pages[off >> 12] = md->getPhysicalSegment(off, nullptr, no_mapper);
        d->addr = d->pages[0];
    }
    return 0;
}

void dma_free(struct gpu *g, struct dma *d)
{
    IOBufferMemoryDescriptor *md = (IOBufferMemoryDescriptor *)d->handle;
    if (md) {
        md->complete();
        md->release();
        __atomic_sub_fetch(&dma_live, d->size, __ATOMIC_RELAXED);
    }
    mem_free(d->pages);
    memset(d, 0, sizeof(*d));
}

// IOFree needs the size, kept in front of the block
void *mem_alloc(size_t size)
{
    size_t *p = (size_t *)IOMalloc(size + 16);
    if (!p)
        return nullptr;
    p[0] = size + 16;
    memset(p + 2, 0, size);
    return p + 2;
}

void mem_free(void *ptr)
{
    if (!ptr)
        return;
    size_t *p = (size_t *)ptr - 2;
    IOFree(p, p[0]);
}

void delay_us(uint32_t us)
{
    if (us >= 1000)
        IOSleep(us / 1000);
    else
        IODelay(us);
}

uint64_t time_us(void)
{
    uint64_t ns;
    absolutetime_to_nanoseconds(mach_absolute_time(), &ns);
    return ns / 1000;
}

// kern.malachite_log: the unified log drops most kernel messages and the kernel buffer wraps, so without a serial
// port this is the only record of where bring-up stopped
static char log_ring[256 * 1024];
static uint64_t log_end;  // bytes ever written; writers reserve their span so lines don't overlap

void log_msg(const char *fmt, ...)
{
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    size_t len = (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1;
    IOLog("%s", line);
    uint64_t at = __sync_fetch_and_add(&log_end, len);
    for (size_t i = 0; i < len; i++)
        log_ring[(at + i) % sizeof(log_ring)] = line[i];
}

static int log_sysctl(struct sysctl_oid *, void *, int, struct sysctl_req *req)
{
    if (!kauth_cred_issuser(kauth_cred_get()))
        return EPERM;
    uint64_t end = log_end, size = sizeof(log_ring), start = end > size ? end - size : 0;
    // the log may have grown since the reader sized its buffer: the newest bytes that fit, as sysctl -b shows ENOMEM
    // as nothing
    if (req->oldptr && end - start > req->oldlen - req->oldidx)
        start = end - (req->oldlen - req->oldidx);
    size_t first = (size_t)(start % size), len = (size_t)(end - start);
    size_t head = len < size - first ? len : size - first;
    int err = SYSCTL_OUT(req, log_ring + first, head);
    if (!err && len > head)
        err = SYSCTL_OUT(req, log_ring, len - head);
    return err;
}

SYSCTL_PROC(_kern, OID_AUTO, malachite_log, CTLTYPE_OPAQUE | CTLFLAG_RD | CTLFLAG_LOCKED, nullptr, 0, log_sysctl, "S",
            "the NVIDIA driver's log since it loaded");

void log_sysctl_register(void)
{
    sysctl__kern_malachite_log.oid_number = OID_AUTO;
    sysctl_register_oid(&sysctl__kern_malachite_log);
}

void log_sysctl_unregister(void)
{
    sysctl_unregister_oid(&sysctl__kern_malachite_log);
}

const void *fw_get(struct gpu *g, const char *name, size_t *size)
{
    static const char *const names[] = MALACHITE_FW_NAMES;
    for (int i = 0; i < MALACHITE_FW_COUNT; i++) {
        if (!strcmp(name, names[i]) && g->plat->fw[i]) {
            *size = g->plat->fw[i]->getLength();
            return g->plat->fw[i]->getBytesNoCopy();
        }
    }
    log_msg("Malachite: firmware %s was not loaded\n", name);
    return nullptr;
}

uint64_t bar_phys(struct gpu *g, int bar)
{
    if (bar == 1 && g->plat->bar1_len)
        return g->plat->bar1_base;
    IODeviceMemory *mem = g->plat->pci->getDeviceMemoryWithRegister(kIOPCIConfigBaseAddress0 + bar * 4);
    return mem ? mem->getPhysicalAddress() : 0;
}

int is_boot_vga(struct gpu *g)
{
    return 0;
}

uint32_t pci_bdf(struct gpu *g)
{
    IOPCIDevice *pci = g->plat->pci;
    return pci->getBusNumber() << 8 | pci->getDeviceNumber() << 3 | pci->getFunctionNumber();
}

uint64_t bar_size(struct gpu *g, int bar)
{
    if (bar == 1 && g->plat->bar1_len)
        return g->plat->bar1_len;
    IODeviceMemory *mem = g->plat->pci->getDeviceMemoryWithRegister(kIOPCIConfigBaseAddress0 + bar * 4);
    return mem ? mem->getLength() : 0;
}

// a page only: a resizable BAR1 is as big as VRAM
volatile uint8_t *bar1_cpu(struct gpu *g, uint64_t off)
{
    OSSafeReleaseNULL(g->plat->bar1);
    IOMemoryDescriptor *page =
        IOMemoryDescriptor::withPhysicalAddress(bar_phys(g, 1) + (off & ~0xfffull), 0x1000, kIODirectionInOut);
    if (page) {
        g->plat->bar1 = page->map(kIOMapWriteCombineCache);
        page->release();
    }
    return g->plat->bar1 ? (volatile uint8_t *)g->plat->bar1->getVirtualAddress() + (off & 0xfff) : NULL;
}

uint16_t pci_device_id(struct gpu *g)
{
    return g->plat->pci->configRead16(kIOPCIConfigDeviceID);
}

// escapes run on the caller's thread, so the current task's address space is the right one
int copy_from_user(void *task, void *dst, uint64_t src, size_t n)
{
    return copyin((user_addr_t)src, dst, n) ? -1 : 0;
}

int copy_to_user(void *task, uint64_t dst, const void *src, size_t n)
{
    return copyout(src, (user_addr_t)dst, n) ? -1 : 0;
}

}
