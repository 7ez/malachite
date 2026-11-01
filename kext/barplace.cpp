// Placing BAR1 where macOS left it unassigned. A resizable BAR1 of over 1 GB is one macOS gives up on, so the driver
// finds a free, size-aligned range in the 64-bit window above the GPU and writes the BAR itself. The window is the
// upstream bridge's prefetchable one, or the host bridge's ACPI memory ranges when the GPU sits on the root bus.
#include <IOKit/IOLib.h>
#include <IOKit/pci/IOPCIDevice.h>

#include "plat.h"

namespace {

enum { PCI_COMMAND = 0x04, PCI_HEADER_TYPE = 0x0e, BAR1_LO = 0x14, BAR1_HI = 0x18 };
enum { BRIDGE_PREF_BASE = 0x24, BRIDGE_PREF_LIMIT = 0x26, BRIDGE_PREF_BASE_HI = 0x28, BRIDGE_PREF_LIMIT_HI = 0x2c };
enum { MAX_RANGES = 96, MAX_ANCESTORS = 8 };

struct range {
    uint64_t lo, hi;  // inclusive
    bool gpu;         // one of the GPU's own BARs
};

struct ranges {
    range r[MAX_RANGES];
    int n;
    bool full;

    void add(uint64_t lo, uint64_t hi, bool gpu = false)
    {
        if (n < MAX_RANGES)
            r[n++] = { lo, hi, gpu };
        else
            full = true;
    }
};

// AppleACPIPlatform's dump of a host bridge's _CRS: one of these per resource, 0 = memory, 1 = I/O, 2 = bus numbers
struct acpi_space {
    uint64_t type, unknown[2], min, max, translation, length, tail[3];
};

bool is_bridge(IOPCIDevice *d)
{
    return (d->configRead8(PCI_HEADER_TYPE) & 0x7f) == 1;
}

bool bridge_window(IOPCIDevice *b, range *w)
{
    uint16_t base = b->configRead16(BRIDGE_PREF_BASE), limit = b->configRead16(BRIDGE_PREF_LIMIT);
    uint64_t lo = (uint64_t)(base & 0xfff0) << 16, hi = (uint64_t)(limit & 0xfff0) << 16 | 0xfffff;
    if ((base & 0xf) == 1) {
        lo |= (uint64_t)b->configRead32(BRIDGE_PREF_BASE_HI) << 32;
        hi |= (uint64_t)b->configRead32(BRIDGE_PREF_LIMIT_HI) << 32;
    }
    *w = { lo, hi };
    return lo <= hi;
}

void set_bridge_window(IOPCIDevice *b, uint64_t lo, uint64_t hi)
{
    b->configWrite32(BRIDGE_PREF_BASE_HI, (uint32_t)(lo >> 32));
    b->configWrite32(BRIDGE_PREF_LIMIT_HI, (uint32_t)(hi >> 32));
    b->configWrite16(BRIDGE_PREF_BASE, (uint16_t)((lo >> 16) & 0xfff0) | 1);
    b->configWrite16(BRIDGE_PREF_LIMIT, (uint16_t)((hi >> 16) & 0xfff0) | 1);
}

// what a bridge (or the host bridge) forwards above 4 GB, where a 64-bit BAR can go
void windows_of(IORegistryEntry *e, ranges *out)
{
    if (IOPCIDevice *b = OSDynamicCast(IOPCIDevice, e)) {
        range w;
        if (bridge_window(b, &w) && w.hi >= 1ull << 32)
            out->add(w.lo, w.hi);
        return;
    }
    OSData *spaces = OSDynamicCast(OSData, e->getProperty("acpi-address-spaces"));
    const acpi_space *s = spaces ? (const acpi_space *)spaces->getBytesNoCopy() : nullptr;
    for (unsigned i = 0; s && i < spaces->getLength() / sizeof(*s); i++) {
        if (s[i].type == 0 && s[i].max >= 1ull << 32 && s[i].min <= s[i].max)
            out->add(s[i].min < 1ull << 32 ? 1ull << 32 : s[i].min, s[i].max);
    }
}

// every range another device owns, from the BARs macOS assigned and the windows of bridges that aren't above the GPU
void collect_used(IOPCIDevice *gpu, IOPCIDevice *const *ancestors, int nancestors, ranges *used)
{
    OSDictionary *match = IOService::serviceMatching("IOPCIDevice");
    OSIterator *it = match ? IOService::getMatchingServices(match) : nullptr;
    while (it) {
        IOPCIDevice *d = OSDynamicCast(IOPCIDevice, it->getNextObject());
        if (!d)
            break;
        bool above = false;
        for (int i = 0; i < nancestors; i++)
            above |= ancestors[i] == d;
        if (OSArray *mem = d->getDeviceMemory()) {
            for (unsigned i = 0; i < mem->getCount(); i++) {
                IOMemoryDescriptor *m = OSDynamicCast(IOMemoryDescriptor, mem->getObject(i));
                if (m && m->getLength())
                    used->add(m->getPhysicalAddress(), m->getPhysicalAddress() + m->getLength() - 1, d == gpu);
            }
        }
        range w;
        if (!above && d != gpu && is_bridge(d) && bridge_window(d, &w))
            used->add(w.lo, w.hi);
    }
    OSSafeReleaseNULL(it);
    OSSafeReleaseNULL(match);
}

// Whether another device needs w. The GPU's BAR3 is in the window macOS sized for it, but nothing here reaches it from
// the CPU, so the window may move out from under it.
bool others_in(const ranges &used, range w)
{
    for (int i = 0; i < used.n; i++) {
        if (!used.r[i].gpu && used.r[i].lo <= w.hi && used.r[i].hi >= w.lo)
            return true;
    }
    return false;
}

// the highest free size-aligned slot: firmware fills a window from the bottom, so the top is where it has put the
// fewest devices macOS doesn't know about
bool pick(const ranges &windows, const ranges &used, uint64_t size, uint64_t *base)
{
    for (int i = windows.n - 1; i >= 0; i--) {
        const range &w = windows.r[i];
        if (w.hi - w.lo + 1 < size)
            continue;
        for (uint64_t c = (w.hi + 1 - size) & ~(size - 1); c >= w.lo;) {
            const range *hit = nullptr;
            for (int j = 0; j < used.n; j++) {
                if (used.r[j].lo <= c + size - 1 && used.r[j].hi >= c && (!hit || used.r[j].lo < hit->lo))
                    hit = &used.r[j];
            }
            if (!hit) {
                *base = c;
                return true;
            }
            if (hit->lo < size)
                break;
            c = (hit->lo - size) & ~(size - 1);
        }
    }
    return false;
}

}  // namespace

// BAR1's size by writing ones with decode off, 0 when it isn't a 64-bit memory BAR
static uint64_t bar1_size_of(IOPCIDevice *pci)
{
    uint32_t lo = pci->configRead32(BAR1_LO), hi = pci->configRead32(BAR1_HI);
    if ((lo & 7) != 4)
        return 0;
    pci->configWrite32(BAR1_LO, 0xffffffff);
    pci->configWrite32(BAR1_HI, 0xffffffff);
    uint64_t mask = (uint64_t)pci->configRead32(BAR1_HI) << 32 | (pci->configRead32(BAR1_LO) & ~0xfu);
    pci->configWrite32(BAR1_LO, lo);
    pci->configWrite32(BAR1_HI, hi);
    return -mask;
}

bool pci_place_bar1(IOPCIDevice *pci, uint64_t *base, uint64_t *size)
{
    uint16_t command = pci->configRead16(PCI_COMMAND);
    uint32_t old_lo = pci->configRead32(BAR1_LO), old_hi = pci->configRead32(BAR1_HI);
    pci->setMemoryEnable(false);
    *size = bar1_size_of(pci);
    if (!*size) {
        pci->configWrite16(PCI_COMMAND, command);
        return false;
    }

    IOPCIDevice *ancestors[MAX_ANCESTORS];
    int nancestors = 0;
    IORegistryEntry *root = nullptr;
    for (IORegistryEntry *e = pci->getParentEntry(gIOServicePlane); e && !root; e = e->getParentEntry(gIOServicePlane)) {
        if (IOPCIDevice *d = OSDynamicCast(IOPCIDevice, e)) {
            if (nancestors < MAX_ANCESTORS)
                ancestors[nancestors++] = d;
        } else if (e->getProperty("acpi-address-spaces")) {
            root = e;
        }
    }

    ranges used = {}, windows = {};
    collect_used(pci, ancestors, nancestors, &used);
    IOPCIDevice *bridge = nancestors ? ancestors[0] : nullptr;
    IORegistryEntry *above = bridge ? (nancestors > 1 ? (IORegistryEntry *)ancestors[1] : root) : root;
    uint32_t saved[4] = {};
    bool widened = false, placed = false;
    range old = {};
    bool had_window = bridge && bridge_window(bridge, &old);
    if (used.full) {
        log_msg("Malachite: too many PCI ranges to place BAR1 safely\n");
    } else if (!bridge) {
        if (root)
            windows_of(root, &windows);
        placed = pick(windows, used, *size, base);
    } else if (had_window && old.hi - old.lo + 1 >= *size && old.lo >= 1ull << 32) {
        windows_of(bridge, &windows);
        placed = pick(windows, used, *size, base);
    } else if ((bridge->configRead16(BRIDGE_PREF_BASE) & 0xf) == 1 && above && !(had_window && others_in(used, old))) {
        // the bridge's window is absent or too small, and no other device lives in it: move it into free space above
        windows_of(above, &windows);
        placed = pick(windows, used, *size, base);
        if (placed) {
            saved[0] = bridge->configRead32(BRIDGE_PREF_BASE_HI);
            saved[1] = bridge->configRead32(BRIDGE_PREF_LIMIT_HI);
            saved[2] = bridge->configRead16(BRIDGE_PREF_BASE);
            saved[3] = bridge->configRead16(BRIDGE_PREF_LIMIT);
            set_bridge_window(bridge, *base, *base + *size - 1);
            bridge->setMemoryEnable(true);
            widened = true;
        }
    }
    if (!placed) {
        log_msg("Malachite: no free %llu MB range for BAR1 in the %s's 64-bit window\n", (unsigned long long)(*size >> 20),
                bridge ? "upstream bridge" : "host bridge");
        log_msg("Malachite: its window %#llx..%#llx%s, %d windows above, %d ranges in use:\n", old.lo, old.hi,
                had_window ? "" : " (none)", windows.n, used.n);
        for (int i = 0; i < windows.n; i++)
            log_msg("Malachite:   window %#llx..%#llx\n", windows.r[i].lo, windows.r[i].hi);
        for (int i = 0; i < used.n; i++)
            log_msg("Malachite:   used %#llx..%#llx\n", used.r[i].lo, used.r[i].hi);
        pci->configWrite16(PCI_COMMAND, command);
        return false;
    }

    pci->configWrite32(BAR1_LO, (uint32_t)*base | (old_lo & 0xf));
    pci->configWrite32(BAR1_HI, (uint32_t)(*base >> 32));
    if (pci->configRead32(BAR1_HI) != (uint32_t)(*base >> 32) || (pci->configRead32(BAR1_LO) & ~0xfu) != (uint32_t)*base) {
        log_msg("Malachite: BAR1 did not take %#llx\n", (unsigned long long)*base);
        pci->configWrite32(BAR1_LO, old_lo);
        pci->configWrite32(BAR1_HI, old_hi);
        if (widened) {
            bridge->configWrite32(BRIDGE_PREF_BASE_HI, saved[0]);
            bridge->configWrite32(BRIDGE_PREF_LIMIT_HI, saved[1]);
            bridge->configWrite16(BRIDGE_PREF_BASE, (uint16_t)saved[2]);
            bridge->configWrite16(BRIDGE_PREF_LIMIT, (uint16_t)saved[3]);
        }
        pci->configWrite16(PCI_COMMAND, command);
        return false;
    }
    log_msg("Malachite: macOS left BAR1 unassigned: %llu MB placed at %#llx in the %s's window%s\n",
            (unsigned long long)(*size >> 20), (unsigned long long)*base, bridge ? "upstream bridge" : "host bridge",
            widened ? " (widened)" : "");
    pci->setMemoryEnable(true);
    return true;
}

void pci_restore_bar1(IOPCIDevice *pci, uint64_t base)
{
    uint32_t lo = pci->configRead32(BAR1_LO);
    if ((lo & ~0xfu) != (uint32_t)base || pci->configRead32(BAR1_HI) != (uint32_t)(base >> 32)) {
        pci->configWrite32(BAR1_LO, (uint32_t)base | (lo & 0xf));
        pci->configWrite32(BAR1_HI, (uint32_t)(base >> 32));
    }
}
