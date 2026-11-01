// IOKit side of the platform: what the kext keeps per GPU
#pragma once
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/pci/IOPCIDevice.h>

extern "C" {
#include "../core/plat.h"
}

#include "abi.h"

struct plat {
    IOPCIDevice *pci;
    IOMemoryMap *bar1;
    uint64_t bar1_base, bar1_len;  // BAR1 as the driver placed it when macOS left it unassigned, else 0
    IOBufferMemoryDescriptor *fw[MALACHITE_FW_COUNT];
};

// BAR1 where macOS left it unassigned, shrunk if the full size has no room; size is what it ended up
bool pci_place_bar1(IOPCIDevice *pci, uint64_t *base, uint64_t *size);
void pci_restore_bar1(IOPCIDevice *pci, uint64_t base);
