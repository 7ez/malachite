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
    uint64_t bar1_base, bar1_len;  // where we placed BAR1, else 0
    IOBufferMemoryDescriptor *fw[MALACHITE_FW_COUNT];
};

// for a BAR1 macOS left unassigned; shrunk if the full size has no room, *size is the result
bool pci_place_bar1(IOPCIDevice *pci, uint64_t *base, uint64_t *size);
void pci_restore_bar1(IOPCIDevice *pci, uint64_t base);
