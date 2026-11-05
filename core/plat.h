// What the driver core needs from its host: register access, DMA memory, time, logging and firmware.
// Implemented by tools/vfio.c (Linux user space through VFIO) and kext/plat.cpp (macOS IOKit).
#pragma once
#include <stddef.h>
#include <stdint.h>

struct plat;

struct chip;

struct gpu {
    volatile uint8_t *bar0;
    struct plat *plat;
    const struct chip *chip;
};

static inline uint32_t rd32(struct gpu *g, uint32_t off) { return *(volatile uint32_t *)(g->bar0 + off); }
static inline void wr32(struct gpu *g, uint32_t off, uint32_t v) { *(volatile uint32_t *)(g->bar0 + off) = v; }

// DMA memory: one device address when contiguous, else one per 4 KB page in pages[]
struct dma {
    void *cpu;
    uint64_t addr;
    uint64_t *pages;
    size_t size;
    void *handle;
};

static inline uint64_t dma_addr(const struct dma *d, size_t off)
{
    return d->pages ? d->pages[off >> 12] + (off & 0xfff) : d->addr + off;
}

// zeroed; contiguous memory has one device address, other memory is only page-contiguous
int dma_alloc(struct gpu *g, struct dma *d, size_t size, int contiguous);
void dma_free(struct gpu *g, struct dma *d);
uint64_t dma_live_bytes(void);  // what dma_alloc has handed out and dma_free not taken back

void *mem_alloc(size_t size);  // zeroed
void mem_free(void *p);

void delay_us(uint32_t us);
uint64_t time_us(void);

void log_msg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void log_sysctl_register(void);  // kern.malachite_log: what log_msg kept
void log_sysctl_unregister(void);

// firmware files by name (gsp.bin, bootloader.bin, booter_load.bin, booter_unload.bin)
const void *fw_get(struct gpu *g, const char *name, size_t *size);


uint64_t bar_phys(struct gpu *g, int bar);
int is_boot_vga(struct gpu *g);
uint32_t pci_bdf(struct gpu *g);  // bus << 8 | device << 3 | function
uint64_t bar_size(struct gpu *g, int bar);
// the CPU's view of the BAR1 page at offset off, valid until the next call; NULL if there is none
volatile uint8_t *bar1_cpu(struct gpu *g, uint64_t off);
uint16_t pci_device_id(struct gpu *g);

// user memory of the process behind an RM call (task is whatever the platform passed to rmfe_open)
int copy_from_user(void *task, void *dst, uint64_t src, size_t n);
int copy_to_user(void *task, uint64_t dst, const void *src, size_t n);

void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
int strcmp(const char *a, const char *b);
size_t strlen(const char *s);
