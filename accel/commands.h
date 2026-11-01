// Shared command ABI between the Metal plugin (records) and CommandQueue (executes). A kernel command is
// {malachite_cmd_hdr header; payload}; the IOAccelCommandQueue segment iterator reads len from header+4 and dispatches
// on opcode at header+0, payload at header+8. len is total (incl. header), 4-aligned, >= 8.
#pragma once
#include <stdint.h>

struct malachite_cmd_hdr {
    uint32_t opcode;
    uint32_t len;  // total record length incl. this header, 4-aligned
};

// Opcodes start at 0x100 to stay clear of the base IOAccelCommandQueue's generic command types (0 NoOp, 1
// resource-ref, 2 timestamp) - a low value would be mis-parsed by the base if our override isn't reached.
enum {
    MALACHITE_CMD_COPY_BUF2BUF = 0x100,
    MALACHITE_CMD_FILL_BUFFER = 0x101,
    MALACHITE_CMD_DISPATCH = 0x102,  // payload: struct nvgpu_dispatch (gpu_bridge.h), passed through to NVGpuDispatch
    MALACHITE_CMD_DRAW = 0x103,      // payload: struct nvgpu_draw, passed through to NVGpuDraw
    MALACHITE_CMD_COPY_2D = 0x104,   // payload: struct nvgpu_copy2d, passed through to NVGpuCopy2D
    MALACHITE_CMD_COMPRESS = 0x105,  // payload: struct malachite_compress
};

// A block-linear colour surface's whole buffer as a compressed kind, before anything is in it: what was written under
// another kind reads back as garbage.
struct malachite_compress {
    uint64_t addr;
};

// Buffer->buffer linear copy on raw GPU virtual addresses (srcRef/dstRef unused, 0); the kernel feeds
// srcAddr/dstAddr/size straight into an AMPERE_DMA_COPY_B (0xC7B5) pushbuffer.
struct malachite_copy_buf2buf {
    uint32_t srcRef;   // +0x00
    uint32_t dstRef;   // +0x04
    uint64_t srcAddr;  // +0x08  gpuAddress(src) + sourceOffset
    uint64_t dstAddr;  // +0x10  gpuAddress(dst) + destinationOffset
    uint64_t size;     // +0x18
};  // 0x20 bytes

// Fill a buffer range with a repeating byte (fillBuffer:range:value:). pattern is the byte replicated to 32 bits.
struct malachite_fill_buffer {
    uint64_t dstAddr;  // +0x00  gpuAddress(dst) + range.location
    uint64_t size;     // +0x08  range.length
    uint32_t pattern;  // +0x10  value * 0x01010101
};
