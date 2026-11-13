// Command records the Metal plugin writes and CommandQueue runs: a header, then the payload. The base's segment
// iterator reads the layout, so it is fixed.
#pragma once
#include <stdint.h>

struct malachite_cmd_hdr {
    uint32_t opcode;
    uint32_t len;  // whole record, 4-aligned
};

// from 0x100, clear of the base's generic types (0 NoOp, 1 resource-ref, 2 timestamp)
enum {
    MALACHITE_CMD_COPY_BUF2BUF = 0x100,
    MALACHITE_CMD_FILL_BUFFER = 0x101,
    MALACHITE_CMD_DISPATCH = 0x102,  // struct nvgpu_dispatch (gpu_bridge.h)
    MALACHITE_CMD_DRAW = 0x103,      // struct nvgpu_draw
    MALACHITE_CMD_COPY_2D = 0x104,   // struct nvgpu_copy2d
    MALACHITE_CMD_COMPRESS = 0x105,  // struct malachite_compress
    MALACHITE_CMD_UPLOAD = 0x106,    // struct malachite_copy_buf2buf, not waiting for draws
    MALACHITE_CMD_PUSH = 0x107,      // struct malachite_push
};

// methods for a run of draws, written by the plugin in its own context's buffer; one GPFIFO entry
struct malachite_push {
    uint64_t va;
    uint32_t words;
    uint32_t draws;
};

// set on every opcode once OpenGL's renderer is loaded: OpenGL reads back before a command buffer would complete, so
// its queues wait for their draws at submit
#define MALACHITE_CMD_OPENGL 0x80000000u

// a block-linear colour surface's buffer as a compressed kind, set while empty: data written under another kind
// reads back as garbage
struct malachite_compress {
    uint64_t addr;
};

// raw GPU VAs, into an AMPERE_DMA_COPY_B (0xC7B5) copy
struct malachite_copy_buf2buf {
    uint32_t srcRef;   // +0x00  unused, 0
    uint32_t dstRef;   // +0x04  unused, 0
    uint64_t srcAddr;  // +0x08  gpuAddress(src) + sourceOffset
    uint64_t dstAddr;  // +0x10  gpuAddress(dst) + destinationOffset
    uint64_t size;     // +0x18
};  // 0x20 bytes

// fillBuffer:range:value:
struct malachite_fill_buffer {
    uint64_t dstAddr;  // +0x00  gpuAddress(dst) + range.location
    uint64_t size;     // +0x08  range.length
    uint32_t pattern;  // +0x10  value * 0x01010101
};
