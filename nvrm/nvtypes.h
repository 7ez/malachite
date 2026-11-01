// SPDX-License-Identifier: MIT
// NVIDIA base types, as in Linux nouveau's include/nvrm/nvtypes.h
#pragma once
#include <stdint.h>

#define NV_ALIGN_BYTES(a) __attribute__((__aligned__(a)))
#define NV_DECLARE_ALIGNED(f, a) f __attribute__((__aligned__(a)))
#ifndef __counted_by
#define __counted_by(x)
#endif

typedef uint32_t NvV32;
typedef uint8_t NvU8;
typedef uint16_t NvU16;
typedef uint32_t NvU32;
typedef uint64_t NvU64;
typedef void *NvP64;
typedef NvU8 NvBool;
typedef NvU32 NvHandle;
typedef NvU64 NvLength;
typedef NvU64 RmPhysAddr;
typedef NvU32 NV_STATUS;
typedef union {
} rpc_generic_union;
