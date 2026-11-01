// SPDX-License-Identifier: MIT
// Suspend/resume interface of GSP-RM r570.144, from Linux nouveau's rm/r570/nvrm/{fbsr,gsp,fifo}.h and
// rm/r535/nvrm/{fbsr,gsp}.h (copies of NVIDIA's MIT headers)
#pragma once

#include "nvtypes.h"


/* Excerpt of RM headers from https://github.com/NVIDIA/open-gpu-kernel-modules/tree/570.144 */

#define GSP_FW_SR_META_MAGIC     0x8a3bb9e6c6c39d93ULL
#define GSP_FW_SR_META_REVISION  2

typedef struct
{
    // Magic for verification by secure ucode
    NvU64 magic;  // = GSP_FW_SR_META_MAGIC;

    //
    // Revision number
    // Bumped up when we change this interface so it is not backward compatible.
    //
    NvU64 revision;  // = GSP_FW_SR_META_MAGIC_REVISION;

    // Members regarding data in SYSMEM
    NvU64 sysmemAddrOfSuspendResumeData;
    NvU64 sizeOfSuspendResumeData;

    //
    // Internal members for use by secure ucode
    // Must be exactly GSP_FW_SR_META_INTERNAL_SIZE bytes.
    //
    NvU32 internal[32];

    // Same as flags of GspFwWprMeta
    NvU32 flags;

    // Subrevision number used by secure ucode
    NvU32 subrevision;

    //
    // Pad structure to exactly 256 bytes (1 DMA chunk).
    // Padding initialized to zero.
    //
    NvU32 padding[22];
} GspFwSRMeta;

#define NV2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_3            (0x00000003U)

#define GPU_STATE_FLAGS_PRESERVING         (1U << 0)  // GPU state is preserved
#define GPU_STATE_FLAGS_PM_TRANSITION      (1U << 2)  // To be used with GPU_STATE_FLAGS_PRESERVING.

#define NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING (0x20800ac3) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_INTERNAL_INTERFACE_ID << 8) | NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING_PARAMS {
    NvBool bDisableActiveChannels;
} NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING_PARAMS;

#define NV2080_CTRL_CMD_INTERNAL_FBSR_INIT (0x20800ac2) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_INTERNAL_INTERFACE_ID << 8) | NV2080_CTRL_INTERNAL_FBSR_INIT_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_INTERNAL_FBSR_INIT_PARAMS {
    NvHandle hClient;
    NvHandle hSysMem;
    NvBool   bEnteringGcoffState;
    NV_DECLARE_ALIGNED(NvU64 sysmemAddrOfSuspendResumeData, 8);
} NV2080_CTRL_INTERNAL_FBSR_INIT_PARAMS;

#define NV0080_CTRL_CMD_FB_GET_COMPBIT_STORE_INFO (0x801306) /* finn: Evaluated from "(FINN_NV01_DEVICE_0_FB_INTERFACE_ID << 8) | NV0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS_MESSAGE_ID" */

typedef struct NV0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS {
    NV_DECLARE_ALIGNED(NvU64 Size, 8);
    NV_DECLARE_ALIGNED(NvU64 Address, 8);
    NvU32 AddressSpace;
    NvU32 MaxCompbitLine;
    NvU32 comptagsPerCacheLine;
    NvU32 cacheLineSize;
    NvU32 cacheLineSizePerSlice;
    NvU32 cacheLineFetchAlignment;
    NV_DECLARE_ALIGNED(NvU64 backingStoreBase, 8);
    NvU32 gobsPerComptagPerSlice;
    NvU32 backingStoreCbcBase;
    NvU32 comptaglineAllocationPolicy;
    NV_DECLARE_ALIGNED(NvU64 privRegionStartOffset, 8);
    NvU32 cbcCoveragePerSlice;
} NV0080_CTRL_FB_GET_COMPBIT_STORE_INFO_PARAMS;

#define NV01_MEMORY_LIST_SYSTEM (0x00000081)

#define NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS (0x00000001 << 4)
#define NVOS02_FLAGS_LOCATION_PCI              (0x00000000 << 8)
#define NVOS02_FLAGS_MAPPING_NO_MAP            (0x00000001 << 30)

struct pte_desc
{
    NvU32 idr:2;
    NvU32 reserved1:14;
    NvU32 length:16;
    union {
        NvU64 pte; // PTE when IDR==0; PDE when IDR > 0
        NvU64 pde; // PTE when IDR==0; PDE when IDR > 0
    } pte_pde[]  NV_ALIGN_BYTES(8); // PTE when IDR==0; PDE when IDR > 0
};

typedef struct rpc_alloc_memory_v13_01
{
    NvHandle   hClient;
    NvHandle   hDevice;
    NvHandle   hMemory;
    NvU32      hClass;
    NvU32      flags;
    NvU32      pteAdjust;
    NvU32      format;
    NvU64      length NV_ALIGN_BYTES(8);
    NvU32      pageCount;
    struct pte_desc pteDesc;
} rpc_alloc_memory_v13_01;
