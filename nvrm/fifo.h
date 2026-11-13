// SPDX-License-Identifier: MIT
// VA space, FIFO and channel interface of GSP-RM r570.144, from Linux nouveau's
// rm/r535/nvrm/{vmm,fifo}.h and rm/r570/nvrm/fifo.h (copies of NVIDIA's MIT headers)
#pragma once

#include "nvtypes.h"

#define BIT(n) (1U << (n))


/* Excerpt of RM headers from https://github.com/NVIDIA/open-gpu-kernel-modules/tree/535.113.01 */

#define FERMI_VASPACE_A                                     (0x000090f1)

typedef struct
{
    NvU32   index;
    NvV32   flags;
    NvU64   vaSize NV_ALIGN_BYTES(8);
    NvU64   vaStartInternal NV_ALIGN_BYTES(8);
    NvU64   vaLimitInternal NV_ALIGN_BYTES(8);
    NvU32   bigPageSize;
    NvU64   vaBase NV_ALIGN_BYTES(8);
} NV_VASPACE_ALLOCATION_PARAMETERS;

#define NV_VASPACE_ALLOCATION_INDEX_GPU_NEW                                 0x00 //<! Create new VASpace, by default

#define NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED                   BIT(3)

#define SPLIT_VAS_SERVER_RM_MANAGED_VA_START   0x100000000ULL  // 4GB
#define SPLIT_VAS_SERVER_RM_MANAGED_VA_SIZE     0x20000000ULL  // 512MB

#define GMMU_FMT_MAX_LEVELS  6U

#define NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES (0x90f10106U) /* finn: Evaluated from "(FINN_FERMI_VASPACE_A_VASPACE_INTERFACE_ID << 8) | NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_MESSAGE_ID" */
typedef struct NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS {
    /*!
     * [in] GPU sub-device handle - this API only supports unicast.
     *      Pass 0 to use subDeviceId instead.
     */
    NvHandle hSubDevice;

    /*!
     * [in] GPU sub-device ID. Ignored if hSubDevice is non-zero.
     */
    NvU32    subDeviceId;

    /*!
     * [in] Page size (VA coverage) of the level to reserve.
     *      This need not be a leaf (page table) page size - it can be
     *      the coverage of an arbitrary level (including root page directory).
     */
    NV_DECLARE_ALIGNED(NvU64 pageSize, 8);

    /*!
     * [in] First GPU virtual address of the range to reserve.
     *      This must be aligned to pageSize.
     */
    NV_DECLARE_ALIGNED(NvU64 virtAddrLo, 8);

    /*!
     * [in] Last GPU virtual address of the range to reserve.
     *      This (+1) must be aligned to pageSize.
     */
    NV_DECLARE_ALIGNED(NvU64 virtAddrHi, 8);

    /*! 
     * [in] Number of PDE levels to copy.
     */
    NvU32    numLevelsToCopy;

   /*!
     * [in] Per-level information.
     */
    struct {
        /*!
         * Physical address of this page level instance.
         */
        NV_DECLARE_ALIGNED(NvU64 physAddress, 8);

        /*!
         * Size in bytes allocated for this level instance.
         */
        NV_DECLARE_ALIGNED(NvU64 size, 8);

        /*!
         * Aperture in which this page level instance resides.
         */
        NvU32 aperture;

        /*!
         * Page shift corresponding to the level
         */
        NvU8  pageShift;
    } levels[GMMU_FMT_MAX_LEVELS];
} NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS;

#define NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY (0x801813U) /* finn: Evaluated from "(FINN_NV01_DEVICE_0_DMA_INTERFACE_
ID << 8) | NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_MESSAGE_ID" */

typedef struct NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS {
    NV_DECLARE_ALIGNED(NvU64 physAddress, 8);
    NvU32    numEntries;
    NvU32    flags;
    NvHandle hVASpace;
    NvU32    chId;
    NvU32    subDeviceId; // ID+1, 0 for BC
    NvU32    pasid;
} NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS;

#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE                  1:0
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_VIDMEM           (0x00000000U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_SYSMEM_COH       (0x00000001U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_APERTURE_SYSMEM_NONCOH    (0x00000002U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_PRESERVE_PDES             2:2
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_PRESERVE_PDES_FALSE       (0x00000000U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_PRESERVE_PDES_TRUE        (0x00000001U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_ALL_CHANNELS              3:3
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_ALL_CHANNELS_FALSE        (0x00000000U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_ALL_CHANNELS_TRUE         (0x00000001U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_IGNORE_CHANNEL_BUSY       4:4
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_IGNORE_CHANNEL_BUSY_FALSE (0x00000000U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_IGNORE_CHANNEL_BUSY_TRUE  (0x00000001U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_EXTEND_VASPACE            5:5
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_EXTEND_VASPACE_FALSE      (0x00000000U)
#define NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_FLAGS_EXTEND_VASPACE_TRUE       (0x00000001U)

#define NV0080_CTRL_CMD_DMA_UNSET_PAGE_DIRECTORY                           (0x801814U) /* finn: Evaluated from "(FINN_NV01_DEVICE_0_DMA_INTERFACE_ID << 8) | NV0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS_MESSAGE_ID" */

typedef struct NV0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS {
    NvHandle hVASpace;
    NvU32    subDeviceId; // ID+1, 0 for BC
} NV0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS;

/* Excerpt of RM headers from https://github.com/NVIDIA/open-gpu-kernel-modules/tree/535.113.01 */

#define NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_MAX_ENTRIES         32

#define NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_DATA_TYPES   16

#define NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_PBDMA    2

#define NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_NAME_LEN 16

typedef struct NV2080_CTRL_FIFO_DEVICE_ENTRY {
    NvU32 engineData[NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_DATA_TYPES];
    NvU32 pbdmaIds[NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_PBDMA];
    NvU32 pbdmaFaultIds[NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_PBDMA];
    NvU32 numPbdmas;
    char  engineName[NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_ENGINE_MAX_NAME_LEN];
} NV2080_CTRL_FIFO_DEVICE_ENTRY;

#define NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE                 (0x20801112) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_FIFO_INTERFACE_ID << 8) | NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS_MESSAGE_ID" */
typedef struct NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS {
    NvU32                         baseIndex;
    NvU32                         numEntries;
    NvBool                        bMore;
    // C form: NV2080_CTRL_FIFO_DEVICE_ENTRY entries[NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_MAX_ENTRIES];
    NV2080_CTRL_FIFO_DEVICE_ENTRY entries[NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_MAX_ENTRIES];
} NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS;

typedef enum
{
    /* *************************************************************************
     * Bug 3820969
     * THINK BEFORE CHANGING ENUM ORDER HERE.
     * VGPU-guest uses this same ordering. Because this enum is not versioned,
     * changing the order here WILL BREAK old-guest-on-newer-host compatibility.
     * ************************************************************************/

    // *ENG_XYZ, e.g.: ENG_GR, ENG_CE etc.,
    ENGINE_INFO_TYPE_ENG_DESC = 0,

    // HW engine ID
    ENGINE_INFO_TYPE_FIFO_TAG,

    // RM_ENGINE_TYPE_*
    ENGINE_INFO_TYPE_RM_ENGINE_TYPE,

    //
    // runlist id (meaning varies by GPU)
    // Valid only for Esched-driven engines
    //
    ENGINE_INFO_TYPE_RUNLIST,

    // NV_PFIFO_INTR_MMU_FAULT_ENG_ID_*
    ENGINE_INFO_TYPE_MMU_FAULT_ID,

    // ROBUST_CHANNEL_*
    ENGINE_INFO_TYPE_RC_MASK,

    // Reset Bit Position. On Ampere, only valid if not _INVALID
    ENGINE_INFO_TYPE_RESET,

    // Interrupt Bit Position
    ENGINE_INFO_TYPE_INTR,

    // log2(MC_ENGINE_*)
    ENGINE_INFO_TYPE_MC,

    // The DEV_TYPE_ENUM for this engine
    ENGINE_INFO_TYPE_DEV_TYPE_ENUM,

    // The particular instance of this engine type
    ENGINE_INFO_TYPE_INSTANCE_ID,

    //
    // The base address for this engine's NV_RUNLIST. Valid only on Ampere+
    // Valid only for Esched-driven engines
    //
    ENGINE_INFO_TYPE_RUNLIST_PRI_BASE,

    //
    // If this entry is a host-driven engine.
    // Update _isEngineInfoTypeValidForOnlyHostDriven when adding any new entry.
    //
    ENGINE_INFO_TYPE_IS_HOST_DRIVEN_ENGINE,

    //
    // The index into the per-engine NV_RUNLIST registers. Valid only on Ampere+
    // Valid only for Esched-driven engines
    //
    ENGINE_INFO_TYPE_RUNLIST_ENGINE_ID,

    //
    // The base address for this engine's NV_CHRAM registers. Valid only on
    // Ampere+
    //
    // Valid only for Esched-driven engines
    //
    ENGINE_INFO_TYPE_CHRAM_PRI_BASE,

    // This entry added to copy data at RMCTRL_EXPORT() call for Kernel RM
    ENGINE_INFO_TYPE_KERNEL_RM_MAX,
    // Used for iterating the engine info table by the index passed.
    ENGINE_INFO_TYPE_INVALID = ENGINE_INFO_TYPE_KERNEL_RM_MAX,

    // Size of FIFO_ENGINE_LIST.engineData
    ENGINE_INFO_TYPE_ENGINE_DATA_ARRAY_SIZE = ENGINE_INFO_TYPE_INVALID,

    // Input-only parameter for kfifoEngineInfoXlate.
    ENGINE_INFO_TYPE_PBDMA_ID

    /* *************************************************************************
     * Bug 3820969
     * THINK BEFORE CHANGING ENUM ORDER HERE.
     * VGPU-guest uses this same ordering. Because this enum is not versioned,
     * changing the order here WILL BREAK old-guest-on-newer-host compatibility.
     * ************************************************************************/
} ENGINE_INFO_TYPE;

#define NV2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE (0x20802a08) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | NV2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS_MESSAGE_ID" */
typedef struct NV2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS {
    NvU32 size;
} NV2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS;

// r570's id; the r535 one, 0x20800a42, is now an NVLink control
#define NV2080_CTRL_GPU_MAX_CONSTRUCTED_FALCONS         0x40

typedef struct NV2080_CTRL_GPU_CONSTRUCTED_FALCON_INFO {
    NvU32 engDesc;
    NvU32 ctxAttr;
    NvU32 ctxBufferSize;
    NvU32 addrSpaceList;
    NvU32 registerBase;
} NV2080_CTRL_GPU_CONSTRUCTED_FALCON_INFO;

#define NV2080_CTRL_CMD_GPU_GET_CONSTRUCTED_FALCON_INFO (0x208001b0) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_GPU_INTERFACE_ID << 8) | NV2080_CTRL_GPU_GET_CONSTRUCTED_FALCON_INFO_PARAMS_MESSAGE_ID" */
typedef struct NV2080_CTRL_GPU_GET_CONSTRUCTED_FALCON_INFO_PARAMS {
    NvU32                                   numConstructedFalcons;
    NV2080_CTRL_GPU_CONSTRUCTED_FALCON_INFO constructedFalconsTable[NV2080_CTRL_GPU_MAX_CONSTRUCTED_FALCONS];
} NV2080_CTRL_GPU_GET_CONSTRUCTED_FALCON_INFO_PARAMS;

#define NVA06F_CTRL_CMD_BIND (0xa06f0104) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_BIND_PARAMS_MESSAGE_ID" */
typedef struct NVA06F_CTRL_BIND_PARAMS {
    NvU32 engineType;
} NVA06F_CTRL_BIND_PARAMS;

#define NVA06F_CTRL_CMD_GPFIFO_SCHEDULE (0xa06f0103) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_MESSAGE_ID" */
typedef struct NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS {
    NvBool bEnable;
    NvBool bSkipSubmit;
} NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;

/* Excerpt of RM headers from https://github.com/NVIDIA/open-gpu-kernel-modules/tree/570.144 */

#define NV_MAX_SUBDEVICES             8

typedef struct NV_MEMORY_DESC_PARAMS {
    NV_DECLARE_ALIGNED(NvU64 base, 8);
    NV_DECLARE_ALIGNED(NvU64 size, 8);
    NvU32 addressSpace;
    NvU32 cacheAttrib;
} NV_MEMORY_DESC_PARAMS;

#define CC_CHAN_ALLOC_IV_SIZE_DWORD    3U

#define CC_CHAN_ALLOC_NONCE_SIZE_DWORD 8U

typedef struct NV_CHANNEL_ALLOC_PARAMS {

    NvHandle hObjectError; // error context DMA
    NvHandle hObjectBuffer; // no longer used
    NV_DECLARE_ALIGNED(NvU64 gpFifoOffset, 8);    // offset to beginning of GP FIFO
    NvU32    gpFifoEntries;    // number of GP FIFO entries

    NvU32    flags;


    NvHandle hContextShare; // context share handle
    NvHandle hVASpace; // VASpace for the channel

    // handle to UserD memory object for channel, ignored if hUserdMemory[0]=0
    NvHandle hUserdMemory[NV_MAX_SUBDEVICES];

    // offset to beginning of UserD within hUserdMemory[x]
    NV_DECLARE_ALIGNED(NvU64 userdOffset[NV_MAX_SUBDEVICES], 8);

    // engine type(NV2080_ENGINE_TYPE_*) with which this channel is associated
    NvU32    engineType;
    // Channel identifier that is unique for the duration of a RM session
    NvU32    cid;
    // One-hot encoded bitmask to match SET_SUBDEVICE_MASK methods
    NvU32    subDeviceId;
    NvHandle hObjectEccError; // ECC error context DMA

    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS instanceMem, 8);
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS userdMem, 8);
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS ramfcMem, 8);
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS mthdbufMem, 8);

    NvHandle hPhysChannelGroup;              // reserved
    NvU32    internalFlags;                 // reserved
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS errorNotifierMem, 8); // reserved
    NV_DECLARE_ALIGNED(NV_MEMORY_DESC_PARAMS eccErrorNotifierMem, 8); // reserved
    NvU32    ProcessID;                 // reserved
    NvU32    SubProcessID;                 // reserved

    // IV used for CPU-side encryption / GPU-side decryption.
    NvU32    encryptIv[CC_CHAN_ALLOC_IV_SIZE_DWORD];          // reserved
    // IV used for CPU-side decryption / GPU-side encryption.
    NvU32    decryptIv[CC_CHAN_ALLOC_IV_SIZE_DWORD];          // reserved
    // Nonce used CPU-side signing / GPU-side signature verification.
    NvU32    hmacNonce[CC_CHAN_ALLOC_NONCE_SIZE_DWORD];       // reserved
    NvU32    tpcConfigID; // TPC Configuration Id as supported by DTD-PG Feature
} NV_CHANNEL_ALLOC_PARAMS;

typedef NV_CHANNEL_ALLOC_PARAMS NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS;

#define NVOS04_FLAGS_CHANNEL_TYPE                                  1:0
#define NVOS04_FLAGS_CHANNEL_TYPE_PHYSICAL                         0x00000000
#define NVOS04_FLAGS_CHANNEL_TYPE_VIRTUAL                          0x00000001  // OBSOLETE
#define NVOS04_FLAGS_CHANNEL_TYPE_PHYSICAL_FOR_VIRTUAL             0x00000002  // OBSOLETE
#define NVOS04_FLAGS_VPR                                           2:2
#define NVOS04_FLAGS_VPR_FALSE                                     0x00000000
#define NVOS04_FLAGS_VPR_TRUE                                      0x00000001
#define NVOS04_FLAGS_CC_SECURE                                     2:2
#define NVOS04_FLAGS_CC_SECURE_FALSE                               0x00000000
#define NVOS04_FLAGS_CC_SECURE_TRUE                                0x00000001
#define NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING                  3:3
#define NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_FALSE            0x00000000
#define NVOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_TRUE             0x00000001
#define NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE                       4:4
#define NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_DEFAULT               0x00000000
#define NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE                   0x00000001
#define NVOS04_FLAGS_PRIVILEGED_CHANNEL                           5:5
#define NVOS04_FLAGS_PRIVILEGED_CHANNEL_FALSE                     0x00000000
#define NVOS04_FLAGS_PRIVILEGED_CHANNEL_TRUE                      0x00000001
#define NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING                     6:6
#define NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING_FALSE               0x00000000
#define NVOS04_FLAGS_DELAY_CHANNEL_SCHEDULING_TRUE                0x00000001
#define NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE                7:7
#define NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE_FALSE          0x00000000
#define NVOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE_TRUE           0x00000001
#define NVOS04_FLAGS_CHANNEL_USERD_INDEX_VALUE                    10:8
#define NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED                    11:11
#define NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED_FALSE              0x00000000
#define NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED_TRUE               0x00000001
#define NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_VALUE               20:12
#define NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED               21:21
#define NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_FALSE         0x00000000
#define NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_TRUE          0x00000001
#define NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV                 22:22
#define NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV_FALSE           0x00000000
#define NVOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV_TRUE            0x00000001
#define NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER                        23:23
#define NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER_FALSE                  0x00000000
#define NVOS04_FLAGS_CHANNEL_SKIP_SCRUBBER_TRUE                   0x00000001
#define NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO                      24:24
#define NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO_FALSE                0x00000000
#define NVOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO_TRUE                 0x00000001
#define NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL           25:25
#define NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL_FALSE     0x00000000
#define NVOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL_TRUE      0x00000001
#define NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT                  26:26
#define NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT_FALSE            0x00000000
#define NVOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT_TRUE             0x00000001
#define NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT                 27:27
#define NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT_FALSE           0x00000000
#define NVOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT_TRUE            0x00000001
#define NVOS04_FLAGS_GROUP_CHANNEL_THREAD                          29:28
#define NVOS04_FLAGS_GROUP_CHANNEL_THREAD_DEFAULT                  0x00000000
#define NVOS04_FLAGS_GROUP_CHANNEL_THREAD_ONE                      0x00000001
#define NVOS04_FLAGS_GROUP_CHANNEL_THREAD_TWO                      0x00000002
#define NVOS04_FLAGS_MAP_CHANNEL                                   30:30
#define NVOS04_FLAGS_MAP_CHANNEL_FALSE                             0x00000000
#define NVOS04_FLAGS_MAP_CHANNEL_TRUE                              0x00000001
#define NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC                          31:31
#define NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC_FALSE                    0x00000000
#define NVOS04_FLAGS_SKIP_CTXBUFFER_ALLOC_TRUE                     0x00000001

typedef enum {
    /*!
     * Initial state as passed in NV_CHANNEL_ALLOC_PARAMS by
     * kernel CPU-RM clients.
     */
    ERROR_NOTIFIER_TYPE_UNKNOWN = 0,
    /*! @brief Error notifier is explicitly not set.
     *
     * The corresponding hErrorContext or hEccErrorContext must be
     * NV01_NULL_OBJECT.
     */
    ERROR_NOTIFIER_TYPE_NONE,
    /*! @brief Error notifier is a ContextDma */
    ERROR_NOTIFIER_TYPE_CTXDMA,
    /*! @brief Error notifier is a NvNotification array in sysmem/vidmem */
    ERROR_NOTIFIER_TYPE_MEMORY
} ErrorNotifierType;

#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_PRIVILEGE                       1:0
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_PRIVILEGE_USER                  0x0
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_PRIVILEGE_ADMIN                 0x1
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_PRIVILEGE_KERNEL                0x2
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ERROR_NOTIFIER_TYPE             3:2
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ERROR_NOTIFIER_TYPE_UNKNOWN     ERROR_NOTIFIER_TYPE_UNKNOWN
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ERROR_NOTIFIER_TYPE_NONE        ERROR_NOTIFIER_TYPE_NONE
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ERROR_NOTIFIER_TYPE_CTXDMA      ERROR_NOTIFIER_TYPE_CTXDMA
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ERROR_NOTIFIER_TYPE_MEMORY      ERROR_NOTIFIER_TYPE_MEMORY
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ECC_ERROR_NOTIFIER_TYPE         5:4
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ECC_ERROR_NOTIFIER_TYPE_UNKNOWN ERROR_NOTIFIER_TYPE_UNKNOWN
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ECC_ERROR_NOTIFIER_TYPE_NONE    ERROR_NOTIFIER_TYPE_NONE
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ECC_ERROR_NOTIFIER_TYPE_CTXDMA  ERROR_NOTIFIER_TYPE_CTXDMA
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_ECC_ERROR_NOTIFIER_TYPE_MEMORY  ERROR_NOTIFIER_TYPE_MEMORY
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_GSP_OWNED                       6:6
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_GSP_OWNED_NO                    0x0
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_GSP_OWNED_YES                   0x1
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_UVM_OWNED                       7:7
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_UVM_OWNED_NO                    0x0
#define NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_UVM_OWNED_YES                   0x1


#define NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN (0xc36f0108)
typedef struct NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS {
    NvU32 workSubmitToken;
} NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS;
typedef struct rpc_rc_triggered_v17_02
{
    NvU32      nv2080EngineType;
    NvU32      chid;
    NvU32      gfid;
    NvU32      exceptLevel;
    NvU32      exceptType;
    NvU32      scope;
    NvU16      partitionAttributionId;
    NvU32      mmuFaultAddrLo;
    NvU32      mmuFaultAddrHi;
    NvU32      mmuFaultType;
    NvBool     bCallbackNeeded;
    NvU32      rcJournalBufferSize;
    NvU8       rcJournalBuffer[];
} rpc_rc_triggered_v17_02;


