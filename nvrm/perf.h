// SPDX-License-Identifier: MIT
// Performance controls of GSP-RM, from NVIDIA's MIT headers: the P-state, boost and aggressive-P-state controls of
// open-gpu-kernel-modules 570.144 (ctrl2080perf.h), the clock domain samples of the 610 SDK (ctrl2080clk.h,
// ctrl2080pmumon.h)
#pragma once

#include "nvtypes.h"


#define NV2080_CTRL_CMD_PERF_BOOST                    (0x2080200a)
#define NV2080_CTRL_PERF_BOOST_FLAGS_CMD_CLEAR        (0x00000000)

typedef struct NV2080_CTRL_PERF_BOOST_PARAMS {
    NvU32 flags;
    NvU32 duration;
} NV2080_CTRL_PERF_BOOST_PARAMS;

// the P-state as a mask of NV2080_CTRL_PERF_PSTATES_P0 (1) to P15 (0x8000)
#define NV2080_CTRL_CMD_PERF_GET_CURRENT_PSTATE (0x20802068)

typedef struct NV2080_CTRL_PERF_GET_CURRENT_PSTATE_PARAMS {
    NvU32 currPstate;
} NV2080_CTRL_PERF_GET_CURRENT_PSTATE_PARAMS;

// bGpuIsIdle caps the GPU to its lowest P-state and GPCCLK, false lifts it; busy/idle times are since the last call
#define NV2080_CTRL_CMD_PERF_AGGRESSIVE_PSTATE_NOTIFY (0x2080208f)

typedef struct NV2080_CTRL_PERF_AGGRESSIVE_PSTATE_NOTIFY_PARAMS {
    NvBool bGpuIsIdle;
    NvBool bRestoreToMax;
    NV_DECLARE_ALIGNED(NvU64 idleTimeUs, 8);
    NV_DECLARE_ALIGNED(NvU64 busyTimeUs, 8);
} NV2080_CTRL_PERF_AGGRESSIVE_PSTATE_NOTIFY_PARAMS;

#define NV2080_CTRL_CMD_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES (0x20801037)
#define NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE_COUNT    (250U)

#define NV2080_CTRL_PMUMON_RETRIEVAL_MODE_LATEST_SAMPLE_ONLY 1

typedef struct NV2080_CTRL_PMUMON_GET_SAMPLES_SUPER {
    NvU32 tailIndex;
    NvU32 sequenceId;
    NvU32 headIndex;
    NvU32 numSamples;
    NvU32 resetCount;
    NvU32 retrievalMode;
} NV2080_CTRL_PMUMON_GET_SAMPLES_SUPER;

// frequencies in kHz, NV_U32_MAX where the chip has no such clock
typedef struct NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE {
    NV_DECLARE_ALIGNED(NvU64 timestamp, 8);
    NvU32 gpcClkFreqKHz;
    NvU32 dramClkFreqKHz;
    NvU32 nvdClkFreqKHz;
    NvU32 activeGpcClkFreqKHz;
} NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE;

typedef struct NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS {
    NV2080_CTRL_PMUMON_GET_SAMPLES_SUPER super;
    NV_DECLARE_ALIGNED(NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE samples[NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_SAMPLE_COUNT], 8);
} NV2080_CTRL_CLK_PMUMON_CLK_DOMAINS_GET_SAMPLES_PARAMS;
