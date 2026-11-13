#pragma once
// NVC372_CTRL_CMD_IS_MODE_POSSIBLE of GSP-RM 570, as in ctrlc372chnc.h
#include "nvtypes.h"

#define NVC372_DISPLAY_SW 0xc372u
#define NVC372_CTRL_CMD_IS_MODE_POSSIBLE 0xc3720101u
#define NVC372_CTRL_MAX_POSSIBLE_HEADS 8
#define NVC372_CTRL_MAX_POSSIBLE_WINDOWS 32
#define NVC372_CTRL_MAX_POSSIBLE_TILES 8
#define NVC372_CTRL_IMP_LUT_USAGE_1025 2
#define NVC372_CTRL_FORMAT_RGB_PACKED_1_BPP 0x1
#define NVC372_CTRL_FORMAT_RGB_PACKED_2_BPP 0x2
#define NVC372_CTRL_FORMAT_RGB_PACKED_4_BPP 0x4
#define NVC372_CTRL_FORMAT_RGB_PACKED_8_BPP 0x8
#define NV_DISP_LOCK_PIN_UNSPECIFIED 0x10

typedef struct {
    NvU8 headIndex;
    NvU32 maxPixelClkKHz;
    struct {
        NvU32 width, height;
    } rasterSize;
    struct {
        NvU32 X, Y;
    } rasterBlankStart, rasterBlankEnd;
    struct {
        NvU32 yStart, yEnd;
    } rasterVertBlank2;
    struct {
        NvU32 masterLockMode, masterLockPin, slaveLockMode, slaveLockPin;
    } control;
    NvU32 maxDownscaleFactorH, maxDownscaleFactorV;
    NvU8 outputScalerVerticalTaps;
    NvBool bUpscalingAllowedV, bOverfetchEnabled, bLtmAllowed;
    struct {
        NvU16 leadingRasterLines, trailingRasterLines;
    } minFrameIdle;
    NvU8 lut, cursorSize32p, tileMask;
    NvBool bEnableDsc;
    NvU16 dscTargetBppX16;
    NvU32 possibleDscSliceCountMask, maxDscSliceWidth;
    NvBool bYUV420Format, bIs2Head1Or, bGetOSLDOutput, bDisableMidFrameAndDWCFWatermark;
} NVC372_CTRL_IMP_HEAD;

typedef struct {
    NvU32 windowIndex, owningHead, formatUsageBound, rotatedFormatUsageBound, maxPixelsFetchedPerLine;
    NvU32 maxDownscaleFactorH, maxDownscaleFactorV;
    NvU8 inputScalerVerticalTaps;
    NvBool bUpscalingAllowedV, bOverfetchEnabled;
    NvU8 lut, tmoLut, surfaceLayout;
} NVC372_CTRL_IMP_WINDOW;

typedef struct {
    NvU32 subdeviceIndex;
    NvU8 numHeads, numWindows;
    NVC372_CTRL_IMP_HEAD head[NVC372_CTRL_MAX_POSSIBLE_HEADS];
    NVC372_CTRL_IMP_WINDOW window[NVC372_CTRL_MAX_POSSIBLE_WINDOWS];
    NvU32 options, testMclkFreqKHz;
    NvBool bIsPossible;
    NvBool bIsOSLDPossible[NVC372_CTRL_MAX_POSSIBLE_HEADS];
    NvU32 minImpVPState, minPState, minRequiredBandwidthKBPS, floorBandwidthKBPS, minRequiredHubclkKHz;
    NvU32 vblankIncreaseInLinesForOSLDMode[NVC372_CTRL_MAX_POSSIBLE_HEADS];
    NvU32 wakeUpRgLineForOSLDMode[NVC372_CTRL_MAX_POSSIBLE_HEADS];
    NvU32 worstCaseMargin, dispClkKHz, numTilingAssignments;
    NvU8 tilingAssignments[NVC372_CTRL_MAX_POSSIBLE_TILES];
    struct {
        NvU8 head, headDscSlices;
    } tileList[NVC372_CTRL_MAX_POSSIBLE_TILES];
    char worstCaseDomain[8];
    NvBool bUseCachedPerfState;
} NVC372_CTRL_IS_MODE_POSSIBLE_PARAMS;
