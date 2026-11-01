// Controls GSP-RM leaves to the CPU-side RM (it answers them NV_ERR_INVALID_ARGUMENT), answered from what
// GSP-RM reported at start-up
#include "nvos.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl0080/ctrl0080gpu.h"
#include "ctrl/ctrl0080/ctrl0080gr.h"
#include "ctrl/ctrl2080/ctrl2080fb.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"
#include "ctrl/ctrla06f/ctrla06fgpfifo.h"

#include "../chip.h"
#include "../gsp.h"
#include "rmfe_priv.h"

// the classes NVK looks for (nvkmd_nvrm_pdev.c), as GSP-RM r570 provides them: the generic ones, then the chip's
// channel, copy, 3D, 2D, inline-to-memory, compute and usermode classes
static const uint32_t generic_classes[] = {
    0x0000, 0x0080, 0x2080, 0x003e, 0x0040, 0x0071, 0x0079, 0x50a0, 0x90f1, 0xa06c, 0x9067,
};

int rmfe_local_control(struct rmfe *fe, struct client *c, uint32_t object, uint32_t cmd, void *params, uint32_t size,
                       NV_STATUS *st)
{
    const struct gsp_info *info = gsp_get_info(fe->s);
    *st = NV_OK;
    switch (cmd) {
    case NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2: {
        NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS *p = params;
        if (size < sizeof(*p) || p->gpuId != RMFE_GPU_ID) {
            *st = NV_ERR_INVALID_ARGUMENT;
            return 1;
        }
        p->gpuFlags = 0;
        p->deviceInstance = 0;
        p->subDeviceInstance = 0;
        p->sliStatus = 0;
        p->boardId = 0;
        p->gpuInstance = 0;
        p->numaId = ~0u;
        return 1;
    }
    case NV2080_CTRL_CMD_GPU_GET_NAME_STRING: {
        NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS *p = params;
        if (size < sizeof(*p)) {
            *st = NV_ERR_INVALID_ARGUMENT;
            return 1;
        }
        memset(&p->gpuNameString, 0, sizeof(p->gpuNameString));
        memcpy(p->gpuNameString.ascii, info->name, sizeof(p->gpuNameString.ascii) - 1);
        return 1;
    }
    case NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2: {
        NV0080_CTRL_GPU_GET_CLASSLIST_V2_PARAMS *p = params;
        if (size < sizeof(*p)) {
            *st = NV_ERR_INVALID_ARGUMENT;
            return 1;
        }
        const struct chip *k = gsp_gpu(fe->s)->chip;
        const uint32_t chip_classes[] = { k->gpfifo, k->copy, k->eng3d, 0x902d, 0xa140, k->compute, k->usermode };
        p->numClasses = 0;
        for (size_t i = 0; i < sizeof(generic_classes) / sizeof(generic_classes[0]); i++)
            p->classList[p->numClasses++] = generic_classes[i];
        for (size_t i = 0; i < sizeof(chip_classes) / sizeof(chip_classes[0]); i++)
            p->classList[p->numClasses++] = chip_classes[i];
        return 1;
    }
    case NV0080_CTRL_CMD_GR_GET_INFO_V2: {
        NV0080_CTRL_GR_GET_INFO_V2_PARAMS *p = params;
        if (size < sizeof(*p) || p->grInfoListSize > NV0080_CTRL_GR_INFO_MAX_SIZE) {
            *st = NV_ERR_INVALID_ARGUMENT;
            return 1;
        }
        for (uint32_t i = 0; i < p->grInfoListSize; i++) {
            uint32_t idx = p->grInfoList[i].index;
            p->grInfoList[i].data = idx < GSP_GR_INFO_MAX ? info->gr[idx] : 0;
        }
        return 1;
    }
    case NV2080_CTRL_CMD_FB_GET_INFO_V2: {
        NV2080_CTRL_FB_GET_INFO_V2_PARAMS *p = params;
        if (size < sizeof(*p) || p->fbInfoListSize > NV2080_CTRL_FB_INFO_MAX_LIST_SIZE) {
            *st = NV_ERR_INVALID_ARGUMENT;
            return 1;
        }
        for (uint32_t i = 0; i < p->fbInfoListSize; i++) {
            NV2080_CTRL_FB_INFO *e = &p->fbInfoList[i];
            switch (e->index) {
            case NV2080_CTRL_FB_INFO_INDEX_RAM_SIZE:
            case NV2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE:
            case NV2080_CTRL_FB_INFO_INDEX_HEAP_SIZE:
                e->data = (uint32_t)(info->fb_size >> 10);
                break;
            case NV2080_CTRL_FB_INFO_INDEX_BAR1_SIZE:
                e->data = (uint32_t)(bar_size(gsp_gpu(fe->s), 1) >> 10);
                break;
            default:
                e->data = 0;
                break;
            }
        }
        return 1;
    }
    case NVA06F_CTRL_CMD_GPFIFO_SCHEDULE: {
        // r570 takes two bytes, 610 headers send three (bSkipEnable)
        void *rep = rm_control(fe->s, c->handle, object, cmd, params, 2);
        *st = rep ? NV_OK : NV_ERR_INVALID_ARGUMENT;
        return 1;
    }
    default:
        return 0;
    }
}
