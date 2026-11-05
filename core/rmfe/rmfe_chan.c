// Channels: NVK's channel alloc rebuilt as the one kernel RM sends GSP-RM, the CPU-side context buffers for
// graphics, and the controls whose answer lands in client memory (the work submit token)
#include "nvos.h"
#include "nvgputypes.h"
#include "alloc/alloc_channel.h"
#include "class/cl0002.h"
#include "class/cl2080_notification.h"
#include "class/cl9067.h"
#include "class/clb0b5sw.h"
#include "class/cla06c.h"
#include "class/clc56f.h"
#include "ctrl/ctrl2080/ctrl2080ce.h"
#include "ctrl/ctrla06f/ctrla06fgpfifo.h"
#include "ctrl/ctrl906f.h"
#include "ctrl/ctrlc36f.h"

#include "../chip.h"
#include "../gsp.h"
#include "rmfe_priv.h"

#define FIRST_CHID 32  // below are the core's own test channels

static int is_gr_class(const struct chip *k, uint32_t cls)
{
    if (cls == k->eng3d || cls == k->compute)
        return 1;
    switch (cls) {
    case 0x902d:  // FERMI_TWOD_A
    case 0xa140:  // KEPLER_INLINE_TO_MEMORY_B
        return 1;
    default:
        return 0;
    }
}

int client_rm(struct client *c, struct rm_client *rc)
{
    *rc = (struct rm_client){ .client = c->handle };
    for (struct obj *o = c->objs; o; o = o->next) {
        if (o->cls == 0x0080)
            rc->device = o->handle;
        if (o->cls == 0x2080)
            rc->subdevice = o->handle;
    }
    return rc->device && rc->subdevice ? 0 : -1;
}

static int chid_get(struct rmfe *fe)
{
    return gsp_chid_get(fe->s, FIRST_CHID);
}

static void chid_put(struct rmfe *fe, uint32_t chid)
{
    gsp_chid_put(fe->s, chid);
}

static struct vmm *vaspace_of(struct client *c, uint32_t handle)
{
    struct obj *o = obj_find(c, handle);
    return o && (o->kind == K_VASPACE || o->cls == FERMI_CONTEXT_SHARE_A) ? o->vmm : NULL;
}

// physical address of offset in a memory object, and whether it is system memory
static int phys_of(struct obj *mem, uint64_t off, uint64_t *addr, int *sys)
{
    if (mem->kind == K_VRAM && off < mem->size) {
        *addr = mem->addr + off;
        *sys = 0;
        return 0;
    }
    if (mem->kind == K_SYSMEM && off < mem->dma.size) {
        *addr = dma_addr(&mem->dma, off);
        *sys = 1;
        return 0;
    }
    return -1;
}

static uint32_t pce_mask(struct rmfe *fe, const struct rm_client *rc, uint32_t engine)
{
    NV2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS req = { .ceEngineType = engine };
    NV2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS *m =
        rm_control(fe->s, rc->client, rc->subdevice, NV2080_CTRL_CMD_CE_GET_CE_PCE_MASK, &req, sizeof(req));
    return m ? m->pceMask : 0;
}

// GSP-RM takes channels on a logical copy engine with no physical engines behind it (LCE4 on GA106), and
// their work never runs; such a channel goes to an async copy engine that has some
static uint32_t copy_engine(struct rmfe *fe, const struct rm_client *rc, uint32_t engine)
{
    if (engine < NV2080_ENGINE_TYPE_COPY0 || engine > NV2080_ENGINE_TYPE_COPY9 || pce_mask(fe, rc, engine))
        return engine;
    uint32_t alt, runlist;
    if (chan_find_engine(fe->s, rc, CHAN_COPY, &alt, &runlist))
        return engine;
    log_msg("rmfe: copy engine %u has no physical engines, using %u\n", engine - NV2080_ENGINE_TYPE_COPY0,
            alt - NV2080_ENGINE_TYPE_COPY0);
    return alt;
}

NV_STATUS rmfe_alloc_chan(struct rmfe *fe, struct client *c, struct obj *parent, uint32_t handle, uint32_t cls, void *params)
{
    const NV_CHANNEL_ALLOC_PARAMS *p = params;
    struct rm_client rc;
    if (client_rm(c, &rc))
        return NV_ERR_INVALID_STATE;

    struct obj *userd = obj_find(c, p->hUserdMemory[0]);
    struct chan_params cp = {
        .gpfifo_va = p->gpFifoOffset, .entries = p->gpFifoEntries, .engine = copy_engine(fe, &rc, p->engineType),
        .vaspace = p->hVASpace, .ctxshare = p->hContextShare,
        .privileged = DRF_VAL(OS04, _FLAGS, _PRIVILEGED_CHANNEL, p->flags),
    };
    if (!userd || phys_of(userd, p->userdOffset[0], &cp.userd, &cp.userd_sys))
        return NV_ERR_INVALID_OBJECT_HANDLE;
    struct vmm *vm = vaspace_of(c, p->hContextShare ? p->hContextShare : p->hVASpace);
    if (!vm)
        return NV_ERR_INVALID_OBJECT_HANDLE;
    uint32_t runlist = chan_runlist(fe->s, &rc, cp.engine);
    if (runlist == ~0u)
        return NV_ERR_INVALID_ARGUMENT;
    int got = chid_get(fe);
    if (got < 0)
        return NV_ERR_INSUFFICIENT_RESOURCES;
    uint32_t chid = (uint32_t)got;
    cp.chid = chid;

    struct rmfe_chan *ch = mem_alloc(sizeof(*ch));
    if (chan_alloc(fe->s, &rc, parent ? parent->handle : c->handle, handle, &cp, &ch->res)) {
        chid_put(fe, chid);
        mem_free(ch);
        return NV_ERR_INVALID_ARGUMENT;
    }
    ch->chid = chid;
    ch->runlist = runlist;
    ch->engine = cp.engine;
    ch->error_ctxdma = p->hObjectError;
    ch->token_index = NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN;
    struct obj *o = obj_add(c, handle, parent ? parent->handle : c->handle, cls, K_CHAN);
    o->ch = ch;
    o->vmm = vm;
    log_msg("rmfe: channel %#x chid %u runlist %u engine %u, USERD %s %#llx\n", handle, chid, runlist, cp.engine,
            cp.userd_sys ? "sysmem" : "vram", (unsigned long long)cp.userd);
    return NV_OK;
}

void rmfe_release_chan(struct rmfe *fe, struct obj *o)
{
    struct rmfe_chan *ch = o->ch;
    gr_ctx_free(fe->s, &ch->gr);
    chan_res_free(fe->s, &ch->res);
    chid_put(fe, ch->chid);
    mem_free(ch);
}

NV_STATUS rmfe_alloc_engine(struct rmfe *fe, struct client *c, struct obj *chan, uint32_t handle, uint32_t cls, void *params,
                            uint32_t size)
{
    struct rmfe_chan *ch = chan->ch;
    struct rm_client rc;
    if (client_rm(c, &rc))
        return NV_ERR_INVALID_STATE;
    // kernel RM's job before the first graphics object: the channel's context buffers
    if (is_gr_class(gsp_gpu(fe->s)->chip, cls) && ch->engine == 1 && !ch->promoted) {
        if (gr_promote(fe->s, &rc, chan->handle, chan->vmm, 0, &ch->gr))
            return NV_ERR_NO_MEMORY;
        ch->promoted = 1;
    }
    // a copy object on a copy-engine channel runs on that engine; GSP-RM wants to be told which
    NVB0B5_ALLOCATION_PARAMETERS ce = { .version = NVB0B5_ALLOCATION_PARAMETERS_VERSION_1, .engineType = ch->engine };
    if (cls == gsp_gpu(fe->s)->chip->copy && !params && ch->engine != 1) {
        params = &ce;
        size = sizeof(ce);
    }
    if (rm_alloc(fe->s, c->handle, chan->handle, handle, cls, params, params ? size : 0))
        return NV_ERR_INVALID_CLASS;
    obj_add(c, handle, chan->handle, cls, K_GSP);
    return NV_OK;
}

// a slot of the channel's notifier array (the error context DMA), where RM reports to the client
static NvNotification *notifier(struct client *c, struct rmfe_chan *ch, uint32_t index)
{
    struct obj *dma = obj_find(c, ch->error_ctxdma);
    struct obj *mem = dma && dma->kind == K_CTXDMA ? obj_find(c, dma->target) : NULL;
    uint64_t off = dma ? dma->addr + index * sizeof(NvNotification) : 0;
    if (!mem || mem->kind != K_SYSMEM || off + sizeof(NvNotification) > mem->dma.size)
        return NULL;
    return (NvNotification *)((uint8_t *)mem->dma.cpu + off);
}

// RM writes the token into the channel's notifier as well; NVK reads it from there
static void write_notifier(struct client *c, struct rmfe_chan *ch, uint32_t value)
{
    NvNotification *n = notifier(c, ch, ch->token_index);
    if (!n)
        return;
    n->info32 = value;
    n->info16 = 0;
    n->status = 0;
}

void rmfe_rc_event(void *ctx, uint32_t engine, uint32_t chid, uint32_t exception, const struct gsp_fault *fault)
{
    struct rmfe *fe = ctx;
    if (fe->rc_n < sizeof(fe->rc) / sizeof(fe->rc[0]))
        fe->rc[fe->rc_n++] = (struct rc_event){ chid, exception };
}

// what RM does for a recovered channel: the fault into the log, the error into the client's notifier
void rmfe_drain_rc(struct rmfe *fe)
{
    while (fe->rc_n) {
        uint32_t chid = fe->rc[0].chid, exception = fe->rc[0].exception;
        fe->rc_n--;
        memmove(&fe->rc[0], &fe->rc[1], fe->rc_n * sizeof(fe->rc[0]));
        for (struct client *c = fe->clients; c; c = c->next) {
            for (struct obj *o = c->objs; o; o = o->next) {
                if (o->kind != K_CHAN || o->ch->chid != chid)
                    continue;
                NV906F_CTRL_GET_MMU_FAULT_INFO_PARAMS req = { 0 };
                NV906F_CTRL_GET_MMU_FAULT_INFO_PARAMS *fi =
                    rm_control(fe->s, c->handle, o->handle, NV906F_CTRL_CMD_GET_MMU_FAULT_INFO, &req, sizeof(req));
                if (fi)
                    log_msg("rmfe: channel %#x faulted at %#llx, type %#x \"%s\"\n", o->handle,
                            (unsigned long long)fi->addrHi << 32 | fi->addrLo, fi->faultType, fi->faultString);
                NvNotification *n = notifier(c, o->ch, NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR);
                if (n) {
                    n->info32 = exception;
                    n->info16 = 0;
                    n->status = 0xffff;
                }
            }
        }
    }
}

int rmfe_chan_control(struct rmfe *fe, struct client *c, struct obj *o, uint32_t cmd, void *params, uint32_t size,
                      NV_STATUS *st)
{
    struct rmfe_chan *ch = o->ch;
    *st = NV_OK;
    switch (cmd) {
    case NVA06F_CTRL_CMD_BIND: {
        NVA06F_CTRL_BIND_PARAMS bind = { .engineType = ch->engine };  // where the channel actually went
        *st = rm_control(fe->s, c->handle, o->handle, cmd, &bind, sizeof(bind)) ? NV_OK : NV_ERR_INVALID_ARGUMENT;
        return 1;
    }
    case NVC36F_CTRL_CMD_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX: {
        NVC36F_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX_PARAMS *p = params;
        if (size < sizeof(*p))
            *st = NV_ERR_INVALID_ARGUMENT;
        else
            ch->token_index = p->index;
        return 1;
    }
    case NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: {
        NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS *p = params;
        if (size < sizeof(*p)) {
            *st = NV_ERR_INVALID_ARGUMENT;
            return 1;
        }
        // GSP-RM's token only carries the channel; the doorbell wants the runlist too
        p->workSubmitToken = ch->runlist << 16 | ch->chid;
        write_notifier(c, ch, p->workSubmitToken);
        return 1;
    }
    default:
        return 0;
    }
}
