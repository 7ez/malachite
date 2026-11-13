// A graphics channel through GSP-RM: context buffers allocated by the client and promoted to GSP-RM, the
// 3D class allocated to trigger golden context init, following nouveau's rm/r535/gr.c
#include "chip.h"
#include "../nvrm/gr.h"
#include "../nvrm/gsp.h"
#include "chan.h"
#include "gr.h"
#include "gsp.h"
#include "mem.h"

#define PAGE 0x1000ull

static uint64_t pow2_ge(uint64_t v)
{
    uint64_t p = 1;
    while (p < v)
        p <<= 1;
    return p;
}

// which of GSP-RM's context buffers the client allocates, per nouveau's r535_gr_get_ctxbuf_info()
static int ctxbuf_list(struct gsp *s, struct ctxbuf *list)
{
    static const struct {
        uint32_t engine_id, buffer_id;
        int global, init, ro;
    } map[] = {
        { NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS, NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_MAIN, 0, 1, 0 },
        { NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH, NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PATCH, 0, 1, 0 },
        { NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_BUNDLE_CB, NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_BUFFER_BUNDLE_CB, 1, 0, 0 },
        { NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PAGEPOOL, NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PAGEPOOL, 1, 0, 0 },
        { NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_ATTRIBUTE_CB, NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_ATTRIBUTE_CB, 1, 0, 0 },
        { NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_RTV_CB_GLOBAL, NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_RTV_CB_GLOBAL, 1, 0, 0 },
        { NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_FECS_EVENT, NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_FECS_EVENT, 1, 1, 0 },
        { NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PRIV_ACCESS_MAP, NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PRIV_ACCESS_MAP, 1, 1, 1 },
    };

    const struct rm_client *in = gsp_internal(s);
    NV2080_CTRL_INTERNAL_STATIC_GR_GET_CONTEXT_BUFFERS_INFO_PARAMS *req = mem_alloc(sizeof(*req));
    NV2080_CTRL_INTERNAL_STATIC_GR_GET_CONTEXT_BUFFERS_INFO_PARAMS *info =
        rm_control(s, in->client, in->subdevice, NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO, req, sizeof(*req));
    mem_free(req);
    if (!info)
        return -1;

    int n = 0;
    for (unsigned m = 0; m < sizeof(map) / sizeof(map[0]); m++) {
        uint64_t size = info->engineContextBuffersInfo[0].engine[map[m].engine_id].size;
        if (map[m].buffer_id == NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_MAIN)
            size = ((size + PAGE - 1) & ~(PAGE - 1)) + 64 * PAGE;  // per-subcontext headers
        uint64_t page = size >= (1 << 21) ? 1 << 21 : size >= (1 << 16) ? 1 << 16 : PAGE;
        list[n] = (struct ctxbuf){
            .id = map[m].buffer_id,
            .size = (size + page - 1) & ~(page - 1),
            .align = map[m].buffer_id == NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_ATTRIBUTE_CB ? pow2_ge(size) : page,
            .global = map[m].global, .init = map[m].init, .ro = map[m].ro,
        };
        n++;
        if (map[m].buffer_id == NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PRIV_ACCESS_MAP) {
            list[n] = list[n - 1];
            list[n].id = NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_UNRESTRICTED_PRIV_ACCESS_MAP;
            n++;
        }
    }
    return n;
}

struct gr_global {
    struct ctxbuf bufs[GR_MAX_CTXBUF];  // the global ones; VRAM only
    int n;
};

static int promote(struct gsp *s, const struct rm_client *c, uint32_t channel, struct vmm *vm, int golden, int user,
                   struct gr_ctx *ctx)
{
    struct gr_global *gl = *gsp_gr(s);
    *ctx = (struct gr_ctx){ .vm = vm };
    struct ctxbuf list[GR_MAX_CTXBUF];
    int n = ctxbuf_list(s, list);
    if (n < 0)
        return -1;

    NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS *p = mem_alloc(sizeof(*p));
    p->engineType = 1;  // GR0
    p->hChanClient = c->client;
    p->hObject = channel;
    for (int i = 0; i < n; i++) {
        struct ctxbuf *b = &ctx->bufs[ctx->n];
        *b = list[i];
        // past golden init the global buffers are the ones it got, already initialised
        int alloc = golden || !b->global;
        if (!alloc) {
            // one access map per channel: unrestricted for admin, restricted for user (kgrctxGetRegisterAccessMapId)
            if (b->id == (user ? NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_UNRESTRICTED_PRIV_ACCESS_MAP
                               : NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PRIV_ACCESS_MAP))
                continue;
            for (int g = 0; g < gl->n; g++)
                if (gl->bufs[g].id == b->id)
                    b->vram = gl->bufs[g].vram;
            b->shared = 1;
        } else {
            b->vram = vram_alloc(gsp_vram(s), b->size, b->align);
            if (!b->vram)
                goto fail;
            if (b->init)
                vram_zero(gsp_gpu(s), b->vram, b->size);
        }
        ctx->n++;

        NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY *e = &p->promoteEntry[p->entryCount++];
        e->bufferId = b->id;
        e->bInitialize = b->init && alloc;
        if (alloc && b->id == NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PRIV_ACCESS_MAP) {
            e->bNonmapped = 1;
        } else {
            b->va = vmm_alloc(vm, b->size, b->align);
            if (!b->va || vmm_map_vram(vm, b->va, b->vram, b->size, VMM_PRIV | (b->ro ? VMM_RO : 0)))
                goto fail;
            e->gpuVirtAddr = b->va;
        }
        if (e->bInitialize) {
            e->gpuPhysAddr = b->vram;
            e->size = b->size;
            e->physAttr = 4;
        }
    }
    void *ok = rm_control(s, c->client, c->subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, p, sizeof(*p));
    mem_free(p);
    if (!ok) {
        gr_ctx_free(s, ctx);
        return -1;
    }
    vmm_flush(gsp_gpu(s));
    return 0;
fail:
    mem_free(p);
    gr_ctx_free(s, ctx);
    return -1;
}

// once per GSP-RM session: allocating the 3D class makes GSP-RM build and cache the golden context, and the global
// buffers stay for later channels. Redone on them after a resume, which loses it (kernel RM redoes it on every
// kgraphicsStateLoad).
static int golden(struct gsp *s)
{
    struct rm_client c;
    struct vmm vm = { 0 };
    struct chan ch = { 0 };
    struct gr_ctx ctx = { 0 };
    uint32_t engine, runlist;
    int first = !*gsp_gr(s);
    int ret = -1;
    if (rm_client(s, &c) || vmm_create(s, &c, 0x90f10000, 0, &vm) ||
        chan_find_engine(s, &c, CHAN_GRAPHICS, &engine, &runlist) ||
        chan_create(s, &c, &vm, engine, runlist, 3, 0xc56f0000, 1, &ch) || chan_schedule(&ch) ||
        promote(s, &c, ch.handle, &vm, first, 0, &ctx))
        goto out;
    if (rm_alloc(s, c.client, ch.handle, 0xc7970000, gsp_gpu(s)->chip->eng3d, NULL, 0))
        goto out;

    ret = 0;
    if (!first) {
        log_msg("gr: golden context made again\n");
        goto out;
    }
    struct gr_global *gl = mem_alloc(sizeof(*gl));
    for (int i = 0; i < ctx.n; i++) {
        if (ctx.bufs[i].global) {
            gl->bufs[gl->n++] = ctx.bufs[i];
            ctx.bufs[i].shared = 1;  // not freed with the golden channel
        }
    }
    *gsp_gr(s) = gl;
    log_msg("gr: golden context initialised, %d global context buffers\n", gl->n);
out:
    rm_free(s, c.client, c.client);
    gr_ctx_free(s, &ctx);
    chan_free(&ch);
    vmm_destroy(&vm);
    return ret;
}

int gr_promote(struct gsp *s, const struct rm_client *c, uint32_t channel, struct vmm *vm, int user, struct gr_ctx *ctx)
{
    if (!*gsp_gr(s) && golden(s))
        return -1;
    return promote(s, c, channel, vm, 0, user, ctx);
}

int gr_resume(struct gsp *s)
{
    return *gsp_gr(s) ? golden(s) : 0;
}

void gr_ctx_free(struct gsp *s, struct gr_ctx *ctx)
{
    for (int i = 0; i < ctx->n; i++) {
        struct ctxbuf *b = &ctx->bufs[i];
        if (b->va) {
            vmm_unmap(ctx->vm, b->va, b->size);
            vmm_free(ctx->vm, b->va, b->size);
        }
        if (b->vram && !b->shared)
            vram_free(gsp_vram(s), b->vram, b->size);
    }
    if (ctx->n)
        vmm_flush(gsp_gpu(s));
    ctx->n = 0;
}

void gr_forget(struct gsp *s)
{
    struct gr_global *gl = *gsp_gr(s);
    if (!gl)
        return;
    for (int i = 0; i < gl->n; i++)
        vram_free(gsp_vram(s), gl->bufs[i].vram, gl->bufs[i].size);
    mem_free(gl);
    *gsp_gr(s) = NULL;
}
