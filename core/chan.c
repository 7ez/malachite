// GPFIFO channels through GSP-RM, following nouveau's GSP-mode FIFO (rm/r535/fifo.c, rm/r570/fifo.c)
#include "chip.h"
#include "../nvrm/engine.h"
#include "../nvrm/fifo.h"
#include "../nvrm/gsp.h"
#include "chan.h"
#include "gsp.h"
#include "mem.h"

#define DRF_LO(f) (0 ? f)
#define NVVAL(d, r, f, v) ((uint32_t)(v) << DRF_LO(d##_##r##_##f))

#define USERMODE 0xbb0000  // GA10x usermode block under GSP-RM, 0xb80000 + 0x30000
#define USERD_GP_GET 0x88
#define USERD_GP_PUT 0x8c

#define PAGE 0x1000ull
#define CHAN_BUF (GPFIFO_BYTES + PUSH_SIZE)

#define CE_GET_CE_PCE_MASK 0x20802a02  // NV2080_CTRL_CMD_CE_GET_CE_PCE_MASK
struct pce_mask {
    uint32_t engine, mask;
};

int chan_find_engine(struct gsp *s, const struct rm_client *c, uint32_t want, uint32_t *engine, uint32_t *runlist)
{
    NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS *req = mem_alloc(sizeof(*req));
    NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS *t =
        rm_control(s, c->client, c->subdevice, NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE, req, sizeof(*req));
    mem_free(req);
    if (!t)
        return -1;

    uint32_t gr_runlist = ~0u;
    for (uint32_t i = 0; i < t->numEntries; i++)
        if (t->entries[i].engineData[ENGINE_INFO_TYPE_RM_ENGINE_TYPE] == RM_ENGINE_TYPE_GR0)
            gr_runlist = t->entries[i].engineData[ENGINE_INFO_TYPE_RUNLIST];

    uint32_t ce[10], ce_runlist[10], n = 0;
    for (uint32_t i = 0; i < t->numEntries; i++) {
        uint32_t type = t->entries[i].engineData[ENGINE_INFO_TYPE_RM_ENGINE_TYPE];
        uint32_t rl = t->entries[i].engineData[ENGINE_INFO_TYPE_RUNLIST];
        if (want == CHAN_GRAPHICS && type == RM_ENGINE_TYPE_GR0) {
            *engine = type;
            *runlist = rl;
            return 0;
        }
        // copy engines sharing the graphics runlist don't run as async copy engines
        if (want == CHAN_COPY && type - RM_ENGINE_TYPE_COPY0 < 10 && rl != gr_runlist && n < 10) {
            ce[n] = type;
            ce_runlist[n++] = rl;
        }
    }
    // GSP-RM lists logical copy engines with no physical ones, whose work never runs, and which differs per chip
    uint32_t pick = 0;
    for (uint32_t i = 0; i < n; i++) {
        struct pce_mask q = { ce[i], 0 };
        const struct pce_mask *m = rm_control(s, c->client, c->subdevice, CE_GET_CE_PCE_MASK, &q, sizeof(q));
        if (!m || m->mask) {
            pick = i;
            break;
        }
    }
    if (n) {
        *engine = ce[pick];
        *runlist = ce_runlist[pick];
        return 0;
    }
    log_msg("chan: no %s engine\n", want == CHAN_COPY ? "async copy" : "graphics");
    return -1;
}

uint32_t chan_runlist(struct gsp *s, const struct rm_client *c, uint32_t engine)
{
    NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS *req = mem_alloc(sizeof(*req));
    NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS *t =
        rm_control(s, c->client, c->subdevice, NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE, req, sizeof(*req));
    mem_free(req);
    for (uint32_t i = 0; t && i < t->numEntries; i++)
        if (t->entries[i].engineData[ENGINE_INFO_TYPE_RM_ENGINE_TYPE] == engine)
            return t->entries[i].engineData[ENGINE_INFO_TYPE_RUNLIST];
    return ~0u;
}

// The device info table uses RM engine numbers, alloc and BIND NV2080 ones; they differ from the video engines on.
static uint32_t nv2080_engine(uint32_t rm)
{
    static const struct { uint32_t rm, nv2080, count; } ranges[] = {
        { RM_ENGINE_TYPE_NVDEC0, NV2080_ENGINE_TYPE_NVDEC0, 8 },
        { RM_ENGINE_TYPE_NVENC0, NV2080_ENGINE_TYPE_NVENC0, 3 },
        { RM_ENGINE_TYPE_NVJPEG0, NV2080_ENGINE_TYPE_NVJPEG0, 8 },
        { RM_ENGINE_TYPE_OFA0, NV2080_ENGINE_TYPE_OFA0, 1 },
    };
    for (unsigned i = 0; i < sizeof(ranges) / sizeof(ranges[0]); i++)
        if (rm - ranges[i].rm < ranges[i].count)
            return ranges[i].nv2080 + (rm - ranges[i].rm);
    return rm;
}

// as kernel RM allocates it: instance block and RAMFC in VRAM, method buffer in system memory
int chan_alloc(struct gsp *s, const struct rm_client *c, uint32_t parent, uint32_t handle, const struct chan_params *cp,
               struct chan_res *res)
{
    struct gpu *g = gsp_gpu(s);
    struct vram *vr = gsp_vram(s);
    *res = (struct chan_res){ 0 };

    NV2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS mreq = { 0 };
    NV2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS *m =
        rm_control(s, c->client, c->subdevice, NV2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE, &mreq, sizeof(mreq));
    if (!m)
        return -1;
    uint32_t mthd_size = m->size;

    res->inst = vram_alloc(vr, PAGE, PAGE);
    if (!res->inst)
        return -1;
    if (dma_alloc(g, &res->mthd, mthd_size, 1)) {
        chan_res_free(s, res);
        return -1;
    }
    vram_zero(g, res->inst, PAGE);

    NV_CHANNEL_ALLOC_PARAMS *a = mem_alloc(sizeof(*a));
    a->gpFifoOffset = cp->gpfifo_va;
    a->gpFifoEntries = cp->entries;
    a->flags = NVVAL(NVOS04, FLAGS, PRIVILEGED_CHANNEL, cp->privileged) |
               NVVAL(NVOS04, FLAGS, CHANNEL_USERD_INDEX_VALUE, cp->chid % 8) |
               NVVAL(NVOS04, FLAGS, CHANNEL_USERD_INDEX_PAGE_VALUE, cp->chid / 8) |
               NVVAL(NVOS04, FLAGS, CHANNEL_USERD_INDEX_PAGE_FIXED, 1);
    a->hVASpace = cp->vaspace;
    a->hContextShare = cp->ctxshare;
    a->engineType = nv2080_engine(cp->engine);
    a->instanceMem = (NV_MEMORY_DESC_PARAMS){ res->inst, PAGE, 2, 1 };
    a->userdMem = (NV_MEMORY_DESC_PARAMS){ cp->userd, 0x200, cp->userd_sys ? 1 : 2, 1 };
    a->ramfcMem = (NV_MEMORY_DESC_PARAMS){ res->inst, 0x200, 2, 1 };
    a->mthdbufMem = (NV_MEMORY_DESC_PARAMS){ res->mthd.addr, mthd_size, 1, 0 };
    // admin or user privilege (NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_PRIVILEGE), no error notifiers
    a->internalFlags = (cp->privileged ? 1 : 0) | (1 << 2) | (1 << 4);
    int ret = rm_alloc(s, c->client, parent, handle, gsp_gpu(s)->chip->gpfifo, a, sizeof(*a));
    mem_free(a);
    if (ret) {
        chan_res_free(s, res);
        return -1;
    }
    return 0;
}

void chan_res_free(struct gsp *s, struct chan_res *res)
{
    if (res->inst)
        vram_free(gsp_vram(s), res->inst, PAGE);
    if (res->mthd.cpu)
        dma_free(gsp_gpu(s), &res->mthd);
    *res = (struct chan_res){ 0 };
}

int chan_create(struct gsp *s, const struct rm_client *c, struct vmm *vm, uint32_t engine, uint32_t runlist,
                uint32_t chid, uint32_t handle, int privileged, struct chan *ch)
{
    struct gpu *g = gsp_gpu(s);
    struct vram *vr = gsp_vram(s);
    *ch = (struct chan){ .s = s, .client = c->client, .handle = handle, .chid = chid, .runlist = runlist, .vm = vm };

    uint64_t userd_page = vram_alloc(vr, PAGE, PAGE);
    ch->userd = userd_page ? userd_page + (chid % 8) * 0x200 : 0;
    // not contiguous, so a rebuild after a fault can't fail for want of 528 KB in one piece
    ch->buf = gsp_dma(s, CHAN_BUF, 0);
    ch->va = vmm_alloc(vm, CHAN_BUF, PAGE);
    // read-only so a user channel's shaders can't rewrite their own methods
    if (!userd_page || !ch->buf || !ch->va ||
        vmm_map_sys(vm, ch->va, ch->buf, 0, CHAN_BUF, privileged == CHAN_KERNEL ? 0 : VMM_RO)) {
        chan_free(ch);
        return -1;
    }
    vram_zero(g, userd_page, PAGE);

    struct chan_params cp = { .gpfifo_va = ch->va, .entries = GPFIFO_ENTRIES, .privileged = privileged != 0, .vaspace = vm->handle,
                              .engine = engine, .chid = chid, .userd = ch->userd };
    if (chan_alloc(s, c, c->device, handle, &cp, &ch->res)) {
        chan_free(ch);
        return -1;
    }

    NVA06F_CTRL_BIND_PARAMS bind = { .engineType = nv2080_engine(engine) };
    if (!rm_control(s, c->client, handle, NVA06F_CTRL_CMD_BIND, &bind, sizeof(bind))) {
        rm_free(s, c->client, handle);
        chan_free(ch);
        return -1;
    }
    return 0;
}

int chan_schedule(struct chan *ch)
{
    NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS sched = { .bEnable = 1 };
    return rm_control(ch->s, ch->client, ch->handle, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, &sched, sizeof(sched)) ? 0 : -1;
}

int chan_half(const struct chan *ch, uint32_t bytes)
{
    uint32_t end = ch->push_off + bytes > PUSH_SIZE ? bytes : ch->push_off + bytes;
    return (end - 1) / (PUSH_SIZE / 2);
}

uint32_t *chan_reserve(struct chan *ch, uint32_t n)
{
    // wraps without waiting; submitters of unwaited work wait first (chan_half)
    if (ch->push_off + n * 4 > PUSH_SIZE)
        ch->push_off = 0;
    return (uint32_t *)((uint8_t *)ch->buf->cpu + GPFIFO_BYTES + ch->push_off);
}

void chan_entry(struct chan *ch, uint64_t va, uint32_t words)
{
    uint32_t *gp = (uint32_t *)ch->buf->cpu + ch->gp_put * 2;
    gp[0] = (uint32_t)va & ~3u;
    gp[1] = (uint32_t)(va >> 32) | (words << 10);
    ch->gp_put = (ch->gp_put + 1) % GPFIFO_ENTRIES;
}

void chan_commit(struct chan *ch, uint32_t n)
{
    uint64_t pbva = ch->va + GPFIFO_BYTES + ch->push_off;
    ch->push_off += (n * 4 + 0xff) & ~0xffu;
    chan_entry(ch, pbva, n);
}

void chan_push(struct chan *ch, const uint32_t *push, uint32_t n)
{
    memcpy(chan_reserve(ch, n), push, n * 4);
    chan_commit(ch, n);
}

void chan_kick(struct chan *ch)
{
    struct gpu *g = gsp_gpu(ch->s);
    __sync_synchronize();
    vram_wr32(g, ch->userd + USERD_GP_PUT, ch->gp_put);
    rd32(g, 0x0);  // flush the PRAMIN write before ringing
    // the doorbell needs the runlist too, which GSP-RM's work submit token lacks
    wr32(g, USERMODE + 0x90, (ch->runlist << 16) | ch->chid);
}

void chan_submit(struct chan *ch, const uint32_t *push, uint32_t n)
{
    chan_push(ch, push, n);
    chan_kick(ch);
}

uint32_t chan_save(const struct chan *ch, uint32_t gp, uint32_t *out)
{
    const uint8_t *cpu = ch->buf->cpu;
    uint32_t n = 0;
    for (; gp != ch->gp_put; gp = (gp + 1) % GPFIFO_ENTRIES) {
        const uint32_t *e = (const uint32_t *)cpu + gp * 2;
        uint64_t va = e[0] | (uint64_t)(e[1] & 0xff) << 32;
        uint32_t len = e[1] >> 10;
        // user memory entries (chan_entry) are deltas on this channel's state, useless on a new one
        if (va < ch->va || va >= ch->va + CHAN_BUF)
            continue;
        out[n++] = len;
        memcpy(out + n, cpu + (va - ch->va), len * 4);
        n += len;
    }
    return n;
}

int chan_replay(struct chan *ch, const uint32_t *saved, uint32_t n)
{
    uint32_t bytes = ch->push_off;
    for (uint32_t i = 0; i < n; i += 1 + saved[i])
        bytes += (saved[i] * 4 + 0xff) & ~0xffu;
    if (bytes > PUSH_SIZE)
        return -1;
    for (uint32_t i = 0; i < n; i += 1 + saved[i])
        chan_push(ch, saved + i + 1, saved[i]);
    chan_kick(ch);
    return 0;
}

uint32_t chan_gp_get(struct chan *ch)
{
    return vram_rd32(gsp_gpu(ch->s), ch->userd + USERD_GP_GET);
}

void chan_free(struct chan *ch)
{
    if (!ch->s)
        return;
    chan_res_free(ch->s, &ch->res);
    if (ch->userd) {
        vram_free(gsp_vram(ch->s), ch->userd & ~(PAGE - 1), PAGE);
        ch->userd = 0;
    }
    if (ch->va) {
        vmm_unmap(ch->vm, ch->va, CHAN_BUF);
        vmm_free(ch->vm, ch->va, CHAN_BUF);
        ch->va = 0;
    }
    if (ch->buf) {
        gsp_dma_free(ch->s, ch->buf);
        ch->buf = NULL;
    }
}

int wait_word(volatile uint32_t *word, uint32_t want, int ms)
{
    uint64_t end = time_us() + (uint64_t)ms * 1000;
    while (*word != want) {
        if (time_us() > end)
            return -1;
        delay_us(10);
    }
    return 0;
}

// The semaphore is never reset, as resetting it before submit races the GPU's release; payload must be new.
int chan_copy2d(struct chan *ch, uint64_t src, uint32_t src_pitch, uint64_t dst, uint32_t dst_pitch, uint32_t bytes,
                uint32_t lines, volatile uint32_t *sem, uint64_t semva, uint32_t payload)
{
    struct copy_surface s = { src, src_pitch, 0, 0 }, d = { dst, dst_pitch, 0, 0 };
    return chan_copy_surface(ch, &s, &d, bytes, lines, 0, sem, semva, payload);
}

int chan_copy_surface(struct chan *ch, const struct copy_surface *src, const struct copy_surface *dst,
                      uint32_t bytes, uint32_t lines, uint32_t remap, volatile uint32_t *sem, uint64_t semva,
                      uint32_t payload)
{
    chan_copy_surface_queue(ch, src, dst, bytes, lines, remap, semva, payload);
    if (wait_word(sem, payload, 2000)) {
        log_msg("chan_copy: did not complete (GP_GET %u)\n", chan_gp_get(ch));
        return -1;
    }
    return 0;
}

void chan_copy_surface_queue(struct chan *ch, const struct copy_surface *src, const struct copy_surface *dst,
                             uint32_t bytes, uint32_t lines, uint32_t remap, uint64_t semva, uint32_t payload)
{
    // with remapping, line length and block-linear width and origin x count elements, not bytes
    uint32_t comp = (remap >> 16 & 3) + 1;
    uint32_t elem[2] = { remap ? comp * ((remap >> 24 & 3) + 1) : 1, remap ? comp * ((remap >> 20 & 3) + 1) : 1 };
    uint32_t push[64];
    unsigned n = 0;
    uint32_t head[] = {
        MTHD(4, 0x000, 1), gsp_gpu(ch->s)->chip->copy,
        // OFFSET_IN, OFFSET_OUT, PITCH_IN, PITCH_OUT, LINE_LENGTH_IN, LINE_COUNT
        MTHD(4, 0x400, 8), (uint32_t)(src->addr >> 32), (uint32_t)src->addr, (uint32_t)(dst->addr >> 32),
        (uint32_t)dst->addr, src->pitch, dst->pitch, bytes / elem[1], lines,
    };
    memcpy(push, head, sizeof(head));
    n += sizeof(head) / 4;
    // one-GOB blocks of 8-row GOBs
    const struct copy_surface *side[2] = { dst, src };
    const uint32_t base[2] = { 0x70c, 0x728 };  // SET_DST_BLOCK_SIZE, SET_SRC_BLOCK_SIZE
    for (int i = 0; i < 2; i++) {
        if (!side[i]->blocklinear)
            continue;
        uint32_t origin = (side[i]->origin & 0xffff) / elem[i] | (side[i]->origin & 0xffff0000);
        uint32_t bl[] = { MTHD(4, base[i], 6), 1u << 12, side[i]->pitch / elem[i], side[i]->rows, 1, 0, origin };
        memcpy(push + n, bl, sizeof(bl));
        n += sizeof(bl) / 4;
    }
    if (remap) {  // SET_REMAP_CONST_A and B, then SET_REMAP_COMPONENTS
        uint32_t r[] = { MTHD(4, 0x700, 3), 0, 0, remap };
        memcpy(push + n, r, sizeof(r));
        n += sizeof(r) / 4;
    }
    uint32_t tail[] = {
        MTHD(4, 0x240, 3), (uint32_t)(semva >> 32), (uint32_t)semva, payload,
        // non-pipelined, flush, release, pitch layouts, multi-line, remap
        MTHD(4, 0x300, 1), 2 | (1 << 2) | (1 << 3) | (src->blocklinear ? 0 : 1 << 7) |
                               (dst->blocklinear ? 0 : 1 << 8) | (lines > 1 ? 1 << 9 : 0) | (remap ? 1 << 10 : 0),
    };
    memcpy(push + n, tail, sizeof(tail));
    n += sizeof(tail) / 4;
    chan_submit(ch, push, n);
}

void chan_release_queue(struct chan *ch, uint64_t semva, uint32_t payload)
{
    uint32_t push[] = {
        MTHD(4, 0x000, 1), gsp_gpu(ch->s)->chip->copy,
        MTHD(4, 0x240, 3), (uint32_t)(semva >> 32), (uint32_t)semva, payload,
        MTHD(4, 0x300, 1), 1 << 2 | 1 << 3,  // no transfer; flush, then release one word
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

// clc56f.h, used as uvm_hal_ampere_host_l2_invalidate does with membars around them; admin channels only
#define MEM_OP_A 0x28
#define MEM_OP_MEMBAR (0x5u << 27)
#define MEM_OP_L2_SYSMEM_INVALIDATE (0xeu << 27)
#define MEM_OP_L2_FLUSH_DIRTY (0x10u << 27)

void chan_l2_sysmem(struct chan *ch)
{
    uint32_t push[] = {
        MTHD(0, MEM_OP_A, 4), 0, 0, 0, MEM_OP_MEMBAR,
        MTHD(0, MEM_OP_A, 4), 0, 0, 0, MEM_OP_L2_FLUSH_DIRTY,
        MTHD(0, MEM_OP_A, 4), 0, 0, 0, MEM_OP_MEMBAR,
        MTHD(0, MEM_OP_A, 4), 0, 0, 0, MEM_OP_L2_SYSMEM_INVALIDATE,
        MTHD(0, MEM_OP_A, 4), 0, 0, 0, MEM_OP_MEMBAR,
    };
    chan_push(ch, push, sizeof(push) / 4);
}

void chan_acquire_queue_switch(struct chan *ch, uint64_t semva, uint32_t payload)
{
    uint32_t push[] = {
        MTHD(0, 0x5c, 5), (uint32_t)semva, (uint32_t)(semva >> 32), payload, 0, 3 | 1u << 12,
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

void chan_acquire_queue(struct chan *ch, uint64_t semva, uint32_t payload)
{
    // SEM_ADDR_LO..SEM_EXECUTE, acquire >= payload circularly; no switch (bit 12), as switching cost 5% on window moves
    uint32_t push[] = {
        MTHD(0, 0x5c, 5), (uint32_t)semva, (uint32_t)(semva >> 32), payload, 0, 3,
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

int chan_copy(struct chan *ch, uint64_t src, uint64_t dst, uint64_t size, volatile uint32_t *sem, uint64_t semva,
              uint32_t payload)
{
    return chan_copy2d(ch, src, (uint32_t)size, dst, (uint32_t)size, (uint32_t)size, 1, sem, semva, payload);
}

int chan_fill(struct chan *ch, uint64_t dst, uint64_t size, uint32_t pattern, volatile uint32_t *sem, uint64_t semva,
              uint32_t payload)
{
    uint32_t push[] = {
        MTHD(4, 0x000, 1), gsp_gpu(ch->s)->chip->copy,
        MTHD(4, 0x700, 1), pattern,  // SET_REMAP_CONST_A, low byte only with 1-byte components
        // SET_REMAP_COMPONENTS, DST_X = CONST_A with one 1-byte component
        MTHD(4, 0x708, 1), 4,
        MTHD(4, 0x408, 2), (uint32_t)(dst >> 32), (uint32_t)dst,  // OFFSET_OUT_UPPER/LOWER
        MTHD(4, 0x418, 1), (uint32_t)size,                        // LINE_LENGTH_IN
        MTHD(4, 0x41c, 1), 1,                                     // LINE_COUNT
        MTHD(4, 0x240, 3), (uint32_t)(semva >> 32), (uint32_t)semva, payload,
        // non-pipelined, flush, release, dst pitch layout, remap
        MTHD(4, 0x300, 1), 2 | (1 << 2) | (1 << 3) | (1 << 8) | (1 << 10),
    };
    chan_submit(ch, push, sizeof(push) / 4);
    if (wait_word(sem, payload, 2000)) {
        log_msg("chan_fill: did not complete (GP_GET %u)\n", chan_gp_get(ch));
        return -1;
    }
    return 0;
}
