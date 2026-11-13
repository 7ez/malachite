#include "../registry.h"
#include "CommandQueue.h"
#include "Accelerator.h"
#include "IOAccelSegmentResourceList.h"
#include "bridge.h"
#include "commands.h"
#include "Task.h"
#include <IOKit/IOLib.h>

// A refused task fails every command, and logging each costs WindowServer time: the first 20, then one in 1000.
#include <stdarg.h>

namespace malachite::accel {
static void failed_log(const char *fmt, ...)
{
    static uint32_t count;
    uint32_t n = __sync_add_and_fetch(&count, 1);
    if (n > 20 && n % 1000)
        return;
    char line[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n > 20)
        IOLog("(failure %u) %s", n, line);
    else
        IOLog("%s", line);
}

MalachiteDefineClass(CommandQueue, IOAccelCommandQueue)

// Feeds Activity Monitor's GPU time. Off by default: macOS 26 doesn't export it, and kmutil won't link a kext with a
// weak import it can't see tested, which fails the whole auxiliary collection.
#ifdef MALACHITE_GPU_STAT
extern "C" void ml_gpu_stat_update(uint64_t gpu_ns_delta) __attribute__((weak_import));
#endif

// Charged at the next submit, as ml_gpu_stat_update bills only the current thread; a task's last few ms go unbilled.
static void charge_gpu_time(uintptr_t ctx)
{
#ifdef MALACHITE_GPU_STAT
    uint64_t busy_us = 0;
    if (ml_gpu_stat_update && gGpu &&
        gGpu->callPlatformFunction(NVGPU_FN_CTX_BUSY, true, (void *)ctx, &busy_us, nullptr, nullptr) ==
            kIOReturnSuccess &&
        busy_us)
        ml_gpu_stat_update(busy_us * 1000);
#endif
}

// only so the abstract base can be instantiated; its exported methods handle a count-0 header
class SegmentResourceList : public IOAccelSegmentResourceList {
    OSDeclareDefaultStructors(SegmentResourceList)
};
MalachiteDefineClass(SegmentResourceList, IOAccelSegmentResourceList)

uint64_t CommandQueue::parseSegmentList(IOAccelShared2 *shared, IOAccelSegmentResourceListHeader const *hdr,
                                               IOAccelDeviceShmem *shmem)
{
    SegmentResourceList *list = OSTypeAlloc(SegmentResourceList);
    if (!list)
        return 0;
    if (!list->initWithSharedResourceList(shared, hdr, shmem)) {
        list->release();
        return 0;
    }
    return (uint64_t)list;
}

static uintptr_t queue_ctx(const IOAccelCommandQueue *q)
{
    Task *t = OSDynamicCast(Task, (OSObject *)q->getGPUTask());
    return t ? t->context() : 0;
}

// copied out, as the client can still write the segment
static bool record_payload(const IOAccelKernelCommand *cmd, const IOAccelKernelCommand *end, uint32_t len,
                           void *out, size_t size)
{
    size_t room = (size_t)((const uint8_t *)end - (const uint8_t *)cmd);
    if (len < sizeof(struct malachite_cmd_hdr) + size || len > room)
        return false;
    memcpy(out, (const uint8_t *)cmd + sizeof(struct malachite_cmd_hdr), size);
    return true;
}

uint64_t CommandQueue::processSegmentKernelCommand(IOAccelSegmentResourceList *resList,
                                                          IOAccelKernelCommand const *cmd,
                                                          IOAccelKernelCommand const *end)
{
    struct malachite_cmd_hdr h;
    if ((size_t)((const uint8_t *)end - (const uint8_t *)cmd) < sizeof(h))
        return IOAccelCommandQueue::processSegmentKernelCommand(resList, cmd, end);
    memcpy(&h, cmd, sizeof(h));
    void *ctx = (void *)queue_ctx(this);
    if (h.opcode & MALACHITE_CMD_OPENGL) {
        async = 0;
        h.opcode &= ~MALACHITE_CMD_OPENGL;
    }

    if (h.opcode == MALACHITE_CMD_COPY_BUF2BUF || h.opcode == MALACHITE_CMD_UPLOAD) {
        struct malachite_copy_buf2buf p;
        if (!record_payload(cmd, end, h.len, &p, sizeof(p)))
            return 0;
        if (gGpu) {
            IOReturn r = gGpu->callPlatformFunction(h.opcode == MALACHITE_CMD_UPLOAD ? NVGPU_FN_UPLOAD : NVGPU_FN_COPY, true,
                                                     (void *)(uintptr_t)p.srcAddr, (void *)(uintptr_t)p.dstAddr,
                                                     (void *)(uintptr_t)p.size, ctx);
            if (r != kIOReturnSuccess)
                failed_log("queue: copy failed 0x%x\n", r);
        } else {
            failed_log("queue: no Gpu, copy skipped\n");
        }
        return 0;  // a clear error word lets the segment continue
    }

    if (h.opcode == MALACHITE_CMD_FILL_BUFFER) {
        struct malachite_fill_buffer p;
        if (record_payload(cmd, end, h.len, &p, sizeof(p)) && gGpu)
            gGpu->callPlatformFunction(NVGPU_FN_FILL, true, (void *)(uintptr_t)p.dstAddr, (void *)(uintptr_t)p.size,
                                        (void *)(uintptr_t)p.pattern, ctx);
        return 0;
    }

    if (h.opcode == MALACHITE_CMD_DISPATCH) {
        struct nvgpu_dispatch d;
        if (!record_payload(cmd, end, h.len, &d, sizeof(d)) || !gGpu)
            return 0;
        IOReturn r = rm::queue_dispatch(gGpu, ctx, &d);
        if (r != kIOReturnSuccess)
            failed_log("queue: dispatch failed 0x%x: program %#llx, %ux%ux%u groups of %ux%ux%u, %u regs, "
                  "%u bytes shared\n",
                  r, d.prog_va, d.grid_x, d.grid_y, d.grid_z, d.block_x, d.block_y, d.block_z, d.reg_count,
                  d.shared_size);
        return 0;
    }

    if (h.opcode == MALACHITE_CMD_DRAW) {
        struct nvgpu_draw d;
        if (!record_payload(cmd, end, h.len, &d, sizeof(d)) || !gGpu)
            return 0;
        const char *why = "";
        IOReturn r = rm::queue_draw(gGpu, ctx, this, &d, &why);
        if (r != kIOReturnSuccess)
            failed_log("queue: draw failed 0x%x %s\n", r, why);
        return 0;
    }

    if (h.opcode == MALACHITE_CMD_PUSH) {
        struct malachite_push p;
        if (!record_payload(cmd, end, h.len, &p, sizeof(p)) || !gGpu)
            return 0;
        IOReturn r = rm::queue_push(gGpu, ctx, this, &p);
        if (r != kIOReturnSuccess)
            failed_log("queue: push failed 0x%x\n", r);
        return 0;
    }

    if (h.opcode == MALACHITE_CMD_COPY_2D) {
        struct nvgpu_copy2d c;
        if (!record_payload(cmd, end, h.len, &c, sizeof(c)) || !gGpu)
            return 0;
        IOReturn r = rm::queue_copy2d(gGpu, ctx, &c);
        if (r != kIOReturnSuccess)
            failed_log("queue: 2D copy failed 0x%x\n", r);
        return 0;
    }

    if (h.opcode == MALACHITE_CMD_COMPRESS) {
        struct malachite_compress p;
        if (record_payload(cmd, end, h.len, &p, sizeof(p)) && gGpu)
            rm::queue_compress(gGpu, ctx, p.addr);
        return 0;
    }

    return IOAccelCommandQueue::processSegmentKernelCommand(resList, cmd, end);
}

// +0x5c0 is the accelerator; +0x5c8 the event the base copies into each command buffer's fence afterwards
static IOGraphicsAccelerator2 *queue_accel(IOAccelCommandQueue *q)
{
    return *(IOGraphicsAccelerator2 **)((char *)q + 0x5c0);
}

uint64_t CommandQueue::processCommandBuffer(unsigned int a, unsigned int b)
{
    if (gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_CB_BEGIN, true, (void *)queue_ctx(this), nullptr, nullptr, nullptr);
    uint64_t r = IOAccelCommandQueue::processCommandBuffer(a, b);
    if (async < 0)
        async = gGpu && gGpu->callPlatformFunction(NVGPU_FN_ASYNC, true, (void *)queue_ctx(this), &app, nullptr,
                                                     nullptr) == kIOReturnSuccess;
    if (gGpu && gStampsFollowGPU && async)
        static_cast<Accelerator *>(queue_accel(this))->stampEvent((IOAccelEvent *)((char *)this + 0x5c8), queue_ctx(this),
                                                                  app);
    else if (gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_SYNC, true, (void *)queue_ctx(this), nullptr, nullptr, nullptr);
    charge_gpu_time(queue_ctx(this));
    return r;
}

}  // namespace malachite::accel
