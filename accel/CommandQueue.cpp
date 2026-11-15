#include "../registry.h"
#include "CommandQueue.h"
#include "Accelerator.h"
#include "IOAccelSegmentResourceList.h"
#include "bridge.h"
#include "commands.h"
#include "Task.h"
#include <IOKit/IOLib.h>
#include <kern/clock.h>

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

// Activity Monitor adds up each queue's accumulatedGPUTime, which the base fills in from the interval between a command
// buffer's fence and the stamp it waits for. A submit that waits for the GPU itself leaves it a few microseconds, so the
// time the kext saw the context busy goes in beside it.
void CommandQueue::bill()
{
    uint64_t busy_us = 0;
    if (!gGpu || gGpu->callPlatformFunction(NVGPU_FN_CTX_BUSY, true, (void *)queue_ctx(this), &busy_us, nullptr,
                                            nullptr) != kIOReturnSuccess)
        return;
    billed_ns += busy_us * 1000;
    uint64_t now;
    absolutetime_to_nanoseconds(mach_absolute_time(), &now);
    if (!billed_ns || now - billed_at < 50 * 1000 * 1000)
        return;
    billed_at = now;
    OSObject *was = copyProperty("accumulatedGPUTime");
    OSNumber *n = OSDynamicCast(OSNumber, was);
    setProperty("accumulatedGPUTime", (n ? n->unsigned64BitValue() : 0) + billed_ns, 64);
    OSSafeReleaseNULL(was);
    billed_ns = 0;
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
    if (!async)
        bill();
    return r;
}

}  // namespace malachite::accel
