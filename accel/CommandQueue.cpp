#include "../registry.h"
#include "CommandQueue.h"
#include "Accelerator.h"
#include "IOAccelSegmentResourceList.h"
#include "bridge.h"
#include "commands.h"
#include "Task.h"
#include <IOKit/IOLib.h>

// A failed command's line, but not every one: a task whose work is refused fails each draw and copy, and logging them
// all (hundreds of thousands) costs WindowServer time of its own. The first few, then one in a thousand.
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

// The kernel's GPU time for the calling thread (xnu osfmk/i386 and arm64 machine_routines): Activity Monitor's GPU
// Time and % GPU are task_gpu_utilisation and the coalition's gpu_time, which add these up. Weak, as it is not in
// every kernel's KPI: without it nothing is charged. Off unless MALACHITE_GPU_STAT is defined: macOS 26 doesn't export
// it, and kmutil won't link a kext with a weak import it can't tell is tested (OSKextSymbolIsResolved has no header
// in the SDK), so the auxiliary collection failed to build and none of our kexts loaded.
#ifdef MALACHITE_GPU_STAT
extern "C" void ml_gpu_stat_update(uint64_t gpu_ns_delta) __attribute__((weak_import));
#endif

// The GPU time the task's work has used since its last command buffer, charged to the thread that submits the next
// one: ml_gpu_stat_update can only bill the current thread, and a retire timer's is the kernel's. A task that
// stops submitting leaves its last few milliseconds unbilled.
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

// Empty segment resource list. IOAccelSegmentResourceList's init/prepare/complete are all exported and handle a
// count-0 header; this only exists to be instantiable (the base is abstract).
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

// the GPU context of the task the queue's work is for: its address space, its channels
static uintptr_t queue_ctx(const IOAccelCommandQueue *q)
{
    Task *t = OSDynamicCast(Task, (OSObject *)q->getGPUTask());
    return t ? t->context() : 0;
}

// a record's payload of `size` bytes, copied out (the segment is memory the client can still write): false when
// the segment doesn't hold the whole record, or the record doesn't hold the payload
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

    if (h.opcode == MALACHITE_CMD_COPY_BUF2BUF) {
        struct malachite_copy_buf2buf p;
        if (!record_payload(cmd, end, h.len, &p, sizeof(p)))
            return 0;
        if (gGpu) {
            IOReturn r = gGpu->callPlatformFunction(NVGPU_FN_COPY, true, (void *)(uintptr_t)p.srcAddr,
                                                     (void *)(uintptr_t)p.dstAddr, (void *)(uintptr_t)p.size, ctx);
            if (r != kIOReturnSuccess)
                failed_log("queue: copy failed 0x%x\n", r);
        } else {
            failed_log("queue: no Gpu, copy skipped\n");
        }
        return 0;  // handled; leave the queue error word clear so the segment continues
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
        IOReturn r = rm::queue_draw(gGpu, ctx, &d, &why);
        if (r != kIOReturnSuccess)
            failed_log("queue: draw failed 0x%x %s\n", r, why);
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

    // Not ours: let the base handle generic command types.
    return IOAccelCommandQueue::processSegmentKernelCommand(resList, cmd, end);
}

// IOAccelCommandQueue keeps its accelerator here, and the event its command buffers' completion follows: the base
// copies it into each one's event fence once processCommandBuffer returns
static IOGraphicsAccelerator2 *queue_accel(IOAccelCommandQueue *q)
{
    return *(IOGraphicsAccelerator2 **)((char *)q + 0x5c0);
}

uint64_t CommandQueue::processCommandBuffer(unsigned int a, unsigned int b)
{
    uint64_t r = IOAccelCommandQueue::processCommandBuffer(a, b);
    if (async < 0)
        async = gGpu && gGpu->callPlatformFunction(NVGPU_FN_ASYNC, true, (void *)queue_ctx(this), nullptr, nullptr,
                                                     nullptr) == kIOReturnSuccess;
    if (gGpu && gStampsFollowGPU && async)
        static_cast<Accelerator *>(queue_accel(this))->stampEvent((IOAccelEvent *)((char *)this + 0x5c8), queue_ctx(this));
    else if (gGpu)
        gGpu->callPlatformFunction(NVGPU_FN_SYNC, true, (void *)queue_ctx(this), nullptr, nullptr, nullptr);
    charge_gpu_time(queue_ctx(this));
    return r;
}

}  // namespace malachite::accel
