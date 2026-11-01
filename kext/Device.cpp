#include "../registry.h"
#include <IOKit/IOLib.h>
#include <IOKit/IOService.h>
#include <IOKit/IOSubMemoryDescriptor.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IOInterruptEventSource.h>
#include <IOKit/IOTimerEventSource.h>
#include <kern/thread_call.h>
#include <pexpert/pexpert.h>
#include <sys/fcntl.h>
#include <sys/vnode.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/pci/IOPCIDevice.h>
#include <mach/kmod.h>
#include <sys/proc.h>
#include <sys/kauth.h>
#include <sys/sysctl.h>

#include "../fbkext/surface.h"
#include "../gpu_bridge.h"
#include "abi.h"
#include "plat.h"
#include "regs.h"

extern "C" {
#include "../core/bar1.h"
#include "../core/chan.h"
#include "../core/chip.h"
#include "../core/disp.h"
#include "../core/compute.h"
#include "../core/texdesc.h"
#include "../core/render.h"
#include "../core/sph.h"
#include "../core/gr.h"
#include "../core/gsp.h"
#include "../core/mem.h"
#include "../core/rmfe/rmfe.h"
#include "../core/video.h"
#include "../core/pwr.h"
}

// kmutil wants a _kmod_info symbol; Xcode generates it, this Makefile doesn't
extern "C" kern_return_t _start(kmod_info_t *ki, void *d) { return KERN_SUCCESS; }
extern "C" kern_return_t _stop(kmod_info_t *ki, void *d) { return KERN_SUCCESS; }
extern "C" { KMOD_EXPLICIT_DECL(pl.aochi.malachite, "0.1", _start, _stop) }

namespace malachite::accel {
size_t pace_report(char *out, size_t cap);
}

namespace malachite::rm {

#define MAX_ESCAPE 0x10000

class UserClient;

// Metal's draws don't wait for each other: each takes a slot of its own for its const banks and texture pools
// (vertex bank, pixel bank, then the header and sampler pools, a page each, then a tessellated draw's tessellation
// init bank) and goes to the GPU unwaited. Whatever touches memory the draws may still be writing waits for the last
// of them first (drain).
static const int DRAW_SLOTS = 256;
static const uint32_t PROF_RING = 2048;
static const uint32_t DRAW_SLOT_BYTES = 0x4000;
// a dispatch's slot the same way: const bank 0 and the texture pools in a page, the QMD in the next
static const int DISPATCH_SLOTS = 16;
static const uint32_t DISPATCH_SLOT_BYTES = 0x2000;
// the kernel context keeps chids 1 (copy) and 4/5 (graphics); tasks' contexts take theirs from here on
static const uint32_t FIRST_CTX_CHID = 6;

// what the GPU calls cost, counted since load, for root through kern.malachite_stats calls,
// microseconds spent waiting on the GPU, and the lock's waits and holds
enum {
    ST_PRESENT, ST_FLIP, ST_DRAW, ST_DISPATCH, ST_COPY2D, ST_COPY, ST_FILL, ST_SYNC, ST_MAP, ST_UNMAP, ST_CURSOR,
    ST_DRAIN_WAITS, ST_DRAIN_US, ST_COPY_US, ST_DRAW_US, ST_COPY2D_US, ST_SYNC_US, ST_MAP_US, ST_UNMAP_US, ST_DRAW_SYS, ST_DRAW_SYS_KPX, ST_DRAW_VRAM, ST_DRAW_VRAM_KPX, ST_LOCKS, ST_LOCK_WAIT_US, ST_LOCK_WAIT_MAX_US,
    ST_LOCK_HOLD_US, ST_LOCK_MAX_US, ST_SLOT_WAIT_US, ST_PUSH_WAIT_US, ST_PRESENT_WAIT_US, ST_PRESENT_TOTAL_US, ST_PRESENT_POST_US, ST_PRESENT_SUBMIT_US,
    ST_PRESENT_COPY_US, ST_FLIP_PEND_US, ST_PRESENT_HEAD,
    ST_COUNT = ST_PRESENT_HEAD + BAR1_FB_SLOTS
};
static const char *const stat_names[ST_COUNT] = {
    "present", "flip", "draw", "dispatch", "copy2d", "copy", "fill", "sync", "map", "unmap", "cursor",
    "drain_waits", "drain_us", "copy_us", "draw_us", "copy2d_us", "sync_us", "map_us", "unmap_us", "draw_sys", "draw_sys_kpx", "draw_vram", "draw_vram_kpx", "locks", "lock_wait_us", "lock_wait_max_us",
    "lock_hold_us", "lock_max_us", "slot_wait_us", "push_wait_us", "present_wait_us",
    "present_total_us", "present_post_us", "present_submit_us", "present_copy_us", "flip_pend_us",
    "present_h0", "present_h1", "present_h2", "present_h3",
};
static uint64_t stats[ST_COUNT];

// the time the call takes, added to a counter
#define TIMED(counter, call)                      \
    ({                                            \
        uint64_t t0_ = time_us();                 \
        auto r_ = (call);                         \
        stats[counter] += time_us() - t0_;        \
        r_;                                       \
    })

static int stats_sysctl(struct sysctl_oid *, void *, int, struct sysctl_req *req)
{
    if (!kauth_cred_issuser(kauth_cred_get()))
        return EPERM;
    char out[ST_COUNT * 40];
    size_t n = 0;
    for (int i = 0; i < ST_COUNT; i++)
        n += snprintf(out + n, sizeof(out) - n, "%s %llu\n", stat_names[i], (unsigned long long)stats[i]);
    // the maxima are since the last read, so a benchmark reading before and after gets its own
    stats[ST_LOCK_WAIT_MAX_US] = stats[ST_LOCK_MAX_US] = 0;
    return SYSCTL_OUT(req, out, n);
}

SYSCTL_PROC(_kern, OID_AUTO, malachite_stats, CTLTYPE_OPAQUE | CTLFLAG_RD | CTLFLAG_LOCKED, nullptr, 0, stats_sysctl, "S",
            "what the NVIDIA driver's GPU calls cost");

// A GPU context: an RM client with an address space of its own, and on it a copy channel and a graphics channel
// (3D and compute) with the memory their launches use. Each is made when first wanted.
struct gpu_ctx {
    char owner[32];  // the process that opened it, for the log
    struct rm_client client;
    struct vmm vm;
    struct chan copy_ch;
    struct dma *copy_sem;
    uint64_t copy_semva;
    uint32_t copy_seq;  // monotonic fence payload so the semaphore word changes every submit
    bool gpu_ready;     // the client, address space and copy channel are up
    struct video video;  // the NVDEC channel, once video_ready
    bool video_ready;
    struct chan compute_ch;
    struct gr_ctx compute_ctx;
    struct dma *compute_qmd;   // one QMD (reused per dispatch), its semaphore in the page's tail
    struct dma *compute_cbuf;  // const bank 0 (kernel params at 0x160+), then the texture pools
    uint64_t compute_qmdva, compute_cbufva, compute_semva;
    struct dma *draw_ring;
    uint64_t draw_ringva;
    // kern.malachite_prof_on: a timestamp the GPU writes after each draw, and what the draw was, for the time between
    // two to be charged to the draw (kern.malachite_prof)
    struct dma *prof;
    uint64_t prof_va, prof_last_ts;
    uint32_t prof_n, prof_seen;
    struct prof_draw {
        uint64_t vs, ps;
        uint32_t verts, flags, area;
        uint16_t width, height;
    } *prof_draws;
    uint32_t draw_slot_seq[DRAW_SLOTS];  // the payload of the slot's last draw, 0 for none
    unsigned draw_next;
    uint32_t draw_pending;  // the last unwaited payload, 0 for none
    uint64_t kind_va;       // the depth target last given its kind
    uint32_t kind_set;
    bool kind_compressed;
    uint64_t kind_lo, kind_hi;  // its mapping
    uint64_t zcull_zt;      // the depth target zcull region 0 follows, 0 for none
    uint64_t zcull_lo, zcull_hi;  // the memory of it; hi 0 once a copy wrote there, which zcull can't know
    // Copies run on the copy channel in the draws' order without the CPU waiting: one waits on the word for the work
    // before it, and releases its own payload there. ce_pending is the last one the graphics channel has yet to be
    // told to wait for, ce_last the last one queued, and ce_half_seq what the copy pushbuffer's halves last held.
    uint32_t ce_pending, ce_last, ce_half_seq[2];
    int ce_half;
    // the latest launches, their last payload and the GPFIFO entry after them: on a new channel, the ones a fault
    // lost run again (the first of them only once, as it may be what faulted)
    struct {
        uint32_t payload, gp_end;
    } issued[128];
    unsigned issued_next;
    uint32_t replayed;
    struct dma *dispatch_ring;
    uint64_t dispatch_ringva;
    // the buffer the shaders' local memory comes out of: VRAM, its VA and size, and the bytes a thread it holds
    uint64_t slm_vram, slm_va, slm_size, slm_per_tpc;
    uint32_t slm_lane, slm_per_warp;
    uint32_t dispatch_slot_seq[DISPATCH_SLOTS];
    unsigned dispatch_next;
    uint32_t half_seq[2];  // the payload of the last work submitted in each half of the graphics pushbuffer
    int push_half;          // the half it's submitting into
    uint64_t pass_targets[9];  // the colour and depth targets the last draw wrote, 0 once anything else ran
    uint64_t pass_vs, pass_ps, pass_ti, pass_ts;  // and its shaders
    volatile uint32_t *compute_sem;
    uint32_t compute_seq;
    // the last launches, for the log when the channel faults
    struct launch_log {
        uint32_t seq, groups[3], block[3], regs;
        uint64_t prog, vs;  // a draw's vs, and its ps in prog
        uint64_t params[4];  // its first two shader argument words, of the vs then the ps
    } launches[8];
    uint64_t busy_since;  // when its outstanding graphics work began (time_us), 0 when none is
    uint64_t busy_unbilled;  // its busy time Accelerator has not yet charged to the task (NVGPU_FN_CTX_BUSY)
    bool compute_ready;
    // a channel GSP-RM tore down after a fault (its RC event), to be rebuilt before the next use
    volatile bool compute_faulted, copy_faulted;
    // a task whose work keeps faulting the GPU: its faults in the current window, and until when its work is refused
    uint32_t faults;
    uint64_t fault_window;
    uint64_t refused_until;
    uint32_t backoff_ms;  // the next refusal's length, doubling while the faults go on
};

// What the draws took on the GPU, by kind, since kern.malachite_prof was last read (kern.malachite_prof_on=1 times them): a draw's
// share is the time from the one before it finishing to its own finishing, and a longer wait (the GPU idle, or held
// up by a copy) is counted apart as a gap
static int prof_on;
static struct prof_row {
    uint64_t vs, ps, n, ns, verts, area, max;
    uint32_t flags;
    uint16_t width, height;
} prof_rows[128];
static unsigned prof_nrows;
static uint64_t prof_gaps, prof_gap_ns, prof_draws_total, prof_ns_total, prof_copies, prof_copy_bytes;
// by what a draw waited for: bit 0 idle, 1 invalidate, 2 serialize
static uint64_t prof_wait_n[8], prof_wait_ns[8];
static const uint64_t PROF_GAP_NS = 3000000;

static void prof_add(const gpu_ctx::prof_draw &d, uint64_t ns)
{
    prof_draws_total++;
    if (ns > PROF_GAP_NS) {
        prof_gaps++;
        prof_gap_ns += ns;
        return;
    }
    prof_ns_total += ns;
    unsigned waits = (d.flags & 3) | (d.flags >> 6 & 1) << 2;
    prof_wait_n[waits]++;
    prof_wait_ns[waits] += ns;
    prof_row *r = nullptr;
    for (unsigned i = 0; i < prof_nrows && !r; i++)
        if (prof_rows[i].vs == d.vs && prof_rows[i].ps == d.ps && prof_rows[i].flags == d.flags &&
            prof_rows[i].width == d.width && prof_rows[i].height == d.height)
            r = &prof_rows[i];
    if (!r) {
        if (prof_nrows == sizeof(prof_rows) / sizeof(prof_rows[0]))
            return;
        r = &prof_rows[prof_nrows++];
        *r = { d.vs, d.ps, 0, 0, 0, 0, 0, d.flags, d.width, d.height };
    }
    r->n++;
    r->ns += ns;
    r->verts += d.verts;
    r->area += d.area;
    if (ns > r->max)
        r->max = ns;
}

static int prof_sysctl(struct sysctl_oid *, void *, int, struct sysctl_req *req)
{
    if (!kauth_cred_issuser(kauth_cred_get()))
        return EPERM;
    if (!req->oldptr)  // sysctl(8) asks for the size first, which isn't a read
        return SYSCTL_OUT(req, nullptr, 16384);
    const size_t cap = 16384;
    char *out = (char *)IOMalloc(cap);
    if (!out)
        return ENOMEM;
    size_t n = snprintf(out, cap, "draws %llu, busy %llu us, gaps %llu (%llu us), copies %llu (%llu KB)\n",
                        (unsigned long long)prof_draws_total, (unsigned long long)prof_ns_total / 1000,
                        (unsigned long long)prof_gaps, (unsigned long long)prof_gap_ns / 1000,
                        (unsigned long long)prof_copies, (unsigned long long)prof_copy_bytes / 1024);
    for (unsigned k = 0; k < 8; k++) {
        if (prof_wait_n[k])
            n += snprintf(out + n, cap - n, "idle=%u invalidate=%u serialize=%u: n=%llu avg=%llu ns\n", k & 1, k >> 1 & 1,
                          k >> 2, (unsigned long long)prof_wait_n[k],
                          (unsigned long long)(prof_wait_ns[k] / prof_wait_n[k]));
        prof_wait_n[k] = prof_wait_ns[k] = 0;
    }
    // by time, the most first
    for (unsigned k = 0; k < 48 && n + 160 < cap; k++) {
        unsigned best = prof_nrows;
        for (unsigned i = 0; i < prof_nrows; i++)
            if (prof_rows[i].n && (best == prof_nrows || prof_rows[i].ns > prof_rows[best].ns))
                best = i;
        if (best == prof_nrows)
            break;
        const prof_row &r = prof_rows[best];
        n += snprintf(out + n, cap - n, "%llu us n=%llu avg=%llu max=%llu vs=%#llx ps=%#llx flags=%#x %ux%u verts=%llu px=%llu\n",
                      (unsigned long long)r.ns / 1000, (unsigned long long)r.n, (unsigned long long)(r.ns / r.n / 1000),
                      (unsigned long long)r.max / 1000, (unsigned long long)r.vs, (unsigned long long)r.ps, r.flags,
                      r.width, r.height, (unsigned long long)r.verts / r.n, (unsigned long long)r.area / r.n);
        prof_rows[best].n = 0;
    }
    prof_nrows = 0;
    prof_gaps = prof_gap_ns = prof_draws_total = prof_ns_total = prof_copies = prof_copy_bytes = 0;
    int r = SYSCTL_OUT(req, out, n);
    IOFree(out, cap);
    return r;
}

SYSCTL_INT(_kern, OID_AUTO, malachite_prof_on, CTLFLAG_RW | CTLFLAG_LOCKED, &prof_on, 0, "time the draws on the GPU");
static int compress_on = 1, zcull_on = 1;
SYSCTL_INT(_kern, OID_AUTO, malachite_zcull, CTLFLAG_RW | CTLFLAG_LOCKED, &zcull_on, 0, "cull with zcull in passes from now on");
SYSCTL_INT(_kern, OID_AUTO, malachite_compress, CTLFLAG_RW | CTLFLAG_LOCKED, &compress_on, 0,
           "compress targets made from now on");

// The same layout with compression, without PLC: GA102's PLC has an erratum on some physical addresses.
static uint8_t compressed_kind(uint8_t kind)
{
    switch (kind) {
    case 0x01: return 0x0b;  // Z16
    case 0x04: return 0x0d;  // ZF32_X24S8
    case 0x06: return 0x09;  // generic
    default: return 0;
    }
}
SYSCTL_PROC(_kern, OID_AUTO, malachite_prof, CTLTYPE_OPAQUE | CTLFLAG_RD | CTLFLAG_LOCKED, nullptr, 0, prof_sysctl, "S",
            "what the draws took on the GPU, by kind");

class Gpu : public IOService
{
    OSDeclareDefaultStructors(Gpu)

    IOMemoryMap *bar0 = nullptr;
    struct plat plat = {};
    struct gpu gpu = {};
    struct gsp *gsp = nullptr;
    struct rmfe *fe = nullptr;

    // the kernel's GPU context: present, the display's surfaces, and work from tasks without one;
    // and each task's that has asked for one (NVGPU_FN_CTX_OPEN), named to Accelerator by slot + 1 and a serial
    gpu_ctx kctx = {};
    static const int MAX_CTXS = 64;
    gpu_ctx *ctxs[MAX_CTXS] = {};
    uint32_t ctx_serial[MAX_CTXS] = {};
    gpu_ctx *ctx_of(uintptr_t handle);
    uintptr_t ctx_open();
    void ctx_close(uintptr_t handle);
    uint32_t present_fence = 0;  // the last present's payload, 0 before the first
    bool copy_done(uint32_t payload) const
    {
        return (int32_t)(*(volatile uint32_t *)kctx.copy_sem->cpu - payload) >= 0;
    }
    struct gpu_mapping {
        gpu_ctx *ctx;  // whose address space it's in
        uint64_t va;
        struct dma dma;    // system memory: its pages (dma.size the mapping's size either way)
        uint64_t vram;     // or the VRAM it maps, 0 for none
        uint8_t kind;  // PTE kind its pages are mapped with
        bool compressed;  // as kind's compressed form, in big pages
        uint8_t shadow;  // the VRAM copy of a surface (shadows) it maps in place of its pages, 1-based, 0 for none
        int32_t hnext;   // next slot + 1 in its hash bucket
    };
    // Metal apps read just outside their buffers and Apple's GPUs let them: a page of zeros either side of each
    // mapping, read only, so such a read gets 0 rather than faulting the channel (a write still faults)
    struct dma *zero_page = nullptr;
    bool map_slack(gpu_ctx *c, uint64_t va)
    {
        if (!zero_page && (zero_page = gsp_dma(gsp, 0x1000, 1)))
            memset(zero_page->cpu, 0, 0x1000);
        return zero_page && !vmm_map_sys(&c->vm, va, zero_page, 0, 0x1000, VMM_RO);
    }
    // A buffer's address range with its two zero pages: a fragment shader's helper lane reads one byte before its
    // buffer (CoreAnimation's text blit does). Returns the buffer's address, 0 when the space is full.
    uint64_t map_va(gpu_ctx *c, uint64_t size)
    {
        uint64_t base = vmm_alloc(&c->vm, size + 0x2000, 0x1000);
        if (!base)
            return 0;
        map_slack(c, base);
        return base + 0x1000;
    }
    // the same on 64 KB, for big pages: the ends of the range it took to get there go back
    uint64_t map_va_big(gpu_ctx *c, uint64_t size)
    {
        uint64_t base = vmm_alloc(&c->vm, size + 0x12000, 0x10000);
        if (!base)
            return 0;
        vmm_free(&c->vm, base, 0xf000);
        vmm_free(&c->vm, base + size + 0x11000, 0x1000);
        map_slack(c, base + 0xf000);
        return base + 0x10000;
    }
    void map_va_free(gpu_ctx *c, uint64_t va, uint64_t size)
    {
        vmm_unmap(&c->vm, va - 0x1000, size + 0x2000);
        vmm_free(&c->vm, va - 0x1000, size + 0x2000);
    }
    IOReturn set_kind(gpu_ctx *c, uint64_t va, uint8_t kind, bool compress);
    // the first bytes of the code at va, for the log of a launch that faulted
    void log_code(uint64_t va)
    {
        for (int j = 0; j < gpu_maps_top; j++) {
            struct gpu_mapping &m = gpu_maps[j];
            if (!m.va || va < m.va || va >= m.va + m.dma.size || !m.dma.handle)
                continue;
            uint32_t w[8];
            for (uint64_t off = 0; off < 0x2400 && va - m.va + off + sizeof(w) <= m.dma.size; off += sizeof(w)) {
                if (((IOMemoryDescriptor *)m.dma.handle)->readBytes(va - m.va + off, w, sizeof(w)) != sizeof(w))
                    break;
                log_msg("Malachite:     code +%03llx: %08x %08x %08x %08x %08x %08x %08x %08x\n", (unsigned long long)off, w[0], w[1],
                        w[2], w[3], w[4], w[5], w[6], w[7]);
            }
            return;
        }
    }
    // Every context's mappings, in a table that doubles when full (a Metal process can hold tens of thousands of
    // buffers), up to a cap past which a map fails with an error. Slots are found by VA through a chained hash
    // (1-based slot numbers, 0 for none), because an unmap per buffer would otherwise scan the whole table.
    static const int MAX_GPU_MAPS = 1 << 18;
    struct gpu_mapping *gpu_maps = nullptr;  // va -> backing dma (pages kept alive while mapped)
    int gpu_maps_cap = 0;
    int gpu_maps_free = 0;  // no slot below it is free
    int gpu_maps_top = 0;   // none at or above it is used, so the scans over the table stop here
    int target_map = 0;     // the mapping the last draw rendered to
    int32_t *map_hash = nullptr;  // gpu_maps_cap buckets
    unsigned map_bucket(uint64_t va) const
    {
        return (unsigned)(((va >> 12) * 0x9e3779b97f4a7c15ull) >> 32) & (unsigned)(gpu_maps_cap - 1);
    }
    int map_slot(uint64_t va) const
    {
        if (!map_hash || !va)
            return -1;
        for (int i = map_hash[map_bucket(va)]; i; i = gpu_maps[i - 1].hnext)
            if (gpu_maps[i - 1].va == va)
                return i - 1;
        return -1;
    }
    void map_link(int slot)
    {
        int32_t &head = map_hash[map_bucket(gpu_maps[slot].va)];
        gpu_maps[slot].hnext = head;
        head = slot + 1;
        if (slot >= gpu_maps_top)
            gpu_maps_top = slot + 1;
    }
    // the slot emptied, and the scans' bound pulled back over the free slots at the end
    void map_clear(int slot)
    {
        for (int32_t *p = &map_hash[map_bucket(gpu_maps[slot].va)]; *p; p = &gpu_maps[*p - 1].hnext)
            if (*p == slot + 1) {
                *p = gpu_maps[slot].hnext;
                break;
            }
        memset(&gpu_maps[slot], 0, sizeof(gpu_maps[slot]));
        if (slot < gpu_maps_free)
            gpu_maps_free = slot;
        while (gpu_maps_top && !gpu_maps[gpu_maps_top - 1].va)
            gpu_maps_top--;
    }
    bool map_grow()
    {
        int cap = gpu_maps_cap ? gpu_maps_cap * 2 : 4096;
        if (cap > MAX_GPU_MAPS)
            return false;
        struct gpu_mapping *maps = (struct gpu_mapping *)IOMallocZero(sizeof(struct gpu_mapping) * cap);
        int32_t *hash = (int32_t *)IOMallocZero(sizeof(int32_t) * cap);
        if (!maps || !hash) {
            if (maps)
                IOFree(maps, sizeof(struct gpu_mapping) * cap);
            if (hash)
                IOFree(hash, sizeof(int32_t) * cap);
            return false;
        }
        if (gpu_maps) {
            memcpy(maps, gpu_maps, sizeof(struct gpu_mapping) * gpu_maps_cap);
            IOFree(gpu_maps, sizeof(struct gpu_mapping) * gpu_maps_cap);
            IOFree(map_hash, sizeof(int32_t) * gpu_maps_cap);
        }
        int old = gpu_maps_cap;
        gpu_maps = maps;
        map_hash = hash;
        gpu_maps_cap = cap;
        for (int i = 0; i < old; i++)
            if (gpu_maps[i].va)
                map_link(i);
        return true;
    }
    // a free slot, growing the table for one; -1 when it's at its cap
    int map_alloc()
    {
        for (;;) {
            for (; gpu_maps_free < gpu_maps_cap; gpu_maps_free++)
                if (!gpu_maps[gpu_maps_free].va)
                    return gpu_maps_free;
            if (!map_grow())
                return -1;
        }
    }

    // GPU busy time: a launch opens its context's interval, and it closes once the context's semaphore has caught
    // up with its last launch (drain, the stamp timer's NVGPU_FN_DRAW_DONE polls, or a stats tick). Lock-free,
    // because DRAW_DONE is. Each context's time goes to busy_unbilled, for the task to be charged; busy_total is
    // the time with any context busy, so contexts that overlap count once.
    uint64_t busy_total = 0;
    // how many contexts are busy (low byte) and the time (above it) that count last changed
    uint64_t busy_state = 0;
    void busy_step(int delta, uint64_t now)
    {
        uint64_t old = __atomic_load_n(&busy_state, __ATOMIC_RELAXED), next;
        do {
            uint64_t mark = old >> 8;
            next = (now > mark ? now : mark) << 8 | ((old & 0xff) + delta);
        } while (!__atomic_compare_exchange_n(&busy_state, &old, next, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        if ((old & 0xff) && now > old >> 8)
            __atomic_fetch_add(&busy_total, now - (old >> 8), __ATOMIC_RELAXED);
    }
    static bool caught_up(const gpu_ctx *c)
    {
        return !c->compute_sem || (int32_t)(*c->compute_sem - c->compute_seq) >= 0;
    }
    IOReturn busy_open(gpu_ctx *c, IOReturn r)
    {
        uint64_t none = 0, now = time_us();
        if (r == kIOReturnSuccess &&
            __atomic_compare_exchange_n(&c->busy_since, &none, now, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            busy_step(1, now);
        return r;
    }
    void busy_close(gpu_ctx *c)
    {
        if (uint64_t since = __atomic_exchange_n(&c->busy_since, 0, __ATOMIC_RELAXED)) {
            uint64_t now = time_us();
            busy_step(-1, now);
            __atomic_fetch_add(&c->busy_unbilled, now > since ? now - since : 0, __ATOMIC_RELAXED);
        }
    }
    // work still outstanding is counted up to now and its interval restarted, so a long draw shows in every tick
    void busy_split(gpu_ctx *c, uint64_t now)
    {
        uint64_t since = __atomic_load_n(&c->busy_since, __ATOMIC_RELAXED);
        while (since && !__atomic_compare_exchange_n(&c->busy_since, &since, now, false, __ATOMIC_RELAXED,
                                                     __ATOMIC_RELAXED))
            ;
        if (since && now > since)
            __atomic_fetch_add(&c->busy_unbilled, now - since, __ATOMIC_RELAXED);
    }
    uint64_t busy_take(gpu_ctx *c)
    {
        if (caught_up(c))
            busy_close(c);
        else
            busy_split(c, time_us());
        return __atomic_exchange_n(&c->busy_unbilled, 0, __ATOMIC_RELAXED);
    }
    uint64_t recoveries = 0;
    IOReturn gpu_stats(struct nvgpu_stats *out);

    static const uint32_t TEX_HEADER_POOL = 0x400, TEX_SAMPLER_POOL = 0x800;  // offsets in kctx.compute_cbuf
    IOReturn drain(gpu_ctx *c);
    IOReturn read_indirect(gpu_ctx *c, uint64_t va, uint32_t *words, unsigned n);
    IOReturn push_room(gpu_ctx *c, uint32_t bytes);
    IOReturn ce_room(gpu_ctx *c, uint32_t bytes);
    IOReturn after_copies(gpu_ctx *c);
    bool prof_ready(gpu_ctx *c);
    void prof_harvest(gpu_ctx *c);
    IOReturn local_memory(gpu_ctx *c, uint32_t lane);
    void free_local_memory(gpu_ctx *c);
    void issued(gpu_ctx *c)
    {
        c->issued[c->issued_next++ % 128] = { c->compute_seq, c->compute_ch.gp_put };
    }
    static void rc_event(void *ctx, uint32_t engine, uint32_t chid, uint32_t exception);
    void recover(gpu_ctx *c);
    int ring_wait(gpu_ctx *c, int counter, uint32_t payload, int ms);
    void resync_screens();
    // work that timed out has likely found its channel faulted: GSP-RM's messages read now, under the lock (where
    // the interrupt handler can't), so the RC event marks the channel. A fault shows up just after the launch that
    // made it returns, so it's the next launch that times out, and runs again on a new channel.
    IOReturn settle(IOReturn r)
    {
        if (r == kIOReturnTimeout && gsp)
            gsp_poll(gsp);
        return r;
    }
    // a dispatch or a draw, again on a new channel when an earlier launch faulted the channel under it. A fault is
    // only reported after the launch that made it has released its semaphore, so there's no telling it from one
    // the timed-out launch made itself: that one runs twice (and faults the new channel too).
    IOReturn launch(gpu_ctx *c, const struct nvgpu_dispatch *d, const struct nvgpu_draw *w, const char **reject = nullptr)
    {
        // launches go to the GPU unwaited, so the word of a fault comes in after the faulting one returned: heard
        // first, it has the channel made anew before this one is lost on it
        if (gsp)
            gsp_poll(gsp);
        IOReturn r = settle(d ? gpu_dispatch(c, d) : gpu_draw(c, w, reject));
        if (r == kIOReturnTimeout && c->compute_faulted)
            r = settle(d ? gpu_dispatch(c, d) : gpu_draw(c, w, reject));
        return r;
    }
    void drain_all()
    {
        drain(&kctx);
        for (gpu_ctx *c : ctxs)
            if (c)
                drain(c);
    }

    bool gpu_bridge_init(gpu_ctx *c);
    bool compute_bridge_init(gpu_ctx *c);
    bool video_init(gpu_ctx *c);
    IOReturn gpu_map(gpu_ctx *c, IOMemoryDescriptor *md, uint64_t *outva);
    IOReturn gpu_map_vram(gpu_ctx *c, uint64_t vram, uint64_t size, uint64_t *outva);
    IOReturn gpu_unmap(uint64_t va, uint64_t size, gpu_ctx *owner = nullptr);
    // An unmap asked for while the task's draws are still in flight is done once they are, rather than by making
    // WindowServer wait for the GPU in the middle of its frame: the mapping (and with it the pages, which the buffer's
    // owner may free) stays until the context's semaphore has passed the sequence the draws had reached.
    struct deferred_unmap {
        gpu_ctx *ctx;
        uint64_t va;
        uint32_t seq;
    };
    static const int MAX_DEFERRED = 256;
    deferred_unmap deferred[MAX_DEFERRED];
    int ndeferred = 0;
    // VRAM freed while any task's draws are in flight is given back once none are (they may be reading it)
    struct deferred_free {
        uint64_t vram, size;
    };
    deferred_free vram_deferred[MAX_DEFERRED];
    int nvram_deferred = 0;
    void retire_deferred();
    void drop_deferred(gpu_ctx *c);
    IOReturn gpu_copy(gpu_ctx *c, uint64_t src, uint64_t dst, uint64_t size);
    IOReturn gpu_copy2d(gpu_ctx *c, const struct nvgpu_copy2d *cp);
    IOReturn gpu_fill(gpu_ctx *c, uint64_t dst, uint64_t size, uint32_t pattern);
    IOReturn gpu_dispatch(gpu_ctx *c, const struct nvgpu_dispatch *d);
    IOReturn gpu_draw(gpu_ctx *c, const struct nvgpu_draw *d, const char **reject);
    IOReturn gpu_present(struct nvgpu_present *p, uint64_t since);
    IOReturn present_frame(struct nvgpu_present *p);

    // NVGpuPresent: WindowServer cycles three or four framebuffer surfaces per display, so each stays mapped in a
    // slot of its own; a new surface takes the least recently shown slot, rewriting the PTEs of that slot's fixed
    // VA range. Fewer slots than the surfaces of all the displays would take one for every present, and a
    // promotion with it. The scanout surface gets a VA of its own too.
    static const int PRESENT_SLOTS = 4 * BAR1_FB_SLOTS;
    static const uint64_t PRESENT_SLOT_SIZE = 64ull << 20;
    struct present_slot {
        IOMemoryDescriptor *md;  // retained and prepared while mapped
        struct dma dma;
        uint64_t va, used;
        uint32_t fence;  // the payload of the last copy made from it
        int count;       // frames shown from it, negative while a promotion that found nothing to move rests
        int shadow;      // its VRAM copy once it has been shown twice, 1-based, 0 for none
    } present[PRESENT_SLOTS] = {};
    // A surface WindowServer keeps showing is moved to VRAM: its pages' contents are copied there and every mapping
    // of the surface made to map the copy, so it is drawn into at VRAM speed and the head flips to it as it is
    // rather than a copy of it going over PCIe each frame. The CPU's view of the pages is then stale; a demotion
    // (the slot taken for another surface) copies it back.
    static const int SHADOWS = PRESENT_SLOTS;
    struct shadow_copy {
        uint64_t *pages;  // the surface's physical pages
        uint32_t npages;
        uint64_t vram;  // 0 for a free entry
    } shadows[SHADOWS] = {};
    uint64_t shadow_va = 0;  // where the copies to and from a shadow map in the kernel's space
    // promotions and demotions since the last line about them, which is logged once per 10 s
    uint32_t promoted = 0, demoted = 0;
    uint64_t promote_mark = 0;
    uint64_t shadow_at(const uint64_t *pages, size_t n, int *which);
    int promote(present_slot *s, uint64_t bytes);
    void demote(present_slot *s);
    uint64_t present_frames = 0;
    struct disp *disp = nullptr;
    // A monitor driven: screen i is shown by head i and seen by the framebuffer through nub i and BAR1 framebuffer slot i
    static const int SCREENS = BAR1_FB_SLOTS;
    struct screen {
        IOService *nub;  // for the framebuffer
        uint64_t shown;  // the surface the head shows, 0 while unlit: output out in mode mode (1-based)
        uint64_t size;
        uint32_t pitch, mode;
        int out;
        bool changed;    // Framebuffer is to be told, outside the lock
        bool asleep;     // display sleep: the head is off and the monitor in standby
        bool unplugged;  // the monitor went away; the head is blanked, the surface kept
        uint64_t scanout_va, scanout_vram;  // the surface shown, mapped for presents
        // Tear-free presents: a frame is copied onto the one of three surfaces (shown, back[0], back[1]) after the one
        // the last present chose, and the head flips to it at the first vblank after the copy has landed, which the
        // copy engine says in a semaphore the window waits for (disp_flip_queue). The window channel takes one
        // flip at a time, so a present registers its flip (pend) and returns once its copy is queued, and pump_flips
        // queues it as the head takes the one before. The frame's own surface is free once the copy has landed, so
        // WindowServer goes on with its next while the head waits for its vblank.
        uint64_t back[2], back_va[2], back_size, sema_va;
        int cur;     // the surface of the last present's flip: 0 shown, 1 and 2 back
        int queued;  // the surface of the last flip queued to the head
        unsigned last_k;       // and its semaphore word's index
        uint32_t last_fence;   // and what the word is to read
        unsigned flips;  // flips registered; the next one's semaphore is flips & 1, never the word the window still holds
        bool fresh;      // no flip queued since the head last had a modeset
        uint32_t flip_fence[3];  // what each semaphore word is waited for to read
        uint32_t fence;          // the payload of the last present's copy
        struct {
            bool valid;
            int to;
            uint32_t fence;
            unsigned k;
            uint64_t since;
        } pend;
        void (*vbl_fn)(void *);  // Framebuffer's, called on each of the head's vblanks, under evlock
        void *vbl_ctx;
        void (*pipe_fn)(void *);  // the accelerator's display pipe's, the same
        void *pipe_ctx;
        uint32_t *cursor;  // Framebuffer's last cursor image (CURS_MAX square room) and position, for after a power cycle
        uint32_t cursor_w, cursor_h;
        int cursor_x, cursor_y;
        uint16_t *gamma;  // the last output LUT asked for (SURFACE_GAMMA_ENTRIES triples), for after a power cycle
    } screens[SCREENS] = {};
    int index(const screen &sc) const { return (int)(&sc - screens); }
    static uint64_t surface_of(const screen &sc, int i) { return i ? sc.back[i - 1] : sc.shown; }
    // no flip registered or queued, the head showing surface cur. The window keeps the semaphore of its last flip for
    // the next update it is sent (a modeset's), so that word is left reading what the update waits for, and the flip
    // is let through, before the state goes.
    void reset_flips(screen &sc, int cur = 0)
    {
        if (disp && sc.flips && sc.sema_va) {
            const int head = index(sc);
            vram_wr32(&gpu, disp->ntfy + DISP_FLIP_SEMA(head, sc.last_k), sc.last_fence);
            vram_flush(&gpu);
            for (int i = 0; i < 250 && !disp_flip_taken(disp, head); i++)
                delay_us(200);
        }
        sc.cur = sc.queued = cur;
        sc.fresh = true;
        sc.pend.valid = false;
    }
    bool lit() const
    {
        for (const screen &sc : screens)
            if (sc.shown)
                return true;
        return false;
    }
    bool want_shown = false;  // asked to light heads: hotplug lights one when a monitor turns up
    uint32_t unlit_polls = 0; // timer ticks spent waiting for a monitor, between connection checks
    bool show_failed = false; // lighting failed with these displays connected: polls don't retry until that changes
    uint32_t failed_connected = 0;
    UserClient *clients = nullptr;
    IOLock *lock = nullptr;
    uint64_t held_since = 0;
    void take()
    {
        uint64_t t = time_us();
        IOLockLock(lock);
        held_since = time_us();
        stats[ST_LOCKS]++;
        stats[ST_LOCK_WAIT_US] += held_since - t;
        if (held_since - t > stats[ST_LOCK_WAIT_MAX_US])
            stats[ST_LOCK_WAIT_MAX_US] = held_since - t;
        if (capped)
            uncap();
    }
    // nvpm PM_IDLE_CAP: GSP-RM is told each second whether the GPU did anything, and the cap is lifted as work comes
    // in, since a GPU held at its lowest clocks would run a frame's draws slowly until the next tick
    uint32_t pm = 0;
    bool capped = false;
    uint64_t cap_mark = 0, cap_work = 0, cap_busy = 0;
    void uncap()
    {
        if (gsp && !suspended)
            pwr_idle_cap(gsp, 0, 0, 0);
        capped = false;
    }
    void idle_tick();
    void give()
    {
        uint64_t held = time_us() - held_since;
        stats[ST_LOCK_HOLD_US] += held;
        if (held > stats[ST_LOCK_MAX_US])
            stats[ST_LOCK_MAX_US] = held;
        IOLockUnlock(lock);
    }
    IOWorkLoop *loop = nullptr;
    IOTimerEventSource *timer = nullptr;
    IOInterruptEventSource *irq = nullptr;
    IOLock *evlock = nullptr;
    uint32_t gr_events = 0;  // graphics nonstall interrupts seen
    uint32_t vbl_wake = 0;   // slept on by presents waiting for a head to take a flip, woken by every vblank
    uint64_t vblanks = 0, vblank_mark = 0;  // head 0's, and when the last 600 started

    void stop_gsp();
    // a power cycle, system sleep's or MALACHITE_PM_CYCLE's: the entry points wait while suspended
    bool suspended = false;
    bool awake();
    void suspend();
    IOReturn resume();
    uint64_t canary = 0;  // VRAM written at suspend and read at resume: whether VRAM kept its contents
    void drop_contexts();
    void forget_contexts(bool free_rm);
    void ctx_forget(gpu_ctx *c);
    bool ctx_rebind(gpu_ctx *c);
    bool copy_channel(gpu_ctx *c, uint32_t chid, uint32_t handle);
    bool graphics_channel(gpu_ctx *c, uint32_t chid);
    void keep_cursor(screen &sc, const uint32_t *argb, uint32_t w, uint32_t h);
    IOReturn set_gamma(screen &sc, const uint16_t *rgb);
    int modeset(screen &sc, const struct disp_mode *t, uint64_t fb = 0);
    void create_nubs();
    void tell_nubs();
    void vblank_fns(int i);
    void publish_surface(screen &sc, bool hotplug);
    IOReturn set_mode(screen &sc, uint32_t id);
    IOReturn set_power(screen &sc, bool on);
    void darken_fb();
    IOReturn show();
    IOReturn light(screen &sc, int out);
    void check_hotplug();
    IOReturn read_shown(uint64_t addr, uint64_t len, task_t task);
    IOReturn h264_decode(const uint64_t *args, task_t task);
    static void poll(OSObject *owner, IOTimerEventSource *sender);
    thread_call_t autostart_call = nullptr;
    thread_call_t pump_call = nullptr;  // queues the flips presents have registered, as the head takes the ones before
    static void pump(thread_call_param_t self, thread_call_param_t);
    void pump_flips();
    uint32_t safe_sync = 0;
    struct disp_size scaled[DISP_MAX_SCALED];  // the images scaled.txt offers on panels they suit
    uint32_t nscaled = 0;
    void add_scaled_size(uint32_t w, uint32_t h);
    void read_scaled();
    static void autostart(thread_call_param_t self, thread_call_param_t);
    static void interrupt(OSObject *owner, IOInterruptEventSource *sender, int count);
    IOMemoryDescriptor *map_descriptor(const struct rmfe_map *m, IOOptionBits *options);

public:
    uint32_t wait_event(uint32_t seen, uint32_t timeout_us);
    IOReturn set_scaled(const uint64_t *sizes, uint32_t n);

    bool start(IOService *provider) override;
    void stop(IOService *provider) override;
    void systemWillShutdown(IOOptionBits specifier) override;
    IOReturn setPowerState(unsigned long ordinal, IOService *device) override;
    IOReturn callPlatformFunction(const OSSymbol *name, bool wait, void *p1, void *p2, void *p3, void *p4) override;
    bool usable();
    IOReturn queue_dispatch(void *ctx, const struct nvgpu_dispatch *d);
    IOReturn queue_draw(void *ctx, const struct nvgpu_draw *d, const char **why);
    IOReturn queue_copy2d(void *ctx, const struct nvgpu_copy2d *p);
    IOReturn queue_compress(void *ctx, uint64_t va);
    // Framebuffer's calls (SURFACE_FN_*) for screen i, through its nub
    IOReturn screen_call(int i, const OSSymbol *name, void *p1, void *p2, void *p3, void *p4);

    using IOService::newUserClient;
    IOReturn newUserClient(task_t task, void *security, UInt32 type, OSDictionary *props,
                           IOUserClient **handler) override;
    IOReturn video_open(uintptr_t *ctx, struct video_h264 *h, const uint64_t *args);
    IOReturn video_decode(uintptr_t ctx, struct video_h264 *h, const uint64_t *args);
    void video_close(uintptr_t ctx, struct video_h264 *h, uint32_t decoded);

    IOReturn command(uint32_t selector, const uint64_t *args, task_t task);
    size_t power_report(char *out, size_t cap);
    IOReturn escape(UserClient *uc, uint32_t cmd, void *arg, uint32_t size);
    void add_client(UserClient *uc);
    void remove_client(UserClient *uc);
    IOMemoryDescriptor *take_pending(UserClient *uc, IOOptionBits *options);
};

// what the framebuffer attaches to for one screen: its platform calls reach Gpu with the screen they're for
class SurfaceNub : public IOService
{
    OSDeclareDefaultStructors(SurfaceNub)

public:
    int screen = 0;
    IOReturn callPlatformFunction(const OSSymbol *name, bool wait, void *p1, void *p2, void *p3, void *p4) override
    {
        Gpu *rm = OSDynamicCast(Gpu, getProvider());
        return rm ? rm->screen_call(screen, name, p1, p2, p3, p4) : kIOReturnNotReady;
    }
    void stop(IOService *provider) override
    {
        PMstop();
        IOService::stop(provider);
    }
};

class UserClient : public IOUserClient
{
    OSDeclareDefaultStructors(UserClient)

public:
    Gpu *owner = nullptr;
    task_t task = nullptr;
    struct rmfe_file *file = nullptr;
    int64_t fd = -2;
    IOMemoryDescriptor *pending = nullptr;  // what the next IOConnectMapMemory64 maps
    IOOptionBits pending_options = 0;
    UserClient *next = nullptr;

    bool initWithTask(task_t owningTask, void *securityID, UInt32 type) override;
    bool start(IOService *provider) override;
    IOReturn clientClose() override;
    IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *args, IOExternalMethodDispatch *, OSObject *,
                            void *) override;
    IOReturn clientMemoryForType(UInt32 type, IOOptionBits *options, IOMemoryDescriptor **memory) override;
};

// H.264 decoding for any process, VideoToolbox's decoder in apps and in VTDecoderXPCService: a GPU context of its
// own with the decode buffers in it, all gone with the client
class VideoUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(VideoUserClient)

public:
    Gpu *owner = nullptr;
    IOLock *busy = nullptr;  // one call at a time
    uintptr_t ctx = 0;
    struct video_h264 dec = {};
    uint32_t decoded = 0;  // pictures, for the log
    bool counted = false;

    bool start(IOService *provider) override;
    void free() override;
    IOReturn clientClose() override;
    IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *args, IOExternalMethodDispatch *, OSObject *,
                            void *) override;
    IOReturn clientMemoryForType(UInt32 type, IOOptionBits *options, IOMemoryDescriptor **memory) override;
};

class DisplayUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(DisplayUserClient)

public:
    Gpu *owner = nullptr;

    bool start(IOService *provider) override;
    IOReturn clientClose() override;
    IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *args, IOExternalMethodDispatch *, OSObject *,
                            void *) override;
};

MalachiteDefineClass(Gpu, IOService)
MalachiteDefineClass(UserClient, IOUserClient)
MalachiteDefineClass(VideoUserClient, IOUserClient)
MalachiteDefineClass(DisplayUserClient, IOUserClient)
MalachiteDefineClass(SurfaceNub, IOService)

// A file of the root volume read whole into wired memory, or nullptr (not there yet, or not readable)
static IOBufferMemoryDescriptor *read_file(const char *path)
{
    vfs_context_t ctx = vfs_context_create(nullptr);
    vnode_t vp = nullptr;
    IOBufferMemoryDescriptor *buf = nullptr;
    if (ctx && !vnode_open(path, FREAD, 0, 0, &vp, ctx)) {
        struct vnode_attr va;
        VATTR_INIT(&va);
        VATTR_WANTED(&va, va_data_size);
        int resid = 0;
        if (!vnode_getattr(vp, &va, ctx) && va.va_data_size && va.va_data_size < (256ull << 20) &&
            (buf = IOBufferMemoryDescriptor::withCapacity(va.va_data_size, kIODirectionNone)) &&
            (vn_rdwr(UIO_READ, vp, (caddr_t)buf->getBytesNoCopy(), (int)va.va_data_size, 0, UIO_SYSSPACE, 0,
                     vfs_context_ucred(ctx), &resid, vfs_context_proc(ctx)) ||
             resid))
            OSSafeReleaseNULL(buf);
        if (buf)
            buf->setLength(va.va_data_size);
        vnode_close(vp, FREAD, ctx);
    }
    if (ctx)
        vfs_context_rele(ctx);
    return buf;
}

// The GPU brought up as the system starts, as a driver's should be: the firmware read off the root volume once it's
// mounted, GSP-RM booted, the displays found and lit. malachitectl does the same steps by hand (nvnoauto=1 leaves them
// to it); once these have run, its own are no-ops.
void Gpu::autostart(thread_call_param_t p0, thread_call_param_t)
{
    Gpu *self = (Gpu *)p0;
    static const char *const names[] = MALACHITE_FW_NAMES;
    IOBufferMemoryDescriptor *fw[MALACHITE_FW_COUNT] = {};
    for (int tries = 0; tries < 600; tries++) {  // the root volume comes up some seconds into boot
        // looking a path up before the root is mounted panics in namei
        vnode_t root = vfs_rootvnode();
        if (!root) {
            IOSleep(1000);
            continue;
        }
        vnode_put(root);
        int have = 0;
        for (int i = 0; i < MALACHITE_FW_COUNT; i++) {
            char path[128];
            snprintf(path, sizeof(path), "/Library/Application Support/Malachite/570/%s", names[i]);
            if (!fw[i])
                fw[i] = read_file(path);
            have += fw[i] != nullptr;
        }
        if (have == MALACHITE_FW_COUNT)
            break;
        IOSleep(1000);
    }
    if (!fw[0] || !fw[1] || !fw[2] || !fw[3]) {
        log_msg("Malachite: no firmware in /Library/Application Support/Malachite/570, waiting for malachitectl\n");
        for (int i = 0; i < MALACHITE_FW_COUNT; i++)
            OSSafeReleaseNULL(fw[i]);
        return;
    }
    IOLockLock(self->lock);
    for (int i = 0; i < MALACHITE_FW_COUNT; i++) {
        if (self->plat.fw[i])
            fw[i]->release();
        else
            self->plat.fw[i] = fw[i];
    }
    self->read_scaled();
    IOLockUnlock(self->lock);
    static const uint32_t steps[] = { MALACHITE_GSP_START, MALACHITE_DISP_PROBE, MALACHITE_DISP_SHOW };
    uint64_t none[4] = {};
    for (uint32_t s : steps) {
        IOReturn r = self->command(s, none, nullptr);
        if (r != kIOReturnSuccess) {
            log_msg("Malachite: bring-up step %u failed (0x%x)\n", s, r);
            return;
        }
    }
    log_msg("Malachite: GPU and displays up\n");
}

void Gpu::add_scaled_size(uint32_t w, uint32_t h)
{
    uint64_t pitch = (w * 4ull + 255) & ~255ull;
    if (w && h && nscaled < DISP_MAX_SCALED && pitch * h <= BAR1_FB_SIZE)
        scaled[nscaled++] = { w, h };
}

// "2560x1440" a line, the list the app last set
void Gpu::read_scaled()
{
    IOBufferMemoryDescriptor *f = read_file("/Library/Application Support/Malachite/scaled.txt");
    if (!f)
        return;
    const char *p = (const char *)f->getBytesNoCopy(), *end = p + f->getLength();
    while (p < end && nscaled < DISP_MAX_SCALED) {
        uint32_t w = 0, h = 0;
        while (p < end && *p >= '0' && *p <= '9' && w < 100000)
            w = w * 10 + (uint32_t)(*p++ - '0');
        if (p < end && *p == 'x')
            p++;
        while (p < end && *p >= '0' && *p <= '9' && h < 100000)
            h = h * 10 + (uint32_t)(*p++ - '0');
        while (p < end && *p++ != '\n') {
        }
        add_scaled_size(w, h);
    }
    f->release();
}

// The extra resolutions replaced while the driver runs. WindowServer can't have the mode it shows go away under it, so
// a scaled one that is lit stays listed until it has left it. Framebuffer tells macOS as a hot-plugged monitor would.
IOReturn Gpu::set_scaled(const uint64_t *sizes, uint32_t n)
{
    if (n % 2 || n > 2 * DISP_MAX_SCALED)
        return kIOReturnBadArgument;
    take();
    nscaled = 0;
    for (uint32_t i = 0; i < n; i += 2)
        add_scaled_size((uint32_t)sizes[i], (uint32_t)sizes[i + 1]);
    if (disp) {
        struct disp_mode lit[SCREENS] = {};
        for (screen &sc : screens) {
            if (!sc.shown || !sc.mode)
                continue;
            const struct disp_mode &m = disp->out[sc.out].modes[sc.mode - 1];
            lit[index(sc)] = m;
            bool kept = !m.view_w;
            for (uint32_t i = 0; i < nscaled; i++)
                kept |= scaled[i].w == m.view_w && scaled[i].h == m.view_h;
            if (!kept)
                add_scaled_size(m.view_w, m.view_h);
        }
        disp_set_scaled(disp, scaled, nscaled);
        for (screen &sc : screens) {
            if (!sc.shown || !sc.mode)
                continue;
            const struct disp_output *o = &disp->out[sc.out];
            for (uint32_t i = 0; i < o->nmodes; i++)
                if (!memcmp(&o->modes[i], &lit[index(sc)], sizeof(o->modes[i])))
                    sc.mode = i + 1;
            publish_surface(sc, false);
            sc.changed = true;
        }
    }
    give();
    tell_nubs();
    return kIOReturnSuccess;
}

// What GSP-RM says the GPU is doing, for kern.malachite_power, next to what the driver itself has sent it (busy time is
// ours, from the semaphores). TryLock so a reader never waits behind a launch.
size_t Gpu::power_report(char *out, size_t cap)
{
    if (!IOLockTryLock(lock))
        return snprintf(out, cap, "busy, try again\n");
    size_t n = 0;
    if (!gsp || suspended) {
        IOLockUnlock(lock);
        return snprintf(out, cap, "GSP-RM is not running\n");
    }
    struct pwr p;
    pwr_read(gsp, &p);
    busy_step(0, time_us());
    uint64_t busy = __atomic_load_n(&busy_total, __ATOMIC_RELAXED);
    n += snprintf(out + n, cap - n, "nvpm %#x capped %d\n", pm, capped);
    if (p.pstate_status)
        n += snprintf(out + n, cap - n, "pstate error %#x\n", p.pstate_status);
    else if (p.pstate == ~0u)
        n += snprintf(out + n, cap - n, "pstate none\n");
    else
        n += snprintf(out + n, cap - n, "pstate P%u\n", p.pstate);
    if (p.clk_status) {
        n += snprintf(out + n, cap - n, "clocks error %#x\n", p.clk_status);
    } else {
        // the mark of a clock the chip doesn't have is all ones
        const char *names[] = { "gpc", "dram", "nvd" };
        const uint32_t khz[] = { p.gpc_khz, p.dram_khz, p.nvd_khz };
        n += snprintf(out + n, cap - n, "clocks");
        for (int i = 0; i < 3; i++)
            if (khz[i] != ~0u)
                n += snprintf(out + n, cap - n, " %s %u MHz", names[i], khz[i] / 1000);
        n += snprintf(out + n, cap - n, "\n");
    }
    n += snprintf(out + n, cap - n, "busy_us %llu now_us %llu\n", (unsigned long long)busy,
                  (unsigned long long)time_us());
    n += snprintf(out + n, cap - n, "present %llu draw %llu dispatch %llu copy %llu copy2d %llu fill %llu\n",
                  (unsigned long long)stats[ST_PRESENT], (unsigned long long)stats[ST_DRAW],
                  (unsigned long long)stats[ST_DISPATCH], (unsigned long long)stats[ST_COPY],
                  (unsigned long long)stats[ST_COPY2D], (unsigned long long)stats[ST_FILL]);
    n += accel::pace_report(out + n, cap - n);
    IOLockUnlock(lock);
    return n;
}

static Gpu *power_owner;

static int power_sysctl(struct sysctl_oid *, void *, int, struct sysctl_req *req)
{
    if (!kauth_cred_issuser(kauth_cred_get()))
        return EPERM;
    char out[2048];
    size_t n = power_owner ? power_owner->power_report(out, sizeof(out)) : snprintf(out, sizeof(out), "no GPU\n");
    return SYSCTL_OUT(req, out, n < sizeof(out) ? n : sizeof(out) - 1);
}

SYSCTL_PROC(_kern, OID_AUTO, malachite_power, CTLTYPE_OPAQUE | CTLFLAG_RD | CTLFLAG_LOCKED, nullptr, 0, power_sysctl, "S",
            "the GPU's P-state, clocks and busy time");

// Once a second, under the lock: GSP-RM told whether the GPU has had work since the last time
void Gpu::idle_tick()
{
    uint64_t now = time_us();
    if (!(pm & PM_IDLE_CAP) || !gsp || suspended || now - cap_mark < 500000)
        return;
    const uint64_t work = stats[ST_PRESENT] + stats[ST_DRAW] + stats[ST_DISPATCH] + stats[ST_COPY2D] + stats[ST_COPY] +
                          stats[ST_FILL];
    busy_step(0, now);
    const uint64_t busy = __atomic_load_n(&busy_total, __ATOMIC_RELAXED);
    const bool idle = work == cap_work && busy == cap_busy;
    uint64_t span = cap_mark ? now - cap_mark : 0, used = busy - cap_busy;
    if (used > span)
        used = span;
    if (!pwr_idle_cap(gsp, idle, span - used, used))
        capped = idle;
    cap_mark = now;
    cap_work = work;
    cap_busy = busy;
}

// A start that fails leaves them behind, and a second one registering an oid twice panics (one did, at boot): once
// each, and an oid keeps the number it got, which may be someone else's by the time it's back.
static bool sysctls_up;

static struct sysctl_oid *const sysctls[] = {
    &sysctl__kern_malachite_power, &sysctl__kern_malachite_stats, &sysctl__kern_malachite_prof_on,
    &sysctl__kern_malachite_compress, &sysctl__kern_malachite_zcull, &sysctl__kern_malachite_prof,
};

static void sysctls_register()
{
    if (!__sync_bool_compare_and_swap(&sysctls_up, false, true))
        return;
    log_sysctl_register();
    for (struct sysctl_oid *o : sysctls) {
        o->oid_number = OID_AUTO;
        sysctl_register_oid(o);
    }
}

static void sysctls_unregister()
{
    if (!__sync_bool_compare_and_swap(&sysctls_up, true, false))
        return;
    log_sysctl_unregister();
    for (struct sysctl_oid *o : sysctls)
        sysctl_unregister_oid(o);
}

bool Gpu::start(IOService *provider)
{
    // leaves the GPU to macOS, for getting back in when a driver build doesn't boot
    int off;
    if (PE_parse_boot_argn("-nonvidia", &off, sizeof(off)) || oldDriverLoaded())
        return false;
    if (!IOService::start(provider))
        return false;
    power_owner = this;
    pm = pm_flags();
    if (pm)
        log_msg("Malachite: nvpm %#x\n", pm);
    sysctls_register();
    pump_call = thread_call_allocate(pump, this);

    plat.pci = OSDynamicCast(IOPCIDevice, provider);
    if (!plat.pci)
        return false;
    if (!plat.pci->getDeviceMemoryWithRegister(kIOPCIConfigBaseAddress1)) {
        uint64_t base, len;
        if (pci_place_bar1(plat.pci, &base, &len)) {
            plat.bar1_base = base;
            plat.bar1_len = len;
        }
    }
    plat.pci->setMemoryEnable(true);
    plat.pci->setBusMasterEnable(true);
    bar0 = plat.pci->mapDeviceMemoryWithRegister(kIOPCIConfigBaseAddress0);
    if (!bar0) {
        log_msg("Malachite: BAR0 map failed\n");
        return false;
    }
    gpu.bar0 = (volatile uint8_t *)bar0->getVirtualAddress();
    gpu.plat = &plat;
    uint32_t boot0 = rd32(&gpu, NV_PMC_BOOT_0), chipset = NV_CHIPSET(boot0);
    log_msg("Malachite: BAR0 %llu MB, PMC_BOOT_0 0x%08x chipset 0x%x\n", bar0->getLength() >> 20, boot0, chipset);
    gpu.chip = chip_find(chipset);
    if (!gpu.chip) {
        log_msg("Malachite: chipset 0x%x not supported\n", chipset);
        OSSafeReleaseNULL(bar0);
        return false;
    }
    // what System Information and apps read the GPU's memory from, on the PCI device as other GPUs' drivers put it
    // (without it: 16 MB, the boot framebuffer's); the usable size the VBIOS left, in MB
    if (uint64_t mb = rd32(&gpu, 0x1183a4)) {
        plat.pci->setProperty("VRAM,totalMB", mb, 32);
        plat.pci->setProperty("VRAM,totalsize", mb << 20, 64);
    }
    log_msg("Malachite: %s\n", gpu.chip->family);
    // what Accelerator and the Metal plugin tell apps: the VRAM's size (in MB from the FB controller, readable before
    // GSP-RM is up), and once it is, the GPU's name
    setProperty("MalachiteVRAMBytes", (unsigned long long)rd32(&gpu, 0x1183a4) << 20, 64);
    lock = IOLockAlloc();
    evlock = IOLockAlloc();

    loop = IOWorkLoop::workLoop();
    timer = loop ? IOTimerEventSource::timerEventSource(this, poll) : nullptr;
    if (!timer || loop->addEventSource(timer) != kIOReturnSuccess)
        return false;

    // the MSI if there is one; without it the timer does the work
    for (int i = 0;; i++) {
        int type = 0;
        if (provider->getInterruptType(i, &type) != kIOReturnSuccess)
            break;
        if (type & kIOInterruptTypePCIMessaged) {
            irq = IOInterruptEventSource::interruptEventSource(this, interrupt, provider, i);
            break;
        }
    }
    if (irq && loop->addEventSource(irq) == kIOReturnSuccess)
        irq->enable();
    else
        OSSafeReleaseNULL(irq);
    log_msg("Malachite: %s\n", irq ? "MSI interrupts" : "no MSI, polling");

    // in the power plane: setPowerState() suspends and resumes the GPU around system sleep, and systemWillShutdown()
    // gets called
    static IOPMPowerState states[2] = {
        { kIOPMPowerStateVersion1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { kIOPMPowerStateVersion1, kIOPMPowerOn | kIOPMDeviceUsable, kIOPMPowerOn, kIOPMPowerOn, 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    PMinit();
    provider->joinPMtree(this);
    registerPowerDriver(this, states, 2);
    changePowerStateTo(1);


    registerService();
    create_nubs();
    uint32_t manual = 0;
    PE_parse_boot_argn("nvnoauto", &manual, sizeof(manual));
    PE_parse_boot_argn("nvsafesync", &safe_sync, sizeof(safe_sync));
    if (!manual && (autostart_call = thread_call_allocate(autostart, this)))
        thread_call_enter(autostart_call);
    return true;
}

// thread_call_cancel_wait isn't exported to kexts in a kernel collection; free fails while the call runs
static void free_call(thread_call_t &call)
{
    if (!call)
        return;
    thread_call_cancel(call);
    while (!thread_call_free(call))
        IOSleep(1);
    call = nullptr;
}

void Gpu::stop(IOService *provider)
{
    sysctls_unregister();
    power_owner = nullptr;
    free_call(pump_call);
    free_call(autostart_call);
    if (lock)
        IOLockLock(lock);
    stop_gsp();
    if (lock)
        IOLockUnlock(lock);
    for (screen &sc : screens)  // the framebuffers on them go first
        if (sc.nub) {
            sc.nub->terminate(kIOServiceSynchronous);
            OSSafeReleaseNULL(sc.nub);
        }
    for (screen &sc : screens) {
        if (sc.cursor)
            IOFree(sc.cursor, CURS_MAX * CURS_MAX * 4);
        if (sc.gamma)
            IOFree(sc.gamma, SURFACE_GAMMA_ENTRIES * 3 * sizeof(uint16_t));
    }
    if (timer) {
        timer->cancelTimeout();
        loop->removeEventSource(timer);
    }
    if (irq) {
        irq->disable();
        loop->removeEventSource(irq);
    }
    OSSafeReleaseNULL(irq);
    OSSafeReleaseNULL(timer);
    OSSafeReleaseNULL(loop);
    for (int i = 0; i < MALACHITE_FW_COUNT; i++)
        OSSafeReleaseNULL(plat.fw[i]);
    if (lock)
        IOLockFree(lock);
    if (evlock)
        IOLockFree(evlock);
    OSSafeReleaseNULL(plat.bar1);
    OSSafeReleaseNULL(bar0);
    PMstop();
    IOService::stop(provider);
}

// a restart with GSP-RM still running leaves the card needing a power cycle
void Gpu::systemWillShutdown(IOOptionBits specifier)
{
    IOLockLock(lock);
    if (gsp)
        log_msg("Malachite: shutting GSP-RM down for system shutdown\n");
    stop_gsp();
    IOLockUnlock(lock);
    IOService::systemWillShutdown(specifier);
}

// System sleep: the GPU suspended before it loses power and resumed after. Framebuffer's nubs are power children, so the
// framebuffers are down first and up last.
IOReturn Gpu::setPowerState(unsigned long ordinal, IOService *device)
{
    IOLockLock(lock);
    if (ordinal)
        resume();
    else
        suspend();
    IOLockUnlock(lock);
    tell_nubs();
    return kIOPMAckImplied;
}

// under the lock: a power cycle in progress waited out, 30 s at most; false if it isn't over
bool Gpu::awake()
{
    uint64_t deadline;
    clock_interval_to_deadline(30, kSecondScale, &deadline);
    while (suspended && IOLockSleepDeadline(lock, &suspended, deadline, THREAD_UNINT) != THREAD_TIMED_OUT)
        ;
    return !suspended;
}

// A power cycle unloads GSP-RM and boots it afresh from the firmware in plat.fw[]. What lived in it is made anew on
// what Gpu keeps itself: each context's page tables, mappings and buffers, every VRAM allocation (the addresses
// Accelerator holds stay valid) and the screens' surfaces. VRAM contents are not saved.
//
// Under the lock: the work drained, the heads blanked, the display and every context's client freed, then GSP-RM
// unloaded with WPR2 released. Framebuffer stays online: dark, it would look like an unplug to WindowServer.
void Gpu::suspend()
{
    if (!gsp || suspended)
        return;
    uint64_t t0 = time_us();
    suspended = true;
    capped = false;
    timer->cancelTimeout();
    if (irq)
        irq->disable();
    drain_all();
    for (int i = 0; i < 1000 && present_fence && !copy_done(present_fence); i++)
        delay_us(100);
    for (screen &sc : screens) {
        if (!disp || !sc.shown)
            continue;
        if (!sc.asleep && !sc.unplugged)
            disp_blank(disp, index(sc), sc.out);
        disp_vblank(disp, index(sc), 0);
    }
    for (UserClient *uc = clients; uc; uc = uc->next)
        uc->file = nullptr;
    if (fe)
        rmfe_destroy(fe);
    fe = nullptr;
    // GSP-RM is unloaded outright for a cold boot at resume: suspended with its state (gsp_suspend) it came back with
    // working channels but refused the display a core channel, with the display objects kept (INVALID_STATE on the
    // first update) or freed (NOT_SUPPORTED).
    if (disp) {
        disp_fini(disp);
        disp_modeset_fini(disp);
        IOFree(disp, sizeof(*disp));
        disp = nullptr;
    }
    for (screen &sc : screens)
        reset_flips(sc);
    canary = vram_alloc(gsp_vram(gsp), 0x1000, 0x1000);
    for (int i = 0; canary && i < 4; i++)
        vram_wr32(&gpu, canary + i * 4, 0x4e565253u ^ i);
    forget_contexts(true);
    if (gsp_unload(gsp))
        log_msg("Malachite: GSP-RM did not unload cleanly, the boot cleans up after it\n");
    disp_release_channels(&gpu);
    log_msg("Malachite: GPU suspended in %llu ms\n", (time_us() - t0) / 1000);
}

// Under the lock: GSP-RM resumed (or booted again), each context's channels made anew on its address space if they
// didn't survive, the display probed and every screen lit as it was. A GPU that doesn't come back is let go of as
// MALACHITE_GSP_STOP would.
IOReturn Gpu::resume()
{
    if (!suspended)
        return kIOReturnSuccess;
    uint64_t t0 = time_us();
    if (plat.bar1_len)
        pci_restore_bar1(plat.pci, plat.bar1_base);
    plat.pci->setMemoryEnable(true);
    plat.pci->setBusMasterEnable(true);
    uint32_t boot0 = rd32(&gpu, NV_PMC_BOOT_0);
    int up = boot0 != 0xffffffff ? gsp_resume(gsp) : -1;
    uint64_t t_gsp = time_us();
    bool vram_kept = canary != 0;
    for (int i = 0; canary && i < 4; i++)
        vram_kept &= vram_rd32(&gpu, canary + i * 4) == (0x4e565253u ^ i);
    if (canary)
        vram_free(gsp_vram(gsp), canary, 0x1000);
    canary = 0;
    if (irq)
        irq->enable();
    suspended = false;
    IOLockWakeup(lock, &suspended, false);
    if (up < 0) {
        log_msg("Malachite: the GPU did not come back (PMC_BOOT_0 %#x), it is lost until a restart\n", boot0);
        forget_contexts(false);
        if (disp) {
            disp_forget(disp);
            IOFree(disp, sizeof(*disp));
            disp = nullptr;
        }
        stop_gsp();
        return kIOReturnNotResponding;
    }
    if (up == 0 && !vram_kept) {
        // GSP-RM's channels survived, but not the instance blocks and contexts of ours in VRAM they run on
        log_msg("Malachite: VRAM lost its contents, making the GPU contexts anew\n");
        forget_contexts(true);
        gr_forget(gsp);
    } else if (up == 1) {
        forget_contexts(false);
    }
    log_msg("Malachite: GSP-RM %s, VRAM %s\n", up ? "booted afresh" : "resumed with its state",
            vram_kept ? "kept" : "lost");
    fe = rmfe_create(gsp);
    // the kernel's context first: the tasks' take their addresses from its address space
    for (int i = -1; i < MAX_CTXS; i++) {
        gpu_ctx *c = i < 0 ? &kctx : ctxs[i];
        if (c && c->copy_sem && !(c->compute_sem ? compute_bridge_init(c) : gpu_bridge_init(c)))
            log_msg("Malachite: %s's GPU context not back, its next use tries again\n", c == &kctx ? "kernel" : c->owner);
    }
    uint64_t t_ctx = time_us();

    if (!disp && (want_shown || lit())) {
        disp = (struct disp *)IOMallocZero(sizeof(*disp));
        if (disp && disp_probe(gsp, disp, scaled, nscaled)) {
            disp_fini(disp);
            IOFree(disp, sizeof(*disp));
            disp = nullptr;
        }
    }
    for (screen &sc : screens) {
        if (!disp || !sc.shown || sc.unplugged)
            continue;
        const int head = index(sc);
        const struct disp_output *o = &disp->out[sc.out];
        if (sc.out >= disp->n || !o->connected || sc.mode > o->nmodes) {  // gone meanwhile: hotplug's to light again
            sc.unplugged = true;
            sc.nub->setProperty(SURFACE_ONLINE, kOSBooleanFalse);
            sc.changed = true;
            continue;
        }
        const struct disp_mode *t = o->nmodes ? &o->modes[sc.mode - 1] : &o->mode;
        vram_zero(&gpu, sc.shown, sc.size);  // black until drawn again; after a sleep IOFramebuffer puts its copy back
        if (int r = modeset(sc, t)) {
            log_msg("Malachite: screen %d's modeset after the power cycle failed\n", head);
            if (r != DISP_NO_OUTPUT)
                disp_reset(disp);
            continue;
        }
        disp_vblank(disp, head, 1);
        if (disp_cursor_init(disp, head) == 0) {
            if (sc.cursor_w && disp_cursor_image(disp, head, sc.cursor, sc.cursor_w, sc.cursor_h) == 0)
                disp_cursor_move(disp, head, sc.cursor_x, sc.cursor_y);
            else
                disp_cursor_move(disp, head, -CURS_OFFSCREEN, -CURS_OFFSCREEN);
        }
        if (sc.gamma)  // the head's LUTs were made anew, as identities
            disp_gamma(disp, head, sc.gamma);
        if (sc.asleep)
            disp_blank(disp, head, sc.out);
        // presents' flips release their semaphore in the new notifier memory
        if (sc.sema_va && vmm_map_vram(&kctx.vm, sc.sema_va, disp->ntfy, 0x1000, 0))
            sc.sema_va = 0;
        publish_surface(sc, false);
    }
    vmm_flush(&gpu);
    if (disp)
        disp->hotplug = ~0u;  // what is connected now, looked at by the next poll
    else if (lit())
        log_msg("Malachite: no display after the power cycle\n");
    if (irq)
        gsp_irq_enable(gsp, 1);
    timer->setTimeoutMS(20);
    log_msg("Malachite: GPU resumed in %llu ms (GSP-RM %llu, contexts %llu, display %llu)\n", (time_us() - t0) / 1000,
            (t_gsp - t0) / 1000, (t_ctx - t_gsp) / 1000, (time_us() - t_ctx) / 1000);
    return kIOReturnSuccess;
}

// our side of a context's channels once GSP-RM is gone with them: their memory unmapped and freed, the work state
// reset. The address space, its mappings and the context's buffers stay for ctx_rebind().
void Gpu::ctx_forget(gpu_ctx *c)
{
    if (c->compute_ready) {
        chan_free(&c->compute_ch);
        gr_ctx_free(gsp, &c->compute_ctx);
    }
    if (c->gpu_ready) {
        chan_free(&c->copy_ch);
    }
    if (c->video_ready) {
        gsp_chid_put(gsp, c->video.ch.chid);
        video_free(&c->video);
        c->video_ready = false;
    }
    c->gpu_ready = c->compute_ready = false;
    c->copy_faulted = false;
    c->compute_faulted = false;
    c->draw_pending = 0;
    memset(c->draw_slot_seq, 0, sizeof(c->draw_slot_seq));
    memset(c->dispatch_slot_seq, 0, sizeof(c->dispatch_slot_seq));
    memset(c->pass_targets, 0, sizeof(c->pass_targets));
    memset(c->half_seq, 0, sizeof(c->half_seq));
    c->push_half = 0;
    c->issued_next = 0;
}

// a context's client and copy channel again after a power cycle, its kept page tables handed to the new GSP-RM:
// every mapping stays where it was, and the channels keep their IDs
bool Gpu::ctx_rebind(gpu_ctx *c)
{
    if (rm_client(gsp, &c->client) || vmm_rebind(gsp, &c->client, &c->vm) ||
        !copy_channel(c, c->copy_ch.chid, c->copy_ch.handle))
        return false;
    vmm_flush(&gpu);
    c->gpu_ready = true;
    log_msg("Malachite: GPU context %s back (copy channel %u)\n", c == &kctx ? "kernel" : c->owner, c->copy_ch.chid);
    return true;
}

bool Gpu::copy_channel(gpu_ctx *c, uint32_t chid, uint32_t handle)
{
    uint32_t engine, runlist;
    if (chan_find_engine(gsp, &c->client, CHAN_COPY, &engine, &runlist) ||
        chan_create(gsp, &c->client, &c->vm, engine, runlist, chid, handle, c == &kctx, &c->copy_ch))
        return false;
    struct {
        uint32_t version, engineType;
    } ce = { 1, engine };
    if (chan_schedule(&c->copy_ch) ||
        rm_alloc(gsp, c->client.client, c->copy_ch.handle, 0xc7b50000, gpu.chip->copy, &ce, sizeof(ce))) {
        rm_free(gsp, c->client.client, c->copy_ch.handle);
        chan_free(&c->copy_ch);
        return false;
    }
    return true;
}

// a graphics channel (3D and compute) for a context whose buffers are mapped already: their pools and its local
// memory bound on it
bool Gpu::graphics_channel(gpu_ctx *c, uint32_t chid)
{
    if (compute_setup(gsp, &c->client, &c->vm, chid, c == &kctx, &c->compute_ch, &c->compute_ctx) ||
        render_setup(gsp, &c->client, &c->compute_ch)) {
        // what failed may be any step: the channel GSP-RM holds, if it got that far, goes before its memory
        if (c->compute_ch.res.inst)
            rm_free(gsp, c->client.client, c->compute_ch.handle);
        gr_ctx_free(gsp, &c->compute_ctx);
        chan_free(&c->compute_ch);
        return false;
    }
    vmm_flush(&gpu);
    compute_set_tex_pools(&c->compute_ch, c->compute_cbufva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES,
                          c->compute_cbufva + TEX_SAMPLER_POOL, NVGPU_MAX_TEXTURES);
    render_set_tex_pools(&c->compute_ch, c->compute_cbufva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES,
                         c->compute_cbufva + TEX_SAMPLER_POOL, NVGPU_MAX_TEXTURES);
    if (c->slm_va) {
        compute_set_local_memory(&c->compute_ch, c->slm_va, c->slm_per_tpc);
        render_set_local_memory(&c->compute_ch, c->slm_va, c->slm_size, c->slm_per_warp);
    }
    return true;
}

// every context's channels let go of on our side, with free_rm their clients in GSP-RM too: GSP-RM then unloads with
// nothing of ours left in it
void Gpu::forget_contexts(bool free_rm)
{
    for (int i = -1; i < MAX_CTXS; i++) {
        gpu_ctx *c = i < 0 ? &kctx : ctxs[i];
        if (!c)
            continue;
        if (free_rm && c->gpu_ready)
            rm_free(gsp, c->client.client, c->client.client);
        ctx_forget(c);
    }
}

// Every context and mapping let go of and the kernel's emptied, before GSP-RM goes for good: Accelerator's handles go
// stale and the next GSP-RM starts from nothing. Under the lock.
void Gpu::drop_contexts()
{
    for (int i = 0; i < gpu_maps_top; i++) {
        struct dma *d = &gpu_maps[i].dma;
        if (d->handle) {
            ((IOMemoryDescriptor *)d->handle)->complete();
            ((IOMemoryDescriptor *)d->handle)->release();
        }
        mem_free(d->pages);
        memset(&gpu_maps[i], 0, sizeof(gpu_maps[i]));
    }
    ndeferred = 0;
    if (map_hash)
        memset(map_hash, 0, sizeof(int32_t) * gpu_maps_cap);
    gpu_maps_free = gpu_maps_top = 0;
    for (gpu_ctx *&c : ctxs) {
        if (!c)
            continue;
        ctx_forget(c);
        vmm_destroy(&c->vm);
        IOFree(c, sizeof(*c));
        c = nullptr;
    }
    ctx_forget(&kctx);
    vmm_destroy(&kctx.vm);
    memset(&kctx, 0, sizeof(kctx));
    for (screen &sc : screens)
        sc.scanout_va = sc.back_va[0] = sc.back_va[1] = sc.sema_va = 0;
    for (present_slot &p : present)
        p.va = 0;
}

// Framebuffer's last cursor image: the head's copy goes with the display state in a power cycle
void Gpu::keep_cursor(screen &sc, const uint32_t *argb, uint32_t w, uint32_t h)
{
    if (!sc.cursor)
        sc.cursor = (uint32_t *)IOMalloc(CURS_MAX * CURS_MAX * 4);
    if (!sc.cursor || w > CURS_MAX || h > CURS_MAX)
        return;
    memcpy(sc.cursor, argb, w * h * 4);
    sc.cursor_w = w;
    sc.cursor_h = h;
}

// under the lock; a head that is off gets the table when it wakes
IOReturn Gpu::set_gamma(screen &sc, const uint16_t *rgb)
{
    const size_t size = SURFACE_GAMMA_ENTRIES * 3 * sizeof(uint16_t);
    if (!sc.gamma)
        sc.gamma = (uint16_t *)IOMalloc(size);
    if (!sc.gamma)
        return kIOReturnNoMemory;
    memcpy(sc.gamma, rgb, size);
    if (!disp || !sc.shown || sc.asleep || sc.unplugged)
        return kIOReturnSuccess;
    return disp_gamma(disp, index(sc), sc.gamma) ? kIOReturnError : kIOReturnSuccess;
}

// the screen's surface onto its head again, or fb, a surface of the same size
int Gpu::modeset(screen &sc, const struct disp_mode *t, uint64_t fb)
{
    reset_flips(sc, sc.cur);  // the modeset leaves no flip queued
    return disp_modeset(disp, index(sc), sc.out, t, fb ? fb : sc.shown, sc.pitch);
}

// a slow sweep behind the interrupt, every 20 ms when there is none
void Gpu::poll(OSObject *owner, IOTimerEventSource *sender)
{
    Gpu *self = (Gpu *)owner;
    IOLockLock(self->lock);
    if (self->fe) {
        rmfe_poll(self->fe);
        // a monitor that GSP-RM posts no plug event for: asked about every couple of seconds while none is lit
        if (self->disp && self->want_shown && !self->lit() && ++self->unlit_polls >= (self->irq ? 2u : 100u)) {
            self->unlit_polls = 0;
            self->disp->hotplug = ~0u;
        }
        self->check_hotplug();
        self->idle_tick();
        sender->setTimeoutMS(self->irq ? 1000 : 20);
    }
    IOLockUnlock(self->lock);
    self->tell_nubs();
}

// Framebuffer told of the screens whose properties changed, outside the lock: it may set a mode in response
void Gpu::tell_nubs()
{
    for (screen &sc : screens) {
        IOLockLock(lock);
        bool tell = sc.changed;
        sc.changed = false;
        IOLockUnlock(lock);
        if (tell && sc.nub)
            sc.nub->messageClients(kSurfaceChanged);
    }
}

void Gpu::vblank_fns(int i)
{
    screen &sc = screens[i];
    IOLockLock(evlock);
    if (sc.vbl_fn)
        sc.vbl_fn(sc.vbl_ctx);
    if (sc.pipe_fn)
        sc.pipe_fn(sc.pipe_ctx);
    IOLockWakeup(evlock, &vbl_wake, false);
    IOLockUnlock(evlock);
}

void Gpu::interrupt(OSObject *owner, IOInterruptEventSource *sender, int count)
{
    Gpu *self = (Gpu *)owner;
    IOLockLock(self->lock);
    int events = self->gsp && !self->suspended ? gsp_irq(self->gsp) : 0;
    if ((events & (GSP_IRQ_MSG | GSP_IRQ_GR)) && self->fe)
        rmfe_poll(self->fe);
    if (events & GSP_IRQ_MSG)
        self->check_hotplug();
    IOLockUnlock(self->lock);
    self->tell_nubs();
    for (int i = 1; i < SCREENS; i++)
        if (events & GSP_IRQ_VBLANK(i))
            self->vblank_fns(i);
    if (events & GSP_IRQ_VBLANK(0)) {
        self->vblank_fns(0);
        uint64_t now = mach_absolute_time();
        if (++self->vblanks % 600 == 1) {
            if (self->vblanks > 1) {
                uint64_t ns;
                absolutetime_to_nanoseconds(now - self->vblank_mark, &ns);
                log_msg("Malachite: 600 vblanks in %llu ms\n", ns / 1000000);
            }
            self->vblank_mark = now;
        }
    }
    if (events & GSP_IRQ_GR) {
        IOLockLock(self->evlock);
        self->gr_events++;
        IOLockWakeup(self->evlock, &self->gr_events, false);
        IOLockUnlock(self->evlock);
    }
}

// nvrm_xnu_wait_event: until a graphics nonstall interrupt has come in since seen (or the timeout); the count now
uint32_t Gpu::wait_event(uint32_t seen, uint32_t timeout_us)
{
    IOLockLock(evlock);
    if (timeout_us && gr_events == seen) {
        uint64_t deadline;
        clock_interval_to_deadline(timeout_us, kMicrosecondScale, &deadline);
        IOLockSleepDeadline(evlock, &gr_events, deadline, THREAD_INTERRUPTIBLE);
    }
    uint32_t now = gr_events;
    IOLockUnlock(evlock);
    return now;
}

// under the lock; open connections lose their RM state with it
void Gpu::stop_gsp()
{
    if (timer)
        timer->cancelTimeout();
    for (UserClient *uc = clients; uc; uc = uc->next)
        uc->file = nullptr;
    darken_fb();  // before the surfaces go
    if (disp) {
        for (screen &sc : screens)
            if (sc.shown)
                disp_vblank(disp, index(sc), 0);
        disp_modeset_fini(disp);
        disp_fini(disp);
        IOFree(disp, sizeof(*disp));
    }
    disp = nullptr;
    want_shown = false;
    for (int i = 0; i < 1000 && kctx.copy_sem && present_fence && !copy_done(present_fence); i++)
        delay_us(100);  // the last frame's copy still reading its slot
    for (int i = 0; i < PRESENT_SLOTS; i++) {
        struct present_slot *s = &present[i];
        if (!s->md)
            continue;
        mem_free(s->dma.pages);
        s->md->complete();
        s->md->release();
        s->md = nullptr;
    }
    for (shadow_copy &sh : shadows) {
        if (sh.vram)
            mem_free(sh.pages);
        sh = {};
    }
    for (present_slot &ps : present)
        ps.shadow = 0;
    for (int i = 0; i < gpu_maps_top; i++)
        gpu_maps[i].shadow = 0;
    shadow_va = 0;
    for (screen &sc : screens) {
        for (uint64_t &b : sc.back) {
            if (b)
                vram_free(gsp_vram(gsp), b, sc.back_size);
            b = 0;
        }
        sc.shown = sc.scanout_vram = sc.back_size = 0;
        sc.asleep = sc.unplugged = false;
        reset_flips(sc);
    }
    if (fe)
        rmfe_destroy(fe);
    fe = nullptr;
    if (gsp)
        forget_contexts(true);
    drop_contexts();
    if (gsp) {
        gsp_off_rc(gsp, this);
        gsp_stop(gsp);
    }
    gsp = nullptr;
}

// the context's client, address space and copy channel, made once and kept
bool Gpu::gpu_bridge_init(gpu_ctx *c)
{
    if (c->gpu_ready && (c->copy_faulted || c->compute_faulted))
        recover(c);
    if (c->gpu_ready)
        return true;
    if (!gsp || suspended)
        return false;
    if (c->copy_sem)  // after a power cycle
        return ctx_rebind(c);
    if (!gpu_maps && !map_grow())
        return false;
    // a task's context hands out addresses from the kernel's, so each VA names one mapping
    if (c != &kctx && !gpu_bridge_init(&kctx))
        return false;
    if (rm_client(gsp, &c->client))
        return false;
    if (vmm_create(gsp, &c->client, 0x90f20000, 0, &c->vm))
        return false;
    if (c != &kctx)
        c->vm.va_from = &kctx.vm;
    int chid = c == &kctx ? 1 : gsp_chid_get(gsp, FIRST_CTX_CHID);
    if (chid < 0 || !copy_channel(c, (uint32_t)chid, 0xc56f0000 | (uint32_t)chid))
        return false;
    c->copy_sem = gsp_dma(gsp, 0x1000, 1);
    if (!c->copy_sem)
        return false;
    if (c == &kctx) {
        present_fence = 0;  // a new semaphore word, which no earlier present will write
        for (screen &sc : screens)
            sc.fence = 0;
        for (present_slot &ps : present)
            ps.fence = 0;
    }
    c->copy_semva = vmm_alloc(&c->vm, c->copy_sem->size, 0x1000);
    if (!c->copy_semva || vmm_map_sys(&c->vm, c->copy_semva, c->copy_sem, 0, c->copy_sem->size, 0))
        return false;
    c->gpu_ready = true;
    log_msg("Malachite: GPU context %s ready (copy channel %u, sem va 0x%llx)\n", c == &kctx ? "kernel" : c->owner,
            c->copy_ch.chid, c->copy_semva);
    return true;
}

// a Metal buffer's pages mapped into the context's address space; the GPU VA out
IOReturn Gpu::gpu_map(gpu_ctx *c, IOMemoryDescriptor *md, uint64_t *outva)
{
    if (!md || !outva || !gpu_bridge_init(c))
        return kIOReturnNotReady;

    int slot = map_alloc();
    if (slot < 0)
        return kIOReturnNoResources;

    struct dma *d = &gpu_maps[slot].dma;
    memset(d, 0, sizeof(*d));
    if (md->prepare() != kIOReturnSuccess)
        return kIOReturnError;
    size_t size = (md->getLength() + 0xfff) & ~(size_t)0xfff;
    d->size = size;
    d->handle = md;  // not owned; the buffer owns it
    d->pages = (uint64_t *)mem_alloc((size >> 12) * sizeof(uint64_t));
    if (!d->pages) {
        md->complete();
        return kIOReturnNoMemory;
    }
    for (size_t off = 0; off < size; off += 0x1000)
        d->pages[off >> 12] = md->getPhysicalSegment(off, nullptr, kIOMemoryMapperNone);
    d->addr = d->pages[0];

    uint64_t va = map_va(c, size);
    int which = 0;
    uint64_t copy = shadow_at(d->pages, size >> 12, &which);  // a surface already moved to VRAM maps there
    if (!va || (copy ? vmm_map_vram(&c->vm, va, copy, size, 0) : vmm_map_sys(&c->vm, va, d, 0, size, 0))) {
        if (va)
            map_va_free(c, va, size);
        mem_free(d->pages);
        md->complete();
        memset(d, 0, sizeof(*d));
        return kIOReturnError;
    }
    gpu_maps[slot].shadow = (uint8_t)which;
    map_slack(c, va + size);
    vmm_flush(&gpu);  // invalidate the GPU TLB so the new PTEs are visible (else later copies MMU-fault "not valid")
    md->retain();  // until the unmap, which may come after the buffer's owner is done with it
    gpu_maps[slot].ctx = c;
    gpu_maps[slot].va = va;
    map_link(slot);
    *outva = va;
    return kIOReturnSuccess;
}

IOReturn Gpu::gpu_map_vram(gpu_ctx *c, uint64_t vram, uint64_t size, uint64_t *outva)
{
    if (!vram || !size || !outva || !gpu_bridge_init(c))
        return kIOReturnBadArgument;
    int slot = map_alloc();
    if (slot < 0)
        return kIOReturnNoResources;
    size = (size + 0xfff) & ~0xfffull;
    uint64_t va = size >= 0x10000 ? map_va_big(c, size) : map_va(c, size);
    if (!va || vmm_map_vram(&c->vm, va, vram, size, 0)) {
        if (va)
            map_va_free(c, va, size);
        return kIOReturnError;
    }
    memset(&gpu_maps[slot], 0, sizeof(gpu_maps[slot]));
    map_slack(c, va + size);
    vmm_flush(&gpu);
    gpu_maps[slot].ctx = c;
    gpu_maps[slot].dma.size = size;
    gpu_maps[slot].vram = vram;
    gpu_maps[slot].va = va;
    map_link(slot);
    *outva = va;
    return kIOReturnSuccess;
}

// owner: the only context whose mapping this may drop, or null for the kernel's own calls
IOReturn Gpu::gpu_unmap(uint64_t va, uint64_t size, gpu_ctx *owner)
{
    int i = map_slot(va);
    if (i < 0 || (owner && gpu_maps[i].ctx != owner))
        return kIOReturnNotFound;
    gpu_maps[i].ctx->kind_va = 0;
    struct dma *d = &gpu_maps[i].dma;
    vmm_unmap(&gpu_maps[i].ctx->vm, va - 0x1000, d->size + 0x2000);
    vmm_flush(&gpu);
    vmm_free(&gpu_maps[i].ctx->vm, va - 0x1000, d->size + 0x2000);  // the address and its page tables are the next map's
    if (d->pages)
        mem_free(d->pages);
    if (d->handle) {
        ((IOMemoryDescriptor *)d->handle)->complete();
        ((IOMemoryDescriptor *)d->handle)->release();
    }
    map_clear(i);
    return kIOReturnSuccess;
}

void Gpu::retire_deferred()
{
    for (int i = 0; i < ndeferred;) {
        deferred_unmap &u = deferred[i];
        if ((int32_t)(*u.ctx->compute_sem - u.seq) < 0) {
            i++;
            continue;
        }
        gpu_unmap(u.va, 0, u.ctx);
        u = deferred[--ndeferred];
    }
    bool idle = caught_up(&kctx);
    for (gpu_ctx *c : ctxs)
        idle = idle && (!c || caught_up(c));
    if (idle) {
        for (int i = 0; i < nvram_deferred; i++)
            vram_free(gsp_vram(gsp), vram_deferred[i].vram, vram_deferred[i].size);
        nvram_deferred = 0;
    }
}

// the context's deferred unmaps are gone with it: its mappings are all unmapped at its close
void Gpu::drop_deferred(gpu_ctx *c)
{
    for (int i = 0; i < ndeferred;) {
        if (deferred[i].ctx == c)
            deferred[i] = deferred[--ndeferred];
        else
            i++;
    }
}

static void zcull_copied(gpu_ctx *c, uint64_t dst)
{
    if (dst >= c->zcull_lo && dst < c->zcull_hi)
        c->zcull_hi = 0;
}

IOReturn Gpu::gpu_copy(gpu_ctx *c, uint64_t src, uint64_t dst, uint64_t size)
{
    if (!gpu_bridge_init(c))
        return kIOReturnNotReady;
    zcull_copied(c, dst);
    volatile uint32_t *sem = (volatile uint32_t *)c->copy_sem->cpu;
    uint64_t t0 = time_us();
    int r = chan_copy(&c->copy_ch, src, dst, size, sem, c->copy_semva, ++c->copy_seq);
    stats[ST_COPY_US] += time_us() - t0;
    return r ? kIOReturnError : kIOReturnSuccess;
}

IOReturn Gpu::gpu_copy2d(gpu_ctx *c, const struct nvgpu_copy2d *cp)
{
    if (!c || !cp->bytes || !cp->lines)
        return kIOReturnBadArgument;
    if (!gpu_bridge_init(c) || !compute_bridge_init(c))
        return kIOReturnNotReady;
    zcull_copied(c, cp->dst);
    struct copy_surface s = { cp->src, cp->src_pitch, cp->src_rows, (int)cp->src_blocklinear, cp->src_origin };
    struct copy_surface d = { cp->dst, cp->dst_pitch, cp->dst_rows, (int)cp->dst_blocklinear, cp->dst_origin };
    if (ce_room(c, 0x400) != kIOReturnSuccess)
        return kIOReturnTimeout;
    if (prof_on) {
        prof_copies++;
        prof_copy_bytes += (uint64_t)cp->bytes * cp->lines;
    }
    // after the draws queued so far, which it may read, and before the ones that follow (after_copies)
    if (c->draw_pending)
        chan_acquire_queue(&c->copy_ch, c->compute_semva, c->draw_pending);
    c->ce_pending = c->ce_last = c->draw_pending = ++c->compute_seq;
    chan_copy_surface_queue(&c->copy_ch, &s, &d, cp->bytes, cp->lines, cp->remap, c->compute_semva, c->compute_seq);
    memset(c->pass_targets, 0, sizeof(c->pass_targets));  // the next draw may sample what it wrote
    return kIOReturnSuccess;
}

// room for `bytes` of methods in the copy channel's pushbuffer: going on into its other half waits for the copies
// queued there a half ago
IOReturn Gpu::ce_room(gpu_ctx *c, uint32_t bytes)
{
    int half = chan_half(&c->copy_ch, bytes);
    if (half == c->ce_half)
        return kIOReturnSuccess;
    c->ce_half_seq[c->ce_half] = c->ce_last;
    if (c->ce_half_seq[half] && ring_wait(c, ST_PUSH_WAIT_US, c->ce_half_seq[half], 2000))
        return kIOReturnTimeout;
    c->ce_half = half;
    return kIOReturnSuccess;
}

// the graphics channel's next work starts once the copies queued before it are done: they wrote what it may read
IOReturn Gpu::after_copies(gpu_ctx *c)
{
    if (!c->ce_pending)
        return kIOReturnSuccess;
    if (push_room(c, 0x100) != kIOReturnSuccess)
        return kIOReturnTimeout;
    chan_acquire_queue(&c->compute_ch, c->compute_semva, c->ce_pending);
    c->ce_pending = 0;
    return kIOReturnSuccess;
}

// Remap the buffer that holds va with another PTE kind: a depth/stencil target's Z kind says how its block-linear
// data is laid out. Buffers are mapped pitch (kind 0) when made, before anyone knows they will hold one.
// Only c's own mappings (another context's aren't c's to change). compress: the buffer holds nothing but this target.
IOReturn Gpu::set_kind(gpu_ctx *c, uint64_t va, uint8_t kind, bool compress)
{
    for (int i = 0; i < gpu_maps_top; i++) {
        struct gpu_mapping *m = &gpu_maps[i];
        if (!m->va || m->ctx != c || va < m->va || va >= m->va + m->dma.size)
            continue;
        c->kind_lo = m->va;
        c->kind_hi = m->va + m->dma.size;
        if (m->kind == kind) {
            c->kind_compressed = m->compressed;
            return kIOReturnSuccess;
        }
        uint8_t packed = 0;
        if (compress && compress_on && m->vram && !((m->va | m->vram | m->dma.size) & 0xffff))
            packed = compressed_kind(kind);
        struct vmm *vm = &m->ctx->vm;
        int failed;
        if (packed) {
            failed = vmm_map_vram_big(vm, m->va, m->vram, m->dma.size, VMM_KINDF(packed));
        } else if (m->vram) {
            vmm_unmap(vm, m->va, m->dma.size);  // big pages would hide the small ones
            failed = vmm_map_vram(vm, m->va, m->vram, m->dma.size, VMM_KINDF(kind));
        } else {
            failed = vmm_map_sys(vm, m->va, &m->dma, 0, m->dma.size, VMM_KINDF(kind));
        }
        if (failed)
            return kIOReturnError;
        vmm_flush(&gpu);
        m->kind = kind;
        m->compressed = packed != 0;
        c->kind_compressed = m->compressed;
        return kIOReturnSuccess;
    }
    return kIOReturnNotFound;
}

// Where the pages of a mapping lie in a surface's VRAM copy: the address there, 0 when they aren't part of one
// (*which then untouched, else the copy's 1-based index)
uint64_t Gpu::shadow_at(const uint64_t *pages, size_t n, int *which)
{
    for (int i = 0; i < SHADOWS; i++) {
        const shadow_copy &sh = shadows[i];
        for (uint32_t k = 0; sh.vram && k + n <= sh.npages; k++)
            if (sh.pages[k] == pages[0]) {
                if (memcmp(sh.pages + k, pages, n * sizeof(*pages)))
                    break;
                *which = i + 1;
                return sh.vram + (uint64_t)k * 0x1000;
            }
    }
    return 0;
}

// Move slot s's surface to VRAM: bytes of it, its mappings in every context taking the copy. 0 when there's no room.
int Gpu::promote(present_slot *s, uint64_t bytes)
{
    int i = 0;
    while (i < SHADOWS && shadows[i].vram)
        i++;
    bytes = (bytes + 0xfff) & ~0xfffull;
    if (i == SHADOWS || bytes > s->dma.size || bytes > PRESENT_SLOT_SIZE)
        return 0;
    uint64_t vram = vram_alloc(gsp_vram(gsp), bytes, 0x10000);
    if (!vram)
        return 0;
    if (!shadow_va)
        shadow_va = vmm_alloc(&kctx.vm, PRESENT_SLOT_SIZE, 0x10000);
    uint32_t n = (uint32_t)(bytes >> 12);
    shadow_copy &sh = shadows[i];
    sh.pages = (uint64_t *)mem_alloc(n * sizeof(uint64_t));
    if (!shadow_va || !sh.pages || vmm_map_vram(&kctx.vm, shadow_va, vram, bytes, 0)) {
        if (sh.pages)
            mem_free(sh.pages);
        sh.pages = nullptr;
        vram_free(gsp_vram(gsp), vram, bytes);
        return 0;
    }
    vmm_flush(&gpu);
    volatile uint32_t *sem = (volatile uint32_t *)kctx.copy_sem->cpu;
    int r = chan_copy(&kctx.copy_ch, s->va, shadow_va, bytes, sem, kctx.copy_semva, ++kctx.copy_seq);
    vmm_unmap(&kctx.vm, shadow_va, bytes);
    if (r) {
        mem_free(sh.pages);
        sh.pages = nullptr;
        vram_free(gsp_vram(gsp), vram, bytes);
        return 0;
    }
    memcpy(sh.pages, s->dma.pages, n * sizeof(uint64_t));
    sh.npages = n;
    sh.vram = vram;
    int moved = 0;
    for (int j = 0; j < gpu_maps_top; j++) {
        struct gpu_mapping *m = &gpu_maps[j];
        int which = 0;
        uint64_t at = m->va && m->dma.pages && !m->shadow ? shadow_at(m->dma.pages, m->dma.size >> 12, &which) : 0;
        if (!at || which != i + 1)
            continue;
        drain(m->ctx);  // draws in flight into the pages, and the caches' view of them, are done with
        if (!vmm_map_vram(&m->ctx->vm, m->va, at, m->dma.size, 0)) {
            m->shadow = (uint8_t)which;
            moved++;
        }
    }
    vmm_flush(&gpu);
    // a surface nothing draws through a mapping of isn't one WindowServer renders on the GPU: it stays as it is
    if (!moved) {
        for (int j = 0, shown = 0; j < gpu_maps_top && shown < 6; j++)
            if (gpu_maps[j].va && gpu_maps[j].dma.pages && gpu_maps[j].dma.pages[0] == s->dma.pages[0]) {
                log_msg("Malachite: surface of %u pages (first page %#llx): a mapping of %llu pages at %#llx doesn't fit it\n",
                        n, (unsigned long long)s->dma.pages[0], (unsigned long long)gpu_maps[j].dma.size >> 12,
                        (unsigned long long)gpu_maps[j].va);
                shown++;
            }
        vram_free(gsp_vram(gsp), vram, bytes);
        mem_free(sh.pages);
        sh = {};
        return 0;
    }
    // presents copy the surface from where it is drawn now
    vmm_map_vram(&kctx.vm, s->va, vram, bytes, 0);
    vmm_flush(&gpu);
    promoted++;
    return i + 1;
}

// Slot s's surface back to its pages, with what was drawn in it meanwhile, and its VRAM copy freed
void Gpu::demote(present_slot *s)
{
    shadow_copy &sh = shadows[s->shadow - 1];
    uint64_t bytes = (uint64_t)sh.npages << 12;
    for (int j = 0; j < gpu_maps_top; j++)
        if (gpu_maps[j].va && gpu_maps[j].shadow == s->shadow)
            drain(gpu_maps[j].ctx);
    volatile uint32_t *sem = (volatile uint32_t *)kctx.copy_sem->cpu;
    vmm_map_sys(&kctx.vm, s->va, &s->dma, 0, s->dma.size, 0);  // the pages take what was drawn in VRAM
    if (!vmm_map_vram(&kctx.vm, shadow_va, sh.vram, bytes, 0)) {
        vmm_flush(&gpu);
        chan_copy(&kctx.copy_ch, shadow_va, s->va, bytes, sem, kctx.copy_semva, ++kctx.copy_seq);
        vmm_unmap(&kctx.vm, shadow_va, bytes);
    }
    for (int j = 0; j < gpu_maps_top; j++) {
        struct gpu_mapping *m = &gpu_maps[j];
        if (m->va && m->shadow == s->shadow) {
            vmm_map_sys(&m->ctx->vm, m->va, &m->dma, 0, m->dma.size, 0);
            m->shadow = 0;
        }
    }
    vmm_flush(&gpu);
    vram_free(gsp_vram(gsp), sh.vram, bytes);
    mem_free(sh.pages);
    sh = {};
    s->shadow = 0;
    demoted++;
}

// Busy while the frame before it is still being copied or flipped to, until a bound after `since`: the caller tries
// again with the lock let go, which the next frame's draws take
IOReturn Gpu::gpu_present(struct nvgpu_present *p, uint64_t since)
{
    IOMemoryDescriptor *md = (IOMemoryDescriptor *)p->memory;
    if (!md || !p->width || !p->height || !p->pitch || p->offset + (uint64_t)p->pitch * p->height > md->getLength())
        return kIOReturnBadArgument;
    if (p->screen >= SCREENS)
        return kIOReturnBadArgument;
    screen &sc = screens[p->screen];
    const int head = (int)p->screen;
    if (!sc.shown || sc.asleep)
        return kIOReturnNotReady;
    if (!gpu_bridge_init(&kctx))
        return kIOReturnNotReady;
    size_t size = (md->getLength() + 0xfff) & ~(size_t)0xfff;
    if (size > PRESENT_SLOT_SIZE)
        return kIOReturnUnsupported;
    if (sc.scanout_vram != sc.shown) {  // the head moved to a new surface (a mode change)
        if (!sc.scanout_va)
            sc.scanout_va = vmm_alloc(&kctx.vm, BAR1_FB_SIZE, 0x10000);
        if (!sc.scanout_va || sc.size > BAR1_FB_SIZE || vmm_map_vram(&kctx.vm, sc.scanout_va, sc.shown, sc.size, 0))
            return kIOReturnNoMemory;
        vmm_flush(&gpu);
        sc.scanout_vram = sc.shown;
    }
    if (sc.back_size != sc.size) {
        for (int i = 0; i < 2; i++) {
            if (sc.back[i]) {
                vmm_unmap(&kctx.vm, sc.back_va[i], sc.back_size);
                vram_free(gsp_vram(gsp), sc.back[i], sc.back_size);
            }
            sc.back[i] = vram_alloc(gsp_vram(gsp), sc.size, 0x10000);
            if (!sc.back_va[i])
                sc.back_va[i] = vmm_alloc(&kctx.vm, BAR1_FB_SIZE, 0x10000);
            if (sc.back[i] && (!sc.back_va[i] || vmm_map_vram(&kctx.vm, sc.back_va[i], sc.back[i], sc.size, 0))) {
                vram_free(gsp_vram(gsp), sc.back[i], sc.size);
                sc.back[i] = 0;
            }
        }
        // the notifier memory, where the head's flip semaphores are
        if (!sc.sema_va && (sc.sema_va = vmm_alloc(&kctx.vm, 0x1000, 0x1000)) &&
            vmm_map_vram(&kctx.vm, sc.sema_va, disp->ntfy, 0x1000, 0))
            sc.sema_va = 0;
        vmm_flush(&gpu);
        sc.back_size = sc.size;
        reset_flips(sc);
        if (!sc.back[0] || !sc.back[1] || !sc.sema_va)
            log_msg("Malachite: no spare surfaces for screen %d, presents may tear\n", head);
    }

    struct present_slot *s = nullptr;
    for (int i = 0; i < PRESENT_SLOTS && !s; i++)
        if (present[i].md == md)
            s = &present[i];
    // a slot's mappings change under a copy that is still reading them, so such a slot waits for it
    auto copying = [&](const present_slot *q) { return q->fence && !copy_done(q->fence); };
    if (s && copying(s))
        return time_us() - since < 2000000 ? kIOReturnBusy : kIOReturnTimeout;
    if (!s) {
        for (int i = 0; i < PRESENT_SLOTS; i++)
            if (!s || present[i].used < s->used)
                s = &present[i];
        if (copying(s))
            return time_us() - since < 2000000 ? kIOReturnBusy : kIOReturnTimeout;
        if (!s->va)
            s->va = vmm_alloc(&kctx.vm, PRESENT_SLOT_SIZE, 0x10000);
        if (s->shadow)
            demote(s);
        s->count = 0;
        if (s->md) {
            vmm_unmap(&kctx.vm, s->va, s->dma.size);
            mem_free(s->dma.pages);
            s->md->complete();
            s->md->release();
            s->md = nullptr;
        }
        if (!s->va || md->prepare() != kIOReturnSuccess)
            return kIOReturnError;
        memset(&s->dma, 0, sizeof(s->dma));
        s->dma.size = size;
        s->dma.pages = (uint64_t *)mem_alloc((size >> 12) * sizeof(uint64_t));
        if (!s->dma.pages) {
            md->complete();
            return kIOReturnNoMemory;
        }
        for (size_t off = 0; off < size; off += 0x1000)
            s->dma.pages[off >> 12] = md->getPhysicalSegment(off, nullptr, kIOMemoryMapperNone);
        s->dma.addr = s->dma.pages[0];
        if (vmm_map_sys(&kctx.vm, s->va, &s->dma, 0, size, 0)) {
            mem_free(s->dma.pages);
            md->complete();
            return kIOReturnError;
        }
        vmm_flush(&gpu);
        md->retain();
        s->md = md;
    }
    s->used = ++present_frames;
    if (time_us() - promote_mark > 10000000) {
        if (promoted || demoted)
            log_msg("Malachite: %u surfaces drawn in VRAM from now on and %u back to memory in the last 10 s\n", promoted,
                    demoted);
        promoted = demoted = 0;
        promote_mark = time_us();
    }

    uint32_t lines = (uint32_t)(sc.size / sc.pitch), bytes = p->width * 4;
    if (bytes > sc.pitch)
        bytes = sc.pitch;
    if (lines > p->height)
        lines = p->height;
    bool flip = sc.back[0] && sc.back[1] && sc.sema_va;
    if (!flip) {
        // no spare surfaces: copies go onto the one shown, one at a time
        if (sc.fence && !copy_done(sc.fence))
            return time_us() - since < 2000000 ? kIOReturnBusy : kIOReturnTimeout;
    } else if (sc.pend.valid) {
        // the frame before's flip is still to be queued; the surface this one is copied onto is free once it is
        return time_us() - since < 2000000 ? kIOReturnBusy : kIOReturnTimeout;
    }
    if (s->shadow && !(shadows[s->shadow - 1].vram && shadows[s->shadow - 1].pages[0] == s->dma.pages[0])) {
        s->shadow = 0;
        vmm_map_sys(&kctx.vm, s->va, &s->dma, 0, s->dma.size, 0);
        vmm_flush(&gpu);
    }
    // a full-screen surface shown again and again is drawn in VRAM from then on, which the copy reads there too
    if (flip && !p->offset && p->pitch == sc.pitch && (uint64_t)p->pitch * p->height == sc.size &&
        !s->shadow && ++s->count > 2 && !(s->shadow = promote(s, s->dma.size)))
        s->count = -64;
    const uint64_t bva[3] = { sc.scanout_va, sc.back_va[0], sc.back_va[1] };
    // flipping, onto the surface after the one the head is to show; otherwise onto the one shown
    const int to = flip ? (sc.cur + 1) % 3 : sc.cur;
    present_fence = sc.fence = s->fence = ++kctx.copy_seq;
    struct copy_surface src = { s->va + p->offset, p->pitch, 0, 0 };
    struct copy_surface dst = { bva[to], sc.pitch, 0, 0 };
    chan_copy_surface_queue(&kctx.copy_ch, &src, &dst, bytes, lines, 0, kctx.copy_semva, present_fence);
    if (flip) {
        const unsigned k = sc.flips++;
        chan_release_queue(&kctx.copy_ch, sc.sema_va + DISP_FLIP_SEMA(head, k), present_fence);
        sc.flip_fence[k % 3] = present_fence;
        sc.pend = { true, to, present_fence, k, time_us() };
        sc.cur = to;
        thread_call_enter(pump_call);
    }
    if (prof_on) {  // waits for the copy, which holds the present up: the copy's time as the GPU gets to it
        const uint64_t t0 = time_us();
        while (!copy_done(present_fence) && time_us() - t0 < 50000)
            ;
        stats[ST_PRESENT_COPY_US] += time_us() - t0;
    }
    p->fence = present_fence;
    return kIOReturnSuccess;
}

void Gpu::pump(thread_call_param_t self, thread_call_param_t)
{
    static_cast<Gpu *>(self)->pump_flips();
}

// The flips presents registered, queued to the head as it takes the one before: the window channel holds the next
// flip's update until the last one is through, which a kick waits for, so it isn't done in the present itself.
void Gpu::pump_flips()
{
    for (;;) {
        bool waiting = false;
        take();
        for (screen &sc : screens) {
            if (!sc.pend.valid)
                continue;
            const int head = index(sc);
            if (!disp || suspended || !sc.shown || sc.asleep || sc.unplugged) {
                sc.pend.valid = false;
                continue;
            }
            if (!sc.fresh && !disp_flip_taken(disp, head)) {
                // a window still waiting after a while is let go, and presents go back to copying onto the
                // surface the head shows
                if (time_us() - sc.pend.since < 100000) {
                    waiting = true;
                    continue;
                }
                const struct dchan &wc = disp->head[head].wndw;
                log_msg("Malachite: screen %d's flip never taken (semaphores %#x %#x, fences %#x %#x, last %u fence %#x, "
                        "pending %u, notifiers %#x %#x, GET %#x PUT %#x, window %#x), presents may tear from now\n",
                        head, vram_rd32(&gpu, disp->ntfy + DISP_FLIP_SEMA(head, 0)),
                        vram_rd32(&gpu, disp->ntfy + DISP_FLIP_SEMA(head, 1)), sc.flip_fence[0], sc.flip_fence[1],
                        sc.last_k, sc.last_fence, sc.pend.k, vram_rd32(&gpu, disp->ntfy + disp->head[head].wndw_ntfy),
                        vram_rd32(&gpu, disp->ntfy + (disp->head[head].wndw_ntfy ^ 0x30)), rd32(&gpu, wc.user + 4),
                        wc.cur * 4, rd32(&gpu, 0x610664 + head * 8));
                for (unsigned k = 0; k < 3; k++)
                    vram_wr32(&gpu, disp->ntfy + DISP_FLIP_SEMA(head, k), sc.flip_fence[k]);
                vram_flush(&gpu);
                sc.sema_va = 0;
                sc.cur = sc.queued;
                sc.pend.valid = false;
                continue;
            }
            if (disp_flip_queue(disp, head, surface_of(sc, sc.pend.to), sc.pend.fence, sc.pend.k) == 0) {
                stats[ST_FLIP_PEND_US] += time_us() - sc.pend.since;
                sc.queued = sc.pend.to;
                sc.fresh = false;
                sc.last_k = sc.pend.k;
                sc.last_fence = sc.pend.fence;
                stats[ST_FLIP]++;
            }
            sc.pend.valid = false;
        }
        give();
        if (!waiting)
            return;
        // a flip is taken at a vblank, and the notifier of it is written just after the interrupt
        uint64_t deadline;
        clock_interval_to_deadline(1000, kMicrosecondScale, &deadline);
        IOLockLock(evlock);
        IOLockSleepDeadline(evlock, &vbl_wake, deadline, THREAD_UNINT);
        IOLockUnlock(evlock);
        delay_us(100);
    }
}

// A frame onto the screen. The draws it holds are waited for with the lock let go, as is the frame before it, so
// WindowServer goes on drawing the next while this one finishes.
IOReturn Gpu::present_frame(struct nvgpu_present *p)
{
    struct pending {
        gpu_ctx *c;
        uint32_t seq;
    } waits[MAX_CTXS + 1];
    unsigned n = 0;
    take();
    stats[ST_PRESENT]++;
    if (p->screen < BAR1_FB_SLOTS)
        stats[ST_PRESENT_HEAD + p->screen]++;
    if (kctx.draw_pending && kctx.compute_sem)
        waits[n++] = { &kctx, kctx.draw_pending };
    for (gpu_ctx *c : ctxs)
        if (c && c->draw_pending && c->compute_sem)
            waits[n++] = { c, c->draw_pending };
    give();
    const uint64_t since = time_us();
    uint64_t drawn_at = 0;  // when the draws it holds were done
    for (;;) {
        take();
        for (unsigned i = 0; i < n;) {
            gpu_ctx *c = waits[i].c;
            bool open = c == &kctx;
            for (gpu_ctx *o : ctxs)
                open |= o == c;
            if (!open || (int32_t)(*c->compute_sem - waits[i].seq) >= 0)
                waits[i] = waits[--n];
            else
                i++;
        }
        IOReturn r = kIOReturnBusy;
        if (!n) {
            if (!drawn_at) {  // the wait for the draws is counted above; what follows is the present's own
                stats[ST_PRESENT_WAIT_US] += (drawn_at = time_us()) - since;
                if (prof_on) {
                    prof_harvest(&kctx);
                    for (gpu_ctx *c : ctxs)
                        if (c)
                            prof_harvest(c);
                }
            }
            const uint64_t t0 = time_us();
            r = gpu_present(p, since);
            if (r != kIOReturnBusy)
                stats[ST_PRESENT_SUBMIT_US] += time_us() - t0;
        } else if (time_us() - since > 200000) {
            drain_all();  // a fault takes its recovery here
        }
        give();
        if (r != kIOReturnBusy) {
            const uint64_t now = time_us();
            stats[ST_PRESENT_TOTAL_US] += now - since;
            if (drawn_at)
                stats[ST_PRESENT_POST_US] += now - drawn_at;
            return r;
        }
        // a frame's draws finish within a few milliseconds. A flip is taken at a vblank, and at 240 Hz that is
        // four milliseconds apart, so the wait ends at one rather than at IOSleep's millisecond tick.
        if (time_us() - since > 4000) {
            uint64_t deadline;
            clock_interval_to_deadline(1000, kMicrosecondScale, &deadline);
            IOLockLock(evlock);
            IOLockSleepDeadline(evlock, &vbl_wake, deadline, THREAD_UNINT);
            IOLockUnlock(evlock);
            delay_us(100);  // the notifier of the flip a vblank took is written just after its interrupt
        } else {
            delay_us(50);
        }
    }
}

IOReturn Gpu::gpu_fill(gpu_ctx *c, uint64_t dst, uint64_t size, uint32_t pattern)
{
    if (!gpu_bridge_init(c))
        return kIOReturnNotReady;
    if (!size)
        return kIOReturnSuccess;
    // doubling fill: seed 4 bytes from a mapped scratch word, then copy the filled prefix onto the next gap until
    // covered. The pattern is one byte repeated, so a range of any length and alignment takes it.
    volatile uint32_t *sem = (volatile uint32_t *)c->copy_sem->cpu;
    uint64_t scratch_va = c->copy_semva + 0x100, seed = size < 4 ? size : 4;
    *(uint32_t *)((uint8_t *)c->copy_sem->cpu + 0x100) = pattern;
    if (chan_copy(&c->copy_ch, scratch_va, dst, seed, sem, c->copy_semva, ++c->copy_seq))
        return kIOReturnError;
    for (uint64_t filled = seed; filled < size;) {
        uint64_t n = filled < (size - filled) ? filled : (size - filled);
        if (chan_copy(&c->copy_ch, dst, dst + filled, n, sem, c->copy_semva, ++c->copy_seq))
            return kIOReturnError;
        filled += n;
    }
    return kIOReturnSuccess;
}

bool Gpu::compute_bridge_init(gpu_ctx *c)
{
    if (c->compute_ready && (c->copy_faulted || c->compute_faulted))
        recover(c);
    if (c->compute_ready)
        return true;
    if (!gpu_bridge_init(c))  // the same client and address space, so mapped buffers are addressable here too
        return false;
    if (c->compute_sem) {  // after a power cycle: its buffers are still mapped
        c->compute_ready = graphics_channel(c, c->compute_ch.chid);
        if (c->compute_ready)
            issued(c);
        return c->compute_ready;
    }
    // the kernel's on chid 4 (copy is 1)
    int chid = c == &kctx ? 4 : gsp_chid_get(gsp, FIRST_CTX_CHID);
    if (chid < 0 || compute_setup(gsp, &c->client, &c->vm, (uint32_t)chid, c == &kctx, &c->compute_ch,
                                     &c->compute_ctx) ||
        render_setup(gsp, &c->client, &c->compute_ch))
        return false;
    c->compute_qmd = gsp_dma(gsp, 0x1000, 1);
    c->compute_cbuf = gsp_dma(gsp, 0x1000, 1);
    c->draw_ring = gsp_dma(gsp, DRAW_SLOTS * DRAW_SLOT_BYTES, 0);
    c->dispatch_ring = gsp_dma(gsp, DISPATCH_SLOTS * DISPATCH_SLOT_BYTES, 0);
    if (!c->compute_qmd || !c->compute_cbuf || !c->draw_ring || !c->dispatch_ring)
        return false;
    c->draw_ringva = vmm_alloc(&c->vm, c->draw_ring->size, 0x1000);
    if (!c->draw_ringva || vmm_map_sys(&c->vm, c->draw_ringva, c->draw_ring, 0, c->draw_ring->size, 0))
        return false;
    c->dispatch_ringva = vmm_alloc(&c->vm, c->dispatch_ring->size, 0x1000);
    if (!c->dispatch_ringva || vmm_map_sys(&c->vm, c->dispatch_ringva, c->dispatch_ring, 0, c->dispatch_ring->size, 0))
        return false;
    c->compute_qmdva = vmm_alloc(&c->vm, c->compute_qmd->size, 0x1000);
    c->compute_cbufva = vmm_alloc(&c->vm, c->compute_cbuf->size, 0x1000);
    if (!c->compute_qmdva || !c->compute_cbufva ||
        vmm_map_sys(&c->vm, c->compute_qmdva, c->compute_qmd, 0, c->compute_qmd->size, 0) ||
        vmm_map_sys(&c->vm, c->compute_cbufva, c->compute_cbuf, 0, c->compute_cbuf->size, 0))
        return false;
    vmm_flush(&gpu);
    c->compute_sem = (volatile uint32_t *)((uint8_t *)c->compute_qmd->cpu + 0x800);  // sem lives in the QMD page tail
    c->compute_semva = c->compute_qmdva + 0x800;
    // the texture header and sampler pools share the const bank's page, after the bank's 0x200 bytes
    compute_set_tex_pools(&c->compute_ch, c->compute_cbufva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES,
                          c->compute_cbufva + TEX_SAMPLER_POOL, NVGPU_MAX_TEXTURES);
    render_set_tex_pools(&c->compute_ch, c->compute_cbufva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES,
                         c->compute_cbufva + TEX_SAMPLER_POOL, NVGPU_MAX_TEXTURES);
    c->compute_ready = true;
    log_msg("Malachite: compute bridge ready (qmd va 0x%llx cbuf va 0x%llx)\n", c->compute_qmdva, c->compute_cbufva);
    issued(c);  // where the first launch starts, should it be run again
    return true;
}

// The context's NVDEC channel, made on first use
bool Gpu::video_init(gpu_ctx *c)
{
    if (c->video_ready)
        return true;
    if (!gpu_bridge_init(c))
        return false;
    int chid = gsp_chid_get(gsp, FIRST_CTX_CHID);
    if (chid < 0)
        return false;
    if (video_create(gsp, &c->client, &c->vm, (uint32_t)chid, c == &kctx, &c->video)) {
        gsp_chid_put(gsp, (uint32_t)chid);
        return false;
    }
    c->video_ready = true;
    return true;
}

// A task's context by the handle Accelerator holds: 0 is the kernel's, a closed one NULL
gpu_ctx *Gpu::ctx_of(uintptr_t handle)
{
    if (!handle)
        return &kctx;
    unsigned slot = (unsigned)(handle & 0xff) - 1;
    if (slot >= MAX_CTXS || !ctxs[slot] || ctx_serial[slot] != (uint32_t)(handle >> 8))
        return nullptr;
    return ctxs[slot];
}

// A context for a task: nothing made on the GPU until its first mapping or launch; 0 when they're all taken.
// Under lock.
uintptr_t Gpu::ctx_open()
{
    for (int i = 0; i < MAX_CTXS; i++) {
        if (ctxs[i])
            continue;
        ctxs[i] = (gpu_ctx *)IOMallocZero(sizeof(gpu_ctx));
        if (!ctxs[i])
            return 0;
        proc_selfname(ctxs[i]->owner, sizeof(ctxs[i]->owner));  // called from the task's own thread
        ctx_serial[i] = (ctx_serial[i] + 1) & 0xffffff;
        return (uintptr_t)(i + 1) | (uintptr_t)ctx_serial[i] << 8;
    }
    return 0;
}

// A task's context gone: its work waited for, the mappings left in it dropped, then its client (and with it the
// channels and address space GSP-RM knows), our page tables and memory freed. Under lock.
void Gpu::ctx_close(uintptr_t handle)
{
    gpu_ctx *c = handle ? ctx_of(handle) : nullptr;
    if (!c)
        return;
    if (c->compute_ready)
        drain(c);
    drop_deferred(c);
    for (int i = 0; i < gpu_maps_top; i++)
        if (gpu_maps[i].va && gpu_maps[i].ctx == c)
            gpu_unmap(gpu_maps[i].va, 0);
    if (c->gpu_ready) {
        rm_free(gsp, c->client.client, c->client.client);
        if (c->compute_ready) {
            gr_ctx_free(gsp, &c->compute_ctx);
            chan_free(&c->compute_ch);
        }
        chan_free(&c->copy_ch);
    }
    if (c->video_ready) {
        gsp_chid_put(gsp, c->video.ch.chid);
        video_free(&c->video);
    }
    // what a power cycle keeps, whether or not the channels are back on it yet
    if (c->compute_sem) {
        gsp_chid_put(gsp, c->compute_ch.chid);
        gsp_dma_free(gsp, c->compute_qmd);
        gsp_dma_free(gsp, c->compute_cbuf);
        gsp_dma_free(gsp, c->draw_ring);
        gsp_dma_free(gsp, c->dispatch_ring);
        if (c->prof) {
            gsp_dma_free(gsp, c->prof);
            IOFree(c->prof_draws, PROF_RING * sizeof(*c->prof_draws));
        }
        free_local_memory(c);
    }
    if (c->copy_sem) {
        gsp_chid_put(gsp, c->copy_ch.chid);
        gsp_dma_free(gsp, c->copy_sem);
        vmm_destroy(&c->vm);
    }
    unsigned slot = (unsigned)(handle & 0xff) - 1;
    IOFree(c, sizeof(*c));
    ctxs[slot] = nullptr;
}

// GSP-RM's word that a channel faulted and is torn down: GSP-RM stops it and it never runs again, so ours are freed
// and made anew before their next use (recover()). Called from message handling, which may be inside a wait.
void Gpu::rc_event(void *ctx, uint32_t engine, uint32_t chid, uint32_t exception)
{
    Gpu *self = (Gpu *)ctx;
    for (int i = -1; i < MAX_CTXS; i++) {
        gpu_ctx *c = i < 0 ? &self->kctx : self->ctxs[i];
        if (c && c->compute_ready && chid == c->compute_ch.chid)
            c->compute_faulted = true;
        if (c && c->gpu_ready && chid == c->copy_ch.chid)
            c->copy_faulted = true;
    }
}

// After the kernel's copy channel was remade: a flip queued behind a copy that was lost waits for its semaphore
// for good, so each lit head has the semaphore released and a modeset onto the surface shown, which flips start
// from again. Under lock.
void Gpu::resync_screens()
{
    for (int head = 0; disp && head < SCREENS; head++) {
        screen &sc = screens[head];
        if (!sc.shown || sc.asleep || sc.unplugged)
            continue;
        for (unsigned k = 0; k < 3; k++)
            vram_wr32(&gpu, disp->ntfy + DISP_FLIP_SEMA(head, k), sc.flip_fence[k]);
        vram_flush(&gpu);
        struct disp_output *o = &disp->out[sc.out];
        const struct disp_mode *t = o->nmodes ? &o->modes[sc.mode - 1] : &o->mode;
        reset_flips(sc);
        sc.fence = 0;
        if (int r = modeset(sc, t)) {
            log_msg("Malachite: screen %d's modeset after the copy channel was remade failed\n", head);
            if (r != DISP_NO_OUTPUT)
                disp_reset(disp);
        } else {
            log_msg("Malachite: screen %d set again after the copy channel was remade\n", head);
        }
    }
}

// A faulted channel freed (its 3D, compute and copy objects with it) and built again on the same VA space, every
// mapping kept; what was queued on it is lost, and its waiters have timed out. Under lock.
void Gpu::recover(gpu_ctx *c)
{
    recoveries++;
    // every channel rebuild is a round of GSP-RM calls: a task faulting on every launch would keep the GPU busy
    // with nothing else, so past 16 in 10 s its work is refused for a while, longer each time it goes on (the
    // kernel's own context never is). Never for good: refusing WindowServer's work leaves the desktop unusable, so
    // its refusals stay short; a quiet 10 s forgets it all
    if (c != &kctx && (c->copy_faulted || c->compute_faulted)) {
        uint64_t now = time_us();
        if (now - c->fault_window > 10000000) {
            if (c->faults <= (!strcmp(c->owner, "WindowServer") ? 16u : 3u))
                c->backoff_ms = 0;
            c->fault_window = now;
            c->faults = 0;
        }
        bool ws = !strcmp(c->owner, "WindowServer");
        if (++c->faults > (ws ? 16 : 3) && now >= c->refused_until) {
            uint32_t cap = ws ? 1000 : 120000;
            c->backoff_ms = c->backoff_ms ? (c->backoff_ms * 2 > cap ? cap : c->backoff_ms * 2) : ws ? 500 : 2000;
            c->refused_until = now + (uint64_t)c->backoff_ms * 1000;
            c->faults = 0;
            log_msg("Malachite: %s keeps faulting the GPU, its work is refused for %u ms\n", c->owner, c->backoff_ms);
        }
    }
    if (c->copy_faulted) {
        c->copy_faulted = false;
        log_msg("Malachite: %s's copy channel faulted, making a new one\n", c == &kctx ? "kernel" : c->owner);
        uint32_t handle = c->copy_ch.handle, chid = c->copy_ch.chid;
        rm_free(gsp, c->client.client, handle);
        chan_free(&c->copy_ch);
        if (!copy_channel(c, chid, handle)) {
            log_msg("Malachite: copy channel not remade\n");
            c->gpu_ready = false;
        }
        // the copies lost never release their payloads, which the graphics channel may be waiting on
        if (c->compute_sem && (c->ce_last || c->ce_pending)) {
            *c->compute_sem = c->compute_seq;
            c->ce_pending = c->ce_last = c->draw_pending = 0;
        }
        memset(c->ce_half_seq, 0, sizeof(c->ce_half_seq));
        c->ce_half = 0;
        vmm_flush(&gpu);
        if (c == &kctx) {  // presents go on the kernel's copy channel: a flip may be waiting on a lost release
            resync_screens();
            present_fence = 0;  // the copy it names is lost: nothing to wait for
        }
    }
    if (c->compute_faulted) {
        c->compute_faulted = false;
        log_msg("Malachite: %s's graphics channel faulted, making a new one\n", c == &kctx ? "kernel" : c->owner);
        for (const auto &l : c->launches) {
            if (!l.seq || (int32_t)(l.seq - *c->compute_sem) <= 0)
                continue;
            log_msg("Malachite:   not done: %s %u of program %#llx%s, %ux%ux%u groups of %ux%ux%u, %u regs\n",
                        l.vs ? "draw" : "dispatch", l.seq, (unsigned long long)l.prog, l.vs ? " (vs after)" : "", l.groups[0],
                        l.groups[1], l.groups[2], l.block[0], l.block[1], l.block[2], l.regs);
            if (l.vs)
                log_msg("Malachite:     vs args %#llx %#llx, ps args %#llx %#llx\n", (unsigned long long)l.params[0],
                        (unsigned long long)l.params[1], (unsigned long long)l.params[2], (unsigned long long)l.params[3]);
            if (l.seq == *c->compute_sem + 1) {
                log_msg("Malachite:   the first one's code, ps or program then vs\n");
                log_code(l.prog);
                if (l.vs)
                    log_code(l.vs);
            }
        }
        rm_free(gsp, c->client.client, c->compute_ch.handle);
        // the launches after the last one done, kept for the new channel: from the first one not done, or once it
        // has run again already, from the one after it
        uint32_t done = *c->compute_sem, *saved = nullptr, nsaved = 0, first = 0;
        for (unsigned i = c->issued_next > 128 ? c->issued_next - 127 : 1; i < c->issued_next; i++) {
            auto &prev = c->issued[(i - 1) % 128], &cur = c->issued[i % 128];
            if ((int32_t)(cur.payload - done) <= 0 || (int32_t)(prev.payload - done) > 0)
                continue;
            unsigned from = cur.payload == c->replayed ? i : i - 1;
            if (from + 1 < c->issued_next && (saved = (uint32_t *)IOMalloc(CHAN_SAVE_WORDS * 4))) {
                nsaved = chan_save(&c->compute_ch, c->issued[from % 128].gp_end, saved);
                first = c->issued[(from + 1) % 128].payload;
            }
            break;
        }
        chan_free(&c->compute_ch);
        gr_ctx_free(gsp, &c->compute_ctx);
        c->draw_pending = c->ce_pending = 0;
        if (!saved) {  // what's lost never releases its slots
            memset(c->draw_slot_seq, 0, sizeof(c->draw_slot_seq));
            memset(c->dispatch_slot_seq, 0, sizeof(c->dispatch_slot_seq));
        }
        memset(c->pass_targets, 0, sizeof(c->pass_targets));
        memset(c->half_seq, 0, sizeof(c->half_seq));
        c->push_half = 0;
        // on another chid (the kernel's: the other of 4 and 5): GSP-RM may still be finishing with the faulted one
        uint32_t old = c->compute_ch.chid;
        int chid = c == &kctx ? (old == 4 ? 5 : 4) : gsp_chid_get(gsp, FIRST_CTX_CHID);
        if (c != &kctx)
            gsp_chid_put(gsp, old);
        if (chid < 0 || !graphics_channel(c, (uint32_t)chid)) {
            log_msg("Malachite: graphics channel not remade\n");
            c->compute_ready = false;
            if (saved)
                IOFree(saved, CHAN_SAVE_WORDS * 4);
            return;
        }
        c->issued_next = 0;
        if (saved) {
            if (chan_replay(&c->compute_ch, saved, nsaved))
                log_msg("Malachite: %s's work lost to the fault doesn't fit the new channel\n",
                        c == &kctx ? "kernel" : c->owner);
            else
                log_msg("Malachite: %s's work from %u on runs again\n", c == &kctx ? "kernel" : c->owner, first);
            IOFree(saved, CHAN_SAVE_WORDS * 4);
            c->replayed = first;
            // and everything up to the latest payload released, past one that isn't run again
            render_release(&c->compute_ch, c->compute_semva, c->compute_seq);
            c->half_seq[0] = c->half_seq[1] = c->draw_pending = c->compute_seq;
            c->push_half = chan_half(&c->compute_ch, 1);
        }
        issued(c);
    }
}

// Waits for a ring slot or pushbuffer half the GPU hasn't finished with, counted: the CPU spins here while the GPU is
// the slower side. The lock is held, so the interrupt that would tell of a fault can't run: GSP-RM's messages are
// read here, and a channel it tore down is made anew, rather than the wait running out its time on a dead one.
int Gpu::ring_wait(gpu_ctx *c, int counter, uint32_t payload, int ms)
{
    uint64_t t0 = time_us();
    int r = -1;
    for (int t = 0; t < ms / 10 && r; t++) {
        r = render_wait(c->compute_sem, payload, 10);
        if (r && gsp) {
            gsp_poll(gsp);
            if (c->compute_faulted || c->copy_faulted)
                recover(c);
        }
    }
    stats[counter] += time_us() - t0;
    return r;
}

// The words an indirect draw or dispatch takes its arguments from, once everything queued before it is done, as
// earlier GPU work may write them: through the copy engine, so they may be anywhere, VRAM too.
IOReturn Gpu::read_indirect(gpu_ctx *c, uint64_t va, uint32_t *words, unsigned n)
{
    if (!gpu_bridge_init(c))
        return kIOReturnNotReady;
    if (drain(c) != kIOReturnSuccess)
        return kIOReturnTimeout;
    volatile uint32_t *sem = (volatile uint32_t *)c->copy_sem->cpu;
    if (chan_copy(&c->copy_ch, va, c->copy_semva + 0x200, n * 4, sem, c->copy_semva, ++c->copy_seq))
        return kIOReturnError;
    memcpy(words, (const uint8_t *)c->copy_sem->cpu + 0x200, n * 4);
    return kIOReturnSuccess;
}

// a heap's pools fit what a handle can name: 2^20 headers, 4096 samplers (TEXSAMP's 12 bits)
static bool heap_ok(const struct nvgpu_heap *h)
{
    return !h->headers || (h->samplers && h->nheaders && h->nheaders <= 1u << 20 && h->nsamplers &&
                           h->nsamplers <= 4096);
}

// room for `bytes` more of methods in the graphics pushbuffer: going on into its other half waits for the work last
// submitted there, a half ago
IOReturn Gpu::push_room(gpu_ctx *c, uint32_t bytes)
{
    int half = chan_half(&c->compute_ch, bytes);
    if (half == c->push_half)
        return kIOReturnSuccess;
    c->half_seq[c->push_half] = c->compute_seq;
    if (c->half_seq[half] && ring_wait(c, ST_PUSH_WAIT_US, c->half_seq[half], 2000))
        return kIOReturnTimeout;
    c->push_half = half;
    return kIOReturnSuccess;
}

// Room for lane bytes of local memory for every thread the GPU can hold at once, as NVK sizes it: a warp's 32 lanes,
// the most warps an SM runs, its TPC's SMs, every TPC. It only grows, while the context is idle; the rest of the time
// a shader that needs no more than there is goes straight on.
IOReturn Gpu::local_memory(gpu_ctx *c, uint32_t lane)
{
    if (lane <= c->slm_lane)
        return kIOReturnSuccess;
    if (lane > 4096)
        return kIOReturnBadArgument;
    const struct gsp_info *in = gsp_get_info(gsp);
    uint64_t warps = in->gr[0x0d], sms = in->gr[0x20], tpcs = in->gr[0x09];  // MAX_WARPS_PER_SM, SM_PER_TPC, TPCs
    // a TPC's share is picked by its index, which can run past the enabled count when some are floorswept (and RM
    // documents 0x09 as SMs per TPC on older chips): room for every TPC the GPCs can hold, LITTER_NUM_GPCS times
    // LITTER_NUM_TPC_PER_GPC, as a share past the end faults on unmapped pages or writes another buffer
    if (in->gr[0x14] * in->gr[0x17] > tpcs)
        tpcs = (uint64_t)in->gr[0x14] * in->gr[0x17];
    if (!warps || !sms || !tpcs) {
        log_msg("Malachite: no SM counts for local memory\n");
        return kIOReturnUnsupported;
    }
    lane = (lane + 0xff) & ~0xffu;  // in steps, so a few shaders don't each grow it
    uint64_t per_warp = (lane * 32ull + 0x1ff) & ~0x1ffull;
    uint64_t per_tpc = (per_warp * warps * sms + 0x7fff) & ~0x7fffull;
    uint64_t size = (per_tpc * tpcs + 0x1ffff) & ~0x1ffffull, va;
    if (drain(c) != kIOReturnSuccess)
        return kIOReturnTimeout;
    uint64_t vram = vram_alloc(gsp_vram(gsp), size, 0x20000);
    if (!vram)
        return kIOReturnNoMemory;
    if (gpu_map_vram(c, vram, size, &va) != kIOReturnSuccess) {
        vram_free(gsp_vram(gsp), vram, size);
        return kIOReturnNoMemory;
    }
    free_local_memory(c);
    c->slm_vram = vram;
    c->slm_va = va;
    c->slm_size = size;
    c->slm_per_tpc = per_tpc;
    c->slm_per_warp = (uint32_t)per_warp;
    c->slm_lane = lane;
    compute_set_local_memory(&c->compute_ch, va, per_tpc);
    render_set_local_memory(&c->compute_ch, va, size, (uint32_t)per_warp);
    return kIOReturnSuccess;
}

void Gpu::free_local_memory(gpu_ctx *c)
{
    if (!c->slm_va)
        return;
    gpu_unmap(c->slm_va, c->slm_size);
    vram_free(gsp_vram(gsp), c->slm_vram, c->slm_size);
    c->slm_va = c->slm_vram = c->slm_size = 0;
    c->slm_lane = 0;
}

IOReturn Gpu::gpu_dispatch(gpu_ctx *c, const struct nvgpu_dispatch *d)
{
    // a CTA gets at most the chip's shared memory and 16 named barriers
    if (!d || d->params_size > NVGPU_PARAM_BYTES || !d->grid_x || !d->block_x || d->shared_size > gpu.chip->max_shared ||
        d->barriers > 16 || d->nheaders > NVGPU_MAX_TEXTURES || d->nsamplers > NVGPU_MAX_TEXTURES ||
        !heap_ok(&d->heap))
        return kIOReturnBadArgument;
    // a CTA whose registers (8-register units a thread) don't fit never launches: the channel faults. The SM's 64K
    // are four partitions of 16K, and a CTA's warps are dealt out among them in turn.
    uint64_t threads = (uint64_t)d->block_x * (d->block_y ? d->block_y : 1) * (d->block_z ? d->block_z : 1);
    if ((threads + 127) / 128 * 32 * ((d->reg_count + 7) & ~7u) > 16384) {
        log_msg("Malachite: dispatch of %llu threads of %u registers doesn't fit the register file\n", threads, d->reg_count);
        return kIOReturnBadArgument;
    }
    if (!compute_bridge_init(c))
        return kIOReturnNotReady;
    IOReturn lr = d->local_size ? local_memory(c, d->local_size) : kIOReturnSuccess;
    if (lr != kIOReturnSuccess)
        return lr;
    // c[0x0][0x118] (-> UR4 global base) stays 0; like CUDA's driver, publish ntid.xyz and nctaid.xyz at 0x0
    uint32_t block[3] = { d->block_x, d->block_y ? d->block_y : 1, d->block_z ? d->block_z : 1 };
    uint32_t grid[3] = { d->grid_x, d->grid_y ? d->grid_y : 1, d->grid_z ? d->grid_z : 1 };
    if (d->indirect) {
        IOReturn r = read_indirect(c, d->indirect, grid, 3);
        if (r != kIOReturnSuccess)
            return r;
        if (!grid[0] || !grid[1] || !grid[2])
            return kIOReturnSuccess;
    }
    // like a draw, it goes to the GPU unwaited, in a slot of its own
    unsigned slot = c->dispatch_next++ % DISPATCH_SLOTS;
    if (c->dispatch_slot_seq[slot] && ring_wait(c, ST_SLOT_WAIT_US, c->dispatch_slot_seq[slot], 2000))
        return kIOReturnTimeout;
    if (after_copies(c) != kIOReturnSuccess)
        return kIOReturnTimeout;
    if (push_room(c, 0x1000) != kIOReturnSuccess)
        return kIOReturnTimeout;
    uint8_t *slot_cpu = (uint8_t *)c->dispatch_ring->cpu + slot * DISPATCH_SLOT_BYTES;
    uint64_t cbva = c->dispatch_ringva + slot * DISPATCH_SLOT_BYTES, qmdva = cbva + 0x1000;
    uint32_t *cb = (uint32_t *)slot_cpu;
    memset(cb, 0, COMPUTE_CBUF_BYTES);
    cb[0] = block[0];
    cb[1] = block[1];
    cb[2] = block[2];
    cb[3] = grid[0];
    cb[4] = grid[1];
    cb[5] = grid[2];
    for (int i = 0; i < 3; i++) {  // NVK_CB_ORIGIN, _GROUP_ORIGIN, _GROUPS, _GRID
        cb[6 + i] = d->origin[i];
        cb[9 + i] = d->group_origin[i];
        cb[12 + i] = d->groups[0] ? d->groups[i] : grid[i];
        cb[15 + i] = d->threads[0] ? d->threads[i] : grid[i] * block[i];
    }
    memcpy((uint8_t *)cb + 0x160, d->params, d->params_size);
    // earlier draws and dispatches finish first, as it may read what they wrote (and its semaphore release then
    // comes after theirs); the pools are channel state the earlier ones may still be using
    compute_wait_idle(&c->compute_ch);
    if (d->heap.headers) {
        memcpy((uint8_t *)cb + 0x200, d->heap.handles, sizeof(d->heap.handles));
        compute_set_tex_pools(&c->compute_ch, d->heap.headers, d->heap.nheaders, d->heap.samplers, d->heap.nsamplers);
    } else {
        memcpy((uint8_t *)cb + TEX_HEADER_POOL, d->headers, d->nheaders * sizeof(d->headers[0]));
        memcpy((uint8_t *)cb + TEX_SAMPLER_POOL, d->samplers, d->nsamplers * sizeof(d->samplers[0]));
        compute_set_tex_pools(&c->compute_ch, cbva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES, cbva + TEX_SAMPLER_POOL,
                              NVGPU_MAX_TEXTURES);
    }
    uint32_t payload = ++c->compute_seq;
    compute_build_qmd((uint32_t *)(slot_cpu + 0x1000), d->prog_va, cbva, c->compute_semva, payload, block, grid,
                      d->reg_count, d->shared_size, d->barriers);
    compute_qmd_set_local((uint32_t *)(slot_cpu + 0x1000), d->local_size);
    c->launches[payload % 8] = { payload, { grid[0], grid[1], grid[2] }, { block[0], block[1], block[2] }, d->reg_count,
                                 d->prog_va, 0 };
    vmm_flush(&gpu);
    compute_launch(&c->compute_ch, qmdva);
    issued(c);
    c->dispatch_slot_seq[slot] = c->draw_pending = payload;
    memset(c->pass_targets, 0, sizeof(c->pass_targets));  // the next draw may read what it writes
    return kIOReturnSuccess;
}

bool Gpu::prof_ready(gpu_ctx *c)
{
    if (c->prof)
        return true;
    const size_t bytes = PROF_RING * 16;
    struct dma *dma = gsp_dma(gsp, bytes, 0);
    if (!dma)
        return false;
    memset(dma->cpu, 0, bytes);
    uint64_t va = vmm_alloc(&c->vm, bytes, 0x1000);
    c->prof_draws = (gpu_ctx::prof_draw *)IOMallocZero(PROF_RING * sizeof(*c->prof_draws));
    if (!va || !c->prof_draws || vmm_map_sys(&c->vm, va, dma, 0, bytes, 0)) {
        if (c->prof_draws)
            IOFree(c->prof_draws, PROF_RING * sizeof(*c->prof_draws));
        c->prof_draws = nullptr;
        gsp_dma_free(gsp, dma);
        return false;
    }
    vmm_flush(&gpu);
    c->prof_va = va;
    c->prof_n = c->prof_seen = 0;
    c->prof_last_ts = 0;
    c->prof = dma;
    return true;
}

// the draws of c the GPU has timestamped since the last harvest
void Gpu::prof_harvest(gpu_ctx *c)
{
    if (!c->prof)
        return;
    if (c->prof_n - c->prof_seen > PROF_RING) {  // not looked at for longer than the ring holds
        c->prof_seen = c->prof_n;
        c->prof_last_ts = 0;
    }
    for (; c->prof_seen != c->prof_n; c->prof_seen++) {
        const volatile uint32_t *rec = (const volatile uint32_t *)((uint8_t *)c->prof->cpu + (c->prof_seen % PROF_RING) * 16);
        if (rec[0] != c->prof_seen + 1) {
            static int shown;
            if (shown++ < 4)
                log_msg("Malachite: prof record %u: %#x %#x %#x %#x (n %u)\n", c->prof_seen, rec[0], rec[1], rec[2], rec[3],
                        c->prof_n);
            break;
        }
        uint64_t ts = rec[2] | (uint64_t)rec[3] << 32;
        if (c->prof_last_ts)
            prof_add(c->prof_draws[c->prof_seen % PROF_RING], ts - c->prof_last_ts);
        c->prof_last_ts = ts;
    }
}

// what makes a draw unfit to send, in a word for the log (null when it fits)
static const char *draw_reject(const struct nvgpu_draw *d)
{
    if (!d)
        return "no draw";
    if (d->vs_params_size > NVGPU_PARAM_BYTES || d->ps_params_size > NVGPU_PARAM_BYTES ||
        d->ti_params_size > NVGPU_PARAM_BYTES)
        return "params size";
    if (d->count && (!d->vs || !d->ps || !d->instances))
        return !d->instances ? "no instances" : "no shader";
    if (d->index_format > 2)
        return "index format";
    if (d->zt & 511)
        return "zt alignment";
    // a tessellation spacing of 3 is no spacing, and loses the channel
    if ((d->tess & ~0x333u) || (d->tess >> 4 & 3) == 3 || (d->tess && (d->primitive != 0xe || !d->ti || !d->ts)))
        return "tessellation";
    if (d->target & 127)
        return "target alignment";
    if (d->pitch & 127)
        return "pitch alignment";
    if (!d->width || !d->height)
        return "empty target";
    if (d->nheaders > NVGPU_MAX_TEXTURES || d->nsamplers > NVGPU_MAX_TEXTURES)
        return "texture count";
    if (!heap_ok(&d->heap))
        return "heap";
    if (d->layers > 512 || d->nviewports > 16)
        return "layers or viewports";
    for (const struct nvgpu_target &t : d->more)
        if ((t.addr & 127) || (t.pitch & 127))
            return "target 1-7 alignment";
    return nullptr;
}

IOReturn Gpu::gpu_draw(gpu_ctx *c, const struct nvgpu_draw *d, const char **reject)
{
    if (const char *why = draw_reject(d)) {
        if (reject)
            *reject = why;
        return kIOReturnBadArgument;
    }
    if (!compute_bridge_init(c))
        return kIOReturnNotReady;
    if (ndeferred || nvram_deferred)
        retire_deferred();
    // where it renders, for the stats: the mapping of the draw before it first, a pass draws to one target
    auto holds = [&](int i) {
        return i < gpu_maps_top && gpu_maps[i].ctx == c && gpu_maps[i].va <= d->target &&
               d->target < gpu_maps[i].va + gpu_maps[i].dma.size;
    };
    if (d->target && !holds(target_map))
        for (int i = 0; i < gpu_maps_top; i++)
            if (holds(i)) {
                target_map = i;
                break;
            }
    if (d->target && holds(target_map)) {
        bool vram = gpu_maps[target_map].vram || gpu_maps[target_map].shadow;  // a promoted surface is drawn in its VRAM copy
        stats[vram ? ST_DRAW_VRAM : ST_DRAW_SYS]++;
        stats[vram ? ST_DRAW_VRAM_KPX : ST_DRAW_SYS_KPX] += (uint64_t)d->width * d->height / 1000;
    }
    IOReturn lr = d->local_size ? local_memory(c, d->local_size) : kIOReturnSuccess;
    if (lr != kIOReturnSuccess)
        return lr;
    // the depth kinds the plugin asks for (Z16, ZF32_X24S8, generic), which set_kind compresses where it can
    if (d->zt && d->zt_kind != 0x01 && d->zt_kind != 0x04 && d->zt_kind != 0x06) {
        if (reject)
            *reject = "zt kind";
        return kIOReturnBadArgument;
    }
    if (d->zt && (d->zt != c->kind_va || d->zt_kind != c->kind_set)) {  // finding the mapping walks them all
        if (set_kind(c, d->zt, (uint8_t)d->zt_kind, d->zt_compress) != kIOReturnSuccess) {
            if (reject)
                *reject = "set kind";
            return kIOReturnBadArgument;
        }
        c->kind_va = d->zt;
        c->kind_set = d->zt_kind;
    }
    unsigned slot = c->draw_next++ % DRAW_SLOTS;
    if (c->draw_slot_seq[slot] && ring_wait(c, ST_SLOT_WAIT_US, c->draw_slot_seq[slot], 2000))
        return kIOReturnTimeout;
    // the draw, its clears (of each layer of every target) and their semaphore releases, at most this much of
    // methods
    const bool after_copy = c->ce_pending != 0;
    if (after_copies(c) != kIOReturnSuccess)
        return kIOReturnTimeout;
    if (push_room(c, 0x4000 + (d->layers > 1 ? d->layers * 9 * 8 : 0)) != kIOReturnSuccess)
        return kIOReturnTimeout;
    struct render_draw r = {};
    r.target = d->target;
    r.pitch = d->pitch;
    r.width = d->width;
    r.height = d->height;
    r.samples = d->samples;
    r.format = d->format;
    r.target_blocklinear = d->target_blocklinear;
    r.alpha_control = d->alpha_control;
    r.per_sample = d->per_sample;
    r.ps_depth = d->ps_depth;
    r.layers = d->layers;
    memcpy(r.layer_pitch, d->layer_pitch, sizeof(r.layer_pitch));
    r.zt_layer_pitch = d->zt_layer_pitch;
    r.shader_layer = d->shader_layer;
    r.nviewports = d->nviewports;
    r.viewports = d->viewports;
    r.scissors = d->scissors;
    r.write_mask = d->write_mask;
    memcpy(r.blend, d->blend, sizeof(r.blend));
    for (int j = 0; j < 7; j++) {
        const struct nvgpu_target *t = &d->more[j];
        r.more[j] = { t->addr, t->pitch, t->format, t->blocklinear, t->write_mask, {} };
        memcpy(r.more[j].blend, t->blend, sizeof(t->blend));
    }
    if (d->target && d->clear) {
        render_clear(&c->compute_ch, &r, 0, d->clear_rgba, nullptr, c->compute_semva, ++c->compute_seq);
        c->draw_pending = c->compute_seq;
    }
    for (unsigned j = 0; j < 7; j++)
        if (d->more[j].addr && d->more[j].clear) {
            render_clear(&c->compute_ch, &r, j + 1, d->more[j].clear_rgba, nullptr, c->compute_semva, ++c->compute_seq);
            c->draw_pending = c->compute_seq;
        }
    // draws into the same targets go on without waiting on each other, as a pass's draws may, unless one reads what
    // the ones before it drew (serialize); other targets are a new pass, which may sample what the last one drew
    uint64_t targets[9] = { d->target, d->zt };
    for (int j = 0; j < 7; j++)
        targets[2 + j] = d->more[j].addr;
    r.wait_idle = memcmp(targets, c->pass_targets, sizeof(targets)) != 0 || d->serialize;
    memcpy(c->pass_targets, targets, sizeof(targets));
    // and the shader caches start over with it, for other shaders, and once a lap of the draw slots (whose
    // constants this one rewrote)
    r.invalidate = r.wait_idle || d->vs != c->pass_vs || d->ps != c->pass_ps || d->ti != c->pass_ti ||
                   d->ts != c->pass_ts || slot == 0;
    if (safe_sync)  // nvsafesync=1: every draw waits for the ones before and starts with clean caches
        r.wait_idle = r.invalidate = 1;
    c->pass_vs = d->vs;
    c->pass_ps = d->ps;
    c->pass_ti = d->ti;
    c->pass_ts = d->ts;
    r.zt = d->zt;
    r.zt_format = d->zt_format;
    r.zt_width = d->zt_width;
    r.zt_height = d->zt_height;
    r.zt_compressed = d->zt && c->kind_compressed;
    const struct gsp_zcull *zcull = nullptr;
    // A pass that loads what the last one left keeps what zcull learnt: the prepass of a frame, for the pass after it
    if (d->pass_first && !(d->zt && d->zt == c->zcull_zt && d->zt < c->zcull_hi && !(d->clear_zs & 1))) {
        const struct gsp_zcull *z = &gsp_get_info(gsp)->zcull;
        if (zcull_on && z->subregion_count && d->zt && (d->clear_zs & 1) && d->layers <= 1)
            zcull = z;
        else if (c->zcull_zt)
            render_zcull_off(&c->compute_ch);
        c->zcull_zt = zcull ? d->zt : 0;
        c->zcull_lo = c->kind_lo;
        c->zcull_hi = c->kind_hi;
    }
    r.zt_zcull = d->zt && d->zt == c->zcull_zt;
    if (d->zt && d->clear_zs) {
        render_clear_zs(&c->compute_ch, &r, zcull, d->clear_zs, d->clear_depth, d->clear_stencil, nullptr, c->compute_semva,
                        ++c->compute_seq);
        c->draw_pending = c->compute_seq;
    }
    uint32_t first = d->first, count = d->count, instances = d->instances;
    int base_vertex = d->base_vertex;
    uint32_t base_instance = d->base_instance;
    if (d->indirect) {  // count, instances, first, (base vertex), base instance
        uint32_t a[5];
        IOReturn ir = read_indirect(c, d->indirect, a, d->index ? 5 : 4);
        if (ir != kIOReturnSuccess)
            return ir;
        count = a[0];
        instances = a[1];
        first = a[2];
        base_vertex = d->index ? (int)a[3] : 0;
        base_instance = d->index ? a[4] : a[3];
    }
    if (!count || !instances)  // a pass that only clears, or nothing drawn
        return kIOReturnSuccess;
    uint8_t *cb = (uint8_t *)c->draw_ring->cpu + slot * DRAW_SLOT_BYTES;
    uint64_t cbva = c->draw_ringva + slot * DRAW_SLOT_BYTES;
    // what shaders read of it: the two banks' words up to the handles, and the pools. A tessellated draw has more
    if (d->tess) {
        memset(cb, 0, DRAW_SLOT_BYTES);
    } else {
        memset(cb, 0, 0x280);
        memset(cb + 0x1000, 0, 0x280);
        memset(cb + 0x2000, 0, NVGPU_MAX_TEXTURES * sizeof(d->headers[0]));
        memset(cb + 0x2800, 0, NVGPU_MAX_TEXTURES * sizeof(d->samplers[0]));
    }
    memcpy(cb + 0x160, d->vs_params, d->vs_params_size);
    // NVK_BASE_VERTEX, NVK_BASE_INSTANCE; a draw without indices has its first vertex for [[base_vertex]], as an
    // Apple GPU gives it
    int shader_base_vertex = d->index ? base_vertex : (int)first;
    memcpy(cb + 0x100, &shader_base_vertex, 4);
    memcpy(cb + 0x104, &base_instance, 4);
    memcpy(cb + 0x1000 + 0x160, d->ps_params, d->ps_params_size);
    if (d->heap.headers) {
        memcpy(cb + 0x200, d->vs_textured ? d->vs_handles : d->heap.handles, sizeof(d->heap.handles));
        memcpy(cb + 0x1000 + 0x200, d->heap.handles, sizeof(d->heap.handles));
        r.tex_headers = d->heap.headers;
        r.tex_samplers = d->heap.samplers;
        r.tex_count = d->heap.nheaders;
        r.sampler_count = d->heap.nsamplers;
    } else {
        memcpy(cb + 0x2000, d->headers, d->nheaders * sizeof(d->headers[0]));
        memcpy(cb + 0x2800, d->samplers, d->nsamplers * sizeof(d->samplers[0]));
        r.tex_headers = cbva + 0x2000;
        r.tex_samplers = cbva + 0x2800;
        r.tex_count = NVGPU_MAX_TEXTURES;
    }
    r.depth_test = d->depth_test;
    r.depth_write = d->depth_write;
    r.depth_func = d->depth_func;
    r.stencil_test = d->stencil_test;
    memcpy(r.stencil, d->stencil, sizeof(r.stencil));
    r.stencil_ref = d->stencil_ref;
    memcpy(r.stencil_read_mask, d->stencil_read_mask, sizeof(r.stencil_read_mask));
    memcpy(r.stencil_write_mask, d->stencil_write_mask, sizeof(r.stencil_write_mask));
    r.vs = d->vs;
    r.ps = d->ps;
    r.vs_regs = d->vs_regs;
    r.ps_regs = d->ps_regs;
    r.vs_cbuf = cbva;
    r.ps_cbuf = cbva + 0x1000;
    if (d->tess) {
        memcpy(cb + 0x3000 + 0x160, d->ti_params, d->ti_params_size);
        r.tess = d->tess;
        r.ti = d->ti;
        r.ts = d->ts;
        r.ti_regs = d->ti_regs;
        r.ts_regs = d->ts_regs;
        r.ti_cbuf = cbva + 0x3000;
    }
    r.primitive = d->primitive;
    r.first = first;
    r.count = count;
    r.instances = instances;
    r.base_vertex = base_vertex;
    r.base_instance = base_instance;
    r.index = d->index;
    r.index_size = d->index_size;
    r.index_format = d->index_format;
    r.cull_face = d->cull_face;
    r.front_ccw = d->front_ccw;
    r.point_size = d->point_size;
    r.line_width = d->line_width;
    r.provoking_last = d->provoking_last;
    r.restart = d->restart;
    r.restart_index = d->restart_index;
    r.window_coords = d->window_coords;
    r.depth_clamp = d->depth_clamp;
    r.fill_lines = d->fill_lines;
    memcpy(r.depth_bias, d->depth_bias, sizeof(r.depth_bias));
    memcpy(r.blend_const, d->blend_const, sizeof(r.blend_const));
    r.visibility = d->visibility;
    r.visibility_reset = d->visibility_reset;
    memcpy(r.viewport, d->viewport, sizeof(r.viewport));
    memcpy(r.scissor, d->scissor, sizeof(r.scissor));
    memcpy(r.attributes, d->attributes, sizeof(r.attributes));
    for (int s = 0; s < RENDER_STREAMS; s++) {
        r.streams[s].addr = d->streams[s].addr;
        r.streams[s].stride = d->streams[s].stride;
        r.streams[s].size = d->streams[s].size;
        r.streams[s].divisor = d->streams[s].divisor;
    }
    render_draw(&c->compute_ch, &r, nullptr, c->compute_semva, ++c->compute_seq);
    if (prof_on && prof_ready(c)) {
        uint32_t i = c->prof_n++;
        c->prof_draws[i % PROF_RING] = { r.vs, r.ps, r.count * r.instances,
                                         (r.wait_idle ? 1u : 0) | (r.invalidate ? 2u : 0) | (after_copy ? 4u : 0) |
                                             (d->clear ? 8u : 0) | (d->tess ? 16u : 0) | (d->heap.headers ? 32u : 0) |
                                             (d->serialize ? 64u : 0),
                                         MIN(r.scissor[2], r.width) * MIN(r.scissor[3], r.height),
                                         (uint16_t)r.width, (uint16_t)r.height };
        render_stamp(&c->compute_ch, c->prof_va + (i % PROF_RING) * 16, i + 1);
        chan_kick(&c->compute_ch);
    }
    c->launches[c->compute_seq % 8] = { c->compute_seq, { r.count, r.instances, 0 }, {}, d->vs_regs, r.ps, r.vs };
    memcpy(c->launches[c->compute_seq % 8].params, cb + 0x160, 16);
    memcpy(c->launches[c->compute_seq % 8].params + 2, cb + 0x1000 + 0x160, 16);
    issued(c);
    c->draw_slot_seq[slot] = c->draw_pending = c->compute_seq;
    return kIOReturnSuccess;
}

IOReturn Gpu::drain(gpu_ctx *c)
{
    memset(c->pass_targets, 0, sizeof(c->pass_targets));  // a copy, a dispatch or the CPU may write memory next
    if (!c->draw_pending)
        return kIOReturnSuccess;
    uint64_t t0 = time_us();
    if ((int32_t)(*c->compute_sem - c->draw_pending) < 0)
        stats[ST_DRAIN_WAITS]++;
    // up to 2 s, but on a fault only until GSP-RM's word of it, which has the channel made anew and the work lost
    // run again
    int r = -1;
    for (int t = 0; t < 200 && r; t++) {
        r = render_wait(c->compute_sem, c->draw_pending, 10);
        if (r && gsp) {
            gsp_poll(gsp);
            if (c->compute_faulted || c->copy_faulted)
                recover(c);
        }
    }
    c->draw_pending = 0;
    stats[ST_DRAIN_US] += time_us() - t0;
    busy_close(c);
    if (r)
        log_msg("Malachite: draws never finished (semaphore 0x%x)\n", *c->compute_sem);
    return r ? kIOReturnTimeout : kIOReturnSuccess;
}

bool Gpu::usable()
{
    if (!suspended)
        return true;
    IOLockLock(lock);
    bool up = awake();
    IOLockUnlock(lock);
    return up;
}

// the GPU work and mappings of a task's context, named by the handle in the call's last argument (0 for the
// kernel's); one closed already gets kIOReturnNotReady
#define WITH_CTX(arg, call)                                  \
    do {                                                     \
        take();                                              \
        gpu_ctx *c = ctx_of((uintptr_t)(arg));               \
        IOReturn r = !c          ? kIOReturnNotReady         \
                     : c->refused_until > time_us() ? kIOReturnNotPermitted \
                                  : (call);                  \
        give();                                              \
        return r;                                            \
    } while (0)

// Draws, dispatches and 2D copies come here directly: callPlatformFunction's name is a symbol looked up under a lock and
// then compared with every call it might be, thousands of times a frame.
IOReturn Gpu::queue_dispatch(void *ctx, const struct nvgpu_dispatch *d)
{
    if (!usable())
        return kIOReturnNotReady;
    WITH_CTX(ctx, (stats[ST_DISPATCH]++, busy_open(c, launch(c, d, nullptr))));
}

IOReturn Gpu::queue_draw(void *ctx, const struct nvgpu_draw *d, const char **why)
{
    if (!usable())
        return kIOReturnNotReady;
    WITH_CTX(ctx, (stats[ST_DRAW]++, busy_open(c, TIMED(ST_DRAW_US, launch(c, nullptr, d, why)))));
}

IOReturn Gpu::queue_copy2d(void *ctx, const struct nvgpu_copy2d *p)
{
    if (!usable())
        return kIOReturnNotReady;
    WITH_CTX(ctx, (stats[ST_COPY2D]++, TIMED(ST_COPY2D_US, gpu_copy2d(c, p))));
}

IOReturn Gpu::queue_compress(void *ctx, uint64_t va)
{
    if (!usable())
        return kIOReturnNotReady;
    WITH_CTX(ctx, set_kind(c, va, 0x06, true));
}

IOReturn queue_dispatch(IOService *gpu, void *ctx, const struct nvgpu_dispatch *d)
{
    return static_cast<Gpu *>(gpu)->queue_dispatch(ctx, d);
}

IOReturn queue_draw(IOService *gpu, void *ctx, const struct nvgpu_draw *d, const char **why)
{
    return static_cast<Gpu *>(gpu)->queue_draw(ctx, d, why);
}

IOReturn queue_copy2d(IOService *gpu, void *ctx, const struct nvgpu_copy2d *p)
{
    return static_cast<Gpu *>(gpu)->queue_copy2d(ctx, p);
}

IOReturn queue_compress(IOService *gpu, void *ctx, uint64_t va)
{
    return static_cast<Gpu *>(gpu)->queue_compress(ctx, va);
}

IOReturn Gpu::callPlatformFunction(const OSSymbol *name, bool wait, void *p1, void *p2, void *p3, void *p4)
{
    // through a power cycle: everything was drained before it, so a semaphore asked about is done; the rest waits
    if (suspended) {
        if (name->isEqualTo(NVGPU_FN_DRAW_DONE) || name->isEqualTo(NVGPU_FN_FENCE))
            return kIOReturnSuccess;
        if (!usable())
            return kIOReturnNotReady;
    }
    // mapping memory is no GPU work: a context whose work is refused still gets its buffers (an allocation that fails
    // crashes the apps that don't expect it to)
#define WITH_CTX_MAP(arg, call)                              \
    do {                                                     \
        take();                                              \
        gpu_ctx *c = ctx_of((uintptr_t)(arg));               \
        IOReturn r = c ? (call) : kIOReturnNotReady;         \
        give();                                              \
        return r;                                            \
    } while (0)
    if (name->isEqualTo(NVGPU_FN_CTX_OPEN)) {
        take();
        *(uintptr_t *)p1 = ctx_open();
        give();
        return *(uintptr_t *)p1 ? kIOReturnSuccess : kIOReturnNoResources;
    }
    if (name->isEqualTo(NVGPU_FN_CTX_CLOSE)) {
        take();
        ctx_close((uintptr_t)p1);
        give();
        return kIOReturnSuccess;
    }
    if (name->isEqualTo(NVGPU_FN_MAP))
        WITH_CTX_MAP(p3, (stats[ST_MAP]++, TIMED(ST_MAP_US, gpu_map(c, (IOMemoryDescriptor *)p1, (uint64_t *)p2))));
    if (name->isEqualTo(NVGPU_FN_VRAM_ALLOC)) {
        take();
        uint64_t size = (uint64_t)(uintptr_t)p1;
        *(uint64_t *)p2 = size && gpu_bridge_init(&kctx) ? vram_alloc(gsp_vram(gsp), size, 0x10000) : 0;
        give();
        return *(uint64_t *)p2 ? kIOReturnSuccess : kIOReturnNoMemory;
    }
    if (name->isEqualTo(NVGPU_FN_VRAM_FREE)) {
        take();
        retire_deferred();
        if (nvram_deferred == MAX_DEFERRED) {
            drain_all();
            retire_deferred();
        }
        if (nvram_deferred < MAX_DEFERRED)  // a draw still in flight, any task's, may be reading it
            vram_deferred[nvram_deferred++] = { (uint64_t)(uintptr_t)p1, (uint64_t)(uintptr_t)p2 };
        else
            vram_free(gsp_vram(gsp), (uint64_t)(uintptr_t)p1, (uint64_t)(uintptr_t)p2);
        retire_deferred();
        give();
        return kIOReturnSuccess;
    }
    if (name->isEqualTo(NVGPU_FN_MAP_VRAM))
        WITH_CTX_MAP(p4, gpu_map_vram(c, (uint64_t)(uintptr_t)p1, (uint64_t)(uintptr_t)p2, (uint64_t *)p3));
    if (name->isEqualTo(NVGPU_FN_UNMAP)) {
        take();
        stats[ST_UNMAP]++;
        uint64_t t0 = time_us();
        gpu_ctx *owner = ctx_of((uintptr_t)p3);
        int slot = owner ? map_slot((uint64_t)(uintptr_t)p1) : -1;
        retire_deferred();
        IOReturn r;
        if (slot >= 0 && gpu_maps[slot].ctx == owner && owner->compute_sem && !caught_up(owner) &&
            ndeferred < MAX_DEFERRED) {  // a draw of its task's still in flight may be using it
            deferred[ndeferred++] = { owner, (uint64_t)(uintptr_t)p1, owner->compute_seq };
            r = kIOReturnSuccess;
        } else {
            if (slot >= 0 && gpu_maps[slot].ctx == owner)
                drain(owner);
            r = owner ? gpu_unmap((uint64_t)(uintptr_t)p1, (uint64_t)(uintptr_t)p2, owner) : kIOReturnNotReady;
        }
        stats[ST_UNMAP_US] += time_us() - t0;
        give();
        return r;
    }
    if (name->isEqualTo(NVGPU_FN_COPY))
        WITH_CTX(p4, (stats[ST_COPY]++, drain(c),
                      gpu_copy(c, (uint64_t)(uintptr_t)p1, (uint64_t)(uintptr_t)p2, (uint64_t)(uintptr_t)p3)));
    if (name->isEqualTo(NVGPU_FN_PIPE_VBL)) {
        uintptr_t i = (uintptr_t)p3;
        if (i >= SCREENS)
            return kIOReturnBadArgument;
        IOLockLock(evlock);
        screens[i].pipe_fn = (void (*)(void *))p1;
        screens[i].pipe_ctx = p2;
        IOLockUnlock(evlock);
        return kIOReturnSuccess;
    }
    if (name->isEqualTo(NVGPU_FN_PRESENT))
        return present_frame((struct nvgpu_present *)p1);
    if (name->isEqualTo(NVGPU_FN_FILL))
        WITH_CTX(p4, (stats[ST_FILL]++, drain(c),
                      gpu_fill(c, (uint64_t)(uintptr_t)p1, (uint64_t)(uintptr_t)p2, (uint32_t)(uintptr_t)p3)));
    if (name->isEqualTo(NVGPU_FN_SYNC))
        WITH_CTX(p1, (stats[ST_SYNC]++, TIMED(ST_SYNC_US, settle(drain(c)))));
    if (name->isEqualTo(NVGPU_FN_DRAW_FENCE))
        // at the end of a command buffer: the next one's draws start a pass, as the CPU may have written memory
        WITH_CTX(p2, (*(uint32_t *)p1 = c->draw_pending, memset(c->pass_targets, 0, sizeof(c->pass_targets)),
                      kIOReturnSuccess));
    if (name->isEqualTo(NVGPU_FN_ASYNC)) {
        take();
        gpu_ctx *c = ctx_of((uintptr_t)p1);
        bool ws = c && !strcmp(c->owner, "WindowServer");
        give();
        return ws ? kIOReturnSuccess : kIOReturnUnsupported;
    }
    if (name->isEqualTo(NVGPU_FN_CTX_BUSY)) {  // no lock, like DRAW_DONE: Accelerator asks as it submits
        gpu_ctx *c = ctx_of((uintptr_t)p1);
        *(uint64_t *)p2 = c ? busy_take(c) : 0;
        return kIOReturnSuccess;
    }
    if (name->isEqualTo(NVGPU_FN_DRAW_DONE)) {  // no lock: a semaphore read
        gpu_ctx *c = ctx_of((uintptr_t)p2);
        if (c && caught_up(c))
            busy_close(c);
        if (!c || !c->compute_sem || (int32_t)(*c->compute_sem - (uint32_t)(uintptr_t)p1) >= 0)
            return kIOReturnSuccess;
        // still going: unless its channel faulted (word GSP-RM may have sent only now), which has it made anew and
        // the work lost run again
        if (gsp && (gsp_pending(gsp) || c->compute_faulted)) {
            take();
            if ((c = ctx_of((uintptr_t)p2))) {
                gsp_poll(gsp);
                if (c->compute_faulted)
                    compute_bridge_init(c);
            }
            give();
        }
        return kIOReturnBusy;
    }
#undef WITH_CTX
    if (name->isEqualTo(NVGPU_FN_STATS))
        return gpu_stats((struct nvgpu_stats *)p1);
    if (name->isEqualTo(NVGPU_FN_FENCE)) {
        uint32_t fence = (uint32_t)(uintptr_t)p1;
        // not under the lock: this runs from the vblank handler, which the lock's holders may be waiting on
        return kctx.copy_sem && !copy_done(fence) ? kIOReturnBusy : kIOReturnSuccess;
    }
    if (name->isEqualTo(NVGPU_FN_GAMMA)) {
        uintptr_t i = (uintptr_t)p2;
        if (i >= SCREENS)
            return kIOReturnBadArgument;
        take();
        IOReturn r = set_gamma(screens[i], (const uint16_t *)p1);
        give();
        return r;
    }
    return IOService::callPlatformFunction(name, wait, p1, p2, p3, p4);
}

// Called every 250 ms by Accelerator, so it only reads: BAR0, our allocator's counters, the mapping table and the
// semaphores. TryLock because a launch can hold the lock for as long as it waits on the GPU, and a missed tick costs
// nothing.
IOReturn Gpu::gpu_stats(struct nvgpu_stats *out)
{
    if (!IOLockTryLock(lock))
        return kIOReturnBusy;
    memset(out, 0, sizeof(*out));
    // TSENS: the core temperature in 1/256 C from bit 8. A GPU in a low power state reads back all ones and an
    // unclocked sensor 0, neither a temperature.
    uint32_t raw = rd32(&gpu, 0x20460);
    uint32_t c = (raw & 0x1fff8) >> 8;
    out->temp_c = raw != ~0u && c >= 1 && c <= 125 ? c : 0;
    if (gsp) {
        const struct vram *v = gsp_vram(gsp);
        out->vram_total = v->total;
        out->vram_used = v->used < v->total ? v->used : v->total;
    }
    for (int i = 0; i < gpu_maps_top; i++)
        if (gpu_maps[i].va && !gpu_maps[i].vram)
            out->sys_mapped += gpu_maps[i].dma.size;
    // a context whose work is done has its interval closed here if nothing else has; the union is brought up to now
    // by a step that changes nothing
    uint64_t now = time_us();
    for (int i = -1; i < MAX_CTXS; i++) {
        gpu_ctx *x = i < 0 ? &kctx : ctxs[i];
        if (x && caught_up(x))
            busy_close(x);
    }
    busy_step(0, now);
    out->busy_us = __atomic_load_n(&busy_total, __ATOMIC_RELAXED);
    out->now_us = now;
    out->recoveries = recoveries;
    out->dma_bytes = dma_live_bytes();
    IOLockUnlock(lock);
    return kIOReturnSuccess;
}

IOReturn Gpu::screen_call(int i, const OSSymbol *name, void *p1, void *p2, void *p3, void *p4)
{
    if (i < 0 || i >= SCREENS)
        return kIOReturnBadArgument;
    screen &sc = screens[i];
    if (name->isEqualTo(SURFACE_FN_GAMMA)) {
        take();
        IOReturn r = set_gamma(sc, (const uint16_t *)p1);
        give();
        return r;
    }
    if (name->isEqualTo(SURFACE_FN_CURSOR_MOVE)) {
        __sync_fetch_and_add(&stats[ST_CURSOR], 1);
        sc.cursor_x = (int)(intptr_t)p1;
        sc.cursor_y = (int)(intptr_t)p2;
        if (!sc.shown || !disp || !disp->head[i].curs_img)
            return kIOReturnNotReady;
        disp_cursor_move(disp, i, (int)(intptr_t)p1, (int)(intptr_t)p2);
        return kIOReturnSuccess;
    }
    if (name->isEqualTo(SURFACE_FN_CURSOR_IMAGE)) {
        take();
        static int logged;
        if (logged++ < 12)
            log_msg("Malachite: screen %d cursor %ux%u%s\n", i, (uint32_t)(uintptr_t)p2, (uint32_t)(uintptr_t)p3,
                    p4 ? ", rows were padded" : "");
        keep_cursor(sc, (const uint32_t *)p1, (uint32_t)(uintptr_t)p2, (uint32_t)(uintptr_t)p3);
        IOReturn r = sc.shown && disp && !disp_cursor_image(disp, i, (const uint32_t *)p1, (uint32_t)(uintptr_t)p2,
                                                             (uint32_t)(uintptr_t)p3)
                         ? kIOReturnSuccess
                         : kIOReturnError;
        give();
        return r;
    }
    if (name->isEqualTo(SURFACE_FN_POWER)) {
        take();
        IOReturn r = set_power(sc, (uintptr_t)p1 >= 2);
        give();
        return r;
    }
    if (name->isEqualTo(SURFACE_FN_SET_MODE)) {
        take();
        IOReturn r = set_mode(sc, (uint32_t)(uintptr_t)p1);
        give();
        return r;
    }
    if (name->isEqualTo(SURFACE_FN_SET_VBL)) {
        IOLockLock(evlock);
        sc.vbl_fn = (void (*)(void *))p1;
        sc.vbl_ctx = p2;
        IOLockUnlock(evlock);
        return kIOReturnSuccess;
    }
    return kIOReturnUnsupported;
}

// the nubs Framebuffer matches on, one per screen,
// published at start so WindowServer finds the framebuffers when it starts; each stays dark until its head is lit
void Gpu::create_nubs()
{
    uint64_t bar1_phys = bar_phys(&gpu, 1), bar1_len = bar_size(&gpu, 1);
    // without them Framebuffer never loads and macOS has no screen, though the GPU works: say why
    if (!bar1_len) {
        log_msg("Malachite: macOS gave the GPU no BAR1 and none could be placed, so no screens (Resizable BAR on? macOS "
                "often leaves a BAR over 1 GB unassigned: turn it off, or limit it with OpenCore's "
                "ResizeAppleGpuBars)\n");
        return;
    }
    if (bar1_len < BAR1_FB_OFF + SCREENS * BAR1_FB_SIZE) {
        log_msg("Malachite: BAR1 is %llu MB, too small for the screens' framebuffers\n",
                (unsigned long long)(bar1_len >> 20));
        return;
    }
    log_msg("Malachite: BAR1 %llu MB at %#llx\n", (unsigned long long)(bar1_len >> 20), (unsigned long long)bar1_phys);
    for (screen &sc : screens) {
        OSDictionary *props = OSDictionary::withCapacity(10);
        if (!props)
            return;
        const struct { const char *key; uint64_t v; } nums[] = {
            { SURFACE_APERTURE, bar1_phys + BAR1_FB_OFF + index(sc) * BAR1_FB_SIZE },
            { SURFACE_APERTURE_SIZE, BAR1_FB_SIZE },
            { SURFACE_SCREEN, (uint64_t)index(sc) },
            // the mode the head will most likely be lit with, so the hotplug doesn't change the geometry under
            // WindowServer; publish_surface() replaces it with the real one
            { SURFACE_WIDTH, 1920 }, { SURFACE_HEIGHT, 1080 }, { SURFACE_PITCH, 7680 }, { SURFACE_REFRESH, 60ull << 16 },
        };
        for (const auto &e : nums) {
            OSNumber *n = OSNumber::withNumber(e.v, 64);
            props->setObject(e.key, n);
            OSSafeReleaseNULL(n);
        }
        props->setObject(SURFACE_ONLINE, kOSBooleanFalse);
        props->setObject(SURFACE_HW_CURSOR, kOSBooleanTrue);  // IOFramebuffer asks when it opens, before the head lights
        SurfaceNub *nub = new SurfaceNub;
        if (nub && nub->init(props) && nub->attach(this)) {
            nub->screen = index(sc);
            nub->setName(SURFACE_NUB_NAME);
            // a power child of Gpu, before Framebuffer comes to join it: the framebuffer then sleeps before Gpu suspends
            // the GPU and wakes after it resumes
            static IOPMPowerState states[2] = {
                { kIOPMPowerStateVersion1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { kIOPMPowerStateVersion1, kIOPMPowerOn, kIOPMPowerOn, kIOPMPowerOn, 0, 0, 0, 0, 0, 0, 0, 0 },
            };
            nub->PMinit();
            joinPMtree(nub);
            nub->registerPowerDriver(nub, states, 2);
            nub->registerService();
            sc.nub = nub;
        } else {
            OSSafeReleaseNULL(nub);
        }
        props->release();
    }
}

static uint64_t refresh_16_16(const struct disp_mode *m)
{
    uint64_t total = (uint64_t)(m->hactive + m->hblank) * (m->vactive + m->vblank);
    return total ? ((uint64_t)m->clock * 1000 << 16) / total : 0;
}

static void set_num(IOService *s, const char *key, uint64_t v)
{
    OSNumber *n = OSNumber::withNumber(v, 64);
    s->setProperty(key, n);
    OSSafeReleaseNULL(n);
}

// EDID detailed timing byte 17: bit 1 hsync positive, bit 2 vsync positive (digital separate sync)
static OSData *timing_of(const struct disp_mode *m)
{
    // as macOS expects of a mode: its own size, and a clock that gives the same refresh rate over it
    uint64_t total = (uint64_t)(m->hactive + m->hblank) * (m->vactive + m->vblank);
    uint64_t shown = (uint64_t)(mode_w(m) + m->hblank) * (mode_h(m) + m->vblank);
    const struct surface_timing t = { (uint32_t)(m->clock * shown / total), (uint16_t)mode_w(m), m->hblank, m->hsync_off, m->hsync_len,
                                   (uint16_t)mode_h(m), m->vblank, m->vsync_off, m->vsync_len,
                                   (uint8_t)(m->flags >> 1 & 1), (uint8_t)(m->flags >> 2 & 1) };
    return OSData::withBytes(&t, sizeof(t));
}

// mode m as SURFACE_MODES lists it
static void list_mode(OSArray *modes, uint32_t id, const struct disp_mode *m)
{
    OSDictionary *e = OSDictionary::withCapacity(5);
    if (!e)
        return;
    const struct { const char *key; uint64_t v; } f[] = {
        { SURFACE_MODE_ID, id },
        { SURFACE_WIDTH, mode_w(m) },
        { SURFACE_HEIGHT, mode_h(m) },
        { SURFACE_REFRESH, refresh_16_16(m) },
    };
    for (const auto &k : f) {
        OSNumber *n = OSNumber::withNumber(k.v, 64);
        e->setObject(k.key, n);
        OSSafeReleaseNULL(n);
    }
    if (OSData *t = timing_of(m)) {
        e->setObject(SURFACE_TIMING, t);
        t->release();
    }
    modes->setObject(e);
    e->release();
}

// what a head shows (mode sc.mode of output sc.out, from surface sc.shown) to Framebuffer through the screen's BAR1
// framebuffer slot, under the lock; hotplug: tell Framebuffer, as a newly lit head looks to it like a plugged-in monitor
void Gpu::publish_surface(screen &sc, bool hotplug)
{
    const struct disp_output *o = &disp->out[sc.out];
    const struct disp_mode *m = o->nmodes ? &o->modes[sc.mode - 1] : &o->mode;
    IOService *nub = sc.nub;
    if (!nub || rmfe_bar1_fb(fe, index(sc), sc.shown, sc.size)) {
        log_msg("Malachite: no framebuffer for screen %d\n", index(sc));
        return;
    }
    set_num(nub, SURFACE_WIDTH, mode_w(m));
    set_num(nub, SURFACE_HEIGHT, mode_h(m));
    set_num(nub, SURFACE_PITCH, sc.pitch);
    set_num(nub, SURFACE_REFRESH, refresh_16_16(m));
    set_num(nub, SURFACE_MODE, sc.mode);
    if (OSData *timing = timing_of(m)) {
        nub->setProperty(SURFACE_TIMING, timing);
        timing->release();
    }
    if (OSArray *modes = OSArray::withCapacity(o->nmodes)) {
        for (uint32_t i = 0; i < o->nmodes; i++)
            if (!o->modes[i].view_w || i + 1 == sc.mode || disp_mode_possible(disp, index(sc), &o->modes[i]))
                list_mode(modes, i + 1, &o->modes[i]);
        nub->setProperty(SURFACE_MODES, modes);
        modes->release();
    }
    OSData *edid = OSData::withBytes(o->edid, o->edid_size);
    nub->setProperty(SURFACE_EDID, edid);
    OSSafeReleaseNULL(edid);
    if (!hotplug)
        return;
    nub->setProperty(SURFACE_ONLINE, kOSBooleanTrue);
    sc.changed = true;  // tell_nubs() tells Framebuffer once the lock is dropped: it may set a mode in response
    log_msg("Malachite: screen %d lit, %ux%u\n", index(sc), mode_w(m), mode_h(m));
}

// display sleep and wake: the head off its OR and the sink in standby, then the same mode and surface back
IOReturn Gpu::set_power(screen &sc, bool on)
{
    if (!disp || !sc.shown)
        return kIOReturnNotReady;
    if (on == !sc.asleep)
        return kIOReturnSuccess;
    if (sc.unplugged) {  // nothing to drive; hotplug lights the head when the monitor is back
        sc.asleep = !on;
        return kIOReturnSuccess;
    }
    struct disp_output *o = &disp->out[sc.out];
    const struct disp_mode *t = o->nmodes ? &o->modes[sc.mode - 1] : &o->mode;
    if (!on) {
        if (disp_blank(disp, index(sc), sc.out))
            return kIOReturnError;
        sc.asleep = true;
        return kIOReturnSuccess;
    }
    // what the head was reading when it went dark: presents put WindowServer's frames on the surfaces it flips
    // between, and shown holds the boot screen
    uint64_t fb = surface_of(sc, sc.cur);
    if (modeset(sc, t, fb))
        return kIOReturnError;
    sc.asleep = false;
    if (sc.gamma)  // one may have come while the head was off
        disp_gamma(disp, index(sc), sc.gamma);
    return kIOReturnSuccess;
}

// a mode id of the screen's output: a new surface of its size, the modeset, then Framebuffer's slot moved onto it
IOReturn Gpu::set_mode(screen &sc, uint32_t id)
{
    if (!disp || !sc.shown)
        return kIOReturnNotReady;
    struct disp_output *o = &disp->out[sc.out];
    if (id < 1 || id > o->nmodes)
        return kIOReturnBadArgument;
    if (id == sc.mode)
        return kIOReturnSuccess;
    // a DisplayPort sink only lists modes its fastest link carries; disp_modeset trains the one this mode needs
    const struct disp_mode *t = &o->modes[id - 1];
    if (t->view_w && !disp_mode_possible(disp, index(sc), t)) {
        log_msg("Malachite: screen %d can't show %ux%u at %u kHz\n", index(sc), mode_w(t), mode_h(t), t->clock);
        return kIOReturnUnsupportedMode;
    }
    uint32_t pitch = (mode_w(t) * 4 + 255) & ~255u;
    uint64_t size = (uint64_t)pitch * mode_h(t);
    if (size > BAR1_FB_SIZE)
        return kIOReturnUnsupportedMode;
    uint64_t fb = vram_alloc(gsp_vram(gsp), size, 0x100000);
    if (!fb)
        return kIOReturnNoMemory;
    vram_zero(&gpu, fb, size);  // black until WindowServer draws
    reset_flips(sc);
    if (int r = disp_modeset(disp, index(sc), sc.out, t, fb, pitch)) {
        vram_free(gsp_vram(gsp), fb, size);
        // no link for the new mode: the old one back, on its surface
        if (r == DISP_NO_OUTPUT && sc.mode && modeset(sc, &o->modes[sc.mode - 1]))
            log_msg("Malachite: screen %d's old mode didn't come back either\n", index(sc));
        return kIOReturnError;
    }
    vram_free(gsp_vram(gsp), sc.shown, sc.size);
    sc.shown = fb;
    sc.size = size;
    sc.pitch = pitch;
    sc.mode = id;
    publish_surface(sc, false);
    log_msg("Malachite: screen %d mode %u: %ux%u\n", index(sc), id, mode_w(t), mode_h(t));
    return kIOReturnSuccess;
}

void Gpu::darken_fb()
{
    for (screen &sc : screens) {
        if (!sc.nub || sc.nub->getProperty(SURFACE_ONLINE) != kOSBooleanTrue)
            continue;
        sc.nub->setProperty(SURFACE_ONLINE, kOSBooleanFalse);
        sc.nub->messageClients(kSurfaceChanged);
    }
}

// what screen 0's head shows into user memory after a struct malachite_shot: the surface Framebuffer sees through BAR1's
// framebuffer slot, or a VRAM surface it was flipped to (the back one, or a shadow), brought over by the copy engine
IOReturn Gpu::read_shown(uint64_t addr, uint64_t len, task_t task)
{
    const screen &sc = screens[0];
    if (!sc.shown)
        return kIOReturnNotReady;
    const struct disp_output *o = &disp->out[sc.out];
    const struct disp_mode *m = o->nmodes ? &o->modes[sc.mode - 1] : &o->mode;
    struct malachite_shot hdr = { mode_w(m), mode_h(m), sc.pitch, 0 };
    uint64_t bytes = (uint64_t)sc.pitch * mode_h(m);
    if (len < sizeof(hdr) + bytes)
        return kIOReturnBadArgument;
    const void *pixels = nullptr;
    IOMemoryDescriptor *fb = nullptr;
    IOMemoryMap *map = nullptr;
    struct dma *tmp = nullptr;
    // the head reads one of the back surfaces while a flip chain runs
    const uint64_t shown = sc.cur ? surface_of(sc, sc.cur) : 0;
    if (shown) {
        uint64_t size = (bytes + 0xfff) & ~0xfffull, tva = 0;
        if (!shadow_va || !(tmp = gsp_dma(gsp, size, 0)))
            return kIOReturnNoMemory;
        tva = vmm_alloc(&kctx.vm, size, 0x1000);
        volatile uint32_t *sem = (volatile uint32_t *)kctx.copy_sem->cpu;
        if (!tva || vmm_map_sys(&kctx.vm, tva, tmp, 0, size, 0) || vmm_map_vram(&kctx.vm, shadow_va, shown, size, 0) ||
            (vmm_flush(&gpu), chan_copy(&kctx.copy_ch, shadow_va, tva, size, sem, kctx.copy_semva, ++kctx.copy_seq))) {
            gsp_dma_free(gsp, tmp);
            return kIOReturnError;
        }
        vmm_unmap(&kctx.vm, shadow_va, size);
        vmm_unmap(&kctx.vm, tva, size);
        pixels = tmp->cpu;
    } else {
        uint64_t bar1 = bar_phys(&gpu, 1);
        fb = bar1 ? IOMemoryDescriptor::withPhysicalAddress(bar1 + BAR1_FB_OFF, bytes, kIODirectionIn) : nullptr;
        map = fb ? fb->map(kIOMapInhibitCache) : nullptr;
        pixels = map ? (const void *)map->getVirtualAddress() : nullptr;
    }
    IOMemoryDescriptor *dst = IOMemoryDescriptor::withAddressRange(addr, sizeof(hdr) + bytes, kIODirectionIn, task);
    IOReturn ret = kIOReturnNoMemory;
    if (pixels && dst && dst->prepare() == kIOReturnSuccess) {
        dst->writeBytes(0, &hdr, sizeof(hdr));
        dst->writeBytes(sizeof(hdr), pixels, bytes);
        dst->complete();
        ret = kIOReturnSuccess;
    }
    OSSafeReleaseNULL(dst);
    OSSafeReleaseNULL(map);
    OSSafeReleaseNULL(fb);
    if (tmp)
        gsp_dma_free(gsp, tmp);
    return ret;
}

// where MALACHITE_H264_DECODE puts the pictures in the caller's memory
struct frame_sink {
    task_t task;
    uint64_t addr, len, off;
    uint32_t count;
    bool fault;
};

static int sink_frame(void *arg, const struct video_frame *f)
{
    frame_sink *o = (frame_sink *)arg;
    struct malachite_frame h = { f->index, f->idr_count, f->poc, f->width, f->height, 0 };
    uint64_t bytes = (uint64_t)f->width * f->height * 3 / 2;
    if (o->off + sizeof(h) + bytes > o->len)
        return 1;
    if (copy_to_user(o->task, o->addr + o->off, &h, sizeof(h)) ||
        copy_to_user(o->task, o->addr + o->off + sizeof(h), f->nv12, bytes)) {
        o->fault = true;
        return 1;
    }
    o->off += sizeof(h) + bytes;
    o->count++;
    return 0;
}

IOReturn Gpu::h264_decode(const uint64_t *args, task_t task)
{
    uint64_t len = args[1];
    if (!len || len > 16u << 20 || args[3] < sizeof(struct malachite_frames))
        return kIOReturnBadArgument;
    uint8_t *stream = (uint8_t *)IOMalloc(len);
    if (!stream)
        return kIOReturnNoMemory;
    frame_sink sink = { task, args[2], args[3], sizeof(struct malachite_frames) };
    IOReturn ret = kIOReturnBadArgument;
    if (!copy_from_user(task, stream, args[0], len)) {
        int n = video_decode_h264(&kctx.video, stream, len, (uint32_t)args[4], sink_frame, &sink);
        // the pictures before a failed one still go out
        struct malachite_frames hdr = { sink.count };
        if (sink.fault || copy_to_user(task, args[2], &hdr, sizeof(hdr)))
            ret = kIOReturnBadArgument;
        else
            ret = n < 0 ? kIOReturnError : kIOReturnSuccess;
    }
    IOFree(stream, len);
    return ret;
}

// a head lit for output out in its preferred mode, on a new surface: screen 0 keeps what the EFI firmware showed,
// the others start black
IOReturn Gpu::light(screen &sc, int out)
{
    const int head = index(sc);
    uint32_t w = disp->out[out].mode.hactive, h = disp->out[out].mode.vactive, pitch = (w * 4 + 255) & ~255u;
    uint64_t size = (uint64_t)pitch * h;
    if (size > BAR1_FB_SIZE)
        return kIOReturnUnsupportedMode;
    uint64_t fb = vram_alloc(gsp_vram(gsp), size, 0x100000);
    if (!fb)
        return kIOReturnNoMemory;
    if (head || disp_copy_firmware(&gpu, fb, pitch, w, h))
        vram_zero(&gpu, fb, size);  // black until WindowServer draws
    reset_flips(sc);
    if (int r = disp_modeset(disp, head, out, &disp->out[out].mode, fb, pitch)) {
        vram_free(gsp_vram(gsp), fb, size);
        return r == DISP_NO_OUTPUT ? kIOReturnOffline : kIOReturnError;
    }
    sc.shown = fb;
    sc.out = out;
    sc.mode = 1;  // modes[0] is the preferred one
    sc.size = size;
    sc.pitch = pitch;
    sc.asleep = sc.unplugged = false;
    disp_vblank(disp, head, 1);
    if (disp_cursor_init(disp, head) == 0)
        disp_cursor_move(disp, head, -CURS_OFFSCREEN, -CURS_OFFSCREEN);
    else
        log_msg("Malachite: no hardware cursor on screen %d\n", head);
    publish_surface(sc, true);
    return kIOReturnSuccess;
}

// every connected output no screen shows yet lit on a free one (screen i is head i, so only as many as there are
// heads), under the lock. What the EFI firmware lit goes first, head by head, so screen 0 keeps the firmware's
// screen. An output that can't be driven (a DisplayPort link that won't train) leaves the others to light.
IOReturn Gpu::show()
{
    if (!disp)
        return kIOReturnNotReady;  // disp first
    int order[DISP_MAX_OUTPUTS + DISP_MAX_HEADS], n = 0;  // the firmware may show one display on two heads
    uint32_t fw = 0;
    for (uint32_t h = 0; h < disp->heads; h++) {
        fw |= disp->fw_display[h];
        for (int i = 0; i < disp->n; i++)
            if (disp->fw_display[h] && disp->out[i].id == disp->fw_display[h])
                order[n++] = i;
    }
    for (int i = 0; i < disp->n; i++)
        if (!(disp->out[i].id & fw))
            order[n++] = i;
    bool skipped = false;
    for (int k = 0; k < n; k++) {
        const int i = order[k];
        if (!disp->out[i].connected || !disp->out[i].mode.clock)
            continue;
        bool driven = false;
        screen *free = nullptr;
        for (screen &sc : screens) {
            driven |= sc.shown && sc.out == i;
            if (!sc.shown && !free && index(sc) < (int)disp->heads)
                free = &sc;
        }
        if (driven || !free)
            continue;
        IOReturn r = light(*free, i);
        if (r == kIOReturnOffline) {
            log_msg("Malachite: display %#x can't be lit\n", disp->out[i].id);
            skipped = true;
        } else if (r != kIOReturnSuccess) {
            disp_reset(disp);
            show_failed = true;
            failed_connected = disp->connected;
            return r;
        }
    }
    // polls don't retry an output that couldn't be lit until what is connected changes
    show_failed = skipped;
    failed_connected = disp->connected;
    return lit() ? kIOReturnSuccess : skipped ? kIOReturnOffline : kIOReturnNoDevice;
}

// after GSP-RM said a display came or went, under the lock: light a head for a monitor that turned up, blank one
// whose monitor went (Framebuffer sees the connection drop), and light it again when that one is back
void Gpu::check_hotplug()
{
    if (!disp)
        return;
    if (uint32_t irq = disp->dp_irq) {
        disp->dp_irq = 0;
        for (screen &sc : screens)
            if (sc.shown && !sc.asleep && !sc.unplugged && (disp->out[sc.out].id & irq))
                disp_dp_service(disp, index(sc), &disp->out[sc.out]);
    }
    if (!disp->hotplug)
        return;
    bool event = disp->hotplug != ~0u;  // GSP-RM said so, rather than a poll asking
    disp->hotplug = 0;
    disp_read_outputs(disp);
    for (screen &sc : screens) {
        if (!sc.shown)
            continue;
        const int head = index(sc);
        const struct disp_output *o = &disp->out[sc.out];
        if (!o->connected && !sc.unplugged) {
            if (!sc.asleep)
                disp_blank(disp, head, sc.out);
            sc.unplugged = true;
            sc.nub->setProperty(SURFACE_ONLINE, kOSBooleanFalse);
            sc.changed = true;
            log_msg("Malachite: hotplug: screen %d's monitor gone, head blanked\n", head);
        } else if (o->connected && sc.unplugged) {
            sc.unplugged = false;
            // the same mode on the same surface if the monitor still has it, else its preferred one
            const struct disp_mode *was = &o->modes[sc.mode - 1];
            if (sc.mode > o->nmodes || mode_w(was) * 4 > sc.pitch || (uint64_t)mode_h(was) * sc.pitch > sc.size) {
                uint32_t old = sc.mode;
                sc.mode = 0;  // set_mode(1) does nothing for the mode already shown
                if (set_mode(sc, 1) != kIOReturnSuccess)
                    sc.mode = old;
            } else if (int r = sc.asleep ? 0 : (reset_flips(sc), modeset(sc, was))) {
                log_msg("Malachite: hotplug: screen %d's modeset failed\n", head);
                if (r != DISP_NO_OUTPUT)
                    disp_reset(disp);
            }
            publish_surface(sc, true);
            log_msg("Malachite: hotplug: screen %d's monitor back, head lit\n", head);
        }
    }
    // a monitor plugged into an output nothing shows
    if (want_shown && (event || !show_failed || disp->connected != failed_connected) && show() == kIOReturnSuccess)
        log_msg("Malachite: hotplug: heads lit\n");
}

IOReturn Gpu::command(uint32_t selector, const uint64_t *args, task_t task)
{
    IOReturn ret = kIOReturnSuccess;
    IOLockLock(lock);
    if (!awake()) {
        IOLockUnlock(lock);
        return kIOReturnNotReady;
    }
    switch (selector) {
    case MALACHITE_LOAD_FIRMWARE: {
        uint64_t index = args[0], addr = args[1], len = args[2];
        if (index >= MALACHITE_FW_COUNT || !len) {
            ret = kIOReturnBadArgument;
            break;
        }
        IOMemoryDescriptor *src = IOMemoryDescriptor::withAddressRange(addr, len, kIODirectionOut, task);
        IOBufferMemoryDescriptor *dst = IOBufferMemoryDescriptor::withCapacity(len, kIODirectionNone);
        if (!src || !dst || src->prepare() != kIOReturnSuccess) {
            OSSafeReleaseNULL(src);
            OSSafeReleaseNULL(dst);
            ret = kIOReturnNoMemory;
            break;
        }
        src->readBytes(0, dst->getBytesNoCopy(), len);
        dst->setLength(len);
        src->complete();
        src->release();
        OSSafeReleaseNULL(plat.fw[index]);
        plat.fw[index] = dst;
        break;
    }
    case MALACHITE_GSP_START:
        if (!gsp)
            gsp = gsp_start(&gpu);
        if (gsp && !fe) {
            setProperty("MalachiteGPUName", gsp_get_info(gsp)->name);
            // the chipset model System Information shows, and the one property a sandboxed process may read; data
            // with its NUL as firmware gives it on real Macs, as Valve's launcher reads its bytes and throws on a string
            const char *name = gsp_get_info(gsp)->name;
            if (OSData *model = OSData::withBytes(name, (unsigned)strlen(name) + 1)) {
                plat.pci->setProperty("model", model);
                model->release();
            }
            gsp_on_rc(gsp, rc_event, this);
            fe = rmfe_create(gsp);
            if (irq)
                gsp_irq_enable(gsp, 1);
            timer->setTimeoutMS(20);
        }
        ret = gsp ? kIOReturnSuccess : kIOReturnError;
        break;
    case MALACHITE_H264_DECODE:
        ret = gsp && video_init(&kctx) ? h264_decode(args, task) : kIOReturnError;
        break;
    case MALACHITE_DISP_PROBE:
        if (!gsp) {
            ret = kIOReturnNotReady;
            break;
        }
        if (!disp) {
            disp = (struct disp *)IOMallocZero(sizeof(*disp));
            if (disp_probe(gsp, disp, scaled, nscaled)) {
                disp_fini(disp);
                IOFree(disp, sizeof(*disp));
                disp = nullptr;
                ret = kIOReturnError;
            }
        } else {  // again: look at what is connected now, as a hotplug would
            disp->hotplug = ~0u;
            check_hotplug();
        }
        break;
    case MALACHITE_DISP_SHOW:
        want_shown = true;
        ret = show();
        break;
    case MALACHITE_GSP_STOP:
        stop_gsp();
        break;
    case MALACHITE_DISP_READ:
        ret = read_shown(args[0], args[1], task);
        break;
    case MALACHITE_PM_CYCLE:
        if (!gsp) {
            ret = kIOReturnNotReady;
            break;
        }
        suspend();
        ret = resume();
        break;
    default:
        ret = kIOReturnUnsupported;
    }
    IOLockUnlock(lock);
    tell_nubs();
    return ret;
}

IOMemoryDescriptor *Gpu::map_descriptor(const struct rmfe_map *m, IOOptionBits *options)
{
    uint64_t size = (m->size + 0xfff) & ~0xfffull;
    switch (m->kind) {
    case RMFE_MAP_BAR0:
        if (m->offset + size > bar0->getLength())
            return nullptr;
        *options = kIOMapInhibitCache;
        return IOMemoryDescriptor::withPhysicalAddress(bar_phys(&gpu, 0) + m->offset, size, kIODirectionInOut);
    case RMFE_MAP_BAR1:
        if (m->offset + size > bar_size(&gpu, 1))
            return nullptr;
        *options = kIOMapWriteCombineCache;
        return IOMemoryDescriptor::withPhysicalAddress(bar_phys(&gpu, 1) + m->offset, size, kIODirectionInOut);
    case RMFE_MAP_SYS:
        *options = 0;
        return IOSubMemoryDescriptor::withSubRange((IOMemoryDescriptor *)m->sys->handle, m->offset, size,
                                                   kIODirectionInOut);
    default:
        return nullptr;
    }
}

IOReturn Gpu::escape(UserClient *uc, uint32_t cmd, void *arg, uint32_t size)
{
    IOLockLock(lock);
    if (!awake() || !fe) {
        IOLockUnlock(lock);
        return kIOReturnNotReady;
    }
    if (!uc->file)
        uc->file = rmfe_open(fe, uc->task);
    IOReturn ret = rmfe_escape(uc->file, cmd, arg, size) ? kIOReturnBadArgument : kIOReturnSuccess;

    int fd;
    struct rmfe_map m;
    if (ret == kIOReturnSuccess && !rmfe_take_map(uc->file, &fd, &m)) {
        UserClient *target = nullptr;
        for (UserClient *c = clients; c; c = c->next)
            if (c->task == uc->task && c->fd == fd)
                target = c;
        IOOptionBits options = 0;
        IOMemoryDescriptor *md = target ? map_descriptor(&m, &options) : nullptr;
        if (md) {
            OSSafeReleaseNULL(target->pending);
            target->pending = md;
            target->pending_options = options;
        } else {
            log_msg("Malachite: no mapping for fd %d (kind %d)\n", fd, m.kind);
            ret = kIOReturnBadArgument;
        }
    }
    IOLockUnlock(lock);
    return ret;
}

void Gpu::add_client(UserClient *uc)
{
    IOLockLock(lock);
    uc->next = clients;
    clients = uc;
    IOLockUnlock(lock);
}

void Gpu::remove_client(UserClient *uc)
{
    IOLockLock(lock);
    for (UserClient **pp = &clients; *pp; pp = &(*pp)->next) {
        if (*pp == uc) {
            *pp = uc->next;
            break;
        }
    }
    if (uc->file)
        rmfe_close(uc->file);
    uc->file = nullptr;
    OSSafeReleaseNULL(uc->pending);
    IOLockUnlock(lock);
}

IOMemoryDescriptor *Gpu::take_pending(UserClient *uc, IOOptionBits *options)
{
    IOLockLock(lock);
    IOMemoryDescriptor *md = uc->pending;
    *options = uc->pending_options;
    uc->pending = nullptr;
    IOLockUnlock(lock);
    return md;
}

// Only for root: the client's escapes are raw GSP-RM calls (any memory, any engine, the display) and its commands
// load firmware, start and stop the GPU and read the screen. Metal and OpenGL never come here.
bool UserClient::initWithTask(task_t owningTask, void *securityID, UInt32 type)
{
    if (IOUserClient::clientHasPrivilege(securityID, kIOClientPrivilegeAdministrator) != kIOReturnSuccess)
        return false;
    task = owningTask;
    return IOUserClient::initWithTask(owningTask, securityID, type);
}

bool UserClient::start(IOService *provider)
{
    owner = OSDynamicCast(Gpu, provider);
    if (!owner || !IOUserClient::start(provider))
        return false;
    owner->add_client(this);
    return true;
}

IOReturn UserClient::clientClose()
{
    owner->remove_client(this);
    terminate();
    return kIOReturnSuccess;
}

IOReturn UserClient::externalMethod(uint32_t selector, IOExternalMethodArguments *args, IOExternalMethodDispatch *,
                                        OSObject *, void *)
{
    switch (selector) {
    case MALACHITE_SEL_ESCAPE: {
        if (args->scalarInputCount < 1)
            return kIOReturnBadArgument;
        uint32_t size = args->structureInputDescriptor ? (uint32_t)args->structureInputDescriptor->getLength()
                                                       : args->structureInputSize;
        if (!size || size > MAX_ESCAPE)
            return kIOReturnBadArgument;
        void *buf = IOMalloc(size);
        if (!buf)
            return kIOReturnNoMemory;
        if (args->structureInputDescriptor)
            args->structureInputDescriptor->readBytes(0, buf, size);
        else
            memcpy(buf, args->structureInput, size);

        IOReturn ret = owner->escape(this, (uint32_t)args->scalarInput[0], buf, size);
        if (ret == kIOReturnSuccess) {
            if (args->structureOutputDescriptor) {
                args->structureOutputDescriptor->writeBytes(0, buf, IOMin(size, args->structureOutputDescriptor->getLength()));
            } else {
                uint32_t n = IOMin(size, args->structureOutputSize);
                memcpy(args->structureOutput, buf, n);
                args->structureOutputSize = n;
            }
        }
        IOFree(buf, size);
        return ret;
    }
    case MALACHITE_SEL_BIND_FD:
        if (args->scalarInputCount < 1)
            return kIOReturnBadArgument;
        fd = (int64_t)args->scalarInput[0];
        return kIOReturnSuccess;
    case MALACHITE_SEL_WAIT_EVENT:
        if (args->scalarInputCount < 2 || args->scalarOutputCount < 1)
            return kIOReturnBadArgument;
        args->scalarOutput[0] = owner->wait_event((uint32_t)args->scalarInput[0], (uint32_t)args->scalarInput[1]);
        return kIOReturnSuccess;
    default:
        if (selector < MALACHITE_LOAD_FIRMWARE || selector >= MALACHITE_END)
            return kIOReturnUnsupported;
        if ((selector == MALACHITE_LOAD_FIRMWARE && args->scalarInputCount < 3) ||
            (selector == MALACHITE_DISP_READ && args->scalarInputCount < 2) ||
            (selector == MALACHITE_H264_DECODE && args->scalarInputCount < 5))
            return kIOReturnBadArgument;
        return owner->command(selector, args->scalarInput, task);
    }
}

IOReturn UserClient::clientMemoryForType(UInt32 type, IOOptionBits *options, IOMemoryDescriptor **memory)
{
    IOMemoryDescriptor *md = type == 0 ? owner->take_pending(this, options) : nullptr;
    if (!md)
        return kIOReturnBadArgument;
    *memory = md;  // the caller takes over the reference
    return kIOReturnSuccess;
}

static volatile SInt32 video_clients;
static const SInt32 MAX_VIDEO_CLIENTS = 8;

IOReturn Gpu::newUserClient(task_t task, void *security, UInt32 type, OSDictionary *props, IOUserClient **handler)
{
    if (type == MALACHITE_DISPLAY_CLIENT) {
        if (IOUserClient::clientHasPrivilege(security, kIOClientPrivilegeLocalUser) != kIOReturnSuccess)
            return kIOReturnNotPrivileged;
        DisplayUserClient *uc = OSTypeAlloc(DisplayUserClient);
        if (!uc)
            return kIOReturnNoMemory;
        if (!uc->initWithTask(task, security, type, props) || !uc->attach(this)) {
            uc->release();
            return kIOReturnError;
        }
        if (!uc->start(this)) {
            uc->detach(this);
            uc->release();
            return kIOReturnError;
        }
        *handler = uc;
        return kIOReturnSuccess;
    }
    if (type != MALACHITE_VIDEO_CLIENT)
        return IOService::newUserClient(task, security, type, props, handler);
    // any process may open one, and each costs a GPU context: few enough at a time that they can't use up
    // the contexts WindowServer and apps need
    if (OSIncrementAtomic(&video_clients) >= MAX_VIDEO_CLIENTS) {
        OSDecrementAtomic(&video_clients);
        return kIOReturnNoResources;
    }
    VideoUserClient *uc = OSTypeAlloc(VideoUserClient);
    if (!uc) {
        OSDecrementAtomic(&video_clients);
        return kIOReturnNoMemory;
    }
    uc->counted = true;
    if (!uc->initWithTask(task, security, type, props) || !uc->attach(this)) {
        uc->release();
        return kIOReturnError;
    }
    if (!uc->start(this)) {
        uc->detach(this);
        uc->release();
        return kIOReturnError;
    }
    *handler = uc;
    return kIOReturnSuccess;
}

// A video client's context, made the first time, and its decode buffers, made again for each stream size
IOReturn Gpu::video_open(uintptr_t *ctx, struct video_h264 *h, const uint64_t *args)
{
    IOLockLock(lock);
    if (!*ctx)
        *ctx = ctx_open();
    gpu_ctx *c = *ctx ? ctx_of(*ctx) : nullptr;
    IOReturn ret = kIOReturnNotReady;
    if (!c || !video_init(c)) {
        char name[32];
        proc_selfname(name, sizeof(name));
        log_msg("video: no %s for %s (pid %d)\n", c ? "NVDEC channel" : "GPU context", name, proc_selfpid());
    } else if (!args[0]) {
        ret = kIOReturnSuccess;  // the context and channel alone, for the decoder to know it can decode
    } else {
        if (h->width_mbs)
            video_h264_close(&c->video, h);
        ret = video_h264_open(&c->video, h, (uint32_t)args[0], (uint32_t)args[1], (uint32_t)args[2], args[3])
                  ? kIOReturnBadArgument
                  : kIOReturnSuccess;
        log_msg("video: %s (pid %d) decodes H.264 of %llux%llu macroblocks into %llu surfaces: %#x\n", c->owner,
                proc_selfpid(), args[0], args[1], args[2], ret);
    }
    IOLockUnlock(lock);
    return ret;
}

IOReturn Gpu::video_decode(uintptr_t ctx, struct video_h264 *h, const uint64_t *args)
{
    uint32_t index = (uint32_t)args[1], payload;
    IOLockLock(lock);
    gpu_ctx *c = ctx ? ctx_of(ctx) : nullptr;  // 0 would be the kernel's
    bool queued = c && c->video_ready && h->width_mbs &&
                  !video_h264_queue(&c->video, h, (uint32_t)args[0], index, (uint32_t)args[2], &payload);
    IOLockUnlock(lock);
    if (!queued)
        return kIOReturnBadArgument;
    // on the client's own semaphore, the context staying until the client goes, so others have the GPU meanwhile
    return video_h264_wait(&c->video, h, payload, index) ? kIOReturnTimeout : kIOReturnSuccess;
}

void Gpu::video_close(uintptr_t ctx, struct video_h264 *h, uint32_t decoded)
{
    IOLockLock(lock);
    gpu_ctx *c = ctx ? ctx_of(ctx) : nullptr;  // 0 would be the kernel's
    if (c)
        log_msg("video: %s decoded %u pictures\n", c->owner, decoded);
    if (c && h->width_mbs)
        video_h264_close(&c->video, h);
    ctx_close(ctx);
    IOLockUnlock(lock);
}

bool DisplayUserClient::start(IOService *provider)
{
    owner = OSDynamicCast(Gpu, provider);
    return owner && IOUserClient::start(provider);
}

IOReturn DisplayUserClient::clientClose()
{
    terminate();
    return kIOReturnSuccess;
}

IOReturn DisplayUserClient::externalMethod(uint32_t selector, IOExternalMethodArguments *args, IOExternalMethodDispatch *,
                                           OSObject *, void *)
{
    if (selector != MALACHITE_DISPLAY_SET_SCALED)
        return kIOReturnUnsupported;
    return owner->set_scaled(args->scalarInput, args->scalarInputCount);
}

bool VideoUserClient::start(IOService *provider)
{
    owner = OSDynamicCast(Gpu, provider);
    busy = IOLockAlloc();
    return owner && busy && IOUserClient::start(provider);
}

void VideoUserClient::free()
{
    if (counted)
        OSDecrementAtomic(&video_clients);
    if (busy)
        IOLockFree(busy);
    IOUserClient::free();
}

IOReturn VideoUserClient::clientClose()
{
    IOLockLock(busy);
    owner->video_close(ctx, &dec, decoded);
    ctx = 0;
    IOLockUnlock(busy);
    terminate();
    return kIOReturnSuccess;
}

IOReturn VideoUserClient::externalMethod(uint32_t selector, IOExternalMethodArguments *args,
                                           IOExternalMethodDispatch *, OSObject *, void *)
{
    if (args->scalarInputCount < (selector == NVVIDEO_OPEN ? 4u : 3u))
        return kIOReturnBadArgument;
    IOReturn ret;
    IOLockLock(busy);
    switch (selector) {
    case NVVIDEO_OPEN:
        ret = owner->video_open(&ctx, &dec, args->scalarInput);
        break;
    case NVVIDEO_DECODE:
        ret = owner->video_decode(ctx, &dec, args->scalarInput);
        decoded += ret == kIOReturnSuccess;
        break;
    default:
        ret = kIOReturnUnsupported;
    }
    IOLockUnlock(busy);
    return ret;
}

IOReturn VideoUserClient::clientMemoryForType(UInt32 type, IOOptionBits *options, IOMemoryDescriptor **memory)
{
    IOLockLock(busy);
    struct vbuf *b = type == NVVIDEO_MAP_SETUP  ? &dec.setup
                     : type == NVVIDEO_MAP_BITS ? &dec.bits
                     : type - NVVIDEO_MAP_SURFACE < dec.slots ? &dec.surf[type - NVVIDEO_MAP_SURFACE]
                                                              : nullptr;
    IOMemoryDescriptor *md = b && b->mem ? (IOMemoryDescriptor *)b->mem->handle : nullptr;
    if (md) {
        md->retain();  // the caller takes it over
        *memory = md;
        *options = type >= NVVIDEO_MAP_SURFACE ? kIOMapReadOnly : 0;
    }
    IOLockUnlock(busy);
    return md ? kIOReturnSuccess : kIOReturnBadArgument;
}

}  // namespace malachite::rm
