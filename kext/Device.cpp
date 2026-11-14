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
#include "../accel/commands.h"
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

// Draws go out unwaited, so each gets a slot: vertex bank, pixel bank, header and sampler pools, a page each, then
// the tessellation init bank.
static const int DRAW_SLOTS = 256;
static const uint32_t PROF_RING = 2048;
static const uint32_t DRAW_SLOT_BYTES = 0x4000;
// const bank 0 and the texture pools in a page, the QMD in the next
static const int DISPATCH_SLOTS = 16;
static const uint32_t DISPATCH_SLOT_BYTES = 0x2000;
// the kernel context has chids 1 (copy) and 4/5 (graphics)
static const uint32_t FIRST_CTX_CHID = 6;

// kern.malachite_stats, counted since load
enum {
    ST_PRESENT, ST_FLIP, ST_DRAW, ST_DISPATCH, ST_COPY2D, ST_COPY, ST_FILL, ST_SYNC, ST_MAP, ST_UNMAP, ST_CURSOR,
    ST_DRAIN_WAITS, ST_DRAIN_US, ST_COPY_US, ST_DRAW_US, ST_COPY2D_US, ST_SYNC_US, ST_MAP_US, ST_UNMAP_US, ST_DRAW_SYS, ST_DRAW_SYS_KPX, ST_DRAW_VRAM, ST_DRAW_VRAM_KPX, ST_LOCKS, ST_LOCK_WAIT_US, ST_LOCK_WAIT_MAX_US,
    ST_LOCK_HOLD_US, ST_LOCK_MAX_US, ST_SLOT_WAIT_US, ST_PUSH_WAIT_US, ST_PRESENT_WAIT_US, ST_PRESENT_TOTAL_US, ST_PRESENT_POST_US, ST_PRESENT_SUBMIT_US,
    ST_PRESENT_COPY_US, ST_FLIP_PEND_US, ST_CTXS, ST_CTXS_PEAK, ST_CTXS_REFUSED, ST_PRESENT_HEAD,
    ST_COUNT = ST_PRESENT_HEAD + BAR1_FB_SLOTS
};
static const char *const stat_names[ST_COUNT] = {
    "present", "flip", "draw", "dispatch", "copy2d", "copy", "fill", "sync", "map", "unmap", "cursor",
    "drain_waits", "drain_us", "copy_us", "draw_us", "copy2d_us", "sync_us", "map_us", "unmap_us", "draw_sys", "draw_sys_kpx", "draw_vram", "draw_vram_kpx", "locks", "lock_wait_us", "lock_wait_max_us",
    "lock_hold_us", "lock_max_us", "slot_wait_us", "push_wait_us", "present_wait_us",
    "present_total_us", "present_post_us", "present_submit_us", "present_copy_us", "flip_pend_us",
    "ctxs", "ctxs_peak", "ctxs_refused",
    "present_h0", "present_h1", "present_h2", "present_h3",
};
static uint64_t stats[ST_COUNT];

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
    // maxima reset on read, so a benchmark reading before and after gets its own
    stats[ST_LOCK_WAIT_MAX_US] = stats[ST_LOCK_MAX_US] = 0;
    return SYSCTL_OUT(req, out, n);
}

SYSCTL_PROC(_kern, OID_AUTO, malachite_stats, CTLTYPE_OPAQUE | CTLFLAG_RD | CTLFLAG_LOCKED, nullptr, 0, stats_sysctl, "S",
            "what the NVIDIA driver's GPU calls cost");

// The primitive engine's fetch state as last set, for the fault log. An off stream keeps its address, one never set
// reads from 0, and streams 16 to 31 stay off.
struct pe_state {
    uint32_t attributes[RENDER_ATTRIBUTES];
    uint32_t format[RENDER_STREAMS], size[RENDER_STREAMS], divisor[RENDER_STREAMS];
    uint64_t addr[RENDER_STREAMS];
    uint32_t instanced;  // a bit a stream
    uint64_t index;
    uint32_t index_size, index_format;
    uint32_t restart, restart_index;
    uint32_t first, count, instances, base_instance;
    int32_t base_vertex;
};
static const uint32_t PE_UNKNOWN = ~0u;
static const uint32_t PE_STREAM_ON = 1u << 12;

static void pe_unknown(pe_state *s)
{
    for (uint32_t i = 0; i < RENDER_ATTRIBUTES; i++)
        s->attributes[i] = ATTR_INACTIVE;
    for (uint32_t j = 0; j < RENDER_STREAMS; j++) {
        s->format[j] = PE_UNKNOWN;
        s->size[j] = 0;
        s->divisor[j] = 0;
        s->addr[j] = 0;
    }
    s->instanced = 0;
    s->index = ~0ull;
    s->index_size = s->index_format = 0;
    s->restart = s->restart_index = 0;
    s->first = s->count = s->instances = s->base_instance = 0;
    s->base_vertex = 0;
}

// as build_draw (core/render.c) sets it
static void pe_draw(pe_state *s, const struct render_draw *r)
{
    pe_unknown(s);
    for (uint32_t i = 0; i < RENDER_ATTRIBUTES; i++)
        s->attributes[i] = r->attributes[i] ? r->attributes[i] : ATTR_INACTIVE | ATTR_R32G32B32A32_FLOAT;
    for (uint32_t j = 0; j < RENDER_STREAMS; j++) {
        s->format[j] = r->streams[j].addr ? r->streams[j].stride | PE_STREAM_ON : 0;
        s->addr[j] = r->streams[j].addr;
        s->size[j] = r->streams[j].size;
        s->divisor[j] = r->streams[j].divisor;
        if (r->streams[j].addr && r->streams[j].divisor)
            s->instanced |= 1u << j;
    }
    s->index = r->index;
    s->index_size = r->index_size;
    s->index_format = r->index_format;
    s->restart = draw_restarts(r->primitive, r->restart, r->index != 0);
    s->restart_index = draw_restart_index(r->restart, r->restart_index, r->index_format);
    s->first = r->first;
    s->count = r->count;
    s->instances = r->instances;
    s->base_instance = r->base_instance;
    s->base_vertex = r->base_vertex;
}

// 1 for a non-indexed draw, 2 for an indexed one, else 0
static int pe_method(pe_state *s, uint32_t m, uint32_t v)
{
    if (m >= R_SET_VERTEX_ATTRIBUTE_A(0) && m < R_SET_VERTEX_ATTRIBUTE_A(RENDER_ATTRIBUTES)) {
        s->attributes[(m - R_SET_VERTEX_ATTRIBUTE_A(0)) / 4] = v;
    } else if (m >= R_SET_VERTEX_STREAM_A_FORMAT(0) && m < R_SET_VERTEX_STREAM_A_FORMAT(RENDER_STREAMS)) {
        uint32_t j = (m - R_SET_VERTEX_STREAM_A_FORMAT(0)) / 16, k = m % 16 / 4;
        if (k == 0)
            s->format[j] = v;
        else if (k == 1)
            s->addr[j] = (uint64_t)v << 32 | (uint32_t)s->addr[j];
        else if (k == 2)
            s->addr[j] = (s->addr[j] & ~0xffffffffull) | v;
        else
            s->divisor[j] = v;
    } else if (m >= R_SET_VERTEX_STREAM_SIZE_A(0) && m < R_SET_VERTEX_STREAM_SIZE_A(RENDER_STREAMS)) {
        if (m % 8)
            s->size[(m - R_SET_VERTEX_STREAM_SIZE_A(0)) / 8] = v;
    } else if (m >= R_SET_VERTEX_STREAM_INSTANCE_A(0) && m < R_SET_VERTEX_STREAM_INSTANCE_A(RENDER_STREAMS)) {
        uint32_t bit = 1u << (m - R_SET_VERTEX_STREAM_INSTANCE_A(0)) / 4;
        s->instanced = v ? s->instanced | bit : s->instanced & ~bit;
    } else if (m == R_SET_INDEX_BUFFER_A) {
        s->index = (uint64_t)v << 32 | (uint32_t)s->index;
    } else if (m == R_SET_INDEX_BUFFER_A + 4) {
        s->index = (s->index & ~0xffffffffull) | v;
    } else if (m == R_SET_INDEX_BUFFER_SIZE_A + 4) {
        s->index_size = v;
    } else if (m == R_SET_INDEX_BUFFER_E) {
        s->index_format = v;
    } else if (m == R_SET_DA_PRIMITIVE_RESTART) {
        s->restart = v;
    } else if (m == R_SET_DA_PRIMITIVE_RESTART + 4) {
        s->restart_index = v;
    } else if (m == R_SET_GLOBAL_BASE_VERTEX_INDEX) {
        s->base_vertex = (int32_t)v;
    } else if (m == R_SET_GLOBAL_BASE_VERTEX_INDEX + 4) {
        s->base_instance = v;
    } else if (m == R_SET_DRAW_CONTROL_A + 4) {
        s->instances = v;
    } else if (m == R_DRAW_VERTEX_ARRAY_BEGIN_END_A || m == R_DRAW_INDEX_BUFFER_BEGIN_END_A) {
        s->first = v;
    } else if (m == R_DRAW_VERTEX_ARRAY_BEGIN_END_A + 4 || m == R_DRAW_INDEX_BUFFER_BEGIN_END_A + 4) {
        s->count = v;
        return m == R_DRAW_INDEX_BUFFER_BEGIN_END_A + 4 ? 2 : 1;
    }
    return 0;
}

// why a draw from s would fetch from nowhere, if it would
static bool pe_wrong(const pe_state *s, bool indexed, char *why, size_t n)
{
    for (uint32_t i = 0; i < RENDER_ATTRIBUTES; i++) {
        uint32_t a = s->attributes[i], j = a & 0x1f;
        if (a & ATTR_INACTIVE)
            continue;
        if (j >= RENDER_STREAMS || !(s->format[j] & PE_STREAM_ON)) {
            snprintf(why, n, "attribute %u reads stream %u, which is off", i, j);
            return true;
        }
    }
    for (uint32_t j = 0; j < RENDER_STREAMS; j++)
        if (s->format[j] != PE_UNKNOWN && (s->format[j] & PE_STREAM_ON) && s->addr[j] < 0x1000) {
            snprintf(why, n, "stream %u is on at %#llx", j, (unsigned long long)s->addr[j]);
            return true;
        }
    if (indexed && s->index < 0x1000) {
        snprintf(why, n, "its index buffer is at %#llx", (unsigned long long)s->index);
        return true;
    }
    return false;
}

// an attribute starting past its stream's end for vertices up to top, which reads the vertex stream substitute
static bool pe_past_end(const pe_state *s, uint32_t top, char *why, size_t n)
{
    for (uint32_t i = 0; i < RENDER_ATTRIBUTES; i++) {
        uint32_t a = s->attributes[i], j = a & 0x1f;
        if ((a & ATTR_INACTIVE) || j >= RENDER_STREAMS || s->format[j] == PE_UNKNOWN || !(s->format[j] & PE_STREAM_ON))
            continue;
        bool per_instance = (s->instanced >> j & 1) && s->divisor[j];
        uint64_t v = per_instance ? s->base_instance + (s->instances ? s->instances - 1 : 0) / s->divisor[j]
                                  : (uint64_t)((int64_t)top + s->base_vertex);
        uint64_t at = v * (s->format[j] & 0xfff) + (a >> 7 & 0x3fff);
        if (at >= s->size[j]) {
            snprintf(why, n, "attribute %u reads vertex %llu of stream %u at +%#llx, past its %#x bytes", i,
                     (unsigned long long)v, j, (unsigned long long)at, s->size[j]);
            return true;
        }
    }
    return false;
}

// An RM client with its own address space, copy channel and graphics channel, each made on first use.
struct gpu_ctx {
    char owner[32];  // the opening process, for the log
    struct rm_client client;
    struct vmm vm;
    struct chan copy_ch;
    struct dma *copy_sem;
    uint64_t copy_semva;
    uint32_t copy_seq;  // monotonic, so the semaphore word changes every submit
    bool gpu_ready;     // client, address space and copy channel up
    struct video video;
    bool video_ready;
    struct chan compute_ch;
    struct gr_ctx compute_ctx;
    struct dma *compute_qmd;   // semaphore in the page's tail
    struct dma *compute_cbuf;  // const bank 0 (kernel params at 0x160+), then the texture pools
    uint64_t compute_qmdva, compute_cbufva, compute_semva;
    uint64_t zero_va;  // Gpu::zero_page, read only, for vertex fetches past a stream's end
    struct dma *draw_ring;
    uint64_t draw_ringva;
    // kern.malachite_prof_on: per-draw GPU timestamps, the gap between two charged to the later draw
    struct dma *prof;
    uint64_t prof_va, prof_last_ts;
    uint32_t prof_n, prof_seen;
    struct prof_draw {
        uint64_t vs, ps;
        uint32_t verts, flags, area;
        uint16_t width, height;
    } *prof_draws;
    uint32_t draw_slot_seq[DRAW_SLOTS];  // 0 for none
    unsigned draw_next;
    uint32_t draw_pending;  // the last unwaited payload, 0 for none
    uint32_t draws_queued;  // not kicked yet
    // VMM_CACHED app memory: command buffers end with l2_sysmem so uncached mappings and the CPU see fresh data
    bool cached_sys;
    uint64_t methods_va;  // where push_run last found a run
    gsp_fault fault;      // last MMU fault, for recover()'s log; type NULL for none
    uint32_t fault_chid;
    uint64_t kind_va;       // the depth target last given its kind
    uint32_t kind_set;
    bool kind_compressed;
    uint64_t kind_lo, kind_hi;
    uint64_t zcull_zt;      // the depth target zcull region 0 follows, 0 for none
    uint64_t zcull_lo, zcull_hi;  // hi 0 once a copy wrote there, which zcull can't know
    // Copies are chained on the copy channel's semaphore in draw order. ce_pending is the last the graphics channel
    // hasn't been told to wait for.
    uint32_t ce_pending, ce_last, ce_half_seq[2];
    int ce_half;
    // L2 is already clean if nothing was queued since the last l2_sysmem; l2_seq is 0 after a new mapping
    uint32_t l2_seq, l2_copy_seq, l2_begun;
    // what a fault lost runs again on the new channel, the first launch only once as it may be the culprit
    struct {
        uint32_t payload, gp_end;
    } issued[128];
    unsigned issued_next;
    uint32_t replayed;
    struct dma *dispatch_ring;
    uint64_t dispatch_ringva;
    uint64_t slm_vram, slm_va, slm_size, slm_per_tpc;
    uint32_t slm_lane, slm_per_warp;
    uint32_t dispatch_slot_seq[DISPATCH_SLOTS];
    unsigned dispatch_next;
    uint32_t half_seq[2];  // last payload in each graphics pushbuffer half
    int push_half;
    uint64_t pass_targets[9];  // what the last draw wrote, 0 once anything else ran
    uint64_t pass_vs, pass_ps, pass_ti, pass_ts;
    volatile uint32_t *compute_sem;
    uint32_t compute_seq;
    // for the fault log; more than a kick holds, so the faulting launch is among them
    static const unsigned LAUNCH_LOG = 64;
    struct launch_log {
        uint32_t seq, groups[3], block[3], regs;
        uint64_t prog, vs;  // a draw's ps is in prog
        uint64_t params[4];  // the first two argument words of the vs, then the ps
        uint64_t target, zt;
        const uint8_t *cbuf;  // a draw's ps bank is 0x1000 on
        pe_state pe;
        uint64_t run_va;  // run_words 0 for anything but a plugin run
        uint32_t run_words, run_draws;
    } launches[LAUNCH_LOG];
    unsigned launch_next;
    // A plugin run only holds changes on the previous draw's state, so it must follow a draw from the same queue;
    // nullptr on a new channel or after a failed draw.
    const void *state_of;
    bool runs_dropped;  // and logged
    uint64_t busy_since;  // 0 when idle
    uint64_t busy_unbilled;  // not yet charged to the task (NVGPU_FN_CTX_BUSY)
    bool compute_ready;
    // set by the RC event; rebuilt before the next use
    volatile bool compute_faulted, copy_faulted;
    // a task that keeps faulting the GPU has its work refused for a while
    uint32_t faults;
    uint64_t fault_window;
    uint64_t refused_until;
    uint32_t backoff_ms;  // doubles while the faults go on
};

// A draw's time runs from the previous one finishing to its own; longer waits count as gaps, except for dispatches
// (flags 0x80), which may well run that long.
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
    if (ns > PROF_GAP_NS && !(d.flags & 0x80)) {
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
    if (!req->oldptr)  // sysctl(8) asks for the size first, which mustn't reset the counters
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
static int present_apps = 1;
SYSCTL_INT(_kern, OID_AUTO, malachite_present_apps, CTLFLAG_RW | CTLFLAG_LOCKED, &present_apps, 0, "test");

// compressed without PLC, as GA102's PLC has an erratum on some physical addresses
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

    // kctx does presents and work from tasks without a context; Accelerator names the others by slot + 1 and serial
    gpu_ctx kctx = {};
    static const int MAX_CTXS = 255;
    gpu_ctx *ctxs[MAX_CTXS] = {};
    uint32_t ctx_serial[MAX_CTXS] = {};
    gpu_ctx *ctx_of(uintptr_t handle);
    uintptr_t ctx_open();
    void ctx_close(uintptr_t handle);
    uint32_t present_fence = 0;  // 0 before the first present
    bool copy_done(uint32_t payload) const
    {
        return (int32_t)(*(volatile uint32_t *)kctx.copy_sem->cpu - payload) >= 0;
    }
    struct gpu_mapping {
        gpu_ctx *ctx;
        uint64_t va;
        struct dma dma;    // dma.size is the mapping's size even for VRAM
        uint64_t vram;     // 0 for system memory
        uint8_t kind;
        bool compressed;  // big pages
        int16_t passes;  // negative while a promotion that found nothing rests
        uint8_t shadow;  // 1-based index in shadows, mapped in place of the pages, 0 for none
        bool uncached;   // the channel reads methods from it (push_run)
        int32_t hnext;   // next slot + 1 in its hash bucket
    };
    unsigned sys_flags(const gpu_mapping *m) const
    {
        return m->ctx->cached_sys && !m->uncached ? VMM_CACHED : 0;
    }
    // Metal apps read just outside their buffers, which Apple's GPUs allow, so read-only zero pages flank each mapping
    // to keep such reads from faulting the channel
    struct dma *zero_page = nullptr;
    bool map_slack(gpu_ctx *c, uint64_t va)
    {
        if (!zero_page && (zero_page = gsp_dma(gsp, 0x1000, 1)))
            memset(zero_page->cpu, 0, 0x1000);
        return zero_page && !vmm_map_sys(&c->vm, va, zero_page, 0, 0x1000, VMM_RO);
    }
    // a helper lane in CoreAnimation's text blit reads a byte before its buffer; 0 when the space is full
    uint64_t map_va(gpu_ctx *c, uint64_t size)
    {
        uint64_t base = vmm_alloc(&c->vm, size + 0x2000, 0x1000);
        if (!base)
            return 0;
        map_slack(c, base);
        map_slack(c, base + 0x1000 + size);
        return base + 0x1000;
    }
    // Image blit helper lanes read a row or two (8 KB) before their source and a compositor reduction reads past its
    // partial sums, so big pages get 64 KB of zeros either side.
    uint64_t map_va_big(gpu_ctx *c, uint64_t size)
    {
        uint64_t base = vmm_alloc(&c->vm, size + 0x20000, 0x10000);
        if (!base)
            return 0;
        for (uint64_t off = 0; off < 0x10000; off += 0x1000) {
            map_slack(c, base + off);
            map_slack(c, base + 0x10000 + size + off);
        }
        return base + 0x10000;
    }
    static uint64_t slack(uint64_t size, bool vram)
    {
        return vram && size >= 0x10000 ? 0x10000 : 0x1000;
    }
    void map_va_free(gpu_ctx *c, uint64_t va, uint64_t size, bool vram)
    {
        uint64_t s = slack(size, vram);
        vmm_unmap(&c->vm, va - s, size + 2 * s);
        vmm_free(&c->vm, va - s, size + 2 * s);
    }
    IOReturn set_kind(gpu_ctx *c, uint64_t va, uint8_t kind, bool compress);
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
    const gpu_mapping *mapping_of(gpu_ctx *c, uint64_t va)
    {
        for (int i = 0; i < gpu_maps_top; i++)
            if (gpu_maps[i].ctx == c && gpu_maps[i].va && gpu_maps[i].va <= va && va < gpu_maps[i].va + gpu_maps[i].dma.size)
                return &gpu_maps[i];
        return nullptr;
    }
    // gpu_bridge.h's tag, looked for in a program's first 256 KB only
    bool program_tag(gpu_ctx *c, uint64_t va, nvgpu_program_tag *t)
    {
        const gpu_mapping *m = va ? mapping_of(c, va) : nullptr;
        if (!m || !m->dma.handle || m->vram)
            return false;
        auto *md = (IOMemoryDescriptor *)m->dma.handle;
        uint64_t words[0x80];
        for (uint64_t at = va - m->va; at < va - m->va + 0x40000 && at + sizeof(words) <= m->dma.size;
             at += sizeof(words)) {
            if (md->readBytes(at, words, sizeof(words)) != sizeof(words))
                return false;
            for (unsigned k = 0; k < 0x80; k += 0x20) {
                if (words[k] != NVGPU_PROGRAM_TAG || md->readBytes(at + k * 8, t, sizeof(*t)) != sizeof(*t))
                    continue;
                t->name[sizeof(t->name) - 1] = 0;
                if (t->nbuffers > NVGPU_PROGRAM_TAG_BUFFERS)
                    t->nbuffers = NVGPU_PROGRAM_TAG_BUFFERS;
                return true;
            }
        }
        return false;
    }
    struct bound_buffer {
        const char *stage;
        unsigned index;
        uint64_t va;
        const gpu_mapping *m;
        uint64_t dist;  // to the fault, 0 when the mapping holds it
    };
    void log_buffers(gpu_ctx *c, const char *stage, const nvgpu_program_tag &t, const uint8_t *cbuf, uint64_t fva,
                     bound_buffer &near)
    {
        if (!t.nbuffers)
            return;
        char line[400] = "";
        size_t n = 0;
        for (unsigned i = 0; i < t.nbuffers && n < sizeof(line); i++) {
            unsigned index = t.buffers[i].index, off = t.buffers[i].cbuf_off;
            if (off < 0x160 || off + 8 > COMPUTE_CBUF_BYTES)
                continue;
            uint64_t va;
            memcpy(&va, cbuf + off, 8);
            const gpu_mapping *m = va ? mapping_of(c, va) : nullptr;
            if (!m) {
                n += snprintf(line + n, sizeof(line) - n, ", %u at %#llx unmapped", index, (unsigned long long)va);
                continue;
            }
            n += snprintf(line + n, sizeof(line) - n, ", %u at %#llx in %#llx+%#llx", index, (unsigned long long)va,
                          (unsigned long long)m->va, (unsigned long long)m->dma.size);
            uint64_t end = m->va + m->dma.size;
            uint64_t dist = fva < m->va ? m->va - fva : fva >= end ? fva - end + 1 : 0;
            if (!near.m || dist < near.dist)
                near = { stage, index, va, m, dist };
        }
        log_msg("Malachite:     %s %s's buffers%s\n", stage, t.name, *line ? line + 1 : " not read");
    }
    void launch_names(gpu_ctx *c, const gpu_ctx::launch_log &l, char *out, size_t n)
    {
        char ps[64] = "unnamed", vs[64] = "unnamed";
        nvgpu_program_tag tag;
        bool named = program_tag(c, l.prog, &tag);
        if (named)
            strlcpy(ps, tag.name, sizeof(ps));
        if (l.vs && program_tag(c, l.vs, &tag)) {
            strlcpy(vs, tag.name, sizeof(vs));
            named = true;
        }
        *out = 0;
        if (l.vs && named)
            snprintf(out, n, " (ps %s, vs %s after)", ps, vs);
        else if (l.vs)
            strlcpy(out, " (vs after)", n);
        else if (named)
            snprintf(out, n, " (kernel %s)", ps);
    }
    void log_bound(gpu_ctx *c, const gpu_ctx::launch_log &l, bool have_fault, uint64_t fault_va)
    {
        if (!l.cbuf)
            return;
        bound_buffer near = {};
        nvgpu_program_tag tag;
        if (l.vs && program_tag(c, l.vs, &tag))
            log_buffers(c, "vs", tag, l.cbuf, fault_va, near);
        if (program_tag(c, l.prog, &tag))
            log_buffers(c, l.vs ? "ps" : "kernel", tag, l.cbuf + (l.vs ? 0x1000 : 0), fault_va, near);
        if (!have_fault || !near.m)
            return;
        uint64_t end = near.m->va + near.m->dma.size;
        char where[64];
        if (fault_va < near.m->va)
            snprintf(where, sizeof(where), "%#llx before", (unsigned long long)(near.m->va - fault_va));
        else if (fault_va >= end)
            snprintf(where, sizeof(where), "%#llx past the end of", (unsigned long long)(fault_va - end));
        else
            snprintf(where, sizeof(where), "%#llx into", (unsigned long long)(fault_va - near.m->va));
        log_msg("Malachite:     the fault at %#llx is %s %s buffer %u at %#llx (%#llx+%#llx)\n",
                (unsigned long long)fault_va, where, near.stage, near.index, (unsigned long long)near.va,
                (unsigned long long)near.m->va, (unsigned long long)near.m->dma.size);
    }
    // false unless one system memory mapping holds them all
    int read_slot = 0;  // tried first
    bool read_mapped(gpu_ctx *c, uint64_t va, void *out, uint32_t len)
    {
        for (int k = -1; k < gpu_maps_top; k++) {
            int j = k < 0 ? read_slot : k;
            if (j >= gpu_maps_top)
                continue;
            const gpu_mapping &m = gpu_maps[j];
            if (m.ctx != c || !m.va || va < m.va || va + len > m.va + m.dma.size || !m.dma.handle || m.vram ||
                m.shadow)
                continue;
            read_slot = j;
            return ((IOMemoryDescriptor *)m.dma.handle)->readBytes(va - m.va, out, len) == len;
        }
        return false;
    }
    // highest index read or the last vertex; -1 when the indices aren't in system memory, -2 past the index buffer
    int64_t pe_top(gpu_ctx *c, const pe_state *s, bool indexed)
    {
        if (!indexed)
            return (int64_t)s->first + s->count - 1;
        uint32_t bytes = 1u << s->index_format, top = 0;
        if (s->index_format > RENDER_INDEX_U32 || ((uint64_t)s->first + s->count) * bytes > s->index_size)
            return -2;
        uint8_t w[256];
        for (uint32_t i = 0; i < s->count && i < 0x10000; i += sizeof(w) / bytes) {
            uint32_t k = s->count - i < sizeof(w) / bytes ? s->count - i : sizeof(w) / bytes;
            if (!read_mapped(c, s->index + ((uint64_t)s->first + i) * bytes, w, k * bytes))
                return -1;
            for (uint32_t e = 0; e < k; e++) {
                uint32_t v = bytes == 1 ? w[e] : bytes == 2 ? ((uint16_t *)w)[e] : ((uint32_t *)w)[e];
                if (!(s->restart && v == s->restart_index) && v > top)
                    top = v;
            }
        }
        return top;
    }
    // replays a plugin run's methods onto s to find its first draw that fetches from nowhere
    void decode_run(gpu_ctx *c, uint64_t va, uint32_t words, pe_state *s, char *out, size_t n)
    {
        *out = 0;
        const gpu_mapping *m = nullptr;
        for (int j = 0; j < gpu_maps_top && !m; j++)
            if (gpu_maps[j].ctx == c && gpu_maps[j].va && gpu_maps[j].va <= va &&
                va + words * 4ull <= gpu_maps[j].va + gpu_maps[j].dma.size && gpu_maps[j].dma.handle)
                m = &gpu_maps[j];
        if (!m) {
            pe_unknown(s);
            snprintf(out, n, "its methods aren't in a mapping of system memory");
            return;
        }
        uint32_t w[64], left = 0, op = 0, mthd = 0, draws = 0;
        for (uint32_t i = 0; i < words; i++) {
            if (i % 64 == 0) {
                uint32_t k = words - i < 64 ? words - i : 64;
                if (((IOMemoryDescriptor *)m->dma.handle)->readBytes(va - m->va + i * 4ull, w, k * 4) != k * 4) {
                    pe_unknown(s);
                    snprintf(out, n, "word %u can't be read", i);
                    return;
                }
            }
            uint32_t v = w[i % 64];
            int drawn = 0;
            if (!left) {
                // INC_METHOD, NON_INC_METHOD, IMMD_DATA_METHOD and ONE_INC
                op = v >> 29;
                mthd = (v & 0xfff) << 2;
                if (op == 4)
                    drawn = pe_method(s, mthd, v >> 16 & 0x1fff);
                else if (op == 1 || op == 3 || op == 5)
                    left = v >> 16 & 0x1fff;
                else {
                    pe_unknown(s);
                    snprintf(out, n, "word %u is no method header: %#x", i, v);
                    return;
                }
            } else {
                drawn = pe_method(s, mthd, v);
                left--;
                if (op == 1 || op == 5)
                    mthd += 4;
                if (op == 5)
                    op = 3;
            }
            if (!drawn)
                continue;
            draws++;
            if (*out)
                continue;
            char why[128];
            int64_t top = pe_top(c, s, drawn == 2);
            if (pe_wrong(s, drawn == 2, why, sizeof(why)) ||
                (top >= 0 && pe_past_end(s, (uint32_t)top, why, sizeof(why))))
                snprintf(out, n, "draw %u of it: %s", draws, why);
            else if (top == -2)
                snprintf(out, n, "draw %u of it: indices %u to %u, past its index buffer's %#x bytes", draws, s->first,
                         s->first + s->count - 1, s->index_size);
        }
    }
    // Every context's mappings, hashed by VA, as a Metal process can hold tens of thousands of buffers and an unmap
    // per buffer would otherwise scan the whole table.
    static const int MAX_GPU_MAPS = 1 << 18;
    struct gpu_mapping *gpu_maps = nullptr;
    int gpu_maps_cap = 0;
    int gpu_maps_free = 0;  // no slot below it is free
    int gpu_maps_top = 0;   // none at or above it is used
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
    // -1 at the cap
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

    // Lock-free because NVGPU_FN_DRAW_DONE is. busy_total counts overlapping contexts once.
    uint64_t busy_total = 0;
    // busy contexts in the low byte, the time that count last changed above it
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
    // so a long draw shows in every tick
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
    uint32_t cached_begins = 0;  // command buffers begun by contexts with VMM_CACHED memory
    IOReturn gpu_stats(struct nvgpu_stats *out);

    static const uint32_t TEX_HEADER_POOL = 0x400, TEX_SAMPLER_POOL = 0x800;  // offsets in kctx.compute_cbuf
    IOReturn drain(gpu_ctx *c, int ms = 2000);
    IOReturn sync_ctx(uintptr_t handle);
    int clear_vram(uint64_t vram, uint64_t size);
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
    static void rc_event(void *ctx, uint32_t engine, uint32_t chid, uint32_t exception, const gsp_fault *fault);
    void log_fault(gpu_ctx *c, const char *channel, uint32_t chid);
    void recover(gpu_ctx *c);
    int ring_wait(gpu_ctx *c, int counter, uint32_t payload, int ms);
    void resync_screens();
    // A timeout likely means a fault; the interrupt handler can't read GSP-RM's messages under the lock, so read them
    // here for the RC event to mark the channel.
    IOReturn settle(IOReturn r)
    {
        if (r == kIOReturnTimeout && gsp)
            gsp_poll(gsp);
        return r;
    }
    // Retried on a new channel after a fault. A fault is reported only after its launch released, so a launch that
    // faulted itself runs twice.
    IOReturn launch(gpu_ctx *c, const struct nvgpu_dispatch *d, const struct nvgpu_draw *w, const char **reject = nullptr,
                    const void *queue = nullptr)
    {
        // an earlier launch's fault arrives late; hear it before this one is lost on the dead channel
        if (gsp)
            gsp_poll(gsp);
        IOReturn r = settle(d ? gpu_dispatch(c, d) : gpu_draw(c, w, queue, reject));
        if (r == kIOReturnTimeout && c->compute_faulted)
            r = settle(d ? gpu_dispatch(c, d) : gpu_draw(c, w, queue, reject));
        return r;
    }
    void kick_draws(gpu_ctx *c)
    {
        if (c->draws_queued) {
            chan_kick(&c->compute_ch);
            c->draws_queued = 0;
        }
    }
    // On the copy channel, as only admin channels may run MEM_OP's L2 operations. Skipped at begin unless lines may
    // have come in since: own work, another app's command buffer (L2 is GPU-wide) or a new mapping.
    void l2_sysmem(gpu_ctx *c, bool begin)
    {
        if (!c->cached_sys)
            return;
        bool clean = c->l2_seq == c->compute_seq && c->l2_copy_seq == c->copy_seq;
        if (begin) {
            clean = clean && c->l2_seq && c->l2_begun == cached_begins;
            c->l2_begun = ++cached_begins;
        }
        if (clean)
            return;
        // the wait, the L2 operations and the release; failing to queue them leaves L2 as dirty as it was
        if (!compute_bridge_init(c) || c->copy_faulted || ce_room(c, 0x300) != kIOReturnSuccess) {
            c->l2_seq = 0;
            return;
        }
        kick_draws(c);
        // the copy channel runs in order, so only the graphics channel needs waiting for
        if (c->draw_pending && c->draw_pending != c->ce_last && (int32_t)(*c->compute_sem - c->draw_pending) < 0)
            chan_acquire_queue_switch(&c->copy_ch, c->compute_semva, c->draw_pending);
        chan_l2_sysmem(&c->copy_ch);
        c->ce_pending = c->ce_last = c->draw_pending = c->l2_seq = ++c->compute_seq;
        c->l2_copy_seq = c->copy_seq;
        chan_release_queue(&c->copy_ch, c->compute_semva, c->compute_seq);
    }
    // A kick (a bus read and a doorbell) is the slowest part of a draw, so up to 32 share one.
    IOReturn draw(gpu_ctx *c, const struct nvgpu_draw *w, const void *queue, const char **reject)
    {
        IOReturn r = launch(c, nullptr, w, reject, queue);
        if (r != kIOReturnSuccess || c->draws_queued >= 32)
            kick_draws(c);
        return r;
    }
    // The channel fetches methods ahead of the semaphore wait that orders them after the L2 invalidate, so a stale
    // cached line of a rewritten block would corrupt the stream.
    void uncache_methods(gpu_ctx *c, uint64_t va)
    {
        if (!c->cached_sys)
            return;
        int last = map_slot(c->methods_va);
        if (last >= 0 && gpu_maps[last].ctx == c && va >= gpu_maps[last].va &&
            va < gpu_maps[last].va + gpu_maps[last].dma.size)
            return;
        for (int i = 0; i < gpu_maps_top; i++) {
            gpu_mapping *m = &gpu_maps[i];
            if (m->ctx != c || !m->va || va < m->va || va >= m->va + m->dma.size)
                continue;
            if (!m->uncached && !m->vram && !m->shadow) {
                m->uncached = true;
                vmm_map_sys(&c->vm, m->va, &m->dma, 0, m->dma.size, VMM_KINDF(m->kind));
                vmm_flush(&gpu);
            }
            c->methods_va = m->va;
            return;
        }
    }
    // Runs hold only state changes and aren't replayed after a fault. Without the state they'd fetch from address 0
    // and fault again, so they're dropped until the queue's next full draw.
    IOReturn push_run(gpu_ctx *c, const struct malachite_push *p, const void *queue)
    {
        if (gsp)
            gsp_poll(gsp);
        if (!compute_bridge_init(c)) {
            c->state_of = nullptr;
            return kIOReturnNotReady;
        }
        uncache_methods(c, p->va);
        // the release's slot only; push_room(0x200) would count a wrap the release doesn't make
        if (after_copies(c) != kIOReturnSuccess || push_room(c, 0x100) != kIOReturnSuccess) {
            c->state_of = nullptr;
            return settle(kIOReturnTimeout);
        }
        if (!queue || c->state_of != queue) {
            if (!c->runs_dropped)
                log_msg("Malachite: %s's draws are dropped until its next full one: %s\n", c->owner,
                        c->state_of ? "another queue's draw came between" : "the channel lost the state they change");
            c->runs_dropped = true;
            return kIOReturnSuccess;
        }
        chan_entry(&c->compute_ch, p->va, p->words);
        render_mark(&c->compute_ch, c->compute_semva, ++c->compute_seq);
        // two GPFIFO entries for one pushbuffer slot, and the slot count is what keeps the GPFIFO from overrunning
        c->compute_ch.push_off += 0x100;
        if (c->compute_ch.push_off > PUSH_SIZE)
            c->compute_ch.push_off = PUSH_SIZE;
        c->draws_queued = 0;
        c->draw_pending = c->compute_seq;
        gpu_ctx::launch_log &l = c->launches[c->launch_next++ % gpu_ctx::LAUNCH_LOG];
        l = {};
        l.seq = c->compute_seq;
        l.run_va = p->va;
        l.run_words = p->words;
        l.run_draws = p->draws;
        issued(c);
        return kIOReturnSuccess;
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
    // Unmaps during in-flight draws wait for the semaphore to pass seq, so WindowServer doesn't stall mid-frame.
    struct deferred_unmap {
        gpu_ctx *ctx;
        uint64_t va;
        uint32_t seq;
    };
    static const int MAX_DEFERRED = 256;
    deferred_unmap deferred[MAX_DEFERRED];
    int ndeferred = 0;
    // held while any task's draws are in flight, as they may be reading it
    struct deferred_free {
        uint64_t vram, size;
    };
    deferred_free vram_deferred[MAX_DEFERRED];
    int nvram_deferred = 0;
    void retire_deferred();
    void drop_deferred(gpu_ctx *c);
    IOReturn gpu_copy(gpu_ctx *c, uint64_t src, uint64_t dst, uint64_t size);
    IOReturn gpu_copy2d(gpu_ctx *c, const struct nvgpu_copy2d *cp);
    IOReturn copy_after_draws(gpu_ctx *c, uint64_t src, uint64_t dst, uint64_t size);
    IOReturn gpu_fill(gpu_ctx *c, uint64_t dst, uint64_t size, uint32_t pattern);
    IOReturn gpu_dispatch(gpu_ctx *c, const struct nvgpu_dispatch *d);
    IOReturn gpu_draw(gpu_ctx *c, const struct nvgpu_draw *d, const void *queue, const char **reject);
    IOReturn gpu_present(struct nvgpu_present *p, uint64_t since);
    void adopt_surface(int i);
    IOReturn present_frame(struct nvgpu_present *p);

    // WindowServer cycles three or four surfaces per display; with fewer slots than that every present would remap,
    // and promote.
    static const int PRESENT_SLOTS = 4 * BAR1_FB_SLOTS;
    static const uint64_t PRESENT_SLOT_SIZE = 64ull << 20;
    struct present_slot {
        IOMemoryDescriptor *md;  // retained and prepared while mapped
        struct dma dma;
        uint64_t va, used;
        uint32_t fence;  // the last copy from it
        int count;       // frames shown, negative while a promotion that found nothing rests
        int shadow;      // 1-based, 0 for none
    } present[PRESENT_SLOTS] = {};
    present_slot *oldest_slot();
    IOReturn fill_slot(present_slot *s, IOMemoryDescriptor *md, size_t size);
    // A surface WindowServer keeps showing moves to VRAM so it isn't copied over PCIe each frame; the CPU's view of
    // the pages is stale until a demotion copies it back.
    static const int SHADOWS = PRESENT_SLOTS;
    struct shadow_copy {
        uint64_t *pages;  // physical
        uint32_t npages;
        uint64_t vram;  // 0 for a free entry
    } shadows[SHADOWS] = {};
    uint64_t shadow_va = 0;  // kernel VA for copies to and from a shadow
    // logged once per 10 s
    uint32_t promoted = 0, demoted = 0;
    uint64_t promote_mark = 0;
    uint64_t shadow_at(const uint64_t *pages, size_t n, int *which);
    int promote(present_slot *s, uint64_t bytes);
    void demote(present_slot *s);
    uint64_t present_frames = 0;
    struct disp *disp = nullptr;
    // screen i is head i, nub i and BAR1 framebuffer slot i
    static const int SCREENS = BAR1_FB_SLOTS;
    struct screen {
        IOService *nub;
        uint64_t shown;  // 0 while unlit; mode is 1-based
        uint64_t size;
        uint32_t pitch, mode;
        int out;
        bool changed;    // Framebuffer is to be told, outside the lock
        bool asleep;     // display sleep: head off, monitor in standby
        bool unplugged;  // head blanked, surface kept
        uint64_t scanout_va, scanout_vram;
        // Tear-free presents copy onto the next of three surfaces and flip at the first vblank after the copy's
        // semaphore. The window channel takes one flip at a time, so presents register theirs (pend) for pump_flips.
        uint64_t back[2], back_va[2], back_size, sema_va;
        int cur;     // 0 shown, 1 and 2 back
        int queued;  // the last flip queued to the head
        unsigned last_k;
        uint32_t last_fence;
        unsigned flips;  // the next semaphore is flips & 1, never the word the window still holds
        bool fresh;      // no flip queued since the last modeset
        uint32_t flip_fence[3];
        uint32_t fence;          // the last present's copy
        struct {
            bool valid;
            int to;
            uint32_t fence;
            unsigned k;
            uint64_t since;
        } pend;
        void (*vbl_fn)(void *);  // called under evlock
        void *vbl_ctx;
        void (*pipe_fn)(void *);  // the accelerator's display pipe, likewise
        void *pipe_ctx;
        uint32_t *cursor;  // kept for after a power cycle
        uint32_t cursor_w, cursor_h;
        int cursor_x, cursor_y;
        uint16_t *gamma;  // kept for after a power cycle
    } screens[SCREENS] = {};
    int index(const screen &sc) const { return (int)(&sc - screens); }
    static uint64_t surface_of(const screen &sc, int i) { return i ? sc.back[i - 1] : sc.shown; }
    // The window reuses its last flip's semaphore on the next update (a modeset's), so release it before forgetting.
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
    bool want_shown = false;  // hotplug lights a head when a monitor turns up
    uint32_t unlit_polls = 0; // ticks since the last connection check
    bool show_failed = false; // polls don't retry until what's connected changes
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
    }
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
    uint32_t gr_events = 0;  // graphics nonstall interrupts
    uint32_t vbl_wake = 0;   // presents waiting for a flip sleep on it, every vblank wakes them
    uint64_t vblanks = 0, vblank_mark = 0;  // head 0's, and when the last 600 started

    void stop_gsp();
    // entry points wait while suspended
    bool suspended = false;
    bool awake();
    void suspend();
    IOReturn resume();
    uint64_t canary = 0;  // tells whether VRAM kept its contents across suspend
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
    IOReturn decode_stream(const uint64_t *args, task_t task);
    static void poll(OSObject *owner, IOTimerEventSource *sender);
    thread_call_t autostart_call = nullptr;
    thread_call_t pump_call = nullptr;
    static void pump(thread_call_param_t self, thread_call_param_t);
    void pump_flips();
    uint32_t safe_sync = 0;
    struct disp_size scaled[DISP_MAX_SCALED];  // from scaled.txt, offered on panels they suit
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
    IOReturn queue_draw(void *ctx, const void *queue, const struct nvgpu_draw *d, const char **why);
    IOReturn queue_push(void *ctx, const void *queue, const struct malachite_push *p);
    IOReturn queue_copy2d(void *ctx, const struct nvgpu_copy2d *p);
    IOReturn queue_compress(void *ctx, uint64_t va);
    // SURFACE_FN_* through screen i's nub
    IOReturn screen_call(int i, const OSSymbol *name, void *p1, void *p2, void *p3, void *p4);

    using IOService::newUserClient;
    IOReturn newUserClient(task_t task, void *security, UInt32 type, OSDictionary *props,
                           IOUserClient **handler) override;
    IOReturn video_open(uintptr_t *ctx, struct video_dec *h, const uint64_t *args);
    IOReturn video_decode(uintptr_t ctx, struct video_dec *h, const uint64_t *args);
    void video_close(uintptr_t ctx, struct video_dec *h, uint32_t decoded);

    IOReturn command(uint32_t selector, const uint64_t *args, task_t task);
    size_t power_report(char *out, size_t cap);
    IOReturn escape(UserClient *uc, uint32_t cmd, void *arg, uint32_t size);
    void add_client(UserClient *uc);
    void remove_client(UserClient *uc);
    IOMemoryDescriptor *take_pending(UserClient *uc, IOOptionBits *options);
};

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

// open to any process, as VideoToolbox decodes in apps and in VTDecoderXPCService; a GPU context per client
class VideoUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(VideoUserClient)

public:
    Gpu *owner = nullptr;
    IOLock *busy = nullptr;  // one call at a time
    uintptr_t ctx = 0;
    struct video_dec dec = {};
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

// the same steps malachitectl runs by hand (nvnoauto=1 leaves them to it), which are then no-ops
void Gpu::autostart(thread_call_param_t p0, thread_call_param_t)
{
    Gpu *self = (Gpu *)p0;
    static const char *const names[] = MALACHITE_FW_NAMES;
    static const char base[] = "/Library/Application Support/Malachite/570";
    const char *dir = self->gpu.chip->fwdir;
    IOBufferMemoryDescriptor *fw[MALACHITE_FW_COUNT] = {};
    for (int tries = 0; tries < 600; tries++) {  // the root volume comes up some seconds into boot
        // a path lookup before the root is mounted panics in namei
        vnode_t root = vfs_rootvnode();
        if (!root) {
            IOSleep(1000);
            continue;
        }
        vnode_put(root);
        int have = 0;
        for (int i = 0; i < MALACHITE_FW_COUNT; i++) {
            char path[128];
            // gsp.bin is the same for every family
            if (dir && i != MALACHITE_FW_GSP)
                snprintf(path, sizeof(path), "%s/%s/%s", base, dir, names[i]);
            else
                snprintf(path, sizeof(path), "%s/%s", base, names[i]);
            if (!fw[i])
                fw[i] = read_file(path);
            have += fw[i] != nullptr;
        }
        if (have == MALACHITE_FW_COUNT)
            break;
        IOSleep(1000);
    }
    if (!fw[0] || !fw[1] || !fw[2] || !fw[3]) {
        log_msg("Malachite: no firmware in %s%s%s, waiting for malachitectl\n", base, dir ? "/" : "", dir ? dir : "");
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

// "2560x1440" a line, as the app last wrote it
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

// A lit scaled mode stays listed, as WindowServer can't have its current mode vanish.
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

// TryLock so a reader never waits behind a launch
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
    if (p.pstate_status)
        n += snprintf(out + n, cap - n, "pstate error %#x\n", p.pstate_status);
    else if (p.pstate == ~0u)
        n += snprintf(out + n, cap - n, "pstate none\n");
    else
        n += snprintf(out + n, cap - n, "pstate P%u\n", p.pstate);
    if (p.clk_status) {
        n += snprintf(out + n, cap - n, "clocks error %#x\n", p.clk_status);
    } else {
        // all ones for a clock the chip lacks
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

// A failed start leaves them registered and registering twice panics. Each re-registers as OID_AUTO, as its old
// number may be taken.
static bool sysctls_up;

static struct sysctl_oid *const sysctls[] = {
    &sysctl__kern_malachite_power, &sysctl__kern_malachite_stats, &sysctl__kern_malachite_prof_on,
    &sysctl__kern_malachite_prof, &sysctl__kern_malachite_present_apps,
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
    // a way back in when a driver build doesn't boot
    int off;
    if (PE_parse_boot_argn("-nonvidia", &off, sizeof(off)) || oldDriverLoaded())
        return false;
    if (!IOService::start(provider))
        return false;
    power_owner = this;
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
    uint16_t device = pci_device_id(&gpu);
    const char *name = chip_name(chipset), *product = chip_product(device);
    log_msg("Malachite: %s (%s), 10de:%04x %s\n", name ? name : "unknown chip", gpu.chip ? gpu.chip->family : "no family",
            device, product ? product : "(not a listed product)");
    if (!gpu.chip || gpu.chip->missing) {
        log_msg("Malachite: chipset 0x%x not supported%s%s\n", chipset, gpu.chip ? ", needs " : "",
                gpu.chip ? gpu.chip->missing : "");
        OSSafeReleaseNULL(bar0);
        return false;
    }
    if (gpu.chip->fwdir)
        setProperty("MalachiteFirmwareDir", gpu.chip->fwdir);
    // System Information and apps read VRAM size here, else they show the boot framebuffer's 16 MB
    if (uint64_t mb = rd32(&gpu, 0x1183a4)) {
        plat.pci->setProperty("VRAM,totalMB", mb, 32);
        plat.pci->setProperty("VRAM,totalsize", mb << 20, 64);
    }
    // for Accelerator and the Metal plugin; the FB controller has it before GSP-RM is up
    setProperty("MalachiteVRAMBytes", (unsigned long long)rd32(&gpu, 0x1183a4) << 20, 64);
    lock = IOLockAlloc();
    evlock = IOLockAlloc();

    loop = IOWorkLoop::workLoop();
    timer = loop ? IOTimerEventSource::timerEventSource(this, poll) : nullptr;
    if (!timer || loop->addEventSource(timer) != kIOReturnSuccess)
        return false;

    // without an MSI the timer does the work
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

    // for setPowerState() around system sleep, and systemWillShutdown()
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
    for (screen &sc : screens)  // their framebuffers go first
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

// the nubs are power children, so framebuffers go down first and come up last
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

// under the lock; waits out a power cycle for up to 30 s
bool Gpu::awake()
{
    uint64_t deadline;
    clock_interval_to_deadline(30, kSecondScale, &deadline);
    while (suspended && IOLockSleepDeadline(lock, &suspended, deadline, THREAD_UNINT) != THREAD_TIMED_OUT)
        ;
    return !suspended;
}

// Gpu keeps page tables, mappings and VRAM allocations so Accelerator's addresses survive; VRAM contents don't.
// Under the lock. Framebuffer stays online, as dark it would look like an unplug to WindowServer.
void Gpu::suspend()
{
    if (!gsp || suspended)
        return;
    uint64_t t0 = time_us();
    suspended = true;
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
    // Unloaded outright, as after gsp_suspend GSP-RM refused the display a core channel (INVALID_STATE or
    // NOT_SUPPORTED).
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

// Under the lock. A GPU that doesn't come back is let go of as MALACHITE_GSP_STOP would.
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
        // GSP-RM's channels survived, but not our instance blocks and contexts in VRAM
        log_msg("Malachite: VRAM lost its contents, making the GPU contexts anew\n");
        forget_contexts(true);
        gr_forget(gsp);
    } else if (up == 1) {
        forget_contexts(false);
    }
    log_msg("Malachite: GSP-RM %s, VRAM %s\n", up ? "booted afresh" : "resumed with its state",
            vram_kept ? "kept" : "lost");
    fe = rmfe_create(gsp);
    // kernel first, as tasks take their addresses from its address space
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
        if (sc.out >= disp->n || !o->connected || sc.mode > o->nmodes) {  // gone meanwhile; hotplug lights it again
            sc.unplugged = true;
            sc.nub->setProperty(SURFACE_ONLINE, kOSBooleanFalse);
            sc.changed = true;
            continue;
        }
        const struct disp_mode *t = o->nmodes ? &o->modes[sc.mode - 1] : &o->mode;
        vram_zero(&gpu, sc.shown, sc.size);  // after a sleep IOFramebuffer puts its copy back
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
        if (sc.gamma)  // the new LUTs are identities
            disp_gamma(disp, head, sc.gamma);
        if (sc.asleep)
            disp_blank(disp, head, sc.out);
        // flip semaphores live in the new notifier memory
        if (sc.sema_va && vmm_map_vram(&kctx.vm, sc.sema_va, disp->ntfy, 0x1000, 0))
            sc.sema_va = 0;
        publish_surface(sc, false);
    }
    vmm_flush(&gpu);
    if (disp)
        disp->hotplug = ~0u;  // the next poll checks connections
    else if (lit())
        log_msg("Malachite: no display after the power cycle\n");
    if (irq)
        gsp_irq_enable(gsp, 1);
    timer->setTimeoutMS(20);
    log_msg("Malachite: GPU resumed in %llu ms (GSP-RM %llu, contexts %llu, display %llu)\n", (time_us() - t0) / 1000,
            (t_gsp - t0) / 1000, (t_ctx - t_gsp) / 1000, (time_us() - t_ctx) / 1000);
    return kIOReturnSuccess;
}

// after GSP-RM lost the channels; the address space, mappings and buffers stay for ctx_rebind()
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

// hands the kept page tables to the new GSP-RM, so mappings and channel IDs stay
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
        chan_create(gsp, &c->client, &c->vm, engine, runlist, chid, handle, c == &kctx ? CHAN_KERNEL : CHAN_ADMIN,
                    &c->copy_ch))
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

// the context's buffers must be mapped already
bool Gpu::graphics_channel(gpu_ctx *c, uint32_t chid)
{
    if (compute_setup(gsp, &c->client, &c->vm, chid, c == &kctx, &c->compute_ch, &c->compute_ctx) ||
        render_setup(gsp, &c->client, &c->compute_ch)) {
        // GSP-RM's channel, if it got that far, goes before its memory
        if (c->compute_ch.res.inst)
            rm_free(gsp, c->client.client, c->compute_ch.handle);
        gr_ctx_free(gsp, &c->compute_ctx);
        chan_free(&c->compute_ch);
        return false;
    }
    vmm_flush(&gpu);
    c->state_of = nullptr;
    compute_set_tex_pools(&c->compute_ch, c->compute_cbufva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES,
                          c->compute_cbufva + TEX_SAMPLER_POOL, NVGPU_MAX_TEXTURES);
    render_set_tex_pools(&c->compute_ch, c->compute_cbufva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES,
                         c->compute_cbufva + TEX_SAMPLER_POOL, NVGPU_MAX_TEXTURES);
    render_set_vertex_substitute(&c->compute_ch, c->zero_va);
    if (c->slm_va) {
        compute_set_local_memory(&c->compute_ch, c->slm_va, c->slm_per_tpc);
        render_set_local_memory(&c->compute_ch, c->slm_va, c->slm_size, c->slm_per_warp);
    }
    return true;
}

// free_rm frees the clients too, so GSP-RM unloads holding nothing of ours
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

// before GSP-RM goes for good, leaving Accelerator's handles stale; under the lock
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
    stats[ST_CTXS] = 0;
    ctx_forget(&kctx);
    vmm_destroy(&kctx.vm);
    memset(&kctx, 0, sizeof(kctx));
    for (screen &sc : screens)
        sc.scanout_va = sc.back_va[0] = sc.back_va[1] = sc.sema_va = 0;
    for (present_slot &p : present)
        p.va = 0;
}

// the head's copy goes with the display state in a power cycle
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

// fb, if given, is a surface of the same size
int Gpu::modeset(screen &sc, const struct disp_mode *t, uint64_t fb)
{
    reset_flips(sc, sc.cur);
    return disp_modeset(disp, index(sc), sc.out, t, fb ? fb : sc.shown, sc.pitch);
}

// a slow sweep behind the interrupt, every 20 ms without one
void Gpu::poll(OSObject *owner, IOTimerEventSource *sender)
{
    Gpu *self = (Gpu *)owner;
    IOLockLock(self->lock);
    if (self->fe) {
        rmfe_poll(self->fe);
        // GSP-RM posts no plug event for some monitors
        if (self->disp && self->want_shown && !self->lit() && ++self->unlit_polls >= (self->irq ? 2u : 100u)) {
            self->unlit_polls = 0;
            self->disp->hotplug = ~0u;
        }
        self->check_hotplug();
        sender->setTimeoutMS(self->irq ? 1000 : 20);
    }
    IOLockUnlock(self->lock);
    self->tell_nubs();
}

// outside the lock, as Framebuffer may set a mode in response
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

// for nvrm_xnu_wait_event
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
    // tasks allocate VAs from the kernel's space, so each VA names one mapping
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
        present_fence = 0;  // a new semaphore word no earlier present will write
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
    d->handle = md;  // not owned
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
    // apps go through L2, as a BVH read uncached over the bus ran ~15 times slower
    unsigned flags = c != &kctx && strcmp(c->owner, "WindowServer") ? VMM_CACHED : 0;
    c->cached_sys |= flags != 0;
    if (flags)
        c->l2_seq = 0;  // L2 may hold lines of these pages from their last owner
    if (!va || (copy ? vmm_map_vram(&c->vm, va, copy, size, 0) : vmm_map_sys(&c->vm, va, d, 0, size, flags))) {
        if (va)
            map_va_free(c, va, size, false);
        mem_free(d->pages);
        md->complete();
        memset(d, 0, sizeof(*d));
        return kIOReturnError;
    }
    gpu_maps[slot].shadow = (uint8_t)which;
    gpu_maps[slot].uncached = false;
    map_slack(c, va + size);
    vmm_flush(&gpu);  // else later copies fault "not valid" on stale TLB entries
    md->retain();  // the unmap may come after the buffer's owner let go
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
            map_va_free(c, va, size, true);
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

// a non-null owner may only drop its own mappings
IOReturn Gpu::gpu_unmap(uint64_t va, uint64_t size, gpu_ctx *owner)
{
    int i = map_slot(va);
    if (i < 0 || (owner && gpu_maps[i].ctx != owner))
        return kIOReturnNotFound;
    gpu_maps[i].ctx->kind_va = 0;
    struct dma *d = &gpu_maps[i].dma;
    uint64_t s = slack(d->size, gpu_maps[i].vram != 0);
    vmm_unmap(&gpu_maps[i].ctx->vm, va - s, d->size + 2 * s);
    vmm_flush(&gpu);
    vmm_free(&gpu_maps[i].ctx->vm, va - s, d->size + 2 * s);
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

// its mappings are all unmapped at close anyway
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
    // ordered after queued draws it may read, and before later ones (after_copies)
    kick_draws(c);
    if (c->draw_pending)
        chan_acquire_queue(&c->copy_ch, c->compute_semva, c->draw_pending);
    c->ce_pending = c->ce_last = c->draw_pending = ++c->compute_seq;
    chan_copy_surface_queue(&c->copy_ch, &s, &d, cp->bytes, cp->lines, cp->remap, c->compute_semva, c->compute_seq);
    memset(c->pass_targets, 0, sizeof(c->pass_targets));  // the next draw may sample what it wrote
    return kIOReturnSuccess;
}

// queued like a 2D copy, since draining first held the lock for all the work before it
IOReturn Gpu::copy_after_draws(gpu_ctx *c, uint64_t src, uint64_t dst, uint64_t size)
{
    if (c == &kctx || !c->compute_ready || size > UINT32_MAX) {
        drain(c);
        return gpu_copy(c, src, dst, size);
    }
    struct nvgpu_copy2d cp = {};
    cp.src = src;
    cp.dst = dst;
    cp.src_pitch = cp.dst_pitch = cp.bytes = (uint32_t)size;
    cp.lines = 1;
    return gpu_copy2d(c, &cp);
}

// entering the other half waits for the copies queued there a lap ago
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

// the copies wrote what the graphics channel may read next
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

// Buffers are mapped pitch before anyone knows they'll hold a depth target, whose Z kind sets its block-linear
// layout. Only c's own mappings; compress means the buffer holds nothing else.
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
        if (compress && m->vram && !((m->va | m->vram | m->dma.size) & 0xffff))
            packed = compressed_kind(kind);
        // queued work on the old PTEs faults mid-remap (PTE not valid, UNSUPPORTED_KIND), seen in 1 launch in 4
        if (drain(c) != kIOReturnSuccess)
            return kIOReturnTimeout;
        struct vmm *vm = &m->ctx->vm;
        int failed;
        if (packed) {
            failed = vmm_map_vram_big(vm, m->va, m->vram, m->dma.size, VMM_KINDF(packed));
        } else if (m->vram) {
            vmm_unmap(vm, m->va, m->dma.size);  // big pages would hide the small ones
            failed = vmm_map_vram(vm, m->va, m->vram, m->dma.size, VMM_KINDF(kind));
        } else {
            failed = vmm_map_sys(vm, m->va, &m->dma, 0, m->dma.size, VMM_KINDF(kind) | sys_flags(m));
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

// 0 when the pages aren't in a shadow, *which then untouched
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

// 0 when there's no room
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
        drain(m->ctx);  // draws in flight into the pages, and caches holding them
        if (!vmm_map_vram(&m->ctx->vm, m->va, at, m->dma.size, 0)) {
            m->shadow = (uint8_t)which;
            moved++;
        }
    }
    vmm_flush(&gpu);
    // no mapping draws to it, so WindowServer doesn't render it on the GPU
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
    vmm_map_vram(&kctx.vm, s->va, vram, bytes, 0);
    vmm_flush(&gpu);
    promoted++;
    return i + 1;
}

void Gpu::demote(present_slot *s)
{
    shadow_copy &sh = shadows[s->shadow - 1];
    uint64_t bytes = (uint64_t)sh.npages << 12;
    for (int j = 0; j < gpu_maps_top; j++)
        if (gpu_maps[j].va && gpu_maps[j].shadow == s->shadow)
            drain(gpu_maps[j].ctx);
    volatile uint32_t *sem = (volatile uint32_t *)kctx.copy_sem->cpu;
    vmm_map_sys(&kctx.vm, s->va, &s->dma, 0, s->dma.size, 0);  // copy target for what was drawn in VRAM
    if (!vmm_map_vram(&kctx.vm, shadow_va, sh.vram, bytes, 0)) {
        vmm_flush(&gpu);
        chan_copy(&kctx.copy_ch, shadow_va, s->va, bytes, sem, kctx.copy_semva, ++kctx.copy_seq);
        vmm_unmap(&kctx.vm, shadow_va, bytes);
    }
    for (int j = 0; j < gpu_maps_top; j++) {
        struct gpu_mapping *m = &gpu_maps[j];
        if (m->va && m->shadow == s->shadow) {
            vmm_map_sys(&m->ctx->vm, m->va, &m->dma, 0, m->dma.size, sys_flags(m));
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

// gpu_present is busy while the previous frame is still copied or flipped; the caller retries without the lock
Gpu::present_slot *Gpu::oldest_slot()
{
    present_slot *s = &present[0];
    for (int i = 1; i < PRESENT_SLOTS; i++)
        if (present[i].used < s->used)
            s = &present[i];
    return s;
}

IOReturn Gpu::fill_slot(present_slot *s, IOMemoryDescriptor *md, size_t size)
{
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
    return kIOReturnSuccess;
}

// GL windows go to VRAM, as WindowServer over PCIe caps GL at 120 swaps a second. Not Metal's, whose drawables the
// CPU may read; being mapped by another process tells a window from GL's offscreen surfaces.
void Gpu::adopt_surface(int i)
{
    struct gpu_mapping &m = gpu_maps[i];
    if (m.vram || m.shadow || !m.dma.handle || m.dma.size < (1u << 20) || m.dma.size > PRESENT_SLOT_SIZE)
        return;
    if (++m.passes < 30)
        return;
    m.passes = -600;
    bool shared = false;
    for (int j = 0; j < gpu_maps_top && !shared; j++)
        shared = gpu_maps[j].va && gpu_maps[j].ctx != m.ctx && gpu_maps[j].dma.pages &&
                 gpu_maps[j].dma.pages[0] == m.dma.pages[0];
    if (!shared)
        return;
    present_slot *s = oldest_slot();
    if ((s->fence && !copy_done(s->fence)) || fill_slot(s, (IOMemoryDescriptor *)m.dma.handle, m.dma.size))
        return;
    s->used = ++present_frames;
    s->shadow = promote(s, s->dma.size);
}

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
    if (sc.scanout_vram != sc.shown) {  // a mode change
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
        // the flip semaphores live in notifier memory
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
    // don't remap a slot a copy is still reading
    auto copying = [&](const present_slot *q) { return q->fence && !copy_done(q->fence); };
    if (s && copying(s))
        return time_us() - since < 2000000 ? kIOReturnBusy : kIOReturnTimeout;
    if (!s) {
        s = oldest_slot();
        if (copying(s))
            return time_us() - since < 2000000 ? kIOReturnBusy : kIOReturnTimeout;
        if (IOReturn r = fill_slot(s, md, size))
            return r;
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
        // without spare surfaces, copies go onto the shown one, one at a time
        if (sc.fence && !copy_done(sc.fence))
            return time_us() - since < 2000000 ? kIOReturnBusy : kIOReturnTimeout;
    } else if (sc.pend.valid) {
        // our target surface is free once the previous flip is queued
        return time_us() - since < 2000000 ? kIOReturnBusy : kIOReturnTimeout;
    }
    if (s->shadow && !(shadows[s->shadow - 1].vram && shadows[s->shadow - 1].pages[0] == s->dma.pages[0])) {
        s->shadow = 0;
        vmm_map_sys(&kctx.vm, s->va, &s->dma, 0, s->dma.size, 0);
        vmm_flush(&gpu);
    }
    // a full-screen surface shown repeatedly is promoted to VRAM
    if (flip && !p->offset && p->pitch == sc.pitch && (uint64_t)p->pitch * p->height == sc.size &&
        !s->shadow && ++s->count > 2 && !(s->shadow = promote(s, s->dma.size)))
        s->count = -64;
    const uint64_t bva[3] = { sc.scanout_va, sc.back_va[0], sc.back_va[1] };
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
    if (prof_on) {  // waiting here holds the present up
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

// Not in the present itself, as the window channel holds a flip's update until the last one is through and the kick
// waits for that.
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
                // after 100 ms give up and copy onto the shown surface from now on
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
        // the flip notifier is written just after the vblank interrupt
        uint64_t deadline;
        clock_interval_to_deadline(1000, kMicrosecondScale, &deadline);
        IOLockLock(evlock);
        IOLockSleepDeadline(evlock, &vbl_wake, deadline, THREAD_UNINT);
        IOLockUnlock(evlock);
        delay_us(100);
    }
}

// waits without the lock, so WindowServer draws the next frame meanwhile
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
        if (c && c->draw_pending && c->compute_sem && (present_apps || !strcmp(c->owner, "WindowServer")))
            waits[n++] = { c, c->draw_pending };
    give();
    const uint64_t since = time_us();
    uint64_t drawn_at = 0;
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
            if (!drawn_at) {
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
            drain_all();  // recovers from a fault
        }
        give();
        if (r != kIOReturnBusy) {
            const uint64_t now = time_us();
            stats[ST_PRESENT_TOTAL_US] += now - since;
            if (drawn_at)
                stats[ST_PRESENT_POST_US] += now - drawn_at;
            return r;
        }
        // vblanks are 4 ms apart at 240 Hz, so wake on one rather than IOSleep's millisecond tick
        if (time_us() - since > 4000) {
            uint64_t deadline;
            clock_interval_to_deadline(1000, kMicrosecondScale, &deadline);
            IOLockLock(evlock);
            IOLockSleepDeadline(evlock, &vbl_wake, deadline, THREAD_UNINT);
            IOLockUnlock(evlock);
            delay_us(100);  // the flip notifier is written just after the vblank interrupt
        } else {
            delay_us(50);
        }
    }
}

// Fresh VRAM holds another process's pixels, which WindowServer's backdrops once showed; zeroed like system memory.
int Gpu::clear_vram(uint64_t vram, uint64_t size)
{
    uint64_t va = vmm_alloc(&kctx.vm, size, 0x10000);
    if (!va)
        return -1;
    int r = vmm_map_vram(&kctx.vm, va, vram, size, 0);
    if (!r) {
        vmm_flush(&gpu);
        r = gpu_fill(&kctx, va, size, 0) != kIOReturnSuccess;
        vmm_unmap(&kctx.vm, va, size);
    }
    vmm_free(&kctx.vm, va, size);
    return r;
}

IOReturn Gpu::gpu_fill(gpu_ctx *c, uint64_t dst, uint64_t size, uint32_t pattern)
{
    if (!gpu_bridge_init(c))
        return kIOReturnNotReady;
    if (!size)
        return kIOReturnSuccess;
    // doubling copies from a 4-byte seed; pattern is one byte repeated, so any length and alignment works
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
    if (!gpu_bridge_init(c))  // shares the client and address space
        return false;
    if (c->compute_sem) {  // after a power cycle, buffers still mapped
        c->compute_ready = graphics_channel(c, c->compute_ch.chid);
        if (c->compute_ready)
            issued(c);
        return c->compute_ready;
    }
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
    if (!c->draw_ringva)
        return false;
    if (vmm_map_sys(&c->vm, c->draw_ringva, c->draw_ring, 0, c->draw_ring->size, 0))
        return false;
    c->dispatch_ringva = vmm_alloc(&c->vm, c->dispatch_ring->size, 0x1000);
    if (!c->dispatch_ringva || vmm_map_sys(&c->vm, c->dispatch_ringva, c->dispatch_ring, 0, c->dispatch_ring->size, 0))
        return false;
    c->compute_qmdva = vmm_alloc(&c->vm, c->compute_qmd->size, 0x1000);
    c->compute_cbufva = vmm_alloc(&c->vm, c->compute_cbuf->size, 0x1000);
    c->zero_va = vmm_alloc(&c->vm, 0x1000, 0x1000);
    if (!c->compute_qmdva || !c->compute_cbufva || !c->zero_va ||
        vmm_map_sys(&c->vm, c->compute_qmdva, c->compute_qmd, 0, c->compute_qmd->size, 0) ||
        vmm_map_sys(&c->vm, c->compute_cbufva, c->compute_cbuf, 0, c->compute_cbuf->size, 0) ||
        !map_slack(c, c->zero_va))
        return false;
    vmm_flush(&gpu);
    c->compute_sem = (volatile uint32_t *)((uint8_t *)c->compute_qmd->cpu + 0x800);
    c->compute_semva = c->compute_qmdva + 0x800;
    compute_set_tex_pools(&c->compute_ch, c->compute_cbufva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES,
                          c->compute_cbufva + TEX_SAMPLER_POOL, NVGPU_MAX_TEXTURES);
    render_set_tex_pools(&c->compute_ch, c->compute_cbufva + TEX_HEADER_POOL, NVGPU_MAX_TEXTURES,
                         c->compute_cbufva + TEX_SAMPLER_POOL, NVGPU_MAX_TEXTURES);
    render_set_vertex_substitute(&c->compute_ch, c->zero_va);
    c->compute_ready = true;
    log_msg("Malachite: compute bridge ready (qmd va 0x%llx cbuf va 0x%llx)\n", c->compute_qmdva, c->compute_cbufva);
    issued(c);  // for replaying the first launch
    return true;
}

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

// handle 0 is the kernel's; a closed one gives NULL
gpu_ctx *Gpu::ctx_of(uintptr_t handle)
{
    if (!handle)
        return &kctx;
    unsigned slot = (unsigned)(handle & 0xff) - 1;
    if (slot >= MAX_CTXS || !ctxs[slot] || ctx_serial[slot] != (uint32_t)(handle >> 8))
        return nullptr;
    return ctxs[slot];
}

// nothing made on the GPU until first use; 0 when all are taken; under lock
uintptr_t Gpu::ctx_open()
{
    for (int i = 0; i < MAX_CTXS; i++) {
        if (ctxs[i])
            continue;
        ctxs[i] = (gpu_ctx *)IOMallocZero(sizeof(gpu_ctx));
        if (!ctxs[i])
            return 0;
        proc_selfname(ctxs[i]->owner, sizeof(ctxs[i]->owner));  // runs on the task's own thread
        ctx_serial[i] = (ctx_serial[i] + 1) & 0xffffff;
        if (++stats[ST_CTXS] > stats[ST_CTXS_PEAK])
            stats[ST_CTXS_PEAK] = stats[ST_CTXS];
        return (uintptr_t)(i + 1) | (uintptr_t)ctx_serial[i] << 8;
    }
    if (stats[ST_CTXS_REFUSED]++ % 100 == 0) {
        char who[32];
        proc_selfname(who, sizeof(who));
        log_msg("Malachite: %s gets no GPU context, all %d are taken\n", who, MAX_CTXS);
    }
    return 0;
}

// under lock
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
    // kept across a power cycle, so freed even if the channels aren't back
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
    stats[ST_CTXS]--;
}

// GSP-RM never runs a torn-down channel again, so recover() rebuilds it before next use. May run inside a wait.
void Gpu::rc_event(void *ctx, uint32_t engine, uint32_t chid, uint32_t exception, const gsp_fault *fault)
{
    Gpu *self = (Gpu *)ctx;
    for (int i = -1; i < MAX_CTXS; i++) {
        gpu_ctx *c = i < 0 ? &self->kctx : self->ctxs[i];
        if (c && c->compute_ready && chid == c->compute_ch.chid)
            c->compute_faulted = true;
        if (c && c->gpu_ready && chid == c->copy_ch.chid)
            c->copy_faulted = true;
        if (c && (chid == c->compute_ch.chid || chid == c->copy_ch.chid)) {
            c->fault = fault ? *fault : gsp_fault{};
            c->fault_chid = chid;
        }
    }
}

void Gpu::log_fault(gpu_ctx *c, const char *channel, uint32_t chid)
{
    const char *owner = c == &kctx ? "kernel" : c->owner;
    if (!c->fault.type || c->fault_chid != chid) {
        log_msg("Malachite: %s's %s channel faulted, making a new one\n", owner, channel);
        return;
    }
    const gsp_fault &f = c->fault;
    log_msg("Malachite: %s's %s channel faulted: %s fault on a %s at %#llx by %s, making a new one\n", owner, channel,
            f.type, f.access, (unsigned long long)f.va, f.client);
    const gpu_mapping *in = nullptr, *before = nullptr, *after = nullptr;
    for (int i = 0; i < gpu_maps_top; i++) {
        const gpu_mapping *m = &gpu_maps[i];
        if (m->ctx != c || !m->va)
            continue;
        if (m->va <= f.va && f.va < m->va + m->dma.size)
            in = m;
        else if (m->va + m->dma.size <= f.va && (!before || m->va > before->va))
            before = m;
        else if (m->va > f.va && (!after || m->va < after->va))
            after = m;
    }
    const gpu_mapping *near[] = { in, before, after };
    for (const gpu_mapping *m : near)
        if (m)
            log_msg("Malachite:   %s the fault at %#llx: %#llx+%#llx %s kind %#x%s%s%s\n",
                    m == in ? "in" : m == before ? "before" : "after", (unsigned long long)f.va,
                    (unsigned long long)m->va, (unsigned long long)m->dma.size, m->vram ? "VRAM" : "system", m->kind,
                    m->compressed ? " compressed" : "", m->shadow ? " shadowed" : "",
                    !m->vram && sys_flags(m) ? " cached" : "");
    c->fault = {};
}

// A flip queued behind a lost copy would wait forever, so release the semaphores and modeset each lit head. Under
// lock.
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

// Rebuilt on the same VA space, mappings kept. Under lock.
void Gpu::recover(gpu_ctx *c)
{
    recoveries++;
    // A task faulting on every launch would keep the GPU busy with rebuilds, so its work is refused, longer each
    // time. WindowServer's refusals stay short, as the desktop is unusable without it.
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
        log_fault(c, "copy", c->copy_ch.chid);
        uint32_t handle = c->copy_ch.handle, chid = c->copy_ch.chid;
        rm_free(gsp, c->client.client, handle);
        chan_free(&c->copy_ch);
        if (!copy_channel(c, chid, handle)) {
            log_msg("Malachite: copy channel not remade\n");
            c->gpu_ready = false;
        }
        // lost copies never release, and the graphics channel may be waiting on them
        if (c->compute_sem && (c->ce_last || c->ce_pending)) {
            *c->compute_sem = c->compute_seq;
            c->ce_pending = c->ce_last = c->draw_pending = 0;
        }
        c->l2_seq = 0;  // a lost write-back or invalidate counts as done
        memset(c->ce_half_seq, 0, sizeof(c->ce_half_seq));
        c->ce_half = 0;
        vmm_flush(&gpu);
        if (c == &kctx) {  // a flip may be waiting on a lost release
            resync_screens();
            present_fence = 0;  // its copy is lost
        }
    }
    if (c->compute_faulted) {
        c->compute_faulted = false;
        bool have_fault = c->fault.type && c->fault_chid == c->compute_ch.chid;
        uint64_t fault_va = c->fault.va;
        bool write_fault = have_fault && c->fault.access && !strcmp(c->fault.access, "write");
        log_fault(c, "graphics", c->compute_ch.chid);
        auto mapped = [&](uint64_t va) {
            if (!va)
                return '-';
            for (int i = 0; i < gpu_maps_top; i++)
                if (gpu_maps[i].ctx == c && gpu_maps[i].va <= va && va < gpu_maps[i].va + gpu_maps[i].dma.size)
                    return 'm';
            return 'X';
        };
        const unsigned logged = gpu_ctx::LAUNCH_LOG;
        // writes are posted, so the faulting launch may have finished already
        unsigned from = c->launch_next > logged ? c->launch_next - logged : 0, last = c->launch_next;
        for (unsigned k = from; k < c->launch_next; k++) {
            const auto &l = c->launches[k % logged];
            if (l.cbuf && (int32_t)(l.seq - *c->compute_sem) <= 0)
                last = k;
        }
        if (write_fault && last < c->launch_next) {
            const auto &l = c->launches[last % logged];
            bool reused = false;
            for (unsigned j = last + 1; j < c->launch_next; j++)
                reused |= c->launches[j % logged].cbuf == l.cbuf;
            char names[160];
            launch_names(c, l, names, sizeof(names));
            log_msg("Malachite:   the last one done: %s %u of program %#llx%s\n", l.vs ? "draw" : "dispatch", l.seq,
                    (unsigned long long)l.prog, names);
            if (!reused)
                log_bound(c, l, have_fault, fault_va);
        }
        // fetch state carries from each draw through the runs after it
        pe_state pe;
        pe_unknown(&pe);
        for (unsigned k = from; k < c->launch_next; k++) {
            const auto &l = c->launches[k % logged];
            bool pending = l.seq && (int32_t)(l.seq - *c->compute_sem) > 0;
            char why[160];
            if (l.run_words) {
                decode_run(c, l.run_va, l.run_words, &pe, why, sizeof(why));
                if (pending)
                    log_msg("Malachite:   not done: run %u of %u draws, %u words at %#llx%s%s\n", l.seq, l.run_draws,
                            l.run_words, (unsigned long long)l.run_va, *why ? ": " : "", why);
                continue;
            }
            if (l.vs)
                pe = l.pe;
            if (!pending)
                continue;
            char names[160];
            launch_names(c, l, names, sizeof(names));
            log_msg("Malachite:   not done: %s %u of program %#llx%s, %ux%ux%u groups of %ux%ux%u, %u regs\n",
                        l.vs ? "draw" : "dispatch", l.seq, (unsigned long long)l.prog, names, l.groups[0],
                        l.groups[1], l.groups[2], l.block[0], l.block[1], l.block[2], l.regs);
            if (l.vs)
                log_msg("Malachite:     vs args %#llx %#llx, ps args %#llx %#llx, target %#llx zt %#llx; mapped: ps %c "
                        "vs %c args %c%c%c%c target %c zt %c\n",
                        (unsigned long long)l.params[0], (unsigned long long)l.params[1],
                        (unsigned long long)l.params[2], (unsigned long long)l.params[3], (unsigned long long)l.target,
                        (unsigned long long)l.zt, mapped(l.prog), mapped(l.vs), mapped(l.params[0]),
                        mapped(l.params[1]), mapped(l.params[2]), mapped(l.params[3]), mapped(l.target), mapped(l.zt));
            if (l.vs) {
                // stream:address/stride/size[/divisor], attribute:stream+offset/SET_VERTEX_ATTRIBUTE_A
                const pe_state &s = l.pe;
                char streams[256] = "", attributes[256] = "";
                for (uint32_t j = 0, n = 0; j < RENDER_STREAMS && n < sizeof(streams); j++) {
                    if (!(s.format[j] & PE_STREAM_ON))
                        continue;
                    n += snprintf(streams + n, sizeof(streams) - n, " %u:%#llx/%u/%#x", j,
                                  (unsigned long long)s.addr[j], s.format[j] & 0xfff, s.size[j]);
                    if (s.instanced >> j & 1 && n < sizeof(streams))
                        n += snprintf(streams + n, sizeof(streams) - n, "/%u", s.divisor[j]);
                }
                for (uint32_t i = 0, n = 0; i < RENDER_ATTRIBUTES && n < sizeof(attributes); i++)
                    if (!(s.attributes[i] & ATTR_INACTIVE))
                        n += snprintf(attributes + n, sizeof(attributes) - n, " %u:%u+%#x/%#x", i,
                                      s.attributes[i] & 0x1f, s.attributes[i] >> 7 & 0x3fff, s.attributes[i]);
                bool indexed = s.index != 0;
                int64_t top = pe_top(c, &s, indexed);
                bool wrong = pe_wrong(&s, indexed, why, sizeof(why)) ||
                             (top >= 0 && pe_past_end(&s, (uint32_t)top, why, sizeof(why)));
                char tops[24] = "not readable";
                if (top >= 0)
                    snprintf(tops, sizeof(tops), "to %lld", (long long)top);
                else if (top == -2)
                    strlcpy(tops, "past the buffer", sizeof(tops));
                log_msg("Malachite:     vs %#llx, index %#llx+%#x format %u restart %u at %#x, %s %u+%u base %d, "
                        "instances %u from %u, %s %s, streams%s\n",
                        (unsigned long long)l.vs, (unsigned long long)s.index, s.index_size, s.index_format, s.restart,
                        s.restart_index, indexed ? "indices" : "vertices", s.first, s.count, s.base_vertex, s.instances,
                        s.base_instance, indexed ? "indices" : "vertices", tops, *streams ? streams : " none");
                log_msg("Malachite:     attributes%s%s%s\n", *attributes ? attributes : " none", wrong ? ": " : "",
                        wrong ? why : "");
            }
            if (l.seq == *c->compute_sem + 1) {
                log_bound(c, l, have_fault, fault_va);
                log_msg("Malachite:   the first one's code, ps or program then vs\n");
                log_code(l.prog);
                if (l.vs)
                    log_code(l.vs);
            }
        }
        rm_free(gsp, c->client.client, c->compute_ch.handle);
        // replay from the first launch not done, or the one after if it was already replayed once
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
        if (!saved) {  // lost work never releases its slots, nor what the copy channel may wait on
            memset(c->draw_slot_seq, 0, sizeof(c->draw_slot_seq));
            memset(c->dispatch_slot_seq, 0, sizeof(c->dispatch_slot_seq));
            if (c->compute_sem)
                *c->compute_sem = c->compute_seq;
        }
        memset(c->pass_targets, 0, sizeof(c->pass_targets));
        memset(c->half_seq, 0, sizeof(c->half_seq));
        c->push_half = 0;
        // a new chid, as GSP-RM may still be finishing with the faulted one
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
        // the payload the last launch to run again releases itself
        const uint32_t last_replayed = c->issued_next ? c->issued[(c->issued_next - 1) % 128].payload : 0;
        c->issued_next = 0;
        c->draws_queued = 0;
        if (saved) {
            if (chan_replay(&c->compute_ch, saved, nsaved))
                log_msg("Malachite: %s's work lost to the fault doesn't fit the new channel\n",
                        c == &kctx ? "kernel" : c->owner);
            else
                log_msg("Malachite: %s's work from %u on runs again (GP_PUT %u, semaphore %#x, issued up to %#x)\n",
                        c == &kctx ? "kernel" : c->owner, first, c->compute_ch.gp_put, *c->compute_sem, c->compute_seq);
            IOFree(saved, CHAN_SAVE_WORDS * 4);
            c->replayed = first;
            // release up to the latest payload, past the one not replayed; when the last launch to run again releases
            // it, a second release could land after another channel's release of a later payload and take it back
            if (c->compute_seq != last_replayed)
                render_release(&c->compute_ch, c->compute_semva, c->compute_seq);
            c->half_seq[0] = c->half_seq[1] = c->draw_pending = c->compute_seq;
            c->push_half = chan_half(&c->compute_ch, 1);
        }
        issued(c);
    }
}

// The lock blocks the interrupt handler, so poll GSP-RM here to rebuild a faulted channel instead of timing out.
int Gpu::ring_wait(gpu_ctx *c, int counter, uint32_t payload, int ms)
{
    if ((int32_t)(*c->compute_sem - payload) >= 0)  // nearly always
        return 0;
    uint64_t t0 = time_us();
    int r = -1;
    kick_draws(c);
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

// after earlier work, which may write them; via the copy engine, so VRAM works too
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

// a handle names at most 2^20 headers and 4096 samplers (TEXSAMP's 12 bits)
static bool heap_ok(const struct nvgpu_heap *h)
{
    return !h->headers || (h->samplers && h->nheaders && h->nheaders <= 1u << 20 && h->nsamplers &&
                           h->nsamplers <= 4096);
}

// entering the other half waits for the work last submitted there
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

// sized as NVK does for every resident thread; only grows, and only while the context is idle
IOReturn Gpu::local_memory(gpu_ctx *c, uint32_t lane)
{
    if (lane <= c->slm_lane)
        return kIOReturnSuccess;
    if (lane > 4096)
        return kIOReturnBadArgument;
    const struct gsp_info *in = gsp_get_info(gsp);
    uint64_t warps = in->gr[0x0d], sms = in->gr[0x20], tpcs = in->gr[0x09];  // MAX_WARPS_PER_SM, SM_PER_TPC, TPCs
    // TPC indices run past the enabled count when some are floorswept, so size for LITTER_NUM_GPCS *
    // LITTER_NUM_TPC_PER_GPC; a share past the end faults or writes another buffer
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
    if (!d || d->params_size > NVGPU_COMPUTE_PARAM_BYTES || !d->grid_x || !d->block_x || d->shared_size > gpu.chip->max_shared ||
        d->barriers > 16 || d->nheaders > NVGPU_MAX_TEXTURES || d->nsamplers > NVGPU_MAX_TEXTURES ||
        !heap_ok(&d->heap))
        return kIOReturnBadArgument;
    // a CTA whose registers don't fit faults the channel; warps are dealt round-robin over four 16K partitions
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
    // c[0x0][0x118] (UR4 global base) stays 0; ntid.xyz and nctaid.xyz at 0x0 as CUDA's driver does
    uint32_t block[3] = { d->block_x, d->block_y ? d->block_y : 1, d->block_z ? d->block_z : 1 };
    uint32_t grid[3] = { d->grid_x, d->grid_y ? d->grid_y : 1, d->grid_z ? d->grid_z : 1 };
    if (d->indirect) {
        IOReturn r = read_indirect(c, d->indirect, grid, 3);
        if (r != kIOReturnSuccess)
            return r;
        if (!grid[0] || !grid[1] || !grid[2])
            return kIOReturnSuccess;
    }
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
    // it may read what earlier launches wrote, and they may still use the pools
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
    if (prof_on && prof_ready(c)) {
        uint32_t i = c->prof_n++;
        c->prof_draws[i % PROF_RING] = { d->prog_va, 0, grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2],
                                         0x80, 0, (uint16_t)block[0], (uint16_t)MIN(grid[0], 65535u) };
        compute_qmd_set_stamp((uint32_t *)(slot_cpu + 0x1000), c->prof_va + (i % PROF_RING) * 16, i + 1);
    }
    gpu_ctx::launch_log &l = c->launches[c->launch_next++ % gpu_ctx::LAUNCH_LOG];
    l = { payload, { grid[0], grid[1], grid[2] }, { block[0], block[1], block[2] }, d->reg_count, d->prog_va, 0 };
    l.cbuf = slot_cpu;
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

void Gpu::prof_harvest(gpu_ctx *c)
{
    if (!c->prof)
        return;
    if (c->prof_n - c->prof_seen > PROF_RING) {  // the ring wrapped
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
    // tessellation spacing 3 isn't one, and loses the channel
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

IOReturn Gpu::gpu_draw(gpu_ctx *c, const struct nvgpu_draw *d, const void *queue, const char **reject)
{
    // runs mustn't build on state this draw hasn't sent yet
    c->state_of = nullptr;
    if (const char *why = draw_reject(d)) {
        if (reject)
            *reject = why;
        return kIOReturnBadArgument;
    }
    if (!compute_bridge_init(c))
        return kIOReturnNotReady;
    if (ndeferred || nvram_deferred)
        retire_deferred();
    // for the stats; try the previous draw's target first, as a pass draws to one
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
        bool vram = gpu_maps[target_map].vram || gpu_maps[target_map].shadow;
        stats[vram ? ST_DRAW_VRAM : ST_DRAW_SYS]++;
        stats[vram ? ST_DRAW_VRAM_KPX : ST_DRAW_SYS_KPX] += (uint64_t)d->width * d->height / 1000;
        if (d->pass_first && d->opengl)
            adopt_surface(target_map);
    }
    IOReturn lr = d->local_size ? local_memory(c, d->local_size) : kIOReturnSuccess;
    if (lr != kIOReturnSuccess)
        return lr;
    // Z16, ZF32_X24S8, generic
    if (d->zt && d->zt_kind != 0x01 && d->zt_kind != 0x04 && d->zt_kind != 0x06) {
        if (reject)
            *reject = "zt kind";
        return kIOReturnBadArgument;
    }
    if (d->zt && (d->zt != c->kind_va || d->zt_kind != c->kind_set)) {  // set_kind walks every mapping
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
    // the most the draw, its per-layer clears and their releases take
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
    r.target_width = d->target_width;
    r.target_height = d->target_height;
    r.alpha_control = d->alpha_control;
    r.per_sample = d->per_sample;
    r.ps_depth = d->ps_depth;
    r.layers = d->layers;
    __builtin_memcpy(r.layer_pitch, d->layer_pitch, sizeof(r.layer_pitch));
    r.zt_layer_pitch = d->zt_layer_pitch;
    __builtin_memcpy(r.target_layers, d->target_layers, sizeof(r.target_layers));
    r.zt_layers = d->zt_layers;
    r.shader_layer = d->shader_layer;
    r.nviewports = d->nviewports;
    r.viewports = d->viewports;
    r.scissors = d->scissors;
    r.write_mask = d->write_mask;
    __builtin_memcpy(r.blend, d->blend, sizeof(r.blend));
    for (int j = 0; j < 7; j++) {
        const struct nvgpu_target *t = &d->more[j];
        r.more[j] = { t->addr, t->pitch, t->format, t->blocklinear, t->write_mask, {}, t->width, t->height };
        __builtin_memcpy(r.more[j].blend, t->blend, sizeof(t->blend));
    }
    uint32_t layers = r.layers;
    if (d->target && d->clear) {
        r.layers = draw_target_layers(&r, 0);
        render_clear(&c->compute_ch, &r, 0, d->clear_rgba, nullptr, c->compute_semva, ++c->compute_seq);
        c->draw_pending = c->compute_seq;
    }
    for (unsigned j = 0; j < 7; j++)
        if (d->more[j].addr && d->more[j].clear) {
            r.layers = layers;
            r.layers = draw_target_layers(&r, j + 1);
            render_clear(&c->compute_ch, &r, j + 1, d->more[j].clear_rgba, nullptr, c->compute_semva, ++c->compute_seq);
            c->draw_pending = c->compute_seq;
        }
    r.layers = layers;
    // draws within a pass don't wait for each other unless serialized; new targets mean a pass that may sample the last
    uint64_t targets[9] = { d->target, d->zt };
    for (int j = 0; j < 7; j++)
        targets[2 + j] = d->more[j].addr;
    r.wait_idle = memcmp(targets, c->pass_targets, sizeof(targets)) != 0 || d->serialize;
    __builtin_memcpy(c->pass_targets, targets, sizeof(targets));
    // also once a lap of the draw slots, whose constants were rewritten
    r.invalidate = r.wait_idle || d->vs != c->pass_vs || d->ps != c->pass_ps || d->ti != c->pass_ti ||
                   d->ts != c->pass_ts || slot == 0;
    if (safe_sync)  // nvsafesync=1
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
    // a pass loading the last one's depth keeps zcull's state, so a prepass helps the pass after it
    if (d->pass_first && !(d->zt && d->zt == c->zcull_zt && d->zt < c->zcull_hi && !(d->clear_zs & 1))) {
        const struct gsp_zcull *z = &gsp_get_info(gsp)->zcull;
        if (z->subregion_count && d->zt && (d->clear_zs & 1) && d->layers <= 1)
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
    if (!count || !instances)  // a clear-only pass
        return kIOReturnSuccess;
    uint8_t *cb = (uint8_t *)c->draw_ring->cpu + slot * DRAW_SLOT_BYTES;
    uint64_t cbva = c->draw_ringva + slot * DRAW_SLOT_BYTES;
    // only what shaders read is cleared; a tessellated draw uses more
    if (d->tess) {
        memset(cb, 0, DRAW_SLOT_BYTES);
    } else {
        memset(cb, 0, 0x280);
        memset(cb + 0x1000, 0, 0x280);
        memset(cb + 0x2000, 0, NVGPU_MAX_TEXTURES * sizeof(d->headers[0]));
        memset(cb + 0x2800, 0, NVGPU_MAX_TEXTURES * sizeof(d->samplers[0]));
    }
    memcpy(cb + 0x160, d->vs_params, d->vs_params_size);
    // NVK_BASE_VERTEX, NVK_BASE_INSTANCE; non-indexed draws give first as [[base_vertex]], as Apple GPUs do
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
    __builtin_memcpy(r.stencil, d->stencil, sizeof(r.stencil));
    r.stencil_ref = d->stencil_ref;
    __builtin_memcpy(r.stencil_read_mask, d->stencil_read_mask, sizeof(r.stencil_read_mask));
    __builtin_memcpy(r.stencil_write_mask, d->stencil_write_mask, sizeof(r.stencil_write_mask));
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
    __builtin_memcpy(r.depth_bias, d->depth_bias, sizeof(r.depth_bias));
    __builtin_memcpy(r.blend_const, d->blend_const, sizeof(r.blend_const));
    r.visibility = d->visibility;
    r.visibility_reset = d->visibility_reset;
    __builtin_memcpy(r.viewport, d->viewport, sizeof(r.viewport));
    __builtin_memcpy(r.scissor, d->scissor, sizeof(r.scissor));
    __builtin_memcpy(r.attributes, d->attributes, sizeof(r.attributes));
    for (int s = 0; s < RENDER_STREAMS; s++) {
        r.streams[s].addr = d->streams[s].addr;
        r.streams[s].stride = d->streams[s].stride;
        r.streams[s].size = d->streams[s].size;
        r.streams[s].divisor = d->streams[s].divisor;
    }
    render_draw_queued(&c->compute_ch, &r, c->compute_semva, ++c->compute_seq);
    c->draws_queued++;
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
    gpu_ctx::launch_log &l = c->launches[c->launch_next++ % gpu_ctx::LAUNCH_LOG];
    l = { c->compute_seq, { r.count, r.instances, 0 }, {}, d->vs_regs, r.ps, r.vs };
    __builtin_memcpy(l.params, d->vs_params, 16);
    __builtin_memcpy(l.params + 2, d->ps_params, 16);
    l.target = d->target;
    l.zt = d->zt;
    l.cbuf = cb;
    pe_draw(&l.pe, &r);
    issued(c);
    c->draw_slot_seq[slot] = c->draw_pending = c->compute_seq;
    c->state_of = queue;
    c->runs_dropped = false;
    return kIOReturnSuccess;
}

IOReturn Gpu::drain(gpu_ctx *c, int ms)
{
    memset(c->pass_targets, 0, sizeof(c->pass_targets));  // a copy, dispatch or the CPU may write next
    kick_draws(c);
    if (!c->draw_pending)
        return kIOReturnSuccess;
    uint64_t t0 = time_us();
    if ((int32_t)(*c->compute_sem - c->draw_pending) < 0)
        stats[ST_DRAIN_WAITS]++;
    // on a fault, stop waiting once GSP-RM reports it and recover
    int r = -1;
    for (int t = 0; t < ms / 10 && r; t++) {
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
    if (prof_on && !r)  // for apps that never present, such as OpenCL
        prof_harvest(c);
    if (r)
        log_msg("Malachite: draws never finished (semaphore 0x%x)\n", *c->compute_sem);
    return r ? kIOReturnTimeout : kIOReturnSuccess;
}

// The lock is only taken to check, as holding it would stall WindowServer. L2 writeback follows the work so the copy
// channel doesn't wait switched out; OpenCL waits per kernel, so short waits poll closely.
IOReturn Gpu::sync_ctx(uintptr_t handle)
{
    uint64_t t0 = time_us();
    auto done = [](const gpu_ctx *c) {
        return !c->draw_pending || !c->compute_sem || (int32_t)(*c->compute_sem - c->draw_pending) >= 0;
    };
    // what this call waits for: work another thread of the task queues meanwhile is for its own sync to wait on
    uint32_t target = 0;
    auto reached = [&](const gpu_ctx *c) {
        return !target || !c->compute_sem || (int32_t)(*c->compute_sem - target) >= 0;
    };
    for (bool flushed = false, first = true;;) {
        take();
        gpu_ctx *c = ctx_of(handle);
        if (!c || c->refused_until > time_us()) {
            give();
            return !c ? kIOReturnNotReady : kIOReturnNotPermitted;
        }
        kick_draws(c);
        if (gsp && (gsp_pending(gsp) || c->compute_faulted || c->copy_faulted)) {
            gsp_poll(gsp);
            if (c->compute_faulted || c->copy_faulted)
                recover(c);
        }
        uint64_t waited = time_us() - t0;
        if (first) {
            target = c->draw_pending;
            first = false;
        }
        if (!flushed && reached(c)) {
            l2_sysmem(c, false);
            target = c->draw_pending;
            flushed = true;
        }
        if ((flushed && reached(c)) || waited > 2000000) {
            stats[ST_SYNC]++;
            IOReturn r;
            if (done(c) || waited > 2000000) {
                r = settle(drain(c, 10));
            } else {
                // more work is queued since: drain would wait for it too and drop it if it ran out of time
                memset(c->pass_targets, 0, sizeof(c->pass_targets));
                r = settle(kIOReturnSuccess);
            }
            stats[ST_SYNC_US] += time_us() - t0;
            give();
            return r;
        }
        give();
        delay_us(waited < 1000 ? 2 : waited < 10000 ? 20 : 1000);
    }
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

// a closed context gets kIOReturnNotReady
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

// called directly, as callPlatformFunction's symbol lookup and compares cost too much thousands of times a frame
IOReturn Gpu::queue_dispatch(void *ctx, const struct nvgpu_dispatch *d)
{
    if (!usable())
        return kIOReturnNotReady;
    WITH_CTX(ctx, (stats[ST_DISPATCH]++, busy_open(c, launch(c, d, nullptr))));
}

IOReturn Gpu::queue_draw(void *ctx, const void *queue, const struct nvgpu_draw *d, const char **why)
{
    if (!usable())
        return kIOReturnNotReady;
    WITH_CTX(ctx, (stats[ST_DRAW]++, busy_open(c, draw(c, d, queue, why))));
}

IOReturn Gpu::queue_push(void *ctx, const void *queue, const struct malachite_push *p)
{
    if (!usable())
        return kIOReturnNotReady;
    if (!p->words || p->words > 0x1fffff || (p->va & 3))
        return kIOReturnBadArgument;
    WITH_CTX(ctx, (stats[ST_DRAW] += p->draws, busy_open(c, push_run(c, p, queue))));
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

IOReturn queue_draw(IOService *gpu, void *ctx, const void *queue, const struct nvgpu_draw *d, const char **why)
{
    return static_cast<Gpu *>(gpu)->queue_draw(ctx, queue, d, why);
}

IOReturn queue_push(IOService *gpu, void *ctx, const void *queue, const struct malachite_push *p)
{
    return static_cast<Gpu *>(gpu)->queue_push(ctx, queue, p);
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
    // everything was drained before a power cycle, so any semaphore is done
    if (suspended) {
        if (name->isEqualTo(NVGPU_FN_DRAW_DONE) || name->isEqualTo(NVGPU_FN_FENCE))
            return kIOReturnSuccess;
        if (!usable())
            return kIOReturnNotReady;
    }
    // refused contexts still map, as apps crash on allocations they don't expect to fail
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
        uint64_t vram = size && gpu_bridge_init(&kctx) ? vram_alloc(gsp_vram(gsp), size, 0x10000) : 0;
        static bool told;
        if (vram && clear_vram(vram, size) && !told) {
            told = true;
            log_msg("Malachite: VRAM of %llu bytes not cleared before it was handed out\n", (unsigned long long)size);
        }
        *(uint64_t *)p2 = vram;
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
        if (nvram_deferred < MAX_DEFERRED)  // any task's draw in flight may be reading it
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
            ndeferred < MAX_DEFERRED) {  // its task's draws in flight may be using it
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
        WITH_CTX(p4, (stats[ST_COPY]++,
                      copy_after_draws(c, (uint64_t)(uintptr_t)p1, (uint64_t)(uintptr_t)p2, (uint64_t)(uintptr_t)p3)));
    if (name->isEqualTo(NVGPU_FN_UPLOAD))
        WITH_CTX(p4, (stats[ST_COPY]++, memset(c->pass_targets, 0, sizeof(c->pass_targets)),
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
        return sync_ctx((uintptr_t)p1);
    if (name->isEqualTo(NVGPU_FN_DRAW_FENCE))
        // the CPU may write memory before the next command buffer, so its draws start a pass
        WITH_CTX(p2, (l2_sysmem(c, false), kick_draws(c), *(uint32_t *)p1 = c->draw_pending,
                      memset(c->pass_targets, 0, sizeof(c->pass_targets)), kIOReturnSuccess));
    if (name->isEqualTo(NVGPU_FN_CB_BEGIN))
        WITH_CTX(p1, (l2_sysmem(c, true), kIOReturnSuccess));
    if (name->isEqualTo(NVGPU_FN_ASYNC)) {
        take();
        gpu_ctx *c = ctx_of((uintptr_t)p1);
        bool ws = c && !strcmp(c->owner, "WindowServer");
        give();
        *(bool *)p2 = !ws;
        return ws ? kIOReturnSuccess : kIOReturnUnsupported;
    }
    if (name->isEqualTo(NVGPU_FN_CTX_BUSY)) {  // lock-free, as Accelerator asks while submitting
        gpu_ctx *c = ctx_of((uintptr_t)p1);
        *(uint64_t *)p2 = c ? busy_take(c) : 0;
        return kIOReturnSuccess;
    }
    if (name->isEqualTo(NVGPU_FN_DRAW_DONE)) {
        // a dying WindowServer closes the context under a timer still polling it; the timer retries on busy
        if (!IOLockTryLock(lock))
            return kIOReturnBusy;
        gpu_ctx *c = ctx_of((uintptr_t)p2);
        if (c && caught_up(c))
            busy_close(c);
        IOReturn r = kIOReturnSuccess;
        if (c && c->compute_sem && (int32_t)(*c->compute_sem - (uint32_t)(uintptr_t)p1) < 0) {
            r = kIOReturnBusy;
            // unless the channel faulted, which GSP-RM may only now report
            if (gsp && (gsp_pending(gsp) || c->compute_faulted)) {
                gsp_poll(gsp);
                if (c->compute_faulted)
                    compute_bridge_init(c);
            }
        }
        IOLockUnlock(lock);
        return r;
    }
#undef WITH_CTX
    if (name->isEqualTo(NVGPU_FN_STATS))
        return gpu_stats((struct nvgpu_stats *)p1);
    if (name->isEqualTo(NVGPU_FN_FENCE)) {
        uint32_t fence = (uint32_t)(uintptr_t)p1;
        // lock-free, as it runs from the vblank handler the lock's holders may be waiting on
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

// Accelerator calls it every 250 ms. TryLock, as a launch can hold the lock for its whole GPU wait and a missed tick
// costs nothing.
IOReturn Gpu::gpu_stats(struct nvgpu_stats *out)
{
    if (!IOLockTryLock(lock))
        return kIOReturnBusy;
    memset(out, 0, sizeof(*out));
    // TSENS in 1/256 C from bit 8; all ones in a low power state, 0 when unclocked
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
    // close finished intervals nothing else closed, then bring busy_total up to now
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

// published at start so WindowServer finds the framebuffers; each stays dark until its head is lit
void Gpu::create_nubs()
{
    uint64_t bar1_phys = bar_phys(&gpu, 1), bar1_len = bar_size(&gpu, 1);
    // without them macOS has no screen though the GPU works, so say why
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
            // the likeliest mode, so lighting the head doesn't change geometry under WindowServer
            { SURFACE_WIDTH, 1920 }, { SURFACE_HEIGHT, 1080 }, { SURFACE_PITCH, 7680 }, { SURFACE_REFRESH, 60ull << 16 },
        };
        for (const auto &e : nums) {
            OSNumber *n = OSNumber::withNumber(e.v, 64);
            props->setObject(e.key, n);
            OSSafeReleaseNULL(n);
        }
        props->setObject(SURFACE_ONLINE, kOSBooleanFalse);
        props->setObject(SURFACE_HW_CURSOR, kOSBooleanTrue);  // IOFramebuffer asks on open, before the head lights
        SurfaceNub *nub = new SurfaceNub;
        if (nub && nub->init(props) && nub->attach(this)) {
            nub->screen = index(sc);
            nub->setName(SURFACE_NUB_NAME);
            // a power child, so the framebuffer sleeps before the GPU suspends and wakes after it resumes
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
    // macOS wants the mode's own size, with the clock scaled to keep the refresh rate
    uint64_t total = (uint64_t)(m->hactive + m->hblank) * (m->vactive + m->vblank);
    uint64_t shown = (uint64_t)(mode_w(m) + m->hblank) * (mode_h(m) + m->vblank);
    const struct surface_timing t = { (uint32_t)(m->clock * shown / total), (uint16_t)mode_w(m), m->hblank, m->hsync_off, m->hsync_len,
                                   (uint16_t)mode_h(m), m->vblank, m->vsync_off, m->vsync_len,
                                   (uint8_t)(m->flags >> 1 & 1), (uint8_t)(m->flags >> 2 & 1) };
    return OSData::withBytes(&t, sizeof(t));
}

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

// under the lock; with hotplug, a newly lit head looks like a plugged-in monitor to Framebuffer
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
    sc.changed = true;  // for tell_nubs()
    log_msg("Malachite: screen %d lit, %ux%u\n", index(sc), mode_w(m), mode_h(m));
}

// display sleep detaches the head from its OR and puts the sink in standby
IOReturn Gpu::set_power(screen &sc, bool on)
{
    if (!disp || !sc.shown)
        return kIOReturnNotReady;
    if (on == !sc.asleep)
        return kIOReturnSuccess;
    if (sc.unplugged) {  // hotplug lights the head when the monitor is back
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
    // presents flip between surfaces, and shown may still hold the boot screen
    uint64_t fb = surface_of(sc, sc.cur);
    if (modeset(sc, t, fb))
        return kIOReturnError;
    sc.asleep = false;
    if (sc.gamma)  // may have changed while the head was off
        disp_gamma(disp, index(sc), sc.gamma);
    return kIOReturnSuccess;
}

IOReturn Gpu::set_mode(screen &sc, uint32_t id)
{
    if (!disp || !sc.shown)
        return kIOReturnNotReady;
    struct disp_output *o = &disp->out[sc.out];
    if (id < 1 || id > o->nmodes)
        return kIOReturnBadArgument;
    if (id == sc.mode)
        return kIOReturnSuccess;
    // DP sinks list modes for their fastest link; disp_modeset trains the one this mode needs
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
    vram_zero(&gpu, fb, size);
    reset_flips(sc);
    if (int r = disp_modeset(disp, index(sc), sc.out, t, fb, pitch)) {
        vram_free(gsp_vram(gsp), fb, size);
        // no link for the new mode, so restore the old one
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

// screen 0 via BAR1, or via the copy engine when the head shows a back surface
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

struct frame_sink {
    task_t task;
    uint64_t addr, len, off;
    uint32_t count;
    bool fault;
};

static int sink_frame(void *arg, const struct video_frame *f)
{
    frame_sink *o = (frame_sink *)arg;
    struct malachite_frame h = { f->index, f->idr_count, f->poc, f->width, f->height, f->bps };
    uint64_t bytes = (uint64_t)f->width * f->height * 3 / 2 * f->bps;
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

IOReturn Gpu::decode_stream(const uint64_t *args, task_t task)
{
    uint64_t len = args[1];
    if (!len || len > 16u << 20 || args[3] < sizeof(struct malachite_frames) || args[5] > NVDEC_VP8)
        return kIOReturnBadArgument;
    uint8_t *stream = (uint8_t *)IOMalloc(len);
    if (!stream)
        return kIOReturnNoMemory;
    frame_sink sink = { task, args[2], args[3], sizeof(struct malachite_frames) };
    IOReturn ret = kIOReturnBadArgument;
    if (!copy_from_user(task, stream, args[0], len)) {
        int n = ::video_decode(&kctx.video, (enum nvdec_codec)args[5], stream, len, (uint32_t)args[4], sink_frame,
                               &sink);
        // pictures before a failed one still go out
        struct malachite_frames hdr = { sink.count };
        if (sink.fault || copy_to_user(task, args[2], &hdr, sizeof(hdr)))
            ret = kIOReturnBadArgument;
        else
            ret = n < 0 ? kIOReturnError : kIOReturnSuccess;
    }
    IOFree(stream, len);
    return ret;
}

// screen 0 keeps what the EFI firmware showed, the others start black
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
        vram_zero(&gpu, fb, size);
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

// Under the lock. Firmware-lit outputs go first so screen 0 keeps the firmware's screen; an output that won't train
// doesn't stop the others.
IOReturn Gpu::show()
{
    if (!disp)
        return kIOReturnNotReady;
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
    // polls retry only once what's connected changes
    show_failed = skipped;
    failed_connected = disp->connected;
    return lit() ? kIOReturnSuccess : skipped ? kIOReturnOffline : kIOReturnNoDevice;
}

// under the lock
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
    bool event = disp->hotplug != ~0u;  // from GSP-RM rather than a poll
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
            // the same mode if the monitor still has it, else its preferred one
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
            // OpenCL's compute units
            setProperty("MalachiteSMCount", (unsigned long long)gsp_get_info(gsp)->gr[0x09] * gsp_get_info(gsp)->gr[0x20], 32);
            // the one property sandboxed processes may read; data with its NUL as on real Macs, as some launchers
            // throw on a string
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
    case MALACHITE_VIDEO_DECODE:
        ret = gsp && video_init(&kctx) ? decode_stream(args, task) : kIOReturnError;
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
        } else {
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

// root only, as escapes are raw GSP-RM calls that reach any memory, engine or display
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
            (selector == MALACHITE_VIDEO_DECODE && args->scalarInputCount < 6))
            return kIOReturnBadArgument;
        return owner->command(selector, args->scalarInput, task);
    }
}

IOReturn UserClient::clientMemoryForType(UInt32 type, IOOptionBits *options, IOMemoryDescriptor **memory)
{
    IOMemoryDescriptor *md = type == 0 ? owner->take_pending(this, options) : nullptr;
    if (!md)
        return kIOReturnBadArgument;
    *memory = md;  // the caller takes the reference
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
    // capped, as any process may open one and each costs a GPU context others need
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

// decode buffers are remade for each stream size
IOReturn Gpu::video_open(uintptr_t *ctx, struct video_dec *h, const uint64_t *args)
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
        // only a capability check for the codec
        ret = args[4] <= NVDEC_VP8 && video_supports(&c->video, (enum nvdec_codec)args[4]) ? kIOReturnSuccess
                                                                                         : kIOReturnUnsupported;
    } else if (args[4] > NVDEC_VP8) {
        ret = kIOReturnBadArgument;
    } else {
        if (h->width_mbs)
            video_dec_close(&c->video, h);
        ret = video_dec_open(&c->video, h, (enum nvdec_codec)args[4], (uint32_t)args[0], (uint32_t)args[1],
                             (uint32_t)args[5], (uint32_t)args[2], args[3])
                  ? kIOReturnBadArgument
                  : kIOReturnSuccess;
        log_msg("video: %s (pid %d) decodes %s of %llux%llu macroblocks into %llu surfaces: %#x\n", c->owner,
                proc_selfpid(), nvdec_codec_name((enum nvdec_codec)args[4]), args[0], args[1], args[2], ret);
    }
    IOLockUnlock(lock);
    return ret;
}

IOReturn Gpu::video_decode(uintptr_t ctx, struct video_dec *h, const uint64_t *args)
{
    uint32_t index = (uint32_t)args[1], payload;
    IOLockLock(lock);
    gpu_ctx *c = ctx ? ctx_of(ctx) : nullptr;  // 0 would be the kernel's
    bool queued = c && c->video_ready && h->width_mbs &&
                  !video_dec_queue(&c->video, h, (uint32_t)args[0], index, (uint32_t)args[2], &payload);
    IOLockUnlock(lock);
    if (!queued)
        return kIOReturnBadArgument;
    // without the lock, on the client's own semaphore, so others have the GPU meanwhile
    return video_dec_wait(&c->video, h, payload, index) ? kIOReturnTimeout : kIOReturnSuccess;
}

void Gpu::video_close(uintptr_t ctx, struct video_dec *h, uint32_t decoded)
{
    IOLockLock(lock);
    gpu_ctx *c = ctx ? ctx_of(ctx) : nullptr;  // 0 would be the kernel's
    if (c)
        log_msg("video: %s decoded %u pictures\n", c->owner, decoded);
    if (c && h->width_mbs)
        video_dec_close(&c->video, h);
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
    if (args->scalarInputCount < (selector == NVVIDEO_OPEN ? 6u : 3u))
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
        md->retain();  // the caller takes it
        *memory = md;
        *options = type >= NVVIDEO_MAP_SURFACE ? kIOMapReadOnly : 0;
    }
    IOLockUnlock(busy);
    return md ? kIOReturnSuccess : kIOReturnBadArgument;
}

}  // namespace malachite::rm
