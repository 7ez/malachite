// AMPERE_COMPUTE_B (0xc7c0): QMD dispatches on the graphics engine
#include "chip.h"
#include "../nvrm/gr.h"
#include "../nvrm/gsp.h"
#include "chan.h"
#include "compute.h"
#include "gr.h"
#include "gsp.h"
#include "mem.h"

#define PAGE 0x1000ull

// NVC7C0 method byte offsets (sdk clc7c0.h)
#define C_SET_OBJECT 0x0000
#define C_INVALIDATE_SKED_CACHES 0x0298
#define C_SHARED_WINDOW_A 0x02a0
#define C_WAIT_FOR_IDLE 0x0110
#define C_SEND_PCAS_A 0x02b4
#define C_SEND_SIGNALING_PCAS2_B 0x02c0
#define C_INVALIDATE_TEXHDR_NO_WFI 0x0244
#define C_LOCAL_WINDOW_A 0x07b0
#define C_SET_SHADER_LOCAL_MEMORY_A 0x0790
#define C_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A 0x02e4
#define C_INVALIDATE_SAMPLER_NO_WFI 0x1424
#define C_SET_RENDER_ENABLE_OVERRIDE 0x1944
#define C_SET_TEX_SAMPLER_POOL_A 0x155c
#define C_SET_TEX_HEADER_POOL_A 0x1574

// nvc0_screen.c's reserved generic-address windows for shared/local memory (Volta+ form)
#define SHARED_MEMORY_WINDOW (0xfeull << 24)  // vmm_alloc keeps both windows clear of mappings
#define LOCAL_MEMORY_WINDOW (0xffull << 24)

#define KERNEL_PARAM_OFFSET 0x160  // where sm_86 nvcc puts the first kernel parameter in const bank 0
#define CBUF_SIZE COMPUTE_CBUF_BYTES

// QMDV03_00 field bit ranges (hi, lo), from sdk clc7c0qmd.h
#define QF_INVALIDATE_TEXTURE_HEADER_CACHE 186, 186
#define QF_INVALIDATE_TEXTURE_SAMPLER_CACHE 187, 187
#define QF_INVALIDATE_TEXTURE_DATA_CACHE 188, 188
#define QF_INVALIDATE_SHADER_DATA_CACHE 189, 189
#define QF_INVALIDATE_INSTRUCTION_CACHE 190, 190
#define QF_INVALIDATE_SHADER_CONSTANT_CACHE 191, 191
#define QF_SM_GLOBAL_CACHING 134, 134
#define QF_CTA_RASTER_WIDTH 415, 384
#define QF_CTA_RASTER_HEIGHT 431, 416
#define QF_CTA_RASTER_DEPTH 463, 448
#define QF_SHARED_MEMORY_SIZE 561, 544
#define QF_MIN_SM_CONFIG_SHARED_MEM_SIZE 567, 562
#define QF_MAX_SM_CONFIG_SHARED_MEM_SIZE 574, 569
#define QF_QMD_VERSION 579, 576
#define QF_QMD_MAJOR_VERSION 583, 580
#define QF_TARGET_SM_CONFIG_SHARED_MEM_SIZE 662, 657
#define QF_CTA_THREAD_DIMENSION0 607, 592
#define QF_CTA_THREAD_DIMENSION1 623, 608
#define QF_CTA_THREAD_DIMENSION2 639, 624
#define QF_CONSTANT_BUFFER0_VALID 640, 640
#define QF_REGISTER_COUNT_V 656, 648
#define QF_SLM_LOW_SIZE 759, 736
#define QF_BARRIER_COUNT 767, 763
#define QF_RELEASE0_ADDRESS_LOWER 799, 768
#define QF_RELEASE0_ADDRESS_UPPER 807, 800
#define QF_RELEASE0_MEMBAR_TYPE 819, 819
#define QF_RELEASE0_ENABLE 823, 823
#define QF_RELEASE0_STRUCTURE_SIZE 831, 830
#define QF_RELEASE0_PAYLOAD_LOWER 863, 832
#define QF_CONSTANT_BUFFER0_ADDR_LOWER 1055, 1024
#define QF_CONSTANT_BUFFER0_ADDR_UPPER 1072, 1056
#define QF_CONSTANT_BUFFER0_SIZE_SHIFTED4 1087, 1075
#define QF_PROGRAM_ADDRESS_LOWER 1567, 1536
#define QF_PROGRAM_ADDRESS_UPPER 1584, 1568
#define QF_SLM_HIGH_SIZE 1623, 1600

static void qmd_set(uint32_t *qmd, unsigned hi, unsigned lo, uint64_t val)
{
    for (unsigned b = lo; b <= hi; b++) {
        if ((val >> (b - lo)) & 1)
            qmd[b / 32] |= 1u << (b % 32);
        else
            qmd[b / 32] &= ~(1u << (b % 32));
    }
}
#define QSET(qmd, field, val) qmd_set((qmd), field, (val))

// smallest GA10x shared-memory carve-out (0, 8, 16, 32, 64, 100 KB) holding bytes, in the QMD's KB / 4 + 1 form
static uint32_t smem_config(uint32_t bytes)
{
    static const uint32_t kb[] = { 0, 8, 16, 32, 64, 100 };
    unsigned i = 0;
    while (i + 1 < sizeof kb / sizeof kb[0] && kb[i] * 1024 < bytes)
        i++;
    return kb[i] / 4 + 1;
}

void compute_build_qmd(uint32_t *q, uint64_t prog_va, uint64_t cbuf_va, uint64_t sem_va, uint32_t payload,
                      const uint32_t block[3], const uint32_t grid[3], uint32_t reg_count, uint32_t shared_size,
                      uint32_t barriers)
{
    memset(q, 0, 256);
    QSET(q, QF_BARRIER_COUNT, barriers);  // BAR.SYNC n traps ("Illegal Instruction Parameter") without them
    if (shared_size) {
        QSET(q, QF_SHARED_MEMORY_SIZE, (shared_size + 0xff) & ~0xffu);
        QSET(q, QF_MIN_SM_CONFIG_SHARED_MEM_SIZE, smem_config(shared_size));
        QSET(q, QF_TARGET_SM_CONFIG_SHARED_MEM_SIZE, smem_config(100 * 1024));
        QSET(q, QF_MAX_SM_CONFIG_SHARED_MEM_SIZE, smem_config(100 * 1024));
    }
    QSET(q, QF_QMD_MAJOR_VERSION, 3);
    QSET(q, QF_QMD_VERSION, 0);
    QSET(q, QF_SM_GLOBAL_CACHING, 1);
    QSET(q, QF_INVALIDATE_INSTRUCTION_CACHE, 1);   // each dispatch re-fetches its program and
    QSET(q, QF_INVALIDATE_SHADER_CONSTANT_CACHE, 1);  // re-reads the const bank (params change per phase)
    QSET(q, QF_INVALIDATE_SHADER_DATA_CACHE, 1);
    QSET(q, QF_INVALIDATE_TEXTURE_HEADER_CACHE, 1);  // texture/sampler descriptors are rewritten per dispatch
    QSET(q, QF_INVALIDATE_TEXTURE_SAMPLER_CACHE, 1);
    QSET(q, QF_INVALIDATE_TEXTURE_DATA_CACHE, 1);
    QSET(q, QF_CTA_RASTER_WIDTH, grid[0]);
    QSET(q, QF_CTA_RASTER_HEIGHT, grid[1]);
    QSET(q, QF_CTA_RASTER_DEPTH, grid[2]);
    QSET(q, QF_CTA_THREAD_DIMENSION0, block[0]);
    QSET(q, QF_CTA_THREAD_DIMENSION1, block[1]);
    QSET(q, QF_CTA_THREAD_DIMENSION2, block[2]);
    QSET(q, QF_REGISTER_COUNT_V, reg_count);
    QSET(q, QF_PROGRAM_ADDRESS_LOWER, prog_va & 0xffffffff);
    QSET(q, QF_PROGRAM_ADDRESS_UPPER, prog_va >> 32);
    QSET(q, QF_CONSTANT_BUFFER0_ADDR_LOWER, cbuf_va & 0xffffffff);
    QSET(q, QF_CONSTANT_BUFFER0_ADDR_UPPER, cbuf_va >> 32);
    QSET(q, QF_CONSTANT_BUFFER0_SIZE_SHIFTED4, CBUF_SIZE >> 4);
    QSET(q, QF_CONSTANT_BUFFER0_VALID, 1);
    QSET(q, QF_RELEASE0_ADDRESS_LOWER, sem_va & 0xffffffff);
    QSET(q, QF_RELEASE0_ADDRESS_UPPER, sem_va >> 32);
    QSET(q, QF_RELEASE0_PAYLOAD_LOWER, payload);
    QSET(q, QF_RELEASE0_STRUCTURE_SIZE, 1);  // one word
    QSET(q, QF_RELEASE0_MEMBAR_TYPE, 1);     // sysmembar: flush the kernel's stores to memory too
    QSET(q, QF_RELEASE0_ENABLE, 1);
}

void compute_qmd_set_local(uint32_t *q, uint32_t bytes)
{
    QSET(q, QF_SLM_LOW_SIZE, (bytes + 15) & ~15u);
}

#define QF_RELEASE1_ADDRESS_LOWER 927, 896
#define QF_RELEASE1_ADDRESS_UPPER 935, 928
#define QF_RELEASE1_ENABLE 951, 951
#define QF_RELEASE1_STRUCTURE_SIZE 959, 958
#define QF_RELEASE1_PAYLOAD_LOWER 991, 960

void compute_qmd_set_stamp(uint32_t *q, uint64_t va, uint32_t payload)
{
    QSET(q, QF_RELEASE1_ADDRESS_LOWER, (uint32_t)va);
    QSET(q, QF_RELEASE1_ADDRESS_UPPER, va >> 32);
    QSET(q, QF_RELEASE1_PAYLOAD_LOWER, payload);
    QSET(q, QF_RELEASE1_STRUCTURE_SIZE, 0);  // four words: the payload, then the time at +8
    QSET(q, QF_RELEASE1_ENABLE, 1);
}

void compute_set_local_memory(struct chan *ch, uint64_t va, uint64_t per_tpc)
{
    uint32_t push[] = {
        MTHD(1, C_WAIT_FOR_IDLE, 1), 0,
        MTHD(1, C_SET_SHADER_LOCAL_MEMORY_A, 2), (uint32_t)(va >> 32), (uint32_t)va,
        MTHD(1, C_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 3), (uint32_t)(per_tpc >> 32), (uint32_t)per_tpc, 0xff,
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

void compute_set_tex_pools(struct chan *ch, uint64_t header_va, uint32_t headers, uint64_t sampler_va,
                           uint32_t samplers)
{
    uint32_t push[] = {
        MTHD(1, C_SET_TEX_HEADER_POOL_A, 3), (uint32_t)(header_va >> 32), (uint32_t)header_va, headers - 1,
        MTHD(1, C_SET_TEX_SAMPLER_POOL_A, 3), (uint32_t)(sampler_va >> 32), (uint32_t)sampler_va, samplers - 1,
    };
    chan_push(ch, push, sizeof(push) / 4);  // the launch or draw after it kicks
}

void compute_wait_idle(struct chan *ch)
{
    uint32_t push[] = { MTHD(1, C_WAIT_FOR_IDLE, 1), 0 };
    chan_push(ch, push, sizeof(push) / 4);  // the launch after it kicks
}

void compute_launch(struct chan *ch, uint64_t qmd_va)
{
    uint32_t push[] = {
        MTHD(1, C_SET_RENDER_ENABLE_OVERRIDE, 1), 1,  // ALWAYS_RENDER around the launch
        MTHD(1, C_SEND_PCAS_A, 1), (uint32_t)(qmd_va >> 8),
        MTHD(1, C_SEND_SIGNALING_PCAS2_B, 1), 3,      // PCAS_ACTION_INVALIDATE_COPY_SCHEDULE
        MTHD(1, C_SET_RENDER_ENABLE_OVERRIDE, 1), 0,  // USE_RENDER_ENABLE
    };
    chan_submit(ch, push, sizeof(push) / 4);
}

int compute_run(struct chan *ch, uint64_t qmd_va, volatile uint32_t *sem, uint32_t payload)
{
    compute_launch(ch, qmd_va);
    return wait_word(sem, payload, 2000);
}

int compute_setup(struct gsp *s, const struct rm_client *c, struct vmm *vm, uint32_t chid, int privileged,
                  struct chan *ch, struct gr_ctx *ctx)
{
    uint32_t engine, runlist;
    if (chan_find_engine(s, c, CHAN_GRAPHICS, &engine, &runlist) ||
        chan_create(s, c, vm, engine, runlist, chid, 0xc56f0000 | chid, privileged, ch) || chan_schedule(ch) ||
        gr_promote(s, c, ch->handle, vm, !privileged, ctx))
        return -1;
    if (rm_alloc(s, c->client, ch->handle, 0xc7c00000, gsp_gpu(s)->chip->compute, NULL, 0))
        return -1;
    // 0xc7c0 goes on subchannel 1, subchannel 0 being the 3D class
    uint32_t preamble[] = {
        MTHD(1, C_SET_OBJECT, 1), gsp_gpu(s)->chip->compute,
        MTHD(1, C_SHARED_WINDOW_A, 2), (uint32_t)(SHARED_MEMORY_WINDOW >> 32), (uint32_t)SHARED_MEMORY_WINDOW,
        MTHD(1, C_LOCAL_WINDOW_A, 2), (uint32_t)(LOCAL_MEMORY_WINDOW >> 32), (uint32_t)LOCAL_MEMORY_WINDOW,
        MTHD(1, C_INVALIDATE_SKED_CACHES, 1), 0,
        MTHD(1, C_INVALIDATE_SAMPLER_NO_WFI, 1), 0,
        MTHD(1, C_INVALIDATE_TEXHDR_NO_WFI, 1), 0,
    };
    chan_submit(ch, preamble, sizeof(preamble) / 4);
    return 0;
}

