// The CPU side of NVIDIA's resource manager, as far as NVK's NVRM backend needs it: the NVRM escape ABI
// (open-gpu-kernel-modules 610 headers, what NVK is built against) over the GSP-RM core.
// Objects GSP-RM owns are forwarded under the caller's handles; memory, VA spaces and usermode live here.
#include "nvos.h"
#include "nvstatus.h"
#include "nv_escape.h"
#include "nv-ioctl.h"
#include "nv-unix-nvos-params-wrappers.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "class/cl90f1.h"
#include "class/cl0040.h"
#include "class/cl003e.h"
#include "class/cl50a0.h"
#include "class/clc461.h"
#include "alloc/alloc_channel.h"
#include "class/cl0002.h"
#include "class/cl0005.h"
#include "class/cl9067.h"
#include "class/cla06c.h"
#include "class/clb0b5sw.h"
#include "class/clc56f.h"

#include "../gsp.h"
#include "../mem.h"
#include "rmfe.h"
#include "rmfe_priv.h"

#define CLIENT_BASE 0xc1d00000u
#define USERMODE_BAR0 0xbb0000  // GA10x usermode block with GSP-RM

struct rmfe *rmfe_create(struct gsp *s)
{
    struct rmfe *fe = mem_alloc(sizeof(*fe));
    fe->s = s;
    gsp_on_rc(s, rmfe_rc_event, fe);
    return fe;
}

void rmfe_destroy(struct rmfe *fe)
{
    while (fe->files)
        rmfe_close(fe->files);
    gsp_off_rc(fe->s, fe);
    if (fe->bar1_up)
        bar1_fini(&fe->bar1);
    mem_free(fe);
}

struct rmfe_file *rmfe_open(struct rmfe *fe, void *task)
{
    struct rmfe_file *f = mem_alloc(sizeof(*f));
    f->fe = fe;
    f->task = task;
    f->next = fe->files;
    fe->files = f;
    return f;
}

struct client *client_find(struct rmfe *fe, uint32_t handle)
{
    for (struct client *c = fe->clients; c; c = c->next)
        if (c->handle == handle)
            return c;
    return NULL;
}

struct obj *obj_find(struct client *c, uint32_t handle)
{
    for (struct obj *o = c->objs; o; o = o->next)
        if (o->handle == handle)
            return o;
    return NULL;
}

struct obj *obj_add(struct client *c, uint32_t handle, uint32_t parent, uint32_t cls, enum kind kind)
{
    struct obj *o = mem_alloc(sizeof(*o));
    *o = (struct obj){ .handle = handle, .parent = parent, .cls = cls, .kind = kind, .next = c->objs };
    c->objs = o;
    return o;
}

static void obj_release(struct rmfe *fe, struct client *c, struct obj *o)
{
    switch (o->kind) {
    case K_VASPACE:
        vmm_destroy(o->vmm);
        mem_free(o->vmm);
        break;
    case K_VRAM:
    case K_SYSMEM:
    case K_VIRTUAL:
        rmfe_release_memory(fe, o);
        break;
    case K_CHAN:
        rmfe_release_chan(fe, o);
        break;
    default:
        break;
    }
}

// drops o and everything below it; GSP-RM frees its own objects when the topmost one goes
static void obj_free_tree(struct rmfe *fe, struct client *c, uint32_t handle)
{
    for (int again = 1; again;) {
        again = 0;
        for (struct obj *o = c->objs; o; o = o->next) {
            if (o->parent == handle && o->handle != handle) {
                obj_free_tree(fe, c, o->handle);
                again = 1;
                break;
            }
        }
    }
    for (struct obj **pp = &c->objs; *pp; pp = &(*pp)->next) {
        if ((*pp)->handle == handle) {
            struct obj *o = *pp;
            *pp = o->next;
            obj_release(fe, c, o);
            mem_free(o);
            return;
        }
    }
}

static void client_free(struct rmfe *fe, struct client *c)
{
    rm_free(fe->s, c->handle, c->handle);
    while (c->objs)
        obj_free_tree(fe, c, c->objs->handle);
    for (struct client **pp = &fe->clients; *pp; pp = &(*pp)->next) {
        if (*pp == c) {
            *pp = c->next;
            break;
        }
    }
    mem_free(c);
}

void rmfe_close(struct rmfe_file *f)
{
    struct rmfe *fe = f->fe;
    for (struct client *c = fe->clients, *next; c; c = next) {
        next = c->next;
        if (c->owner == f)
            client_free(fe, c);
    }
    for (struct rmfe_file **pp = &fe->files; *pp; pp = &(*pp)->next) {
        if (*pp == f) {
            *pp = f->next;
            break;
        }
    }
    mem_free(f);
}

static uint32_t new_handle(struct client *c)
{
    return 0xcaf00000u | ++c->next_handle;
}

// alloc parameter sizes for the classes passed through, as RM's class tables define them
static uint32_t params_size(uint32_t cls)
{
    switch (cls) {
    case NV01_ROOT_CLIENT: return sizeof(NV0000_ALLOC_PARAMETERS);
    case NV01_DEVICE_0: return sizeof(NV0080_ALLOC_PARAMETERS);
    case NV20_SUBDEVICE_0: return sizeof(NV2080_ALLOC_PARAMETERS);
    case FERMI_VASPACE_A: return sizeof(NV_VASPACE_ALLOCATION_PARAMETERS);
    case NV01_MEMORY_LOCAL_USER:
    case NV01_MEMORY_SYSTEM:
    case NV50_MEMORY_VIRTUAL: return sizeof(NV_MEMORY_ALLOCATION_PARAMS);
    case NV01_CONTEXT_DMA: return sizeof(NV_CONTEXT_DMA_ALLOCATION_PARAMS);
    case NV01_EVENT_OS_EVENT: return sizeof(NV0005_ALLOC_PARAMETERS);
    case KEPLER_CHANNEL_GROUP_A: return sizeof(NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS);
    case FERMI_CONTEXT_SHARE_A: return sizeof(NV_CTXSHARE_ALLOCATION_PARAMETERS);
    case AMPERE_CHANNEL_GPFIFO_A: return sizeof(NV_CHANNEL_ALLOC_PARAMS);
    case 0xc7b5: return sizeof(NVB0B5_ALLOCATION_PARAMETERS);  // AMPERE_DMA_COPY_B
    default: return 0;
    }
}

static NV_STATUS alloc_root(struct rmfe_file *f, NVOS21_PARAMETERS *p)
{
    struct rmfe *fe = f->fe;
    uint32_t handle = p->hObjectNew ? p->hObjectNew : CLIENT_BASE + ++fe->next_client;
    if (client_find(fe, handle))
        return NV_ERR_INSERT_DUPLICATE_NAME;
    NV0000_ALLOC_PARAMETERS root = { .hClient = handle, .processID = ~0u };
    if (rm_alloc(fe->s, handle, handle, handle, NV01_ROOT, &root, sizeof(root)))
        return NV_ERR_OPERATING_SYSTEM;
    struct client *c = mem_alloc(sizeof(*c));
    *c = (struct client){ .handle = handle, .owner = f, .next = fe->clients };
    fe->clients = c;
    p->hObjectNew = handle;
    return NV_OK;
}

static NV_STATUS alloc_vaspace(struct rmfe *fe, struct client *c, struct obj *dev, uint32_t handle, const void *params)
{
    const NV_VASPACE_ALLOCATION_PARAMETERS *vas = params;
    struct obj *sub = NULL;
    for (struct obj *o = c->objs; o; o = o->next)
        if (o->cls == NV20_SUBDEVICE_0 && o->parent == dev->handle)
            sub = o;
    struct rm_client rc = { c->handle, dev->handle, sub ? sub->handle : 0 };
    struct vmm *v = mem_alloc(sizeof(*v));
    if (vmm_create(fe->s, &rc, handle, vas ? vas->vaBase : 0, v)) {
        vmm_destroy(v);
        mem_free(v);
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }
    obj_add(c, handle, dev->handle, FERMI_VASPACE_A, K_VASPACE)->vmm = v;
    return NV_OK;
}

static struct vmm *vaspace_vmm(struct client *c, uint32_t handle)
{
    struct obj *o = obj_find(c, handle);
    return o && o->kind == K_VASPACE ? o->vmm : NULL;
}

static NV_STATUS rm_alloc_escape(struct rmfe_file *f, NVOS21_PARAMETERS *p)
{
    struct rmfe *fe = f->fe;
    if (p->hClass == NV01_ROOT_CLIENT || p->hClass == NV01_ROOT)
        return alloc_root(f, p);

    struct client *c = client_find(fe, p->hRoot);
    if (!c)
        return NV_ERR_INVALID_CLIENT;
    struct obj *parent = p->hObjectParent == c->handle ? NULL : obj_find(c, p->hObjectParent);
    if (p->hObjectParent != c->handle && !parent)
        return NV_ERR_INVALID_OBJECT_PARENT;
    uint32_t handle = p->hObjectNew ? p->hObjectNew : new_handle(c);
    if (obj_find(c, handle))
        return NV_ERR_INSERT_DUPLICATE_NAME;

    uint32_t size = p->paramsSize ? p->paramsSize : params_size(p->hClass);
    uint8_t *params = NULL;
    if (p->pAllocParms && size) {
        params = mem_alloc(size);
        if (copy_from_user(f->task, params, (uintptr_t)p->pAllocParms, size)) {
            mem_free(params);
            return NV_ERR_INVALID_ADDRESS;
        }
    }

    NV_STATUS st = NV_OK;
    switch (p->hClass) {
    case FERMI_VASPACE_A:
        st = parent && parent->cls == NV01_DEVICE_0 ? alloc_vaspace(fe, c, parent, handle, params) : NV_ERR_INVALID_OBJECT_PARENT;
        break;
    case TURING_USERMODE_A:
        obj_add(c, handle, p->hObjectParent, p->hClass, K_USERMODE);
        break;
    case NV01_EVENT_OS_EVENT:
        obj_add(c, handle, p->hObjectParent, p->hClass, K_EVENT);
        break;
    case NV01_MEMORY_LOCAL_USER:
    case NV01_MEMORY_SYSTEM:
    case NV50_MEMORY_VIRTUAL:
        st = params ? rmfe_alloc_memory(fe, c, p->hObjectParent, handle, p->hClass, params) : NV_ERR_INVALID_ARGUMENT;
        // these hand results back in their parameters
        if (st == NV_OK && copy_to_user(f->task, (uintptr_t)p->pAllocParms, params, size))
            st = NV_ERR_INVALID_ADDRESS;
        break;
    case NV01_CONTEXT_DMA: {
        const NV_CONTEXT_DMA_ALLOCATION_PARAMS *cd = (const void *)params;
        if (!cd || !obj_find(c, cd->hMemory)) {
            st = NV_ERR_INVALID_OBJECT_HANDLE;
            break;
        }
        struct obj *o = obj_add(c, handle, p->hObjectParent, p->hClass, K_CTXDMA);
        o->target = cd->hMemory;
        o->addr = cd->offset;
        o->size = cd->limit + 1;
        break;
    }
    case AMPERE_CHANNEL_GPFIFO_A:
        st = params ? rmfe_alloc_chan(fe, c, parent, handle, p->hClass, params) : NV_ERR_INVALID_ARGUMENT;
        break;
    default:
        if (parent && parent->kind == K_CHAN) {
            st = rmfe_alloc_engine(fe, c, parent, handle, p->hClass, params, size);
            break;
        }
        if (rm_alloc(fe->s, c->handle, p->hObjectParent, handle, p->hClass, params, params ? size : 0)) {
            st = NV_ERR_INVALID_CLASS;
            break;
        }
        struct obj *o = obj_add(c, handle, p->hObjectParent, p->hClass, K_GSP);
        // channels in a group or subcontext run in the group's VA space
        if (params && p->hClass == KEPLER_CHANNEL_GROUP_A)
            o->vmm = vaspace_vmm(c, ((NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS *)params)->hVASpace);
        if (params && p->hClass == FERMI_CONTEXT_SHARE_A)
            o->vmm = vaspace_vmm(c, ((NV_CTXSHARE_ALLOCATION_PARAMETERS *)params)->hVASpace);
        if (p->hClass == FERMI_CONTEXT_SHARE_A && !o->vmm && parent)
            o->vmm = parent->vmm;
        break;
    }
    mem_free(params);
    if (st == NV_OK)
        p->hObjectNew = handle;
    return st;
}

static NV_STATUS rm_free_escape(struct rmfe_file *f, NVOS00_PARAMETERS *p)
{
    struct client *c = client_find(f->fe, p->hRoot);
    if (!c)
        return NV_ERR_INVALID_CLIENT;
    if (p->hObjectOld == c->handle) {
        client_free(f->fe, c);
        return NV_OK;
    }
    struct obj *o = obj_find(c, p->hObjectOld);
    if (!o)
        return NV_ERR_INVALID_OBJECT_HANDLE;
    // local objects GSP-RM also knows (the VA space) still need freeing there
    if (o->kind == K_GSP || o->kind == K_VASPACE || o->kind == K_CHAN)
        rm_free(f->fe->s, c->handle, o->handle);
    obj_free_tree(f->fe, c, o->handle);
    return NV_OK;
}

static NV_STATUS rm_control_escape(struct rmfe_file *f, NVOS54_PARAMETERS *p)
{
    struct client *c = client_find(f->fe, p->hClient);
    if (!c)
        return NV_ERR_INVALID_CLIENT;
    uint8_t *params = mem_alloc(p->paramsSize ? p->paramsSize : 1);
    if (p->paramsSize && copy_from_user(f->task, params, (uintptr_t)p->params, p->paramsSize)) {
        mem_free(params);
        return NV_ERR_INVALID_ADDRESS;
    }

    NV_STATUS st;
    struct obj *o = obj_find(c, p->hObject);
    int local = (o && o->kind == K_CHAN && rmfe_chan_control(f->fe, c, o, p->cmd, params, p->paramsSize, &st)) ||
                rmfe_local_control(f->fe, c, p->hObject, p->cmd, params, p->paramsSize, &st);
    if (!local) {
        void *rep = rm_control(f->fe->s, c->handle, p->hObject, p->cmd, params, p->paramsSize);
        st = rep ? NV_OK : NV_ERR_NOT_SUPPORTED;
        if (rep)
            memcpy(params, rep, p->paramsSize);
    }
    if (st == NV_OK && p->paramsSize && copy_to_user(f->task, (uintptr_t)p->params, params, p->paramsSize))
        st = NV_ERR_INVALID_ADDRESS;
    mem_free(params);
    return st;
}

static NV_STATUS card_info(struct rmfe_file *f, void *arg, uint32_t size)
{
    struct gpu *g = gsp_gpu(f->fe->s);
    nv_ioctl_card_info_t *ci = arg;
    if (size < sizeof(*ci))
        return NV_ERR_INVALID_ARGUMENT;
    memset(arg, 0, size);
    uint32_t bdf = pci_bdf(g);
    ci->valid = 1;
    ci->pci_info.bus = bdf >> 8;
    ci->pci_info.slot = (bdf >> 3) & 0x1f;
    ci->pci_info.function = bdf & 7;
    ci->pci_info.vendor_id = 0x10de;
    ci->pci_info.device_id = pci_device_id(g);
    ci->gpu_id = RMFE_GPU_ID;
    ci->reg_address = bar_phys(g, 0);
    ci->reg_size = bar_size(g, 0);
    ci->fb_address = bar_phys(g, 1);
    ci->fb_size = bar_size(g, 1);
    return NV_OK;
}

static NV_STATUS map_memory(struct rmfe_file *f, nv_ioctl_nvos33_parameters_with_fd *w)
{
    NVOS33_PARAMETERS *p = &w->params;
    struct client *c = client_find(f->fe, p->hClient);
    struct obj *o = c ? obj_find(c, p->hMemory) : NULL;
    if (!o)
        return NV_ERR_INVALID_OBJECT_HANDLE;
    f->map_fd = w->fd;
    switch (o->kind) {
    case K_USERMODE:
        f->map = (struct rmfe_map){ .kind = RMFE_MAP_BAR0, .offset = USERMODE_BAR0 + p->offset, .size = p->length };
        return NV_OK;
    case K_VRAM: {
        uint64_t bar1;
        NV_STATUS st = rmfe_map_bar1(f->fe, o, p->offset, p->length, &bar1);
        if (st == NV_OK)
            f->map = (struct rmfe_map){ .kind = RMFE_MAP_BAR1, .offset = bar1, .size = p->length };
        return st;
    }
    case K_SYSMEM:
        if (p->offset + p->length > o->dma.size)
            return NV_ERR_INVALID_LIMIT;
        f->map = (struct rmfe_map){ .kind = RMFE_MAP_SYS, .offset = p->offset, .size = p->length, .sys = &o->dma };
        return NV_OK;
    default:
        return NV_ERR_NOT_SUPPORTED;
    }
}

// the CPU side of the mapping is the process's own; only a BAR1 window needs taking down
static NV_STATUS unmap_memory(struct rmfe_file *f, NVOS34_PARAMETERS *p)
{
    struct client *c = client_find(f->fe, p->hClient);
    struct obj *o = c ? obj_find(c, p->hMemory) : NULL;
    if (!o)
        return NV_ERR_INVALID_OBJECT_HANDLE;
    return o->kind == K_VRAM ? rmfe_unmap_bar1(f->fe, o) : NV_OK;
}

int rmfe_escape(struct rmfe_file *f, uint32_t cmd, void *arg, uint32_t size)
{
    NV_STATUS st;
    rmfe_drain_rc(f->fe);
    switch (cmd) {
    case NV_ESC_CARD_INFO:
        return card_info(f, arg, size) == NV_OK ? 0 : -1;
    case NV_ESC_REGISTER_FD:
        return 0;
    case NV_ESC_ALLOC_OS_EVENT:
    case NV_ESC_FREE_OS_EVENT:
        // the platform's wait (graphics nonstall interrupts) stands behind every OS event
        if (size < sizeof(nv_ioctl_alloc_os_event_t))
            return -1;
        ((nv_ioctl_alloc_os_event_t *)arg)->Status = NV_OK;
        return 0;
    case NV_ESC_RM_ALLOC:
        if (size < sizeof(NVOS21_PARAMETERS))
            return -1;
        st = rm_alloc_escape(f, arg);
        ((NVOS21_PARAMETERS *)arg)->status = st;
        break;
    case NV_ESC_RM_FREE:
        if (size < sizeof(NVOS00_PARAMETERS))
            return -1;
        st = rm_free_escape(f, arg);
        ((NVOS00_PARAMETERS *)arg)->status = st;
        break;
    case NV_ESC_RM_CONTROL:
        if (size < sizeof(NVOS54_PARAMETERS))
            return -1;
        st = rm_control_escape(f, arg);
        ((NVOS54_PARAMETERS *)arg)->status = st;
        break;
    case NV_ESC_RM_MAP_MEMORY:
        if (size < sizeof(nv_ioctl_nvos33_parameters_with_fd))
            return -1;
        st = map_memory(f, arg);
        ((nv_ioctl_nvos33_parameters_with_fd *)arg)->params.status = st;
        break;
    case NV_ESC_RM_UNMAP_MEMORY:
        if (size < sizeof(NVOS34_PARAMETERS))
            return -1;
        st = unmap_memory(f, arg);
        ((NVOS34_PARAMETERS *)arg)->status = st;
        break;
    case NV_ESC_RM_MAP_MEMORY_DMA:
    case NV_ESC_RM_UNMAP_MEMORY_DMA: {
        uint32_t need = cmd == NV_ESC_RM_MAP_MEMORY_DMA ? sizeof(NVOS46_PARAMETERS) : sizeof(NVOS47_PARAMETERS);
        if (size < need)
            return -1;
        NvHandle hClient = cmd == NV_ESC_RM_MAP_MEMORY_DMA ? ((NVOS46_PARAMETERS *)arg)->hClient : ((NVOS47_PARAMETERS *)arg)->hClient;
        struct client *c = client_find(f->fe, hClient);
        st = !c ? NV_ERR_INVALID_CLIENT
             : cmd == NV_ESC_RM_MAP_MEMORY_DMA ? rmfe_map_dma(f->fe, c, arg) : rmfe_unmap_dma(f->fe, c, arg);
        if (cmd == NV_ESC_RM_MAP_MEMORY_DMA)
            ((NVOS46_PARAMETERS *)arg)->status = st;
        else
            ((NVOS47_PARAMETERS *)arg)->status = st;
        break;
    }
    case NV_ESC_RM_ALLOC_MEMORY:
        if (size < sizeof(nv_ioctl_nvos02_parameters_with_fd))
            return -1;
        st = NV_ERR_NOT_SUPPORTED;
        ((nv_ioctl_nvos02_parameters_with_fd *)arg)->params.status = st;
        break;
    default:
        log_msg("rmfe: escape %#x not handled\n", cmd);
        return -1;
    }
    if (st != NV_OK)
        log_msg("rmfe: escape %#x -> status %#x\n", cmd, st);
    return 0;
}

void rmfe_poll(struct rmfe *fe)
{
    gsp_poll(fe->s);
    rmfe_drain_rc(fe);
}

int rmfe_take_map(struct rmfe_file *f, int *fd, struct rmfe_map *out)
{
    if (!f->map.kind)
        return -1;
    *out = f->map;
    *fd = f->map_fd;
    f->map.kind = RMFE_MAP_NONE;
    return 0;
}
