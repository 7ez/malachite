// Booter-Load/Unload on SEC2, following nouveau's ga102_gsp_booter_ctor() and nova-core (firmware/booter.rs)
#include "booter.h"
#include "falcon.h"

struct bin_hdr {
    uint32_t bin_magic, bin_ver, bin_size, header_offset, data_offset, data_size;
};

struct hs_header_v2 {
    uint32_t sig_prod_offset, sig_prod_size, patch_loc, patch_sig;
    uint32_t meta_data_offset, meta_data_size, num_sig, header_offset, header_size;
};

struct hs_load_header_v2 {
    uint32_t os_code_offset, os_code_size, os_data_offset, os_data_size, num_apps;
    struct {
        uint32_t offset, size, data_offset, data_size;
    } app[];
};

int booter_run(struct gpu *g, const char *path, uint64_t arg, uint32_t *mbox0_out)
{
    size_t size;
    const uint8_t *blob = fw_get(g, path, &size);
    if (!blob)
        return -1;
    const struct bin_hdr *hdr = (const void *)blob;
    if (hdr->bin_magic != 0x10de) {
        log_msg("%s: not an nvfw binary\n", path);
        return -1;
    }
    const struct hs_header_v2 *hs = (const void *)(blob + hdr->header_offset);
    const struct hs_load_header_v2 *lh = (const void *)(blob + hs->header_offset);
    const uint32_t *meta = (const void *)(blob + hs->meta_data_offset);
    uint32_t loc = *(const uint32_t *)(blob + hs->patch_loc);
    uint32_t sig = *(const uint32_t *)(blob + hs->patch_sig);
    uint32_t cnt = *(const uint32_t *)(blob + hs->num_sig);
    uint32_t sig_size = hs->sig_prod_size / cnt;
    uint32_t fuse_ver = meta[0], engine = meta[1], ucode_id = meta[2];

    int reg_fuse = fuse_version(g, engine, ucode_id);
    int idx = reg_fuse ? (int)fuse_ver - reg_fuse : (int)cnt - 1;
    if (idx < 0 || idx >= (int)cnt) {
        log_msg("%s: no signature for fuse %d (image fuse %u, %u signatures)\n", path, reg_fuse, fuse_ver, cnt);
        return -1;
    }

    struct dma buf;
    if (dma_alloc(g, &buf, hdr->data_size, 1))
        return -1;
    memcpy(buf.cpu, blob + hdr->data_offset, hdr->data_size);
    memcpy((uint8_t *)buf.cpu + loc, blob + hs->sig_prod_offset + sig + idx * sig_size, sig_size);

    struct falcon_load l = {
        .iova = buf.addr,
        .imem_src = lh->app[0].offset, .imem_dst = 0, .imem_len = lh->app[0].size,
        .dmem_src = lh->os_data_offset, .dmem_dst = 0, .dmem_len = lh->os_data_size,
        .pkc_offset = loc - lh->os_data_offset, .engine_mask = engine, .ucode_id = ucode_id,
        .boot_addr = lh->app[0].offset,
    };
    uint32_t mbox0 = (uint32_t)arg, mbox1 = arg >> 32;
    int ret = falcon_reset(g, &sec2_falcon) || falcon_load(g, &sec2_falcon, &l) ||
              falcon_boot(g, &sec2_falcon, &mbox0, &mbox1);
    dma_free(g, &buf);
    if (ret)
        return -1;

    log_msg("%s: signature %d of %u (fuse %d), mailbox0 %#x mailbox1 %#x\n", path, idx, cnt, reg_fuse, mbox0, mbox1);
    *mbox0_out = mbox0;
    return 0;
}
