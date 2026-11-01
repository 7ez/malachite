// FWSEC-FRTS and FWSEC-SB from the VBIOS on the GSP falcon, following nova-core (firmware/fwsec.rs)
#include "falcon.h"
#include "fwsec.h"
#include "vbios.h"

static int patch_cmd(uint8_t *img, const struct fwsec_desc_v3 *d, uint32_t cmd, uint64_t frts_addr, uint64_t frts_size)
{
    uint8_t *hdr = img + d->imem_load_size + d->interface_offset;
    if (hdr[0] != 1) {
        log_msg("fwsec: appif header version %d\n", hdr[0]);
        return -1;
    }
    for (int i = 0; i < hdr[3]; i++) {
        uint32_t *app = (uint32_t *)(hdr + hdr[1] + i * hdr[2]);
        if (app[0] != 4)  // DMEMMAPPER
            continue;
        uint32_t *mapper = (uint32_t *)(img + d->imem_load_size + app[1]);
        mapper[11] = cmd;
        struct {
            uint32_t ver, hdr;
            uint64_t addr;
            uint32_t size, flags;
            uint32_t rver, rhdr, raddr, rsize, rtype;
        } __attribute__((packed)) frts = {
            1, 24, 0, 0, 2,
            1, 20, (uint32_t)(frts_addr >> 12), (uint32_t)(frts_size >> 12), 2,
        };
        // FWSEC-SB only takes the read_vbios part
        memcpy(img + d->imem_load_size + mapper[2], &frts, cmd == FWSEC_FRTS ? sizeof(frts) : 24);
        return 0;
    }
    log_msg("fwsec: no DMEM mapper interface\n");
    return -1;
}

int fwsec_run(struct gpu *g, uint32_t cmd, uint64_t frts_addr, uint64_t frts_size)
{
    if (cmd == FWSEC_FRTS && rd32(g, 0x1fa828)) {
        log_msg("fwsec: WPR2 already set, the GPU needs a reset\n");
        return -1;
    }

    struct fwsec_ucode fw;
    if (vbios_fwsec(g, &fw))
        return -1;
    const struct fwsec_desc_v3 *d = &fw.desc;
    int ret = -1;
    struct dma buf = { 0 };
    if (patch_cmd(fw.image, d, cmd, frts_addr, frts_size))
        goto out;

    int fuse = fuse_version(g, d->engine_id_mask, d->ucode_id);
    uint32_t bit = 1u << fuse;
    if (!(d->signature_versions & bit)) {
        log_msg("fwsec: no signature for fuse version %d (have %#x)\n", fuse, d->signature_versions);
        goto out;
    }
    int idx = __builtin_popcount(d->signature_versions & (bit - 1));
    memcpy(fw.image + d->imem_load_size + d->pkc_data_offset, fw.sigs + idx * 384, 384);

    if (dma_alloc(g, &buf, fw.image_size, 1))
        goto out;
    memcpy(buf.cpu, fw.image, fw.image_size);

    struct falcon_load l = {
        .iova = buf.addr,
        .imem_src = 0, .imem_dst = d->imem_phys_base, .imem_len = d->imem_load_size,
        .dmem_src = d->imem_load_size, .dmem_dst = d->dmem_phys_base, .dmem_len = d->dmem_load_size,
        .pkc_offset = d->pkc_data_offset, .engine_mask = d->engine_id_mask, .ucode_id = d->ucode_id,
        .boot_addr = 0,
    };
    uint32_t mbox0 = 0, mbox1 = 0;
    if (falcon_reset(g, &gsp_falcon) || falcon_load(g, &gsp_falcon, &l) || falcon_boot(g, &gsp_falcon, &mbox0, &mbox1))
        goto out;

    if (cmd == FWSEC_SB) {
        uint32_t err = rd32(g, 0x1400 + 0x15 * 4) & 0xffff;
        log_msg("fwsec-sb: mailbox0 %#x, error %#x\n", mbox0, err);
        ret = mbox0 || err ? -1 : 0;
        goto out;
    }

    uint32_t err = rd32(g, 0x1400 + 0xe * 4) >> 16;
    uint64_t lo = (uint64_t)(rd32(g, 0x1fa824) >> 4) << 12;
    uint64_t hi = (uint64_t)(rd32(g, 0x1fa828) >> 4) << 12;
    log_msg("fwsec-frts: signature %d (fuse %d), mailbox0 %#x, error %#x, WPR2 %#llx..%#llx\n",
            idx, fuse, mbox0, err, (unsigned long long)lo, (unsigned long long)hi);
    ret = mbox0 || err || !hi || lo != frts_addr ? -1 : 0;
out:
    if (buf.cpu)
        dma_free(g, &buf);
    vbios_fwsec_free(&fw);
    return ret;
}
