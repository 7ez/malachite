#pragma once
#include "plat.h"

struct fwsec_desc_v3 {
    uint32_t hdr, stored_size, pkc_data_offset, interface_offset;
    uint32_t imem_phys_base, imem_load_size, imem_virt_base;
    uint32_t dmem_phys_base, dmem_load_size;
    uint16_t engine_id_mask;
    uint8_t ucode_id, signature_count;
    uint16_t signature_versions, reserved;
} __attribute__((packed));

struct fwsec_ucode {
    struct fwsec_desc_v3 desc;
    uint8_t *sigs;  // signature_count RSA-3K signatures of 384 bytes
    uint8_t *image;  // IMEM then DMEM
    uint32_t image_size;
};

int vbios_fwsec(struct gpu *g, struct fwsec_ucode *fw);
void vbios_fwsec_free(struct fwsec_ucode *fw);
