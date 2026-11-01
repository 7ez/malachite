// FWSEC from the VBIOS in the PROM window, laid out as in nova-core's vbios.rs and
// Documentation/gpu/nova/core/vbios.rst: PCI-AT image (BIT falcon data pointer), EFI, then FwSec images
#include "vbios.h"

#define PROM 0x300000
#define PROM_SIZE 0x100000

#define TYPE_PCI_AT 0x00
#define TYPE_FWSEC 0xe0
#define TYPE_NBSI 0x70
#define BIT_FALCON_DATA 0x70
#define APPID_FWSEC_PROD 0x85

static uint16_t u16at(const uint8_t *p) { return p[0] | p[1] << 8; }
static uint32_t u32at(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

static uint8_t *read_prom(struct gpu *g)
{
    uint8_t *rom = mem_alloc(PROM_SIZE);
    // with the PCI ROM shadow off (config mirror 0x088000 + 0x50) the whole flash shows at 0x300000
    uint32_t shadow = rd32(g, 0x088050);
    wr32(g, 0x088050, shadow & ~1u);
    for (uint32_t i = 0; i < PROM_SIZE; i += 4) {
        uint32_t v = rd32(g, PROM + i);
        memcpy(rom + i, &v, 4);
    }
    wr32(g, 0x088050, shadow);
    return rom;
}

static uint32_t rom_offset(const uint8_t *prom)
{
    if (u16at(prom) == 0xaa55)
        return 0;
    uint32_t fixed1 = u32at(prom + 4), fixed2 = u32at(prom + 8);
    if (u32at(prom) != 0x4947564e)  // "NVGI"
        return ~0u;
    uint32_t ver = (fixed1 >> 8) & 0xff;
    if (ver == 1 || ver == 2)
        return u32at(prom + ((fixed1 >> 16) & 0x7fff) + 4);
    uint32_t dir = u32at(prom + (fixed2 & 0xfffff)) + 4096;
    return memcmp(prom + dir, "RFRD", 4) ? ~0u : u32at(prom + dir + 8);
}

static int bit_token(const uint8_t *img, uint32_t len, uint8_t id, uint16_t *data_off)
{
    static const uint8_t sig[] = { 0xff, 0xb8, 'B', 'I', 'T', 0 };
    for (uint32_t bit = 0; bit + 12 < len; bit++) {
        if (memcmp(img + bit, sig, sizeof(sig)))
            continue;
        uint8_t hdr = img[bit + 8], size = img[bit + 9], count = img[bit + 10];
        for (uint32_t i = 0; i < count; i++) {
            const uint8_t *t = img + bit + hdr + i * size;
            if (t[0] == id) {
                *data_off = u16at(t + 4);
                return 0;
            }
        }
        return -1;
    }
    return -1;
}

int vbios_fwsec(struct gpu *g, struct fwsec_ucode *fw)
{
    uint8_t *prom = read_prom(g);
    int ret = -1;
    uint32_t base = rom_offset(prom);
    if (base >= PROM_SIZE) {
        log_msg("vbios: no PCI ROM in the PROM\n");
        goto out;
    }

    // PCI-AT image, then everything from the first FwSec image on as one block
    const uint8_t *rom = prom + base, *pci_at = NULL, *fwsec = NULL;
    uint32_t pci_at_len = 0, off = 0, end = PROM_SIZE - base;
    while (off + 0x1a < end) {
        uint16_t sig = u16at(rom + off);
        if (sig != 0xaa55 && sig != 0xbb77 && sig != 0x4e56)
            break;
        const uint8_t *pcir = rom + off + u16at(rom + off + 0x18);
        uint32_t len = u16at(pcir + 0x10) * 512;
        uint8_t type = pcir[0x14];
        int last = pcir[0x15] & 0x80;
        if (type != TYPE_NBSI) {
            uint32_t npde = off + ((u16at(rom + off + 0x18) + u16at(pcir + 0x0a) + 0xf) & ~0xfu);
            if (!memcmp(rom + npde, "NPDE", 4)) {
                len = u16at(rom + npde + 8) * 512;
                last = rom[npde + 10] & 0x80;
            }
        } else {
            last = 1;
        }
        if (type == TYPE_PCI_AT && !pci_at) {
            pci_at = rom + off;
            pci_at_len = len;
        } else if (type == TYPE_FWSEC && !fwsec) {
            fwsec = rom + off;
        }
        if (last || !len)
            break;
        off += len;
    }

    uint16_t tok;
    if (!pci_at || !fwsec || bit_token(pci_at, pci_at_len, BIT_FALCON_DATA, &tok)) {
        log_msg("vbios: no PCI-AT image, FwSec image or falcon data token\n");
        goto out;
    }
    // the falcon data pointer treats PCI-AT and FwSec as contiguous, skipping the EFI image between them
    const uint8_t *table = fwsec + (u32at(pci_at + tok) - pci_at_len);
    uint8_t hdr_len = table[1], ent_len = table[2], count = table[3];
    uint32_t ucode = 0;
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t *e = table + hdr_len + i * ent_len;
        if (e[0] == APPID_FWSEC_PROD)
            ucode = u32at(e + 2);
    }
    if (!ucode) {
        log_msg("vbios: no FWSEC-PROD entry\n");
        goto out;
    }

    const uint8_t *desc = fwsec + (ucode - pci_at_len);
    uint32_t hdr = u32at(desc), vers = (hdr >> 8) & 0xff, desc_size = hdr >> 16;
    if (vers != 3) {
        log_msg("vbios: FWSEC descriptor v%u, only v3 is handled\n", vers);
        goto out;
    }
    memcpy(&fw->desc, desc, sizeof(fw->desc));
    fw->sigs = mem_alloc(fw->desc.signature_count * 384);
    memcpy(fw->sigs, desc + sizeof(fw->desc), fw->desc.signature_count * 384);
    fw->image_size = fw->desc.imem_load_size + fw->desc.dmem_load_size;
    fw->image = mem_alloc(fw->image_size);
    memcpy(fw->image, desc + desc_size, fw->image_size);
    ret = 0;
out:
    mem_free(prom);
    return ret;
}

void vbios_fwsec_free(struct fwsec_ucode *fw)
{
    mem_free(fw->sigs);
    mem_free(fw->image);
}
