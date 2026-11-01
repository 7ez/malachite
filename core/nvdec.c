#include "nvdec.h"

#define ALIGN(x, a) (((x) + (a) - 1) / (a) * (a))

// an end of sequence NAL unit twice, which NVDEC puts after the slices
static const uint8_t eos[16] = { 0, 0, 1, 0x0b, 0, 0, 0, 0, 0, 0, 1, 0x0b };

void nvdec_h264_surface(uint32_t width_mbs, uint32_t height_mbs, struct nvdec_surface *surf)
{
    surf->pitch = ALIGN(width_mbs * 16, 64);
    surf->luma_rows = height_mbs * 16;
    surf->chroma_rows = ALIGN(height_mbs * 8, 16);
    surf->chroma_offset = ALIGN((uint64_t)surf->pitch * surf->luma_rows, 0x1000);
    surf->size = surf->chroma_offset + (uint64_t)surf->pitch * surf->chroma_rows;
}

void nvdec_h264_sizes(uint32_t width_mbs, uint32_t height_mbs, struct nvdec_h264_sizes *sizes)
{
    sizes->coloc = ALIGN(ALIGN(height_mbs, 2) * width_mbs * 64 - 63, 0x100);
    sizes->mbhist = ALIGN(width_mbs * 104, 0x100);
    sizes->history = ALIGN(width_mbs * 0x300, 0x200);
}

void nvdec_h264_setup(const struct h264_dec *d, const struct h264_picture *pic, const struct nvdec_surface *surf,
                      struct nvdec_h264_pic *out)
{
    const struct h264_sps *s = pic->sps;
    const struct h264_pps *p = pic->pps;
    struct nvdec_h264_sizes sizes;
    nvdec_h264_sizes(s->width_mbs, s->height_mbs, &sizes);

    *out = (struct nvdec_h264_pic){
        .explicit_eos = 1,
        .stream_len = pic->bytes + sizeof(eos),
        .slice_count = pic->nslices,
        .mbhist_buffer_size = sizes.mbhist,
        // with the timer on, a timeout of 0 never fires and a stuck decode hangs the engine (nvk)
        .gptimer_timeout = 81000000,
        .log2_max_poc_lsb_minus4 = s->poc_type == 0 ? (int32_t)s->log2_max_poc_lsb - 4 : 0,
        .delta_pic_order_always_zero = s->delta_pic_order_always_zero,
        .frame_mbs_only = s->frame_mbs_only,
        .width_mbs = (int32_t)s->width_mbs,
        .height_mbs = (int32_t)s->height_mbs,
        .tile_format = 1,
        .gob_height = 0,
        .cabac = p->cabac,
        .pic_order_present = p->bottom_field_pic_order,
        .num_ref_idx_l0_minus1 = (int32_t)p->num_ref_idx_l0,
        .num_ref_idx_l1_minus1 = (int32_t)p->num_ref_idx_l1,
        .deblocking_control = p->deblocking_control,
        .redundant_pic_cnt = p->redundant_pic_cnt,
        .transform_8x8 = p->transform_8x8,
        .pitch_luma = surf->pitch,
        .pitch_chroma = surf->pitch,
        .luma_bot_offset = s->width_mbs * 16,
        .chroma_bot_offset = surf->pitch / 2,
        .hist_buffer_size = sizes.history >> 8,
        .direct_8x8_inference = (uint32_t)s->direct_8x8_inference,
        .weighted_pred = (uint32_t)p->weighted_pred,
        .constrained_intra_pred = (uint32_t)p->constrained_intra_pred,
        .ref_pic = pic->hdr.nal_ref_idc != 0,
        .log2_max_frame_num_minus4 = s->log2_max_frame_num - 4,
        .chroma_format_idc = s->chroma_format,
        .poc_type = s->poc_type,
        .pic_init_qp_minus26 = p->pic_init_qp,
        .chroma_qp_index_offset = p->chroma_qp_index_offset,
        .second_chroma_qp_index_offset = p->second_chroma_qp_index_offset,
        .weighted_bipred_idc = p->weighted_bipred_idc,
        .curr_pic_idx = pic->slot,
        .curr_col_idx = pic->slot,
        .frame_num = pic->hdr.frame_num,
        .curr_field_order_cnt = { pic->poc[0], pic->poc[1] },
        .transform_bypass = (uint32_t)s->transform_bypass,
    };
    memcpy(out->eos, eos, sizeof(eos));
    memset(out->weight_scale, 16, sizeof(out->weight_scale));
    memset(out->weight_scale_8x8, 16, sizeof(out->weight_scale_8x8));

    for (int i = 0; i < H264_DPB; i++) {
        const struct h264_ref *r = &d->dpb[i];
        if (!r->used)
            continue;
        uint32_t slot = r->non_existing ? pic->slot : r->slot, marking = r->long_term ? 2 : 1;
        out->dpb[i] = (struct nvdec_dpb_entry){
            .index = slot,
            .col_idx = slot,
            .state = 3,
            .is_long_term = (uint32_t)r->long_term,
            .not_existing = (uint32_t)r->non_existing,
            .top_field_marking = marking,
            .bottom_field_marking = marking,
            .field_order_cnt = { r->poc[0], r->poc[1] },
            .frame_idx = (int32_t)(r->long_term ? r->long_term_frame_idx : r->frame_num),
        };
    }
}

uint32_t nvdec_crop_width(const struct h264_sps *s)
{
    return s->width_mbs * 16 - s->crop_left - s->crop_right;
}

uint32_t nvdec_crop_height(const struct h264_sps *s)
{
    return s->height_mbs * 16 - s->crop_top - s->crop_bottom;
}

// where byte x of row y lies: two-GOB blocks, each GOB 64 bytes by 8 rows of 16-byte sectors
static size_t bl_offset(uint32_t pitch, uint32_t x, uint32_t y)
{
    size_t block = (size_t)(y / 16) * (pitch / 64) + x / 64;
    return block * 1024 + y % 16 / 8 * 512 + x % 64 / 32 * 256 + y % 8 / 4 * 128 + x % 32 / 16 * 64 + y % 4 * 16 +
           x % 16;
}

static void detile(const uint8_t *src, uint32_t pitch, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                   uint8_t *out, size_t out_pitch)
{
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w;) {
            uint32_t sx = x0 + x, run = 16 - sx % 16;
            if (run > w - x)
                run = w - x;
            memcpy(out + y * out_pitch + x, src + bl_offset(pitch, sx, y0 + y), run);
            x += run;
        }
    }
}

void nvdec_detile_nv12(const uint8_t *surface, const struct nvdec_surface *surf, const struct h264_sps *s,
                       uint8_t *luma, size_t luma_pitch, uint8_t *chroma, size_t chroma_pitch)
{
    uint32_t w = nvdec_crop_width(s), h = nvdec_crop_height(s);
    detile(surface, surf->pitch, s->crop_left, s->crop_top, w, h, luma, luma_pitch);
    detile(surface + surf->chroma_offset, surf->pitch, s->crop_left, s->crop_top / 2, w, h / 2, chroma, chroma_pitch);
}
