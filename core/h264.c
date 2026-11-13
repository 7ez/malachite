// H.264 parameter sets, slice headers, picture order count and reference marking, ITU-T H.264 7.3, 8.2.1, 8.2.5
#include "h264.h"
#include "bitreader.h"

// whether a scaling_list() was anything but the flat default
static int scaling_list(struct bits *b, int size)
{
    int last = 8, next = 8, flat = 1;
    for (int j = 0; j < size; j++) {
        if (next) {
            next = (last + bits_se(b) + 256) % 256;
            if (j == 0 && !next)
                return 0;  // the default list, not flat either
        }
        if ((next ? next : last) != 16)
            flat = 0;
        last = next ? next : last;
    }
    return flat;
}

// scaling matrices present, each list read; -1 for any but flat ones
static int scaling_matrix(struct bits *b, int lists)
{
    int flat = 1;
    for (int i = 0; i < lists; i++)
        if (bits_u(b, 1))
            flat &= scaling_list(b, i < 6 ? 16 : 64);
    return flat ? 0 : -1;
}

static int parse_sps(struct h264_dec *d, struct bits *b)
{
    struct h264_sps s = { .valid = 1, .chroma_format = 1 };
    s.profile = bits_u(b, 8);
    bits_u(b, 8);
    s.level = bits_u(b, 8);
    uint32_t id = bits_ue(b);
    if (id >= H264_MAX_SPS)
        return -1;
    // profiles whose SPS carries chroma format and bit depth
    static const uint8_t high[] = { 100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135 };
    int is_high = 0;
    for (unsigned i = 0; i < sizeof(high); i++)
        is_high |= s.profile == high[i];
    if (is_high) {
        s.chroma_format = bits_ue(b);
        if (s.chroma_format == 3)
            bits_u(b, 1);  // separate_colour_plane_flag
        s.bit_depth_luma = bits_ue(b);
        s.bit_depth_chroma = bits_ue(b);
        s.transform_bypass = bits_u(b, 1);
        if (bits_u(b, 1) && scaling_matrix(b, s.chroma_format != 3 ? 8 : 12)) {
            log_msg("h264: SPS %u has scaling matrices\n", id);
            return -1;
        }
    }
    s.log2_max_frame_num = bits_ue(b) + 4;
    s.poc_type = bits_ue(b);
    if (s.poc_type == 0) {
        s.log2_max_poc_lsb = bits_ue(b) + 4;
    } else if (s.poc_type == 1) {
        s.delta_pic_order_always_zero = bits_u(b, 1);
        s.offset_for_non_ref_pic = bits_se(b);
        s.offset_for_top_to_bottom = bits_se(b);
        s.poc_cycle = bits_ue(b);
        if (s.poc_cycle > 255)
            return -1;
        for (uint32_t i = 0; i < s.poc_cycle; i++)
            s.offset_for_ref_frame[i] = bits_se(b);
    }
    s.max_num_ref_frames = bits_ue(b);
    s.gaps_allowed = bits_u(b, 1);
    s.width_mbs = bits_ue(b) + 1;
    s.height_mbs = bits_ue(b) + 1;
    s.frame_mbs_only = bits_u(b, 1);
    if (!s.frame_mbs_only)
        s.mbaff = bits_u(b, 1);
    s.direct_8x8_inference = bits_u(b, 1);
    if (bits_u(b, 1)) {
        s.crop_left = bits_ue(b) * 2;
        s.crop_right = bits_ue(b) * 2;
        s.crop_top = bits_ue(b) * 2;
        s.crop_bottom = bits_ue(b) * 2;
    }
    if (b->over || s.max_num_ref_frames > H264_DPB || s.log2_max_frame_num > 16 || s.log2_max_poc_lsb > 16 ||
        s.width_mbs > 512 || s.height_mbs > 512 || s.crop_left + s.crop_right >= s.width_mbs * 16 ||
        s.crop_top + s.crop_bottom >= s.height_mbs * 16)
        return -1;
    if (!s.frame_mbs_only || s.chroma_format != 1 || s.bit_depth_luma || s.bit_depth_chroma) {
        log_msg("h264: SPS %u is interlaced, not 4:2:0 or not 8-bit\n", id);
        return -1;
    }
    d->sps[id] = s;
    return 0;
}

static int parse_pps(struct h264_dec *d, struct bits *b)
{
    struct h264_pps p = { .valid = 1 };
    uint32_t id = bits_ue(b);
    p.sps = bits_ue(b);
    if (id >= H264_MAX_PPS || p.sps >= H264_MAX_SPS || !d->sps[p.sps].valid)
        return -1;
    p.cabac = bits_u(b, 1);
    p.bottom_field_pic_order = bits_u(b, 1);
    if (bits_ue(b)) {
        log_msg("h264: PPS %u has slice groups\n", id);
        return -1;
    }
    p.num_ref_idx_l0 = bits_ue(b);
    p.num_ref_idx_l1 = bits_ue(b);
    p.weighted_pred = bits_u(b, 1);
    p.weighted_bipred_idc = bits_u(b, 2);
    p.pic_init_qp = bits_se(b);
    bits_se(b);  // pic_init_qs_minus26
    p.chroma_qp_index_offset = bits_se(b);
    p.deblocking_control = bits_u(b, 1);
    p.constrained_intra_pred = bits_u(b, 1);
    p.redundant_pic_cnt = bits_u(b, 1);
    p.second_chroma_qp_index_offset = p.chroma_qp_index_offset;
    if (bits_more_rbsp_data(b)) {
        p.transform_8x8 = bits_u(b, 1);
        if (bits_u(b, 1) && scaling_matrix(b, 6 + 2 * p.transform_8x8)) {
            log_msg("h264: PPS %u has scaling matrices\n", id);
            return -1;
        }
        p.second_chroma_qp_index_offset = bits_se(b);
    }
    if (b->over || p.num_ref_idx_l0 > 31 || p.num_ref_idx_l1 > 31)
        return -1;
    d->pps[id] = p;
    return 0;
}

static void skip_ref_list_modification(struct bits *b)
{
    if (!bits_u(b, 1))
        return;
    for (int i = 0; i < 100 && !b->over; i++) {
        uint32_t idc = bits_ue(b);
        if (idc == 3)
            return;
        bits_ue(b);
    }
    b->over = 1;
}

static void skip_pred_weight_table(struct bits *b, const struct h264_slice *h, uint32_t l0, uint32_t l1)
{
    bits_ue(b);  // luma_log2_weight_denom
    bits_ue(b);  // chroma_log2_weight_denom, 4:2:0 only
    for (int list = 0; list < (h->type == 1 ? 2 : 1); list++) {
        for (uint32_t i = 0; i <= (list ? l1 : l0); i++) {
            if (bits_u(b, 1)) {
                bits_se(b);
                bits_se(b);
            }
            if (bits_u(b, 1))
                for (int j = 0; j < 4; j++)
                    bits_se(b);
        }
    }
}

// the slice header up to dec_ref_pic_marking(), which is all the driver needs
static int parse_slice(const struct h264_dec *d, struct bits *b, uint32_t nal_type, uint32_t nal_ref_idc,
                       struct h264_slice *h)
{
    *h = (struct h264_slice){ .nal_type = nal_type, .nal_ref_idc = nal_ref_idc };
    h->first_mb = bits_ue(b);
    h->type = bits_ue(b) % 5;
    h->pps = bits_ue(b);
    if (h->pps >= H264_MAX_PPS || !d->pps[h->pps].valid)
        return -1;
    const struct h264_pps *p = &d->pps[h->pps];
    const struct h264_sps *s = &d->sps[p->sps];
    if (!s->valid)
        return -1;
    h->frame_num = bits_u(b, s->log2_max_frame_num);
    if (nal_type == 5)
        h->idr_pic_id = bits_ue(b);
    if (s->poc_type == 0) {
        h->poc_lsb = bits_u(b, s->log2_max_poc_lsb);
        if (p->bottom_field_pic_order)
            h->delta_poc_bottom = bits_se(b);
    }
    if (s->poc_type == 1 && !s->delta_pic_order_always_zero) {
        h->delta_poc[0] = bits_se(b);
        if (p->bottom_field_pic_order)
            h->delta_poc[1] = bits_se(b);
    }
    if (p->redundant_pic_cnt && bits_ue(b)) {
        log_msg("h264: redundant pictures\n");
        return -1;
    }
    if (h->type == 1)
        bits_u(b, 1);  // direct_spatial_mv_pred_flag
    uint32_t l0 = p->num_ref_idx_l0, l1 = p->num_ref_idx_l1;
    if (h->type == 0 || h->type == 1 || h->type == 3) {
        if (bits_u(b, 1)) {
            l0 = bits_ue(b);
            if (h->type == 1)
                l1 = bits_ue(b);
        }
    }
    if (h->type != 2 && h->type != 4) {
        skip_ref_list_modification(b);
        if (h->type == 1)
            skip_ref_list_modification(b);
    }
    if ((p->weighted_pred && (h->type == 0 || h->type == 3)) || (p->weighted_bipred_idc == 1 && h->type == 1))
        skip_pred_weight_table(b, h, l0, l1);
    if (nal_ref_idc) {
        if (nal_type == 5) {
            bits_u(b, 1);  // no_output_of_prior_pics_flag
            h->long_term_reference = bits_u(b, 1);
        } else if ((h->adaptive_marking = bits_u(b, 1))) {
            for (;;) {
                uint32_t op = bits_ue(b);
                if (!op || b->over)
                    break;
                if (h->nmmco == H264_MMCO_MAX || op > 6)
                    return -1;
                struct h264_mmco *m = &h->mmco[h->nmmco++];
                m->op = op;
                if (op == 1 || op == 3)
                    m->pic_num_diff = bits_ue(b) + 1;
                if (op == 2)
                    m->long_term_pic_num = bits_ue(b);
                if (op == 3 || op == 6)
                    m->long_term_frame_idx = bits_ue(b);
                if (op == 4)
                    m->max_long_term_frame_idx = bits_ue(b);
            }
        }
    }
    return b->over || l0 > 31 || l1 > 31 ? -1 : 0;
}

void h264_init(struct h264_dec *d)
{
    *d = (struct h264_dec){ 0 };
}


static uint32_t max_frame_num(const struct h264_sps *s)
{
    return 1u << s->log2_max_frame_num;
}

// 8.2.1, frames only
static void picture_order(struct h264_dec *d, struct h264_picture *pic)
{
    const struct h264_sps *s = pic->sps;
    const struct h264_slice *h = &pic->hdr;
    if (s->poc_type == 0) {
        int32_t prev_msb = d->prev_poc_msb, prev_lsb = d->prev_poc_lsb;
        if (pic->idr)
            prev_msb = prev_lsb = 0;
        int32_t max = 1 << s->log2_max_poc_lsb, lsb = (int32_t)h->poc_lsb, msb = prev_msb;
        if (lsb < prev_lsb && prev_lsb - lsb >= max / 2)
            msb += max;
        else if (lsb > prev_lsb && lsb - prev_lsb > max / 2)
            msb -= max;
        pic->poc[0] = msb + lsb;
        pic->poc[1] = pic->poc[0] + h->delta_poc_bottom;
        if (h->nal_ref_idc) {
            d->prev_poc_msb = msb;
            d->prev_poc_lsb = lsb;
        }
        return;
    }

    int32_t offset = d->prev_mmco5 ? 0 : d->prev_frame_num_offset;
    if (pic->idr)
        offset = 0;
    else if (d->prev_frame_num > h->frame_num)
        offset += (int32_t)max_frame_num(s);
    d->prev_frame_num_offset = offset;

    if (s->poc_type == 2) {
        int32_t poc = pic->idr ? 0 : 2 * (offset + (int32_t)h->frame_num) - !h->nal_ref_idc;
        pic->poc[0] = pic->poc[1] = poc;
        return;
    }
    int32_t abs = s->poc_cycle ? offset + (int32_t)h->frame_num : 0;
    if (!h->nal_ref_idc && abs > 0)
        abs--;
    int32_t expected = 0;
    if (abs > 0) {
        int32_t delta = 0;
        for (uint32_t i = 0; i < s->poc_cycle; i++)
            delta += s->offset_for_ref_frame[i];
        expected = (abs - 1) / (int32_t)s->poc_cycle * delta;
        for (uint32_t i = 0; i <= (uint32_t)(abs - 1) % s->poc_cycle; i++)
            expected += s->offset_for_ref_frame[i];
    }
    if (!h->nal_ref_idc)
        expected += s->offset_for_non_ref_pic;
    pic->poc[0] = expected + h->delta_poc[0];
    pic->poc[1] = pic->poc[0] + s->offset_for_top_to_bottom + h->delta_poc[1];
}

static void frame_num_wrap(struct h264_dec *d, const struct h264_sps *s, uint32_t frame_num)
{
    for (int i = 0; i < H264_DPB; i++) {
        struct h264_ref *r = &d->dpb[i];
        if (r->used && !r->long_term)
            r->frame_num_wrap = r->frame_num > frame_num ? (int32_t)(r->frame_num - max_frame_num(s))
                                                         : (int32_t)r->frame_num;
    }
}

// 8.2.5.3: room for one more reference frame by dropping the oldest short-term one
static void sliding_window(struct h264_dec *d, const struct h264_sps *s)
{
    uint32_t n = 0;
    struct h264_ref *oldest = NULL;
    for (int i = 0; i < H264_DPB; i++) {
        struct h264_ref *r = &d->dpb[i];
        if (!r->used)
            continue;
        n++;
        if (!r->long_term && (!oldest || r->frame_num_wrap < oldest->frame_num_wrap))
            oldest = r;
    }
    if (n >= (s->max_num_ref_frames ? s->max_num_ref_frames : 1) && oldest)
        oldest->used = 0;
}

static struct h264_ref *free_entry(struct h264_dec *d)
{
    for (int i = 0; i < H264_DPB; i++)
        if (!d->dpb[i].used)
            return &d->dpb[i];
    return NULL;
}

// 8.2.5.2: frame_num gaps filled with frames that don't exist, each going through the sliding window
static int fill_gaps(struct h264_dec *d, const struct h264_picture *pic)
{
    const struct h264_sps *s = pic->sps;
    uint32_t max = max_frame_num(s), want = (d->prev_ref_frame_num + 1) % max;
    if (pic->idr || pic->hdr.frame_num == d->prev_ref_frame_num || pic->hdr.frame_num == want)
        return 0;
    if (!s->gaps_allowed) {
        log_msg("h264: frame_num %u after %u\n", pic->hdr.frame_num, d->prev_ref_frame_num);
        return -1;
    }
    for (uint32_t fn = want; fn != pic->hdr.frame_num; fn = (fn + 1) % max) {
        frame_num_wrap(d, s, fn);
        sliding_window(d, s);
        struct h264_ref *r = free_entry(d);
        if (!r)
            return -1;
        *r = (struct h264_ref){ .used = 1, .non_existing = 1, .frame_num = fn, .frame_num_wrap = (int32_t)fn,
                                .slot = H264_SLOTS };
        d->prev_ref_frame_num = fn;
    }
    return 0;
}

static int pick_slot(const struct h264_dec *d, uint32_t *slot)
{
    uint32_t busy = 0;
    for (int i = 0; i < H264_DPB; i++)
        if (d->dpb[i].used && !d->dpb[i].non_existing)
            busy |= 1u << d->dpb[i].slot;
    for (uint32_t i = 0; i < H264_SLOTS; i++) {
        if (!(busy & 1u << i)) {
            *slot = i;
            return 0;
        }
    }
    return -1;
}

static int same_picture(const struct h264_slice *a, const struct h264_slice *b)
{
    return a->frame_num == b->frame_num && a->pps == b->pps && (a->nal_type == 5) == (b->nal_type == 5) &&
           !a->nal_ref_idc == !b->nal_ref_idc && a->poc_lsb == b->poc_lsb && a->idr_pic_id == b->idr_pic_id &&
           a->delta_poc_bottom == b->delta_poc_bottom && a->delta_poc[0] == b->delta_poc[0] &&
           a->delta_poc[1] == b->delta_poc[1];
}

int h264_next(struct h264_dec *d, const uint8_t *s, size_t len, size_t *pos, struct h264_picture *pic)
{
    struct h264_slice *h = mem_alloc(sizeof(*h));
    size_t at = *pos, start, size;
    int ret = 0;
    pic->nslices = 0;
    while (bits_next_nal(s, len, &at, &start, &size)) {
        if (!size)
            continue;
        uint32_t type = s[start] & 0x1f, ref_idc = s[start] >> 5 & 3;
        struct bits b;
        bits_init(&b, s + start + 1, size - 1);
        if (type == 1 || type == 5) {
            if (parse_slice(d, &b, type, ref_idc, h)) {
                ret = -1;
                break;
            }
            if (pic->nslices && (h->first_mb == 0 || !same_picture(h, &pic->hdr)))
                break;  // the next picture's
            if (!d->started && type != 5) {
                *pos = at;
                continue;  // nothing to predict from before the first IDR
            }
            d->started = 1;
            if (!pic->nslices) {
                pic->hdr = *h;
                pic->bytes = 0;
            }
            if (pic->nslices == H264_MAX_SLICES) {
                ret = -1;
                break;
            }
            pic->slice_pos[pic->nslices] = (uint32_t)start;
            pic->slice_len[pic->nslices++] = (uint32_t)size;
            pic->bytes += (uint32_t)size + 3;
        } else if (pic->nslices && (type == 6 || type == 7 || type == 8 || type == 9 || (type >= 14 && type <= 18))) {
            break;  // a new access unit
        } else if (type == 7 && parse_sps(d, &b)) {
            ret = -1;
            break;
        } else if (type == 8 && parse_pps(d, &b)) {
            ret = -1;
            break;
        } else if (type == 2 || type == 3 || type == 4 || type == 20) {
            log_msg("h264: NAL unit type %u isn't supported\n", type);
            ret = -1;
            break;
        }
        *pos = at;
    }
    mem_free(h);
    if (ret || !pic->nslices)
        return ret;

    pic->pps = &d->pps[pic->hdr.pps];
    pic->sps = &d->sps[pic->pps->sps];
    pic->idr = pic->hdr.nal_type == 5;
    if (pic->idr) {
        for (int i = 0; i < H264_DPB; i++)
            d->dpb[i].used = 0;
        d->max_long_term_frame_idx = 0;
        d->prev_ref_frame_num = 0;
        d->idrs++;
    } else if (fill_gaps(d, pic)) {
        return -1;
    }
    picture_order(d, pic);
    frame_num_wrap(d, pic->sps, pic->hdr.frame_num);
    if (pick_slot(d, &pic->slot))
        return -1;
    pic->index = d->pictures++;
    pic->idr_count = d->idrs;
    return 1;
}

static struct h264_ref *short_term(struct h264_dec *d, int32_t pic_num)
{
    for (int i = 0; i < H264_DPB; i++)
        if (d->dpb[i].used && !d->dpb[i].long_term && d->dpb[i].frame_num_wrap == pic_num)
            return &d->dpb[i];
    return NULL;
}

static void drop_long_term(struct h264_dec *d, uint32_t idx, const struct h264_ref *keep)
{
    for (int i = 0; i < H264_DPB; i++)
        if (d->dpb[i].used && d->dpb[i].long_term && d->dpb[i].long_term_frame_idx == idx && &d->dpb[i] != keep)
            d->dpb[i].used = 0;
}

void h264_done(struct h264_dec *d, const struct h264_picture *pic)
{
    const struct h264_slice *h = &pic->hdr;
    int32_t poc[2] = { pic->poc[0], pic->poc[1] };
    int mmco5 = 0, long_term = 0;
    uint32_t long_term_idx = 0;

    if (pic->idr) {
        long_term = h->long_term_reference;
        d->max_long_term_frame_idx = long_term;
    } else if (h->nal_ref_idc && h->adaptive_marking) {
        for (uint32_t i = 0; i < h->nmmco; i++) {
            const struct h264_mmco *m = &h->mmco[i];
            struct h264_ref *r;
            switch (m->op) {
            case 1:
                if ((r = short_term(d, (int32_t)h->frame_num - (int32_t)m->pic_num_diff)))
                    r->used = 0;
                break;
            case 2:
                for (int j = 0; j < H264_DPB; j++)
                    if (d->dpb[j].used && d->dpb[j].long_term &&
                        d->dpb[j].long_term_frame_idx == m->long_term_pic_num)
                        d->dpb[j].used = 0;
                break;
            case 3:
                if ((r = short_term(d, (int32_t)h->frame_num - (int32_t)m->pic_num_diff))) {
                    drop_long_term(d, m->long_term_frame_idx, r);
                    r->long_term = 1;
                    r->long_term_frame_idx = m->long_term_frame_idx;
                }
                break;
            case 4:
                d->max_long_term_frame_idx = m->max_long_term_frame_idx;
                for (int j = 0; j < H264_DPB; j++)
                    if (d->dpb[j].used && d->dpb[j].long_term &&
                        d->dpb[j].long_term_frame_idx >= m->max_long_term_frame_idx)
                        d->dpb[j].used = 0;
                break;
            case 5:
                for (int j = 0; j < H264_DPB; j++)
                    d->dpb[j].used = 0;
                d->max_long_term_frame_idx = 0;
                mmco5 = 1;
                break;
            case 6:
                drop_long_term(d, m->long_term_frame_idx, NULL);
                long_term = 1;
                long_term_idx = m->long_term_frame_idx;
                break;
            }
        }
    } else if (h->nal_ref_idc) {
        sliding_window(d, pic->sps);
    }

    if (mmco5) {
        // the picture counts as frame_num 0 at picture order 0 from here on
        int32_t min = poc[0] < poc[1] ? poc[0] : poc[1];
        poc[0] -= min;
        poc[1] -= min;
        d->prev_poc_msb = 0;
        d->prev_poc_lsb = poc[0];
    }
    d->prev_mmco5 = mmco5;
    d->prev_frame_num = mmco5 ? 0 : h->frame_num;
    if (!h->nal_ref_idc)
        return;
    d->prev_ref_frame_num = mmco5 ? 0 : h->frame_num;

    struct h264_ref *r = free_entry(d);
    if (!r) {  // a stream marking more frames than it said it would
        sliding_window(d, pic->sps);
        r = free_entry(d);
    }
    if (r)
        *r = (struct h264_ref){ .used = 1, .long_term = long_term, .long_term_frame_idx = long_term_idx,
                                .frame_num = mmco5 ? 0 : h->frame_num, .poc = { poc[0], poc[1] },
                                .slot = pic->slot };
}

void h264_slices(const struct h264_picture *pic, const uint8_t *s, uint8_t *out, uint32_t *offsets)
{
    uint32_t at = 0;
    for (uint32_t i = 0; i < pic->nslices; i++) {
        offsets[i] = at;
        out[at++] = 0;
        out[at++] = 0;
        out[at++] = 1;
        memcpy(out + at, s + pic->slice_pos[i], pic->slice_len[i]);
        at += pic->slice_len[i];
    }
    offsets[pic->nslices] = at;
}
