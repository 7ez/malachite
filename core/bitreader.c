#include "bitreader.h"

void bits_init(struct bits *b, const uint8_t *p, size_t len)
{
    *b = (struct bits){ .p = p, .len = len };
    while (len && !p[len - 1])
        len--;
    if (len) {
        int bit = 0;
        while (!(p[len - 1] >> bit & 1))
            bit++;
        b->stop = (uint64_t)len * 8 - 1 - bit;
    }
}

void bits_init_raw(struct bits *b, const uint8_t *p, size_t len)
{
    *b = (struct bits){ .p = p, .len = len, .raw = 1 };
}

static uint32_t bit(struct bits *b)
{
    if (!b->left) {
        if (!b->raw && b->pos < b->len && b->zeros >= 2 && b->p[b->pos] == 3) {
            b->pos++;
            b->skipped++;
            b->zeros = 0;
        }
        if (b->pos >= b->len) {
            b->over = 1;
            return 0;
        }
        b->cur = b->p[b->pos++];
        b->zeros = b->cur ? 0 : b->zeros + 1;
        b->left = 8;
    }
    return b->cur >> --b->left & 1;
}

uint32_t bits_u(struct bits *b, int n)
{
    uint32_t v = 0;
    while (n--)
        v = v << 1 | bit(b);
    return v;
}

uint32_t bits_ue(struct bits *b)
{
    int n = 0;
    while (!bit(b) && n < 32 && !b->over)
        n++;
    if (n >= 32) {
        b->over = 1;
        return 0;
    }
    return (uint32_t)((1ull << n) - 1 + bits_u(b, n));
}

int32_t bits_se(struct bits *b)
{
    uint32_t v = bits_ue(b);
    return v & 1 ? (int32_t)((v + 1) / 2) : -(int32_t)(v / 2);
}

int bits_more_rbsp_data(struct bits *b)
{
    return (uint64_t)b->pos * 8 - b->left < b->stop;
}

uint64_t bits_pos(const struct bits *b)
{
    return ((uint64_t)b->pos - b->skipped) * 8 - b->left;
}

int bits_next_nal(const uint8_t *s, size_t len, size_t *pos, size_t *start, size_t *size)
{
    size_t i = *pos;
    while (i + 3 <= len && !(s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1))
        i++;
    if (i + 3 > len)
        return 0;
    *start = i + 3;
    for (i = *start; i + 3 <= len; i++)
        if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] <= 1)
            break;
    if (i + 3 > len)
        i = len;
    *size = i - *start;
    *pos = i;
    return 1;
}

// A sample is consumed whole, so a position past 0 and short of the end is in an IVF file
int bits_next_ivf(const uint8_t *s, size_t len, size_t *pos, size_t *start, size_t *size)
{
    if (*pos >= len)
        return 0;
    if (*pos == 0 && (len < 32 || memcmp(s, "DKIF", 4))) {
        *start = 0;
        *size = len;
        *pos = len;
        return 1;
    }
    if (*pos == 0)
        *pos = (size_t)(s[6] | s[7] << 8);
    if (*pos >= len)
        return 0;
    if (len - *pos < 12)
        return -1;
    const uint8_t *h = s + *pos;
    size_t n = (size_t)h[0] | (size_t)h[1] << 8 | (size_t)h[2] << 16 | (size_t)h[3] << 24;
    if (n > len - *pos - 12)
        return -1;
    *start = *pos + 12;
    *size = n;
    *pos = *start + n;
    return 1;
}

void bool_init(struct boolreader *r, const uint8_t *p, size_t len)
{
    bits_init_raw(&r->b, p, len);
    r->value = bits_u(&r->b, 8);
    r->range = 255;
}

int bool_read(struct boolreader *r, uint32_t prob)
{
    uint32_t split = 1 + (((r->range - 1) * prob) >> 8);
    int bit = r->value >= split;
    if (bit) {
        r->range -= split;
        r->value -= split;
    } else {
        r->range = split;
    }
    while (r->range < 128) {
        r->value = r->value << 1 | bits_u(&r->b, 1);
        r->range <<= 1;
    }
    return bit;
}

uint32_t bool_literal(struct boolreader *r, int n)
{
    uint32_t v = 0;
    while (n--)
        v = v << 1 | (uint32_t)bool_read(r, 128);
    return v;
}
