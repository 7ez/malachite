// Kernels covering what airc lowers, plus constructs the bitcode reader must parse even when airc can't lower
// them yet (vectors, half, select, threadgroup ids). Compiled with tools/mtl/mtlair.m into the AIR corpus.
#include <metal_stdlib>
using namespace metal;

kernel void copy(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = a[i];
}

kernel void add(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], device const uint *b [[buffer(2)]],
                uint i [[thread_position_in_grid]])
{
    o[i] = a[i] + b[i];
}

kernel void iop(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], device const uint *b [[buffer(2)]],
                uint i [[thread_position_in_grid]])
{
    o[i] = a[i] * b[i] - a[i];
}

kernel void bits(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], device const uint *b [[buffer(2)]],
                 uint i [[thread_position_in_grid]])
{
    o[i] = ((a[i] & b[i]) | (a[i] << 3)) ^ (b[i] >> 2);
}

kernel void fop(device float *o [[buffer(0)]], device const float *a [[buffer(1)]], device const float *b [[buffer(2)]],
                uint i [[thread_position_in_grid]])
{
    o[i] = a[i] * b[i] + b[i];
}

kernel void relu(device float *o [[buffer(0)]], device const float *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = fmax(a[i], 0.0f);
}

kernel void math(device float *o [[buffer(0)]], device const float *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float x = a[i];
    o[i] = exp2(x) + log2(x) + sin(x) * cos(x) + rsqrt(x) + fma(x, x, 1.5f);
}

kernel void imin(device int *o [[buffer(0)]], device const int *a [[buffer(1)]], device const int *b [[buffer(2)]],
                 uint i [[thread_position_in_grid]])
{
    o[i] = max(min(a[i], b[i]), -7);
}

kernel void conv(device float *o [[buffer(0)]], device int *p [[buffer(1)]], device const int *a [[buffer(2)]],
                 uint i [[thread_position_in_grid]])
{
    o[i] = float(a[i]) * 0.5f;
    p[i] = int(o[i] * 3.0f);
}

kernel void itof(device float *o [[buffer(0)]], device const int *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = float(a[i]);
}

kernel void itof_tid(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    o[i] = float(int(i));
}

kernel void itof_add(device float *o [[buffer(0)]], device const int *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = float(a[i] + 1);
}

kernel void ftoi(device int *o [[buffer(0)]], device const float *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = int(a[i]);
}

kernel void guard(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], constant uint &n [[buffer(2)]],
                  uint i [[thread_position_in_grid]])
{
    if (i < n)
        o[i] = a[i];
}

kernel void loop(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], constant uint &n [[buffer(2)]],
                 uint i [[thread_position_in_grid]])
{
    uint acc = 0;
    for (uint j = 0; j < n; j++)
        acc += a[j];
    o[i] = acc;
}

kernel void divloop(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    uint acc = 0;
    uint n = a[i] & 7;  // a different trip count in every lane
    for (uint j = 0; j < n; j++)
        acc += j + i;
    o[i] = acc;
}

kernel void pick(device float *o [[buffer(0)]], device const float *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = a[i] > 1.0f ? a[i] : -a[i];
}

kernel void ipick(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = a[i] < 100 ? a[i] : 77u;
}

kernel void vec4(device float4 *o [[buffer(0)]], device const float4 *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = a[i] * 2.0f + float4(1, 2, 3, 4);
}

// enough live values to need more than 64 registers
kernel void chain(device float4 *o [[buffer(0)]], device const float4 *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float4 s = a[i], x = s;
    for (int k = 0; k < 12; k++)
        x = x * 0.5f + s;
    o[i] = x;
}

struct Params {
    float k, b;
};

kernel void rounding(device float4 *o [[buffer(0)]], device const float *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float x = a[i];
    o[i] = float4(floor(x), ceil(x), round(x), fract(x));
}

kernel void affine(device float *o [[buffer(0)]], constant Params &p [[buffer(1)]], device const float *a [[buffer(2)]],
                   uint i [[thread_position_in_grid]])
{
    o[i] = a[i] * p.k + p.b;
}

kernel void swizzle(device float4 *o [[buffer(0)]], device const float4 *a [[buffer(1)]],
                    device const float4 *b [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    o[i] = a[i].wzyx * b[i].x + float4(a[i].xy, b[i].zw);
}

kernel void vec3(device float3 *o [[buffer(0)]], device const float3 *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = a[i] * 0.5f + a[i].zxy;
}

kernel void uvec(device uint4 *o [[buffer(0)]], device const uint4 *a [[buffer(1)]], device const uint4 *b [[buffer(2)]],
                 uint i [[thread_position_in_grid]])
{
    o[i] = (a[i] ^ b[i]) + uint4(i);
}

kernel void dot4(device float *o [[buffer(0)]], device const float4 *a [[buffer(1)]], device const float4 *b [[buffer(2)]],
                 uint i [[thread_position_in_grid]])
{
    o[i] = dot(a[i], b[i]);
}

kernel void hwide(device float4 *o [[buffer(0)]], device const half4 *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = float4(a[i]) * 2.0f;
}

kernel void hmath(device half *o [[buffer(0)]], device const half *a [[buffer(1)]], device const half *b [[buffer(2)]],
                  uint i [[thread_position_in_grid]])
{
    o[i] = fma(a[i], b[i], a[i]) - b[i];
}

kernel void hnarrow(device half *o [[buffer(0)]], device const float *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = half(a[i] * 3.0f);
}

kernel void reverse(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]],
                    uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]])
{
    threadgroup uint tile[64];
    tile[t] = a[g * 64 + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[g * 64 + t] = tile[63 - t];
}

kernel void reduce(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]],
                   uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]])
{
    threadgroup uint tile[64];
    tile[t] = a[g * 64 + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (t < s)
            tile[t] += tile[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0)
        o[g] = tile[0];
}

kernel void halfs(device half *o [[buffer(0)]], device const half *a [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = a[i] * half(0.25);
}

kernel void groups(device uint *o [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                   uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]])
{
    o[g * n + t] = g;
}

// RenderBox's segment lookup: a binary search whose pointer is a phi and a select, then a negative index
struct Seg {
    int start;
    int n;
    float v[2];
};

kernel void bsearch(device uint *o [[buffer(0)]], device const Seg *s [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    device const Seg *p = s + 1;
    int len = s[0].start;
    while (len > 1) {
        int mid = len >> 1;
        device const Seg *m = p + mid;
        bool before = m->start > int(i);
        len = before ? mid : len - mid;
        p = before ? p : m;
    }
    o[i] = uint(p->n + p[-1].n);
}

// a private array written and read at indices only known at run time, as RenderBox fills its layer transforms
kernel void parray(device uint *o [[buffer(0)]], device const uint *in [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    uint2 a[5];
    for (uint k = 0; k < 5; k++)
        a[k] = uint2(in[i] + k, k * 7);
    a[in[i] & 3].y += 100;
    uint2 v = a[(in[i] >> 2) & 3];
    o[i] = v.x * 1000 + v.y;
}

// RenderBox's layer transform: a struct holding a pointer and a float2[5] filled in a loop, read by constant index
struct Layer {
    device const uint *tex;
    float2 m[5];
};

kernel void layer(device float *o [[buffer(0)]], device const uint *in [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    Layer l;
    l.tex = in;
    for (uint k = 0; k < 5; k++)
        l.m[k] = float2(float(in[2 * k + (i & 7)]), float(in[2 * k + 1 + (i & 7)]));
    float2 p = float2(float(i), float(i) * 0.5f);
    float2 r = clamp(p.x * l.m[0] + (p.y * l.m[1] + l.m[2]), l.m[3], l.m[4]);
    o[2 * i] = r.x;
    o[2 * i + 1] = r.y;
}

// 16-bit integers, as a ushort [[vertex_id]] is used
kernel void ushorts(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    ushort v = ushort(i * 3);
    o[i] = uint(v >> 2) * 13 + uint(v & 3);
}

// SIMD-group functions: a 32-lane sum, a butterfly shuffle and a max, with the lane and group builtins
kernel void simdops(device uint *o [[buffer(0)]], device const uint *a [[buffer(1)]], uint i [[thread_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]], uint group [[simdgroup_index_in_threadgroup]])
{
    o[i] = simd_sum(a[i]) + simd_shuffle_xor(a[i], 5) * 3 + simd_max(int(a[i])) * 7 + (lane | group << 5) * 11;
}
