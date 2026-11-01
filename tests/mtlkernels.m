// Run each kernel of tools/mtl/kernels.metal on MalachiteDevice and check it against the same math on the CPU.
// Kernels airc can't lower yet must fail at pipeline creation with an error, not run wrong.
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation tools/mtl/mtlkernels.m -o build/mtlkernels
//   build/mtlkernels tools/mtl/kernels.metal
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <math.h>

#define N 256      // threads
#define W (4 * N)  // words per buffer: room for a float4 per thread

static float f(uint32_t u)
{
    float x;
    memcpy(&x, &u, 4);
    return x;
}

static uint32_t u(float x)
{
    uint32_t v;
    memcpy(&v, &x, 4);
    return v;
}

static int near(float got, float want, float rel)
{
    return fabsf(got - want) <= rel * fabsf(want) + 1e-4f;
}

static void ints(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++) {
        b[1][i] = i * 7 + 3;
        b[2][i] = i * 13 + 1;
    }
}

static void signed_ints(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++) {
        b[1][i] = (uint32_t)((int)i - 100);
        b[2][i] = (uint32_t)(100 - (int)i * 2);
    }
}

static void floats(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++) {
        b[1][i] = u(0.5f + i * 0.01f);
        b[2][i] = u(1.5f - i * 0.125f);
    }
}

static int copy(uint32_t **b, unsigned i)
{
    return b[0][i] == b[1][i];
}

static int add(uint32_t **b, unsigned i)
{
    return b[0][i] == b[1][i] + b[2][i];
}

static int iop(uint32_t **b, unsigned i)
{
    return b[0][i] == b[1][i] * b[2][i] - b[1][i];
}

static int bits(uint32_t **b, unsigned i)
{
    uint32_t a = b[1][i], c = b[2][i];
    return b[0][i] == (((a & c) | (a << 3)) ^ (c >> 2));
}

static int fop(uint32_t **b, unsigned i)
{
    float a = f(b[1][i]), c = f(b[2][i]);
    return near(f(b[0][i]), a * c + c, 1e-6f);
}

static void relu_init(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++)
        b[1][i] = u(i * 0.5f - 40.0f);
}

static int relu(uint32_t **b, unsigned i)
{
    return f(b[0][i]) == fmaxf(f(b[1][i]), 0.0f);
}

static int math(uint32_t **b, unsigned i)
{
    float x = f(b[1][i]);
    float want = exp2f(x) + log2f(x) + sinf(x) * cosf(x) + 1.0f / sqrtf(x) + fmaf(x, x, 1.5f);
    return near(f(b[0][i]), want, 2e-3f);
}

static int imin(uint32_t **b, unsigned i)
{
    int a = (int)b[1][i], c = (int)b[2][i];
    int m = a < c ? a : c;
    return (int)b[0][i] == (m > -7 ? m : -7);
}

static void conv_init(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++)
        b[2][i] = (uint32_t)((int)i * 3 - 200);
}

static int conv(uint32_t **b, unsigned i)
{
    float o = (float)(int)b[2][i] * 0.5f;
    return f(b[0][i]) == o && (int)b[1][i] == (int)(o * 3.0f);
}

static void itof_init(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++)
        b[1][i] = (uint32_t)((int)i * 3 - 200);
}

static int itof(uint32_t **b, unsigned i)
{
    return f(b[0][i]) == (float)(int)b[1][i];
}

static int itof_tid(uint32_t **b, unsigned i)
{
    return f(b[0][i]) == (float)i;
}

static int itof_add(uint32_t **b, unsigned i)
{
    return f(b[0][i]) == (float)((int)b[1][i] + 1);
}

static void ftoi_init(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++)
        b[1][i] = u(i * 1.5f - 100.25f);
}

static int ftoi(uint32_t **b, unsigned i)
{
    return (int)b[0][i] == (int)f(b[1][i]);
}

static void guard_init(uint32_t **b)
{
    ints(b);
    b[2][0] = 200;  // n
}

static int guard(uint32_t **b, unsigned i)
{
    return b[0][i] == (i < 200 ? b[1][i] : 0xdeadbeef);
}

static void pick_init(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++)
        b[1][i] = u(i * 0.02f - 1.0f);
}

static int pick(uint32_t **b, unsigned i)
{
    float a = f(b[1][i]);
    return f(b[0][i]) == (a > 1.0f ? a : -a);
}

static int ipick(uint32_t **b, unsigned i)
{
    return b[0][i] == (b[1][i] < 100 ? b[1][i] : 77u);
}

static void loop_init(uint32_t **b)
{
    ints(b);
    b[2][0] = 10;  // n
}

static int loop(uint32_t **b, unsigned i)
{
    uint32_t acc = 0;
    for (unsigned j = 0; j < b[2][0]; j++)
        acc += b[1][j];
    return b[0][i] == acc;
}

static void loop0_init(uint32_t **b)
{
    ints(b);
    b[2][0] = 0;
}

static int divloop(uint32_t **b, unsigned i)
{
    uint32_t n = b[1][i] & 7;
    return b[0][i] == n * (n - 1) / 2 + n * i;
}

static int groups(uint32_t **b, unsigned i)
{
    return b[0][i] == i / 64;
}

static void vfloats(uint32_t **b)
{
    for (unsigned k = 0; k < W; k++) {
        b[1][k] = u(0.5f + k * 0.01f);
        b[2][k] = u(1.5f - k * 0.0625f);
    }
}

static void vints(uint32_t **b)
{
    for (unsigned k = 0; k < W; k++) {
        b[1][k] = k * 2654435761u;
        b[2][k] = k * 7 + 1;
    }
}

static void halves(uint32_t **b)
{
    _Float16 *a = (_Float16 *)b[1], *c = (_Float16 *)b[2];
    for (unsigned k = 0; k < 2 * W; k++) {
        a[k] = (_Float16)(k * 0.125f - 3.0f);
        c[k] = (_Float16)(0.75f + k * 0.03125f);
    }
}

static int vec4(uint32_t **b, unsigned i)
{
    for (unsigned l = 0; l < 4; l++)
        if (f(b[0][4 * i + l]) != f(b[1][4 * i + l]) * 2.0f + (float)(l + 1))
            return 0;
    return 1;
}

static int chain(uint32_t **b, unsigned i)
{
    for (unsigned l = 0; l < 4; l++) {
        float s = f(b[1][4 * i + l]), x = s;
        for (int k = 0; k < 12; k++)
            x = x * 0.5f + s;
        if (!near(f(b[0][4 * i + l]), x, 1e-5f))
            return 0;
    }
    return 1;
}

static int swizzle(uint32_t **b, unsigned i)
{
    const float *a = (const float *)b[1] + 4 * i, *c = (const float *)b[2] + 4 * i;
    float want[4] = { a[3] * c[0] + a[0], a[2] * c[0] + a[1], a[1] * c[0] + c[2], a[0] * c[0] + c[3] };
    for (unsigned l = 0; l < 4; l++)
        if (!near(f(b[0][4 * i + l]), want[l], 1e-6f))
            return 0;
    return 1;
}

static int vec3(uint32_t **b, unsigned i)  // float3 is 16 bytes in a buffer; lane 3 is padding
{
    const float *a = (const float *)b[1] + 4 * i;
    float want[3] = { a[0] * 0.5f + a[2], a[1] * 0.5f + a[0], a[2] * 0.5f + a[1] };
    for (unsigned l = 0; l < 3; l++)
        if (!near(f(b[0][4 * i + l]), want[l], 1e-6f))
            return 0;
    return 1;
}

static int uvec(uint32_t **b, unsigned i)
{
    for (unsigned l = 0; l < 4; l++)
        if (b[0][4 * i + l] != ((b[1][4 * i + l] ^ b[2][4 * i + l]) + i))
            return 0;
    return 1;
}

static int dot4(uint32_t **b, unsigned i)
{
    const float *a = (const float *)b[1] + 4 * i, *c = (const float *)b[2] + 4 * i;
    return near(f(b[0][i]), a[0] * c[0] + a[1] * c[1] + a[2] * c[2] + a[3] * c[3], 1e-5f);
}

static int hwide(uint32_t **b, unsigned i)
{
    const _Float16 *a = (const _Float16 *)b[1] + 4 * i;
    for (unsigned l = 0; l < 4; l++)
        if (f(b[0][4 * i + l]) != (float)a[l] * 2.0f)
            return 0;
    return 1;
}

static int hmath(uint32_t **b, unsigned i)
{
    _Float16 a = ((const _Float16 *)b[1])[i], c = ((const _Float16 *)b[2])[i];
    _Float16 want = (_Float16)fmaf((float)a, (float)c, (float)a) - c;  // fma rounds once, then the sub
    return ((const _Float16 *)b[0])[i] == want;
}

static void hnarrow_init(uint32_t **b)
{
    for (unsigned k = 0; k < N; k++)
        b[1][k] = u(k * 0.37f - 20.0f);
}

static int hnarrow(uint32_t **b, unsigned i)
{
    return ((const _Float16 *)b[0])[i] == (_Float16)(f(b[1][i]) * 3.0f);
}

static int reverse(uint32_t **b, unsigned i)
{
    return b[0][i] == b[1][i / 64 * 64 + 63 - i % 64];
}

static int reduce(uint32_t **b, unsigned i)  // one sum per threadgroup in o[0..3]; the rest untouched
{
    if (i >= N / 64)
        return b[0][i] == 0xdeadbeef;
    uint32_t sum = 0;
    for (unsigned k = 0; k < 64; k++)
        sum += b[1][i * 64 + k];
    return b[0][i] == sum;
}

static int halfs(uint32_t **b, unsigned i)
{
    return ((const _Float16 *)b[0])[i] == ((const _Float16 *)b[1])[i] * (_Float16)0.25f;
}

// values either side of zero, with halves (round goes away from zero) and integers
static void round_init(uint32_t **b)
{
    for (unsigned i = 0; i < N; i++)
        b[1][i] = u(-8.0f + i * 0.25f + (i % 3 == 0 ? 0.1f : 0.0f));
}

static int rounding(uint32_t **b, unsigned i)
{
    float x = f(b[1][i]), want[4] = { floorf(x), ceilf(x), roundf(x), x - floorf(x) };
    for (unsigned l = 0; l < 4; l++)
        if (!near(f(b[0][4 * i + l]), want[l], 1e-6f))
            return 0;
    return 1;
}

static const float params[2] = { 3.0f, -0.5f };  // affine's constant Params, passed with setBytes

static int affine(uint32_t **b, unsigned i)
{
    return near(f(b[0][i]), f(b[2][i]) * params[0] + params[1], 1e-6f);
}

// segment 0 holds the count; segment k (1..200, four words each) starts at k - 1 (more than RenderBox's 130, so
// the search takes 8 steps), its n 100k
static void bsearch_init(uint32_t **b)
{
    b[1][0] = 200;
    for (unsigned k = 1; k <= 200; k++) {
        b[1][4 * k] = k - 1;
        b[1][4 * k + 1] = 100 * k;
    }
}

static int bsearch_(uint32_t **b, unsigned i)
{
    const uint32_t *s = b[1];
    int p = 1, len = (int)s[0];
    while (len > 1) {
        int half = len >> 1;
        int before = (int)s[4 * (p + half)] > (int)i;
        len = before ? half : len - half;
        p = before ? p : p + half;
    }
    return b[0][i] == s[4 * p + 1] + s[4 * (p - 1) + 1];
}

static int parray(uint32_t **b, unsigned i)
{
    uint32_t x = b[1][i], a[5][2];
    for (uint32_t k = 0; k < 5; k++) {
        a[k][0] = x + k;
        a[k][1] = k * 7;
    }
    a[x & 3][1] += 100;
    const uint32_t *v = a[(x >> 2) & 3];
    return b[0][i] == v[0] * 1000 + v[1];
}

static int layer(uint32_t **b, unsigned i)
{
    float m[5][2];
    for (unsigned k = 0; k < 5; k++) {
        m[k][0] = (float)b[1][2 * k + (i & 7)];
        m[k][1] = (float)b[1][2 * k + 1 + (i & 7)];
    }
    float p[2] = { (float)i, (float)i * 0.5f };
    for (unsigned c = 0; c < 2; c++) {
        float r = p[0] * m[0][c] + (p[1] * m[1][c] + m[2][c]);
        r = r < m[3][c] ? m[3][c] : r > m[4][c] ? m[4][c] : r;
        if (!near(f(b[0][2 * i + c]), r, 1e-3f * fabsf(r) + 1e-3f))
            return 0;
    }
    return 1;
}

// the 32 lanes of i's warp; threadgroups are 64 threads, two warps
static int simd(uint32_t **b, unsigned i)
{
    unsigned base = i & ~31u;
    uint32_t sum = 0, max = 0;
    for (unsigned j = base; j < base + 32; j++) {
        sum += b[1][j];
        max = b[1][j] > max ? b[1][j] : max;
    }
    return b[0][i] == sum + b[1][i ^ 5] * 3 + max * 7 + (i % 64) * 11;
}

static int short16(uint32_t **b, unsigned i)
{
    uint16_t v = (uint16_t)(i * 3);
    return b[0][i] == (uint32_t)(v >> 2) * 13 + (v & 3);
}

static const struct test {
    const char *name;
    void (*init)(uint32_t **b);  // fills the inputs; buffer 0, the output, starts as 0xdeadbeef
    int (*ok)(uint32_t **b, unsigned i);
    const void *bytes;  // set with setBytes at index 1 in place of buffer 1
    size_t nbytes;
} tests[] = {
    { "copy", ints, copy },        { "add", ints, add },         { "iop", ints, iop },
    { "bsearch", bsearch_init, bsearch_ }, { "ushorts", NULL, short16 }, { "parray", ints, parray }, { "layer", ints, layer },
    { "bits", ints, bits },        { "fop", floats, fop },       { "relu", relu_init, relu },
    { "math", floats, math },      { "imin", signed_ints, imin }, { "itof", itof_init, itof },
    { "itof_tid", NULL, itof_tid }, { "itof_add", itof_init, itof_add },
    { "ftoi", ftoi_init, ftoi },   { "conv", conv_init, conv },
    { "guard", guard_init, guard }, { "groups", NULL, groups },  { "pick", pick_init, pick },
    { "ipick", ints, ipick },      { "loop", loop_init, loop },  { "loop", loop0_init, loop },
    { "divloop", ints, divloop },  { "vec4", vfloats, vec4 },    { "swizzle", vfloats, swizzle },
    { "vec3", vfloats, vec3 },     { "uvec", vints, uvec },      { "dot4", vfloats, dot4 },
    { "hwide", halves, hwide },    { "hmath", halves, hmath },   { "hnarrow", hnarrow_init, hnarrow },
    { "halfs", halves, halfs },    { "reverse", ints, reverse }, { "reduce", ints, reduce },
    { "chain", vfloats, chain },   { "affine", floats, affine, params, sizeof params },
    { "rounding", round_init, rounding }, { "simdops", ints, simd },
};

int main(int argc, char **argv)
{
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s <kernels.metal>\n", argv[0]);
            return 2;
        }
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> lib = src ? [dev newLibraryWithSource:src options:nil error:&err] : nil;
        if (!lib) {
            fprintf(stderr, "%s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> bufs[3];
        uint32_t *b[3];
        for (int i = 0; i < 3; i++) {
            bufs[i] = [dev newBufferWithLength:W * 4 options:MTLResourceStorageModeShared];
            b[i] = bufs[i].contents;
        }
        int bad = 0;
        for (unsigned t = 0; t < sizeof tests / sizeof tests[0]; t++) {
            const struct test *T = &tests[t];
            id<MTLFunction> fn = [lib newFunctionWithName:@(T->name)];
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!T->ok) {
                printf("%-7s %s\n", T->name, ps ? "UNEXPECTEDLY COMPILED" : err.localizedDescription.UTF8String);
                bad |= ps != nil;
                continue;
            }
            if (!ps) {
                printf("%-7s FAILED to compile: %s\n", T->name, err.localizedDescription.UTF8String);
                bad = 1;
                continue;
            }
            for (int i = 0; i < 3; i++)
                memset(b[i], 0, W * 4);
            for (unsigned i = 0; i < W; i++)
                b[0][i] = 0xdeadbeef;
            if (T->init)
                T->init(b);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            for (int i = 0; i < 3; i++)
                [enc setBuffer:bufs[i] offset:0 atIndex:i];
            if (T->bytes)
                [enc setBytes:T->bytes length:T->nbytes atIndex:1];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            unsigned i = 0;
            while (i < N && T->ok(b, i))
                i++;
            if (i == N) {
                printf("%-7s ok\n", T->name);
            } else {
                printf("%-7s MISMATCH at %u: out 0x%08x (in 0x%08x 0x%08x)\n", T->name, i, b[0][i], b[1][i], b[2][i]);
                bad = 1;
            }
        }
        return bad;
    }
}
