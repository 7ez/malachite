// Tessellation on MalachiteDevice: a triangle patch of three control points with factors 4, drawn into a 64x64 target
// and checked against the same triangle drawn untessellated (the barycentric colours interpolate linearly, so the
// two match); drawn as edges, it has more of them than with factors 1; a factor of 0 or NaN culls it; under back-face
// culling one output winding draws and the other doesn't, as the untessellated triangle says; read through a control
// point index buffer it draws the same; a quad patch covers its square; and an instanced patch picked by a patch
// index buffer, or by its patch start, lands where its per-patch offset and instance put it.
// Then: functions without a control point count draw as those with one; each edge factor subdivides its own edge;
// each partition mode, maximum factor and factor scale tessellates as the integer factor it should come to; each
// factor step function finds a patch's factors where Metal says, and a patch index buffer picks factors with the
// rest of the patch's data.
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation tools/mtl/mtltess.m -o build/mtltess
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define N 64

static NSString *const src = @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct CP { float4 p [[attribute(0)]]; };\n"
    "struct PatchIn { patch_control_point<CP> cp; };\n"
    "struct OffsetIn { patch_control_point<CP> cp; float4 off [[attribute(1)]]; };\n"
    "struct V { float4 pos [[position]]; float4 col; };\n"
    "[[patch(triangle, 3)]] vertex V ttri(PatchIn in [[stage_in]], float3 b [[position_in_patch]])\n"
    "{\n"
    "    V o;\n"
    "    o.pos = b.x * in.cp[0].p + b.y * in.cp[1].p + b.z * in.cp[2].p;\n"
    "    o.col = float4(b, 1);\n"
    "    return o;\n"
    "}\n"
    "[[patch(quad, 4)]] vertex V tquad(PatchIn in [[stage_in]], float2 uv [[position_in_patch]])\n"
    "{\n"
    "    V o;\n"
    "    o.pos = mix(mix(in.cp[0].p, in.cp[1].p, uv.x), mix(in.cp[3].p, in.cp[2].p, uv.x), uv.y);\n"
    "    o.col = float4(1);\n"
    "    return o;\n"
    "}\n"
    "[[patch(triangle)]] vertex V ttrin(PatchIn in [[stage_in]], float3 b [[position_in_patch]])\n"
    "{\n"
    "    V o;\n"
    "    o.pos = b.x * in.cp[0].p + b.y * in.cp[1].p + b.z * in.cp[2].p;\n"
    "    o.col = float4(b, 1);\n"
    "    return o;\n"
    "}\n"
    "[[patch(quad)]] vertex V tquadn(PatchIn in [[stage_in]], float2 uv [[position_in_patch]])\n"
    "{\n"
    "    V o;\n"
    "    o.pos = mix(mix(in.cp[0].p, in.cp[1].p, uv.x), mix(in.cp[3].p, in.cp[2].p, uv.x), uv.y);\n"
    "    o.col = float4(1);\n"
    "    return o;\n"
    "}\n"
    "[[patch(triangle, 3)]] vertex V toff(OffsetIn in [[stage_in]], float3 b [[position_in_patch]],\n"
    "                                     uint pid [[patch_id]], uint iid [[instance_id]])\n"
    "{\n"
    "    V o;\n"
    "    o.pos = b.x * in.cp[0].p + b.y * in.cp[1].p + b.z * in.cp[2].p + in.off - float4(0, float(iid), 0, 0);\n"
    "    o.col = float4(float(pid == 0), float(pid == 1), float(iid), 1);\n"
    "    return o;\n"
    "}\n"
    "vertex V vref(uint v [[vertex_id]], const device float4 *p [[buffer(0)]])\n"
    "{\n"
    "    V o;\n"
    "    o.pos = p[v];\n"
    "    o.col = float4(float(v == 0), float(v == 1), float(v == 2), 1);\n"
    "    return o;\n"
    "}\n"
    "fragment float4 fcol(V in [[stage_in]]) { return in.col; }\n";

static id<MTLDevice> dev;
static id<MTLLibrary> lib;
static id<MTLCommandQueue> queue;
static id<MTLTexture> target;

// vfn and the colour fragment function; a post-tessellation function reads attribute 0 (float4) per control point
// from buffer 0 and attribute 1 (float4) per patch from buffer 1, its factors per patch, integer partitioning up to
// 16, clockwise, unless set changes them
static id<MTLRenderPipelineState> pipeline(NSString *vfn, void (^set)(MTLRenderPipelineDescriptor *pd))
{
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:vfn];
    pd.fragmentFunction = [lib newFunctionWithName:@"fcol"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    if (![vfn isEqualToString:@"vref"]) {
        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        for (int i = 0; i < 2; i++) {
            vd.attributes[i].format = MTLVertexFormatFloat4;
            vd.attributes[i].bufferIndex = i;
            vd.layouts[i].stride = 16;
        }
        vd.layouts[0].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
        vd.layouts[1].stepFunction = MTLVertexStepFunctionPerPatch;
        pd.vertexDescriptor = vd;
        pd.tessellationPartitionMode = MTLTessellationPartitionModeInteger;
        pd.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
        pd.tessellationOutputWindingOrder = MTLWindingClockwise;
        pd.maxTessellationFactor = 16;
        if (set)
            set(pd);
    }
    NSError *err;
    id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps)
        printf("%s: no pipeline: %s\n", vfn.UTF8String, err.localizedDescription.UTF8String);
    return ps;
}

static id<MTLBuffer> buffer(const void *bytes, NSUInteger length)
{
    return [dev newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
}

// the target cleared to black, encode's draws into it, read back into out
static void draw(void (^encode)(id<MTLRenderCommandEncoder> e), uint8_t out[N][N][4])
{
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = target;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
    encode(e);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    [target getBytes:out bytesPerRow:N * 4 fromRegion:MTLRegionMake2D(0, 0, N, N) mipmapLevel:0];
}

static int lit(uint8_t px[N][N][4])
{
    int n = 0;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            n += px[y][x][0] || px[y][x][1] || px[y][x][2];
    return n;
}

// pixels lit in both whose colours differ by more than tolerance, and pixels lit in one only
static void compare(uint8_t a[N][N][4], uint8_t b[N][N][4], int tolerance, int *colour, int *coverage)
{
    *colour = *coverage = 0;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            int la = a[y][x][0] || a[y][x][1] || a[y][x][2], lb = b[y][x][0] || b[y][x][1] || b[y][x][2];
            if (la != lb) {
                ++*coverage;
                continue;
            }
            for (int c = 0; c < 3; c++)
                if (la && abs(a[y][x][c] - b[y][x][c]) > tolerance) {
                    ++*colour;
                    break;
                }
        }
}

// the pixel at NDC x, y
static const uint8_t *at(uint8_t px[N][N][4], float x, float y)
{
    return px[(int)((1 - y) / 2 * N)][(int)((x + 1) / 2 * N)];
}

// f as a half, for 0 and the normal halves the factors here are
static uint16_t half(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    return f == 0 ? 0 : (uint16_t)((b >> 16 & 0x8000) | ((b >> 23 & 0xff) - 112) << 10 | (b >> 13 & 0x3ff));
}

static id<MTLBuffer> tri_factors(float e0, float e1, float e2, float inside)
{
    MTLTriangleTessellationFactorsHalf f = { { half(e0), half(e1), half(e2) }, half(inside) };
    return buffer(&f, sizeof f);
}

static id<MTLBuffer> quad_factors(float e0, float e1, float e2, float e3, float inside)
{
    MTLQuadTessellationFactorsHalf f = { { half(e0), half(e1), half(e2), half(e3) }, { half(inside), half(inside) } };
    return buffer(&f, sizeof f);
}

// lit pixels near each edge of a patch drawn over corners (NDC), away from its other edges: edge k's are those whose
// coordinate that is 0 on it (a triangle's barycentric weight of corner k; a quad's u, v, 1 - u, 1 - v, the square
// from corner 0 to corner 2) is between 0.04 and 0.2, with all the others over 0.15
static void bands(uint8_t px[N][N][4], const float (*c)[4], bool quad, int band[4])
{
    memset(band, 0, 4 * sizeof(int));
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++) {
            if (!px[row][col][0] && !px[row][col][1] && !px[row][col][2])
                continue;
            float x = (col + 0.5f) / N * 2 - 1, y = 1 - (row + 0.5f) / N * 2, w[4];
            if (quad) {
                w[0] = (x - c[0][0]) / (c[2][0] - c[0][0]);
                w[1] = (y - c[0][1]) / (c[2][1] - c[0][1]);
                w[2] = 1 - w[0];
                w[3] = 1 - w[1];
            } else {
                float d = (c[1][1] - c[2][1]) * (c[0][0] - c[2][0]) + (c[2][0] - c[1][0]) * (c[0][1] - c[2][1]);
                w[0] = ((c[1][1] - c[2][1]) * (x - c[2][0]) + (c[2][0] - c[1][0]) * (y - c[2][1])) / d;
                w[1] = ((c[2][1] - c[0][1]) * (x - c[2][0]) + (c[0][0] - c[2][0]) * (y - c[2][1])) / d;
                w[2] = 1 - w[0] - w[1];
                w[3] = 1;
            }
            for (int k = 0; k < (quad ? 4 : 3); k++) {
                bool away = true;
                for (int j = 0; j < 4; j++)
                    away &= j == k || w[j] > 0.15f;
                band[k] += away && w[k] > 0.04f && w[k] < 0.2f;
            }
        }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    @autoreleasepool {
        NSError *err;
        dev = MTLCreateSystemDefaultDevice();
        lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            printf("%s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        queue = [dev newCommandQueue];
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:N
                                                                                     height:N
                                                                                  mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        target = [dev newTextureWithDescriptor:td];
        id<MTLRenderPipelineState> ref = pipeline(@"vref", nil);
        id<MTLRenderPipelineState> tri = pipeline(@"ttri", nil);
        id<MTLRenderPipelineState> ccw = pipeline(@"ttri", ^(MTLRenderPipelineDescriptor *pd) {
            pd.tessellationOutputWindingOrder = MTLWindingCounterClockwise;
        });
        id<MTLRenderPipelineState> indexed = pipeline(@"ttri", ^(MTLRenderPipelineDescriptor *pd) {
            pd.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeUInt16;
        });
        id<MTLRenderPipelineState> quad = pipeline(@"tquad", nil);
        id<MTLRenderPipelineState> off = pipeline(@"toff", nil);
        id<MTLRenderPipelineState> trin = pipeline(@"ttrin", nil);
        id<MTLRenderPipelineState> quadn = pipeline(@"tquadn", nil);
        if (!ref || !tri || !ccw || !indexed || !quad || !off || !trin || !quadn)
            return 1;

        // clockwise on the framebuffer, Metal's front faces
        const float p[3][4] = { { -0.75f, -0.75f, 0, 1 }, { 0, 0.75f, 0, 1 }, { 0.75f, -0.75f, 0, 1 } };
        id<MTLBuffer> points = buffer(p, sizeof p);
        const float q[3][4] = { { 0, 0.75f, 0, 1 }, { 0.75f, -0.75f, 0, 1 }, { -0.75f, -0.75f, 0, 1 } };
        id<MTLBuffer> permuted = buffer(q, sizeof q);  // point j at index[j]
        const uint16_t index[3] = { 2, 0, 1 };
        id<MTLBuffer> indices = buffer(index, sizeof index);
        const uint16_t h4 = 0x4400, h1 = 0x3c00, hnan = 0x7e00;  // halves 4, 1, NaN
        MTLTriangleTessellationFactorsHalf f4 = { { h4, h4, h4 }, h4 }, f1 = { { h1, h1, h1 }, h1 };
        MTLTriangleTessellationFactorsHalf f0 = { { 0, h4, h4 }, h4 }, fnan = { { hnan, h4, h4 }, h4 };
        NSArray<id<MTLBuffer>> *factors = @[ buffer(&f4, sizeof f4), buffer(&f1, sizeof f1), buffer(&f0, sizeof f0),
                                              buffer(&fnan, sizeof fnan) ];
        int bad = 0;

        // a patch of p with factors[f] through pipeline ps
        void (^one)(id<MTLRenderPipelineState>, int, MTLCullMode, MTLTriangleFillMode, uint8_t[N][N][4]) =
            ^(id<MTLRenderPipelineState> ps, int f, MTLCullMode cull, MTLTriangleFillMode fill, uint8_t out[N][N][4]) {
                draw(^(id<MTLRenderCommandEncoder> e) {
                    [e setRenderPipelineState:ps];
                    [e setCullMode:cull];
                    [e setTriangleFillMode:fill];
                    [e setVertexBuffer:points offset:0 atIndex:0];
                    [e setTessellationFactorBuffer:factors[f] offset:0 instanceStride:0];
                    [e drawPatches:3 patchStart:0 patchCount:1 patchIndexBuffer:nil patchIndexBufferOffset:0
                         instanceCount:1 baseInstance:0];
                }, out);
            };

        static uint8_t want[N][N][4], got[N][N][4], other[N][N][4];
        draw(^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:ref];
            [e setVertexBuffer:points offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }, want);
        one(tri, 0, MTLCullModeNone, MTLTriangleFillModeFill, got);
        int colour, coverage;
        compare(want, got, 2, &colour, &coverage);
        int ok = lit(got) > N * N / 4 && !colour && coverage <= N;  // the subdivided edges may round apart
        printf("tess fill %s (%d lit, untessellated %d; %d colours off, %d edge pixels apart)\n",
               ok ? "ok" : "MISMATCH", lit(got), lit(want), colour, coverage);
        bad |= !ok;

        one(tri, 0, MTLCullModeNone, MTLTriangleFillModeLines, got);
        one(tri, 1, MTLCullModeNone, MTLTriangleFillModeLines, other);
        ok = lit(other) > 0 && lit(got) > lit(other) * 3 / 2;
        printf("tess lines %s (%d lit with factors 4, %d with 1)\n", ok ? "ok" : "MISMATCH", lit(got), lit(other));
        bad |= !ok;

        one(tri, 2, MTLCullModeNone, MTLTriangleFillModeFill, got);
        one(tri, 3, MTLCullModeNone, MTLTriangleFillModeFill, other);
        ok = !lit(got) && !lit(other);
        printf("tess cull %s (%d lit with an edge factor 0, %d with NaN)\n", ok ? "ok" : "MISMATCH", lit(got),
               lit(other));
        bad |= !ok;

        // under back-face culling the untessellated triangle draws, the clockwise output triangles are culled and the
        // counter-clockwise ones draw, as on an Apple GPU
        draw(^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:ref];
            [e setCullMode:MTLCullModeBack];
            [e setVertexBuffer:points offset:0 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }, want);
        one(tri, 0, MTLCullModeBack, MTLTriangleFillModeFill, got);
        one(ccw, 0, MTLCullModeBack, MTLTriangleFillModeFill, other);
        ok = lit(want) && !lit(got) && lit(other);
        printf("tess winding %s (back faces culled: %d lit untessellated, %d clockwise, %d counter-clockwise%s)\n",
               ok ? "ok" : "MISMATCH", lit(want), lit(got), lit(other),
               lit(got) && !lit(other) ? "; swap the output primitives the plugin maps the windings to" : "");
        bad |= !ok;

        one(tri, 0, MTLCullModeNone, MTLTriangleFillModeFill, want);
        draw(^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:indexed];
            [e setVertexBuffer:permuted offset:0 atIndex:0];
            [e setTessellationFactorBuffer:factors[0] offset:0 instanceStride:0];
            [e drawIndexedPatches:3 patchStart:0 patchCount:1 patchIndexBuffer:nil patchIndexBufferOffset:0
                controlPointIndexBuffer:indices controlPointIndexBufferOffset:0 instanceCount:1 baseInstance:0];
        }, got);
        compare(want, got, 1, &colour, &coverage);
        ok = lit(got) && !colour && !coverage;
        printf("tess indexed %s (%d lit; %d colours off, %d pixels apart)\n", ok ? "ok" : "MISMATCH", lit(got), colour,
               coverage);
        bad |= !ok;

        // the square from NDC -0.5 to 0.5: pixels 16..47
        const float c[4][4] = {
            { -0.5f, -0.5f, 0, 1 }, { 0.5f, -0.5f, 0, 1 }, { 0.5f, 0.5f, 0, 1 }, { -0.5f, 0.5f, 0, 1 },
        };
        MTLQuadTessellationFactorsHalf fq = { { h4, h4, h4, h4 }, { h4, h4 } };
        id<MTLBuffer> corners = buffer(c, sizeof c), qfactors = buffer(&fq, sizeof fq);
        draw(^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:quad];
            [e setVertexBuffer:corners offset:0 atIndex:0];
            [e setTessellationFactorBuffer:qfactors offset:0 instanceStride:0];
            [e drawPatches:4 patchStart:0 patchCount:1 patchIndexBuffer:nil patchIndexBufferOffset:0 instanceCount:1
                baseInstance:0];
        }, got);
        int holes = 0, spill = 0;
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int on = got[y][x][0] != 0;
                holes += x > 16 && x < 47 && y > 16 && y < 47 && !on;
                spill += (x < 15 || x > 48 || y < 15 || y > 48) && on;
            }
        ok = !holes && !spill;
        printf("tess quad %s (%d lit, %d holes inside, %d outside)\n", ok ? "ok" : "MISMATCH", lit(got), holes, spill);
        bad |= !ok;

        // patch 1 of a small triangle at per-patch offsets, two instances a unit apart: picked by a patch index
        // buffer, then by its patch start (its control points are the second three, as a patch's are fetched by its
        // id); patch 0's place stays black
        const float s[6][4] = { { -0.25f, -0.25f, 0, 1 }, { 0, 0.25f, 0, 1 }, { 0.25f, -0.25f, 0, 1 },
                                { -0.25f, -0.25f, 0, 1 }, { 0, 0.25f, 0, 1 }, { 0.25f, -0.25f, 0, 1 } };
        const float o[2][4] = { { -0.5f, 0.5f, 0, 0 }, { 0.5f, 0.5f, 0, 0 } };
        const uint32_t pick = 1;
        MTLTriangleTessellationFactorsHalf two[2] = { f4, f4 };
        id<MTLBuffer> small = buffer(s, sizeof s), offsets = buffer(o, sizeof o), picks = buffer(&pick, 4);
        id<MTLBuffer> tfactors = buffer(two, sizeof two);
        for (int byStart = 0; byStart < 2; byStart++) {
            draw(^(id<MTLRenderCommandEncoder> e) {
                [e setRenderPipelineState:off];
                [e setVertexBuffer:small offset:0 atIndex:0];
                [e setVertexBuffer:offsets offset:0 atIndex:1];
                [e setTessellationFactorBuffer:tfactors offset:0 instanceStride:0];
                [e drawPatches:3 patchStart:byStart patchCount:1 patchIndexBuffer:byStart ? nil : picks
                    patchIndexBufferOffset:0 instanceCount:2 baseInstance:0];
            }, got);
            const uint8_t *i0 = at(got, 0.5f, 0.42f), *i1 = at(got, 0.5f, -0.58f), *p0 = at(got, -0.5f, 0.42f);
            ok = !i0[0] && i0[1] == 255 && !i0[2] && !i1[0] && i1[1] == 255 && i1[2] == 255 && !p0[1] &&
                 lit(got) > 0;
            printf("tess %s %s (instance 0 %d %d %d, instance 1 %d %d %d, patch 0's place %d %d %d)\n",
                   byStart ? "patch start" : "patch index", ok ? "ok" : "MISMATCH", i0[0], i0[1], i0[2], i1[0], i1[1],
                   i1[2], p0[0], p0[1], p0[2]);
            bad |= !ok;
        }

        // a patch of n control points cps with factors f scaled by scale through ps, drawn as edges
        void (^patch)(id<MTLRenderPipelineState>, id<MTLBuffer>, NSUInteger, id<MTLBuffer>, float, uint8_t[N][N][4]) =
            ^(id<MTLRenderPipelineState> ps, id<MTLBuffer> cps, NSUInteger n, id<MTLBuffer> f, float scale,
              uint8_t out[N][N][4]) {
                draw(^(id<MTLRenderCommandEncoder> e) {
                    [e setRenderPipelineState:ps];
                    [e setTriangleFillMode:MTLTriangleFillModeLines];
                    [e setVertexBuffer:cps offset:0 atIndex:0];
                    [e setTessellationFactorBuffer:f offset:0 instanceStride:0];
                    [e setTessellationFactorScale:scale];
                    [e drawPatches:n patchStart:0 patchCount:1 patchIndexBuffer:nil patchIndexBufferOffset:0
                         instanceCount:1 baseInstance:0];
                }, out);
            };

        // the control point count from the draw, against the functions with one: also 16 a patch, the corners
        // first, which no pipeline has been made for
        float c16[16][4] = { 0 };
        memcpy(c16, c, sizeof c);
        id<MTLBuffer> corners16 = buffer(c16, sizeof c16);
        const struct {
            const char *what;
            __unsafe_unretained id<MTLRenderPipelineState> counted, uncounted;
            __unsafe_unretained id<MTLBuffer> cps, uncounted_cps, factors;
            NSUInteger n, uncounted_n;
        } counts[] = {
            { "triangle", tri, trin, points, points, factors[0], 3, 3 },
            { "quad", quad, quadn, corners, corners, qfactors, 4, 4 },
            { "quad of 16", quad, quadn, corners, corners16, qfactors, 4, 16 },
        };
        for (size_t i = 0; i < sizeof counts / sizeof counts[0]; i++) {
            patch(counts[i].counted, counts[i].cps, counts[i].n, counts[i].factors, 1, want);
            patch(counts[i].uncounted, counts[i].uncounted_cps, counts[i].uncounted_n, counts[i].factors, 1, got);
            compare(want, got, 0, &colour, &coverage);
            ok = lit(got) && !colour && !coverage;
            printf("tess no count %s %s (%d lit, with a count %d; %d colours off, %d pixels apart)\n", counts[i].what,
                   ok ? "ok" : "MISMATCH", lit(got), lit(want), colour, coverage);
            bad |= !ok;
        }

        // one edge factor 8 and the others 1, inside 2: the edges fan out from that edge's vertices only. Edge k is
        // the one where the coordinate k of the position in the patch is 0 (a quad's u, v, 1 - u, 1 - v), as in D3D
        for (int q = 0; q < 2; q++)
            for (int k = 0; k < (q ? 4 : 3); k++) {
                float e[4] = { 1, 1, 1, 1 };
                e[k] = 8;
                patch(q ? quad : tri, q ? corners : points, q ? 4 : 3,
                      q ? quad_factors(e[0], e[1], e[2], e[3], 2) : tri_factors(e[0], e[1], e[2], 2), 1, got);
                int band[4];
                bands(got, q ? c : p, q, band);
                ok = true;
                for (int j = 0; j < (q ? 4 : 3); j++)
                    ok &= j == k || band[k] > 2 * band[j];
                printf("tess %s edge %d %s (lit near each edge: %d %d %d %d)\n", q ? "quad" : "triangle", k,
                       ok ? "ok" : "MISMATCH", band[0], band[1], band[2], band[3]);
                bad |= !ok;
            }

        // each partition mode, maximum and scale against the integer factors the factors should come to: the same
        // edges, or (for fractional factors between two integers) clearly different ones
        id<MTLRenderPipelineState> (^tuned)(MTLTessellationPartitionMode, NSUInteger, bool) =
            ^(MTLTessellationPartitionMode mode, NSUInteger max, bool scaled) {
                return pipeline(@"ttri", ^(MTLRenderPipelineDescriptor *pd) {
                    pd.tessellationPartitionMode = mode;
                    pd.maxTessellationFactor = max;
                    pd.tessellationFactorScaleEnabled = scaled;
                });
            };
        id<MTLRenderPipelineState> pow2 = tuned(MTLTessellationPartitionModePow2, 16, false);
        id<MTLRenderPipelineState> odd = tuned(MTLTessellationPartitionModeFractionalOdd, 16, false);
        id<MTLRenderPipelineState> even = tuned(MTLTessellationPartitionModeFractionalEven, 16, false);
        id<MTLRenderPipelineState> int8 = tuned(MTLTessellationPartitionModeInteger, 8, false);
        id<MTLRenderPipelineState> pow2_8 = tuned(MTLTessellationPartitionModePow2, 8, false);
        id<MTLRenderPipelineState> odd8 = tuned(MTLTessellationPartitionModeFractionalOdd, 8, false);
        id<MTLRenderPipelineState> even8 = tuned(MTLTessellationPartitionModeFractionalEven, 8, false);
        id<MTLRenderPipelineState> scaled = tuned(MTLTessellationPartitionModeInteger, 16, true);
        id<MTLRenderPipelineState> scaled8 = tuned(MTLTessellationPartitionModeInteger, 8, true);
        if (!pow2 || !odd || !even || !int8 || !pow2_8 || !odd8 || !even8 || !scaled || !scaled8)
            return 1;
        const struct {
            const char *what;
            __unsafe_unretained id<MTLRenderPipelineState> ps;
            float factor, scale, as;  // drawn with factor and scale, against integer factor as
            bool same;
        } parts[] = {
            { "pow2 3 as integer 4", pow2, 3, 1, 4, true },
            { "integer 3.5 as 4", tri, 3.5f, 1, 4, true },
            { "fractional odd 3 as integer 3", odd, 3, 1, 3, true },
            { "fractional odd 4 unlike integer 3", odd, 4, 1, 3, false },
            { "fractional odd 4 unlike integer 5", odd, 4, 1, 5, false },
            { "fractional even 4 as integer 4", even, 4, 1, 4, true },
            { "fractional even 5 unlike integer 4", even, 5, 1, 4, false },
            { "fractional even 5 unlike integer 6", even, 5, 1, 6, false },
            { "max 8 integer 16 as 8", int8, 16, 1, 8, true },
            { "max 8 pow2 16 as integer 8", pow2_8, 16, 1, 8, true },
            { "max 8 fractional odd 16 as integer 7", odd8, 16, 1, 7, true },
            { "max 8 fractional even 16 as integer 8", even8, 16, 1, 8, true },
            { "scale 0.5 of 8 as 4", scaled, 8, 0.5f, 4, true },
            { "scale 0.5 not enabled, 4 as 4", tri, 4, 0.5f, 4, true },
            { "scale 4 of 4 under max 8 as 8", scaled8, 4, 4, 8, true },
        };
        for (size_t i = 0; i < sizeof parts / sizeof parts[0]; i++) {
            float f = parts[i].factor, a = parts[i].as;
            patch(parts[i].ps, points, 3, tri_factors(f, f, f, f), parts[i].scale, got);
            patch(tri, points, 3, tri_factors(a, a, a, a), 1, want);
            compare(want, got, 2, &colour, &coverage);
            int apart = colour + coverage;
            ok = lit(got) && (parts[i].same ? apart <= 8 : apart > 24);
            printf("tess %s %s (%d pixels apart)\n", parts[i].what, ok ? "ok" : "MISMATCH", apart);
            bad |= !ok;
        }

        // factors 4, 0, 0, 4 with an instance stride of two of them, for patches 0 and 1 of two instances: which of
        // the four draw (patch 0 and 1 of instance 0, then of instance 1) says where each step function read them
        id<MTLRenderPipelineState> (^stepped)(MTLTessellationFactorStepFunction) =
            ^(MTLTessellationFactorStepFunction step) {
                return pipeline(@"toff", ^(MTLRenderPipelineDescriptor *pd) {
                    pd.tessellationFactorStepFunction = step;
                });
            };
        MTLTriangleTessellationFactorsHalf four[4] = { f4, f0, f0, f4 };
        id<MTLBuffer> stepfactors = buffer(four, sizeof four);
        const struct {
            const char *what;
            MTLTessellationFactorStepFunction step;
            bool drawn[4];
        } steps[] = {
            { "constant", MTLTessellationFactorStepFunctionConstant, { true, true, true, true } },
            { "per patch", MTLTessellationFactorStepFunctionPerPatch, { true, false, true, false } },
            { "per instance", MTLTessellationFactorStepFunctionPerInstance, { true, true, false, false } },
            { "per patch and instance", MTLTessellationFactorStepFunctionPerPatchAndPerInstance,
              { true, false, false, true } },
        };
        for (size_t i = 0; i < sizeof steps / sizeof steps[0]; i++) {
            id<MTLRenderPipelineState> ps = stepped(steps[i].step);
            if (!ps)
                return 1;
            draw(^(id<MTLRenderCommandEncoder> e) {
                [e setRenderPipelineState:ps];
                [e setVertexBuffer:small offset:0 atIndex:0];
                [e setVertexBuffer:offsets offset:0 atIndex:1];
                [e setTessellationFactorBuffer:stepfactors offset:0 instanceStride:2 * sizeof f4];
                [e drawPatches:3 patchStart:0 patchCount:2 patchIndexBuffer:nil patchIndexBufferOffset:0
                     instanceCount:2 baseInstance:0];
            }, got);
            bool drawn[4];
            ok = true;
            for (int j = 0; j < 4; j++) {
                const uint8_t *px = at(got, j & 1 ? 0.5f : -0.5f, 0.42f - (j >> 1));
                drawn[j] = px[0] || px[1] || px[2];
                ok &= drawn[j] == steps[i].drawn[j];
            }
            printf("tess factors %s %s (drawn: instance 0 patches %d %d, instance 1 patches %d %d)\n", steps[i].what,
                   ok ? "ok" : "MISMATCH", drawn[0], drawn[1], drawn[2], drawn[3]);
            bad |= !ok;
        }

        // a patch index buffer picking patch 1 still takes the factors at the draw position, patch 0's 0, which cull it
        // (as an Apple GPU does)
        MTLTriangleTessellationFactorsHalf picked[2] = { f0, f4 };
        id<MTLBuffer> pfactors = buffer(picked, sizeof picked);
        draw(^(id<MTLRenderCommandEncoder> e) {
            [e setRenderPipelineState:off];
            [e setVertexBuffer:small offset:0 atIndex:0];
            [e setVertexBuffer:offsets offset:0 atIndex:1];
            [e setTessellationFactorBuffer:pfactors offset:0 instanceStride:0];
            [e drawPatches:3 patchStart:0 patchCount:1 patchIndexBuffer:picks patchIndexBufferOffset:0
                 instanceCount:1 baseInstance:0];
        }, got);
        const uint8_t *p1 = at(got, 0.5f, 0.42f);
        ok = !lit(got);
        printf("tess patch index factors %s (patch 1's place %d %d %d%s)\n", ok ? "ok" : "MISMATCH", p1[0], p1[1],
               p1[2], lit(got) ? "; drawn, so the factors went by patch id" : "");
        bad |= !ok;
        return bad;
    }
}
