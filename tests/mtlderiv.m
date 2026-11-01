// Derivatives along the edge two triangles share: a quad of two triangles whose fragment function writes the
// screen-space derivatives of its interpolated coordinates, or of a value loaded from a buffer, which are the same at
// every pixel, the edge's included (the threads of the other triangle's pixels there are helpers).
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation tools/mtl/mtlderiv.m -o build/mtlderiv
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>

static NSString *source = @"#include <metal_stdlib>\n"
                           "using namespace metal;\n"
                           "struct V { float4 pos [[position]]; float2 uv; };\n"
                           "vertex V vs(uint id [[vertex_id]], constant float2 *p [[buffer(0)]])\n"
                           "{\n"
                           "    V o;\n"
                           "    o.pos = float4(p[id] * 2 - 1, 0, 1);\n"
                           "    o.uv = p[id] * 64;\n"
                           "    return o;\n"
                           "}\n"
                           "fragment float4 fs(V in [[stage_in]])\n"
                           "{\n"
                           "    float2 dx = dfdx(in.uv), dy = dfdy(in.uv);\n"
                           "    return float4(dx.x, dy.y, fwidth(in.uv.x), fabs(dx.y) + fabs(dy.x));\n"
                           "}\n"
                           "// the same value made on two paths a pixel picks by where it is, its derivative taken once they join\n"
                           "__attribute__((noinline)) float pick(float2 uv)\n"
                           "{\n"
                           "    if (uv.x - uv.y > 0.5)\n"
                           "        return uv.x + sin(uv.y * 0.0001) * 0.0001;\n"
                           "    return uv.x;\n"
                           "}\n"
                           "fragment float4 fs_branch(V in [[stage_in]])\n"
                           "{\n"
                           "    float d = pick(in.uv);\n"
                           "    float w = fwidth(d);\n"
                           "    return float4(dfdx(d), 0, w, 0);\n"
                           "}\n"
                           "// the same through a value loaded from a buffer, which a thread outside the primitive reads too\n"
                           "fragment float4 fs_load(V in [[stage_in]], constant float &k [[buffer(0)]])\n"
                           "{\n"
                           "    float d = in.uv.x * k;\n"
                           "    return float4(dfdx(d), 0, fwidth(d), 0);\n"
                           "}\n";

int main(void)
{
    @autoreleasepool {
        NSError *err;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> lib = [dev newLibraryWithSource:source options:nil error:&err];
        enum { W = 64 };
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                      width:W
                                                                                     height:W
                                                                                  mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> rt = [dev newTextureWithDescriptor:td];
        // two triangles sharing the diagonal from (0,0) to (1,1)
        static const float quad[12] = { 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1 };
        id<MTLBuffer> out = [dev newBufferWithLength:W * W * 16 options:MTLResourceStorageModeShared];
        int bad = 0;
        static const char *const names[] = { "fs", "fs_branch", "fs_load" };
        for (int mode = 0; mode < 3; mode++) {
            MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [lib newFunctionWithName:@"vs"];
            pd.fragmentFunction = [lib newFunctionWithName:@(names[mode])];
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
            id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
            if (!ps) {
                printf("deriv: no pipeline: %s\n", err.localizedDescription.UTF8String);
                return 1;
            }
            id<MTLCommandBuffer> cb = [[dev newCommandQueue] commandBuffer];
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = rt;
            rp.colorAttachments[0].loadAction = MTLLoadActionClear;
            rp.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
            [e setRenderPipelineState:ps];
            [e setVertexBytes:quad length:sizeof(quad) atIndex:0];
            static const float one = 1;
            [e setFragmentBytes:&one length:sizeof(one) atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            [e endEncoding];
            id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
            [b copyFromTexture:rt sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                        sourceSize:MTLSizeMake(W, W, 1) toBuffer:out destinationOffset:0
                destinationBytesPerRow:W * 16 destinationBytesPerImage:W * W * 16];
            [b endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const float *p = out.contents;
            int edge_bad = 0, inside_bad = 0, first = -1;
            for (int y = 0; y < W; y++)
                for (int x = 0; x < W; x++) {
                    const float *c = p + (y * W + x) * 4;
                    // uv grows by 1 a pixel across, and by -1 a pixel down (the viewport flips y)
                    bool ok = mode ? fabsf(c[0] - 1) < 1e-2f && fabsf(c[2] - 1) < 1e-2f
                                     : fabsf(c[0] - 1) < 1e-3f && fabsf(fabsf(c[1]) - 1) < 1e-3f &&
                                           fabsf(c[2] - 1) < 1e-3f && c[3] < 1e-3f;
                    if (ok)
                        continue;
                    if (abs(x - (W - 1 - y)) <= 1)
                        edge_bad++;
                    else
                        inside_bad++;
                    if (first < 0)
                        first = y * W + x;
                }
            static const char *const what_[] = { "deriv", "deriv after a branch", "deriv of a loaded value" };
            const char *what = what_[mode];
            if (edge_bad || inside_bad) {
                const float *c = p + first * 4;
                printf("%s MISMATCH (%d pixels by the shared edge, %d elsewhere; (%d,%d) dx %g dy %g fwidth %g cross %g)\n",
                       what, edge_bad, inside_bad, first % W, first / W, c[0], c[1], c[2], c[3]);
            } else
                printf("%s ok (all %d pixels, the shared edge's included)\n", what, W * W);
            bad |= edge_bad || inside_bad;
        }
        return bad;
    }
}
