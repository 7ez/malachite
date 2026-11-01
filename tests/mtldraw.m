// End-to-end Metal render test on MalachiteDevice: clear a 64x64 RGBA8 target and draw one triangle with vertex
// colours through tools/mtl/gfx.metal's vmain/fmain, then check every pixel on the CPU.
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation tools/mtl/mtldraw.m -o build/mtldraw
//   build/mtldraw tools/mtl/gfx.metal
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define W 64
#define H 64

// count the pixels of the target that differ from want(x, y) by more than one step, printing the first few
static int check(const char *name, const uint8_t *px, NSUInteger pitch, void (^want)(int x, int y, uint8_t out[4]))
{
    int bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const uint8_t *p = px + y * pitch + x * 4;
            uint8_t w[4];
            want(x, y, w);
            for (int c = 0; c < 4; c++)
                if (abs(p[c] - w[c]) > 1) {
                    if (bad++ < 4)
                        printf("%s pixel %d,%d ch %d: got %d want %d\n", name, x, y, c, p[c], w[c]);
                    break;
                }
        }
    printf("%s %s (%d pixels wrong)\n", name, bad ? "MISMATCH" : "ok", bad);
    return bad;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    @autoreleasepool {
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> lib = src ? [dev newLibraryWithSource:src options:nil error:&err] : nil;
        if (!lib) {
            fprintf(stderr, "%s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"vmain"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fmain"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!ps) {
            printf("no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }

        // a linear texture over a buffer; the render target wants a 128-byte aligned pitch
        NSUInteger pitch = W * 4;
        id<MTLBuffer> rtbuf = [dev newBufferWithLength:pitch * H options:MTLResourceStorageModeShared];
        memset(rtbuf.contents, 0xee, pitch * H);
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:W
                                                                                    height:H
                                                                                 mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> rt = [rtbuf newTextureWithDescriptor:td offset:0 bytesPerRow:pitch];

        const float pos[3][4] = { { -1, -1, 0, 1 }, { 1, -1, 0, 1 }, { -1, 1, 0, 1 } };
        const float col[3][4] = { { 1, 0, 0, 1 }, { 0, 1, 0, 1 }, { 0, 0, 1, 1 } };
        id<MTLBuffer> pb = [dev newBufferWithLength:sizeof pos options:MTLResourceStorageModeShared];
        id<MTLBuffer> cb = [dev newBufferWithLength:sizeof col options:MTLResourceStorageModeShared];
        memcpy(pb.contents, pos, sizeof pos);
        memcpy(cb.contents, col, sizeof col);

        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = rt;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0.2, 0.4, 0.6, 1.0);

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:ps];
        [enc setVertexBuffer:pb offset:0 atIndex:0];
        [enc setVertexBuffer:cb offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // +y is up: (-1,-1) is the bottom left corner, (-1,1) the top left
        const uint8_t clear[4] = { 51, 102, 153, 255 };
        int lit = 0, bad = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                const uint8_t *p = (const uint8_t *)rtbuf.contents + y * pitch + x * 4;
                float g = (x + 0.5f) / W, b = (H - 0.5f - y) / H, r = 1 - g - b;
                if (r > -0.5f / W && r < 0.5f / W)
                    continue;  // on the diagonal edge either way is right
                int want[4] = { (int)(r * 255 + 0.5f), (int)(g * 255 + 0.5f), (int)(b * 255 + 0.5f), 255 };
                if (r < 0)
                    for (int c = 0; c < 4; c++)
                        want[c] = clear[c];
                else
                    lit++;
                for (int c = 0; c < 4; c++)
                    if (abs(p[c] - want[c]) > 3) {
                        if (bad++ < 4)
                            printf("pixel %d,%d ch %d: got %d want %d\n", x, y, c, p[c], want[c]);
                        break;
                    }
            }
        printf("draw %s (%d pixels lit, %d wrong)\n", bad ? "MISMATCH" : "ok", lit, bad);
        uint8_t *first = malloc(pitch * H);
        memcpy(first, rtbuf.contents, pitch * H);

        // the same triangle from one interleaved buffer described by a vertex descriptor
        struct {
            float x, y;
            uint8_t rgba[4];
        } verts[3] = { { -1, -1, { 255, 0, 0, 255 } }, { 1, -1, { 0, 255, 0, 255 } }, { -1, 1, { 0, 0, 255, 255 } } };
        id<MTLBuffer> vb = [dev newBufferWithLength:sizeof verts options:MTLResourceStorageModeShared];
        memcpy(vb.contents, verts, sizeof verts);
        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        vd.attributes[0].format = MTLVertexFormatFloat2;
        vd.attributes[0].offset = 0;
        vd.attributes[0].bufferIndex = 0;
        vd.attributes[1].format = MTLVertexFormatUChar4Normalized;
        vd.attributes[1].offset = 8;
        vd.attributes[1].bufferIndex = 0;
        vd.layouts[0].stride = sizeof verts[0];
        pd.vertexFunction = [lib newFunctionWithName:@"vattr"];
        pd.vertexDescriptor = vd;
        id<MTLRenderPipelineState> aps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!aps) {
            printf("stage_in: no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        memset(rtbuf.contents, 0xee, pitch * H);
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:aps];
        [enc setVertexBuffer:vb offset:0 atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        int abad = 0;
        for (int i = 0; i < H * (int)pitch; i++)
            if (abs(((const uint8_t *)rtbuf.contents)[i] - first[i]) > 1 && abad++ < 4)
                printf("stage_in byte %d: got %d want %d\n", i, ((const uint8_t *)rtbuf.contents)[i], first[i]);
        free(first);
        printf("stage_in %s (%d bytes differ from the first draw)\n", abad ? "MISMATCH" : "ok", abad);
        pd.vertexDescriptor = nil;
        bad |= abad;

        // a fullscreen pass whose fragment shader copies a pattern texture by its pixel position
        pd.vertexFunction = [lib newFunctionWithName:@"vfull"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fcopy"];
        id<MTLRenderPipelineState> cps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!cps) {
            printf("copy: no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLBuffer> srcbuf = [dev newBufferWithLength:pitch * H options:MTLResourceStorageModeShared];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                uint8_t *p = (uint8_t *)srcbuf.contents + y * pitch + x * 4;
                p[0] = x * 4;
                p[1] = y * 4;
                p[2] = (x ^ y) * 4;
                p[3] = 255;
            }
        td.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> src2 = [srcbuf newTextureWithDescriptor:td offset:0 bytesPerRow:pitch];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        sd.normalizedCoordinates = NO;
        id<MTLSamplerState> samp = [dev newSamplerStateWithDescriptor:sd];

        memset(rtbuf.contents, 0, pitch * H);
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:cps];
        [enc setFragmentTexture:src2 atIndex:0];
        [enc setFragmentSamplerState:samp atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        int cbad = 0;
        for (int i = 0; i < H * (int)pitch; i++) {
            uint8_t got = ((const uint8_t *)rtbuf.contents)[i], want = ((const uint8_t *)srcbuf.contents)[i];
            if (got != want && cbad++ < 4)
                printf("copy byte %d (pixel %d,%d): got %d want %d\n", i, i % (int)pitch / 4, i / (int)pitch, got,
                       want);
        }
        printf("copy %s (%d bytes wrong)\n", cbad ? "MISMATCH" : "ok", cbad);

        // half-transparent red blended over the copy: src * a + dst * (1 - a), alpha src + dst * (1 - a)
        pd.fragmentFunction = [lib newFunctionWithName:@"fhalf"];
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        id<MTLRenderPipelineState> bps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!bps) {
            printf("blend: no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:bps];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        int bbad = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                const uint8_t *p = (const uint8_t *)rtbuf.contents + y * pitch + x * 4;
                const uint8_t *s = (const uint8_t *)srcbuf.contents + y * pitch + x * 4;
                int want[4] = { (255 + s[0]) / 2, s[1] / 2, s[2] / 2, 255 };
                for (int c = 0; c < 4; c++)
                    if (abs(p[c] - want[c]) > 1) {
                        if (bbad++ < 4)
                            printf("blend pixel %d,%d ch %d: got %d want %d\n", x, y, c, p[c], want[c]);
                        break;
                    }
            }
        printf("blend %s (%d pixels wrong)\n", bbad ? "MISMATCH" : "ok", bbad);

        // the copy again between textures the device made, in BGRA8, through replaceRegion and getBytes
        MTLTextureDescriptor *bd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                     width:W
                                                                                    height:H
                                                                                 mipmapped:NO];
        bd.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> dsrc = [dev newTextureWithDescriptor:bd];
        bd.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> ddst = [dev newTextureWithDescriptor:bd];
        if (!dsrc || !ddst) {
            printf("device textures: not made\n");
            return 1;
        }
        [dsrc replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:srcbuf.contents bytesPerRow:pitch];
        pd.fragmentFunction = [lib newFunctionWithName:@"fcopy"];
        pd.colorAttachments[0].blendingEnabled = NO;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        id<MTLRenderPipelineState> dps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        rp.colorAttachments[0].texture = ddst;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:dps];
        [enc setFragmentTexture:dsrc atIndex:0];
        [enc setFragmentSamplerState:samp atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        uint8_t *back = calloc(H, pitch);
        [ddst getBytes:back bytesPerRow:pitch fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
        int dbad = 0;
        for (int i = 0; i < H * (int)pitch; i++)
            if (back[i] != ((const uint8_t *)srcbuf.contents)[i] && dbad++ < 4)
                printf("device copy byte %d: got %d want %d\n", i, back[i], ((const uint8_t *)srcbuf.contents)[i]);
        free(back);
        printf("device copy %s (%d bytes wrong)\n", dbad ? "MISMATCH" : "ok", dbad);

        // once more with the sampler the shader declares (constexpr), and no sampler bound
        pd.fragmentFunction = [lib newFunctionWithName:@"fcopy_cx"];
        id<MTLRenderPipelineState> xps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!xps) {
            printf("constexpr: no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        uint8_t zero[W * 4] = { 0 };
        for (int y = 0; y < H; y++)
            [ddst replaceRegion:MTLRegionMake2D(0, y, W, 1) mipmapLevel:0 withBytes:zero bytesPerRow:W * 4];
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:xps];
        [enc setFragmentTexture:dsrc atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        back = calloc(H, pitch);
        [ddst getBytes:back bytesPerRow:pitch fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
        int xbad = 0;
        for (int i = 0; i < H * (int)pitch; i++)
            if (back[i] != ((const uint8_t *)srcbuf.contents)[i] && xbad++ < 4)
                printf("constexpr copy byte %d: got %d want %d\n", i, back[i], ((const uint8_t *)srcbuf.contents)[i]);
        free(back);
        printf("constexpr copy %s (%d bytes wrong)\n", xbad ? "MISMATCH" : "ok", xbad);

        // derivatives: (d/dx x/4, d/dy y/2, d/dx y) = (0.25, 0.5, 0) at every pixel
        pd.fragmentFunction = [lib newFunctionWithName:@"fderiv"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        id<MTLRenderPipelineState> gps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!gps) {
            printf("deriv: no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        rp.colorAttachments[0].texture = rt;
        memset(rtbuf.contents, 0, pitch * H);
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:gps];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        int gbad = 0;
        const uint8_t dwant[4] = { 64, 128, 0, 255 };
        for (int i = 0; i < W * H; i++) {
            const uint8_t *p = (const uint8_t *)rtbuf.contents + (i / W) * pitch + (i % W) * 4;
            for (int c = 0; c < 4; c++)
                if (abs(p[c] - dwant[c]) > 1) {
                    if (gbad++ < 4)
                        printf("deriv pixel %d,%d ch %d: got %d want %d\n", i % W, i / W, c, p[c], dwant[c]);
                    break;
                }
        }
        printf("deriv %s (%d pixels wrong)\n", gbad ? "MISMATCH" : "ok", gbad);

        // framebuffer fetch: clear to (0.2, 0.4, 0.6, 1), then a draw returning 1 - [[color(0)]]
        pd.fragmentFunction = [lib newFunctionWithName:@"finvert"];
        id<MTLRenderPipelineState> ips = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!ips) {
            printf("fetch: no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0.2, 0.4, 0.6, 1.0);
        memset(rtbuf.contents, 0, pitch * H);
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:ips];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        int fbad = 0;
        const uint8_t fwant[4] = { 204, 153, 102, 0 };
        for (int i = 0; i < W * H; i++) {
            const uint8_t *p = (const uint8_t *)rtbuf.contents + (i / W) * pitch + (i % W) * 4;
            for (int c = 0; c < 4; c++)
                if (abs(p[c] - fwant[c]) > 1) {
                    if (fbad++ < 4)
                        printf("fetch pixel %d,%d ch %d: got %d want %d\n", i % W, i / W, c, p[c], fwant[c]);
                    break;
                }
        }
        printf("fetch %s (%d pixels wrong)\n", fbad ? "MISMATCH" : "ok", fbad);

        // an indexed quad, four instances, its corners two entries into the buffer (base vertex 2)
        pd.colorAttachments[0].blendingEnabled = NO;
        pd.vertexFunction = [lib newFunctionWithName:@"vquad"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fflat"];
        id<MTLRenderPipelineState> qps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!qps) {
            printf("instanced: no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        const float corners[6][2] = { { 9, 9 }, { 9, 9 }, { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
        const uint16_t indices[6] = { 0, 1, 2, 2, 1, 3 };
        id<MTLBuffer> qb = [dev newBufferWithBytes:corners length:sizeof corners options:MTLResourceStorageModeShared];
        id<MTLBuffer> ib = [dev newBufferWithBytes:indices length:sizeof indices options:MTLResourceStorageModeShared];
        memset(rtbuf.contents, 0, pitch * H);
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:qps];
        [enc setVertexBuffer:qb offset:0 atIndex:0];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:6
                         indexType:MTLIndexTypeUInt16
                       indexBuffer:ib
                 indexBufferOffset:0
                     instanceCount:4
                        baseVertex:2
                      baseInstance:0];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        int ibad = check("instanced", rtbuf.contents, pitch, ^(int x, int y, uint8_t out[4]) {
            int iid = (x >= W / 2) + (y < H / 2) * 2;  // the top half is instances 2 and 3
            out[0] = iid == 0 ? 255 : 0;
            out[1] = iid == 1 ? 255 : 0;
            out[2] = iid == 2 ? 255 : 0;
            out[3] = 255;
        });

        // an indexed strip cut by the restart index: the first triangle of the quad, then one vertex that makes
        // nothing; without the restart the strip goes on through the vertex the index reads to corner 3
        const uint16_t strip[5] = { 0, 1, 2, 0xffff, 3 };
        id<MTLBuffer> sb = [dev newBufferWithBytes:strip length:sizeof strip options:MTLResourceStorageModeShared];
        memset(rtbuf.contents, 0, pitch * H);
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:qps];
        [enc setVertexBuffer:qb offset:0 atIndex:0];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangleStrip
                        indexCount:5
                         indexType:MTLIndexTypeUInt16
                       indexBuffer:sb
                 indexBufferOffset:0
                     instanceCount:1
                        baseVertex:2
                      baseInstance:0];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        const uint8_t *rbg = clear;
        int rbad = check("restart", rtbuf.contents, pitch, ^(int x, int y, uint8_t out[4]) {
            float u = (x + 0.5f) / (W / 2), v = 2 - (y + 0.5f) / (H / 2);  // the quad's corner space
            int in = x < W / 2 && y >= H / 2 && u + v < 1;
            if (fabsf(u + v - 1) < 2.0f / (W / 2)) {  // the diagonal edge: whatever was drawn
                const uint8_t *p = (const uint8_t *)rtbuf.contents + y * pitch + x * 4;
                in = p[0] > 127;
            }
            static const uint8_t red[4] = { 255, 0, 0, 255 };
            memcpy(out, in ? red : rbg, 4);
        });

        // a full-target triangle through a viewport on the right half, cut by a scissor at three quarters
        pd.vertexFunction = [lib newFunctionWithName:@"vfull"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fhalf"];
        id<MTLRenderPipelineState> vps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        memset(rtbuf.contents, 0, pitch * H);
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:vps];
        [enc setViewport:(MTLViewport){ W / 2, 0, W / 2, H, 0, 1 }];
        [enc setScissorRect:(MTLScissorRect){ 0, 0, W * 3 / 4, H }];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        const uint8_t *bg = clear;
        int vbad = check("viewport", rtbuf.contents, pitch, ^(int x, int y, uint8_t out[4]) {
            static const uint8_t drawn[4] = { 255, 0, 0, 128 };
            memcpy(out, x >= W / 2 && x < W * 3 / 4 ? drawn : bg, 4);
        });
        // depth: one full-target triangle at a time through vmain (positions, flat colours), depth test less;
        // ZD32=1 tests a depth-only target and skips the stencil test
        int stencil = !getenv("ZD32");
        MTLPixelFormat zf = stencil ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
        MTLTextureDescriptor *zd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:zf
                                                                                     width:W
                                                                                    height:H
                                                                                 mipmapped:NO];
        zd.usage = MTLTextureUsageRenderTarget;
        zd.storageMode = MTLStorageModePrivate;
        id<MTLTexture> zt = [dev newTextureWithDescriptor:zd];
        pd.vertexFunction = [lib newFunctionWithName:@"vmain"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fmain"];
        pd.depthAttachmentPixelFormat = zf;
        pd.stencilAttachmentPixelFormat = stencil ? zf : MTLPixelFormatInvalid;
        id<MTLRenderPipelineState> zps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        pd.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
        id<MTLRenderPipelineState> sps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        pd.colorAttachments[0].writeMask = MTLColorWriteMaskAll;
        if (!zt || !zps || !sps) {
            printf("depth: no texture or pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        // a triangle covering the target at depth z, and its flat colour
        id<MTLBuffer> (^tri)(float) = ^(float z) {
            const float p[3][4] = { { -1, -1, z, 1 }, { 3, -1, z, 1 }, { -1, 3, z, 1 } };
            return [dev newBufferWithBytes:p length:sizeof p options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> (^flat)(float, float, float) = ^(float r, float g, float b) {
            const float c[3][4] = { { r, g, b, 1 }, { r, g, b, 1 }, { r, g, b, 1 } };
            return [dev newBufferWithBytes:c length:sizeof c options:MTLResourceStorageModeShared];
        };
        MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
        dsd.depthCompareFunction = MTLCompareFunctionLess;
        dsd.depthWriteEnabled = YES;
        id<MTLDepthStencilState> less = [dev newDepthStencilStateWithDescriptor:dsd];
        MTLRenderPassDescriptor *zp = [MTLRenderPassDescriptor renderPassDescriptor];
        zp.colorAttachments[0].texture = rt;
        zp.colorAttachments[0].loadAction = MTLLoadActionClear;
        zp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        zp.depthAttachment.texture = zt;
        zp.depthAttachment.loadAction = MTLLoadActionClear;
        zp.depthAttachment.clearDepth = 1.0;
        zp.stencilAttachment.texture = stencil ? zt : nil;
        zp.stencilAttachment.loadAction = MTLLoadActionClear;
        zp.stencilAttachment.clearStencil = 0;
        memset(rtbuf.contents, 0, pitch * H);
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:zp];
        [enc setRenderPipelineState:zps];
        [enc setDepthStencilState:less];
        [enc setVertexBuffer:tri(0.5f) offset:0 atIndex:0];
        [enc setVertexBuffer:flat(1, 0, 0) offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc setVertexBuffer:tri(0.75f) offset:0 atIndex:0];  // behind: rejected
        [enc setVertexBuffer:flat(0, 1, 0) offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc setScissorRect:(MTLScissorRect){ 0, 0, W / 2, H }];
        [enc setVertexBuffer:tri(0.25f) offset:0 atIndex:0];  // in front, left half
        [enc setVertexBuffer:flat(0, 0, 1) offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        int zbad = check("depth", rtbuf.contents, pitch, ^(int x, int y, uint8_t out[4]) {
            out[0] = x < W / 2 ? 0 : 255;
            out[1] = 0;
            out[2] = x < W / 2 ? 255 : 0;
            out[3] = 255;
        });

        if (!stencil)
            return bad || cbad || bbad || dbad || xbad || gbad || fbad || ibad || rbad || vbad || zbad;

        // stencil: mark the top half with 1 (no colour written), then draw green where it is 1
        MTLDepthStencilDescriptor *mark = [MTLDepthStencilDescriptor new];
        mark.frontFaceStencil.depthStencilPassOperation = MTLStencilOperationReplace;
        mark.backFaceStencil.depthStencilPassOperation = MTLStencilOperationReplace;
        MTLDepthStencilDescriptor *test = [MTLDepthStencilDescriptor new];
        test.frontFaceStencil.stencilCompareFunction = MTLCompareFunctionEqual;
        test.backFaceStencil.stencilCompareFunction = MTLCompareFunctionEqual;
        memset(rtbuf.contents, 0, pitch * H);
        cmd = [q commandBuffer];
        enc = [cmd renderCommandEncoderWithDescriptor:zp];
        [enc setStencilReferenceValue:1];
        [enc setRenderPipelineState:sps];
        [enc setDepthStencilState:[dev newDepthStencilStateWithDescriptor:mark]];
        [enc setScissorRect:(MTLScissorRect){ 0, 0, W, H / 2 }];
        [enc setVertexBuffer:tri(0.5f) offset:0 atIndex:0];
        [enc setVertexBuffer:flat(1, 0, 0) offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc setRenderPipelineState:zps];
        [enc setDepthStencilState:[dev newDepthStencilStateWithDescriptor:test]];
        [enc setScissorRect:(MTLScissorRect){ 0, 0, W, H }];
        [enc setVertexBuffer:flat(0, 1, 0) offset:0 atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        int sbad = check("stencil", rtbuf.contents, pitch, ^(int x, int y, uint8_t out[4]) {
            out[0] = 0;
            out[1] = y < H / 2 ? 255 : 0;
            out[2] = 0;
            out[3] = 255;
        });
        return bad || cbad || bbad || dbad || xbad || gbad || fbad || ibad || rbad || vbad || zbad || sbad;
    }
}
