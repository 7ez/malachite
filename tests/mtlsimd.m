// SIMD-group and quad functions on MalachiteDevice, each thread's result checked against what the lanes' values give on
// the CPU: shuffles, broadcasts, reductions, prefix sums, ballots and quad shuffles.
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation tools/mtl/mtlsimd.m -o build/mtlsimd
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString *source = @"#include <metal_stdlib>\n"
                           "using namespace metal;\n"
                           "kernel void k(device uint *out [[buffer(0)]], uint t [[thread_position_in_grid]],\n"
                           "              uint lane [[thread_index_in_simdgroup]])\n"
                           "{\n"
                           "    uint v = t * 7 + 3;\n"
                           "    float f = float(v);\n"
                           "    device uint *o = out + t * 16;\n"
                           "    o[0] = simd_shuffle(v, (lane + 5) & 31);\n"
                           "    o[1] = simd_broadcast(v, 9);\n"
                           "    o[2] = simd_sum(v);\n"
                           "    o[3] = simd_max(v);\n"
                           "    o[4] = simd_prefix_exclusive_sum(v);\n"
                           "    o[5] = simd_prefix_inclusive_sum(v);\n"
                           "    o[6] = uint(simd_vote::vote_t(simd_ballot((v & 3) == 1)));\n"
                           "    o[7] = quad_shuffle(v, (lane + 1) & 3);\n"
                           "    o[8] = quad_broadcast(v, 2);\n"
                           "    o[9] = quad_sum(v);\n"
                           "    o[10] = quad_shuffle_xor(v, 1);\n"
                           "    o[11] = simd_shuffle_xor(v, 3);\n"
                           "    o[12] = simd_shuffle_down(v, 2);\n"
                           "    o[13] = simd_shuffle_up(v, 2);\n"
                           "    o[14] = uint(simd_sum(f));\n"
                           "    o[15] = simd_any(v == 10) ? 1 : 0;\n"
                           "}\n";

int main(void)
{
    @autoreleasepool {
        NSError *err;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> lib = [dev newLibraryWithSource:source options:nil error:&err];
        id<MTLComputePipelineState> ps = lib ? [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"]
                                                                                  error:&err]
                                             : nil;
        if (!ps) {
            printf("simd: no pipeline: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        enum { N = 64 };
        id<MTLBuffer> out = [dev newBufferWithLength:N * 16 * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [[dev newCommandQueue] commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(N, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint32_t *o = out.contents;
        static const char *names[16] = { "shuffle", "broadcast", "sum", "max", "prefix excl", "prefix incl", "ballot",
                                         "quad shuffle", "quad broadcast", "quad sum", "quad xor", "shuffle xor",
                                         "shuffle down", "shuffle up", "float sum", "any" };
        int bad = 0;
        for (int k = 0; k < 16; k++) {
            int wrong = 0, first = -1;
            uint32_t want0 = 0;
            for (uint32_t t = 0; t < N; t++) {
                uint32_t lane = t & 31, base = t - lane, v[32], w = 0;
                for (uint32_t i = 0; i < 32; i++)
                    v[i] = (base + i) * 7 + 3;
                uint32_t q = lane & ~3u;
                switch (k) {
                case 0: w = v[(lane + 5) & 31]; break;
                case 1: w = v[9]; break;
                case 2: for (int i = 0; i < 32; i++) w += v[i]; break;
                case 3: w = v[31]; break;
                case 4: for (uint32_t i = 0; i < lane; i++) w += v[i]; break;
                case 5: for (uint32_t i = 0; i <= lane; i++) w += v[i]; break;
                case 6: for (int i = 0; i < 32; i++) w |= ((v[i] & 3) == 1) << i; break;
                case 7: w = v[q + ((lane + 1) & 3)]; break;
                case 8: w = v[q + 2]; break;
                case 9: w = v[q] + v[q + 1] + v[q + 2] + v[q + 3]; break;
                case 10: w = v[lane ^ 1]; break;
                case 11: w = v[lane ^ 3]; break;
                case 12: w = lane + 2 < 32 ? v[lane + 2] : v[lane]; break;
                case 13: w = lane >= 2 ? v[lane - 2] : v[lane]; break;
                case 14: { float s = 0; for (int i = 0; i < 32; i++) s += (float)v[i]; w = (uint32_t)s; break; }
                case 15: for (int i = 0; i < 32; i++) w |= v[i] == 10; break;
                }
                if (o[t * 16 + k] != w) {
                    if (first < 0) {
                        first = (int)t;
                        want0 = w;
                    }
                    wrong++;
                }
            }
            bad += wrong != 0;
            if (wrong)
                printf("simd %-14s MISMATCH (%d of %d; thread %d got %u want %u)\n", names[k], wrong, N, first,
                       o[first * 16 + k], want0);
            else
                printf("simd %-14s ok\n", names[k]);
        }
        return bad;
    }
}
