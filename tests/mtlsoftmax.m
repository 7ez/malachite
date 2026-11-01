// MPSGraph softmax on MalachiteDevice: each row of the result against the CPU's, and each summing to 1.
//   xcrun clang -fobjc-arc -framework Metal -framework MetalPerformanceShadersGraph -framework Foundation \
//       tools/mtl/mtlsoftmax.m -o build/mtlsoftmax
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <math.h>

int main(void)
{
    @autoreleasepool {
        enum { ROWS = 8, COLS = 1000 };
        static float in[ROWS * COLS], out[ROWS * COLS];
        for (int i = 0; i < ROWS * COLS; i++)
            in[i] = sinf(i * 0.37f) * 6.0f + (i % COLS) * 0.002f;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MPSGraph *g = [MPSGraph new];
        MPSGraphTensor *x = [g placeholderWithShape:@[ @(ROWS), @(COLS) ] dataType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor *y = [g softMaxWithTensor:x axis:1 name:nil];
        NSData *bytes = [NSData dataWithBytes:in length:sizeof(in)];
        MPSGraphTensorData *xd = [[MPSGraphTensorData alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:dev]
                                                                       data:bytes
                                                                      shape:@[ @(ROWS), @(COLS) ]
                                                                   dataType:MPSDataTypeFloat32];
        NSDictionary *res = [g runWithFeeds:@{ x : xd } targetTensors:@[ y ] targetOperations:nil];
        [[res[y] mpsndarray] readBytes:out strideBytes:nil];
        int bad = 0;
        float worst = 0, worst_sum = 1;
        for (int r = 0; r < ROWS; r++) {
            const float *a = in + r * COLS;
            float m = a[0], s = 0, sum = 0;
            for (int c = 1; c < COLS; c++)
                m = fmaxf(m, a[c]);
            for (int c = 0; c < COLS; c++)
                s += expf(a[c] - m);
            for (int c = 0; c < COLS; c++) {
                float want = expf(a[c] - m) / s, got = out[r * COLS + c];
                sum += got;
                float err = fabsf(got - want);
                if (!(err <= 1e-4f + want * 1e-2f))
                    bad++;
                if (!(err <= worst))
                    worst = err;
            }
            if (!(fabsf(sum - 1) <= fabsf(worst_sum - 1)))
                worst_sum = sum;
        }
        printf("softmax %s (%d of %d wrong, worst error %g, row sum %g)\n", bad ? "MISMATCH" : "ok", bad, ROWS * COLS,
               worst, worst_sum);
        return bad != 0;
    }
}
