// The probability table NVDEC decodes an AV1 frame with and writes back adapted once the frame is decoded: every CDF
// as 16-bit inverse values (32768 minus each but the last), without libaom's counters, each table of CDFs at a fixed
// byte offset and stride. The layout is the one NVIDIA's CUVID library builds, as mesa-nova's NVK AV1 decoder found
// it (src/nouveau/vulkan/rust/video/decode/av1_probs.rs), which decodes bit-exact on GA10x.
#pragma once
#include <stdint.h>

#define AV1_PROBS_SIZE 0x2fc0

// the table a frame without a primary reference frame starts from: the defaults, the coefficient ones for its
// base_q_idx
void av1_default_probs(uint32_t base_qindex, uint8_t *table);
