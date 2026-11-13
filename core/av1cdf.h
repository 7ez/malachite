// NVDEC's AV1 probability table, read per frame and written back adapted: CDFs as 16-bit 32768 - value, last dropped,
// no counters, each table at a fixed byte offset and stride. Layout is NVIDIA's CUVID one, as mesa-nova's NVK AV1
// decoder found it (src/nouveau/vulkan/rust/video/decode/av1_probs.rs); bit-exact on GA10x.
#pragma once
#include <stdint.h>

#define AV1_PROBS_SIZE 0x2fc0

// defaults for a frame with no primary reference; coefficient CDFs picked by base_qindex
void av1_default_probs(uint32_t base_qindex, uint8_t *table);
