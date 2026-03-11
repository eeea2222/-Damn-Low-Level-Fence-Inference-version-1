#pragma once
// Q8_0 block type for GGUF quantized weights (used for token embeddings)
// 32 elements per block, 34 bytes. val[i] = d * qs[i]

#include <cstdint>
#include <cuda_fp16.h>

#define QK8_0 32

#pragma pack(push, 1)
struct block_q8_0 {
    __half  d;          // 2 bytes: FP16 scale
    int8_t  qs[QK8_0];  // 32 bytes: quantized values
};
#pragma pack(pop)

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");
