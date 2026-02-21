#pragma once
// =============================================================================
// q6k_types.h — Q6_K block type for GGUF quantized weights
//
// Matches the binary layout used by llama.cpp / ggml.
// Each super-block encodes 256 weights at ~6.5625 bits per weight.
//
// Memory layout (210 bytes total):
//   ql[128]    — lower 4 bits of each 6-bit quant (2 nibbles per byte)
//   qh[64]     — upper 2 bits of each 6-bit quant (4 crumbs per byte)
//   scales[16] — signed 8-bit scale per sub-block of 16 elements
//   d          — FP16 super-block scale
//
// Reconstruction:  weight[i] = d * scales[i/16] * (q6[i] - 32)
// =============================================================================

#include <cstdint>
#include <cuda_fp16.h>

// Number of elements per super-block (K-quant standard)
#define QK_K 256

// ---- Q6_K super-block ----
// Packed identically to llama.cpp's block_q6_K.
// IMPORTANT: This struct MUST be packed to match GGUF on-disk layout.
#pragma pack(push, 1)
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // 128 bytes: lower 4 bits of quants
    uint8_t qh[QK_K / 4];      //  64 bytes: upper 2 bits of quants
    int8_t  scales[QK_K / 16]; //  16 bytes: 8-bit signed sub-block scales
    __half  d;                  //   2 bytes: FP16 super-block scale
};
#pragma pack(pop)

static_assert(sizeof(block_q6_K) == 210,
    "block_q6_K must be exactly 210 bytes to match GGUF binary layout");

// Bytes per weight (for bandwidth calculations)
constexpr float Q6K_BPW = static_cast<float>(sizeof(block_q6_K) * 8) / QK_K; // ~6.5625
