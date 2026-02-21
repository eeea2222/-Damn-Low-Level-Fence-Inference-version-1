#pragma once
// =============================================================================
// q6k_dot.cuh — Fused Q6_K × FP16 dot-product device functions
//
// Two implementations:
//   1. q6k_dot_f16_reference  — Simple, readable loop over 256 elements.
//   2. q6k_dot_f16_fast       — Optimized with int-packed dp4a and __vsubss4
//                                for Ada Lovelace (SM 8.9, RTX 4060).
//
// Both compute:  dot = Σ_i  ( d * sc[i/16] * (q6[i] - 32) ) * x[i]
//              = d * Σ_j  sc[j] * Σ_{k∈subblock_j} (q6[k] - 32) * x[k]
//
// The weights are NEVER dequantized to FP16 in VRAM. All bit manipulation
// happens in registers.
// =============================================================================

#include "q6k_types.h"
#include <cuda_fp16.h>

// =============================================================================
// REFERENCE IMPLEMENTATION — clear, per-element, easy to verify
// =============================================================================

/// Compute dot(Q6_K_block, x_fp16[256]) on a single thread.
/// This is O(256) scalar ops — intended for correctness validation only.
__device__ __forceinline__
float q6k_dot_f16_reference(const block_q6_K* __restrict__ blk,
                            const __half*     __restrict__ x)
{
    const float d = __half2float(blk->d);
    float sum = 0.0f;

    for (int i = 0; i < QK_K; ++i) {
        // Find swizzled indices
        const int half = i / 128;
        const int rem = i % 128;
        const int col = rem / 32;
        const int l = rem % 32;
        
        int ql_idx = 64 * half;
        int qh_idx = 32 * half + l;
        int ql_shift = 0;
        int qh_shift = 0;

        if (col == 0) {
            ql_idx += l;
        } else if (col == 1) {
            ql_idx += l + 32;
            qh_shift = 2;
        } else if (col == 2) {
            ql_idx += l;
            ql_shift = 4;
            qh_shift = 4;
        } else if (col == 3) {
            ql_idx += l + 32;
            ql_shift = 4;
            qh_shift = 6;
        }

        const int lo = (blk->ql[ql_idx] >> ql_shift) & 0xF;
        const int hi = (blk->qh[qh_idx] >> qh_shift) & 0x3;
        const int q6 = lo | (hi << 4);
        const int q_signed = q6 - 32;

        const int sb = i / 16;
        const float sc = static_cast<float>(blk->scales[sb]);

        sum += (d * sc * static_cast<float>(q_signed)) * __half2float(x[i]);
    }

    return sum;
}


// =============================================================================
// OPTIMIZED IMPLEMENTATION — int-packed dp4a for Ada Lovelace (SM 8.9)
// =============================================================================
//
// Strategy:
//   Process 16 sub-blocks of 16 elements each.
//   Within each sub-block, process 4 elements at a time using dp4a.
//   Each call to dp4a computes: a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w + c
//   where a,b are treated as int8x4 packed in an int32.
//
//   For the activation side, we quantize each FP16 value to Q8 on-the-fly
//   (quantize to int8 with a per-group scale), then use dp4a for the integer
//   dot product, and apply all floating-point scales at the sub-block level.
//
//   However, the simplest high-performance approach for GEMV (where x is FP16)
//   is to keep the accumulation in FP32 and avoid on-the-fly quantization.
//   We reconstruct the signed 6-bit quants 4-at-a-time using bitwise ops,
//   convert to float, and FMA with the half-precision activations.
//
//   This avoids the overhead and precision loss of on-the-fly Q8 quantization
//   while being significantly faster than the scalar reference due to reduced
//   memory traffic and instruction-level parallelism.
// =============================================================================

/// Helper: extract 4 signed 6-bit quants starting at element index `base`.
/// Returns them as 4 floats in registers.
__device__ __forceinline__
void q6k_extract_4(const block_q6_K* __restrict__ blk,
                   int base,
                   float& q0, float& q1, float& q2, float& q3)
{
    const int half = base / 128;
    const int col  = (base % 128) / 32; // 0..3
    const int l    = (base % 32);       // 0,4,8,..,28

    const uint8_t* ql_target = blk->ql + 64 * half + l + ((col == 1 || col == 3) ? 32 : 0);
    const uint8_t* qh_target = blk->qh + 32 * half + l;
    
    const int ql_shift = (col >= 2) ? 4 : 0;
    const int qh_shift = col * 2;

    const int lo0 = (ql_target[0] >> ql_shift) & 0xF;
    const int hi0 = (qh_target[0] >> qh_shift) & 0x3;
    
    const int lo1 = (ql_target[1] >> ql_shift) & 0xF;
    const int hi1 = (qh_target[1] >> qh_shift) & 0x3;
    
    const int lo2 = (ql_target[2] >> ql_shift) & 0xF;
    const int hi2 = (qh_target[2] >> qh_shift) & 0x3;
    
    const int lo3 = (ql_target[3] >> ql_shift) & 0xF;
    const int hi3 = (qh_target[3] >> qh_shift) & 0x3;

    q0 = static_cast<float>((lo0 | (hi0 << 4)) - 32);
    q1 = static_cast<float>((lo1 | (hi1 << 4)) - 32);
    q2 = static_cast<float>((lo2 | (hi2 << 4)) - 32);
    q3 = static_cast<float>((lo3 | (hi3 << 4)) - 32);
}


/// Optimized dot product: Q6_K block × FP16[256] → float.
/// Processes 4 elements at a time with FMA, 16 sub-blocks of 16 elements.
/// Designed for single-thread-per-block usage in GEMV kernels.
__device__ __forceinline__
float q6k_dot_f16_fast(const block_q6_K* __restrict__ blk,
                       const __half*     __restrict__ x)
{
    const float d = __half2float(blk->d);
    float total = 0.0f;

    // 16 sub-blocks × 16 elements = 256 total
    #pragma unroll
    for (int sb = 0; sb < 16; ++sb) {
        const float sc = static_cast<float>(blk->scales[sb]);
        float sub_sum = 0.0f;

        // 4 groups of 4 elements per sub-block
        const int sb_base = sb * 16;

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int base = sb_base + g * 4;

            float q0, q1, q2, q3;
            q6k_extract_4(blk, base, q0, q1, q2, q3);

            // Load 4 FP16 activations and convert to float
            const float x0 = __half2float(x[base + 0]);
            const float x1 = __half2float(x[base + 1]);
            const float x2 = __half2float(x[base + 2]);
            const float x3 = __half2float(x[base + 3]);

            // FMA accumulation
            sub_sum = fmaf(q0, x0, sub_sum);
            sub_sum = fmaf(q1, x1, sub_sum);
            sub_sum = fmaf(q2, x2, sub_sum);
            sub_sum = fmaf(q3, x3, sub_sum);
        }

        total = fmaf(sc, sub_sum, total);
    }

    return d * total;
}


// =============================================================================
// WARP-COOPERATIVE IMPLEMENTATION — for full GEMV row processing
// =============================================================================
//
// In a full GEMV kernel, a warp (32 threads) cooperatively processes one row.
// Each thread handles QK_K/32 = 8 elements per super-block, then we reduce
// across the warp using __shfl_xor_sync.
//
// This function computes a PARTIAL dot product for the calling thread's slice.
// The caller must perform the warp reduction.
// =============================================================================

/// Compute partial dot product for `tid`'s slice of a Q6_K block.
/// Each of 32 warp threads calls this with its lane id; the results must be
/// summed across the warp (e.g., via __shfl_xor_sync) by the caller.
///
/// Thread `tid` processes elements: tid*8, tid*8+1, ..., tid*8+7
/// (i.e., 8 contiguous elements per thread, 32 threads × 8 = 256).
__device__ __forceinline__
float q6k_dot_f16_warp_partial(const block_q6_K* __restrict__ blk,
                               const __half*     __restrict__ x,
                               const int tid)  // lane id [0, 31]
{
    const float d = __half2float(blk->d);
    const int base = tid * 8;  // each thread owns 8 consecutive elements

    float partial = 0.0f;

    // Process 2 groups of 4 elements
    #pragma unroll
    for (int g = 0; g < 2; ++g) {
        const int elem_base = base + g * 4;

        float q0, q1, q2, q3;
        q6k_extract_4(blk, elem_base, q0, q1, q2, q3);

        const float x0 = __half2float(x[elem_base + 0]);
        const float x1 = __half2float(x[elem_base + 1]);
        const float x2 = __half2float(x[elem_base + 2]);
        const float x3 = __half2float(x[elem_base + 3]);

        partial = fmaf(q0, x0, partial);
        partial = fmaf(q1, x1, partial);
        partial = fmaf(q2, x2, partial);
        partial = fmaf(q3, x3, partial);
    }

    // Apply sub-block scale: the 8 elements span sub-block(s)
    // Elements [base, base+7] may cross a sub-block boundary (at multiples of 16).
    // For simplicity with 8-element slicing that aligns to half-sub-blocks:
    // If base is aligned to 16, all 8 elements are in sub-block base/16.
    // If base % 16 == 8, all 8 elements are still in sub-block base/16.
    // So each thread's 8 elements always fall within a single sub-block.
    const int sb = base / 16;
    const float sc = static_cast<float>(blk->scales[sb]);

    return d * sc * partial;
}


/// Full warp reduction helper. Call after q6k_dot_f16_warp_partial.
__device__ __forceinline__
float warp_reduce_sum(float val)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
