#pragma once
// =============================================================================
// gemv_q6k.cuh — Fused Q6_K Matrix-Vector Multiply CUDA Kernel
//
// Computes:  y[M] = W_q6k[M × K] · x[K]
//
// Where:
//   W is stored as Q6_K blocks (each block = 256 elements, 210 bytes)
//   x is FP16 (the activation vector)
//   y is FP32 (the output vector)
//
// Launch configuration:
//   - WARPS_PER_BLOCK warps per thread block (each warp = 32 threads)
//   - 1 warp per output row
//   - Each warp iterates over K/256 Q6_K blocks along its row
//
// Target: RTX 4060 (Ada Lovelace, SM 8.9)
// =============================================================================

#include "q6k_types.h"
#include "q6k_dot.cuh"
#include <cuda_fp16.h>

// Number of warps per thread block
// 4 warps = 128 threads — good occupancy on SM 8.9 (max 48 warps/SM)
#ifndef GEMV_Q6K_WARPS_PER_BLOCK
#define GEMV_Q6K_WARPS_PER_BLOCK 4
#endif

#define GEMV_Q6K_THREADS_PER_BLOCK (GEMV_Q6K_WARPS_PER_BLOCK * 32)

// =============================================================================
// Main GEMV kernel: y[M] = W_q6k[M × K] · x_fp16[K]
//
// W_q6k layout: row-major, M rows, each row has (K / 256) Q6_K blocks
//               stored contiguously.
//
// Parameters:
//   W      - Q6_K weight matrix [M rows × (K/256) blocks per row]
//   x      - FP16 input vector [K]
//   y      - FP32 output vector [M]
//   M      - number of output rows
//   K      - input dimension (must be multiple of 256)
// =============================================================================
__global__ void gemv_q6k_kernel(
    const block_q6_K* __restrict__ W,
    const __half*     __restrict__ x,
    float*            __restrict__ y,
    const int M,
    const int K)
{
    // Identify which row this warp handles
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * GEMV_Q6K_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    const int blocks_per_row = K / QK_K;  // K / 256

    // Pointer to this row's Q6_K blocks
    const block_q6_K* row_blocks = W + static_cast<size_t>(row) * blocks_per_row;

    float row_sum = 0.0f;

    // Iterate over all Q6_K blocks in this row
    for (int b = 0; b < blocks_per_row; ++b) {
        const block_q6_K* blk = &row_blocks[b];
        const __half* x_seg = x + b * QK_K;  // x offset for this block

        // Each lane computes partial dot over its 8 elements
        float partial = q6k_dot_f16_warp_partial(blk, x_seg, lane_id);
        row_sum += partial;
    }

    // Warp-level reduction
    row_sum = warp_reduce_sum(row_sum);

    // Lane 0 writes output
    if (lane_id == 0) {
        y[row] = row_sum;
    }
}

// =============================================================================
// Host-side launch wrapper
// =============================================================================

inline void gemv_q6k(
    const block_q6_K* d_W,    // device: Q6_K weight matrix
    const __half*     d_x,    // device: FP16 input vector
    float*            d_y,    // device: FP32 output vector
    int M,                    // output dimension (rows)
    int K,                    // input dimension (cols, must be multiple of 256)
    cudaStream_t stream = 0)
{
    const int warps_per_block = GEMV_Q6K_WARPS_PER_BLOCK;
    const int threads = GEMV_Q6K_THREADS_PER_BLOCK;
    const int blocks = (M + warps_per_block - 1) / warps_per_block;

    gemv_q6k_kernel<<<blocks, threads, 0, stream>>>(d_W, d_x, d_y, M, K);
}

// =============================================================================
// Bias-fused variant: y[M] = W_q6k[M × K] · x[K] + bias[M]
// =============================================================================
__global__ void gemv_q6k_bias_kernel(
    const block_q6_K* __restrict__ W,
    const __half*     __restrict__ x,
    const float*      __restrict__ bias,
    float*            __restrict__ y,
    const int M,
    const int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * GEMV_Q6K_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    const int blocks_per_row = K / QK_K;
    const block_q6_K* row_blocks = W + static_cast<size_t>(row) * blocks_per_row;

    float row_sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        float partial = q6k_dot_f16_warp_partial(&row_blocks[b],
                                                  x + b * QK_K,
                                                  lane_id);
        row_sum += partial;
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        y[row] = row_sum + bias[row];
    }
}

inline void gemv_q6k_bias(
    const block_q6_K* d_W,
    const __half*     d_x,
    const float*      d_bias,
    float*            d_y,
    int M, int K,
    cudaStream_t stream = 0)
{
    const int threads = GEMV_Q6K_THREADS_PER_BLOCK;
    const int blocks = (M + GEMV_Q6K_WARPS_PER_BLOCK - 1) / GEMV_Q6K_WARPS_PER_BLOCK;
    gemv_q6k_bias_kernel<<<blocks, threads, 0, stream>>>(d_W, d_x, d_bias, d_y, M, K);
}
