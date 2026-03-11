#pragma once
// =============================================================================
// gemv_q8_0.cuh — Q8_0 Matrix-Vector Multiply CUDA Kernel
// Used for: embedding lookup (single row) and LM head (151936×2560)
// =============================================================================

#include "q8_0_types.h"
#include <cuda_fp16.h>

// ---- Single-row dequantize (embedding lookup) ----
// Dequantize row `row_id` from Q8_0 matrix into float output
__global__ void embed_q8_0_kernel(
    float*              __restrict__ out,
    const block_q8_0*  __restrict__ weight,
    const int row_id,
    const int dim)  // embedding dimension
{
    const int blocks_per_row = dim / QK8_0;
    const block_q8_0* row = weight + static_cast<size_t>(row_id) * blocks_per_row;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) return;

    const int block_idx = tid / QK8_0;
    const int elem_idx  = tid % QK8_0;

    const float d = __half2float(row[block_idx].d);
    out[tid] = d * static_cast<float>(row[block_idx].qs[elem_idx]);
}

inline void embed_q8_0(float* d_out, const block_q8_0* d_weight,
                        int token_id, int dim, cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    embed_q8_0_kernel<<<blocks, threads, 0, stream>>>(d_out, d_weight, token_id, dim);
}

// ---- Warp-cooperative Q8_0 dot product (for GEMV) ----
// Each thread in a warp handles (dim/32) elements
__device__ __forceinline__
float q8_0_dot_f32_warp_partial(const block_q8_0* __restrict__ row_blocks,
                                 const float*     __restrict__ x,
                                 const int blocks_per_row,
                                 const int lane_id)
{
    float sum = 0.0f;
    // Each lane processes blocks: lane_id, lane_id+32, lane_id+64, ...
    for (int b = lane_id; b < blocks_per_row; b += 32) {
        const float d = __half2float(row_blocks[b].d);
        const float* x_seg = x + b * QK8_0;

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            block_sum += static_cast<float>(row_blocks[b].qs[i]) * x_seg[i];
        }
        sum += d * block_sum;
    }
    return sum;
}

__device__ __forceinline__
float warp_reduce_sum_q8(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---- Full GEMV: y[M] = W_q8_0[M × K] · x[K] ----
#ifndef GEMV_Q8_WARPS_PER_BLOCK
#define GEMV_Q8_WARPS_PER_BLOCK 4
#endif

__global__ void gemv_q8_0_kernel(
    const block_q8_0* __restrict__ W,
    const float*      __restrict__ x,
    float*            __restrict__ y,
    const int M,
    const int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * GEMV_Q8_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    const int blocks_per_row = K / QK8_0;
    const block_q8_0* row_blocks = W + static_cast<size_t>(row) * blocks_per_row;

    float partial = q8_0_dot_f32_warp_partial(row_blocks, x, blocks_per_row, lane_id);
    float total = warp_reduce_sum_q8(partial);

    if (lane_id == 0) {
        y[row] = total;
    }
}

inline void gemv_q8_0(const block_q8_0* d_W, const float* d_x, float* d_y,
                       int M, int K, cudaStream_t stream = 0) {
    const int threads = GEMV_Q8_WARPS_PER_BLOCK * 32;
    const int blocks = (M + GEMV_Q8_WARPS_PER_BLOCK - 1) / GEMV_Q8_WARPS_PER_BLOCK;
    gemv_q8_0_kernel<<<blocks, threads, 0, stream>>>(d_W, d_x, d_y, M, K);
}
