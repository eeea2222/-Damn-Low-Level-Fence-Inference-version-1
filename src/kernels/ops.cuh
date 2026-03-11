#pragma once
// =============================================================================
// ops.cuh — Helper CUDA kernels for Qwen3 inference
//
// All operations work on FP32 buffers (hidden states stay in FP32 throughout
// the forward pass for numerical stability; FP16 is only used for weight
// storage and KV cache).
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// =============================================================================
// RMSNorm:  out[i] = w[i] * x[i] / sqrt(mean(x^2) + eps)
// =============================================================================
__global__ void rms_norm_kernel(
    float*       __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const int n,
    const float eps)
{
    // Single block, cooperative reduction
    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += x[i] * x[i];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / static_cast<float>(n) + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = w[i] * x[i] * rms;
    }
}

inline void rms_norm(float* out, const float* x, const float* w,
                     int n, float eps, cudaStream_t stream = 0) {
    const int threads = (n < 1024) ? n : 1024;
    rms_norm_kernel<<<1, threads, threads * sizeof(float), stream>>>(out, x, w, n, eps);
}

// In-place variant
inline void rms_norm_inplace(float* x, const float* w,
                              int n, float eps, cudaStream_t stream = 0) {
    // Use out-of-place with a temp buffer, or we do it differently
    // For simplicity, we allow out == x in the kernel (reads before writes per-element
    // after the reduction completes, so this is safe)
    rms_norm(x, x, w, n, eps, stream);
}

// =============================================================================
// Per-head RMSNorm (for QK-Norm)
// Apply RMSNorm independently to each head of size head_dim
// =============================================================================
__global__ void rms_norm_per_head_kernel(
    float*       __restrict__ x,     // [n_heads * head_dim], modified in place
    const float* __restrict__ w,     // [head_dim], shared across heads
    const int head_dim,
    const float eps)
{
    // One block per head
    const int head = blockIdx.x;
    float* head_x = x + head * head_dim;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        local_sum += head_x[i] * head_x[i];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / static_cast<float>(head_dim) + eps);

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head_x[i] = w[i] * head_x[i] * rms;
    }
}

inline void rms_norm_per_head(float* x, const float* w,
                               int n_heads, int head_dim, float eps,
                               cudaStream_t stream = 0) {
    const int threads = (head_dim < 256) ? head_dim : 256;
    rms_norm_per_head_kernel<<<n_heads, threads, threads * sizeof(float), stream>>>(
        x, w, head_dim, eps);
}

// =============================================================================
// RoPE (Rotary Positional Embedding)
// Applied to Q and K after projection, before attention
//
// For each head, for dimension pair (i, i+1) where i is even:
//   freq = 1.0 / (freq_base ^ (i / head_dim))
//   cos_val = cos(pos * freq), sin_val = sin(pos * freq)
//   q[i]   = q[i] * cos_val - q[i+1] * sin_val
//   q[i+1] = q[i] * sin_val + q[i+1] * cos_val
// =============================================================================
__global__ void apply_rope_kernel(
    float* __restrict__ q,    // [n_q_heads * head_dim]
    float* __restrict__ k,    // [n_kv_heads * head_dim]
    const int head_dim,
    const int n_q_heads,
    const int n_kv_heads,
    const int pos,
    const float freq_base)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;

    // Process Q heads
    const int total_q_pairs = n_q_heads * half_dim;
    if (tid < total_q_pairs) {
        const int head = tid / half_dim;
        const int pair = tid % half_dim;

        const float freq = 1.0f / powf(freq_base, static_cast<float>(2 * pair) / static_cast<float>(head_dim));
        const float angle = static_cast<float>(pos) * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);

        const int idx0 = head * head_dim + pair;
        const int idx1 = idx0 + half_dim;

        const float q0 = q[idx0];
        const float q1 = q[idx1];
        q[idx0] = q0 * cos_val - q1 * sin_val;
        q[idx1] = q0 * sin_val + q1 * cos_val;
    }

    // Process K heads (offset by total_q_pairs)
    const int k_tid = tid - total_q_pairs;
    const int total_k_pairs = n_kv_heads * half_dim;
    if (k_tid >= 0 && k_tid < total_k_pairs) {
        const int head = k_tid / half_dim;
        const int pair = k_tid % half_dim;

        const float freq = 1.0f / powf(freq_base, static_cast<float>(2 * pair) / static_cast<float>(head_dim));
        const float angle = static_cast<float>(pos) * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);

        const int idx0 = head * head_dim + pair;
        const int idx1 = idx0 + half_dim;

        const float k0 = k[idx0];
        const float k1 = k[idx1];
        k[idx0] = k0 * cos_val - k1 * sin_val;
        k[idx1] = k0 * sin_val + k1 * cos_val;
    }
}

inline void apply_rope(float* q, float* k, int head_dim,
                        int n_q_heads, int n_kv_heads, int pos,
                        float freq_base, cudaStream_t stream = 0) {
    const int total = (n_q_heads + n_kv_heads) * (head_dim / 2);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    apply_rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k, head_dim, n_q_heads, n_kv_heads, pos, freq_base);
}

// =============================================================================
// SwiGLU:  out[i] = silu(gate[i]) * up[i]
//          silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// =============================================================================
__global__ void silu_elementwise_mul_kernel(
    float*       __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        const float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

inline void silu_elementwise_mul(float* out, const float* gate, const float* up,
                                  int n, cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    silu_elementwise_mul_kernel<<<blocks, threads, 0, stream>>>(out, gate, up, n);
}

// =============================================================================
// Softmax:  x[i] = exp(x[i] - max) / sum(exp(x - max))
// =============================================================================
__global__ void softmax_kernel(float* __restrict__ x, const int n) {
    extern __shared__ float sdata[];

    // Find max
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(x[i] - max_val);
        x[i] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float total = sdata[0];

    // Normalize
    float inv_total = 1.0f / total;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] *= inv_total;
    }
}

inline void softmax(float* x, int n, cudaStream_t stream = 0) {
    const int threads = (n < 1024) ? ((n + 31) / 32 * 32) : 1024;
    softmax_kernel<<<1, threads, threads * sizeof(float), stream>>>(x, n);
}

// =============================================================================
// Residual add:  x[i] += residual[i]
// =============================================================================
__global__ void residual_add_kernel(
    float*       __restrict__ x,
    const float* __restrict__ residual,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += residual[i];
}

inline void residual_add(float* x, const float* residual, int n,
                          cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(x, residual, n);
}

// =============================================================================
// Float → Half conversion (for KV cache storage)
// =============================================================================
__global__ void float_to_half_kernel(
    __half*      __restrict__ out,
    const float* __restrict__ in,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

inline void float_to_half(half* out, const float* in, int n,
                           cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads, 0, stream>>>(out, in, n);
}

// =============================================================================
// Half → Float conversion (for reading KV cache)
// =============================================================================
__global__ void half_to_float_kernel(
    float*        __restrict__ out,
    const __half* __restrict__ in,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

inline void half_to_float(float* out, const half* in, int n,
                           cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    half_to_float_kernel<<<blocks, threads, 0, stream>>>(out, in, n);
}

// =============================================================================
// GQA Attention for single query position (decode phase)
//
// For one query head:
//   score[t] = dot(q_head, kv_cache_k[t][kv_head]) / sqrt(head_dim)
//   attn = softmax(score[0..seq_len-1])
//   out_head = sum_t(attn[t] * kv_cache_v[t][kv_head])
//
// We process all query heads in one kernel launch.
// =============================================================================
__global__ void gqa_attention_kernel(
    float*        __restrict__ attn_out,     // [n_q_heads * head_dim]
    const float*  __restrict__ q,            // [n_q_heads * head_dim]
    const __half* __restrict__ k_cache,      // [max_ctx * n_kv_heads * head_dim]
    const __half* __restrict__ v_cache,      // [max_ctx * n_kv_heads * head_dim]
    const int head_dim,
    const int n_q_heads,
    const int n_kv_heads,
    const int seq_len,         // number of valid positions (1..pos+1)
    const int max_ctx,
    float* __restrict__ score_buf)  // [n_q_heads * max_ctx] scratch
{
    // One block per query head
    const int qh = blockIdx.x;
    if (qh >= n_q_heads) return;

    const int kv_head = qh / (n_q_heads / n_kv_heads);  // GQA mapping
    const float* q_head = q + qh * head_dim;
    float* scores = score_buf + qh * max_ctx;
    float* out_head = attn_out + qh * head_dim;

    const float scale = rsqrtf(static_cast<float>(head_dim));

    // Compute scores: dot(q, k[t]) for each cached position
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        const __half* k_t = k_cache + static_cast<size_t>(t) * n_kv_heads * head_dim
                            + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_head[d] * __half2float(k_t[d]);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Softmax over scores[0..seq_len-1]
    // (simplified: single block, use shared memory)
    extern __shared__ float smem[];

    // Find max
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = smem[0];

    // Exp + sum
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of values
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            const __half* v_t = v_cache + static_cast<size_t>(t) * n_kv_heads * head_dim
                                + kv_head * head_dim;
            acc += scores[t] * __half2float(v_t[d]);
        }
        out_head[d] = acc;
    }
}

inline void gqa_attention(float* attn_out, const float* q,
                           const half* k_cache, const half* v_cache,
                           int head_dim, int n_q_heads, int n_kv_heads,
                           int seq_len, int max_ctx, float* score_buf,
                           cudaStream_t stream = 0) {
    const int threads = 256;
    gqa_attention_kernel<<<n_q_heads, threads, threads * sizeof(float), stream>>>(
        attn_out, q, k_cache, v_cache, head_dim, n_q_heads, n_kv_heads,
        seq_len, max_ctx, score_buf);
}

// =============================================================================
// Store K,V into KV cache at given position
// =============================================================================
__global__ void store_kv_kernel(
    __half*      __restrict__ k_cache,   // [max_ctx * kv_dim]
    __half*      __restrict__ v_cache,
    const float* __restrict__ k,         // [kv_dim]
    const float* __restrict__ v,
    const int pos,
    const int kv_dim,    // n_kv_heads * head_dim
    const int max_ctx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;

    k_cache[static_cast<size_t>(pos) * kv_dim + i] = __float2half(k[i]);
    v_cache[static_cast<size_t>(pos) * kv_dim + i] = __float2half(v[i]);
}

inline void store_kv(half* k_cache, half* v_cache,
                      const float* k, const float* v,
                      int pos, int kv_dim, int max_ctx,
                      cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (kv_dim + threads - 1) / threads;
    store_kv_kernel<<<blocks, threads, 0, stream>>>(
        k_cache, v_cache, k, v, pos, kv_dim, max_ctx);
}

// =============================================================================
// Convert float buffer to __half (for GEMV input)
// =============================================================================
__global__ void f32_to_f16_kernel(
    __half*      __restrict__ out,
    const float* __restrict__ in,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

inline void f32_to_f16(half* out, const float* in, int n,
                        cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    f32_to_f16_kernel<<<blocks, threads, 0, stream>>>(out, in, n);
}

// =============================================================================
// Argmax (for greedy sampling)
// =============================================================================
__global__ void argmax_kernel(
    const float* __restrict__ x,
    int* __restrict__ result,
    const int n)
{
    // Simple single-block argmax
    extern __shared__ float sdata2[];
    int* sidx = reinterpret_cast<int*>(sdata2 + blockDim.x);

    float local_max = -1e30f;
    int local_idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > local_max) {
            local_max = x[i];
            local_idx = i;
        }
    }
    sdata2[threadIdx.x] = local_max;
    sidx[threadIdx.x] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata2[threadIdx.x + s] > sdata2[threadIdx.x]) {
                sdata2[threadIdx.x] = sdata2[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) *result = sidx[0];
}

inline void argmax(const float* x, int* result, int n, cudaStream_t stream = 0) {
    const int threads = 256;
    argmax_kernel<<<1, threads, threads * (sizeof(float) + sizeof(int)), stream>>>(
        x, result, n);
}
