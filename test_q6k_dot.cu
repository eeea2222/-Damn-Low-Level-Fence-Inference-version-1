// =============================================================================
// test_q6k_dot.cu — Verification harness for Q6_K × FP16 dot product kernels
//
// Tests:
//   1. All quants = 32 (zero-centered) → dot ≈ 0
//   2. All quants = 0  (minimum value)
//   3. All quants = 63 (maximum value)
//   4. Random quants with known FP16 vector
//   5. Mixed positive/negative scales
//   6. Warp-cooperative kernel correctness
//
// Build:  make test_q6k_dot
// Run:    ./build/test_q6k_dot
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../src/kernels/q6k_types.h"
#include "../src/kernels/q6k_dot.cuh"

// ---------- Error checking ----------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------- CPU golden reference ----------
// Dequantize + accumulate in FP64 for maximum precision.
static double cpu_q6k_dot(const block_q6_K& blk, const __half* x_host) {
    const double d = static_cast<double>(__half2float(blk.d));
    double sum = 0.0;

    for (int i = 0; i < QK_K; ++i) {
        // Lower 4 bits
        int byte_lo  = i / 2;
        int shift_lo = (i % 2) * 4;
        int lo4 = (blk.ql[byte_lo] >> shift_lo) & 0xF;

        // Upper 2 bits
        int byte_hi  = i / 4;
        int shift_hi = (i % 4) * 2;
        int hi2 = (blk.qh[byte_hi] >> shift_hi) & 0x3;

        int q6 = lo4 | (hi2 << 4);       // [0, 63]
        int q_signed = q6 - 32;           // [-32, 31]

        int sb = i / 16;
        double sc = static_cast<double>(blk.scales[sb]);
        double xi = static_cast<double>(__half2float(x_host[i]));

        sum += d * sc * static_cast<double>(q_signed) * xi;
    }

    return sum;
}

// ---------- Helper: pack a 6-bit quant into the block ----------
static void set_q6(block_q6_K& blk, int i, int val) {
    assert(val >= 0 && val <= 63);
    int lo4 = val & 0xF;
    int hi2 = (val >> 4) & 0x3;

    // Set lower nibble
    int byte_lo = i / 2;
    int shift_lo = (i % 2) * 4;
    blk.ql[byte_lo] &= ~(0xF << shift_lo);
    blk.ql[byte_lo] |= (lo4 << shift_lo);

    // Set upper crumb
    int byte_hi = i / 4;
    int shift_hi = (i % 4) * 2;
    blk.qh[byte_hi] &= ~(0x3 << shift_hi);
    blk.qh[byte_hi] |= (hi2 << shift_hi);
}

// ---------- GPU test kernels ----------

// Single-thread kernel that runs both reference and fast implementations
__global__ void kernel_test_single_thread(
    const block_q6_K* __restrict__ blk,
    const __half*     __restrict__ x,
    float* result_ref,
    float* result_fast)
{
    *result_ref  = q6k_dot_f16_reference(blk, x);
    *result_fast = q6k_dot_f16_fast(blk, x);
}

// Warp-cooperative kernel: 1 warp (32 threads) processes one block
__global__ void kernel_test_warp(
    const block_q6_K* __restrict__ blk,
    const __half*     __restrict__ x,
    float* result)
{
    const int tid = threadIdx.x;  // lane id [0, 31]
    float partial = q6k_dot_f16_warp_partial(blk, x, tid);
    float total = warp_reduce_sum(partial);

    if (tid == 0) {
        *result = total;
    }
}

// ---------- Test framework ----------
struct TestCase {
    const char* name;
    block_q6_K blk;
    __half x_host[QK_K];
};

static int tests_passed = 0;
static int tests_total  = 0;

static void run_test(const TestCase& tc, float atol = 1e-2f, float rtol = 1e-3f) {
    tests_total++;

    // CPU reference (FP64)
    double expected = cpu_q6k_dot(tc.blk, tc.x_host);

    // Allocate device memory
    block_q6_K* d_blk;
    __half* d_x;
    float* d_ref;
    float* d_fast;
    float* d_warp;

    CUDA_CHECK(cudaMalloc(&d_blk,  sizeof(block_q6_K)));
    CUDA_CHECK(cudaMalloc(&d_x,    QK_K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ref,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fast, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_warp, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_blk, &tc.blk, sizeof(block_q6_K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, tc.x_host, QK_K * sizeof(__half), cudaMemcpyHostToDevice));

    // Run single-thread kernels
    kernel_test_single_thread<<<1, 1>>>(d_blk, d_x, d_ref, d_fast);
    CUDA_CHECK(cudaGetLastError());

    // Run warp kernel
    kernel_test_warp<<<1, 32>>>(d_blk, d_x, d_warp);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back results
    float h_ref, h_fast, h_warp;
    CUDA_CHECK(cudaMemcpy(&h_ref,  d_ref,  sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_fast, d_fast, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_warp, d_warp, sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    float exp_f = static_cast<float>(expected);
    float denom = fmaxf(fabsf(exp_f), 1e-6f);

    float err_ref  = fabsf(h_ref - exp_f);
    float err_fast = fabsf(h_fast - exp_f);
    float err_warp = fabsf(h_warp - exp_f);

    bool pass_ref  = (err_ref  < atol) || (err_ref / denom  < rtol);
    bool pass_fast = (err_fast < atol) || (err_fast / denom < rtol);
    bool pass_warp = (err_warp < atol) || (err_warp / denom < rtol);
    bool pass = pass_ref && pass_fast && pass_warp;

    if (pass) {
        printf("  [PASS] %-35s  cpu=%.6f  ref=%.6f  fast=%.6f  warp=%.6f\n",
               tc.name, exp_f, h_ref, h_fast, h_warp);
        tests_passed++;
    } else {
        printf("  [FAIL] %-35s  cpu=%.6f  ref=%.6f(e=%.2e,%s)  "
               "fast=%.6f(e=%.2e,%s)  warp=%.6f(e=%.2e,%s)\n",
               tc.name, exp_f,
               h_ref,  err_ref,  pass_ref  ? "ok" : "FAIL",
               h_fast, err_fast, pass_fast ? "ok" : "FAIL",
               h_warp, err_warp, pass_warp ? "ok" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_blk));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_fast));
    CUDA_CHECK(cudaFree(d_warp));
}

// ---------- Test cases ----------

// TC1: All quants = 32 (zero-centered) → dot should be ~0
static TestCase make_test_zero_centered() {
    TestCase tc;
    tc.name = "all_quants_32_zero_centered";
    memset(&tc.blk, 0, sizeof(block_q6_K));
    tc.blk.d = __float2half(0.1f);
    for (int i = 0; i < 16; ++i) tc.blk.scales[i] = 5;
    for (int i = 0; i < QK_K; ++i) set_q6(tc.blk, i, 32);
    for (int i = 0; i < QK_K; ++i) tc.x_host[i] = __float2half(1.0f);
    return tc;
}

// TC2: All quants = 0 (minimum, q-32 = -32)
static TestCase make_test_all_min() {
    TestCase tc;
    tc.name = "all_quants_0_minimum";
    memset(&tc.blk, 0, sizeof(block_q6_K));
    tc.blk.d = __float2half(0.05f);
    for (int i = 0; i < 16; ++i) tc.blk.scales[i] = 3;
    for (int i = 0; i < QK_K; ++i) set_q6(tc.blk, i, 0);
    for (int i = 0; i < QK_K; ++i) tc.x_host[i] = __float2half(0.5f);
    return tc;
}

// TC3: All quants = 63 (maximum, q-32 = 31)
static TestCase make_test_all_max() {
    TestCase tc;
    tc.name = "all_quants_63_maximum";
    memset(&tc.blk, 0, sizeof(block_q6_K));
    tc.blk.d = __float2half(0.05f);
    for (int i = 0; i < 16; ++i) tc.blk.scales[i] = 3;
    for (int i = 0; i < QK_K; ++i) set_q6(tc.blk, i, 63);
    for (int i = 0; i < QK_K; ++i) tc.x_host[i] = __float2half(0.5f);
    return tc;
}

// TC4: Random quants, uniform activations
static TestCase make_test_random() {
    TestCase tc;
    tc.name = "random_quants_uniform_x";
    memset(&tc.blk, 0, sizeof(block_q6_K));
    tc.blk.d = __float2half(0.02f);

    srand(42);
    for (int i = 0; i < 16; ++i) {
        tc.blk.scales[i] = static_cast<int8_t>((rand() % 21) - 10); // [-10, 10]
    }
    for (int i = 0; i < QK_K; ++i) {
        set_q6(tc.blk, i, rand() % 64);  // [0, 63]
    }
    for (int i = 0; i < QK_K; ++i) {
        float v = -1.0f + 2.0f * (static_cast<float>(rand()) / RAND_MAX);
        tc.x_host[i] = __float2half(v);
    }
    return tc;
}

// TC5: Mixed positive/negative scales, d=1.0 for easy verification
static TestCase make_test_mixed_scales() {
    TestCase tc;
    tc.name = "mixed_pos_neg_scales";
    memset(&tc.blk, 0, sizeof(block_q6_K));
    tc.blk.d = __float2half(1.0f);

    // Alternating positive/negative scales
    for (int i = 0; i < 16; ++i) {
        tc.blk.scales[i] = (i % 2 == 0) ? 4 : -4;
    }
    for (int i = 0; i < QK_K; ++i) {
        set_q6(tc.blk, i, 40);  // q-32 = 8
    }
    for (int i = 0; i < QK_K; ++i) {
        tc.x_host[i] = __float2half(0.25f);
    }
    return tc;
}

// TC6: Single non-zero sub-block (isolates sub-block indexing)
static TestCase make_test_single_subblock() {
    TestCase tc;
    tc.name = "single_nonzero_subblock";
    memset(&tc.blk, 0, sizeof(block_q6_K));
    tc.blk.d = __float2half(0.1f);

    // Only sub-block 7 has a non-zero scale
    for (int i = 0; i < 16; ++i) tc.blk.scales[i] = 0;
    tc.blk.scales[7] = 10;

    for (int i = 0; i < QK_K; ++i) set_q6(tc.blk, i, 45); // q-32 = 13
    for (int i = 0; i < QK_K; ++i) tc.x_host[i] = __float2half(1.0f);
    return tc;
}

// ---------- Main ----------
int main() {
    printf("=== Q6_K Dot Product Kernel Tests ===\n");
    printf("Struct size check: block_q6_K = %zu bytes (expected 210)\n\n",
           sizeof(block_q6_K));

    if (sizeof(block_q6_K) != 210) {
        fprintf(stderr, "FATAL: block_q6_K size mismatch!\n");
        return 1;
    }

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    TestCase tc1 = make_test_zero_centered();
    TestCase tc2 = make_test_all_min();
    TestCase tc3 = make_test_all_max();
    TestCase tc4 = make_test_random();
    TestCase tc5 = make_test_mixed_scales();
    TestCase tc6 = make_test_single_subblock();

    run_test(tc1);
    run_test(tc2);
    run_test(tc3);
    run_test(tc4);
    run_test(tc5);
    run_test(tc6);

    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
