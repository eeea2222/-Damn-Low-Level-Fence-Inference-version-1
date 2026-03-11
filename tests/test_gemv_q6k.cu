// =============================================================================
// test_gemv_q6k.cu — GEMV kernel verification
//
// Tests:
//   1. Synthetic small matrix (64×256) with known values
//   2. Random synthetic matrix (128×512)
//   3. Real tensor from GGUF file (blk.0.attn_k.weight: 1024×2560)
//
// Build:  make test_gemv_q6k
// Run:    ./build/test_gemv_q6k
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../src/kernels/q6k_types.h"
#include "../src/kernels/q6k_dot.cuh"
#include "../src/kernels/gemv_q6k.cuh"
#include "../src/gguf/gguf_parser.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const char* MODEL_PATH =
    "/home/efeaydin/Desktop/fence-inference-1/"
    "p-e-w_Qwen3-4B-Instruct-2507-heretic-Q6_K_L.gguf";

// ---------- Helper: set a 6-bit quant in a block ----------
static void set_q6(block_q6_K& blk, int i, int val) {
    int lo4 = val & 0xF;
    int hi2 = (val >> 4) & 0x3;
    int byte_lo = i / 2;
    int shift_lo = (i % 2) * 4;
    blk.ql[byte_lo] &= ~(0xF << shift_lo);
    blk.ql[byte_lo] |= (lo4 << shift_lo);
    int byte_hi = i / 4;
    int shift_hi = (i % 4) * 2;
    blk.qh[byte_hi] &= ~(0x3 << shift_hi);
    blk.qh[byte_hi] |= (hi2 << shift_hi);
}

// ---------- CPU golden reference ----------
static double cpu_q6k_dot_block(const block_q6_K& blk, const __half* x) {
    double d = __half2float(blk.d);
    double sum = 0.0;
    for (int i = 0; i < QK_K; ++i) {
        int lo4 = (blk.ql[i/2] >> ((i%2)*4)) & 0xF;
        int hi2 = (blk.qh[i/4] >> ((i%4)*2)) & 0x3;
        int q   = (lo4 | (hi2 << 4)) - 32;
        double sc = blk.scales[i/16];
        sum += d * sc * q * __half2float(x[i]);
    }
    return sum;
}

// CPU GEMV: y[M] = W[M × K] · x[K]
static void cpu_gemv_q6k(const block_q6_K* W, const __half* x,
                          double* y, int M, int K) {
    int blocks_per_row = K / QK_K;
    for (int row = 0; row < M; ++row) {
        double sum = 0.0;
        for (int b = 0; b < blocks_per_row; ++b) {
            sum += cpu_q6k_dot_block(W[row * blocks_per_row + b],
                                      x + b * QK_K);
        }
        y[row] = sum;
    }
}

// ---------- Test framework ----------
static int tests_passed = 0;
static int tests_total  = 0;

static bool check_gemv(const char* name, const double* cpu_y, const float* gpu_y,
                        int M, float atol = 0.5f, float rtol = 5e-3f) {
    tests_total++;
    int errors = 0;
    float max_err = 0;

    for (int i = 0; i < M; ++i) {
        float exp_f = static_cast<float>(cpu_y[i]);
        float err = fabsf(gpu_y[i] - exp_f);
        float denom = fmaxf(fabsf(exp_f), 1.0f);
        float rel = err / denom;
        max_err = fmaxf(max_err, rel);
        if (err > atol && rel > rtol) {
            if (errors < 3) {
                printf("       row %d: cpu=%.6f gpu=%.6f err=%.2e\n",
                       i, exp_f, gpu_y[i], err);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  [PASS] %-45s max_rel_err=%.2e\n", name, max_err);
        tests_passed++;
        return true;
    } else {
        printf("  [FAIL] %-45s %d/%d rows exceeded tolerance, max_rel=%.2e\n",
               name, errors, M, max_err);
        return false;
    }
}

// ---------- Test 1: Synthetic 64×256 (1 block per row) ----------
static void test_synthetic_small() {
    printf("\n--- Test: Synthetic 64x256 ---\n");
    const int M = 64;
    const int K = 256;
    const int blocks_per_row = K / QK_K;  // 1

    // Create weight matrix
    std::vector<block_q6_K> W(M * blocks_per_row);
    srand(123);
    for (int r = 0; r < M; ++r) {
        auto& blk = W[r];
        memset(&blk, 0, sizeof(block_q6_K));
        blk.d = __float2half(0.05f);
        for (int s = 0; s < 16; ++s) {
            blk.scales[s] = static_cast<int8_t>((rand() % 21) - 10);
        }
        for (int i = 0; i < QK_K; ++i) {
            set_q6(blk, i, rand() % 64);
        }
    }

    // Create input vector
    std::vector<__half> x(K);
    for (int i = 0; i < K; ++i) {
        x[i] = __float2half(-1.0f + 2.0f * (float)rand() / RAND_MAX);
    }

    // CPU reference
    std::vector<double> cpu_y(M);
    cpu_gemv_q6k(W.data(), x.data(), cpu_y.data(), M, K);

    // GPU
    block_q6_K* d_W; __half* d_x; float* d_y;
    CUDA_CHECK(cudaMalloc(&d_W, M * blocks_per_row * sizeof(block_q6_K)));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W, W.data(), M * blocks_per_row * sizeof(block_q6_K),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), K * sizeof(__half), cudaMemcpyHostToDevice));

    gemv_q6k(d_W, d_x, d_y, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_y(M);
    CUDA_CHECK(cudaMemcpy(gpu_y.data(), d_y, M * sizeof(float),
                           cudaMemcpyDeviceToHost));

    check_gemv("Synthetic 64x256 (1 block/row)", cpu_y.data(), gpu_y.data(), M);

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// ---------- Test 2: Synthetic 128×512 (2 blocks per row) ----------
static void test_synthetic_multi_block() {
    printf("\n--- Test: Synthetic 128x512 ---\n");
    const int M = 128;
    const int K = 512;
    const int blocks_per_row = K / QK_K;  // 2

    std::vector<block_q6_K> W(M * blocks_per_row);
    srand(456);
    for (auto& blk : W) {
        memset(&blk, 0, sizeof(block_q6_K));
        blk.d = __float2half(0.03f);
        for (int s = 0; s < 16; ++s) {
            blk.scales[s] = static_cast<int8_t>((rand() % 31) - 15);
        }
        for (int i = 0; i < QK_K; ++i) {
            set_q6(blk, i, rand() % 64);
        }
    }

    std::vector<__half> x(K);
    for (int i = 0; i < K; ++i) {
        x[i] = __float2half(-0.5f + (float)rand() / RAND_MAX);
    }

    std::vector<double> cpu_y(M);
    cpu_gemv_q6k(W.data(), x.data(), cpu_y.data(), M, K);

    block_q6_K* d_W; __half* d_x; float* d_y;
    CUDA_CHECK(cudaMalloc(&d_W, M * blocks_per_row * sizeof(block_q6_K)));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W, W.data(), M * blocks_per_row * sizeof(block_q6_K),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), K * sizeof(__half), cudaMemcpyHostToDevice));

    gemv_q6k(d_W, d_x, d_y, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_y(M);
    CUDA_CHECK(cudaMemcpy(gpu_y.data(), d_y, M * sizeof(float),
                           cudaMemcpyDeviceToHost));

    check_gemv("Synthetic 128x512 (2 blocks/row)", cpu_y.data(), gpu_y.data(), M);

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// ---------- Test 3: Real tensor from GGUF ----------
static void test_real_tensor() {
    printf("\n--- Test: Real GGUF tensor (blk.0.attn_k.weight 1024x2560) ---\n");

    GGUFFile gguf;
    if (!gguf.open(MODEL_PATH)) {
        tests_total++;
        printf("  [SKIP] Cannot open GGUF file\n");
        return;
    }

    // blk.0.attn_k.weight: Q6_K, dims=[2560, 1024]
    // In GGUF, dims are [cols, rows] = [input_dim, output_dim]
    // So this is a 1024×2560 matrix (1024 output rows, 2560 input cols)
    auto* ti = gguf.find_tensor("blk.0.attn_k.weight");
    if (!ti) {
        tests_total++;
        printf("  [SKIP] Tensor not found\n");
        gguf.close();
        return;
    }

    const int K = static_cast<int>(ti->dims[0]);  // 2560 (input dim)
    const int M = static_cast<int>(ti->dims[1]);   // 1024 (output dim)
    const int blocks_per_row = K / QK_K;           // 10

    printf("  Matrix: %d rows × %d cols (%d Q6_K blocks per row)\n",
           M, K, blocks_per_row);

    const block_q6_K* W_host = static_cast<const block_q6_K*>(ti->data);

    // Create a random FP16 input vector
    std::vector<__half> x(K);
    srand(789);
    for (int i = 0; i < K; ++i) {
        x[i] = __float2half(-0.1f + 0.2f * (float)rand() / RAND_MAX);
    }

    // CPU reference (only first 32 rows for speed)
    const int test_rows = 32;
    std::vector<double> cpu_y(test_rows);
    cpu_gemv_q6k(W_host, x.data(), cpu_y.data(), test_rows, K);

    // GPU: upload weight matrix and input
    block_q6_K* d_W; __half* d_x; float* d_y;
    size_t W_bytes = static_cast<size_t>(M) * blocks_per_row * sizeof(block_q6_K);

    CUDA_CHECK(cudaMalloc(&d_W, W_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_W, W_host, W_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), K * sizeof(__half), cudaMemcpyHostToDevice));

    // Run full GEMV (all M rows)
    gemv_q6k(d_W, d_x, d_y, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back first test_rows
    std::vector<float> gpu_y(M);
    CUDA_CHECK(cudaMemcpy(gpu_y.data(), d_y, M * sizeof(float),
                           cudaMemcpyDeviceToHost));

    check_gemv("Real attn_k (first 32 rows)", cpu_y.data(), gpu_y.data(), test_rows);

    // Benchmark: run 100 iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int ITERS = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        gemv_q6k(d_W, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float us_per_iter = ms * 1000.0f / ITERS;

    // Bandwidth calculation
    double bytes_read = static_cast<double>(W_bytes) + K * 2.0; // weights + input
    double gb_per_s = bytes_read / (us_per_iter * 1e-6) / 1e9;

    printf("  Benchmark: %.1f µs/iter (%.2f GB/s effective bandwidth)\n",
           us_per_iter, gb_per_s);
    printf("  Matrix: %d×%d, %zu bytes weight data\n", M, K, W_bytes);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    gguf.close();
}

// ---------- Main ----------
int main() {
    printf("=== GEMV Q6_K Kernel Tests ===\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB VRAM)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    test_synthetic_small();
    test_synthetic_multi_block();
    test_real_tensor();

    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
