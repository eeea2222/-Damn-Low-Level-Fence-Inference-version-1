// =============================================================================
// test_gguf_parser.cpp — GGUF parser verification against the real model file
//
// Tests:
//   1. Header validation (magic, version, tensor count)
//   2. Metadata extraction (architecture, dimensions, attention params)
//   3. Tensor lookup (known tensors by name, verify dims/type)
//   4. Data pointer validity (first bytes of tensor data)
//
// Build:  make test_gguf_parser
// Run:    ./build/test_gguf_parser
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../src/gguf/gguf_parser.h"

static const char* MODEL_PATH =
    "/home/efeaydin/Desktop/fence-inference-1/"
    "p-e-w_Qwen3-4B-Instruct-2507-heretic-Q6_K_L.gguf";

static int tests_passed = 0;
static int tests_total  = 0;

#define CHECK(cond, msg)                                       \
    do {                                                        \
        tests_total++;                                          \
        if (cond) {                                             \
            printf("  [PASS] %s\n", msg);                       \
            tests_passed++;                                     \
        } else {                                                \
            printf("  [FAIL] %s\n", msg);                       \
        }                                                       \
    } while (0)

#define CHECK_EQ(actual, expected, msg)                        \
    do {                                                        \
        tests_total++;                                          \
        if ((actual) == (expected)) {                            \
            printf("  [PASS] %s = %s\n", msg, std::to_string(expected).c_str()); \
            tests_passed++;                                     \
        } else {                                                \
            printf("  [FAIL] %s: expected %s, got %s\n", msg,  \
                   std::to_string(expected).c_str(),            \
                   std::to_string(actual).c_str());             \
        }                                                       \
    } while (0)

int main() {
    printf("=== GGUF Parser Tests ===\n\n");

    GGUFFile gguf;

    // ---- Test 1: Open file ----
    printf("--- File Opening ---\n");
    bool opened = gguf.open(MODEL_PATH);
    CHECK(opened, "File opened successfully");
    if (!opened) {
        printf("\nFATAL: Cannot open GGUF file, aborting.\n");
        return 1;
    }

    // ---- Test 2: Header values ----
    printf("\n--- Header ---\n");
    CHECK_EQ(gguf.version(), (uint32_t)3, "Version");
    CHECK_EQ(gguf.tensor_count(), (uint64_t)398, "Tensor count");
    CHECK(gguf.file_size() > 3000000000ULL, "File size > 3GB");

    // ---- Test 3: Metadata ----
    printf("\n--- Metadata ---\n");

    std::string arch = gguf.get_string("general.architecture");
    CHECK(arch == "qwen3", "Architecture = qwen3");

    uint32_t block_count = gguf.get_u32("qwen3.block_count");
    CHECK_EQ(block_count, (uint32_t)36, "Block count");

    uint32_t embed_len = gguf.get_u32("qwen3.embedding_length");
    CHECK_EQ(embed_len, (uint32_t)2560, "Embedding length");

    uint32_t ff_len = gguf.get_u32("qwen3.feed_forward_length");
    CHECK_EQ(ff_len, (uint32_t)9728, "Feed-forward length");

    uint32_t n_heads = gguf.get_u32("qwen3.attention.head_count");
    CHECK_EQ(n_heads, (uint32_t)32, "Attention heads");

    uint32_t n_kv_heads = gguf.get_u32("qwen3.attention.head_count_kv");
    CHECK_EQ(n_kv_heads, (uint32_t)8, "KV heads");

    uint32_t key_len = gguf.get_u32("qwen3.attention.key_length");
    CHECK_EQ(key_len, (uint32_t)128, "Key length (head dim)");

    float rms_eps = gguf.get_f32("qwen3.attention.layer_norm_rms_epsilon");
    CHECK(fabsf(rms_eps - 1e-6f) < 1e-8f, "RMS norm epsilon ~ 1e-6");

    float rope_freq = gguf.get_f32("qwen3.rope.freq_base");
    CHECK(rope_freq > 4e6f, "RoPE freq base > 4M");

    // ---- Test 4: Tensor lookups ----
    printf("\n--- Tensor Lookups ---\n");

    // output_norm.weight: F32, [2560]
    auto* t_onorm = gguf.find_tensor("output_norm.weight");
    CHECK(t_onorm != nullptr, "Found output_norm.weight");
    if (t_onorm) {
        CHECK_EQ(t_onorm->n_dims, (uint32_t)1, "  ndims");
        CHECK_EQ(t_onorm->dims[0], (uint64_t)2560, "  dim[0]");
        CHECK(t_onorm->type == GGMLType::F32, "  type = F32");
        CHECK(t_onorm->data != nullptr, "  data pointer non-null");
        CHECK(t_onorm->data_size_bytes == 2560 * 4, "  data size = 10240 bytes");
    }

    // token_embd.weight: Q8_0, [2560, 151936]
    auto* t_embd = gguf.find_tensor("token_embd.weight");
    CHECK(t_embd != nullptr, "Found token_embd.weight");
    if (t_embd) {
        CHECK_EQ(t_embd->n_dims, (uint32_t)2, "  ndims");
        CHECK_EQ(t_embd->dims[0], (uint64_t)2560, "  dim[0] = 2560");
        CHECK_EQ(t_embd->dims[1], (uint64_t)151936, "  dim[1] = 151936");
        CHECK(t_embd->type == GGMLType::Q8_0, "  type = Q8_0");
        CHECK(t_embd->data != nullptr, "  data pointer non-null");
    }

    // blk.0.attn_q.weight: Q6_K, [2560, 4096]
    auto* t_attn_q = gguf.find_tensor("blk.0.attn_q.weight");
    CHECK(t_attn_q != nullptr, "Found blk.0.attn_q.weight");
    if (t_attn_q) {
        CHECK_EQ(t_attn_q->n_dims, (uint32_t)2, "  ndims");
        CHECK_EQ(t_attn_q->dims[0], (uint64_t)2560, "  dim[0] = 2560");
        CHECK_EQ(t_attn_q->dims[1], (uint64_t)4096, "  dim[1] = 4096");
        CHECK(t_attn_q->type == GGMLType::Q6_K, "  type = Q6_K");
        CHECK(t_attn_q->data != nullptr, "  data pointer non-null");
        // Q6_K: 210 bytes per 256 elements. Total = (2560*4096)/256 * 210
        size_t expected_size = (static_cast<size_t>(2560) * 4096 / 256) * 210;
        CHECK_EQ(t_attn_q->data_size_bytes, expected_size, "  data size");
    }

    // blk.0.attn_k.weight: Q6_K, [2560, 1024]
    auto* t_attn_k = gguf.find_tensor("blk.0.attn_k.weight");
    CHECK(t_attn_k != nullptr, "Found blk.0.attn_k.weight");
    if (t_attn_k) {
        CHECK_EQ(t_attn_k->dims[0], (uint64_t)2560, "  dim[0] = 2560");
        CHECK_EQ(t_attn_k->dims[1], (uint64_t)1024, "  dim[1] = 1024");
        CHECK(t_attn_k->type == GGMLType::Q6_K, "  type = Q6_K");
    }

    // blk.35.ffn_down.weight (last layer)
    auto* t_last = gguf.find_tensor("blk.35.ffn_down.weight");
    CHECK(t_last != nullptr, "Found blk.35.ffn_down.weight (last layer)");

    // Non-existent tensor
    auto* t_bad = gguf.find_tensor("nonexistent.tensor");
    CHECK(t_bad == nullptr, "Non-existent tensor returns nullptr");

    // ---- Test 5: Data pointer validity ----
    printf("\n--- Data Pointer Validity ---\n");
    if (t_onorm && t_onorm->data) {
        // output_norm.weight is F32 at offset 0 — first float should be non-zero
        const float* norm_data = static_cast<const float*>(t_onorm->data);
        CHECK(norm_data[0] != 0.0f, "output_norm.weight[0] is non-zero");
        printf("       output_norm.weight[0] = %f\n", norm_data[0]);
    }
    if (t_attn_q && t_attn_q->data) {
        // First byte should be non-zero (ql data)
        const uint8_t* raw = static_cast<const uint8_t*>(t_attn_q->data);
        bool has_nonzero = false;
        for (int i = 0; i < 16; ++i) {
            if (raw[i] != 0) { has_nonzero = true; break; }
        }
        CHECK(has_nonzero, "blk.0.attn_q.weight first 16 bytes have non-zero data");
    }

    // ---- Summary ----
    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_total);

    gguf.close();
    return (tests_passed == tests_total) ? 0 : 1;
}
