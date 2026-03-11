#pragma once
// =============================================================================
// qwen3.h — Qwen3 model definition: config, weights, KV cache, forward pass
// =============================================================================

class Tokenizer;  // forward declaration

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../gguf/gguf_parser.h"
#include "../kernels/q6k_types.h"
#include "../kernels/q8_0_types.h"

// ---- Model config (extracted from GGUF metadata) ----
struct Qwen3Config {
    int n_layers       = 36;
    int embed_dim      = 2560;
    int ff_dim         = 9728;
    int n_heads        = 32;
    int n_kv_heads     = 8;
    int head_dim       = 128;
    int vocab_size     = 151936;
    int max_ctx        = 4096;   // max context length we allocate
    float rope_freq_base = 5000000.0f;
    float rms_eps      = 1e-6f;

    // Sampling parameters
    float temperature  = 0.7f;
    float top_p        = 0.8f;
    int   top_k        = 50;
    float repetition_penalty = 1.15f;
    float min_p        = 0.05f;      // Drop tokens with probability < 5% of the most likely token
    bool  dynamic_temp = true;       // Enable entropy-scaled temperature



    int q_dim() const { return n_heads * head_dim; }     // 4096
    int kv_dim() const { return n_kv_heads * head_dim; } // 1024
    int heads_per_group() const { return n_heads / n_kv_heads; } // 4
};

// ---- Per-layer weight pointers (device) ----
struct Qwen3LayerWeights {
    // Attention
    const float*      attn_norm;       // F32 [embed_dim]
    const block_q6_K* attn_q;          // Q6_K [embed_dim, q_dim]
    const block_q6_K* attn_k;          // Q6_K [embed_dim, kv_dim]
    const block_q6_K* attn_v;          // Q6_K [embed_dim, kv_dim]
    const float*      attn_q_bias;     // F32 [q_dim]
    const float*      attn_k_bias;     // F32 [kv_dim]
    const float*      attn_v_bias;     // F32 [kv_dim]
    const float*      attn_q_norm;     // F32 [head_dim] — QK-Norm
    const float*      attn_k_norm;     // F32 [head_dim] — QK-Norm
    const block_q6_K* attn_output;     // Q6_K [q_dim, embed_dim]

    // MLP (SwiGLU)
    const float*      ffn_norm;        // F32 [embed_dim]
    const block_q6_K* ffn_gate;        // Q6_K [embed_dim, ff_dim]
    const block_q6_K* ffn_up;          // Q6_K [embed_dim, ff_dim]
    const block_q6_K* ffn_down;        // Q6_K [ff_dim, embed_dim]
};

// ---- Model weights ----
struct Qwen3Weights {
    // Global
    const block_q8_0* token_embd;      // Q8_0 [embed_dim, vocab_size]
    const float*      output_norm;     // F32  [embed_dim]
    // token_embd is also used as LM head (tied weights)

    // Per-layer
    std::vector<Qwen3LayerWeights> layers;
};

// ---- KV cache (FP16, device memory) ----
struct KVCache {
    // k_cache[layer]: [max_ctx * kv_dim] FP16
    // v_cache[layer]: [max_ctx * kv_dim] FP16
    std::vector<__half*> k;  // per-layer
    std::vector<__half*> v;  // per-layer

    void allocate(int n_layers, int max_ctx, int kv_dim);
    void free();
    ~KVCache() { free(); }
};

// ---- Scratch buffers (pre-allocated FP32 device memory) ----
struct ScratchBuffers {
    float* hidden;       // [embed_dim]
    float* residual;     // [embed_dim]
    float* normed;       // [embed_dim]
    float* q;            // [q_dim]
    float* k;            // [kv_dim]
    float* v;            // [kv_dim]
    float* attn_out;     // [q_dim]
    float* gate;         // [ff_dim]
    float* up;           // [ff_dim]
    float* down;         // [embed_dim]
    float* logits;       // [vocab_size]  (GPU)
    float* cpu_logits;   // [vocab_size]  (CPU, pinned)
    float* score_buf;    // [n_heads * max_ctx]
    __half* x_half;      // [max(embed_dim, ff_dim)] — FP16 conversion buffer
    int*   argmax_out;   // [1]


    void allocate(const Qwen3Config& cfg);
    void free();
    ~ScratchBuffers() { free(); }
};

// ---- Main model class ----
class Qwen3Model {
public:
    Qwen3Config  config;
    Qwen3Weights weights;
    KVCache      kv_cache;
    ScratchBuffers scratch;

    /// Load model from GGUF file. Copies weights to GPU.
    bool load(const std::string& gguf_path);
    void unload();

    ~Qwen3Model() { unload(); }


    /// Run forward pass for a single token at given position.
    /// Returns logits (device pointer, [vocab_size]).
    float* forward(int token_id, int pos);

    /// Greedy decode: returns the argmax token from logits.
    int sample_greedy(float* logits);

    /// Nucleus sampling: Top-K, Top-p, and Temperature with Repetition Penalty.
    int sample_nucleus(float* logits, const std::vector<int>& history);

    /// Full autoregressive generation.
    std::vector<int> generate(const std::vector<int>& prompt_tokens,
                              int max_new_tokens,
                              bool print_tokens = false,
                              const Tokenizer* tokenizer = nullptr);

private:
    // Device memory for weight copies
    std::vector<void*> device_allocs_;
    GGUFFile gguf_;

    void* copy_tensor_to_gpu(const GGUFTensorInfo* ti);
    
    // Internal forward pass helpers
    void forward_attention(int l, int pos, bool dl);
    void forward_mlp(int l, bool dl);
};
