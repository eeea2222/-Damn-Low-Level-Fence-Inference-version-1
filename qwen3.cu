// =============================================================================
// qwen3.cu — Qwen3 model: weight loading, forward pass, generation
// =============================================================================

#include "qwen3.h"
#include "../tokenizer/tokenizer.h"
#include "../kernels/q6k_dot.cuh"
#include "../kernels/gemv_q6k.cuh"
#include "../kernels/gemv_q8_0.cuh"
#include "../kernels/ops.cuh"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

bool g_debug = false;

static void dbg(const char* name, float* d_buf, int n) {
    if (!g_debug) return;
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_buf, n*sizeof(float), cudaMemcpyDeviceToHost));
    float mn=1e30f, mx=-1e30f, sum=0;
    int nans=0;
    for (int i=0;i<n;i++) {
        if (std::isnan(h[i])) nans++;
        else { mn=fminf(mn,h[i]); mx=fmaxf(mx,h[i]); sum+=h[i]; }
    }
    printf("  DBG %-25s min=%.4f max=%.4f mean=%.6f nans=%d\n",
           name, mn, mx, sum/n, nans);
}

// ---- KV Cache ----
void KVCache::allocate(int n_layers, int max_ctx, int kv_dim) {
    k.resize(n_layers);
    v.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        CUDA_CHECK(cudaMalloc(&k[i], (size_t)max_ctx * kv_dim * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&v[i], (size_t)max_ctx * kv_dim * sizeof(__half)));
        CUDA_CHECK(cudaMemset(k[i], 0, (size_t)max_ctx * kv_dim * sizeof(__half)));
        CUDA_CHECK(cudaMemset(v[i], 0, (size_t)max_ctx * kv_dim * sizeof(__half)));
    }
}

void KVCache::free() {
    for (auto* p : k) if (p) cudaFree(p);
    for (auto* p : v) if (p) cudaFree(p);
    k.clear(); v.clear();
}

// ---- Scratch Buffers ----
void ScratchBuffers::allocate(const Qwen3Config& cfg) {
    CUDA_CHECK(cudaMalloc(&hidden,    cfg.embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&residual,  cfg.embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&normed,    cfg.embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q,         cfg.q_dim() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&k,         cfg.kv_dim() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v,         cfg.kv_dim() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attn_out,  cfg.q_dim() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gate,      cfg.ff_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up,        cfg.ff_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&down,      cfg.embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&logits,    cfg.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaHostAlloc((void**)&cpu_logits, cfg.vocab_size * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&score_buf, cfg.n_heads * cfg.max_ctx * sizeof(float)));
    int max_half = cfg.ff_dim > cfg.embed_dim ? cfg.ff_dim : cfg.embed_dim;
    CUDA_CHECK(cudaMalloc(&x_half,    max_half * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&argmax_out, sizeof(int)));
}

void ScratchBuffers::free() {
    auto F = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F(reinterpret_cast<void*&>(hidden));
    F(reinterpret_cast<void*&>(residual));
    F(reinterpret_cast<void*&>(normed));
    F(reinterpret_cast<void*&>(q));
    F(reinterpret_cast<void*&>(k));
    F(reinterpret_cast<void*&>(v));
    F(reinterpret_cast<void*&>(attn_out));
    F(reinterpret_cast<void*&>(gate));
    F(reinterpret_cast<void*&>(up));
    F(reinterpret_cast<void*&>(down));
    F(reinterpret_cast<void*&>(logits));
    if (cpu_logits) { cudaFreeHost(cpu_logits); cpu_logits = nullptr; }
    F(reinterpret_cast<void*&>(score_buf));
    F(reinterpret_cast<void*&>(x_half));
    F(reinterpret_cast<void*&>(argmax_out));
}

// ---- Copy tensor data to GPU ----
void* Qwen3Model::copy_tensor_to_gpu(const GGUFTensorInfo* ti) {
    if (!ti) return nullptr; // Safe null check
    void* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, ti->data_size_bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, ti->data, ti->data_size_bytes, cudaMemcpyHostToDevice));
    device_allocs_.push_back(d_ptr);
    return d_ptr;
}

// ===========================================================================
// Load model from GGUF
// ===========================================================================
bool Qwen3Model::load(const std::string& gguf_path) {
    printf("Loading GGUF: %s\n", gguf_path.c_str());
    if (!gguf_.open(gguf_path)) return false;

    // Find architecture prefix dynamically
    std::string arch = "qwen2"; // default fallback
    try {
        arch = gguf_.get_string("general.architecture");
    } catch (...) {}

    // Config from metadata
    config.n_layers       = gguf_.get_u32(arch + ".block_count");
    config.embed_dim      = gguf_.get_u32(arch + ".embedding_length");
    config.ff_dim         = gguf_.get_u32(arch + ".feed_forward_length");
    config.n_heads        = gguf_.get_u32(arch + ".attention.head_count");
    config.n_kv_heads     = gguf_.get_u32(arch + ".attention.head_count_kv");
    config.head_dim       = gguf_.get_u32(arch + ".attention.key_length");
    config.rope_freq_base = gguf_.get_f32(arch + ".rope.freq_base");
    config.rms_eps        = gguf_.get_f32(arch + ".attention.layer_norm_rms_epsilon");

    auto* embd_ti = gguf_.find_tensor("token_embd.weight");
    if (!embd_ti) { fprintf(stderr, "Missing token_embd.weight\n"); return false; }
    config.vocab_size = (int)embd_ti->dims[1];

    printf("Config: layers=%d embed=%d ff=%d heads=%d/%d head_dim=%d vocab=%d\n",
           config.n_layers, config.embed_dim, config.ff_dim,
           config.n_heads, config.n_kv_heads, config.head_dim, config.vocab_size);
    printf("VRAM estimate: ~%.1f GB weights + %.0f MB KV cache\n",
           3.2, (double)config.n_layers * 2 * config.max_ctx * config.kv_dim() * 2 / 1e6);

    // Copy weights to GPU
    printf("Copying weights to GPU...\n");
    auto t0 = std::chrono::steady_clock::now();

    weights.token_embd = (const block_q8_0*)copy_tensor_to_gpu(embd_ti);
    weights.output_norm = (const float*)copy_tensor_to_gpu(gguf_.find_tensor("output_norm.weight"));

    weights.layers.resize(config.n_layers);
    for (int l = 0; l < config.n_layers; ++l) {
        auto& lw = weights.layers[l];
        std::string p = "blk." + std::to_string(l) + ".";

        auto lf = [&](const char* s) -> const float* {
            return (const float*)copy_tensor_to_gpu(gguf_.find_tensor(p + s));
        };
        auto lq = [&](const char* s) -> const block_q6_K* {
            return (const block_q6_K*)copy_tensor_to_gpu(gguf_.find_tensor(p + s));
        };

        lw.attn_norm    = lf("attn_norm.weight");
        lw.attn_q       = lq("attn_q.weight");
        lw.attn_k       = lq("attn_k.weight");
        lw.attn_v       = lq("attn_v.weight");
        
        // Biases are optional in some models
        lw.attn_q_bias  = lf("attn_q.bias");
        lw.attn_k_bias  = lf("attn_k.bias");
        lw.attn_v_bias  = lf("attn_v.bias");

        lw.attn_q_norm  = lf("attn_q_norm.weight");
        lw.attn_k_norm  = lf("attn_k_norm.weight");
        lw.attn_output  = lq("attn_output.weight");
        lw.ffn_norm     = lf("ffn_norm.weight");
        lw.ffn_gate     = lq("ffn_gate.weight");
        lw.ffn_up       = lq("ffn_up.weight");
        lw.ffn_down     = lq("ffn_down.weight");

        if ((l + 1) % 12 == 0 || l == config.n_layers - 1)
            printf("  Layer %d/%d\n", l + 1, config.n_layers);
    }

    auto t1 = std::chrono::steady_clock::now();
    printf("Weights loaded in %.1f s\n", std::chrono::duration<double>(t1 - t0).count());

    // Allocate KV cache
    kv_cache.allocate(config.n_layers, config.max_ctx, config.kv_dim());
    printf("KV cache allocated\n");

    // Allocate scratch
    scratch.allocate(config);
    printf("Model ready!\n\n");
    return true;
}

void Qwen3Model::unload() {
    for (auto* p : device_allocs_) cudaFree(p);
    device_allocs_.clear();
    kv_cache.free();
    scratch.free();
    gguf_.close();
}

// ===========================================================================
// Forward pass: single token at position pos → logits
// ===========================================================================
float* Qwen3Model::forward(int token_id, int pos) {
    const auto& c = config;
    auto& s = scratch;

    // 1. Embed
    embed_q8_0(s.hidden, weights.token_embd, token_id, c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (g_debug && pos == 0) { printf("--- Forward token=%d pos=%d ---\n", token_id, pos); dbg("embed", s.hidden, c.embed_dim); }

    // 2. Layers
    for (int l = 0; l < c.n_layers; ++l) {
        bool dl = g_debug && pos == 0 && l == 0;
        forward_attention(l, pos, dl);
        forward_mlp(l, dl);
    }

    rms_norm_inplace(s.hidden, (float*)weights.output_norm, c.embed_dim, c.rms_eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (g_debug && pos == 0) dbg("final_norm", s.hidden, c.embed_dim);

    gemv_q8_0(weights.token_embd, s.hidden, s.logits, c.vocab_size, c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    return s.logits;
}

void Qwen3Model::forward_attention(int l, int pos, bool dl) {
    const auto& c = config;
    auto& s = scratch;
    const auto& lw = weights.layers[l];

    cudaMemcpyAsync(s.residual, s.hidden, c.embed_dim * sizeof(float), cudaMemcpyDeviceToDevice);

    rms_norm(s.normed, s.hidden, (float*)lw.attn_norm, c.embed_dim, c.rms_eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) dbg("rms_norm(attn)", s.normed, c.embed_dim);

    f32_to_f16(s.x_half, s.normed, c.embed_dim);
    // Q, K, V projections (with optional bias)
    if (lw.attn_q_bias) gemv_q6k_bias(lw.attn_q, s.x_half, lw.attn_q_bias, s.q, c.q_dim(), c.embed_dim);
    else                gemv_q6k(lw.attn_q, s.x_half, s.q, c.q_dim(), c.embed_dim);
    
    if (lw.attn_k_bias) gemv_q6k_bias(lw.attn_k, s.x_half, lw.attn_k_bias, s.k, c.kv_dim(), c.embed_dim);
    else                gemv_q6k(lw.attn_k, s.x_half, s.k, c.kv_dim(), c.embed_dim);
    
    if (lw.attn_v_bias) gemv_q6k_bias(lw.attn_v, s.x_half, lw.attn_v_bias, s.v, c.kv_dim(), c.embed_dim);
    else                gemv_q6k(lw.attn_v, s.x_half, s.v, c.kv_dim(), c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) { dbg("Q proj", s.q, c.q_dim()); dbg("K proj", s.k, c.kv_dim()); dbg("V proj", s.v, c.kv_dim()); }

    rms_norm_per_head(s.q, (float*)lw.attn_q_norm, c.n_heads, c.head_dim, c.rms_eps);
    rms_norm_per_head(s.k, (float*)lw.attn_k_norm, c.n_kv_heads, c.head_dim, c.rms_eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) { dbg("Q qk-norm", s.q, c.q_dim()); dbg("K qk-norm", s.k, c.kv_dim()); }

    apply_rope(s.q, s.k, c.head_dim, c.n_heads, c.n_kv_heads, pos, c.rope_freq_base);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) { dbg("Q rope", s.q, c.q_dim()); dbg("K rope", s.k, c.kv_dim()); }

    store_kv(kv_cache.k[l], kv_cache.v[l], s.k, s.v, pos, c.kv_dim(), c.max_ctx);

    gqa_attention(s.attn_out, s.q, kv_cache.k[l], kv_cache.v[l],
                  c.head_dim, c.n_heads, c.n_kv_heads, pos + 1, c.max_ctx, s.score_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) dbg("attn_out", s.attn_out, c.q_dim());

    f32_to_f16(s.x_half, s.attn_out, c.q_dim());
    gemv_q6k(lw.attn_output, s.x_half, s.hidden, c.embed_dim, c.q_dim());
    residual_add(s.hidden, s.residual, c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) dbg("after attn+res", s.hidden, c.embed_dim);
}

void Qwen3Model::forward_mlp(int l, bool dl) {
    const auto& c = config;
    auto& s = scratch;
    const auto& lw = weights.layers[l];

    cudaMemcpyAsync(s.residual, s.hidden, c.embed_dim * sizeof(float), cudaMemcpyDeviceToDevice);

    rms_norm(s.normed, s.hidden, (float*)lw.ffn_norm, c.embed_dim, c.rms_eps);
    f32_to_f16(s.x_half, s.normed, c.embed_dim);

    gemv_q6k(lw.ffn_gate, s.x_half, s.gate, c.ff_dim, c.embed_dim);
    gemv_q6k(lw.ffn_up,   s.x_half, s.up,   c.ff_dim, c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) { dbg("gate", s.gate, c.ff_dim); dbg("up", s.up, c.ff_dim); }

    silu_elementwise_mul(s.gate, s.gate, s.up, c.ff_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) dbg("silu*up", s.gate, c.ff_dim);

    f32_to_f16(s.x_half, s.gate, c.ff_dim);
    gemv_q6k(lw.ffn_down, s.x_half, s.hidden, c.embed_dim, c.ff_dim);
    residual_add(s.hidden, s.residual, c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (dl) dbg("after ffn+res", s.hidden, c.embed_dim);
}

// ---- Sampling ----
int Qwen3Model::sample_greedy(float* logits_ptr) {
    argmax(logits_ptr, scratch.argmax_out, config.vocab_size);
    int token;
    CUDA_CHECK(cudaMemcpy(&token, scratch.argmax_out, sizeof(int), cudaMemcpyDeviceToHost));
    return token;
}

int Qwen3Model::sample_nucleus(float* logits_ptr, const std::vector<int>& history) {
    if (config.temperature <= 0.0f || config.top_k <= 1) {
        return sample_greedy(logits_ptr);
    }
    
    CUDA_CHECK(cudaMemcpy(scratch.cpu_logits, logits_ptr, config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Apply repetition penalty
    if (config.repetition_penalty > 1.0f && !history.empty()) {
        for (int tok : history) {
            if (tok >= 0 && tok < config.vocab_size) {
                if (scratch.cpu_logits[tok] > 0) {
                    scratch.cpu_logits[tok] /= config.repetition_penalty;
                } else {
                    scratch.cpu_logits[tok] *= config.repetition_penalty;
                }
            }
        }
    }
    
    float max_l = -1e30f;
    for (int i = 0; i < config.vocab_size; ++i) {
        max_l = std::max(max_l, scratch.cpu_logits[i]);
    }
    
    // 1. Min-P Sampling (Drop tokens < config.min_p * max_prob)
    if (config.min_p > 0.0f) {
        float min_p_threshold = max_l + logf(config.min_p); 
        for (int i = 0; i < config.vocab_size; ++i) {
            if (scratch.cpu_logits[i] < min_p_threshold) {
                scratch.cpu_logits[i] = -1e30f;
            }
        }
    }
    
    float current_temp = config.temperature;
    
    // 2. Dynamic Temperature (Entropy-scaled)
    if (config.dynamic_temp) {
        float dyn_sum = 0.0f;
        for (int i = 0; i < config.vocab_size; ++i) {
            if (scratch.cpu_logits[i] > -1e20f) {
                dyn_sum += expf(scratch.cpu_logits[i] - max_l);
            }
        }
        float entropy = 0.0f;
        for (int i = 0; i < config.vocab_size; ++i) {
            if (scratch.cpu_logits[i] > -1e20f) {
                float p = expf(scratch.cpu_logits[i] - max_l) / dyn_sum;
                if (p > 1e-6f) entropy -= p * logf(p);
            }
        }
        // Scale temperature based on confusion: High entropy = higher temp, Low entropy = lower temp.
        float scale = std::clamp(entropy / 2.0f, 0.5f, 2.0f);
        current_temp *= scale;
    }
    
    // Softmax
    float sum = 0.0f;
    for (int i = 0; i < config.vocab_size; ++i) {
        if (scratch.cpu_logits[i] < -1e20f) {
            scratch.cpu_logits[i] = 0.0f;
            continue;
        }
        float p = expf((scratch.cpu_logits[i] - max_l) / current_temp);
        scratch.cpu_logits[i] = p;
        sum += p;
    }
    for (int i = 0; i < config.vocab_size; ++i) {
        scratch.cpu_logits[i] /= sum;
    }
    
    std::vector<std::pair<float, int>> probs;
    probs.reserve(config.vocab_size);
    for (int i = 0; i < config.vocab_size; ++i) {
        probs.push_back({scratch.cpu_logits[i], i});
    }
    
    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    int limit = std::min(config.vocab_size, config.top_k);
    
    float cumul = 0.0f;
    int actual_k = limit;
    for (int i = 0; i < limit; ++i) {
        cumul += probs[i].first;
        if (cumul >= config.top_p) {
            actual_k = i + 1;
            break;
        }
    }
    
    std::vector<float> final_probs(actual_k);
    for (int i = 0; i < actual_k; ++i) final_probs[i] = probs[i].first;
    
    static std::mt19937 rand_engine(std::random_device{}());
    std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
    
    int chosen_idx = dist(rand_engine);
    return probs[chosen_idx].second;
}

// ---- Generation loop ----
std::vector<int> Qwen3Model::generate(
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    bool print_tokens,
    const Tokenizer* tokenizer)
{
    std::vector<int> output;
    output.reserve(prompt_tokens.size() + max_new_tokens);

    const int eos_id = 151645;  // <|im_end|>

    auto t_start = std::chrono::steady_clock::now();
    int prompt_len = (int)prompt_tokens.size();

    // Prefill: process all prompt tokens (positions 0..prompt_len-1)
    for (int i = 0; i < prompt_len; ++i) {
        forward(prompt_tokens[i], i);
        output.push_back(prompt_tokens[i]);
    }

    auto t_prefill = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
    printf("[prefill %d tokens in %.0f ms (%.1f tok/s)]\n",
           prompt_len, prefill_ms, prompt_len / (prefill_ms / 1000.0));

    // Decode: generate new tokens
    int pos = prompt_len;
    int generated = 0;
    auto t_decode_start = std::chrono::steady_clock::now();

    for (int i = 0; i < max_new_tokens && pos < config.max_ctx; ++i) {
        int next_token = sample_nucleus(scratch.logits, output);

        if (next_token == eos_id) break;

        output.push_back(next_token);
        generated++;

        if (print_tokens && tokenizer) {
            std::string text = tokenizer->decode(next_token);
            printf("%s", text.c_str());
            fflush(stdout);
        }

        forward(next_token, pos);
        pos++;
    }

    auto t_end = std::chrono::steady_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_decode_start).count();
    if (generated > 0) {
        printf("\n[generated %d tokens in %.0f ms (%.1f tok/s)]\n",
               generated, decode_ms, generated / (decode_ms / 1000.0));
    }

    return output;
}
