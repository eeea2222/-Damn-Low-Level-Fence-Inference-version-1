#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/model/qwen3.h"
#include "../src/tokenizer/tokenizer.h"
#include "../src/kernels/q6k_dot.cuh"
#include "../src/kernels/gemv_q6k.cuh"
#include "../src/kernels/gemv_q8_0.cuh"
#include "../src/kernels/ops.cuh"

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)

void print_stats(const char* name, float* d_buf, int n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_buf, n*sizeof(float), cudaMemcpyDeviceToHost));
    float mn=1e30f, mx=-1e30f, sum=0, sum2=0;
    int nans=0, infs=0;
    for (int i=0;i<n;i++) {
        if (std::isnan(h[i])) nans++;
        else if (std::isinf(h[i])) infs++;
        else { mn=fminf(mn,h[i]); mx=fmaxf(mx,h[i]); sum+=h[i]; sum2+=h[i]*h[i]; }
    }
    float mean=sum/n, std_=sqrtf(sum2/n-mean*mean);
    printf("  %-25s n=%d min=%.4f max=%.4f mean=%.4f std=%.4f nans=%d infs=%d\n",
           name, n, mn, mx, mean, std_, nans, infs);
    printf("    first 8: ");
    for (int i=0;i<8&&i<n;i++) printf("%.4f ", h[i]);
    printf("\n");
}

int main() {
    const char* model_path = "/home/efeaydin/Desktop/fence-inference-1/"
        "p-e-w_Qwen3-4B-Instruct-2507-heretic-Q6_K_L.gguf";

    Qwen3Model model;
    model.config.max_ctx = 64;
    if (!model.load(model_path)) return 1;

    const auto& c = model.config;
    auto& s = model.scratch;

    int token_id = 9707; // "Hello"
    int pos = 0;

    printf("\n=== Debug Forward Pass ===\n");
    printf("Token: %d, Pos: %d\n\n", token_id, pos);

    // 1. Embed
    embed_q8_0(s.hidden, model.weights.token_embd, token_id, c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("embed", s.hidden, c.embed_dim);

    // 2. Layer 0
    const auto& lw = model.weights.layers[0];

    // RMSNorm
    rms_norm(s.normed, s.hidden, (float*)lw.attn_norm, c.embed_dim, c.rms_eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("rms_norm(attn)", s.normed, c.embed_dim);

    // Convert to FP16
    f32_to_f16(s.x_half, s.normed, c.embed_dim);

    // Q projection
    gemv_q6k(lw.attn_q, s.x_half, s.q, c.q_dim(), c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("Q proj", s.q, c.q_dim());

    // K projection
    gemv_q6k(lw.attn_k, s.x_half, s.k, c.kv_dim(), c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("K proj", s.k, c.kv_dim());

    // V projection
    gemv_q6k(lw.attn_v, s.x_half, s.v, c.kv_dim(), c.embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("V proj", s.v, c.kv_dim());

    // QK-Norm
    rms_norm_per_head(s.q, (float*)lw.attn_q_norm, c.n_heads, c.head_dim, c.rms_eps);
    rms_norm_per_head(s.k, (float*)lw.attn_k_norm, c.n_kv_heads, c.head_dim, c.rms_eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("Q after QK-norm", s.q, c.q_dim());
    print_stats("K after QK-norm", s.k, c.kv_dim());

    // RoPE
    apply_rope(s.q, s.k, c.head_dim, c.n_heads, c.n_kv_heads, pos, c.rope_freq_base);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("Q after RoPE", s.q, c.q_dim());
    print_stats("K after RoPE", s.k, c.kv_dim());

    // Store KV
    store_kv(model.kv_cache.k[0], model.kv_cache.v[0], s.k, s.v, pos, c.kv_dim(), c.max_ctx);

    // GQA
    gqa_attention(s.attn_out, s.q, model.kv_cache.k[0], model.kv_cache.v[0],
                  c.head_dim, c.n_heads, c.n_kv_heads, 1, c.max_ctx, s.score_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("attn_out", s.attn_out, c.q_dim());

    // Output proj
    f32_to_f16(s.x_half, s.attn_out, c.q_dim());
    gemv_q6k(lw.attn_output, s.x_half, s.hidden, c.embed_dim, c.q_dim());
    CUDA_CHECK(cudaDeviceSynchronize());
    print_stats("attn O proj", s.hidden, c.embed_dim);

    // Skip the rest, just run the full forward pass
    printf("\n=== Running full forward ===\n");
    float* logits = model.forward(token_id, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Print top-5 logits
    std::vector<float> h_logits(c.vocab_size);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), logits, c.vocab_size*sizeof(float), cudaMemcpyDeviceToHost));

    int nans=0;
    for (int i=0;i<c.vocab_size;i++) if(std::isnan(h_logits[i])) nans++;
    printf("Logits: %d NaNs out of %d\n", nans, c.vocab_size);

    // Find top 5
    std::vector<std::pair<float,int>> top;
    for (int i=0;i<c.vocab_size;i++) {
        if (!std::isnan(h_logits[i]) && !std::isinf(h_logits[i]))
            top.push_back({h_logits[i], i});
    }
    std::sort(top.begin(), top.end(), [](auto&a, auto&b){return a.first>b.first;});

    Tokenizer tok;
    tok.load_from_gguf(model_path);

    printf("\nTop 10 tokens:\n");
    for (int i=0;i<10&&i<(int)top.size();i++) {
        printf("  %d: id=%d logit=%.4f '%s'\n", i, top[i].second, top[i].first,
               tok.decode(top[i].second).c_str());
    }

    model.unload();
    return 0;
}
