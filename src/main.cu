// =============================================================================
// main.cu — Fence Inference Engine: Interactive Qwen3 Chat
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

#include "model/qwen3.h"
#include "tokenizer/tokenizer.h"

extern bool g_debug;  // defined in qwen3.cu

static const char* MODEL_PATH =
    "/home/efeaydin/Desktop/fence-inference-1/"
    "p-e-w_Qwen3-4B-Instruct-2507-heretic-Q6_K_L.gguf";

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH    Model GGUF file (default: built-in path)\n");
    printf("  --prompt TEXT    Single prompt (non-interactive mode)\n");
    printf("  --max-tokens N   Max new tokens to generate (default: 512)\n");
    printf("  --max-ctx N      Max context length (default: 4096)\n");
    printf("  --system TEXT    System prompt\n");
}

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║   Fence Inference Engine v0.1                 ║\n");
    printf("║   Qwen3-4B Q6_K_L — RTX 4060 CUDA            ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");

    std::string model_path = MODEL_PATH;
    std::string single_prompt;
    std::string system_prompt = "You are a helpful assistant.";
    int max_tokens = 512;
    int max_ctx = 4096;
    bool debug = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            single_prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-ctx") == 0 && i + 1 < argc) {
            max_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    g_debug = debug;

    // Load tokenizer
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model_path)) return 1;

    // Load model
    Qwen3Model model;
    model.config.max_ctx = max_ctx;
    if (!model.load(model_path)) return 1;

    // Single prompt mode
    if (!single_prompt.empty()) {
        auto tokens = tokenizer.format_chat(system_prompt, single_prompt);
        printf("Prompt tokens: %zu\n", tokens.size());
        printf("Assistant: ");
        fflush(stdout);

        auto output = model.generate(tokens, max_tokens, true, &tokenizer);
        printf("\n");
        model.unload();
        return 0;
    }

    // Interactive chat loop
    printf("System: %s\n", system_prompt.c_str());
    printf("Type your message (empty line to quit).\n\n");

    std::vector<int> history_tokens;
    
    // Convert system prompt into ChatML tokens
    if (!system_prompt.empty()) {
        history_tokens.push_back(Tokenizer::IM_START);
        auto sys_toks = tokenizer.encode("system\n" + system_prompt);
        history_tokens.insert(history_tokens.end(), sys_toks.begin(), sys_toks.end());
        history_tokens.push_back(Tokenizer::IM_END);
        auto nl_toks = tokenizer.encode("\n");
        history_tokens.insert(history_tokens.end(), nl_toks.begin(), nl_toks.end());
    }

    while (true) {
        printf("You: ");
        fflush(stdout);

        std::string line;
        if (!std::getline(std::cin, line) || line.empty()) break;

        // Append user turn to history
        history_tokens.push_back(Tokenizer::IM_START);
        auto usr_toks = tokenizer.encode("user\n" + line);
        history_tokens.insert(history_tokens.end(), usr_toks.begin(), usr_toks.end());
        history_tokens.push_back(Tokenizer::IM_END);
        auto nl_toks = tokenizer.encode("\n");
        history_tokens.insert(history_tokens.end(), nl_toks.begin(), nl_toks.end());

        // Append assistant header
        history_tokens.push_back(Tokenizer::IM_START);
        auto asst_toks = tokenizer.encode("assistant\n");
        history_tokens.insert(history_tokens.end(), asst_toks.begin(), asst_toks.end());

        printf("  [%zu prompt tokens in active context]\n", history_tokens.size());
        printf("Assistant: ");
        fflush(stdout);

        // Generate assistant response
        auto output = model.generate(history_tokens, max_tokens, true, &tokenizer);
        printf("\n\n");

        // `generate` returns the combined sequence of history + new tokens.
        // It breaks *before* appending IM_END, so we append the cutoff manually.
        history_tokens = output;
        history_tokens.push_back(Tokenizer::IM_END);
        history_tokens.insert(history_tokens.end(), nl_toks.begin(), nl_toks.end());
    }

    model.unload();
    printf("Goodbye!\n");
    return 0;
}
