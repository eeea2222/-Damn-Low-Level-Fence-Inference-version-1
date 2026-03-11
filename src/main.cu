// =============================================================================
// main.cu — Fence Inference Engine: Multi-architecture interactive chat
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

static void print_usage(const char* prog) {
    printf("Usage: %s --model PATH [options]\n\n", prog);
    printf("Required:\n");
    printf("  --model PATH       GGUF model file\n\n");
    printf("Options:\n");
    printf("  --prompt TEXT      Single prompt (non-interactive mode)\n");
    printf("  --system TEXT      System prompt (default: 'You are a helpful assistant.')\n");
    printf("  --max-tokens N     Max new tokens to generate (default: 512)\n");
    printf("  --max-ctx N        Max context length in tokens (default: 4096)\n");
    printf("  --temp F           Sampling temperature (default: 0.7)\n");
    printf("  --top-p F          Top-p nucleus cutoff (default: 0.8)\n");
    printf("  --top-k N          Top-k candidates (default: 50)\n");
    printf("  --rep-penalty F    Repetition penalty (default: 1.15)\n");
    printf("  --debug            Print per-layer activation statistics\n");
    printf("  --help             Show this help\n\n");
    printf("Supported architectures: Qwen2, Qwen3, LLaMA-3, LLaMA-2, Mistral, Mixtral\n");
}

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║   Fence Inference Engine v0.2                     ║\n");
    printf("║   Multi-architecture GGUF — CUDA inference        ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n\n");

    std::string model_path;
    std::string single_prompt;
    std::string system_prompt = "You are a helpful assistant.";
    int   max_tokens = 512;
    int   max_ctx    = 4096;
    bool  debug      = false;

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--model")      == 0 && i + 1 < argc) model_path    = argv[++i];
        else if (strcmp(argv[i], "--prompt")     == 0 && i + 1 < argc) single_prompt = argv[++i];
        else if (strcmp(argv[i], "--system")     == 0 && i + 1 < argc) system_prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-ctx")    == 0 && i + 1 < argc) max_ctx       = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp")       == 0 && i + 1 < argc) { /* handled below */ ++i; }
        else if (strcmp(argv[i], "--top-p")      == 0 && i + 1 < argc) { ++i; }
        else if (strcmp(argv[i], "--top-k")      == 0 && i + 1 < argc) { ++i; }
        else if (strcmp(argv[i], "--rep-penalty")== 0 && i + 1 < argc) { ++i; }
        else if (strcmp(argv[i], "--debug")      == 0) debug = true;
        else if (strcmp(argv[i], "--help")       == 0) { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model PATH is required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Re-parse sampling flags now that we know they're valid
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--temp")        == 0 && i + 1 < argc) { /* loaded after model */ ++i; }
        else if (strcmp(argv[i], "--top-p")       == 0 && i + 1 < argc) { ++i; }
        else if (strcmp(argv[i], "--top-k")       == 0 && i + 1 < argc) { ++i; }
        else if (strcmp(argv[i], "--rep-penalty")== 0 && i + 1 < argc) { ++i; }
    }

    g_debug = debug;

    // ---- Load tokenizer ----
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model_path)) return 1;

    printf("Chat format : %s\n\n", chat_format_name(tokenizer.chat_format()));

    // ---- Load model ----
    Qwen3Model model;
    model.config.max_ctx = max_ctx;

    // Apply sampling parameters from CLI
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--temp")        == 0 && i + 1 < argc)
            model.config.temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p")       == 0 && i + 1 < argc)
            model.config.top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k")       == 0 && i + 1 < argc)
            model.config.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc)
            model.config.repetition_penalty = (float)atof(argv[++i]);
        else if (argv[i][0] == '-' && argv[i][1] == '-' && i + 1 < argc) ++i;
    }

    if (!model.load(model_path)) return 1;

    // ---- Single-prompt mode ----
    if (!single_prompt.empty()) {
        auto tokens = tokenizer.format_chat(system_prompt, single_prompt);
        printf("Prompt tokens: %zu\n", tokens.size());
        printf("Assistant: ");
        fflush(stdout);
        model.generate(tokens, max_tokens, true, &tokenizer);
        printf("\n");
        model.unload();
        return 0;
    }

    // ---- Interactive chat loop ----
    printf("System: %s\n", system_prompt.c_str());
    printf("Type your message (empty line to quit).\n\n");

    std::vector<int> history;

    // Initialise context with sequence-start token(s) + system prompt
    tokenizer.begin_sequence(history);
    if (!system_prompt.empty())
        tokenizer.append_system_turn(history, system_prompt);

    while (true) {
        printf("You: ");
        fflush(stdout);

        std::string line;
        if (!std::getline(std::cin, line) || line.empty()) break;

        // Truncate context if we're approaching the limit (keep at least 256 tokens free)
        if ((int)history.size() > max_ctx - 256) {
            fprintf(stderr, "\n[Context limit approaching — clearing history]\n");
            history.clear();
            tokenizer.begin_sequence(history);
            if (!system_prompt.empty())
                tokenizer.append_system_turn(history, system_prompt);
        }

        tokenizer.append_user_turn(history, line);
        tokenizer.append_assistant_header(history);

        printf("  [%zu tokens in context]\n", history.size());
        printf("Assistant: ");
        fflush(stdout);

        // Generate: returns history + new tokens (stops before turn-end marker)
        auto output = model.generate(history, max_tokens, true, &tokenizer);
        printf("\n\n");

        // Commit the generated response into history and close the turn
        history = std::move(output);
        tokenizer.append_turn_end(history);
    }

    model.unload();
    printf("Goodbye!\n");
    return 0;
}

