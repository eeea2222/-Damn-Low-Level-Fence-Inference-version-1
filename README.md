# Fence Inference Engine

Low-level CUDA/C++ LLM inference engine using GGUF quantised models.
No external ML frameworks — just raw CUDA kernels.

## Supported Architectures

| Architecture | Chat Format | Example Models |
|---|---|---|
| Qwen3 | ChatML | Qwen3-4B, Qwen3-8B, Qwen3-14B |
| Qwen2 | ChatML | Qwen2-7B, Qwen2.5-7B |
| LLaMA 3 / 3.1 / 3.2 | LLaMA-3 | Meta-Llama-3-8B-Instruct, Llama-3.1-8B-Instruct |
| LLaMA 2 | LLaMA-2 | Llama-2-7b-chat, CodeLlama-7b-Instruct |
| Mistral / Mixtral | Mistral | Mistral-7B-Instruct-v0.3, Mixtral-8x7B-Instruct |

All models must be in **GGUF format** (Q6_K or Q8_0 quantisation).
The chat format is auto-detected from the `tokenizer.chat_template` field in the GGUF file.

## Build

```bash
make fence          # optimised build
make DEBUG=1 fence  # debug build with CUDA device-side symbols
```

Requires CUDA 12+ and an NVIDIA GPU (Makefile defaults to SM 8.9 / RTX 4060;
edit `SM` in the Makefile for other GPUs).

## Run

```bash
# Interactive chat
./build/fence --model /path/to/model.gguf

# Single prompt
./build/fence --model /path/to/model.gguf --prompt "What is the capital of France?"

# All options
./build/fence --help
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | *(required)* | GGUF model file |
| `--prompt TEXT` | — | Single prompt, then exit |
| `--system TEXT` | You are a helpful assistant. | System prompt |
| `--max-tokens N` | 512 | Max new tokens per response |
| `--max-ctx N` | 4096 | Context window size |
| `--temp F` | 0.7 | Sampling temperature |
| `--top-p F` | 0.8 | Nucleus sampling cutoff |
| `--top-k N` | 50 | Top-K candidates |
| `--rep-penalty F` | 1.15 | Repetition penalty |
| `--debug` | off | Print per-layer activation stats |

## Tests

```bash
make test_q6k_dot                               # kernel unit test (no model needed)
make test_gguf_parser MODEL=/path/to/model.gguf # GGUF parser integration test
make test_gemv_q6k    MODEL=/path/to/model.gguf # GEMV kernel integration test
make debug_forward && ./build/debug_forward /path/to/model.gguf
```

## Source Layout

```
src/
  main.cu                   — CLI entry point, interactive chat loop
  model/
    qwen3.h / qwen3.cu      — Model loading, forward pass, sampling
    chat_template.h         — Chat format detection (ChatML / LLaMA-3 / Mistral / LLaMA-2)
  tokenizer/
    tokenizer.h / tokenizer.cpp  — GPT-2 BPE tokenizer, multi-format prompt building
  gguf/
    gguf_parser.h / gguf_parser.cpp  — Zero-copy mmap GGUF v2/v3 parser
  kernels/
    ops.cuh                 — RMSNorm, RoPE, SwiGLU, GQA attention, softmax
    gemv_q6k.cuh            — Q6_K matrix-vector multiply
    gemv_q8_0.cuh           — Q8_0 matrix-vector multiply (embedding lookup)
    q6k_dot.cuh             — Q6_K dot-product primitives
    q6k_types.h / q8_0_types.h  — Quantised block structs
tests/
  test_q6k_dot.cu           — Q6_K kernel correctness tests
  test_gguf_parser.cpp      — GGUF parser integration tests
  test_gemv_q6k.cu          — GEMV Q6_K integration tests
  debug_forward.cu          — Layer-by-layer activation statistics
```

## Credits & Licenses

- GGUF format / quantisation types: [llama.cpp / GGML](https://github.com/ggerganov/llama.cpp) — MIT License  
- GPT-2 byte-level BPE: [OpenAI tiktoken](https://github.com/openai/tiktoken) — MIT License  
- Engine source code: **MIT License**  
- Model weights: see each model's individual licence (typically Apache 2.0 or Llama Community Licence)
