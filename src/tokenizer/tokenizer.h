#pragma once
// =============================================================================
// tokenizer.h — BPE tokenizer with multi-architecture chat template support
//
// Reads vocabulary, merge rules, BOS/EOS token IDs, and the optional
// tokenizer.chat_template string from a GGUF file.  Exposes format-aware
// helpers so the caller never needs to hard-code model-specific tokens.
// =============================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include "../model/chat_template.h"

class Tokenizer {
public:
    // ---- Loading ----

    /// Load vocabulary, BPE merge rules, and tokenizer metadata from a GGUF file.
    bool load_from_gguf(const std::string& gguf_path);

    // ---- Encoding / Decoding ----

    /// Encode raw UTF-8 text to token IDs using BPE (with byte-level fallback).
    std::vector<int> encode(const std::string& text) const;

    /// Decode a single token ID to a UTF-8 string.
    std::string decode(int token_id) const;

    /// Decode a sequence of token IDs.
    std::string decode(const std::vector<int>& tokens) const;

    // ---- Single-turn prompt formatting ----

    /// Build a complete single-turn prompt in the model's native chat format.
    std::vector<int> format_chat(const std::string& system_prompt,
                                  const std::string& user_message) const;

    // ---- Multi-turn history helpers ----
    // These abstract over the detected chat format so that callers do not
    // need to hard-code any model-specific token IDs.

    /// Append any required sequence-start token (e.g. <|begin_of_text|>).
    /// Call once before the first turn.
    void begin_sequence(std::vector<int>& tokens) const;

    /// Append a system turn.  For formats that embed the system prompt inside
    /// the first user message (Mistral, LLaMA-2), the content is stored and
    /// prepended automatically when append_user_turn() is first called.
    void append_system_turn(std::vector<int>& tokens,
                            const std::string& content) const;

    /// Append a user turn.
    void append_user_turn(std::vector<int>& tokens,
                          const std::string& content) const;

    /// Append the opening tokens of an assistant response (without closing).
    void append_assistant_header(std::vector<int>& tokens) const;

    /// Append the end-of-turn marker after a completed assistant response.
    void append_turn_end(std::vector<int>& tokens) const;

    // ---- Accessors ----

    /// Detected chat format (set after load_from_gguf succeeds).
    ChatFormat chat_format() const { return chat_format_; }

    /// EOS token ID read from GGUF metadata (falls back to Qwen default).
    int eos_id() const { return eos_id_; }

    /// BOS token ID read from GGUF metadata (falls back to Qwen default).
    int bos_id() const { return bos_id_; }

    /// Vocabulary size.
    int vocab_size() const { return (int)id_to_token_.size(); }

    /// Full vocabulary for display / debugging.
    const std::vector<std::string>& vocab() const { return id_to_token_; }

    // ---- Legacy fixed token IDs (ChatML / Qwen3 defaults) ----
    static constexpr int IM_START = 151644;  ///< <|im_start|>
    static constexpr int IM_END   = 151645;  ///< <|im_end|>
    static constexpr int BOS      = 151643;  ///< Qwen BOS / <|endoftext|>
    static constexpr int EOS      = 151645;  ///< Qwen EOS  / <|im_end|>
    static constexpr int NEWLINE  = 198;     ///< '\n' in GPT-2 BPE

private:
    // ---- Vocabulary tables ----
    std::vector<std::string>            id_to_token_;
    std::unordered_map<std::string,int> token_to_id_;
    std::unordered_map<std::string,int> merges_;  ///< "tok1 tok2" → rank

    // ---- Tokenizer metadata (read from GGUF) ----
    ChatFormat chat_format_  = ChatFormat::CHATML;
    int        bos_id_       = BOS;
    int        eos_id_       = EOS;

    // LLaMA-3 special token IDs (0 = not present in this vocab)
    int llama3_begin_text_id_   = 0;  ///< <|begin_of_text|>
    int llama3_start_header_id_ = 0;  ///< <|start_header_id|>
    int llama3_end_header_id_   = 0;  ///< <|end_header_id|>
    int llama3_eot_id_          = 0;  ///< <|eot_id|>

    // Mistral / LLaMA-2 special token IDs (0 = not present)
    int mistral_bos_id_  = 0;  ///< <s>   (BOS for Mistral / LLaMA-2)
    int mistral_eos_id_  = 0;  ///< </s>  (EOS for Mistral / LLaMA-2)
    int inst_start_id_   = 0;  ///< [INST]
    int inst_end_id_     = 0;  ///< [/INST]

    // Pending system prompt for formats that embed it in the first user turn
    mutable std::string pending_system_;
    mutable bool        system_used_ = false;

    // ---- Internal helpers ----

    /// Append encoded `text` into `tokens`.
    void push_encoded(std::vector<int>& tokens, const std::string& text) const;

    /// Look up a token string in the vocabulary; returns -1 if not found.
    int lookup_token(const std::string& tok_str) const;

    /// Detect and store chat format + look up special tokens after vocab is loaded.
    void detect_format_and_special_tokens(const std::string& chat_template_raw,
                                          const std::string& arch_hint);

    // ---- Core BPE implementation ----
    int  find_longest_match(const std::string& text, size_t pos) const;
    void bpe_merge(std::vector<std::string>& words) const;
};
