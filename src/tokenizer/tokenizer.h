#pragma once
// =============================================================================
// tokenizer.h — Minimal BPE tokenizer for Qwen3 (reads vocab from GGUF)
// =============================================================================

#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    /// Load vocabulary from GGUF file (reads the raw bytes directly)
    bool load_from_gguf(const std::string& gguf_path);

    /// Encode text to token IDs (byte-level fallback)
    std::vector<int> encode(const std::string& text) const;

    /// Decode token ID to string
    std::string decode(int token_id) const;

    /// Decode sequence of tokens
    std::string decode(const std::vector<int>& tokens) const;

    /// Format a ChatML prompt
    std::vector<int> format_chat(const std::string& system_prompt,
                                  const std::string& user_message) const;

    /// Vocabulary size
    int vocab_size() const { return (int)id_to_token_.size(); }

    /// Get full vocab for display
    const std::vector<std::string>& vocab() const { return id_to_token_; }

    // Special token IDs
    static constexpr int IM_START = 151644;
    static constexpr int IM_END   = 151645;
    static constexpr int BOS      = 151643;
    static constexpr int EOS      = 151645;
    static constexpr int NEWLINE  = 198;  // '\n' in GPT-2 BPE

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merges_; // "token1 token2" -> rank

    // Find longest token match at position (legacy fallback)
    int find_longest_match(const std::string& text, size_t pos) const;
    
    // Core BPE merge function
    void bpe_merge(std::vector<std::string>& words) const;
};
