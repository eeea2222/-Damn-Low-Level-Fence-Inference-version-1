// =============================================================================
// tokenizer.cpp — BPE tokenizer for multiple model architectures
//
// Supports Qwen2/3 (ChatML), LLaMA-3, LLaMA-2, and Mistral instruct formats.
// Vocabulary, merge rules, BOS/EOS IDs, and the chat template are all read
// from the GGUF file; no model-specific constants are hard-coded.
//
// GPT-2 byte-to-unicode mapping:
//   Each raw byte 0x00–0xFF is mapped to a unicode codepoint so that every
//   possible byte value has a printable representation.  Printable ASCII (and
//   the Latin-1 supplement range used by GPT-2) map to themselves; the rest
//   are shifted into U+0100+.
// =============================================================================

#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <algorithm>

// ---- GPT-2 byte ↔ unicode mapping ----
// This is the standard bytes_to_unicode() from GPT-2/tiktoken.
// Maps each byte value 0-255 to a unicode codepoint.
static int byte_to_unicode[256];
static int unicode_to_byte[65536];  // sparse, only ~256 entries used
static bool mapping_initialized = false;

static void init_byte_unicode_mapping() {
    if (mapping_initialized) return;

    // GPT-2 bytes_to_unicode():
    // - Printable ASCII chars and Latin-1 supplement keep their codepoints
    // - Other bytes (control chars, 0x80-0x9F range) get mapped to U+0100+
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= 0x21 && b <= 0x7E) ||   // '!' to '~'
            (b >= 0xA1 && b <= 0xAC) ||   // '¡' to '¬'
            (b >= 0xAE && b <= 0xFF)) {   // '®' to 'ÿ'
            byte_to_unicode[b] = b;
        } else {
            byte_to_unicode[b] = 256 + n;
            n++;
        }
    }

    memset(unicode_to_byte, -1, sizeof(unicode_to_byte));
    for (int b = 0; b < 256; ++b) {
        unicode_to_byte[byte_to_unicode[b]] = b;
    }
    mapping_initialized = true;
}

// Convert raw bytes to GPT-2 unicode string (for lookup in vocab)
static std::string bytes_to_gpt2_str(const std::string& raw) {
    init_byte_unicode_mapping();
    std::string result;
    for (unsigned char c : raw) {
        int cp = byte_to_unicode[c];
        // Encode codepoint as UTF-8
        if (cp < 0x80) {
            result += (char)cp;
        } else if (cp < 0x800) {
            result += (char)(0xC0 | (cp >> 6));
            result += (char)(0x80 | (cp & 0x3F));
        } else {
            result += (char)(0xE0 | (cp >> 12));
            result += (char)(0x80 | ((cp >> 6) & 0x3F));
            result += (char)(0x80 | (cp & 0x3F));
        }
    }
    return result;
}

// Convert GPT-2 unicode string back to raw bytes (for decoding)
static std::string gpt2_str_to_bytes(const std::string& s) {
    init_byte_unicode_mapping();
    std::string result;
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int cp;
        int char_len;
        if (c < 0x80) {
            cp = c; char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            // Truncated 2-byte sequence: skip only this one malformed leading byte and
            // restart the loop so the next byte is re-examined from scratch.
            if (i + 1 >= n) { result += '?'; ++i; continue; }
            cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i+1]) & 0x3F);
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= n) { result += '?'; ++i; continue; }  // truncated 3-byte sequence
            cp = ((c & 0x0F) << 12) |
                 ((static_cast<unsigned char>(s[i+1]) & 0x3F) << 6) |
                  (static_cast<unsigned char>(s[i+2]) & 0x3F);
            char_len = 3;
        } else {
            if (i + 3 >= n) { result += '?'; ++i; continue; }  // truncated 4-byte sequence
            cp = ((c & 0x07) << 18) |
                 ((static_cast<unsigned char>(s[i+1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(s[i+2]) & 0x3F) << 6) |
                  (static_cast<unsigned char>(s[i+3]) & 0x3F);
            char_len = 4;
        }
        i += char_len;

        if (cp < 65536 && unicode_to_byte[cp] >= 0) {
            result += static_cast<char>(static_cast<unsigned char>(unicode_to_byte[cp]));
        } else {
            result += '?';  // unmappable codepoint
        }
    }
    return result;
}

// ---- GGUF reading helpers (tokenizer's own lightweight reader) ----

static bool read_u32_from_file(FILE* f, uint32_t& out) {
    if (fread(&out, 4, 1, f) != 1) {
        fprintf(stderr, "Tokenizer: unexpected end of file reading u32\n");
        return false;
    }
    return true;
}

static bool read_u64_from_file(FILE* f, uint64_t& out) {
    if (fread(&out, 8, 1, f) != 1) {
        fprintf(stderr, "Tokenizer: unexpected end of file reading u64\n");
        return false;
    }
    return true;
}

// Legacy wrappers used by skip_value (return 0 on error; callers check ferror)
static uint64_t read_u64(FILE* f) { uint64_t v = 0; read_u64_from_file(f, v); return v; }
static uint32_t read_u32(FILE* f) { uint32_t v = 0; read_u32_from_file(f, v); return v; }

static std::string read_gguf_string(FILE* f) {
    uint64_t len = read_u64(f);
    if (ferror(f) || feof(f)) return {};
    std::string s(len, '\0');
    if (len > 0 && fread(&s[0], 1, len, f) != len) {
        fprintf(stderr, "Tokenizer: unexpected end of file reading string of length %llu\n",
                (unsigned long long)len);
        return {};
    }
    return s;
}

static void skip_value(FILE* f, uint32_t vtype) {
    static const int sizes[] = {1,1,2,2,4,4,4,1, 0, 0, 8,8,8};
    if (vtype <= 7 || (vtype >= 10 && vtype <= 12)) {
        fseek(f, sizes[vtype], SEEK_CUR);
    } else if (vtype == 8) {
        uint64_t len = read_u64(f);
        fseek(f, (long)len, SEEK_CUR);
    } else if (vtype == 9) {
        uint32_t at = read_u32(f);
        uint64_t al = read_u64(f);
        for (uint64_t i = 0; i < al; ++i) skip_value(f, at);
    }
}

bool Tokenizer::load_from_gguf(const std::string& gguf_path) {
    FILE* f = fopen(gguf_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Tokenizer: cannot open %s\n", gguf_path.c_str()); return false; }

    fseek(f, 4, SEEK_CUR);  // magic
    fseek(f, 4, SEEK_CUR);  // version
    read_u64(f);             // tensor_count
    uint64_t metadata_kv_count = read_u64(f);

    std::string chat_template_raw;
    std::string arch_hint;

    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        if (ferror(f) || feof(f)) break;

        std::string key = read_gguf_string(f);
        uint32_t vtype  = read_u32(f);

        if (key == "tokenizer.ggml.tokens" && vtype == 9) {
            uint32_t arr_type = read_u32(f);
            uint64_t arr_len  = read_u64(f);
            if (arr_type != 8) {
                fprintf(stderr, "Tokenizer: tokens array not string type\n");
                fclose(f); return false;
            }
            id_to_token_.resize(arr_len);
            for (uint64_t t = 0; t < arr_len; ++t) {
                id_to_token_[t] = read_gguf_string(f);
            }
            printf("Tokenizer: loaded %zu tokens\n", (size_t)arr_len);

        } else if (key == "tokenizer.ggml.merges" && vtype == 9) {
            uint32_t arr_type = read_u32(f);
            uint64_t arr_len  = read_u64(f);
            if (arr_type != 8) {
                fprintf(stderr, "Tokenizer: merges array not string type\n");
                fclose(f); return false;
            }
            for (uint64_t m = 0; m < arr_len; ++m) {
                merges_[read_gguf_string(f)] = (int)m;
            }
            printf("Tokenizer: loaded %zu BPE merges\n", (size_t)arr_len);

        } else if (key == "tokenizer.ggml.bos_token_id" && (vtype == 4 || vtype == 5)) {
            // UINT32 (4) or INT32 (5)
            uint32_t v = read_u32(f);
            bos_id_ = (int)v;

        } else if (key == "tokenizer.ggml.eos_token_id" && (vtype == 4 || vtype == 5)) {
            uint32_t v = read_u32(f);
            eos_id_ = (int)v;

        } else if (key == "tokenizer.chat_template" && vtype == 8) {
            chat_template_raw = read_gguf_string(f);

        } else if (key == "general.architecture" && vtype == 8) {
            arch_hint = read_gguf_string(f);

        } else {
            skip_value(f, vtype);
        }
    }

    fclose(f);

    if (id_to_token_.empty()) {
        fprintf(stderr, "Tokenizer: no tokens found in GGUF\n");
        return false;
    }

    // Build reverse map
    for (int idx = 0; idx < (int)id_to_token_.size(); ++idx) {
        token_to_id_[id_to_token_[idx]] = idx;
    }

    init_byte_unicode_mapping();

    // Detect chat format and look up special token IDs
    detect_format_and_special_tokens(chat_template_raw, arch_hint);

    printf("Tokenizer: format=%s  bos=%d  eos=%d\n",
           chat_format_name(chat_format_), bos_id_, eos_id_);
    return true;
}

// Greedy longest-match tokenization against GPT-2-encoded text
int Tokenizer::find_longest_match(const std::string& text, size_t pos) const {
    size_t remaining = text.size() - pos;
    size_t max_len = std::min(remaining, (size_t)128);

    for (size_t len = max_len; len > 0; --len) {
        std::string candidate = text.substr(pos, len);
        auto it = token_to_id_.find(candidate);
        if (it != token_to_id_.end()) {
            return (int)len;
        }
    }
    return 0;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // Convert raw text to GPT-2 unicode representation
    std::string encoded = bytes_to_gpt2_str(text);

    // Initial split: each GPT-2 unicode character becomes a word
    std::vector<std::string> words;
    for (size_t i = 0; i < encoded.size(); ) {
        unsigned char c = encoded[i];
        int char_len = 1;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        
        words.push_back(encoded.substr(i, char_len));
        i += char_len;
    }

    if (!merges_.empty()) {
        bpe_merge(words);
    } else {
        // Fallback to greedy longest-match if no merges
        // (Re-build `words` using legacy approach for robustness)
        words.clear();
        size_t pos = 0;
        while (pos < encoded.size()) {
            int match_len = find_longest_match(encoded, pos);
            if (match_len > 0) {
                words.push_back(encoded.substr(pos, match_len));
                pos += match_len;
            } else {
                words.push_back(encoded.substr(pos, 1));
                pos++;
            }
        }
    }

    std::vector<int> tokens;
    for (const auto& w : words) {
        auto it = token_to_id_.find(w);
        if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            // Unmappable byte fallback
            for (size_t i = 0; i < w.size(); ) {
                unsigned char c = w[i];
                int char_len = 1;
                if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                
                auto cit = token_to_id_.find(w.substr(i, char_len));
                if (cit != token_to_id_.end()) tokens.push_back(cit->second);
                i += char_len;
            }
        }
    }

    return tokens;
}

void Tokenizer::bpe_merge(std::vector<std::string>& words) const {
    if (words.size() < 2) return;

    struct Symbol {
        int prev, next;
        int rank; // rank of pair (this, next)
    };

    std::vector<Symbol> syms(words.size());
    for (int i = 0; i < (int)words.size(); ++i) {
        syms[i].prev = i - 1;
        syms[i].next = i + 1;
        syms[i].rank = 1e9;
    }
    syms.back().next = -1;

    auto eval_pair = [&](int i) {
        if (i < 0) return;
        if (syms[i].next < 0) {
            syms[i].rank = 1e9;
            return;
        }
        std::string pair = words[i] + " " + words[syms[i].next];
        auto it = merges_.find(pair);
        syms[i].rank = (it != merges_.end()) ? it->second : 1e9;
    };

    // Initial evaluation
    for (int i = 0; i < (int)words.size() - 1; ++i) {
        eval_pair(i);
    }

    while (true) {
        int best_rank = 1e9;
        int best_i = -1;
        
        // Find pair with lowest rank using index traversal
        int curr = 0;
        while (curr >= 0) {
            if (syms[curr].rank < best_rank) {
                best_rank = syms[curr].rank;
                best_i = curr;
            }
            curr = syms[curr].next;
        }

        if (best_i == -1 || best_rank == 1e9) break;

        // Merge best_i and best_i.next
        int right_i = syms[best_i].next;
        
        // Combine text in place
        words[best_i] += words[right_i];
        
        // Update linked list pointers
        syms[best_i].next = syms[right_i].next;
        if (syms[right_i].next >= 0) {
            syms[syms[right_i].next].prev = best_i;
        }

        // Re-evaluate affected pairs immediately bordering the merged pair
        eval_pair(syms[best_i].prev);
        eval_pair(best_i);
    }

    // Collect result
    std::vector<std::string> res;
    int curr = 0;
    while (curr >= 0) {
        res.push_back(std::move(words[curr]));
        curr = syms[curr].next;
    }
    words = std::move(res);
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id >= 0 && token_id < (int)id_to_token_.size()) {
        // Convert from GPT-2 unicode back to raw bytes
        return gpt2_str_to_bytes(id_to_token_[token_id]);
    }
    return "";
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int t : tokens) result += decode(t);
    return result;
}

std::vector<int> Tokenizer::format_chat(
    const std::string& system_prompt,
    const std::string& user_message) const
{
    std::vector<int> tokens;
    system_used_ = false;
    pending_system_.clear();

    begin_sequence(tokens);
    if (!system_prompt.empty())
        append_system_turn(tokens, system_prompt);
    append_user_turn(tokens, user_message);
    append_assistant_header(tokens);
    return tokens;
}

// ============================================================================
// Internal helpers
// ============================================================================

int Tokenizer::lookup_token(const std::string& tok_str) const {
    auto it = token_to_id_.find(tok_str);
    return (it != token_to_id_.end()) ? it->second : -1;
}

void Tokenizer::push_encoded(std::vector<int>& tokens,
                             const std::string& text) const {
    auto toks = encode(text);
    tokens.insert(tokens.end(), toks.begin(), toks.end());
}

void Tokenizer::detect_format_and_special_tokens(
    const std::string& chat_template_raw,
    const std::string& arch_hint)
{
    chat_format_ = detect_chat_format(arch_hint, chat_template_raw);

    // LLaMA-3 special tokens
    int id;
    if ((id = lookup_token("<|begin_of_text|>"))  >= 0) llama3_begin_text_id_   = id;
    if ((id = lookup_token("<|start_header_id|>")) >= 0) llama3_start_header_id_ = id;
    if ((id = lookup_token("<|end_header_id|>"))   >= 0) llama3_end_header_id_   = id;
    if ((id = lookup_token("<|eot_id|>"))           >= 0) llama3_eot_id_          = id;

    // Mistral / LLaMA-2 special tokens
    if ((id = lookup_token("<s>"))     >= 0) mistral_bos_id_  = id;
    if ((id = lookup_token("</s>"))    >= 0) mistral_eos_id_  = id;
    if ((id = lookup_token("[INST]"))  >= 0) inst_start_id_   = id;
    if ((id = lookup_token("[/INST]")) >= 0) inst_end_id_      = id;

    // If GGUF provided explicit BOS/EOS but we still have the defaults, keep GGUF values.
    // If GGUF gave no special IDs (still at BOS/EOS defaults from constructor) and this
    // looks like a LLaMA-3 model, set them from the special token IDs we just found.
    if (chat_format_ == ChatFormat::LLAMA3 && llama3_begin_text_id_ > 0)
        bos_id_ = llama3_begin_text_id_;
    if (chat_format_ == ChatFormat::LLAMA3 && llama3_eot_id_ > 0)
        eos_id_ = llama3_eot_id_;
    if ((chat_format_ == ChatFormat::MISTRAL || chat_format_ == ChatFormat::LLAMA2)
        && mistral_eos_id_ > 0)
        eos_id_ = mistral_eos_id_;
}

// ============================================================================
// Multi-turn history helpers
// ============================================================================

void Tokenizer::begin_sequence(std::vector<int>& tokens) const {
    system_used_ = false;
    pending_system_.clear();

    switch (chat_format_) {
        case ChatFormat::LLAMA3:
            if (llama3_begin_text_id_ > 0)
                tokens.push_back(llama3_begin_text_id_);
            break;
        case ChatFormat::LLAMA2:
        case ChatFormat::MISTRAL:
            if (mistral_bos_id_ > 0)
                tokens.push_back(mistral_bos_id_);
            break;
        default:
            break;  // ChatML / Raw: no explicit sequence-start token
    }
}

void Tokenizer::append_system_turn(std::vector<int>& tokens,
                                   const std::string& content) const {
    switch (chat_format_) {
        case ChatFormat::CHATML:
            tokens.push_back(IM_START);
            push_encoded(tokens, "system\n" + content);
            tokens.push_back(IM_END);
            push_encoded(tokens, "\n");
            break;

        case ChatFormat::LLAMA3:
            if (llama3_start_header_id_ > 0) tokens.push_back(llama3_start_header_id_);
            push_encoded(tokens, "system");
            if (llama3_end_header_id_ > 0)   tokens.push_back(llama3_end_header_id_);
            push_encoded(tokens, "\n\n" + content);
            if (llama3_eot_id_ > 0)           tokens.push_back(llama3_eot_id_);
            break;

        case ChatFormat::LLAMA2:
        case ChatFormat::MISTRAL:
            // These formats embed the system prompt inside the first [INST] block.
            // Store it and include it when append_user_turn is first called.
            pending_system_ = content;
            break;

        case ChatFormat::RAW:
            push_encoded(tokens, content + "\n");
            break;
    }
}

void Tokenizer::append_user_turn(std::vector<int>& tokens,
                                 const std::string& content) const {
    switch (chat_format_) {
        case ChatFormat::CHATML:
            tokens.push_back(IM_START);
            push_encoded(tokens, "user\n" + content);
            tokens.push_back(IM_END);
            push_encoded(tokens, "\n");
            break;

        case ChatFormat::LLAMA3:
            if (llama3_start_header_id_ > 0) tokens.push_back(llama3_start_header_id_);
            push_encoded(tokens, "user");
            if (llama3_end_header_id_ > 0)   tokens.push_back(llama3_end_header_id_);
            push_encoded(tokens, "\n\n" + content);
            if (llama3_eot_id_ > 0)           tokens.push_back(llama3_eot_id_);
            break;

        case ChatFormat::LLAMA2: {
            // First user turn includes system prompt if present
            if (inst_start_id_ > 0) tokens.push_back(inst_start_id_);
            if (!pending_system_.empty() && !system_used_) {
                push_encoded(tokens, " <<SYS>>\n" + pending_system_ + "\n<</SYS>>\n\n");
                system_used_ = true;
            } else {
                push_encoded(tokens, " ");
            }
            push_encoded(tokens, content + " ");
            if (inst_end_id_ > 0) tokens.push_back(inst_end_id_);
            break;
        }

        case ChatFormat::MISTRAL: {
            if (inst_start_id_ > 0) tokens.push_back(inst_start_id_);
            if (!pending_system_.empty() && !system_used_) {
                push_encoded(tokens, " " + pending_system_ + "\n\n");
                system_used_ = true;
            } else {
                push_encoded(tokens, " ");
            }
            push_encoded(tokens, content + " ");
            if (inst_end_id_ > 0) tokens.push_back(inst_end_id_);
            break;
        }

        case ChatFormat::RAW:
            push_encoded(tokens, content + "\n");
            break;
    }
}

void Tokenizer::append_assistant_header(std::vector<int>& tokens) const {
    switch (chat_format_) {
        case ChatFormat::CHATML:
            tokens.push_back(IM_START);
            push_encoded(tokens, "assistant\n");
            break;

        case ChatFormat::LLAMA3:
            if (llama3_start_header_id_ > 0) tokens.push_back(llama3_start_header_id_);
            push_encoded(tokens, "assistant");
            if (llama3_end_header_id_ > 0)   tokens.push_back(llama3_end_header_id_);
            push_encoded(tokens, "\n\n");
            break;

        case ChatFormat::LLAMA2:
        case ChatFormat::MISTRAL:
            // After [INST]…[/INST] the model generates directly; no explicit header.
            push_encoded(tokens, " ");
            break;

        case ChatFormat::RAW:
            break;
    }
}

void Tokenizer::append_turn_end(std::vector<int>& tokens) const {
    switch (chat_format_) {
        case ChatFormat::CHATML:
            tokens.push_back(IM_END);
            push_encoded(tokens, "\n");
            break;

        case ChatFormat::LLAMA3:
            if (llama3_eot_id_ > 0)
                tokens.push_back(llama3_eot_id_);
            break;

        case ChatFormat::LLAMA2:
        case ChatFormat::MISTRAL:
            if (mistral_eos_id_ > 0)
                tokens.push_back(mistral_eos_id_);
            break;

        case ChatFormat::RAW:
            break;
    }
}
