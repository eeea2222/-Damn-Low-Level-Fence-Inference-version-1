// =============================================================================
// tokenizer.cpp — BPE tokenizer for Qwen3 (GPT-2 byte-level encoding)
//
// Qwen3 (and GPT-2/tiktoken) use a byte-to-unicode mapping where each
// byte 0x00-0xFF is mapped to a unicode character. Printable ASCII stays
// the same, but bytes like space (0x20), newline (0x0A), tab (0x09) etc.
// are shifted to the U+0100+ range.
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
    while (i < s.size()) {
        unsigned char c = s[i];
        int cp;
        if (c < 0x80) {
            cp = c; i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F); i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); i += 3;
        } else {
            cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) |
                 ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); i += 4;
        }

        if (cp < 65536 && unicode_to_byte[cp] >= 0) {
            result += (char)(unsigned char)unicode_to_byte[cp];
        } else {
            result += '?';  // unmappable
        }
    }
    return result;
}

// ---- GGUF reading helpers ----
static uint64_t read_u64(FILE* f) { uint64_t v; if(fread(&v, 8, 1, f)!=1){}; return v; }
static uint32_t read_u32(FILE* f) { uint32_t v; if(fread(&v, 4, 1, f)!=1){}; return v; }

static std::string read_gguf_string(FILE* f) {
    uint64_t len = read_u64(f);
    std::string s(len, '\0');
    if(fread(&s[0], 1, len, f)!=len){};
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
    read_u64(f);  // tensor_count
    uint64_t metadata_kv_count = read_u64(f);

    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        std::string key = read_gguf_string(f);
        uint32_t vtype = read_u32(f);

        if (key == "tokenizer.ggml.tokens" && vtype == 9) {
            uint32_t arr_type = read_u32(f);
            uint64_t arr_len = read_u64(f);
            if (arr_type != 8) {
                fprintf(stderr, "Tokenizer: tokens array not string\n");
                fclose(f); return false;
            }
            id_to_token_.resize(arr_len);
            for (uint64_t t = 0; t < arr_len; ++t) {
                id_to_token_[t] = read_gguf_string(f);
            }
            printf("Tokenizer: loaded %zu tokens\n", arr_len);
        } else if (key == "tokenizer.ggml.merges" && vtype == 9) {
            uint32_t arr_type = read_u32(f);
            uint64_t arr_len = read_u64(f);
            if (arr_type != 8) {
                fprintf(stderr, "Tokenizer: merges array not string\n");
                fclose(f); return false;
            }
            for (uint64_t m = 0; m < arr_len; ++m) {
                merges_[read_gguf_string(f)] = (int)m;
            }
            printf("Tokenizer: loaded %zu merges\n", arr_len);
        } else {
            skip_value(f, vtype);
        }
    }

    fclose(f);

    if (id_to_token_.empty()) {
        fprintf(stderr, "Tokenizer: no tokens found\n");
        return false;
    }

    // Build reverse map
    for (int i = 0; i < (int)id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = i;
    }

    init_byte_unicode_mapping();
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

    // <|im_start|>system\n{system}<|im_end|>\n
    if (!system_prompt.empty()) {
        tokens.push_back(IM_START);
        auto part = encode("system\n" + system_prompt);
        tokens.insert(tokens.end(), part.begin(), part.end());
        tokens.push_back(IM_END);
        auto nl = encode("\n");
        tokens.insert(tokens.end(), nl.begin(), nl.end());
    }

    // <|im_start|>user\n{message}<|im_end|>\n
    tokens.push_back(IM_START);
    auto usr = encode("user\n" + user_message);
    tokens.insert(tokens.end(), usr.begin(), usr.end());
    tokens.push_back(IM_END);
    auto nl2 = encode("\n");
    tokens.insert(tokens.end(), nl2.begin(), nl2.end());

    // <|im_start|>assistant\n
    tokens.push_back(IM_START);
    auto asst = encode("assistant\n");
    tokens.insert(tokens.end(), asst.begin(), asst.end());

    return tokens;
}
