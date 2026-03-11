#pragma once
// =============================================================================
// chat_template.h — Chat prompt format detection for various model families
//
// Supported formats (all open-source, architecture documented in public papers):
//
//   CHATML   — Qwen2, Qwen3, InternLM2, DeepSeek-V2
//              <|im_start|>role\ncontent<|im_end|>\n
//
//   LLAMA3   — Meta LLaMA 3, LLaMA 3.1, LLaMA 3.2
//              <|begin_of_text|><|start_header_id|>role<|end_header_id|>
//              \n\ncontent<|eot_id|>
//
//   LLAMA2   — Meta LLaMA 2, CodeLlama Instruct
//              <s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST] {resp}</s>
//
//   MISTRAL  — Mistral / Mixtral instruct v0.1+
//              <s>[INST] {user} [/INST] {resp}</s>[INST] …
//
//   RAW      — No template — base / completion models
// =============================================================================

#include <string>

/// Detected chat prompt format
enum class ChatFormat {
    CHATML,   ///< Qwen2/3, InternLM2, DeepSeek-V2
    LLAMA3,   ///< Meta LLaMA 3 / 3.1 / 3.2
    LLAMA2,   ///< Meta LLaMA 2, CodeLlama Instruct
    MISTRAL,  ///< Mistral / Mixtral instruct
    RAW,      ///< No template
};

/// Detect chat format from architecture name and optional jinja2 template string.
/// The template string is taken from the GGUF `tokenizer.chat_template` key.
inline ChatFormat detect_chat_format(const std::string& arch,
                                     const std::string& chat_template_str = "")
{
    // Prefer the explicit jinja template when available
    if (!chat_template_str.empty()) {
        if (chat_template_str.find("im_start") != std::string::npos)
            return ChatFormat::CHATML;
        if (chat_template_str.find("begin_of_text") != std::string::npos ||
            chat_template_str.find("start_header_id") != std::string::npos)
            return ChatFormat::LLAMA3;
        if (chat_template_str.find("<<SYS>>") != std::string::npos)
            return ChatFormat::LLAMA2;
        if (chat_template_str.find("[INST]") != std::string::npos)
            return ChatFormat::MISTRAL;
    }

    // Fall back to architecture name heuristics
    if (arch == "qwen2"  || arch == "qwen3" ||
        arch == "internlm2" || arch == "deepseek2")
        return ChatFormat::CHATML;
    if (arch == "llama")
        return ChatFormat::LLAMA3;   // LLaMA 3 is more prevalent now
    if (arch == "mistral" || arch == "mixtral")
        return ChatFormat::MISTRAL;

    return ChatFormat::RAW;
}

/// Human-readable label for a ChatFormat value
inline const char* chat_format_name(ChatFormat fmt) {
    switch (fmt) {
        case ChatFormat::CHATML:  return "ChatML";
        case ChatFormat::LLAMA3:  return "LLaMA-3";
        case ChatFormat::LLAMA2:  return "LLaMA-2";
        case ChatFormat::MISTRAL: return "Mistral";
        case ChatFormat::RAW:     return "Raw";
    }
    return "Unknown";
}
