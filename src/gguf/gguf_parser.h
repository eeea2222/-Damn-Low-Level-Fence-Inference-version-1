#pragma once
// =============================================================================
// gguf_parser.h — Zero-copy GGUF v3 file parser with mmap
//
// Provides:
//   - GGUFFile class: mmap a GGUF file, parse header/metadata/tensor_info
//   - O(1) lookup of tensors by name
//   - O(1) lookup of metadata by key
//   - Direct const void* pointers into mmap region for tensor data
//
// Usage:
//   GGUFFile gguf;
//   gguf.open("model.gguf");
//   auto* tensor = gguf.find_tensor("blk.0.attn_q.weight");
//   const block_q6_K* data = (const block_q6_K*)tensor->data;
//   uint32_t layers = gguf.get_u32("qwen3.block_count");
//   gguf.close();
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>

// ---- GGML tensor type enum (matches GGUF spec) ----
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
    MXFP4   = 39,
};

// ---- GGUF metadata value type enum ----
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// ---- Quantization type info ----
struct GGMLTypeInfo {
    uint32_t block_size;   // elements per block
    uint32_t type_size;    // bytes per block
};

// Returns block_size and type_size for a given GGMLType
GGMLTypeInfo ggml_type_info(GGMLType type);

// Returns human-readable name for a GGMLType
const char* ggml_type_name(GGMLType type);

// ---- Tensor info ----
struct GGUFTensorInfo {
    std::string name;
    uint32_t    n_dims;
    uint64_t    dims[4];         // up to 4 dimensions
    GGMLType    type;
    uint64_t    offset;          // offset relative to tensor_data start
    const void* data;            // direct pointer into mmap region
    size_t      data_size_bytes; // total bytes for this tensor's data

    // Convenience: total number of elements
    uint64_t n_elements() const {
        uint64_t n = 1;
        for (uint32_t i = 0; i < n_dims; ++i) n *= dims[i];
        return n;
    }
};

// ---- Metadata value (simplified — stores scalars and strings) ----
using GGUFMetadataValue = std::variant<
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    float,
    bool,
    std::string,
    uint64_t, int64_t,
    double
    // Arrays are skipped (logged but not stored for now)
>;

struct GGUFMetadataKV {
    std::string        key;
    GGUFValueType      type;
    GGUFMetadataValue  value;
};

// ---- Main GGUF file class ----
class GGUFFile {
public:
    GGUFFile() = default;
    ~GGUFFile();

    // Non-copyable
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    // Move OK
    GGUFFile(GGUFFile&& other) noexcept;
    GGUFFile& operator=(GGUFFile&& other) noexcept;

    // ---- Core API ----

    /// Open and parse a GGUF file. Returns false on error.
    bool open(const std::string& path);

    /// Close the file and unmap memory.
    void close();

    /// Is the file currently open?
    bool is_open() const { return mmap_ptr_ != nullptr; }

    // ---- Header accessors ----
    uint32_t version()      const { return version_; }
    uint64_t tensor_count() const { return tensor_count_; }
    uint64_t metadata_count() const { return metadata_kv_count_; }

    // ---- Metadata accessors ----
    const std::vector<GGUFMetadataKV>& metadata() const { return metadata_; }

    /// Find metadata by key. Returns nullptr if not found.
    const GGUFMetadataKV* find_metadata(const std::string& key) const;

    /// Type-safe metadata getters. Throw std::runtime_error if not found or wrong type.
    std::string get_string(const std::string& key) const;
    uint32_t    get_u32(const std::string& key) const;
    int32_t     get_i32(const std::string& key) const;
    float       get_f32(const std::string& key) const;
    bool        get_bool(const std::string& key) const;
    uint64_t    get_u64(const std::string& key) const;

    // ---- Tensor accessors ----
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }

    /// Find tensor by name. Returns nullptr if not found.
    const GGUFTensorInfo* find_tensor(const std::string& name) const;

    /// File path
    const std::string& path() const { return path_; }

    /// Total file size
    size_t file_size() const { return file_size_; }

    /// Start of tensor data region
    size_t data_offset() const { return tensor_data_offset_; }

private:
    std::string path_;
    void*       mmap_ptr_       = nullptr;
    size_t      file_size_      = 0;
    int         fd_             = -1;

    uint32_t version_           = 0;
    uint64_t tensor_count_      = 0;
    uint64_t metadata_kv_count_ = 0;
    uint32_t alignment_         = 32;  // default GGUF alignment
    size_t   tensor_data_offset_ = 0;

    std::vector<GGUFMetadataKV>  metadata_;
    std::vector<GGUFTensorInfo>  tensors_;

    // O(1) lookup maps
    std::unordered_map<std::string, size_t> metadata_index_;
    std::unordered_map<std::string, size_t> tensor_index_;

    // Internal parsing helpers
    bool parse_header(const uint8_t*& cursor, const uint8_t* end);
    bool parse_metadata(const uint8_t*& cursor, const uint8_t* end);
    bool parse_tensor_infos(const uint8_t*& cursor, const uint8_t* end);
    bool skip_value(GGUFValueType type, const uint8_t*& cursor, const uint8_t* end);

    // Read helpers (advance cursor)
    static std::string read_string(const uint8_t*& cursor, const uint8_t* end);
    template<typename T>
    static T read_scalar(const uint8_t*& cursor, const uint8_t* end);
};
