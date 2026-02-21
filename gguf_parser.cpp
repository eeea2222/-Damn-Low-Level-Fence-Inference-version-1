// =============================================================================
// gguf_parser.cpp — GGUF v3 file parser implementation
//
// Memory-maps a GGUF file and parses the header, metadata KV pairs, and
// tensor info entries. Tensor data pointers point directly into the mmap
// region for zero-copy access.
// =============================================================================

#include "gguf_parser.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <algorithm>

// POSIX
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ---- GGML type info table ----
GGMLTypeInfo ggml_type_info(GGMLType type) {
    // (block_size, type_size_in_bytes)
    switch (type) {
        case GGMLType::F32:     return {1, 4};
        case GGMLType::F16:     return {1, 2};
        case GGMLType::Q4_0:    return {32, 18};
        case GGMLType::Q4_1:    return {32, 20};
        case GGMLType::Q5_0:    return {32, 22};
        case GGMLType::Q5_1:    return {32, 24};
        case GGMLType::Q8_0:    return {32, 34};
        case GGMLType::Q8_1:    return {32, 40};
        case GGMLType::Q2_K:    return {256, 84};
        case GGMLType::Q3_K:    return {256, 110};
        case GGMLType::Q4_K:    return {256, 144};
        case GGMLType::Q5_K:    return {256, 176};
        case GGMLType::Q6_K:    return {256, 210};
        case GGMLType::Q8_K:    return {256, 292};
        case GGMLType::BF16:    return {1, 2};
        case GGMLType::I8:      return {1, 1};
        case GGMLType::I16:     return {1, 2};
        case GGMLType::I32:     return {1, 4};
        case GGMLType::I64:     return {1, 8};
        case GGMLType::F64:     return {1, 8};
        default:
            fprintf(stderr, "WARNING: unknown GGMLType %u, assuming F32\n",
                    static_cast<uint32_t>(type));
            return {1, 4};
    }
}

const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:     return "F32";
        case GGMLType::F16:     return "F16";
        case GGMLType::Q4_0:    return "Q4_0";
        case GGMLType::Q4_1:    return "Q4_1";
        case GGMLType::Q5_0:    return "Q5_0";
        case GGMLType::Q5_1:    return "Q5_1";
        case GGMLType::Q8_0:    return "Q8_0";
        case GGMLType::Q8_1:    return "Q8_1";
        case GGMLType::Q2_K:    return "Q2_K";
        case GGMLType::Q3_K:    return "Q3_K";
        case GGMLType::Q4_K:    return "Q4_K";
        case GGMLType::Q5_K:    return "Q5_K";
        case GGMLType::Q6_K:    return "Q6_K";
        case GGMLType::Q8_K:    return "Q8_K";
        case GGMLType::BF16:    return "BF16";
        case GGMLType::I8:      return "I8";
        case GGMLType::I16:     return "I16";
        case GGMLType::I32:     return "I32";
        case GGMLType::I64:     return "I64";
        case GGMLType::F64:     return "F64";
        default:                return "UNKNOWN";
    }
}

// ---- Read helpers ----

template<typename T>
T GGUFFile::read_scalar(const uint8_t*& cursor, const uint8_t* end) {
    if (cursor + sizeof(T) > end) {
        throw std::runtime_error("GGUF: unexpected end of file while reading scalar");
    }
    T val;
    memcpy(&val, cursor, sizeof(T));
    cursor += sizeof(T);
    return val;
}

std::string GGUFFile::read_string(const uint8_t*& cursor, const uint8_t* end) {
    uint64_t len = read_scalar<uint64_t>(cursor, end);
    if (cursor + len > end) {
        throw std::runtime_error("GGUF: unexpected end of file while reading string");
    }
    std::string s(reinterpret_cast<const char*>(cursor), len);
    cursor += len;
    return s;
}

// ---- Skip a metadata value (for arrays and unsupported types) ----
bool GGUFFile::skip_value(GGUFValueType type, const uint8_t*& cursor, const uint8_t* end) {
    switch (type) {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
        case GGUFValueType::BOOL:
            cursor += 1; break;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
            cursor += 2; break;
        case GGUFValueType::UINT32:
        case GGUFValueType::INT32:
        case GGUFValueType::FLOAT32:
            cursor += 4; break;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            cursor += 8; break;
        case GGUFValueType::STRING: {
            uint64_t len = read_scalar<uint64_t>(cursor, end);
            cursor += len;
            break;
        }
        case GGUFValueType::ARRAY: {
            auto arr_type = read_scalar<GGUFValueType>(cursor, end);
            uint64_t arr_len = read_scalar<uint64_t>(cursor, end);
            for (uint64_t i = 0; i < arr_len; ++i) {
                if (!skip_value(arr_type, cursor, end)) return false;
            }
            break;
        }
        default:
            fprintf(stderr, "GGUF: unknown value type %u\n",
                    static_cast<uint32_t>(type));
            return false;
    }
    return cursor <= end;
}

// ---- Parse header ----
bool GGUFFile::parse_header(const uint8_t*& cursor, const uint8_t* end) {
    // Magic: "GGUF" = 0x46554747
    uint32_t magic = read_scalar<uint32_t>(cursor, end);
    if (magic != 0x46554747) {
        fprintf(stderr, "GGUF: invalid magic 0x%08X (expected 0x46554747 'GGUF')\n", magic);
        return false;
    }

    version_ = read_scalar<uint32_t>(cursor, end);
    if (version_ < 2 || version_ > 3) {
        fprintf(stderr, "GGUF: unsupported version %u (expected 2 or 3)\n", version_);
        return false;
    }

    tensor_count_      = read_scalar<uint64_t>(cursor, end);
    metadata_kv_count_ = read_scalar<uint64_t>(cursor, end);

    return true;
}

// ---- Parse metadata KV pairs ----
bool GGUFFile::parse_metadata(const uint8_t*& cursor, const uint8_t* end) {
    metadata_.reserve(metadata_kv_count_);

    for (uint64_t i = 0; i < metadata_kv_count_; ++i) {
        GGUFMetadataKV kv;
        kv.key  = read_string(cursor, end);
        kv.type = read_scalar<GGUFValueType>(cursor, end);

        bool stored = true;

        switch (kv.type) {
            case GGUFValueType::UINT8:
                kv.value = read_scalar<uint8_t>(cursor, end); break;
            case GGUFValueType::INT8:
                kv.value = read_scalar<int8_t>(cursor, end); break;
            case GGUFValueType::UINT16:
                kv.value = read_scalar<uint16_t>(cursor, end); break;
            case GGUFValueType::INT16:
                kv.value = read_scalar<int16_t>(cursor, end); break;
            case GGUFValueType::UINT32:
                kv.value = read_scalar<uint32_t>(cursor, end); break;
            case GGUFValueType::INT32:
                kv.value = read_scalar<int32_t>(cursor, end); break;
            case GGUFValueType::FLOAT32:
                kv.value = read_scalar<float>(cursor, end); break;
            case GGUFValueType::BOOL:
                kv.value = static_cast<bool>(read_scalar<uint8_t>(cursor, end)); break;
            case GGUFValueType::STRING:
                kv.value = read_string(cursor, end); break;
            case GGUFValueType::UINT64:
                kv.value = read_scalar<uint64_t>(cursor, end); break;
            case GGUFValueType::INT64:
                kv.value = read_scalar<int64_t>(cursor, end); break;
            case GGUFValueType::FLOAT64:
                kv.value = read_scalar<double>(cursor, end); break;
            case GGUFValueType::ARRAY: {
                // Skip arrays (tokenizer data, etc.) — too large to store in variant
                auto arr_type = read_scalar<GGUFValueType>(cursor, end);
                uint64_t arr_len = read_scalar<uint64_t>(cursor, end);
                for (uint64_t a = 0; a < arr_len; ++a) {
                    if (!skip_value(arr_type, cursor, end)) return false;
                }
                stored = false;
                break;
            }
            default:
                fprintf(stderr, "GGUF: unknown metadata value type %u for key '%s'\n",
                        static_cast<uint32_t>(kv.type), kv.key.c_str());
                return false;
        }

        // Check for alignment override
        if (kv.key == "general.alignment" && stored) {
            if (kv.type == GGUFValueType::UINT32) {
                alignment_ = std::get<uint32_t>(kv.value);
            }
        }

        if (stored) {
            metadata_index_[kv.key] = metadata_.size();
            metadata_.push_back(std::move(kv));
        }
    }

    return true;
}

// ---- Parse tensor info entries ----
bool GGUFFile::parse_tensor_infos(const uint8_t*& cursor, const uint8_t* end) {
    tensors_.reserve(tensor_count_);

    for (uint64_t i = 0; i < tensor_count_; ++i) {
        GGUFTensorInfo ti{};
        ti.name   = read_string(cursor, end);
        ti.n_dims = read_scalar<uint32_t>(cursor, end);

        if (ti.n_dims > 4) {
            fprintf(stderr, "GGUF: tensor '%s' has %u dims (max 4)\n",
                    ti.name.c_str(), ti.n_dims);
            return false;
        }

        for (uint32_t d = 0; d < ti.n_dims; ++d) {
            ti.dims[d] = read_scalar<uint64_t>(cursor, end);
        }
        // Zero out unused dims
        for (uint32_t d = ti.n_dims; d < 4; ++d) {
            ti.dims[d] = 1;
        }

        ti.type   = read_scalar<GGMLType>(cursor, end);
        ti.offset = read_scalar<uint64_t>(cursor, end);

        // data pointer and size will be set after mmap
        ti.data = nullptr;
        ti.data_size_bytes = 0;

        tensor_index_[ti.name] = tensors_.size();
        tensors_.push_back(std::move(ti));
    }

    return true;
}

// ---- Open & parse ----
bool GGUFFile::open(const std::string& path) {
    if (mmap_ptr_) close();

    path_ = path;

    // Open file
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "GGUF: cannot open '%s': %s\n", path.c_str(), strerror(errno));
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(fd_, &st) < 0) {
        fprintf(stderr, "GGUF: fstat failed: %s\n", strerror(errno));
        ::close(fd_); fd_ = -1;
        return false;
    }
    file_size_ = static_cast<size_t>(st.st_size);

    // mmap the entire file
    mmap_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        fprintf(stderr, "GGUF: mmap failed: %s\n", strerror(errno));
        mmap_ptr_ = nullptr;
        ::close(fd_); fd_ = -1;
        return false;
    }

    // Advise kernel: we'll read sequentially during parsing
    madvise(mmap_ptr_, file_size_, MADV_SEQUENTIAL);

    const auto* base = static_cast<const uint8_t*>(mmap_ptr_);
    const auto* end  = base + file_size_;
    const auto* cursor = base;

    try {
        if (!parse_header(cursor, end))      { close(); return false; }
        if (!parse_metadata(cursor, end))    { close(); return false; }
        if (!parse_tensor_infos(cursor, end)) { close(); return false; }
    } catch (const std::exception& ex) {
        fprintf(stderr, "GGUF: parse error: %s\n", ex.what());
        close();
        return false;
    }

    // Compute tensor data start (aligned from current position)
    size_t header_end = static_cast<size_t>(cursor - base);
    tensor_data_offset_ = header_end + (alignment_ - (header_end % alignment_)) % alignment_;

    // Now resolve tensor data pointers and sizes
    for (auto& ti : tensors_) {
        size_t abs_offset = tensor_data_offset_ + ti.offset;
        if (abs_offset >= file_size_) {
            fprintf(stderr, "GGUF: tensor '%s' offset %zu exceeds file size %zu\n",
                    ti.name.c_str(), abs_offset, file_size_);
            close();
            return false;
        }
        ti.data = base + abs_offset;

        // Compute data size
        auto info = ggml_type_info(ti.type);
        uint64_t n_elem = ti.n_elements();
        uint64_t n_blocks = (n_elem + info.block_size - 1) / info.block_size;
        ti.data_size_bytes = static_cast<size_t>(n_blocks * info.type_size);
    }

    // Switch to random access mode (tensors are accessed on demand)
    madvise(mmap_ptr_, file_size_, MADV_RANDOM);

    return true;
}

// ---- Close ----
void GGUFFile::close() {
    if (mmap_ptr_) {
        munmap(mmap_ptr_, file_size_);
        mmap_ptr_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
    tensor_data_offset_ = 0;
    version_ = 0;
    tensor_count_ = 0;
    metadata_kv_count_ = 0;
    alignment_ = 32;

    metadata_.clear();
    tensors_.clear();
    metadata_index_.clear();
    tensor_index_.clear();
}

// ---- Move semantics ----
GGUFFile::~GGUFFile() { close(); }

GGUFFile::GGUFFile(GGUFFile&& other) noexcept {
    *this = std::move(other);
}

GGUFFile& GGUFFile::operator=(GGUFFile&& other) noexcept {
    if (this != &other) {
        close();
        path_               = std::move(other.path_);
        mmap_ptr_           = other.mmap_ptr_;           other.mmap_ptr_ = nullptr;
        file_size_          = other.file_size_;           other.file_size_ = 0;
        fd_                 = other.fd_;                  other.fd_ = -1;
        version_            = other.version_;
        tensor_count_       = other.tensor_count_;
        metadata_kv_count_  = other.metadata_kv_count_;
        alignment_          = other.alignment_;
        tensor_data_offset_ = other.tensor_data_offset_;
        metadata_           = std::move(other.metadata_);
        tensors_            = std::move(other.tensors_);
        metadata_index_     = std::move(other.metadata_index_);
        tensor_index_       = std::move(other.tensor_index_);
    }
    return *this;
}

// ---- Lookup ----
const GGUFTensorInfo* GGUFFile::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return (it != tensor_index_.end()) ? &tensors_[it->second] : nullptr;
}

const GGUFMetadataKV* GGUFFile::find_metadata(const std::string& key) const {
    auto it = metadata_index_.find(key);
    return (it != metadata_index_.end()) ? &metadata_[it->second] : nullptr;
}

// ---- Type-safe metadata getters ----
std::string GGUFFile::get_string(const std::string& key) const {
    auto* kv = find_metadata(key);
    if (!kv) throw std::runtime_error("GGUF: metadata key not found: " + key);
    if (kv->type != GGUFValueType::STRING)
        throw std::runtime_error("GGUF: key '" + key + "' is not a string");
    return std::get<std::string>(kv->value);
}

uint32_t GGUFFile::get_u32(const std::string& key) const {
    auto* kv = find_metadata(key);
    if (!kv) throw std::runtime_error("GGUF: metadata key not found: " + key);
    if (kv->type != GGUFValueType::UINT32)
        throw std::runtime_error("GGUF: key '" + key + "' is not u32");
    return std::get<uint32_t>(kv->value);
}

int32_t GGUFFile::get_i32(const std::string& key) const {
    auto* kv = find_metadata(key);
    if (!kv) throw std::runtime_error("GGUF: metadata key not found: " + key);
    if (kv->type != GGUFValueType::INT32)
        throw std::runtime_error("GGUF: key '" + key + "' is not i32");
    return std::get<int32_t>(kv->value);
}

float GGUFFile::get_f32(const std::string& key) const {
    auto* kv = find_metadata(key);
    if (!kv) throw std::runtime_error("GGUF: metadata key not found: " + key);
    if (kv->type != GGUFValueType::FLOAT32)
        throw std::runtime_error("GGUF: key '" + key + "' is not f32");
    return std::get<float>(kv->value);
}

bool GGUFFile::get_bool(const std::string& key) const {
    auto* kv = find_metadata(key);
    if (!kv) throw std::runtime_error("GGUF: metadata key not found: " + key);
    if (kv->type != GGUFValueType::BOOL)
        throw std::runtime_error("GGUF: key '" + key + "' is not bool");
    return std::get<bool>(kv->value);
}

uint64_t GGUFFile::get_u64(const std::string& key) const {
    auto* kv = find_metadata(key);
    if (!kv) throw std::runtime_error("GGUF: metadata key not found: " + key);
    if (kv->type != GGUFValueType::UINT64)
        throw std::runtime_error("GGUF: key '" + key + "' is not u64");
    return std::get<uint64_t>(kv->value);
}
