#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mit2 {

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    std::string sha256;
    std::string layout;
    std::string component;
};

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    const uint8_t* data() const { return static_cast<const uint8_t*>(data_); }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }

private:
    int fd_ = -1;
    void* data_ = nullptr;
    size_t size_ = 0;
};

// True when path is a regular file starting with the MIT2 magic
// (a single-file bundle as written by the native clone pipeline).
bool bundle_path_is_single_file(const std::string& path);

class Bundle {
public:
    // Accepts either a bundle directory (manifest.json + weights.bin) or a
    // single-file bundle (weights stream with embedded manifest + footer).
    explicit Bundle(const std::string& bundle_dir);

    const std::vector<TensorInfo>& tensors() const { return d_->tensors; }
    const TensorInfo* find(const std::string& name) const;
    const uint8_t* tensor_data(const TensorInfo& info) const;
    size_t tensor_count() const { return d_->tensors.size(); }
    size_t total_tensor_bytes() const;
    const std::string& root() const { return d_->root; }
    const std::string& weights_file() const { return d_->weights_file; }
    size_t weights_size() const { return d_->weights.size(); }
    uint32_t version() const { return d_->version; }
    uint32_t alignment() const { return d_->alignment; }
    const std::string& endianness() const { return d_->endianness; }

    // Internal shared data; public so bundle.cpp can access without friendship
    struct Data {
        std::string root;
        std::string weights_file;
        uint32_t version = 0;
        uint32_t alignment = 0;
        std::string endianness;
        MappedFile weights;
        std::vector<TensorInfo> tensors;
        std::unordered_map<std::string, size_t> by_name;
    };

private:
    std::shared_ptr<Data> d_;
};

}  // namespace mit2
