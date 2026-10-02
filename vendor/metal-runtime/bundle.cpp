#include "mit2/bundle.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <cstring>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace mit2 {

namespace {
// Thread-local cache: reuse the loaded bundle data when the same bundle_dir
// is opened multiple times in sequence (e.g. inside run_tts_clone_w2v_encoder
// which calls ~200 sub-functions each constructing a Bundle).
// Keyed by directory: synthesis alternates between the model bundle and the
// voice bundle within one request — a single-slot cache would evict the 1.5MB
// model manifest (4521 tensors) on every alternation and re-parse it several
// times per request (~0.4s each).
thread_local std::unordered_map<std::string, std::shared_ptr<Bundle::Data>> tl_bundle_cache_map;
}  // namespace cache

namespace {

std::string read_text(const std::string& path) {
    std::ifstream fp(path);
    if (!fp) {
        throw std::runtime_error("failed to open " + path);
    }
    std::ostringstream ss;
    ss << fp.rdbuf();
    return ss.str();
}

std::string json_string(const std::string& obj, const std::string& key, const std::string& fallback = "") {
    std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    return std::regex_search(obj, m, re) ? m[1].str() : fallback;
}

uint64_t json_uint(const std::string& obj, const std::string& key) {
    std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch m;
    if (!std::regex_search(obj, m, re)) {
        throw std::runtime_error("missing numeric key " + key);
    }
    return static_cast<uint64_t>(std::stoull(m[1].str()));
}

std::vector<int64_t> json_shape(const std::string& obj) {
    std::regex re("\"shape\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch m;
    if (!std::regex_search(obj, m, re)) {
        throw std::runtime_error("missing tensor shape");
    }
    std::vector<int64_t> shape;
    std::stringstream ss(m[1].str());
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            shape.push_back(std::stoll(item));
        }
    }
    return shape;
}

std::vector<std::string> tensor_objects(const std::string& json) {
    const auto key = json.find("\"tensors\"");
    if (key == std::string::npos) {
        throw std::runtime_error("manifest missing tensors");
    }
    const auto begin = json.find('[', key);
    const auto end = json.rfind(']');
    if (begin == std::string::npos || end == std::string::npos || end <= begin) {
        throw std::runtime_error("invalid tensors array");
    }
    std::vector<std::string> out;
    int depth = 0;
    size_t obj_begin = std::string::npos;
    for (size_t i = begin; i <= end; ++i) {
        if (json[i] == '{') {
            if (depth == 0) {
                obj_begin = i;
            }
            ++depth;
        } else if (json[i] == '}') {
            --depth;
            if (depth == 0 && obj_begin != std::string::npos) {
                out.push_back(json.substr(obj_begin, i - obj_begin + 1));
                obj_begin = std::string::npos;
            }
        }
    }
    return out;
}

}  // namespace

MappedFile::MappedFile(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("failed to open mapped file: " + path);
    }
    struct stat st {};
    if (fstat(fd_, &st) != 0) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("failed to stat mapped file: " + path);
    }
    size_ = static_cast<size_t>(st.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("failed to mmap file: " + path);
    }
}

MappedFile::~MappedFile() {
    if (data_) {
        munmap(data_, size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

MappedFile::MappedFile(MappedFile&& other) noexcept {
    *this = std::move(other);
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        fd_ = other.fd_;
        data_ = other.data_;
        size_ = other.size_;
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

namespace {
// Single-file bundles append the manifest JSON after the tensor data and end
// with a fixed footer so the file stays a valid MIT2 weights stream:
//   [u64 manifest_offset][u64 manifest_size]["MIT2VOIC"]
constexpr char kSingleFileFooterMagic[8] = {'M', 'I', 'T', '2', 'V', 'O', 'I', 'C'};
constexpr size_t kSingleFileFooterSize = 24;

uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}
}  // namespace

bool bundle_path_is_single_file(const std::string& path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        return false;
    }
    std::ifstream fp(path, std::ios::binary);
    char magic[4] = {};
    fp.read(magic, 4);
    return fp && std::memcmp(magic, "MIT2", 4) == 0;
}

Bundle::Bundle(const std::string& bundle_dir) {
    // Return cached data if this bundle_dir was opened before on this thread.
    {
        const auto it = tl_bundle_cache_map.find(bundle_dir);
        if (it != tl_bundle_cache_map.end()) {
            d_ = it->second;
            return;
        }
    }

    d_ = std::make_shared<Data>();
    d_->root = bundle_dir;

    struct stat st {};
    const bool single_file = stat(bundle_dir.c_str(), &st) == 0 && S_ISREG(st.st_mode);

    std::string json;
    if (single_file) {
        d_->weights_file = bundle_dir;
        d_->weights = MappedFile(bundle_dir);
        if (d_->weights.size() < 12 + kSingleFileFooterSize) {
            throw std::runtime_error("single-file bundle is too small: " + bundle_dir);
        }
        const uint8_t* footer = d_->weights.data() + d_->weights.size() - kSingleFileFooterSize;
        if (std::memcmp(footer + 16, kSingleFileFooterMagic, 8) != 0) {
            throw std::runtime_error("single-file bundle footer magic missing: " + bundle_dir);
        }
        const uint64_t manifest_offset = read_u64_le(footer);
        const uint64_t manifest_size = read_u64_le(footer + 8);
        if (manifest_offset + manifest_size > d_->weights.size() - kSingleFileFooterSize) {
            throw std::runtime_error("single-file bundle manifest exceeds file: " + bundle_dir);
        }
        json.assign(reinterpret_cast<const char*>(d_->weights.data() + manifest_offset),
                    static_cast<size_t>(manifest_size));
    } else {
        json = read_text(d_->root + "/manifest.json");
    }
    const std::string format = json_string(json, "format");
    if (format != "MIT2") {
        throw std::runtime_error("unsupported bundle format: " + format);
    }
    d_->version = static_cast<uint32_t>(json_uint(json, "version"));
    d_->alignment = static_cast<uint32_t>(json_uint(json, "alignment"));
    d_->endianness = json_string(json, "endianness");
    if (d_->version != 1) {
        throw std::runtime_error("unsupported bundle version: " + std::to_string(d_->version));
    }
    if (d_->alignment == 0) {
        throw std::runtime_error("bundle alignment must be non-zero");
    }
    if (d_->endianness != "little") {
        throw std::runtime_error("unsupported bundle endianness: " + d_->endianness);
    }
    if (!single_file) {
        d_->weights_file = json_string(json, "weights_file", "weights.bin");
        d_->weights = MappedFile(d_->root + "/" + d_->weights_file);
    }
    if (d_->weights.size() < 12) {
        throw std::runtime_error("weights file is too small for MIT2 header");
    }
    const uint8_t* header = d_->weights.data();
    if (std::memcmp(header, "MIT2", 4) != 0) {
        throw std::runtime_error("weights file has invalid MIT2 magic");
    }
    const uint32_t file_version =
        static_cast<uint32_t>(header[4]) |
        (static_cast<uint32_t>(header[5]) << 8) |
        (static_cast<uint32_t>(header[6]) << 16) |
        (static_cast<uint32_t>(header[7]) << 24);
    const uint32_t file_alignment =
        static_cast<uint32_t>(header[8]) |
        (static_cast<uint32_t>(header[9]) << 8) |
        (static_cast<uint32_t>(header[10]) << 16) |
        (static_cast<uint32_t>(header[11]) << 24);
    if (file_version != d_->version) {
        throw std::runtime_error("weights header version mismatch");
    }
    if (file_alignment != d_->alignment) {
        throw std::runtime_error("weights header alignment mismatch");
    }

    for (const auto& obj : tensor_objects(json)) {
        TensorInfo info;
        info.name = json_string(obj, "name");
        info.shape = json_shape(obj);
        info.dtype = json_string(obj, "dtype");
        info.offset = json_uint(obj, "offset");
        info.nbytes = json_uint(obj, "nbytes");
        info.sha256 = json_string(obj, "sha256");
        info.layout = json_string(obj, "layout", "row_major");
        info.component = json_string(obj, "component", "unknown");
        if (info.name.empty()) {
            throw std::runtime_error("tensor record has empty name");
        }
        if (d_->by_name.find(info.name) != d_->by_name.end()) {
            throw std::runtime_error("duplicate tensor in manifest: " + info.name);
        }
        if (info.offset % d_->alignment != 0) {
            throw std::runtime_error("tensor offset is not bundle-aligned: " + info.name);
        }
        if (info.nbytes == 0) {
            throw std::runtime_error("tensor has zero bytes: " + info.name);
        }
        if (info.offset + info.nbytes > d_->weights.size()) {
            throw std::runtime_error("tensor extends beyond weights file: " + info.name);
        }
        d_->by_name[info.name] = d_->tensors.size();
        d_->tensors.push_back(std::move(info));
    }

    // Cache for subsequent calls with the same path on this thread.
    // Bounded: each entry holds an mmap + parsed manifest; 16 distinct bundles
    // (model + many voices) is far beyond normal use.
    if (tl_bundle_cache_map.size() >= 16) {
        tl_bundle_cache_map.clear();
    }
    tl_bundle_cache_map.emplace(bundle_dir, d_);
}

const TensorInfo* Bundle::find(const std::string& name) const {
    const auto it = d_->by_name.find(name);
    if (it == d_->by_name.end()) {
        return nullptr;
    }
    return &d_->tensors[it->second];
}

const uint8_t* Bundle::tensor_data(const TensorInfo& info) const {
    return d_->weights.data() + info.offset;
}

size_t Bundle::total_tensor_bytes() const {
    size_t total = 0;
    for (const auto& t : d_->tensors) {
        total += static_cast<size_t>(t.nbytes);
    }
    return total;
}

}  // namespace mit2
