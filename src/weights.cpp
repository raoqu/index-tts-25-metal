#include "itts25/weights.hpp"
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>

namespace itts25 {
Weights::Weights(const std::string& path, bool require_model) : bundle_(path) {
    if (require_model) {
        std::ifstream stream(path + "/manifest.json");
        const std::string text((std::istreambuf_iterator<char>(stream)), {});
        if (!std::regex_search(text, std::regex("\"target\"\\s*:\\s*\"index-tts2.5\"")))
            throw std::runtime_error("Expected an IndexTTS 2.5 model bundle");
    }
    for (const auto& t : bundle_.tensors()) {
        const size_t element_bytes = t.dtype == "f32" || t.dtype == "u32" ? 4 : 0;
        if (!element_bytes) throw std::runtime_error("Initial native backend requires f32/u32: " + t.name);
        size_t count = 1;
        for (auto n : t.shape) {
            if (n <= 0 || static_cast<uint64_t>(n) > std::numeric_limits<size_t>::max() / count)
                throw std::runtime_error("Invalid shape: " + t.name);
            count *= static_cast<size_t>(n);
        }
        if (count > std::numeric_limits<size_t>::max()/element_bytes || count*element_bytes != t.nbytes)
            throw std::runtime_error("Tensor size does not match shape: " + t.name);
    }
}
const mit2::TensorInfo& Weights::info(const std::string& name) const {
    const auto* p = bundle_.find(name);
    if (!p) throw std::runtime_error("Missing tensor: " + name);
    return *p;
}
const std::vector<float>& Weights::get(const std::string& name) {
    auto found = cache_.find(name);
    if (found != cache_.end()) return found->second;
    const auto& t = info(name);
    if (t.dtype != "f32") throw std::runtime_error("Expected float32: " + name);
    std::vector<float> v(t.nbytes/sizeof(float));
    std::memcpy(v.data(), bundle_.tensor_data(t), t.nbytes);
    return cache_.emplace(name, std::move(v)).first->second;
}
std::vector<uint32_t> Weights::ids(const std::string& name) const {
    const auto& t = info(name);
    if (t.dtype != "u32") throw std::runtime_error("Expected uint32: " + name);
    std::vector<uint32_t> v(t.nbytes/sizeof(uint32_t));
    std::memcpy(v.data(), bundle_.tensor_data(t), t.nbytes);
    return v;
}
}
