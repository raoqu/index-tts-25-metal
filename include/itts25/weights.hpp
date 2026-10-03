#pragma once
#include "mit2/bundle.hpp"
#include <unordered_map>

namespace itts25 {
class Weights {
public:
    explicit Weights(const std::string& path, bool require_model = true);
    const std::vector<float>& get(const std::string& name);
    const mit2::TensorInfo& info(const std::string& name) const;
    void release(const std::string& name) { cache_.erase(name); }
    std::vector<uint32_t> ids(const std::string& name) const;
    const mit2::Bundle& bundle() const { return bundle_; }
private:
    mit2::Bundle bundle_;
    std::unordered_map<std::string, std::vector<float>> cache_;
};
}
