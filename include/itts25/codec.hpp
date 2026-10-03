#pragma once
#include "itts25/weights.hpp"
#include "itts25/codec_ops.hpp"
#include "mit2/metal_context.hpp"

namespace itts25 {
struct EncodedSemantic {
    std::vector<uint32_t> codes;
    std::vector<float> projected; // time-major [codes.size(), 1024]
};
class EnhancedCodec {
public:
    EnhancedCodec(Weights& weights, mit2::MetalContext& metal);
    std::vector<float> decode(const std::vector<uint32_t>& codes); // [2*T,1024]
    EncodedSemantic encode(const std::vector<float>& features, uint32_t tokens);
    uint64_t extra_gpu_submissions() const { return ops_.submissions(); }
private:
    Weights& weights_;
    mit2::MetalContext& metal_;
    CodecOps ops_;
    std::vector<float> vocos(std::vector<float> x, uint32_t tokens, const std::string& prefix);
    std::vector<float> linear(const std::vector<float>& x, uint32_t tokens, const std::string& prefix);
    std::vector<float> norm(const std::vector<float>& x, uint32_t tokens, const std::string& prefix);
    std::vector<float> conv(const std::vector<float>& x, uint32_t tokens, const std::string& prefix, bool depthwise = false);
    std::vector<float> lookup(const std::vector<uint32_t>& codes);
};
}
