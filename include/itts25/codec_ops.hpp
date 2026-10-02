#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace itts25 {
// Additional 2.5 kernels: exact-form GELU, channel scales, stride-2 convolution,
// and normalized factorized vector quantization. All arithmetic runs on Metal.
class CodecOps {
public:
    CodecOps();
    ~CodecOps();
    std::vector<float> gelu(const std::vector<float>& x);
    std::vector<float> scale(const std::vector<float>& x, const std::vector<float>& gamma);
    std::vector<float> downsample(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens);
    std::vector<uint32_t> quantize(const std::vector<float>& latents, const std::vector<float>& codebook);
    uint64_t submissions() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
