#pragma once
#include "itts25/weights.hpp"
#include <memory>
namespace itts25 {
class CampConvolution {
public:
    CampConvolution();~CampConvolution();
    std::vector<float> run(const std::vector<float>& input,const std::vector<float>& weight,const std::vector<uint32_t>& dimensions);
    uint64_t submissions() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
class CampPlus {
public:
    explicit CampPlus(Weights& weights):weights_(weights) {}
    std::vector<float> encode(const std::vector<float>& fbank,uint32_t frames);
    uint64_t gpu_submissions() const { return convolution_.submissions(); }
private:
    Weights& weights_;CampConvolution convolution_;
};
}
