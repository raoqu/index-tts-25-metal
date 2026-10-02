#pragma once
#include "itts25/weights.hpp"
#include "mit2/metal_context.hpp"
#include <memory>
namespace itts25 {
class RelativeAttention {
public:
    RelativeAttention(); ~RelativeAttention();
    std::vector<float> run(const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,const std::vector<uint32_t>& mask,const std::vector<float>& distances,uint32_t tokens);
    uint64_t submissions() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
class SpeechFeatures {
public:
    SpeechFeatures(Weights& weights,mit2::MetalContext& metal):weights_(weights),metal_(metal) {}
    // W2V-BERT hidden_states[17]: output of layers 0..16, normalized with 2.5 stats.
    std::vector<float> encode(const std::vector<float>& input,const std::vector<uint32_t>& mask,uint32_t tokens);
    bool trace_enabled=false;
    std::vector<std::vector<float>> layer_trace;
    uint64_t extra_gpu_submissions() const { return attention_.submissions(); }
private:
    Weights& weights_;mit2::MetalContext& metal_;RelativeAttention attention_;
    std::vector<float> linear(const std::vector<float>& x,uint32_t tokens,const std::string& prefix,bool bias=true);
    std::vector<float> norm(const std::vector<float>& x,uint32_t tokens,const std::string& prefix);
    std::vector<float> ff(const std::vector<float>& x,uint32_t tokens,const std::string& prefix);
};
}
