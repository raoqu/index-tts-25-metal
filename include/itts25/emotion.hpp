#pragma once
#include "itts25/weights.hpp"
#include "mit2/metal_context.hpp"
namespace itts25 {
class EmotionEncoder {
public:
    EmotionEncoder(Weights& weights,mit2::MetalContext& metal):weights_(weights),metal_(metal) {}
    std::vector<float> encode(const std::vector<float>& semantic_features,uint32_t tokens);
private:
    Weights& weights_;
    mit2::MetalContext& metal_;
};
namespace emotion_detail {
std::vector<float> run_gpt_emovec_metal_linear(mit2::MetalContext&,const mit2::Bundle&,const std::vector<float>&,uint32_t);
}
}
