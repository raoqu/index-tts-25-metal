#include "itts25/emotion.hpp"
#include <stdexcept>
namespace itts25 {
std::vector<float> EmotionEncoder::encode(const std::vector<float>& features,uint32_t tokens) {
    if(tokens<3 || tokens>1600 || features.size()!=static_cast<size_t>(tokens)*1024) throw std::invalid_argument("Invalid emotion feature dimensions");
    return emotion_detail::run_gpt_emovec_metal_linear(metal_,weights_.bundle(),features,tokens);
}
}
