#pragma once
#include "itts25/weights.hpp"
#include "mit2/metal_context.hpp"
namespace itts25 {
class QwenDecoder {
public:
    explicit QwenDecoder(Weights& weights);
    std::vector<float> prefill(const std::vector<uint32_t>& ids);
    std::vector<float> step(uint32_t id);
    std::vector<uint32_t> generate(const std::vector<uint32_t>& ids,uint32_t maximum=1024);
    uint64_t submissions() const {return metal_.command_buffers_submitted();}
private:
    std::unordered_map<uint32_t,std::vector<float>> zeros_;
    Weights& weights_;mit2::MetalContext metal_;uint32_t cached_=0,selected_=0;
    std::vector<float> run(const std::vector<uint32_t>& ids,uint32_t offset,bool only_token=false);
    mit2::PassSlot linear(mit2::PassSlot x,uint32_t tokens,const std::string& name);
    mit2::PassSlot norm(mit2::PassSlot x,uint32_t tokens,uint32_t width,const std::string& name);
};
}
