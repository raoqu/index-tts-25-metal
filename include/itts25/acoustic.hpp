#pragma once
#include "itts25/weights.hpp"
#include "mit2/metal_context.hpp"
#include <utility>
#include <unordered_map>
namespace itts25 {
// All interfaces use row-major token/channel layout, batch one.
class AcousticModel {
public:
    AcousticModel(Weights& weights,mit2::MetalContext& metal);
    std::vector<float> regulate(const std::vector<float>& features,uint32_t input_tokens,uint32_t output_tokens);
    std::pair<std::vector<float>,std::vector<float>> estimate_cfg(const std::vector<float>& x,const std::vector<float>& prompt,const std::vector<float>& condition,const std::vector<float>& style,uint32_t tokens,float timestep);
    std::vector<float> flow(std::vector<float> noise,const std::vector<float>& prompt,const std::vector<float>& condition,const std::vector<float>& style,uint32_t tokens,uint32_t steps=25,float cfg_rate=0.7f);
    std::vector<float> vocode(const std::vector<float>& mel,uint32_t tokens);
private:
    Weights& weights_;
    mit2::MetalContext& metal_;
    std::unordered_map<uint32_t, std::pair<std::vector<float>, std::vector<float>>> timestep_cache_;
    std::unordered_map<uint32_t, std::vector<float>> modulation_cache_;
};
namespace acoustic_detail {
std::vector<float> run_cfm_trajectory_metal_pass(mit2::MetalContext&, const mit2::Bundle&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, uint32_t, uint32_t, uint32_t, float, const std::vector<float>&);
std::vector<float> prepare_cfm_modulation(mit2::MetalContext&, const mit2::Bundle&, const std::vector<float>&, uint32_t);
std::vector<float> run_length_regulator_full_metal(mit2::MetalContext&,const mit2::Bundle&,const std::vector<float>&,uint32_t,uint32_t);
std::vector<float> run_timestep_embedder_metal(mit2::MetalContext&,const mit2::Bundle&,const std::vector<float>&,const std::string&);
std::vector<float> run_bigvgan_vocoder_metal_single_pass(mit2::MetalContext&,const mit2::Bundle&,const std::vector<float>&,uint32_t);
std::pair<std::vector<float>,std::vector<float>> run_dit_estimator_step_metal_cfg_transformer_batched_pass(mit2::MetalContext&,const mit2::Bundle&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&,const std::vector<uint32_t>&,uint32_t);
}
}
