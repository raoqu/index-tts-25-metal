#include "itts25/acoustic.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace itts25 {
using namespace acoustic_detail;
AcousticModel::AcousticModel(Weights& w,mit2::MetalContext& m):weights_(w),metal_(m) {
    for(const auto& p:std::vector<std::pair<std::string,std::vector<int64_t>>>{
        {"s2mel.net.length_regulator.content_in_proj.weight",{512,1024}},
        {"s2mel.net.cfm.estimator.transformer.layers.12.attention.wqkv.weight",{1536,512}},
        {"bigvgan.conv_pre.weight_v",{1536,80,7}}})
        if(w.info(p.first).shape!=p.second) throw std::runtime_error("Unsupported 2.5 acoustic shape: "+p.first);
}
std::vector<float> AcousticModel::regulate(const std::vector<float>& x,uint32_t in,uint32_t out) {
    if(!in || !out || out>4096 || x.size()!=static_cast<size_t>(in)*1024) throw std::invalid_argument("Invalid length regulator dimensions");
    return run_length_regulator_full_metal(metal_,weights_.bundle(),x,in,out);
}
std::pair<std::vector<float>,std::vector<float>> AcousticModel::estimate_cfg(const std::vector<float>& x,const std::vector<float>& prompt,const std::vector<float>& cond,const std::vector<float>& style,uint32_t tokens,float time) {
    if(!tokens || tokens>4096 || x.size()!=static_cast<size_t>(tokens)*80 || prompt.size()!=x.size() || cond.size()!=static_cast<size_t>(tokens)*512 || style.size()!=192 || !std::isfinite(time)) throw std::invalid_argument("Invalid CFM estimator dimensions");
    auto t1=run_timestep_embedder_metal(metal_,weights_.bundle(),{time},"s2mel.net.cfm.estimator.t_embedder");
    auto t2=run_timestep_embedder_metal(metal_,weights_.bundle(),{time},"s2mel.net.cfm.estimator.t_embedder2");
    return run_dit_estimator_step_metal_cfg_transformer_batched_pass(metal_,weights_.bundle(),x,prompt,cond,style,std::vector<float>(x.size(),0),std::vector<float>(cond.size(),0),std::vector<float>(192,0),t1,t2,std::vector<uint32_t>(tokens,1),tokens);
}
std::vector<float> AcousticModel::flow(std::vector<float> noise,const std::vector<float>& prompt,const std::vector<float>& cond,const std::vector<float>& style,uint32_t tokens,uint32_t steps,float cfg) {
    if(!steps || steps>200 || prompt.size()%80 || prompt.size()>noise.size() || !std::isfinite(cfg) || cfg<0) throw std::invalid_argument("Invalid CFM schedule or prompt");
    const uint32_t prompt_tokens=prompt.size()/80;
    std::vector<float> px(noise.size(),0); std::copy(prompt.begin(),prompt.end(),px.begin());
    std::fill(noise.begin(),noise.begin()+prompt.size(),0);
    if (!tokens || tokens > 4096 || noise.size() != static_cast<size_t>(tokens) * 80 ||
        cond.size() != static_cast<size_t>(tokens) * 512 || style.size() != 192)
        throw std::invalid_argument("Invalid CFM dimensions");
    auto found = timestep_cache_.find(steps);
    if (found == timestep_cache_.end()) {
        if (timestep_cache_.size() >= 8) { timestep_cache_.clear(); modulation_cache_.clear(); conditioning_cache_.clear(); }
        std::pair<std::vector<float>, std::vector<float>> table;
        std::vector<float> timesteps(steps);
        for (uint32_t i = 0; i < steps; ++i) timesteps[i] = static_cast<float>(i) / steps;
        table.first = run_timestep_embedder_metal(metal_, weights_.bundle(), timesteps, "s2mel.net.cfm.estimator.t_embedder");
        table.second = run_timestep_embedder_metal(metal_, weights_.bundle(), timesteps, "s2mel.net.cfm.estimator.t_embedder2");
        found = timestep_cache_.emplace(steps, std::move(table)).first;
    }
    auto mod = modulation_cache_.find(steps);
    if (mod == modulation_cache_.end()) mod = modulation_cache_.emplace(steps, prepare_cfm_modulation(metal_, weights_.bundle(), found->second.first, steps)).first;
    auto conditioning=conditioning_cache_.find(steps);
    if(conditioning==conditioning_cache_.end())conditioning=conditioning_cache_.emplace(steps,prepare_cfm_conditioning(metal_,weights_.bundle(),found->second.first,found->second.second,steps)).first;
    return run_cfm_trajectory_metal_pass(metal_, weights_.bundle(), noise, px, cond, style,
        found->second.first, found->second.second, tokens, prompt_tokens, steps, cfg, mod->second, conditioning->second);
}
std::vector<float> AcousticModel::vocode(const std::vector<float>& mel,uint32_t tokens) {
    if(!tokens || tokens>4096 || mel.size()!=static_cast<size_t>(tokens)*80) throw std::invalid_argument("Invalid BigVGAN mel dimensions");
    return run_bigvgan_vocoder_metal_single_pass(metal_,weights_.bundle(),mel,tokens);
}
}
