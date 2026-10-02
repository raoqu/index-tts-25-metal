#include "itts25/speech_features.hpp"
#include <stdexcept>
namespace itts25 {
std::vector<float> SpeechFeatures::linear(const std::vector<float>& x,uint32_t t,const std::string& p,bool bias){
    const auto& s=weights_.info(p+".weight").shape;
    const uint32_t out=s[0],in=s[1];
    auto b=bias?weights_.get(p+".bias"):std::vector<float>(out,0);
    return metal_.linear_rows_f32_resident(p+".weight",weights_.get(p+".weight"),p+(bias?".bias":".zero"),b,x,t,out,in);
}
std::vector<float> SpeechFeatures::norm(const std::vector<float>& x,uint32_t t,const std::string& p){
    const auto& g=weights_.get(p+".weight");
    return metal_.layernorm_rows_f32_resident(p+".weight",g,p+".bias",weights_.get(p+".bias"),x,t,g.size(),1e-5f);
}
std::vector<float> SpeechFeatures::ff(const std::vector<float>& x,uint32_t t,const std::string& p){
    auto h=norm(x,t,p+"_layer_norm");
    h=metal_.silu_f32(linear(h,t,p+".intermediate_dense"));
    return metal_.add_scaled_f32(x,linear(h,t,p+".output_dense"),0.5f);
}
std::vector<float> SpeechFeatures::encode(const std::vector<float>& input,const std::vector<uint32_t>& mask,uint32_t t){
    if(!t || t>1600 || input.size()!=static_cast<size_t>(t)*160 || mask.size()!=t)throw std::invalid_argument("Invalid W2V feature input");
    auto x=linear(norm(input,t,"w2v_bert.feature_projection.layer_norm"),t,"w2v_bert.feature_projection.projection");
    x=metal_.mask_rows_f32(x,mask,t,1024);
    layer_trace.clear();
    for(uint32_t layer=0;layer<17;layer++) {
        const auto p="w2v_bert.encoder.layers."+std::to_string(layer);
        x=ff(x,t,p+".ffn1");
        auto h=norm(x,t,p+".self_attn_layer_norm");
        auto q=linear(h,t,p+".self_attn.linear_q"),k=linear(h,t,p+".self_attn.linear_k"),v=linear(h,t,p+".self_attn.linear_v");
        h=attention_.run(q,k,v,mask,weights_.get(p+".self_attn.distance_embedding.weight"),t);
        x=metal_.add_f32(x,linear(h,t,p+".self_attn.linear_out"));
        h=metal_.mask_rows_f32(norm(x,t,p+".conv_module.layer_norm"),mask,t,1024);
        h=metal_.glu_split_f32(linear(h,t,p+".conv_module.pointwise_conv1",false),t,1024);
        h=metal_.depthwise_conv1d_causal_f32_resident(p+".conv_module.depthwise_conv.weight",weights_.get(p+".conv_module.depthwise_conv.weight"),p+".conv_module.depthwise_conv.zero",std::vector<float>(1024,0),h,t,1024,31);
        h=metal_.silu_f32(norm(h,t,p+".conv_module.depthwise_layer_norm"));
        x=metal_.add_f32(x,linear(h,t,p+".conv_module.pointwise_conv2",false));
        x=norm(ff(x,t,p+".ffn2"),t,p+".final_layer_norm");
        if(trace_enabled)layer_trace.push_back(x);
    }
    return metal_.w2v_bert_normalize_f32(x,weights_.get("w2v_bert.stats.mean"),weights_.get("w2v_bert.stats.std"),t);
}
}
