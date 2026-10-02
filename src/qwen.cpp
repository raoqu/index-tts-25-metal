#include "itts25/qwen.hpp"
#include <algorithm>
#include <stdexcept>
namespace itts25 {
QwenDecoder::QwenDecoder(Weights& w):weights_(w) {
    auto check=[&](const std::string& name,const std::vector<int64_t>& shape){if(w.info(name).shape!=shape)throw std::invalid_argument("Unsupported Qwen shape: "+name);};
    check("qwen.model.embed_tokens.weight",{151936,1024});
    for(uint32_t i=0;i<28;i++){
        auto p="qwen.model.layers."+std::to_string(i);
        check(p+".self_attn.q_proj.weight",{2048,1024});check(p+".self_attn.k_proj.weight",{1024,1024});check(p+".self_attn.v_proj.weight",{1024,1024});check(p+".self_attn.o_proj.weight",{1024,2048});
        check(p+".self_attn.q_norm.weight",{128});check(p+".self_attn.k_norm.weight",{128});
        check(p+".mlp.gate_proj.weight",{3072,1024});check(p+".mlp.up_proj.weight",{3072,1024});check(p+".mlp.down_proj.weight",{1024,3072});
    }
}
mit2::PassSlot QwenDecoder::linear(mit2::PassSlot x,uint32_t t,const std::string& p){
    const auto& s=weights_.info(p+".weight").shape;auto& bias=zeros_[s[0]];if(bias.empty())bias.resize(s[0],0);
    return metal_.linear_rows_f32_pass(p+".weight",weights_.get(p+".weight"),"qwen.zero."+std::to_string(s[0]),bias,x,t,s[0],s[1]);
}
mit2::PassSlot QwenDecoder::norm(mit2::PassSlot x,uint32_t t,uint32_t width,const std::string& p){return metal_.rmsnorm_rows_eps_f32_pass(p+".weight",weights_.get(p+".weight"),x,t,width,1e-6f);}
std::vector<float> QwenDecoder::prefill(const std::vector<uint32_t>& ids){
    if(ids.empty() || ids.size()>3072)throw std::invalid_argument("Qwen prompt capacity is 3072 tokens");
    metal_.gptKvCacheCreate(28,4096,1024);auto result=run(ids,0);cached_=ids.size();return result;
}
std::vector<float> QwenDecoder::step(uint32_t id){if(!cached_ || cached_>=4096)throw std::invalid_argument("Qwen cache capacity exceeded");auto result=run({id},cached_);cached_++;return result;}
std::vector<float> QwenDecoder::run(const std::vector<uint32_t>& ids,uint32_t offset){
    if(std::any_of(ids.begin(),ids.end(),[](auto id){return id>=151936;}))throw std::invalid_argument("Invalid Qwen token");
    const uint32_t t=ids.size();auto embeddings=metal_.embedding_f32_resident("qwen.model.embed_tokens.weight",weights_.get("qwen.model.embed_tokens.weight"),ids,1024);metal_.beginPass((static_cast<size_t>(t)*48*1024+200000)*4);
    auto hidden=metal_.passUploadAlloc(embeddings);
    auto a=metal_.passAlloc(t*1024),b=metal_.passAlloc(t*1024);metal_.passSetScratchBase();
    for(uint32_t layer=0;layer<28;layer++){
        auto p="qwen.model.layers."+std::to_string(layer);auto normalized=norm(hidden,t,1024,p+".input_layernorm");
        auto q=norm(linear(normalized,t,p+".self_attn.q_proj"),t*16,128,p+".self_attn.q_norm");
        auto k=norm(linear(normalized,t,p+".self_attn.k_proj"),t*8,128,p+".self_attn.k_norm");
        auto v=linear(normalized,t,p+".self_attn.v_proj");
        q=metal_.qwen_rope_f32_pass(q,t,16,offset);k=metal_.qwen_rope_f32_pass(k,t,8,offset);
        auto attention=metal_.qwen_attention_f32_pass(layer,q,k,v,t,offset);
        auto residual=metal_.add_f32_pass(hidden,linear(attention,t,p+".self_attn.o_proj"));
        auto ln=norm(residual,t,1024,p+".post_attention_layernorm");
        auto ff=metal_.silu_mul_f32_pass(linear(ln,t,p+".mlp.gate_proj"),linear(ln,t,p+".mlp.up_proj"));
        auto out=layer%2?a:b;metal_.add_f32_pass_into(residual,linear(ff,t,p+".mlp.down_proj"),out);hidden=out;metal_.passResetScratch();
    }
    auto last=norm(hidden.slice((t-1)*1024,1024),1,1024,"qwen.model.norm");
    auto logits=linear(last,1,"qwen.model.embed_tokens");metal_.endPass();return metal_.passRead(logits);
}
std::vector<uint32_t> QwenDecoder::generate(const std::vector<uint32_t>& ids,uint32_t maximum){
    if(!maximum || maximum>1024 || ids.size()+maximum>4096)throw std::invalid_argument("Qwen generation capacity exceeded");
    auto logits=prefill(ids);std::vector<uint32_t> result;
    for(uint32_t i=0;i<maximum;i++){uint32_t id=std::max_element(logits.begin(),logits.end())-logits.begin();result.push_back(id);if(id==151643)return result;if(i+1<maximum)logits=step(id);}
    throw std::runtime_error("Qwen did not reach EOS within 1024 generated tokens");
}
}
