#include "itts25/gpt.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace itts25 {
constexpr uint32_t width=1280, heads=20, head_dim=64, layers=24, cache_capacity=4096;
GptDecoder::GptDecoder(Weights& w,mit2::MetalContext& m):weights_(w),metal_(m) {
    const auto check=[&](const std::string& name,const std::vector<int64_t>& shape) {
        if(w.info(name).shape!=shape) throw std::runtime_error("Unsupported 2.5 GPT shape: "+name);
    };
    check("gpt.spk_emb_proj.weight",{1280,192});
    check("gpt.text_embedding.weight",{60510,1280});
    check("gpt.lang_embedding.weight",{107,1280});
    check("gpt.mel_embedding.weight",{8194,1280});
    for(uint32_t i=0;i<layers;i++) {
        const auto p="gpt.gpt.h."+std::to_string(i);
        check(p+".attn.c_attn.weight",{1280,3840});
        check(p+".attn.c_proj.weight",{1280,1280});
        check(p+".mlp.c_fc.weight",{1280,5120});
        check(p+".mlp.c_proj.weight",{5120,1280});
    }
}
const std::vector<float>& GptDecoder::conv_weight(const std::string& name) {
    const auto found=transposed_.find(name);
    if(found!=transposed_.end()) return found->second;
    const auto& shape=weights_.info(name).shape;
    const auto& raw=weights_.get(name);
    std::vector<float> out(raw.size());
    for(size_t in=0;in<static_cast<size_t>(shape[0]);in++)
        for(size_t col=0;col<static_cast<size_t>(shape[1]);col++) out[col*shape[0]+in]=raw[in*shape[1]+col];
    return transposed_.emplace(name,std::move(out)).first->second;
}
mit2::PassSlot GptDecoder::linear(mit2::PassSlot x,uint32_t tokens,const std::string& p,bool transpose) {
    const auto& s=weights_.info(p+".weight").shape;
    const auto& w=transpose?conv_weight(p+".weight"):weights_.get(p+".weight");
    return metal_.linear_rows_f32_pass(p+".weight.rows",w,p+".bias",weights_.get(p+".bias"),x,tokens,transpose?s[1]:s[0],transpose?s[0]:s[1]);
}
mit2::PassSlot GptDecoder::norm(mit2::PassSlot x,uint32_t t,const std::string& p) {
    return metal_.layernorm_rows_f32_pass(p+".weight",weights_.get(p+".weight"),p+".bias",weights_.get(p+".bias"),x,t,width,1e-5f);
}
std::vector<float> GptDecoder::prepare(const std::vector<float>& speaker,const std::vector<float>& emotion,const std::vector<uint32_t>& text,uint32_t language) {
    if(speaker.size()!=192 || emotion.size()!=width || language>=107) throw std::invalid_argument("Invalid GPT conditioning dimensions");
    std::vector<uint32_t> ids{0};
    for(auto id:text) {
        if(id>=60510) throw std::invalid_argument("Text token is outside 2.5 vocabulary");
        if(id!=0 && id!=1) ids.push_back(id);
    }
    ids.push_back(1);
    if(ids.size()>602) throw std::invalid_argument("Text segment exceeds GPT position capacity");
    // Compact the masked left-padding rows: GPT has no global position embedding,
    // so removing masked keys is equivalent and permits the native causal kernel.
    auto speaker_proj=metal_.linear_rows_f32_resident("gpt.spk_emb_proj.weight",weights_.get("gpt.spk_emb_proj.weight"),"gpt.spk_emb_proj.bias",weights_.get("gpt.spk_emb_proj.bias"),speaker,1,width,192);
    auto speaker_cond=metal_.add_f32(speaker_proj,emotion);
    std::vector<float> prefix(3*width,0);
    std::copy(speaker_cond.begin(),speaker_cond.end(),prefix.begin());
    auto embeddings=metal_.embedding_f32_resident("gpt.text_embedding.weight",weights_.get("gpt.text_embedding.weight"),ids,width);
    std::vector<uint32_t> positions(ids.size()); std::iota(positions.begin(),positions.end(),0);
    auto pos=metal_.embedding_f32_resident("gpt.text_pos_embedding.emb.weight",weights_.get("gpt.text_pos_embedding.emb.weight"),positions,width);
    embeddings=metal_.add_f32(embeddings,pos);
    auto language_embedding=metal_.embedding_f32_resident("gpt.lang_embedding.weight",weights_.get("gpt.lang_embedding.weight"),{language},width);
    std::vector<float> languages(embeddings.size());
    for(size_t row=0;row<ids.size();row++) std::copy(language_embedding.begin(),language_embedding.end(),languages.begin()+row*width);
    embeddings=metal_.add_f32(embeddings,languages);
    prefix.insert(prefix.end(),embeddings.begin(),embeddings.end());
    auto start=metal_.embedding_f32_resident("gpt.mel_embedding.weight",weights_.get("gpt.mel_embedding.weight"),{8192},width);
    auto start_pos=metal_.embedding_f32_resident("gpt.mel_pos_embedding.emb.weight",weights_.get("gpt.mel_pos_embedding.emb.weight"),{0},width);
    start=metal_.add_f32(start,start_pos);
    prefix.insert(prefix.end(),start.begin(),start.end());
    return prefix;
}
std::vector<float> GptDecoder::prefill(const std::vector<float>& prefix) {
    if(prefix.empty() || prefix.size()%width || prefix.size()/width>1024) throw std::invalid_argument("Invalid GPT prefill dimensions");
    metal_.gptKvCacheCreate(layers,cache_capacity,width);
    auto result=run(prefix,true,0,0);
    cached_tokens_=prefix.size()/width;
    return result;
}
std::vector<float> GptDecoder::step(uint32_t code,uint32_t position) {
    if(!cached_tokens_ || cached_tokens_>=cache_capacity || code>=8194 || position>=weights_.info("gpt.mel_pos_embedding.emb.weight").shape[0])
        throw std::invalid_argument("GPT step requires valid cache, code and position");
    auto result=run({},false,code,position);
    cached_tokens_++;
    return result;
}
GptState GptDecoder::checkpoint() const {return {cached_tokens_,metal_.gptKvSnapshot(cached_tokens_)};}
void GptDecoder::restore(const GptState& state) {metal_.gptKvRestore(state.kv,state.tokens);cached_tokens_=state.tokens;}
std::vector<float> GptDecoder::run(const std::vector<float>& input,bool initial,uint32_t code,uint32_t position) {
    // Resolve/transpose weights before beginning the command buffer. This avoids
    // CPU conversion and allocation stalls while a GPU pass is being encoded.
    for(uint32_t i=0;i<layers;i++) for(const auto* name:{"attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"})
        conv_weight("gpt.gpt.h."+std::to_string(i)+"."+name+".weight");
    const uint32_t t=initial?static_cast<uint32_t>(input.size()/width):1;
    metal_.beginPass((static_cast<size_t>(t)*24*width+16384)*4);
    mit2::PassSlot hidden;
    if(initial) hidden=metal_.passUploadAlloc(input);
    else {
        auto id=metal_.passUploadAllocU32(std::vector<uint32_t>{code});
        hidden=metal_.gpt_build_current_pass(id,"gpt.mel_embedding.weight",weights_.get("gpt.mel_embedding.weight"),"gpt.mel_pos_embedding.emb.weight",weights_.get("gpt.mel_pos_embedding.emb.weight"),width,position);
    }
    auto a=metal_.passAlloc(t*width),b=metal_.passAlloc(t*width);
    metal_.passSetScratchBase();
    for(uint32_t layer=0;layer<layers;layer++) {
        const auto p="gpt.gpt.h."+std::to_string(layer);
        auto qkv=linear(norm(hidden,t,p+".ln_1"),t,p+".attn.c_attn",true);
        mit2::PassSlot attention;
        if(initial) {
            metal_.gptKvStoreFromQkv_pass(layer,qkv,t);
            attention=metal_.gpt_causal_attention_f32_pass(qkv,t,heads,head_dim);
        } else attention=metal_.gpt_cached_attention_resident_pass(layer,qkv,cached_tokens_,heads,head_dim);
        auto residual=metal_.add_f32_pass(hidden,linear(attention,t,p+".attn.c_proj",true));
        auto ff=linear(norm(residual,t,p+".ln_2"),t,p+".mlp.c_fc",true);
        ff=metal_.gelu_f32_pass(ff,t*5120);
        auto projected=linear(ff,t,p+".mlp.c_proj",true);
        auto out=layer%2?a:b;
        metal_.add_f32_pass_into(residual,projected,out);
        hidden=out;
        metal_.passResetScratch();
    }
    auto last=hidden.slice((t-1)*width,width);
    last=norm(norm(last,1,"gpt.gpt.ln_f"),1,"gpt.final_norm");
    auto logits=linear(last,1,"gpt.mel_head",false);
    metal_.endPass();
    return metal_.passRead(logits);
}
}
