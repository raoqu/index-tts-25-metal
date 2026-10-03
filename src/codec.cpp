#include "itts25/codec.hpp"
#include <algorithm>
#include <stdexcept>

namespace itts25 {
EnhancedCodec::EnhancedCodec(Weights& w, mit2::MetalContext& m) : weights_(w), metal_(m) {
    const auto shape = [&](const std::string& name, std::vector<int64_t> expected) {
        if(w.info(name).shape != expected) throw std::runtime_error("Unsupported 2.5 codec shape: "+name);
    };
    shape("codec.down.weight",{1024,1024,3});
    shape("codec.up.weight",{1024,1024,3});
    shape("codec.quantizer.quantizers.0.codebook.weight",{8192,8});
    for(const auto* block : {"encoder", "decoder"}) {
        std::string prefix=std::string("codec.")+block+".0";
        shape(prefix+".embed.weight",{384,1024,7});
        for(unsigned i=0;i<12;i++) {
            auto layer=prefix+".convnext."+std::to_string(i);
            shape(layer+".dwconv.weight",{384,1,7});
            shape(layer+".pwconv1.weight",{2048,384});
            shape(layer+".pwconv2.weight",{384,2048});
        }
    }
}
std::vector<float> EnhancedCodec::linear(const std::vector<float>& x,uint32_t t,const std::string& p) {
    const auto& s=weights_.info(p+".weight").shape;
    return metal_.linear_rows_f32_resident(p+".weight",weights_.get(p+".weight"),p+".bias",weights_.get(p+".bias"),x,t,s[0],s[1]);
}
std::vector<float> EnhancedCodec::norm(const std::vector<float>& x,uint32_t t,const std::string& p) {
    const auto& g=weights_.get(p+".weight");
    return metal_.layernorm_rows_f32_resident(p+".weight",g,p+".bias",weights_.get(p+".bias"),x,t,g.size(),1e-6f);
}
std::vector<float> EnhancedCodec::conv(const std::vector<float>& x,uint32_t t,const std::string& p,bool depthwise) {
    const auto& s=weights_.info(p+".weight").shape;
    if(depthwise) return metal_.depthwise_conv1d_same_f32_resident(p+".weight",weights_.get(p+".weight"),p+".bias",weights_.get(p+".bias"),x,t,s[0],s[2]);
    return metal_.conv1d_same_f32_resident(p+".weight",weights_.get(p+".weight"),p+".bias",weights_.get(p+".bias"),x,t,s[1],s[0],s[2]);
}
mit2::PassSlot EnhancedCodec::vocos_pass(mit2::PassSlot x,uint32_t t,const std::string& p,mit2::PassSlot a,mit2::PassSlot b) {
    const auto norm_pass=[&](mit2::PassSlot v,const std::string& k){const auto& g=weights_.get(k+".weight");return metal_.layernorm_rows_f32_pass(k+".weight",g,k+".bias",weights_.get(k+".bias"),v,t,g.size(),1e-6f);};
    const auto linear_pass=[&](mit2::PassSlot v,const std::string& k){const auto& sh=weights_.info(k+".weight").shape;return metal_.linear_rows_f32_pass(k+".weight",weights_.get(k+".weight"),k+".bias",weights_.get(k+".bias"),v,t,sh[0],sh[1]);};
    const auto conv_pass=[&](mit2::PassSlot v,const std::string& k,bool dw){const auto& sh=weights_.info(k+".weight").shape;const auto& w=weights_.get(k+".weight");const auto& bias=weights_.get(k+".bias");return dw?metal_.depthwise_conv1d_same_pass(k+".weight",w,k+".bias",bias,v,t,sh[0],sh[2]):metal_.conv1d_same_f32_pass(k+".weight",w,k+".bias",bias,v,t,sh[1],sh[0],sh[2]);};
    auto initial=norm_pass(conv_pass(x,p+".0.embed",false),p+".0.norm");
    metal_.copy_f32_pass_into(initial,a,t*384);x=a;metal_.passResetScratch();
    for(uint32_t i=0;i<12;i++) {
        auto layer=p+".0.convnext."+std::to_string(i);
        auto h=norm_pass(conv_pass(x,layer+".dwconv",true),layer+".norm");
        h=linear_pass(metal_.codec_gelu_pass(linear_pass(h,layer+".pwconv1")),layer+".pwconv2");
        auto out=i%2?a:b;metal_.add_f32_pass_into(x,metal_.codec_scale_pass(h,layer+".gamma",weights_.get(layer+".gamma")),out);
        x=out;metal_.passResetScratch();
    }
    return linear_pass(norm_pass(x,p+".0.final_layer_norm"),p+".1");
}
std::vector<float> EnhancedCodec::vocos(std::vector<float> x,uint32_t t,const std::string& p) {
    metal_.beginPass(static_cast<size_t>(t)*12000*4+65536);
    auto input=metal_.passUploadAlloc(x),a=metal_.passAlloc(t*384),b=metal_.passAlloc(t*384);metal_.passSetScratchBase();
    auto out=vocos_pass(input,t,p,a,b);metal_.endPass();return metal_.passRead(out);
}
std::vector<float> EnhancedCodec::lookup(const std::vector<uint32_t>& codes) {
    if(codes.empty() || codes.size()>5000 || *std::max_element(codes.begin(),codes.end())>=8192)
        throw std::invalid_argument("Codec expects 1..5000 code IDs in [0,8192)");
    const std::string p="codec.quantizer.quantizers.0";
    auto embeddings=metal_.embedding_f32_resident(p+".codebook.weight",weights_.get(p+".codebook.weight"),codes,8);
    return linear(embeddings,codes.size(),p+".out_project");
}
std::vector<float> EnhancedCodec::decode(const std::vector<uint32_t>& codes) {
    auto x=lookup(codes);const uint32_t t=codes.size();
    metal_.beginPass(static_cast<size_t>(t)*12000*4+65536);
    auto input=metal_.passUploadAlloc(x),a=metal_.passAlloc(t*384),b=metal_.passAlloc(t*384);metal_.passSetScratchBase();
    auto hidden=vocos_pass(input,t,"codec.decoder",a,b);
    hidden=metal_.nearest_interpolate_pass(hidden,t,t*2,1024);
    auto out=metal_.conv1d_same_f32_pass("codec.up.weight",weights_.get("codec.up.weight"),"codec.up.bias",weights_.get("codec.up.bias"),hidden,t*2,1024,1024,3);
    metal_.endPass();return metal_.passRead(out);
}
EncodedSemantic EnhancedCodec::encode(const std::vector<float>& features,uint32_t t) {
    if(!t || t>10000 || features.size()!=static_cast<size_t>(t)*1024) throw std::invalid_argument("Codec features must have shape [1,T,1024]");
    auto x=ops_.gelu(ops_.downsample(features,weights_.get("codec.down.weight"),weights_.get("codec.down.bias"),t));
    const uint32_t n=(t+1)/2;
    x=vocos(std::move(x),n,"codec.encoder");
    const std::string p="codec.quantizer.quantizers.0";
    auto latents=linear(x,n,p+".in_project");
    auto codes=ops_.quantize(latents,weights_.get(p+".codebook.weight"));
    auto projected=lookup(codes);
    return {std::move(codes),std::move(projected)};
}
}
