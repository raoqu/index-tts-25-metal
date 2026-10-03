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
std::vector<float> EnhancedCodec::vocos(std::vector<float> x,uint32_t t,const std::string& p) {
    x=norm(conv(x,t,p+".0.embed"),t,p+".0.norm");
    for(uint32_t i=0;i<12;i++) {
        const auto layer=p+".0.convnext."+std::to_string(i);
        auto hidden=norm(conv(x,t,layer+".dwconv",true),t,layer+".norm");
        hidden=linear(ops_.gelu(linear(hidden,t,layer+".pwconv1")),t,layer+".pwconv2");
        x=metal_.add_f32(x,ops_.scale(hidden,weights_.get(layer+".gamma")));
    }
    return linear(norm(x,t,p+".0.final_layer_norm"),t,p+".1");
}
std::vector<float> EnhancedCodec::lookup(const std::vector<uint32_t>& codes) {
    if(codes.empty() || codes.size()>5000 || *std::max_element(codes.begin(),codes.end())>=8192)
        throw std::invalid_argument("Codec expects 1..5000 code IDs in [0,8192)");
    const std::string p="codec.quantizer.quantizers.0";
    auto embeddings=metal_.embedding_f32_resident(p+".codebook.weight",weights_.get(p+".codebook.weight"),codes,8);
    return linear(embeddings,codes.size(),p+".out_project");
}
std::vector<float> EnhancedCodec::decode(const std::vector<uint32_t>& codes) {
    auto x=vocos(lookup(codes),codes.size(),"codec.decoder");
    x=metal_.nearest_interpolate_f32(x,codes.size(),codes.size()*2,1024);
    return conv(x,codes.size()*2,"codec.up");
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
