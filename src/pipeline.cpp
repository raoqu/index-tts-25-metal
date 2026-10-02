#include "itts25/pipeline.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <algorithm>
#include <stdexcept>
namespace itts25 {
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
Synthesizer::Synthesizer(const std::string& path):weights_(path),features_(weights_,metal_),camp_(weights_),emotion_(weights_,metal_),gpt_(weights_,metal_),codec_(weights_,metal_),acoustic_(weights_,metal_) {}
uint64_t Synthesizer::submissions() const{return metal_.command_buffers_submitted()+features_.extra_gpu_submissions()+camp_.gpu_submissions()+codec_.extra_gpu_submissions();}
SynthesisResult Synthesizer::synthesize(Weights& input,const GenerationOptions& options,uint32_t maximum,float duration){
    if(!std::isfinite(duration) || duration<0.25f || duration>4)throw std::invalid_argument("Invalid duration factor");
    const auto count=[](const mit2::TensorInfo& t,int64_t width){if(t.shape.size()!=3 || t.shape[0]!=1 || t.shape[2]!=width || t.shape[1]<=0)throw std::invalid_argument("Invalid synthesis input shape: "+t.name);return static_cast<uint32_t>(t.shape[1]);};
    const auto f=count(input.info("speech.input"),160),bank=count(input.info("camp.input"),80),prompt_tokens=count(input.info("prompt.mel"),80);
    const auto gpu_before=submissions();SynthesisResult result;auto start=Clock::now();
    auto reference=features_.encode(input.get("speech.input"),input.ids("speech.mask"),f);
    auto style=camp_.encode(input.get("camp.input"),bank);
    auto emovec=emotion_.encode(reference,f);
    auto prompt_condition=acoustic_.regulate(reference,f,prompt_tokens);
    auto prefix=gpt_.prepare(style,emovec,input.ids("text.ids"),input.ids("text.language").at(0));
    result.features_seconds=elapsed(start);start=Clock::now();
    auto generated=gpt_.generate(prefix,options,maximum);result.codes=std::move(generated.codes);result.stopped=generated.stopped;
    result.gpt_seconds=elapsed(start);start=Clock::now();
    if(result.codes.empty())throw std::runtime_error("GPT produced no semantic codes");
    auto semantic=codec_.decode(result.codes);
    const auto target_tokens=static_cast<uint32_t>(static_cast<double>(semantic.size()/1024)*1.72*duration);
    if(!target_tokens || target_tokens+prompt_tokens>4096)throw std::invalid_argument("Synthesis acoustic token capacity exceeded");
    auto cond=acoustic_.regulate(semantic,semantic.size()/1024,target_tokens);
    prompt_condition.insert(prompt_condition.end(),cond.begin(),cond.end());
    const auto total=target_tokens+prompt_tokens;std::vector<float> noise(static_cast<size_t>(total)*80);
    std::mt19937_64 rng(options.seed);std::normal_distribution<float> normal(0,1);
    for(auto& value:noise)value=normal(rng);
    auto mel=acoustic_.flow(std::move(noise),input.get("prompt.mel"),prompt_condition,style,total);
    mel.erase(mel.begin(),mel.begin()+static_cast<size_t>(prompt_tokens)*80);
    result.acoustic_seconds=elapsed(start);start=Clock::now();
    result.wave=acoustic_.vocode(mel,target_tokens);result.vocoder_seconds=elapsed(start);result.gpu_submissions=submissions()-gpu_before;
    return result;
}
void save_wave(const std::string& path,const std::vector<float>& wave,uint32_t rate){
    if(wave.empty() || wave.size()>0x7fffffff/2)throw std::invalid_argument("Invalid WAV sample count");
    auto parent=std::filesystem::path(path).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("Cannot open WAV output");
    auto u32=[&](uint32_t x){char bytes[4];for(int i=0;i<4;i++)bytes[i]=static_cast<char>(x>>(8*i));out.write(bytes,4);};
    auto u16=[&](uint16_t x){char bytes[2]{static_cast<char>(x),static_cast<char>(x>>8)};out.write(bytes,2);};
    out.write("RIFF",4);u32(36+wave.size()*2);out.write("WAVEfmt ",8);u32(16);u16(1);u16(1);u32(rate);u32(rate*2);u16(2);u16(16);out.write("data",4);u32(wave.size()*2);
    for(float x:wave){if(!std::isfinite(x))throw std::runtime_error("Non-finite synthesis waveform");u16(static_cast<uint16_t>(static_cast<int16_t>(std::clamp(x,-1.0f,1.0f)*32767.0f)));}
    if(!out)throw std::runtime_error("WAV write failed");
}
}
