#include "itts25/codec.hpp"
#include "itts25/gpt.hpp"
#include "itts25/acoustic.hpp"
#include "itts25/emotion.hpp"
#include "itts25/speech_features.hpp"
#include "itts25/campplus.hpp"
#include "itts25/pipeline.hpp"
#include "itts25/server.hpp"
#include "itts25/qwen.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using Clock=std::chrono::steady_clock;
static bool compare(const std::string& name,const std::vector<float>& actual,const std::vector<float>& expected,float tolerance=1e-4f,double rms_tolerance=1e-5) {
    if(actual.size()!=expected.size()) throw std::runtime_error("Output shape mismatch: "+name);
    double sum=0, square=0; float peak=0;
    for(size_t i=0;i<actual.size();i++) {
        if(!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw std::runtime_error("Non-finite output: "+name);
        const auto d=std::abs(actual[i]-expected[i]); peak=std::max(peak,d); sum+=d; square+=d*d;
    }
    const double mean=sum/actual.size(), rmse=std::sqrt(square/actual.size());
    const bool pass=peak<tolerance && rmse<rms_tolerance;
    std::cout << "{\"case\":\"" << name << "\",\"max_abs_error\":" << peak << ",\"mean_abs_error\":" << mean << ",\"rmse\":" << rmse << ",\"pass\":" << (pass?"true":"false") << "}\n";
    return pass;
}
static bool gpt_test(const std::string& model,const std::string& golden) {
    itts25::Weights weights(model), fixture(golden,false);
    mit2::MetalContext metal; itts25::GptDecoder gpt(weights,metal);
    bool pass=true;
    std::vector<float> chinese;
    for(const auto* lang:{"ZH","EN","JA","ES","AR"}) {
        const auto name=std::string("prefix.")+lang;
        auto prefix=gpt.prepare(fixture.get("speaker"),fixture.get("emotion"),fixture.ids(name+".tokens"),fixture.ids(name+".language")[0]);
        pass=compare(name,prefix,fixture.get(name+".embeddings")) && pass;
        if(std::string(lang)=="ZH") chinese=std::move(prefix);
    }
    auto check_logits=[&](const std::string& name,const std::vector<float>& logits) {
        const auto& ref=fixture.get(name+".logits");
        auto argmax=[](const std::vector<float>& x) {return std::max_element(x.begin(),x.end())-x.begin();};
        const bool match=argmax(logits)==argmax(ref);
        std::cout << "{\"case\":\"" << name << ".argmax\",\"exact_match\":" << (match?"true":"false") << "}\n";
        return compare(name,logits,ref,1e-4f,2e-5) && match;
    };
    pass=check_logits("decode.prefill",gpt.prefill(chinese)) && pass;
    for(uint32_t i=1;i<=4;i++) {
        auto name="decode."+std::to_string(i);
        pass=check_logits(name,gpt.step(fixture.ids(name+".code")[0],i)) && pass;
    }
    itts25::GenerationOptions config;config.do_sample=false;config.repetition_penalty=1;
    auto generated=gpt.generate(chinese,config,8);
    auto expected_codes=fixture.ids("generation.greedy.codes");
    if(!expected_codes.empty() && expected_codes.back()==8193)expected_codes.pop_back();
    bool generated_match=generated.codes==expected_codes;pass=generated_match && pass;
    std::cout << "{\"case\":\"generation.greedy\",\"exact_match\":" << (generated_match?"true":"false") << "}\n";
    config.num_beams=3;auto beam=gpt.generate(chinese,config,8);
    auto expected_beam=fixture.ids("generation.beam.codes");if(!expected_beam.empty() && expected_beam.back()==8193)expected_beam.pop_back();
    bool beam_match=beam.codes==expected_beam;pass=beam_match && pass;
    std::cout << "{\"case\":\"generation.beam\",\"exact_match\":" << (beam_match?"true":"false") << "}\n";
    std::cout << "{\"stage\":\"gpt_2.5\",\"cached_tokens\":" << gpt.cached_tokens() << ",\"gpu_submissions\":" << metal.command_buffers_submitted() << ",\"pass\":" << (pass?"true":"false") << "}\n";
    return pass && metal.command_buffers_submitted()>0;
}
static bool codec_test(const std::string& model,const std::string& golden) {
    itts25::Weights weights(model), fixture(golden,false);
    mit2::MetalContext metal;
    itts25::EnhancedCodec codec(weights,metal);
    bool pass=true; unsigned count=0;
    for(const auto& t:fixture.bundle().tensors()) {
        if(t.name.rfind("decode.",0)==0 && t.name.find(".codes")!=std::string::npos) {
            const auto prefix=t.name.substr(0,t.name.size()-6);
            pass=compare(prefix,codec.decode(fixture.ids(t.name)),fixture.get(prefix+".output")) && pass;
            count++;
        } else if(t.name.rfind("encode.",0)==0 && t.name.find(".input")!=std::string::npos) {
            const auto prefix=t.name.substr(0,t.name.size()-6);
            if(t.shape.size()!=3 || t.shape[0]!=1 || t.shape[2]!=1024) throw std::runtime_error("Invalid encode fixture shape");
            auto actual=codec.encode(fixture.get(t.name),t.shape[1]);
            const bool codes_match=actual.codes==fixture.ids(prefix+".codes");
            std::cout << "{\"case\":\"" << prefix << ".codes\",\"exact_match\":" << (codes_match?"true":"false") << "}\n";
            pass=compare(prefix+".projected",actual.projected,fixture.get(prefix+".projected")) && codes_match && pass;
            count++;
        }
    }
    const auto gpu=metal.command_buffers_submitted()+codec.extra_gpu_submissions();
    pass=pass && count>=9 && gpu>0;
    std::cout << "{\"stage\":\"enhanced_codec_2.5\",\"cases\":" << count << ",\"gpu_submissions\":" << gpu << ",\"pass\":" << (pass?"true":"false") << "}\n";
    return pass;
}
static bool acoustic_test(const std::string& model,const std::string& golden) {
    itts25::Weights weights(model),fixture(golden,false);
    mit2::MetalContext metal; itts25::AcousticModel acoustic(weights,metal);
    bool pass=true;
    for(const auto& pair:std::vector<std::pair<uint32_t,uint32_t>>{{1,3},{5,9},{7,17}}) {
        const auto name="regulator."+std::to_string(pair.first)+"."+std::to_string(pair.second);
        pass=compare(name,acoustic.regulate(fixture.get(name+".input"),pair.first,pair.second),fixture.get(name+".output")) && pass;
    }
    auto derivatives=acoustic.estimate_cfg(fixture.get("cfm.noise"),fixture.get("cfm.prompt_full"),fixture.get("cfm.condition"),fixture.get("cfm.style"),20,fixture.get("cfm.timestep")[0]);
    auto expected=fixture.get("cfm.estimate");
    std::vector<float> combined=derivatives.first;combined.insert(combined.end(),derivatives.second.begin(),derivatives.second.end());
    pass=compare("cfm.estimate",combined,expected) && pass;
    pass=compare("cfm.25steps",acoustic.flow(fixture.get("cfm.noise"),fixture.get("cfm.prompt"),fixture.get("cfm.condition"),fixture.get("cfm.style"),20,25,0.7f),fixture.get("cfm.output")) && pass;
    for(const auto tokens:{3u,9u,13u}) {
        const auto name="vocoder."+std::to_string(tokens);
        pass=compare(name,acoustic.vocode(fixture.get(name+".input"),tokens),fixture.get(name+".output")) && pass;
    }
    std::cout << "{\"stage\":\"acoustic_2.5\",\"gpu_submissions\":" << metal.command_buffers_submitted() << ",\"pass\":" << (pass?"true":"false") << "}\n";
    return pass && metal.command_buffers_submitted()>0;
}
static bool emotion_test(const std::string& model,const std::string& golden) {
    itts25::Weights weights(model),fixture(golden,false); mit2::MetalContext metal;
    itts25::EmotionEncoder encoder(weights,metal);bool pass=true;
    for(const auto n:{3u,7u,17u,50u}) {
        const auto name="emotion."+std::to_string(n);
        pass=compare(name,encoder.encode(fixture.get(name+".input"),n),fixture.get(name+".output")) && pass;
    }
    std::cout << "{\"stage\":\"emotion_2.5\",\"gpu_submissions\":" << metal.command_buffers_submitted() << ",\"pass\":" << (pass?"true":"false") << "}\n";
    return pass && metal.command_buffers_submitted()>0;
}
static bool speech_test(const std::string& model,const std::string& golden) {
    itts25::Weights weights(model),fixture(golden,false);mit2::MetalContext metal;
    itts25::SpeechFeatures encoder(weights,metal);itts25::EnhancedCodec codec(weights,metal);bool pass=true;
    for(const auto n:{7u,17u,51u}) {
        const auto name="speech."+std::to_string(n);
        encoder.trace_enabled=n==51;
        auto features=encoder.encode(fixture.get(name+".input"),fixture.ids(name+".mask"),n);
        pass=compare(name,features,fixture.get(name+".output"),1e-3f,2e-5) && pass;
        const bool match=codec.encode(features,n).codes==fixture.ids(name+".codes");
        std::cout << "{\"case\":\"" << name << ".codec_codes\",\"exact_match\":" << (match?"true":"false") << "}\n";
        pass=match && pass;
        if(n==51)for(uint32_t i=0;i<17;i++)pass=compare(name+".layer."+std::to_string(i),encoder.layer_trace[i],fixture.get(name+".layer."+std::to_string(i)),1e-3,1e-5) && pass;
    }
    std::cout << "{\"stage\":\"w2v_bert_2.5\",\"gpu_submissions\":" << metal.command_buffers_submitted()+encoder.extra_gpu_submissions()+codec.extra_gpu_submissions() << ",\"pass\":" << (pass?"true":"false") << "}\n";
    return pass;
}
static bool camp_test(const std::string& model,const std::string& golden) {
    itts25::Weights weights(model),fixture(golden,false);itts25::CampPlus camp(weights);bool pass=true;
    for(const auto n:{11u,47u,105u}) {
        const auto name="camp."+std::to_string(n);
        pass=compare(name,camp.encode(fixture.get(name+".input"),n),fixture.get(name+".output")) && pass;
    }
    std::cout << "{\"stage\":\"campplus_2.5\",\"gpu_submissions\":" << camp.gpu_submissions() << ",\"pass\":" << (pass?"true":"false") << "}\n";
    return pass && camp.gpu_submissions()>0;
}
static bool sampling_test(const std::string& path) {
    itts25::Weights fixture(path,false);bool pass=true;
    for(const auto* name:{"standard","beam","beam_edge","unfiltered"}) {
        const std::string n=name;const auto& params=fixture.get(n+".config");itts25::GenerationOptions options;
        options.temperature=params[0];options.top_k=params[1];options.top_p=params[2];options.repetition_penalty=params[3];options.num_beams=params[4];
        auto scores=itts25::apply_gpt_sampling_processors(fixture.get("logits"),fixture.ids("history"),options);auto allowed=fixture.ids(n+".allowed");bool exact=true;
        for(size_t i=0;i<scores.size();i++){exact=exact && std::isfinite(scores[i])==bool(allowed[i]);if(!std::isfinite(scores[i]))scores[i]=0;}
        std::cout << "{\"case\":\"" << n << ".filter\",\"exact_match\":" << (exact?"true":"false") << "}\n";
        pass=compare(n,scores,fixture.get(n+".values"),1e-5,1e-6) && exact && pass;
    }
    return pass;
}
static bool qwen_test(const std::string& model,const std::string& golden) {
    itts25::Weights weights(model),fixture(golden,false);itts25::QwenDecoder qwen(weights);bool pass=true;
    for(const auto* label:{"happy","sad"}) {
        const std::string name=label;auto input=fixture.ids(name+".input");
        auto check=[&](const std::string& test,const std::vector<float>& actual){const auto& expected=fixture.get(test);bool exact=std::max_element(actual.begin(),actual.end())-actual.begin()==std::max_element(expected.begin(),expected.end())-expected.begin();std::cout << "{\"case\":\"" << test << ".argmax\",\"exact_match\":" << (exact?"true":"false") << "}\n";return compare(test,actual,expected,5e-4,1e-4) && exact;};
        pass=check(name+".prefill",qwen.prefill(input)) && pass;
        for(uint32_t i=1;i<=3;i++){auto p=name+".step."+std::to_string(i);pass=check(p+".logits",qwen.step(fixture.ids(p+".input")[0])) && pass;}
        auto actual=qwen.generate(input,128);bool exact=actual==fixture.ids(name+".generated");pass=exact && pass;
        std::cout << "{\"case\":\"" << name << ".generated\",\"exact_match\":" << (exact?"true":"false") << "}\n";
    }
    std::cout << "{\"stage\":\"qwen3_0.6b\",\"gpu_submissions\":" << qwen.submissions() << ",\"pass\":" << (pass?"true":"false") << "}\n";
    return pass;
}
static bool mps_zero_bias_reuse_test() {
    mit2::MetalContext metal;
    constexpr uint32_t cols=512, rows=1536;
    std::vector<float> w(rows*cols,0),bias(rows,0);
    for(uint32_t i=0;i<rows;i++) w[i*cols+i%cols]=1;
    for(uint32_t tokens:{714u,1580u}) {
        const uint32_t capacity=tokens*(cols+rows)+1024;
        metal.beginPass(capacity*4);
        metal.passUploadAlloc(std::vector<float>(capacity,std::numeric_limits<float>::quiet_NaN()));
        metal.endPass();
        std::vector<float> x(tokens*cols),expected(tokens*rows);
        for(uint32_t i=0;i<x.size();i++) x[i]=static_cast<float>(i%1024+1)/cols;
        for(uint32_t t=0;t<tokens;t++) for(uint32_t c=0;c<rows;c++) expected[t*rows+c]=x[t*cols+c%cols];
        metal.beginPass(capacity*4);
        auto input=metal.passUploadAlloc(x);
        auto output=metal.linear_rows_f32_pass("test.identity",w,"test.zero_bias",bias,input,tokens,rows,cols);
        metal.endPass();
        auto result=metal.passRead(output);
        size_t nonfinite=0;for(float v:result) if(!std::isfinite(v)) ++nonfinite;
        std::cout << "{\"case\":\"mps_zero_bias_reuse\",\"tokens\":" << tokens << ",\"nonfinite\":" << nonfinite << "}\n";
        if(nonfinite || !compare("mps_zero_bias_reused_nan_workspace",result,expected,1e-6f,1e-7)) return false;
    }
    return true;
}
int main(int argc,char** argv) {
    try {
        mit2::AutoreleasePool pool;
        if(argc==2 && std::string(argv[1])=="--test-mps-zero-bias") return mps_zero_bias_reuse_test()?0:1;
        if(argc==2 && std::string(argv[1])=="--diagnostics") {
            mit2::MetalContext metal; auto d=metal.diagnostics();
            std::vector<float> test{1,-2,3,0.25f};
            if(!metal.smoke_copy(test)) throw std::runtime_error("Metal copy test failed");
            std::cout << "{\"device\":\"" << d.device_name << "\",\"unified_memory\":" << (d.unified_memory?"true":"false") << ",\"gpu_submissions\":" << metal.command_buffers_submitted() << "}\n";
            return 0;
        }
        if(argc==3 && std::string(argv[1])=="--test-sampling") return sampling_test(argv[2])?0:1;
        if(argc==3 && std::string(argv[1])=="--serve"){itts25::serve(argv[2]);return 0;}
        if(argc==4 && std::string(argv[1])=="--test-codec") return codec_test(argv[2],argv[3])?0:1;
        if(argc==4 && std::string(argv[1])=="--test-gpt") return gpt_test(argv[2],argv[3])?0:1;
        if(argc==4 && std::string(argv[1])=="--test-acoustic") return acoustic_test(argv[2],argv[3])?0:1;
        if(argc==4 && std::string(argv[1])=="--test-emotion") return emotion_test(argv[2],argv[3])?0:1;
        if(argc==4 && std::string(argv[1])=="--test-speech") return speech_test(argv[2],argv[3])?0:1;
        if(argc==4 && std::string(argv[1])=="--test-qwen") return qwen_test(argv[2],argv[3])?0:1;
        if(argc==4 && std::string(argv[1])=="--test-camp") return camp_test(argv[2],argv[3])?0:1;
        if(argc==5 && std::string(argv[1])=="--synthesize") {
            auto start=Clock::now();itts25::Synthesizer synthesizer(argv[2]);itts25::Weights input(argv[3],false);
            auto result=synthesizer.synthesize(input,itts25::GenerationOptions{});itts25::save_wave(argv[4],result.wave);
            std::cout << "{\"stage\":\"native_synthesis\",\"codes\":" << result.codes.size() << ",\"stopped\":" << (result.stopped?"true":"false") << ",\"duration_seconds\":" << result.wave.size()/22050.0 << ",\"features_seconds\":" << result.features_seconds << ",\"gpt_seconds\":" << result.gpt_seconds << ",\"acoustic_seconds\":" << result.acoustic_seconds << ",\"vocoder_seconds\":" << result.vocoder_seconds << ",\"gpu_submissions\":" << result.gpu_submissions << ",\"total_seconds\":" << std::chrono::duration<double>(Clock::now()-start).count() << "}\n";
            return 0;
        }
        if(argc==5 && std::string(argv[1])=="--codec-decode") {
            itts25::Weights weights(argv[2]); mit2::MetalContext metal; itts25::EnhancedCodec codec(weights,metal);
            std::ifstream input(argv[3],std::ios::binary|std::ios::ate);
            auto size=input.tellg();
            if(size<=0 || size%4 || size>5000*4) throw std::runtime_error("Invalid code ID file");
            std::vector<uint32_t> codes(static_cast<size_t>(size)/4); input.seekg(0);
            input.read(reinterpret_cast<char*>(codes.data()),size);
            auto output=codec.decode(codes);
            std::ofstream dest(argv[4],std::ios::binary);
            dest.write(reinterpret_cast<const char*>(output.data()),output.size()*4);
            if(!dest) throw std::runtime_error("Could not write codec output");
            return 0;
        }
        std::cout << "IndexTTS 2.5 Metal runtime\n  --diagnostics\n  --test-codec MODEL_BUNDLE GOLDEN_BUNDLE\n  --test-gpt MODEL_BUNDLE GOLDEN_BUNDLE\n  --codec-decode MODEL_BUNDLE CODES_U32 OUTPUT_F32\n";
        return argc==1 || (argc==2 && std::string(argv[1])=="--help") ? 0 : 2;
    } catch(const std::exception& e) { std::cerr << "itts25-metal: " << e.what() << '\n'; return 1; }
}
