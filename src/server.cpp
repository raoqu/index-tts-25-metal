#include "itts25/server.hpp"
#include "itts25/pipeline.hpp"
#include "itts25/qwen.hpp"
#include <memory>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
namespace itts25 {
static std::string json_string(const std::string& value) {
    std::string out="\"";
    for(unsigned char c:value){if(c=='"' || c=='\\'){out+='\\';out+=c;}else if(c<32)out+=' ';else out+=c;}
    return out+'"';
}
void serve(const std::string& model) {
    Weights weights(model);mit2::MetalContext metal;
    SpeechFeatures speech(weights,metal);CampPlus camp(weights);EmotionEncoder emotion(weights,metal);
    GptDecoder gpt(weights,metal);EnhancedCodec codec(weights,metal);AcousticModel acoustic(weights,metal);
    std::unique_ptr<QwenDecoder> qwen;
    auto submissions=[&](){return metal.command_buffers_submitted()+speech.extra_gpu_submissions()+camp.gpu_submissions()+codec.extra_gpu_submissions()+(qwen?qwen->submissions():0);};
    std::cout << "{\"ready\":true,\"backend\":\"metal\",\"model\":\"2.5\"}" << std::endl;
    std::string line;
    while(std::getline(std::cin,line)) {
        try {
            mit2::AutoreleasePool request_pool;
            std::istringstream fields(line);std::string op,input_path,output_path,extra;
            if(!std::getline(fields,op,'\t') || !std::getline(fields,input_path,'\t') || !std::getline(fields,output_path,'\t') || std::getline(fields,extra,'\t'))throw std::invalid_argument("Expected operation, input bundle and output file");
            Weights input(input_path,false);std::vector<float> values;std::vector<uint32_t> ids;std::vector<uint32_t> shape;
            const auto before=submissions();const auto start=std::chrono::steady_clock::now();
            auto tokens=[&](const std::string& name,uint32_t width){const auto& s=input.info(name).shape;if(s.size()!=3 || s[0]!=1 || s[2]!=width || s[1]<=0 || s[1]>4096)throw std::invalid_argument("Invalid BTC shape: "+name);return static_cast<uint32_t>(s[1]);};
            auto scalar=[&](const std::string& name){auto x=input.ids(name);if(x.size()!=1)throw std::invalid_argument("Expected scalar: "+name);return x[0];};
            if(op=="qwen") {if(!qwen)qwen=std::make_unique<QwenDecoder>(weights);ids=qwen->generate(input.ids("input"));shape={1,static_cast<uint32_t>(ids.size())};}
            else if(op=="speech") {auto t=tokens("input",160);values=speech.encode(input.get("input"),input.ids("mask"),t);shape={1,t,1024};}
            else if(op=="camp") {values=camp.encode(input.get("input"),tokens("input",80));shape={1,192};}
            else if(op=="emotion") {values=emotion.encode(input.get("input"),tokens("input",1024));shape={1,1280};}
            else if(op=="codec") {auto codes=input.ids("codes");values=codec.decode(codes);shape={1,static_cast<uint32_t>(codes.size()*2),1024};}
            else if(op=="regulate") {auto out=scalar("length");values=acoustic.regulate(input.get("input"),tokens("input",1024),out);shape={1,out,512};}
            else if(op=="flow") {auto t=tokens("noise",80);tokens("prompt",80);if(tokens("condition",512)!=t)throw std::invalid_argument("Flow length mismatch");auto config=input.get("config");if(config.size()!=2)throw std::invalid_argument("Invalid flow config");values=acoustic.flow(input.get("noise"),input.get("prompt"),input.get("condition"),input.get("style"),t,static_cast<uint32_t>(config[0]),config[1]);shape={1,t,80};}
            else if(op=="vocode") {auto t=tokens("input",80);values=acoustic.vocode(input.get("input"),t);shape={1,1,static_cast<uint32_t>(values.size())};}
            else if(op=="gpt") {
                const auto& config=input.get("config");auto seed=input.ids("seed");if(config.size()!=8 || seed.size()!=2)throw std::invalid_argument("Invalid generation config");
                GenerationOptions options;options.do_sample=config[0]!=0;options.temperature=config[1];options.top_k=static_cast<uint32_t>(config[2]);options.top_p=config[3];options.repetition_penalty=config[4];options.num_beams=static_cast<uint32_t>(config[5]);options.length_penalty=config[6];options.seed=uint64_t(seed[0])|(uint64_t(seed[1])<<32);
                auto prefix=gpt.prepare(input.get("speaker"),input.get("emotion"),input.ids("text"),scalar("language"));
                auto generated=gpt.generate(prefix,options,static_cast<uint32_t>(config[7]));ids=std::move(generated.codes);
                if(generated.stopped)ids.push_back(8193);if(ids.empty())throw std::runtime_error("Empty generated speech");shape={1,static_cast<uint32_t>(ids.size())};
            } else throw std::invalid_argument("Unknown native operation: "+op);
            std::ofstream out(output_path,std::ios::binary);if(!out)throw std::runtime_error("Cannot create operation output");
            if(ids.empty())out.write(reinterpret_cast<const char*>(values.data()),values.size()*4);else out.write(reinterpret_cast<const char*>(ids.data()),ids.size()*4);
            out.close();if(!out)throw std::runtime_error("Native output write failed");
            std::cout << "{\"ok\":true,\"dtype\":\"" << (ids.empty()?"f32":"u32") << "\",\"shape\":[";
            for(size_t i=0;i<shape.size();i++)std::cout << (i?",":"") << shape[i];
            std::cout << "],\"gpu_submissions\":" << submissions()-before << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << "}" << std::endl;
        }catch(const std::exception& e){std::cout << "{\"ok\":false,\"error\":" << json_string(e.what()) << "}" << std::endl;}
    }
}
}
