#include "itts25/runtime.hpp"
#include <mach/mach.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
using itts25::Json;
static Json memory() {
    task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    if(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)!=KERN_SUCCESS)throw std::runtime_error("task_info failed");
    return {{"rss_bytes",info.resident_size},{"footprint_bytes",info.phys_footprint}};
}
static std::string hash_file(const std::string& path) {
    std::ifstream f(path,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(f)),{});
    return itts25::sha256(bytes.data(),bytes.size());
}
int main(int argc,char** argv) {
    try {
        if(argc!=5)throw std::invalid_argument("Usage: itts25-runtime-check MODEL FRONTEND VOICE OUTPUT_DIR");
        std::filesystem::create_directories(argv[4]);Json report;std::map<std::string,std::string> hashes;
        for(unsigned epoch=0;epoch<2;epoch++) {
            Json row{{"before",memory()}};
            {
                mit2::AutoreleasePool pool;itts25::NativeRuntime runtime(argv[1],argv[2]);auto voice=itts25::read_native_voice(argv[3]);
                const std::string emotional="我今天非常高兴！";auto expected=runtime.emotion_text(emotional);
                std::vector<std::pair<std::string,Json>> classifications{{emotional,expected}};
                for(unsigned i=0;i<18;i++){auto text="这是第"+std::to_string(i)+"次平静自然的测试。";classifications.emplace_back(text,runtime.emotion_text(text));}
                // There are 19 distinct keys and only 16 cache slots, so replay
                // every key to include classifications that were evicted.
                for(const auto& entry:classifications)if(runtime.emotion_text(entry.first)!=entry.second)throw std::runtime_error("Classification changed on eviction replay");
                if(runtime.emotion_text(emotional)!=expected)throw std::runtime_error("Classification changed after cache eviction");
                bool rejected=false;try{runtime.emotion_text("");}catch(const std::invalid_argument&){rejected=true;}
                if(!rejected)throw std::runtime_error("Empty emotion text accepted");
                auto invalid=voice;invalid.prompt.pop_back();rejected=false;
                try{runtime.synthesize(invalid,"你好世界",std::string(argv[4])+"/invalid.wav",{{"seed",424242}});}catch(const std::invalid_argument&){rejected=true;}
                if(!rejected)throw std::runtime_error("Malformed voice prompt accepted");
                for(const auto& name:{"recovery","emotion","beam","audio"}) {
                    auto output=std::string(argv[4])+"/"+name+"-"+std::to_string(epoch)+".wav";
                    Json options{{"seed",424242}};std::string text="你好世界";
                    if(std::string(name)=="emotion")options["emo_text"]=emotional;
                    if(std::string(name)=="audio"){options["emo_audio_prompt"]="examples/voice_01.wav";options["emo_alpha"]=0.5;}
                    if(std::string(name)=="beam"){options["num_beams"]=3;text="你好，今天我们一起测试本地语音合成的性能，并检查较长文本的输出是否稳定。";}
                    auto result=runtime.synthesize(voice,text,output,options);auto hash=hash_file(output);
                    if(epoch==0)hashes[name]=hash;else if(hashes.at(name)!=hash)throw std::runtime_error("Output changed after runtime reconstruction");
                    row["outputs"][name]={{"sha256",hash},{"total_seconds",result.at("total_seconds")}};
                }
                row["active"]=memory();
            }
            row["destroyed"]=memory();report["epochs"].push_back(row);
        }
        report["passed"]=true;std::cout<<report.dump(2)<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
