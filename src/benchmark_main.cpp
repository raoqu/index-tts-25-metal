#include "itts25/runtime.hpp"
#include <mach/mach.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
using itts25::Json;
static Json memory() {
    task_vm_info_data_t info{}; mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    if(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)!=KERN_SUCCESS)
        throw std::runtime_error("task_info failed");
    return {{"rss_bytes",info.resident_size},{"footprint_bytes",info.phys_footprint}};
}
int main(int argc,char** argv) {
    try {
        if(argc!=7)throw std::invalid_argument("Usage: itts25-benchmark MODEL FRONTEND CASES_JSON OUTPUT_DIR CYCLES WARMUPS");
        std::ifstream input(argv[3]);Json cases;input>>cases;
        const unsigned cycles=std::stoul(argv[5]),warmups=std::stoul(argv[6]);
        if(!cases.is_array()||cases.empty()||cycles<1||cycles>1000||warmups>100)throw std::invalid_argument("Invalid benchmark configuration");
        std::filesystem::create_directories(argv[4]);
        auto t=std::chrono::steady_clock::now();itts25::NativeRuntime runtime(argv[1],argv[2]);
        std::cout<<Json{{"kind","load"},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()},{"memory",memory()}}.dump()<<std::endl;
        std::vector<itts25::NativeVoice> voices;
        for(const auto& c:cases)voices.push_back(itts25::read_native_voice(c.at("voice")));
        for(unsigned cycle=0;cycle<cycles+warmups;cycle++)for(size_t i=0;i<cases.size();i++) {
            const auto& c=cases[i];auto id=c.at("id").get<std::string>();
            auto output=std::string(argv[4])+"/"+id+".wav";
            auto before=runtime.resource_stats();auto result=runtime.synthesize(voices[i],c.at("text"),output,c.value("options",Json::object()));
            auto after=runtime.resource_stats();
            std::ifstream wav(output,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(wav)),{});result["wav_sha256"]=itts25::sha256(bytes.data(),bytes.size());
            result["kind"]="synthesis";result["case"]=id;result["cycle"]=cycle;result["warmup"]=cycle<warmups;result["memory"]=memory();
            result["metal"]={{"allocations",after.buffer_allocations-before.buffer_allocations},{"allocated_bytes",after.buffer_bytes_allocated-before.buffer_bytes_allocated},{"gpu_seconds",after.gpu_elapsed_seconds-before.gpu_elapsed_seconds}};
            std::cout<<result.dump()<<std::endl;
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
