#include "itts25/runtime.hpp"
#include "itts25/http.hpp"
#include "itts25/model_download.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <mach-o/dyld.h>
#include <cstring>
#include <cstdlib>
namespace {
std::filesystem::path resource_root() {
    uint32_t size=0;
    _NSGetExecutablePath(nullptr,&size);
    std::string executable(size,'\0');
    if(_NSGetExecutablePath(executable.data(),&size)!=0)throw std::runtime_error("Cannot locate native executable");
    executable.resize(std::strlen(executable.c_str()));
    auto directory=std::filesystem::weakly_canonical(executable).parent_path();
    const auto fallback=directory.parent_path();
    for(unsigned depth=0;depth<8;depth++){
        if(std::filesystem::exists(directory/"web/index.html")&&
           (std::filesystem::exists(directory/"bundles/full/manifest.json")||std::filesystem::exists(directory/"CMakeLists.txt")))return directory;
        if(directory==directory.parent_path())break;
        directory=directory.parent_path();
    }
    return fallback;
}
}
int main(int argc,char** argv) {
    try {
        const auto root=resource_root();
        itts25::HttpConfig config;bool http=false,cli=false,download_only=false,allow_download=true;
        config.store=(root/"voices").string();config.web_file=(root/"web/index.html").string();config.example_audio=(root/"examples/voice_01.wav").string();
        std::string model=(root/"bundles/full").string(),resources=(root/"bundles/frontend").string(),voice,text,output=(root/"outputs/native.wav").string(),emotion_text,clone_output;itts25::Json options=itts25::Json::object();
        if(const char* value=std::getenv("ITTS25_METAL_BUNDLE"))model=value;
        if(const char* value=std::getenv("MODEL_BUNDLE"))model=value;
        if(const char* value=std::getenv("ITTS25_FRONTEND"))resources=value;
        for(int i=1;i<argc;i++){std::string arg=argv[i];auto value=[&](){if(i+1>=argc)throw std::invalid_argument("Missing argument: "+arg);return std::string(argv[++i]);};
            if(arg=="--help"||arg=="-h"){std::cout<<"IndexTTS 2.5 native C++/Metal (default: web service)\n  Resources are located relative to the executable. Missing default resources download from ModelScope.\n  --download-only  Prepare model, frontend and example audio, then exit.\n  --no-download    Require local resources without network access.\n  --cli --model_bundle DIR --frontend DIR --voice AUDIO_OR_BUNDLE --text TEXT --output WAV\n  --clone OUTPUT_VOICE --emotion-text TEXT --options JSON --options-file FILE\n  --http --host HOST --port PORT --voice_store DIR --webkey KEY --web | --server\n  --web_file FILE --example_audio FILE --seed_example\n  --queue_size N --voice_cache_size N --tts_concurrency N --clone_concurrency N\n";return 0;}
            if(arg=="--download-only"){download_only=true;continue;}
            if(arg=="--no-download"){allow_download=false;continue;}
            if(arg=="--cli"){cli=true;continue;}if(arg=="--web_file"){config.web_file=value();continue;}if(arg=="--example_audio"){config.example_audio=value();continue;}
            if(arg=="--http"){http=true;continue;}if(arg=="--host"){config.host=value();continue;}if(arg=="--port"){auto n=std::stoul(value());if(n<1||n>65535)throw std::invalid_argument("Invalid port");config.port=n;continue;}
            if(arg=="--voice_store"){config.store=value();continue;}if(arg=="--webkey"){config.webkey=value();continue;}if(arg=="--web"){config.web=true;continue;}if(arg=="--server"){config.web=false;continue;}if(arg=="--seed_example"){config.seed_example=true;continue;}
            if(arg=="--queue_size"){config.queue_size=std::stoul(value());continue;}if(arg=="--voice_cache_size"){config.voice_cache_size=std::stoul(value());continue;}if(arg=="--tts_concurrency"){config.tts_concurrency=std::stoul(value());continue;}if(arg=="--clone_concurrency"){config.clone_concurrency=std::stoul(value());continue;}
            if(arg=="--prompt"){voice=value();continue;}
            if(arg=="--emotion_text"){emotion_text=value();continue;}
            if(arg=="--greedy"){options["do_sample"]=false;continue;}
            if(arg=="--no-text-normalization"){options["text_normalization"]=false;continue;}
            const std::vector<std::string> numeric{"--duration_factor","--temperature","--top_k","--top_p","--num_beams","--repetition_penalty","--length_penalty","--max_mel_tokens","--max_text_tokens_per_segment","--interval_silence","--emo_alpha"};
            if(std::find(numeric.begin(),numeric.end(),arg)!=numeric.end()){options[arg.substr(2)]=itts25::Json::parse(value());continue;}
            if(arg=="--model_bundle")model=value();else if(arg=="--frontend")resources=value();else if(arg=="--voice")voice=value();else if(arg=="--text")text=value();else if(arg=="--output")output=value();else if(arg=="--clone")clone_output=value();else if(arg=="--emotion-text")emotion_text=value();else if(arg=="--lang")options["lang"]=value();else if(arg=="--seed")options["seed"]=std::stoull(value());else if(arg=="--options")options=itts25::Json::parse(value());else if(arg=="--options-file"){std::ifstream in(value());in>>options;}else throw std::invalid_argument("Unknown option: "+arg);
        }
        const bool run_server=http||(voice.empty()&&text.empty()&&clone_output.empty()&&emotion_text.empty()&&!cli);
        if(!download_only&&!run_server&&voice.empty()&&emotion_text.empty())throw std::invalid_argument("--voice is required");
        if(!download_only&&!run_server&&text.empty()&&clone_output.empty()&&emotion_text.empty())throw std::invalid_argument("--text or --clone is required");
        itts25::ensure_model_resources(root,model,resources,config.example_audio,download_only||(run_server&&config.seed_example),allow_download);
        if(download_only){std::cout<<"Native model/frontend packages and example audio are ready.\n";return 0;}
        if(run_server){config.model=model;config.frontend=resources;return itts25::run_http(config);}
        itts25::NativeRuntime runtime(model,resources);
        if(!emotion_text.empty()){if(voice.empty()){std::cout<<runtime.emotion_text(emotion_text).dump()<<'\n';return 0;}options["emo_text"]=emotion_text;}
        auto v=mit2::bundle_path_is_single_file(voice)||std::filesystem::is_directory(voice)?itts25::read_native_voice(voice):runtime.clone(voice);
        if(!clone_output.empty()){itts25::save_native_voice(clone_output,v);std::cout<<itts25::Json{{"status","ok"},{"voice",clone_output}}.dump()<<'\n';}
        if(!text.empty())std::cout<<runtime.synthesize(v,text,output,options).dump()<<'\n';else if(clone_output.empty())throw std::invalid_argument("--text or --clone is required");return 0;
    }catch(const std::exception& e){std::cerr<<itts25::Json{{"error",e.what()}}.dump()<<'\n';return 1;}
}
