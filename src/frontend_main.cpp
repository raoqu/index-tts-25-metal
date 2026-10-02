#include "itts25/frontend.hpp"
#include "itts25/text_processor.hpp"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc==6&&std::string(argv[1])=="--process") {itts25::ByteTokenizer t(argv[2]);itts25::TextProcessor p(argv[2]);auto text=p.process(argv[4],argv[3]);auto segments=p.split(text,"<|"+std::string(argv[3])+"|> ",t,std::stoul(argv[5]));std::cout<<itts25::Json{{"text",text},{"segments",segments}}.dump()<<'\n';return 0;}
        if(argc==4&&std::string(argv[1])=="--tokenize") {itts25::ByteTokenizer t(argv[2]);std::cout<<itts25::Json(t.encode(argv[3])).dump()<<'\n';return 0;}
        if(argc==4&&std::string(argv[1])=="--qwen-tokenize") {itts25::ByteTokenizer t(argv[2],true);std::cout<<itts25::Json(t.encode(argv[3])).dump()<<'\n';return 0;}
        if(argc==7&&std::string(argv[1])=="--prepare") {itts25::ByteTokenizer t(argv[2]);itts25::AudioFrontend f(argv[2]);auto ids=t.encode("<|"+std::string(argv[4])+"|> "+argv[5]);ids.push_back(1);auto a=f.prepare(argv[3]);itts25::write_input_bundle(argv[6],a,ids,t.language(argv[4]));std::cout<<itts25::Json{{"speech_frames",a.speech_frames},{"camp_frames",a.camp_frames},{"mel_frames",a.mel_frames},{"text_ids",ids}}.dump()<<'\n';return 0;}
        std::cerr<<"Usage: itts25-frontend --tokenize RESOURCES TEXT | --qwen-tokenize RESOURCES TEXT | --prepare RESOURCES AUDIO LANG TEXT OUTPUT_BUNDLE\n";return 2;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
