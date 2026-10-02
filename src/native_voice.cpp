#include "itts25/runtime.hpp"
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <set>
#include <algorithm>
namespace itts25 {
namespace {
constexpr uint64_t limit=128*1024*1024;
Json manifest(const std::string& path) {
    if(std::filesystem::is_directory(path)){std::ifstream in(path+"/manifest.json");Json j;in>>j;return j;}
    std::ifstream in(path,std::ios::binary);if(!in)throw std::invalid_argument("Cannot open voice bundle");in.seekg(0,std::ios::end);uint64_t size=in.tellg();if(size<36||size>limit)throw std::invalid_argument("Invalid voice bundle size");
    in.seekg(size-24);uint64_t off,len;char magic[8];in.read(reinterpret_cast<char*>(&off),8);in.read(reinterpret_cast<char*>(&len),8);in.read(magic,8);
    if(std::memcmp(magic,"MIT2VOIC",8)||off<12||len>1024*1024||off>size-24||len!=size-24-off)throw std::invalid_argument("Invalid voice footer");std::string raw(len,'\0');in.seekg(off);in.read(raw.data(),len);return Json::parse(raw);
}
}
NativeVoice read_native_voice(const std::string& path) {
    auto j=manifest(path);auto meta=j.at("metadata");if(meta.at("target")!="index-tts2.5"||meta.at("kind")!="voice"||meta.at("sample_rate")!=22050)throw std::invalid_argument("Expected IndexTTS 2.5 voice; clone 2.0 voices again");
    const bool directory=std::filesystem::is_directory(path);
    uint64_t payload_end=std::filesystem::file_size(directory?path+"/weights.bin":path);
    if(payload_end>limit)throw std::invalid_argument("Voice bundle exceeds capacity");
    if(!directory){std::ifstream in(path,std::ios::binary);in.seekg(payload_end-24);in.read(reinterpret_cast<char*>(&payload_end),8);}
    std::vector<std::pair<uint64_t,uint64_t>> ranges;
    for(const auto& record:j.at("tensors")){
        if(!record.at("offset").is_number_unsigned()||!record.at("nbytes").is_number_unsigned()||record.at("dtype")!="f32")throw std::invalid_argument("Invalid voice tensor record");
        auto begin=record.at("offset").get<uint64_t>(),bytes=record.at("nbytes").get<uint64_t>();
        if(begin<4096||begin%4096||begin>payload_end||!bytes||bytes>payload_end-begin)throw std::invalid_argument("Voice tensor exceeds payload bounds");ranges.emplace_back(begin,begin+bytes);
    }
    std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();i++)if(ranges[i-1].second>ranges[i].first)throw std::invalid_argument("Overlapping voice tensors");
    Weights w(path,false);std::set<std::string> required{"voice.spk_cond","voice.emo_cond","voice.style","voice.prompt_condition","voice.mel","voice.audio"};
    if(w.bundle().tensor_count()!=required.size())throw std::invalid_argument("Incomplete 2.5 voice bundle");
    for(const auto& name:required){const auto& t=w.info(name);const auto& v=w.get(name);if(t.nbytes>limit||t.sha256!=sha256(v.data(),v.size()*4)||!std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}))throw std::invalid_argument("Invalid voice tensor/checksum: "+name);}
    auto btc=[&](const std::string& name,int64_t width,uint32_t low,uint32_t high){const auto& s=w.info(name).shape;if(s.size()!=3||s[0]!=1||s[2]!=width||s[1]<low||s[1]>high)throw std::invalid_argument("Invalid voice shape: "+name);return static_cast<uint32_t>(s[1]);};
    NativeVoice v;v.speaker_frames=btc("voice.spk_cond",1024,3,1600);v.emotion_frames=btc("voice.emo_cond",1024,3,1600);
    auto s=w.info("voice.mel").shape;if(s.size()!=3||s[0]!=1||s[1]!=80||s[2]<1||s[2]>1600)throw std::invalid_argument("Invalid voice mel");v.mel_frames=s[2];
    if(w.info("voice.style").shape!=std::vector<int64_t>{1,192}||btc("voice.prompt_condition",512,1,1600)!=v.mel_frames)throw std::invalid_argument("Invalid voice style/prompt");
    s=w.info("voice.audio").shape;if(s.size()!=2||s[0]!=1||s[1]<2205||s[1]>15*22050)throw std::invalid_argument("Invalid preview audio");
    v.speaker=w.get("voice.spk_cond");v.emotion=w.get("voice.emo_cond");v.style=w.get("voice.style");v.prompt=w.get("voice.prompt_condition");v.audio=w.get("voice.audio");v.mel.resize(v.mel_frames*80);auto& mel=w.get("voice.mel");for(size_t t=0;t<v.mel_frames;t++)for(size_t c=0;c<80;c++)v.mel[t*80+c]=mel[c*v.mel_frames+t];return v;
}
void save_native_voice(const std::string& path,const NativeVoice& v) {
    auto parent=std::filesystem::path(path).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("Cannot save voice");uint32_t header[3]{0x3254494d,1,4096};out.write(reinterpret_cast<char*>(header),12);Json tensors=Json::array();
    auto add=[&](const std::string& name,const std::vector<float>& data,const std::vector<uint32_t>& shape){uint64_t pos=out.tellp(),off=(pos+4095)/4096*4096;std::string pad(off-pos,'\0');out.write(pad.data(),pad.size());out.write(reinterpret_cast<const char*>(data.data()),data.size()*4);tensors.push_back(Json{{"name",name},{"shape",shape},{"dtype","f32"},{"offset",off},{"nbytes",data.size()*4},{"sha256",sha256(data.data(),data.size()*4)},{"component","voice"},{"layout","row_major"}});};
    add("voice.spk_cond",v.speaker,{1,v.speaker_frames,1024});add("voice.emo_cond",v.emotion,{1,v.emotion_frames,1024});add("voice.style",v.style,{1,192});add("voice.prompt_condition",v.prompt,{1,v.mel_frames,512});
    std::vector<float> mel(v.mel.size());for(size_t t=0;t<v.mel_frames;t++)for(size_t c=0;c<80;c++)mel[c*v.mel_frames+t]=v.mel[t*80+c];add("voice.mel",mel,{1,80,v.mel_frames});add("voice.audio",v.audio,{1,static_cast<uint32_t>(v.audio.size())});
    Json j{{"format","MIT2"},{"version",1},{"alignment",4096},{"endianness","little"},{"weights_file","weights.bin"},{"metadata",{{"target","index-tts2.5"},{"kind","voice"},{"sample_rate",22050}}},{"tensors",tensors}};
    auto raw=j.dump();uint64_t off=out.tellp(),len=raw.size();out.write(raw.data(),raw.size());out.write(reinterpret_cast<char*>(&off),8);out.write(reinterpret_cast<char*>(&len),8);out.write("MIT2VOIC",8);out.close();if(!out)throw std::runtime_error("Voice write failed");(void)read_native_voice(path);
}
}
