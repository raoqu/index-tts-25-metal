#include "itts25/runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <regex>
#include <stdexcept>
namespace itts25 {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double number(const Json& o,const char* name,double def,double lo,double hi,bool integer=false){if(!o.contains(name))return def;const auto& x=o.at(name);if(!x.is_number()||x.is_boolean())throw std::invalid_argument(std::string("Invalid generation field: ")+name);double v=x.get<double>();if(!std::isfinite(v)||v<lo||v>hi||(integer&&std::trunc(v)!=v))throw std::invalid_argument(std::string("Invalid generation field: ")+name);return v;}
bool boolean(const Json& o,const char* name,bool def){if(!o.contains(name))return def;auto x=o.at(name);if(x.is_boolean())return x.get<bool>();if(x.is_string()){auto s=unicode_transform(x.get<std::string>(),1);if(s=="true"||s=="1"||s=="yes"||s=="on")return true;if(s=="false"||s=="0"||s=="no"||s=="off"||s.empty())return false;}throw std::invalid_argument(std::string("Invalid boolean: ")+name);}
}
NativeRuntime::NativeRuntime(const std::string& model,const std::string& frontend):weights_(model),speech_(weights_,metal_),camp_(weights_),emotion_(weights_,metal_),gpt_(weights_,metal_),codec_(weights_,metal_),acoustic_(weights_,metal_),audio_(frontend),tokenizer_(frontend),text_(frontend),resources_(frontend) {
    std::ifstream in(frontend+"/frontend.json");Json cfg;in>>cfg;emotion_sizes_=cfg.at("emo_num").get<std::vector<uint32_t>>();
    if(emotion_sizes_.size()!=8)throw std::invalid_argument("Invalid emotion table partition");
}
std::vector<float> NativeRuntime::encode_emotion(const std::vector<float>& features,uint32_t frames) {
    auto key=sha256(features.data(),features.size()*sizeof(float));auto cached=emotion_cache_.find(key);
    if(cached!=emotion_cache_.end()){emotion_cache_hits_++;return cached->second;}
    auto value=emotion_.encode(features,frames);if(emotion_cache_.size()>=16)emotion_cache_.erase(emotion_cache_.begin());emotion_cache_[key]=value;return value;
}
uint64_t NativeRuntime::submissions() const{return metal_.command_buffers_submitted()+speech_.extra_gpu_submissions()+camp_.gpu_submissions()+codec_.extra_gpu_submissions()+(qwen_?qwen_->submissions():0);}
NativeVoice NativeRuntime::clone(const std::string& path) {
    mit2::AutoreleasePool pool;auto a=audio_.prepare(path);auto e=audio_.prepare_emotion(path);NativeVoice v;
    v.speaker=speech_.encode(a.speech,a.mask,a.speech_frames);v.speaker_frames=a.speech_frames;
    v.emotion=speech_.encode(e.speech,e.mask,e.speech_frames);v.emotion_frames=e.speech_frames;
    v.style=camp_.encode(a.camp,a.camp_frames);v.prompt=acoustic_.regulate(v.speaker,v.speaker_frames,a.mel_frames);v.mel_frames=a.mel_frames;v.mel=std::move(a.mel);v.audio=std::move(a.audio);return v;
}
Json NativeRuntime::emotion_text(const std::string& text) {
    if(text.empty()||text.size()>16000||std::count_if(text.begin(),text.end(),[](unsigned char c){return (c&0xc0)!=0x80;})>4000)throw std::invalid_argument("Invalid emotion text length");
    auto cached=emotion_text_cache_.find(text);if(cached!=emotion_text_cache_.end())return cached->second;
    if(!qwen_tokenizer_)qwen_tokenizer_=std::make_unique<ByteTokenizer>(resources_,true);if(!qwen_)qwen_=std::make_unique<QwenDecoder>(weights_);
    // Exact chat template from the bundled 2.5 Qwen tokenizer (not Qwen3 defaults).
    const auto input="System: 文本情感分类<|endoftext|>\nHuman: "+text+"<|endoftext|>\nAssistant:";
    auto ids=qwen_->generate(qwen_tokenizer_->encode(input));auto last=std::find(ids.rbegin(),ids.rend(),151668u);if(last!=ids.rend())ids.erase(ids.begin(),last.base());
    auto decoded=qwen_tokenizer_->decode(ids);Json content=Json::parse(decoded,nullptr,false);
    if(content.is_discarded()){
        content=Json::object();const std::regex pair(R"rx(([^\s":.,]+?)"?\s*:\s*([\d.]+))rx");
        for(auto it=std::sregex_iterator(decoded.begin(),decoded.end(),pair);it!=std::sregex_iterator();++it){try{content[(*it)[1].str()]=std::stof((*it)[2].str());}catch(const std::exception&){}}
    }
    const std::vector<std::string> cn{"高兴","愤怒","悲伤","恐惧","反感","低落","惊讶","自然"},en{"happy","angry","sad","afraid","disgusted","melancholic","surprised","calm"};
    auto label=[&](const Json& v)->int{if(!v.is_string())return -1;auto s=unicode_transform(v.get<std::string>(),1);for(size_t i=0;i<cn.size();i++)if(s==cn[i]||s==en[i])return i;return -1;};
    std::vector<float> values(8,0);int detected=label(content);
    if(content.is_object()){for(const auto& alias:{"emotion","emotion_label","label","情感","情绪"})if(content.contains(alias)&&detected<0)detected=label(content[alias]);
        bool any_key=false;for(size_t i=0;i<8;i++)if(content.contains(cn[i])){any_key=true;auto& v=content[cn[i]];if(v.is_number())values[i]=std::clamp(v.get<float>(),0.f,1.2f);else {int other=label(v);if(other>=0)values[other]=1;}}
        if(any_key)detected=-1;
    }if(detected>=0)values[detected]=1;
    auto lower=unicode_transform(text,1);for(const auto& word:{"低落","melancholy","melancholic","depression","depressed","gloomy"})if(lower.find(word)!=std::string::npos){std::swap(values[2],values[5]);break;}
    if(std::all_of(values.begin(),values.end(),[](float v){return v<=0;}))values[7]=1;
    Json result=Json::object();for(size_t i=0;i<8;i++)result[en[i]]=values[i];Json classified{{"content",decoded},{"vector",values},{"emotions",result}};
    if(emotion_text_cache_.size()>=16)emotion_text_cache_.erase(emotion_text_cache_.begin());
    emotion_text_cache_.emplace(text,classified);return classified;
}
Json NativeRuntime::synthesize(const NativeVoice& voice,const std::string& raw,const std::string& output,const Json& options) {
    if(!options.is_object()||raw.empty()||raw.size()>40000||std::count_if(raw.begin(),raw.end(),[](unsigned char c){return (c&0xc0)!=0x80;})>10000)throw std::invalid_argument("Invalid speech input/options");
    mit2::AutoreleasePool pool;auto start=Clock::now();const auto before=submissions(),hits_before=emotion_cache_hits_;
    std::string lang=options.value("lang",options.value("language",std::string("ZH")));lang=unicode_transform(lang,1);
    if(lang!="zh"&&lang!="en"&&lang!="ja"&&lang!="es"&&lang!="ar")throw std::invalid_argument("lang must be ZH, EN, JA, ES, or AR");
    GenerationOptions generation;generation.do_sample=boolean(options,"do_sample",true);generation.temperature=number(options,"temperature",.8,.01,10);generation.top_k=number(options,"top_k",30,0,8194,true);generation.top_p=number(options,"top_p",.8,.001,1);generation.repetition_penalty=number(options,"repetition_penalty",10,.01,100);generation.num_beams=number(options,"num_beams",1,1,10,true);generation.length_penalty=number(options,"length_penalty",0,-10,10);
    if(options.contains("seed")){number(options,"seed",0,0,9223372036854775807.0,true);generation.seed=options.at("seed").get<uint64_t>();}else generation.seed=(uint64_t(std::random_device{}())<<32)|std::random_device{}();
    const auto maximum=number(options,"max_mel_tokens",1500,1,1500,true),text_max=number(options,"max_text_tokens_per_segment",120,4,600,true);const auto silence=number(options,"interval_silence",200,0,10000,true);
    float duration=number(options,"duration_factor",1,.25,4);if(options.contains("speed")&&!options.contains("duration_factor"))duration=1/number(options,"speed",1,.25,4);
    float alpha=number(options,"emo_alpha",1,0,1);
    std::vector<float> vector;bool use_text=boolean(options,"use_emo_text",options.contains("emo_text")&&!options.at("emo_text").get<std::string>().empty());
    if(use_text)vector=emotion_text(options.value("emo_text",raw)).at("vector").get<std::vector<float>>();
    else if(options.contains("emo_vector")){if(!options.at("emo_vector").is_array()||options.at("emo_vector").size()!=8)throw std::invalid_argument("emo_vector requires eight values");for(const auto& x:options.at("emo_vector")){if(!x.is_number())throw std::invalid_argument("Invalid emotion vector");float v=x.get<float>();if(!std::isfinite(v)||v<0||v>1.2)throw std::invalid_argument("Invalid emotion vector");vector.push_back(v);}if(std::accumulate(vector.begin(),vector.end(),0.f)>.8000001f)throw std::invalid_argument("Emotion vector total must be <=0.8");}
    if(!vector.empty()&&alpha!=1)for(auto& v:vector)v=std::trunc(v*alpha*10000)/10000;
    std::vector<float> emovec;double audio_dsp_time=0,audio_encoder_time=0;
    if(vector.empty()&&options.contains("emo_audio_prompt")&&!options.at("emo_audio_prompt").get<std::string>().empty()){auto base=encode_emotion(voice.speaker,voice.speaker_frames);auto dsp_start=Clock::now();auto a=audio_.prepare_emotion(options.at("emo_audio_prompt"));audio_dsp_time=elapsed(dsp_start);auto encoder_start=Clock::now();auto f=speech_.encode(a.speech,a.mask,a.speech_frames);audio_encoder_time=elapsed(encoder_start);auto other=encode_emotion(f,a.speech_frames);emovec=base;for(size_t i=0;i<emovec.size();i++)emovec[i]+=alpha*(other[i]-base[i]);}
    else emovec=encode_emotion(voice.emotion,voice.emotion_frames);
    if(!vector.empty()){
        const auto& sizes=emotion_sizes_;auto& speakers=weights_.get("emotion.speaker");auto& emos=weights_.get("emotion.matrix");std::mt19937_64 random(generation.seed);size_t offset=0;std::vector<float> mix(emovec.size(),0);
        for(size_t e=0;e<8;e++){uint32_t chosen=0;if(boolean(options,"use_random",false))chosen=std::uniform_int_distribution<uint32_t>(0,sizes[e]-1)(random);else {double best=-std::numeric_limits<double>::infinity();for(uint32_t n=0;n<sizes[e];n++){double dot=0,norm=0;for(size_t j=0;j<192;j++){double v=speakers[(offset+n)*192+j];dot+=v*voice.style[j];norm+=v*v;}double score=dot/std::max(std::sqrt(norm),1e-8);if(score>best){best=score;chosen=n;}}}for(size_t j=0;j<mix.size();j++)mix[j]+=vector[e]*emos[(offset+chosen)*mix.size()+j];offset+=sizes[e];}
        float total=std::accumulate(vector.begin(),vector.end(),0.f);for(size_t j=0;j<emovec.size();j++)emovec[j]=mix[j]+(1-total)*emovec[j];
    }
    const double emotion_time=elapsed(start);auto text_start=Clock::now();
    auto processed=text_.process(raw,lang,boolean(options,"text_normalization",true));if(processed.empty())throw std::invalid_argument("Text normalization produced empty input");auto prefix="<|"+lang+"|> ";auto segments=text_.split(processed,prefix,tokenizer_,text_max);
    const double text_time=elapsed(text_start);
    const double features_time=elapsed(start);std::mt19937_64 rng(generation.seed);std::normal_distribution<float> normal(0,1);std::vector<float> wave;Json segment_results=Json::array();double gpt_time=0,acoustic_time=0,vocoder_time=0,codec_time=0,regulate_time=0,flow_time=0;
    for(size_t index=0;index<segments.size();index++){
        auto ids=tokenizer_.encode(prefix+segments[index]);ids.push_back(1);auto t=Clock::now();auto prepared=gpt_.prepare(voice.style,emovec,ids,tokenizer_.language(lang));auto opts=generation;opts.seed+=index;auto generated=gpt_.generate(prepared,opts,maximum);gpt_time+=elapsed(t);
        if(generated.codes.empty())throw std::runtime_error("GPT produced no semantic codes");t=Clock::now();auto semantic=codec_.decode(generated.codes);codec_time+=elapsed(t);auto regulate_start=Clock::now();auto target=static_cast<uint32_t>(double(semantic.size()/1024)*1.72*duration);if(!target||target+voice.mel_frames>4096)throw std::invalid_argument("Acoustic token capacity exceeded");auto condition=voice.prompt;auto cond=acoustic_.regulate(semantic,semantic.size()/1024,target);condition.insert(condition.end(),cond.begin(),cond.end());std::vector<float> noise((target+voice.mel_frames)*80);for(auto& x:noise)x=normal(rng);
        regulate_time+=elapsed(regulate_start);auto flow_start=Clock::now();auto mel=acoustic_.flow(std::move(noise),voice.mel,condition,voice.style,target+voice.mel_frames,25,.7f);flow_time+=elapsed(flow_start);mel.erase(mel.begin(),mel.begin()+voice.mel_frames*80);acoustic_time+=elapsed(t);t=Clock::now();auto w=acoustic_.vocode(mel,target);vocoder_time+=elapsed(t);
        if(index)wave.insert(wave.end(),static_cast<size_t>(22050*silence/1000),0);for(auto x:w)wave.push_back(std::clamp(x,-1.f,1.f));segment_results.push_back(Json{{"text",segments[index]},{"text_ids",ids},{"codes",generated.codes},{"stopped",generated.stopped},{"mel_frames",target}});
    }
    size_t fade=std::min(size_t(441),wave.size());if(fade>1)for(size_t i=0;i<fade;i++)wave[wave.size()-fade+i]*=.5*(1+std::cos(3.14159265358979323846*i/(fade-1)));
    save_wave(output,wave);double seconds=elapsed(start),audio_seconds=wave.size()/22050.0;
    return Json{{"status","ok"},{"backend","native-metal"},{"model","2.5"},{"output",output},{"audio_seconds",audio_seconds},{"total_seconds",seconds},{"rtf",seconds/audio_seconds},{"gpu_submissions",submissions()-before},{"emotion_cache_hits",emotion_cache_hits_-hits_before},{"segments",segment_results},{"stage_seconds",{{"features",features_time},{"gpt",gpt_time},{"acoustic",acoustic_time},{"vocoder",vocoder_time},{"emotion",emotion_time},{"audio_dsp",audio_dsp_time},{"audio_encoder",audio_encoder_time},{"text",text_time},{"codec",codec_time},{"regulate",regulate_time},{"flow",flow_time}}},{"seed",generation.seed}};
}
}
