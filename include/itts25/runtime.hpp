#pragma once
#include "itts25/frontend.hpp"
#include "itts25/text_processor.hpp"
#include "itts25/pipeline.hpp"
#include "itts25/qwen.hpp"
namespace itts25 {
struct NativeVoice {
    std::vector<float> speaker,emotion,style,prompt,mel,audio;
    uint32_t speaker_frames=0,emotion_frames=0,mel_frames=0;
};
NativeVoice read_native_voice(const std::string& path);
void save_native_voice(const std::string& path,const NativeVoice& voice);
std::string sha256(const void* data,size_t size);
class NativeRuntime {
public:
    NativeRuntime(const std::string& model,const std::string& frontend);
    NativeVoice clone(const std::string& audio);
    Json synthesize(const NativeVoice& voice,const std::string& text,const std::string& output,const Json& options=Json::object());
    Json emotion_text(const std::string& text);
private:
    Weights weights_;mit2::MetalContext metal_;SpeechFeatures speech_;CampPlus camp_;
    EmotionEncoder emotion_;GptDecoder gpt_;EnhancedCodec codec_;AcousticModel acoustic_;
    AudioFrontend audio_;ByteTokenizer tokenizer_;TextProcessor text_;std::string resources_;
    std::unique_ptr<ByteTokenizer> qwen_tokenizer_;std::unique_ptr<QwenDecoder> qwen_;
    std::unordered_map<std::string,std::vector<float>> emotion_cache_;
    std::vector<uint32_t> emotion_sizes_;
    uint64_t emotion_cache_hits_=0;
    std::vector<float> encode_emotion(const std::vector<float>& features,uint32_t frames);
    uint64_t submissions() const;
};
}
