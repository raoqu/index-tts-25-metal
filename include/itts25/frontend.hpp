#pragma once
#include "itts25/weights.hpp"
#include "json.hpp"
#include <memory>
namespace itts25 {
using Json = nlohmann::json;
class ByteTokenizer {
public:
    ByteTokenizer(const std::string& path, bool qwen=false);
    ~ByteTokenizer();
    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& ids, bool skip_special=true) const;
    uint32_t language(const std::string& lang) const;
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
struct AudioFeatures {
    std::vector<float> audio, speech, camp, mel;
    std::vector<uint32_t> mask;
    uint32_t speech_frames=0, camp_frames=0, mel_frames=0;
};
class AudioFrontend {
public:
    explicit AudioFrontend(const std::string& resources);
    AudioFeatures prepare(const std::string& audio_path);
    AudioFeatures prepare_emotion(const std::string& audio_path);
    std::vector<float> load(const std::string& path, uint32_t rate);
private:
    Weights constants_;
    std::vector<float> resample16(const std::vector<float>& audio);
    std::vector<float> fbank(const std::vector<float>& audio, bool speech);
};
void write_input_bundle(const std::string& path,const AudioFeatures& audio,
                        const std::vector<uint32_t>& ids,uint32_t language);
}
