#pragma once
#include "itts25/weights.hpp"
#include "itts25/gpt.hpp"
#include "itts25/codec.hpp"
#include "itts25/acoustic.hpp"
#include "itts25/emotion.hpp"
#include "itts25/speech_features.hpp"
#include "itts25/campplus.hpp"
namespace itts25 {
struct SynthesisResult {
    std::vector<float> wave;std::vector<uint32_t> codes;bool stopped=false;
    double features_seconds=0,gpt_seconds=0,acoustic_seconds=0,vocoder_seconds=0;
    uint64_t gpu_submissions=0;
};
class Synthesizer {
public:
    explicit Synthesizer(const std::string& model);
    SynthesisResult synthesize(Weights& input,const GenerationOptions& options,uint32_t max_tokens=1500,float duration_factor=1.0f);
private:
    Weights weights_;mit2::MetalContext metal_;SpeechFeatures features_;CampPlus camp_;
    EmotionEncoder emotion_;GptDecoder gpt_;EnhancedCodec codec_;AcousticModel acoustic_;
    uint64_t submissions() const;
};
void save_wave(const std::string& path,const std::vector<float>& wave,uint32_t rate=22050);
}
