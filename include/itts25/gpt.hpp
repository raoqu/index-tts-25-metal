#pragma once
#include "itts25/weights.hpp"
#include "mit2/metal_context.hpp"

namespace itts25 {
struct GenerationOptions {
    bool do_sample=true;
    uint64_t seed=20261002;
    float temperature=0.8f;
    uint32_t top_k=30;
    float top_p=0.8f;
    float repetition_penalty=10.0f;
    uint32_t num_beams=1;
    float length_penalty=0;
};
struct GenerationResult {std::vector<uint32_t> codes;bool stopped=false;};
struct GptState {uint32_t tokens=0;std::vector<float> kv;};
std::vector<float> apply_gpt_sampling_processors(const std::vector<float>& logits,const std::vector<uint32_t>& history,const GenerationOptions& options);
// Batch-one GPT2 decoder with an entirely GPU-resident 24-layer KV cache.
// Speaker/emotion feature extraction is handled by separate native components.
class GptDecoder {
public:
    GptDecoder(Weights& weights, mit2::MetalContext& metal);
    std::vector<float> prepare(const std::vector<float>& speaker, const std::vector<float>& emotion,
                               const std::vector<uint32_t>& text, uint32_t language);
    std::vector<float> prefill(const std::vector<float>& prefix);
    std::vector<float> step(uint32_t code, uint32_t position);
    GenerationResult generate(const std::vector<float>& prefix,const GenerationOptions& options,uint32_t max_tokens=1500);
    GptState checkpoint() const;
    void restore(const GptState& state);
    uint32_t cached_tokens() const { return cached_tokens_; }
private:
    Weights& weights_;
    mit2::MetalContext& metal_;
    std::unordered_map<std::string,std::vector<float>> transposed_;
    uint32_t cached_tokens_ = 0;
    const std::vector<float>& conv_weight(const std::string& name);
    mit2::PassSlot linear(mit2::PassSlot x, uint32_t tokens, const std::string& name, bool transpose);
    mit2::PassSlot norm(mit2::PassSlot x, uint32_t tokens, const std::string& name);
    std::vector<float> run(const std::vector<float>& x, bool initial, uint32_t code, uint32_t position);
    GenerationResult generate_beams(const std::vector<float>& prefix,const GenerationOptions& options,uint32_t maximum);
};
}
