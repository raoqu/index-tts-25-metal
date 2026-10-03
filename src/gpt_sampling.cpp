// Sampling processors adapted from reference commit eda855b2e9d7cfaa4269380f42880d41b060c4d2.
#include "itts25/gpt.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <memory>
#include <stdexcept>
namespace itts25 {
struct SplitMix64 {
    uint64_t state;

    explicit SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next_u64() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    double next_unit() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
};

std::vector<float> apply_gpt_sampling_processors(const std::vector<float>& logits,
                                                 const std::vector<uint32_t>& history,
                                                 const GenerationOptions& config) {
    std::vector<float> scores = logits;
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    if (config.repetition_penalty != 1.0f) {
        if (config.repetition_penalty <= 0.0f) {
            throw std::runtime_error("GPT repetition penalty must be positive");
        }
        std::vector<uint8_t> seen(scores.size(), 0);
        for (uint32_t token : history) {
            if (token >= scores.size() || seen[token]) {
                continue;
            }
            seen[token] = 1;
            if (scores[token] < 0.0f) {
                scores[token] *= config.repetition_penalty;
            } else {
                scores[token] /= config.repetition_penalty;
            }
        }
    }
    if (config.temperature <= 0.0f) {
        throw std::runtime_error("GPT sampling temperature must be positive");
    }
    if (config.temperature != 1.0f) {
        for (float& score : scores) {
            score /= config.temperature;
        }
    }
    if (config.top_k > 0 && config.top_k < scores.size()) {
        std::vector<float> sorted = scores;
        const size_t kth = static_cast<size_t>(std::max(config.top_k, config.num_beams > 1 ? 2u : 1u) - 1);
        std::nth_element(sorted.begin(), sorted.begin() + kth, sorted.end(), std::greater<float>());
        const float pivot = sorted[kth];
        for (float& score : scores) {
            if (score < pivot) {
                score = neg_inf;
            }
        }
    }
    if (config.top_p < 1.0f) {
        if (config.top_p <= 0.0f) {
            throw std::runtime_error("GPT top_p must be in (0, 1]");
        }
        std::vector<size_t> order(scores.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (scores[a] == scores[b]) {
                return a < b;
            }
            return scores[a] < scores[b];
        });
        float max_score = neg_inf;
        for (float score : scores) {
            if (std::isfinite(score)) {
                max_score = std::max(max_score, score);
            }
        }
        if (!std::isfinite(max_score)) {
            throw std::runtime_error("GPT top_p filtering removed all logits");
        }
        double total = 0.0;
        std::vector<double> probs(scores.size(), 0.0);
        for (size_t i = 0; i < scores.size(); ++i) {
            if (std::isfinite(scores[i])) {
                probs[i] = std::exp(static_cast<double>(scores[i] - max_score));
                total += probs[i];
            }
        }
        if (total <= 0.0) {
            throw std::runtime_error("GPT sampling probability mass is zero");
        }
        double cumulative = 0.0;
        std::vector<uint8_t> remove(scores.size(), 0);
        const double low_tail_threshold = 1.0 - static_cast<double>(config.top_p);
        size_t finite_seen = 0;
        for (size_t idx : order) {
            if (!std::isfinite(scores[idx])) {
                remove[idx] = 1;
                continue;
            }
            ++finite_seen;
            cumulative += probs[idx] / total;
            if (cumulative <= low_tail_threshold) {
                remove[idx] = 1;
            }
        }
        if (finite_seen > 0) {
            for(size_t i=0;i<std::min(order.size(), config.num_beams>1?size_t(2):size_t(1));i++) remove[order[order.size()-1-i]] = 0;
        }
        for (size_t i = 0; i < scores.size(); ++i) {
            if (remove[i]) {
                scores[i] = neg_inf;
            }
        }
    }
    return scores;
}

uint32_t sample_gpt_token_from_logits(const std::vector<float>& processed_logits, SplitMix64& rng) {
    float max_score = -std::numeric_limits<float>::infinity();
    for (float score : processed_logits) {
        if (std::isfinite(score)) {
            max_score = std::max(max_score, score);
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("GPT sampling received no finite logits");
    }
    double total = 0.0;
    std::vector<double> probs(processed_logits.size(), 0.0);
    for (size_t i = 0; i < processed_logits.size(); ++i) {
        if (std::isfinite(processed_logits[i])) {
            probs[i] = std::exp(static_cast<double>(processed_logits[i] - max_score));
            total += probs[i];
        }
    }
    if (total <= 0.0) {
        throw std::runtime_error("GPT sampling probability mass is zero");
    }
    const double target = rng.next_unit() * total;
    double cumulative = 0.0;
    uint32_t last_finite = 0;
    for (uint32_t i = 0; i < probs.size(); ++i) {
        if (probs[i] <= 0.0) {
            continue;
        }
        last_finite = i;
        cumulative += probs[i];
        if (target < cumulative) {
            return i;
        }
    }
    return last_finite;
}


GenerationResult GptDecoder::generate(const std::vector<float>& prefix,const GenerationOptions& options,uint32_t maximum) {
    if(!maximum || maximum+1>=weights_.info("gpt.mel_pos_embedding.emb.weight").shape[0] || prefix.size()/1280+maximum>=4096)throw std::invalid_argument("GPT generation capacity exceeded");
    auto config=options;
    if(!config.do_sample){config.temperature=1;config.top_k=0;config.top_p=1;}
    if(!std::isfinite(config.temperature) || !std::isfinite(config.top_p) || !std::isfinite(config.repetition_penalty) || config.temperature<=0 || config.top_p<=0 || config.top_p>1 || config.repetition_penalty<=0)throw std::invalid_argument("Invalid GPT generation parameters");
    if(!config.num_beams || config.num_beams>10 || !std::isfinite(config.length_penalty))throw std::invalid_argument("Invalid beam search configuration");
    if(config.num_beams>1)return generate_beams(prefix,config,maximum);
    SplitMix64 random(config.seed);GenerationResult result;
    std::vector<uint32_t> history{1,8192};
    auto logits=prefill(prefix);
    for(uint32_t i=0;i<maximum;i++) {
        auto scores=apply_gpt_sampling_processors(logits,history,config);
        uint32_t token=config.do_sample?sample_gpt_token_from_logits(scores,random):std::max_element(scores.begin(),scores.end())-scores.begin();
        if(token==8193){result.stopped=true;break;}
        if(token>=8192)throw std::runtime_error("GPT generated invalid semantic code");
        result.codes.push_back(token);history.push_back(token);
        // Official GPT2InferenceModel cached generation uses attention_mask length
        // minus prefix length: first generated code has mel position 2, not 1.
        if(i+1<maximum)logits=step(token,i+2);
    }
    return result;
}
GenerationResult GptDecoder::generate_beams(const std::vector<float>& prefix,const GenerationOptions& config,uint32_t maximum) {
    struct SnapshotScope { mit2::MetalContext& metal; ~SnapshotScope(){metal.gptKvSnapshotsClear();} } snapshots{metal_};
    struct Beam {std::vector<uint32_t> codes;float score=0;std::vector<float> logits;std::shared_ptr<GptState> state;};
    struct Finished {std::vector<uint32_t> codes;double score;bool eos;};
    struct Candidate {uint32_t parent,token;float score;};
    SplitMix64 random(config.seed);std::vector<Finished> finished;
    Beam initial;initial.logits=prefill(prefix);initial.state=std::make_shared<GptState>(checkpoint());
    std::vector<Beam> beams(config.num_beams,initial);
    for(uint32_t i=1;i<beams.size();i++)beams[i].score=-1e9f;
    auto finish=[&](const Beam& beam,float score,uint32_t length,bool eos) {
        finished.push_back({beam.codes,score/std::pow(static_cast<double>(std::max(length,1u)),config.length_penalty),eos});
        std::sort(finished.begin(),finished.end(),[](const auto& a,const auto& b){return a.score>b.score;});
        if(finished.size()>config.num_beams)finished.resize(config.num_beams);
    };
    for(uint32_t step_index=0;step_index<maximum;step_index++) {
        std::vector<Candidate> candidates;std::vector<float> all_scores;
        for(uint32_t parent=0;parent<beams.size();parent++) {
            auto log_probs=beams[parent].logits;
            float peak=*std::max_element(log_probs.begin(),log_probs.end()),sum=0;
            for(auto value:log_probs)sum+=std::exp(value-peak);
            float log_sum=peak+std::log(sum);for(auto& value:log_probs)value-=log_sum;
            std::vector<uint32_t> history{1,8192};history.insert(history.end(),beams[parent].codes.begin(),beams[parent].codes.end());
            auto scores=apply_gpt_sampling_processors(log_probs,history,config);
            for(auto& value:scores)value+=beams[parent].score;
            all_scores.insert(all_scores.end(),scores.begin(),scores.end());
        }
        const uint32_t take=2*config.num_beams;
        if(config.do_sample) {
            auto available=all_scores;
            for(uint32_t i=0;i<take;i++) {
                uint32_t id=sample_gpt_token_from_logits(available,random);
                candidates.push_back({id/8194,id%8194,all_scores[id]});available[id]=-std::numeric_limits<float>::infinity();
            }
        } else {
            std::vector<uint32_t> order(all_scores.size());std::iota(order.begin(),order.end(),0);
            std::partial_sort(order.begin(),order.begin()+take,order.end(),[&](auto a,auto b){return all_scores[a]==all_scores[b]?a<b:all_scores[a]>all_scores[b];});
            for(uint32_t i=0;i<take;i++){auto id=order[i];candidates.push_back({id/8194,id%8194,all_scores[id]});}
        }
        std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.score>b.score;});
        std::vector<Beam> next;
        for(uint32_t rank=0;rank<candidates.size();rank++) {
            auto candidate=candidates[rank];const auto& parent=beams[candidate.parent];
            if(candidate.token==8193){if(rank<config.num_beams)finish(parent,candidate.score,step_index+1,true);continue;}
            if(candidate.token>=8192)throw std::runtime_error("Beam generated invalid semantic code");
            Beam child;child.codes=parent.codes;child.codes.push_back(candidate.token);child.score=candidate.score;
            if(step_index+1<maximum){restore(*parent.state);child.logits=step(candidate.token,step_index+2);child.state=std::make_shared<GptState>(checkpoint());}
            next.push_back(std::move(child));if(next.size()==config.num_beams)break;
        }
        if(next.empty())break;
        beams=std::move(next);
        if(finished.size()==config.num_beams && finished.back().score>=candidates[0].score/std::pow(static_cast<double>(step_index+1),config.length_penalty)) {beams.clear();break;}
    }
    for(const auto& beam:beams)finish(beam,beam.score,beam.codes.size(),false);
    if(finished.empty())throw std::runtime_error("Beam search produced no hypotheses");
    return {std::move(finished[0].codes),finished[0].eos};
}
}
