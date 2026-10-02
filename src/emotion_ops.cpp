// Adapted GPU emotion operators from index-tts2-metal, commit eda855b2e9d7cfaa4269380f42880d41b060c4d2.
// This module loads the actual 2.5 emotion Conformer/Perceiver tensors, independently of the 2.0 pipeline.
#include "itts25/emotion.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace itts25::emotion_detail {
static std::vector<float> tensor_as_f32(const mit2::Bundle& bundle,const std::string& name) {
    const auto* tensor=bundle.find(name);
    if(!tensor || tensor->dtype!="f32") throw std::runtime_error("Missing FP32 emotion tensor: "+name);
    std::vector<float> values(tensor->nbytes/4);
    std::memcpy(values.data(),bundle.tensor_data(*tensor),tensor->nbytes);
    return values;
}
struct GptSubsamplingOutput {
    std::vector<float> subsampling;
    std::vector<float> pos_emb;
    std::vector<uint32_t> mask;
    uint32_t output_tokens = 0;
};
struct GptConformerFfOutput {
    std::vector<float> normed;
    std::vector<float> raw;
    std::vector<float> tail;
};
struct GptConformerAttnOutput {
    std::vector<float> normed;
    std::vector<float> attn;
    std::vector<float> residual;
};
struct GptConformerConvOutput {
    std::vector<float> normed;
    std::vector<float> raw;
    std::vector<float> residual;
};
std::vector<float> finish_gpt_emovec_from_subsampling_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& bundle,
                                                            const GptSubsamplingOutput& sub);
GptSubsamplingOutput run_gpt_conditioning_subsampling_metal_linear(mit2::MetalContext& metal,
                                                                   const mit2::Bundle& bundle,
                                                                   const std::vector<float>& input,
                                                                   uint32_t input_tokens,
                                                                   const std::string& encoder_prefix);
GptConformerAttnOutput run_gpt_conformer_attention_metal_proj(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layer_index,
                                                              const std::string& encoder_prefix,
                                                              uint32_t heads);
std::vector<float> run_gpt_conformer_block_metal_attn_conv_ff(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layer_index,
                                                              const std::string& encoder_prefix,
                                                              uint32_t heads);
GptConformerConvOutput run_gpt_conformer_conv_metal(mit2::MetalContext& metal,
                                                    const mit2::Bundle& bundle,
                                                    const std::vector<float>& input,
                                                    const std::vector<uint32_t>& mask,
                                                    uint32_t tokens,
                                                    uint32_t layer_index,
                                                    const std::string& encoder_prefix);
GptConformerFfOutput run_gpt_conformer_ff_tail_metal(mit2::MetalContext& metal,
                                                     const mit2::Bundle& bundle,
                                                     const std::vector<float>& input,
                                                     uint32_t tokens,
                                                     uint32_t layer_index,
                                                     const std::string& encoder_prefix);
std::vector<float> run_gpt_conformer_stack_metal_attn_conv_ff(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layers,
                                                              const std::string& encoder_prefix,
                                                              uint32_t heads);
std::vector<float> run_gpt_emovec_metal_linear(mit2::MetalContext& metal,
                                               const mit2::Bundle& bundle,
                                               const std::vector<float>& spk_cond_emb,
                                               uint32_t input_tokens);
std::vector<float> run_gpt_perceiver_metal(mit2::MetalContext& metal,
                                           const mit2::Bundle& bundle,
                                           const std::vector<float>& context,
                                           const std::vector<uint32_t>& mask,
                                           uint32_t context_tokens,
                                           const std::string& perceiver_prefix,
                                           uint32_t context_dim,
                                           uint32_t dim,
                                           uint32_t latents,
                                           uint32_t heads);

std::vector<float> finish_gpt_emovec_from_subsampling_metal(mit2::MetalContext& metal,
                                                            const mit2::Bundle& bundle,
                                                            const GptSubsamplingOutput& sub) {
    constexpr uint32_t conformer_dim = 512;
    constexpr uint32_t emo_dim = 1024;
    constexpr uint32_t model_dim = 1280;
    auto context = run_gpt_conformer_stack_metal_attn_conv_ff(metal,
                                                              bundle,
                                                              sub.subsampling,
                                                              sub.pos_emb,
                                                              sub.mask,
                                                              sub.output_tokens,
                                                              4,
                                                              "gpt.emo_conditioning_encoder",
                                                              4);
    std::vector<uint32_t> perceiver_mask;
    perceiver_mask.reserve(static_cast<size_t>(sub.output_tokens) + 1);
    perceiver_mask.push_back(1);
    perceiver_mask.insert(perceiver_mask.end(), sub.mask.begin(), sub.mask.end());
    auto emo_perceiver = run_gpt_perceiver_metal(metal,
                                                 bundle,
                                                 context,
                                                 perceiver_mask,
                                                 sub.output_tokens,
                                                 "gpt.emo_perceiver_encoder",
                                                 conformer_dim,
                                                 emo_dim,
                                                 1,
                                                 4);
    auto emovec_w = tensor_as_f32(bundle, "gpt.emovec_layer.weight");
    auto emovec_b = tensor_as_f32(bundle, "gpt.emovec_layer.bias");
    auto emo_w = tensor_as_f32(bundle, "gpt.emo_layer.weight");
    auto emo_b = tensor_as_f32(bundle, "gpt.emo_layer.bias");
    auto projected = metal.linear_rows_f32_resident(
        "gpt.emovec_layer.weight.resident",
        emovec_w,
        "gpt.emovec_layer.bias.resident",
        emovec_b,
        emo_perceiver,
        1,
        model_dim,
        emo_dim);
    return metal.linear_rows_f32_resident(
        "gpt.emo_layer.weight.resident",
        emo_w,
        "gpt.emo_layer.bias.resident",
        emo_b,
        projected,
        1,
        model_dim,
        model_dim);
}

GptSubsamplingOutput run_gpt_conditioning_subsampling_metal_linear(mit2::MetalContext& metal,
                                                                   const mit2::Bundle& bundle,
                                                                   const std::vector<float>& input,
                                                                   uint32_t input_tokens,
                                                                   const std::string& encoder_prefix) {
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t channels = 512;
    constexpr uint32_t conv_kernel = 3;
    constexpr uint32_t conv_stride = 2;
    constexpr uint32_t conv_freq = 511;
    constexpr uint32_t flat_dim = channels * conv_freq;
    constexpr uint32_t output_dim = 512;
    if (input_tokens < conv_kernel) {
        throw std::runtime_error("GPT subsampling input_tokens must be at least 3");
    }
    if (input.size() != static_cast<size_t>(input_tokens) * input_dim) {
        throw std::runtime_error("GPT subsampling input must have shape [tokens,1024]");
    }
    const uint32_t output_tokens = ((input_tokens - conv_kernel) / conv_stride) + 1;

    auto conv_w = tensor_as_f32(bundle, encoder_prefix + ".embed.conv.0.weight");
    auto conv_b = tensor_as_f32(bundle, encoder_prefix + ".embed.conv.0.bias");
    auto linear_w = tensor_as_f32(bundle, encoder_prefix + ".embed.out.0.weight");
    auto linear_b = tensor_as_f32(bundle, encoder_prefix + ".embed.out.0.bias");
    auto pe = tensor_as_f32(bundle, encoder_prefix + ".embed.pos_enc.pe");
    if (conv_w.size() != static_cast<size_t>(channels) * conv_kernel * conv_kernel || conv_b.size() != channels) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.conv.0 shape mismatch");
    }
    if (linear_w.size() != static_cast<size_t>(output_dim) * flat_dim || linear_b.size() != output_dim) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.out.0 shape mismatch");
    }
    if (pe.size() < static_cast<size_t>(output_tokens) * output_dim) {
        throw std::runtime_error("gpt.conditioning_encoder.embed.pos_enc.pe shape mismatch");
    }

    auto flat = metal.subsampling_conv2d_relu_flat_f32_resident(
        encoder_prefix + ".embed.conv.0.weight.resident",
        conv_w,
        encoder_prefix + ".embed.conv.0.bias.resident",
        conv_b,
        input,
        input_tokens,
        input_dim,
        channels,
        conv_kernel,
        conv_stride);

    auto subsampling = metal.linear_rows_f32_resident(
        encoder_prefix + ".embed.out.0.weight.resident",
        linear_w,
        encoder_prefix + ".embed.out.0.bias.resident",
        linear_b,
        flat,
        output_tokens,
        output_dim,
        flat_dim);
    const float positional_scale = std::sqrt(static_cast<float>(output_dim));
    for (float& v : subsampling) {
        v *= positional_scale;
    }
    std::vector<float> pos_emb(static_cast<size_t>(output_tokens) * output_dim);
    std::copy(pe.begin(), pe.begin() + pos_emb.size(), pos_emb.begin());
    std::vector<uint32_t> mask(output_tokens, 1);
    return GptSubsamplingOutput{std::move(subsampling), std::move(pos_emb), std::move(mask), output_tokens};
}

GptConformerAttnOutput run_gpt_conformer_attention_metal_proj(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layer_index,
                                                              const std::string& encoder_prefix,
                                                              uint32_t heads) {
    constexpr uint32_t dim = 512;
    const uint32_t head_dim = dim / heads;
    if (heads == 0 || dim % heads != 0) {
        throw std::runtime_error("GPT conformer attention heads must divide dim");
    }
    if (input.size() != static_cast<size_t>(tokens) * dim ||
        pos_emb.size() != static_cast<size_t>(tokens) * dim ||
        mask.size() != tokens) {
        throw std::runtime_error("GPT conformer attention input shape mismatch");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    auto norm_w = tensor_as_f32(bundle, base + ".norm_mha.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_mha.bias");
    auto q_w = tensor_as_f32(bundle, base + ".self_attn.linear_q.weight");
    auto q_b = tensor_as_f32(bundle, base + ".self_attn.linear_q.bias");
    auto k_w = tensor_as_f32(bundle, base + ".self_attn.linear_k.weight");
    auto k_b = tensor_as_f32(bundle, base + ".self_attn.linear_k.bias");
    auto v_w = tensor_as_f32(bundle, base + ".self_attn.linear_v.weight");
    auto v_b = tensor_as_f32(bundle, base + ".self_attn.linear_v.bias");
    auto pos_w = tensor_as_f32(bundle, base + ".self_attn.linear_pos.weight");
    auto out_w = tensor_as_f32(bundle, base + ".self_attn.linear_out.weight");
    auto out_b = tensor_as_f32(bundle, base + ".self_attn.linear_out.bias");
    auto bias_u = tensor_as_f32(bundle, base + ".self_attn.pos_bias_u");
    auto bias_v = tensor_as_f32(bundle, base + ".self_attn.pos_bias_v");
    if (bias_u.size() != static_cast<size_t>(heads) * head_dim ||
        bias_v.size() != static_cast<size_t>(heads) * head_dim) {
        throw std::runtime_error("GPT conformer attention bias shape mismatch");
    }

    auto normed = metal.layernorm_rows_f32_resident(
        base + ".norm_mha.weight.resident",
        norm_w,
        base + ".norm_mha.bias.resident",
        norm_b,
        input,
        tokens,
        dim,
        1e-5f);
    auto q_linear = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_q.weight.resident",
        q_w,
        base + ".self_attn.linear_q.bias.resident",
        q_b,
        normed,
        tokens,
        dim,
        dim);
    auto k_linear = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_k.weight.resident",
        k_w,
        base + ".self_attn.linear_k.bias.resident",
        k_b,
        normed,
        tokens,
        dim,
        dim);
    auto v_linear = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_v.weight.resident",
        v_w,
        base + ".self_attn.linear_v.bias.resident",
        v_b,
        normed,
        tokens,
        dim,
        dim);
    std::vector<float> zero_dim_bias(dim, 0.0f);
    auto p_linear = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_pos.weight.resident",
        pos_w,
        base + ".self_attn.linear_pos.zero_bias.resident",
        zero_dim_bias,
        pos_emb,
        tokens,
        dim,
        dim);

    auto attn_context = metal.conformer_rel_attention_context_f32_resident(
        base + ".self_attn.pos_bias_u.resident",
        bias_u,
        base + ".self_attn.pos_bias_v.resident",
        bias_v,
        q_linear,
        k_linear,
        v_linear,
        p_linear,
        mask,
        tokens,
        heads,
        head_dim);
    auto attn = metal.linear_rows_f32_resident(
        base + ".self_attn.linear_out.weight.resident",
        out_w,
        base + ".self_attn.linear_out.bias.resident",
        out_b,
        attn_context,
        tokens,
        dim,
        dim);
    auto residual = metal.add_f32(input, attn);
    return GptConformerAttnOutput{std::move(normed), std::move(attn), std::move(residual)};
}

std::vector<float> run_gpt_conformer_block_metal_attn_conv_ff(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layer_index,
                                                              const std::string& encoder_prefix,
                                                              uint32_t heads) {
    auto attn = run_gpt_conformer_attention_metal_proj(metal, bundle, input, pos_emb, mask, tokens, layer_index, encoder_prefix, heads);
    auto conv = run_gpt_conformer_conv_metal(metal, bundle, attn.residual, mask, tokens, layer_index, encoder_prefix);
    auto ff = run_gpt_conformer_ff_tail_metal(metal, bundle, conv.residual, tokens, layer_index, encoder_prefix);
    return ff.tail;
}

GptConformerConvOutput run_gpt_conformer_conv_metal(mit2::MetalContext& metal,
                                                    const mit2::Bundle& bundle,
                                                    const std::vector<float>& input,
                                                    const std::vector<uint32_t>& mask,
                                                    uint32_t tokens,
                                                    uint32_t layer_index,
                                                    const std::string& encoder_prefix) {
    constexpr uint32_t dim = 512;
    constexpr uint32_t doubled = 1024;
    constexpr uint32_t kernel = 15;
    if (input.size() != static_cast<size_t>(tokens) * dim || mask.size() != tokens) {
        throw std::runtime_error("GPT conformer conv input shape mismatch");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    const std::string conv = base + ".conv_module";
    auto norm_w = tensor_as_f32(bundle, base + ".norm_conv.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_conv.bias");
    auto pw1_w = tensor_as_f32(bundle, conv + ".pointwise_conv1.weight");
    auto pw1_b = tensor_as_f32(bundle, conv + ".pointwise_conv1.bias");
    auto depth_w = tensor_as_f32(bundle, conv + ".depthwise_conv.weight");
    auto depth_b = tensor_as_f32(bundle, conv + ".depthwise_conv.bias");
    auto conv_norm_w = tensor_as_f32(bundle, conv + ".norm.weight");
    auto conv_norm_b = tensor_as_f32(bundle, conv + ".norm.bias");
    auto pw2_w = tensor_as_f32(bundle, conv + ".pointwise_conv2.weight");
    auto pw2_b = tensor_as_f32(bundle, conv + ".pointwise_conv2.bias");
    if (pw1_w.size() != static_cast<size_t>(doubled) * dim ||
        pw2_w.size() != static_cast<size_t>(dim) * dim ||
        depth_w.size() != static_cast<size_t>(dim) * kernel ||
        depth_b.size() != dim) {
        throw std::runtime_error("GPT conformer conv weight shape mismatch");
    }

    auto normed = metal.layernorm_rows_f32_resident(
        base + ".norm_conv.weight.resident",
        norm_w,
        base + ".norm_conv.bias.resident",
        norm_b,
        input,
        tokens,
        dim,
        1e-5f);
    auto masked_normed = metal.mask_rows_f32(normed, mask, tokens, dim);
    auto pw1 = metal.linear_rows_f32_resident(
        conv + ".pointwise_conv1.weight.resident",
        pw1_w,
        conv + ".pointwise_conv1.bias.resident",
        pw1_b,
        masked_normed,
        tokens,
        doubled,
        dim);
    auto glu = metal.glu_split_f32(pw1, tokens, dim);
    auto depth = metal.depthwise_conv1d_same_f32_resident(
        conv + ".depthwise_conv.weight.resident",
        depth_w,
        conv + ".depthwise_conv.bias.resident",
        depth_b,
        glu,
        tokens,
        dim,
        kernel);
    auto conv_normed = metal.layernorm_rows_f32_resident(
        conv + ".norm.weight.resident",
        conv_norm_w,
        conv + ".norm.bias.resident",
        conv_norm_b,
        depth,
        tokens,
        dim,
        1e-5f);
    auto conv_act = metal.silu_f32(conv_normed);
    auto raw_unmasked = metal.linear_rows_f32_resident(
        conv + ".pointwise_conv2.weight.resident",
        pw2_w,
        conv + ".pointwise_conv2.bias.resident",
        pw2_b,
        conv_act,
        tokens,
        dim,
        dim);
    auto raw = metal.mask_rows_f32(raw_unmasked, mask, tokens, dim);
    auto residual = metal.add_f32(input, raw);
    return GptConformerConvOutput{std::move(normed), std::move(raw), std::move(residual)};
}

GptConformerFfOutput run_gpt_conformer_ff_tail_metal(mit2::MetalContext& metal,
                                                     const mit2::Bundle& bundle,
                                                     const std::vector<float>& input,
                                                     uint32_t tokens,
                                                     uint32_t layer_index,
                                                     const std::string& encoder_prefix) {
    constexpr uint32_t dim = 512;
    if (input.size() != static_cast<size_t>(tokens) * dim) {
        throw std::runtime_error("GPT conformer FF input must have shape [tokens,512]");
    }
    const std::string base = encoder_prefix + ".encoders." + std::to_string(layer_index);
    auto norm_w = tensor_as_f32(bundle, base + ".norm_ff.weight");
    auto norm_b = tensor_as_f32(bundle, base + ".norm_ff.bias");
    auto w1 = tensor_as_f32(bundle, base + ".feed_forward.w_1.weight");
    auto b1 = tensor_as_f32(bundle, base + ".feed_forward.w_1.bias");
    auto w2 = tensor_as_f32(bundle, base + ".feed_forward.w_2.weight");
    auto b2 = tensor_as_f32(bundle, base + ".feed_forward.w_2.bias");
    auto final_norm_w = tensor_as_f32(bundle, base + ".norm_final.weight");
    auto final_norm_b = tensor_as_f32(bundle, base + ".norm_final.bias");
    const uint32_t hidden = static_cast<uint32_t>(b1.size());
    if (hidden == 0 ||
        w1.size() != static_cast<size_t>(hidden) * dim ||
        w2.size() != static_cast<size_t>(dim) * hidden ||
        b2.size() != dim) {
        throw std::runtime_error("GPT conformer FF weight shape mismatch");
    }

    auto normed = metal.layernorm_rows_f32_resident(
        base + ".norm_ff.weight.resident",
        norm_w,
        base + ".norm_ff.bias.resident",
        norm_b,
        input,
        tokens,
        dim,
        1e-5f);
    auto hidden_pre = metal.linear_rows_f32_resident(
        base + ".feed_forward.w_1.weight.resident",
        w1,
        base + ".feed_forward.w_1.bias.resident",
        b1,
        normed,
        tokens,
        hidden,
        dim);
    auto hidden_act = metal.silu_f32(hidden_pre);
    auto raw = metal.linear_rows_f32_resident(
        base + ".feed_forward.w_2.weight.resident",
        w2,
        base + ".feed_forward.w_2.bias.resident",
        b2,
        hidden_act,
        tokens,
        dim,
        hidden);
    auto residual = metal.add_f32(input, raw);
    auto tail = metal.layernorm_rows_f32_resident(
        base + ".norm_final.weight.resident",
        final_norm_w,
        base + ".norm_final.bias.resident",
        final_norm_b,
        residual,
        tokens,
        dim,
        1e-5f);
    return GptConformerFfOutput{std::move(normed), std::move(raw), std::move(tail)};
}

std::vector<float> run_gpt_conformer_stack_metal_attn_conv_ff(mit2::MetalContext& metal,
                                                              const mit2::Bundle& bundle,
                                                              const std::vector<float>& input,
                                                              const std::vector<float>& pos_emb,
                                                              const std::vector<uint32_t>& mask,
                                                              uint32_t tokens,
                                                              uint32_t layers,
                                                              const std::string& encoder_prefix,
                                                              uint32_t heads) {
    constexpr uint32_t dim = 512;
    if (input.size() != static_cast<size_t>(tokens) * dim) {
        throw std::runtime_error("GPT conformer stack input must have shape [tokens,512]");
    }
    std::vector<float> state = input;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        state = run_gpt_conformer_block_metal_attn_conv_ff(metal, bundle, state, pos_emb, mask, tokens, layer, encoder_prefix, heads);
    }
    auto after_w = tensor_as_f32(bundle, encoder_prefix + ".after_norm.weight");
    auto after_b = tensor_as_f32(bundle, encoder_prefix + ".after_norm.bias");
    return metal.layernorm_rows_f32_resident(
        encoder_prefix + ".after_norm.weight.resident",
        after_w,
        encoder_prefix + ".after_norm.bias.resident",
        after_b,
        state,
        tokens,
        dim,
        1e-5f);
}

std::vector<float> run_gpt_emovec_metal_linear(mit2::MetalContext& metal,
                                               const mit2::Bundle& bundle,
                                               const std::vector<float>& spk_cond_emb,
                                               uint32_t input_tokens) {
    constexpr uint32_t input_dim = 1024;
    if (spk_cond_emb.size() != static_cast<size_t>(input_tokens) * input_dim) {
        throw std::runtime_error("GPT emovec input must have shape [tokens,1024]");
    }
    auto sub = run_gpt_conditioning_subsampling_metal_linear(metal, bundle, spk_cond_emb, input_tokens, "gpt.emo_conditioning_encoder");
    return finish_gpt_emovec_from_subsampling_metal(metal, bundle, sub);
}

std::vector<float> run_gpt_perceiver_metal(mit2::MetalContext& metal,
                                           const mit2::Bundle& bundle,
                                           const std::vector<float>& context,
                                           const std::vector<uint32_t>& mask,
                                           uint32_t context_tokens,
                                           const std::string& perceiver_prefix,
                                           uint32_t context_dim,
                                           uint32_t dim,
                                           uint32_t latents,
                                           uint32_t heads) {
    constexpr uint32_t head_dim = 64;
    const uint32_t inner = heads * head_dim;
    if (context.size() != static_cast<size_t>(context_tokens) * context_dim) {
        throw std::runtime_error("GPT perceiver context shape mismatch");
    }
    if (mask.size() != static_cast<size_t>(latents) + context_tokens) {
        throw std::runtime_error("GPT perceiver mask shape mismatch");
    }

    const std::vector<float> zero_inner_bias(inner, 0.0f);
    const std::vector<float> zero_kv_bias(inner * 2, 0.0f);
    const std::vector<float> zero_dim_bias(dim, 0.0f);

    auto proj_w = tensor_as_f32(bundle, perceiver_prefix + ".proj_context.weight");
    auto proj_b = tensor_as_f32(bundle, perceiver_prefix + ".proj_context.bias");
    auto projected_context = metal.linear_rows_f32_resident(
        perceiver_prefix + ".proj_context.weight.resident",
        proj_w,
        perceiver_prefix + ".proj_context.bias.resident",
        proj_b,
        context,
        context_tokens,
        dim,
        context_dim);

    auto latents_tensor = tensor_as_f32(bundle, perceiver_prefix + ".latents");
    if (latents_tensor.size() != static_cast<size_t>(latents) * dim) {
        throw std::runtime_error("gpt.perceiver_encoder.latents shape mismatch");
    }
    std::vector<float> latent_state = latents_tensor;

    for (uint32_t layer = 0; layer < 2; ++layer) {
        const std::string base = perceiver_prefix + ".layers." + std::to_string(layer);
        auto to_q = tensor_as_f32(bundle, base + ".0.to_q.weight");
        auto to_kv = tensor_as_f32(bundle, base + ".0.to_kv.weight");
        auto to_out = tensor_as_f32(bundle, base + ".0.to_out.weight");
        auto ff0_w = tensor_as_f32(bundle, base + ".1.0.weight");
        auto ff0_b = tensor_as_f32(bundle, base + ".1.0.bias");
        auto ff2_w = tensor_as_f32(bundle, base + ".1.2.weight");
        auto ff2_b = tensor_as_f32(bundle, base + ".1.2.bias");
        if ((ff0_b.size() % 2) != 0) {
            throw std::runtime_error("GPT perceiver GEGLU bias shape mismatch");
        }
        const uint32_t ff_inner = static_cast<uint32_t>(ff0_b.size() / 2);

        const uint32_t key_tokens = latents + context_tokens;
        std::vector<float> combined_context(static_cast<size_t>(key_tokens) * dim);
        std::copy(latent_state.begin(), latent_state.end(), combined_context.begin());
        std::copy(projected_context.begin(), projected_context.end(), combined_context.begin() + static_cast<size_t>(latents) * dim);

        auto q = metal.linear_rows_f32_resident(
            base + ".0.to_q.weight.resident",
            to_q,
            perceiver_prefix + ".zero_inner_bias.resident",
            zero_inner_bias,
            latent_state,
            latents,
            inner,
            dim);
        auto kv = metal.linear_rows_f32_resident(
            base + ".0.to_kv.weight.resident",
            to_kv,
            perceiver_prefix + ".zero_kv_bias.resident",
            zero_kv_bias,
            combined_context,
            key_tokens,
            inner * 2,
            dim);
        std::vector<float> k(static_cast<size_t>(key_tokens) * inner);
        std::vector<float> v(static_cast<size_t>(key_tokens) * inner);
        for (uint32_t t = 0; t < key_tokens; ++t) {
            std::copy(kv.begin() + static_cast<size_t>(t) * inner * 2,
                      kv.begin() + static_cast<size_t>(t) * inner * 2 + inner,
                      k.begin() + static_cast<size_t>(t) * inner);
            std::copy(kv.begin() + static_cast<size_t>(t) * inner * 2 + inner,
                      kv.begin() + static_cast<size_t>(t + 1) * inner * 2,
                      v.begin() + static_cast<size_t>(t) * inner);
        }

        auto attn = metal.cross_attention_heads_masked_f32(q, k, v, mask, latents, key_tokens, heads, head_dim);
        auto attn_projected = metal.linear_rows_f32_resident(
            base + ".0.to_out.weight.resident",
            to_out,
            perceiver_prefix + ".zero_dim_bias.resident",
            zero_dim_bias,
            attn,
            latents,
            dim,
            inner);
        latent_state = metal.add_f32(latent_state, attn_projected);

        auto ff0 = metal.linear_rows_f32_resident(
            base + ".1.0.weight.resident",
            ff0_w,
            base + ".1.0.bias.resident",
            ff0_b,
            latent_state,
            latents,
            ff_inner * 2,
            dim);
        auto ff_mid = metal.geglu_erf_split_f32(ff0, latents, ff_inner);
        auto ff_out = metal.linear_rows_f32_resident(
            base + ".1.2.weight.resident",
            ff2_w,
            base + ".1.2.bias.resident",
            ff2_b,
            ff_mid,
            latents,
            dim,
            ff_inner);
        latent_state = metal.add_f32(latent_state, ff_out);
    }

    auto gamma = tensor_as_f32(bundle, perceiver_prefix + ".norm.gamma");
    return metal.rmsnorm_rows_f32_resident(
        perceiver_prefix + ".norm.gamma.resident",
        gamma,
        latent_state,
        latents,
        dim);
}
}
