// Adapted GPU acoustic operators from index-tts2-metal, commit eda855b2e9d7cfaa4269380f42880d41b060c4d2.
// This module loads the actual 2.5 S2Mel and BigVGAN tensors, independently of the 2.0 pipeline.
#include "itts25/acoustic.hpp"
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
namespace itts25::acoustic_detail {
constexpr uint32_t kFusedDitAttentionMaxTokens=4096;
static bool cfm_skip(const char*) { return false; }
static std::vector<float> tensor_as_f32(const mit2::Bundle& bundle,const std::string& name) {
    const auto* tensor=bundle.find(name);
    if(!tensor || tensor->dtype!="f32") throw std::runtime_error("Missing FP32 acoustic tensor: "+name);
    std::vector<float> values(tensor->nbytes/4);
    std::memcpy(values.data(),bundle.tensor_data(*tensor),tensor->nbytes);
    return values;
}
struct DitPostTransformerCoreOutputs {
    std::vector<float> long_skip;
    std::vector<float> conv1;
    std::vector<float> res_projection;
};
struct BigVGANUpsampleSpec {
    const char* prefix;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel;
    uint32_t stride;
    uint32_t padding;
};

const BigVGANUpsampleSpec kBigVGANUpsamplers[] = {
    {"bigvgan.ups.0.0", 1536, 768, 8, 4, 2},
    {"bigvgan.ups.1.0", 768, 384, 8, 4, 2},
    {"bigvgan.ups.2.0", 384, 192, 4, 2, 1},
    {"bigvgan.ups.3.0", 192, 96, 4, 2, 1},
    {"bigvgan.ups.4.0", 96, 48, 4, 2, 1},
    {"bigvgan.ups.5.0", 48, 24, 4, 2, 1},
};
struct BigVGANResblockSpec {
    const char* prefix;
    uint32_t channels;
    uint32_t kernel;
};

const BigVGANResblockSpec kBigVGANResblocks[] = {
    {"bigvgan.resblocks.0", 768, 3},
    {"bigvgan.resblocks.1", 768, 7},
    {"bigvgan.resblocks.2", 768, 11},
    {"bigvgan.resblocks.3", 384, 3},
    {"bigvgan.resblocks.4", 384, 7},
    {"bigvgan.resblocks.5", 384, 11},
    {"bigvgan.resblocks.6", 192, 3},
    {"bigvgan.resblocks.7", 192, 7},
    {"bigvgan.resblocks.8", 192, 11},
    {"bigvgan.resblocks.9", 96, 3},
    {"bigvgan.resblocks.10", 96, 7},
    {"bigvgan.resblocks.11", 96, 11},
    {"bigvgan.resblocks.12", 48, 3},
    {"bigvgan.resblocks.13", 48, 7},
    {"bigvgan.resblocks.14", 48, 11},
    {"bigvgan.resblocks.15", 24, 3},
    {"bigvgan.resblocks.16", 24, 7},
    {"bigvgan.resblocks.17", 24, 11},
};
static mit2::PassSlot bigvgan_activation_pass(mit2::MetalContext& metal,
                                              const mit2::Bundle& bundle,
                                              const std::string& prefix,
                                              mit2::PassSlot x, uint32_t tokens, uint32_t channels);
static std::vector<float> bigvgan_conv_weight_for_resident(mit2::MetalContext& metal,
                                                           const mit2::Bundle& bundle,
                                                           const std::string& prefix,
                                                           uint32_t out_ch, uint32_t in_ch, uint32_t kernel);
static std::vector<float> bigvgan_tensor_for_resident(mit2::MetalContext& metal,
                                                      const mit2::Bundle& bundle,
                                                      const std::string& name);
std::vector<float> run_bigvgan_vocoder_metal_single_pass(mit2::MetalContext& metal,
                                                         const mit2::Bundle& bundle,
                                                         const std::vector<float>& mel,
                                                         uint32_t tokens);
std::pair<std::vector<float>, std::vector<float>>
run_dit_estimator_step_metal_cfg_transformer_batched_pass(
    mit2::MetalContext& metal,
    const mit2::Bundle& bundle,
    const std::vector<float>& x,
    const std::vector<float>& prompt_x,
    const std::vector<float>& cond,
    const std::vector<float>& style,
    const std::vector<float>& null_prompt_x,
    const std::vector<float>& null_cond,
    const std::vector<float>& null_style,
    const std::vector<float>& t1,
    const std::vector<float>& t2,
    const std::vector<uint32_t>& mask,
    uint32_t tokens);
std::vector<float> run_length_regulator_front_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens);
std::vector<float> run_length_regulator_full_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens);
std::vector<float> run_timestep_embedder_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& timesteps, const std::string& prefix);
static void run_transformer_block_pass_into(
    mit2::MetalContext& metal,
    const mit2::Bundle& bundle,
    mit2::PassSlot x,
    mit2::PassSlot attn_wb,   // precomputed adaLN modulation [1024] = proj(t1)
    mit2::PassSlot ffn_wb,    // precomputed adaLN modulation [1024] = proj(t1)
    mit2::PassSlot mask,
    uint32_t batch,
    uint32_t tokens,
    uint32_t layer,
    mit2::PassSlot skip_in,
    bool has_skip,
    mit2::PassSlot output);
static std::vector<float> tensor_for_resident(mit2::MetalContext& metal,
                                              const mit2::Bundle& bundle,
                                              const std::string& name);
std::vector<float> weight_norm_conv_transpose_weight(const mit2::Bundle& bundle, const std::string& prefix, uint32_t in_channels, uint32_t out_channels, uint32_t kernel);
std::vector<float> weight_norm_conv_weight(const mit2::Bundle& bundle, const std::string& prefix, uint32_t out_channels, uint32_t in_channels, uint32_t kernel);
std::vector<float> weight_norm_rowmajor(const std::vector<float>& g, const std::vector<float>& v, uint32_t rows, uint32_t cols);

static mit2::PassSlot bigvgan_activation_pass(mit2::MetalContext& metal,
                                              const mit2::Bundle& bundle,
                                              const std::string& prefix,
                                              mit2::PassSlot x, uint32_t tokens, uint32_t channels) {
    auto up = bigvgan_tensor_for_resident(metal, bundle, prefix + ".upsample.filter");
    auto down = bigvgan_tensor_for_resident(metal, bundle, prefix + ".downsample.lowpass.filter");
    auto alpha = bigvgan_tensor_for_resident(metal, bundle, prefix + ".act.alpha");
    auto beta = bigvgan_tensor_for_resident(metal, bundle, prefix + ".act.beta");
    return metal.bigvgan_activation_f32_pass(
        prefix + ".upsample.filter.resident", up,
        prefix + ".downsample.lowpass.filter.resident", down,
        prefix + ".act.alpha.resident", alpha,
        prefix + ".act.beta.resident", beta,
        x, tokens, channels);
}

static std::vector<float> bigvgan_conv_weight_for_resident(mit2::MetalContext& metal,
                                                           const mit2::Bundle& bundle,
                                                           const std::string& prefix,
                                                           uint32_t out_ch, uint32_t in_ch, uint32_t kernel) {
    if (metal.residentExists(prefix + ".weight_norm.resident") ||
        metal.residentExists(prefix + ".weight_norm.resident.tap0.f32") ||
        metal.residentExists(prefix + ".weight_norm.resident.tap0.f16")) {
        return {};
    }
    return weight_norm_conv_weight(bundle, prefix, out_ch, in_ch, kernel);
}

static std::vector<float> bigvgan_tensor_for_resident(mit2::MetalContext& metal,
                                                      const mit2::Bundle& bundle,
                                                      const std::string& name) {
    if (metal.residentExists(name + ".resident")) {
        return {};
    }
    return tensor_as_f32(bundle, name);
}

std::vector<float> run_bigvgan_vocoder_metal_single_pass(mit2::MetalContext& metal,
                                                         const mit2::Bundle& bundle,
                                                         const std::vector<float>& mel,
                                                         uint32_t tokens) {
    // Stage sizes: conv_pre 1536*T; stage outs: 768*4T, 384*16T, 192*32T, 96*64T, 48*128T, 24*256T.
    const size_t max_stage = static_cast<size_t>(tokens) * 6144;
    // Per-stage scratch: upsample out + 3 resblocks x (3 pairs x 4 temps + ping/pong) + avg.
    const size_t ws = (2 * max_stage + 48 * max_stage + static_cast<size_t>(tokens) * 1600 + 4096) * sizeof(float);
    metal.beginPass(ws);
    auto stage_a = metal.passAlloc(static_cast<uint32_t>(max_stage));
    auto stage_b = metal.passAlloc(static_cast<uint32_t>(max_stage));
    auto mel_slot = metal.passUploadAlloc(mel);
    metal.passSetScratchBase();

    // conv_pre
    {
        auto w = bigvgan_conv_weight_for_resident(metal, bundle, "bigvgan.conv_pre", 1536, 80, 7);
        auto b = bigvgan_tensor_for_resident(metal, bundle, "bigvgan.conv_pre.bias");
        auto pre = metal.conv1d_same_f32_pass(
            "bigvgan.conv_pre.weight_norm.resident", w,
            "bigvgan.conv_pre.bias.resident", b,
            mel_slot, tokens, 80, 1536, 7);
        metal.copy_f32_pass_into(pre, stage_a.slice(0, tokens * 1536), tokens * 1536);
        metal.passResetScratch();
    }

    mit2::PassSlot cur = stage_a;
    mit2::PassSlot other = stage_b;
    uint32_t cur_tokens = tokens;
    uint32_t cur_channels = 1536;

    for (uint32_t i = 0; i < 6; ++i) {
        const auto& spec = kBigVGANUpsamplers[i];
        const std::string ups_prefix(spec.prefix);
        auto uw = (metal.residentExists(ups_prefix + ".weight_norm.resident") ||
                   metal.residentExists(ups_prefix + ".weight_norm.resident.tap0.f32") ||
                   metal.residentExists(ups_prefix + ".weight_norm.resident.tap0.f16"))
                      ? std::vector<float>{}
                      : weight_norm_conv_transpose_weight(bundle, spec.prefix, spec.in_channels, spec.out_channels, spec.kernel);
        auto ub = bigvgan_tensor_for_resident(metal, bundle, ups_prefix + ".bias");
        auto up = metal.conv_transpose1d_f32_pass(
            ups_prefix + ".weight_norm.resident", uw,
            ups_prefix + ".bias.resident", ub,
            cur.slice(0, cur_tokens * cur_channels),
            cur_tokens, spec.in_channels, spec.out_channels, spec.kernel, spec.stride, spec.padding);
        cur_tokens = (cur_tokens - 1) * spec.stride + spec.kernel - 2 * spec.padding;
        cur_channels = spec.out_channels;
        const uint32_t n = cur_tokens * cur_channels;

        // 3 resblocks over `up`, averaged.
        mit2::PassSlot rb_out[3];
        for (uint32_t bidx = 0; bidx < 3; ++bidx) {
            const auto& rspec = kBigVGANResblocks[i * 3 + bidx];
            const std::string block_prefix(rspec.prefix);
            const uint32_t dilations[3] = {1, 3, 5};
            mit2::PassSlot x_rb = up;
            for (uint32_t pair = 0; pair < 3; ++pair) {
                const std::string ps = std::to_string(pair);
                auto t1 = bigvgan_activation_pass(metal, bundle, block_prefix + ".activations." + std::to_string(pair * 2), x_rb, cur_tokens, cur_channels);
                auto c1w = bigvgan_conv_weight_for_resident(metal, bundle, block_prefix + ".convs1." + ps, cur_channels, cur_channels, rspec.kernel);
                auto c1b = bigvgan_tensor_for_resident(metal, bundle, block_prefix + ".convs1." + ps + ".bias");
                auto t2 = metal.conv1d_dilated_same_f32_pass(
                    block_prefix + ".convs1." + ps + ".weight_norm.resident", c1w,
                    block_prefix + ".convs1." + ps + ".bias.resident", c1b,
                    t1, cur_tokens, cur_channels, cur_channels, rspec.kernel, dilations[pair]);
                auto t3 = bigvgan_activation_pass(metal, bundle, block_prefix + ".activations." + std::to_string(pair * 2 + 1), t2, cur_tokens, cur_channels);
                auto c2w = bigvgan_conv_weight_for_resident(metal, bundle, block_prefix + ".convs2." + ps, cur_channels, cur_channels, rspec.kernel);
                auto c2b = bigvgan_tensor_for_resident(metal, bundle, block_prefix + ".convs2." + ps + ".bias");
                auto t4 = metal.conv1d_dilated_same_f32_pass(
                    block_prefix + ".convs2." + ps + ".weight_norm.resident", c2w,
                    block_prefix + ".convs2." + ps + ".bias.resident", c2b,
                    t3, cur_tokens, cur_channels, cur_channels, rspec.kernel, 1);
                x_rb = metal.add_f32_pass(t4, x_rb);
            }
            rb_out[bidx] = x_rb;
        }
        auto avg = metal.avg3_f32_pass(rb_out[0], rb_out[1], rb_out[2]);
        metal.copy_f32_pass_into(avg, other.slice(0, n), n);
        std::swap(cur, other);
        metal.passResetScratch();
    }

    // Post: activation_post -> conv_post -> clamp.
    auto act = bigvgan_activation_pass(metal, bundle, "bigvgan.activation_post",
                                       cur.slice(0, cur_tokens * 24), cur_tokens, 24);
    auto pw = bigvgan_conv_weight_for_resident(metal, bundle, "bigvgan.conv_post", 1, 24, 7);
    const std::vector<float> post_zero_bias{0.0f};
    auto wave_raw = metal.conv1d_same_f32_pass(
        "bigvgan.conv_post.weight_norm.resident", pw,
        "bigvgan.conv_post.bias.zero.resident", post_zero_bias,
        act, cur_tokens, 24, 1, 7);
    auto wave_slot = metal.clamp_f32_pass(wave_raw, -1.0f, 1.0f);
    metal.endPass();
    return metal.passRead(wave_slot);
}

static std::pair<std::vector<float>, std::vector<float>> run_cfg_pass(
    mit2::MetalContext& metal,
    const mit2::Bundle& bundle,
    const std::vector<float>& x,
    const std::vector<float>& prompt_x,
    const std::vector<float>& cond,
    const std::vector<float>& style,
    const std::vector<float>& null_prompt_x,
    const std::vector<float>& null_cond,
    const std::vector<float>& null_style,
    const std::vector<float>& t1,
    const std::vector<float>& t2,
    const std::vector<uint32_t>& mask,
    uint32_t tokens, uint32_t steps, uint32_t prompt_tokens, float cfg_rate,
    const std::vector<float>& modulation = {})
{
    if (tokens > kFusedDitAttentionMaxTokens) throw std::invalid_argument("CFM token capacity exceeded");

    constexpr uint32_t batch = 2;
    const uint32_t rows = batch * tokens;

    // Workspace: persistent region (~26000 el/token) + reusable scratch (~21500 el/token, reset per block).
    // Formula: (persistent + scratch) × 4 bytes/element + alignment padding.
    const size_t ws = static_cast<size_t>(tokens) * (26000 + 27000 + 160) * 4 +
                      static_cast<size_t>(std::max(steps, 1u)) * 1024 * 4 + 256 * 208 + modulation.size() * sizeof(float);
    metal.beginPass(ws);

    // ----------------------------------------------------------------
    // Prepare batched inputs on CPU side, then upload into workspace.
    // ----------------------------------------------------------------
    std::vector<float> x_mel_bat(x.size() * batch);
    std::copy(x.begin(), x.end(), x_mel_bat.begin());
    std::copy(x.begin(), x.end(), x_mel_bat.begin() + static_cast<std::ptrdiff_t>(x.size()));

    std::vector<float> px_bat(prompt_x.size() * batch);
    std::copy(prompt_x.begin(), prompt_x.end(), px_bat.begin());
    std::copy(null_prompt_x.begin(), null_prompt_x.end(), px_bat.begin() + static_cast<std::ptrdiff_t>(prompt_x.size()));

    std::vector<float> cond_bat(cond.size() * batch);
    std::copy(cond.begin(), cond.end(), cond_bat.begin());
    std::copy(null_cond.begin(), null_cond.end(), cond_bat.begin() + static_cast<std::ptrdiff_t>(cond.size()));

    std::vector<float> style_bat(style.size() * batch);
    std::copy(style.begin(), style.end(), style_bat.begin());
    std::copy(null_style.begin(), null_style.end(), style_bat.begin() + static_cast<std::ptrdiff_t>(style.size()));

    std::vector<uint32_t> mask_bat;
    mask_bat.reserve(static_cast<size_t>(batch) * tokens);
    mask_bat.insert(mask_bat.end(), mask.begin(), mask.end());
    mask_bat.insert(mask_bat.end(), mask.begin(), mask.end());

    // ----------------------------------------------------------------
    // Pre-allocate ALL persistent output slots (must come before passSetScratchBase).
    // Uploaded inputs live here too — they need to survive for the entire pass.
    // ----------------------------------------------------------------
    auto x_mel_slot = metal.passUploadAlloc(x_mel_bat);
    auto x_cur_slot = metal.passUploadAlloc(x);
    auto x_nxt_slot = metal.passAlloc(tokens * 80);
    auto px_slot    = metal.passUploadAlloc(px_bat);
    auto cond_slot  = metal.passUploadAlloc(cond_bat);
    auto style_slot = metal.passUploadAlloc(style_bat.data(), static_cast<uint32_t>(style_bat.size()));
    auto mask_slot  = metal.passUploadAllocU32(mask_bat);
    auto t1_all_slot = metal.passUploadAlloc(t1);
    auto t2_all_slot = metal.passUploadAlloc(t2);
    const auto modulation_slot = modulation.empty() ? mit2::PassSlot{} : metal.passUploadAlloc(modulation);

    // Persistent outputs for input merge
    auto cond_proj_slot = metal.passAlloc(rows * 512);
    auto x_in_slot      = metal.passAlloc(rows * 512);

    // Transformer persistent: 6 skip slots (layers 0-5 outputs),
    // 2 ping-pong slots (layers 6-12 outputs), 1 raw output of layer 12.
    mit2::PassSlot skip_slot[6];
    for (int i = 0; i < 6; ++i) skip_slot[i] = metal.passAlloc(rows * 512);
    auto x_ping   = metal.passAlloc(rows * 512);
    auto x_pong   = metal.passAlloc(rows * 512);
    auto x_raw_12 = metal.passAlloc(rows * 512);  // raw layer-12 output before final norm

    // Post-transformer persistent
    auto long_skip_slot = metal.passAlloc(rows * 512);
    auto conv1_slot     = metal.passAlloc(rows * 512);
    auto res_proj_slot  = metal.passAlloc(rows * 512);

    // Wavenet persistent
    auto wn_cond_slot = metal.passAlloc(8192);
    const std::vector<float> wn_zeros_vec(static_cast<size_t>(rows) * 512, 0.0f);
    auto wn_zeros     = metal.passUploadAlloc(wn_zeros_vec);  // initial output accumulator
    auto wn_state_a   = metal.passAlloc(rows * 1024);   // ping-pong A: [new_x | new_out]
    auto wn_state_b   = metal.passAlloc(rows * 1024);   // ping-pong B

    // Final layer persistent
    auto final_in_slot     = metal.passAlloc(rows * 512);
    auto final_hidden_slot = metal.passAlloc(rows * 512);
    auto dphi_slot         = metal.passAlloc(rows * 80);

    // All subsequent allocations are scratch — reset per-block to reuse memory.
    metal.passSetScratchBase();

    // ----------------------------------------------------------------
    // Input merge: cond_projection → dit_input_merge → cond_x_merge_linear
    // ----------------------------------------------------------------
    {
        auto cp_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_projection.weight");
        auto cp_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_projection.bias");
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.cond_projection.weight.resident", cp_w,
            "s2mel.net.cfm.estimator.cond_projection.bias.resident", cp_b,
            cond_slot, rows, 512, 512, cond_proj_slot);
        // cond_proj_slot is persistent; scratch still empty — no reset needed yet.

    }

    // Keep the trajectory and both CFG branches on GPU through the whole schedule.
    // Conditioning is independent of time and is projected once per request.
    for (uint32_t step = 0; step < std::max(steps, 1u); ++step) {
        const auto t1_slot = t1_all_slot.slice(step * 512, 512);
        const auto t2_slot = t2_all_slot.slice(step * 512, 512);
        if (steps) {
            metal.copy_f32_pass_into(x_cur_slot, x_mel_slot.slice(0, tokens * 80), tokens * 80);
            metal.copy_f32_pass_into(x_cur_slot, x_mel_slot.slice(tokens * 80, tokens * 80), tokens * 80);
        }
    {
        auto merged_tmp = metal.dit_input_merge_batched_f32_pass(
            x_mel_slot, px_slot, cond_proj_slot, style_slot, batch, tokens);

        auto mg_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.weight");
        auto mg_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.cond_x_merge_linear.bias");
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.cond_x_merge_linear.weight.resident", mg_w,
            "s2mel.net.cfm.estimator.cond_x_merge_linear.bias.resident", mg_b,
            merged_tmp, rows, 512, 864, x_in_slot);

        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Transformer stack: 13 layers with skip connections (LIFO).
    // Layer outputs:
    //   0..5  → skip_slot[0..5]  (saved for skip connections at layers 7..12)
    //   6     → x_ping           (no skip)
    //   7     → x_pong           (skip from skip_slot[5])
    //   8     → x_ping           (skip from skip_slot[4])
    //   9     → x_pong           (skip from skip_slot[3])
    //   10    → x_ping           (skip from skip_slot[2])
    //   11    → x_pong           (skip from skip_slot[1])
    //   12    → x_raw_12         (skip from skip_slot[0])
    // ----------------------------------------------------------------
    {
        const mit2::PassSlot layer_inputs[13] = {
            x_in_slot,
            skip_slot[0], skip_slot[1], skip_slot[2], skip_slot[3], skip_slot[4],
            skip_slot[5], x_ping, x_pong, x_ping, x_pong, x_ping, x_pong
        };
        const mit2::PassSlot layer_outputs[13] = {
            skip_slot[0], skip_slot[1], skip_slot[2], skip_slot[3], skip_slot[4], skip_slot[5],
            x_ping, x_pong, x_ping, x_pong, x_ping, x_pong, x_raw_12
        };
        const int skip_from[6] = {5, 4, 3, 2, 1, 0};  // for layers 7..12

        for (uint32_t layer = 0; layer < 13; ++layer) {
            const bool has_skip = (layer > 6);
            const mit2::PassSlot si = has_skip ? skip_slot[skip_from[layer - 7]] : mit2::PassSlot{};

            const std::string lb = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
            auto attn_pw = tensor_for_resident(metal, bundle, lb + ".attention_norm.project_layer.weight");
            auto attn_pb = tensor_for_resident(metal, bundle, lb + ".attention_norm.project_layer.bias");
            auto attn_wb = modulation_slot.valid() ? modulation_slot.slice((step * 27 + layer * 2) * 1024, 1024) : metal.linear_f32_pass(
                lb + ".attention_norm.project_layer.weight.resident", attn_pw,
                lb + ".attention_norm.project_layer.bias.resident", attn_pb,
                t1_slot, 1024, 512);
            auto ffn_pw = tensor_for_resident(metal, bundle, lb + ".ffn_norm.project_layer.weight");
            auto ffn_pb = tensor_for_resident(metal, bundle, lb + ".ffn_norm.project_layer.bias");
            auto ffn_wb = modulation_slot.valid() ? modulation_slot.slice((step * 27 + layer * 2 + 1) * 1024, 1024) : metal.linear_f32_pass(
                lb + ".ffn_norm.project_layer.weight.resident", ffn_pw,
                lb + ".ffn_norm.project_layer.bias.resident", ffn_pb,
                t1_slot, 1024, 512);

            run_transformer_block_pass_into(
                metal, bundle,
                layer_inputs[layer], attn_wb, ffn_wb, mask_slot,
                batch, tokens, layer,
                si, has_skip,
                layer_outputs[layer]);

            metal.passResetScratch();
        }

        // Final transformer norm (adaptive rmsnorm with t1 conditioning).
        // x_raw_12 is persistent; norm output is scratch but consumed immediately below.
        const std::string norm_base = "s2mel.net.cfm.estimator.transformer.norm";
        auto norm_g  = tensor_for_resident(metal, bundle, norm_base + ".norm.weight");
        auto norm_pw = tensor_for_resident(metal, bundle, norm_base + ".project_layer.weight");
        auto norm_pb = tensor_for_resident(metal, bundle, norm_base + ".project_layer.bias");
        auto norm_wb = modulation_slot.valid() ? modulation_slot.slice((step * 27 + 26) * 1024, 1024) : metal.linear_f32_pass(
            norm_base + ".project_layer.weight.resident", norm_pw,
            norm_base + ".project_layer.bias.resident", norm_pb,
            t1_slot, 1024, 512);
        // x_normed is in scratch — consumed by post-transformer below without resetting.
        auto x_normed = metal.adaptive_rmsnorm_rows_f32_pass(
            norm_base + ".norm.weight.resident", norm_g,
            x_raw_12, norm_wb.slice(0, 512), norm_wb.slice(512, 512), rows, 512, 1e-5f);
        // Do NOT passResetScratch() here — x_normed must survive into post-transformer.

        // ----------------------------------------------------------------
        // Post-transformer projection.
        // x_normed is in scratch (valid until next passResetScratch).
        // ----------------------------------------------------------------
        auto skip_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.skip_linear.weight");
        auto skip_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.skip_linear.bias");
        auto c1_w   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv1.weight");
        auto c1_b   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv1.bias");
        auto rp_w   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.res_projection.weight");
        auto rp_b   = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.res_projection.bias");

        auto skip_in_tmp = metal.concat_rows_f32_pass(x_normed, x_mel_slot, rows, 512, 80);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.skip_linear.weight.resident", skip_w,
            "s2mel.net.cfm.estimator.skip_linear.bias.resident", skip_b,
            skip_in_tmp, rows, 512, 592, long_skip_slot);
        // long_skip_slot is persistent; scratch (x_normed, skip_in_tmp) no longer needed.
        metal.passResetScratch();

        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.conv1.weight.resident", c1_w,
            "s2mel.net.cfm.estimator.conv1.bias.resident", c1_b,
            long_skip_slot, rows, 512, 512, conv1_slot);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.res_projection.weight.resident", rp_w,
            "s2mel.net.cfm.estimator.res_projection.bias.resident", rp_b,
            long_skip_slot, rows, 512, 512, res_proj_slot);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Wavenet conditioning: conv1d(t2, 1, 512, 8192, 1) → wn_cond_slot
    // ----------------------------------------------------------------
    {
        const std::string cond_pfx = "s2mel.net.cfm.estimator.wavenet.cond_layer.conv.conv";
        auto cond_w = (metal.residentExists(cond_pfx + ".weight_norm.resident") ||
                       metal.residentExists(cond_pfx + ".weight_norm.resident.f16"))
                          ? std::vector<float>{}
                          : weight_norm_conv_weight(bundle, cond_pfx, 8192, 512, 1);
        auto cond_b = tensor_for_resident(metal, bundle, cond_pfx + ".bias");
        metal.conv1d_same_f32_pass_into(
            cond_pfx + ".weight_norm.resident", cond_w,
            cond_pfx + ".bias.resident", cond_b,
            t2_slot, 1, 512, 8192, 1, wn_cond_slot);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Wavenet stack: 8 layers, ping-pong state buffers.
    // wn_state[0] = wn_state_a, wn_state[1] = wn_state_b.
    // Layer i writes to wn_state[i%2], reads prev from wn_state[(i+1)%2].
    // Layer 0 special: x=conv1_slot, out_acc=wn_zeros.
    // ----------------------------------------------------------------
    {
        const mit2::PassSlot wn_states[2] = {wn_state_a, wn_state_b};

        for (uint32_t layer = 0; layer < 8; ++layer) {
            const uint32_t res_skip_ch = (layer < 7) ? 1024u : 512u;
            const mit2::PassSlot& out_state = wn_states[layer % 2];

            mit2::PassSlot wn_x, wn_out_acc;
            if (layer == 0) {
                wn_x       = conv1_slot;
                wn_out_acc = wn_zeros;
            } else {
                const mit2::PassSlot& prev = wn_states[(layer - 1) % 2];
                wn_x       = prev.slice(0,             rows * 512);
                wn_out_acc = prev.slice(rows * 512,    rows * 512);
            }

            const std::string in_pfx = "s2mel.net.cfm.estimator.wavenet.in_layers."  + std::to_string(layer) + ".conv.conv";
            const std::string rs_pfx = "s2mel.net.cfm.estimator.wavenet.res_skip_layers." + std::to_string(layer) + ".conv.conv";

            // k=5 conv runs via per-tap residents (MPS branch); only skip the CPU
            // weight-norm compute once the taps exist. (Base resident alone is not
            // enough — the MPS branch would have no data to build taps from.)
            const std::string in_res = in_pfx + ".weight_norm.resident";
            auto in_w  = (metal.residentExists(in_res + ".tap0") || metal.residentExists(in_res + ".tap0.f16") || metal.residentExists(in_res + ".wconv.f32"))
                             ? std::vector<float>{}
                             : weight_norm_conv_weight(bundle, in_pfx, 1024, 512, 5);
            auto in_b  = tensor_for_resident(metal, bundle, in_pfx + ".bias");
            auto rs_w  = (metal.residentExists(rs_pfx + ".weight_norm.resident") ||
                          metal.residentExists(rs_pfx + ".weight_norm.resident.f16"))
                             ? std::vector<float>{}
                             : weight_norm_conv_weight(bundle, rs_pfx, res_skip_ch, 512, 1);
            auto rs_b  = tensor_for_resident(metal, bundle, rs_pfx + ".bias");

            const uint32_t cond_off = layer * 1024;

            auto in_layer = metal.conv1d_reflect_same_batched_f32_pass(
                in_pfx + ".weight_norm.resident", in_w,
                in_pfx + ".bias.resident", in_b,
                wn_x, batch, tokens, 512, 1024, 5);
            auto gate = metal.wavenet_gate_f32_pass(
                in_layer, wn_cond_slot, rows, 512, 8192, cond_off, 1);
            auto res_skip = metal.conv1d_same_f32_pass(
                rs_pfx + ".weight_norm.resident", rs_w,
                rs_pfx + ".bias.resident", rs_b,
                gate, rows, 512, res_skip_ch, 1);

            metal.wavenet_res_skip_update_f32_pass_into(
                wn_x, wn_out_acc, res_skip, mask_slot,
                rows, 512, layer < 7,
                out_state);

            metal.passResetScratch();
        }
    }

    // ----------------------------------------------------------------
    // Final add: wavenet_out + res_proj → final_in_slot
    // Wavenet output = "output" half of the last state (wn_state_b, layer 7 writes to wn_state[7%2]).
    // ----------------------------------------------------------------
    {
        const mit2::PassSlot& last_state = (7 % 2 == 0) ? wn_state_a : wn_state_b;
        auto wn_final = last_state.slice(rows * 512, rows * 512);
        metal.add_f32_pass_into(wn_final, res_proj_slot, final_in_slot);
        metal.passResetScratch();
    }

    // ----------------------------------------------------------------
    // Final layer: silu(t1) → ada_modulation → adaptive_layernorm → linear → conv2 → dphi
    // ----------------------------------------------------------------
    {
        auto ada_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight");
        auto ada_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias");
        const bool lin_res = metal.residentExists("s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident") ||
                             metal.residentExists("s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident.f16");
        auto lin_g = lin_res ? std::vector<float>{} : tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_g");
        auto lin_v = lin_res ? std::vector<float>{} : tensor_as_f32(bundle, "s2mel.net.cfm.estimator.final_layer.linear.weight_v");
        auto lin_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.final_layer.linear.bias");
        auto lin_w = lin_res ? std::vector<float>{} : weight_norm_rowmajor(lin_g, lin_v, 512, 512);

        auto silu_t1     = metal.silu_f32_pass(t1_slot, 512);
        auto shift_scale = metal.linear_f32_pass(
            "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.weight.resident", ada_w,
            "s2mel.net.cfm.estimator.final_layer.adaLN_modulation.1.bias.resident", ada_b,
            silu_t1, 1024, 512);
        auto modulated = metal.adaptive_layernorm_rows_f32_pass(
            final_in_slot, shift_scale.slice(0, 512), shift_scale.slice(512, 512),
            rows, 512, 1e-6f);
        metal.linear_rows_f32_pass_into(
            "s2mel.net.cfm.estimator.final_layer.linear.weight_norm.resident", lin_w,
            "s2mel.net.cfm.estimator.final_layer.linear.bias.resident", lin_b,
            modulated, rows, 512, 512, final_hidden_slot);
        metal.passResetScratch();

        auto c2_w = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv2.weight");
        auto c2_b = tensor_for_resident(metal, bundle, "s2mel.net.cfm.estimator.conv2.bias");
        metal.conv1d_same_f32_pass_into(
            "s2mel.net.cfm.estimator.conv2.weight.resident", c2_w,
            "s2mel.net.cfm.estimator.conv2.bias.resident", c2_b,
            final_hidden_slot, rows, 512, 80, 1, dphi_slot);
        metal.passResetScratch();
    }

    if (steps) {
        // Match the original 2.5 linspace schedule, including FP32 interval rounding.
        const float t = static_cast<float>(step) / steps;
        const float next = static_cast<float>(step + 1) / steps;
        metal.cfm_euler_update_f32_pass_into(x_cur_slot,
            dphi_slot.slice(0, tokens * 80), dphi_slot.slice(tokens * 80, tokens * 80),
            x_nxt_slot, tokens, 80, prompt_tokens, next - t, cfg_rate);
        std::swap(x_cur_slot, x_nxt_slot);
        metal.passResetScratch();
    }
    } // step
    metal.endPass();
    if (steps) return {metal.passRead(x_cur_slot), {}};

    // ----------------------------------------------------------------
    // Read outputs and split CFG branches.
    // ----------------------------------------------------------------
    auto dphi_full = metal.passRead(dphi_slot);
    const size_t branch = static_cast<size_t>(tokens) * 80;
    std::vector<float> dphi(dphi_full.begin(), dphi_full.begin() + static_cast<std::ptrdiff_t>(branch));
    std::vector<float> null_dphi(dphi_full.begin() + static_cast<std::ptrdiff_t>(branch),
                                 dphi_full.begin() + static_cast<std::ptrdiff_t>(branch * 2));
    return {std::move(dphi), std::move(null_dphi)};
}

std::pair<std::vector<float>, std::vector<float>>
run_dit_estimator_step_metal_cfg_transformer_batched_pass(
    mit2::MetalContext& metal, const mit2::Bundle& bundle,
    const std::vector<float>& x, const std::vector<float>& prompt_x,
    const std::vector<float>& cond, const std::vector<float>& style,
    const std::vector<float>& null_prompt_x, const std::vector<float>& null_cond,
    const std::vector<float>& null_style, const std::vector<float>& t1,
    const std::vector<float>& t2, const std::vector<uint32_t>& mask, uint32_t tokens) {
    return run_cfg_pass(metal, bundle, x, prompt_x, cond, style, null_prompt_x,
                        null_cond, null_style, t1, t2, mask, tokens, 0, 0, 0);
}

std::vector<float> prepare_cfm_modulation(mit2::MetalContext& metal, const mit2::Bundle& bundle,
                                          const std::vector<float>& t1, uint32_t steps) {
    if (!steps || t1.size() != steps * 512u) throw std::invalid_argument("Invalid CFM modulation schedule");
    // Preserve the original single-row FP32 reduction order. Cache is owned by
    // AcousticModel, bounded with its timestep cache and freed on destruction.
    metal.beginPass((t1.size() + steps * 27u * 1024u) * sizeof(float) + 4096);
    const auto times = metal.passUploadAlloc(t1);
    mit2::PassSlot first{};
    for (uint32_t step = 0; step < steps; ++step) {
        for (uint32_t i = 0; i < 27; ++i) {
            const std::string p = i == 26 ? "s2mel.net.cfm.estimator.transformer.norm.project_layer" :
                "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(i / 2) +
                (i % 2 ? ".ffn_norm.project_layer" : ".attention_norm.project_layer");
            auto slot = metal.linear_f32_pass(p + ".weight.resident", tensor_for_resident(metal,bundle,p+".weight"),
                p + ".bias.resident", tensor_for_resident(metal,bundle,p+".bias"), times.slice(step*512,512),1024,512);
            if (!first.valid()) first = slot;
        }
    }
    metal.endPass();
    first.element_count = steps * 27u * 1024u;
    return metal.passRead(first);
}

std::vector<float> run_cfm_trajectory_metal_pass(
    mit2::MetalContext& metal, const mit2::Bundle& bundle,
    const std::vector<float>& x, const std::vector<float>& prompt_x,
    const std::vector<float>& cond, const std::vector<float>& style,
    const std::vector<float>& t1, const std::vector<float>& t2,
    uint32_t tokens, uint32_t prompt_tokens, uint32_t steps, float cfg_rate,
    const std::vector<float>& modulation) {
    return run_cfg_pass(metal, bundle, x, prompt_x, cond, style,
        std::vector<float>(x.size(), 0), std::vector<float>(cond.size(), 0),
        std::vector<float>(192, 0), t1, t2, std::vector<uint32_t>(tokens, 1),
        tokens, steps, prompt_tokens, cfg_rate, modulation).first;
}

std::vector<float> run_length_regulator_front_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens) {
    auto weight = tensor_as_f32(bundle, "s2mel.net.length_regulator.content_in_proj.weight");
    auto bias = tensor_as_f32(bundle, "s2mel.net.length_regulator.content_in_proj.bias");
    auto projected = metal.linear_rows_f32_resident(
        "s2mel.net.length_regulator.content_in_proj.weight.resident",
        weight,
        "s2mel.net.length_regulator.content_in_proj.bias.resident",
        bias,
        input,
        in_tokens,
        512,
        1024);
    return metal.nearest_interpolate_f32(projected, in_tokens, out_tokens, 512);
}

std::vector<float> run_length_regulator_full_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& input, uint32_t in_tokens, uint32_t out_tokens) {
    auto x = run_length_regulator_front_metal(metal, bundle, input, in_tokens, out_tokens);
    const int conv_indices[] = {0, 3, 6, 9};
    const int norm_indices[] = {1, 4, 7, 10};
    for (int block = 0; block < 4; ++block) {
        const std::string conv = "s2mel.net.length_regulator.model." + std::to_string(conv_indices[block]);
        const std::string norm = "s2mel.net.length_regulator.model." + std::to_string(norm_indices[block]);
        x = metal.conv1d_same_f32_resident(
            conv + ".weight.resident",
            tensor_as_f32(bundle, conv + ".weight"),
            conv + ".bias.resident",
            tensor_as_f32(bundle, conv + ".bias"),
            x,
            out_tokens,
            512,
            512,
            3);
        x = metal.groupnorm1_f32_resident(
            norm + ".weight.resident",
            tensor_as_f32(bundle, norm + ".weight"),
            norm + ".bias.resident",
            tensor_as_f32(bundle, norm + ".bias"),
            x,
            out_tokens,
            512,
            1e-5f);
        x = metal.mish_f32(x);
    }
    return metal.conv1d_same_f32_resident(
        "s2mel.net.length_regulator.model.12.weight.resident",
        tensor_as_f32(bundle, "s2mel.net.length_regulator.model.12.weight"),
        "s2mel.net.length_regulator.model.12.bias.resident",
        tensor_as_f32(bundle, "s2mel.net.length_regulator.model.12.bias"),
        x,
        out_tokens,
        512,
        512,
        1);
}

std::vector<float> run_timestep_embedder_metal(mit2::MetalContext& metal, const mit2::Bundle& bundle, const std::vector<float>& timesteps, const std::string& prefix) {
    auto freqs = tensor_as_f32(bundle, prefix + ".freqs");
    auto w0 = tensor_as_f32(bundle, prefix + ".mlp.0.weight");
    auto b0 = tensor_as_f32(bundle, prefix + ".mlp.0.bias");
    auto w2 = tensor_as_f32(bundle, prefix + ".mlp.2.weight");
    auto b2 = tensor_as_f32(bundle, prefix + ".mlp.2.bias");
    auto emb = metal.timestep_embedding_f32(timesteps, freqs, 1000.0f);
    auto h = metal.linear_rows_f32_resident(
        prefix + ".mlp.0.weight.resident",
        w0,
        prefix + ".mlp.0.bias.resident",
        b0,
        emb,
        static_cast<uint32_t>(timesteps.size()),
        512,
        256);
    h = metal.silu_f32(h);
    return metal.linear_rows_f32_resident(
        prefix + ".mlp.2.weight.resident",
        w2,
        prefix + ".mlp.2.bias.resident",
        b2,
        h,
        static_cast<uint32_t>(timesteps.size()),
        512,
        512);
}

static void run_transformer_block_pass_into(
    mit2::MetalContext& metal,
    const mit2::Bundle& bundle,
    mit2::PassSlot x,
    mit2::PassSlot attn_wb,   // precomputed adaLN modulation [1024] = proj(t1)
    mit2::PassSlot ffn_wb,    // precomputed adaLN modulation [1024] = proj(t1)
    mit2::PassSlot mask,
    uint32_t batch,
    uint32_t tokens,
    uint32_t layer,
    mit2::PassSlot skip_in,
    bool has_skip,
    mit2::PassSlot output)
{
    const uint32_t rows = batch * tokens;
    const std::string lb = "s2mel.net.cfm.estimator.transformer.layers." + std::to_string(layer);
    const std::vector<float> zero_qkv_bias(1536, 0.0f);
    const std::vector<float> zero_out_bias(512, 0.0f);
    const std::vector<float> zero_mid_bias(1536, 0.0f);

    mit2::PassSlot cur_x = x;
    if (has_skip) {
        auto skip_w = tensor_for_resident(metal, bundle, lb + ".skip_in_linear.weight");
        auto skip_b = tensor_for_resident(metal, bundle, lb + ".skip_in_linear.bias");
        auto cat = metal.concat_rows_f32_pass(x, skip_in, rows, 512, 512);
        cur_x = metal.linear_rows_f32_pass(
            lb + ".skip_in_linear.weight.resident", skip_w,
            lb + ".skip_in_linear.bias.resident", skip_b,
            cat, rows, 512, 1024);
    }

    // Attention norm (modulation vector precomputed per (step, layer))
    auto attn_norm_g = tensor_for_resident(metal, bundle, lb + ".attention_norm.norm.weight");
    auto attn_normed = metal.adaptive_rmsnorm_rows_f32_pass(
        lb + ".attention_norm.norm.weight.resident", attn_norm_g,
        cur_x, attn_wb.slice(0, 512), attn_wb.slice(512, 512), rows, 512, 1e-5f);

    mit2::PassSlot h{};
    if (cfm_skip("attnblock")) {
        h = metal.silu_f32_pass(attn_normed, rows * 512);  // placeholder
    } else {
        // Attention
        auto wqkv = tensor_for_resident(metal, bundle, lb + ".attention.wqkv.weight");
        auto wo = tensor_for_resident(metal, bundle, lb + ".attention.wo.weight");
        auto qkv = metal.linear_rows_f32_pass(
            lb + ".attention.wqkv.weight.resident", wqkv,
            "s2mel.net.cfm.estimator.transformer.attention.wqkv.zero_bias.resident", zero_qkv_bias,
            attn_normed, rows, 1536, 512);
        mit2::PassSlot attn_out{};
        if (cfm_skip("attn")) {
            attn_out = metal.silu_f32_pass(attn_normed, rows * 512);  // placeholder
        } else {
            attn_out = metal.dit_attention_qkv_rope_batched_f32_pass(qkv, mask, batch, tokens, 8, 64);
        }
        auto attn_proj = metal.linear_rows_f32_pass(
            lb + ".attention.wo.weight.resident", wo,
            "s2mel.net.cfm.estimator.transformer.attention.wo.zero_bias.resident", zero_out_bias,
            attn_out, rows, 512, 512);
        h = metal.add_f32_pass(cur_x, attn_proj);
    }

    // FFN norm (modulation vector precomputed per (step, layer))
    auto ffn_norm_g = tensor_for_resident(metal, bundle, lb + ".ffn_norm.norm.weight");
    auto ffn_normed = metal.adaptive_rmsnorm_rows_f32_pass(
        lb + ".ffn_norm.norm.weight.resident", ffn_norm_g,
        h, ffn_wb.slice(0, 512), ffn_wb.slice(512, 512), rows, 512, 1e-5f);

    if (cfm_skip("ffn")) {
        auto ffn_ph = metal.silu_f32_pass(ffn_normed, rows * 512);
        metal.add_f32_pass_into(h, ffn_ph, output);
        return;
    }
    // FFN: w1 and w3 merged into a single [3072,512] GEMM + SwiGLU split.
    const std::string w13_key = lb + ".feed_forward.w1w3.weight";
    std::vector<float> w13;
    if (!metal.residentExists(w13_key + ".resident") && !metal.residentExists(w13_key + ".resident.f16")) {
        auto w1 = tensor_as_f32(bundle, lb + ".feed_forward.w1.weight");
        auto w3 = tensor_as_f32(bundle, lb + ".feed_forward.w3.weight");
        w13.reserve(w1.size() + w3.size());
        w13.insert(w13.end(), w1.begin(), w1.end());
        w13.insert(w13.end(), w3.begin(), w3.end());
    }
    auto w2 = tensor_for_resident(metal, bundle, lb + ".feed_forward.w2.weight");
    const std::vector<float> zero_w13_bias(3072, 0.0f);
    auto h13 = metal.linear_rows_f32_pass(
        w13_key + ".resident", w13,
        "s2mel.net.cfm.estimator.transformer.feed_forward.zero_w13_bias.resident", zero_w13_bias,
        ffn_normed, rows, 3072, 512);
    auto gated = metal.silu_mul_split_f32_pass(h13, rows, 1536);
    auto ffn_out = metal.linear_rows_f32_pass(
        lb + ".feed_forward.w2.weight.resident", w2,
        "s2mel.net.cfm.estimator.transformer.feed_forward.zero_out_bias.resident", zero_out_bias,
        gated, rows, 512, 1536);

    metal.add_f32_pass_into(h, ffn_out, output);
}

static std::vector<float> tensor_for_resident(mit2::MetalContext& metal,
                                              const mit2::Bundle& bundle,
                                              const std::string& name) {
    if (metal.residentExists(name + ".resident") || metal.residentExists(name + ".resident.f16")) {
        return {};
    }
    return tensor_as_f32(bundle, name);
}

std::vector<float> weight_norm_conv_transpose_weight(const mit2::Bundle& bundle, const std::string& prefix, uint32_t in_channels, uint32_t out_channels, uint32_t kernel) {
    auto g = tensor_as_f32(bundle, prefix + ".weight_g");
    auto v = tensor_as_f32(bundle, prefix + ".weight_v");
    return weight_norm_rowmajor(g, v, in_channels, out_channels * kernel);
}

std::vector<float> weight_norm_conv_weight(const mit2::Bundle& bundle, const std::string& prefix, uint32_t out_channels, uint32_t in_channels, uint32_t kernel) {
    auto g = tensor_as_f32(bundle, prefix + ".weight_g");
    auto v = tensor_as_f32(bundle, prefix + ".weight_v");
    return weight_norm_rowmajor(g, v, out_channels, in_channels * kernel);
}

std::vector<float> weight_norm_rowmajor(const std::vector<float>& g, const std::vector<float>& v, uint32_t rows, uint32_t cols) {
    std::vector<float> weight(static_cast<size_t>(rows) * cols);
    for (uint32_t row = 0; row < rows; ++row) {
        float norm = 0.0f;
        for (uint32_t col = 0; col < cols; ++col) {
            const float value = v[static_cast<size_t>(row) * cols + col];
            norm += value * value;
        }
        norm = std::sqrt(norm);
        const float scale = g[row] / norm;
        for (uint32_t col = 0; col < cols; ++col) {
            weight[static_cast<size_t>(row) * cols + col] = v[static_cast<size_t>(row) * cols + col] * scale;
        }
    }
    return weight;
}
}
