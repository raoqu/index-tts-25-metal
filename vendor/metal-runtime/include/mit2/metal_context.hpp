#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mit2 {

// RAII drain scope for Objective-C autoreleased objects. The long-lived server
// worker threads (TTS dispatcher, acoustic worker) are plain std::threads with
// no run loop, so without an explicit pool every Metal/MPS command buffer,
// encoder and descriptor autoreleased during a request lives until the thread
// exits — i.e. forever. Wrapping each request's work in one of these drains
// them per request. Implemented in metal_context.mm via the objc runtime so it
// is usable from plain .cpp translation units.
class AutoreleasePool {
public:
    AutoreleasePool();
    ~AutoreleasePool();
    AutoreleasePool(const AutoreleasePool&) = delete;
    AutoreleasePool& operator=(const AutoreleasePool&) = delete;

private:
    void* token_ = nullptr;
};

// A handle to a region within the active pass workspace buffer.
// element_count is the number of float32 (or uint32) elements; byte_offset is the byte position.
struct PassSlot {
    uint32_t byte_offset = 0;
    uint32_t element_count = 0;
    bool valid() const noexcept { return element_count > 0; }
    // Sub-slice: element_offset and count are in 4-byte elements
    PassSlot slice(uint32_t element_offset, uint32_t count) const noexcept {
        return {byte_offset + element_offset * 4u, count};
    }
};

struct MetalDiagnostics {
    std::string device_name;
    uint64_t recommended_max_working_set = 0;
    bool unified_memory = false;
    bool low_power = false;
    bool headless = false;
};

struct MetalResourceStats {
    uint64_t command_buffers_submitted = 0;
    uint64_t buffer_allocations = 0;
    uint64_t buffer_bytes_allocated = 0;
    double gpu_elapsed_seconds = 0.0;
};

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    MetalDiagnostics diagnostics() const;
    uint64_t command_buffers_submitted() const;
    MetalResourceStats resource_stats() const;
    bool smoke_copy(std::vector<float>& values);
    bool smoke_scratch_arena(size_t capacity_bytes,
                             size_t source_offset_bytes,
                             size_t destination_offset_bytes,
                             const std::vector<float>& values);
    std::vector<float> add_f32(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<float> add_scaled_f32(const std::vector<float>& a, const std::vector<float>& b, float scale);
    std::vector<float> avg3_f32(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c);
    std::vector<float> w2v_bert_normalize_f32(const std::vector<float>& hidden, const std::vector<float>& mean, const std::vector<float>& std, uint32_t tokens);
    std::vector<float> silu_f32(const std::vector<float>& x);
    std::vector<float> silu_mul_f32(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<float> mask_rows_f32(const std::vector<float>& x, const std::vector<uint32_t>& mask, uint32_t tokens, uint32_t width);
    std::vector<float> glu_split_f32(const std::vector<float>& x, uint32_t tokens, uint32_t width);
    std::vector<float> wavenet_gate_f32(const std::vector<float>& in_layer, const std::vector<float>& cond, uint32_t tokens, uint32_t width, uint32_t cond_width, uint32_t cond_offset, uint32_t cond_tokens);
    std::vector<float> wavenet_res_skip_update_f32(const std::vector<float>& x, const std::vector<float>& output, const std::vector<float>& res_skip, const std::vector<uint32_t>& mask, uint32_t tokens, uint32_t width, bool has_residual);
    std::vector<float> geglu_erf_split_f32(const std::vector<float>& x, uint32_t tokens, uint32_t width);
    std::vector<float> gelu_f32(const std::vector<float>& x);
    std::vector<float> tanh_f32(const std::vector<float>& x);
    std::vector<float> clamp_f32(const std::vector<float>& x, float min_value, float max_value);
    std::vector<float> softmax_f32(const std::vector<float>& x);
    std::vector<float> embedding_f32(const std::vector<float>& table, const std::vector<uint32_t>& ids, uint32_t width);
    std::vector<float> embedding_f32_resident(const std::string& table_key, const std::vector<float>& table, const std::vector<uint32_t>& ids, uint32_t width);
    std::vector<float> semantic_quantize_f32_resident(const std::string& in_weight_key, const std::vector<float>& in_weight, const std::string& in_bias_key, const std::vector<float>& in_bias, const std::string& codebook_key, const std::vector<float>& codebook, const std::string& out_weight_key, const std::vector<float>& out_weight, const std::string& out_bias_key, const std::vector<float>& out_bias, const std::vector<float>& spk_cond, uint32_t tokens, std::vector<uint32_t>& codes);
    std::vector<float> layernorm_f32(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta, float eps);
    std::vector<float> layernorm_rows_f32(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta, uint32_t tokens, uint32_t width, float eps);
    std::vector<float> layernorm_f32_resident(const std::string& gamma_key, const std::vector<float>& gamma, const std::string& beta_key, const std::vector<float>& beta, const std::vector<float>& x, float eps);
    std::vector<float> layernorm_rows_f32_resident(const std::string& gamma_key, const std::vector<float>& gamma, const std::string& beta_key, const std::vector<float>& beta, const std::vector<float>& x, uint32_t tokens, uint32_t width, float eps);
    std::vector<float> adaptive_layernorm_rows_f32(const std::vector<float>& x, const std::vector<float>& shift, const std::vector<float>& scale, uint32_t tokens, uint32_t width, float eps);
    std::vector<float> adaptive_rmsnorm_rows_f32(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& adaptive_weight, const std::vector<float>& adaptive_bias, uint32_t tokens, uint32_t width, float eps);
    std::vector<float> adaptive_rmsnorm_rows_f32_resident(const std::string& gamma_key, const std::vector<float>& gamma, const std::vector<float>& x, const std::vector<float>& adaptive_weight, const std::vector<float>& adaptive_bias, uint32_t tokens, uint32_t width, float eps);
    std::vector<float> cfm_euler_update_f32(const std::vector<float>& x, const std::vector<float>& dphi, const std::vector<float>& cfg_dphi, uint32_t tokens, uint32_t width, uint32_t prompt_tokens, float dt, float cfg_rate);
    std::vector<float> concat_rows_f32(const std::vector<float>& a, const std::vector<float>& b, uint32_t tokens, uint32_t a_width, uint32_t b_width);
    std::vector<float> hot_condition_merge_f32(const std::vector<float>& prompt, const std::vector<float>& generated, uint32_t prompt_tokens, uint32_t generated_tokens, uint32_t width);
    std::vector<float> dit_input_merge_f32(const std::vector<float>& x, const std::vector<float>& prompt_x, const std::vector<float>& cond_proj, const std::vector<float>& style, uint32_t tokens);
    std::vector<float> dit_input_merge_batched_f32(const std::vector<float>& x, const std::vector<float>& prompt_x, const std::vector<float>& cond_proj, const std::vector<float>& style, uint32_t batch, uint32_t tokens);
    std::vector<float> rmsnorm_f32(const std::vector<float>& x, const std::vector<float>& gamma, float eps);
    std::vector<float> rmsnorm_rows_f32(const std::vector<float>& x, const std::vector<float>& gamma, uint32_t tokens, uint32_t width);
    std::vector<float> rmsnorm_rows_f32_resident(const std::string& gamma_key, const std::vector<float>& gamma, const std::vector<float>& x, uint32_t tokens, uint32_t width);
    std::vector<float> rmsnorm_rows_eps_f32(const std::vector<float>& x, const std::vector<float>& gamma, uint32_t tokens, uint32_t width, float eps);
    std::vector<float> linear_f32(const std::vector<float>& weight, const std::vector<float>& bias, const std::vector<float>& x, uint32_t rows, uint32_t cols);
    std::vector<float> linear_rows_f32(const std::vector<float>& weight, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t rows, uint32_t cols);
    std::vector<float> linear_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t rows, uint32_t cols);
    std::vector<float> linear_rows_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t rows, uint32_t cols);
    std::vector<float> nearest_interpolate_f32(const std::vector<float>& x, uint32_t in_tokens, uint32_t out_tokens, uint32_t width);
    std::vector<float> conv1d_same_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel);
    std::vector<float> conv1d_same_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel);
    std::vector<float> conv1d_reflect_same_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel);
    std::vector<float> conv1d_reflect_same_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel);
    std::vector<float> conv1d_reflect_same_batched_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t batch, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel);
    std::vector<float> conv1d_reflect_same_batched_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t batch, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel);
    std::vector<float> depthwise_conv1d_same_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t channels, uint32_t kernel);
    std::vector<float> depthwise_conv1d_same_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t channels, uint32_t kernel);
    std::vector<float> depthwise_conv1d_causal_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t channels, uint32_t kernel);
    std::vector<float> subsampling_conv2d_relu_flat_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t input_tokens, uint32_t input_dim, uint32_t out_channels, uint32_t kernel, uint32_t stride);
    std::vector<float> subsampling_conv2d_relu_flat_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t input_tokens, uint32_t input_dim, uint32_t out_channels, uint32_t kernel, uint32_t stride);
    std::vector<float> conv1d_dilated_same_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t dilation);
    std::vector<float> conv1d_dilated_same_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t dilation);
    std::vector<float> conv_transpose1d_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t stride, uint32_t padding);
    std::vector<float> conv_transpose1d_f32_resident(const std::string& weight_key, const std::vector<float>& weight, const std::string& bias_key, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t stride, uint32_t padding);
    std::vector<float> bigvgan_activation_f32(const std::vector<float>& x, const std::vector<float>& up_filter, const std::vector<float>& down_filter, const std::vector<float>& alpha_log, const std::vector<float>& beta_log, uint32_t tokens, uint32_t channels);
    std::vector<float> bigvgan_activation_f32_resident(const std::string& up_filter_key, const std::vector<float>& up_filter, const std::string& down_filter_key, const std::vector<float>& down_filter, const std::string& alpha_key, const std::vector<float>& alpha_log, const std::string& beta_key, const std::vector<float>& beta_log, const std::vector<float>& x, uint32_t tokens, uint32_t channels);
    std::vector<float> groupnorm1_f32(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta, uint32_t tokens, uint32_t channels, float eps);
    std::vector<float> groupnorm1_f32_resident(const std::string& gamma_key, const std::vector<float>& gamma, const std::string& beta_key, const std::vector<float>& beta, const std::vector<float>& x, uint32_t tokens, uint32_t channels, float eps);
    std::vector<float> mish_f32(const std::vector<float>& x);
    std::vector<float> timestep_embedding_f32(const std::vector<float>& timesteps, const std::vector<float>& freqs, float scale);
    std::vector<float> attention_single_head_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t tokens, uint32_t head_dim);
    std::vector<float> attention_single_query_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t key_tokens, uint32_t head_dim);
    std::vector<float> gpt_cached_attention_f32(const std::vector<float>& cache_k, const std::vector<float>& cache_v, const std::vector<float>& current_qkv, uint32_t cache_tokens, uint32_t heads, uint32_t head_dim);
    // Full-sequence causal multi-head attention over packed QKV [tokens, 3*heads*head_dim].
    std::vector<float> gpt_causal_attention_f32(const std::vector<float>& qkv, uint32_t tokens, uint32_t heads, uint32_t head_dim);
    std::vector<float> attention_single_head_causal_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t tokens, uint32_t head_dim);
    std::vector<float> attention_single_head_masked_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t head_dim);
    std::vector<float> conformer_rel_attention_context_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, const std::vector<float>& p, const std::vector<float>& bias_u, const std::vector<float>& bias_v, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t heads, uint32_t head_dim);
    std::vector<float> conformer_rel_attention_context_f32_resident(const std::string& bias_u_key, const std::vector<float>& bias_u, const std::string& bias_v_key, const std::vector<float>& bias_v, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, const std::vector<float>& p, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t heads, uint32_t head_dim);
    std::vector<float> dit_attention_qkv_rope_f32(const std::vector<float>& qkv, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t heads, uint32_t head_dim);
    std::vector<float> dit_attention_qkv_rope_batched_f32(const std::vector<float>& qkv, const std::vector<uint32_t>& key_mask, uint32_t batch, uint32_t tokens, uint32_t heads, uint32_t head_dim);
    std::vector<float> cross_attention_heads_masked_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, const std::vector<uint32_t>& key_mask, uint32_t query_tokens, uint32_t key_tokens, uint32_t heads, uint32_t head_dim);

    // True if a resident GPU buffer exists for this key; callers may then pass
    // an empty weight vector to *_resident/_pass ops to skip the CPU-side copy.
    bool residentExists(const std::string& key) const;

    void beginBatch();
    void endBatch();

    // -----------------------------------------------------------------------
    // Pass API: batch multiple GPU dispatches in one MTLCommandBuffer.
    // Call beginPass() before dispatching, endPass() to commit+wait once.
    // passBarrier() inserts a GPU-side buffer barrier (no CPU stall).
    // passSetScratchBase() marks the current cursor as reusable scratch start.
    // passResetScratch() resets cursor to scratch base (for loop reuse).
    // -----------------------------------------------------------------------
    bool inPass() const;
    void beginPass(size_t workspace_bytes);
    void endPass();
    void passBarrier();
    void passSetScratchBase();
    void passResetScratch();

    PassSlot passAlloc(uint32_t element_count);
    PassSlot passUploadAlloc(const float* data, uint32_t count);
    PassSlot passUploadAlloc(const std::vector<float>& data);
    PassSlot passUploadAllocU32(const uint32_t* data, uint32_t count);
    PassSlot passUploadAllocU32(const std::vector<uint32_t>& data);
    void passUploadInto(PassSlot slot, const float* data, uint32_t count);
    void passUploadInto(PassSlot slot, const std::vector<float>& data);
    std::vector<float> passRead(PassSlot slot) const;

    // Pass-mode operations: each dispatches a kernel into the current pass encoder.
    // All return a newly allocated PassSlot for the output; call passBarrier()
    // before the NEXT op that reads from the returned slot (automatic inside each op).
    PassSlot add_f32_pass(PassSlot a, PassSlot b);
    PassSlot silu_f32_pass(PassSlot x, uint32_t count);
    PassSlot silu_mul_f32_pass(PassSlot a, PassSlot b);
    // SwiGLU over packed [tokens, 2*width]: out = silu(x[:, :width]) * x[:, width:]
    PassSlot silu_mul_split_f32_pass(PassSlot x, uint32_t tokens, uint32_t width);
    PassSlot concat_rows_f32_pass(PassSlot a, PassSlot b, uint32_t tokens, uint32_t a_width, uint32_t b_width);
    PassSlot linear_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t rows, uint32_t cols);
    PassSlot linear_rows_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t rows, uint32_t cols);
    PassSlot rmsnorm_rows_eps_f32_pass(const std::string& gk,const std::vector<float>& g,PassSlot x,uint32_t tokens,uint32_t width,float eps);
    PassSlot qwen_rope_f32_pass(PassSlot x,uint32_t tokens,uint32_t heads,uint32_t offset);
    PassSlot qwen_attention_f32_pass(uint32_t layer,PassSlot q,PassSlot k,PassSlot v,uint32_t tokens,uint32_t offset);
    PassSlot rmsnorm_rows_f32_pass(const std::string& gk, const std::vector<float>& g, PassSlot x, uint32_t tokens, uint32_t width);
    PassSlot adaptive_rmsnorm_rows_f32_pass(const std::string& gk, const std::vector<float>& g, PassSlot x, PassSlot aw, PassSlot ab, uint32_t tokens, uint32_t width, float eps);
    PassSlot adaptive_layernorm_rows_f32_pass(PassSlot x, PassSlot shift, PassSlot scale, uint32_t tokens, uint32_t width, float eps);
    PassSlot dit_attention_qkv_rope_batched_f32_pass(PassSlot qkv, PassSlot mask, uint32_t batch, uint32_t tokens, uint32_t heads, uint32_t head_dim);
    PassSlot dit_input_merge_batched_f32_pass(PassSlot x, PassSlot prompt_x, PassSlot cond_proj, PassSlot style, uint32_t batch, uint32_t tokens);
    PassSlot conv1d_same_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel);
    PassSlot conv1d_reflect_same_batched_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t batch, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel);
    PassSlot wavenet_gate_f32_pass(PassSlot in_layer, PassSlot cond, uint32_t tokens, uint32_t width, uint32_t cond_width, uint32_t cond_offset, uint32_t cond_tokens);
    // Returns combined [new_x | new_output] slot (element_count = tokens * width * 2)
    PassSlot wavenet_res_skip_update_f32_pass(PassSlot x, PassSlot out, PassSlot res_skip, PassSlot mask, uint32_t tokens, uint32_t width, bool has_residual);

    // GPU-resident GPT KV cache: one persistent buffer for all layers, written
    // once after prefill; the resident attention op appends each new token's
    // K/V on-GPU (no per-token CPU uploads/readbacks).
    void gptKvCacheCreate(uint32_t layers, uint32_t max_tokens, uint32_t width);
    std::vector<float> gptKvCacheRead(uint32_t tokens) const;
    void gptKvCacheWrite(const std::vector<float>& values,uint32_t tokens);
    void gptKvCacheUpload(uint32_t layer, const std::vector<float>& k, const std::vector<float>& v, uint32_t tokens);
    PassSlot gpt_cached_attention_resident_pass(uint32_t layer, PassSlot current_qkv,
                                                uint32_t cache_tokens, uint32_t heads, uint32_t head_dim);

    // Batch (rows) layernorm and full-sequence causal attention pass ops.
    PassSlot layernorm_rows_f32_pass(const std::string& gamma_key, const std::vector<float>& gamma,
                                     const std::string& beta_key, const std::vector<float>& beta,
                                     PassSlot x, uint32_t tokens, uint32_t width, float eps);
    PassSlot gpt_causal_attention_f32_pass(PassSlot qkv, uint32_t tokens, uint32_t heads, uint32_t head_dim);

    // Fused single-row decode op: [LayerNorm ->] GEMV(fp16 W) [-> GELU] [+ residual].
    // ln keys/vectors ignored unless has_ln; residual ignored unless has_residual.
    PassSlot gpt_fused_gemv_f16w_pass(const std::string& wk, const std::vector<float>& w,
                                      const std::string& bk, const std::vector<float>& b,
                                      PassSlot x, uint32_t rows, uint32_t cols,
                                      bool has_ln, const std::string& gk, const std::vector<float>& g,
                                      const std::string& bek, const std::vector<float>& be,
                                      bool fuse_gelu, bool has_residual, PassSlot residual, float eps);
    // Store K/V sections of a full-sequence QKV slot into the resident KV cache layer.
    void gptKvStoreFromQkv_pass(uint32_t layer, PassSlot qkv, uint32_t tokens);

    // GPU-side greedy decode helpers: argmax(logits) -> token slot; build next
    // input embedding from the token slot (chains K tokens in one command buffer).
    PassSlot gpt_argmax_pass(PassSlot logits, uint32_t vocab);
    PassSlot gpt_build_current_pass(PassSlot token_slot,
                                    const std::string& mel_emb_key, const std::vector<float>& mel_emb,
                                    const std::string& pos_emb_key, const std::vector<float>& pos_emb,
                                    uint32_t width, uint32_t position);
    std::vector<uint32_t> passReadU32(PassSlot slot) const;

    // CFM in-pass helpers: euler update into a slot, and raw copy between slots.
    void cfm_euler_update_f32_pass_into(PassSlot x, PassSlot dphi, PassSlot cfg_dphi, PassSlot out,
                                        uint32_t tokens, uint32_t width, uint32_t prompt_tokens,
                                        float dt, float cfg_rate);
    void copy_f32_pass_into(PassSlot src, PassSlot dst, uint32_t count);

    // BigVGAN pass ops (vocoder in a single command buffer).
    PassSlot conv1d_dilated_same_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel, uint32_t dilation);
    PassSlot conv_transpose1d_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel, uint32_t stride, uint32_t padding);
    PassSlot bigvgan_activation_f32_pass(const std::string& upk, const std::vector<float>& up, const std::string& downk, const std::vector<float>& down, const std::string& ak, const std::vector<float>& alpha, const std::string& bek, const std::vector<float>& beta, PassSlot x, uint32_t tokens, uint32_t channels);
    PassSlot avg3_f32_pass(PassSlot a, PassSlot b, PassSlot c);
    PassSlot clamp_f32_pass(PassSlot x, float min_value, float max_value);

    // ------------------------------------------------------------------
    // GPT decode ICB: record the per-token dispatch graph ONCE into an
    // MTLIndirectCommandBuffer over a dedicated stable workspace, then execute
    // it N times per command buffer. Varying scalars (kv_tokens / position /
    // step) live in a GPU state buffer advanced by a recorded kernel.
    // ------------------------------------------------------------------
    bool gptIcbAvailable() const;
    void gptIcbInvalidate();
    void gptIcbBeginRecord(uint32_t max_commands, size_t ws_bytes, uint32_t max_history);
    PassSlot gptIcbAlloc(uint32_t element_count);
    PassSlot gptIcb_build_current(PassSlot token_slot,
                                  const std::string& mel_emb_key, const std::vector<float>& mel_emb,
                                  const std::string& pos_emb_key, const std::vector<float>& pos_emb,
                                  uint32_t width);
    PassSlot gptIcb_fused_gemv_f16w(const std::string& wk, const std::vector<float>& w,
                                    const std::string& bk, const std::vector<float>& b,
                                    PassSlot x, uint32_t rows, uint32_t cols,
                                    bool has_ln, const std::string& gk, const std::vector<float>& g,
                                    const std::string& bek, const std::vector<float>& be,
                                    bool fuse_gelu, bool has_residual, PassSlot residual, float eps);
    PassSlot gptIcb_attention_resident(uint32_t layer, PassSlot qkv, uint32_t heads, uint32_t head_dim);
    PassSlot gptIcb_layernorm(const std::string& gamma_key, const std::vector<float>& gamma,
                              const std::string& beta_key, const std::vector<float>& beta,
                              PassSlot x, uint32_t count, float eps);
    void gptIcb_argmax_into(PassSlot logits, uint32_t vocab, PassSlot token_slot);
    void gptIcb_record_token(PassSlot token_slot);
    void gptIcb_advance_state();
    void gptIcbEndRecord(PassSlot token_slot, PassSlot logits_slot);
    struct GptIcbResult {
        std::vector<uint32_t> tokens;
        std::vector<float> last_logits;
    };
    GptIcbResult gptIcbExecute(uint32_t n_tokens, uint32_t seed_token,
                               uint32_t kv_tokens_start, uint32_t step_start, uint32_t vocab);

    // GPT decode pass ops (single-row layernorm, GELU, KV-cache attention).
    PassSlot layernorm_f32_pass(const std::string& gamma_key, const std::vector<float>& gamma,
                                const std::string& beta_key, const std::vector<float>& beta,
                                PassSlot x, uint32_t count, float eps);
    PassSlot gelu_f32_pass(PassSlot x, uint32_t count);
    PassSlot codec_gelu_pass(PassSlot x);
    PassSlot codec_scale_pass(PassSlot x,const std::string& key,const std::vector<float>& gamma);
    PassSlot depthwise_conv1d_same_pass(const std::string& wk,const std::vector<float>& w,
        const std::string& bk,const std::vector<float>& b,PassSlot x,uint32_t tokens,uint32_t channels,uint32_t kernel);
    PassSlot nearest_interpolate_pass(PassSlot x,uint32_t in_tokens,uint32_t out_tokens,uint32_t width);
    PassSlot gpt_cached_attention_f32_pass(const std::vector<float>& cache_k,
                                           const std::vector<float>& cache_v,
                                           PassSlot current_qkv,
                                           uint32_t cache_tokens, uint32_t heads, uint32_t head_dim);

    // "Into" variants: write output into a pre-allocated slot (no new allocation).
    void add_f32_pass_into(PassSlot a, PassSlot b, PassSlot out);
    void linear_rows_f32_pass_into(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t rows, uint32_t cols, PassSlot out);
    void wavenet_res_skip_update_f32_pass_into(PassSlot x, PassSlot out, PassSlot res_skip, PassSlot mask, uint32_t tokens, uint32_t width, bool has_residual, PassSlot combined_out);
    void conv1d_same_f32_pass_into(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel, PassSlot out);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace mit2
