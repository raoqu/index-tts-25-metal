#include "mit2/metal_context.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mit2 {

// Push/pop a real NSAutoreleasePool via the objc runtime. Using the runtime
// entry points (rather than @autoreleasepool) lets this be a normal C++ RAII
// object callable from plain .cpp worker loops.
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

AutoreleasePool::AutoreleasePool() : token_(objc_autoreleasePoolPush()) {}
AutoreleasePool::~AutoreleasePool() { objc_autoreleasePoolPop(token_); }

static id<MTLBuffer> new_counted_buffer_with_bytes(id<MTLDevice> device,
                                                   uint64_t& buffer_allocations,
                                                   uint64_t& buffer_bytes_allocated,
                                                   const void* data,
                                                   NSUInteger length);
static id<MTLBuffer> new_counted_buffer_with_length(id<MTLDevice> device,
                                                    uint64_t& buffer_allocations,
                                                    uint64_t& buffer_bytes_allocated,
                                                    NSUInteger length);

static std::string command_buffer_error_message(id<MTLCommandBuffer> cb) {
    NSError* err = [cb error];
    if (!err) {
        return "";
    }
    NSString* desc = [err localizedDescription];
    if (!desc) {
        return "";
    }
    return std::string([desc UTF8String]);
}

static std::string linear_rows_resident_failure_message(const char* backend,
                                                        const std::string& weight_key,
                                                        const std::string& bias_key,
                                                        uint32_t tokens,
                                                        uint32_t rows,
                                                        uint32_t cols,
                                                        size_t weight_bytes,
                                                        size_t bias_bytes,
                                                        size_t x_bytes,
                                                        size_t out_bytes,
                                                        size_t arena_bytes,
                                                        const std::string& metal_error) {
    std::ostringstream oss;
    oss << "linear_rows_f32_resident command failed"
        << " backend=" << backend
        << " weight_key=" << weight_key
        << " bias_key=" << bias_key
        << " tokens=" << tokens
        << " rows=" << rows
        << " cols=" << cols
        << " weight_bytes=" << weight_bytes
        << " bias_bytes=" << bias_bytes
        << " x_bytes=" << x_bytes
        << " out_bytes=" << out_bytes
        << " arena_bytes=" << arena_bytes;
    if (!metal_error.empty()) {
        oss << " metal_error=\"" << metal_error << "\"";
    }
    return oss.str();
}

struct MetalContext::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> copy_pipeline = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> add_scaled_pipeline = nil;
    id<MTLComputePipelineState> avg3_pipeline = nil;
    id<MTLComputePipelineState> w2v_bert_normalize_pipeline = nil;
    id<MTLComputePipelineState> silu_pipeline = nil;
    id<MTLComputePipelineState> silu_mul_pipeline = nil;
    id<MTLComputePipelineState> mask_rows_pipeline = nil;
    id<MTLComputePipelineState> glu_split_pipeline = nil;
    id<MTLComputePipelineState> wavenet_gate_pipeline = nil;
    id<MTLComputePipelineState> wavenet_res_skip_update_pipeline = nil;
    id<MTLComputePipelineState> geglu_erf_split_pipeline = nil;
    id<MTLComputePipelineState> gelu_pipeline = nil;
    id<MTLComputePipelineState> tanh_pipeline = nil;
    id<MTLComputePipelineState> clamp_pipeline = nil;
    id<MTLComputePipelineState> softmax_pipeline = nil;
    id<MTLComputePipelineState> embedding_pipeline = nil;
    id<MTLComputePipelineState> semantic_quantize_pipeline = nil;
    id<MTLComputePipelineState> layernorm_pipeline = nil;
    id<MTLComputePipelineState> layernorm_rows_pipeline = nil;
    id<MTLComputePipelineState> layernorm_rows_serial_pipeline = nil;
    id<MTLComputePipelineState> adaptive_layernorm_rows_pipeline = nil;
    id<MTLComputePipelineState> adaptive_rmsnorm_rows_pipeline = nil;
    id<MTLComputePipelineState> cfm_euler_update_pipeline = nil;
    id<MTLComputePipelineState> concat_rows_pipeline = nil;
    id<MTLComputePipelineState> hot_condition_merge_pipeline = nil;
    id<MTLComputePipelineState> dit_input_merge_pipeline = nil;
    id<MTLComputePipelineState> dit_input_merge_batched_pipeline = nil;
    id<MTLComputePipelineState> rmsnorm_pipeline = nil;
    id<MTLComputePipelineState> qwen_rope_pipeline = nil;
    id<MTLComputePipelineState> qwen_store_pipeline = nil;
    id<MTLComputePipelineState> qwen_attention_pipeline = nil;
    id<MTLComputePipelineState> rmsnorm_rows_pipeline = nil;
    id<MTLComputePipelineState> rmsnorm_rows_eps_pipeline = nil;
    id<MTLComputePipelineState> linear_pipeline = nil;
    id<MTLComputePipelineState> linear_rows_pipeline = nil;
    id<MTLComputePipelineState> nearest_interpolate_pipeline = nil;
    id<MTLComputePipelineState> conv1d_same_pipeline = nil;
    id<MTLComputePipelineState> conv1d_reflect_same_pipeline = nil;
    id<MTLComputePipelineState> conv1d_reflect_same_batched_pipeline = nil;
    id<MTLComputePipelineState> depthwise_conv1d_same_pipeline = nil;
    id<MTLComputePipelineState> depthwise_conv1d_causal_pipeline = nil;
    id<MTLComputePipelineState> subsampling_conv2d_relu_flat_pipeline = nil;
    id<MTLComputePipelineState> conv1d_dilated_same_pipeline = nil;
    id<MTLComputePipelineState> conv_transpose1d_pipeline = nil;
    id<MTLComputePipelineState> bigvgan_activation_pipeline = nil;
    id<MTLComputePipelineState> bigvgan_upsample_snake_pipeline = nil;
    id<MTLComputePipelineState> bigvgan_downsample_pipeline = nil;
    id<MTLBuffer> bigvgan_activation_workspace = nil;
    NSUInteger bigvgan_activation_capacity = 0;
    id<MTLComputePipelineState> groupnorm1_pipeline = nil;
    id<MTLComputePipelineState> mish_pipeline = nil;
    id<MTLComputePipelineState> timestep_embedding_pipeline = nil;
    id<MTLComputePipelineState> attention_single_head_pipeline = nil;
    id<MTLComputePipelineState> attention_single_query_pipeline = nil;
    id<MTLComputePipelineState> gpt_cached_attention_pipeline = nil;
    id<MTLComputePipelineState> attention_single_head_causal_pipeline = nil;
    id<MTLComputePipelineState> attention_single_head_masked_pipeline = nil;
    id<MTLComputePipelineState> conformer_rel_attention_context_pipeline = nil;
    id<MTLComputePipelineState> dit_attention_qkv_rope_pipeline = nil;
    id<MTLComputePipelineState> dit_attention_qkv_rope_batched_pipeline = nil;
    id<MTLComputePipelineState> cross_attention_heads_masked_pipeline = nil;
    id<MTLComputePipelineState> broadcast_bias_rows_pipeline = nil;
    id<MTLComputePipelineState> reflect_pad_rows_batched_pipeline = nil;
    id<MTLComputePipelineState> dit_rope_rotate_qk_batched_pipeline = nil;
    id<MTLComputePipelineState> gpt_cached_attention_resident_pipeline = nil;
    id<MTLComputePipelineState> linear_gemv_pipeline = nil;
    id<MTLComputePipelineState> gpt_causal_attention_pipeline = nil;
    id<MTLComputePipelineState> linear_gemv_f16w_pipeline = nil;
    id<MTLComputePipelineState> gpt_fused_gemv_f16w_pipeline = nil;
    id<MTLComputePipelineState> gpt_kv_store_pipeline = nil;
    id<MTLComputePipelineState> gpt_argmax_pipeline = nil;
    id<MTLComputePipelineState> gpt_build_current_pipeline = nil;
    id<MTLComputePipelineState> gemm_f16w_pipeline = nil;
    id<MTLComputePipelineState> silu_mul_split_pipeline = nil;
    id<MTLComputePipelineState> dit_attention_sgq_pipeline = nil;
    id<MTLComputePipelineState> dit_attention_sgq_fp32_pipeline = nil;
    id<MTLComputePipelineState> gpt_causal_attention_sgq_pipeline = nil;
    id<MTLComputePipelineState> gpt_record_token_pipeline = nil;
    id<MTLComputePipelineState> gpt_icb_advance_pipeline = nil;
    id<MTLComputePipelineState> overlap_rows_pipeline = nil;
    // GPT decode ICB state.
    id<MTLIndirectCommandBuffer> gpt_icb = nil;
    id<MTLBuffer> gpt_icb_ws = nil;          // dedicated, never-reallocated decode workspace
    NSUInteger gpt_icb_ws_capacity = 0;
    NSUInteger gpt_icb_ws_cursor = 0;
    id<MTLBuffer> gpt_icb_state = nil;       // 3 uints: kv_tokens, position, step
    id<MTLBuffer> gpt_icb_history = nil;     // token ids per step
    uint32_t gpt_icb_history_capacity = 0;
    uint32_t gpt_icb_cmd_cursor = 0;
    uint32_t gpt_icb_max_commands = 0;
    bool gpt_icb_recording = false;
    bool gpt_icb_ready = false;
    NSUInteger gpt_icb_token_off = 0;
    NSUInteger gpt_icb_logits_off = 0;
    std::vector<id<MTLResource>> gpt_icb_read_resources;
    std::unordered_map<uint32_t, id<MTLBuffer>> icb_const_cache;

    id<MTLBuffer> icb_const_u32(uint32_t value) {
        auto it = icb_const_cache.find(value);
        if (it != icb_const_cache.end()) {
            if (gpt_icb_recording) {
                gpt_icb_read_resources.push_back(it->second);
            }
            return it->second;
        }
        id<MTLBuffer> buf = new_counted_buffer_with_bytes(device, buffer_allocations, buffer_bytes_allocated,
                                                          &value, sizeof(value));
        icb_const_cache.emplace(value, buf);
        if (gpt_icb_recording) {
            gpt_icb_read_resources.push_back(buf);
        }
        return buf;
    }
    id<MTLBuffer> icb_const_f32(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return icb_const_u32(bits);
    }
    mit2::PassSlot icb_alloc_raw(uint32_t element_count) {
        constexpr NSUInteger kAlign = 256;
        const NSUInteger offset = gpt_icb_ws_cursor;
        const size_t raw = static_cast<size_t>(offset) + static_cast<size_t>(element_count) * 4u;
        gpt_icb_ws_cursor = static_cast<NSUInteger>((raw + kAlign - 1) & ~(kAlign - 1));
        if (gpt_icb_ws_cursor > gpt_icb_ws_capacity) {
            throw std::runtime_error("gpt icb workspace overflow");
        }
        return mit2::PassSlot{static_cast<uint32_t>(offset), element_count};
    }
    id<MTLIndirectComputeCommand> icb_next_command() {
        if (!gpt_icb_recording || gpt_icb_cmd_cursor >= gpt_icb_max_commands) {
            throw std::logic_error("gpt icb: not recording or command capacity exceeded");
        }
        return [gpt_icb indirectComputeCommandAtIndex:gpt_icb_cmd_cursor++];
    }
    void icb_track_read(id<MTLResource> r) { gpt_icb_read_resources.push_back(r); }

    // GPU-resident GPT KV cache: layout [layer][K|V][max_tokens][width] f32.
    id<MTLBuffer> gpt_kv_buffer = nil;
    uint32_t gpt_kv_layers = 0;
    uint32_t gpt_kv_max_tokens = 0;
    uint32_t gpt_kv_width = 0;

    NSUInteger gpt_kv_k_offset(uint32_t layer) const {
        return static_cast<NSUInteger>(layer) * 2 * gpt_kv_max_tokens * gpt_kv_width * sizeof(float);
    }
    NSUInteger gpt_kv_v_offset(uint32_t layer) const {
        return gpt_kv_k_offset(layer) + static_cast<NSUInteger>(gpt_kv_max_tokens) * gpt_kv_width * sizeof(float);
    }
    // MPSMatrixMultiplication cache keyed by (M,N,K) — alpha=1, beta=1, A x B^T.
    std::unordered_map<uint64_t, MPSMatrixMultiplication*> mps_gemm_cache;
    // Pass mode state
    id<MTLBuffer> pass_workspace = nil;
    NSUInteger pass_workspace_capacity = 0;
    NSUInteger pass_cursor = 0;
    NSUInteger pass_scratch_base = 0;
    id<MTLCommandBuffer> pass_cb = nil;
    id<MTLComputeCommandEncoder> pass_enc = nil;
    bool pass_mode = false;

    bool batch_mode = false;
    uint64_t command_buffers_submitted = 0;
    uint64_t buffer_allocations = 0;
    uint64_t buffer_bytes_allocated = 0;
    double gpu_elapsed_seconds = 0.0;
    id<MTLBuffer> scratch_arena = nil;
    NSUInteger scratch_arena_capacity = 0;
    std::unordered_map<NSUInteger, std::vector<id<MTLBuffer>>> constant_buffer_pool;
    std::unordered_map<NSUInteger, size_t> constant_buffer_cursor;
    std::unordered_map<std::string, id<MTLBuffer>> resident_buffers;
    std::unordered_map<std::string, NSUInteger> resident_buffer_lengths;
    std::unordered_map<std::string, bool> bias_zero_flags;

    // Zero-ness of a bias, surviving skipped CPU copies (empty b after first upload).
    bool bias_is_zero_for(const std::string& bk, const std::vector<float>& b) {
        auto it = bias_zero_flags.find(bk);
        if (it != bias_zero_flags.end()) {
            return it->second;
        }
        if (b.empty()) {
            return false;  // unknown: broadcast from the resident buffer (always correct)
        }
        bool zero = true;
        for (float v : b) {
            if (v != 0.0f) { zero = false; break; }
        }
        bias_zero_flags.emplace(bk, zero);
        return zero;
    }

    id<MTLBuffer> scratch_arena_buffer(NSUInteger length);
    id<MTLBuffer> constant_buffer_with_bytes(const void* data, NSUInteger length);
    id<MTLBuffer> resident_buffer_with_bytes(const std::string& key, const void* data, NSUInteger length);
    bool has_resident(const std::string& key) const { return resident_buffers.find(key) != resident_buffers.end(); }
    // fp32 -> fp16 converting resident upload. Key convention: base key + ".f16".
    // Empty data is allowed when the buffer already exists (skip-copy pattern).
    id<MTLBuffer> resident_buffer_f16_from_f32(const std::string& key, const float* data, size_t count);
    // Prefer the fp16 resident when enabled; fall back to the fp32 resident when
    // the caller skipped the CPU copy (empty w) and only the fp32 buffer exists.
    id<MTLBuffer> weight_buffer_pref_f16(const std::string& wk, const std::vector<float>& w, bool fp16_wanted, bool& used_f16) {
        if (fp16_wanted) {
            const std::string k16 = wk + ".f16";
            if (has_resident(k16) || !w.empty()) {
                used_f16 = true;
                return resident_buffer_f16_from_f32(k16, w.data(), w.size());
            }
        }
        used_f16 = false;
        return resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    }

    MPSMatrixMultiplication* mps_gemm(uint32_t M, uint32_t N, uint32_t K, bool accumulate) {
        const uint64_t key = (static_cast<uint64_t>(accumulate) << 63) |
                             (static_cast<uint64_t>(M) << 42) | (static_cast<uint64_t>(N) << 21) | K;
        auto it = mps_gemm_cache.find(key);
        if (it != mps_gemm_cache.end()) {
            return it->second;
        }
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:YES
            resultRows:M
            resultColumns:N
            interiorColumns:K
            alpha:1.0
            beta:(accumulate ? 1.0 : 0.0)];
        mps_gemm_cache.emplace(key, mm);
        return mm;
    }

    // Ensure a live compute encoder (reopen after an MPS encode left it nil).
    void pass_ensure_encoder() {
        if (!pass_enc) {
            pass_enc = [pass_cb computeCommandEncoder];
        } else {
            [pass_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    }

    // Dispatch the custom simdgroup-MMA GEMM into the CURRENT pass encoder:
    // C[M,N] = A[M,K] x B[N,K]^T (+bias) (+= C). No encoder split.
    void pass_gemm_f16w(id<MTLBuffer> wbuf, id<MTLBuffer> bbuf, bool add_bias, bool accumulate,
                        NSUInteger x_off, NSUInteger out_off,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t lda = 0) {
        if (lda == 0) {
            lda = K;
        }
        const uint32_t flags = (add_bias ? 1u : 0u) | (accumulate ? 2u : 0u);
        [pass_enc setComputePipelineState:gemm_f16w_pipeline];
        [pass_enc setBuffer:pass_workspace offset:x_off atIndex:0];
        [pass_enc setBuffer:wbuf offset:0 atIndex:1];
        [pass_enc setBuffer:bbuf offset:0 atIndex:2];
        [pass_enc setBuffer:pass_workspace offset:out_off atIndex:3];
        [pass_enc setBytes:&M length:sizeof(M) atIndex:4];
        [pass_enc setBytes:&N length:sizeof(N) atIndex:5];
        [pass_enc setBytes:&K length:sizeof(K) atIndex:6];
        [pass_enc setBytes:&flags length:sizeof(flags) atIndex:7];
        [pass_enc setBytes:&lda length:sizeof(lda) atIndex:8];
        [pass_enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 31) / 32, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    // Encode out[tokens,rows] = x[tokens,cols] @ W[rows,cols]^T + bias via MPS.
    // Splits the pass encoder around the MPS encode (MPS needs the command buffer);
    // hazard tracking on the workspace buffer orders the dispatches.
    void pass_linear_rows_mps(id<MTLBuffer> wbuf, bool w_is_fp16, id<MTLBuffer> bbuf,
                              NSUInteger x_off, NSUInteger out_off,
                              uint32_t tokens, uint32_t rows, uint32_t cols) {
        {
            // Explicitly initialize C, including zero biases. Skipping this write
            // produced non-finite CFM outputs after interleaved GPT requests.
            pass_ensure_encoder();
            uint32_t t = tokens, r = rows;
            [pass_enc setComputePipelineState:broadcast_bias_rows_pipeline];
            [pass_enc setBuffer:bbuf offset:0 atIndex:0];
            [pass_enc setBuffer:pass_workspace offset:out_off atIndex:1];
            [pass_enc setBytes:&t length:sizeof(t) atIndex:2];
            [pass_enc setBytes:&r length:sizeof(r) atIndex:3];
            [pass_enc dispatchThreads:MTLSizeMake(rows, tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(rows, broadcast_bias_rows_pipeline.threadExecutionWidth), 1, 1)];
        }
        if (pass_enc) {
            [pass_enc endEncoding];
            pass_enc = nil;
        }

        MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:tokens
                                                                        columns:cols
                                                                       rowBytes:static_cast<NSUInteger>(cols) * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                        columns:cols
                                                                       rowBytes:static_cast<NSUInteger>(cols) * (w_is_fp16 ? sizeof(uint16_t) : sizeof(float))
                                                                       dataType:(w_is_fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32)];
        MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:tokens
                                                                        columns:rows
                                                                       rowBytes:static_cast<NSUInteger>(rows) * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrix* A = [[MPSMatrix alloc] initWithBuffer:pass_workspace offset:x_off descriptor:dA];
        MPSMatrix* B = [[MPSMatrix alloc] initWithBuffer:wbuf offset:0 descriptor:dB];
        MPSMatrix* C = [[MPSMatrix alloc] initWithBuffer:pass_workspace offset:out_off descriptor:dC];
        [mps_gemm(tokens, rows, cols, true) encodeToCommandBuffer:pass_cb
                                                                leftMatrix:A
                                                               rightMatrix:B
                                                             resultMatrix:C];
        // pass_enc stays nil; PASS_REQUIRE_AND_BARRIER reopens lazily so
        // back-to-back MPS GEMMs share no encoder churn.
    }

    // Batched reflect-same conv1d as kernel-tap GEMM accumulation:
    // out[b,t,:] = bias + sum_k X_pad[b, t+k, :] @ Wk^T  (Wk dense [out_ch, in_ch] per tap)
    void pass_conv1d_reflect_batched_mps(NSUInteger x_off, NSUInteger x_pad_off, NSUInteger out_off,
                                         const std::vector<id<MTLBuffer>>& tap_bufs, bool taps_fp16,
                                         id<MTLBuffer> bbuf,
                                         uint32_t batch, uint32_t tokens,
                                         uint32_t in_ch, uint32_t out_ch, uint32_t kernel) {
        const uint32_t pad = kernel / 2;
        const uint32_t padded_tokens = tokens + 2 * pad;
        pass_ensure_encoder();
        // 1) reflect-pad into x_pad
        uint32_t b_ = batch, t_ = tokens, w_ = in_ch, p_ = pad;
        [pass_enc setComputePipelineState:reflect_pad_rows_batched_pipeline];
        [pass_enc setBuffer:pass_workspace offset:x_off atIndex:0];
        [pass_enc setBuffer:pass_workspace offset:x_pad_off atIndex:1];
        [pass_enc setBytes:&b_ length:sizeof(b_) atIndex:2];
        [pass_enc setBytes:&t_ length:sizeof(t_) atIndex:3];
        [pass_enc setBytes:&w_ length:sizeof(w_) atIndex:4];
        [pass_enc setBytes:&p_ length:sizeof(p_) atIndex:5];
        [pass_enc dispatchThreads:MTLSizeMake(in_ch, padded_tokens, batch)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(in_ch, reflect_pad_rows_batched_pipeline.threadExecutionWidth), 1, 1)];
        [pass_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 2) broadcast bias into out [batch*tokens, out_ch]
        uint32_t rows_total = batch * tokens, oc_ = out_ch;
        [pass_enc setComputePipelineState:broadcast_bias_rows_pipeline];
        [pass_enc setBuffer:bbuf offset:0 atIndex:0];
        [pass_enc setBuffer:pass_workspace offset:out_off atIndex:1];
        [pass_enc setBytes:&rows_total length:sizeof(rows_total) atIndex:2];
        [pass_enc setBytes:&oc_ length:sizeof(oc_) atIndex:3];
        [pass_enc dispatchThreads:MTLSizeMake(out_ch, rows_total, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_ch, broadcast_bias_rows_pipeline.threadExecutionWidth), 1, 1)];
        [pass_enc endEncoding];
        pass_enc = nil;
        // 3) accumulate one GEMM per (batch element, tap)
        MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:tokens
                                                                        columns:in_ch
                                                                       rowBytes:static_cast<NSUInteger>(in_ch) * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:out_ch
                                                                        columns:in_ch
                                                                       rowBytes:static_cast<NSUInteger>(in_ch) * (taps_fp16 ? sizeof(uint16_t) : sizeof(float))
                                                                       dataType:(taps_fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32)];
        MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:tokens
                                                                        columns:out_ch
                                                                       rowBytes:static_cast<NSUInteger>(out_ch) * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrixMultiplication* mm = mps_gemm(tokens, out_ch, in_ch, true);
        for (uint32_t b = 0; b < batch; ++b) {
            const NSUInteger c_off = out_off + static_cast<NSUInteger>(b) * tokens * out_ch * sizeof(float);
            for (uint32_t k = 0; k < kernel; ++k) {
                const NSUInteger a_off = x_pad_off +
                    (static_cast<NSUInteger>(b) * padded_tokens + k) * in_ch * sizeof(float);
                MPSMatrix* A = [[MPSMatrix alloc] initWithBuffer:pass_workspace offset:a_off descriptor:dA];
                MPSMatrix* B = [[MPSMatrix alloc] initWithBuffer:tap_bufs[k] offset:0 descriptor:dB];
                MPSMatrix* C = [[MPSMatrix alloc] initWithBuffer:pass_workspace offset:c_off descriptor:dC];
                [mm encodeToCommandBuffer:pass_cb leftMatrix:A rightMatrix:B resultMatrix:C];
            }
        }
        // pass_enc stays nil; reopened lazily by the next compute op.
    }

    mit2::PassSlot pass_alloc_raw(uint32_t element_count) {
        constexpr NSUInteger kAlign = 256;
        const NSUInteger bytes = static_cast<NSUInteger>(element_count) * 4u;
        const NSUInteger offset = pass_cursor;
        const size_t raw = static_cast<size_t>(offset) + static_cast<size_t>(bytes);
        const size_t new_cursor = (raw + kAlign - 1) & ~(kAlign - 1);
        pass_cursor = static_cast<NSUInteger>(new_cursor);
        if (pass_cursor > pass_workspace_capacity) {
            throw std::runtime_error("pass workspace overflow: need " + std::to_string(pass_cursor) +
                                     " bytes, capacity " + std::to_string(pass_workspace_capacity));
        }
        return mit2::PassSlot{static_cast<uint32_t>(offset), element_count};
    }
};

static void commit_and_count(uint64_t& command_buffers_submitted, id<MTLCommandBuffer> cb) {
    ++command_buffers_submitted;
    [cb commit];
}

static void wait_and_record(double& gpu_elapsed_seconds, id<MTLCommandBuffer> cb, bool skip_wait = false) {
    if (!skip_wait) {
        [cb waitUntilCompleted];
    }
    const double start = [cb GPUStartTime];
    const double end = [cb GPUEndTime];
    if (end > start) {
        gpu_elapsed_seconds += end - start;
    }
}

static void dispatch_dit_rope_rotate_qk(id<MTLComputeCommandEncoder> enc,
                                         id<MTLComputePipelineState> pipeline,
                                         id<MTLBuffer> qkv_buf, NSUInteger qkv_off,
                                         uint32_t batch, uint32_t tokens,
                                         uint32_t heads, uint32_t head_dim) {
    uint32_t b_ = batch, t_ = tokens, h_ = heads, d_ = head_dim;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:qkv_buf offset:qkv_off atIndex:0];
    [enc setBytes:&b_ length:sizeof(b_) atIndex:1];
    [enc setBytes:&t_ length:sizeof(t_) atIndex:2];
    [enc setBytes:&h_ length:sizeof(h_) atIndex:3];
    [enc setBytes:&d_ length:sizeof(d_) atIndex:4];
    const uint32_t half_dim = head_dim / 2;
    [enc dispatchThreads:MTLSizeMake(half_dim, tokens, static_cast<NSUInteger>(batch) * heads)
        threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(half_dim, pipeline.threadExecutionWidth), 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static bool parse_env_bool_override(const char* name, bool& value) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0' || std::strcmp(v, "auto") == 0) {
        return false;
    }
    if (v[0] == '0' || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0) {
        value = false;
        return true;
    }
    if (v[0] == '1' || std::strcmp(v, "true") == 0 || std::strcmp(v, "on") == 0) {
        value = true;
        return true;
    }
    return false;
}

static const std::string& system_metal_device_name() {
    static const std::string name = [] {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                return std::string();
            }
            return std::string([[device name] UTF8String]);
        }
    }();
    return name;
}

static bool device_name_contains(const char* needle) {
    return system_metal_device_name().find(needle) != std::string::npos;
}

// FP32 is the 2.5 validation/default path. FP16 is an explicit opt-in.
static bool fp16_weights_enabled() {
    static const bool on = []() {
        bool forced = false;
        if (parse_env_bool_override("ITTS25_FP16_WEIGHTS", forced)) {
            return forced;
        }
        return false;
    }();
    return on;
}

// Simdgroup-per-query attention (head_dim==64). MIT2_SGQ_ATTN=0 falls back to
// the per-query-threadgroup kernel.
static bool sgq_attention_enabled() {
    static const bool on = []() {
        bool forced = false;
        if (parse_env_bool_override("MIT2_SGQ_ATTN", forced)) {
            return forced;
        }
        return true;
    }();
    return on;
}

// Custom simdgroup-MMA GEMM (fp16 weights) for in-pass GEMMs: avoids the MPS
// encoder split and its whole-buffer hazard serialization. MIT2_CUSTOM_GEMM=1
// forces it, MIT2_CUSTOM_GEMM=0 forces MPS, unset/auto uses the device policy.
static bool custom_gemm_enabled() {
    static const bool on = []() {
        bool forced = false;
        if (parse_env_bool_override("MIT2_CUSTOM_GEMM", forced)) {
            return forced;
        }
        // Current measured policy:
        // - M3 Ultra: MPS wins for CFM GEMMs.
        // - M1 Max: MPS is at least as good in steady-state bench; the larger
        //   regression is BigVGAN, not CFM GEMM.
        // Leave custom GEMM opt-in until a device/OS pair proves otherwise.
        if (device_name_contains("M3 Ultra") || device_name_contains("M1 Max")) {
            return false;
        }
        return false;
    }();
    return on;
}

static bool bigvgan_im2col_enabled_for_device() {
    static const bool on = []() {
        bool forced = false;
        if (parse_env_bool_override("MIT2_BIGVGAN_IM2COL", forced)) {
            return forced;
        }
        // M3 Ultra keeps the measured single-pass custom-conv path by default.
        // M1 Max is vocoder-bound, so auto enables the MPS tap-GEMM backend only there.
        if (device_name_contains("M1 Max")) {
            return true;
        }
        return false;
    }();
    return on;
}

// Use MPS when the GEMM is large enough to amortize the encoder split.
// MIT2_DISABLE_MPS=1 forces the naive kernel (for A/B debugging).
static bool linear_rows_use_mps(uint32_t tokens, uint32_t rows, uint32_t cols) {
    static const bool disabled = []() {
        const char* v = std::getenv("MIT2_DISABLE_MPS");
        return v && v[0] == '1';
    }();
    if (disabled) {
        return false;
    }
    const uint64_t weight_elems = static_cast<uint64_t>(rows) * cols;
    if (weight_elems >= 1048576u) {
        return true;
    }
    return tokens >= 64 && weight_elems >= 16384u;
}

static id<MTLBuffer> new_counted_buffer_with_bytes(id<MTLDevice> device,
                                                   uint64_t& buffer_allocations,
                                                   uint64_t& buffer_bytes_allocated,
                                                   const void* data,
                                                   NSUInteger length) {
    ++buffer_allocations;
    buffer_bytes_allocated += static_cast<uint64_t>(length);
    return [device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> new_counted_buffer_with_length(id<MTLDevice> device,
                                                    uint64_t& buffer_allocations,
                                                    uint64_t& buffer_bytes_allocated,
                                                    NSUInteger length) {
    ++buffer_allocations;
    buffer_bytes_allocated += static_cast<uint64_t>(length);
    return [device newBufferWithLength:length options:MTLResourceStorageModeShared];
}

id<MTLBuffer> MetalContext::Impl::scratch_arena_buffer(NSUInteger length) {
    if (!scratch_arena || scratch_arena_capacity < length) {
        scratch_arena = new_counted_buffer_with_length(
            device, buffer_allocations, buffer_bytes_allocated, length);
        scratch_arena_capacity = length;
    }
    return scratch_arena;
}

id<MTLBuffer> MetalContext::Impl::constant_buffer_with_bytes(const void* data, NSUInteger length) {
    if (length == 0) {
        throw std::invalid_argument("constant buffer length must be non-zero");
    }
    constexpr size_t slots_per_size = 32;
    auto& pool = constant_buffer_pool[length];
    if (pool.size() < slots_per_size) {
        pool.push_back(new_counted_buffer_with_length(device, buffer_allocations, buffer_bytes_allocated, length));
    }
    size_t& cursor = constant_buffer_cursor[length];
    id<MTLBuffer> buffer = pool[cursor % pool.size()];
    cursor = (cursor + 1) % slots_per_size;
    std::memcpy([buffer contents], data, length);
    return buffer;
}

id<MTLBuffer> MetalContext::Impl::resident_buffer_with_bytes(const std::string& key, const void* data, NSUInteger length) {
    if (key.empty()) {
        throw std::invalid_argument("resident buffer key must be non-empty");
    }
    const auto existing = resident_buffers.find(key);
    if (existing != resident_buffers.end()) {
        // length==0 means "caller skipped the CPU copy because the buffer is resident".
        if (length != 0 && resident_buffer_lengths[key] != length) {
            throw std::invalid_argument("resident buffer length changed for key: " + key);
        }
        return existing->second;
    }
    if (length == 0) {
        throw std::invalid_argument("resident buffer missing for key with empty data: " + key);
    }
    id<MTLBuffer> buffer = new_counted_buffer_with_bytes(device, buffer_allocations, buffer_bytes_allocated, data, length);
    resident_buffers.emplace(key, buffer);
    resident_buffer_lengths.emplace(key, length);
    return buffer;
}

id<MTLBuffer> MetalContext::Impl::resident_buffer_f16_from_f32(const std::string& key, const float* data, size_t count) {
    const auto existing = resident_buffers.find(key);
    if (existing != resident_buffers.end()) {
        return existing->second;
    }
    if (count == 0) {
        throw std::invalid_argument("resident f16 buffer missing for key with empty data: " + key);
    }
    const NSUInteger bytes = static_cast<NSUInteger>(count) * sizeof(uint16_t);
    id<MTLBuffer> buffer = new_counted_buffer_with_length(device, buffer_allocations, buffer_bytes_allocated, bytes);
    __fp16* dst = static_cast<__fp16*>([buffer contents]);
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<__fp16>(data[i]);
    }
    resident_buffers.emplace(key, buffer);
    resident_buffer_lengths.emplace(key, bytes);
    return buffer;
}

static size_t align_up_size(size_t value, size_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument("alignment must be non-zero");
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    if (value > std::numeric_limits<size_t>::max() - (alignment - remainder)) {
        throw std::overflow_error("aligned size overflow");
    }
    return value + (alignment - remainder);
}

static size_t arena_capacity_size(size_t required, size_t alignment) {
    return std::max<size_t>(align_up_size(required, alignment), alignment);
}

// Metal kernel source embedded at build time (cmake/embed_metal.cmake);
// the binary has no runtime dependency on metal/kernels.metal.
#include "metal_kernels_embedded.inc"

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
        throw std::runtime_error("missing Metal function: " + std::string([name UTF8String]));
    }
    MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;
    desc.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithDescriptor:desc
                                              options:MTLPipelineOptionNone
                                           reflection:nil
                                                error:&error];
    if (!pipeline) {
        throw std::runtime_error("failed to create pipeline " + std::string([name UTF8String]) + ": " +
                                 std::string([[error localizedDescription] UTF8String]));
    }
    return pipeline;
}

MetalContext::MetalContext() : impl_(new Impl()) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw std::runtime_error("Metal device unavailable");
        }
        impl_->queue = [impl_->device newCommandQueue];
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:mit2::kMetalKernelsSource];
        if (!source) {
            throw std::runtime_error("embedded Metal source is not valid UTF-8");
        }
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else options.fastMathEnabled = NO;
        impl_->library = [impl_->device newLibraryWithSource:source options:options error:&error];
        if (!impl_->library) {
            throw std::runtime_error("failed to compile Metal library: " + std::string([[error localizedDescription] UTF8String]));
        }
        impl_->qwen_rope_pipeline=make_pipeline(impl_->device,impl_->library,@"itts25_qwen_rope");
        impl_->qwen_store_pipeline=make_pipeline(impl_->device,impl_->library,@"itts25_qwen_store");
        impl_->qwen_attention_pipeline=make_pipeline(impl_->device,impl_->library,@"itts25_qwen_attention");
        impl_->copy_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_copy_f32");
        impl_->add_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_add_f32");
        impl_->add_scaled_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_add_scaled_f32");
        impl_->avg3_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_avg3_f32");
        impl_->w2v_bert_normalize_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_w2v_bert_normalize_f32");
        impl_->silu_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_silu_f32");
        impl_->silu_mul_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_silu_mul_f32");
        impl_->mask_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_mask_rows_f32");
        impl_->glu_split_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_glu_split_f32");
        impl_->wavenet_gate_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_wavenet_gate_f32");
        impl_->wavenet_res_skip_update_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_wavenet_res_skip_update_f32");
        impl_->geglu_erf_split_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_geglu_erf_split_f32");
        impl_->gelu_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gelu_f32");
        impl_->tanh_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_tanh_f32");
        impl_->clamp_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_clamp_f32");
        impl_->softmax_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_softmax_f32_one_row");
        impl_->embedding_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_embedding_f32");
        impl_->semantic_quantize_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_semantic_quantize_f32");
        impl_->layernorm_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_layernorm_f32_one_row");
        impl_->layernorm_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_layernorm_f32_rows");
        impl_->layernorm_rows_serial_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_layernorm_f32_rows_serial");
        impl_->adaptive_layernorm_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_adaptive_layernorm_f32_rows");
        impl_->adaptive_rmsnorm_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_adaptive_rmsnorm_f32_rows");
        impl_->cfm_euler_update_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_cfm_euler_update_f32");
        impl_->concat_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_concat_rows_f32");
        impl_->hot_condition_merge_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_hot_condition_merge_f32");
        impl_->dit_input_merge_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_dit_input_merge_f32");
        impl_->dit_input_merge_batched_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_dit_input_merge_batched_f32");
        impl_->rmsnorm_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_rmsnorm_f32_one_row");
        impl_->rmsnorm_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_rmsnorm_f32_rows");
        impl_->rmsnorm_rows_eps_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_rmsnorm_f32_rows_eps");
        impl_->linear_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_linear_f32_rowmajor");
        impl_->linear_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_linear_rows_f32_rowmajor");
        impl_->nearest_interpolate_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_nearest_interpolate_f32");
        impl_->conv1d_same_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_conv1d_same_f32");
        impl_->conv1d_reflect_same_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_conv1d_reflect_same_f32");
        impl_->conv1d_reflect_same_batched_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_conv1d_reflect_same_batched_f32");
        impl_->depthwise_conv1d_same_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_depthwise_conv1d_same_f32");
        impl_->depthwise_conv1d_causal_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_depthwise_conv1d_causal_f32");
        impl_->subsampling_conv2d_relu_flat_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_subsampling_conv2d_relu_flat_f32");
        impl_->conv1d_dilated_same_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_conv1d_dilated_same_f32");
        impl_->conv_transpose1d_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_conv_transpose1d_f32");
        impl_->bigvgan_activation_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_bigvgan_activation_f32");
        impl_->bigvgan_upsample_snake_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_bigvgan_upsample_snake_f32");
        impl_->bigvgan_downsample_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_bigvgan_downsample_f32");
        impl_->groupnorm1_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_groupnorm1_f32");
        impl_->mish_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_mish_f32");
        impl_->timestep_embedding_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_timestep_embedding_f32");
        impl_->attention_single_head_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_attention_single_head_f32");
        impl_->attention_single_query_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_attention_single_query_f32");
        impl_->gpt_cached_attention_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_cached_attention_f32");
        impl_->attention_single_head_causal_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_attention_single_head_causal_f32");
        impl_->attention_single_head_masked_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_attention_single_head_masked_f32");
        impl_->conformer_rel_attention_context_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_conformer_rel_attention_context_f32");
        impl_->dit_attention_qkv_rope_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_dit_attention_qkv_rope_f32");
        impl_->dit_attention_qkv_rope_batched_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_dit_attention_qkv_rope_batched_f32");
        impl_->cross_attention_heads_masked_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_cross_attention_heads_masked_f32");
        impl_->broadcast_bias_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_broadcast_bias_rows_f32");
        impl_->reflect_pad_rows_batched_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_reflect_pad_rows_batched_f32");
        impl_->dit_rope_rotate_qk_batched_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_dit_rope_rotate_qk_batched_f32");
        impl_->gpt_cached_attention_resident_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_cached_attention_resident_f32");
        impl_->linear_gemv_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_linear_gemv_f32");
        impl_->gpt_causal_attention_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_causal_attention_f32");
        impl_->linear_gemv_f16w_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_linear_gemv_f16w_f32");
        impl_->gpt_fused_gemv_f16w_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_fused_gemv_f16w_f32");
        impl_->gpt_kv_store_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_kv_store_f32");
        impl_->gpt_argmax_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_argmax_f32");
        impl_->gpt_build_current_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_build_current_f32");
        impl_->gemm_f16w_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gemm_f16w_f32");
        impl_->silu_mul_split_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_silu_mul_split_f32");
        impl_->dit_attention_sgq_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_dit_attention_sgq_batched_f32");
        impl_->dit_attention_sgq_fp32_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_dit_attention_sgq_fp32_batched_f32");
        impl_->gpt_causal_attention_sgq_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_causal_attention_sgq_f32");
        impl_->gpt_record_token_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_record_token_f32");
        impl_->gpt_icb_advance_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_gpt_icb_advance_f32");
        impl_->overlap_rows_pipeline = make_pipeline(impl_->device, impl_->library, @"mit2_overlap_rows_f32");
    }
}


bool MetalContext::residentExists(const std::string& key) const {
    return impl_->has_resident(key);
}

void MetalContext::beginBatch() {
    impl_->batch_mode = true;
}

void MetalContext::endBatch() {
    impl_->batch_mode = false;
    // Wait for all pending command buffers
    id<MTLCommandBuffer> barrier = [impl_->queue commandBuffer];
    [barrier commit];
    [barrier waitUntilCompleted];
}

MetalContext::~MetalContext() {
    delete impl_;
}

MetalDiagnostics MetalContext::diagnostics() const {
    @autoreleasepool {
        MetalDiagnostics d;
        d.device_name = [[impl_->device name] UTF8String];
        d.recommended_max_working_set = [impl_->device recommendedMaxWorkingSetSize];
        d.unified_memory = [impl_->device hasUnifiedMemory];
        d.low_power = [impl_->device isLowPower];
        d.headless = [impl_->device isHeadless];
        return d;
    }
}

uint64_t MetalContext::command_buffers_submitted() const {
    return impl_->command_buffers_submitted;
}

MetalResourceStats MetalContext::resource_stats() const {
    MetalResourceStats stats;
    stats.command_buffers_submitted = impl_->command_buffers_submitted;
    stats.buffer_allocations = impl_->buffer_allocations;
    stats.buffer_bytes_allocated = impl_->buffer_bytes_allocated;
    stats.gpu_elapsed_seconds = impl_->gpu_elapsed_seconds;
    return stats;
}

bool MetalContext::smoke_copy(std::vector<float>& values) {
    @autoreleasepool {
        const NSUInteger bytes = values.size() * sizeof(float);
        id<MTLBuffer> src = new_counted_buffer_with_bytes(impl_->device, impl_->buffer_allocations, impl_->buffer_bytes_allocated, values.data(), bytes);
        id<MTLBuffer> dst = new_counted_buffer_with_length(impl_->device, impl_->buffer_allocations, impl_->buffer_bytes_allocated, bytes);
        uint32_t count = static_cast<uint32_t>(values.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->copy_pipeline];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];

        const NSUInteger width = impl_->copy_pipeline.threadExecutionWidth;
        MTLSize threads = MTLSizeMake(values.size(), 1, 1);
        MTLSize groups = MTLSizeMake(width, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            return false;
        }

        float* out = static_cast<float*>([dst contents]);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = out[i];
        }
        return true;
    }
}

bool MetalContext::smoke_scratch_arena(size_t capacity_bytes,
                                       size_t source_offset_bytes,
                                       size_t destination_offset_bytes,
                                       const std::vector<float>& values) {
    @autoreleasepool {
        if (values.empty()) {
            throw std::invalid_argument("scratch arena smoke values must be non-empty");
        }
        const size_t bytes = values.size() * sizeof(float);
        const size_t source_end = source_offset_bytes + bytes;
        const size_t destination_end = destination_offset_bytes + bytes;
        if (source_end > capacity_bytes || destination_end > capacity_bytes ||
            source_end < source_offset_bytes || destination_end < destination_offset_bytes) {
            throw std::invalid_argument("scratch arena smoke slice exceeds capacity");
        }
        if (std::max(source_offset_bytes, destination_offset_bytes) < std::min(source_end, destination_end)) {
            throw std::invalid_argument("scratch arena smoke slices must not overlap");
        }

        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(capacity_bytes));
        std::memset([arena contents], 0, capacity_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + source_offset_bytes, values.data(), bytes);
        uint32_t count = static_cast<uint32_t>(values.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->copy_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(source_offset_bytes) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(destination_offset_bytes) atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];

        const NSUInteger width = impl_->copy_pipeline.threadExecutionWidth;
        MTLSize threads = MTLSizeMake(values.size(), 1, 1);
        MTLSize groups = MTLSizeMake(width, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            return false;
        }

        const auto* copied = reinterpret_cast<const float*>(static_cast<const uint8_t*>([arena contents]) + destination_offset_bytes);
        for (size_t i = 0; i < values.size(); ++i) {
            if (copied[i] != values[i]) {
                return false;
            }
        }
        return true;
    }
}

std::vector<float> MetalContext::add_f32(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("add_f32 input size mismatch");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t bytes = a.size() * sizeof(float);
        const size_t a_offset = 0;
        const size_t b_offset = align_up_size(a_offset + bytes, alignment);
        const size_t out_offset = align_up_size(b_offset + bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + a_offset, a.data(), bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + b_offset, b.data(), bytes);
        uint32_t count = static_cast<uint32_t>(a.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->add_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(a_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(b_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:nbuf offset:0 atIndex:3];
        MTLSize threads = MTLSizeMake(a.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->add_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("add_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + a.size());
    }
}

std::vector<float> MetalContext::add_scaled_f32(const std::vector<float>& a,
                                                const std::vector<float>& b,
                                                float scale) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("add_scaled_f32 input size mismatch");
    }
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("add_scaled_f32 scale must be finite");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t bytes = a.size() * sizeof(float);
        const size_t a_offset = 0;
        const size_t b_offset = align_up_size(a_offset + bytes, alignment);
        const size_t out_offset = align_up_size(b_offset + bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + a_offset, a.data(), bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + b_offset, b.data(), bytes);
        uint32_t count = static_cast<uint32_t>(a.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLBuffer> scale_buf = impl_->constant_buffer_with_bytes(&scale, sizeof(scale));

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->add_scaled_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(a_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(b_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:nbuf offset:0 atIndex:3];
        [enc setBuffer:scale_buf offset:0 atIndex:4];
        MTLSize threads = MTLSizeMake(a.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->add_scaled_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("add_scaled_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + a.size());
    }
}

std::vector<float> MetalContext::avg3_f32(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c) {
    if (a.size() != b.size() || a.size() != c.size()) {
        throw std::invalid_argument("avg3_f32 input size mismatch");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = a.size() * sizeof(float);
        const size_t a_offset = 0;
        const size_t b_offset = align_up_size(a_offset + bytes, alignment);
        const size_t c_offset = align_up_size(b_offset + bytes, alignment);
        const size_t out_offset = align_up_size(c_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + a_offset, a.data(), bytes);
        std::memcpy(base + b_offset, b.data(), bytes);
        std::memcpy(base + c_offset, c.data(), bytes);
        uint32_t count = static_cast<uint32_t>(a.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->avg3_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(a_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(b_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(c_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:nbuf offset:0 atIndex:4];
        MTLSize threads = MTLSizeMake(a.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->avg3_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("avg3_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + a.size());
    }
}

std::vector<float> MetalContext::w2v_bert_normalize_f32(
    const std::vector<float>& hidden,
    const std::vector<float>& mean,
    const std::vector<float>& std,
    uint32_t tokens) {
    const size_t value_count = static_cast<size_t>(tokens) * 1024u;
    if (tokens == 0 ||
        hidden.size() != value_count ||
        mean.size() != 1024 ||
        std.size() != 1024) {
        throw std::invalid_argument("w2v_bert_normalize_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t hidden_bytes = hidden.size() * sizeof(float);
        const size_t stats_bytes = mean.size() * sizeof(float);
        const size_t hidden_offset = 0;
        const size_t mean_offset = align_up_size(hidden_offset + hidden_bytes, alignment);
        const size_t std_offset = align_up_size(mean_offset + stats_bytes, alignment);
        const size_t out_offset = align_up_size(std_offset + stats_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + hidden_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + hidden_offset, hidden.data(), hidden_bytes);
        std::memcpy(base + mean_offset, mean.data(), stats_bytes);
        std::memcpy(base + std_offset, std.data(), stats_bytes);
        uint32_t count = static_cast<uint32_t>(value_count);
        id<MTLBuffer> countbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->w2v_bert_normalize_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(hidden_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mean_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(std_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:countbuf offset:0 atIndex:4];
        MTLSize threads = MTLSizeMake(value_count, 1, 1);
        MTLSize groups = MTLSizeMake(impl_->w2v_bert_normalize_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("w2v_bert_normalize_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + value_count);
    }
}

std::vector<float> MetalContext::silu_f32(const std::vector<float>& x) {
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->silu_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];
        MTLSize threads = MTLSizeMake(x.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->silu_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("silu_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::silu_mul_f32(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("silu_mul_f32 input size mismatch");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = a.size() * sizeof(float);
        const size_t a_offset = 0;
        const size_t b_offset = align_up_size(a_offset + bytes, alignment);
        const size_t out_offset = align_up_size(b_offset + bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + a_offset, a.data(), bytes);
        std::memcpy(base + b_offset, b.data(), bytes);
        uint32_t count = static_cast<uint32_t>(a.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->silu_mul_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(a_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(b_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:nbuf offset:0 atIndex:3];
        MTLSize threads = MTLSizeMake(a.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->silu_mul_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("silu_mul_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + a.size());
    }
}

std::vector<float> MetalContext::mask_rows_f32(const std::vector<float>& x, const std::vector<uint32_t>& mask, uint32_t tokens, uint32_t width) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        mask.size() != tokens) {
        throw std::invalid_argument("mask_rows_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger mask_bytes = mask.size() * sizeof(uint32_t);
        const size_t x_offset = 0;
        const size_t mask_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t out_offset = align_up_size(mask_offset + mask_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + mask_offset, mask.data(), mask_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->mask_rows_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mask_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:tokensbuf offset:0 atIndex:3];
        [enc setBuffer:widthbuf offset:0 atIndex:4];
        MTLSize threads = MTLSizeMake(width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->mask_rows_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("mask_rows_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::glu_split_f32(const std::vector<float>& x, uint32_t tokens, uint32_t width) {
    if (tokens == 0 || width == 0 || x.size() != static_cast<size_t>(tokens) * width * 2) {
        throw std::invalid_argument("glu_split_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * width;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->glu_split_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:tokensbuf offset:0 atIndex:2];
        [enc setBuffer:widthbuf offset:0 atIndex:3];
        MTLSize threads = MTLSizeMake(width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->glu_split_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("glu_split_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::wavenet_gate_f32(
    const std::vector<float>& in_layer,
    const std::vector<float>& cond,
    uint32_t tokens,
    uint32_t width,
    uint32_t cond_width,
    uint32_t cond_offset,
    uint32_t cond_tokens) {
    if (tokens == 0 || width == 0 || cond_width == 0 ||
        in_layer.size() != static_cast<size_t>(tokens) * width * 2 ||
        (cond_tokens != 1 && cond_tokens != tokens) ||
        cond.size() != static_cast<size_t>(cond_tokens) * cond_width ||
        static_cast<uint64_t>(cond_offset) + static_cast<uint64_t>(width) * 2ULL > cond_width) {
        throw std::invalid_argument("wavenet_gate_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t in_bytes = in_layer.size() * sizeof(float);
        const size_t cond_bytes = cond.size() * sizeof(float);
        const size_t out_count = static_cast<size_t>(tokens) * width;
        const size_t out_bytes = out_count * sizeof(float);
        const size_t in_offset = 0;
        const size_t cond_offset_bytes = align_up_size(in_offset + in_bytes, alignment);
        const size_t out_offset = align_up_size(cond_offset_bytes + cond_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + in_offset, in_layer.data(), in_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + cond_offset_bytes, cond.data(), cond_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> condwidthbuf = impl_->constant_buffer_with_bytes(&cond_width, sizeof(cond_width));
        id<MTLBuffer> condoffsetbuf = impl_->constant_buffer_with_bytes(&cond_offset, sizeof(cond_offset));
        id<MTLBuffer> condtokensbuf = impl_->constant_buffer_with_bytes(&cond_tokens, sizeof(cond_tokens));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->wavenet_gate_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(in_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(cond_offset_bytes) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:tokensbuf offset:0 atIndex:3];
        [enc setBuffer:widthbuf offset:0 atIndex:4];
        [enc setBuffer:condwidthbuf offset:0 atIndex:5];
        [enc setBuffer:condoffsetbuf offset:0 atIndex:6];
        [enc setBuffer:condtokensbuf offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->wavenet_gate_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("wavenet_gate_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::wavenet_res_skip_update_f32(
    const std::vector<float>& x,
    const std::vector<float>& output,
    const std::vector<float>& res_skip,
    const std::vector<uint32_t>& mask,
    uint32_t tokens,
    uint32_t width,
    bool has_residual) {
    const size_t row_count = static_cast<size_t>(tokens) * width;
    const size_t res_skip_count = has_residual ? row_count * 2 : row_count;
    if (tokens == 0 || width == 0 ||
        x.size() != row_count ||
        output.size() != row_count ||
        res_skip.size() != res_skip_count ||
        mask.size() != tokens) {
        throw std::invalid_argument("wavenet_res_skip_update_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t state_bytes = row_count * sizeof(float);
        const size_t res_skip_bytes = res_skip_count * sizeof(float);
        const size_t mask_bytes = mask.size() * sizeof(uint32_t);
        const size_t out_count = row_count * 2;
        const size_t out_bytes = out_count * sizeof(float);
        uint32_t has_residual_u32 = has_residual ? 1U : 0U;
        const size_t x_offset = 0;
        const size_t output_offset = align_up_size(x_offset + state_bytes, alignment);
        const size_t res_skip_offset = align_up_size(output_offset + state_bytes, alignment);
        const size_t mask_offset = align_up_size(res_skip_offset + res_skip_bytes, alignment);
        const size_t out_offset = align_up_size(mask_offset + mask_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), state_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + output_offset, output.data(), state_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + res_skip_offset, res_skip.data(), res_skip_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + mask_offset, mask.data(), mask_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> hasresbuf = impl_->constant_buffer_with_bytes(&has_residual_u32, sizeof(has_residual_u32));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->wavenet_res_skip_update_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(output_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(res_skip_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mask_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:4];
        [enc setBuffer:tokensbuf offset:0 atIndex:5];
        [enc setBuffer:widthbuf offset:0 atIndex:6];
        [enc setBuffer:hasresbuf offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->wavenet_res_skip_update_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("wavenet_res_skip_update_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::geglu_erf_split_f32(const std::vector<float>& x, uint32_t tokens, uint32_t width) {
    if (tokens == 0 || width == 0 || x.size() != static_cast<size_t>(tokens) * width * 2) {
        throw std::invalid_argument("geglu_erf_split_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * width;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->geglu_erf_split_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:tokensbuf offset:0 atIndex:2];
        [enc setBuffer:widthbuf offset:0 atIndex:3];
        MTLSize threads = MTLSizeMake(width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->geglu_erf_split_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("geglu_erf_split_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::gelu_f32(const std::vector<float>& x) {
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->gelu_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];
        MTLSize threads = MTLSizeMake(x.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->gelu_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("gelu_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::tanh_f32(const std::vector<float>& x) {
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->tanh_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];
        MTLSize threads = MTLSizeMake(x.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->tanh_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("tanh_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::clamp_f32(const std::vector<float>& x, float min_value, float max_value) {
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLBuffer> minbuf = impl_->constant_buffer_with_bytes(&min_value, sizeof(min_value));
        id<MTLBuffer> maxbuf = impl_->constant_buffer_with_bytes(&max_value, sizeof(max_value));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->clamp_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];
        [enc setBuffer:minbuf offset:0 atIndex:3];
        [enc setBuffer:maxbuf offset:0 atIndex:4];
        MTLSize threads = MTLSizeMake(x.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->clamp_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("clamp_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::softmax_f32(const std::vector<float>& x) {
    if (x.empty() || x.size() > 1024 * 1024) {
        throw std::invalid_argument("softmax_f32 expects 1..1048576 elements");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->softmax_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("softmax_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::embedding_f32(const std::vector<float>& table, const std::vector<uint32_t>& ids, uint32_t width) {
    if (width == 0 || table.size() % width != 0) {
        throw std::invalid_argument("embedding_f32 invalid table width");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger table_bytes = table.size() * sizeof(float);
        const NSUInteger ids_bytes = ids.size() * sizeof(uint32_t);
        const NSUInteger out_count = static_cast<NSUInteger>(ids.size()) * width;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t table_offset = 0;
        const size_t ids_offset = align_up_size(table_offset + table_bytes, alignment);
        const size_t out_offset = align_up_size(ids_offset + ids_bytes, alignment);
        const size_t arena_bytes = std::max<size_t>(align_up_size(out_offset + out_bytes, alignment), alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + table_offset, table.data(), table_bytes);
        std::memcpy(base + ids_offset, ids.data(), ids_bytes);
        id<MTLBuffer> width_buf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->embedding_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(table_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(ids_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:width_buf offset:0 atIndex:3];
        MTLSize threads = MTLSizeMake(width, ids.size(), 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->embedding_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("embedding_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::embedding_f32_resident(const std::string& table_key, const std::vector<float>& table, const std::vector<uint32_t>& ids, uint32_t width) {
    if (width == 0 || table.size() % width != 0) {
        throw std::invalid_argument("embedding_f32_resident invalid table width");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger table_bytes = table.size() * sizeof(float);
        const NSUInteger ids_bytes = ids.size() * sizeof(uint32_t);
        const NSUInteger out_count = static_cast<NSUInteger>(ids.size()) * width;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t ids_offset = 0;
        const size_t out_offset = align_up_size(ids_offset + ids_bytes, alignment);
        const size_t arena_bytes = std::max<size_t>(align_up_size(out_offset + out_bytes, alignment), alignment);
        id<MTLBuffer> table_buffer = impl_->resident_buffer_with_bytes(table_key, table.data(), table_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + ids_offset, ids.data(), ids_bytes);
        id<MTLBuffer> width_buf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->embedding_pipeline];
        [enc setBuffer:table_buffer offset:0 atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(ids_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:width_buf offset:0 atIndex:3];
        MTLSize threads = MTLSizeMake(width, ids.size(), 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->embedding_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("embedding_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::semantic_quantize_f32_resident(
    const std::string& in_weight_key,
    const std::vector<float>& in_weight,
    const std::string& in_bias_key,
    const std::vector<float>& in_bias,
    const std::string& codebook_key,
    const std::vector<float>& codebook,
    const std::string& out_weight_key,
    const std::vector<float>& out_weight,
    const std::string& out_bias_key,
    const std::vector<float>& out_bias,
    const std::vector<float>& spk_cond,
    uint32_t tokens,
    std::vector<uint32_t>& codes) {
    if (tokens == 0 ||
        in_weight.size() != 8u * 1024u ||
        in_bias.size() != 8u ||
        codebook.size() != 8192u * 8u ||
        out_weight.size() != 1024u * 8u ||
        out_bias.size() != 1024u ||
        spk_cond.size() != static_cast<size_t>(tokens) * 1024u) {
        throw std::invalid_argument("semantic_quantize_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger in_weight_bytes = in_weight.size() * sizeof(float);
        const NSUInteger in_bias_bytes = in_bias.size() * sizeof(float);
        const NSUInteger codebook_bytes = codebook.size() * sizeof(float);
        const NSUInteger out_weight_bytes = out_weight.size() * sizeof(float);
        const NSUInteger out_bias_bytes = out_bias.size() * sizeof(float);
        const size_t spk_bytes = spk_cond.size() * sizeof(float);
        const size_t sref_count = static_cast<size_t>(tokens) * 1024u;
        const size_t sref_bytes = sref_count * sizeof(float);
        const size_t codes_bytes = static_cast<size_t>(tokens) * sizeof(uint32_t);
        id<MTLBuffer> in_weight_buffer = impl_->resident_buffer_with_bytes(in_weight_key, in_weight.data(), in_weight_bytes);
        id<MTLBuffer> in_bias_buffer = impl_->resident_buffer_with_bytes(in_bias_key, in_bias.data(), in_bias_bytes);
        id<MTLBuffer> codebook_buffer = impl_->resident_buffer_with_bytes(codebook_key, codebook.data(), codebook_bytes);
        id<MTLBuffer> out_weight_buffer = impl_->resident_buffer_with_bytes(out_weight_key, out_weight.data(), out_weight_bytes);
        id<MTLBuffer> out_bias_buffer = impl_->resident_buffer_with_bytes(out_bias_key, out_bias.data(), out_bias_bytes);
        const size_t spk_offset = 0;
        const size_t sref_offset = align_up_size(spk_offset + spk_bytes, alignment);
        const size_t codes_offset = align_up_size(sref_offset + sref_bytes, alignment);
        const size_t arena_bytes = align_up_size(codes_offset + codes_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + spk_offset, spk_cond.data(), spk_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->semantic_quantize_pipeline];
        [enc setBuffer:in_weight_buffer offset:0 atIndex:0];
        [enc setBuffer:in_bias_buffer offset:0 atIndex:1];
        [enc setBuffer:codebook_buffer offset:0 atIndex:2];
        [enc setBuffer:out_weight_buffer offset:0 atIndex:3];
        [enc setBuffer:out_bias_buffer offset:0 atIndex:4];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(spk_offset) atIndex:5];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(sref_offset) atIndex:6];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(codes_offset) atIndex:7];
        [enc setBuffer:tokensbuf offset:0 atIndex:8];
        MTLSize threads = MTLSizeMake(tokens, 1, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(tokens, impl_->semantic_quantize_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("semantic_quantize_f32_resident command failed");
        }
        float* sref_ptr = reinterpret_cast<float*>(base + sref_offset);
        uint32_t* codes_ptr = reinterpret_cast<uint32_t*>(base + codes_offset);
        codes.assign(codes_ptr, codes_ptr + tokens);
        return std::vector<float>(sref_ptr, sref_ptr + sref_count);
    }
}

std::vector<float> MetalContext::layernorm_f32(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta, float eps) {
    if (x.empty() || x.size() != gamma.size() || x.size() != beta.size()) {
        throw std::invalid_argument("layernorm_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t gamma_offset = align_up_size(x_offset + bytes, alignment);
        const size_t beta_offset = align_up_size(gamma_offset + bytes, alignment);
        const size_t out_offset = align_up_size(beta_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        std::memcpy(base + gamma_offset, gamma.data(), bytes);
        std::memcpy(base + beta_offset, beta.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->layernorm_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(gamma_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(beta_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:nbuf offset:0 atIndex:4];
        [enc setBuffer:epsbuf offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("layernorm_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::layernorm_rows_f32(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta, uint32_t tokens, uint32_t width, float eps) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        gamma.size() != width ||
        beta.size() != width) {
        throw std::invalid_argument("layernorm_rows_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t param_bytes = static_cast<size_t>(width) * sizeof(float);
        const size_t x_offset = 0;
        const size_t gamma_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t beta_offset = align_up_size(gamma_offset + param_bytes, alignment);
        const size_t out_offset = align_up_size(beta_offset + param_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + gamma_offset, gamma.data(), param_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + beta_offset, beta.data(), param_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const bool use_serial_rows = width > 1024;
        id<MTLComputePipelineState> pipeline = use_serial_rows ? impl_->layernorm_rows_serial_pipeline : impl_->layernorm_rows_pipeline;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(gamma_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(beta_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:widthbuf offset:0 atIndex:5];
        [enc setBuffer:epsbuf offset:0 atIndex:6];
        if (use_serial_rows) {
            MTLSize threads = MTLSizeMake(tokens, 1, 1);
            MTLSize groups = MTLSizeMake(std::min<uint32_t>(tokens, pipeline.threadExecutionWidth), 1, 1);
            [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("layernorm_rows_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::layernorm_f32_resident(
    const std::string& gamma_key,
    const std::vector<float>& gamma,
    const std::string& beta_key,
    const std::vector<float>& beta,
    const std::vector<float>& x,
    float eps) {
    if (x.empty() || x.size() != gamma.size() || x.size() != beta.size()) {
        throw std::invalid_argument("layernorm_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        id<MTLBuffer> gamma_buffer = impl_->resident_buffer_with_bytes(gamma_key, gamma.data(), bytes);
        id<MTLBuffer> beta_buffer = impl_->resident_buffer_with_bytes(beta_key, beta.data(), bytes);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->layernorm_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:gamma_buffer offset:0 atIndex:1];
        [enc setBuffer:beta_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:nbuf offset:0 atIndex:4];
        [enc setBuffer:epsbuf offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("layernorm_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::layernorm_rows_f32_resident(
    const std::string& gamma_key,
    const std::vector<float>& gamma,
    const std::string& beta_key,
    const std::vector<float>& beta,
    const std::vector<float>& x,
    uint32_t tokens,
    uint32_t width,
    float eps) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        gamma.size() != width ||
        beta.size() != width) {
        throw std::invalid_argument("layernorm_rows_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t param_bytes = static_cast<size_t>(width) * sizeof(float);
        id<MTLBuffer> gamma_buffer = impl_->resident_buffer_with_bytes(gamma_key, gamma.data(), static_cast<NSUInteger>(param_bytes));
        id<MTLBuffer> beta_buffer = impl_->resident_buffer_with_bytes(beta_key, beta.data(), static_cast<NSUInteger>(param_bytes));
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        const bool use_serial_rows = width > 1024;
        id<MTLComputePipelineState> pipeline = use_serial_rows ? impl_->layernorm_rows_serial_pipeline : impl_->layernorm_rows_pipeline;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:gamma_buffer offset:0 atIndex:1];
        [enc setBuffer:beta_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:widthbuf offset:0 atIndex:5];
        [enc setBuffer:epsbuf offset:0 atIndex:6];
        if (use_serial_rows) {
            MTLSize threads = MTLSizeMake(tokens, 1, 1);
            MTLSize groups = MTLSizeMake(std::min<uint32_t>(tokens, pipeline.threadExecutionWidth), 1, 1);
            [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("layernorm_rows_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::adaptive_layernorm_rows_f32(
    const std::vector<float>& x,
    const std::vector<float>& shift,
    const std::vector<float>& scale,
    uint32_t tokens,
    uint32_t width,
    float eps) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        shift.size() != width ||
        scale.size() != width) {
        throw std::invalid_argument("adaptive_layernorm_rows_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t param_bytes = static_cast<size_t>(width) * sizeof(float);
        const size_t x_offset = 0;
        const size_t shift_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t scale_offset = align_up_size(shift_offset + param_bytes, alignment);
        const size_t out_offset = align_up_size(scale_offset + param_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + shift_offset, shift.data(), param_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + scale_offset, scale.data(), param_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->adaptive_layernorm_rows_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(shift_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(scale_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:widthbuf offset:0 atIndex:5];
        [enc setBuffer:epsbuf offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("adaptive_layernorm_rows_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::adaptive_rmsnorm_rows_f32(
    const std::vector<float>& x,
    const std::vector<float>& gamma,
    const std::vector<float>& adaptive_weight,
    const std::vector<float>& adaptive_bias,
    uint32_t tokens,
    uint32_t width,
    float eps) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        gamma.size() != width ||
        adaptive_weight.size() != width ||
        adaptive_bias.size() != width) {
        throw std::invalid_argument("adaptive_rmsnorm_rows_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t param_bytes = static_cast<size_t>(width) * sizeof(float);
        const size_t x_offset = 0;
        const size_t gamma_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t adaptive_weight_offset = align_up_size(gamma_offset + param_bytes, alignment);
        const size_t adaptive_bias_offset = align_up_size(adaptive_weight_offset + param_bytes, alignment);
        const size_t out_offset = align_up_size(adaptive_bias_offset + param_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + gamma_offset, gamma.data(), param_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + adaptive_weight_offset, adaptive_weight.data(), param_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + adaptive_bias_offset, adaptive_bias.data(), param_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->adaptive_rmsnorm_rows_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(gamma_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(adaptive_weight_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(adaptive_bias_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:4];
        [enc setBuffer:tokensbuf offset:0 atIndex:5];
        [enc setBuffer:widthbuf offset:0 atIndex:6];
        [enc setBuffer:epsbuf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("adaptive_rmsnorm_rows_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::adaptive_rmsnorm_rows_f32_resident(
    const std::string& gamma_key,
    const std::vector<float>& gamma,
    const std::vector<float>& x,
    const std::vector<float>& adaptive_weight,
    const std::vector<float>& adaptive_bias,
    uint32_t tokens,
    uint32_t width,
    float eps) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        gamma.size() != width ||
        adaptive_weight.size() != width ||
        adaptive_bias.size() != width) {
        throw std::invalid_argument("adaptive_rmsnorm_rows_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t param_bytes = static_cast<size_t>(width) * sizeof(float);
        const size_t x_offset = 0;
        const size_t adaptive_weight_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t adaptive_bias_offset = align_up_size(adaptive_weight_offset + param_bytes, alignment);
        const size_t out_offset = align_up_size(adaptive_bias_offset + param_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> gamma_buffer = impl_->resident_buffer_with_bytes(gamma_key, gamma.data(), param_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + adaptive_weight_offset, adaptive_weight.data(), param_bytes);
        std::memcpy(base + adaptive_bias_offset, adaptive_bias.data(), param_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->adaptive_rmsnorm_rows_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:gamma_buffer offset:0 atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(adaptive_weight_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(adaptive_bias_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:4];
        [enc setBuffer:tokensbuf offset:0 atIndex:5];
        [enc setBuffer:widthbuf offset:0 atIndex:6];
        [enc setBuffer:epsbuf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("adaptive_rmsnorm_rows_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::cfm_euler_update_f32(
    const std::vector<float>& x,
    const std::vector<float>& dphi,
    const std::vector<float>& cfg_dphi,
    uint32_t tokens,
    uint32_t width,
    uint32_t prompt_tokens,
    float dt,
    float cfg_rate) {
    if (tokens == 0 || width == 0 || prompt_tokens > tokens ||
        x.size() != static_cast<size_t>(tokens) * width ||
        dphi.size() != x.size() ||
        cfg_dphi.size() != x.size()) {
        throw std::invalid_argument("cfm_euler_update_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t dphi_offset = align_up_size(x_offset + bytes, alignment);
        const size_t cfg_dphi_offset = align_up_size(dphi_offset + bytes, alignment);
        const size_t out_offset = align_up_size(cfg_dphi_offset + bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + dphi_offset, dphi.data(), bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + cfg_dphi_offset, cfg_dphi.data(), bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> promptbuf = impl_->constant_buffer_with_bytes(&prompt_tokens, sizeof(prompt_tokens));
        id<MTLBuffer> dtbuf = impl_->constant_buffer_with_bytes(&dt, sizeof(dt));
        id<MTLBuffer> cfgbuf = impl_->constant_buffer_with_bytes(&cfg_rate, sizeof(cfg_rate));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->cfm_euler_update_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(dphi_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(cfg_dphi_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:widthbuf offset:0 atIndex:5];
        [enc setBuffer:promptbuf offset:0 atIndex:6];
        [enc setBuffer:dtbuf offset:0 atIndex:7];
        [enc setBuffer:cfgbuf offset:0 atIndex:8];
        MTLSize threads = MTLSizeMake(width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->cfm_euler_update_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("cfm_euler_update_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::concat_rows_f32(
    const std::vector<float>& a,
    const std::vector<float>& b,
    uint32_t tokens,
    uint32_t a_width,
    uint32_t b_width) {
    if (tokens == 0 || a_width == 0 || b_width == 0 ||
        a.size() != static_cast<size_t>(tokens) * a_width ||
        b.size() != static_cast<size_t>(tokens) * b_width) {
        throw std::invalid_argument("concat_rows_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t a_bytes = a.size() * sizeof(float);
        const size_t b_bytes = b.size() * sizeof(float);
        const uint32_t out_width = a_width + b_width;
        const size_t out_count = static_cast<size_t>(tokens) * out_width;
        const size_t out_bytes = out_count * sizeof(float);
        const size_t a_offset = 0;
        const size_t b_offset = align_up_size(a_offset + a_bytes, alignment);
        const size_t out_offset = align_up_size(b_offset + b_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + a_offset, a.data(), a_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + b_offset, b.data(), b_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> awidthbuf = impl_->constant_buffer_with_bytes(&a_width, sizeof(a_width));
        id<MTLBuffer> bwidthbuf = impl_->constant_buffer_with_bytes(&b_width, sizeof(b_width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->concat_rows_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(a_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(b_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:tokensbuf offset:0 atIndex:3];
        [enc setBuffer:awidthbuf offset:0 atIndex:4];
        [enc setBuffer:bwidthbuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(out_width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_width, impl_->concat_rows_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("concat_rows_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::hot_condition_merge_f32(
    const std::vector<float>& prompt,
    const std::vector<float>& generated,
    uint32_t prompt_tokens,
    uint32_t generated_tokens,
    uint32_t width) {
    if (prompt_tokens == 0 || generated_tokens == 0 || width == 0 ||
        prompt.size() < static_cast<size_t>(prompt_tokens) * width ||
        generated.size() != static_cast<size_t>(generated_tokens) * width) {
        throw std::invalid_argument("hot_condition_merge_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t prompt_values = static_cast<size_t>(prompt_tokens) * width;
        const size_t prompt_bytes = prompt_values * sizeof(float);
        const size_t generated_bytes = generated.size() * sizeof(float);
        const uint32_t tokens = prompt_tokens + generated_tokens;
        const size_t out_count = static_cast<size_t>(tokens) * width;
        const size_t out_bytes = out_count * sizeof(float);
        const size_t prompt_offset = 0;
        const size_t generated_offset = align_up_size(prompt_offset + prompt_bytes, alignment);
        const size_t out_offset = align_up_size(generated_offset + generated_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + prompt_offset, prompt.data(), prompt_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + generated_offset, generated.data(), generated_bytes);
        id<MTLBuffer> prompttokensbuf = impl_->constant_buffer_with_bytes(&prompt_tokens, sizeof(prompt_tokens));
        id<MTLBuffer> generatedtokensbuf = impl_->constant_buffer_with_bytes(&generated_tokens, sizeof(generated_tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->hot_condition_merge_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(prompt_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(generated_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:prompttokensbuf offset:0 atIndex:3];
        [enc setBuffer:generatedtokensbuf offset:0 atIndex:4];
        [enc setBuffer:widthbuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->hot_condition_merge_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("hot_condition_merge_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::dit_input_merge_f32(
    const std::vector<float>& x,
    const std::vector<float>& prompt_x,
    const std::vector<float>& cond_proj,
    const std::vector<float>& style,
    uint32_t tokens) {
    constexpr uint32_t mel_width = 80;
    constexpr uint32_t cond_width = 512;
    constexpr uint32_t style_width = 192;
    constexpr uint32_t out_width = mel_width + mel_width + cond_width + style_width;
    if (tokens == 0 ||
        x.size() != static_cast<size_t>(tokens) * mel_width ||
        prompt_x.size() != static_cast<size_t>(tokens) * mel_width ||
        cond_proj.size() != static_cast<size_t>(tokens) * cond_width ||
        style.size() != style_width) {
        throw std::invalid_argument("dit_input_merge_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t prompt_bytes = prompt_x.size() * sizeof(float);
        const size_t cond_bytes = cond_proj.size() * sizeof(float);
        const size_t style_bytes = style.size() * sizeof(float);
        const size_t out_count = static_cast<size_t>(tokens) * out_width;
        const size_t out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t prompt_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t cond_offset = align_up_size(prompt_offset + prompt_bytes, alignment);
        const size_t style_offset = align_up_size(cond_offset + cond_bytes, alignment);
        const size_t out_offset = align_up_size(style_offset + style_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + prompt_offset, prompt_x.data(), prompt_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + cond_offset, cond_proj.data(), cond_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + style_offset, style.data(), style_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->dit_input_merge_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(prompt_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(cond_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(style_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:4];
        [enc setBuffer:tokensbuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(out_width, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_width, impl_->dit_input_merge_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("dit_input_merge_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::dit_input_merge_batched_f32(
    const std::vector<float>& x,
    const std::vector<float>& prompt_x,
    const std::vector<float>& cond_proj,
    const std::vector<float>& style,
    uint32_t batch,
    uint32_t tokens) {
    constexpr uint32_t mel_width = 80;
    constexpr uint32_t cond_width = 512;
    constexpr uint32_t style_width = 192;
    constexpr uint32_t out_width = mel_width + mel_width + cond_width + style_width;
    const uint32_t rows = batch * tokens;
    if (batch == 0 || tokens == 0 ||
        x.size() != static_cast<size_t>(rows) * mel_width ||
        prompt_x.size() != static_cast<size_t>(rows) * mel_width ||
        cond_proj.size() != static_cast<size_t>(rows) * cond_width ||
        style.size() != static_cast<size_t>(batch) * style_width) {
        throw std::invalid_argument("dit_input_merge_batched_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t prompt_bytes = prompt_x.size() * sizeof(float);
        const size_t cond_bytes = cond_proj.size() * sizeof(float);
        const size_t style_bytes = style.size() * sizeof(float);
        const size_t out_count = static_cast<size_t>(rows) * out_width;
        const size_t out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t prompt_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t cond_offset = align_up_size(prompt_offset + prompt_bytes, alignment);
        const size_t style_offset = align_up_size(cond_offset + cond_bytes, alignment);
        const size_t out_offset = align_up_size(style_offset + style_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + prompt_offset, prompt_x.data(), prompt_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + cond_offset, cond_proj.data(), cond_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + style_offset, style.data(), style_bytes);
        id<MTLBuffer> batchbuf = impl_->constant_buffer_with_bytes(&batch, sizeof(batch));
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->dit_input_merge_batched_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(prompt_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(cond_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(style_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:4];
        [enc setBuffer:batchbuf offset:0 atIndex:5];
        [enc setBuffer:tokensbuf offset:0 atIndex:6];
        MTLSize threads = MTLSizeMake(out_width, rows, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_width, impl_->dit_input_merge_batched_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("dit_input_merge_batched_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::rmsnorm_f32(const std::vector<float>& x, const std::vector<float>& gamma, float eps) {
    if (x.empty() || x.size() != gamma.size()) {
        throw std::invalid_argument("rmsnorm_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t gamma_offset = align_up_size(x_offset + bytes, alignment);
        const size_t out_offset = align_up_size(gamma_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        std::memcpy(base + gamma_offset, gamma.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->rmsnorm_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(gamma_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:nbuf offset:0 atIndex:3];
        [enc setBuffer:epsbuf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("rmsnorm_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::rmsnorm_rows_f32(const std::vector<float>& x, const std::vector<float>& gamma, uint32_t tokens, uint32_t width) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        gamma.size() != width) {
        throw std::invalid_argument("rmsnorm_rows_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t param_bytes = static_cast<size_t>(width) * sizeof(float);
        const size_t x_offset = 0;
        const size_t gamma_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t out_offset = align_up_size(gamma_offset + param_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + gamma_offset, gamma.data(), param_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->rmsnorm_rows_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(gamma_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:tokensbuf offset:0 atIndex:3];
        [enc setBuffer:widthbuf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("rmsnorm_rows_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::rmsnorm_rows_f32_resident(
    const std::string& gamma_key,
    const std::vector<float>& gamma,
    const std::vector<float>& x,
    uint32_t tokens,
    uint32_t width) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        gamma.size() != width) {
        throw std::invalid_argument("rmsnorm_rows_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t param_bytes = static_cast<size_t>(width) * sizeof(float);
        id<MTLBuffer> gamma_buffer = impl_->resident_buffer_with_bytes(gamma_key, gamma.data(), static_cast<NSUInteger>(param_bytes));
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->rmsnorm_rows_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:gamma_buffer offset:0 atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:tokensbuf offset:0 atIndex:3];
        [enc setBuffer:widthbuf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("rmsnorm_rows_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::rmsnorm_rows_eps_f32(const std::vector<float>& x, const std::vector<float>& gamma, uint32_t tokens, uint32_t width, float eps) {
    if (tokens == 0 || width == 0 ||
        x.size() != static_cast<size_t>(tokens) * width ||
        gamma.size() != width) {
        throw std::invalid_argument("rmsnorm_rows_eps_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t param_bytes = static_cast<size_t>(width) * sizeof(float);
        const size_t x_offset = 0;
        const size_t gamma_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t out_offset = align_up_size(gamma_offset + param_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + gamma_offset, gamma.data(), param_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->rmsnorm_rows_eps_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(gamma_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:tokensbuf offset:0 atIndex:3];
        [enc setBuffer:widthbuf offset:0 atIndex:4];
        [enc setBuffer:epsbuf offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("rmsnorm_rows_eps_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::linear_f32(const std::vector<float>& weight, const std::vector<float>& bias, const std::vector<float>& x, uint32_t rows, uint32_t cols) {
    if (weight.size() != static_cast<size_t>(rows) * cols || bias.size() != rows || x.size() != cols) {
        throw std::invalid_argument("linear_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger weight_bytes = weight.size() * sizeof(float);
        const NSUInteger bias_bytes = bias.size() * sizeof(float);
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger out_bytes = rows * sizeof(float);
        const size_t weight_offset = 0;
        const size_t bias_offset = align_up_size(weight_offset + weight_bytes, alignment);
        const size_t x_offset = align_up_size(bias_offset + bias_bytes, alignment);
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + weight_offset, weight.data(), weight_bytes);
        std::memcpy(base + bias_offset, bias.data(), bias_bytes);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> rowsbuf = impl_->constant_buffer_with_bytes(&rows, sizeof(rows));
        id<MTLBuffer> colsbuf = impl_->constant_buffer_with_bytes(&cols, sizeof(cols));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->linear_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:rowsbuf offset:0 atIndex:4];
        [enc setBuffer:colsbuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(rows, 1, 1);
        MTLSize groups = MTLSizeMake(impl_->linear_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("linear_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + rows);
    }
}

std::vector<float> MetalContext::linear_rows_f32(const std::vector<float>& weight, const std::vector<float>& bias, const std::vector<float>& x, uint32_t tokens, uint32_t rows, uint32_t cols) {
    if (tokens == 0 || rows == 0 || cols == 0 ||
        weight.size() != static_cast<size_t>(rows) * cols ||
        bias.size() != rows ||
        x.size() != static_cast<size_t>(tokens) * cols) {
        throw std::invalid_argument("linear_rows_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t weight_bytes = weight.size() * sizeof(float);
        const size_t bias_bytes = bias.size() * sizeof(float);
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t out_count = static_cast<size_t>(tokens) * rows;
        const size_t out_bytes = out_count * sizeof(float);
        const size_t weight_offset = 0;
        const size_t bias_offset = align_up_size(weight_offset + weight_bytes, alignment);
        const size_t x_offset = align_up_size(bias_offset + bias_bytes, alignment);
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + weight_offset, weight.data(), weight_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + bias_offset, bias.data(), bias_bytes);
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> rowsbuf = impl_->constant_buffer_with_bytes(&rows, sizeof(rows));
        id<MTLBuffer> colsbuf = impl_->constant_buffer_with_bytes(&cols, sizeof(cols));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->linear_rows_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:rowsbuf offset:0 atIndex:5];
        [enc setBuffer:colsbuf offset:0 atIndex:6];
        MTLSize threads = MTLSizeMake(rows, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(rows, impl_->linear_rows_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("linear_rows_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::linear_f32_resident(
    const std::string& weight_key,
    const std::vector<float>& weight,
    const std::string& bias_key,
    const std::vector<float>& bias,
    const std::vector<float>& x,
    uint32_t rows,
    uint32_t cols) {
    if (weight.size() != static_cast<size_t>(rows) * cols || bias.size() != rows || x.size() != cols) {
        throw std::invalid_argument("linear_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger weight_bytes = weight.size() * sizeof(float);
        const NSUInteger bias_bytes = bias.size() * sizeof(float);
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger out_bytes = rows * sizeof(float);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), weight_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), bias_bytes);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> rowsbuf = impl_->constant_buffer_with_bytes(&rows, sizeof(rows));
        id<MTLBuffer> colsbuf = impl_->constant_buffer_with_bytes(&cols, sizeof(cols));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->linear_pipeline];
        [enc setBuffer:weight_buffer offset:0 atIndex:0];
        [enc setBuffer:bias_buffer offset:0 atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:rowsbuf offset:0 atIndex:4];
        [enc setBuffer:colsbuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(rows, 1, 1);
        MTLSize groups = MTLSizeMake(impl_->linear_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("linear_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + rows);
    }
}

std::vector<float> MetalContext::linear_rows_f32_resident(
    const std::string& weight_key,
    const std::vector<float>& weight,
    const std::string& bias_key,
    const std::vector<float>& bias,
    const std::vector<float>& x,
    uint32_t tokens,
    uint32_t rows,
    uint32_t cols) {
    if (tokens == 0 || rows == 0 || cols == 0 ||
        weight.size() != static_cast<size_t>(rows) * cols ||
        bias.size() != rows ||
        x.size() != static_cast<size_t>(tokens) * cols) {
        throw std::invalid_argument("linear_rows_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const size_t weight_bytes = weight.size() * sizeof(float);
        const size_t bias_bytes = bias.size() * sizeof(float);
        const size_t x_bytes = x.size() * sizeof(float);
        const size_t out_count = static_cast<size_t>(tokens) * rows;
        const size_t out_bytes = out_count * sizeof(float);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), static_cast<NSUInteger>(weight_bytes));
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), static_cast<NSUInteger>(bias_bytes));
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = align_up_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        std::memcpy(static_cast<uint8_t*>([arena contents]) + x_offset, x.data(), x_bytes);
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        if (linear_rows_use_mps(tokens, rows, cols)) {
            const bool bias_zero = impl_->bias_is_zero_for(bias_key, bias);
            if (!bias_zero) {
                id<MTLComputeCommandEncoder> benc = [cb computeCommandEncoder];
                uint32_t t = tokens, r = rows;
                [benc setComputePipelineState:impl_->broadcast_bias_rows_pipeline];
                [benc setBuffer:bias_buffer offset:0 atIndex:0];
                [benc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
                [benc setBytes:&t length:sizeof(t) atIndex:2];
                [benc setBytes:&r length:sizeof(r) atIndex:3];
                [benc dispatchThreads:MTLSizeMake(rows, tokens, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(rows, impl_->broadcast_bias_rows_pipeline.threadExecutionWidth), 1, 1)];
                [benc endEncoding];
            }
            bool f16w = false;
            id<MTLBuffer> gemm_weight = impl_->weight_buffer_pref_f16(weight_key, weight, fp16_weights_enabled(), f16w);
            MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:tokens columns:cols rowBytes:static_cast<NSUInteger>(cols) * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:static_cast<NSUInteger>(cols) * (f16w ? sizeof(uint16_t) : sizeof(float)) dataType:(f16w ? MPSDataTypeFloat16 : MPSDataTypeFloat32)];
            MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:tokens columns:rows rowBytes:static_cast<NSUInteger>(rows) * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrix* A = [[MPSMatrix alloc] initWithBuffer:arena offset:x_offset descriptor:dA];
            MPSMatrix* B = [[MPSMatrix alloc] initWithBuffer:gemm_weight offset:0 descriptor:dB];
            MPSMatrix* C = [[MPSMatrix alloc] initWithBuffer:arena offset:out_offset descriptor:dC];
            [impl_->mps_gemm(tokens, rows, cols, !bias_zero) encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
            commit_and_count(impl_->command_buffers_submitted, cb);
            wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
            if ([cb status] != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error(linear_rows_resident_failure_message("mps",
                                                                              weight_key,
                                                                              bias_key,
                                                                              tokens,
                                                                              rows,
                                                                              cols,
                                                                              weight_bytes,
                                                                              bias_bytes,
                                                                              x_bytes,
                                                                              out_bytes,
                                                                              arena_bytes,
                                                                              command_buffer_error_message(cb)));
            }
            float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
            return std::vector<float>(ptr, ptr + out_count);
        }
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> rowsbuf = impl_->constant_buffer_with_bytes(&rows, sizeof(rows));
        id<MTLBuffer> colsbuf = impl_->constant_buffer_with_bytes(&cols, sizeof(cols));
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->linear_rows_pipeline];
        [enc setBuffer:weight_buffer offset:0 atIndex:0];
        [enc setBuffer:bias_buffer offset:0 atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:rowsbuf offset:0 atIndex:5];
        [enc setBuffer:colsbuf offset:0 atIndex:6];
        MTLSize threads = MTLSizeMake(rows, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(rows, impl_->linear_rows_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error(linear_rows_resident_failure_message("compute",
                                                                          weight_key,
                                                                          bias_key,
                                                                          tokens,
                                                                          rows,
                                                                          cols,
                                                                          weight_bytes,
                                                                          bias_bytes,
                                                                          x_bytes,
                                                                          out_bytes,
                                                                          arena_bytes,
                                                                          command_buffer_error_message(cb)));
        }
        float* ptr = reinterpret_cast<float*>(static_cast<uint8_t*>([arena contents]) + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::nearest_interpolate_f32(const std::vector<float>& x, uint32_t in_tokens, uint32_t out_tokens, uint32_t width) {
    if (in_tokens == 0 || out_tokens == 0 || width == 0 || x.size() != static_cast<size_t>(in_tokens) * width) {
        throw std::invalid_argument("nearest_interpolate_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger in_bytes = x.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(out_tokens) * width;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + in_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), in_bytes);
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_tokens, sizeof(in_tokens));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_tokens, sizeof(out_tokens));
        id<MTLBuffer> widthbuf = impl_->constant_buffer_with_bytes(&width, sizeof(width));
        const float scale = static_cast<float>(in_tokens) / static_cast<float>(out_tokens);
        id<MTLBuffer> scalebuf = impl_->constant_buffer_with_bytes(&scale, sizeof(scale));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->nearest_interpolate_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:inbuf offset:0 atIndex:2];
        [enc setBuffer:outbuf offset:0 atIndex:3];
        [enc setBuffer:widthbuf offset:0 atIndex:4];
        [enc setBuffer:scalebuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(width, out_tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(width, impl_->nearest_interpolate_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("nearest_interpolate_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv1d_same_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel) {
    if (tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(tokens) * in_channels ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv1d_same_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t weight_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t bias_offset = align_up_size(weight_offset + w_bytes, alignment);
        const size_t out_offset = align_up_size(bias_offset + b_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + weight_offset, weight.data(), w_bytes);
        std::memcpy(base + bias_offset, bias.data(), b_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv1d_same_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:inbuf offset:0 atIndex:5];
        [enc setBuffer:outbuf offset:0 atIndex:6];
        [enc setBuffer:kbuf offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(out_channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv1d_same_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv1d_same_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv1d_same_f32_resident(const std::string& weight_key,
                                                          const std::vector<float>& weight,
                                                          const std::string& bias_key,
                                                          const std::vector<float>& bias,
                                                          const std::vector<float>& x,
                                                          uint32_t tokens,
                                                          uint32_t in_channels,
                                                          uint32_t out_channels,
                                                          uint32_t kernel) {
    if (tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(tokens) * in_channels ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv1d_same_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), w_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), b_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv1d_same_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:weight_buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:inbuf offset:0 atIndex:5];
        [enc setBuffer:outbuf offset:0 atIndex:6];
        [enc setBuffer:kbuf offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(out_channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv1d_same_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv1d_same_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv1d_reflect_same_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel) {
    if (tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(tokens) * in_channels ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv1d_reflect_same_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t weight_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t bias_offset = align_up_size(weight_offset + w_bytes, alignment);
        const size_t out_offset = align_up_size(bias_offset + b_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + weight_offset, weight.data(), w_bytes);
        std::memcpy(base + bias_offset, bias.data(), b_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv1d_reflect_same_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:inbuf offset:0 atIndex:5];
        [enc setBuffer:outbuf offset:0 atIndex:6];
        [enc setBuffer:kbuf offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(out_channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv1d_reflect_same_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv1d_reflect_same_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv1d_reflect_same_f32_resident(const std::string& weight_key,
                                                                  const std::vector<float>& weight,
                                                                  const std::string& bias_key,
                                                                  const std::vector<float>& bias,
                                                                  const std::vector<float>& x,
                                                                  uint32_t tokens,
                                                                  uint32_t in_channels,
                                                                  uint32_t out_channels,
                                                                  uint32_t kernel) {
    if (tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(tokens) * in_channels ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv1d_reflect_same_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), w_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), b_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv1d_reflect_same_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:weight_buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:inbuf offset:0 atIndex:5];
        [enc setBuffer:outbuf offset:0 atIndex:6];
        [enc setBuffer:kbuf offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(out_channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv1d_reflect_same_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv1d_reflect_same_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv1d_reflect_same_batched_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t batch, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel) {
    if (batch == 0 || tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(batch) * tokens * in_channels ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv1d_reflect_same_batched_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(batch) * tokens * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t weight_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t bias_offset = align_up_size(weight_offset + w_bytes, alignment);
        const size_t out_offset = align_up_size(bias_offset + b_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + weight_offset, weight.data(), w_bytes);
        std::memcpy(base + bias_offset, bias.data(), b_bytes);
        id<MTLBuffer> batchbuf = impl_->constant_buffer_with_bytes(&batch, sizeof(batch));
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv1d_reflect_same_batched_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:batchbuf offset:0 atIndex:4];
        [enc setBuffer:tokensbuf offset:0 atIndex:5];
        [enc setBuffer:inbuf offset:0 atIndex:6];
        [enc setBuffer:outbuf offset:0 atIndex:7];
        [enc setBuffer:kbuf offset:0 atIndex:8];
        MTLSize threads = MTLSizeMake(out_channels, tokens, batch);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv1d_reflect_same_batched_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv1d_reflect_same_batched_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv1d_reflect_same_batched_f32_resident(const std::string& weight_key,
                                                                          const std::vector<float>& weight,
                                                                          const std::string& bias_key,
                                                                          const std::vector<float>& bias,
                                                                          const std::vector<float>& x,
                                                                          uint32_t batch,
                                                                          uint32_t tokens,
                                                                          uint32_t in_channels,
                                                                          uint32_t out_channels,
                                                                          uint32_t kernel) {
    if (batch == 0 || tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(batch) * tokens * in_channels ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv1d_reflect_same_batched_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(batch) * tokens * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), w_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), b_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> batchbuf = impl_->constant_buffer_with_bytes(&batch, sizeof(batch));
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv1d_reflect_same_batched_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:weight_buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:batchbuf offset:0 atIndex:4];
        [enc setBuffer:tokensbuf offset:0 atIndex:5];
        [enc setBuffer:inbuf offset:0 atIndex:6];
        [enc setBuffer:outbuf offset:0 atIndex:7];
        [enc setBuffer:kbuf offset:0 atIndex:8];
        MTLSize threads = MTLSizeMake(out_channels, tokens, batch);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv1d_reflect_same_batched_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv1d_reflect_same_batched_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::depthwise_conv1d_same_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t channels, uint32_t kernel) {
    if (tokens == 0 || channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(tokens) * channels ||
        weight.size() != static_cast<size_t>(channels) * kernel ||
        bias.size() != channels) {
        throw std::invalid_argument("depthwise_conv1d_same_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t weight_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t bias_offset = align_up_size(weight_offset + w_bytes, alignment);
        const size_t out_offset = align_up_size(bias_offset + b_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + weight_offset, weight.data(), w_bytes);
        std::memcpy(base + bias_offset, bias.data(), b_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&channels, sizeof(channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->depthwise_conv1d_same_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:channelsbuf offset:0 atIndex:5];
        [enc setBuffer:kbuf offset:0 atIndex:6];
        MTLSize threads = MTLSizeMake(channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(channels, impl_->depthwise_conv1d_same_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("depthwise_conv1d_same_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::depthwise_conv1d_same_f32_resident(
    const std::string& weight_key,
    const std::vector<float>& weight,
    const std::string& bias_key,
    const std::vector<float>& bias,
    const std::vector<float>& x,
    uint32_t tokens,
    uint32_t channels,
    uint32_t kernel) {
    if (tokens == 0 || channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(tokens) * channels ||
        weight.size() != static_cast<size_t>(channels) * kernel ||
        bias.size() != channels) {
        throw std::invalid_argument("depthwise_conv1d_same_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), w_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), b_bytes);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&channels, sizeof(channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->depthwise_conv1d_same_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:weight_buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:channelsbuf offset:0 atIndex:5];
        [enc setBuffer:kbuf offset:0 atIndex:6];
        MTLSize threads = MTLSizeMake(channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(channels, impl_->depthwise_conv1d_same_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("depthwise_conv1d_same_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::depthwise_conv1d_causal_f32_resident(
    const std::string& weight_key,
    const std::vector<float>& weight,
    const std::string& bias_key,
    const std::vector<float>& bias,
    const std::vector<float>& x,
    uint32_t tokens,
    uint32_t channels,
    uint32_t kernel) {
    if (tokens == 0 || channels == 0 || kernel == 0 ||
        x.size() != static_cast<size_t>(tokens) * channels ||
        weight.size() != static_cast<size_t>(channels) * kernel ||
        bias.size() != channels) {
        throw std::invalid_argument("depthwise_conv1d_causal_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), w_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), b_bytes);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&channels, sizeof(channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->depthwise_conv1d_causal_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:weight_buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:channelsbuf offset:0 atIndex:5];
        [enc setBuffer:kbuf offset:0 atIndex:6];
        MTLSize threads = MTLSizeMake(channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(channels, impl_->depthwise_conv1d_causal_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("depthwise_conv1d_causal_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::subsampling_conv2d_relu_flat_f32(const std::vector<float>& x,
                                                                  const std::vector<float>& weight,
                                                                  const std::vector<float>& bias,
                                                                  uint32_t input_tokens,
                                                                  uint32_t input_dim,
                                                                  uint32_t out_channels,
                                                                  uint32_t kernel,
                                                                  uint32_t stride) {
    if (input_tokens < kernel || input_dim < kernel || out_channels == 0 || kernel == 0 || stride == 0 ||
        x.size() != static_cast<size_t>(input_tokens) * input_dim ||
        weight.size() != static_cast<size_t>(out_channels) * kernel * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("subsampling_conv2d_relu_flat_f32 invalid input sizes");
    }
    const uint32_t output_tokens = ((input_tokens - kernel) / stride) + 1;
    const uint32_t output_freq = ((input_dim - kernel) / stride) + 1;
    const uint32_t flat_dim = out_channels * output_freq;
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(output_tokens) * flat_dim;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t weight_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t bias_offset = align_up_size(weight_offset + w_bytes, alignment);
        const size_t out_offset = align_up_size(bias_offset + b_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + weight_offset, weight.data(), w_bytes);
        std::memcpy(base + bias_offset, bias.data(), b_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&input_tokens, sizeof(input_tokens));
        id<MTLBuffer> indimbuf = impl_->constant_buffer_with_bytes(&input_dim, sizeof(input_dim));
        id<MTLBuffer> outtokensbuf = impl_->constant_buffer_with_bytes(&output_tokens, sizeof(output_tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> freqbuf = impl_->constant_buffer_with_bytes(&output_freq, sizeof(output_freq));
        id<MTLBuffer> kernelbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLBuffer> stridebuf = impl_->constant_buffer_with_bytes(&stride, sizeof(stride));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->subsampling_conv2d_relu_flat_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:indimbuf offset:0 atIndex:5];
        [enc setBuffer:outtokensbuf offset:0 atIndex:6];
        [enc setBuffer:channelsbuf offset:0 atIndex:7];
        [enc setBuffer:freqbuf offset:0 atIndex:8];
        [enc setBuffer:kernelbuf offset:0 atIndex:9];
        [enc setBuffer:stridebuf offset:0 atIndex:10];
        MTLSize threads = MTLSizeMake(flat_dim, output_tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(flat_dim, impl_->subsampling_conv2d_relu_flat_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("subsampling_conv2d_relu_flat_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::subsampling_conv2d_relu_flat_f32_resident(const std::string& weight_key,
                                                                           const std::vector<float>& weight,
                                                                           const std::string& bias_key,
                                                                           const std::vector<float>& bias,
                                                                           const std::vector<float>& x,
                                                                           uint32_t input_tokens,
                                                                           uint32_t input_dim,
                                                                           uint32_t out_channels,
                                                                           uint32_t kernel,
                                                                           uint32_t stride) {
    if (input_tokens < kernel || input_dim < kernel || out_channels == 0 || kernel == 0 || stride == 0 ||
        x.size() != static_cast<size_t>(input_tokens) * input_dim ||
        weight.size() != static_cast<size_t>(out_channels) * kernel * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("subsampling_conv2d_relu_flat_f32_resident invalid input sizes");
    }
    const uint32_t output_tokens = ((input_tokens - kernel) / stride) + 1;
    const uint32_t output_freq = ((input_dim - kernel) / stride) + 1;
    const uint32_t flat_dim = out_channels * output_freq;
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(output_tokens) * flat_dim;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), w_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), b_bytes);
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&input_tokens, sizeof(input_tokens));
        id<MTLBuffer> indimbuf = impl_->constant_buffer_with_bytes(&input_dim, sizeof(input_dim));
        id<MTLBuffer> outtokensbuf = impl_->constant_buffer_with_bytes(&output_tokens, sizeof(output_tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> freqbuf = impl_->constant_buffer_with_bytes(&output_freq, sizeof(output_freq));
        id<MTLBuffer> kernelbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLBuffer> stridebuf = impl_->constant_buffer_with_bytes(&stride, sizeof(stride));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->subsampling_conv2d_relu_flat_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:weight_buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:indimbuf offset:0 atIndex:5];
        [enc setBuffer:outtokensbuf offset:0 atIndex:6];
        [enc setBuffer:channelsbuf offset:0 atIndex:7];
        [enc setBuffer:freqbuf offset:0 atIndex:8];
        [enc setBuffer:kernelbuf offset:0 atIndex:9];
        [enc setBuffer:stridebuf offset:0 atIndex:10];
        MTLSize threads = MTLSizeMake(flat_dim, output_tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(flat_dim, impl_->subsampling_conv2d_relu_flat_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("subsampling_conv2d_relu_flat_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv1d_dilated_same_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t dilation) {
    if (tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 || dilation == 0 ||
        x.size() != static_cast<size_t>(tokens) * in_channels ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv1d_dilated_same_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t weight_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t bias_offset = align_up_size(weight_offset + w_bytes, alignment);
        const size_t out_offset = align_up_size(bias_offset + b_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + weight_offset, weight.data(), w_bytes);
        std::memcpy(base + bias_offset, bias.data(), b_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLBuffer> dilationbuf = impl_->constant_buffer_with_bytes(&dilation, sizeof(dilation));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv1d_dilated_same_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:inbuf offset:0 atIndex:5];
        [enc setBuffer:outbuf offset:0 atIndex:6];
        [enc setBuffer:kbuf offset:0 atIndex:7];
        [enc setBuffer:dilationbuf offset:0 atIndex:8];
        MTLSize threads = MTLSizeMake(out_channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv1d_dilated_same_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv1d_dilated_same_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv1d_dilated_same_f32_resident(const std::string& weight_key,
                                                                  const std::vector<float>& weight,
                                                                  const std::string& bias_key,
                                                                  const std::vector<float>& bias,
                                                                  const std::vector<float>& x,
                                                                  uint32_t tokens,
                                                                  uint32_t in_channels,
                                                                  uint32_t out_channels,
                                                                  uint32_t kernel,
                                                                  uint32_t dilation) {
    if (tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 || dilation == 0 ||
        x.size() != static_cast<size_t>(tokens) * in_channels ||
        weight.size() != static_cast<size_t>(out_channels) * in_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv1d_dilated_same_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(tokens) * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), w_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), b_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLBuffer> dilationbuf = impl_->constant_buffer_with_bytes(&dilation, sizeof(dilation));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv1d_dilated_same_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:weight_buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:inbuf offset:0 atIndex:5];
        [enc setBuffer:outbuf offset:0 atIndex:6];
        [enc setBuffer:kbuf offset:0 atIndex:7];
        [enc setBuffer:dilationbuf offset:0 atIndex:8];
        MTLSize threads = MTLSizeMake(out_channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv1d_dilated_same_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv1d_dilated_same_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv_transpose1d_f32(const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias, uint32_t tokens, uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t stride, uint32_t padding) {
    if (tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 || stride == 0 ||
        x.size() != static_cast<size_t>(tokens) * in_channels ||
        weight.size() != static_cast<size_t>(in_channels) * out_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv_transpose1d_f32 invalid input sizes");
    }
    const uint32_t out_tokens = (tokens - 1) * stride + kernel - 2 * padding;
    if (out_tokens == 0) {
        throw std::invalid_argument("conv_transpose1d_f32 invalid output size");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(out_tokens) * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t weight_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t bias_offset = align_up_size(weight_offset + w_bytes, alignment);
        const size_t out_offset = align_up_size(bias_offset + b_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + weight_offset, weight.data(), w_bytes);
        std::memcpy(base + bias_offset, bias.data(), b_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> outtokensbuf = impl_->constant_buffer_with_bytes(&out_tokens, sizeof(out_tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLBuffer> stridebuf = impl_->constant_buffer_with_bytes(&stride, sizeof(stride));
        id<MTLBuffer> padbuf = impl_->constant_buffer_with_bytes(&padding, sizeof(padding));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv_transpose1d_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(weight_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:outtokensbuf offset:0 atIndex:5];
        [enc setBuffer:inbuf offset:0 atIndex:6];
        [enc setBuffer:outbuf offset:0 atIndex:7];
        [enc setBuffer:kbuf offset:0 atIndex:8];
        [enc setBuffer:stridebuf offset:0 atIndex:9];
        [enc setBuffer:padbuf offset:0 atIndex:10];
        MTLSize threads = MTLSizeMake(out_channels, out_tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv_transpose1d_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv_transpose1d_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::conv_transpose1d_f32_resident(const std::string& weight_key,
                                                               const std::vector<float>& weight,
                                                               const std::string& bias_key,
                                                               const std::vector<float>& bias,
                                                               const std::vector<float>& x,
                                                               uint32_t tokens,
                                                               uint32_t in_channels,
                                                               uint32_t out_channels,
                                                               uint32_t kernel,
                                                               uint32_t stride,
                                                               uint32_t padding) {
    if (tokens == 0 || in_channels == 0 || out_channels == 0 || kernel == 0 || stride == 0 ||
        x.size() != static_cast<size_t>(tokens) * in_channels ||
        weight.size() != static_cast<size_t>(in_channels) * out_channels * kernel ||
        bias.size() != out_channels) {
        throw std::invalid_argument("conv_transpose1d_f32_resident invalid input sizes");
    }
    const uint32_t out_tokens = (tokens - 1) * stride + kernel - 2 * padding;
    if (out_tokens == 0) {
        throw std::invalid_argument("conv_transpose1d_f32_resident invalid output size");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger w_bytes = weight.size() * sizeof(float);
        const NSUInteger b_bytes = bias.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(out_tokens) * out_channels;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> weight_buffer = impl_->resident_buffer_with_bytes(weight_key, weight.data(), w_bytes);
        id<MTLBuffer> bias_buffer = impl_->resident_buffer_with_bytes(bias_key, bias.data(), b_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> outtokensbuf = impl_->constant_buffer_with_bytes(&out_tokens, sizeof(out_tokens));
        id<MTLBuffer> inbuf = impl_->constant_buffer_with_bytes(&in_channels, sizeof(in_channels));
        id<MTLBuffer> outbuf = impl_->constant_buffer_with_bytes(&out_channels, sizeof(out_channels));
        id<MTLBuffer> kbuf = impl_->constant_buffer_with_bytes(&kernel, sizeof(kernel));
        id<MTLBuffer> stridebuf = impl_->constant_buffer_with_bytes(&stride, sizeof(stride));
        id<MTLBuffer> padbuf = impl_->constant_buffer_with_bytes(&padding, sizeof(padding));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conv_transpose1d_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:weight_buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:outtokensbuf offset:0 atIndex:5];
        [enc setBuffer:inbuf offset:0 atIndex:6];
        [enc setBuffer:outbuf offset:0 atIndex:7];
        [enc setBuffer:kbuf offset:0 atIndex:8];
        [enc setBuffer:stridebuf offset:0 atIndex:9];
        [enc setBuffer:padbuf offset:0 atIndex:10];
        MTLSize threads = MTLSizeMake(out_channels, out_tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(out_channels, impl_->conv_transpose1d_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conv_transpose1d_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::bigvgan_activation_f32(const std::vector<float>& x, const std::vector<float>& up_filter, const std::vector<float>& down_filter, const std::vector<float>& alpha_log, const std::vector<float>& beta_log, uint32_t tokens, uint32_t channels) {
    if (tokens == 0 || channels == 0 || x.size() != static_cast<size_t>(tokens) * channels ||
        up_filter.size() != 12 || down_filter.size() != 12 || alpha_log.size() != channels || beta_log.size() != channels) {
        throw std::invalid_argument("bigvgan_activation_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger filter_bytes = 12 * sizeof(float);
        const NSUInteger param_bytes = channels * sizeof(float);
        const size_t x_offset = 0;
        const size_t up_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t down_offset = align_up_size(up_offset + filter_bytes, alignment);
        const size_t alpha_offset = align_up_size(down_offset + filter_bytes, alignment);
        const size_t beta_offset = align_up_size(alpha_offset + param_bytes, alignment);
        const size_t out_offset = align_up_size(beta_offset + param_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + up_offset, up_filter.data(), filter_bytes);
        std::memcpy(base + down_offset, down_filter.data(), filter_bytes);
        std::memcpy(base + alpha_offset, alpha_log.data(), param_bytes);
        std::memcpy(base + beta_offset, beta_log.data(), param_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&channels, sizeof(channels));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->bigvgan_activation_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(up_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(down_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(alpha_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(beta_offset) atIndex:4];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:5];
        [enc setBuffer:tokensbuf offset:0 atIndex:6];
        [enc setBuffer:channelsbuf offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(channels, impl_->bigvgan_activation_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("bigvgan_activation_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::bigvgan_activation_f32_resident(const std::string& up_filter_key,
                                                                 const std::vector<float>& up_filter,
                                                                 const std::string& down_filter_key,
                                                                 const std::vector<float>& down_filter,
                                                                 const std::string& alpha_key,
                                                                 const std::vector<float>& alpha_log,
                                                                 const std::string& beta_key,
                                                                 const std::vector<float>& beta_log,
                                                                 const std::vector<float>& x,
                                                                 uint32_t tokens,
                                                                 uint32_t channels) {
    if (tokens == 0 || channels == 0 || x.size() != static_cast<size_t>(tokens) * channels ||
        up_filter.size() != 12 || down_filter.size() != 12 || alpha_log.size() != channels || beta_log.size() != channels) {
        throw std::invalid_argument("bigvgan_activation_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger filter_bytes = 12 * sizeof(float);
        const NSUInteger param_bytes = channels * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> up_buffer = impl_->resident_buffer_with_bytes(up_filter_key, up_filter.data(), filter_bytes);
        id<MTLBuffer> down_buffer = impl_->resident_buffer_with_bytes(down_filter_key, down_filter.data(), filter_bytes);
        id<MTLBuffer> alpha_buffer = impl_->resident_buffer_with_bytes(alpha_key, alpha_log.data(), param_bytes);
        id<MTLBuffer> beta_buffer = impl_->resident_buffer_with_bytes(beta_key, beta_log.data(), param_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&channels, sizeof(channels));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->bigvgan_activation_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:up_buffer offset:0 atIndex:1];
        [enc setBuffer:down_buffer offset:0 atIndex:2];
        [enc setBuffer:alpha_buffer offset:0 atIndex:3];
        [enc setBuffer:beta_buffer offset:0 atIndex:4];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:5];
        [enc setBuffer:tokensbuf offset:0 atIndex:6];
        [enc setBuffer:channelsbuf offset:0 atIndex:7];
        MTLSize threads = MTLSizeMake(channels, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(channels, impl_->bigvgan_activation_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("bigvgan_activation_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::groupnorm1_f32(const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta, uint32_t tokens, uint32_t channels, float eps) {
    if (tokens == 0 || channels == 0 || x.size() != static_cast<size_t>(tokens) * channels ||
        gamma.size() != channels || beta.size() != channels) {
        throw std::invalid_argument("groupnorm1_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger param_bytes = gamma.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t gamma_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t beta_offset = align_up_size(gamma_offset + param_bytes, alignment);
        const size_t out_offset = align_up_size(beta_offset + param_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        std::memcpy(base + gamma_offset, gamma.data(), param_bytes);
        std::memcpy(base + beta_offset, beta.data(), param_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&channels, sizeof(channels));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->groupnorm1_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(gamma_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(beta_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:channelsbuf offset:0 atIndex:5];
        [enc setBuffer:epsbuf offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("groupnorm1_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::groupnorm1_f32_resident(const std::string& gamma_key,
                                                         const std::vector<float>& gamma,
                                                         const std::string& beta_key,
                                                         const std::vector<float>& beta,
                                                         const std::vector<float>& x,
                                                         uint32_t tokens,
                                                         uint32_t channels,
                                                         float eps) {
    if (tokens == 0 || channels == 0 || x.size() != static_cast<size_t>(tokens) * channels ||
        gamma.size() != channels || beta.size() != channels) {
        throw std::invalid_argument("groupnorm1_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger x_bytes = x.size() * sizeof(float);
        const NSUInteger param_bytes = gamma.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + x_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + x_bytes, alignment);
        id<MTLBuffer> gamma_buffer = impl_->resident_buffer_with_bytes(gamma_key, gamma.data(), param_bytes);
        id<MTLBuffer> beta_buffer = impl_->resident_buffer_with_bytes(beta_key, beta.data(), param_bytes);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), x_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> channelsbuf = impl_->constant_buffer_with_bytes(&channels, sizeof(channels));
        id<MTLBuffer> epsbuf = impl_->constant_buffer_with_bytes(&eps, sizeof(eps));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->groupnorm1_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:gamma_buffer offset:0 atIndex:1];
        [enc setBuffer:beta_buffer offset:0 atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:channelsbuf offset:0 atIndex:5];
        [enc setBuffer:epsbuf offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("groupnorm1_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::mish_f32(const std::vector<float>& x) {
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = x.size() * sizeof(float);
        const size_t x_offset = 0;
        const size_t out_offset = align_up_size(x_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + x_offset, x.data(), bytes);
        uint32_t count = static_cast<uint32_t>(x.size());
        id<MTLBuffer> nbuf = impl_->constant_buffer_with_bytes(&count, sizeof(count));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->mish_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(x_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBuffer:nbuf offset:0 atIndex:2];
        MTLSize threads = MTLSizeMake(x.size(), 1, 1);
        MTLSize groups = MTLSizeMake(impl_->mish_pipeline.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("mish_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + x.size());
    }
}

std::vector<float> MetalContext::timestep_embedding_f32(const std::vector<float>& timesteps, const std::vector<float>& freqs, float scale) {
    if (timesteps.empty() || freqs.empty()) {
        throw std::invalid_argument("timestep_embedding_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const uint32_t batch = static_cast<uint32_t>(timesteps.size());
        const uint32_t half = static_cast<uint32_t>(freqs.size());
        const NSUInteger t_bytes = timesteps.size() * sizeof(float);
        const NSUInteger f_bytes = freqs.size() * sizeof(float);
        const NSUInteger out_count = static_cast<NSUInteger>(batch) * half * 2;
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t t_offset = 0;
        const size_t f_offset = align_up_size(t_offset + t_bytes, alignment);
        const size_t out_offset = align_up_size(f_offset + f_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + t_offset, timesteps.data(), t_bytes);
        std::memcpy(base + f_offset, freqs.data(), f_bytes);
        id<MTLBuffer> batchbuf = impl_->constant_buffer_with_bytes(&batch, sizeof(batch));
        id<MTLBuffer> halfbuf = impl_->constant_buffer_with_bytes(&half, sizeof(half));
        id<MTLBuffer> scalebuf = impl_->constant_buffer_with_bytes(&scale, sizeof(scale));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->timestep_embedding_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(t_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(f_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:batchbuf offset:0 atIndex:3];
        [enc setBuffer:halfbuf offset:0 atIndex:4];
        [enc setBuffer:scalebuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(half * 2, batch, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(half * 2, impl_->timestep_embedding_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("timestep_embedding_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::attention_single_head_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t tokens, uint32_t head_dim) {
    const size_t count = static_cast<size_t>(tokens) * head_dim;
    if (tokens == 0 || head_dim == 0 || q.size() != count || k.size() != count || v.size() != count) {
        throw std::invalid_argument("attention_single_head_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = count * sizeof(float);
        const size_t q_offset = 0;
        const size_t k_offset = align_up_size(q_offset + bytes, alignment);
        const size_t v_offset = align_up_size(k_offset + bytes, alignment);
        const size_t out_offset = align_up_size(v_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + q_offset, q.data(), bytes);
        std::memcpy(base + k_offset, k.data(), bytes);
        std::memcpy(base + v_offset, v.data(), bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->attention_single_head_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(q_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(k_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(v_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:dimbuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(head_dim, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(head_dim, impl_->attention_single_head_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("attention_single_head_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + count);
    }
}

std::vector<float> MetalContext::attention_single_head_causal_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t tokens, uint32_t head_dim) {
    const size_t count = static_cast<size_t>(tokens) * head_dim;
    if (tokens == 0 || head_dim == 0 || q.size() != count || k.size() != count || v.size() != count) {
        throw std::invalid_argument("attention_single_head_causal_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = count * sizeof(float);
        const size_t q_offset = 0;
        const size_t k_offset = align_up_size(q_offset + bytes, alignment);
        const size_t v_offset = align_up_size(k_offset + bytes, alignment);
        const size_t out_offset = align_up_size(v_offset + bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + q_offset, q.data(), bytes);
        std::memcpy(base + k_offset, k.data(), bytes);
        std::memcpy(base + v_offset, v.data(), bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->attention_single_head_causal_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(q_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(k_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(v_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:dimbuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(head_dim, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(head_dim, impl_->attention_single_head_causal_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("attention_single_head_causal_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + count);
    }
}

std::vector<float> MetalContext::attention_single_query_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, uint32_t key_tokens, uint32_t head_dim) {
    const size_t kv_count = static_cast<size_t>(key_tokens) * head_dim;
    if (key_tokens == 0 || head_dim == 0 || q.size() != head_dim || k.size() != kv_count || v.size() != kv_count) {
        throw std::invalid_argument("attention_single_query_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger q_bytes = q.size() * sizeof(float);
        const NSUInteger kv_bytes = kv_count * sizeof(float);
        const NSUInteger out_bytes = head_dim * sizeof(float);
        const size_t q_offset = 0;
        const size_t k_offset = align_up_size(q_offset + q_bytes, alignment);
        const size_t v_offset = align_up_size(k_offset + kv_bytes, alignment);
        const size_t out_offset = align_up_size(v_offset + kv_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + q_offset, q.data(), q_bytes);
        std::memcpy(base + k_offset, k.data(), kv_bytes);
        std::memcpy(base + v_offset, v.data(), kv_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&key_tokens, sizeof(key_tokens));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->attention_single_query_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(q_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(k_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(v_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:dimbuf offset:0 atIndex:5];
        MTLSize threads = MTLSizeMake(head_dim, 1, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(head_dim, impl_->attention_single_query_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("attention_single_query_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + head_dim);
    }
}

std::vector<float> MetalContext::gpt_cached_attention_f32(
    const std::vector<float>& cache_k,
    const std::vector<float>& cache_v,
    const std::vector<float>& current_qkv,
    uint32_t cache_tokens,
    uint32_t heads,
    uint32_t head_dim) {
    const uint32_t width = heads * head_dim;
    const size_t cache_count = static_cast<size_t>(cache_tokens) * width;
    if (cache_tokens == 0 || heads == 0 || head_dim == 0 || cache_tokens > 1023 ||
        cache_k.size() != cache_count || cache_v.size() != cache_count ||
        current_qkv.size() != static_cast<size_t>(width) * 3) {
        throw std::invalid_argument("gpt_cached_attention_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger cache_bytes = cache_count * sizeof(float);
        const NSUInteger qkv_bytes = current_qkv.size() * sizeof(float);
        const NSUInteger out_bytes = static_cast<NSUInteger>(width) * sizeof(float);
        const size_t k_offset = 0;
        const size_t v_offset = align_up_size(k_offset + cache_bytes, alignment);
        const size_t qkv_offset = align_up_size(v_offset + cache_bytes, alignment);
        const size_t out_offset = align_up_size(qkv_offset + qkv_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + k_offset, cache_k.data(), cache_bytes);
        std::memcpy(base + v_offset, cache_v.data(), cache_bytes);
        std::memcpy(base + qkv_offset, current_qkv.data(), qkv_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&cache_tokens, sizeof(cache_tokens));
        id<MTLBuffer> headsbuf = impl_->constant_buffer_with_bytes(&heads, sizeof(heads));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->gpt_cached_attention_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(k_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(v_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(qkv_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:headsbuf offset:0 atIndex:5];
        [enc setBuffer:dimbuf offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("gpt_cached_attention_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + width);
    }
}

std::vector<float> MetalContext::attention_single_head_masked_f32(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, const std::vector<uint32_t>& key_mask, uint32_t tokens, uint32_t head_dim) {
    const size_t count = static_cast<size_t>(tokens) * head_dim;
    if (tokens == 0 || head_dim == 0 || q.size() != count || k.size() != count || v.size() != count || key_mask.size() != tokens) {
        throw std::invalid_argument("attention_single_head_masked_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = count * sizeof(float);
        const NSUInteger mask_bytes = static_cast<NSUInteger>(tokens) * sizeof(uint32_t);
        const size_t q_offset = 0;
        const size_t k_offset = align_up_size(q_offset + bytes, alignment);
        const size_t v_offset = align_up_size(k_offset + bytes, alignment);
        const size_t mask_offset = align_up_size(v_offset + bytes, alignment);
        const size_t out_offset = align_up_size(mask_offset + mask_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + q_offset, q.data(), bytes);
        std::memcpy(base + k_offset, k.data(), bytes);
        std::memcpy(base + v_offset, v.data(), bytes);
        std::memcpy(base + mask_offset, key_mask.data(), mask_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->attention_single_head_masked_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(q_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(k_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(v_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mask_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:4];
        [enc setBuffer:tokensbuf offset:0 atIndex:5];
        [enc setBuffer:dimbuf offset:0 atIndex:6];
        MTLSize threads = MTLSizeMake(head_dim, tokens, 1);
        MTLSize groups = MTLSizeMake(std::min<uint32_t>(head_dim, impl_->attention_single_head_masked_pipeline.threadExecutionWidth), 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:groups];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("attention_single_head_masked_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + count);
    }
}

std::vector<float> MetalContext::dit_attention_qkv_rope_f32(const std::vector<float>& qkv,
                                                            const std::vector<uint32_t>& key_mask,
                                                            uint32_t tokens,
                                                            uint32_t heads,
                                                            uint32_t head_dim) {
    const size_t inner = static_cast<size_t>(heads) * head_dim;
    const size_t qkv_count = static_cast<size_t>(tokens) * inner * 3;
    const size_t out_count = static_cast<size_t>(tokens) * inner;
    if (tokens == 0 || tokens > 4096 || heads == 0 || head_dim == 0 || (head_dim % 2) != 0 ||
        qkv.size() != qkv_count || key_mask.size() != tokens) {
        throw std::invalid_argument("dit_attention_qkv_rope_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger qkv_bytes = qkv_count * sizeof(float);
        const NSUInteger out_bytes = out_count * sizeof(float);
        const NSUInteger mask_bytes = static_cast<NSUInteger>(tokens) * sizeof(uint32_t);
        const size_t qkv_offset = 0;
        const size_t mask_offset = align_up_size(qkv_offset + qkv_bytes, alignment);
        const size_t out_offset = align_up_size(mask_offset + mask_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + qkv_offset, qkv.data(), qkv_bytes);
        std::memcpy(base + mask_offset, key_mask.data(), mask_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> headsbuf = impl_->constant_buffer_with_bytes(&heads, sizeof(heads));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        dispatch_dit_rope_rotate_qk(enc, impl_->dit_rope_rotate_qk_batched_pipeline,
                                    arena, static_cast<NSUInteger>(qkv_offset), 1, tokens, heads, head_dim);
        uint32_t one = 1;
        id<MTLBuffer> batchbuf = impl_->constant_buffer_with_bytes(&one, sizeof(one));
        [enc setComputePipelineState:impl_->dit_attention_qkv_rope_batched_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(qkv_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mask_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:batchbuf offset:0 atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:headsbuf offset:0 atIndex:5];
        [enc setBuffer:dimbuf offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("dit_attention_qkv_rope_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::dit_attention_qkv_rope_batched_f32(const std::vector<float>& qkv,
                                                                    const std::vector<uint32_t>& key_mask,
                                                                    uint32_t batch,
                                                                    uint32_t tokens,
                                                                    uint32_t heads,
                                                                    uint32_t head_dim) {
    const size_t inner = static_cast<size_t>(heads) * head_dim;
    const size_t qkv_count = static_cast<size_t>(batch) * tokens * inner * 3;
    const size_t out_count = static_cast<size_t>(batch) * tokens * inner;
    if (batch == 0 || tokens == 0 || tokens > 4096 || heads == 0 || head_dim == 0 || (head_dim % 2) != 0 ||
        qkv.size() != qkv_count || key_mask.size() != static_cast<size_t>(batch) * tokens) {
        throw std::invalid_argument("dit_attention_qkv_rope_batched_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger qkv_bytes = qkv_count * sizeof(float);
        const NSUInteger out_bytes = out_count * sizeof(float);
        const NSUInteger mask_bytes = static_cast<NSUInteger>(batch) * tokens * sizeof(uint32_t);
        const size_t qkv_offset = 0;
        const size_t mask_offset = align_up_size(qkv_offset + qkv_bytes, alignment);
        const size_t out_offset = align_up_size(mask_offset + mask_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + qkv_offset, qkv.data(), qkv_bytes);
        std::memcpy(base + mask_offset, key_mask.data(), mask_bytes);
        id<MTLBuffer> batchbuf = impl_->constant_buffer_with_bytes(&batch, sizeof(batch));
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> headsbuf = impl_->constant_buffer_with_bytes(&heads, sizeof(heads));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        dispatch_dit_rope_rotate_qk(enc, impl_->dit_rope_rotate_qk_batched_pipeline,
                                    arena, static_cast<NSUInteger>(qkv_offset), batch, tokens, heads, head_dim);
        [enc setComputePipelineState:impl_->dit_attention_qkv_rope_batched_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(qkv_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mask_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:2];
        [enc setBuffer:batchbuf offset:0 atIndex:3];
        [enc setBuffer:tokensbuf offset:0 atIndex:4];
        [enc setBuffer:headsbuf offset:0 atIndex:5];
        [enc setBuffer:dimbuf offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("dit_attention_qkv_rope_batched_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

std::vector<float> MetalContext::cross_attention_heads_masked_f32(const std::vector<float>& q,
                                                                  const std::vector<float>& k,
                                                                  const std::vector<float>& v,
                                                                  const std::vector<uint32_t>& key_mask,
                                                                  uint32_t query_tokens,
                                                                  uint32_t key_tokens,
                                                                  uint32_t heads,
                                                                  uint32_t head_dim) {
    const size_t inner = static_cast<size_t>(heads) * head_dim;
    const size_t q_count = static_cast<size_t>(query_tokens) * inner;
    const size_t kv_count = static_cast<size_t>(key_tokens) * inner;
    if (query_tokens == 0 || key_tokens == 0 || key_tokens > 1024 || heads == 0 || head_dim == 0 ||
        q.size() != q_count || k.size() != kv_count || v.size() != kv_count || key_mask.size() != key_tokens) {
        throw std::invalid_argument("cross_attention_heads_masked_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger q_bytes = q_count * sizeof(float);
        const NSUInteger kv_bytes = kv_count * sizeof(float);
        const NSUInteger mask_bytes = static_cast<NSUInteger>(key_tokens) * sizeof(uint32_t);
        const size_t q_offset = 0;
        const size_t k_offset = align_up_size(q_offset + q_bytes, alignment);
        const size_t v_offset = align_up_size(k_offset + kv_bytes, alignment);
        const size_t mask_offset = align_up_size(v_offset + kv_bytes, alignment);
        const size_t out_offset = align_up_size(mask_offset + mask_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + q_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + q_offset, q.data(), q_bytes);
        std::memcpy(base + k_offset, k.data(), kv_bytes);
        std::memcpy(base + v_offset, v.data(), kv_bytes);
        std::memcpy(base + mask_offset, key_mask.data(), mask_bytes);
        id<MTLBuffer> qtokensbuf = impl_->constant_buffer_with_bytes(&query_tokens, sizeof(query_tokens));
        id<MTLBuffer> ktokensbuf = impl_->constant_buffer_with_bytes(&key_tokens, sizeof(key_tokens));
        id<MTLBuffer> headsbuf = impl_->constant_buffer_with_bytes(&heads, sizeof(heads));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->cross_attention_heads_masked_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(q_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(k_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(v_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mask_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:4];
        [enc setBuffer:qtokensbuf offset:0 atIndex:5];
        [enc setBuffer:ktokensbuf offset:0 atIndex:6];
        [enc setBuffer:headsbuf offset:0 atIndex:7];
        [enc setBuffer:dimbuf offset:0 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(heads, query_tokens, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("cross_attention_heads_masked_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + q_count);
    }
}

std::vector<float> MetalContext::conformer_rel_attention_context_f32(const std::vector<float>& q,
                                                                     const std::vector<float>& k,
                                                                     const std::vector<float>& v,
                                                                     const std::vector<float>& p,
                                                                     const std::vector<float>& bias_u,
                                                                     const std::vector<float>& bias_v,
                                                                     const std::vector<uint32_t>& key_mask,
                                                                     uint32_t tokens,
                                                                     uint32_t heads,
                                                                     uint32_t head_dim) {
    const size_t dim = static_cast<size_t>(heads) * head_dim;
    const size_t count = static_cast<size_t>(tokens) * dim;
    if (tokens == 0 || tokens > 1024 || heads == 0 || head_dim == 0 ||
        q.size() != count || k.size() != count || v.size() != count || p.size() != count ||
        bias_u.size() != dim || bias_v.size() != dim || key_mask.size() != tokens) {
        throw std::invalid_argument("conformer_rel_attention_context_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = count * sizeof(float);
        const NSUInteger bias_bytes = dim * sizeof(float);
        const NSUInteger mask_bytes = static_cast<NSUInteger>(tokens) * sizeof(uint32_t);
        const size_t q_offset = 0;
        const size_t k_offset = align_up_size(q_offset + bytes, alignment);
        const size_t v_offset = align_up_size(k_offset + bytes, alignment);
        const size_t p_offset = align_up_size(v_offset + bytes, alignment);
        const size_t bias_u_offset = align_up_size(p_offset + bytes, alignment);
        const size_t bias_v_offset = align_up_size(bias_u_offset + bias_bytes, alignment);
        const size_t mask_offset = align_up_size(bias_v_offset + bias_bytes, alignment);
        const size_t out_offset = align_up_size(mask_offset + mask_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + q_offset, q.data(), bytes);
        std::memcpy(base + k_offset, k.data(), bytes);
        std::memcpy(base + v_offset, v.data(), bytes);
        std::memcpy(base + p_offset, p.data(), bytes);
        std::memcpy(base + bias_u_offset, bias_u.data(), bias_bytes);
        std::memcpy(base + bias_v_offset, bias_v.data(), bias_bytes);
        std::memcpy(base + mask_offset, key_mask.data(), mask_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> headsbuf = impl_->constant_buffer_with_bytes(&heads, sizeof(heads));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conformer_rel_attention_context_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(q_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(k_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(v_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(p_offset) atIndex:3];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_u_offset) atIndex:4];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(bias_v_offset) atIndex:5];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mask_offset) atIndex:6];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:7];
        [enc setBuffer:tokensbuf offset:0 atIndex:8];
        [enc setBuffer:headsbuf offset:0 atIndex:9];
        [enc setBuffer:dimbuf offset:0 atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conformer_rel_attention_context_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + count);
    }
}

std::vector<float> MetalContext::conformer_rel_attention_context_f32_resident(
    const std::string& bias_u_key,
    const std::vector<float>& bias_u,
    const std::string& bias_v_key,
    const std::vector<float>& bias_v,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& p,
    const std::vector<uint32_t>& key_mask,
    uint32_t tokens,
    uint32_t heads,
    uint32_t head_dim) {
    const size_t dim = static_cast<size_t>(heads) * head_dim;
    const size_t count = static_cast<size_t>(tokens) * dim;
    if (tokens == 0 || tokens > 1024 || heads == 0 || head_dim == 0 ||
        q.size() != count || k.size() != count || v.size() != count || p.size() != count ||
        bias_u.size() != dim || bias_v.size() != dim || key_mask.size() != tokens) {
        throw std::invalid_argument("conformer_rel_attention_context_f32_resident invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger bytes = count * sizeof(float);
        const NSUInteger bias_bytes = dim * sizeof(float);
        const NSUInteger mask_bytes = static_cast<NSUInteger>(tokens) * sizeof(uint32_t);
        id<MTLBuffer> bias_u_buffer = impl_->resident_buffer_with_bytes(bias_u_key, bias_u.data(), bias_bytes);
        id<MTLBuffer> bias_v_buffer = impl_->resident_buffer_with_bytes(bias_v_key, bias_v.data(), bias_bytes);
        const size_t q_offset = 0;
        const size_t k_offset = align_up_size(q_offset + bytes, alignment);
        const size_t v_offset = align_up_size(k_offset + bytes, alignment);
        const size_t p_offset = align_up_size(v_offset + bytes, alignment);
        const size_t mask_offset = align_up_size(p_offset + bytes, alignment);
        const size_t out_offset = align_up_size(mask_offset + mask_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + q_offset, q.data(), bytes);
        std::memcpy(base + k_offset, k.data(), bytes);
        std::memcpy(base + v_offset, v.data(), bytes);
        std::memcpy(base + p_offset, p.data(), bytes);
        std::memcpy(base + mask_offset, key_mask.data(), mask_bytes);
        id<MTLBuffer> tokensbuf = impl_->constant_buffer_with_bytes(&tokens, sizeof(tokens));
        id<MTLBuffer> headsbuf = impl_->constant_buffer_with_bytes(&heads, sizeof(heads));
        id<MTLBuffer> dimbuf = impl_->constant_buffer_with_bytes(&head_dim, sizeof(head_dim));
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->conformer_rel_attention_context_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(q_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(k_offset) atIndex:1];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(v_offset) atIndex:2];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(p_offset) atIndex:3];
        [enc setBuffer:bias_u_buffer offset:0 atIndex:4];
        [enc setBuffer:bias_v_buffer offset:0 atIndex:5];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(mask_offset) atIndex:6];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:7];
        [enc setBuffer:tokensbuf offset:0 atIndex:8];
        [enc setBuffer:headsbuf offset:0 atIndex:9];
        [enc setBuffer:dimbuf offset:0 atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("conformer_rel_attention_context_f32_resident command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + count);
    }
}

// ---------------------------------------------------------------------------
// Pass API implementation
// ---------------------------------------------------------------------------

bool MetalContext::inPass() const { return impl_->pass_mode; }

void MetalContext::beginPass(size_t workspace_bytes) {
    if (impl_->pass_mode) {
        throw std::logic_error("beginPass: already in pass mode");
    }
    @autoreleasepool {
        if (!impl_->pass_workspace || impl_->pass_workspace_capacity < static_cast<NSUInteger>(workspace_bytes)) {
            impl_->pass_workspace = new_counted_buffer_with_length(
                impl_->device, impl_->buffer_allocations, impl_->buffer_bytes_allocated,
                static_cast<NSUInteger>(workspace_bytes));
            impl_->pass_workspace_capacity = static_cast<NSUInteger>(workspace_bytes);
        }
        impl_->pass_cursor = 0;
        impl_->pass_scratch_base = 0;
        impl_->pass_cb = [impl_->queue commandBuffer];
        impl_->pass_enc = [impl_->pass_cb computeCommandEncoder];
        impl_->pass_mode = true;
    }
}

void MetalContext::endPass() {
    if (!impl_->pass_mode) {
        throw std::logic_error("endPass: not in pass mode");
    }
    @autoreleasepool {
        if (impl_->pass_enc) {
            [impl_->pass_enc endEncoding];
        }
        commit_and_count(impl_->command_buffers_submitted, impl_->pass_cb);
        wait_and_record(impl_->gpu_elapsed_seconds, impl_->pass_cb);
        if ([impl_->pass_cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("pass command buffer failed");
        }
        impl_->pass_mode = false;
        impl_->pass_enc = nil;
        impl_->pass_cb = nil;
    }
}

void MetalContext::passBarrier() {
    if (!impl_->pass_mode) throw std::logic_error("passBarrier: not in pass mode");
    impl_->pass_ensure_encoder();
}

void MetalContext::passSetScratchBase() {
    if (!impl_->pass_mode) throw std::logic_error("passSetScratchBase: not in pass mode");
    impl_->pass_scratch_base = impl_->pass_cursor;
}

void MetalContext::passResetScratch() {
    if (!impl_->pass_mode) throw std::logic_error("passResetScratch: not in pass mode");
    impl_->pass_cursor = impl_->pass_scratch_base;
}

PassSlot MetalContext::passAlloc(uint32_t element_count) {
    if (!impl_->pass_mode) throw std::logic_error("passAlloc: not in pass mode");
    return impl_->pass_alloc_raw(element_count);
}

PassSlot MetalContext::passUploadAlloc(const float* data, uint32_t count) {
    if (!impl_->pass_mode) throw std::logic_error("passUploadAlloc: not in pass mode");
    auto slot = impl_->pass_alloc_raw(count);
    std::memcpy(static_cast<uint8_t*>([impl_->pass_workspace contents]) + slot.byte_offset,
                data, static_cast<size_t>(count) * sizeof(float));
    return slot;
}

PassSlot MetalContext::passUploadAlloc(const std::vector<float>& data) {
    return passUploadAlloc(data.data(), static_cast<uint32_t>(data.size()));
}

PassSlot MetalContext::passUploadAllocU32(const uint32_t* data, uint32_t count) {
    if (!impl_->pass_mode) throw std::logic_error("passUploadAllocU32: not in pass mode");
    auto slot = impl_->pass_alloc_raw(count);
    std::memcpy(static_cast<uint8_t*>([impl_->pass_workspace contents]) + slot.byte_offset,
                data, static_cast<size_t>(count) * sizeof(uint32_t));
    return slot;
}

PassSlot MetalContext::passUploadAllocU32(const std::vector<uint32_t>& data) {
    return passUploadAllocU32(data.data(), static_cast<uint32_t>(data.size()));
}

void MetalContext::passUploadInto(PassSlot slot, const float* data, uint32_t count) {
    if (!impl_->pass_mode) throw std::logic_error("passUploadInto: not in pass mode");
    std::memcpy(static_cast<uint8_t*>([impl_->pass_workspace contents]) + slot.byte_offset,
                data, static_cast<size_t>(count) * sizeof(float));
}

void MetalContext::passUploadInto(PassSlot slot, const std::vector<float>& data) {
    passUploadInto(slot, data.data(), static_cast<uint32_t>(data.size()));
}

std::vector<float> MetalContext::passRead(PassSlot slot) const {
    if (impl_->pass_mode) throw std::logic_error("passRead: cannot read while pass is active");
    if (!slot.valid() || !impl_->pass_workspace) {
        throw std::invalid_argument("passRead: invalid slot or no workspace");
    }
    const float* ptr = reinterpret_cast<const float*>(
        static_cast<const uint8_t*>([impl_->pass_workspace contents]) + slot.byte_offset);
    return std::vector<float>(ptr, ptr + slot.element_count);
}

// Helper macro: require pass mode and insert barrier before dispatch.
// After an MPS encode the encoder is nil; reopen lazily (encoder boundary
// already orders the MPS work via hazard tracking).
#define PASS_REQUIRE_AND_BARRIER()                                         \
    do {                                                                   \
        if (!impl_->pass_mode) throw std::logic_error("pass op outside pass"); \
        impl_->pass_ensure_encoder();                                      \
    } while (0)

PassSlot MetalContext::add_f32_pass(PassSlot a, PassSlot b) {
    PASS_REQUIRE_AND_BARRIER();
    auto out = impl_->pass_alloc_raw(a.element_count);
    uint32_t count = a.element_count;
    [impl_->pass_enc setComputePipelineState:impl_->add_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:a.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:b.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:3];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(impl_->add_pipeline.threadExecutionWidth, 1, 1)];
    return out;
}

void MetalContext::add_f32_pass_into(PassSlot a, PassSlot b, PassSlot out) {
    PASS_REQUIRE_AND_BARRIER();
    uint32_t count = a.element_count;
    [impl_->pass_enc setComputePipelineState:impl_->add_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:a.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:b.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:3];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(impl_->add_pipeline.threadExecutionWidth, 1, 1)];
}

PassSlot MetalContext::silu_f32_pass(PassSlot x, uint32_t count) {
    PASS_REQUIRE_AND_BARRIER();
    auto out = impl_->pass_alloc_raw(count);
    [impl_->pass_enc setComputePipelineState:impl_->silu_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:2];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(impl_->silu_pipeline.threadExecutionWidth, 1, 1)];
    return out;
}

PassSlot MetalContext::silu_mul_f32_pass(PassSlot a, PassSlot b) {
    PASS_REQUIRE_AND_BARRIER();
    auto out = impl_->pass_alloc_raw(a.element_count);
    uint32_t count = a.element_count;
    [impl_->pass_enc setComputePipelineState:impl_->silu_mul_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:a.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:b.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:3];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(impl_->silu_mul_pipeline.threadExecutionWidth, 1, 1)];
    return out;
}

PassSlot MetalContext::silu_mul_split_f32_pass(PassSlot x, uint32_t tokens, uint32_t width) {
    PASS_REQUIRE_AND_BARRIER();
    auto out = impl_->pass_alloc_raw(tokens * width);
    [impl_->pass_enc setComputePipelineState:impl_->silu_mul_split_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:2];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:3];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(width, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(width, impl_->silu_mul_split_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

PassSlot MetalContext::concat_rows_f32_pass(PassSlot a, PassSlot b, uint32_t tokens, uint32_t a_width, uint32_t b_width) {
    PASS_REQUIRE_AND_BARRIER();
    const uint32_t out_width = a_width + b_width;
    auto out = impl_->pass_alloc_raw(static_cast<uint32_t>(tokens) * out_width);
    [impl_->pass_enc setComputePipelineState:impl_->concat_rows_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:a.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:b.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [impl_->pass_enc setBytes:&a_width length:sizeof(a_width) atIndex:4];
    [impl_->pass_enc setBytes:&b_width length:sizeof(b_width) atIndex:5];
    [impl_->pass_enc setBytes:&out_width length:sizeof(out_width) atIndex:6];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(out_width, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_width, impl_->concat_rows_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

PassSlot MetalContext::linear_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t rows, uint32_t cols) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(rows);
    if (rows >= 256 && cols >= 64) {
        // simdgroup-per-row GEMV: 8 simdgroups (256 threads) per threadgroup.
        constexpr uint32_t sgs_per_tg = 8;
        const uint32_t tgs = (rows + sgs_per_tg - 1) / sgs_per_tg;
        bool f16g = false;
        id<MTLBuffer> gemv_w = impl_->weight_buffer_pref_f16(wk, w, fp16_weights_enabled(), f16g);
        [impl_->pass_enc setComputePipelineState:(f16g ? impl_->linear_gemv_f16w_pipeline : impl_->linear_gemv_pipeline)];
        [impl_->pass_enc setBuffer:gemv_w offset:0 atIndex:0];
        [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:1];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:2];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
        [impl_->pass_enc setBytes:&rows length:sizeof(rows) atIndex:4];
        [impl_->pass_enc setBytes:&cols length:sizeof(cols) atIndex:5];
        [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(sgs_per_tg * 32, 1, 1)];
        return out;
    }
    id<MTLBuffer> wbuf = impl_->resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    [impl_->pass_enc setComputePipelineState:impl_->linear_pipeline];
    [impl_->pass_enc setBuffer:wbuf offset:0 atIndex:0];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&rows length:sizeof(rows) atIndex:4];
    [impl_->pass_enc setBytes:&cols length:sizeof(cols) atIndex:5];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(rows, impl_->linear_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

static void dispatch_linear_rows_pass(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pipeline,
                                      id<MTLBuffer> workspace,
                                      id<MTLBuffer> wbuf, id<MTLBuffer> bbuf,
                                      NSUInteger x_offset, NSUInteger out_offset,
                                      uint32_t tokens, uint32_t rows, uint32_t cols) {
    uint32_t t = tokens, r = rows, c = cols;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:wbuf offset:0 atIndex:0];
    [enc setBuffer:bbuf offset:0 atIndex:1];
    [enc setBuffer:workspace offset:x_offset atIndex:2];
    [enc setBuffer:workspace offset:out_offset atIndex:3];
    [enc setBytes:&t length:sizeof(t) atIndex:4];
    [enc setBytes:&r length:sizeof(r) atIndex:5];
    [enc setBytes:&c length:sizeof(c) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(rows, tokens, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(rows, pipeline.threadExecutionWidth), 1, 1)];
}

PassSlot MetalContext::linear_rows_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t rows, uint32_t cols) {
    // Autoregressive decoding is GEMV, not a one-row MPS GEMM. Use the
    // existing coalesced simdgroup reduction (FP32 unless explicitly opted in).
    if (tokens == 1) {
        return linear_f32_pass(wk, w, bk, b, x, rows, cols);
    }
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(tokens * rows);
    if (linear_rows_use_mps(tokens, rows, cols)) {
        bool f16 = false;
        id<MTLBuffer> wbuf = impl_->weight_buffer_pref_f16(wk, w, fp16_weights_enabled(), f16);
        const bool bias_zero = impl_->bias_is_zero_for(bk, b);
        if (f16 && custom_gemm_enabled()) {
            impl_->pass_gemm_f16w(wbuf, bbuf, !bias_zero, false, x.byte_offset, out.byte_offset, tokens, rows, cols);
            return out;
        }
        impl_->pass_linear_rows_mps(wbuf, f16, bbuf, x.byte_offset, out.byte_offset, tokens, rows, cols);
        return out;
    }
    id<MTLBuffer> wbuf = impl_->resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    dispatch_linear_rows_pass(impl_->pass_enc, impl_->linear_rows_pipeline, impl_->pass_workspace,
                              wbuf, bbuf, x.byte_offset, out.byte_offset, tokens, rows, cols);
    return out;
}

void MetalContext::linear_rows_f32_pass_into(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t rows, uint32_t cols, PassSlot out) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    if (linear_rows_use_mps(tokens, rows, cols)) {
        bool f16 = false;
        id<MTLBuffer> wbuf = impl_->weight_buffer_pref_f16(wk, w, fp16_weights_enabled(), f16);
        const bool bias_zero = impl_->bias_is_zero_for(bk, b);
        if (f16 && custom_gemm_enabled()) {
            impl_->pass_gemm_f16w(wbuf, bbuf, !bias_zero, false, x.byte_offset, out.byte_offset, tokens, rows, cols);
            return;
        }
        impl_->pass_linear_rows_mps(wbuf, f16, bbuf, x.byte_offset, out.byte_offset, tokens, rows, cols);
        return;
    }
    id<MTLBuffer> wbuf = impl_->resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    dispatch_linear_rows_pass(impl_->pass_enc, impl_->linear_rows_pipeline, impl_->pass_workspace,
                              wbuf, bbuf, x.byte_offset, out.byte_offset, tokens, rows, cols);
}


PassSlot MetalContext::rmsnorm_rows_eps_f32_pass(const std::string& gk,const std::vector<float>& g,PassSlot x,uint32_t tokens,uint32_t width,float eps) {
    PASS_REQUIRE_AND_BARRIER();
    auto gbuf=impl_->resident_buffer_with_bytes(gk,g.data(),g.size()*4);auto out=impl_->pass_alloc_raw(tokens*width);
    [impl_->pass_enc setComputePipelineState:impl_->rmsnorm_rows_eps_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:gbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&tokens length:4 atIndex:3];[impl_->pass_enc setBytes:&width length:4 atIndex:4];[impl_->pass_enc setBytes:&eps length:4 atIndex:5];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(tokens,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];return out;
}
PassSlot MetalContext::qwen_rope_f32_pass(PassSlot x,uint32_t tokens,uint32_t heads,uint32_t offset) {
    PASS_REQUIRE_AND_BARRIER();auto out=impl_->pass_alloc_raw(x.element_count);
    [impl_->pass_enc setComputePipelineState:impl_->qwen_rope_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];[impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
    [impl_->pass_enc setBytes:&tokens length:4 atIndex:2];[impl_->pass_enc setBytes:&heads length:4 atIndex:3];[impl_->pass_enc setBytes:&offset length:4 atIndex:4];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(tokens*heads*128,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];return out;
}
PassSlot MetalContext::qwen_attention_f32_pass(uint32_t layer,PassSlot q,PassSlot k,PassSlot v,uint32_t tokens,uint32_t offset) {
    PASS_REQUIRE_AND_BARRIER();
    if(!impl_->gpt_kv_buffer || layer>=impl_->gpt_kv_layers || impl_->gpt_kv_width!=1024 || offset+tokens>4096)throw std::invalid_argument("Invalid Qwen attention cache");
    auto enc=impl_->pass_enc;
    [enc setComputePipelineState:impl_->qwen_store_pipeline];
    [enc setBuffer:impl_->pass_workspace offset:k.byte_offset atIndex:0];[enc setBuffer:impl_->pass_workspace offset:v.byte_offset atIndex:1];
    [enc setBuffer:impl_->gpt_kv_buffer offset:impl_->gpt_kv_k_offset(layer) atIndex:2];[enc setBuffer:impl_->gpt_kv_buffer offset:impl_->gpt_kv_v_offset(layer) atIndex:3];
    [enc setBytes:&tokens length:4 atIndex:4];[enc setBytes:&offset length:4 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(tokens*1024,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];auto out=impl_->pass_alloc_raw(tokens*2048);
    [enc setComputePipelineState:impl_->qwen_attention_pipeline];
    [enc setBuffer:impl_->pass_workspace offset:q.byte_offset atIndex:0];[enc setBuffer:impl_->gpt_kv_buffer offset:impl_->gpt_kv_k_offset(layer) atIndex:1];[enc setBuffer:impl_->gpt_kv_buffer offset:impl_->gpt_kv_v_offset(layer) atIndex:2];[enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [enc setBytes:&tokens length:4 atIndex:4];[enc setBytes:&offset length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(16,tokens,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];return out;
}

PassSlot MetalContext::rmsnorm_rows_f32_pass(const std::string& gk, const std::vector<float>& g, PassSlot x, uint32_t tokens, uint32_t width) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> gbuf = impl_->resident_buffer_with_bytes(gk, g.data(), g.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(tokens * width);
    [impl_->pass_enc setComputePipelineState:impl_->rmsnorm_rows_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:gbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:4];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::adaptive_rmsnorm_rows_f32_pass(const std::string& gk, const std::vector<float>& g, PassSlot x, PassSlot aw, PassSlot ab, uint32_t tokens, uint32_t width, float eps) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> gbuf = impl_->resident_buffer_with_bytes(gk, g.data(), g.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(tokens * width);
    [impl_->pass_enc setComputePipelineState:impl_->adaptive_rmsnorm_rows_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:gbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:aw.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:ab.byte_offset atIndex:3];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:4];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:5];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:6];
    [impl_->pass_enc setBytes:&eps length:sizeof(eps) atIndex:7];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::adaptive_layernorm_rows_f32_pass(PassSlot x, PassSlot shift, PassSlot scale, uint32_t tokens, uint32_t width, float eps) {
    PASS_REQUIRE_AND_BARRIER();
    auto out = impl_->pass_alloc_raw(tokens * width);
    [impl_->pass_enc setComputePipelineState:impl_->adaptive_layernorm_rows_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:shift.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:scale.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:5];
    [impl_->pass_enc setBytes:&eps length:sizeof(eps) atIndex:6];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::dit_attention_qkv_rope_batched_f32_pass(PassSlot qkv, PassSlot mask, uint32_t batch, uint32_t tokens, uint32_t heads, uint32_t head_dim) {
    PASS_REQUIRE_AND_BARRIER();
    const uint32_t inner = heads * head_dim;
    auto out = impl_->pass_alloc_raw(batch * tokens * inner);
    dispatch_dit_rope_rotate_qk(impl_->pass_enc, impl_->dit_rope_rotate_qk_batched_pipeline,
                                impl_->pass_workspace, qkv.byte_offset, batch, tokens, heads, head_dim);
    if (head_dim == 64 && sgq_attention_enabled()) {
        [impl_->pass_enc setComputePipelineState:fp16_weights_enabled() ? impl_->dit_attention_sgq_pipeline : impl_->dit_attention_sgq_fp32_pipeline];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:qkv.byte_offset atIndex:0];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:mask.byte_offset atIndex:1];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
        [impl_->pass_enc setBytes:&batch length:sizeof(batch) atIndex:3];
        [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
        [impl_->pass_enc setBytes:&heads length:sizeof(heads) atIndex:5];
        [impl_->pass_enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(heads, (tokens + 31) / 32, batch)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return out;
    }
    [impl_->pass_enc setComputePipelineState:impl_->dit_attention_qkv_rope_batched_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:qkv.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:mask.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&batch length:sizeof(batch) atIndex:3];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [impl_->pass_enc setBytes:&heads length:sizeof(heads) atIndex:5];
    [impl_->pass_enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(heads, tokens, batch)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::dit_input_merge_batched_f32_pass(PassSlot x, PassSlot prompt_x, PassSlot cond_proj, PassSlot style, uint32_t batch, uint32_t tokens) {
    PASS_REQUIRE_AND_BARRIER();
    constexpr uint32_t mel_width = 80;
    constexpr uint32_t cond_width = 512;
    constexpr uint32_t style_width = 192;
    constexpr uint32_t out_width = mel_width + mel_width + cond_width + style_width;
    const uint32_t rows = batch * tokens;
    auto out = impl_->pass_alloc_raw(rows * out_width);
    [impl_->pass_enc setComputePipelineState:impl_->dit_input_merge_batched_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:prompt_x.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:cond_proj.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:style.byte_offset atIndex:3];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:4];
    [impl_->pass_enc setBytes:&batch length:sizeof(batch) atIndex:5];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:6];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(out_width, rows, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_width, impl_->dit_input_merge_batched_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

PassSlot MetalContext::conv1d_same_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> wbuf = impl_->resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(tokens * out_ch);
    if (kernel == 1 && linear_rows_use_mps(tokens, out_ch, in_ch)) {
        // kernel-1 conv1d == GEMM: weight [out_ch, in_ch, 1] has GEMM layout already.
        bool f16c = false;
        id<MTLBuffer> wgemm = impl_->weight_buffer_pref_f16(wk, w, fp16_weights_enabled(), f16c);
        const bool bias_zero_c = impl_->bias_is_zero_for(bk, b);
        if (f16c && custom_gemm_enabled()) {
            impl_->pass_gemm_f16w(wgemm, bbuf, !bias_zero_c, false, x.byte_offset, out.byte_offset, tokens, out_ch, in_ch);
            return out;
        }
        impl_->pass_linear_rows_mps(wgemm, f16c, bbuf, x.byte_offset, out.byte_offset, tokens, out_ch, in_ch);
        return out;
    }
    [impl_->pass_enc setComputePipelineState:impl_->conv1d_same_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:wbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [impl_->pass_enc setBytes:&in_ch length:sizeof(in_ch) atIndex:5];
    [impl_->pass_enc setBytes:&out_ch length:sizeof(out_ch) atIndex:6];
    [impl_->pass_enc setBytes:&kernel length:sizeof(kernel) atIndex:7];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(out_ch, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_ch, impl_->conv1d_same_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

void MetalContext::conv1d_same_f32_pass_into(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel, PassSlot out) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> wbuf = impl_->resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    if (kernel == 1 && linear_rows_use_mps(tokens, out_ch, in_ch)) {
        bool f16c = false;
        id<MTLBuffer> wgemm = impl_->weight_buffer_pref_f16(wk, w, fp16_weights_enabled(), f16c);
        const bool bias_zero_c = impl_->bias_is_zero_for(bk, b);
        if (f16c && custom_gemm_enabled()) {
            impl_->pass_gemm_f16w(wgemm, bbuf, !bias_zero_c, false, x.byte_offset, out.byte_offset, tokens, out_ch, in_ch);
            return;
        }
        impl_->pass_linear_rows_mps(wgemm, f16c, bbuf, x.byte_offset, out.byte_offset, tokens, out_ch, in_ch);
        return;
    }
    [impl_->pass_enc setComputePipelineState:impl_->conv1d_same_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:wbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [impl_->pass_enc setBytes:&in_ch length:sizeof(in_ch) atIndex:5];
    [impl_->pass_enc setBytes:&out_ch length:sizeof(out_ch) atIndex:6];
    [impl_->pass_enc setBytes:&kernel length:sizeof(kernel) atIndex:7];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(out_ch, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_ch, impl_->conv1d_same_pipeline.threadExecutionWidth), 1, 1)];
}

PassSlot MetalContext::conv1d_reflect_same_batched_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t batch, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> bbuf_mps = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    if ((kernel % 2) == 1 && linear_rows_use_mps(tokens, out_ch, in_ch)) {
        // Per-tap dense weight matrices Wk[out_ch, in_ch], cached as residents.
        const bool taps_fp16 = fp16_weights_enabled();
        std::vector<id<MTLBuffer>> tap_bufs(kernel);
        for (uint32_t k = 0; k < kernel; ++k) {
            const std::string tap_key = wk + ".tap" + std::to_string(k) + (taps_fp16 ? ".f16" : "");
            if (impl_->has_resident(tap_key)) {
                tap_bufs[k] = taps_fp16
                    ? impl_->resident_buffer_f16_from_f32(tap_key, nullptr, 0)
                    : impl_->resident_buffer_with_bytes(tap_key, nullptr, 0);
            } else {
                if (w.size() < static_cast<size_t>(out_ch) * in_ch * kernel) {
                    throw std::invalid_argument("conv tap resident missing and weight data not provided: " + tap_key);
                }
                std::vector<float> tap(static_cast<size_t>(out_ch) * in_ch);
                for (uint32_t oc = 0; oc < out_ch; ++oc) {
                    for (uint32_t ic = 0; ic < in_ch; ++ic) {
                        tap[static_cast<size_t>(oc) * in_ch + ic] =
                            w[(static_cast<size_t>(oc) * in_ch + ic) * kernel + k];
                    }
                }
                tap_bufs[k] = taps_fp16
                    ? impl_->resident_buffer_f16_from_f32(tap_key, tap.data(), tap.size())
                    : impl_->resident_buffer_with_bytes(tap_key, tap.data(), tap.size() * sizeof(float));
            }
        }
        const uint32_t pad = kernel / 2;
        auto x_pad = impl_->pass_alloc_raw(batch * (tokens + 2 * pad) * in_ch);
        auto out_mps = impl_->pass_alloc_raw(batch * tokens * out_ch);
        if (taps_fp16 && !custom_gemm_enabled()) {
            // Default fast path: reflect-pad -> overlapping-row expansion (im2col)
            // -> ONE MPS GEMM (K = kernel*in_ch) per batch element, replacing
            // `kernel` serialized accumulate-GEMMs per batch.
            const uint32_t padded_tokens = tokens + 2 * pad;
            const uint32_t big_k = kernel * in_ch;
            uint32_t b_ = batch, t_ = tokens, w_ = in_ch, p_ = pad;
            [impl_->pass_enc setComputePipelineState:impl_->reflect_pad_rows_batched_pipeline];
            [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
            [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x_pad.byte_offset atIndex:1];
            [impl_->pass_enc setBytes:&b_ length:sizeof(b_) atIndex:2];
            [impl_->pass_enc setBytes:&t_ length:sizeof(t_) atIndex:3];
            [impl_->pass_enc setBytes:&w_ length:sizeof(w_) atIndex:4];
            [impl_->pass_enc setBytes:&p_ length:sizeof(p_) atIndex:5];
            [impl_->pass_enc dispatchThreads:MTLSizeMake(in_ch, padded_tokens, batch)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(in_ch, impl_->reflect_pad_rows_batched_pipeline.threadExecutionWidth), 1, 1)];
            [impl_->pass_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Tap-permuted merged weight resident (fp16), W'[oc][k*in_ch+ic].
            const std::string wconv_key = wk + ".wconv.f16";
            id<MTLBuffer> wconv;
            if (impl_->has_resident(wconv_key)) {
                wconv = impl_->resident_buffer_f16_from_f32(wconv_key, nullptr, 0);
            } else {
                if (w.size() < static_cast<size_t>(out_ch) * in_ch * kernel) {
                    throw std::invalid_argument("conv wconv resident missing and weight data not provided: " + wconv_key);
                }
                std::vector<float> wperm(static_cast<size_t>(out_ch) * in_ch * kernel);
                for (uint32_t oc = 0; oc < out_ch; ++oc) {
                    for (uint32_t k = 0; k < kernel; ++k) {
                        for (uint32_t ic = 0; ic < in_ch; ++ic) {
                            wperm[(static_cast<size_t>(oc) * kernel + k) * in_ch + ic] =
                                w[(static_cast<size_t>(oc) * in_ch + ic) * kernel + k];
                        }
                    }
                }
                wconv = impl_->resident_buffer_f16_from_f32(wconv_key, wperm.data(), wperm.size());
            }
            auto a_big = impl_->pass_alloc_raw(batch * tokens * big_k);
            uint32_t rl = big_k;
            for (uint32_t bb = 0; bb < batch; ++bb) {
                const NSUInteger in_off = x_pad.byte_offset + static_cast<NSUInteger>(bb) * padded_tokens * in_ch * sizeof(float);
                const NSUInteger out_off = a_big.byte_offset + static_cast<NSUInteger>(bb) * tokens * big_k * sizeof(float);
                [impl_->pass_enc setComputePipelineState:impl_->overlap_rows_pipeline];
                [impl_->pass_enc setBuffer:impl_->pass_workspace offset:in_off atIndex:0];
                [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out_off atIndex:1];
                [impl_->pass_enc setBytes:&t_ length:sizeof(t_) atIndex:2];
                [impl_->pass_enc setBytes:&w_ length:sizeof(w_) atIndex:3];
                [impl_->pass_enc setBytes:&rl length:sizeof(rl) atIndex:4];
                [impl_->pass_enc dispatchThreads:MTLSizeMake(big_k, tokens, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            for (uint32_t bb = 0; bb < batch; ++bb) {
                const NSUInteger a_off = a_big.byte_offset + static_cast<NSUInteger>(bb) * tokens * big_k * sizeof(float);
                const NSUInteger c_off = out_mps.byte_offset + static_cast<NSUInteger>(bb) * tokens * out_ch * sizeof(float);
                impl_->pass_linear_rows_mps(wconv, true, bbuf_mps, a_off, c_off, tokens, out_ch, big_k);
            }
            return out_mps;
        }
        if (taps_fp16 && custom_gemm_enabled()) {
            // Reflect-pad, then ONE overlapping-row GEMM per batch element:
            // A row t = X_pad[t .. t+kernel-1] flattened (contiguous since the
            // padded rows are dense), so lda = in_ch < K = kernel*in_ch, against
            // the tap-permuted weight W'[oc][k*in_ch+ic] = conv_w[oc][ic][k].
            const uint32_t padded_tokens = tokens + 2 * pad;
            uint32_t b_ = batch, t_ = tokens, w_ = in_ch, p_ = pad;
            [impl_->pass_enc setComputePipelineState:impl_->reflect_pad_rows_batched_pipeline];
            [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
            [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x_pad.byte_offset atIndex:1];
            [impl_->pass_enc setBytes:&b_ length:sizeof(b_) atIndex:2];
            [impl_->pass_enc setBytes:&t_ length:sizeof(t_) atIndex:3];
            [impl_->pass_enc setBytes:&w_ length:sizeof(w_) atIndex:4];
            [impl_->pass_enc setBytes:&p_ length:sizeof(p_) atIndex:5];
            [impl_->pass_enc dispatchThreads:MTLSizeMake(in_ch, padded_tokens, batch)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(in_ch, impl_->reflect_pad_rows_batched_pipeline.threadExecutionWidth), 1, 1)];
            [impl_->pass_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Tap-permuted merged weight (resident, fp16).
            const std::string wconv_key = wk + ".wconv.f16";
            id<MTLBuffer> wconv;
            if (impl_->has_resident(wconv_key)) {
                wconv = impl_->resident_buffer_f16_from_f32(wconv_key, nullptr, 0);
            } else {
                if (w.size() < static_cast<size_t>(out_ch) * in_ch * kernel) {
                    throw std::invalid_argument("conv wconv resident missing and weight data not provided: " + wconv_key);
                }
                std::vector<float> wperm(static_cast<size_t>(out_ch) * in_ch * kernel);
                for (uint32_t oc = 0; oc < out_ch; ++oc) {
                    for (uint32_t k = 0; k < kernel; ++k) {
                        for (uint32_t ic = 0; ic < in_ch; ++ic) {
                            wperm[(static_cast<size_t>(oc) * kernel + k) * in_ch + ic] =
                                w[(static_cast<size_t>(oc) * in_ch + ic) * kernel + k];
                        }
                    }
                }
                wconv = impl_->resident_buffer_f16_from_f32(wconv_key, wperm.data(), wperm.size());
            }
            const uint32_t big_k = kernel * in_ch;
            for (uint32_t bb = 0; bb < batch; ++bb) {
                const NSUInteger a_off = x_pad.byte_offset +
                    static_cast<NSUInteger>(bb) * padded_tokens * in_ch * sizeof(float);
                const NSUInteger c_off = out_mps.byte_offset +
                    static_cast<NSUInteger>(bb) * tokens * out_ch * sizeof(float);
                impl_->pass_gemm_f16w(wconv, bbuf_mps, true, false, a_off, c_off,
                                      tokens, out_ch, big_k, in_ch);
            }
            [impl_->pass_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            return out_mps;
        }
        impl_->pass_conv1d_reflect_batched_mps(x.byte_offset, x_pad.byte_offset, out_mps.byte_offset,
                                               tap_bufs, taps_fp16, bbuf_mps, batch, tokens, in_ch, out_ch, kernel);
        return out_mps;
    }
    id<MTLBuffer> wbuf = impl_->resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    id<MTLBuffer> bbuf = bbuf_mps;
    auto out = impl_->pass_alloc_raw(batch * tokens * out_ch);
    [impl_->pass_enc setComputePipelineState:impl_->conv1d_reflect_same_batched_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:wbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&batch length:sizeof(batch) atIndex:4];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:5];
    [impl_->pass_enc setBytes:&in_ch length:sizeof(in_ch) atIndex:6];
    [impl_->pass_enc setBytes:&out_ch length:sizeof(out_ch) atIndex:7];
    [impl_->pass_enc setBytes:&kernel length:sizeof(kernel) atIndex:8];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(out_ch, tokens, batch)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_ch, impl_->conv1d_reflect_same_batched_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

PassSlot MetalContext::wavenet_gate_f32_pass(PassSlot in_layer, PassSlot cond, uint32_t tokens, uint32_t width, uint32_t cond_width, uint32_t cond_offset_val, uint32_t cond_tokens) {
    PASS_REQUIRE_AND_BARRIER();
    auto out = impl_->pass_alloc_raw(tokens * width);
    [impl_->pass_enc setComputePipelineState:impl_->wavenet_gate_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:in_layer.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:cond.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:4];
    [impl_->pass_enc setBytes:&cond_width length:sizeof(cond_width) atIndex:5];
    [impl_->pass_enc setBytes:&cond_offset_val length:sizeof(cond_offset_val) atIndex:6];
    [impl_->pass_enc setBytes:&cond_tokens length:sizeof(cond_tokens) atIndex:7];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(width, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(width, impl_->wavenet_gate_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

static void dispatch_wavenet_res_skip_update(id<MTLComputeCommandEncoder> enc,
                                              id<MTLComputePipelineState> pipeline,
                                              id<MTLBuffer> workspace,
                                              NSUInteger x_off, NSUInteger out_off,
                                              NSUInteger res_skip_off, NSUInteger mask_off,
                                              NSUInteger combined_out_off,
                                              uint32_t tokens, uint32_t width, bool has_residual) {
    uint32_t has_res_u32 = has_residual ? 1u : 0u;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:workspace offset:x_off atIndex:0];
    [enc setBuffer:workspace offset:out_off atIndex:1];
    [enc setBuffer:workspace offset:res_skip_off atIndex:2];
    [enc setBuffer:workspace offset:mask_off atIndex:3];
    [enc setBuffer:workspace offset:combined_out_off atIndex:4];
    [enc setBytes:&tokens length:sizeof(tokens) atIndex:5];
    [enc setBytes:&width length:sizeof(width) atIndex:6];
    [enc setBytes:&has_res_u32 length:sizeof(has_res_u32) atIndex:7];
    [enc dispatchThreads:MTLSizeMake(width, tokens, 1)
       threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(width, pipeline.threadExecutionWidth), 1, 1)];
}

PassSlot MetalContext::wavenet_res_skip_update_f32_pass(PassSlot x, PassSlot out, PassSlot res_skip, PassSlot mask, uint32_t tokens, uint32_t width, bool has_residual) {
    PASS_REQUIRE_AND_BARRIER();
    auto combined = impl_->pass_alloc_raw(tokens * width * 2);
    dispatch_wavenet_res_skip_update(impl_->pass_enc, impl_->wavenet_res_skip_update_pipeline,
                                     impl_->pass_workspace, x.byte_offset, out.byte_offset,
                                     res_skip.byte_offset, mask.byte_offset, combined.byte_offset,
                                     tokens, width, has_residual);
    return combined;
}

void MetalContext::wavenet_res_skip_update_f32_pass_into(PassSlot x, PassSlot out, PassSlot res_skip, PassSlot mask, uint32_t tokens, uint32_t width, bool has_residual, PassSlot combined_out) {
    PASS_REQUIRE_AND_BARRIER();
    dispatch_wavenet_res_skip_update(impl_->pass_enc, impl_->wavenet_res_skip_update_pipeline,
                                     impl_->pass_workspace, x.byte_offset, out.byte_offset,
                                     res_skip.byte_offset, mask.byte_offset, combined_out.byte_offset,
                                     tokens, width, has_residual);
}

PassSlot MetalContext::layernorm_rows_f32_pass(const std::string& gamma_key, const std::vector<float>& gamma,
                                               const std::string& beta_key, const std::vector<float>& beta,
                                               PassSlot x, uint32_t tokens, uint32_t width, float eps) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> gbuf = impl_->resident_buffer_with_bytes(gamma_key, gamma.data(), gamma.size() * sizeof(float));
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(beta_key, beta.data(), beta.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(tokens * width);
    [impl_->pass_enc setComputePipelineState:impl_->layernorm_rows_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:gbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:5];
    [impl_->pass_enc setBytes:&eps length:sizeof(eps) atIndex:6];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::gpt_causal_attention_f32_pass(PassSlot qkv, uint32_t tokens, uint32_t heads, uint32_t head_dim) {
    PASS_REQUIRE_AND_BARRIER();
    const uint32_t width = heads * head_dim;
    if (tokens == 0 || tokens > 4096 || (1024 % head_dim) != 0) {
        throw std::invalid_argument("gpt_causal_attention_f32_pass invalid args");
    }
    auto out = impl_->pass_alloc_raw(tokens * width);
    if (head_dim == 64 && sgq_attention_enabled()) {
        [impl_->pass_enc setComputePipelineState:impl_->gpt_causal_attention_sgq_pipeline];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:qkv.byte_offset atIndex:0];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
        [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:2];
        [impl_->pass_enc setBytes:&heads length:sizeof(heads) atIndex:3];
        [impl_->pass_enc setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
        [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(heads, (tokens + 31) / 32, 1)
                   threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        return out;
    }
    [impl_->pass_enc setComputePipelineState:impl_->gpt_causal_attention_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:qkv.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:2];
    [impl_->pass_enc setBytes:&heads length:sizeof(heads) atIndex:3];
    [impl_->pass_enc setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::layernorm_f32_pass(
    const std::string& gamma_key, const std::vector<float>& gamma,
    const std::string& beta_key, const std::vector<float>& beta,
    PassSlot x, uint32_t count, float eps) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> gbuf = impl_->resident_buffer_with_bytes(gamma_key, gamma.data(), gamma.size() * sizeof(float));
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(beta_key, beta.data(), beta.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(count);
    [impl_->pass_enc setComputePipelineState:impl_->layernorm_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:gbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:4];
    [impl_->pass_enc setBytes:&eps length:sizeof(eps) atIndex:5];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::gelu_f32_pass(PassSlot x, uint32_t count) {
    PASS_REQUIRE_AND_BARRIER();
    auto out = impl_->pass_alloc_raw(count);
    [impl_->pass_enc setComputePipelineState:impl_->gelu_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:2];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(impl_->gelu_pipeline.threadExecutionWidth, 1, 1)];
    return out;
}

std::vector<float> MetalContext::gpt_causal_attention_f32(const std::vector<float>& qkv, uint32_t tokens, uint32_t heads, uint32_t head_dim) {
    const uint32_t width = heads * head_dim;
    const size_t qkv_count = static_cast<size_t>(tokens) * width * 3;
    const size_t out_count = static_cast<size_t>(tokens) * width;
    if (tokens == 0 || tokens > 4096 || heads == 0 || head_dim == 0 || (1024 % head_dim) != 0 ||
        qkv.size() != qkv_count) {
        throw std::invalid_argument("gpt_causal_attention_f32 invalid input sizes");
    }
    @autoreleasepool {
        constexpr size_t alignment = 256;
        const NSUInteger qkv_bytes = qkv_count * sizeof(float);
        const NSUInteger out_bytes = out_count * sizeof(float);
        const size_t qkv_offset = 0;
        const size_t out_offset = align_up_size(qkv_offset + qkv_bytes, alignment);
        const size_t arena_bytes = arena_capacity_size(out_offset + out_bytes, alignment);
        id<MTLBuffer> arena = impl_->scratch_arena_buffer(static_cast<NSUInteger>(arena_bytes));
        uint8_t* base = static_cast<uint8_t*>([arena contents]);
        std::memcpy(base + qkv_offset, qkv.data(), qkv_bytes);
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->gpt_causal_attention_pipeline];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(qkv_offset) atIndex:0];
        [enc setBuffer:arena offset:static_cast<NSUInteger>(out_offset) atIndex:1];
        [enc setBytes:&tokens length:sizeof(tokens) atIndex:2];
        [enc setBytes:&heads length:sizeof(heads) atIndex:3];
        [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1)
                   threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [enc endEncoding];
        commit_and_count(impl_->command_buffers_submitted, cb);
        wait_and_record(impl_->gpu_elapsed_seconds, cb, impl_->batch_mode);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("gpt_causal_attention_f32 command failed");
        }
        float* ptr = reinterpret_cast<float*>(base + out_offset);
        return std::vector<float>(ptr, ptr + out_count);
    }
}

bool MetalContext::gptIcbAvailable() const {
    return impl_->gpt_icb_ready;
}

void MetalContext::gptIcbInvalidate() {
    impl_->gpt_icb_ready = false;
}

void MetalContext::gptIcbBeginRecord(uint32_t max_commands, size_t ws_bytes, uint32_t max_history) {
    Impl* im = impl_;
    if (!im->gpt_icb_ws || im->gpt_icb_ws_capacity < ws_bytes) {
        im->gpt_icb_ws = new_counted_buffer_with_length(im->device, im->buffer_allocations,
                                                        im->buffer_bytes_allocated, ws_bytes);
        im->gpt_icb_ws_capacity = ws_bytes;
    }
    if (!im->gpt_icb_state) {
        const uint32_t zeros[3] = {0, 0, 0};
        im->gpt_icb_state = new_counted_buffer_with_bytes(im->device, im->buffer_allocations,
                                                          im->buffer_bytes_allocated, zeros, sizeof(zeros));
    }
    if (!im->gpt_icb_history || im->gpt_icb_history_capacity < max_history) {
        im->gpt_icb_history = new_counted_buffer_with_length(im->device, im->buffer_allocations,
                                                             im->buffer_bytes_allocated,
                                                             static_cast<NSUInteger>(max_history) * sizeof(uint32_t));
        im->gpt_icb_history_capacity = max_history;
    }
    MTLIndirectCommandBufferDescriptor* desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    desc.inheritPipelineState = NO;
    desc.inheritBuffers = NO;
    desc.maxKernelBufferBindCount = 11;
    im->gpt_icb = [im->device newIndirectCommandBufferWithDescriptor:desc
                                                     maxCommandCount:max_commands
                                                             options:0];
    if (!im->gpt_icb) {
        throw std::runtime_error("failed to create GPT indirect command buffer");
    }
    im->gpt_icb_max_commands = max_commands;
    im->gpt_icb_cmd_cursor = 0;
    im->gpt_icb_ws_cursor = 0;
    im->gpt_icb_recording = true;
    im->gpt_icb_ready = false;
    im->gpt_icb_read_resources.clear();
}

PassSlot MetalContext::gptIcbAlloc(uint32_t element_count) {
    return impl_->icb_alloc_raw(element_count);
}

PassSlot MetalContext::gptIcb_build_current(PassSlot token_slot,
                                            const std::string& mel_emb_key, const std::vector<float>& mel_emb,
                                            const std::string& pos_emb_key, const std::vector<float>& pos_emb,
                                            uint32_t width) {
    Impl* im = impl_;
    id<MTLBuffer> mbuf = im->resident_buffer_with_bytes(mel_emb_key, mel_emb.data(), mel_emb.size() * sizeof(float));
    id<MTLBuffer> pbuf = im->resident_buffer_with_bytes(pos_emb_key, pos_emb.data(), pos_emb.size() * sizeof(float));
    im->icb_track_read(mbuf);
    im->icb_track_read(pbuf);
    auto out = im->icb_alloc_raw(width);
    auto cmd = im->icb_next_command();
    [cmd setComputePipelineState:im->gpt_build_current_pipeline];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:token_slot.byte_offset atIndex:0];
    [cmd setKernelBuffer:mbuf offset:0 atIndex:1];
    [cmd setKernelBuffer:pbuf offset:0 atIndex:2];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:out.byte_offset atIndex:3];
    [cmd setKernelBuffer:im->icb_const_u32(width) offset:0 atIndex:4];
    [cmd setKernelBuffer:im->gpt_icb_state offset:4 atIndex:5];  // position
    [cmd setBarrier];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake((width + 255) / 256, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return out;
}

PassSlot MetalContext::gptIcb_fused_gemv_f16w(const std::string& wk, const std::vector<float>& w,
                                              const std::string& bk, const std::vector<float>& b,
                                              PassSlot x, uint32_t rows, uint32_t cols,
                                              bool has_ln, const std::string& gk, const std::vector<float>& g,
                                              const std::string& bek, const std::vector<float>& be,
                                              bool fuse_gelu, bool has_residual, PassSlot residual, float eps) {
    Impl* im = impl_;
    if (cols > 1280 && has_ln) {
        throw std::invalid_argument("gptIcb_fused_gemv_f16w: ln staging supports cols <= 1280");
    }
    id<MTLBuffer> wbuf = im->resident_buffer_f16_from_f32(wk + ".f16", w.data(), w.size());
    id<MTLBuffer> bbuf = im->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    id<MTLBuffer> gbuf = has_ln ? im->resident_buffer_with_bytes(gk, g.data(), g.size() * sizeof(float)) : bbuf;
    id<MTLBuffer> bebuf = has_ln ? im->resident_buffer_with_bytes(bek, be.data(), be.size() * sizeof(float)) : bbuf;
    im->icb_track_read(wbuf);
    im->icb_track_read(bbuf);
    im->icb_track_read(gbuf);
    im->icb_track_read(bebuf);
    auto out = im->icb_alloc_raw(rows);
    const uint32_t flags = (has_ln ? 1u : 0u) | (fuse_gelu ? 2u : 0u) | (has_residual ? 4u : 0u);
    const NSUInteger res_off = has_residual ? residual.byte_offset : x.byte_offset;
    auto cmd = im->icb_next_command();
    [cmd setComputePipelineState:im->gpt_fused_gemv_f16w_pipeline];
    [cmd setKernelBuffer:wbuf offset:0 atIndex:0];
    [cmd setKernelBuffer:bbuf offset:0 atIndex:1];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:x.byte_offset atIndex:2];
    [cmd setKernelBuffer:gbuf offset:0 atIndex:3];
    [cmd setKernelBuffer:bebuf offset:0 atIndex:4];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:res_off atIndex:5];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:out.byte_offset atIndex:6];
    [cmd setKernelBuffer:im->icb_const_u32(rows) offset:0 atIndex:7];
    [cmd setKernelBuffer:im->icb_const_u32(cols) offset:0 atIndex:8];
    [cmd setKernelBuffer:im->icb_const_f32(eps) offset:0 atIndex:9];
    [cmd setKernelBuffer:im->icb_const_u32(flags) offset:0 atIndex:10];
    [cmd setBarrier];
    constexpr uint32_t sgs_per_tg = 8;
    [cmd concurrentDispatchThreadgroups:MTLSizeMake((rows + sgs_per_tg - 1) / sgs_per_tg, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(sgs_per_tg * 32, 1, 1)];
    return out;
}

PassSlot MetalContext::gptIcb_attention_resident(uint32_t layer, PassSlot qkv, uint32_t heads, uint32_t head_dim) {
    Impl* im = impl_;
    if (!im->gpt_kv_buffer || layer >= im->gpt_kv_layers) {
        throw std::invalid_argument("gptIcb_attention_resident: kv cache missing");
    }
    const uint32_t width = heads * head_dim;
    auto out = im->icb_alloc_raw(width);
    auto cmd = im->icb_next_command();
    [cmd setComputePipelineState:im->gpt_cached_attention_resident_pipeline];
    [cmd setKernelBuffer:im->gpt_kv_buffer offset:im->gpt_kv_k_offset(layer) atIndex:0];
    [cmd setKernelBuffer:im->gpt_kv_buffer offset:im->gpt_kv_v_offset(layer) atIndex:1];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:qkv.byte_offset atIndex:2];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:out.byte_offset atIndex:3];
    [cmd setKernelBuffer:im->gpt_icb_state offset:0 atIndex:4];  // kv_tokens
    [cmd setKernelBuffer:im->icb_const_u32(heads) offset:0 atIndex:5];
    [cmd setKernelBuffer:im->icb_const_u32(head_dim) offset:0 atIndex:6];
    [cmd setBarrier];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(heads, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::gptIcb_layernorm(const std::string& gamma_key, const std::vector<float>& gamma,
                                        const std::string& beta_key, const std::vector<float>& beta,
                                        PassSlot x, uint32_t count, float eps) {
    Impl* im = impl_;
    id<MTLBuffer> gbuf = im->resident_buffer_with_bytes(gamma_key, gamma.data(), gamma.size() * sizeof(float));
    id<MTLBuffer> bbuf = im->resident_buffer_with_bytes(beta_key, beta.data(), beta.size() * sizeof(float));
    im->icb_track_read(gbuf);
    im->icb_track_read(bbuf);
    auto out = im->icb_alloc_raw(count);
    auto cmd = im->icb_next_command();
    [cmd setComputePipelineState:im->layernorm_pipeline];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:x.byte_offset atIndex:0];
    [cmd setKernelBuffer:gbuf offset:0 atIndex:1];
    [cmd setKernelBuffer:bbuf offset:0 atIndex:2];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:out.byte_offset atIndex:3];
    [cmd setKernelBuffer:im->icb_const_u32(count) offset:0 atIndex:4];
    [cmd setKernelBuffer:im->icb_const_f32(eps) offset:0 atIndex:5];
    [cmd setBarrier];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

void MetalContext::gptIcb_argmax_into(PassSlot logits, uint32_t vocab, PassSlot token_slot) {
    Impl* im = impl_;
    auto cmd = im->icb_next_command();
    [cmd setComputePipelineState:im->gpt_argmax_pipeline];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:logits.byte_offset atIndex:0];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:token_slot.byte_offset atIndex:1];
    [cmd setKernelBuffer:im->icb_const_u32(vocab) offset:0 atIndex:2];
    [cmd setBarrier];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

void MetalContext::gptIcb_record_token(PassSlot token_slot) {
    Impl* im = impl_;
    auto cmd = im->icb_next_command();
    [cmd setComputePipelineState:im->gpt_record_token_pipeline];
    [cmd setKernelBuffer:im->gpt_icb_ws offset:token_slot.byte_offset atIndex:0];
    [cmd setKernelBuffer:im->gpt_icb_history offset:0 atIndex:1];
    [cmd setKernelBuffer:im->gpt_icb_state offset:0 atIndex:2];
    [cmd setBarrier];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void MetalContext::gptIcb_advance_state() {
    Impl* im = impl_;
    auto cmd = im->icb_next_command();
    [cmd setComputePipelineState:im->gpt_icb_advance_pipeline];
    [cmd setKernelBuffer:im->gpt_icb_state offset:0 atIndex:0];
    [cmd setBarrier];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void MetalContext::gptIcbEndRecord(PassSlot token_slot, PassSlot logits_slot) {
    Impl* im = impl_;
    im->gpt_icb_token_off = token_slot.byte_offset;
    im->gpt_icb_logits_off = logits_slot.byte_offset;
    im->gpt_icb_recording = false;
    im->gpt_icb_ready = true;
}

MetalContext::GptIcbResult MetalContext::gptIcbExecute(uint32_t n_tokens, uint32_t seed_token,
                                                       uint32_t kv_tokens_start, uint32_t step_start,
                                                       uint32_t vocab) {
    Impl* im = impl_;
    if (!im->gpt_icb_ready) {
        throw std::logic_error("gptIcbExecute: ICB not recorded");
    }
    if (step_start + n_tokens > im->gpt_icb_history_capacity) {
        throw std::invalid_argument("gptIcbExecute: history capacity exceeded");
    }
    @autoreleasepool {
        uint32_t* st = static_cast<uint32_t*>([im->gpt_icb_state contents]);
        st[0] = kv_tokens_start;
        st[2] = step_start;
        st[1] = (step_start == 0) ? 0 : step_start + 1;
        uint32_t* tok = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>([im->gpt_icb_ws contents]) + im->gpt_icb_token_off);
        tok[0] = seed_token;

        id<MTLCommandBuffer> cb = [im->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (id<MTLResource> r : im->gpt_icb_read_resources) {
            [enc useResource:r usage:MTLResourceUsageRead];
        }
        [enc useResource:im->gpt_icb_ws usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        [enc useResource:im->gpt_icb_state usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        [enc useResource:im->gpt_icb_history usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        [enc useResource:im->gpt_kv_buffer usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        for (uint32_t i = 0; i < n_tokens; ++i) {
            [enc executeCommandsInBuffer:im->gpt_icb withRange:NSMakeRange(0, im->gpt_icb_cmd_cursor)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [enc endEncoding];
        commit_and_count(im->command_buffers_submitted, cb);
        wait_and_record(im->gpu_elapsed_seconds, cb);
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("gpt icb command buffer failed");
        }

        GptIcbResult result;
        const uint32_t* hist = static_cast<const uint32_t*>([im->gpt_icb_history contents]);
        result.tokens.assign(hist + step_start, hist + step_start + n_tokens);
        const float* logits = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>([im->gpt_icb_ws contents]) + im->gpt_icb_logits_off);
        result.last_logits.assign(logits, logits + vocab);
        return result;
    }
}

void MetalContext::gptKvCacheCreate(uint32_t layers, uint32_t max_tokens, uint32_t width) {
    if (layers == 0 || max_tokens == 0 || width == 0) {
        throw std::invalid_argument("gptKvCacheCreate invalid dims");
    }
    // Layout stability: per-layer offsets depend on gpt_kv_max_tokens, and the
    // decode ICB bakes those offsets in. Use a generous capacity that only
    // grows, so offsets stay stable across segments with different lengths.
    constexpr uint32_t kKvCapacityFloor = 2048;
    uint32_t capacity = std::max(max_tokens, kKvCapacityFloor);
    if (impl_->gpt_kv_buffer && impl_->gpt_kv_layers == layers && impl_->gpt_kv_width == width &&
        impl_->gpt_kv_max_tokens >= capacity) {
        return;  // existing layout still valid (and ICB stays valid)
    }
    capacity = std::max(capacity, impl_->gpt_kv_max_tokens);
    // Preserve FP32 keys/values for parity with the actual 2.5 reference.
    const NSUInteger bytes = static_cast<NSUInteger>(layers) * 2 * capacity * width * sizeof(float);
    if (!impl_->gpt_kv_buffer || [impl_->gpt_kv_buffer length] < bytes) {
        impl_->gpt_kv_buffer = new_counted_buffer_with_length(
            impl_->device, impl_->buffer_allocations, impl_->buffer_bytes_allocated, bytes);
    }
    impl_->gpt_kv_layers = layers;
    impl_->gpt_kv_max_tokens = capacity;
    impl_->gpt_kv_width = width;
    // Offsets changed — any recorded decode ICB is stale.
    impl_->gpt_icb_ready = false;
}

std::vector<float> MetalContext::gptKvCacheRead(uint32_t tokens) const {
    if(impl_->pass_mode || !impl_->gpt_kv_buffer || !tokens || tokens>impl_->gpt_kv_max_tokens)throw std::invalid_argument("Invalid GPT cache snapshot");
    const size_t count=static_cast<size_t>(tokens)*impl_->gpt_kv_width;
    std::vector<float> values(count*2*impl_->gpt_kv_layers);
    const auto* bytes=static_cast<const uint8_t*>([impl_->gpt_kv_buffer contents]);
    for(uint32_t layer=0;layer<impl_->gpt_kv_layers;layer++) {
        std::memcpy(values.data()+(2*layer)*count,bytes+impl_->gpt_kv_k_offset(layer),count*4);
        std::memcpy(values.data()+(2*layer+1)*count,bytes+impl_->gpt_kv_v_offset(layer),count*4);
    }
    return values;
}
void MetalContext::gptKvCacheWrite(const std::vector<float>& values,uint32_t tokens) {
    const size_t count=static_cast<size_t>(tokens)*impl_->gpt_kv_width;
    if(impl_->pass_mode || !impl_->gpt_kv_buffer || !tokens || tokens>impl_->gpt_kv_max_tokens || values.size()!=count*2*impl_->gpt_kv_layers)throw std::invalid_argument("Invalid GPT cache restore");
    auto* bytes=static_cast<uint8_t*>([impl_->gpt_kv_buffer contents]);
    for(uint32_t layer=0;layer<impl_->gpt_kv_layers;layer++) {
        std::memcpy(bytes+impl_->gpt_kv_k_offset(layer),values.data()+(2*layer)*count,count*4);
        std::memcpy(bytes+impl_->gpt_kv_v_offset(layer),values.data()+(2*layer+1)*count,count*4);
    }
}
void MetalContext::gptKvCacheUpload(uint32_t layer, const std::vector<float>& k, const std::vector<float>& v, uint32_t tokens) {
    if (!impl_->gpt_kv_buffer || layer >= impl_->gpt_kv_layers || tokens > impl_->gpt_kv_max_tokens ||
        k.size() < static_cast<size_t>(tokens) * impl_->gpt_kv_width ||
        v.size() < static_cast<size_t>(tokens) * impl_->gpt_kv_width) {
        throw std::invalid_argument("gptKvCacheUpload invalid args");
    }
    uint8_t* base = static_cast<uint8_t*>([impl_->gpt_kv_buffer contents]);
    const size_t count = static_cast<size_t>(tokens) * impl_->gpt_kv_width;
    float* dk = reinterpret_cast<float*>(base + impl_->gpt_kv_k_offset(layer));
    float* dv = reinterpret_cast<float*>(base + impl_->gpt_kv_v_offset(layer));
    for (size_t i = 0; i < count; ++i) {
        dk[i] = k[i];
        dv[i] = v[i];
    }
}

PassSlot MetalContext::gpt_fused_gemv_f16w_pass(const std::string& wk, const std::vector<float>& w,
                                                const std::string& bk, const std::vector<float>& b,
                                                PassSlot x, uint32_t rows, uint32_t cols,
                                                bool has_ln, const std::string& gk, const std::vector<float>& g,
                                                const std::string& bek, const std::vector<float>& be,
                                                bool fuse_gelu, bool has_residual, PassSlot residual, float eps) {
    PASS_REQUIRE_AND_BARRIER();
    if (cols > 1280 && has_ln) {
        throw std::invalid_argument("gpt_fused_gemv_f16w_pass: ln staging supports cols <= 1280");
    }
    id<MTLBuffer> wbuf = impl_->resident_buffer_f16_from_f32(wk + ".f16", w.data(), w.size());
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    // gamma/beta: bind a valid buffer even when unused (Metal requires bindings).
    id<MTLBuffer> gbuf = has_ln
        ? impl_->resident_buffer_with_bytes(gk, g.data(), g.size() * sizeof(float))
        : bbuf;
    id<MTLBuffer> bebuf = has_ln
        ? impl_->resident_buffer_with_bytes(bek, be.data(), be.size() * sizeof(float))
        : bbuf;
    auto out = impl_->pass_alloc_raw(rows);
    const uint32_t flags = (has_ln ? 1u : 0u) | (fuse_gelu ? 2u : 0u) | (has_residual ? 4u : 0u);
    const NSUInteger res_off = has_residual ? residual.byte_offset : x.byte_offset;
    [impl_->pass_enc setComputePipelineState:impl_->gpt_fused_gemv_f16w_pipeline];
    [impl_->pass_enc setBuffer:wbuf offset:0 atIndex:0];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:gbuf offset:0 atIndex:3];
    [impl_->pass_enc setBuffer:bebuf offset:0 atIndex:4];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:res_off atIndex:5];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:6];
    [impl_->pass_enc setBytes:&rows length:sizeof(rows) atIndex:7];
    [impl_->pass_enc setBytes:&cols length:sizeof(cols) atIndex:8];
    [impl_->pass_enc setBytes:&eps length:sizeof(eps) atIndex:9];
    [impl_->pass_enc setBytes:&flags length:sizeof(flags) atIndex:10];
    constexpr uint32_t sgs_per_tg = 8;
    const uint32_t tgs = (rows + sgs_per_tg - 1) / sgs_per_tg;
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(sgs_per_tg * 32, 1, 1)];
    return out;
}

void MetalContext::gptKvStoreFromQkv_pass(uint32_t layer, PassSlot qkv, uint32_t tokens) {
    PASS_REQUIRE_AND_BARRIER();
    if (!impl_->gpt_kv_buffer || layer >= impl_->gpt_kv_layers || tokens > impl_->gpt_kv_max_tokens) {
        throw std::invalid_argument("gptKvStoreFromQkv_pass invalid args");
    }
    const uint32_t width = impl_->gpt_kv_width;
    [impl_->pass_enc setComputePipelineState:impl_->gpt_kv_store_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:qkv.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->gpt_kv_buffer offset:impl_->gpt_kv_k_offset(layer) atIndex:1];
    [impl_->pass_enc setBuffer:impl_->gpt_kv_buffer offset:impl_->gpt_kv_v_offset(layer) atIndex:2];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:4];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(width, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(width, impl_->gpt_kv_store_pipeline.threadExecutionWidth), 1, 1)];
}

void MetalContext::cfm_euler_update_f32_pass_into(PassSlot x, PassSlot dphi, PassSlot cfg_dphi, PassSlot out,
                                                  uint32_t tokens, uint32_t width, uint32_t prompt_tokens,
                                                  float dt, float cfg_rate) {
    PASS_REQUIRE_AND_BARRIER();
    [impl_->pass_enc setComputePipelineState:impl_->cfm_euler_update_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:dphi.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:cfg_dphi.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:5];
    [impl_->pass_enc setBytes:&prompt_tokens length:sizeof(prompt_tokens) atIndex:6];
    [impl_->pass_enc setBytes:&dt length:sizeof(dt) atIndex:7];
    [impl_->pass_enc setBytes:&cfg_rate length:sizeof(cfg_rate) atIndex:8];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(width, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(width, impl_->cfm_euler_update_pipeline.threadExecutionWidth), 1, 1)];
}

void MetalContext::copy_f32_pass_into(PassSlot src, PassSlot dst, uint32_t count) {
    PASS_REQUIRE_AND_BARRIER();
    [impl_->pass_enc setComputePipelineState:impl_->copy_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:src.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:dst.byte_offset atIndex:1];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:2];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(impl_->copy_pipeline.threadExecutionWidth, 1, 1)];
}

PassSlot MetalContext::conv1d_dilated_same_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel, uint32_t dilation) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> wbuf = impl_->resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    const float* weight_data = static_cast<const float*>([wbuf contents]);
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(tokens * out_ch);
    const bool bigvgan_resblock = wk.find("bigvgan.resblocks.") != std::string::npos;
    const bool mps_tap_backend =
        bigvgan_resblock &&
        bigvgan_im2col_enabled_for_device() &&
        (kernel % 2) == 1 &&
        (linear_rows_use_mps(tokens, out_ch, in_ch) ||
         (tokens >= 256 && static_cast<uint64_t>(tokens) * out_ch * in_ch >= 1048576));
    if (mps_tap_backend) {
        const bool taps_fp16 = fp16_weights_enabled();
        std::vector<id<MTLBuffer>> tap_bufs(kernel);
        for (uint32_t k = 0; k < kernel; ++k) {
            const std::string tap_key = wk + ".tap" + std::to_string(k) + (taps_fp16 ? ".f16" : ".f32");
            if (impl_->has_resident(tap_key)) {
                tap_bufs[k] = taps_fp16 ? impl_->resident_buffer_f16_from_f32(tap_key, nullptr, 0)
                                        : impl_->resident_buffer_with_bytes(tap_key, nullptr, 0);
            } else {
                std::vector<float> tap(static_cast<size_t>(out_ch) * in_ch);
                for (uint32_t oc = 0; oc < out_ch; ++oc) {
                    for (uint32_t ic = 0; ic < in_ch; ++ic) {
                        tap[static_cast<size_t>(oc) * in_ch + ic] =
                            weight_data[(static_cast<size_t>(oc) * in_ch + ic) * kernel + k];
                    }
                }
                tap_bufs[k] = taps_fp16 ? impl_->resident_buffer_f16_from_f32(tap_key, tap.data(), tap.size())
                                        : impl_->resident_buffer_with_bytes(tap_key, tap.data(), tap.size() * sizeof(float));
            }
        }

        impl_->pass_ensure_encoder();
        uint32_t t_ = tokens, r_ = out_ch;
        [impl_->pass_enc setComputePipelineState:impl_->broadcast_bias_rows_pipeline];
        [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:0];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
        [impl_->pass_enc setBytes:&t_ length:sizeof(t_) atIndex:2];
        [impl_->pass_enc setBytes:&r_ length:sizeof(r_) atIndex:3];
        [impl_->pass_enc dispatchThreads:MTLSizeMake(out_ch, tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_ch, impl_->broadcast_bias_rows_pipeline.threadExecutionWidth), 1, 1)];
        [impl_->pass_enc endEncoding];
        impl_->pass_enc = nil;

        const int pad = static_cast<int>((kernel - 1) * dilation / 2);
        for (uint32_t k = 0; k < kernel; ++k) {
            const int kd = static_cast<int>(k * dilation);
            const int t_start_i = std::max(0, pad - kd);
            const int t_end_i = std::min(static_cast<int>(tokens), static_cast<int>(tokens) + pad - kd);
            if (t_end_i <= t_start_i) {
                continue;
            }
            const uint32_t valid_tokens = static_cast<uint32_t>(t_end_i - t_start_i);
            const uint32_t src_start = static_cast<uint32_t>(t_start_i + kd - pad);
            const NSUInteger a_off = x.byte_offset + static_cast<NSUInteger>(src_start) * in_ch * sizeof(float);
            const NSUInteger c_off = out.byte_offset + static_cast<NSUInteger>(t_start_i) * out_ch * sizeof(float);
            MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:valid_tokens
                                                                            columns:in_ch
                                                                           rowBytes:static_cast<NSUInteger>(in_ch) * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:out_ch
                                                                            columns:in_ch
                                                                           rowBytes:static_cast<NSUInteger>(in_ch) * (taps_fp16 ? sizeof(uint16_t) : sizeof(float))
                                                                           dataType:(taps_fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32)];
            MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:valid_tokens
                                                                            columns:out_ch
                                                                           rowBytes:static_cast<NSUInteger>(out_ch) * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
            MPSMatrix* A = [[MPSMatrix alloc] initWithBuffer:impl_->pass_workspace offset:a_off descriptor:dA];
            MPSMatrix* B = [[MPSMatrix alloc] initWithBuffer:tap_bufs[k] offset:0 descriptor:dB];
            MPSMatrix* C = [[MPSMatrix alloc] initWithBuffer:impl_->pass_workspace offset:c_off descriptor:dC];
            [impl_->mps_gemm(valid_tokens, out_ch, in_ch, true) encodeToCommandBuffer:impl_->pass_cb
                                                                           leftMatrix:A
                                                                          rightMatrix:B
                                                                        resultMatrix:C];
        }
        return out;
    }

    [impl_->pass_enc setComputePipelineState:impl_->conv1d_dilated_same_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:wbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [impl_->pass_enc setBytes:&in_ch length:sizeof(in_ch) atIndex:5];
    [impl_->pass_enc setBytes:&out_ch length:sizeof(out_ch) atIndex:6];
    [impl_->pass_enc setBytes:&kernel length:sizeof(kernel) atIndex:7];
    [impl_->pass_enc setBytes:&dilation length:sizeof(dilation) atIndex:8];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(out_ch, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_ch, impl_->conv1d_dilated_same_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

PassSlot MetalContext::conv_transpose1d_f32_pass(const std::string& wk, const std::vector<float>& w, const std::string& bk, const std::vector<float>& b, PassSlot x, uint32_t tokens, uint32_t in_ch, uint32_t out_ch, uint32_t kernel, uint32_t stride, uint32_t padding) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> wbuf = impl_->resident_buffer_with_bytes(wk, w.data(), w.size() * sizeof(float));
    const float* weight_data = static_cast<const float*>([wbuf contents]);
    id<MTLBuffer> bbuf = impl_->resident_buffer_with_bytes(bk, b.data(), b.size() * sizeof(float));
    const uint32_t out_tokens = (tokens - 1) * stride + kernel - 2 * padding;
    auto out = impl_->pass_alloc_raw(out_tokens * out_ch);
    // Each kernel tap is a GEMM into one strided output phase. Accumulate
    // overlapping phases with beta=1, retaining bias and edge cropping.
    const bool fast = wk.rfind("bigvgan.ups.", 0) == 0 &&
        bigvgan_im2col_enabled_for_device() && linear_rows_use_mps(tokens, out_ch, in_ch);
    if (fast) {
        [impl_->pass_enc setComputePipelineState:impl_->broadcast_bias_rows_pipeline];
        [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:0];
        [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
        [impl_->pass_enc setBytes:&out_tokens length:sizeof(out_tokens) atIndex:2];
        [impl_->pass_enc setBytes:&out_ch length:sizeof(out_ch) atIndex:3];
        [impl_->pass_enc dispatchThreads:MTLSizeMake(out_ch, out_tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_ch, impl_->broadcast_bias_rows_pipeline.threadExecutionWidth), 1, 1)];
        [impl_->pass_enc endEncoding]; impl_->pass_enc = nil;
        const bool half_weights = fp16_weights_enabled();
        for (uint32_t k = 0; k < kernel; ++k) {
            const std::string key = wk + ".tap" + std::to_string(k) + (half_weights ? ".f16" : ".f32");
            id<MTLBuffer> tap;
            if (impl_->has_resident(key)) {
                tap = half_weights ? impl_->resident_buffer_f16_from_f32(key, nullptr, 0)
                                   : impl_->resident_buffer_with_bytes(key, nullptr, 0);
            } else {
                if ([wbuf length] != static_cast<size_t>(in_ch) * out_ch * kernel * sizeof(float))
                    throw std::invalid_argument("Invalid transposed convolution weight size");
                std::vector<float> values(static_cast<size_t>(out_ch) * in_ch);
                for (uint32_t oc = 0; oc < out_ch; ++oc)
                    for (uint32_t ic = 0; ic < in_ch; ++ic)
                        values[static_cast<size_t>(oc) * in_ch + ic] = weight_data[(static_cast<size_t>(ic) * out_ch + oc) * kernel + k];
                tap = half_weights ? impl_->resident_buffer_f16_from_f32(key, values.data(), values.size())
                                   : impl_->resident_buffer_with_bytes(key, values.data(), values.size() * sizeof(float));
            }
            const int shift = static_cast<int>(k) - static_cast<int>(padding);
            const int begin = shift < 0 ? (-shift + static_cast<int>(stride) - 1) / static_cast<int>(stride) : 0;
            const int end = std::min(static_cast<int>(tokens),
                (static_cast<int>(out_tokens) - 1 - shift) / static_cast<int>(stride) + 1);
            if (end <= begin) continue;
            const uint32_t count = end - begin;
            const NSUInteger input_offset = x.byte_offset + static_cast<NSUInteger>(begin) * in_ch * sizeof(float);
            const NSUInteger output_offset = out.byte_offset + static_cast<NSUInteger>(begin * stride + shift) * out_ch * sizeof(float);
            MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:count columns:in_ch
                rowBytes:static_cast<NSUInteger>(in_ch) * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:out_ch columns:in_ch
                rowBytes:static_cast<NSUInteger>(in_ch) * (half_weights ? sizeof(uint16_t) : sizeof(float))
                dataType:(half_weights ? MPSDataTypeFloat16 : MPSDataTypeFloat32)];
            MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:count columns:out_ch
                rowBytes:static_cast<NSUInteger>(stride) * out_ch * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrix* A = [[MPSMatrix alloc] initWithBuffer:impl_->pass_workspace offset:input_offset descriptor:da];
            MPSMatrix* B = [[MPSMatrix alloc] initWithBuffer:tap offset:0 descriptor:db];
            MPSMatrix* C = [[MPSMatrix alloc] initWithBuffer:impl_->pass_workspace offset:output_offset descriptor:dc];
            [impl_->mps_gemm(count, out_ch, in_ch, true) encodeToCommandBuffer:impl_->pass_cb
                leftMatrix:A rightMatrix:B resultMatrix:C];
        }
        return out;
    }
    [impl_->pass_enc setComputePipelineState:impl_->conv_transpose1d_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:wbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:bbuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
    [impl_->pass_enc setBytes:&out_tokens length:sizeof(out_tokens) atIndex:5];
    [impl_->pass_enc setBytes:&in_ch length:sizeof(in_ch) atIndex:6];
    [impl_->pass_enc setBytes:&out_ch length:sizeof(out_ch) atIndex:7];
    [impl_->pass_enc setBytes:&kernel length:sizeof(kernel) atIndex:8];
    [impl_->pass_enc setBytes:&stride length:sizeof(stride) atIndex:9];
    [impl_->pass_enc setBytes:&padding length:sizeof(padding) atIndex:10];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(out_ch, out_tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(out_ch, impl_->conv_transpose1d_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

PassSlot MetalContext::bigvgan_activation_f32_pass(const std::string& upk, const std::vector<float>& up, const std::string& downk, const std::vector<float>& down, const std::string& ak, const std::vector<float>& alpha, const std::string& bek, const std::vector<float>& beta, PassSlot x, uint32_t tokens, uint32_t channels) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> upbuf = impl_->resident_buffer_with_bytes(upk, up.data(), up.size() * sizeof(float));
    id<MTLBuffer> downbuf = impl_->resident_buffer_with_bytes(downk, down.data(), down.size() * sizeof(float));
    id<MTLBuffer> abuf = impl_->resident_buffer_with_bytes(ak, alpha.data(), alpha.size() * sizeof(float));
    id<MTLBuffer> bebuf = impl_->resident_buffer_with_bytes(bek, beta.data(), beta.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(tokens * channels);
    const NSUInteger intermediate_bytes = static_cast<NSUInteger>(tokens) * channels * 2 * sizeof(float);
    if (!impl_->bigvgan_activation_workspace || impl_->bigvgan_activation_capacity < intermediate_bytes) {
        impl_->bigvgan_activation_workspace = new_counted_buffer_with_length(
            impl_->device, impl_->buffer_allocations, impl_->buffer_bytes_allocated, intermediate_bytes);
        if (!impl_->bigvgan_activation_workspace) throw std::runtime_error("BigVGAN activation allocation failed");
        impl_->bigvgan_activation_capacity = intermediate_bytes;
    }
    [impl_->pass_enc setComputePipelineState:impl_->bigvgan_upsample_snake_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:upbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:abuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:bebuf offset:0 atIndex:3];
    [impl_->pass_enc setBuffer:impl_->bigvgan_activation_workspace offset:0 atIndex:4];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:5];
    [impl_->pass_enc setBytes:&channels length:sizeof(channels) atIndex:6];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(channels, tokens * 2, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(channels, impl_->bigvgan_upsample_snake_pipeline.threadExecutionWidth), 1, 1)];
    [impl_->pass_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [impl_->pass_enc setComputePipelineState:impl_->bigvgan_downsample_pipeline];
    [impl_->pass_enc setBuffer:impl_->bigvgan_activation_workspace offset:0 atIndex:0];
    [impl_->pass_enc setBuffer:downbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:2];
    [impl_->pass_enc setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [impl_->pass_enc setBytes:&channels length:sizeof(channels) atIndex:4];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(channels, tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(channels, impl_->bigvgan_downsample_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

PassSlot MetalContext::avg3_f32_pass(PassSlot a, PassSlot b, PassSlot c) {
    PASS_REQUIRE_AND_BARRIER();
    uint32_t count = a.element_count;
    auto out = impl_->pass_alloc_raw(count);
    [impl_->pass_enc setComputePipelineState:impl_->avg3_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:a.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:b.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:c.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:4];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(impl_->avg3_pipeline.threadExecutionWidth, 1, 1)];
    return out;
}

PassSlot MetalContext::clamp_f32_pass(PassSlot x, float min_value, float max_value) {
    PASS_REQUIRE_AND_BARRIER();
    uint32_t count = x.element_count;
    auto out = impl_->pass_alloc_raw(count);
    [impl_->pass_enc setComputePipelineState:impl_->clamp_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:x.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
    [impl_->pass_enc setBytes:&count length:sizeof(count) atIndex:2];
    [impl_->pass_enc setBytes:&min_value length:sizeof(min_value) atIndex:3];
    [impl_->pass_enc setBytes:&max_value length:sizeof(max_value) atIndex:4];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(impl_->clamp_pipeline.threadExecutionWidth, 1, 1)];
    return out;
}

PassSlot MetalContext::gpt_argmax_pass(PassSlot logits, uint32_t vocab) {
    PASS_REQUIRE_AND_BARRIER();
    auto out = impl_->pass_alloc_raw(1);
    [impl_->pass_enc setComputePipelineState:impl_->gpt_argmax_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:logits.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:1];
    [impl_->pass_enc setBytes:&vocab length:sizeof(vocab) atIndex:2];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::gpt_build_current_pass(PassSlot token_slot,
                                              const std::string& mel_emb_key, const std::vector<float>& mel_emb,
                                              const std::string& pos_emb_key, const std::vector<float>& pos_emb,
                                              uint32_t width, uint32_t position) {
    PASS_REQUIRE_AND_BARRIER();
    id<MTLBuffer> mbuf = impl_->resident_buffer_with_bytes(mel_emb_key, mel_emb.data(), mel_emb.size() * sizeof(float));
    id<MTLBuffer> pbuf = impl_->resident_buffer_with_bytes(pos_emb_key, pos_emb.data(), pos_emb.size() * sizeof(float));
    auto out = impl_->pass_alloc_raw(width);
    [impl_->pass_enc setComputePipelineState:impl_->gpt_build_current_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:token_slot.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:mbuf offset:0 atIndex:1];
    [impl_->pass_enc setBuffer:pbuf offset:0 atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&width length:sizeof(width) atIndex:4];
    [impl_->pass_enc setBytes:&position length:sizeof(position) atIndex:5];
    [impl_->pass_enc dispatchThreads:MTLSizeMake(width, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(width, impl_->gpt_build_current_pipeline.threadExecutionWidth), 1, 1)];
    return out;
}

std::vector<uint32_t> MetalContext::passReadU32(PassSlot slot) const {
    if (impl_->pass_mode) throw std::logic_error("passReadU32: cannot read while pass is active");
    if (!slot.valid() || !impl_->pass_workspace) {
        throw std::invalid_argument("passReadU32: invalid slot or no workspace");
    }
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>([impl_->pass_workspace contents]) + slot.byte_offset);
    return std::vector<uint32_t>(ptr, ptr + slot.element_count);
}

PassSlot MetalContext::gpt_cached_attention_resident_pass(uint32_t layer, PassSlot current_qkv,
                                                          uint32_t cache_tokens, uint32_t heads, uint32_t head_dim) {
    PASS_REQUIRE_AND_BARRIER();
    if (!impl_->gpt_kv_buffer || layer >= impl_->gpt_kv_layers ||
        cache_tokens + 1 > impl_->gpt_kv_max_tokens || cache_tokens + 1 > 4096 ||
        heads * head_dim != impl_->gpt_kv_width) {
        throw std::invalid_argument("gpt_cached_attention_resident_pass invalid args");
    }
    const uint32_t width = heads * head_dim;
    auto out = impl_->pass_alloc_raw(width);
    [impl_->pass_enc setComputePipelineState:impl_->gpt_cached_attention_resident_pipeline];
    [impl_->pass_enc setBuffer:impl_->gpt_kv_buffer offset:impl_->gpt_kv_k_offset(layer) atIndex:0];
    [impl_->pass_enc setBuffer:impl_->gpt_kv_buffer offset:impl_->gpt_kv_v_offset(layer) atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:current_qkv.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&cache_tokens length:sizeof(cache_tokens) atIndex:4];
    [impl_->pass_enc setBytes:&heads length:sizeof(heads) atIndex:5];
    [impl_->pass_enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(heads, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

PassSlot MetalContext::gpt_cached_attention_f32_pass(
    const std::vector<float>& cache_k,
    const std::vector<float>& cache_v,
    PassSlot current_qkv,
    uint32_t cache_tokens, uint32_t heads, uint32_t head_dim) {
    if (!impl_->pass_mode) throw std::logic_error("pass op outside pass");
    const uint32_t width = heads * head_dim;
    // Upload KV caches into workspace (CPU→GPU via shared memory, no GPU barrier needed).
    auto k_slot = passUploadAlloc(cache_k);
    auto v_slot = passUploadAlloc(cache_v);
    auto out = impl_->pass_alloc_raw(width);
    // GPU barrier: previous kernel writing current_qkv must complete before attention reads it.
    [impl_->pass_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [impl_->pass_enc setComputePipelineState:impl_->gpt_cached_attention_pipeline];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:k_slot.byte_offset atIndex:0];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:v_slot.byte_offset atIndex:1];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:current_qkv.byte_offset atIndex:2];
    [impl_->pass_enc setBuffer:impl_->pass_workspace offset:out.byte_offset atIndex:3];
    [impl_->pass_enc setBytes:&cache_tokens length:sizeof(cache_tokens) atIndex:4];
    [impl_->pass_enc setBytes:&heads length:sizeof(heads) atIndex:5];
    [impl_->pass_enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
    [impl_->pass_enc dispatchThreadgroups:MTLSizeMake(heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    return out;
}

}  // namespace mit2
