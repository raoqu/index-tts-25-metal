#include <metal_stdlib>
using namespace metal;

static inline float mit2_erf_approx_f32(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - poly * exp(-ax * ax));
}

#include <metal_simdgroup_matrix>

// Tiled GEMM with fp16 weights: C[M,N] = A[M,K] x B[N,K]^T (+bias) (+=C).
// A fp32 (workspace activations), B fp16 row-major [N,K] (resident weights),
// C fp32. Threadgroup computes a 64x64 C tile: 4 simdgroups in a 2x2 grid,
// each accumulating 32x32 via 4x4 simdgroup_float8x8 MMA. A is staged to
// threadgroup memory as half (fp16 activation staging); epilogue is scalar
// over a threadgroup C tile (bias / accumulate handled there).
// flags: bit0 = add bias, bit1 = accumulate onto existing C.
kernel void mit2_gemm_f16w_f32(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& flags [[buffer(7)]],
    constant uint& lda [[buffer(8)]],   // A row stride (may be < K: overlapping rows for conv-as-GEMM)
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // v2: threadgroup computes a 32(M) x 64(N) C tile with 4 simdgroups, each
    // owning a 32x16 N-slice (4x2 of 8x8 fp32 accumulators). Bias is loaded
    // straight into the accumulators (replicated-row tile); interior tiles
    // simdgroup_store directly to C; only M-edge tiles spill through a small
    // scratch. ~6KB threadgroup memory (was 20KB) for real occupancy.
    const uint m0 = tg.y * 32;
    const uint n0 = tg.x * 64;
    const bool add_bias = (flags & 1u) != 0;

    threadgroup half Atile[32 * 16];
    threadgroup half Btile[64 * 16];
    threadgroup float BiasTile[8 * 64];
    threadgroup float Edge[4][8 * 8];

    // Bias tile: 8 identical rows of bias[n0 .. n0+64).
    for (uint idx = tid; idx < 8 * 64; idx += 128) {
        const uint c = idx % 64;
        const uint gn = n0 + c;
        BiasTile[idx] = (add_bias && gn < N) ? bias[gn] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulators initialized with bias.
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 2; ++j) {
            simdgroup_load(acc[i][j], BiasTile + sg * 16 + j * 8, 64);
        }
    }

    for (uint k0 = 0; k0 < K; k0 += 16) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Stage A[32,16] (fp32 -> half): 512 elems / 128 threads = 4 each.
        for (uint idx = tid; idx < 32 * 16; idx += 128) {
            const uint r = idx / 16;
            const uint c = idx % 16;
            const uint gm = m0 + r;
            const uint gk = k0 + c;
            Atile[idx] = (gm < M && gk < K) ? half(A[gm * lda + gk]) : half(0.0f);
        }
        // Stage B[64,16] (half): 1024 elems / 128 threads = 8 each.
        for (uint idx = tid; idx < 64 * 16; idx += 128) {
            const uint r = idx / 16;
            const uint c = idx % 16;
            const uint gn = n0 + r;
            const uint gk = k0 + c;
            Btile[idx] = (gn < N && gk < K) ? B[gn * K + gk] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 16; kk += 8) {
            simdgroup_half8x8 a_frag[4];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a_frag[i], Atile + (i * 8) * 16 + kk, 16);
            }
            simdgroup_half8x8 b_frag[2];
            for (uint j = 0; j < 2; ++j) {
                simdgroup_load(b_frag[j], Btile + (sg * 16 + j * 8) * 16 + kk, 16, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
    }

    // Store: interior 8x8 tiles go straight to C; M-edge tiles via scratch.
    for (uint i = 0; i < 4; ++i) {
        const uint gm = m0 + i * 8;
        for (uint j = 0; j < 2; ++j) {
            const uint gn = n0 + sg * 16 + j * 8;
            if (gn >= N) {
                continue;
            }
            if (gm + 8 <= M && gn + 8 <= N) {
                simdgroup_store(acc[i][j], C + gm * N + gn, N);
            } else {
                simdgroup_store(acc[i][j], Edge[sg], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64; e += 32) {
                    const uint r = e / 8;
                    const uint c = e % 8;
                    if (gm + r < M && gn + c < N) {
                        C[(gm + r) * N + gn + c] = Edge[sg][e];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// SwiGLU over a packed FFN output [tokens, 2*width]: out = silu(a) * b with
// a = x[:, :width], b = x[:, width:]. Pairs with the merged w1|w3 GEMM.
kernel void mit2_silu_mul_split_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= width || token >= tokens) {
        return;
    }
    const uint base = token * width * 2;
    const float a = x[base + col];
    const float b = x[base + width + col];
    const float s = a / (1.0f + exp(-a));
    out[token * width + col] = s * b;
}

kernel void mit2_copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

kernel void mit2_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        out[gid] = a[gid] + b[gid];
    }
}

// Reflect-pad rows: out[b, t, c] = x[b, reflect(t - pad), c] for t in [0, tokens + 2*pad).
// Matches conv1d reflect-same boundary handling.
kernel void mit2_reflect_pad_rows_batched_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& pad [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint t = gid.y;
    const uint b = gid.z;
    const uint padded_tokens = tokens + 2 * pad;
    if (c >= width || t >= padded_tokens || b >= batch) {
        return;
    }
    int src_t = int(t) - int(pad);
    const int last = int(tokens) - 1;
    if (tokens > 1) {
        while (src_t < 0 || src_t > last) {
            if (src_t < 0) {
                src_t = -src_t;
            }
            if (src_t > last) {
                src_t = 2 * last - src_t;
            }
        }
    } else {
        src_t = 0;
    }
    out[(b * padded_tokens + t) * width + c] = x[(b * tokens + uint(src_t)) * width + c];
}

// Overlapping-row expansion (im2col for k-tap conv): out[t] = in[t*row_stride ..
// t*row_stride + row_len), i.e. row t gathers `kernel` consecutive padded rows.
kernel void mit2_overlap_rows_f32(
    device const float* in_data [[buffer(0)]],
    device float* out_data [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& row_stride [[buffer(3)]],
    constant uint& row_len [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint t = gid.y;
    if (i >= row_len || t >= tokens) {
        return;
    }
    out_data[t * row_len + i] = in_data[t * row_stride + i];
}

// Fill each row of out[tokens, rows] with bias[rows] (pre-pass for beta=1 MPS GEMM).
kernel void mit2_broadcast_bias_rows_f32(
    device const float* bias [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint row = gid.x;
    const uint token = gid.y;
    if (row < rows && token < tokens) {
        out[token * rows + row] = bias[row];
    }
}

kernel void mit2_add_scaled_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        out[gid] = a[gid] + scale * b[gid];
    }
}

kernel void mit2_avg3_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        out[gid] = (a[gid] + b[gid] + c[gid]) * (1.0f / 3.0f);
    }
}

kernel void mit2_w2v_bert_normalize_f32(
    device const float* hidden [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* std [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        const uint dim = gid % 1024;
        out[gid] = (hidden[gid] - mean[dim]) / std[dim];
    }
}

kernel void mit2_embedding_f32(
    device const float* table [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint token = gid.y;
    const uint col = gid.x;
    if (col < width) {
        const uint id = ids[token];
        out[token * width + col] = table[id * width + col];
    }
}

kernel void mit2_semantic_quantize_f32(
    device const float* in_weight [[buffer(0)]],
    device const float* in_bias [[buffer(1)]],
    device const float* codebook [[buffer(2)]],
    device const float* out_weight [[buffer(3)]],
    device const float* out_bias [[buffer(4)]],
    device const float* spk_cond [[buffer(5)]],
    device float* sref [[buffer(6)]],
    device uint* codes [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    uint token [[thread_position_in_grid]]
) {
    if (token >= tokens) {
        return;
    }

    float latent[8];
    const uint spk_base = token * 1024;
    for (uint out = 0; out < 8; ++out) {
        float acc = in_bias[out];
        const uint w_base = out * 1024;
        for (uint in = 0; in < 1024; ++in) {
            acc += in_weight[w_base + in] * spk_cond[spk_base + in];
        }
        latent[out] = acc;
    }

    uint best_code = 0;
    float best_dist = INFINITY;
    for (uint code = 0; code < 8192; ++code) {
        const uint c_base = code * 8;
        float dist = 0.0f;
        for (uint dim = 0; dim < 8; ++dim) {
            const float delta = latent[dim] - codebook[c_base + dim];
            dist += delta * delta;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_code = code;
        }
    }
    codes[token] = best_code;

    const uint best_base = best_code * 8;
    const uint sref_base = token * 1024;
    for (uint row = 0; row < 1024; ++row) {
        float acc = out_bias[row];
        const uint w_base = row * 8;
        for (uint dim = 0; dim < 8; ++dim) {
            acc += out_weight[w_base + dim] * codebook[best_base + dim];
        }
        sref[sref_base + row] = acc;
    }
}

kernel void mit2_silu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        const float v = x[gid];
        out[gid] = v / (1.0f + exp(-v));
    }
}

kernel void mit2_silu_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        const float v = a[gid];
        out[gid] = (v / (1.0f + exp(-v))) * b[gid];
    }
}

kernel void mit2_mask_rows_f32(
    device const float* x [[buffer(0)]],
    device const uint* mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= width || token >= tokens) {
        return;
    }
    const uint idx = token * width + col;
    out[idx] = mask[token] == 0 ? 0.0f : x[idx];
}

kernel void mit2_glu_split_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= width || token >= tokens) {
        return;
    }
    const uint base = token * width * 2;
    const float a = x[base + col];
    const float b = x[base + width + col];
    out[token * width + col] = a / (1.0f + exp(-b));
}

kernel void mit2_wavenet_gate_f32(
    device const float* in_layer [[buffer(0)]],
    device const float* cond [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& cond_width [[buffer(5)]],
    constant uint& cond_offset [[buffer(6)]],
    constant uint& cond_tokens [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= width || token >= tokens) {
        return;
    }
    const uint in_base = token * width * 2;
    const uint cond_base = (cond_tokens == 1 ? 0 : token * cond_width) + cond_offset;
    const float a = in_layer[in_base + col] + cond[cond_base + col];
    const float b = in_layer[in_base + width + col] + cond[cond_base + width + col];
    const float sigmoid_b = 1.0f / (1.0f + exp(-b));
    const float tanh_a = a > 20.0f ? 1.0f : (a < -20.0f ? -1.0f : tanh(a));
    out[token * width + col] = tanh_a * sigmoid_b;
}

kernel void mit2_wavenet_res_skip_update_f32(
    device const float* x [[buffer(0)]],
    device const float* output [[buffer(1)]],
    device const float* res_skip [[buffer(2)]],
    device const uint* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& has_residual [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= width || token >= tokens) {
        return;
    }
    const uint idx = token * width + col;
    const float m = mask[token] == 0 ? 0.0f : 1.0f;
    const uint out_x_base = 0;
    const uint out_output_base = tokens * width;
    if (has_residual != 0) {
        const uint rs_base = token * width * 2;
        out[out_x_base + idx] = (x[idx] + res_skip[rs_base + col]) * m;
        out[out_output_base + idx] = output[idx] + res_skip[rs_base + width + col];
    } else {
        out[out_x_base + idx] = x[idx];
        out[out_output_base + idx] = (output[idx] + res_skip[idx]) * m;
    }
}

kernel void mit2_geglu_erf_split_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= width || token >= tokens) {
        return;
    }
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    const uint base = token * width * 2;
    const float a = x[base + col];
    const float b = x[base + width + col];
    const float gelu = 0.5f * b * (1.0f + mit2_erf_approx_f32(b * inv_sqrt2));
    out[token * width + col] = a * gelu;
}

kernel void mit2_gelu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        const float v = x[gid];
        const float c = 0.7978845608028654f;
        if (v > 10.0f) {
            out[gid] = v;
        } else if (v < -10.0f) {
            out[gid] = 0.0f;
        } else {
            out[gid] = 0.5f * v * (1.0f + tanh(c * (v + 0.044715f * v * v * v)));
        }
    }
}

kernel void mit2_tanh_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        out[gid] = tanh(x[gid]);
    }
}

kernel void mit2_clamp_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& min_value [[buffer(3)]],
    constant float& max_value [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        out[gid] = min(max(x[gid], min_value), max_value);
    }
}

kernel void mit2_softmax_f32_one_row(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float scratch[1024];
    float local_max = -INFINITY;
    for (uint i = tid; i < count; i += 1024) {
        local_max = max(local_max, x[i]);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint i = tid; i < count; i += 1024) {
        local_sum += exp(x[i] - row_max);
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = scratch[0];

    for (uint i = tid; i < count; i += 1024) {
        out[i] = exp(x[i] - row_max) / denom;
    }
}

kernel void mit2_layernorm_f32_one_row(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float scratch[1024];
    float local_sum = 0.0f;
    for (uint i = tid; i < count; i += 1024) {
        local_sum += x[i];
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = scratch[0] / float(count);
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_var = 0.0f;
    for (uint i = tid; i < count; i += 1024) {
        const float d = x[i] - mean;
        local_var += d * d;
    }
    scratch[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = rsqrt(scratch[0] / float(count) + eps);
    for (uint i = tid; i < count; i += 1024) {
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

kernel void mit2_layernorm_f32_rows(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint token = group.x;
    if (token >= tokens) {
        return;
    }
    threadgroup float scratch[1024];
    const uint base = token * width;
    float local_sum = 0.0f;
    for (uint i = tid; i < width; i += 1024) {
        local_sum += x[base + i];
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = scratch[0] / float(width);
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_var = 0.0f;
    for (uint i = tid; i < width; i += 1024) {
        const float d = x[base + i] - mean;
        local_var += d * d;
    }
    scratch[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = rsqrt(scratch[0] / float(width) + eps);
    for (uint i = tid; i < width; i += 1024) {
        out[base + i] = (x[base + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

kernel void mit2_layernorm_f32_rows_serial(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint token [[thread_position_in_grid]]
) {
    if (token >= tokens) {
        return;
    }
    const uint base = token * width;
    float mean = 0.0f;
    for (uint i = 0; i < width; ++i) {
        mean += x[base + i];
    }
    mean /= float(width);

    float var = 0.0f;
    for (uint i = 0; i < width; ++i) {
        const float d = x[base + i] - mean;
        var += d * d;
    }
    const float inv_std = rsqrt(var / float(width) + eps);
    for (uint i = 0; i < width; ++i) {
        out[base + i] = (x[base + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

kernel void mit2_adaptive_layernorm_f32_rows(
    device const float* x [[buffer(0)]],
    device const float* shift [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint token = group.x;
    if (token >= tokens) {
        return;
    }
    threadgroup float scratch[1024];
    const uint base = token * width;
    float local_sum = 0.0f;
    for (uint i = tid; i < width; i += 1024) {
        local_sum += x[base + i];
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = scratch[0] / float(width);
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_var = 0.0f;
    for (uint i = tid; i < width; i += 1024) {
        const float d = x[base + i] - mean;
        local_var += d * d;
    }
    scratch[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = rsqrt(scratch[0] / float(width) + eps);
    for (uint i = tid; i < width; i += 1024) {
        out[base + i] = (x[base + i] - mean) * inv_std * (1.0f + scale[i]) + shift[i];
    }
}

kernel void mit2_adaptive_rmsnorm_f32_rows(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* adaptive_weight [[buffer(2)]],
    device const float* adaptive_bias [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint token = group.x;
    if (token >= tokens) {
        return;
    }
    threadgroup float scratch[1024];
    const uint base = token * width;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < width; i += 1024) {
        const float v = x[base + i];
        local_sum_sq += v * v;
    }
    scratch[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(scratch[0] / float(width) + eps);
    for (uint i = tid; i < width; i += 1024) {
        const float normed = x[base + i] * inv_rms * gamma[i];
        out[base + i] = adaptive_weight[i] * normed + adaptive_bias[i];
    }
}

kernel void mit2_cfm_euler_update_f32(
    device const float* x [[buffer(0)]],
    device const float* dphi [[buffer(1)]],
    device const float* cfg_dphi [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& prompt_tokens [[buffer(6)]],
    constant float& dt [[buffer(7)]],
    constant float& cfg_rate [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    if (col >= width || token >= tokens) {
        return;
    }
    const uint idx = token * width + col;
    if (token < prompt_tokens) {
        out[idx] = 0.0f;
        return;
    }
    const float guided = (1.0f + cfg_rate) * dphi[idx] - cfg_rate * cfg_dphi[idx];
    out[idx] = x[idx] + dt * guided;
}

kernel void mit2_concat_rows_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& a_width [[buffer(4)]],
    constant uint& b_width [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    const uint out_width = a_width + b_width;
    if (token >= tokens || col >= out_width) {
        return;
    }
    if (col < a_width) {
        out[token * out_width + col] = a[token * a_width + col];
    } else {
        const uint b_col = col - a_width;
        out[token * out_width + col] = b[token * b_width + b_col];
    }
}

kernel void mit2_hot_condition_merge_f32(
    device const float* prompt [[buffer(0)]],
    device const float* generated [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& prompt_tokens [[buffer(3)]],
    constant uint& generated_tokens [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    const uint tokens = prompt_tokens + generated_tokens;
    if (token >= tokens || col >= width) {
        return;
    }
    if (token < prompt_tokens) {
        out[token * width + col] = prompt[token * width + col];
    } else {
        const uint generated_token = token - prompt_tokens;
        out[token * width + col] = generated[generated_token * width + col];
    }
}

kernel void mit2_dit_input_merge_f32(
    device const float* x [[buffer(0)]],
    device const float* prompt_x [[buffer(1)]],
    device const float* cond_proj [[buffer(2)]],
    device const float* style [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint token = gid.y;
    constexpr uint mel_width = 80;
    constexpr uint cond_width = 512;
    constexpr uint style_width = 192;
    constexpr uint out_width = mel_width + mel_width + cond_width + style_width;
    if (token >= tokens || col >= out_width) {
        return;
    }
    if (col < mel_width) {
        out[token * out_width + col] = x[token * mel_width + col];
    } else if (col < mel_width * 2) {
        const uint prompt_col = col - mel_width;
        out[token * out_width + col] = prompt_x[token * mel_width + prompt_col];
    } else if (col < mel_width * 2 + cond_width) {
        const uint cond_col = col - mel_width * 2;
        out[token * out_width + col] = cond_proj[token * cond_width + cond_col];
    } else {
        const uint style_col = col - mel_width * 2 - cond_width;
        out[token * out_width + col] = style[style_col];
    }
}

kernel void mit2_dit_input_merge_batched_f32(
    device const float* x [[buffer(0)]],
    device const float* prompt_x [[buffer(1)]],
    device const float* cond_proj [[buffer(2)]],
    device const float* style [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint row = gid.y;
    constexpr uint mel_width = 80;
    constexpr uint cond_width = 512;
    constexpr uint style_width = 192;
    constexpr uint out_width = mel_width + mel_width + cond_width + style_width;
    const uint rows = batch * tokens;
    if (row >= rows || col >= out_width) {
        return;
    }
    if (col < mel_width) {
        out[row * out_width + col] = x[row * mel_width + col];
    } else if (col < mel_width * 2) {
        const uint prompt_col = col - mel_width;
        out[row * out_width + col] = prompt_x[row * mel_width + prompt_col];
    } else if (col < mel_width * 2 + cond_width) {
        const uint cond_col = col - mel_width * 2;
        out[row * out_width + col] = cond_proj[row * cond_width + cond_col];
    } else {
        const uint branch = row / tokens;
        const uint style_col = col - mel_width * 2 - cond_width;
        out[row * out_width + col] = style[branch * style_width + style_col];
    }
}

kernel void mit2_rmsnorm_f32_one_row(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float scratch[1024];
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < count; i += 1024) {
        local_sum_sq += x[i] * x[i];
    }
    scratch[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(scratch[0] / float(count) + eps);
    for (uint i = tid; i < count; i += 1024) {
        out[i] = x[i] * inv_rms * gamma[i];
    }
}

kernel void mit2_rmsnorm_f32_rows(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint token = group.x;
    if (token >= tokens) {
        return;
    }
    threadgroup float scratch[1024];
    const uint base = token * width;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < width; i += 1024) {
        const float v = x[base + i];
        local_sum_sq += v * v;
    }
    scratch[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_norm = sqrt(float(width)) * rsqrt(scratch[0] + 1.0e-12f);
    for (uint i = tid; i < width; i += 1024) {
        out[base + i] = x[base + i] * inv_norm * gamma[i];
    }
}

kernel void mit2_rmsnorm_f32_rows_eps(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint token = group.x;
    if (token >= tokens) {
        return;
    }
    threadgroup float scratch[1024];
    const uint base = token * width;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < width; i += 1024) {
        const float v = x[base + i];
        local_sum_sq += v * v;
    }
    scratch[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms = rsqrt(scratch[0] / float(width) + eps);
    for (uint i = tid; i < width; i += 1024) {
        out[base + i] = x[base + i] * inv_rms * gamma[i];
    }
}

kernel void mit2_linear_f32_rowmajor(
    device const float* weight [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float acc = bias[row];
    const uint base = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc += weight[base + col] * x[col];
    }
    out[row] = acc;
}

// GEMV with one simdgroup per output row: 32 lanes stride the columns and
// simd_sum-reduce. ~32x more parallelism than one-thread-per-row, and
// coalesced weight reads — GPT decode is bandwidth-bound on these.
kernel void mit2_linear_gemv_f32(
    device const float* weight [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint sg_in_grid [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    const uint sgs_per_tg = tg_size.x / 32;
    const uint row = group.x * sgs_per_tg + sg_in_grid;
    if (row >= rows) {
        return;
    }
    const uint base = row * cols;
    float acc = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        acc += weight[base + col] * x[col];
    }
    const float total = simd_sum(acc);
    if (lane == 0) {
        out[row] = total + bias[row];
    }
}

// Half-precision-weight GEMV: fp16 weights (half the bandwidth), fp32
// activations and accumulation. GPT decode is bandwidth-bound on these reads.
kernel void mit2_linear_gemv_f16w_f32(
    device const half* weight [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint sg_in_grid [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    const uint sgs_per_tg = tg_size.x / 32;
    const uint row = group.x * sgs_per_tg + sg_in_grid;
    if (row >= rows) {
        return;
    }
    const uint base = row * cols;
    float acc = 0.0f;
    if ((cols & 3u) == 0u) {
        // Vectorized: each lane reads half4/float4 (4x fewer memory transactions).
        const device half4* w4 = reinterpret_cast<const device half4*>(weight + base);
        const device float4* x4 = reinterpret_cast<const device float4*>(x);
        const uint cols4 = cols / 4;
        for (uint c = lane; c < cols4; c += 32) {
            acc += dot(float4(w4[c]), x4[c]);
        }
    } else {
        for (uint col = lane; col < cols; col += 32) {
            acc += float(weight[base + col]) * x[col];
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0) {
        out[row] = total + bias[row];
    }
}

// Fused single-row GPT op: optional LayerNorm(x) -> GEMV(half weights) ->
// optional GELU -> optional residual add. Collapses ln+linear+gelu+add chains
// (10 dispatches/layer -> 5) for the bandwidth-bound decode loop.
// flags: bit0 = apply layernorm to x first, bit1 = gelu on output, bit2 = add residual.
kernel void mit2_gpt_fused_gemv_f16w_f32(
    device const half* weight [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device const float* ln_gamma [[buffer(3)]],
    device const float* ln_beta [[buffer(4)]],
    device const float* residual [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& flags [[buffer(10)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    const uint tid = tid3.x;
    const bool has_ln = (flags & 1u) != 0;
    const bool fuse_gelu = (flags & 2u) != 0;
    const bool has_residual = (flags & 4u) != 0;
    threadgroup float xnorm[1280];
    threadgroup float red[8];
    threadgroup float bcast2[2];

    if (has_ln) {
        // Cooperative LayerNorm of x[cols] into threadgroup memory (cols <= 1280).
        float lsum = 0.0f;
        for (uint i = tid; i < cols; i += tg_size.x) {
            lsum += x[i];
        }
        float ssum = simd_sum(lsum);
        if (lane == 0) {
            red[sg] = ssum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint i = 0; i < tg_size.x / 32; ++i) {
                total += red[i];
            }
            bcast2[0] = total / float(cols);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float mean = bcast2[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lvar = 0.0f;
        for (uint i = tid; i < cols; i += tg_size.x) {
            const float d = x[i] - mean;
            lvar += d * d;
        }
        float svar = simd_sum(lvar);
        if (lane == 0) {
            red[sg] = svar;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint i = 0; i < tg_size.x / 32; ++i) {
                total += red[i];
            }
            bcast2[1] = rsqrt(total / float(cols) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_std = bcast2[1];
        for (uint i = tid; i < cols; i += tg_size.x) {
            xnorm[i] = (x[i] - mean) * inv_std * ln_gamma[i] + ln_beta[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint sgs_per_tg = tg_size.x / 32;
    const uint row = group.x * sgs_per_tg + sg;
    if (row >= rows) {
        return;
    }
    const uint base = row * cols;
    float acc = 0.0f;
    if ((cols & 3u) == 0u) {
        const device half4* w4 = reinterpret_cast<const device half4*>(weight + base);
        const uint cols4 = cols / 4;
        if (has_ln) {
            const threadgroup float4* xn4 = reinterpret_cast<const threadgroup float4*>(xnorm);
            for (uint c = lane; c < cols4; c += 32) {
                acc += dot(float4(w4[c]), xn4[c]);
            }
        } else {
            const device float4* x4 = reinterpret_cast<const device float4*>(x);
            for (uint c = lane; c < cols4; c += 32) {
                acc += dot(float4(w4[c]), x4[c]);
            }
        }
    } else if (has_ln) {
        for (uint col = lane; col < cols; col += 32) {
            acc += float(weight[base + col]) * xnorm[col];
        }
    } else {
        for (uint col = lane; col < cols; col += 32) {
            acc += float(weight[base + col]) * x[col];
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0) {
        float v = total + bias[row];
        if (fuse_gelu) {
            const float c = 0.7978845608028654f;
            if (v > 10.0f) {
                // identity
            } else if (v < -10.0f) {
                v = 0.0f;
            } else {
                v = 0.5f * v * (1.0f + tanh(c * (v + 0.044715f * v * v * v)));
            }
        }
        if (has_residual) {
            v += residual[row];
        }
        out[row] = v;
    }
}

// Store the K/V sections of a full-sequence QKV tensor [tokens, 3*width] into
// the resident GPT KV cache (prefill writes the cache on-GPU, no CPU upload).
kernel void mit2_gpt_kv_store_f32(
    device const float* qkv [[buffer(0)]],
    device float* cache_k [[buffer(1)]],
    device float* cache_v [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (d >= width || t >= tokens) {
        return;
    }
    const uint qkv_width = width * 3;
    cache_k[t * width + d] = qkv[t * qkv_width + width + d];
    cache_v[t * width + d] = qkv[t * qkv_width + width * 2 + d];
}

// Argmax over a logits row -> token id. Single threadgroup.
kernel void mit2_gpt_argmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* token_out [[buffer(1)]],
    constant uint& vocab [[buffer(2)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    const uint tid = tid3.x;
    threadgroup float best_val[1024];
    threadgroup uint best_idx[1024];
    float bv = -INFINITY;
    uint bi = 0;
    for (uint i = tid; i < vocab; i += tg_size.x) {
        const float v = logits[i];
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    best_val[tid] = bv;
    best_idx[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float ov = best_val[tid + stride];
            const uint oi = best_idx[tid + stride];
            if (ov > best_val[tid] || (ov == best_val[tid] && oi < best_idx[tid])) {
                best_val[tid] = ov;
                best_idx[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        token_out[0] = best_idx[0];
    }
}

// Build the next decode-step input embedding on-GPU:
// out[i] = mel_embedding[token][i] + mel_pos_embedding[position][i],
// where token is read from a GPU slot written by the argmax kernel.
kernel void mit2_gpt_build_current_f32(
    device const uint* token_slot [[buffer(0)]],
    device const float* mel_embedding [[buffer(1)]],
    device const float* mel_pos_embedding [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& position [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= width) {
        return;
    }
    const uint token = token_slot[0];
    out[gid] = mel_embedding[token * width + gid] + mel_pos_embedding[position * width + gid];
}

// ICB decode state: state = {kv_tokens, position, step}. Advance after each
// token; position follows the HF generate rule (step==0 -> 0, else step+1).
kernel void mit2_gpt_icb_advance_f32(
    device uint* state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        state[0] += 1;                 // kv_tokens
        state[2] += 1;                 // step
        const uint step = state[2];
        state[1] = (step == 0) ? 0 : step + 1;  // position for the NEXT token
    }
}

// Record the current token id into history[state.step] (read back per chunk).
kernel void mit2_gpt_record_token_f32(
    device const uint* token_slot [[buffer(0)]],
    device uint* history [[buffer(1)]],
    device const uint* state [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        history[state[2]] = token_slot[0];
    }
}

kernel void mit2_linear_rows_f32_rowmajor(
    device const float* weight [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint row = gid.x;
    const uint token = gid.y;
    if (row >= rows || token >= tokens) {
        return;
    }
    float acc = bias[row];
    const uint weight_base = row * cols;
    const uint x_base = token * cols;
    for (uint col = 0; col < cols; ++col) {
        acc += weight[weight_base + col] * x[x_base + col];
    }
    out[token * rows + row] = acc;
}

kernel void mit2_nearest_interpolate_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& in_tokens [[buffer(2)]],
    constant uint& out_tokens [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint out_t = gid.y;
    if (col >= width || out_t >= out_tokens) {
        return;
    }
    uint in_t = uint(floor(float(out_t) * scale));
    if (in_t >= in_tokens) {
        in_t = in_tokens - 1;
    }
    out[out_t * width + col] = x[in_t * width + col];
}

kernel void mit2_conv1d_same_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& in_channels [[buffer(5)]],
    constant uint& out_channels [[buffer(6)]],
    constant uint& kernel_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint out_c = gid.x;
    const uint t = gid.y;
    if (out_c >= out_channels || t >= tokens) {
        return;
    }
    const int pad = int(kernel_size / 2);
    float acc = bias[out_c];
    for (uint in_c = 0; in_c < in_channels; ++in_c) {
        for (uint k = 0; k < kernel_size; ++k) {
            const int src_t = int(t) + int(k) - pad;
            if (src_t >= 0 && src_t < int(tokens)) {
                const uint x_idx = uint(src_t) * in_channels + in_c;
                const uint w_idx = (out_c * in_channels + in_c) * kernel_size + k;
                acc += x[x_idx] * weight[w_idx];
            }
        }
    }
    out[t * out_channels + out_c] = acc;
}

kernel void mit2_conv1d_reflect_same_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& in_channels [[buffer(5)]],
    constant uint& out_channels [[buffer(6)]],
    constant uint& kernel_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint out_c = gid.x;
    const uint t = gid.y;
    if (out_c >= out_channels || t >= tokens) {
        return;
    }
    const int pad = int(kernel_size / 2);
    const int last = int(tokens) - 1;
    float acc = bias[out_c];
    for (uint in_c = 0; in_c < in_channels; ++in_c) {
        for (uint k = 0; k < kernel_size; ++k) {
            int src_t = int(t) + int(k) - pad;
            if (tokens > 1) {
                while (src_t < 0 || src_t > last) {
                    if (src_t < 0) {
                        src_t = -src_t;
                    }
                    if (src_t > last) {
                        src_t = 2 * last - src_t;
                    }
                }
            } else {
                src_t = 0;
            }
            const uint x_idx = uint(src_t) * in_channels + in_c;
            const uint w_idx = (out_c * in_channels + in_c) * kernel_size + k;
            acc += x[x_idx] * weight[w_idx];
        }
    }
    out[t * out_channels + out_c] = acc;
}

kernel void mit2_conv1d_reflect_same_batched_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& in_channels [[buffer(6)]],
    constant uint& out_channels [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint out_c = gid.x;
    const uint t = gid.y;
    const uint b = gid.z;
    if (out_c >= out_channels || t >= tokens || b >= batch) {
        return;
    }
    const int pad = int(kernel_size / 2);
    const int last = int(tokens) - 1;
    float acc = bias[out_c];
    const uint x_batch_base = b * tokens * in_channels;
    const uint out_batch_base = b * tokens * out_channels;
    for (uint in_c = 0; in_c < in_channels; ++in_c) {
        for (uint k = 0; k < kernel_size; ++k) {
            int src_t = int(t) + int(k) - pad;
            if (tokens > 1) {
                while (src_t < 0 || src_t > last) {
                    if (src_t < 0) {
                        src_t = -src_t;
                    }
                    if (src_t > last) {
                        src_t = 2 * last - src_t;
                    }
                }
            } else {
                src_t = 0;
            }
            const uint x_idx = x_batch_base + uint(src_t) * in_channels + in_c;
            const uint w_idx = (out_c * in_channels + in_c) * kernel_size + k;
            acc += x[x_idx] * weight[w_idx];
        }
    }
    out[out_batch_base + t * out_channels + out_c] = acc;
}

kernel void mit2_depthwise_conv1d_same_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& kernel_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= channels || t >= tokens) {
        return;
    }
    const int pad = int(kernel_size / 2);
    float acc = bias[c];
    for (uint k = 0; k < kernel_size; ++k) {
        const int src_t = int(t) + int(k) - pad;
        if (src_t >= 0 && src_t < int(tokens)) {
            acc += x[uint(src_t) * channels + c] * weight[c * kernel_size + k];
        }
    }
    out[t * channels + c] = acc;
}

kernel void mit2_depthwise_conv1d_causal_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& kernel_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= channels || t >= tokens) {
        return;
    }
    const int left_pad = int(kernel_size) - 1;
    float acc = bias[c];
    for (uint k = 0; k < kernel_size; ++k) {
        const int src_t = int(t) + int(k) - left_pad;
        if (src_t >= 0 && src_t < int(tokens)) {
            acc += x[uint(src_t) * channels + c] * weight[c * kernel_size + k];
        }
    }
    out[t * channels + c] = acc;
}

kernel void mit2_subsampling_conv2d_relu_flat_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& input_tokens [[buffer(4)]],
    constant uint& input_dim [[buffer(5)]],
    constant uint& output_tokens [[buffer(6)]],
    constant uint& out_channels [[buffer(7)]],
    constant uint& output_freq [[buffer(8)]],
    constant uint& kernel_size [[buffer(9)]],
    constant uint& stride [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint out_t = gid.y;
    const uint flat_dim = out_channels * output_freq;
    if (out_t >= output_tokens || col >= flat_dim) {
        return;
    }
    const uint out_c = col / output_freq;
    const uint out_f = col - out_c * output_freq;
    const uint src_t0 = out_t * stride;
    const uint src_f0 = out_f * stride;
    float acc = bias[out_c];
    for (uint kt = 0; kt < kernel_size; ++kt) {
        for (uint kf = 0; kf < kernel_size; ++kf) {
            const uint src_t = src_t0 + kt;
            const uint src_f = src_f0 + kf;
            if (src_t < input_tokens && src_f < input_dim) {
                const uint x_idx = src_t * input_dim + src_f;
                const uint w_idx = (out_c * kernel_size + kt) * kernel_size + kf;
                acc += x[x_idx] * weight[w_idx];
            }
        }
    }
    out[out_t * flat_dim + col] = max(acc, 0.0f);
}

kernel void mit2_conv1d_dilated_same_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& in_channels [[buffer(5)]],
    constant uint& out_channels [[buffer(6)]],
    constant uint& kernel_size [[buffer(7)]],
    constant uint& dilation [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint out_c = gid.x;
    const uint t = gid.y;
    if (out_c >= out_channels || t >= tokens) {
        return;
    }
    const int pad = int((kernel_size - 1) * dilation / 2);
    float acc = bias[out_c];
    for (uint in_c = 0; in_c < in_channels; ++in_c) {
        for (uint k = 0; k < kernel_size; ++k) {
            const int src_t = int(t) + int(k * dilation) - pad;
            if (src_t >= 0 && src_t < int(tokens)) {
                const uint x_idx = uint(src_t) * in_channels + in_c;
                const uint w_idx = (out_c * in_channels + in_c) * kernel_size + k;
                acc += x[x_idx] * weight[w_idx];
            }
        }
    }
    out[t * out_channels + out_c] = acc;
}

kernel void mit2_conv_transpose1d_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& out_tokens [[buffer(5)]],
    constant uint& in_channels [[buffer(6)]],
    constant uint& out_channels [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    constant uint& padding [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint out_c = gid.x;
    const uint out_t = gid.y;
    if (out_c >= out_channels || out_t >= out_tokens) {
        return;
    }
    float acc = bias[out_c];
    for (uint in_c = 0; in_c < in_channels; ++in_c) {
        for (uint k = 0; k < kernel_size; ++k) {
            const int numerator = int(out_t) + int(padding) - int(k);
            if (numerator >= 0 && (uint(numerator) % stride) == 0) {
                const uint in_t = uint(numerator) / stride;
                if (in_t < tokens) {
                    const uint x_idx = in_t * in_channels + in_c;
                    const uint w_idx = (in_c * out_channels + out_c) * kernel_size + k;
                    acc += x[x_idx] * weight[w_idx];
                }
            }
        }
    }
    out[out_t * out_channels + out_c] = acc;
}

static inline float mit2_clamped_channel_sample(
    device const float* x,
    int t,
    uint c,
    uint tokens,
    uint channels
) {
    if (t < 0) {
        t = 0;
    } else if (t >= int(tokens)) {
        t = int(tokens) - 1;
    }
    return x[uint(t) * channels + c];
}

static inline float mit2_bigvgan_upsample_value(
    device const float* x,
    device const float* up_filter,
    uint up_t,
    uint c,
    uint tokens,
    uint channels
) {
    constexpr uint ratio = 2;
    constexpr uint kernel_size = 12;
    constexpr uint input_pad = 5;
    constexpr uint crop_left = 15;
    const uint full_t = up_t + crop_left;
    float acc = 0.0f;
    for (uint k = 0; k < kernel_size; ++k) {
        if (full_t >= k && ((full_t - k) % ratio) == 0) {
            const uint padded_t = (full_t - k) / ratio;
            const int src_t = int(padded_t) - int(input_pad);
            acc += mit2_clamped_channel_sample(x, src_t, c, tokens, channels) * up_filter[k];
        }
    }
    return float(ratio) * acc;
}

kernel void mit2_bigvgan_activation_f32(
    device const float* x [[buffer(0)]],
    device const float* up_filter [[buffer(1)]],
    device const float* down_filter [[buffer(2)]],
    device const float* alpha_log [[buffer(3)]],
    device const float* beta_log [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= channels || t >= tokens) {
        return;
    }
    constexpr int down_pad_left = 5;
    const uint up_tokens = tokens * 2;
    const float alpha = exp(alpha_log[c]);
    const float beta = exp(beta_log[c]);
    float acc = 0.0f;
    for (uint k = 0; k < 12; ++k) {
        int up_t = int(t * 2 + k) - down_pad_left;
        if (up_t < 0) {
            up_t = 0;
        } else if (up_t >= int(up_tokens)) {
            up_t = int(up_tokens) - 1;
        }
        const float y = mit2_bigvgan_upsample_value(x, up_filter, uint(up_t), c, tokens, channels);
        const float s = sin(y * alpha);
        const float activated = y + (s * s) / (beta + 1.0e-9f);
        acc += activated * down_filter[k];
    }
    out[t * channels + c] = acc;
}

// Materialize the 2x upsampled Snake activation once. The original fused
// expression evaluated overlapping samples twelve times per output position.
kernel void mit2_bigvgan_upsample_snake_f32(
    device const float* x [[buffer(0)]],
    device const float* up_filter [[buffer(1)]],
    device const float* alpha_log [[buffer(2)]],
    device const float* beta_log [[buffer(3)]],
    device float* activated [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& channels [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, t = gid.y;
    if (c >= channels || t >= tokens * 2) return;
    const float y = mit2_bigvgan_upsample_value(x, up_filter, t, c, tokens, channels);
    const float alpha = exp(alpha_log[c]);
    const float beta = exp(beta_log[c]);
    const float s = sin(y * alpha);
    activated[t * channels + c] = y + (s * s) / (beta + 1.0e-9f);
}

kernel void mit2_bigvgan_downsample_f32(
    device const float* activated [[buffer(0)]],
    device const float* down_filter [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, t = gid.y;
    if (c >= channels || t >= tokens) return;
    float acc = 0.0f;
    for (uint k = 0; k < 12; ++k) {
        const int up_t = clamp(int(t * 2 + k) - 5, 0, int(tokens * 2) - 1);
        acc += activated[uint(up_t) * channels + c] * down_filter[k];
    }
    out[t * channels + c] = acc;
}

kernel void mit2_groupnorm1_f32(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float scratch[1024];
    const uint count = tokens * channels;
    float local_sum = 0.0f;
    for (uint i = tid; i < count; i += 1024) {
        local_sum += x[i];
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = scratch[0] / float(count);
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_var = 0.0f;
    for (uint i = tid; i < count; i += 1024) {
        const float d = x[i] - mean;
        local_var += d * d;
    }
    scratch[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = rsqrt(scratch[0] / float(count) + eps);

    for (uint i = tid; i < count; i += 1024) {
        const uint c = i % channels;
        out[i] = (x[i] - mean) * inv_std * gamma[c] + beta[c];
    }
}

kernel void mit2_mish_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        const float v = x[gid];
        const float sp = log(1.0f + exp(v));
        out[gid] = v * tanh(sp);
    }
}

kernel void mit2_timestep_embedding_f32(
    device const float* timesteps [[buffer(0)]],
    device const float* freqs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& half_dim [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint row = gid.y;
    if (row >= batch || col >= half_dim * 2) {
        return;
    }
    const uint f = col < half_dim ? col : col - half_dim;
    const float arg = scale * timesteps[row] * freqs[f];
    out[row * half_dim * 2 + col] = col < half_dim ? cos(arg) : sin(arg);
}

kernel void mit2_attention_single_head_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint tq = gid.y;
    if (tq >= tokens || d >= head_dim) {
        return;
    }
    float max_score = -INFINITY;
    const float scale = rsqrt(float(head_dim));
    for (uint tk = 0; tk < tokens; ++tk) {
        float score = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            score += q[tq * head_dim + i] * k[tk * head_dim + i];
        }
        score *= scale;
        max_score = max(max_score, score);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (uint tk = 0; tk < tokens; ++tk) {
        float score = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            score += q[tq * head_dim + i] * k[tk * head_dim + i];
        }
        const float w = exp(score * scale - max_score);
        denom += w;
        acc += w * v[tk * head_dim + d];
    }
    out[tq * head_dim + d] = acc / denom;
}

kernel void mit2_attention_single_head_causal_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint tq = gid.y;
    if (tq >= tokens || d >= head_dim) {
        return;
    }
    float max_score = -INFINITY;
    const float scale = rsqrt(float(head_dim));
    for (uint tk = 0; tk <= tq; ++tk) {
        float score = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            score += q[tq * head_dim + i] * k[tk * head_dim + i];
        }
        score *= scale;
        max_score = max(max_score, score);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (uint tk = 0; tk <= tq; ++tk) {
        float score = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            score += q[tq * head_dim + i] * k[tk * head_dim + i];
        }
        const float w = exp(score * scale - max_score);
        denom += w;
        acc += w * v[tk * head_dim + d];
    }
    out[tq * head_dim + d] = acc / denom;
}

kernel void mit2_attention_single_query_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& key_tokens [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint d = gid;
    if (d >= head_dim) {
        return;
    }
    float max_score = -INFINITY;
    const float scale = rsqrt(float(head_dim));
    for (uint tk = 0; tk < key_tokens; ++tk) {
        float score = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            score += q[i] * k[tk * head_dim + i];
        }
        score *= scale;
        max_score = max(max_score, score);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (uint tk = 0; tk < key_tokens; ++tk) {
        float score = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            score += q[i] * k[tk * head_dim + i];
        }
        const float w = exp(score * scale - max_score);
        denom += w;
        acc += w * v[tk * head_dim + d];
    }
    out[d] = acc / denom;
}

kernel void mit2_gpt_cached_attention_f32(
    device const float* cache_k [[buffer(0)]],
    device const float* cache_v [[buffer(1)]],
    device const float* current_qkv [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& cache_tokens [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint h = group.x;
    if (h >= heads) {
        return;
    }
    const uint width = heads * head_dim;
    const uint key_tokens = cache_tokens + 1;
    const uint head_base = h * head_dim;
    const float scale = rsqrt(float(head_dim));
    threadgroup float scores[1024];
    threadgroup float scratch[1024];

    float local_max = -INFINITY;
    for (uint tk = tid; tk < key_tokens; tk += 1024) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            const float qv = current_qkv[head_base + d];
            const float kv = tk < cache_tokens
                ? cache_k[tk * width + head_base + d]
                : current_qkv[width + head_base + d];
            score += qv * kv;
        }
        score *= scale;
        scores[tk] = score;
        local_max = max(local_max, score);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint tk = tid; tk < key_tokens; tk += 1024) {
        const float weight = exp(scores[tk] - row_max);
        scores[tk] = weight;
        local_sum += weight;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = scratch[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint tk = 0; tk < key_tokens; ++tk) {
            const float vv = tk < cache_tokens
                ? cache_v[tk * width + head_base + tid]
                : current_qkv[width * 2 + head_base + tid];
            acc += scores[tk] * vv;
        }
        out[head_base + tid] = acc / denom;
    }
}

// GPT cached attention with GPU-resident KV cache. Reads K/V from the
// persistent cache buffer and appends the current token's K/V in-place
// (each head writes its own disjoint slice; attention reads the current
// K/V from current_qkv, so the append has no ordering hazard).
kernel void mit2_gpt_cached_attention_resident_f32(
    device float* cache_k [[buffer(0)]],
    device float* cache_v [[buffer(1)]],
    device const float* current_qkv [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& cache_tokens [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint h = group.x;
    if (h >= heads) {
        return;
    }
    const uint width = heads * head_dim;
    const uint key_tokens = cache_tokens + 1;
    const uint head_base = h * head_dim;
    const float scale = rsqrt(float(head_dim));
    threadgroup float scores[4096];
    threadgroup float scratch[1024];

    // Append current K/V into the cache (disjoint per head, no barrier needed
    // w.r.t. the attention math below which reads current from current_qkv).
    if (tid < head_dim) {
        cache_k[cache_tokens * width + head_base + tid] = current_qkv[width + head_base + tid];
        cache_v[cache_tokens * width + head_base + tid] = current_qkv[width * 2 + head_base + tid];
    }

    float local_max = -INFINITY;
    for (uint tk = tid; tk < key_tokens; tk += 1024) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            const float qv = current_qkv[head_base + d];
            const float kv = tk < cache_tokens
                ? float(cache_k[tk * width + head_base + d])
                : current_qkv[width + head_base + d];
            score += qv * kv;
        }
        score *= scale;
        scores[tk] = score;
        local_max = max(local_max, score);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint tk = tid; tk < key_tokens; tk += 1024) {
        const float weight = exp(scores[tk] - row_max);
        scores[tk] = weight;
        local_sum += weight;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = scratch[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint tk = 0; tk < key_tokens; ++tk) {
            const float vv = tk < cache_tokens
                ? float(cache_v[tk * width + head_base + tid])
                : current_qkv[width * 2 + head_base + tid];
            acc += scores[tk] * vv;
        }
        out[head_base + tid] = acc / denom;
    }
}

// Full-sequence causal multi-head attention over packed QKV [tokens, 3*width]
// (GPT layout: q | k | v sections, head h at offset h*head_dim within each).
// One threadgroup per (head, query token); replaces the per-head host loop.
kernel void mit2_gpt_causal_attention_f32(
    device const float* qkv [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint tid = tid3.x;
    const uint h = group.x;
    const uint tq = group.y;
    if (h >= heads || tq >= tokens) {
        return;
    }
    const uint width = heads * head_dim;
    const uint qkv_width = width * 3;
    const uint key_tokens = tq + 1;  // causal
    const uint head_base = h * head_dim;
    const float scale = rsqrt(float(head_dim));
    threadgroup float scores[4096];
    threadgroup float red[32];
    threadgroup float partial[1024];
    threadgroup float bcast;

    const uint q_base = tq * qkv_width + head_base;
    float local_max = -INFINITY;
    for (uint tk = tid; tk < key_tokens; tk += 1024) {
        const uint k_base = tk * qkv_width + width + head_base;
        float acc = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            acc += qkv[q_base + d] * qkv[k_base + d];
        }
        const float score = acc * scale;
        scores[tk] = score;
        local_max = max(local_max, score);
    }
    const float sg_max = simd_max(local_max);
    if (simd_lane == 0) {
        red[simd_group] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < 32; ++i) {
            m = max(m, red[i]);
        }
        bcast = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = bcast;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint tk = tid; tk < key_tokens; tk += 1024) {
        const float w = exp(scores[tk] - row_max);
        scores[tk] = w;
        local_sum += w;
    }
    const float sg_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        red[simd_group] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            t += red[i];
        }
        bcast = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denom = bcast;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint chunks = 1024 / head_dim;
    const uint chunk = tid / head_dim;
    const uint d = tid % head_dim;
    float acc = 0.0f;
    for (uint tk = chunk; tk < key_tokens; tk += chunks) {
        acc += scores[tk] * qkv[tk * qkv_width + width * 2 + head_base + d];
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = chunks / 2; stride > 0; stride >>= 1) {
        if (chunk < stride) {
            partial[tid] += partial[tid + stride * head_dim];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (chunk == 0) {
        out[tq * width + head_base + d] = denom > 0.0f ? partial[d] / denom : 0.0f;
    }
}

// Simdgroup-per-query causal attention (head_dim == 64) over packed QKV —
// same online-softmax design as the DiT sgq kernel, causal (keys <= query).
kernel void mit2_gpt_causal_attention_sgq_f32(
    device const float* qkv [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint h = tg.x;
    const uint tq = tg.y * 32 + sg;
    if (h >= heads || tq >= tokens) {
        return;
    }
    const uint width = heads * head_dim;
    const uint qkv_width = width * 3;
    const float scale = rsqrt(float(head_dim));
    const uint q_base = tq * qkv_width + h * head_dim;
    const float q0 = qkv[q_base + 2 * lane];
    const float q1 = qkv[q_base + 2 * lane + 1];

    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint tk = 0; tk <= tq; ++tk) {
        const uint k_base = tk * qkv_width + width + h * head_dim;
        const float partial = q0 * qkv[k_base + 2 * lane] + q1 * qkv[k_base + 2 * lane + 1];
        const float score = simd_sum(partial) * scale;
        const float m_new = max(m, score);
        const float corr = exp(m - m_new);
        const float w = exp(score - m_new);
        const uint v_base = tk * qkv_width + width * 2 + h * head_dim;
        acc0 = acc0 * corr + w * qkv[v_base + 2 * lane];
        acc1 = acc1 * corr + w * qkv[v_base + 2 * lane + 1];
        l = l * corr + w;
        m = m_new;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    const uint out_base = tq * width + h * head_dim;
    out[out_base + 2 * lane] = acc0 * inv;
    out[out_base + 2 * lane + 1] = acc1 * inv;
}

kernel void mit2_attention_single_head_masked_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const uint* key_mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint tq = gid.y;
    if (tq >= tokens || d >= head_dim) {
        return;
    }
    float max_score = -INFINITY;
    const float scale = rsqrt(float(head_dim));
    for (uint tk = 0; tk < tokens; ++tk) {
        if (key_mask[tk] == 0) {
            continue;
        }
        float score = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            score += q[tq * head_dim + i] * k[tk * head_dim + i];
        }
        score *= scale;
        max_score = max(max_score, score);
    }
    float denom = 0.0f;
    float acc = 0.0f;
    for (uint tk = 0; tk < tokens; ++tk) {
        if (key_mask[tk] == 0) {
            continue;
        }
        float score = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            score += q[tq * head_dim + i] * k[tk * head_dim + i];
        }
        const float w = exp(score * scale - max_score);
        denom += w;
        acc += w * v[tk * head_dim + d];
    }
    out[tq * head_dim + d] = acc / denom;
}

kernel void mit2_conformer_rel_attention_context_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* p [[buffer(3)]],
    device const float* bias_u [[buffer(4)]],
    device const float* bias_v [[buffer(5)]],
    device const uint* key_mask [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = thread_pos.x;
    const uint h = group.x;
    const uint tq = group.y;
    const uint dim = heads * head_dim;
    threadgroup float scores[1024];
    threadgroup float scratch[1024];

    float local_max = -INFINITY;
    const float scale = rsqrt(float(head_dim));
    for (uint tk = tid; tk < tokens; tk += 1024) {
        float score = -INFINITY;
        if (key_mask[tk] != 0) {
            float ac = 0.0f;
            float bd = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                const uint q_idx = tq * dim + h * head_dim + d;
                const uint k_idx = tk * dim + h * head_dim + d;
                const uint b_idx = h * head_dim + d;
                const float qv = q[q_idx];
                ac += (qv + bias_u[b_idx]) * k[k_idx];
                bd += (qv + bias_v[b_idx]) * p[k_idx];
            }
            score = (ac + bd) * scale;
        }
        scores[tk] = score;
        local_max = max(local_max, score);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint tk = tid; tk < tokens; tk += 1024) {
        const float score = scores[tk];
        const float w = isfinite(score) ? exp(score - row_max) : 0.0f;
        scores[tk] = w;
        local_sum += w;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = scratch[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        if (denom > 0.0f) {
            for (uint tk = 0; tk < tokens; ++tk) {
                const uint v_idx = tk * dim + h * head_dim + tid;
                acc += (scores[tk] / denom) * v[v_idx];
            }
        }
        out[tq * dim + h * head_dim + tid] = acc;
    }
}

// Apply RoPE rotation in-place to the Q and K sections of a packed QKV buffer
// [batch, tokens, 3*dim]. Run once before the fused attention kernel so the
// O(T^2) score loop does pure dot products (no per-pair trig).
kernel void mit2_dit_rope_rotate_qk_batched_f32(
    device float* qkv [[buffer(0)]],
    constant uint& batch [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;            // head_dim/2 pairs
    const uint t = gid.y;               // token
    const uint bh = gid.z;              // batch * heads
    const uint half_dim = head_dim / 2;
    if (pair >= half_dim || t >= tokens || bh >= batch * heads) {
        return;
    }
    const uint b = bh / heads;
    const uint h = bh % heads;
    const uint dim = heads * head_dim;
    const uint qkv_width = dim * 3;
    const uint d0 = pair * 2;
    const float theta = float(t) / pow(10000.0f, float(d0) / float(head_dim));
    const float c = cos(theta);
    const float s = sin(theta);
    const uint base = (b * tokens + t) * qkv_width + h * head_dim + d0;
    // Q section
    {
        const float a = qkv[base];
        const float bb = qkv[base + 1];
        qkv[base] = a * c - bb * s;
        qkv[base + 1] = bb * c + a * s;
    }
    // K section (offset by dim)
    {
        const float a = qkv[base + dim];
        const float bb = qkv[base + dim + 1];
        qkv[base + dim] = a * c - bb * s;
        qkv[base + dim + 1] = bb * c + a * s;
    }
}

kernel void mit2_dit_attention_qkv_rope_f32(
    device const float* qkv [[buffer(0)]],
    device const uint* key_mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = thread_pos.x;
    const uint h = group.x;
    const uint tq = group.y;
    const uint dim = heads * head_dim;
    const uint qkv_width = dim * 3;
    // scores capacity bounds max supported tokens (host: kFusedDitAttentionMaxTokens)
    threadgroup float scores[4096];
    threadgroup float scratch[1024];

    float local_max = -INFINITY;
    const float scale = rsqrt(float(head_dim));
    // Q/K are pre-rotated by mit2_dit_rope_rotate_qk_batched_f32.
    for (uint tk = tid; tk < tokens; tk += 1024) {
        float score = -INFINITY;
        if (key_mask[tk] != 0) {
            const uint q_base = tq * qkv_width + h * head_dim;
            const uint k_base = tk * qkv_width + dim + h * head_dim;
            float acc = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                acc += qkv[q_base + d] * qkv[k_base + d];
            }
            score = acc * scale;
        }
        scores[tk] = score;
        local_max = max(local_max, score);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint tk = tid; tk < tokens; tk += 1024) {
        const float score = scores[tk];
        const float w = isfinite(score) ? exp(score - row_max) : 0.0f;
        scores[tk] = w;
        local_sum += w;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = scratch[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        if (denom > 0.0f) {
            for (uint tk = 0; tk < tokens; ++tk) {
                const uint v_idx = tk * qkv_width + dim * 2 + h * head_dim + tid;
                acc += (scores[tk] / denom) * qkv[v_idx];
            }
        }
        out[tq * dim + h * head_dim + tid] = acc;
    }
}

kernel void mit2_dit_attention_qkv_rope_batched_f32(
    device const float* qkv [[buffer(0)]],
    device const uint* key_mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint tid = thread_pos.x;
    const uint h = group.x;
    const uint tq = group.y;
    const uint b = group.z;
    if (b >= batch) {
        return;
    }
    const uint dim = heads * head_dim;
    const uint qkv_width = dim * 3;
    const uint qkv_batch_base = b * tokens * qkv_width;
    const uint out_batch_base = b * tokens * dim;
    const uint mask_batch_base = b * tokens;
    // scores capacity bounds max supported tokens (host: kFusedDitAttentionMaxTokens)
    threadgroup float scores[4096];
    threadgroup float red[32];
    threadgroup float partial[1024];
    threadgroup float bcast;

    // Q/K are pre-rotated by mit2_dit_rope_rotate_qk_batched_f32.
    float local_max = -INFINITY;
    const float scale = rsqrt(float(head_dim));
    const uint q_base = qkv_batch_base + tq * qkv_width + h * head_dim;
    for (uint tk = tid; tk < tokens; tk += 1024) {
        float score = -INFINITY;
        if (key_mask[mask_batch_base + tk] != 0) {
            const uint k_base = qkv_batch_base + tk * qkv_width + dim + h * head_dim;
            float acc = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                acc += qkv[q_base + d] * qkv[k_base + d];
            }
            score = acc * scale;
        }
        scores[tk] = score;
        local_max = max(local_max, score);
    }
    // simdgroup max reduction (1024 threads = 32 simdgroups)
    const float sg_max = simd_max(local_max);
    if (simd_lane == 0) {
        red[simd_group] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < 32; ++i) {
            m = max(m, red[i]);
        }
        bcast = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = bcast;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint tk = tid; tk < tokens; tk += 1024) {
        const float score = scores[tk];
        const float w = isfinite(score) ? exp(score - row_max) : 0.0f;
        scores[tk] = w;
        local_sum += w;
    }
    const float sg_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        red[simd_group] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            t += red[i];
        }
        bcast = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denom = bcast;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // V accumulation: threads = (chunk, dim); each chunk strides over tokens.
    const uint chunks = 1024 / head_dim;
    const uint chunk = tid / head_dim;
    const uint d = tid % head_dim;
    float acc = 0.0f;
    if (denom > 0.0f) {
        const uint v_col = dim * 2 + h * head_dim + d;
        for (uint tk = chunk; tk < tokens; tk += chunks) {
            acc += scores[tk] * qkv[qkv_batch_base + tk * qkv_width + v_col];
        }
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = chunks / 2; stride > 0; stride >>= 1) {
        if (chunk < stride) {
            partial[tid] += partial[tid + stride * head_dim];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (chunk == 0) {
        out[out_batch_base + tq * dim + h * head_dim + d] = denom > 0.0f ? partial[d] / denom : 0.0f;
    }
}

// Simdgroup-per-query DiT attention (head_dim == 64): one threadgroup carries
// 32 queries, each handled by one simdgroup with ONLINE softmax — no scores
// array, no threadgroup barriers. Each lane owns 2 of the 64 dims; the dot
// product is a 2-MAC partial + simd_sum per key. Q/K must be pre-rotated.
kernel void mit2_dit_attention_sgq_batched_f32(
    device const float* qkv [[buffer(0)]],
    device const uint* key_mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // FlashAttention-2 style simdgroup-matrix kernel (head_dim == 64).
    // Threadgroup = 128 threads = 4 simdgroups; covers 32 queries of one
    // (batch, head). Each simdgroup owns 8 query rows with an fp32 8x64 O
    // accumulator held as simdgroup matrices. K/V staged to threadgroup memory
    // as half (shared by all queries); scores computed as 8x8 MMA tiles with
    // online softmax — rescale applied via diagonal-matrix MMA.
    const uint h = tg.x;
    const uint b = tg.z;
    const uint q0_tile = tg.y * 32;          // first query of this TG
    const uint q0_sg = q0_tile + sg * 8;     // first query of this simdgroup
    const uint dim = heads * head_dim;
    const uint qkv_width = dim * 3;
    const uint batch_base = b * tokens * qkv_width;
    const uint mask_base = b * tokens;
    const float scale = rsqrt(float(head_dim));

    constexpr uint TK = 32;                  // staged keys per tile
    threadgroup half Qs[32 * 64];            // staged (pre-scaled) Q for the TG
    threadgroup half Ks[TK * 64];
    threadgroup half Vs[TK * 64];
    threadgroup uint Ms[TK];
    threadgroup float Sspill[4][8 * 16];     // per-SG score spill (16-key stage)
    threadgroup half Pspill[4][8 * 16];      // per-SG probability tiles
    threadgroup half Dspill[4][8 * 8];       // per-SG diag(corr) tile
    threadgroup float MLrow[4][8][2];        // per-SG running (m, l) per row
    threadgroup float Ospill[4][8 * 64];     // final O spill for epilogue

    // Stage Q (32 queries x 64 dims), pre-multiplied by scale; zero-pad tail.
    for (uint idx = tid; idx < 32 * 64; idx += 128) {
        const uint r = idx / 64;
        const uint d = idx % 64;
        const uint q = q0_tile + r;
        Qs[idx] = (q < tokens) ? half(qkv[batch_base + q * qkv_width + h * head_dim + d] * scale)
                               : half(0.0f);
    }
    if (lane < 8) {
        MLrow[sg][lane][0] = -INFINITY;  // m
        MLrow[sg][lane][1] = 0.0f;       // l
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Q fragments for this SG: 8 d-chunks of 8x8.
    simdgroup_half8x8 Qf[8];
    for (uint d = 0; d < 8; ++d) {
        simdgroup_load(Qf[d], Qs + (sg * 8) * 64 + d * 8, 64);
    }
    // O accumulator: 8 d-chunks of 8x8 (fp32).
    simdgroup_float8x8 Of[8];
    for (uint d = 0; d < 8; ++d) {
        Of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint kt = 0; kt < tokens; kt += TK) {
        const uint tile_n = min(TK, tokens - kt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < TK * 64; idx += 128) {
            const uint kk = idx / 64;
            const uint d = idx % 64;
            if (kk < tile_n) {
                const uint src = batch_base + (kt + kk) * qkv_width + h * head_dim + d;
                Ks[idx] = half(qkv[src + dim]);
                Vs[idx] = half(qkv[src + 2 * dim]);
            } else {
                Ks[idx] = half(0.0f);
                Vs[idx] = half(0.0f);
            }
        }
        for (uint idx = tid; idx < TK; idx += 128) {
            Ms[idx] = (idx < tile_n) ? key_mask[mask_base + kt + idx] : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kc = 0; kc < TK; kc += 16) {
            // Two S(8x8) tiles = Q(8x64) x K^T for 16 keys per softmax stage.
            for (uint half_idx = 0; half_idx < 2; ++half_idx) {
                simdgroup_float8x8 Sf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint d = 0; d < 8; ++d) {
                    simdgroup_half8x8 Kf;
                    simdgroup_load(Kf, Ks + (kc + half_idx * 8) * 64 + d * 8, 64, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf, Qf[d], Kf, Sf);
                }
                simdgroup_store(Sf, Sspill[sg] + half_idx * 8, 16);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Scalar online softmax over 16 columns: lane r (r < 8) owns row r.
            float corr = 1.0f;
            if (lane < 8) {
                const uint r = lane;
                float rowmax = -INFINITY;
                float sc[16];
                for (uint c = 0; c < 16; ++c) {
                    const uint kk = kc + c;
                    const bool valid = (kk < tile_n) && (Ms[kk] != 0u);
                    sc[c] = valid ? Sspill[sg][r * 16 + c] : -INFINITY;
                    rowmax = max(rowmax, sc[c]);
                }
                float mr = MLrow[sg][r][0];
                float lr = MLrow[sg][r][1];
                if (rowmax > -INFINITY) {
                    const float m_new = max(mr, rowmax);
                    corr = exp(mr - m_new);
                    float wsum = 0.0f;
                    for (uint c = 0; c < 16; ++c) {
                        const float w = (sc[c] > -INFINITY) ? exp(sc[c] - m_new) : 0.0f;
                        Pspill[sg][r * 16 + c] = half(w);
                        wsum += w;
                    }
                    lr = lr * corr + wsum;
                    mr = m_new;
                } else {
                    for (uint c = 0; c < 16; ++c) {
                        Pspill[sg][r * 16 + c] = half(0.0f);
                    }
                }
                MLrow[sg][r][0] = mr;
                MLrow[sg][r][1] = lr;
                for (uint c = 0; c < 8; ++c) {
                    Dspill[sg][r * 8 + c] = half((c == r) ? corr : 0.0f);
                }
            }
            // Skip the rescale MMAs when no row max changed (corr == 1 exactly).
            const bool need_rescale = simd_any(lane < 8 && corr != 1.0f);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 Pf0;
            simdgroup_half8x8 Pf1;
            simdgroup_load(Pf0, Pspill[sg], 16);
            simdgroup_load(Pf1, Pspill[sg] + 8, 16);
            if (need_rescale) {
                simdgroup_half8x8 Df;
                simdgroup_load(Df, Dspill[sg], 8);
                for (uint d = 0; d < 8; ++d) {
                    simdgroup_float8x8 Oscaled;
                    simdgroup_multiply(Oscaled, Df, Of[d]);
                    simdgroup_half8x8 Vf;
                    simdgroup_load(Vf, Vs + kc * 64 + d * 8, 64);
                    simdgroup_multiply_accumulate(Oscaled, Pf0, Vf, Oscaled);
                    simdgroup_load(Vf, Vs + (kc + 8) * 64 + d * 8, 64);
                    simdgroup_multiply_accumulate(Of[d], Pf1, Vf, Oscaled);
                }
            } else {
                for (uint d = 0; d < 8; ++d) {
                    simdgroup_half8x8 Vf;
                    simdgroup_load(Vf, Vs + kc * 64 + d * 8, 64);
                    simdgroup_multiply_accumulate(Of[d], Pf0, Vf, Of[d]);
                    simdgroup_load(Vf, Vs + (kc + 8) * 64 + d * 8, 64);
                    simdgroup_multiply_accumulate(Of[d], Pf1, Vf, Of[d]);
                }
            }
        }
    }

    // Epilogue: O rows / l, bounds-checked writes.
    for (uint d = 0; d < 8; ++d) {
        simdgroup_store(Of[d], Ospill[sg] + d * 8, 64);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < 8 * 64; idx += 32) {
        const uint r = idx / 64;
        const uint d = idx % 64;
        const uint q = q0_sg + r;
        if (q >= tokens) {
            continue;
        }
        const float lr = MLrow[sg][r][1];
        const float inv = lr > 0.0f ? 1.0f / lr : 0.0f;
        out[b * tokens * dim + q * dim + h * head_dim + d] = Ospill[sg][r * 64 + d] * inv;
    }
}

// FP32: reuse staged Q storage for the output epilogue to fit 32 keys
// below 32 KiB. Softmax stages remain 16 keys, preserving reduction order.
kernel void mit2_dit_attention_sgq_fp32_batched_f32(
    device const float* qkv [[buffer(0)]],
    device const uint* key_mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // FP32 tiled online-softmax attention (head_dim == 64).
    // Threadgroup = 128 threads = 4 simdgroups; covers 32 queries of one
    // (batch, head). Each simdgroup owns 8 query rows with an fp32 8x64 O
    // accumulator held as simdgroup matrices. K/V staged to threadgroup memory
    // as float (shared by all queries); scores computed as 8x8 MMA tiles with
    // online softmax — rescale applied via diagonal-matrix MMA.
    const uint h = tg.x;
    const uint b = tg.z;
    const uint q0_tile = tg.y * 32;          // first query of this TG
    const uint q0_sg = q0_tile + sg * 8;     // first query of this simdgroup
    const uint dim = heads * head_dim;
    const uint qkv_width = dim * 3;
    const uint batch_base = b * tokens * qkv_width;
    const uint mask_base = b * tokens;
    const float scale = rsqrt(float(head_dim));

    constexpr uint TK = 32;                  // staged keys per tile
    threadgroup float Qs[32 * 64];            // staged (pre-scaled) Q for the TG
    threadgroup float Ks[TK * 64];
    threadgroup float Vs[TK * 64];
    threadgroup uint Ms[TK];
    threadgroup float Sspill[4][8 * 16];     // per-SG score spill (16-key stage)
    threadgroup float Pspill[4][8 * 16];      // per-SG probability tiles
    threadgroup float Dspill[4][8 * 8];       // per-SG diag(corr) tile
    threadgroup float MLrow[4][8][2];        // per-SG running (m, l) per row
    // Qf retains Q in registers after staging. Qs is reused for the epilogue.

    // Stage Q (32 queries x 64 dims), pre-multiplied by scale; zero-pad tail.
    for (uint idx = tid; idx < 32 * 64; idx += 128) {
        const uint r = idx / 64;
        const uint d = idx % 64;
        const uint q = q0_tile + r;
        Qs[idx] = (q < tokens) ? float(qkv[batch_base + q * qkv_width + h * head_dim + d] * scale)
                               : float(0.0f);
    }
    if (lane < 8) {
        MLrow[sg][lane][0] = -INFINITY;  // m
        MLrow[sg][lane][1] = 0.0f;       // l
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Q fragments for this SG: 8 d-chunks of 8x8.
    simdgroup_float8x8 Qf[8];
    for (uint d = 0; d < 8; ++d) {
        simdgroup_load(Qf[d], Qs + (sg * 8) * 64 + d * 8, 64);
    }
    // O accumulator: 8 d-chunks of 8x8 (fp32).
    simdgroup_float8x8 Of[8];
    for (uint d = 0; d < 8; ++d) {
        Of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint kt = 0; kt < tokens; kt += TK) {
        const uint tile_n = min(TK, tokens - kt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < TK * 64; idx += 128) {
            const uint kk = idx / 64;
            const uint d = idx % 64;
            if (kk < tile_n) {
                const uint src = batch_base + (kt + kk) * qkv_width + h * head_dim + d;
                Ks[idx] = float(qkv[src + dim]);
                Vs[idx] = float(qkv[src + 2 * dim]);
            } else {
                Ks[idx] = float(0.0f);
                Vs[idx] = float(0.0f);
            }
        }
        for (uint idx = tid; idx < TK; idx += 128) {
            Ms[idx] = (idx < tile_n) ? key_mask[mask_base + kt + idx] : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kc = 0; kc < TK; kc += 16) {
            // Two S(8x8) tiles = Q(8x64) x K^T for 16 keys per softmax stage.
            for (uint half_idx = 0; half_idx < 2; ++half_idx) {
                simdgroup_float8x8 Sf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint d = 0; d < 8; ++d) {
                    simdgroup_float8x8 Kf;
                    simdgroup_load(Kf, Ks + (kc + half_idx * 8) * 64 + d * 8, 64, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf, Qf[d], Kf, Sf);
                }
                simdgroup_store(Sf, Sspill[sg] + half_idx * 8, 16);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Scalar online softmax over 16 columns: lane r (r < 8) owns row r.
            float corr = 1.0f;
            if (lane < 8) {
                const uint r = lane;
                float rowmax = -INFINITY;
                float sc[16];
                for (uint c = 0; c < 16; ++c) {
                    const uint kk = kc + c;
                    const bool valid = (kk < tile_n) && (Ms[kk] != 0u);
                    sc[c] = valid ? Sspill[sg][r * 16 + c] : -INFINITY;
                    rowmax = max(rowmax, sc[c]);
                }
                float mr = MLrow[sg][r][0];
                float lr = MLrow[sg][r][1];
                if (rowmax > -INFINITY) {
                    const float m_new = max(mr, rowmax);
                    corr = exp(mr - m_new);
                    float wsum = 0.0f;
                    for (uint c = 0; c < 16; ++c) {
                        const float w = (sc[c] > -INFINITY) ? exp(sc[c] - m_new) : 0.0f;
                        Pspill[sg][r * 16 + c] = float(w);
                        wsum += w;
                    }
                    lr = lr * corr + wsum;
                    mr = m_new;
                } else {
                    for (uint c = 0; c < 16; ++c) {
                        Pspill[sg][r * 16 + c] = float(0.0f);
                    }
                }
                MLrow[sg][r][0] = mr;
                MLrow[sg][r][1] = lr;
                for (uint c = 0; c < 8; ++c) {
                    Dspill[sg][r * 8 + c] = float((c == r) ? corr : 0.0f);
                }
            }
            // Skip the rescale MMAs when no row max changed (corr == 1 exactly).
            const bool need_rescale = simd_any(lane < 8 && corr != 1.0f);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 Pf0;
            simdgroup_float8x8 Pf1;
            simdgroup_load(Pf0, Pspill[sg], 16);
            simdgroup_load(Pf1, Pspill[sg] + 8, 16);
            if (need_rescale) {
                simdgroup_float8x8 Df;
                simdgroup_load(Df, Dspill[sg], 8);
                for (uint d = 0; d < 8; ++d) {
                    simdgroup_float8x8 Oscaled;
                    simdgroup_multiply(Oscaled, Df, Of[d]);
                    simdgroup_float8x8 Vf;
                    simdgroup_load(Vf, Vs + kc * 64 + d * 8, 64);
                    simdgroup_multiply_accumulate(Oscaled, Pf0, Vf, Oscaled);
                    simdgroup_load(Vf, Vs + (kc + 8) * 64 + d * 8, 64);
                    simdgroup_multiply_accumulate(Of[d], Pf1, Vf, Oscaled);
                }
            } else {
                for (uint d = 0; d < 8; ++d) {
                    simdgroup_float8x8 Vf;
                    simdgroup_load(Vf, Vs + kc * 64 + d * 8, 64);
                    simdgroup_multiply_accumulate(Of[d], Pf0, Vf, Of[d]);
                    simdgroup_load(Vf, Vs + (kc + 8) * 64 + d * 8, 64);
                    simdgroup_multiply_accumulate(Of[d], Pf1, Vf, Of[d]);
                }
            }
        }
    }

    // Epilogue: O rows / l, bounds-checked writes.
    for (uint d = 0; d < 8; ++d) {
        simdgroup_store(Of[d], Qs + sg * 8 * 64 + d * 8, 64);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < 8 * 64; idx += 32) {
        const uint r = idx / 64;
        const uint d = idx % 64;
        const uint q = q0_sg + r;
        if (q >= tokens) {
            continue;
        }
        const float lr = MLrow[sg][r][1];
        const float inv = lr > 0.0f ? 1.0f / lr : 0.0f;
        out[b * tokens * dim + q * dim + h * head_dim + d] = Qs[sg * 8 * 64 + r * 64 + d] * inv;
    }
}

kernel void mit2_cross_attention_heads_masked_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const uint* key_mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& query_tokens [[buffer(5)]],
    constant uint& key_tokens [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const uint tid = thread_pos.x;
    const uint h = group.x;
    const uint tq = group.y;
    const uint inner = heads * head_dim;
    threadgroup float scores[1024];
    threadgroup float scratch[1024];

    float local_max = -INFINITY;
    const float scale = rsqrt(float(head_dim));
    for (uint tk = tid; tk < key_tokens; tk += 1024) {
        float score = -INFINITY;
        if (key_mask[tk] != 0) {
            score = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                const uint qi = tq * inner + h * head_dim + d;
                const uint ki = tk * inner + h * head_dim + d;
                score += q[qi] * k[ki];
            }
            score *= scale;
        }
        scores[tk] = score;
        local_max = max(local_max, score);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];
    // barrier: scratch[0] broadcast must complete before scratch is reused
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint tk = tid; tk < key_tokens; tk += 1024) {
        const float score = scores[tk];
        const float w = isfinite(score) ? exp(score - row_max) : 0.0f;
        scores[tk] = w;
        local_sum += w;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = scratch[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        if (denom > 0.0f) {
            for (uint tk = 0; tk < key_tokens; ++tk) {
                const uint vi = tk * inner + h * head_dim + tid;
                acc += (scores[tk] / denom) * v[vi];
            }
        }
        out[tq * inner + h * head_dim + tid] = acc;
    }
}

// Qwen3 0.6B: half-rotation RoPE, eight KV heads shared by sixteen query heads.
kernel void itts25_qwen_rope(device const float* x [[buffer(0)]],device float* out [[buffer(1)]],constant uint& tokens [[buffer(2)]],constant uint& heads [[buffer(3)]],constant uint& offset [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=tokens*heads*128)return;uint d=id%128,base=id-d,t=id/(heads*128);
    float angle=float(offset+t)*pow(1000000.0f,-float(d%64)/64.0f);
    float rotated=d<64?-x[base+d+64]:x[base+d-64];out[id]=x[id]*cos(angle)+rotated*sin(angle);
}
kernel void itts25_qwen_store(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],device float* keys [[buffer(2)]],device float* values [[buffer(3)]],constant uint& tokens [[buffer(4)]],constant uint& offset [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(id<tokens*1024){keys[offset*1024+id]=k[id];values[offset*1024+id]=v[id];}
}
kernel void itts25_qwen_attention(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],device float* out [[buffer(3)]],constant uint& tokens [[buffer(4)]],constant uint& offset [[buffer(5)]],uint tid [[thread_index_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    uint head=group.x,query=group.y,base=query*2048+head*128,count=offset+query+1,kv=head/2;
    threadgroup float scores[4096],scratch[128];float peak=-INFINITY;
    for(uint key=tid;key<count;key+=128){float dot=0;for(uint d=0;d<128;d++)dot+=q[base+d]*k[key*1024+kv*128+d];float score=dot*0.08838834764831844f;scores[key]=score;peak=max(peak,score);}
    scratch[tid]=peak;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=64;stride;stride/=2){if(tid<stride)scratch[tid]=max(scratch[tid],scratch[tid+stride]);threadgroup_barrier(mem_flags::mem_threadgroup);}
    float maximum=scratch[0];threadgroup_barrier(mem_flags::mem_threadgroup);float sum=0;
    for(uint key=tid;key<count;key+=128){float p=exp(scores[key]-maximum);scores[key]=p;sum+=p;}
    scratch[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=64;stride;stride/=2){if(tid<stride)scratch[tid]+=scratch[tid+stride];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float value=0;for(uint key=0;key<count;key++)value+=(scores[key]/scratch[0])*v[key*1024+kv*128+tid];out[base+tid]=value;
}

// Same erf polynomial and operation order as CodecOps::gelu.
kernel void itts25_codec_gelu(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant uint& n [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=n)return;float v=x[i]*0.7071067811865475f,a=abs(v),t=1.0f/(1.0f+0.3275911f*a);
    float p=(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;
    float erf=1.0f-p*exp(-a*a);if(v<0)erf=-erf;y[i]=0.5f*x[i]*(1.0f+erf);
}
kernel void itts25_codec_scale(device const float* x [[buffer(0)]],device const float* g [[buffer(1)]],device float* y [[buffer(2)]],constant uint& n [[buffer(3)]],constant uint& c [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i<n)y[i]=x[i]*g[i%c];
}
