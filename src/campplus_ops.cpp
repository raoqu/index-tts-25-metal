// Adapted native CAMPPlus graph from index-tts2-metal, commit eda855b2e9d7cfaa4269380f42880d41b060c4d2.
// This module loads actual CAMPPlus weights and uses Metal convolutions, independently of the 2.0 pipeline.
#include "itts25/campplus.hpp"
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
namespace itts25::camp_detail {
thread_local CampConvolution* active_convolution=nullptr;
static std::vector<float> tensor_as_f32(const mit2::Bundle& bundle,const std::string& name) {
    const auto* tensor=bundle.find(name);
    if(!tensor || tensor->dtype!="f32") throw std::runtime_error("Missing FP32 CAMPPlus tensor: "+name);
    std::vector<float> values(tensor->nbytes/4);
    std::memcpy(values.data(),bundle.tensor_data(*tensor),tensor->nbytes);
    return values;
}
struct NchwShape {
    uint32_t channels = 0;
    uint32_t height = 0;
    uint32_t width = 0;
};
std::vector<float> add_relu_nchw(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> concat_channels_ncw(const std::vector<float>& a,
                                       uint32_t a_channels,
                                       const std::vector<float>& b,
                                       uint32_t b_channels,
                                       uint32_t width);
void cpu_batchnorm1d_affine_false_inplace(std::vector<float>& x,
                                          uint32_t channels,
                                          uint32_t width,
                                          const std::vector<float>& running_mean,
                                          const std::vector<float>& running_var,
                                          float eps = 1.0e-5f);
void cpu_batchnorm1d_inplace(std::vector<float>& x,
                             uint32_t channels,
                             uint32_t width,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias,
                             const std::vector<float>& running_mean,
                             const std::vector<float>& running_var,
                             bool relu,
                             float eps = 1.0e-5f);
void cpu_batchnorm2d_inplace(std::vector<float>& x,
                             NchwShape shape,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias,
                             const std::vector<float>& running_mean,
                             const std::vector<float>& running_var,
                             bool relu,
                             float eps = 1.0e-5f);
std::vector<float> cpu_cam_context_ncw(const std::vector<float>& x, uint32_t channels, uint32_t width);
std::vector<float> cpu_campplus_dense_tdnn_layer(const std::string& debug_name,
                                                 const std::vector<float>& input,
                                                 uint32_t in_channels,
                                                 uint32_t width,
                                                 const std::vector<float>& nonlinear1_bn_weight,
                                                 const std::vector<float>& nonlinear1_bn_bias,
                                                 const std::vector<float>& nonlinear1_bn_running_mean,
                                                 const std::vector<float>& nonlinear1_bn_running_var,
                                                 const std::vector<float>& linear1_weight,
                                                 const std::vector<float>& nonlinear2_bn_weight,
                                                 const std::vector<float>& nonlinear2_bn_bias,
                                                 const std::vector<float>& nonlinear2_bn_running_mean,
                                                 const std::vector<float>& nonlinear2_bn_running_var,
                                                 const std::vector<float>& cam_linear_local_weight,
                                                 const std::vector<float>& cam_linear1_weight,
                                                 const std::vector<float>& cam_linear1_bias,
                                                 const std::vector<float>& cam_linear2_weight,
                                                 const std::vector<float>& cam_linear2_bias,
                                                 uint32_t local_padding = 1,
                                                 uint32_t local_dilation = 1);
std::vector<float> cpu_campplus_head_conv1_bn_relu(const std::vector<float>& fbank,
                                                   const std::vector<float>& conv_weight,
                                                   const std::vector<float>& bn_weight,
                                                   const std::vector<float>& bn_bias,
                                                   const std::vector<float>& bn_running_mean,
                                                   const std::vector<float>& bn_running_var,
                                                   uint32_t frames,
                                                   float eps = 1.0e-5f);
std::vector<float> cpu_campplus_head_layer1_block0(const std::vector<float>& input,
                                                   NchwShape input_shape,
                                                   const std::vector<float>& conv1_weight,
                                                   const std::vector<float>& bn1_weight,
                                                   const std::vector<float>& bn1_bias,
                                                   const std::vector<float>& bn1_running_mean,
                                                   const std::vector<float>& bn1_running_var,
                                                   const std::vector<float>& conv2_weight,
                                                   const std::vector<float>& bn2_weight,
                                                   const std::vector<float>& bn2_bias,
                                                   const std::vector<float>& bn2_running_mean,
                                                   const std::vector<float>& bn2_running_var,
                                                   const std::vector<float>& shortcut_weight,
                                                   const std::vector<float>& shortcut_bn_weight,
                                                   const std::vector<float>& shortcut_bn_bias,
                                                   const std::vector<float>& shortcut_bn_running_mean,
                                                   const std::vector<float>& shortcut_bn_running_var,
                                                   NchwShape& output_shape);
std::vector<float> cpu_campplus_head_layer1_block1(const std::vector<float>& input,
                                                   NchwShape input_shape,
                                                   const std::vector<float>& conv1_weight,
                                                   const std::vector<float>& bn1_weight,
                                                   const std::vector<float>& bn1_bias,
                                                   const std::vector<float>& bn1_running_mean,
                                                   const std::vector<float>& bn1_running_var,
                                                   const std::vector<float>& conv2_weight,
                                                   const std::vector<float>& bn2_weight,
                                                   const std::vector<float>& bn2_bias,
                                                   const std::vector<float>& bn2_running_mean,
                                                   const std::vector<float>& bn2_running_var,
                                                   NchwShape& output_shape);
std::vector<float> cpu_conv1d_ncw_bias(const std::vector<float>& x,
                                       const std::vector<float>& weight,
                                       const std::vector<float>& bias,
                                       uint32_t in_channels,
                                       uint32_t input_width,
                                       uint32_t out_channels,
                                       uint32_t kernel,
                                       uint32_t stride,
                                       uint32_t padding,
                                       uint32_t dilation,
                                       uint32_t& output_width);
std::vector<float> cpu_conv1d_ncw_no_bias(const std::vector<float>& x,
                                          const std::vector<float>& weight,
                                          uint32_t in_channels,
                                          uint32_t input_width,
                                          uint32_t out_channels,
                                          uint32_t kernel,
                                          uint32_t stride,
                                          uint32_t padding,
                                          uint32_t dilation,
                                          uint32_t& output_width);
std::vector<float> cpu_conv2d_nchw_no_bias(const std::vector<float>& x,
                                           NchwShape input_shape,
                                           const std::vector<float>& weight,
                                           uint32_t out_channels,
                                           uint32_t kernel_h,
                                           uint32_t kernel_w,
                                           uint32_t stride_h,
                                           uint32_t stride_w,
                                           uint32_t pad_h,
                                           uint32_t pad_w,
                                           NchwShape& output_shape);
std::vector<float> cpu_stats_pool_ncw_unbiased(const std::vector<float>& x,
                                               uint32_t channels,
                                               uint32_t width);
void relu_inplace(std::vector<float>& x);
std::vector<float> run_campplus_dense_block_cpu(const mit2::Bundle& model,
                                                const std::string& block_prefix,
                                                std::vector<float> input,
                                                uint32_t in_channels,
                                                uint32_t layer_count,
                                                uint32_t width,
                                                uint32_t local_padding,
                                                uint32_t local_dilation);
std::vector<float> run_campplus_style_forward_cpu(const mit2::Bundle& model,
                                                  const std::vector<float>& fbank,
                                                  uint32_t fbank_frames);
std::vector<float> run_campplus_transit_cpu(const mit2::Bundle& model,
                                            const std::string& prefix,
                                            std::vector<float> input,
                                            uint32_t in_channels,
                                            uint32_t out_channels,
                                            uint32_t width);
void sigmoid_inplace(std::vector<float>& x);

std::vector<float> add_relu_nchw(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("add_relu_nchw size mismatch");
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = std::max(a[i] + b[i], 0.0f);
    }
    return out;
}

std::vector<float> concat_channels_ncw(const std::vector<float>& a,
                                       uint32_t a_channels,
                                       const std::vector<float>& b,
                                       uint32_t b_channels,
                                       uint32_t width) {
    if (a.size() != static_cast<size_t>(a_channels) * width ||
        b.size() != static_cast<size_t>(b_channels) * width) {
        throw std::runtime_error("concat_channels_ncw shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(a_channels + b_channels) * width);
    std::copy(a.begin(), a.end(), out.begin());
    std::copy(b.begin(), b.end(), out.begin() + static_cast<std::ptrdiff_t>(a.size()));
    return out;
}

void cpu_batchnorm1d_affine_false_inplace(std::vector<float>& x,
                                          uint32_t channels,
                                          uint32_t width,
                                          const std::vector<float>& running_mean,
                                          const std::vector<float>& running_var,
                                          float eps) {
    if (x.size() != static_cast<size_t>(channels) * width ||
        running_mean.size() != channels ||
        running_var.size() != channels) {
        throw std::runtime_error("cpu_batchnorm1d_affine_false_inplace shape mismatch");
    }
    for (uint32_t c = 0; c < channels; ++c) {
        const float scale = 1.0f / std::sqrt(running_var[c] + eps);
        const float shift = -running_mean[c] * scale;
        for (uint32_t t = 0; t < width; ++t) {
            x[static_cast<size_t>(c) * width + t] =
                x[static_cast<size_t>(c) * width + t] * scale + shift;
        }
    }
}

void cpu_batchnorm1d_inplace(std::vector<float>& x,
                             uint32_t channels,
                             uint32_t width,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias,
                             const std::vector<float>& running_mean,
                             const std::vector<float>& running_var,
                             bool relu,
                             float eps) {
    if (x.size() != static_cast<size_t>(channels) * width ||
        weight.size() != channels ||
        bias.size() != channels ||
        running_mean.size() != channels ||
        running_var.size() != channels) {
        throw std::runtime_error("cpu_batchnorm1d_inplace shape mismatch");
    }
    for (uint32_t c = 0; c < channels; ++c) {
        const float scale = weight[c] / std::sqrt(running_var[c] + eps);
        const float shift = bias[c] - running_mean[c] * scale;
        for (uint32_t t = 0; t < width; ++t) {
            float y = x[static_cast<size_t>(c) * width + t] * scale + shift;
            if (relu) {
                y = std::max(y, 0.0f);
            }
            x[static_cast<size_t>(c) * width + t] = y;
        }
    }
}

void cpu_batchnorm2d_inplace(std::vector<float>& x,
                             NchwShape shape,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias,
                             const std::vector<float>& running_mean,
                             const std::vector<float>& running_var,
                             bool relu,
                             float eps) {
    if (x.size() != static_cast<size_t>(shape.channels) * shape.height * shape.width ||
        weight.size() != shape.channels ||
        bias.size() != shape.channels ||
        running_mean.size() != shape.channels ||
        running_var.size() != shape.channels) {
        throw std::runtime_error("cpu_batchnorm2d_inplace shape mismatch");
    }
    for (uint32_t c = 0; c < shape.channels; ++c) {
        const float scale = weight[c] / std::sqrt(running_var[c] + eps);
        const float shift = bias[c] - running_mean[c] * scale;
        for (uint32_t h = 0; h < shape.height; ++h) {
            for (uint32_t w = 0; w < shape.width; ++w) {
                float y = x[(static_cast<size_t>(c) * shape.height + h) * shape.width + w] * scale + shift;
                if (relu) {
                    y = std::max(y, 0.0f);
                }
                x[(static_cast<size_t>(c) * shape.height + h) * shape.width + w] = y;
            }
        }
    }
}

std::vector<float> cpu_cam_context_ncw(const std::vector<float>& x, uint32_t channels, uint32_t width) {
    if (x.size() != static_cast<size_t>(channels) * width || width == 0) {
        throw std::runtime_error("cpu_cam_context_ncw shape mismatch");
    }
    constexpr uint32_t seg_len = 100;
    const uint32_t pooled = (width + seg_len - 1u) / seg_len;
    std::vector<float> context(static_cast<size_t>(channels) * width);
    for (uint32_t c = 0; c < channels; ++c) {
        double mean_acc = 0.0;
        for (uint32_t t = 0; t < width; ++t) {
            mean_acc += x[static_cast<size_t>(c) * width + t];
        }
        const float mean = static_cast<float>(mean_acc / static_cast<double>(width));
        for (uint32_t p = 0; p < pooled; ++p) {
            const uint32_t start = p * seg_len;
            const uint32_t end = std::min(width, start + seg_len);
            double seg_acc = 0.0;
            for (uint32_t t = start; t < end; ++t) {
                seg_acc += x[static_cast<size_t>(c) * width + t];
            }
            const float seg_mean = static_cast<float>(seg_acc / static_cast<double>(end - start));
            for (uint32_t offset = 0; offset < seg_len; ++offset) {
                const uint32_t t = start + offset;
                if (t >= width) {
                    break;
                }
                context[static_cast<size_t>(c) * width + t] = mean + seg_mean;
            }
        }
    }
    return context;
}

std::vector<float> cpu_campplus_dense_tdnn_layer(const std::string& debug_name,
                                                 const std::vector<float>& input,
                                                 uint32_t in_channels,
                                                 uint32_t width,
                                                 const std::vector<float>& nonlinear1_bn_weight,
                                                 const std::vector<float>& nonlinear1_bn_bias,
                                                 const std::vector<float>& nonlinear1_bn_running_mean,
                                                 const std::vector<float>& nonlinear1_bn_running_var,
                                                 const std::vector<float>& linear1_weight,
                                                 const std::vector<float>& nonlinear2_bn_weight,
                                                 const std::vector<float>& nonlinear2_bn_bias,
                                                 const std::vector<float>& nonlinear2_bn_running_mean,
                                                 const std::vector<float>& nonlinear2_bn_running_var,
                                                 const std::vector<float>& cam_linear_local_weight,
                                                 const std::vector<float>& cam_linear1_weight,
                                                 const std::vector<float>& cam_linear1_bias,
                                                 const std::vector<float>& cam_linear2_weight,
                                                 const std::vector<float>& cam_linear2_bias,
                                                 uint32_t local_padding,
                                                 uint32_t local_dilation) {
    constexpr uint32_t bn_channels = 128;
    constexpr uint32_t reduction_channels = 64;
    constexpr uint32_t out_channels = 32;
    if (input.size() != static_cast<size_t>(in_channels) * width) {
        throw std::runtime_error(debug_name + " input shape mismatch");
    }
    auto x = input;
    cpu_batchnorm1d_inplace(x,
                            in_channels,
                            width,
                            nonlinear1_bn_weight,
                            nonlinear1_bn_bias,
                            nonlinear1_bn_running_mean,
                            nonlinear1_bn_running_var,
                            true);
    uint32_t linear1_width = 0;
    x = cpu_conv1d_ncw_no_bias(x, linear1_weight, in_channels, width, bn_channels, 1, 1, 0, 1, linear1_width);
    if (linear1_width != width) {
        throw std::runtime_error(debug_name + " linear1 width mismatch");
    }
    cpu_batchnorm1d_inplace(x,
                            bn_channels,
                            width,
                            nonlinear2_bn_weight,
                            nonlinear2_bn_bias,
                            nonlinear2_bn_running_mean,
                            nonlinear2_bn_running_var,
                            true);
    uint32_t local_width = 0;
    auto local = cpu_conv1d_ncw_no_bias(
        x, cam_linear_local_weight, bn_channels, width, out_channels, 3, 1, local_padding, local_dilation, local_width);
    if (local_width != width) {
        throw std::runtime_error(debug_name + " local width mismatch");
    }
    auto context = cpu_cam_context_ncw(x, bn_channels, width);
    uint32_t context1_width = 0;
    auto context1 = cpu_conv1d_ncw_bias(
        context, cam_linear1_weight, cam_linear1_bias, bn_channels, width, reduction_channels, 1, 1, 0, 1, context1_width);
    if (context1_width != width) {
        throw std::runtime_error(debug_name + " context1 width mismatch");
    }
    relu_inplace(context1);
    uint32_t context2_width = 0;
    auto mask = cpu_conv1d_ncw_bias(
        context1, cam_linear2_weight, cam_linear2_bias, reduction_channels, width, out_channels, 1, 1, 0, 1, context2_width);
    if (context2_width != width) {
        throw std::runtime_error(debug_name + " context2 width mismatch");
    }
    sigmoid_inplace(mask);
    for (size_t i = 0; i < local.size(); ++i) {
        local[i] *= mask[i];
    }
    return local;
}

std::vector<float> cpu_campplus_head_conv1_bn_relu(const std::vector<float>& fbank,
                                                   const std::vector<float>& conv_weight,
                                                   const std::vector<float>& bn_weight,
                                                   const std::vector<float>& bn_bias,
                                                   const std::vector<float>& bn_running_mean,
                                                   const std::vector<float>& bn_running_var,
                                                   uint32_t frames,
                                                   float eps) {
    std::vector<float> input(fbank.size());
    for(uint32_t f=0;f<80;f++)for(uint32_t t=0;t<frames;t++)input[f*frames+t]=fbank[t*80+f];
    auto out=active_convolution->run(input,conv_weight,{1,80,frames,32,3,3,1,1,1,1,1,1,80,frames});
    cpu_batchnorm2d_inplace(out,{32,80,frames},bn_weight,bn_bias,bn_running_mean,bn_running_var,true,eps);
    return out;
}

std::vector<float> cpu_campplus_head_layer1_block0(const std::vector<float>& input,
                                                   NchwShape input_shape,
                                                   const std::vector<float>& conv1_weight,
                                                   const std::vector<float>& bn1_weight,
                                                   const std::vector<float>& bn1_bias,
                                                   const std::vector<float>& bn1_running_mean,
                                                   const std::vector<float>& bn1_running_var,
                                                   const std::vector<float>& conv2_weight,
                                                   const std::vector<float>& bn2_weight,
                                                   const std::vector<float>& bn2_bias,
                                                   const std::vector<float>& bn2_running_mean,
                                                   const std::vector<float>& bn2_running_var,
                                                   const std::vector<float>& shortcut_weight,
                                                   const std::vector<float>& shortcut_bn_weight,
                                                   const std::vector<float>& shortcut_bn_bias,
                                                   const std::vector<float>& shortcut_bn_running_mean,
                                                   const std::vector<float>& shortcut_bn_running_var,
                                                   NchwShape& output_shape) {
    NchwShape conv1_shape;
    auto out = cpu_conv2d_nchw_no_bias(input, input_shape, conv1_weight, 32, 3, 3, 2, 1, 1, 1, conv1_shape);
    cpu_batchnorm2d_inplace(out, conv1_shape, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, true);
    NchwShape conv2_shape;
    out = cpu_conv2d_nchw_no_bias(out, conv1_shape, conv2_weight, 32, 3, 3, 1, 1, 1, 1, conv2_shape);
    cpu_batchnorm2d_inplace(out, conv2_shape, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, false);
    NchwShape shortcut_shape;
    auto shortcut = cpu_conv2d_nchw_no_bias(input, input_shape, shortcut_weight, 32, 1, 1, 2, 1, 0, 0, shortcut_shape);
    cpu_batchnorm2d_inplace(shortcut,
                            shortcut_shape,
                            shortcut_bn_weight,
                            shortcut_bn_bias,
                            shortcut_bn_running_mean,
                            shortcut_bn_running_var,
                            false);
    if (conv2_shape.channels != shortcut_shape.channels ||
        conv2_shape.height != shortcut_shape.height ||
        conv2_shape.width != shortcut_shape.width) {
        throw std::runtime_error("campplus layer1 block0 shortcut shape mismatch");
    }
    output_shape = conv2_shape;
    return add_relu_nchw(out, shortcut);
}

std::vector<float> cpu_campplus_head_layer1_block1(const std::vector<float>& input,
                                                   NchwShape input_shape,
                                                   const std::vector<float>& conv1_weight,
                                                   const std::vector<float>& bn1_weight,
                                                   const std::vector<float>& bn1_bias,
                                                   const std::vector<float>& bn1_running_mean,
                                                   const std::vector<float>& bn1_running_var,
                                                   const std::vector<float>& conv2_weight,
                                                   const std::vector<float>& bn2_weight,
                                                   const std::vector<float>& bn2_bias,
                                                   const std::vector<float>& bn2_running_mean,
                                                   const std::vector<float>& bn2_running_var,
                                                   NchwShape& output_shape) {
    NchwShape conv1_shape;
    auto out = cpu_conv2d_nchw_no_bias(input, input_shape, conv1_weight, 32, 3, 3, 1, 1, 1, 1, conv1_shape);
    cpu_batchnorm2d_inplace(out, conv1_shape, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, true);
    NchwShape conv2_shape;
    out = cpu_conv2d_nchw_no_bias(out, conv1_shape, conv2_weight, 32, 3, 3, 1, 1, 1, 1, conv2_shape);
    cpu_batchnorm2d_inplace(out, conv2_shape, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, false);
    if (conv2_shape.channels != input_shape.channels ||
        conv2_shape.height != input_shape.height ||
        conv2_shape.width != input_shape.width) {
        throw std::runtime_error("campplus layer1 block1 residual shape mismatch");
    }
    output_shape = conv2_shape;
    return add_relu_nchw(out, input);
}

std::vector<float> cpu_conv1d_ncw_bias(const std::vector<float>& x,
                                       const std::vector<float>& weight,
                                       const std::vector<float>& bias,
                                       uint32_t in_channels,
                                       uint32_t input_width,
                                       uint32_t out_channels,
                                       uint32_t kernel,
                                       uint32_t stride,
                                       uint32_t padding,
                                       uint32_t dilation,
                                       uint32_t& output_width) {
    auto out = cpu_conv1d_ncw_no_bias(x,
                                      weight,
                                      in_channels,
                                      input_width,
                                      out_channels,
                                      kernel,
                                      stride,
                                      padding,
                                      dilation,
                                      output_width);
    if (bias.size() != out_channels) {
        throw std::runtime_error("cpu_conv1d_ncw_bias bias shape mismatch");
    }
    for (uint32_t oc = 0; oc < out_channels; ++oc) {
        for (uint32_t t = 0; t < output_width; ++t) {
            out[static_cast<size_t>(oc) * output_width + t] += bias[oc];
        }
    }
    return out;
}

std::vector<float> cpu_conv1d_ncw_no_bias(const std::vector<float>& x,
                                          const std::vector<float>& weight,
                                          uint32_t in_channels,
                                          uint32_t input_width,
                                          uint32_t out_channels,
                                          uint32_t kernel,
                                          uint32_t stride,
                                          uint32_t padding,
                                          uint32_t dilation,
                                          uint32_t& output_width) {
    output_width=(input_width+2*padding-dilation*(kernel-1)-1)/stride+1;
    return active_convolution->run(x,weight,{in_channels,1,input_width,out_channels,1,kernel,1,stride,0,padding,1,dilation,1,output_width});
}

std::vector<float> cpu_conv2d_nchw_no_bias(const std::vector<float>& x,
                                           NchwShape input_shape,
                                           const std::vector<float>& weight,
                                           uint32_t out_channels,
                                           uint32_t kernel_h,
                                           uint32_t kernel_w,
                                           uint32_t stride_h,
                                           uint32_t stride_w,
                                           uint32_t pad_h,
                                           uint32_t pad_w,
                                           NchwShape& output_shape) {
    output_shape={out_channels,(input_shape.height+2*pad_h-kernel_h)/stride_h+1,(input_shape.width+2*pad_w-kernel_w)/stride_w+1};
    return active_convolution->run(x,weight,{input_shape.channels,input_shape.height,input_shape.width,out_channels,kernel_h,kernel_w,stride_h,stride_w,pad_h,pad_w,1,1,output_shape.height,output_shape.width});
}

std::vector<float> cpu_stats_pool_ncw_unbiased(const std::vector<float>& x,
                                               uint32_t channels,
                                               uint32_t width) {
    if (width < 2u || x.size() != static_cast<size_t>(channels) * width) {
        throw std::runtime_error("cpu_stats_pool_ncw_unbiased shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels) * 2u);
    for (uint32_t c = 0; c < channels; ++c) {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(c) * width;
        for (uint32_t t = 0; t < width; ++t) {
            sum += x[base + t];
        }
        const double mean = sum / static_cast<double>(width);
        double sq = 0.0;
        for (uint32_t t = 0; t < width; ++t) {
            const double delta = static_cast<double>(x[base + t]) - mean;
            sq += delta * delta;
        }
        out[c] = static_cast<float>(mean);
        out[channels + c] = static_cast<float>(std::sqrt(sq / static_cast<double>(width - 1u)));
    }
    return out;
}

void relu_inplace(std::vector<float>& x) {
    for (float& value : x) {
        value = std::max(value, 0.0f);
    }
}

std::vector<float> run_campplus_dense_block_cpu(const mit2::Bundle& model,
                                                const std::string& block_prefix,
                                                std::vector<float> input,
                                                uint32_t in_channels,
                                                uint32_t layer_count,
                                                uint32_t width,
                                                uint32_t local_padding,
                                                uint32_t local_dilation) {
    uint32_t channels = in_channels;
    for (uint32_t i = 1; i <= layer_count; ++i) {
        const std::string prefix = block_prefix + ".tdnnd" + std::to_string(i);
        const auto tdnnd = cpu_campplus_dense_tdnn_layer(
            "cpu_campplus_" + block_prefix + "_tdnnd" + std::to_string(i),
            input,
            channels,
            width,
            tensor_as_f32(model, prefix + ".nonlinear1.batchnorm.weight"),
            tensor_as_f32(model, prefix + ".nonlinear1.batchnorm.bias"),
            tensor_as_f32(model, prefix + ".nonlinear1.batchnorm.running_mean"),
            tensor_as_f32(model, prefix + ".nonlinear1.batchnorm.running_var"),
            tensor_as_f32(model, prefix + ".linear1.weight"),
            tensor_as_f32(model, prefix + ".nonlinear2.batchnorm.weight"),
            tensor_as_f32(model, prefix + ".nonlinear2.batchnorm.bias"),
            tensor_as_f32(model, prefix + ".nonlinear2.batchnorm.running_mean"),
            tensor_as_f32(model, prefix + ".nonlinear2.batchnorm.running_var"),
            tensor_as_f32(model, prefix + ".cam_layer.linear_local.weight"),
            tensor_as_f32(model, prefix + ".cam_layer.linear1.weight"),
            tensor_as_f32(model, prefix + ".cam_layer.linear1.bias"),
            tensor_as_f32(model, prefix + ".cam_layer.linear2.weight"),
            tensor_as_f32(model, prefix + ".cam_layer.linear2.bias"),
            local_padding,
            local_dilation);
        input = concat_channels_ncw(input, channels, tdnnd, 32u, width);
        channels += 32u;
    }
    return input;
}

std::vector<float> run_campplus_style_forward_cpu(const mit2::Bundle& model,
                                                  const std::vector<float>& fbank,
                                                  uint32_t fbank_frames) {
    if (fbank_frames == 0 || fbank.size() != static_cast<size_t>(fbank_frames) * 80u) {
        throw std::runtime_error("campplus style fbank shape mismatch");
    }
    const auto conv1 = cpu_campplus_head_conv1_bn_relu(
        fbank,
        tensor_as_f32(model, "campplus.head.conv1.weight"),
        tensor_as_f32(model, "campplus.head.bn1.weight"),
        tensor_as_f32(model, "campplus.head.bn1.bias"),
        tensor_as_f32(model, "campplus.head.bn1.running_mean"),
        tensor_as_f32(model, "campplus.head.bn1.running_var"),
        fbank_frames);
    NchwShape layer0_shape{32u, 80u, fbank_frames};
    NchwShape layer1_block0_shape;
    auto layer1 = cpu_campplus_head_layer1_block0(
        conv1,
        layer0_shape,
        tensor_as_f32(model, "campplus.head.layer1.0.conv1.weight"),
        tensor_as_f32(model, "campplus.head.layer1.0.bn1.weight"),
        tensor_as_f32(model, "campplus.head.layer1.0.bn1.bias"),
        tensor_as_f32(model, "campplus.head.layer1.0.bn1.running_mean"),
        tensor_as_f32(model, "campplus.head.layer1.0.bn1.running_var"),
        tensor_as_f32(model, "campplus.head.layer1.0.conv2.weight"),
        tensor_as_f32(model, "campplus.head.layer1.0.bn2.weight"),
        tensor_as_f32(model, "campplus.head.layer1.0.bn2.bias"),
        tensor_as_f32(model, "campplus.head.layer1.0.bn2.running_mean"),
        tensor_as_f32(model, "campplus.head.layer1.0.bn2.running_var"),
        tensor_as_f32(model, "campplus.head.layer1.0.shortcut.0.weight"),
        tensor_as_f32(model, "campplus.head.layer1.0.shortcut.1.weight"),
        tensor_as_f32(model, "campplus.head.layer1.0.shortcut.1.bias"),
        tensor_as_f32(model, "campplus.head.layer1.0.shortcut.1.running_mean"),
        tensor_as_f32(model, "campplus.head.layer1.0.shortcut.1.running_var"),
        layer1_block0_shape);
    NchwShape layer1_block1_shape;
    layer1 = cpu_campplus_head_layer1_block1(
        layer1,
        layer1_block0_shape,
        tensor_as_f32(model, "campplus.head.layer1.1.conv1.weight"),
        tensor_as_f32(model, "campplus.head.layer1.1.bn1.weight"),
        tensor_as_f32(model, "campplus.head.layer1.1.bn1.bias"),
        tensor_as_f32(model, "campplus.head.layer1.1.bn1.running_mean"),
        tensor_as_f32(model, "campplus.head.layer1.1.bn1.running_var"),
        tensor_as_f32(model, "campplus.head.layer1.1.conv2.weight"),
        tensor_as_f32(model, "campplus.head.layer1.1.bn2.weight"),
        tensor_as_f32(model, "campplus.head.layer1.1.bn2.bias"),
        tensor_as_f32(model, "campplus.head.layer1.1.bn2.running_mean"),
        tensor_as_f32(model, "campplus.head.layer1.1.bn2.running_var"),
        layer1_block1_shape);

    NchwShape layer2_block0_shape;
    auto layer2 = cpu_campplus_head_layer1_block0(
        layer1,
        layer1_block1_shape,
        tensor_as_f32(model, "campplus.head.layer2.0.conv1.weight"),
        tensor_as_f32(model, "campplus.head.layer2.0.bn1.weight"),
        tensor_as_f32(model, "campplus.head.layer2.0.bn1.bias"),
        tensor_as_f32(model, "campplus.head.layer2.0.bn1.running_mean"),
        tensor_as_f32(model, "campplus.head.layer2.0.bn1.running_var"),
        tensor_as_f32(model, "campplus.head.layer2.0.conv2.weight"),
        tensor_as_f32(model, "campplus.head.layer2.0.bn2.weight"),
        tensor_as_f32(model, "campplus.head.layer2.0.bn2.bias"),
        tensor_as_f32(model, "campplus.head.layer2.0.bn2.running_mean"),
        tensor_as_f32(model, "campplus.head.layer2.0.bn2.running_var"),
        tensor_as_f32(model, "campplus.head.layer2.0.shortcut.0.weight"),
        tensor_as_f32(model, "campplus.head.layer2.0.shortcut.1.weight"),
        tensor_as_f32(model, "campplus.head.layer2.0.shortcut.1.bias"),
        tensor_as_f32(model, "campplus.head.layer2.0.shortcut.1.running_mean"),
        tensor_as_f32(model, "campplus.head.layer2.0.shortcut.1.running_var"),
        layer2_block0_shape);
    NchwShape layer2_block1_shape;
    layer2 = cpu_campplus_head_layer1_block1(
        layer2,
        layer2_block0_shape,
        tensor_as_f32(model, "campplus.head.layer2.1.conv1.weight"),
        tensor_as_f32(model, "campplus.head.layer2.1.bn1.weight"),
        tensor_as_f32(model, "campplus.head.layer2.1.bn1.bias"),
        tensor_as_f32(model, "campplus.head.layer2.1.bn1.running_mean"),
        tensor_as_f32(model, "campplus.head.layer2.1.bn1.running_var"),
        tensor_as_f32(model, "campplus.head.layer2.1.conv2.weight"),
        tensor_as_f32(model, "campplus.head.layer2.1.bn2.weight"),
        tensor_as_f32(model, "campplus.head.layer2.1.bn2.bias"),
        tensor_as_f32(model, "campplus.head.layer2.1.bn2.running_mean"),
        tensor_as_f32(model, "campplus.head.layer2.1.bn2.running_var"),
        layer2_block1_shape);

    NchwShape conv2_shape;
    auto conv2 = cpu_conv2d_nchw_no_bias(
        layer2,
        layer2_block1_shape,
        tensor_as_f32(model, "campplus.head.conv2.weight"),
        32u,
        3u,
        3u,
        2u,
        1u,
        1u,
        1u,
        conv2_shape);
    cpu_batchnorm2d_inplace(conv2,
                            conv2_shape,
                            tensor_as_f32(model, "campplus.head.bn2.weight"),
                            tensor_as_f32(model, "campplus.head.bn2.bias"),
                            tensor_as_f32(model, "campplus.head.bn2.running_mean"),
                            tensor_as_f32(model, "campplus.head.bn2.running_var"),
                            true);

    uint32_t tdnn_frames = 0;
    auto tdnn = cpu_conv1d_ncw_no_bias(conv2,
                                       tensor_as_f32(model, "campplus.xvector.tdnn.linear.weight"),
                                       320u,
                                       fbank_frames,
                                       128u,
                                       5u,
                                       2u,
                                       2u,
                                       1u,
                                       tdnn_frames);
    cpu_batchnorm1d_inplace(tdnn,
                            128u,
                            tdnn_frames,
                            tensor_as_f32(model, "campplus.xvector.tdnn.nonlinear.batchnorm.weight"),
                            tensor_as_f32(model, "campplus.xvector.tdnn.nonlinear.batchnorm.bias"),
                            tensor_as_f32(model, "campplus.xvector.tdnn.nonlinear.batchnorm.running_mean"),
                            tensor_as_f32(model, "campplus.xvector.tdnn.nonlinear.batchnorm.running_var"),
                            true);

    const auto block1 = run_campplus_dense_block_cpu(model, "campplus.xvector.block1", tdnn, 128u, 12u, tdnn_frames, 1u, 1u);
    const auto transit1 = run_campplus_transit_cpu(model, "campplus.xvector.transit1", block1, 512u, 256u, tdnn_frames);
    const auto block2 = run_campplus_dense_block_cpu(model, "campplus.xvector.block2", transit1, 256u, 24u, tdnn_frames, 2u, 2u);
    const auto transit2 = run_campplus_transit_cpu(model, "campplus.xvector.transit2", block2, 1024u, 512u, tdnn_frames);
    const auto block3 = run_campplus_dense_block_cpu(model, "campplus.xvector.block3", transit2, 512u, 16u, tdnn_frames, 2u, 2u);
    auto transit3 = run_campplus_transit_cpu(model, "campplus.xvector.transit3", block3, 1024u, 512u, tdnn_frames);
    cpu_batchnorm1d_inplace(transit3,
                            512u,
                            tdnn_frames,
                            tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.weight"),
                            tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.bias"),
                            tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.running_mean"),
                            tensor_as_f32(model, "campplus.xvector.out_nonlinear.batchnorm.running_var"),
                            true);
    const auto stats = cpu_stats_pool_ncw_unbiased(transit3, 512u, tdnn_frames);
    uint32_t dense_width = 0;
    auto dense = cpu_conv1d_ncw_no_bias(stats,
                                        tensor_as_f32(model, "campplus.xvector.dense.linear.weight"),
                                        1024u,
                                        1u,
                                        192u,
                                        1u,
                                        1u,
                                        0u,
                                        1u,
                                        dense_width);
    if (dense_width != 1u) {
        throw std::runtime_error("campplus xvector dense width mismatch");
    }
    cpu_batchnorm1d_affine_false_inplace(dense,
                                         192u,
                                         1u,
                                         tensor_as_f32(model, "campplus.xvector.dense.nonlinear.batchnorm.running_mean"),
                                         tensor_as_f32(model, "campplus.xvector.dense.nonlinear.batchnorm.running_var"));
    return dense;
}

std::vector<float> run_campplus_transit_cpu(const mit2::Bundle& model,
                                            const std::string& prefix,
                                            std::vector<float> input,
                                            uint32_t in_channels,
                                            uint32_t out_channels,
                                            uint32_t width) {
    cpu_batchnorm1d_inplace(input,
                            in_channels,
                            width,
                            tensor_as_f32(model, prefix + ".nonlinear.batchnorm.weight"),
                            tensor_as_f32(model, prefix + ".nonlinear.batchnorm.bias"),
                            tensor_as_f32(model, prefix + ".nonlinear.batchnorm.running_mean"),
                            tensor_as_f32(model, prefix + ".nonlinear.batchnorm.running_var"),
                            true);
    uint32_t out_width = 0;
    const auto out = cpu_conv1d_ncw_no_bias(input,
                                           tensor_as_f32(model, prefix + ".linear.weight"),
                                           in_channels,
                                           width,
                                           out_channels,
                                           1u,
                                           1u,
                                           0u,
                                           1u,
                                           out_width);
    if (out_width != width) {
        throw std::runtime_error(prefix + " width mismatch");
    }
    return out;
}

void sigmoid_inplace(std::vector<float>& x) {
    for (float& value : x) {
        value = 1.0f / (1.0f + std::exp(-value));
    }
}
}
