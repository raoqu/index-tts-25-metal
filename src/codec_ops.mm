#include "itts25/codec_ops.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace itts25 {
static const char* kernels = R"METAL(
#include <metal_stdlib>
using namespace metal;
float erf_poly(float x) {
  float a=abs(x), t=1.0f/(1.0f+0.3275911f*a);
  float p=(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;
  float y=1.0f-p*exp(-a*a);
  return x < 0 ? -y : y;
}
kernel void gelu_erf(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if(i<n) y[i]=0.5f*x[i]*(1.0f+erf_poly(x[i]*0.7071067811865475f));
}
kernel void scale_channels(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]], device float* y [[buffer(2)]], constant uint& n [[buffer(3)]], constant uint& c [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if(i<n) y[i]=x[i]*g[i%c];
}
kernel void codec_down(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* b [[buffer(2)]], device float* y [[buffer(3)]], constant uint& t [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  uint rows=(t+1)/2;
  if(i>=rows*1024) return;
  uint row=i/1024, out=i%1024;
  float v=b[out];
  for(uint ch=0;ch<1024;ch++) for(uint k=0;k<3;k++) {
    int pos=int(row*2+k)-1;
    if(pos>=0 && pos<int(t)) v+=x[uint(pos)*1024+ch]*w[(out*1024+ch)*3+k];
  }
  y[i]=v;
}
kernel void normalized_fvq(device const float* z [[buffer(0)]], device const float* cb [[buffer(1)]], device uint* ids [[buffer(2)]], constant uint& t [[buffer(3)]], uint row [[thread_position_in_grid]]) {
  if(row>=t) return;
  float v[8], norm=0;
  for(uint d=0;d<8;d++) { v[d]=z[row*8+d]; norm+=v[d]*v[d]; }
  norm=max(sqrt(norm),1e-12f);
  for(uint d=0;d<8;d++) v[d]/=norm;
  uint best=0; float distance=INFINITY;
  for(uint c=0;c<8192;c++) {
    float cn=0; for(uint d=0;d<8;d++) cn+=cb[c*8+d]*cb[c*8+d];
    cn=max(sqrt(cn),1e-12f);
    float dist=0; for(uint d=0;d<8;d++) {float diff=v[d]-cb[c*8+d]/cn; dist+=diff*diff;}
    if(dist<distance) {distance=dist;best=c;}
  }
  ids[row]=best;
}
)METAL";
struct CodecOps::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::unordered_map<std::string,id<MTLComputePipelineState>> pipelines;
    uint64_t submitted = 0;
    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Metal device is unavailable");
        queue = [device newCommandQueue];
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
        } else {
            opts.fastMathEnabled = NO;
        }
        id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:kernels] options:opts error:&err];
        if (!lib) throw std::runtime_error(std::string("Codec Metal compile failed: ") + [[err localizedDescription] UTF8String]);
        for (const auto* name : {"gelu_erf", "scale_channels", "codec_down", "normalized_fvq"}) {
            auto fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            auto p = [device newComputePipelineStateWithFunction:fn error:&err];
            if (!p) throw std::runtime_error("Codec pipeline creation failed");
            pipelines.emplace(name,p);
        }
    }
    id<MTLBuffer> input(const std::vector<float>& v) {
        auto buffer = [device newBufferWithBytes:v.data() length:v.size()*4 options:MTLResourceStorageModeShared];
        if (!buffer) throw std::runtime_error("Codec input buffer allocation failed");
        return buffer;
    }
    id<MTLBuffer> output(size_t n) {
        if (!n || n>std::numeric_limits<uint32_t>::max()) throw std::runtime_error("Invalid codec output size");
        auto buffer = [device newBufferWithLength:n*4 options:MTLResourceStorageModeShared];
        if (!buffer) throw std::runtime_error("Codec output buffer allocation failed");
        return buffer;
    }
    void run(const std::string& name, const std::vector<id<MTLBuffer>>& buffers, const std::vector<uint32_t>& args, uint32_t n) {
        auto cb = [queue commandBuffer];
        auto enc = [cb computeCommandEncoder];
        auto p = pipelines.at(name);
        [enc setComputePipelineState:p];
        NSUInteger i=0;
        for(auto b:buffers) [enc setBuffer:b offset:0 atIndex:i++];
        for(const auto& a:args) [enc setBytes:&a length:sizeof(a) atIndex:i++];
        const auto width = std::min<NSUInteger>(p.maxTotalThreadsPerThreadgroup,64);
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; submitted++;
        if(cb.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error(std::string("Codec Metal execution failed: ") + [[cb.error localizedDescription] UTF8String]);
    }
};
CodecOps::CodecOps() { @autoreleasepool { impl_ = std::make_unique<Impl>(); } }
CodecOps::~CodecOps() = default;
uint64_t CodecOps::submissions() const { return impl_->submitted; }
std::vector<float> CodecOps::gelu(const std::vector<float>& x) {
    @autoreleasepool {
        auto out=impl_->output(x.size());
        impl_->run("gelu_erf",{impl_->input(x),out},{static_cast<uint32_t>(x.size())},static_cast<uint32_t>(x.size()));
        auto p=static_cast<const float*>(out.contents); return {p,p+x.size()};
    }
}
std::vector<float> CodecOps::scale(const std::vector<float>& x,const std::vector<float>& g) {
    if(g.empty() || x.size()%g.size()) throw std::invalid_argument("Invalid channel scale dimensions");
    @autoreleasepool {
        auto out=impl_->output(x.size());
        impl_->run("scale_channels",{impl_->input(x),impl_->input(g),out},{static_cast<uint32_t>(x.size()),static_cast<uint32_t>(g.size())},static_cast<uint32_t>(x.size()));
        auto p=static_cast<const float*>(out.contents); return {p,p+x.size()};
    }
}
std::vector<float> CodecOps::downsample(const std::vector<float>& x,const std::vector<float>& w,const std::vector<float>& b,uint32_t t) {
    if(!t || t>100000 || x.size()!=static_cast<size_t>(t)*1024 || w.size()!=1024*1024*3 || b.size()!=1024)
        throw std::invalid_argument("Invalid codec downsample dimensions");
    @autoreleasepool {
        const uint32_t n=((t+1)/2)*1024;
        auto out=impl_->output(n);
        impl_->run("codec_down",{impl_->input(x),impl_->input(w),impl_->input(b),out},{t},n);
        auto p=static_cast<const float*>(out.contents); return {p,p+n};
    }
}
std::vector<uint32_t> CodecOps::quantize(const std::vector<float>& z,const std::vector<float>& cb) {
    if(z.empty() || z.size()%8 || cb.size()!=8192*8) throw std::invalid_argument("Invalid FVQ dimensions");
    @autoreleasepool {
        auto t=static_cast<uint32_t>(z.size()/8); auto out=impl_->output(t);
        impl_->run("normalized_fvq",{impl_->input(z),impl_->input(cb),out},{t},t);
        auto p=static_cast<const uint32_t*>(out.contents); return {p,p+t};
    }
}
}
