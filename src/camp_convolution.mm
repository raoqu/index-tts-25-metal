#include "itts25/campplus.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <stdexcept>
namespace itts25 {
static const char* source=R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void convolution(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* out [[buffer(2)]],constant uint* shape [[buffer(3)]],uint i [[thread_position_in_grid]]) {
  uint ic=shape[0],ih=shape[1],iw=shape[2],oc=shape[3],kh=shape[4],kw=shape[5];
  uint sh=shape[6],sw=shape[7],ph=shape[8],pw=shape[9],dh=shape[10],dw=shape[11],oh=shape[12],ow=shape[13];
  if(i>=oc*oh*ow)return;
  uint c=i/(oh*ow),row=(i/ow)%oh,col=i%ow;float acc=0;
  for(uint ci=0;ci<ic;ci++)for(uint ky=0;ky<kh;ky++) {
    int y=int(row*sh+ky*dh)-int(ph);if(y<0 || y>=int(ih))continue;
    for(uint kx=0;kx<kw;kx++) {
      int pos=int(col*sw+kx*dw)-int(pw);if(pos>=0 && pos<int(iw))acc+=x[(ci*ih+uint(y))*iw+uint(pos)]*w[((c*ic+ci)*kh+ky)*kw+kx];
    }
  }
  out[i]=acc;
}
)METAL";
struct CampConvolution::Impl {
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLComputePipelineState> pipeline;uint64_t submitted=0;
    Impl(){
        device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("Metal device unavailable");queue=[device newCommandQueue];NSError* error=nil;
        MTLCompileOptions* opts=[MTLCompileOptions new];if(@available(macOS 15.0,*))opts.mathMode=MTLMathModeSafe;else opts.fastMathEnabled=NO;
        auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:opts error:&error];
        if(!library)throw std::runtime_error(std::string("CAMP convolution compile failed: ")+[[error localizedDescription] UTF8String]);
        pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"convolution"] error:&error];
        if(!pipeline)throw std::runtime_error("CAMP convolution pipeline creation failed");
    }
};
CampConvolution::CampConvolution(){@autoreleasepool{impl_=std::make_unique<Impl>();}}
CampConvolution::~CampConvolution()=default;
uint64_t CampConvolution::submissions() const{return impl_->submitted;}
std::vector<float> CampConvolution::run(const std::vector<float>& x,const std::vector<float>& w,const std::vector<uint32_t>& s) {
    if(s.size()!=14 || x.size()!=static_cast<size_t>(s[0])*s[1]*s[2] || w.size()!=static_cast<size_t>(s[3])*s[0]*s[4]*s[5])throw std::invalid_argument("Invalid CAMP convolution shape");
    size_t n=static_cast<size_t>(s[3])*s[12]*s[13];if(!n || n>200000000)throw std::invalid_argument("CAMP output size out of range");
    @autoreleasepool {
        auto a=[impl_->device newBufferWithBytes:x.data() length:x.size()*4 options:MTLResourceStorageModeShared];
        auto b=[impl_->device newBufferWithBytes:w.data() length:w.size()*4 options:MTLResourceStorageModeShared];
        auto out=[impl_->device newBufferWithLength:n*4 options:MTLResourceStorageModeShared];
        if(!a || !b || !out)throw std::runtime_error("CAMP convolution allocation failed");
        auto cb=[impl_->queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:impl_->pipeline];
        [enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:b offset:0 atIndex:1];[enc setBuffer:out offset:0 atIndex:2];[enc setBytes:s.data() length:s.size()*4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];impl_->submitted++;
        if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("CAMP convolution execution failed");
        auto ptr=static_cast<const float*>(out.contents);return {ptr,ptr+n};
    }
}
}
