#include "itts25/speech_features.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <stdexcept>
namespace itts25 {
static const char* source=R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void relative_attention(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device const float* v [[buffer(2)]],device const uint* mask [[buffer(3)]],device const float* distances [[buffer(4)]],device float* out [[buffer(5)]],constant uint& tokens [[buffer(6)]],uint tid [[thread_index_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
  uint head=group.x, query=group.y,base=query*1024+head*64;
  threadgroup float scores[2048],scratch[128];
  float local_max=-INFINITY;
  for(uint key=tid;key<tokens;key+=128) {
    float dot=0,relative=0;
    int distance=clamp(int(key)-int(query),-64,8)+64;
    for(uint d=0;d<64;d++) {
      dot+=q[base+d]*k[key*1024+head*64+d];
      relative+=q[base+d]*distances[uint(distance)*64+d];
    }
    float score=mask[key] ? dot*0.125f+relative*0.125f : -INFINITY;
    scores[key]=score;local_max=max(local_max,score);
  }
  scratch[tid]=local_max;threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint stride=64;stride;stride/=2) {
    if(tid<stride)scratch[tid]=max(scratch[tid],scratch[tid+stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max=scratch[0];threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum=0;
  for(uint key=tid;key<tokens;key+=128){float p=exp(scores[key]-row_max);scores[key]=p;sum+=p;}
  scratch[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint stride=64;stride;stride/=2){if(tid<stride)scratch[tid]+=scratch[tid+stride];threadgroup_barrier(mem_flags::mem_threadgroup);}
  float denominator=scratch[0];
  if(tid<64){float total=0;for(uint key=0;key<tokens;key++)total+=(scores[key]/denominator)*v[key*1024+head*64+tid];out[base+tid]=total;}
}
)METAL";
struct RelativeAttention::Impl {
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLComputePipelineState> pipeline;uint64_t submitted=0;
    Impl(){
        device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("Metal device unavailable");queue=[device newCommandQueue];
        NSError* error=nil;MTLCompileOptions* opts=[MTLCompileOptions new];
        if(@available(macOS 15.0,*))opts.mathMode=MTLMathModeSafe;else opts.fastMathEnabled=NO;
        auto library=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:opts error:&error];
        if(!library)throw std::runtime_error(std::string("Relative attention compile failed: ")+[[error localizedDescription] UTF8String]);
        pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"relative_attention"] error:&error];
        if(!pipeline)throw std::runtime_error("Relative attention pipeline creation failed");
    }
};
RelativeAttention::RelativeAttention(){@autoreleasepool{impl_=std::make_unique<Impl>();}}
RelativeAttention::~RelativeAttention()=default;
uint64_t RelativeAttention::submissions() const{return impl_->submitted;}
std::vector<float> RelativeAttention::run(const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,const std::vector<uint32_t>& mask,const std::vector<float>& distances,uint32_t tokens){
    if(!tokens || tokens>1600 || q.size()!=static_cast<size_t>(tokens)*1024 || k.size()!=q.size() || v.size()!=q.size() || distances.size()!=73*64 || mask.size()!=tokens || std::none_of(mask.begin(),mask.end(),[](auto x){return x!=0;}))throw std::invalid_argument("Invalid W2V relative attention dimensions");
    @autoreleasepool {
        std::vector<id<MTLBuffer>> buffers;
        for(const auto* x:{&q,&k,&v})buffers.push_back([impl_->device newBufferWithBytes:x->data() length:x->size()*4 options:MTLResourceStorageModeShared]);
        buffers.push_back([impl_->device newBufferWithBytes:mask.data() length:mask.size()*4 options:MTLResourceStorageModeShared]);
        buffers.push_back([impl_->device newBufferWithBytes:distances.data() length:distances.size()*4 options:MTLResourceStorageModeShared]);
        auto output=[impl_->device newBufferWithLength:q.size()*4 options:MTLResourceStorageModeShared];buffers.push_back(output);
        auto cb=[impl_->queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:impl_->pipeline];
        for(NSUInteger i=0;i<buffers.size();i++){if(!buffers[i])throw std::runtime_error("Attention allocation failed");[enc setBuffer:buffers[i] offset:0 atIndex:i];}
        [enc setBytes:&tokens length:4 atIndex:6];[enc dispatchThreadgroups:MTLSizeMake(16,tokens,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];impl_->submitted++;
        if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Relative attention execution failed");
        auto ptr=static_cast<const float*>(output.contents);return {ptr,ptr+q.size()};
    }
}
}
