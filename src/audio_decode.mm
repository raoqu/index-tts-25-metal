#import <AudioToolbox/AudioToolbox.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
namespace itts25 {
// macOS native decoder covers AAC/M4A that libsndfile cannot open. No subprocess.
std::vector<float> decode_apple_audio(const std::string& path,uint32_t& rate) {
    CFURLRef url=CFURLCreateFromFileSystemRepresentation(nullptr,reinterpret_cast<const UInt8*>(path.data()),path.size(),false);
    if(!url)throw std::invalid_argument("Invalid audio path");
    struct File {ExtAudioFileRef handle=nullptr;~File(){if(handle)ExtAudioFileDispose(handle);}} file;
    OSStatus status=ExtAudioFileOpenURL(url,&file.handle);CFRelease(url);
    if(status!=noErr)throw std::invalid_argument("Cannot decode reference audio (AudioToolbox "+std::to_string(status)+")");
    AudioStreamBasicDescription source{};UInt32 size=sizeof(source);
    if(ExtAudioFileGetProperty(file.handle,kExtAudioFileProperty_FileDataFormat,&size,&source)!=noErr||source.mSampleRate<8000||source.mSampleRate>384000||source.mChannelsPerFrame<1||source.mChannelsPerFrame>64)throw std::invalid_argument("Unsupported audio stream");
    rate=source.mSampleRate;AudioStreamBasicDescription client{};client.mSampleRate=source.mSampleRate;client.mFormatID=kAudioFormatLinearPCM;client.mFormatFlags=kAudioFormatFlagIsFloat|kAudioFormatFlagIsPacked|kAudioFormatFlagsNativeEndian;client.mChannelsPerFrame=source.mChannelsPerFrame;client.mBitsPerChannel=32;client.mFramesPerPacket=1;client.mBytesPerFrame=4*client.mChannelsPerFrame;client.mBytesPerPacket=client.mBytesPerFrame;
    if(ExtAudioFileSetProperty(file.handle,kExtAudioFileProperty_ClientDataFormat,sizeof(client),&client)!=noErr)throw std::runtime_error("Cannot configure native audio decoder");
    std::vector<float> mono;const size_t max_frames=16*rate;mono.reserve(max_frames);std::vector<float> interleaved(8192*client.mChannelsPerFrame);
    while(mono.size()<max_frames){UInt32 count=std::min(size_t(8192),max_frames-mono.size());AudioBufferList buffer{};buffer.mNumberBuffers=1;buffer.mBuffers[0].mNumberChannels=client.mChannelsPerFrame;buffer.mBuffers[0].mDataByteSize=count*client.mBytesPerFrame;buffer.mBuffers[0].mData=interleaved.data();auto result=ExtAudioFileRead(file.handle,&count,&buffer);if(result!=noErr)throw std::invalid_argument("Native audio decoding failed");if(!count)break;for(size_t i=0;i<count;i++){double sum=0;for(size_t c=0;c<client.mChannelsPerFrame;c++)sum+=interleaved[i*client.mChannelsPerFrame+c];mono.push_back(sum/client.mChannelsPerFrame);}}
    return mono;
}
}
