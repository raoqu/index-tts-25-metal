#include "itts25/frontend.hpp"
#include <sndfile.h>
#include <soxr.h>
#include <complex>
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
namespace itts25 {
std::vector<float> decode_apple_audio(const std::string&,uint32_t&);
namespace {
constexpr double pi=3.14159265358979323846;
struct SpectralWorkspace {
    FFTSetupD small=vDSP_create_fftsetupD(9,kFFTRadix2),large=vDSP_create_fftsetupD(10,kFFTRadix2);
    std::vector<double> real,imag;
    ~SpectralWorkspace(){if(small)vDSP_destroy_fftsetupD(small);if(large)vDSP_destroy_fftsetupD(large);}
};
std::vector<double> spectrum(const std::vector<double>& frame,size_t n,bool power) {
    if((n!=512&&n!=1024)||frame.size()>n)throw std::invalid_argument("Unsupported FFT size");
    thread_local SpectralWorkspace workspace;
    if(!workspace.small||!workspace.large)throw std::runtime_error("FFT setup allocation failed");
    workspace.real.resize(n);workspace.imag.resize(n);
    std::fill(workspace.real.begin(),workspace.real.end(),0);std::fill(workspace.imag.begin(),workspace.imag.end(),0);
    std::copy(frame.begin(),frame.end(),workspace.real.begin());
    DSPDoubleSplitComplex split{workspace.real.data(),workspace.imag.data()};
    vDSP_fft_zipD(n==512?workspace.small:workspace.large,&split,1,n==512?9:10,FFT_FORWARD);
    std::vector<double> out(n/2+1);
    for(size_t i=0;i<out.size();i++){double v=workspace.real[i]*workspace.real[i]+workspace.imag[i]*workspace.imag[i];out[i]=power?v:std::sqrt(v+1e-9);}
    return out;
}
std::vector<float> soxr_resample(const std::vector<float>& input,uint32_t from,uint32_t to) {
    if(from==to)return input;std::vector<float> out(static_cast<size_t>(std::ceil(input.size()*double(to)/from)));
    auto io=soxr_io_spec(SOXR_FLOAT32_I,SOXR_FLOAT32_I);auto quality=soxr_quality_spec(SOXR_HQ,0);
    size_t used=0,made=0;auto err=soxr_oneshot(from,to,1,input.data(),input.size(),&used,out.data(),out.size(),&made,&io,&quality,nullptr);
    if(err||used!=input.size())throw std::runtime_error(err?err:"Incomplete audio resampling");
    // librosa fixes the ceil-length by zero padding when libsoxr emits one fewer sample.
    std::fill(out.begin()+made,out.end(),0);return out;
}
}
AudioFrontend::AudioFrontend(const std::string& resources):constants_(resources,false) {}
std::vector<float> AudioFrontend::load(const std::string& path,uint32_t rate) {
    SF_INFO info{};SNDFILE* raw=sf_open(path.c_str(),SFM_READ,&info);
    if(!raw){uint32_t source_rate=0;auto mono=decode_apple_audio(path,source_rate);mono=soxr_resample(mono,source_rate,rate);if(mono.size()>15*rate)mono.resize(15*rate);if(mono.size()<rate/10||!std::all_of(mono.begin(),mono.end(),[](float x){return std::isfinite(x);}))throw std::invalid_argument("Invalid/short reference audio");return mono;}
    std::unique_ptr<SNDFILE,decltype(&sf_close)> file(raw,sf_close);
    if(info.samplerate<8000||info.samplerate>384000||info.channels<1||info.channels>64)throw std::invalid_argument("Unsupported reference audio format");
    // Decode only what can contribute to the 15-second reference, plus filter context.
    auto frames=std::min<sf_count_t>(info.frames,16LL*info.samplerate);std::vector<float> interleaved(static_cast<size_t>(frames)*info.channels);
    if(sf_readf_float(file.get(),interleaved.data(),frames)!=frames)throw std::runtime_error("Truncated reference audio");
    std::vector<float> mono(frames);for(size_t i=0;i<mono.size();i++){double v=0;for(int c=0;c<info.channels;c++)v+=interleaved[i*info.channels+c];mono[i]=v/info.channels;if(!std::isfinite(mono[i]))throw std::invalid_argument("Non-finite reference audio");}
    mono=soxr_resample(mono,info.samplerate,rate);if(mono.size()>15*rate)mono.resize(15*rate);
    if(mono.size()<rate/10)throw std::invalid_argument("Reference audio must contain at least 0.1 seconds");return mono;
}
std::vector<float> AudioFrontend::resample16(const std::vector<float>& audio) {
    const auto& kernel=constants_.get("resample.kernel");const int width=constants_.ids("resample.width")[0],orig=441,dest=320;
    const size_t taps=2*width+orig;std::vector<float> out(static_cast<size_t>(std::ceil(audio.size()*double(dest)/orig)));
    for(size_t j=0;j<out.size();j++){const auto phase=j%dest;const int64_t start=(j/dest)*orig-width;double sum=0;
        for(size_t k=0;k<taps;k++){int64_t i=start+k;if(i>=0&&i<static_cast<int64_t>(audio.size()))sum+=double(audio[i])*kernel[phase*taps+k];}out[j]=sum;}
    return out;
}
std::vector<float> AudioFrontend::fbank(const std::vector<float>& audio,bool speech) {
    const size_t frames=1+(audio.size()-400)/160;const auto& window=constants_.get(speech?"speech.window":"camp.window");const auto& filters=constants_.get(speech?"speech.filters":"camp.filters");
    std::vector<double> bank(frames*80);
    for(size_t t=0;t<frames;t++){std::vector<double> frame(400);for(size_t j=0;j<400;j++)frame[j]=speech?static_cast<float>(audio[t*160+j]*32768.f):audio[t*160+j];
        double mean=std::accumulate(frame.begin(),frame.end(),0.0)/400;
        for(auto& x:frame)x=speech?x-mean:static_cast<float>(x-static_cast<float>(mean));
        for(size_t j=400;j-->0;){double v=frame[j]-(speech?.97:double(.97f))*frame[j?j-1:0];frame[j]=speech?v*window[j]:static_cast<float>(static_cast<float>(v)*window[j]);}
        auto spec=spectrum(frame,512,true);
        for(size_t b=0;b<80;b++){double sum=0;for(size_t j=0;j<257;j++)sum+=spec[j]*filters[b*257+j];bank[t*80+b]=std::log(std::max(sum,double(1.1920928955078125e-7f)));}
    }
    std::vector<float> out((speech?(frames+1)/2*2:frames)*80,0);
    for(size_t b=0;b<80;b++){double mean=0;for(size_t t=0;t<frames;t++)mean+=bank[t*80+b];mean/=frames;double var=0;
        if(speech){for(size_t t=0;t<frames;t++)var+=std::pow(bank[t*80+b]-mean,2);var=std::sqrt(var/(frames-1)+1e-7);}
        for(size_t t=0;t<frames;t++)out[t*80+b]=(bank[t*80+b]-mean)/(speech?var:1.0);
    }return out;
}
AudioFeatures AudioFrontend::prepare(const std::string& path) {
    AudioFeatures f;f.audio=load(path,22050);auto audio16=resample16(f.audio);
    f.speech=fbank(audio16,true);f.speech_frames=f.speech.size()/160;
    const size_t frames=1+(audio16.size()-400)/160;f.mask.assign(f.speech_frames,1);if(frames%2)f.mask.back()=0;
    f.camp=fbank(audio16,false);f.camp_frames=f.camp.size()/80;
    f.mel_frames=(f.audio.size()-256)/256+1;f.mel.resize(f.mel_frames*80);
    const auto& window=constants_.get("mel.window");const auto& filters=constants_.get("mel.filters");
    for(size_t t=0;t<f.mel_frames;t++){std::vector<double> frame(1024);for(size_t j=0;j<1024;j++){int64_t i=static_cast<int64_t>(t*256+j)-384;if(i<0)i=-i;if(i>=static_cast<int64_t>(f.audio.size()))i=2*f.audio.size()-2-i;frame[j]=static_cast<float>(f.audio[i]*window[j]);}
        auto spec=spectrum(frame,1024,false);for(size_t b=0;b<80;b++){double sum=0;for(size_t j=0;j<513;j++)sum+=spec[j]*filters[b*513+j];f.mel[t*80+b]=std::log(std::max(sum,1e-5));}
    }return f;
}
AudioFeatures AudioFrontend::prepare_emotion(const std::string& path) {
    AudioFeatures f;auto audio=load(path,16000);f.speech=fbank(audio,true);f.speech_frames=f.speech.size()/160;
    auto frames=1+(audio.size()-400)/160;f.mask.assign(f.speech_frames,1);if(frames%2)f.mask.back()=0;return f;
}
void write_input_bundle(const std::string& path,const AudioFeatures& a,const std::vector<uint32_t>& ids,uint32_t language) {
    std::filesystem::create_directories(path);std::ofstream out(path+"/weights.bin",std::ios::binary);
    const uint32_t header[3]{0x3254494d,1,4096};out.write(reinterpret_cast<const char*>(header),12);
    Json tensors=Json::array();auto add=[&](const std::string& name,const void* data,size_t count,const std::vector<uint32_t>& shape,const std::string& dtype){
        size_t pos=out.tellp(),offset=(pos+4095)/4096*4096;std::string pad(offset-pos,'\0');out.write(pad.data(),pad.size());out.write(reinterpret_cast<const char*>(data),count*4);
        tensors.push_back(Json{{"name",name},{"shape",shape},{"dtype",dtype},{"offset",offset},{"nbytes",count*4},{"layout","row_major"},{"component","input"}});
    };
    add("speech.input",a.speech.data(),a.speech.size(),{1,a.speech_frames,160},"f32");add("speech.mask",a.mask.data(),a.mask.size(),{1,a.speech_frames},"u32");
    add("camp.input",a.camp.data(),a.camp.size(),{1,a.camp_frames,80},"f32");add("prompt.mel",a.mel.data(),a.mel.size(),{1,a.mel_frames,80},"f32");
    add("text.ids",ids.data(),ids.size(),{static_cast<uint32_t>(ids.size())},"u32");add("text.language",&language,1,{1},"u32");
    if(!out)throw std::runtime_error("Cannot write frontend input bundle");
    std::ofstream manifest(path+"/manifest.json");manifest<<Json{{"format","MIT2"},{"version",1},{"alignment",4096},{"endianness","little"},{"weights_file","weights.bin"},{"metadata",{{"target","index-tts2.5"},{"kind","native_input"}}},{"tensors",tensors}}.dump(2)<<'\n';if(!manifest)throw std::runtime_error("Cannot write frontend manifest");
}
}
