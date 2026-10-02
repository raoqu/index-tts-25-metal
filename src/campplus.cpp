#include "itts25/campplus.hpp"
#include <stdexcept>
namespace itts25 {
namespace camp_detail {
extern thread_local CampConvolution* active_convolution;
std::vector<float> run_campplus_style_forward_cpu(const mit2::Bundle&,const std::vector<float>&,uint32_t);
}
std::vector<float> CampPlus::encode(const std::vector<float>& fbank,uint32_t frames) {
    if(frames<4 || frames>1600 || fbank.size()!=static_cast<size_t>(frames)*80)throw std::invalid_argument("Invalid CAMPPlus fbank input");
    if(camp_detail::active_convolution)throw std::logic_error("Nested CAMPPlus invocation");
    struct Scope {~Scope(){camp_detail::active_convolution=nullptr;}} scope;
    camp_detail::active_convolution=&convolution_;
    return camp_detail::run_campplus_style_forward_cpu(weights_.bundle(),fbank,frames);
}
}
