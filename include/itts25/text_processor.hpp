#pragma once
#include "itts25/frontend.hpp"
namespace itts25 {
class TextProcessor {
public:
    explicit TextProcessor(const std::string& resources);
    ~TextProcessor();
    std::string process(const std::string& text,const std::string& language,bool normalize=true) const;
    std::vector<std::string> split(const std::string& text,const std::string& prefix,const ByteTokenizer& tokenizer,uint32_t maximum=120) const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
std::string unicode_transform(const std::string&,int);
}
