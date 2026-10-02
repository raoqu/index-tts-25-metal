#include "itts25/frontend.hpp"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
namespace itts25 {
std::string unicode_transform(const std::string&,int);
namespace {
std::string unbase64(const std::string& input) {
    const std::string alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; uint32_t acc=0; int bits=0;
    for(char c:input){if(c=='=')break;auto n=alphabet.find(c);if(n==std::string::npos)throw std::runtime_error("Invalid base64 vocabulary");acc=(acc<<6)|n;bits+=6;if(bits>=8){bits-=8;out+=static_cast<char>(acc>>bits);}}
    return out;
}
std::string utf8(uint32_t c) {
    if(c<128)return std::string(1,static_cast<char>(c));
    if(c<2048)return std::string{static_cast<char>(192|(c>>6)),static_cast<char>(128|(c&63))};
    return std::string{static_cast<char>(224|(c>>12)),static_cast<char>(128|((c>>6)&63)),static_cast<char>(128|(c&63))};
}
}
struct ByteTokenizer::Impl {
    bool nfc=false;
    pcre2_code* pattern=nullptr;
    std::unordered_map<std::string,uint32_t> ranks,specials,languages;
    std::unordered_map<uint32_t,std::string> pieces;
    ~Impl(){if(pattern)pcre2_code_free(pattern);}
    std::vector<uint32_t> bpe(const std::string& piece) const {
        auto whole=ranks.find(piece);if(whole!=ranks.end())return {whole->second};
        std::vector<std::string> parts;for(char c:piece)parts.emplace_back(1,c);
        while(parts.size()>1){size_t best=parts.size();uint32_t rank=std::numeric_limits<uint32_t>::max();
            for(size_t i=0;i+1<parts.size();i++){auto f=ranks.find(parts[i]+parts[i+1]);if(f!=ranks.end()&&f->second<rank){rank=f->second;best=i;}}
            if(best==parts.size())break;parts[best]+=parts[best+1];parts.erase(parts.begin()+best+1);
        }
        std::vector<uint32_t> out;for(const auto& p:parts){auto f=ranks.find(p);if(f==ranks.end())throw std::runtime_error("Vocabulary lacks byte token");out.push_back(f->second);}return out;
    }
    void ordinary(const std::string& text,std::vector<uint32_t>& out) const {
        auto* m=pcre2_match_data_create_from_pattern(pattern,nullptr);
        size_t pos=0;
        while(pos<text.size()){
            int r=pcre2_match(pattern,reinterpret_cast<PCRE2_SPTR>(text.data()),text.size(),pos,0,m,nullptr);
            if(r<0){pcre2_match_data_free(m);throw std::invalid_argument("Invalid UTF-8 or tokenizer regex mismatch");}
            auto* offsets=pcre2_get_ovector_pointer(m);if(offsets[0]!=pos||offsets[1]<=pos){pcre2_match_data_free(m);throw std::runtime_error("Tokenizer failed to cover input");}
            auto ids=bpe(text.substr(pos,offsets[1]-pos));out.insert(out.end(),ids.begin(),ids.end());pos=offsets[1];
        }pcre2_match_data_free(m);
    }
};
ByteTokenizer::ByteTokenizer(const std::string& path,bool qwen):impl_(std::make_unique<Impl>()) {
    std::ifstream in(path+"/frontend.json");Json cfg;in>>cfg;
    if(cfg.at("target")!="index-tts2.5"||cfg.at("version")!=1)throw std::runtime_error("Expected 2.5 frontend resources");
    impl_->languages=cfg.at("languages").get<decltype(impl_->languages)>();std::string pat;
    if(!qwen){pat=cfg.at("pattern");impl_->specials=cfg.at("specials").get<decltype(impl_->specials)>();
        std::ifstream vocab(path+"/text.tiktoken");std::string bytes;uint32_t rank;while(vocab>>bytes>>rank){bytes=unbase64(bytes);impl_->ranks.emplace(bytes,rank);impl_->pieces.emplace(rank,bytes);}
    }else{
        impl_->nfc=true;
        auto& q=cfg.at("qwen");pat=q.at("pre_tokenizer").at("pretokenizers").at(0).at("pattern").at("Regex");
        std::unordered_map<std::string,std::string> byte_map;uint32_t extra=256;
        for(uint32_t b=0;b<256;b++){uint32_t c=(b>=33&&b<=126)||(b>=161&&b<=172)||b>=174?b:extra++;byte_map[utf8(c)]=std::string(1,static_cast<char>(b));}
        for(auto it=q.at("model").at("vocab").begin();it!=q.at("model").at("vocab").end();++it){std::string bytes;auto s=it.key();for(size_t i=0;i<s.size();){size_t n=(static_cast<unsigned char>(s[i])<128?1:2);bytes+=byte_map.at(s.substr(i,n));i+=n;}uint32_t id=it.value();impl_->ranks.emplace(bytes,id);impl_->pieces.emplace(id,bytes);}
        for(const auto& t:q.at("added_tokens"))impl_->specials[t.at("content")]=t.at("id");
    }
    for(const auto& s:impl_->specials)impl_->pieces[s.second]=s.first;
    if(impl_->ranks.size()<256)throw std::runtime_error("Incomplete tokenizer vocabulary");
    int error;PCRE2_SIZE offset;impl_->pattern=pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pat.data()),pat.size(),PCRE2_UTF|PCRE2_UCP,&error,&offset,nullptr);
    if(!impl_->pattern)throw std::runtime_error("Cannot compile tokenizer regex at "+std::to_string(offset));
}
ByteTokenizer::~ByteTokenizer()=default;
std::vector<uint32_t> ByteTokenizer::encode(const std::string& input) const {
    const std::string text=impl_->nfc?unicode_transform(input,0):input;
    std::vector<uint32_t> out;size_t pos=0;
    while(pos<text.size()){size_t next=text.size();const std::pair<const std::string,uint32_t>* special=nullptr;
        for(const auto& s:impl_->specials){size_t f=text.find(s.first,pos);if(f<next){next=f;special=&s;}}
        impl_->ordinary(text.substr(pos,next-pos),out);if(!special)break;out.push_back(special->second);pos=next+special->first.size();
    }return out;
}
std::string ByteTokenizer::decode(const std::vector<uint32_t>& ids,bool skip) const {
    std::string out;for(auto id:ids){const auto& p=impl_->pieces.at(id);if(skip&&impl_->specials.count(p))continue;out+=p;}return out;
}
uint32_t ByteTokenizer::language(const std::string& lang) const {
    std::string s=lang;std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return std::tolower(c);});auto f=impl_->languages.find(s);return f==impl_->languages.end()?impl_->languages.at("common"):f->second;
}
}
