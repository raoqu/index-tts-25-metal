#import <Foundation/Foundation.h>
#include <string>
#include <stdexcept>
namespace itts25 {
std::string unicode_transform(const std::string& input,int mode) {
    @autoreleasepool {
        NSString* text=[[NSString alloc] initWithBytes:input.data() length:input.size() encoding:NSUTF8StringEncoding];
        if(!text)throw std::invalid_argument("Invalid UTF-8 text");
        if(mode==0)text=[text precomposedStringWithCanonicalMapping];
        else if(mode==1)text=[text lowercaseString];else text=[text uppercaseString];
        NSData* bytes=[text dataUsingEncoding:NSUTF8StringEncoding];return std::string(static_cast<const char*>([bytes bytes]),[bytes length]);
    }
}
}
#include <CommonCrypto/CommonDigest.h>
namespace itts25 {
std::string sha256(const void* data,size_t size) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_CTX ctx;CC_SHA256_Init(&ctx);
    auto* bytes=static_cast<const unsigned char*>(data);while(size){auto chunk=static_cast<CC_LONG>(std::min(size,size_t(1u<<30)));CC_SHA256_Update(&ctx,bytes,chunk);bytes+=chunk;size-=chunk;}CC_SHA256_Final(digest,&ctx);
    const char* hex="0123456789abcdef";std::string out;for(auto c:digest){out+=hex[c>>4];out+=hex[c&15];}return out;
}
}
