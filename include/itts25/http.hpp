#pragma once
#include "itts25/runtime.hpp"
namespace itts25 {
struct HttpConfig {
    std::string host="127.0.0.1",model="backends/metal25/bundles/full",frontend="backends/metal25/bundles/frontend",store="backends/metal25/voices",web_file="backends/metal25/web/index.html",webkey;
    std::string example_audio="examples/voice_01.wav";
    uint16_t port=3456;uint32_t queue_size=16,voice_cache_size=20,tts_concurrency=1,clone_concurrency=1;
    bool web=true,seed_example=false;
};
int run_http(const HttpConfig& config);
}
