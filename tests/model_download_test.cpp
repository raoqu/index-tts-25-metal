#include "itts25/model_download.hpp"
#include "httplib.h"
#include <CommonCrypto/CommonDigest.h>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::string digest(const std::string& data) {
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.data(), static_cast<CC_LONG>(data.size()), bytes);
    std::ostringstream output;
    for (auto byte : bytes) output << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return output.str();
}
void write(const fs::path& path, const std::string& data) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << data;
}
std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
template<class F> void fails(F action, const char* message) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

int main() {
    httplib::Server server;
    std::thread thread;
    char directory[] = "/tmp/itts25-download-test-XXXXXX";
    const char* temporary = mkdtemp(directory);
    if (!temporary) return 1;
    const fs::path root = temporary;
    try {
        const std::string payload(32768, 'x');
        const std::vector<itts25::ModelFile> files{{"nested/weights.bin", payload.size(), digest(payload)}};
        enum { normal, interrupted, no_range, corrupt, not_found };
        std::atomic<int> mode{normal}, requests{0}, resumed{0};
        server.Get("/nested/weights.bin", [&](const httplib::Request& req, httplib::Response& res) {
            ++requests;
            uint64_t offset = 0;
            const auto range = req.get_header_value("Range");
            if (!range.empty()) {
                ++resumed;
                offset = std::stoull(range.substr(6));
            }
            if (mode == not_found) { res.status = 404; res.set_content("missing", "text/plain"); return; }
            if (mode == no_range) { offset = 0; res.status = 200; }
            if (offset) {
                res.status = 206;
                res.set_header("Content-Range", "bytes " + std::to_string(offset) + "-" +
                               std::to_string(payload.size() - 1) + "/" + std::to_string(payload.size()));
            }
            if (mode == interrupted) {
                res.set_content_provider(payload.size(), "application/octet-stream",
                    [&](size_t, size_t, httplib::DataSink& sink) {
                        sink.write(payload.data(), 1024);
                        return false;
                    });
            } else {
                res.set_content(std::string(payload.size(), mode == corrupt ? 'z' : 'x'), "application/octet-stream");
            }
        });
        const auto port = server.bind_to_any_port("127.0.0.1");
        check(port > 0, "Cannot bind test HTTP server");
        thread = std::thread([&] { server.listen_after_bind(); });
        server.wait_until_ready();
        const std::string base = "http://127.0.0.1:" + std::to_string(port) + "/";
        auto download = [&](const char* dir, bool enabled = true) {
            itts25::download_model_files(base, root / dir, files, enabled);
        };

        fails([&] { download("offline", false); }, "Offline mode accepted missing resources");
        check(requests == 0 && !fs::exists(root / "offline"), "Offline mode accessed network or wrote files");
        download("fresh");
        check(read(root / "fresh/nested/weights.bin") == payload, "Fresh download content differs");
        const auto initial_requests = requests.load();
        download("fresh", false);
        download("fresh");
        check(requests == initial_requests, "Complete resources accessed the network");

        write(root / "resume/nested/weights.bin.part", payload.substr(0, 4096));
        const auto initial_resumed = resumed.load();
        download("resume");
        check(resumed > initial_resumed && read(root / "resume/nested/weights.bin") == payload,
              "Partial download did not resume correctly");
        write(root / "truncated/nested/weights.bin", payload.substr(0, 4096));
        download("truncated");
        check(read(root / "truncated/nested/weights.bin") == payload, "Truncated final file did not resume");
        write(root / "promote/nested/weights.bin.part", payload);
        const auto before_promote = requests.load();
        download("promote");
        check(requests == before_promote && read(root / "promote/nested/weights.bin") == payload,
              "Complete partial file was not verified and promoted offline");

        mode = no_range;
        write(root / "no-range/nested/weights.bin.part", payload.substr(0, 4096));
        download("no-range");
        check(read(root / "no-range/nested/weights.bin") == payload, "Ignored Range corrupted the file");

        mode = interrupted;
        fails([&] { download("interrupted"); }, "Interrupted response was accepted");
        check(!fs::exists(root / "interrupted/nested/weights.bin") &&
              fs::file_size(root / "interrupted/nested/weights.bin.part") > 0,
              "Interrupted download was promoted or lost its partial bytes");
        mode = normal;
        download("interrupted");
        check(read(root / "interrupted/nested/weights.bin") == payload, "Retry did not recover from interruption");

        mode = corrupt;
        write(root / "corrupt/nested/weights.bin", "old resource with invalid size");
        fails([&] { download("corrupt"); }, "Corrupt SHA-256 was accepted");
        check(read(root / "corrupt/nested/weights.bin") == "old resource with invalid size" &&
              !fs::exists(root / "corrupt/nested/weights.bin.part"), "Corruption replaced an existing file");
        mode = not_found;
        const auto before_404 = requests.load();
        fails([&] { download("404"); }, "HTTP error was accepted");
        check(requests == before_404 + 1 && !fs::exists(root / "404/nested/weights.bin"),
              "HTTP 404 was retried or saved as a resource");

        mode = normal;
        const auto before_concurrent = requests.load();
        std::exception_ptr first_error, second_error;
        std::thread first([&] { try { download("concurrent"); } catch (...) { first_error = std::current_exception(); } });
        std::thread second([&] { try { download("concurrent"); } catch (...) { second_error = std::current_exception(); } });
        first.join(); second.join();
        if (first_error) std::rethrow_exception(first_error);
        if (second_error) std::rethrow_exception(second_error);
        check(requests == before_concurrent + 1 && read(root / "concurrent/nested/weights.bin") == payload,
              "Concurrent processes duplicated or corrupted the download");
        fails([&] { itts25::download_model_files(base, root / "unsafe", {{"../escape", 1, digest("x")}}); },
              "Parent traversal was accepted");
        check(!fs::exists(root / "escape"), "Unsafe path escaped its resource directory");
        fails([&] { itts25::ensure_model_resources(root, root / "bundles/full", root / "missing-custom-frontend",
                                                  root / "examples/voice_01.wav", false, true); },
              "Missing custom frontend was accepted");
        check(!fs::exists(root / "bundles"), "Missing custom resources started a default download");
        server.stop(); thread.join(); fs::remove_all(root);
        std::cout << "Model downloads: fresh, local/offline, resume, interrupted responses, range fallback, SHA-256, HTTP errors and concurrency passed.\n";
        return 0;
    } catch (const std::exception& e) {
        server.stop(); if (thread.joinable()) thread.join(); fs::remove_all(root);
        std::cerr << e.what() << '\n'; return 1;
    }
}
