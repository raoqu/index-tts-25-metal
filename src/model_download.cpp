#include "itts25/model_download.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <curl/curl.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace itts25 {
namespace {
namespace fs = std::filesystem;
constexpr const char* source = "https://modelscope.cn/models/iwannaido/index-tts25-metal/resolve/master/";
const std::vector<ModelFile> resources = {
#include "itts25/model_files.inc"
};

bool complete(const fs::path& path, const ModelFile& file) {
    std::error_code ec;
    return fs::is_regular_file(path, ec) && fs::file_size(path, ec) == file.size && !ec;
}

std::string sha256(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read downloaded resource: " + path.string());
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::vector<char> buffer(1024 * 1024);
    while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
        CC_SHA256_Update(&context, buffer.data(), static_cast<CC_LONG>(input.gcount()));
    }
    if (!input.eof()) throw std::runtime_error("Cannot hash downloaded resource: " + path.string());
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    std::ostringstream output;
    for (auto byte : digest) output << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return output.str();
}

class DownloadLock {
    int fd_ = -1;
public:
    explicit DownloadLock(const fs::path& directory) {
        fs::create_directories(directory);
        fd_ = open((directory / ".download.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd_ < 0) throw std::runtime_error("Cannot lock model directory: " + directory.string());
        if (flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            std::cerr << "[models] Waiting for another download in " << directory << '\n';
            if (flock(fd_, LOCK_EX) != 0) {
                close(fd_); fd_ = -1;
                throw std::runtime_error("Cannot acquire model download lock");
            }
        }
    }
    ~DownloadLock() { if (fd_ >= 0) close(fd_); }
    DownloadLock(const DownloadLock&) = delete;
    DownloadLock& operator=(const DownloadLock&) = delete;
};

struct Transfer {
    FILE* output;
    std::string label;
    uint64_t offset;
    uint64_t size;
    uint64_t written = 0;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

size_t write_file(char* data, size_t size, size_t count, void* opaque) {
    auto& transfer = *static_cast<Transfer*>(opaque);
    const auto bytes = size * count;
    if (bytes > transfer.size - transfer.offset - transfer.written) return 0;
    const auto written = std::fwrite(data, 1, bytes, transfer.output);
    transfer.written += written;
    return written;
}

int progress(void* opaque, curl_off_t, curl_off_t downloaded, curl_off_t, curl_off_t) {
    auto& transfer = *static_cast<Transfer*>(opaque);
    const auto now = std::chrono::steady_clock::now();
    if (now - transfer.last < std::chrono::seconds(2)) return 0;
    transfer.last = now;
    const auto bytes = transfer.offset + uint64_t(std::max<curl_off_t>(0, downloaded));
    std::cerr << "[models] " << transfer.label << ' ' << (bytes * 100 / transfer.size)
              << "% (" << bytes / (1024 * 1024) << '/' << transfer.size / (1024 * 1024) << " MiB)\n";
    return 0;
}

void download_file(const std::string& url, const fs::path& destination, const ModelFile& file) {
    fs::create_directories(destination.parent_path());
    const fs::path partial = destination.string() + ".part";
    if (fs::exists(destination)) {
        if (!fs::is_regular_file(destination)) throw std::runtime_error("Resource is not a file: " + destination.string());
        if (!fs::exists(partial) && fs::file_size(destination) < file.size) fs::copy_file(destination, partial);
        // Keep an existing file until its verified replacement is ready.
    }
    std::string error;
    for (unsigned attempt = 0; attempt < 3; ++attempt) {
        uint64_t offset = fs::exists(partial) ? fs::file_size(partial) : 0;
        if (offset > file.size) { fs::remove(partial); offset = 0; }
        if (offset == file.size) {
            if (sha256(partial) == file.sha256) { fs::rename(partial, destination); return; }
            fs::remove(partial); offset = 0;
        }
        std::unique_ptr<FILE, decltype(&std::fclose)> output(std::fopen(partial.c_str(), "ab"), &std::fclose);
        if (!output) throw std::runtime_error("Cannot write model resource: " + partial.string());
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
        if (!curl) throw std::runtime_error("Cannot initialize model download");
        Transfer transfer{output.get(), file.path, offset, file.size};
        char detail[CURL_ERROR_SIZE] = {};
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "index-tts25-metal/1.0");
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_MAXREDIRS, 8L);
        curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_LIMIT, 1024L);
        curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_TIME, 60L);
        curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_file);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &transfer);
        curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, progress);
        curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &transfer);
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_ERRORBUFFER, detail);
#if LIBCURL_VERSION_NUM >= 0x075500
        curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
#else
        curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
        curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
#endif
        if (offset) curl_easy_setopt(curl.get(), CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(offset));
        std::cerr << "[models] Downloading " << file.path << (offset ? " (resuming)" : "") << '\n';
        const auto result = curl_easy_perform(curl.get());
        long status = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
        const bool flushed = std::fflush(output.get()) == 0 && !std::ferror(output.get());
        output.reset();
        if (offset && (result == CURLE_RANGE_ERROR || status == 200)) {
            fs::remove(partial);
            error = "Server did not honor the resume range";
        } else if (result != CURLE_OK || !flushed) {
            error = result == CURLE_OK ? "Cannot flush downloaded file" : (detail[0] ? detail : curl_easy_strerror(result));
            if (status >= 400 && status < 500 && status != 408 && status != 429) break;
        } else if (fs::file_size(partial) != file.size) {
            error = "Downloaded size mismatch";
        } else if (sha256(partial) != file.sha256) {
            fs::remove(partial);
            error = "Downloaded SHA-256 mismatch";
        } else {
            fs::rename(partial, destination);
            std::cerr << "[models] Verified " << file.path << '\n';
            return;
        }
        std::cerr << "[models] " << file.path << ": " << error << '\n';
    }
    throw std::runtime_error("Model download failed for " + file.path + ": " + error +
                             "; rerun to resume, or install the resources from https://modelscope.cn/models/iwannaido/index-tts25-metal");
}

bool same_path(const fs::path& left, const fs::path& right) {
    return fs::weakly_canonical(left) == fs::weakly_canonical(right);
}

void ensure_group(const fs::path& directory, const std::string& prefix, bool allow_download) {
    std::vector<ModelFile> files;
    for (const auto& resource : resources) {
        if (resource.path.compare(0, prefix.size(), prefix) == 0)
            files.push_back({resource.path.substr(prefix.size()), resource.size, resource.sha256});
    }
    download_model_files(std::string(source) + prefix, directory, files, allow_download);
}

void require_files(const fs::path& directory, const std::vector<std::string>& paths) {
    for (const auto& path : paths) {
        std::error_code ec;
        const auto file = directory / path;
        if (!fs::is_regular_file(file, ec) || fs::file_size(file, ec) == 0 || ec)
            throw std::runtime_error("Missing resource in custom directory: " + file.string());
    }
}
}

void download_model_files(const std::string& base_url, const fs::path& directory,
                          const std::vector<ModelFile>& files, bool allow_download) {
    bool ready = true;
    for (const auto& file : files) {
        const fs::path relative(file.path);
        if (relative.empty() || relative.is_absolute()) throw std::invalid_argument("Invalid model resource path");
        for (const auto& component : relative)
            if (component == "..") throw std::invalid_argument("Invalid model resource path");
        if (!file.size || file.sha256.size() != 64) throw std::invalid_argument("Invalid model resource metadata");
        if (!complete(directory / relative, file)) {
            if (!allow_download) throw std::runtime_error("Missing or incomplete resource: " + (directory / relative).string() + "; automatic download is disabled (--no-download)");
            ready = false;
        }
    }
    if (ready) return;
    DownloadLock lock(directory);
    static const auto curl_status = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (curl_status != CURLE_OK) throw std::runtime_error("Cannot initialize libcurl");
    for (const auto& file : files)
        if (!complete(directory / file.path, file)) download_file(base_url + file.path, directory / file.path, file);
}

void ensure_model_resources(const fs::path& root, const fs::path& model, const fs::path& frontend,
                            const fs::path& example, bool include_example, bool allow_download) {
    const bool default_model = same_path(model, root / "bundles/full");
    const bool default_frontend = same_path(frontend, root / "bundles/frontend");
    const bool default_example = same_path(example, root / "examples/voice_01.wav");
    // Reject missing custom resources before starting a large default download.
    if (!default_model) require_files(model, {"manifest.json", "weights.bin"});
    if (!default_frontend) require_files(frontend, {"manifest.json", "weights.bin", "frontend.json", "text.tiktoken", "libmecab.2.dylib",
                                  "fsts/zh/tn/tagger.fst", "fsts/zh/tn/verbalizer.fst", "fsts/en/tn/tagger.fst", "fsts/en/tn/verbalizer.fst",
                                  "unidic/sys.dic", "unidic/unk.dic", "unidic/char.bin", "unidic/matrix.bin", "unidic/dicrc", "unidic/mecabrc"});
    if (include_example && !default_example) require_files(example.parent_path(), {example.filename().string()});
    if (default_model) ensure_group(model, "bundles/full/", allow_download);
    if (default_frontend) ensure_group(frontend, "bundles/frontend/", allow_download);
    if (include_example && default_example) ensure_group(example.parent_path(), "examples/", allow_download);
}
}
