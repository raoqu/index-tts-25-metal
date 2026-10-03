#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace itts25 {
struct ModelFile {
    std::string path;
    uint64_t size;
    std::string sha256;
};

// Complete local files need no network access. Partial downloads remain in .part files.
void download_model_files(const std::string& base_url, const std::filesystem::path& directory,
                          const std::vector<ModelFile>& files, bool allow_download = true);
void ensure_model_resources(const std::filesystem::path& root,
                            const std::filesystem::path& model,
                            const std::filesystem::path& frontend,
                            const std::filesystem::path& example,
                            bool include_example, bool allow_download);
}
