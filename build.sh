#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
task_build="${BUILD_DIR:-$task_root/build}"
case "$task_build" in /*) ;; *) task_build="$task_root/$task_build" ;; esac
task_cmake="${CMAKE:-cmake}"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Build standalone IndexTTS 2.5 C++/Metal.
Usage: ./build.sh [CMake configuration options]
Environment: BUILD_DIR, CMAKE_BUILD_TYPE (Release), JOBS (4), CMAKE.
Requires Xcode command line tools, CMake, PCRE2, libsndfile and libsoxr.
Example: brew install cmake pcre2 libsndfile libsoxr
HELP
  exit 0
fi

command -v "$task_cmake" >/dev/null || { echo 'CMake is required: brew install cmake' >&2; exit 1; }
"$task_cmake" -S "$task_root" -B "$task_build" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" "$@"
"$task_cmake" --build "$task_build" --parallel "${JOBS:-4}"
echo "Built: $task_build/itts25-native"
