#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$task_root"
task_build="${BUILD_DIR:-$task_root/build}"
case "$task_build" in /*) ;; *) task_build="$task_root/$task_build" ;; esac
task_model="${MODEL_BUNDLE:-${ITTS25_METAL_BUNDLE:-$task_root/bundles/full}}"
task_frontend="${ITTS25_FRONTEND:-$task_root/bundles/frontend}"
task_store="${MIT2_VOICE_STORE:-$task_root/voices}"
task_web="$task_root/web/index.html"
task_example="$task_root/examples/voice_01.wav"
task_cli=0
task_prepare=0
task_build_needed=1
task_args=()

while (($#)); do
  case "$1" in
    --help|-h)
      cat <<'HELP'
Run standalone native IndexTTS 2.5. Default: http://127.0.0.1:3456/web
Usage: ./dev.sh [native options]
  --prepare-only       Build and check resources, then exit.
  --no-build           Run the existing native binary.
  --cli                Synthesize/clone with --voice, --text, --output.
  --host HOST --port N --web --server --webkey KEY
  --model_bundle DIR --frontend DIR --voice_store DIR --web_file FILE
Resources are located beside this script; it can be called from any directory.
Environment: BUILD_DIR, HOST, PORT, MODEL_BUNDLE, ITTS25_FRONTEND,
             MIT2_VOICE_STORE, MIT2_WEBKEY, JOBS.
HELP
      exit 0 ;;
    --cli) task_cli=1; shift ;;
    --prepare-only) task_prepare=1; shift ;;
    --no-build) task_build_needed=0; shift ;;
    --model_bundle|--frontend|--voice_store|--web_file|--example_audio)
      if (($#<2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --model_bundle) task_model="$2" ;;
        --frontend) task_frontend="$2" ;;
        --voice_store) task_store="$2" ;;
        --web_file) task_web="$2" ;;
        --example_audio) task_example="$2" ;;
      esac
      shift 2 ;;
    --model_bundle=*) task_model="${1#*=}"; shift ;;
    --frontend=*) task_frontend="${1#*=}"; shift ;;
    --voice_store=*) task_store="${1#*=}"; shift ;;
    --web_file=*) task_web="${1#*=}"; shift ;;
    --example_audio=*) task_example="${1#*=}"; shift ;;
    *) task_args+=("$1"); shift ;;
  esac
done

if ((task_build_needed)); then "$task_root/build.sh"; fi
[[ -x "$task_build/itts25-native" ]] || { echo "Native binary missing: $task_build/itts25-native; run ./build.sh" >&2; exit 1; }
for task_file in "$task_model/manifest.json" "$task_model/weights.bin" \
  "$task_frontend/manifest.json" "$task_frontend/weights.bin" "$task_frontend/frontend.json" \
  "$task_frontend/text.tiktoken" "$task_frontend/libmecab.2.dylib" "$task_frontend/unidic/sys.dic" \
  "$task_frontend/fsts/zh/tn/tagger.fst" "$task_frontend/fsts/zh/tn/verbalizer.fst" \
  "$task_frontend/fsts/en/tn/tagger.fst" "$task_frontend/fsts/en/tn/verbalizer.fst"; do
  [[ -f "$task_file" ]] || { echo "Missing resource: $task_file" >&2; exit 1; }
done
if ((task_prepare)); then echo 'Native binary and model/frontend packages are ready.'; exit 0; fi

if ((task_cli)); then
  exec "$task_build/itts25-native" --cli --model_bundle "$task_model" --frontend "$task_frontend" ${task_args[@]+"${task_args[@]}"}
fi
exec "$task_build/itts25-native" --http --web --model_bundle "$task_model" --frontend "$task_frontend" \
  --voice_store "$task_store" --web_file "$task_web" --example_audio "$task_example" --seed_example \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-3456}" --webkey "${MIT2_WEBKEY:-}" \
  --queue_size "${MIT2_QUEUE_SIZE:-16}" --voice_cache_size "${MIT2_VOICE_CACHE_SIZE:-20}" \
  --tts_concurrency "${MIT2_TTS_CONCURRENCY:-1}" --clone_concurrency "${MIT2_CLONE_CONCURRENCY:-1}" \
  ${task_args[@]+"${task_args[@]}"}
