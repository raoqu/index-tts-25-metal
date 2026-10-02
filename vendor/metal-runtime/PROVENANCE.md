# Metal runtime source

Vendored from `/Users/raoqu/mylab/index-tts2-metal`, commit `eda855b2e9d7cfaa4269380f42880d41b060c4d2`.

Only the bundle loader, Metal/MPS tensor primitives, embedded kernels and embedding CMake script are reused. The IndexTTS2 model, tokenizer, server and production pipeline are not imported. The bundle header adds its missing direct `<memory>` include. Original attribution comments are retained. This checkout is owned by the requesting user; its root contains no separate license file.

2.5 adaptations use safe Metal math, default FP32 resident weights (FP16 explicitly opt-in via ITTS25_FP16_WEIGHTS), and FP32 GPU-resident GPT KV buffers/kernels. These changes preserve the precision of the exported 2.5 weights and CPU golden outputs.

The backend also adapts the required GPU acoustic functions from runtime/impl/dit_cfm_bigvgan.cpp and the length regulator/time embedder/weight folding helpers from runtime/impl/gpt.cpp into src/acoustic_ops.cpp. The FP32 path uses a new float MMA variant of the reference tiled DiT attention with 16-key tiles to fit 32 KiB threadgroup memory. The full CFM trajectory stays GPU resident, following the reference single-pass design while preserving the 2.5 time grid. BigVGAN has FP32 MPS tap GEMMs for dilated and strided transposed convolution, plus a separable anti-alias activation with reusable intermediate storage. GPT single-token projection uses the reference FP32 simdgroup GEMV. The implementation does not depend on the reference checkout at build or inference time.

src/emotion_ops.cpp adapts the nine required emotion Conformer/Perceiver GPU functions from runtime/impl/gpt.cpp. It does not import the 2.0 speaker conditioning or TTS CLI.

src/campplus_ops.cpp adapts the required CAMPPlus graph and small normalization/pooling helpers from runtime/impl/tts_synthesis/campplus_speaker_embedding.cpp and runtime/impl/tokenizer_tests.cpp. Its 1D/2D convolution helpers are replaced with the new Metal implementation in src/camp_convolution.mm. W2V-BERT is implemented independently from the locally installed official transformers model source, with a new FP32 relative attention kernel.

The native Qwen3 graph and RoPE/GQA kernels are independently implemented from the locally installed Transformers 4.52.1 Qwen3 source (Apache-2.0, Qwen/Alibaba and Hugging Face). GPT sampling processors follow the installed Transformers semantics; the RNG helper derives from the reference GPT runtime.
