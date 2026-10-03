# IndexTTS 2.5 Native Metal

独立 macOS C++17/Objective-C++ 项目。HTTP、SQLite、文本前端、音频处理、
音色克隆和所有神经模型在同一原生进程执行。使用完整 IndexTTS 2.5 FP32
权重与25步CFM，运行和构建均不需要Python。

## 构建和运行

```sh
cd ~/mylab/index-tts-25-metal
./build.sh
./dev.sh
```

默认打开 `http://127.0.0.1:7860/web`。`dev.sh` 会自动增量构建，资源根据
脚本位置定位，与调用时的工作目录无关。Ctrl+C停止服务。
依赖：Xcode命令行工具、CMake、PCRE2、libsndfile、libsoxr。

```sh
brew install cmake pcre2 libsndfile libsoxr
./dev.sh --prepare-only
./dev.sh --port 7861
./dev.sh --no-build
./dev.sh --server                     # 仅HTTP API
./dev.sh --cli --voice examples/voice_01.wav \
  --text '你好，这是原生语音合成测试。' --seed 42 --output outputs/test.wav
```

直接执行 `build/itts25-native` 默认启动Web服务，自动从可执行文件附近定位
模型、前端和网页；指定 `--voice`/`--text` 或 `--cli` 可使用CLI模式。
`./build.sh --help`、`./dev.sh --help` 和 `build/itts25-native --help` 显示选项。

HTTP 和 CLI 每次合成成功后会向 stderr 输出一行 TTS 摘要日志：

```text
[TTS] text="你好，这是原生语音合成测试。" audio=3.200s elapsed=1.600s RTF=0.500
```

`text` 保留原文前 50 个 Unicode 字符，超长时追加 `…`，换行等控制字符转义显示。
`audio` 为生成音频时长，`elapsed` 为合成耗时（含文本处理、情感推理和 WAV 写入，
不含模型加载、音色加载或克隆、HTTP 排队），`RTF = elapsed / audio`，越小越快。

可通过 `--model_bundle`、`--frontend`、`--voice_store`、`--web_file` 覆盖路径；
支持 `MODEL_BUNDLE`、`ITTS25_FRONTEND`、`MIT2_VOICE_STORE`、`HOST`、`PORT`、
`MIT2_WEBKEY`、`JOBS`、`BUILD_DIR` 环境变量。脚本中的相对路径以项目根目录为基准。

## 项目内容

| 路径 | 内容 |
| --- | --- |
| `src/`、`include/`、`vendor/` | 完整原生源码与随附第三方源码 |
| `bundles/full/` | 2.5神经模型包，约8.44GiB |
| `bundles/frontend/` | Whisper/Qwen词表、FST、UniDic、MeCab及DSP常量 |
| `web/index.html` | 音色管理、克隆和合成页面 |
| `voices/` | 音色SQLite数据库和可移植音色文件 |
| `examples/voice_01.wav` | 官方示例参考音频 |
| `docs/` | 接口说明、来源及验收记录 |

模型包、音色、示例音频、编译产物和生成音频放在本地，已从Git中忽略。
复制整个项目时须包含 `bundles/`；仅克隆Git源码不会包含模型包。
源码和模型来源见 `SOURCE.md`；许可证见 `LICENSE`、`LICENSE_ZH.txt` 和各vendor目录。

```sh
ctest --test-dir build --output-on-failure
shasum -a 256 -c docs/model-packages.sha256
```

默认CTest检查Metal设备与MPS工作区。完整模型数值测试需另行提供golden
fixture，已有10项数值验收记录保存在 `docs/ctest-native-final.txt`。
原生同种子可复现，随机序列与PyTorch不同；参考音频截取前15秒。
API参数见 [docs/API.md](docs/API.md)。
