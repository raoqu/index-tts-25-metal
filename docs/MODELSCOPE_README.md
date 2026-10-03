---
license: other
tags:
- IndexTTS
- Metal
- Apple-Silicon
- FP32
tasks:
- text-to-speech
---

# IndexTTS 2.5 Native Metal 模型资源

供 [index-tts-25-metal](https://github.com/raoqu/index-tts-25-metal) 原生
macOS C++17/Objective-C++、Metal/MPS 运行时使用的完整 FP32 模型包、文本前端
和官方示例参考音频。模型未量化，原生推理使用 25 步 CFM。
模型包使用 MIT2 manifest 与 mmap 权重格式。

## 文件布局

| 路径 | 内容 | 大小 |
| --- | --- | --- |
| `bundles/full/` | IndexTTS 2.5 完整神经模型、manifest 和配置 | 约 8.44 GiB |
| `bundles/frontend/` | DSP 常量、Whisper/Qwen 词表、中英文 FST、MeCab、UniDic 及许可 | 约 265 MiB |
| `examples/voice_01.wav` | 官方示例参考音频 | 约 467 KiB |
| `docs/model-packages.sha256` | 32 个模型及前端文件的 SHA-256 清单 | — |

## 自动下载与运行

在原生项目中执行：

```sh
brew install cmake pcre2 libsndfile libsoxr
./dev.sh
```

默认打开 `http://127.0.0.1:3456/web`。缺失的默认模型、前端和示例音频会从
此仓库自动下载，支持断点续传与下载后 SHA-256 校验。运行和下载不需要 Python。
项目的 `./dev.sh --prepare-only` 或 `build/itts25-native --download-only` 可以
仅准备资源；`--no-download` 可以要求使用完整的本地资源。

也可以通过 ModelScope SDK 下载整个模型仓库：

```python
from modelscope import snapshot_download
model_dir = snapshot_download('iwannaido/index-tts25-metal')
```

使用自定义资源路径时，将模型与前端目录分别传给原生运行时：

```sh
build/itts25-native --model_bundle /path/to/model_dir/bundles/full \
  --frontend /path/to/model_dir/bundles/frontend
```

## 来源与许可

这是供原生运行时使用的转换模型资源。原模型与源码来源见 `SOURCE.md`。
模型许可见 `LICENSE`（bilibili Model Use License Agreement）与 `LICENSE_ZH.txt`；
前端依赖的许可保留在对应目录中。

Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.
