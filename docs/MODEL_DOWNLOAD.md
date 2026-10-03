# 模型自动下载

模型仓库：[iwannaido/index-tts25-metal](https://modelscope.cn/models/iwannaido/index-tts25-metal)。
下载方式参考 `~/mylab/index-tts2-metal` 的原生启动检查与 libcurl 下载流程。
构建不访问模型仓库，运行与下载均不需要 Python、ModelScope SDK 或 Git LFS。

## 首次运行

```sh
./dev.sh                       # 构建、准备缺失资源、启动 Web 服务
./dev.sh --prepare-only         # 构建并准备资源，然后退出
./dev.sh --no-build --prepare-only
build/itts25-native --download-only
```

资源默认写入可执行文件所在项目的目录，脚本可从任意工作目录调用。

| 默认目录 | 内容 | 大小 |
| --- | --- | --- |
| `bundles/full/` | 完整 IndexTTS 2.5 FP32 权重、manifest、配置 | 约 8.44 GiB |
| `bundles/frontend/` | DSP 常量、Whisper/Qwen 词表、中文/英文 FST、MeCab、UniDic 和许可文件 | 约 265 MiB |
| `examples/voice_01.wav` | 官方示例参考音频 | 约 467 KiB |

共 33 个资源文件。Web 脚本启动及 `--download-only` 会准备示例音频；
直接运行二进制时，`--seed_example` 会准备示例音频。
用户音色、音色数据库与生成音频保存在本地。

## 中断、校验与离线运行

下载时显示文件名、进度与已接收大小。临时数据保存在目标文件旁的 `.part`
文件中；网络中断或退出后重新运行同一命令即可续传。服务端不支持 Range
时自动从头下载。下载完成后必须通过长度及 SHA-256 校验，才会替换正式文件。
同一目标目录的多个进程通过 `.download.lock` 文件锁串行补齐资源。

启动时按完整文件清单核对长度，已准备的资源无需网络访问。不会每次启动
重新计算 8.44 GiB 权重的哈希。需要检查本地内容时可以执行：

```sh
shasum -a 256 -c docs/model-packages.sha256
./dev.sh --no-download
build/itts25-native --download-only --no-download
```

`--no-download` 下资源缺失或长度不符会明确报错并退出。`--help` 与无效参数
不会触发下载。下载沿用 libcurl 的代理环境变量与 TLS 证书验证。

## 自定义资源路径

`--model_bundle`、`MODEL_BUNDLE`、`ITTS25_METAL_BUNDLE` 以及 `--frontend`、
`ITTS25_FRONTEND` 可以覆盖资源目录。命令行优先于环境变量。
自动下载只补齐项目默认目录；自定义目录检查必需文件后直接使用，
需自行准备兼容的完整资源。`--example_audio` 指定的自定义参考音频也须自行准备。

```sh
./dev.sh --model_bundle /path/to/full --frontend /path/to/frontend
```

资源文件名、长度和 SHA-256 固定在 `include/itts25/model_files.inc`。
模型与前端哈希来自 `docs/model-packages.sha256`，示例音频哈希为
`e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8`。
仓库资源变更时须同步更新清单。原模型来源及许可见 `SOURCE.md`、`LICENSE`、
`LICENSE_ZH.txt` 和前端资源中随附的许可文件。

## 验证

```sh
ctest --test-dir build -R model_download --output-on-failure
```

本地 HTTP 测试覆盖首次下载、完整资源跳过、离线缺失、续传、截断文件、
已完整 `.part` 的校验与提升、Range 回退、中断恢复、哈希错误、HTTP 错误、
并发下载以及非法路径。测试无需外部网络或真实权重下载。

实际 ModelScope 下载、前端权重续传、远端文件校验及原生合成记录见
[`model-download-validation.json`](model-download-validation.json)。
恢复资源后的同种子合成 WAV 与原本地资源生成的 WAV 字节一致。
ModelScope 模型卡片的源文件保存在 [`MODELSCOPE_README.md`](MODELSCOPE_README.md)。
