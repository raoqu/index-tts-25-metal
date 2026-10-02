# 完全原生运行迁移（2026-10-02）

本轮目标替代旧计划允许保留 Python 前处理/UI 的边界：默认 CLI、HTTP、
音色提取、文本前端、情绪与整条生成管线都在 C++/Objective-C++ 中运行。
Python 仅用于离线转换模型/语法资源和独立数值验证，不用于启动或请求处理。

## 审计

起点工作区干净。仓库和参考 `~/mylab/index-tts2-metal/` 均没有 `.codegraph/`。
当前神经模型已原生化，但 `dev.sh` 加载 Python FastAPI、PyTorch、librosa、
transformers；adapter 继承 Python `infer_generator`，通过临时 bundle 与原生
进程逐算子通信。音频重采样、两种 fbank、mel、文本归一化、Whisper/tiktoken、
分句、Qwen tokenizer、情绪矩阵和 SQLite/HTTP 均有运行时 Python 依赖。

参考项目可借鉴原生 HTTP/SQLite、音频输入和 FST 文本归一化。
其 SentencePiece、2.0 模型与特征不能替代 2.5 的 Whisper tokenizer、
192 维 CAMPPlus 条件、EnhancedCodec、107 语言嵌入和已验证模型。

## 分步骤实施与提交

1. 审计并记录新的完成标准（本文）。
2. 原生 2.5 文本编码、音频 DSP；与安装的 Python 库和官方资源对齐。
3. 原生音色缓存、分句、音频/向量/文字情绪、完整 CLI；保留 25 步 CFM。
4. 原生 HTTP/SQLite/UI 与启动脚本；保留原音色数据，错误后可继续服务。
5. 独立构建、数值回归、多语种/长文/情绪/API 和无 Python 运行验收。

每个完成步骤独立提交。失败及剩余项如实记录，未通过整体验收前不标记完成。

## 步骤 2：原生分词与音频 DSP

新增 `itts25-frontend`，以 PCRE2 Unicode 正则 + byte BPE 实现原始 Whisper
60509 词表和 Qwen tokenizer。Qwen 使用 Foundation NFC（验证发现组合重音
必须先归一化）。离线 `export_frontend.py` 只导出词表/特殊 token、语言映射、
滤波矩阵、窗函数及原推理入口的 torchaudio 重采样核，不改模型权重。

音频读取使用 libsndfile，默认加载重采样使用 libsoxr HQ（与 librosa 默认一致）；
22.05→16k 使用原 torchaudio sinc 核。两种 Kaldi fbank 保留 DC 移除、0.97
预加重、Povey 窗及原始归一化；W2V ddof=1、双帧拼接及奇数末帧 mask；
mel 保留 1024 FFT、256 hop、reflect padding、Slaney 滤波器和 log 压缩。

Release 构建成功。独立验证 20 个多语种/特殊符号/Unicode token 用例完全一致。
官方短参考音频验证：W2V 最大误差 0.00135、RMSE 1.08e-5；CAMP 最大
0.000622、RMSE 1.07e-5；mel 最大 0.000279、RMSE 4.19e-6。阈值固定为
max_abs<0.002、RMSE<0.0002；并检查 shape、mask 和文本 ID 精确相等。
报告：`native-frontend-validation.json`。该证据不替代整条推理或长参考验收。

构建需要 CMake、PCRE2、libsndfile、libsoxr；这些是原生库，无 Python 依赖。
资源再生成：`.venv/bin/python backends/metal25/tools/export_frontend.py`。
验证：`TOKENIZERS_PARALLELISM=false .venv/bin/python backends/metal25/tools/validate_frontend.py`。
归一化/FST、分句、情绪、原生 HTTP 和默认启动切换仍在后续步骤。

## 步骤 3：归一化与整条原生管线

使用参考项目的 OpenFST/kaldifst 核心与文本 token parser，保留 2.5 的
技术词、拼音、人名、词汇表、发音标注保护；最终编码仍为 Whisper。
日语使用离线导出的 UniDic 和原生 MeCab C API，保持 g2p_ratio=0 的空格分词。
西语 NeMo 在当前 Python 环境本身不可用；原生保持相同的大写透传行为。
12 个多语种/数字/发音标注/长文本用例的处理文本和分句完全一致，见
`native-text-validation.json`。

`itts25-native` 在同一进程执行音频解码、特征、音色缓存、情绪与全部模型。
直接音频克隆与旧 `voice_demo.pt` 均已产生正常非静音 22050Hz PCM16 WAV。
首个“你好世界”生成 46 个 code、正常 EOS，416 次 GPU 提交，1.834 秒音频。
Qwen 使用 checkpoint 的真实 `System:/Human:/Assistant:` 模板，而非通用 Qwen3
ChatML；原生“开心而自然”返回高兴 0.95、惊讶 0.02、自然 0.03。

原生管线保持 2.5 CAMPPlus/情绪混合、EnhancedCodec、1.72长度系数、25步
CFG 0.7 CFM、BigVGAN、句间静音与20ms尾部升余弦淡出。
随机采样与高斯噪声改用 C++ RNG，同种子原生内可复现，不承诺与 PyTorch RNG
逐样本相同。模型参数未压缩、未更换2.0权重、未减少扩散步数。

## 步骤 4：原生服务与默认入口

原生 cpp-httplib HTTP、SQLite、串行有界队列和音色 LRU 已接入同一 C++ 进程。
继续使用原 `voices.sqlite3` 表结构和 MIT2VOIC 格式，旧 `voice_demo`/`qin`
记录及锁定状态读回成功。没有启动 Python/FastAPI 或逐算子子进程。
`dev.sh`/`start-local.sh` 默认构建并 exec 原生服务，`--cli` 为原生合成入口；
`--help` 和正常启动不导入 Python。缺少权重/前端资源时明确给出一次性离线转换
命令，不自动初始化 Python 环境。

复用原有真实 HTTP 验收工具通过：健康/网页、真实音频克隆、portable voice、
预听、锁定删除保护、保存 WAV、真实 GPU 提交、客户端上传缓存、同种子 WAV
字节一致及临时记录清理。报告 `native-http-validation.json`。
正式7860目前运行原生服务；独立构建与更多语言/情绪/错误/重启验收仍待完成。

## 步骤 5：最终验收完成

已在 `PATH=/usr/bin:/bin:/usr/sbin:/sbin` 下完成独立 Release 从零编译，
包括OpenFST/kaldifst、前端和主程序；构建不寻找Python、不下载源码。
CLI从 `/tmp` 执行独立二进制，显式指定模型和前端资源，完成文字情绪及
语速0.9的真实生成。默认7860服务也已重启为最终C++版本。

| 验收 | 结果与证据 |
| --- | --- |
| 2.5神经组件/GPU数值回归 | 10/10 CTest，`ctest-native-final.txt` |
| Whisper/Qwen分词、音频DSP | 20个token用例精确一致，特征误差达标，`native-frontend-validation.json` |
| 文本规范化、日语、分句、发音标注 | 13个用例精确一致，含30个不同发音标注，`native-text-validation.json` |
| 完整真实生成 | 12组，五语言、快慢语速、向量/音频/文字情绪、三段长文及beam，`native-runtime-validation.json` |
| 原管理接口兼容性 | 实际克隆/预听/锁定/生成/客户端缓存，`native-http-validation.json` |
| 协议与持久化 | 10类，鉴权/别名/LRU/校验/FIFO/429/恢复/重启/文件归属/API-only，`native-protocol-validation.json` |
| AAC/M4A输入 | AudioToolbox原生克隆及生成通过，暖重复WAV字节相同、情感缓存命中，`native-aac-validation.json` |
| CLI独立运行 | 无Python环境路径、仓库外工作目录，正常EOS及480次GPU提交，`native-cli-validation.json` |
| 浏览器真实操作 | qin、文字情绪生成3.228秒音频，播放器就绪、浏览器日志为空，`native-ui-validation.json` |
| 从零构建与链接 | `native-clean-build.txt`、`native-linkage.txt`；不链接libpython/libtorch |
| 最终服务进程 | 单C++进程、无子进程/已映射Python或Torch、旧两个音色与锁定状态保留，`native-process-validation.json` |

AAC暖重复生成1.834秒音频耗时1.450秒，RTF0.791；该值针对本机短参考，
不包括首次加载，不作为所有输入的速度保证。队列/旧音色数据及锁定状态保持。
模型组件的数值对齐、前端对照和端到端测试均通过；原生RNG与PyTorch序列
不同，因此不宣称不同后端随机生成的WAV逐样本一致，也没有宣称人工听感已验收。

交付范围是本机macOS/Metal原生运行。部署需二进制、原生动态库、转换后的
模型/前端资源及网页/音色库；不需要Python源码、虚拟环境或原始checkpoint。
西语NeMo缺失时的大写透传、日语g2p_ratio=0及容量限制详见根目录 `LOCAL_RUN.md`。

实现提交：`0e63c46`（审计）、`d168ba5`（分词/DSP）、`34c992d`（完整管线与服务）、
`8103236`（AAC、缓存、兼容性及验证工具）；最终文档和验收证据另行提交。
