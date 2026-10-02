# 迁移进度

## 基线与审计

- 工作目录：`/Users/raoqu/research/itts25`；上游 commit `d9e41aa`。
- 基线提交 `f6394b9`，保留原始可运行 UI、MPS CLI 与 2.5 模型配置。
- 参考项目 commit `eda855b2e9d7cfaa4269380f42880d41b060c4d2`。
- 当前仓库及参考项目没有 `.codegraph/`，不执行索引。
- 真实 GPT 有 456 个参数/缓冲区；首层说话人投影为 `[1280,192]`。
- codec 有 243 个参数/缓冲区，包含 stride=2 下采样、12 层 Vocos 编解码器。
- S2Mel checkpoint 外层为 `net`，需按各子模块展开而非当成普通 state_dict。

## 原生基础与 EnhancedCodec 验收

- 独立基础提交：`0570ea5`，仅复制参考项目的 Metal/MPS 算子和 bundle reader。
- CMake Release build 成功；设备为 Apple M1 Max，统一内存可用。
- 转换 codec 得到 245 个 FP32 张量、202,398,784 字节；保留原 g/v 参数并计算
  两个 FVQ weight_norm 投影的推理权重。训练 optimizer 没有导出。
- C++ 实现了下采样、12 层 encoder/decoder Vocos、归一化 FVQ、最近邻上采样。
- 独立 CPU PyTorch golden 严格加载全部 checkpoint 参数，覆盖解码长度 1/2/7/31
  和编码长度 1/2/3/17/62，包含奇数长度的 ceil(T/2) 下采样。
- 9 个用例通过；所有编码 token 完全一致。解码最大绝对误差 6.92e-6，
  投影最大绝对误差 4.77e-7。验收阈值 max_abs < 1e-4、RMSE < 1e-5。
- 执行 838 次真实 GPU command buffer。
- `otool -L` 确认仅依赖 macOS 框架、libc++ 与系统运行库，无 libtorch/libpython。
- 首次测试发现 golden 投影被多转置一次；修正 oracle 的 BTC 布局，并增加
  shape 断言后重新生成 golden。没有放宽容差或修改模型计算掩盖失败。
- 固定命令和再生成步骤见后端 README；逐项结果保存在 `docs/codec-validation.jsonl`。

## GPT 原生解码验收

- 完整 FP32 bundle 导出 3,482 个张量、6,675,607,260 字节，无 optimizer。
- 新建原生 24 层 GPT decoder，GPU 常驻 KV，prefill 和每个 decode step 单次提交。
- 严格加载真实 GPT checkpoint 的 CPU oracle 覆盖中文、英文、日文、西班牙文、
  阿拉伯文前缀，以及一次 prefill 和四次 teacher-forced KV 解码。
- 五种前缀最大误差 1.44e-6；五次 logits 最大误差 3.13e-5、最大 RMSE 9.55e-6，
  全部 argmax token 一致。验收阈值 max_abs < 1e-4、RMSE < 2e-5。
- 发现并修复参考 runtime 默认 FP16 权重/KV 的隐藏压缩；详见决策 D06。
- 随机采样、音色情绪编码、声学管线、声码器、UI 接入和完整音频验收尚未完成。
迁移目标保持 active；组件测试不会替代这些完成标准。

## S2Mel 与 BigVGAN 原生验收

- 移植独立长度调节器、13 层 DiT + 8 层 WaveNet 的 CFM estimator、25 步 Euler
  和 6 级上采样 BigVGAN。复用参考 GPU 算子，加载真实 2.5 参数。
- 严格 CPU oracle 验证三组变长调节、CFG 双分支 estimator、25 步完整 CFM
  及三组 BigVGAN 波形。最高误差分别为 7.64e-8、6.74e-6、2.65e-5、1.63e-5。
- 默认阈值 max_abs < 1e-4、RMSE < 1e-5，执行 307 次 GPU 提交。
- 同时发现参考 SGQ DiT attention 使用 half MMA；FP32 默认路径改用纯浮点注意力。
  优化内核只在明确开启 FP16 时启用。没有放宽容差。
- 这些测试是组件验收；完整原生音频、音色提取和 UI 接入仍需完成。

## 原生情绪音频编码验收

- 真实 2.5 emo_conditioning_encoder 的 4 层 Conformer、4 层 Perceiver、
  emovec_layer 和 emo_layer 已在原生 Metal 路径执行。
- 严格加载官方权重的 get_emovec CPU oracle，覆盖长度 3/7/17/50。
- 四组均通过；最高误差 4.71e-6、最高 RMSE 8.37e-7，GPU 提交 484 次。
- 推荐复用经验证的 Conformer/Perceiver GPU 函数；不引入参考项目的 speaker
  Conformer，2.5 的 speaker 条件使用 CAMPPlus 192 维特征。

## 原生 W2V-BERT 与 CAMPPlus 验收

- 实现 17 层 W2V-BERT，包括 73 行距离嵌入的 relative_key attention、causal
  depthwise convolution、两个半残差 FFN 和 checkpoint 统计归一化。
- 三组含 mask/无 mask 用例通过，且经过 EnhancedCodec 后的 code IDs 全部一致。
  51 帧逐层记录和现有 PyTorch MPS 对照说明见 D08；组件特定容差已明确记录。
- CAMPPlus 保留官方 ResNet/TDNN/CAM 图，全部卷积与 dense 投影在 Metal，
  batchnorm/池化等小操作在原生 C++，不调用 Python 模型。
- 三组长度 11/47/105 包含非整除池化宽度；最高误差 1.25e-5、RMSE 4.70e-6，
  共 675 次卷积 GPU 提交，使用与此前组件相同的严格默认阈值。
- 待完成：GPT 采样/beam、完整原生 WAV、持久进程/UI、可选文字情绪模型和
  多语种/语速/情绪端到端验收。完整迁移目标继续 active。

## 原生生成与完整中文音频

- GPT 贪心与三路 beam 的八个实际生成 token 与官方 inference_speech 完全一致。
- 原生管线产生 75 个语义 token 并正常 EOS；22050 Hz 单声道 PCM16，2.995 秒，
  总耗时 28.79 秒，1268 次 GPU 提交。阶段时间及文件检查见 native-synthesis-zh.json。
- WAV 所有值有限，RMS 0.172，单个触及满幅样本占比 0.0015%；尚未进行听感验收。
- 待完成：持久原生服务、保持 UI 的 adapter、Qwen 文字情绪原生移植，以及多语种、
  语速、情绪和分句回归。完整目标继续 active。

## 持久服务与原有推理流程

- 新增 --serve，串行执行同一进程中的模型，隔离临时输入并返回原生耗时/GPU计数。
- adapter 复用官方 infer_generator，保留分句、音色/情绪缓存、语速及音频拼接。
- 三路 beam 默认流程生成 2.914 秒中文 WAV，41.57 秒（首请求）；所有模型操作
  均返回 GPU 提交证据。测试拦截 Python nn.Module 调用，仅允许 Resample DSP，
  神经网络 Python 调用为零。详细记录见 adapter-smoke.json。
- 已提供 --backend metal 选择，默认暂未切换；Qwen 和 UI 端到端仍待验收。

## Qwen3 文字情绪原生验收

- 310 个 Qwen 张量已导出，完整 bundle 共 3792 张量、9,059,806,940 字节。
- 28 层与独立 FP32 GPU KV 实现完成；两种真实情绪文本各检查 prefill、
  三次缓存解码及完整贪心 JSON。所有 argmax/完整 token 与 CPU oracle 完全一致。
- 数值门槛及 MPS 基线详见 D11；逐项报告见 qwen-validation.jsonl。
- NativeQwenEmotion 保留官方 tokenization、JSON 解析、情绪转换和低落规则；
  不实例化 AutoModelForCausalLM。服务首次调用时加载 Qwen，后续常驻。

## 完整回归与默认切换

- 12组集成回归通过：ZH/EN/JA/ES/AR、重复、两种语速、情绪音频/向量/文字、
  三段英文。同种子重复WAV SHA完全一致；0.8/1.2系数对应1.846/2.775秒，
  原始2.310秒。所有音频为22050Hz单声道PCM16、有限且非静音。
- 拦截验证神经网络Python调用为零；错误请求后服务仍可继续；模型及音色缓存
  在同一原生进程反复使用。详细各阶段GPU计数见 integration-validation.json。
- 内存抽样：原生服务RSS最高观察约23572 MiB；包括Qwen，非总GPU内存指标。
- 9项CTest全通过，包含额外Qwen和采样对照。并行回归期间总耗时88.26秒，
  不将其作为独立性能基准。数值验收结果均保留在对应validation.jsonl。
- 正式start-local.sh及本地CLI默认Metal；webui.py对2.5默认Metal、v2保留PyTorch。
  已实际启动127.0.0.1:7860。保留 --backend pytorch 显式回退。
- UI保留119组件和22接口，正式/测试配置与原PyTorch配置结构一致。
- 自动数值/文件/控制回归完成；没有宣称人工听感、ASR识别率或性能优化已完成。
- 后端边界、非神经DSP、容量和可选西语规范化回退见决策D10–D12。

正式7860端口的文字情绪 Gradio 请求通过：三条beam，2.670秒音频，47.69秒首请求（含首次Qwen加载）。见 ui-final-validation.json。

默认CLI最终验证通过：原始中文测试句，22050Hz单声道PCM16，3.065秒，28.26秒推理/29.1秒含加载。见 cli-final-validation.json。

## 按新要求复用 MTTS 管理页面与 API

- 直接复用参考项目 web_index_html，来源提交和原始哈希记录在 web/PROVENANCE.json。
  保留 Status、Voices、预听、锁定、克隆、路径导入、编辑和测试面板；新增2.5语言、
  时长系数及文字情绪控件。决策D13取代此前默认保留Gradio的界面要求。
- 独立HTTP/SQLite层接入已有C++ Metal进程，没有复制2.0推理入口。
  音色以MIT2VOIC单文件保存真实2.5特征及预听波形，验证版本、维度、SHA256。
  支持原API别名、客户端音色LRU、持久音色库、授权记录和串行有界队列。
- 10项HTTP契约测试通过：CRUD/锁定/预听、数据库重启、LRU、客户端上传、
  授权记录文件归属、网页密钥/API-only模式、输入错误、队列429、推理异常恢复。
- 真实HTTP验收通过：官方音频克隆，预听/锁定，原生合成4.5047秒PCM16 WAV，
  合成返回603次真实GPU提交；下载音色后上传客户端缓存，固定种子生成WAV完全相同。
  临时音色已清理，详情见 http-validation.json。
- 浏览器实际操作Status/Voices/测试面板，文字情绪“开心而自然”、时长系数0.9
  生成2.6935秒WAV，播放器readyState=4，浏览器错误/警告日志为空。
  验收截图与测试音频保存在本次聊天outputs中。自动检查不替代人工听感验收。
- dev.sh默认无参数启动、--prepare-only、--help和--port 7861均已实际验证。
  已适配macOS Bash 3.2的空参数数组；uv sync --extra http --extra webui --frozen通过。
  原Gradio和临时HTTP服务均正常退出，原生子进程已关闭；正式7860管理服务已启动，
  SQLite中的voice_demo及锁定状态在重启后保留。原2.0项目无改动。

## FP32性能优化

- 已补齐GPT单token GEMV、FP32分块DiT attention、CFM GPU驻留调度、BigVGAN
  扩张/转置卷积MPS及抗混叠激活；情绪编码和时间embedding使用有界缓存。
- 核心优化提交e35ec77，共享MPS输出初始化修复提交272d811。
- 10项CTest通过，包括新增零bias/NaN工作区测试；12项HTTP/缓存单元测试通过。
- 12项真实集成回归通过，五语言、两种语速、三种情绪和两段英文分句；
  神经网络Python调用为0，固定种子重复WAV完全一致。
- “你好世界”固定种子、完整参考和25步CFM：暖请求RTF短参考8.738→0.811，
  qin长参考14.152→1.403，分别约10.8/10.1倍；详细原始结果见performance-benchmarks.json。
- 数值回放及最终网页/API结果见PERFORMANCE.md；此处不将冷加载时间计入暖请求。
- 正式dev.sh已恢复7860，真实API同种子qin暖请求RTF1.399、WAV与性能基准完全相同。
  官方示例较长句的API暖请求RTF0.633；网页qin随机请求显示RTF1.37、播放器就绪。
  SQLite原两条声音及锁定状态保持，临时音色清理，队列空闲，失败数0。
- 最终数值回放通过：GPT codes精确一致；CFM/vocoder及PCM16满足原波形
  max_abs<1e-4、RMSE<1e-5；短参考PCM16峰值3量化单位，长参考1量化单位。

## 完全原生启动与推理（2026-10-02）

已去除默认运行路径的 Python/FastAPI/adapter：独立 `itts25-native` 同进程
执行 HTTP/SQLite、Whisper/Qwen分词、FST/UniDic文本前端、原生音频DSP、
音色缓存及全部2.5模型。`dev.sh` 和 `start-local.sh` 默认构建/执行该程序。
权重仍为原2.5 FP32，CFM仍25步；Python只参与一次性资源转换和独立oracle。

20个分词、13个文本/分句用例精确对齐；10项神经/GPU CTest、12组实际生成、
10类HTTP协议验收通过。AAC克隆/合成、同种子一致、情感缓存、原音色/锁定
状态和UI播放器也已验证。无Python环境路径下从零构建及独立CLI运行成功。
详细命令、误差、报告、提交与边界见 [NATIVE_RUNTIME.md](NATIVE_RUNTIME.md)。
