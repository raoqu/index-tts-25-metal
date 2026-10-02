# 实现选项与决策

## D01：运行时选择

选项 A：复用参考项目 C++/Objective-C++ Metal 与 MPS 算子。
选项 B：引入 MLX C++ 并重写模型。

推荐并采用 A。参考项目已包含权重常驻、矩阵乘法、卷积、注意力、归一化等
可复用基础，且本机已验证 Metal 可用。它能减少新增依赖，并符合 C++ Metal
后端要求。只复制基础源码到独立目录，构建不依赖参考项目的绝对路径。
保留将高层算子替换为 MLX 的后续空间，当前不同时维护两个推理实现。

## D02：模型版本与精度

选项 A：在 2.0 原生管线中套用 2.5 文件。
选项 B：按真实 2.5 权重形状、模块实现及 golden 移植。

推荐并采用 B。2.5 的 codec 编解码器、GPT 前缀和语言嵌入存在实质变化，
直接复用 2.0 模型管线会产生错误。初始权重/累加为 FP32，数值验收后再优化精度。

## D03：UI 边界

选项 A：用参考项目内置网页替换 Gradio。
选项 B：保留当前 UI，Python adapter 调用持久 C++ 推理进程。

推荐并采用 B，符合“保持 UI 界面”。Python 只承担 UI/文本处理和协议，
最终声学模型必须全部在原生进程执行。逐步组件测试期间暂不把混合推理称为原生后端。

## D04：配置和论文描述的差异

论文/模型卡可能使用 Zipformer 描述解码架构；本次本地 `s2mel.pth` 中实际存在
`net.cfm`、`net.length_regulator`、`net.gpt_layer`，配置指定 `dit_type: DiT`。
采用本地权重与其实际加载源码作为权威依据，审计参数键后决定原生结构，
不根据宣传名称选择不匹配的网络。

## D05：权重转换

采用离线 Python 转换训练 checkpoint 为 mmap bundle；推理二进制不链接 libtorch，
也不启动 Python。只导出模型参数，不导出 optimizer。bundle 使用参考项目的
MIT2 容器，metadata 必须明确 `index-tts2.5`，避免错误加载 2.0 权重。

## D06：禁止默认继承 FP16 优化

GPT 对照发现，参考算子默认把权重和 KV 缓存压缩到 FP16，即使输入 bundle 是
FP32。这导致 logits 误差最高约 4.9e-3，超出初始数值验收阈值。
推荐并采用 FP32 权重和 FP32 KV；使用安全 Metal 数学编译模式。修复后最高误差
3.13e-5、最高 RMSE 9.55e-6，没有通过放宽容差掩盖精度问题。
权重 FP16 改为显式 `ITTS25_FP16_WEIGHTS=1` 选项，当前不作为验收默认值。
KV 暂不提供 FP16 选项，以免静默影响后续 token。语言输入删除被 attention mask
屏蔽的左侧 pad，保留有效行；五种语言前缀对照验证此变换。

## D07：声学注意力精度与模块复用

只移植参考声学实现所需的 13 个 GPU 模型函数及结构，移除旧 TTS、测试 CLI、
文件中转和超过容量的 CPU fallback。新 AcousticModel 负责尺寸检查和 2.5 参数。
参考 DiT SGQ attention 虽然名称含 f32，实际 Q/K/V 和概率经过 half MMA；
使 CFG 误差达到 3.9e-3。采用完整 FP32 attention 后最高误差降为 6.74e-6。
默认保留官方 25 步和 0.7 CFG，暂不采用参考项目的 16 步速度优化。

## D08：W2V-BERT 的逐层与离散 token 验收

原生实现按真实 transformers 源码执行 0..16 共 17 层，取 hidden_states[17]。
新增 FP32 relative_key attention，距离范围为 [-64,8]；卷积为左侧填充的 causal
depthwise convolution。参考项目的部分裸 attention helper 不含距离项，故不能直接复用。

最初暂定的 2e-4 峰值阈值在 51 帧随机输入上失败。逐层检查发现误差随 FP32
矩阵乘法/注意力的累加放大；相同真实权重与输入下，PyTorch MPS 对 CPU 的
误差为 max 1.735e-3、RMSE 4.44e-5，原生为 max 5.90e-4、RMSE 1.54e-5。
推荐采用针对该深层前端的 max < 1e-3、RMSE < 2e-5，原始逐层 max < 1e-3、
RMSE < 1e-5，并增加三组 codec 离散 token 完全一致的独立门槛。
这是根据设备基线设定组件容差；不会修改 GPT、codec、CFM、BigVGAN 的较严阈值。

## D09：CAMPPlus 的原生执行边界

推荐保留参考 CAMPPlus 图的原生 C++ 实现，替换全部 1D/2D 卷积与最终 dense
投影为 Metal；小型 batchnorm、池化、sigmoid 和布局处理暂保留 C++。
这样全部学习参数均由原生后端加载和执行，最重的卷积在 GPU，推理不调用 Python。
后续如分析表明小算子造成延迟，再合并为 GPU pass；不为减少这些算子而改模型结构。

## D10：生成与音频预处理边界

推荐保留现有分词及 librosa/torchaudio 的非神经 DSP，Python UI 只传输 fbank、mel、
文本 ID 和调度参数。全部神经模型在 C++ Metal，避免重写 DSP 引入新的音质变量。
这细化了初始计划中的 DSP 迁移边界；推理二进制仍不依赖 Python/PyTorch。

GPT 支持官方默认三路 beam、重复惩罚、温度、top-k/top-p。缓存分支仅复制已使用
的 FP32 KV 前缀，初始三路共享快照；后续可进一步优化为 GPU 分页。严格对照官方
inference_speech 的八个贪心和三路 beam token 均完全一致。官方缓存生成的第一个
语义 code 使用 mel position 2；本实现遵循该行为。随机生成采用 SplitMix64，
CFM CLI 噪声采用原生正态分布，固定种子可重复，但不声称跨框架随机序列一致。

## D11：文字情绪完整迁移和数值门槛

推荐按本地 Qwen3 0.6B checkpoint 实现原生 28 层、16 个 Q head、8 个 KV head，
包括逐 head RMSNorm、half-rotation RoPE（theta=1e6）、GQA、SwiGLU 和共享词表投影。
使用独立 Metal context/KV，避免文字情绪与语音 GPT 缓存互相覆盖。

最初套用 GPT 的 1e-4/2e-5 logits 门槛失败。与同一 CPU oracle 对照，官方
PyTorch MPS FP32 峰值 3.47e-4/RMSE 7.29e-5；原生峰值 4.19e-4/RMSE 7.91e-5。
两者都是 28 层及 151936 词表的浮点累积误差。推荐本组件 max < 5e-4、
RMSE < 1e-4，同时必须八组 argmax 全相等、两段完整 77-token JSON 全相等。
这是基于实测设备基线的专用门槛，不放宽任何语音模型阈值。

推荐文字情绪最多 3072 输入、1024 生成 token，4096 KV 容量。常见分类 JSON
约 77 token；若未结束，明确报错，避免官方 max_new_tokens=32768 在异常
文本上长期占用本机。正常描述及原有情绪规则、低落词处理均保持。

## D12：默认入口、回归与容量

推荐启动脚本及本地 CLI 默认 Metal，同时保留显式 --backend pytorch 回退。
Gradio 组件、参数默认值、语言/情绪/语速入口不改；通过原有 callback 实际发起
生成请求。组件 id/类型/标签/可见性/choices/上下限对照：119 组件、22 接口完全相同。

回归采用一条 beam 控制测试时长；真实 UI 请求保持三条 beam。检查同种子重复
WAV 的 SHA 完全一致、0.8/1.2时长系数及多语种/三类情绪/多句合成。用 Python
Module 拦截证明没有 Python 神经网络推理，唯一允许项是 Resample 非神经DSP。

容量推荐沿用原有分句而非扩大所有内核工作区。当前声学/声码器最多4096mel帧，
GPT KV4096、参考W2V1600帧；超长单段明确报错。与上游8192声学缓存上限相比
存在该边界，已在运行说明公开；保持默认120-token分句可覆盖常见长文本。
西班牙文可选 NeMo 规范化依赖缺失时保留上游回退，简单文本合成已验证。
不在本次迁移中安装额外 ASR 模型，质量证据以严格组件对齐及自动音频回归为准，
不声称人工听感或识别率已经验收。

## D13：按新要求复用参考网页和 HTTP API

本次用户明确将此前“保留 Gradio”改为复用 index-tts2-metal 网页 UI 和 API 设计。
推荐直接复制参考 web_index_html，保留 Status/Voices/预听/克隆/导入/锁定/测试流程，
并实现 /web/api/*、/api/*、/v1/audio/* 的原有路径和 JSON/音频约定。

选项A：复制完整2.0 C++ HTTP server，同时移植其2.0 clone/product入口。
选项B：复用网页源码和API契约，独立HTTP/SQLite调度层连接已验收的2.5原生进程。
采用B，避免将2.0模型入口和2.5权重混用。HTTP、SQLite和音频DSP使用Python，
全部神经模型继续由C++ Metal执行；工作队列串行保持Metal/KV稳定。

复用MIT2单文件容器及MIT2VOIC footer，音色metadata显式target=index-tts2.5，
保存真实原生参考特征和预听波形。上传和导入均验证形状/哈希/版本，不加载pickle。
2.0音色特征不兼容2.5，明确报错并建议用原参考音频重新克隆。

dev.sh推荐默认保留现有7860端口，/重定向到/web；--port可用3456等端口。
一条命令负责依赖检查、增量构建、必要的权重转换及前台启动；Ctrl+C关闭原生子进程。
原Gradio入口仍可显式运行，但start-local.sh将转到新管理页面。

## D14：在 FP32 精度下补齐性能优化

用户反馈 RTF 14.55 后，固定声音、文本“你好世界”、seed=424242、num_beams=1、
duration_factor=1、CFM 25步/CFG0.7测量阶段耗时。两种完整参考音频分别2.4387秒
和7.3041秒。基线暖请求 RTF 分别8.7377和14.1518。此前版本只完成原生迁移和
数值对齐，默认FP32使原项目的部分FP16快速路径未启用，GPT单token仍走MPS GEMM。

选项A：全局切换FP16并缩短参考音频/减少CFM步数。
选项B：保留FP32权重、FP32 KV、完整参考音频和原采样参数，优化实际算子与缓存。
推荐并采用B，避免通过改变生成条件获得不等价的性能数字。

- GPT单token复用参考的并行FP32 GEMV；多token仍用原有矩阵乘法。
- DiT参考分块注意力新增FP32 MMA变体，16个key一块使共享存储低于32KiB。
- 沿参考single-pass设计，整条CFM轨迹在GPU内完成，条件投影每请求一次；
  时间embedding按步数缓存，遵守原2.5的i/steps及next-t，不改为累加时间。
- BigVGAN扩张卷积及转置卷积使用FP32 MPS tap GEMM，保留边缘裁剪、stride和bias。
  原始卷积权重与tap均常驻，支持同一进程先短后长、先长后短时切换算子。
- 抗混叠Snake分为2x上采样激活和下采样，两者使用相同公式；中间缓冲区反复复用。
- 情绪embedding以输入FP32字节及形状为键，最多8条缓存；返回副本防止调用方修改。
  时间表也最多8组。不同声音、参考音频、插值alpha仍遵循原始融合数学。
- HTTP合成JSON新增stage_seconds及emotion_cache_hits，便于定位后续瓶颈。

保留基线二进制和逐阶段输入，最终要求所有既有数值门槛不变、GPT codes完全相同、
真实输入CFM/vocoder误差通过，固定种子重复WAV完全一致。数值对齐和自动回归不替代
人工听感验收。最终性能数字及复现方式记录在PERFORMANCE.md。

D14补充：真实模型交错执行时，旧MPS零bias路径稳定出现非有限值；显式广播bias
（包括全零bias）并用beta=1累加后，同一GPT/CFM切换序列通过。推荐总是初始化C，
不依赖已用工作区的内容或零bias快速路径。新增硬件回归mps_zero_bias_reuse验证
被NaN填满的工作区可产生精确输出；独立identity GEMM旧版也能通过，因此不把
0*NaN作为已独立证实的MPS底层原因。真实模型交错回放负责覆盖实际失败场景。
