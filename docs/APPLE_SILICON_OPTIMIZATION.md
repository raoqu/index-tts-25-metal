# Apple Silicon 逐项优化记录

设备：M1 Max（32 GPU cores），64 GiB；起点 `2c7a9a6`。

保留条件：排除模型构造/加载，固定模型 FP32、FP32 KV、25 步 CFM、CFG0.7、完整参考音频和种子。整体 TTS 暖请求中位数改善至少 0.5%，且适用场景无有意义的性能退化。边界附近采用独立进程交错 A/B 复测，不凭单次结果保留。专用优化同时测其适用请求（beam、文字情绪、音频情绪等），不以加载时间或仅子算子加速代替整体 TTS 收益。

质量门槛：文本 IDs、语义 codes、帧数完全相同；PCM16 max_abs < 1e-4、RMSE < 1e-5；已有组件 golden 门槛保持不变；固定种子重复稳定。数值测试不代替人工听感。

内存：记录每次请求 RSS / physical footprint、Metal 分配次数及累计分配字节。交错声音和长度、重复请求检查缓存有界、工作区复用和稳定后内存增长；大改动再做销毁/重建及系统 leaks 检查。

每项实验先独立提交，结果低于门槛、质量失败或风险未解决则另行提交回退；验证工具和证据独立保留用于追溯。

顺序：
1. 细分计时、独立持久进程基准及 golden 基线。
2. CFM 时间调制投影缓存。
3. CFM 内核融合 / 分块与 GEMM 选择。
4. 权重重复存储与分配。
5. Codec 连续 GPU pass。
6. GPT 解码调度、beam KV。
7. Qwen GPU embedding / argmax 和文本分类缓存。
8. Metal 冷启动编译 / 对象共享（按排除加载后的门槛决定）。
9. Accelerate DSP、克隆/音频情绪路径。
10. 按设备/形状选择算子与存储模式。

复现：`itts25-benchmark MODEL FRONTEND CASES_JSON OUTPUT_DIR CYCLES WARMUPS` 输出 JSONL；`tools/compare_performance.py BASELINE_JSONL CANDIDATE_JSONL` 对照耗时、阶段、tokens、WAV 和内存。benchmark 不参与默认服务。

基线已有 10 项 CTest 全通过（26.83s）。附加 `tools/run_performance.py` 以 ABBA 顺序独立进程复测，避免同时推理。默认用例保存在 `tools/performance_cases.json`。

## 02：CFM 时间调制缓存（保留）

实现 `d58cc9e`。先验 A/B 整体改善 1.17%；独立 ABBA 两组均超过门槛（15.92% / 2.21%，合并 9.65%），波动较大，不把 9.65% 解释为纯缓存算子的稳定加速比。所有语义 codes 和 PCM 字节一致；acoustic golden 通过。暖请求主 Metal 分配次数为零。缓存与模型实例同寿命，最多8组；默认25步每组约2.64MiB，新增轨迹上传同等大小。阶段与内存详情见 `apple-silicon/02-cfm-modulation.json`。

## 03：FP32 attention 32-key staging（保留，最终复核）

实现 `d8b241f`。复用已经加载到寄存器的 Q 共享存储作为 O epilogue，保持低于32KiB和原16-key softmax计算顺序。ABBA整体改善2.21%（单组0.24%/5.42%，仍有系统波动），三个场景全部改善且 PCM 字节一致，acoustic golden通过。最终基准复核仍包含此项。见 `apple-silicon/03-attention-tiles.json`。

## 04：512通道 RMSNorm 线程数（待最终复核）

实现 `a9ad3c0`。原1024线程上半部分只加零，改512线程保持同一有效加法树。ABBA两组整体改善0.508%/0.608%，处于门槛边缘，最终再复测才决定是否保留。三个场景 PCM 字节一致，acoustic golden通过。见 `apple-silicon/04-rms-threads.json`。

## 05：释放GPT权重CPU副本（回退）

实验 `018ad5b`，GPT golden通过、PCM字节相同。physical footprint明显下降，但ABBA整体改善 -0.53%，单组 -0.60% / +0.75%，未满足整体TTS至少0.5%的保留门槛。按用户要求回退，即使内存下降也不单独例外保留。见 `apple-silicon/05-weight-copies.json`。
