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
