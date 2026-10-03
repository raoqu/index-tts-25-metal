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

## 02：CFM 时间调制缓存（最终回退）

实现 `d58cc9e`。先验 A/B 整体改善 1.17%；独立 ABBA 两组均超过门槛（15.92% / 2.21%，合并 9.65%），波动较大，不把 9.65% 解释为纯缓存算子的稳定加速比。所有语义 codes 和 PCM 字节一致；acoustic golden 通过。暖请求主 Metal 分配次数为零。缓存与模型实例同寿命，最多8组；默认25步每组约2.64MiB，新增轨迹上传同等大小。阶段与内存详情见 `apple-silicon/02-cfm-modulation.json`。

## 03：FP32 attention 32-key staging（最终回退）

实现 `d8b241f`。复用已经加载到寄存器的 Q 共享存储作为 O epilogue，保持低于32KiB和原16-key softmax计算顺序。ABBA整体改善2.21%（单组0.24%/5.42%，仍有系统波动），三个场景全部改善且 PCM 字节一致，acoustic golden通过。最终基准复核仍包含此项。见 `apple-silicon/03-attention-tiles.json`。

## 04：512通道 RMSNorm 线程数（最终回退）

实现 `a9ad3c0`。原1024线程上半部分只加零，改512线程保持同一有效加法树。ABBA两组整体改善0.508%/0.608%，处于门槛边缘，最终再复测才决定是否保留。三个场景 PCM 字节一致，acoustic golden通过。见 `apple-silicon/04-rms-threads.json`。

## 05：释放GPT权重CPU副本（回退）

实验 `018ad5b`，GPT golden通过、PCM字节相同。physical footprint明显下降，但ABBA整体改善 -0.53%，单组 -0.60% / +0.75%，未满足整体TTS至少0.5%的保留门槛。按用户要求回退，即使内存下降也不单独例外保留。见 `apple-silicon/05-weight-copies.json`。

## 06：Codec GPU pass（最终保留）

实现 `7d5deef`。7次暖请求整体改善0.99%；Codec阶段约从80–90ms降为48–62ms，提交次数从92降为3（lookup两次、主体一次）。保持原erf GELU多项式和各算子计算顺序，不用GPT tanh GELU替换。复用主体工作区、gamma常驻，避免每个小算子重新分配上传/读回。9组Codec及W2V语义token golden均通过，整句PCM字节完全相同。见 `apple-silicon/06-codec-pass.json`。

## 07：GPT FP32 ICB 重放

实验 `80414df`。每token重放同一组FP32 GEMV、LayerNorm、GELU、残差和KV attention；保持原CPU采样/RNG与显式mel position。ICB使用独立固定8MiB工作区，KV布局变更自动失效，采样路径不读取未写入的history。结果见 `apple-silicon/07-gpt-icb.json`，GPT golden日志见 `07-ctest-gpt.txt`。

07结论：初版ICB整体 -1.04%；去重并批量声明GPU资源（`ea7dae1`）后整体仅 +0.188%，三个场景均未达到0.5%。质量全通过，但按门槛将两次实验实现均回退。未采用改FP16或更改采样作为补救。

## 08：Beam KV快照（保留）

实验 `e50779e` 的全GPU版本短句退化2.89%、长句改善19.86%；调整 `e178423` 为128个cached tokens以下沿用CPU拷贝，以上GPU blit。混合版本三用例整体改善8.89%，长句11秒左右降到9.38秒（13.51%），短句+0.44%/+0.47%基本持平。GPT golden通过，beam语义codes与PCM字节完全相同。长句footprint约14.19GB→12.73GB；快照池最多24项，按容量替换闲置小buffer，每次生成退出时清空（含异常退出）。新增分配9次/长句是请求内池化的代价，避免保留最大请求的快照内存。见 `apple-silicon/08-beam-kv.json`。

## 09：Qwen GPU embedding / argmax（保留）

实现 `52be7e8`。embedding不再单独提交/读回/重新上传；生成模式把151936 logits的greedy argmax放在同一pass，prefill/step golden接口仍返回完整logits。保持FP32与最小ID平局规则。文字情绪两场景ABBA整体改善1.45%，两轮1.75%/0.85%，语义codes和PCM字节一致，Qwen logits/EOS golden通过。主Metal分配计数不覆盖Qwen自身context，整进程footprint仍记录在结果内。见 `apple-silicon/09-qwen-gpu.json`。

## 10：文字情绪分类缓存（保留）

实现 `ee84f8a`。以完整原始情绪文本为key，仅缓存成功分类的JSON；长度校验在查缓存前，最多16项并随runtime销毁。缓存不包含声音、alpha或随机采样结果。重复文字情绪请求ABBA整体改善41.39%（41.66%/41.09%），PCM字节一致，未缓存的新文本仍走同一Qwen。该收益仅适用于缓存命中的重复情绪文本。见 `apple-silicon/10-emotion-cache.json`；最终验证包含容量淘汰、错误输入及runtime销毁重建。

## 11：Accelerate双精度FFT（最终回退）

实现 `2e5bde1`。固定512/1024点double FFT与setup复用，thread_local工作区最多1024个double的real/imag两组，setup析构释放。完整音频情绪TTS的ABBA改善1.07%，两轮0.90%/1.80%，PCM字节一致。示例参考音频的speech/CAMP/mel特征均逐元素完全一致；没有改resample、窗函数、归一化、裁剪和参考长度。缓存声音的普通请求不走此FFT，因此不宣称有收益。见 `apple-silicon/11-accelerate-fft.json` 和 `11-frontend-parity.json`。

## 12：M1 Max FP32合并卷积GEMM（回退）

实验 `e0ba728`。真实357/790帧交错flow与已有acoustic、MPS bias复用测试通过；flow最大误差4.77e-6，最终PCM在原门槛内。ABBA整体 -0.105%（+0.502%/-0.248%），没有达到0.5%，按门槛回退，不启用这个按设备/形状选择的路径。见 `apple-silicon/12-merged-convolution.json`、`12-real-components.json`。

## 13：剩余CFM时间投影缓存（回退）

实验 `bf13dd2`。额外缓存Wavenet conditioning和末层silu(t1)投影，保持原算子，真实组件和PCM均逐元素一致。ABBA整体改善0.408%，两轮-0.345%/+0.470%，低于门槛，因此回退；不继续保留额外缓存与上传内存。见 `apple-silicon/13-conditioning-cache.json`。

## 14：Metal编译库共享（回退）

实验 `7c8f93a`。跨MetalContext只编译一次库，model/context初始化属于排除范围。暖TTS ABBA -0.946%（+0.263%/-1.865%），输出完全相同；回退。AOT metallib同样只移走首次编译，不会减少暖TTS算子工作，因此不引入额外构建/部署复杂度。见 `apple-silicon/14-shared-library.json`。

## 15：GPU-only权重private存储（回退）

实验 `52140d5`。BigVGAN用于CPU展开tap的基础权重仍shared，其余常驻FP32 buffer通过一次staging blit上传private，临时staging完成后释放。10项CTest全部通过、PCM字节一致。暖TTS ABBA -0.233%（-1.141%/+0.735%），未达门槛，回退。没有更改工作区/KV的CPU可见布局。见 `apple-silicon/15-private-residents.json`。

## 16：RMSNorm最终组合复核（回退）

对当前组合逐项撤下并比较，7次暖请求/每轮，ABBA共两轮。512线程方案整体改善仅0.292%（1.125%/0.048%），PCM字节相同。初测0.507%的结果未重复，保留回退提交 `f8c6182`，不把首轮1.13%作为保留理由。见 `apple-silicon/16-rms-final.json`。

## 17：CFM调制缓存最终组合复核（回退）

7次暖请求/每轮ABBA，整体改善仅0.117%（+2.008%/-1.807%）。初测收益未在当前组合中重复，按门槛保留回退 `0992405`。语义token和PCM字节仍相同，同时移除新增调制缓存及上传分配。初测与最终结果均归档；最终取 `apple-silicon/17-cfm-final.json`。

## 18：Attention分块最终组合复核（回退）

7次暖请求/每轮ABBA，整体改善0.281%（+0.582%/-0.042%），输出逐字节一致。初测2.21%未重复，按门槛保留回退 `b381d1e`，恢复原FP32 attention分块和共享存储。见 `apple-silicon/18-attention-final.json`。

## 19：Codec最终组合复核（保留）

撤下Codec pass后，以7次暖请求/轮ABBA复核其实际贡献。整体改善1.948%（2.543%/1.390%），每个场景均改善、语义token和PCM字节完全相同。恢复 `7d5deef` 的实现，保留连续GPU pass与复用工作区。见 `apple-silicon/19-codec-final.json`。

## 20：FFT最终组合复核（保守回退）

混合ABBA中音频情绪整体-0.916%，因此再次做12次暖请求/轮的FFT单项ABBA，并拆出DSP与语音编码计时。整体点估计+0.972%，两轮-0.715%/+3.339%，无法确认稳定超过0.5%的收益；直接DSP仅6.386ms→4.721ms，节省1.665ms（占整句1.579s的0.105%），其他GPU阶段的波动贡献了表面整体收益。按严格门槛保守保留回退 `0481674`，撤去新增FFT setup/工作区与Accelerate链接。质量仍逐字节一致。见 `apple-silicon/20-fft-final.json`。

最终实现仅保留：Codec GPU pass、混合beam KV快照、Qwen GPU embedding/argmax、有界文字情绪分类缓存。计时和验证工具保留作为基础设施。

## 21：Beam拷贝独立队列（保留）

实验 `c8aceee`。相同共享KV布局与同步等待，把GPU snapshot/restore blit从推理主队列移到按需创建的独立queue。混合ABBA整体+4.373%，长beam两轮+0.596%/+6.954%，质量逐字节一致。两个候选进程的qin flow稳定约1.92s，反向顺序的旧队列进程出现2.4–2.6s；这是观察结果，不据此断言驱动调度的具体原因。队列与MetalContext同寿命，无新增常驻快照；每次生成结束仍清空池。见 `apple-silicon/21-kv-copy-queue.json`。
