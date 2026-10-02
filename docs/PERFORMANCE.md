# IndexTTS 2.5 FP32 Metal 性能验收

优化前主要完成原生迁移和数值对齐。部分参考项目快速路径依赖FP16，默认FP32路径未充分优化，且GPT单token使用一行MPS GEMM。此次在完整参考音频、相同生成参数和FP32精度下补齐实际性能优化。

设备：Apple M1 Max / 64 GB。代码路径：`/Users/raoqu/research/itts25`。基线提交`be3c7ef`；算子优化`e35ec77`；交错请求修复`272d811`。模型由getmodel从ModelScope下载，权重未变化。

## 相同配置的前后对照

文本“你好世界”，seed=424242，ZH，num_beams=1，duration_factor=1，CFM25步、CFG0.7。两种声音分别使用完整2.4387秒和7.3041秒参考，禁止截短参考或降低生成步数。每组独立启动一进程，测一次冷请求及两次暖请求，暖值取中位数；测试期间停止HTTP服务和其他GPU推理。RTF=生成墙钟秒数/音频秒数。

|声音|音频秒数|基线暖耗时|优化暖耗时|基线RTF|优化RTF|加速|
|---|---:|---:|---:|---:|---:|---:|
|voice_demo（2.44秒参考）|1.7067|14.912s|1.384s|8.738|0.811|10.78×|
|qin（7.30秒参考）|1.8692|26.453s|2.623s|14.152|1.403|10.09×|

短参考已快于实时；qin长参考、极短输出仍高于RTF1。参考段参与CFM计算，因此长参考对短句的固定成本仍较大。上述两个数字来自同声音同配置对照，不直接用不同声音的RTF作前后比较。

## 阶段耗时（qin，暖请求中位数）

|阶段|基线|优化|
|---|---:|---:|
|情绪编码|0.696s|缓存命中|
|GPT|3.956s|0.452s|
|Codec|0.110s|0.086s|
|长度调节|0.033s|0.032s|
|CFM|13.665s|1.914s|
|BigVGAN|7.952s|0.116s|

阶段时间来自原生进程；整请求还含RPC、文本/音频DSP和文件处理。UI/API JSON新增`stage_seconds`、`emotion_cache_hits`。

## 冷请求边界

上述速度不包含首次模型加载。首次请求还会构建常驻权重、卷积tap和时间表。原基线冷请求开启输入捕获，不能用来做严格的冷请求加速比。

- demo：模型加载4.019s；加载后首请求4.249s；首次合计约8.268s。
- qin：模型加载3.976s；加载后首请求5.431s；首次合计约9.407s。

## 实现与回归

- GPT：单token投影切换为参考FP32 simdgroup GEMV。
- DiT：参考分块注意力新增FP32 MMA内核，16-key块适配32KiB共享存储。
- CFM：25步轨迹留在GPU，条件投影一次；缓存准确的2.5时间embedding和步长。
- BigVGAN：FP32 MPS扩张卷积及按stride相位的转置卷积；低通激活拆分避免重复Snake计算，中间工作区复用。
- 情绪embedding按输入FP32字节及形状缓存，最多8条并返回副本；不替换模型或改变情绪融合数学。
- 真实交错GPT/CFM请求暴露旧零bias快速路径异常；先显式广播bias，再矩阵乘法可修复。独立identity GEMM旧版也通过，底层MPS原因未独立证实。
- 10项CTest和12项HTTP/缓存单元测试通过；原有组件数值门槛不变。
- 12组真实集成回归通过：五语言、重复、语速0.8/1.2、情绪参考/向量/文字、多段合成。Python神经网络调用为0；重复WAV完全一致。
- 逐阶段相同真实输入对照要求GPT token完全一致、CFM和vocoder max_abs<1e-4、RMSE<1e-5；PCM16也使用该既有波形门槛。短参考峰值差异3个PCM16量化单位，长参考1个；RMSE均<1e-5，SNR均>83dB。自动数值检查不声称人工听感或ASR验收。

## 复现

在项目根目录执行：

```sh
.venv/bin/python -m backends.metal25.tools.benchmark_performance \
  --output backends/metal25/artifacts/performance/recheck-demo
.venv/bin/python -m backends.metal25.tools.benchmark_performance \
  --voice backends/metal25/voices/bundles/voice_92c85f8a49b34f1ca2e13a7c1028ee54.pt \
  --output backends/metal25/artifacts/performance/recheck-qin
.venv/bin/python -m backends.metal25.tools.validate_performance
ctest --test-dir backends/metal25/build --output-on-failure
./dev.sh
```

原始证据：`performance-benchmarks.json`、`performance-validation.json`、`performance-integration-validation.json`、`ctest-performance.txt`。基线二进制和捕获输入位于`backends/metal25/artifacts/performance`，未提交大文件。`validate_performance.py`会在同一进程内交错回放GPT/CFM/vocoder并切换两种声音，覆盖实际复用问题。

## 正式dev.sh服务验收

`./dev.sh`无参数在127.0.0.1:7860启动，/跳转/web；原两条声音及锁定状态保持。
真实API完成克隆/预听/锁定保护、MIT2音色上传缓存和同种子重复验证，临时音色已删除。
4.5047秒中文测试音频的暖请求2.854秒，RTF0.633。qin同种子API暖请求2.616秒，
RTF1.399；WAV与独立性能基准完全相同。失败任务数0，队列空闲。
API的冷/首声音请求和暖缓存数字分开记录，见performance-http-validation.json及
performance-qin-http.json。所有神经模型继续在C++ Metal运行。

网页实际选择qin点击Generate WAV，2.0秒音频用时2.74秒，页面RTF1.37；播放器
readyState=4。UI请求使用网页原默认随机种子，作为功能/状态验证，不替代固定种子
性能基准。完整截图保存在聊天outputs/itts25-optimized-ui.png，验收记录为
performance-ui-validation.json。正式服务保持运行，等待数0，失败数0。
