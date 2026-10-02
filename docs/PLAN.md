# IndexTTS 2.5 C++ / Metal 后端迁移

> 2026-10-02 新要求已完成：默认 CLI、HTTP、文本和音频前端全部原生化。
> 下文允许保留 Python UI/前处理的旧边界已被替代；本轮验收与提交见
> [NATIVE_RUNTIME.md](NATIVE_RUNTIME.md)。Python 仅用于离线转换及验证。

## 目标和完成标准

在独立的 `backends/metal25/` 中运行真正的 IndexTTS 2.5 原生推理，保留现有
Gradio UI 和语音克隆、语言选择、语速与情绪控制。最终模型推理不能调用
PyTorch/Python，也不能用 IndexTTS2 权重替代 2.5。Python 可以保留 UI、
文本前处理、离线权重转换及测试基准。

每个重要步骤提交 Git；所有待决问题记录推荐选择与依据，不向用户提问。

## 实施顺序

1. 保存现有可运行基线；审计真实权重、配置和参考项目。
2. 构建独立 C++17/Objective-C++ Metal 运行时与权重 bundle 工具。
3. 移植 EnhancedCodec：下采样、Vocos 编码器、FVQ、Vocos 解码器、上采样。
4. 移植 2.5 GPT：说话人投影、情绪 Conformer/Perceiver、语言嵌入、
   KV cache、自回归采样；逐层及 logits 对齐。
5. 移植当前 checkpoint 的 S2Mel/CFM 和 BigVGAN；固定噪声对齐 mel/波形。
6. 移植 W2V-BERT、CAMPPlus 和参考音频前处理；测试新音色和情绪音频。
7. 通过持久原生进程连接现有 UI，保留参数与进度/错误处理；移植可选文字情绪模型。
8. 多语种、长文本、语速、情绪、重复请求和 UI 端到端测试；记录与基线的质量、
   延迟、内存和 GPU 执行证据。原生后端成为默认启动路径。

## 验证原则

- PyTorch 仅为离线数值 oracle，golden 使用真实 2.5 checkpoint。
- 原生测试必须记录设备及真实 GPU command buffer 执行次数。
- 先用 FP32 对齐，再评估 FP16、缓存与融合优化。
- 部分组件成功不代表完整迁移完成；UI 切换前需完整原生 WAV 通过验收。
- 完成证据包括 build/CTest、组件误差、生成音频及原生 UI 请求记录。

## 状态

全部实施阶段已完成。基线提交 `f6394b9`；阶段证据、命令和边界见 `PROGRESS.md` 与 `DECISIONS.md`。
