# IndexTTS 2.5 管理页面与 HTTP API

## 1. 来源和启动

复用 `/Users/raoqu/mylab/index-tts2-metal` 的 `web_index_html` 页面及
`docs/api.md` 接口设计，来源提交为 `eda855b2e9d7cfaa4269380f42880d41b060c4d2`。
页面来源哈希见 `../web/PROVENANCE.json`，实现取舍见决策 D13。
原 2.0 项目保持不变，2.5 所有神经模型仍由本项目的 C++ Metal 进程计算。
默认入口的 HTTP、SQLite、文本和音频 DSP 均在 `itts25-native` C++ 进程运行。
本独立项目只包含原生运行源码及已转换资源，不包含Python HTTP/adapter入口。

```sh
cd ~/mylab/index-tts-25-metal
./dev.sh
```

默认地址 `http://127.0.0.1:7860/web`，`/` 重定向到 `/web`。
`./dev.sh --port 3456` 可用参考项目端口。
`--server` 关闭网页，`--web` 显式启用网页，`--prepare-only` 只构建并检查离线资源。
`build.sh` 独立编译，`dev.sh` 自动构建并启动Web服务；二者从脚本位置定位项目。

服务首次创建音色库时克隆官方示例，保存锁定的 `voice_demo`。
数据默认位于项目根目录的 `voices/`：
`voices.sqlite3`、`bundles/*.pt` 和授权记录的 `samples/*`。
`--voice_store` 或 `MIT2_VOICE_STORE` 可改变目录。

## 2. 路径约定

|功能|主要路径|兼容别名|
|---|---|---|
|健康状态 GET|`/health`|`/v1/health`|
|队列状态 GET|`/api/status`|`/status`, `/web/api/status`|
|音色列表 GET / 创建 POST|`/api/voices`|`/v1/audio/voices`, `/voices`, `/web/api/voices`|
|音色 GET/PATCH/PUT/DELETE|`/api/voices/{id}`|以上各音色路径的 `{id}`|
|锁定 POST|`/api/voices/{id}/lock`|以上各音色路径的 `{id}/lock`|
|预听 GET|`/api/voices/{id}/source-audio`|以上各音色路径的 `{id}/source-audio`|
|客户端音色 POST|`/v1/audio/client_voices`|`/v1/audio/remote_voices`, `/client_voices`, `/web/api/client_voices`|
|合成 POST|`/v1/audio/speech`|`/speech`, `/web/api/speech`|
|授权记录 GET/POST及单项CRUD|`/v1/audio/voice_consents`|`/voice_consents`, `/web/api/voice_consents`|
|网页登录 POST|`/web/api/login`|无|

错误格式 `{"error":{"message":"..."}}`，包括 400 参数错误、401 网页密钥错误、
404 未找到、405 方法不支持、409 音色锁定、429 等待队列已满、500 推理错误。

`--webkey` 或 `MIT2_WEBKEY` 设置可选网页登录密钥。页面/API 使用
`X-MTTS-Web-Key: ...` 或 `Authorization: Bearer ...`；登录请求为
`{"key":"..."}`，成功返回 `{"ok":true}`。
与参考实现一致，密钥仅保护 `/web/api/*`，其他 API 路径保持原接口语义。

## 3. 音色管理

音色记录包含 `id, object, name, description, bundle_path, sample_path,
source_audio_seconds, source, locked, created_at, updated_at`。
列表返回 `{"object":"list","data":[...]}`。

```sh
curl http://127.0.0.1:7860/api/voices
curl http://127.0.0.1:7860/api/voices \
  -F audio_sample=@examples/voice_01.wav -F name=我的音色 -F description=测试
curl http://127.0.0.1:7860/api/voices \
  -H 'Content-Type: application/json' \
  -d '{"bundle_path":"/绝对路径/voice.pt","name":"已保存的音色"}'
curl -X POST http://127.0.0.1:7860/api/voices/voice_demo/lock \
  -H 'Content-Type: application/json' -d '{"locked":true}'
curl http://127.0.0.1:7860/api/voices/voice_demo/source-audio --output /tmp/preview.wav
```

multipart 上传也接受 `recording/file/audio`。参考音频保留最多15秒，
预听为原参考波形22050Hz单声道PCM16。支持修改名称、描述、bundle路径和sample路径。
锁定音色无法删除；原文件路径导入会复制到音色库，删除仅清理库内文件。

## 4. 可下载的客户端音色

```sh
curl http://127.0.0.1:7860/v1/audio/client_voices \
  -F audio_sample=@examples/voice_01.wav -F voice_id=client_demo \
  -F consent=demo -F persist=false --output /tmp/voice25.pt
curl http://127.0.0.1:7860/v1/audio/client_voices \
  -F pt=@/tmp/voice25.pt -F voice_id=client_demo -F persist=false
```

音频克隆返回 MIT2 音色二进制，响应头 `X-Voice-Id` 包含ID；默认仅缓存。
上传 `pt` 返回 JSON `status, voice_id, persisted, cached`。
`persist=true` 保存到 SQLite 音色库；已有锁定ID时直接返回已有音色二进制，
保留原记录。非持久客户端音色采用 LRU，默认20项，重启后消失。

`.pt` 后缀沿用参考接口，但内容为 MIT2 + MIT2VOIC footer，不是 pickle。
文件包含2.5音色/情绪特征、style、声学条件、mel和预听波形。
校验 target/version、形状、边界、有限值及 SHA256，上传限制128MiB。
**2.0 的音色文件不能直接复用；需要用原参考音频重新克隆为2.5。**

## 5. 合成

```sh
curl http://127.0.0.1:7860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"index-tts2.5","input":"你好，这是本地测试。","voice":"voice_demo","response_format":"wav","lang":"ZH","seed":42,"duration_factor":1}' \
  --output /tmp/test25.wav
```

`input` 必填，最多10000字符；`voice` 可为音色ID、名称、`{"id":"..."}`、
2.5 MIT2单文件/目录路径；省略时使用最早创建的持久音色。
`model` 保留兼容字段，实际固定使用本地2.5。只接受 `response_format="wav"`。

无 `output` 返回 WAV。设置服务器本地 `output` 路径并省略 `stream` 时，
写 WAV 并返回 JSON `status, output, audio_seconds, total_seconds, rtf,
gpu_submissions, stage_seconds, emotion_cache_hits`；`stream=true` 同时保存并返回 WAV。
`stage_seconds` 为当前请求实际执行的各原生阶段耗时，缓存命中的情绪编码不重复计算。
这里的 stream 沿用原约定，返回生成后的完整文件。

|2.5参数|行为|
|---|---|
|`lang` 或 `language`|`ZH/EN/JA/ES/AR`，默认ZH，接受小写|
|`duration_factor`|0.25–4，默认1，越小越快|
|`speed`|0.25–4，与时长系数互为倒数；显式duration_factor优先|
|`emo_text`|文字情绪描述，非空时启用原生Qwen；可用use_emo_text控制|
|`emo_audio_prompt`, `emo_alpha`|服务器上的情绪参考音频路径及0–1强度|
|`emo_vector`|8个非负值，总和<=0.8：高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然|
|`seed`, `use_random`|控制采样复现及情绪向量查表方式|
|`num_beams`|默认1，1–10|
|`do_sample`, `temperature`, `top_k`, `top_p`|采样设置，沿用原推理流程|
|`repetition_penalty`, `length_penalty`, `max_mel_tokens`|GPT生成参数|
|`max_text_tokens_per_segment`, `interval_silence`, `text_normalization`|分句、句间静音和文本规范化|

正常结果为22050Hz单声道PCM16。参考音频和推理长度边界见 `LOCAL_RUN.md`。

## 6. 授权记录和队列

授权记录 POST 支持 multipart `recording/audio_sample/file/audio`，
或 JSON `recording_path`。字段为 `name, language, create_voice`，
默认 `create_voice=true`，返回关联的 `voice_id`。支持列表和单项查询、编辑、删除。
这是本地记录管理接口，不执行外部身份验证。

`/api/status` 返回模型/音色路径、网页状态、configured并行设置、queue、totals、recent。
队列含 `max_waiting, waiting, running, current`，recent保存最近32条完成/失败/拒绝任务。
单一 C++ 进程中的 Metal 模型串行执行，`effective_executor_concurrency=1`；
兼容 `--tts_concurrency/--clone_concurrency` 配置字段。
`--queue_size` 默认16，`--voice_cache_size` 默认20，均可通过同名 `MIT2_*` 环境变量设置。
Ctrl+C等待在途请求结束并关闭服务。

## 7. 验收

```sh
.venv/bin/python -m unittest backends.metal25.tools.test_http_service -v
.venv/bin/python -m backends.metal25.tools.validate_http --url http://127.0.0.1:7860
```

单元测试使用替代引擎验证 HTTP、音色、SQLite、LRU、锁定和队列契约；
实时验收真实克隆、保存/二进制返回、GPU提交、固定种子复现及清理。
完整原生服务报告见 `native-http-validation.json` 和 `native-runtime-validation.json`；
`http-validation.json` 和 `PROGRESS.md` 保留此前 Python HTTP 入口的历史验收。
