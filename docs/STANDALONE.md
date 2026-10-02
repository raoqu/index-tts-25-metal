# 独立项目验收（2026-10-02）

原生源码、模型包及前端资源整体复制至 `~/mylab/index-tts-25-metal`。
新目录没有原Python入口、虚拟环境或原始checkpoint，没有借用原项目构建缓存。
`build.sh`、`dev.sh` 根据脚本所在目录定位项目；可执行文件也从自身位置
向上定位模型、前端和网页。无参数默认Web服务，地址 `127.0.0.1:7860/web`。

## 已完成检查

- 从新目录全量Release编译成功，记录 `standalone-build.txt`。
- 32个模型/前端资源文件的SHA-256与来源完全相同，清单
  `model-packages.sha256`，检查记录 `standalone-model-copy-check.txt`。
- 默认两项Metal/MPS CTest通过，记录 `standalone-ctest.txt`。
  原10项模型数值报告仍见 `ctest-native-final.txt`。
- 在 `/tmp` 及受限 `PATH=/usr/bin:/bin:/usr/sbin:/sbin` 中，直接执行二进制
  自动开启Web模式并定位本项目模型、前端、音色与网页。
- 真实文字情绪请求生成非静音22050Hz单声道PCM16，正常EOS，467次GPU提交；
  同种子重复WAV字节一致，暖请求情感缓存命中。
- 进程没有推理子进程、Python/Torch映射或原项目文件映射。
  详见 `standalone-runtime-validation.json`、`standalone-linkage.txt`。
- SQLite使用在线backup复制；两条原音色与锁定状态保留，新数据库中的管理
  音色路径已指向本项目 `voices/`，源数据库保持原路径。
- shell语法检查、`--help`、仓库外 `--no-build --prepare-only` 均通过。
- 从 `/tmp` 无参数运行 `dev.sh` 成功接管正式7860 Web服务，模型/音色/网页
  路径均为新目录；生成WAV与直接二进制运行的同种子结果一致。
  默认入口验收见 `standalone-default-service.json`。

源码版本见根目录 `SOURCE.md`。模型、音色及编译产物本地保留并被Git忽略；
Git保存全部源码、脚本、许可、校验清单和验收记录。
