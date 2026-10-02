# 原生项目来源

- 原项目：`/Users/raoqu/research/itts25`。
- 原生源码来源：该项目 `backends/metal25/`，提交 `87ce6ed`。
- 移植目标：独立 macOS C++17/Objective-C++、Metal/MPS 原生运行。
- 模型：原2.5 FP32 bundle与前端资源逐文件复制，未重新转换或量化。
- 音色：原SQLite数据库使用SQLite backup复制，保留两条现有音色及锁定状态；
  仅在新数据库内将本地音色路径改为本项目 `voices/` 目录。
- 不包含原Python运行入口、虚拟环境、原始checkpoint及原构建缓存。
- 原模型数值回归报告和历史迁移文档复制至 `docs/`；其中的原项目路径是
  历史记录或来源信息，运行依赖为本项目内的 `bundles/`、`web/`、`voices/`。
- 运行脚本和可执行文件都能自动定位本项目资源；源码未依赖原项目文件。

Metal底层、FST与HTTP等第三方代码来源详见各 `vendor/*/PROVENANCE*`、
`vendor/FST_PROVENANCE.md` 及随附许可证。
网页来源见 `web/PROVENANCE.json`。模型许可保留在项目根目录。
模型及前端文件校验清单为 `docs/model-packages.sha256`。
