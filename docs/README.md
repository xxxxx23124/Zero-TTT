# Zero-TTT 文档索引

文档区只描述可验证的当前状态、已接受的设计和明确标注的未来工作。单个 Markdown 文件
不得超过 150 行或 12 KiB；主题变大时继续拆分。

## 当前架构

- [系统边界](architecture/overview.md)
- [内部格式版本](architecture/versioning.md)
- [模型与训练](architecture/model-and-training.md)
- [严格 FP32 精度策略](architecture/precision.md)
- [Learner 与流程边界](architecture/learner-and-workflows.md)
- [公共契约](architecture/contracts.md)
- [Docker 运维](operations/docker.md)
- [Docker 训练控制台](operations/training-console.md)

## 数据采集与训练路线

- [统一训练生命周期](workflows/training-lifecycle.md)
- [监督冷启动](workflows/offline-imitation.md)
- [主动选点与在线蒸馏](workflows/online-distillation.md)
- [Human-SL 分级教师](workflows/curriculum-teachers.md)
- [项目路线图](roadmap/README.md)

## 外部集成

- [KataGo](integrations/katago.md)
- [局域网教师协议](integrations/lan-teacher.md)

## 源码旁实现文档

- [数据边界与采样](../src/zero_ttt/data/README.md)
- [数据 Importer](../src/zero_ttt/data/importers/README.md)
- [序列化训练数据](../src/zero_ttt/data/trajectory-storage.md)
- [模型前向](../src/zero_ttt/model/README.md)
- [Learner 与训练产物](../src/zero_ttt/training/README.md)
- [Publication 推理与聚批](../src/zero_ttt/inference/README.md)
- [OpenSpiel 搜索适配](../src/zero_ttt/search/README.md)
- [MCTS 自博弈采集](../src/zero_ttt/selfplay/README.md)
- [训练控制台内部编排](../src/zero_ttt/console/README.md)

## 未来研究

- [快权重总览](research/fast-weights/overview.md)
- [历史与战略记忆](research/fast-weights/trajectory-memory.md)
- [搜索摘要记忆](research/fast-weights/search-memory.md)
- [实验与风险](research/fast-weights/evaluation.md)

## 项目记录

- [架构决策](decisions/README.md)
- [开发日志](devlog/README.md)
- [论文清单](../paper/README.md)

用 `python scripts/check_docs.py` 检查全局与源码旁文档的大小和本地链接；该命令只作为 Docker
容器内命令维护。
