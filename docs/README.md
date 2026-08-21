# Zero-TTT 文档索引

文档区只描述可验证的当前状态、已接受的设计和明确标注的未来工作。单个 Markdown 文件
不得超过 150 行或 12 KiB；主题变大时继续拆分。

## 当前架构

- [系统边界](architecture/overview.md)
- [模型与训练](architecture/model-and-training.md)
- [Learner 与流程边界](architecture/learner-and-workflows.md)
- [公共契约](architecture/contracts.md)
- [Docker 运维](operations/docker.md)

## 数据采集与训练路线

- [阶段一：离线模仿](workflows/offline-imitation.md)
- [阶段二：在线采集与蒸馏](workflows/online-distillation.md)
- [学生自博弈采集](workflows/student-selfplay.md)
- [Human-SL 分级教师](workflows/curriculum-teachers.md)
- [项目路线图](roadmap/README.md)

## 外部集成

- [KataGo](integrations/katago.md)
- [本地 MCTS 兼容性](integrations/mcts-compatibility.md)
- [局域网教师协议](integrations/lan-teacher.md)

## 未来研究

- [快权重总览](research/fast-weights/overview.md)
- [历史与战略记忆](research/fast-weights/trajectory-memory.md)
- [搜索摘要记忆](research/fast-weights/search-memory.md)
- [实验与风险](research/fast-weights/evaluation.md)

## 项目记录

- [架构决策](decisions/README.md)
- [开发日志](devlog/README.md)
- [论文清单](../paper/README.md)

用 `python scripts/check_docs.py` 检查大小和本地链接；该命令只作为 Docker 容器内命令维护。
