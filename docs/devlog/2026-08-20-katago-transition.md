# 2026-08-20：转向 KataGo 教师与解耦训练

## 目标

停止维护内部 MCTS，把训练研究路线、KataGo 集成和未来快权重思路转成小型文档与代码骨架。

## 完成内容

- 删除内部搜索、MCTS self-play、搜索型 replay 和 CoreLoop。
- 配置升级为 schema v3，CLI 只保留真实可运行的检查与 smoke。
- Trainer 改为 `BatchSource`，增加 `TrainBatch` 和 `PositionEvaluator` 契约。
- Tromp–Taylor 多子自杀进入棋规和单元测试。
- KataGo v1.17.2 子模块、CUDA 多阶段镜像、Analysis/GTP Compose 服务和外部权重目录落地。
- 离线模仿、在线蒸馏、局域网教师、MCTS 兼容性与快权重方向完成文档化。

## 兼容性

旧运行文件未主动删除，但旧 replay 和 checkpoint schema 不迁移。删除的源码、TeX 和 PDF
可从 Git 历史恢复。项目从本次变更起只支持 Docker。

## 延期

SGF/g170 数据、真实学生自博弈、教师 worker、在线蒸馏、Human-SL 评测、快权重和学生搜索
后端均未实现。
