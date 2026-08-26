# 系统边界

## 项目定位

Zero-TTT 负责学生模型、训练基础设施、准确棋规、数据资产和学生 MCTS 自博弈。OpenSpiel
提供通用 Python PUCT/MCTS 控制流；本地 `GameState` 仍是规则真源。未修改的 KataGo 只作为
Human-SL/强搜索教师和独立 GTP 引擎。

```text
历史棋谱 ───────────────┐
OpenSpiel MCTS 自博弈 ──┼─→ immutable trajectory shards ─┐
                        │          SQLite index/control ──┼─→ BatchSource → Learner
KataGo 教师 ─→ annotation shards ─────────────────────────┘          │
                                                                     ↓
                                                        EMA publication
                                                               │
                                                  PositionEvaluator
                                                               │
                                                OpenSpiel Evaluator/MCTS
```

## 当前已实现

- 固定 19×19 的 Tromp–Taylor 状态、合法着、历史、数子与特征编码。
- 策略、价值、所有权和目差输出的 Transformer。
- 与数据来源无关的 `Learner`、CPU FP32 EMA、checkpoint 和 BF16 publication；旧 Trainer
  路径只是兼容导出。
- `TrajectoryRecord`/`AnnotationRecord`、NPZ ShardStore、SQLite Catalog、不可变 snapshot 和
  `CatalogBatchSource` 垂直切片。
- `BatchSource`、带逐样本标签 mask 的 `TrainBatch`、`PositionEvaluator` 三个稳定边界。
- KataGo v1.17.2 的独立 Docker 入口，以及固定版本的 OpenSpiel 源码子模块。

## 当前未实现

- 跨 shard compaction、KataGo rich-NPZ 连接、Leela 混合采样与自动数据窗口监控。
- `pyspiel.Game/State` 薄适配层、OpenSpiel evaluator 和 MCTS 自博弈采集。
- KataGo 教师 worker、局域网队列、混合采样和课程调度。
- 快权重和评级系统。

## 解耦原则

- 本地 `GameState` 唯一决定合法着、历史、终局和计分，不采用 OpenSpiel 内置 Go 状态。
- OpenSpiel 只负责搜索，不采用其内置模型、Learner 或完整 AlphaZero runner。
- Learner 只看 `BatchSource`/`TrainBatch`，不知道样本来自棋谱、自博弈还是教师。
- 模型不依赖 KataGo 协议、OpenSpiel 状态或搜索树类型。
- 采集器先写持久数据；KataGo 标签以 sidecar annotation 追加，不重写原始棋局。
- Learner 与 evaluator 可同时驻留 GPU，但默认分阶段执行，不并发提交 CUDA 工作。
- 所有运行、构建和测试只通过 Docker 支持。

旧 replay/checkpoint 不自动删除，但 schema v2 及旧搜索记录不再受支持，也不提供迁移器。
目标组件见 [Learner 与流程边界](learner-and-workflows.md)、
[序列化训练数据](trajectory-storage.md)与[统一训练生命周期](../workflows/training-lifecycle.md)。
