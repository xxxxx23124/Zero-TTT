# 数据、Importer、SQLite 与 Learner 目录边界

状态：首版离线垂直切片已实现。本文描述当前模块边界；仍未实现的扩展以
[项目路线图](../roadmap/README.md)为准。

## 当前目录

```text
src/zero_ttt/
├── data/
│   ├── contracts.py          # BatchSource / TrainBatch
│   ├── records.py            # TrajectoryRecord / AnnotationRecord / ImportEvent
│   ├── manifest.py           # SourceManifest 与资源哈希
│   ├── importers/katago_sgf.py
│   ├── shards.py             # content-addressed NPZ
│   ├── catalog.py            # SQLite migrations、snapshot 与 GC
│   ├── catalog_source.py     # snapshot → TrainBatch
│   └── pipeline.py           # importer 与持久层协调
├── learner.py                # 唯一 Learner 实现
└── training/                 # loss、梯度、EMA、checkpoint 与兼容导出
```

目录表达职责所有权；旧 `zero_ttt.training.trainer.Trainer` 是 `Learner` 的兼容别名，不保留
第二份训练实现。

## `data` 的内部边界

`data/contracts.py` 保留现有 `TrainBatch` 与 `BatchSource`，作为数据流程和 Learner 之间的
稳定边界。`records.py` 定义 `TrajectoryRecord`、`AnnotationRecord` 等持久逻辑记录；
`manifest.py` 定义来源许可声明、原文件哈希和格式身份。完整字段语义服从
[公共契约](contracts.md)与[序列化训练数据](trajectory-storage.md)。

记录类在构造边界校验身份、内容 SHA-256、shape、mask 和稀疏 policy；trajectory 只允许
空棋盘、黑方先行，并由本地 `GameState` 按记录的 `max_moves` 与 moves 重建局面。外部预计算
特征不是事实来源；未来非空初始局面需要能表达行棋方、历史与劫争身份的独立 schema。

## 来源专属 Importer

Importer 接收源文件与显式 manifest，流式
产出 `TrajectoryRecord`、可选的 `AnnotationRecord` 及导入统计。它只负责解析、来源语义转换
和必要的格式级校验，不写 NPZ/SQLite，不构造 `TrainBatch`，也不依赖 Learner。

- `KataGoSgfImporter` 只接受 manifest 明确声明的 `katago-g170-sgfs-zip`，流式读取 zip 中
  line-delimited SGF。sgfmill typed-property 解析、领域 replay/标签转换、ZIP/provenance 编排是
  单向边界；单局格式错误成为结构化 rejection，asset I/O 与完整性错误仍整体失败。
- KataGo rich NPZ、Leela 和普通职业棋谱尚无 Importer，不根据扩展名或内容自动猜测来源。

新增来源通过新增独立 Importer 接入，不修改现有 Importer。Leela raw training data 与 ELF
OpenGo 首版不设专属模块；它们未来仍须实现同一协议。

## 分片、SQLite 与采样

`shards.py` 负责版本化具名数组、不可变分片、原子封存和 SHA-256；`catalog.py` 只保存来源、
shard、game、索引、校验与状态等控制信息，并负责 schema migration、半写恢复和孤立文件报告。

SQLite 不保存大型训练 BLOB，也不为每个 position 建行。随机采样基于 shard/game 的 step
范围和 offset，不使用 `ORDER BY RANDOM()`。教师任务、租约和评级表由未来 migration 增加，
不进入首版 catalog。

`catalog_source.py` 是持久记录到 `BatchSource` 的唯一桥梁。纯 `SnapshotPositionIndex` 先按
eligible position 数加权选择一个 trajectory shard，再在 shard 内采满 microbatch；
`TrajectoryBatchMaterializer` 对本批每个 shard 至多读取一次，并调用本地 `GameState` 与
`game.symmetry` 生成 `TrainBatch`。单个 position 的边际分布仍均匀，microbatch 内允许相关。
多来源 mixture 通过不可变 manifest 和 `MixtureBatchSource` 实现；每个 microbatch 只选择一个
snapshot source，因此仍保持 shard-local 读取。混合与当前增强都不得进入 Importer、SQLite 或 Learner。

NPZ 的读取细节封装在 shard reader 后面。sampling identity 明确包含 shard-local 算法版本、
annotation 模式与 D4；物理重分片不改变 snapshot 的逻辑内容身份。

## Learner 与工作流

`learner.py` 是训练若干 step、保存、恢复和发布的小型门面。`training` 包承接损失、梯度、
EMA、checkpoint 与旧导入路径。Learner
独占训练模型和优化状态，但只消费 `BatchSource`，不知道 Importer、shard、SQLite 或 workflow。

`data.pipeline` 与 CLI 负责组合 Importer、shard store、catalog、`BatchSource` 和 Learner，
落实[监督冷启动](../workflows/offline-imitation.md)。底层组件不得反向导入协调层。

## 依赖方向与首版范围

```text
source files + manifest
          ↓
Importer → records → shard store + catalog → BatchSource → TrainBatch → Learner
                         ↑                     ↓                         ↓
                       SQLite          replay + validation       publication
```

- `game` 可被 replay 和 validation 使用；`model` 只被 Learner 与 inference 使用。
- storage 与 sampling 依赖稳定数据契约，不能依赖具体 Importer。
- annotation 在 sampling 阶段按 `(game_id, ply, teacher_fingerprint)` 连接，不由 Learner join。
- 首版只实现 g170 SGF policy-first、NPZ shard、SQLite catalog 和离线 `BatchSource` 垂直切片。
- 自博弈与 OpenSpiel 已接入；教师队列、自动循环与课程调度延期。
