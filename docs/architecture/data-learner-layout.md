# 数据、Importer、SQLite 与 Learner 目录边界

状态：已接受的目标设计，尚未实现。本文只规定未来代码的目录、职责与依赖方向，不表示这些
模块已经存在。现有实现状态仍以[系统边界](overview.md)和[项目路线图](../roadmap/README.md)
为准。

## 目标目录

```text
src/zero_ttt/
├── data/
│   ├── __init__.py
│   ├── contracts.py
│   ├── records.py
│   ├── manifests.py
│   ├── validation.py
│   ├── replay.py
│   ├── importers/
│   │   ├── __init__.py
│   │   ├── contracts.py
│   │   ├── registry.py
│   │   ├── sgf.py
│   │   └── katago.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── contracts.py
│   │   ├── npz_shards.py
│   │   ├── sqlite_catalog.py
│   │   ├── recovery.py
│   │   └── migrations/
│   └── sampling/
│       ├── __init__.py
│       ├── source.py
│       ├── mixture.py
│       └── symmetry.py
├── learner/
│   ├── __init__.py
│   ├── contracts.py
│   ├── learner.py
│   ├── optimizer.py
│   ├── schedule.py
│   ├── losses.py
│   ├── gradients.py
│   ├── ema.py
│   └── checkpoint.py
└── workflows/
    └── offline_imitation.py
```

目录表达职责所有权，不要求每个目标文件在同一提交中创建。公开名称只从各包的
`__init__.py` 导出；内部实现文件不作为稳定导入面。

## `data` 的内部边界

`data/contracts.py` 保留现有 `TrainBatch` 与 `BatchSource`，作为数据流程和 Learner 之间的
稳定边界。`records.py` 定义 `TrajectoryRecord`、`AnnotationRecord` 等持久逻辑记录；
`manifests.py` 定义来源 URL、许可声明、原文件哈希、格式身份和导入配置。完整字段语义服从
[公共契约](contracts.md)与[序列化训练数据](trajectory-storage.md)。

`validation.py` 校验记录身份、标签视角、mask 和搜索预算。`replay.py` 只使用本地
`GameState` 从初始状态与 moves 重建局面、合法着和模型特征；外部数据的预计算输入特征
不能成为事实来源。

## 来源专属 Importer

`importers/contracts.py` 定义窄 `Importer` 协议。Importer 接收源文件与显式 manifest，流式
产出 `TrajectoryRecord`、可选的 `AnnotationRecord` 及导入统计。它只负责解析、来源语义转换
和必要的格式级校验，不写 NPZ/SQLite，不构造 `TrainBatch`，也不依赖 Learner。

`registry.py` 只根据 manifest 中显式声明的 `source_format` 选择 Importer；不得根据扩展名或
内容自动猜测。未知格式必须直接失败，不能退回通用 SGF 解析。

- `SgfImporter` 处理普通 SGF，包括许可明确的职业棋谱与 Leela Zero SGF。实际落子形成
  one-hot policy；没有可信 score 或 ownership 时，对应 mask 为无效。
- `KataGoImporter` 匹配 KataGo `.sgfs` 与配套 NPZ，按 game identity 和 ply 转换访问分布、
  value、score 与 ownership。随原始自博弈同时生成的搜索标签属于基础 trajectory；后来
  独立生成的教师标签才写成 `AnnotationRecord`。

新增来源通过新增独立 Importer 接入，不修改现有 Importer。Leela raw training data 与 ELF
OpenGo 首版不设专属模块；它们未来仍须实现同一协议。

## 分片、SQLite 与采样

`storage/contracts.py` 定义 shard store 与 catalog 的窄边界。`npz_shards.py` 负责版本化具名
数组、不可变分片、原子封存和 SHA-256；`sqlite_catalog.py` 只保存来源、shard、game、offset、
校验与状态等控制信息。`recovery.py` 隔离半写文件、孤立文件和校验失败的分片；schema 变化
通过 `migrations/` 显式升级。

SQLite 不保存大型训练 BLOB，也不为每个 position 建行。随机采样基于 shard/game 的 step
范围和 offset，不使用 `ORDER BY RANDOM()`。教师任务、租约和评级表由未来 migration 增加，
不进入首版 catalog。

`sampling/source.py` 是持久记录到 `BatchSource` 的唯一桥梁。它通过 catalog 选择 game/ply，
通过 shard reader 读取记录，再调用 replay 与 validation 生成 `TrainBatch`。`mixture.py` 组合
不同来源及阶段的采样比例；`symmetry.py` 负责 D4 增强。混合与增强不得进入 Importer、SQLite
或 Learner。

compressed NPZ 的读取细节封装在 shard reader 后面。扩大数据前必须 benchmark 随机读取、
缓存命中和解压开销；物理格式变化不能影响 catalog、sampling 或 Learner 的公共契约。

## Learner 与工作流

`learner/learner.py` 是训练若干 step、保存、恢复和发布的小型门面。其余模块承接当前
`training` 包中的优化器、调度、损失、梯度、EMA 与 checkpoint 能力。迁移完成后 Learner
独占训练模型和优化状态，但只消费 `BatchSource`，不知道 Importer、shard、SQLite 或 workflow。

`workflows/offline_imitation.py` 负责组合 Importer、validation、shard store、catalog、
`BatchSource` 和 Learner，落实[监督冷启动](../workflows/offline-imitation.md)。它是协调层，底层
组件不得反向导入 workflow。

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
- 首版只实现 SGF、KataGo rich-label、NPZ shard、SQLite catalog 和离线 `BatchSource` 垂直切片。
- 自博弈、OpenSpiel、教师队列与课程调度延期；记录契约只为其搜索标签保留可选字段和 mask。
