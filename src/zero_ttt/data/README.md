# `zero_ttt.data`

`data` 包拥有训练数据从来源声明到 `TrainBatch` 的完整边界：校验逻辑记录、封存不可变
分片、维护 SQLite 控制面、创建 snapshot，并把持久记录物化为训练批量。它不负责模型优化、
搜索、教师网络协议或工作流状态机。

公共数据接口及张量语义见[公共契约](../../../docs/architecture/contracts.md)，物理格式和恢复规则见
[序列化训练数据](trajectory-storage.md)。

## 稳定边界

- `contracts.py` 定义 `TrainBatch` 与 `BatchSource`，是数据流程和 Learner 之间的稳定边界。
- `records.py` 定义 `TrajectoryRecord`、`AnnotationRecord` 与 `ImportEvent`，构造时校验身份、
  shape、mask、稀疏 policy 和内容 SHA-256。
- `manifest.py` 保存来源、许可、原始资产哈希和格式身份；来源不能靠扩展名或内容猜测。
- `pipeline.py` 只协调 Importer、`TrajectoryShardSink`、`ShardStore` 和 `Catalog`。
- `mixture.py` 用不可变 manifest 组合多个 snapshot；混合比例不进入 Importer、Catalog 或 Learner。

底层组件可以依赖 `game` 重放和编码局面，但不能依赖 `learner`、`training`、`selfplay`、
`console` 或 CLI。具体来源解析规则属于 [`importers`](importers/README.md)。

## 持久层与控制面

`ShardStore` 只处理安全路径、原子提交和内容寻址；`TrajectoryNpzCodec` 与
`AnnotationNpzCodec` 拥有数组布局。导入和自博弈都通过 `TrajectoryShardSink` 累计、估算并
封片，不能各自实现另一套 writer。

公开 `Catalog` 是兼容门面，内部组合：

- `CatalogSession`：连接、PRAGMA 和当前 schema 初始化；
- `CatalogRepository`：来源、shard、game、任务及普通 SQL；
- `SnapshotService`：确定性成员选择与 snapshot 身份；
- `ShardLifecycle`：半写恢复、tombstone 和垃圾回收。

SQLite 只保存索引和控制信息，不保存大型训练数组。任何恢复或删除顺序必须服从
[持久化说明](trajectory-storage.md)，不能绕过门面直接操作数据库与分片。

## Snapshot 采样与批量物化

`SnapshotPositionIndex` 先按各 trajectory shard 的 eligible position 数选择一个 shard，再在该
shard 内有放回地抽满一个 microbatch。这样每个 position 的边际概率仍均匀，同时
`TrajectoryBatchMaterializer` 可保证一个 microbatch 只读取一个 trajectory shard。

物化过程从 moves 确定性重放 `GameState`，按 `(game_id, ply, teacher_fingerprint)` 精确连接
annotation，转换当前行棋方视角，应用标签 mask，并对整个样本执行一个 D4 变换。外部预计算
特征不是事实来源；annotation 缺失时不得伪造有效零标签。

`MixtureBatchSource` 每次先按 manifest 权重选择一个 snapshot source，再请求一个完整
microbatch，因此继续保持 shard-local 读取。snapshot、annotation 模式、教师指纹、D4 和采样
算法版本共同构成 sampling identity；改变任一项都必须产生新的训练数据身份。

## 失败边界

- Catalog、分片、record 或内容哈希不一致时立即失败，不进行宽松修复。
- 旧 schema 只给出明确的重建错误，不自动迁移或删除产物。
- 单局解析错误可以成为结构化 rejection；资产 I/O、许可声明或完整性错误使整次导入失败。
- snapshot 只引用已封存、已校验的数据；采集中的半份任务不能进入训练 mixture。
