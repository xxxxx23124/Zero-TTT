# 序列化训练数据

当前 trajectory、annotation、NPZ shard、SQLite Catalog 与 snapshot 均为 v4，只接受中央
版本表登记的当前版本。“一盘一个序列”是逻辑边界，不代表一盘一个文件。

## 逻辑记录

`TrajectoryRecord` 表示一盘完整、有序的棋局；每个 step 是落子前状态，关联实际动作、标签和
搜索摘要。记录至少包含规则、贴目、`max_moves`、终止原因、结果、moves、随机种子、双方
agent、publication/特征/搜索身份、来源和逐标签 mask。

当前记录只接受空棋盘黑方先行，并从 moves 确定性重建局面。重复的 `25×19×19` float 特征不
是权威数据。`AnnotationRecord` 以 `(game_id, ply, teacher_fingerprint)` 连接基础 trajectory，
允许多个教师版本共存而不重写原始棋局。

trajectory 与 annotation 都计算逻辑内容 SHA-256。snapshot 哈希逻辑内容身份而非文件路径，
因此内容变化必须产生新 snapshot，纯物理重分片则不改变逻辑身份。

## NPZ 分片

- 多盘记录拼入只读 NPZ shard，offset 数组标记 game、position 和稀疏 policy 的边界。
- codec 只使用具名 NumPy 数组，并始终以 `allow_pickle=False` 读取。
- `TrajectoryShardSink` 是导入和自博弈共同使用的累计、估算与封片入口。
- `ShardStore` 先原子提交文件并计算 SHA-256，再把已封存 shard 交给 Catalog 登记。
- trajectory 与 annotation 分片分离；读取时统一检查 schema、object array、文件哈希和记录身份。

目标 shard 上限约为 128 MiB 未压缩数据。codec 拥有数组 schema，store 不理解 record 字段，
调用方也不能绕开验证读取任意 NPZ。

## SQLite 控制面与恢复

SQLite 保存来源、分片路径、game/step offset、snapshot、任务、租约和校验信息，不保存大型
训练 BLOB，也不为每个 position 建行。`Catalog` 内部职责划分见[数据包说明](README.md)。

GC 必须先在事务中写 shard tombstone，提交后才删除文件；如果进程在两步之间退出，下一次
`recover()` 完成删除。恢复只清理超过安全时限的 `.shard-*.tmp`，不会删除仍可能属于活跃
writer 的临时文件。被 snapshot 引用的 shard 不得回收。

自博弈 task 只有在全部请求棋局完成并进入 `sealed` 后才对训练 snapshot 可见。失败或采集中的
task 可以保留用于恢复与审计，但不能贡献 games 或 positions。

## 采样、序列与版本边界

普通训练随机抽取单个局面；具体的 shard-local 算法见[采样说明](README.md)。未来快权重训练
可以抽取不跨局的 `[burn-in + train]` 连续子序列，且整个序列只能使用一个 D4 变换。快状态
依赖模型版本，不写入分片。

棋力不是棋局的稳定属性。trajectory 保存生成时的 agent、publication 和搜索配置；未来评级
应写独立 `RatingSnapshot`，评级更新不得重写历史分片。

新写入方必须从 `zero_ttt.versioning` 取得当前版本，读取方必须严格校验。旧 record、shard、
Catalog、snapshot 或 mixture 不迁移、不自动删除，只能由原始资产重新导入或重新采集。
