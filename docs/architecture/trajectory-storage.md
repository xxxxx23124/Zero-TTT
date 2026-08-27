# 序列化训练数据

状态：v3 trajectory/annotation、NPZ shard、SQLite catalog 和 snapshot 已实现，并兼容读取 v2。
[RLDS](https://github.com/google-research/rlds/blob/main/README.md) 的“episode 包含有序 steps”
语义，但不引入 TensorFlow 或 TFDS。“一盘一个序列”是逻辑边界，不等于一盘一个文件。

## 逻辑记录

`TrajectoryRecord` 表示一盘完整棋局。每个 step 是落子前状态，关联实际动作、标签与搜索
摘要。序列至少保留：

- `game_id`、schema、规则、贴目、`max_moves`、终止原因和结果；固定空棋盘黑方先行；
- 有序 moves、随机种子和黑白双方完整 `agent_id`；
- 每步搜索预算、稀疏访问分布、根 value/score、温度、根噪声及有效 mask；
- publication、特征 schema、搜索配置、标签来源和教师指纹；
- 来源许可、原文件哈希，或自博弈任务身份。

v3 自博弈记录额外保存 publication/agent/搜索配置身份、整局与逐步种子、实际 simulation 数、
根 value/score、温度及噪声/可用 mask。`root_noise_mask` 只在对应搜索实际使用非零
Dirichlet epsilon 时为真。新 shard 只写 v3；v2 内容哈希和旧 snapshot 身份按原算法读取，
不重写旧文件。

不把重复的 `25×19×19` float 特征当作权威数据。当前编码器从空棋盘、规则和 moves
确定性重建全部局面；未来可增加可删除的派生特征缓存。

## 物理分片

- 多盘棋拼入只读 NPZ shard；offset 数组标记各局 moves、position 和稀疏 policy 的边界。
- 只使用具名 NumPy 数组并以 `allow_pickle=False` 读取；数组 schema 独立版本化。
- 目标上限约 128 MiB 未压缩数据；到限后原子封存、计算 SHA-256，再登记为可读。
- 基础 trajectory 与教师 annotation 分片分离；后者以
  `(game_id, ply, teacher_fingerprint)` 连接，允许多教师版本共存。
- trajectory 与 annotation 都有逻辑内容 SHA-256；snapshot 哈希这些内容身份而非 shard 路径，
  因而内容变化必然产生新 snapshot，纯物理重分片不会改变 snapshot ID。

NPZ 与现有 NumPy/PyTorch 栈直接兼容；其基本语义见
[NumPy NPZ](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)。KataGo 的
[自博弈训练说明](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)也采用
NPZ 分片，但本项目 schema 独立。

## SQLite 控制面与窗口

SQLite 只保存分片路径、game/step offset、长度、来源、身份、任务、租约和校验信息，不保存
大型训练 BLOB。replay 按总 step 数与磁盘字节数维护滑动窗口，只以完整棋局为单位淘汰；
冷启动锚点、重要旧阶段 rehearsal 和正式评测棋局可标记 `pinned`。

## 棋力元数据

棋力不是棋局的稳定属性，不设置必填 `strength`。黑白与教师记录模型、publication、搜索
预算和配置哈希；课程阶段与 Human-SL profile 只是类别标签。人类数据保留来源声明的 rank、
平台和评级体系，不自动换算 Elo。

未来正式 arena 可独立产生 `RatingSnapshot`：评级池、agent、时间/版本、rating、误差或 RD、
有效局数与评测配置。历史 trajectory 最多引用生成时已有的 snapshot；评级更新不重写分片。
[Glicko](https://www.glicko.net/glicko/glicko.pdf) 可表达不确定性，但并非已选定实现。未知评级
不阻塞入库、采样或训练。

## 序列训练兼容

普通训练可随机抽取单个局面；快权重训练抽取不跨局的连续 `[burn-in + train]` 子序列。
burn-in 只恢复当前模型快状态，不计算 loss；快状态本身不落盘，因为依赖具体模型版本。
每段序列增强只能使用同一个 D4 变换，不能逐 step 独立旋转。

## 已覆盖与后续验收

- 已覆盖 moves 重建、合法着、按 position 权重的 shard-local microbatch、D4、annotation 精确
  命中和逐标签 mask；每个 position 的边际抽样概率均匀。
- 已覆盖 schema/checksum、原子 shard、半写临时文件恢复、snapshot 引用保护与 whole-shard GC。
- v1 catalog/shard 不迁移；v2 catalog 原地做只增加表/列的 v3 迁移，v2 shard 保持只读兼容。
- 连续子序列、burn-in、跨 shard compaction 和长期窗口监控仍是后续工作。

提交 `e2b3017` 的整盘 `GameRecord`、棋步重放和缓存仅作行为参考；不恢复“每盘 NPZ BLOB
直接塞入 SQLite”的物理设计。
