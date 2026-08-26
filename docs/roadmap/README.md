# 路线图

状态只使用 `已完成`、`下一阶段`、`未来`，避免把已接受设计写成现有功能。

## 已完成：2026-08 文档化与依赖决策

- 当前 Trainer 已解耦为 `BatchSource`，并具有 EMA、checkpoint 和 publication。
- 保留本地 19×19 Tromp–Taylor 棋规、特征编码与批量推理契约。
- KataGo v1.17.2 保持教师/GTP；OpenSpiel v2.0.1 指定提交作为学生 MCTS 源码基线。
- 接受统一训练生命周期、序列优先 NPZ 分片和可选棋力评级设计。

## 已完成：2026-08 离线数据垂直切片

- g170 SGF manifest/import、版本化 trajectory/annotation NPZ 和 SQLite catalog。
- 内容寻址的不可变 snapshot、均匀 position 边际的 shard-local 采样、D4、标签 mask 与
  `CatalogBatchSource`。
- 单一 `Learner`、样本尺度 warmup/EMA/publication、有效 batch 4096 生产配置和数据身份恢复检查。
- Docker 合成闭环与首个 g170 归档 64 盘真实 smoke。

## 下一阶段一：序列数据扩展

- benchmark 全量随机读取、LRU 命中和 shard 大小，增加 compaction 与窗口监控。
- 实现连续子序列、burn-in 和整段统一 D4。
- 仅在获得可靠连接键后接入 KataGo rich NPZ；再评估 Leela 混合采样。

## 下一阶段二：扩大监督冷启动

- 执行全量 manifest/import，冻结正式 train/validation snapshot 并记录吞吐。
- 在 RTX 4090 上验证累积 256、有效 batch 4096 的长时稳定性与恢复。
- 实现只读 BF16 publication evaluator，并重新确认双模型驻留验收线。

## 下一阶段三：OpenSpiel AlphaZero

- 实现本地 `GameState` 的薄 `pyspiel.Game/State` 与自定义 Evaluator。
- 以 64 simulations 完成 MCTS 自博弈 tiny 闭环，再标定至约 100。
- 覆盖合法着、价值视角、根噪声、访问标签、终局与异常分片测试。
- 扩大采集前实现多棋局并发和统一 GPU 推理聚批。

## 下一阶段四：主动教师辅导

- 实现 SQLite 任务/租约、本机 Analysis adapter，再接入局域网教师 worker。
- 按 70% 高熵与 30% 分阶段随机选点，写独立 annotation shards。
- 标定教师 profile/visits、混合比例、400 局/Wilson 升级与可选 rating snapshots。

## 未来

- 数据窗口监控、故障恢复、更大教师集群与长期分级 publication。
- 历史/战略快权重和搜索摘要快权重；从 trajectory 重算状态，不保存模型相关快状态。
- 在固定预算下对比无快状态、更长历史、缓存和 MCTS 强度。

每阶段先完成可回放的 tiny 垂直切片，再扩大数据、搜索预算或并发度。
