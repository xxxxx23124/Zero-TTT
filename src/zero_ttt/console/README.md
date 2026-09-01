# `zero_ttt.console`

`console` 是 Docker 交互式训练的编排层。用户配置和操作见
[训练控制台指南](../../../docs/operations/training-console.md)；本文只说明内部状态、恢复和阶段
切换约束。

## 状态与单进程所有权

`ConsoleState` 区分训练阶段 `COLD_START`/`MIXTURE` 与操作状态 `READY`、`COLLECTING`、
`TRAINING`、`WARM_STARTING`、`SOFT_STOPPING`、`FAILED`。所有状态变化必须通过显式转移表，
并由 `StateStore` 原子写入当前 schema 的 JSON。

`ConsoleLock` 保证同一 run 同时只有一个控制台编排者。状态文件只服务菜单和审计；checkpoint、
publication、Catalog、snapshot 和 shard 仍是事实源。

## 启动恢复

`TrainingConsole.reconcile()` 先验证 cold snapshot 和存储路径，再由
`ArtifactCoordinator` 检查最新完整 checkpoint/publication：

- 阶段从 checkpoint 的数据身份推导，不盲信 console JSON；
- publication 缺失或落后时从 checkpoint 的 slow/EMA 补发并登记；
- publication 超前、同 step 冲突、跨 run 或配置不一致时失败；
- 上次中断或 `FAILED` 只有在事实源重新验证后才回到 `READY`。

## 数据规划与阶段切换

`TrainingDataPlanner` 在 cold-start 阶段构造单 snapshot `CatalogBatchSource`；mixture 阶段从全部
已 sealed 自博弈数据创建新 snapshot，并按 `configs/console.toml` 的 `mixture` 权重写不可变
mixture manifest；当前模板是 80% self-play / 20% cold rehearsal。

数据身份未变时严格 resume；新自博弈棋局导致 snapshot 变化时，只能通过
`restore_for_data_transition` 保留模型、优化器、RNG、step 和 samples，并写 `MigrationRecord`。
warm-start 必须同时存在完整 cold-start checkpoint 和至少一盘 sealed 自博弈数据，且至少成功
完成一个新 optimizer step 后才提交 `MIXTURE` 阶段。

## 软停止

`RuntimeBudget` 使用单调时钟；首次 SIGINT/SIGTERM 只设置协作式停止标志。

- 采集在当前 `actor_count` 整盘轮次结束、分片封存且 task sealed 后停止；
- 训练在当前 optimizer step 结束后保存 checkpoint、强制发布并再次保存发布边界；
- 未开始任何 optimizer step 时不提交阶段迁移；
- 未处理异常转入 `FAILED`，不把控制台状态伪装成成功产物。

控制台只调用 [`training`](../training/README.md)、[`selfplay`](../selfplay/README.md) 和
[`data`](../data/README.md) 的应用服务，底层包不得反向导入 `console`。

## Web 控制边界

`zero_ttt.control` 只通过独立子进程调用非交互 console 命令，并转发 SIGTERM 请求软停止；
`zero_ttt.dashboard` 只消费 control API。两者都不得进入 Learner、搜索或数据实现。TensorBoard
适配器消费结构化事件，UI 轮询代理缓存，不轮询加载完整 checkpoint。
