# Docker 训练控制台

控制台是 `src/zero_ttt/console/` 中的纯编排层。它复用现有 Learner、不可变 snapshot/mixture、
publication 和自博弈采集器，不管理学习率、梯度累积、MCTS 或模型结构。

## 首次配置

编辑 `configs/console.toml`：

- `experiment_config` 相对控制台 TOML 所在目录解析，默认指向 `rtx4090l.toml`；
- `catalog_path` 和 `store_root` 默认使用 Compose 的 `/datasets/work` 命名卷；
- `cold_start_snapshot_id` 必须替换为已经创建并验证的 64 位 train snapshot ID；
- `max_runtime_hours` 默认 8 小时，每次选择采集、训练或 warm-start 时重新计时。

占位的全零 snapshot 会被配置校验拒绝。控制台不会猜测监督数据集，也不会自动导入原始数据。

## 启动与菜单

```bash
docker compose run --rm training-console
```

启动时控制台锁定实验 `run_dir`，校验 catalog/cold snapshot，并从最新完整 checkpoint 恢复阶段。
产物协调器按 run、step、samples 和完整配置身份核对 checkpoint/publication；如果 publication
缺失、较旧或属于其他 run，会用 checkpoint 的 slow/EMA 权重补发，并幂等修复 catalog 登记和
checkpoint 中的发布边界。相同 run 的 publication 超前或同 step 冲突不会自动回退。菜单包括：

1. 刷新状态；
2. 收集数据；
3. 开始训练；
4. warm-start；
5. 退出。

状态中的“未纳入最新训练 snapshot”是集合差：它表示尚未进入当前 checkpoint 数据身份的完整
自博弈棋局，不表示随机采样已经逐一训练过 snapshot 内的所有棋局。
其中 games、positions、pending 和新建 mixture snapshot 只统计已 `sealed` 的自博弈 task；
`collecting`/`failed` task 仍显示任务数，但不会进入训练数据。

## 软停止和阶段切换

- 采集以 `selfplay.actor_count` 个并发完整棋局为一轮；到时后完成当前轮、封存 shard 和 task，
  再返回菜单。
- 训练到时后完成当前 optimizer step，原子保存 checkpoint，强制发布 slow/EMA 权重，再保存
  已更新 publication 边界的 checkpoint。
- 首次 SIGINT/SIGTERM 与到时使用相同的软停止路径；异常则记录 `FAILED`，下次启动验证产物后
  恢复 `READY`。
- warm-start 只在冷启动完整 checkpoint 和自博弈数据都存在时开放。它保留权重、optimizer、
  RNG、step、samples 和样本尺度调度，建立 80% self-play / 20% cold rehearsal mixture，并在
  至少一个新 step 成功保存后提交 `MIXTURE` 阶段。
- Mixture 阶段每次训练都会从当前全部自博弈完整棋局创建新 snapshot；身份未变则严格 resume，
  有新棋局则执行显式数据身份迁移。

控制台内部把产物恢复和 catalog 登记集中在 artifact coordinator，把 cold/mixture 数据源构造
集中在 training-data planner；交互菜单只协调状态转换和执行边界。

控制台状态写入 `runtime.run_dir/console/state.json`，mixture manifest 写入同目录的 `mixtures/`。
模型与数据产物仍由原有 checkpoint、publication、shard 和 catalog 负责，控制台状态丢失时不得
替代这些事实源。
