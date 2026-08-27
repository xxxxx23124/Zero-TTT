# `zero_ttt.training` 与 Learner

`zero_ttt.learner.Learner` 是唯一优化实现；`training` 包提供损失、梯度、EMA、checkpoint、
publication 和可恢复应用服务。二者只消费 `BatchSource`/`TrainBatch`，不知道样本来自 SGF、
自博弈还是教师。跨模块职责见[模型与训练](../../../docs/architecture/model-and-training.md)。

## 优化边界

每个 `TrainBatch` 在进入 Learner 前已经完成规则解释、视角转换、合法着处理、标签归一化和
数据增强。value、ownership、score 的缺失由逐样本布尔 mask 表示，`compute_losses` 不能把
缺失标签当作有效零值。

Learner 独占训练模型、AdamW、梯度缓冲、CPU FP32 slow/EMA、随机状态和样本尺度调度：

- GPU fast 参数、梯度和优化器状态保持 FP32，BF16 只用于 autocast 矩阵计算；
- 梯度按模型声明的互斥参数组裁剪，非有限梯度使该次提交失败；
- EMA、warmup、checkpoint 和 publication 边界按 `samples_seen` 定义；
- microbatch 或累积步数变化不能偷偷改变样本时间尺度。

模型结构与参数组属于 [`model`](../model/README.md)，采样身份属于
[`data`](../data/README.md)。

## 应用服务与产物提交

`TrainingSession` 统一 Learner 构造、RNG、严格恢复、optimizer step 和发布触发。普通 restore
必须精确匹配训练数据身份；只有显式数据阶段切换才允许保留训练状态并迁移到新的不可变
snapshot/mixture 身份。

`CheckpointManager` 负责原子文件、保留策略与当前 schema 读取。`ArtifactCoordinator` 负责：

- 校验 checkpoint、publication、run、step、samples 和配置身份；
- 从最新完整 checkpoint 补发缺失或落后的 slow/EMA publication；
- 幂等登记 publication，并修复 checkpoint 中的发布边界；
- 拒绝 publication 超前、同 step 身份冲突或跨 run 混用。

发布顺序是“保存完整 checkpoint → 写不可变 publication → 登记 publication → 再保存已更新
发布边界的 checkpoint”。控制台和 CLI 必须复用这些应用服务，不能各自实现恢复协议。

## 失败边界

- checkpoint、publication 和嵌入配置只接受中央登记的当前版本并严格加载。
- 数据身份不一致时默认失败；迁移必须由显式工作流发起并留下记录。
- Learner 不解析 SGF/KataGo JSON，不运行游戏、MCTS、教师队列或控制台状态机。
- 采集器与 evaluator 只通过不可变 publication 和持久数据交接，不共享模型地址。
