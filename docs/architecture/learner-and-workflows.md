# Learner 与流程边界

状态：Learner、离线数据、OpenSpiel 自博弈闭环和交互式控制台已实现；教师队列、无人值守自动循环和课程调度仍未实现。

## 组件职责

| 组件 | 唯一职责 |
| --- | --- |
| `model` | 定义纯 `nn.Module`、前向输出和可关闭的模型实验 |
| `learner` | 监督优化、EMA、compile 与梯度提交 |
| `training` application service | learner 生命周期、checkpoint、publication 与 catalog 登记 |
| `inference` | 从不可变 publication 构造只读 `PositionEvaluator` |
| `data` | trajectory/annotation 校验、混合和 `BatchSource` |
| OpenSpiel adapter | 把本地棋局和 evaluator 接到 Python MCTS |
| 采集流程 | 棋谱导入、MCTS 自博弈、KataGo 标注和课程调度 |
| KataGo adapter | JSON/坐标/视角与 annotation 之间的转换 |

`Learner` 不包含游戏、搜索、网络协议或调度；输入仍只有 `BatchSource`/`TrainBatch`。采集器
不得持有训练模型、优化器或 EMA 的 Python 引用。具体实现分别见
[训练包](../../src/zero_ttt/training/README.md)与
[自博弈包](../../src/zero_ttt/selfplay/README.md)。

## 数据交接

一盘棋先成为不可变 `TrajectoryRecord`；后来的教师结果成为独立 `AnnotationRecord`。
SQLite 只承担索引、任务、租约和校验控制，不保存大型训练 BLOB。进入 Learner 前，数据源
按运行 manifest 的比例采样局面或连续子序列，并归一化为 `TrainBatch`。

每份可训练数据至少可追溯：

- 来源许可、规则、初始状态、动作序列、终局结果和随机种子；
- 黑白 agent、publication、特征与搜索配置身份；
- 每步标签来源、搜索预算、视角、有效 mask 和教师指纹；
- 原始文件哈希或自博弈任务身份，以及分片 SHA-256。

失败、缺失必需身份或校验不一致的数据不能静默混入。完整规范见
[序列化训练数据](../../src/zero_ttt/data/trajectory-storage.md)。

## 模型与显存所有权

只读 evaluator 从 FP32 slow publication 加载独立模型，不访问 Learner 内部地址。训练和采集
按阶段分别创建 Learner 或 evaluator，不并发提交 CUDA 工作，也不以双 GPU 模型同时常驻作为
支持目标。

严格 FP32 的正式 batch-16 冒烟按阶段验收：默认模型 eager/compiled 训练峰值分别为
`13.127/13.307 GiB reserved`，推理为 `3.979/4.295 GiB`；baseline eager 训练/推理为
`11.766/3.053 GiB`。生产配置仍累积 256，单阶段继续遵守 14.5 GiB 验收线。

## 流程组合

协调器只传 publication 路径/版本、分片 manifest 和阶段结果。整盘 publication 在采集期间
冻结；采集封存后才训练并发布下一版。多棋局并发、统一 GPU 推理聚批是扩大采集前的性能
要求，但不会改变上述职责边界。

Docker 控制台位于独立 `zero_ttt.console` 包中；底层模块不导入它。普通 checkpoint restore
严格绑定数据身份，只有显式滚动不可变 snapshot 时才允许保留完整训练状态的数据迁移。
checkpoint、Catalog 和不可变产物始终是事实源；控制台状态只用于编排与审计。内部状态机和
恢复规则见[控制台包说明](../../src/zero_ttt/console/README.md)。
