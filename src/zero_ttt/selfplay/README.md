# `zero_ttt.selfplay`

`selfplay` 负责用冻结 publication 运行并发 MCTS 对局，把完整搜索轨迹封存为可恢复的训练数据。
它不持有训练模型、优化器或 EMA 的 Python 引用。

## 应用服务与单盘流程

`SelfPlayService` 从一份不可变 publication 创建 evaluator，使用 publication、特征、规则和搜索
配置计算完整 evaluator identity，并拥有 `BatchedInferenceBroker` 生命周期。相关实现边界见
[`inference`](../inference/README.md)与[`search`](../search/README.md)。

`SelfPlayCollector` 对每盘棋执行：

1. 用本地 `GameState` 和 OpenSpiel adapter 从空棋盘推进至终局；
2. 每步运行 MCTS，保存实际 simulation 数、访问分布、根 value/score、温度、噪声 mask 和种子；
3. 以访问分布作为 policy，以终局结果作为 value，并保存可用的 ownership/score；
4. 构造完整 `TrajectoryRecord`，经共享 `TrajectoryShardSink` 原子封存；
5. 全部请求棋局完成后才把 task 标记为 `sealed`。

整盘期间 publication、规则、特征和搜索配置保持冻结。学生 raw policy 不能再次作为自身的改进
标签。

## 幂等恢复与并发

self-play task ID 哈希 publication、evaluator、搜索/游戏/自博弈配置、请求局数和基础种子；
manifest 已存在时必须逐字节一致。game ID 由 task ID 和稳定 ordinal 派生，每步搜索与选着种子
也从这些身份确定性派生。

恢复时 Catalog 跳过已存在 game ID，只采集缺失 ordinal。每轮最多以 `actor_count` 个线程并发
推进完整棋局，共享一个 inference broker；完成记录按 ordinal 交给共同 shard sink。

异常时 task 标记为 `failed`，已原子提交的完整棋局可供下一次同身份任务复用；未封存内存记录
不会进入 Catalog。只有 `sealed` task 能进入 source-filter snapshot 和训练 mixture。物理存储
与可见性规则见[数据持久化](../data/trajectory-storage.md)。

## 验收边界

测试应覆盖终局、非法着屏蔽、pass/手数上限、价值视角、根噪声、访问分布、固定种子、版本
失配、异常不产生半份分片，以及 moves 无损重建全部局面。采集摘要还应报告 simulations/s、
棋规耗时、批处理利用率、推理延迟和 GPU peak allocated bytes。
