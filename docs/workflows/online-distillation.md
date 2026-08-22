# 主动选点与在线蒸馏

状态：未来实现。该阶段用学生真实访问的状态发现盲点，再让远程 KataGo 教师进行有限预算的
针对性辅导。它是数据选择和标注流程，不拥有训练模型或优化器。

## 目标循环

1. 固定 slow publication，运行 OpenSpiel MCTS 自博弈并封存 trajectory shards。
2. 对每个可查询 step 计算归一化学生策略熵及开/中/终局阶段。
3. 在教师查询预算内，70% 取最高熵局面，30% 分阶段随机抽取；做内容去重和每局限额。
4. 教师机领取持久任务，用当前 Human-SL/profile 与高预算搜索返回分布和价值摘要。
5. 校验指纹、预算、视角和响应哈希后写不可变 annotation shard，不重写基础 trajectory。
6. `BatchSource` 按生命周期配方混合 MCTS、教师和 rehearsal 数据，训练并发布下一版。

## 方法定位

学生生成状态分布、教师重标这些状态，属于 DAgger 风格状态聚合；把搜索改进策略蒸馏回
网络接近 Expert Iteration。熵高不一定等于最有教学价值，因此保留 30% 分层随机对照，并
记录选择分数、候选集合版本与抽样种子。

## 不变量

- 一盘棋不切换 publication；训练与采集默认不并发提交 CUDA 工作。
- 教师标签保存完整分布、视角、profile、实际预算和 `teacher_fingerprint`。
- 相同 `(game_id, ply)` 可存在多个教师版本；缺失或失败任务不覆盖旧标注。
- 教师样本不能完全替代 MCTS 与冷启动 rehearsal；实际混合比例进入 manifest。
- 任务和租约落 SQLite 控制面，训练大数组只进入 NPZ 分片。

初始混合比例见[统一训练生命周期](training-lifecycle.md)，远程任务语义见
[局域网教师协议](../integrations/lan-teacher.md)。
