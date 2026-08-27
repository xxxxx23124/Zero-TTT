# MCTS 学生自博弈采集

状态：首版手动分阶段的可训练闭环已实现。默认 AlphaZero 数据由 OpenSpiel Python MCTS 生成；
纯策略对局只作为诊断、人机对战或消融，不作为默认自我改进路径。

## 单盘流程

1. 加载不可变 BF16 slow publication，整盘冻结 evaluator 与搜索配置。
2. 用本地 `GameState` 维护 Tromp–Taylor 合法着、完整历史、终局与计分。
3. 通过薄 `pyspiel.Game/State` 适配层让 `MCTSBot` 搜索；初始预算为每步 64 simulations，
   标定后可提高到约 100。
4. 自博弈根节点加入 Dirichlet 噪声，前 30 ply 以温度 1、之后温度 0 从访问数选着；访问分布成为 policy 标签。
5. 终局结果为所有有效 step 提供 value；可用的 score/ownership 按 mask 保存。
6. 把整盘 moves、每步搜索摘要和完整身份写入逻辑 `TrajectoryRecord`，原子封存分片后才训练。

学生 raw policy 只作为 MCTS prior，不能再当作自身的改进标签。每步实际搜索预算、根 value、
访问分布、温度、噪声和随机种子都必须可审计。

## 搜索与性能边界

- OpenSpiel 每次 `step` 从新根开始；首版不维护跨着复杂树复用，也不恢复旧 Python MCTS。
- 一次搜索内 publication、特征 schema、规则与快状态身份固定。
- 默认 16 个棋局线程共享 evaluator 队列；GPU publication 后端固定补齐 batch 16。
- publication 或未来快权重版本变化时，任何 evaluator cache 都必须隔离或丢弃。

## 验收边界

首版测试至少覆盖：完整 tiny 棋局、非法着屏蔽、两次 pass/手数上限、价值视角、根噪声、
访问分布、固定种子、版本失配、异常不产生半份分片，以及 moves 无损重建全部状态和终局。

搜索适配细节见 [OpenSpiel MCTS](../integrations/mcts-compatibility.md)，物理数据设计见
[序列化训练数据](../architecture/trajectory-storage.md)。

命令入口为 `zero-ttt selfplay-collect`。任务 manifest、稳定 game ordinal 和 catalog 去重使已提交
分片可在中断后复用；训练通过 source-filter snapshot 和 mixture manifest 显式接入，不直接把
采集器对象传给 Learner。
