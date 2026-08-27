# `zero_ttt.search`

`search` 是本地棋规、`PositionEvaluator` 与 OpenSpiel Python MCTS 之间的薄适配层。项目固定
使用 OpenSpiel v2.0.1 的 `MCTSBot`，不采用其内置 Go 规则、模型、Learner 或 AlphaZero runner。

## 状态适配

`OpenSpielGoGame`/`OpenSpielGoState` 只把 clone、apply action、legal actions、current player、
returns 和终局查询转发给本地不可变 `GameState`。动作空间固定为 361 个交叉点加 pass，坐标
转换只有这一处实现。

本地状态仍唯一决定 Tromp–Taylor 合法着、历史、终局和计分。不能换用 OpenSpiel 内置 Go
状态，因为它的规则行为不是本项目契约。

`OpenSpielEvaluator` 从 [`inference`](../inference/README.md) 取得合法着 logits 和当前行棋方
value，只在合法动作上 softmax 为 prior，并把 value 转为 OpenSpiel 所需的双方数组。

## 搜索语义

`search_position` 负责调用 MCTS、读取根节点访问数并选择动作：

- 根访问分布归一化为训练 policy，网络 raw policy 只作为 prior；
- 根 Dirichlet 噪声、温度、simulation 数和种子来自显式搜索配置；
- value/score 的视角在 adapter 边界转换，不依赖隐式符号猜测；
- publication、特征 schema、规则和搜索配置共同构成 evaluator 身份。

上游 `MCTSBot.step` 每次创建新根，`restart_at` 为空操作，因此当前实现默认每着清树，不承诺
跨着树复用或旧搜索恢复。性能扩展应优先使用多棋局并发和共享 evaluator 聚批，而不是在本包
复制推理后端或棋规。

具体整盘采集、审计字段和 task 恢复见 [`selfplay`](../selfplay/README.md)。
