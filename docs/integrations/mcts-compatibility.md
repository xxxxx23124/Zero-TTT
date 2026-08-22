# OpenSpiel MCTS 边界

状态：OpenSpiel 源码已固定，运行时 adapter 尚未实现。项目采用 v2.0.1 的 Python
[`MCTSBot`](https://github.com/google-deepmind/open_spiel/blob/v2.0.1/open_spiel/python/algorithms/mcts.py)
与自定义 Evaluator，不采用 OpenSpiel 内置模型、Learner 或完整 AlphaZero runner。

源码位于 `third_party/open_spiel`，固定提交
`112b77704631fc2ce7ad8e4581f6ca09798ce15a`。子模块用于版本审计和未来集成，本轮不把它
加入 Docker 运行时安装步骤。

## 状态与 evaluator 适配

- 本地 `GameState` 继续唯一维护合法着、完整历史、终局与 Tromp–Taylor 计分。
- 薄 `pyspiel.Game/State` 只转发 clone、apply_action、legal_actions、current_player、returns
  和终局查询，不重新实现棋规。
- 自定义 Evaluator 用 `PositionEvaluator` 取得合法着 logits 和当前行棋方 value；logits 只在
  合法动作上 softmax 为 prior，value 转为 OpenSpiel 需要的双方数组。
- 动作空间固定为 361 个交叉点加 pass；坐标转换只有一个受测实现。

不使用 OpenSpiel 内置 Go 状态，因为其多子自杀和 superko 行为与当前契约不一致。相关实现
见上游 [`go.cc`](https://github.com/google-deepmind/open_spiel/blob/v2.0.1/open_spiel/games/go/go.cc)。

## 搜索默认值

- AlphaZero 自博弈初始 64 simulations，并标定至最多约 100。
- 根加入 Dirichlet 噪声；子节点访问数归一化后成为 policy 标签。
- publication、特征 schema、规则、搜索配置和未来快状态版本组成 evaluator 身份。
- value/score 的存储视角明确；OpenSpiel 备份使用双方 value，不依赖隐式符号猜测。

上游 `MCTSBot.step` 每次都创建新根，`restart_at` 也是空操作，因此默认每着清树，不维护
动态清空 API 或复杂树复用。publication/快状态变化时，外部 evaluator cache 仍须按完整
身份隔离。

## 性能约束

Python 自定义游戏适合先做 tiny 垂直切片，但逐局 batch 1 会浪费 GPU。扩大采集前必须实现
多个棋局并发推进与统一 evaluator 聚批；这属于项目 adapter/采集器，不是重新实现 MCTS。
上游自定义游戏示例也提示 Python 游戏执行 MCTS 会较慢，见
[`tic_tac_toe.py`](https://github.com/google-deepmind/open_spiel/blob/v2.0.1/open_spiel/python/games/tic_tac_toe.py)。

提交 `e2b3017` 的旧 PUCT 仅保留为测试行为参考，不恢复其搜索、CoreLoop 或 replay 代码。
