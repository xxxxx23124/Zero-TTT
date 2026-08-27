# OpenSpiel MCTS 边界

状态：首版运行时 adapter、publication evaluator 与 batch-16 自博弈已实现。项目采用 v2.0.1 的 Python
[`MCTSBot`](https://github.com/google-deepmind/open_spiel/blob/v2.0.1/open_spiel/python/algorithms/mcts.py)
与自定义 Evaluator，不采用 OpenSpiel 内置模型、Learner 或完整 AlphaZero runner。

源码位于 `third_party/open_spiel`，固定提交
`112b77704631fc2ce7ad8e4581f6ca09798ce15a`。Docker 从该源码构建 `pyspiel`；Abseil、JSON、
pybind 和 bridge DDS 依赖也锁定到 Dockerfile 中的具体提交。

## 状态与 evaluator 适配

- 本地 `GameState` 继续唯一维护合法着、完整历史、终局与 Tromp–Taylor 计分。
- 薄 `pyspiel.Game/State` 只转发 clone、apply_action、legal_actions、current_player、returns
  和终局查询，不重新实现棋规。
- `OpenSpielGoGame/State` 转发本地不可变状态；自定义 Evaluator 用 `PositionEvaluator` 取得合法着 logits 和当前行棋方 value；logits 只在
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

采集器以 16 个棋局线程并发推进，共享一个阻塞 evaluator broker；请求最多等待 2 ms，真实
请求去重后由 publication 后端补齐到固定 batch 16。尾批的 padding 不进入返回结果或统计中的
真实 evaluation 数。`selfplay-collect` 输出有效/补齐比例、满批比例、平均/最大推理延迟、
simulations/s、棋规 CPU 总耗时和 CUDA peak allocated bytes。Python 游戏和棋规仍可能成为吞吐
瓶颈，首版不预先重写 C++ 棋规。
上游自定义游戏示例也提示 Python 游戏执行 MCTS 会较慢，见
[`tic_tac_toe.py`](https://github.com/google-deepmind/open_spiel/blob/v2.0.1/open_spiel/python/games/tic_tac_toe.py)。

提交 `e2b3017` 的旧 PUCT 仅保留为测试行为参考，不恢复其搜索、CoreLoop 或 replay 代码。
