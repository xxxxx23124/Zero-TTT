# 学生自博弈采集

状态：已接受的目标设计，尚未实现。最近实现先做单线程、推理 batch size 1 的纯策略版本；
本页不恢复旧 CoreLoop、replay schema 或 MCTS 源码。

## 纯策略首版

1. 加载一个不可变 BF16 slow publication，并固定 evaluator 身份。
2. 用本地 `GameState` 维护 19×19 Tromp–Taylor 棋局、合法着、历史和终局。
3. 每步编码局面，执行一次 batch size 1 推理，按运行 manifest 的固定选着规则落子。
4. 整盘不得切换 publication；保存棋步、模型版本、随机种子、选着参数和终局标签。
5. 原始棋局与派生样本写成不可变分片，之后才由 `BatchSource` 读取或提交教师标注。

纯策略自博弈首先用于发现学生实际访问的状态。把学生自己的 raw policy 再当作改进标签不会
凭空增强策略；默认由最终结果提供 value/score/ownership 标签，policy 改进来自后续 KataGo
标注或可选的本地 MCTS 访问分布。

## 可选本地 MCTS

未来以提交 `e2b3017` 中的旧搜索代码和测试为行为参考，选择性重写：

- 首版只做单线程、batch size 1、固定访问数的 PUCT/FPU；
- 自博弈根节点可加 Dirichlet 噪声，访问次数归一化后作为 policy 标签；
- 暂不恢复线程池、虚拟损失、推理拼批、动态预算、旧 SQLite replay 或 CoreLoop；
- 搜索只依赖 `GameState` 与 `PositionEvaluator`，不引用 Learner。

一次搜索的身份为：

```text
(base_model_version, fast_state_version, feature_schema, rules)
```

身份任一部分变化就清空树和评价缓存。冻结 publication 且没有快权重变化时，可以把实际落子
对应的合法子节点提升为新根；快权重每步写入时默认不复用树或缓存。

## 验收边界

首版至少覆盖完整 tiny 棋局、非法着屏蔽、两次 pass/手数上限、固定种子复现、publication
版本失配和异常不产生半份数据。未来 MCTS 另测价值符号、访问分布、根噪声以及树复用失效。
