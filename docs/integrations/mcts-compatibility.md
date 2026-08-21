# 本地 MCTS 兼容性

状态：已接受的未来方向，当前没有搜索源码。学生最近的自博弈实现仍是单线程、batch size 1
的纯策略版本；MCTS 不阻塞该里程碑。

## 边界

KataGo 只搜索其官方网络，项目不修改 KataGo 或导出 Transformer 供其加载。未来的学生搜索
由 Zero-TTT 本地实现，通过统一 `PositionEvaluator` 读取 policy/value，继续复用本地
`GameState` 的规则、历史和合法着。

提交 `e2b3017` 中曾实现 Python PUCT、推理队列、缓存、动态预算和 CoreLoop。未来只把旧代码
与测试当作行为参考，先选择性重写单线程、batch size 1、固定访问数的 PUCT/FPU、根噪声与
访问次数 policy 标签；不直接恢复旧 replay/CoreLoop，也不在首版恢复并发设施。

## 搜索身份

一次搜索由以下不可变元组标识：

```text
(base_model_version, fast_state_version, feature_schema, rules)
```

- 从根节点开始到搜索结束，基础权重和快权重都冻结。
- evaluator 不能在进行中的搜索里热切换 publication。
- 推理缓存和树节点必须包含完整身份；任一部分变化就作废。
- positional superko 依赖完整局面历史，缓存键不能只有棋盘哈希。
- value 和 score 保持当前行棋方视角，备份算法负责逐层翻转 value。

## 树复用

冻结 publication 且没有快权重变化时，可以研究把实际合法落子的子节点提升为新根。启用
逐步写入的快权重后，每次真实落子都会改变 evaluator，默认丢弃旧树和网络评价缓存。除非
以后证明状态迁移语义正确，否则不能跨快权重版本复用。

详细采集顺序和验收见[学生自博弈](../workflows/student-selfplay.md)。
