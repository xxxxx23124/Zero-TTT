# MCTS 兼容性

## “纯直觉”的准确含义

学生默认批量计算 policy，然后直接采样或取最大概率动作；这是一种运行模式，不是网络能力
限制。模型同时给出 policy、value、ownership 和 score，因此未来搜索器可以通过统一的
`PositionEvaluator` 扩展节点。

项目已经删除自己的 PUCT/MCTS 实现。当前 KataGo 只搜索官方网络；要让 KataGo 搜索
Zero-TTT Transformer，未来必须实现自定义推理后端或可靠导出层，不能假设现有 checkpoint
格式可以直接加载。

## 搜索不变量

一次搜索由以下不可变元组标识：

```text
(base_model_version, fast_state_version, feature_schema, rules)
```

- 从根节点开始到搜索结束，基础权重和快权重都冻结。
- evaluator 不能在进行中的搜索里热切换 publication。
- 推理缓存和树节点必须包含上述版本；任一部分变化就作废。
- positional superko 依赖完整局面历史，缓存键不能只有棋盘哈希。
- value 和 score 必须保持当前行棋方视角，备份算法负责逐层转换。

## 树复用

无快权重时，未来搜索器可以在同一模型版本下研究合法的子树复用。有快权重时，每次真实
落子后的写入都会改变 evaluator；因此默认重新搜索并丢弃旧树。除非以后证明状态迁移语义
正确，否则不能跨快权重版本复用树或网络评价缓存。
