# 公共契约

数据与推理接口位于 `zero_ttt.data` 和 `zero_ttt.inference`；模型的稳定导入面位于
`zero_ttt.model`。Learner 与首版持久 schema 已实现；OpenSpiel adapter 和教师服务仍未实现。

## TrainBatch

批量维度为 `B`：

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| `board` | `B×25×19×19` | 当前行棋方特征 |
| `global_features` | `B×5` | 贴目、手数和行棋方 |
| `legal` | `B×362` | Tromp–Taylor 合法着 |
| `policy` | `B×362` | 合法着上的归一化目标分布 |
| `value` | `B` | 当前行棋方结果 |
| `ownership` | `B×361` | 可选归属标签 |
| `score_margin` | `B` | 可选目差标签 |
| `value_mask` | `B` | value 是否有效 |
| `ownership_mask` | `B` | ownership 是否有效 |
| `score_mask` | `B` | score margin 是否有效 |

policy 来源可以是人类实战落子、学生 MCTS 访问分布或教师搜索分布；必须通过来源与 mask
区分，学生 raw policy 不作为自身的改进标签。

## PositionEvaluator

`InferenceBatch` 接收相同特征张量与合法着掩码；`InferenceOutput` 返回 policy logits、value，
并可返回 ownership/score。实现必须暴露不可混淆的 `model_version`。

未来 OpenSpiel Evaluator 将合法着 logits 归一化为 prior，并把“当前行棋方”value 转成双方
value 数组。一次搜索内 publication、特征 schema、规则和未来快状态版本必须冻结。

## 持久逻辑契约

以下名称已作为版本化 Python 类型实现：

- `TrajectoryRecord`：一盘完整、有序、从空棋盘和 moves 确定性重放的棋局；v2 明确不接受
  setup/handicap/initial-position，且持久化本局 `max_moves`。
- `AnnotationRecord`：以 `(game_id, ply, teacher_fingerprint)` 连接的可追加教师标签。
- `RatingSnapshot`：可选、评级池相关、带误差或 RD 的 agent 评测结果。

物理格式、必备身份和淘汰规则见[序列化训练数据](trajectory-storage.md)。局域网任务与结果仍
只有[文档协议](../integrations/lan-teacher.md)；Learner 永远不接触 KataGo 原始 JSON。
