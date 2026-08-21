# 公共契约

代码接口位于 `zero_ttt.data` 和 `zero_ttt.inference`。它们是当前骨架，不代表真实数据工作流、
持久样本格式或目标 `zero_ttt.learner` 已完成。

## TrainBatch

批量维度为 `B`：

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| `board` | `B×25×19×19` | 当前行棋方特征 |
| `global_features` | `B×5` | 贴目、手数和行棋方 |
| `legal` | `B×362` | Tromp–Taylor 合法着 |
| `policy` | `B×362` | 合法着上的归一化教师分布 |
| `value` | `B` | 当前行棋方结果 |
| `ownership` | `B×361` | 可选归属标签 |
| `score_margin` | `B` | 可选目差标签 |
| `*_mask` | `B` | 辅助标签是否有效 |

`BatchSource` 只承诺 `next_batch`。离线棋谱、在线教师和学生自博弈将分别通过持久分片实现该
协议。分片 manifest 必须保留来源、规则、publication/教师指纹、搜索预算、标签视角和 mask；
具体持久 schema 在实现前另行版本化，本轮不把草案字段固化成 Python 公共类型。

## PositionEvaluator

`InferenceBatch` 接收同一特征张量与合法着掩码；`InferenceOutput` 返回 policy logits、value，
并可返回 ownership/score。实现必须暴露不可混淆的 `model_version`。

纯策略调用者直接按运行 manifest 的选着规则使用 policy。本地 MCTS 可读取 policy/value
扩展节点，因此不需要改变模型结构。一次搜索中的完整 evaluator 身份必须保持不变。

## 教师数据

局域网任务和结果暂时只有[文档协议](../integrations/lan-teacher.md)，本轮没有 Python 类型、
数据库表或网络 API。KataGo adapter 负责坐标、JSON 和视角转换，再写持久分片；Learner
永远不接触原始协议。
