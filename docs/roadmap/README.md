# 路线图

状态只使用 `已完成`、`下一阶段`、`未来`，避免把已接受设计写成现有功能。

## 已完成：2026-08 文档化重构

- 当前 Trainer 已解耦为 `BatchSource`，并具有 EMA、checkpoint 和 publication。
- 保留本地 19×19 Tromp–Taylor 棋规、特征编码与批量推理契约。
- 删除旧搜索型 replay/CoreLoop；固定 KataGo v1.17.2 教师/GTP Docker 骨架。
- 接受 Learner、学生自博弈、分级教师和可选本地 MCTS 的新目标架构。

## 下一阶段一：Learner 与数据契约

- 将训练生命周期整理为 `zero_ttt.learner.Learner` 小型门面，`model` 保持纯网络定义。
- 设计不可变棋局/样本分片和 manifest；用 `BatchSource` 完成校验、混合与 batch 化。
- 实现只读 BF16 publication evaluator，并保持 14.5 GiB 双模型驻留验收线。

## 下一阶段二：纯策略学生自博弈

- 用本地 `GameState` 实现单线程、batch size 1 的完整棋局采集。
- 固定整盘 publication、种子和选着参数，原子写入可恢复的数据分片。
- 为完整 tiny 棋局、非法着、终局、版本失配和中断恢复增加测试。

## 下一阶段三：KataGo 标注与课程

- 实现本机 Analysis 客户端、坐标/视角 adapter、持久任务和结果审计。
- 加入 Human-SL profile、高/低搜索预算和明确 human policy 影响参数。
- 混合在线、离线与旧阶段 rehearsal；按 400 盘和 Wilson 下界执行教师晋级。

## 未来

- 局域网教师 worker、监控、故障恢复和数据保留策略。
- 参考 `e2b3017` 选择性重写单线程、batch size 1 的本地 MCTS。
- 历史/战略快权重、搜索摘要快权重及严格的树/缓存版本语义。

具体 Human-SL rank 阶梯、搜索访问数和数据比例由标定实验写入运行 manifest，不在路线图中
预设。各阶段都先完成 tiny 垂直切片，再扩大数据量或并发度。
