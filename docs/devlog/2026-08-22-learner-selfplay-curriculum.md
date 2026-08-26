# 2026-08-22：Learner、自博弈与分级教师文档重构

## 目标

把模型生命周期、数据采集、学生自博弈、本地 MCTS 和 Human-SL 课程的讨论固化为已接受的
目标架构，同时保持当前实现状态可核验。

## 完成内容

- 将目标 `Learner` 定义为监督优化、EMA、compile、梯度与 checkpoint/publication 的唯一所有者。
- 规定采集器先写不可变分片，再通过 `BatchSource` 导入；KataGo adapter 不进入 Learner。
- 保留本地 `GameState`，把单线程 batch 1 纯策略自博弈列为最近实现。
- 检查提交 `e2b3017` 及删除前历史，决定只把旧 MCTS/测试作为未来选择性重写参考。
- 固定 KataGo 不改上游，只负责 Human-SL/强搜索教师和独立对弈。
- 接受同 profile 高预算教学、低预算对战，以及 400 盘/Wilson 下界毕业门槛。
- 重写架构、集成、工作流、路线图和入口文档，明确所有新组件均未实现。

## 实验与结果

文档修改前的 Docker 基线通过 `scripts/check_docs.py` 与 `tests/quality/test_docs.py`。修改后再次执行
相同检查；没有重跑 GPU 冒烟。显存边界引用既有 625M 正式验收的 14.246 GiB 结果。

## 产生的决策

- D-032：本地学生自博弈与可选 MCTS。
- D-033：Learner 与数据采集分离。
- D-034：训练与采集同进程分阶段。
- D-035：Human-SL 分级教师。

## 问题

本轮没有新增 Python 包、持久 schema、KataGo 客户端、搜索代码或运行配置。具体 rank 阶梯、
访问数、human policy 搜索参数和数据比例仍需标定。

## 下一步

先实现 Learner/持久数据契约的 tiny 垂直切片，再实现纯策略完整棋局采集；之后接入 KataGo
标注与课程调度，最后按需要选择性重写本地 MCTS。
