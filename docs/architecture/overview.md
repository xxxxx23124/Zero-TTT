# 系统边界

## 项目定位

Zero-TTT 负责学生模型、训练基础设施、本地棋局和学生自博弈。学生的默认落子路径是网络
前向后直接按固定规则从 policy 选着，即“纯直觉”；未来可选本地 MCTS。KataGo 保持未修改
的独立进程，只负责官方模型的强教师/Human-SL 分析和独立对弈。

```text
棋谱 ─┐                         KataGo 教师
学生自博弈 ─→ 不可变数据分片 ←──────┘
              │
         BatchSource
              │
     目标 Learner ── checkpoint/EMA publication
                                      │
                              PositionEvaluator
                                      │
                         纯策略落子 / 可选本地 MCTS
```

## 当前已实现

- 固定 19×19 的 Tromp–Taylor 状态、合法着、数子与特征编码。
- 策略、价值、所有权和目差输出的 Transformer。
- 与数据来源无关的 Trainer、CPU FP32 EMA、checkpoint 和 BF16 publication；未来将演进为
  `Learner` 门面。
- `BatchSource`、`TrainBatch`、`PositionEvaluator` 三个稳定边界。
- KataGo v1.17.2 的独立 Docker 构建和服务入口。

## 当前未实现

- 真实棋谱数据源、持久语料库和学生纯策略自博弈。
- KataGo 教师 worker、局域网队列和在线训练调度。
- 目标 `learner` 包、只读 publication evaluator 和本地 MCTS。
- Human-SL 分级教师、快权重和受版本约束的树复用。

## 解耦原则

- Learner 不知道样本来自棋谱、自博弈还是教师；当前 Trainer 已遵循这一原则。
- 模型不依赖 KataGo 协议或任何搜索树类型。
- KataGo 不进入 Python 包，也不能加载本项目的 Transformer checkpoint。
- 本地 `GameState` 是学生自博弈的棋局与合法着来源；不修改 KataGo 来托管学生搜索。
- 采集器先写持久数据，工作流只组装数据源和阶段；基础组件不反向导入工作流。
- Learner 与 evaluator 可以同时驻留 GPU，但默认分阶段执行，不并发提交 CUDA 工作。
- 所有运行、构建和测试只通过 Docker 支持。

旧 replay/checkpoint 不自动删除，但 schema v2 及旧搜索记录不再受支持，也不提供迁移器。

目标组件和流程详见 [Learner 与流程边界](learner-and-workflows.md)、
[学生自博弈](../workflows/student-selfplay.md)与[分级教师](../workflows/curriculum-teachers.md)。
