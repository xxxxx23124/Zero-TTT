# 系统边界

## 项目定位

Zero-TTT 负责学生模型与训练基础设施。学生的默认落子路径是批量网络前向后直接从
policy 采样，即“纯直觉”。KataGo 是独立进程，只负责官方模型的强教师分析和 GTP 对弈。

```text
棋谱/未来自博弈/未来教师结果
              │
         BatchSource
              │
     Trainer ─┴─ checkpoint/EMA publication
              │
       PositionEvaluator
              │
    纯策略落子或未来的外部搜索器
```

## 当前已实现

- 固定 19×19 的 Tromp–Taylor 状态、合法着、数子与特征编码。
- 策略、价值、所有权和目差输出的 Transformer。
- 与数据来源无关的 Trainer、CPU FP32 EMA、checkpoint 和 BF16 publication。
- `BatchSource`、`TrainBatch`、`PositionEvaluator` 三个稳定边界。
- KataGo v1.17.2 的独立 Docker 构建和服务入口。

## 当前未实现

- 真实棋谱数据源、持久语料库和学生纯策略自博弈。
- KataGo 教师 worker、局域网队列和在线训练调度。
- 快权重、学生模型 MCTS 后端、跨搜索树复用。

## 解耦原则

- Trainer 不知道样本来自棋谱、自博弈还是教师。
- 模型不依赖 KataGo 协议或任何搜索树类型。
- KataGo 不进入 Python 包，也不能加载本项目的 Transformer checkpoint。
- 工作流负责组装数据源和生命周期；基础组件不反向导入工作流。
- 所有运行、构建和测试只通过 Docker 支持。

旧 replay/checkpoint 不自动删除，但 schema v2 及旧搜索记录不再受支持，也不提供迁移器。
