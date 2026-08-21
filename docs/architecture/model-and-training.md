# 模型与训练

## 学生模型

当前正式配置是 32 层、`d_model=1280`、20 头、`d_ff=3328` 的 19×19 Transformer。
输入为 25 个点特征平面和 5 个全局特征，输出为：

- 362 维合法着掩码后的 policy logits；
- 当前行棋方视角的 value；
- 361 点 ownership；
- 当前行棋方视角的 score margin。

共享局面条件低秩超网络和稀疏深度混合仍是可关闭实验；基线配置同时关闭两者。

## 当前训练器与目标 Learner

Trainer 只通过 `BatchSource.next_batch(batch_size, rng)` 取样。每个 `TrainBatch` 都已经完成
规则解释、视角转换、合法着处理和标签归一化，Trainer 不负责解析原始数据。

目标架构把这些能力迁入 `zero_ttt.learner.Learner`，但不扩大职责。Learner 独占训练模型、
优化器、CPU FP32 EMA、训练 compile、梯度生命周期、checkpoint 和 publication；它不解析
SGF/KataGo JSON，不运行游戏、MCTS、教师队列或课程调度。

缺失的 ownership/score 由逐样本布尔 mask 表示，不能伪造为有效零标签。policy 必须是合法着
上的非负归一化分布，value 与辅助标签均使用当前行棋方视角。

`fast` 权重接受梯度；CPU FP32 `slow` 按样本数做 EMA。小学习率负责在线吸收新标签，EMA
负责发布和评估稳定性，两者不是互相替代的训练策略。

推理由另一个只读 `PositionEvaluator` 加载 BF16 slow publication。正式显存冒烟已经在训练
batch 16、累积 16 时同时驻留 fast、CPU EMA 和 GPU publication，峰值为 14.246 GiB；因此
目标默认同进程常驻、训练与采集分阶段运行，不预先实现复杂的模型/优化器换入换出。

## 复现边界

- schema v3 TOML 是模型、训练和运行参数的唯一事实来源。
- checkpoint schema v3 保存配置哈希、fast/slow、优化器和随机状态。
- publication 是不可变 BF16 slow 快照，并带模型版本。
- 采集器写不可变样本分片，新的数据来源实现 `BatchSource`，不复制 Learner 或损失函数。
- 采集器和 evaluator 只通过 publication 身份及持久数据与 Learner 交接，不共享模型地址。

详细所有权与目标调用面见 [Learner 与流程边界](learner-and-workflows.md)。
