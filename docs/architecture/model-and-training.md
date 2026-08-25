# 模型与训练

## 学生模型

当前正式配置是 32 层、`d_model=1280`、20 头、`d_ff=3328` 的 19×19 Transformer。
输入为 25 个点特征平面和 5 个全局特征，输出为：

- 362 维合法着掩码后的 policy logits；
- 当前行棋方视角的 value；
- 361 点 ownership；
- 当前行棋方视角的 score margin。

共享局面条件低秩超网络和稀疏深度混合仍是可关闭实验；基线配置同时关闭两者。

宏观前向由 `BasePolicyValueModel` 固定为输入校验、token 编码、backbone、heads、合法着屏蔽
和诊断封装。当前 token layout 是 361 个棋盘 token 加一个 summary token；RoPE 只作用于棋盘
区间。共享超网络实现块内动态残差接口，稀疏 DWA 实现层后深度混合接口，两者不能注册任意
forward 钩子。

模型数学、执行策略、训练优化和持久化分别拥有不同职责：

- `model` 定义纯前向、子模块初始化、token layout、插件插入点和完整互斥的参数分组；
- `ExecutionConfig` 与 block executor 决定 activation checkpoint 和 compile，不改变模型数学；
- Trainer 决定学习率、weight decay、分组梯度裁剪、FP32 EMA 和优化器生命周期；
- checkpoint manager 只负责原子保存、保留策略和 publication 文件。

## 当前训练器与目标 Learner

Trainer 只通过 `BatchSource.next_batch(batch_size, rng)` 取样。每个 `TrainBatch` 都已经完成
规则解释、视角转换、合法着处理和标签归一化，Trainer 不负责解析原始数据。

目标架构把这些能力迁入 `zero_ttt.learner.Learner`，但不扩大职责。Learner 独占训练模型、
优化器、CPU FP32 EMA、训练 compile、梯度生命周期、checkpoint 和 publication；它不解析
SGF/KataGo JSON，不运行游戏、MCTS、教师队列或课程调度。

缺失的 ownership/score 由逐样本布尔 mask 表示，不能伪造为有效零标签。policy 必须是合法着
上的非负归一化分布，value 与辅助标签均使用当前行棋方视角。

GPU `fast` 参数、梯度缓冲和 AdamW 状态保持 FP32，BF16 只用于 autocast 下的矩阵计算；CPU
FP32 `slow` 按样本数做 EMA。FP32 slow 能保留小于 BF16 权重分辨率的 EMA 增量，但不能恢复
BF16 前后向已经舍入的信息。小学习率负责在线吸收新标签，EMA 负责发布和评估稳定性，两者
不是互相替代的训练策略。

推理由另一个只读 `PositionEvaluator` 加载 BF16 slow publication。正式显存冒烟已经在训练
batch 16、累积 16 时同时驻留 fast、CPU EMA 和 GPU publication，峰值为 14.246 GiB；因此
目标默认同进程常驻、训练与采集分阶段运行，不预先实现复杂的模型/优化器换入换出。

## 复现边界

- schema v4 TOML 是模型、训练、执行和运行参数的唯一事实来源。
- checkpoint schema v4 保存配置哈希、fast/slow、优化器和随机状态；不迁移 v3 权重。
- publication 是不可变 BF16 slow 快照，并带模型版本。
- 采集器写不可变样本分片，新的数据来源实现 `BatchSource`，不复制 Learner 或损失函数。
- 采集器和 evaluator 只通过 publication 身份及持久数据与 Learner 交接，不共享模型地址。

详细所有权与目标调用面见 [Learner 与流程边界](learner-and-workflows.md)。
