# `zero_ttt.model`

`model` 包只定义策略—价值网络的数学前向、子模块和稳定输出，不拥有优化器、EMA、持久化、
数据采样或搜索。公共输出契约见[公共契约](../../../docs/architecture/contracts.md)，跨组件关系见
[模型与训练](../../../docs/architecture/model-and-training.md)。

## 前向结构

正式模型接收 25 个点特征平面和 5 个全局特征，输出 362 维 policy logits、当前行棋方视角的
value、361 点 ownership 和当前行棋方视角的 score margin。当前 token layout 是 361 个棋盘
token 加一个 summary token；二维 RoPE 只作用于棋盘区间。

`BasePolicyValueModel` 固定宏观前向模板：

1. 校验输入 shape 与合法着 mask；
2. `GoTokenEncoder` 构造棋盘和 summary token；
3. backbone 逐块执行并产生诊断；
4. `PolicyValueHeads` 生成各预测头；
5. 对非法动作屏蔽 policy logits，封装稳定 `ModelOutput`。

`PolicyValueTransformer` 负责组装具体 encoder、backbone、heads 和可关闭实验。调用方不得复制
这套前向，也不得依赖内部中间张量。

## 扩展点与执行策略

共享局面条件低秩超网络实现 `BlockResidualPlugin`，稀疏深度混合实现 `DepthMixer`；关闭时分别
使用 `NoOpBlockResidualPlugin` 和 `IdentityDepthMixer`。扩展只能通过这些显式接口接入，不能
注册任意 forward hook 或改变稳定输出形状。

`ExecutionConfig` 与 `BlockExecutor` 决定 activation checkpoint 和 compile 等执行策略，但不
改变模型数学。参数初始化与互斥的 `ModelParameterGroup` 由模型定义，学习率、weight decay、
梯度裁剪和优化器生命周期属于 [`training`](../training/README.md)。

## 不变量

- 模型不导入数据来源、KataGo、OpenSpiel、Learner、CLI 或控制台。
- 合法着屏蔽属于稳定前向；搜索层不能另造未屏蔽的公共输出语义。
- 实验开关关闭后的基线不注册无用参数，参数分组必须完整且互斥。
- checkpoint/publication 的模型配置必须足以重新构造同一网络后再严格加载权重。
