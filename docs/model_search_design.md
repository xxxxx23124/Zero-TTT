# Zero-TTT 模型、训练与搜索设计

> 状态：共享超网络、稀疏 DWA、CPU EMA 与发布生命周期已落地，生产 GPU 验收通过
>
> 最近更新：2026-08-19

本文集中规定当前主线的模型结构、训练权重生命周期、MCTS 调度和性能边界。围棋规则、数据许可和总体里程碑仍分别由[实施计划](implementation_plan.md)与[设计决策](design_decisions.md)管理。

## 1. 设计边界

当前闭环不再运行候选模型与冠军模型之间的独立竞技评估，也不存在晋升门槛或自动回退。训练维护三份同构但职责、精度和驻留位置不同的权重：

| 名称 | 职责 | 更新方式 |
| --- | --- | --- |
| `fast` | 接收训练梯度，快速吸收新数据 | 每个优化器步骤更新 |
| `slow` | 为发布提供平滑源模型 | CPU FP32，从 `fast` 做稀疏等效 EMA |
| `publication` | 为自博弈和未来 GTP 提供冻结推理权重 | 从 `slow` 发布，GPU BF16，只在阶段边界替换 |

验证损失、固定局面回归、非有限值检查和梯度健康检查继续保留，但只负责暴露问题，不参与模型晋升、拒绝或回退。非有限值属于运行错误：训练立即停止并保存故障 checkpoint，不自动加载旧权重继续训练。

文档将性能措施分为两类：

- **工程加速：** 精确缓存、批量推理、请求去重、异步传输、SDPA、`torch.compile`、激活 checkpoint 和梯度积累。这些措施不应主动改变模型接口或训练目标。
- **算法优化：** 虚拟损失、根噪声、FPU、动态搜索预算、滑动窗口回放、数据增强和辅助损失。这些措施会改变搜索轨迹或训练分布，必须配置化、记录到 checkpoint，并接受单独测试。

“工程加速”表示设计目标是保持语义，不表示不同浮点内核之间逐位一致。Eager、编译和混合精度路径以明确数值容差验证。

## 2. 配置与复现

一次运行只从一个版本化 TOML 文件读取模型、训练、搜索、回放和实验参数。命令行可以选择运行入口和配置文件，但不能逐项覆盖配置值；环境变量也不能改写模型或算法参数。

配置加载遵守以下规则：

- TOML 顶层包含 `schema_version=2`，未知字段、缺失必填字段、错误类型和越界值都直接报错；旧结构 checkpoint 不迁移。
- 加载后转换为不可变的类型化配置对象；业务代码不得读取散落的环境变量或自行补默认值。
- 规范化后的完整配置计算 SHA-256，并与原始 TOML、schema 版本一起写入 checkpoint。
- 恢复训练时，结构相关字段必须与 checkpoint 完全一致；允许变化的运行字段必须列入显式白名单并记录差异。

首个正式配置面向 NVIDIA GeForce RTX 4090 Laptop GPU 16 GB。配置文件是运行时事实来源；本文记录该硬件档的默认值和不可跨越的语义。

## 3. 棋盘 Transformer

### 3.1 主干规格

默认主干是非因果 Transformer Encoder：

| 参数 | 默认值 |
| --- | ---: |
| Transformer 层数 | 32 |
| 隐藏宽度 `d_model` | 1280 |
| 注意力头数 | 20 |
| 每头宽度 | 64 |
| SwiGLU 中间宽度 `d_ff` | 3328 |
| 主体 dropout | 0 |
| 关闭实验的基础参数量 | 620,432,901 |
| 默认总参数量 | 625,357,745 |

每个交叉点对应一个 token，另加一个全局 token，总序列长度为 362。全局 token 严格初始化为 `learned_cls + global_projection(global_features)`：只叠加贴目、手数、连续 pass 和当前方等不能从当前棋盘唯一恢复的全局特征，不加入棋盘均值、额外池化或绝对位置向量。输入和值目标始终采用当前行棋方视角。

每层使用 Pre-RMSNorm、QK-Norm、无 bias 的主体线性投影和 SwiGLU。注意力与 FFN 的残差输出投影按 `1 / sqrt(2L)` 做深度缩放初始化，其中 `L=32`。网络内部不使用 dropout，也不在 checkpoint 区域内放置其他随机操作。

### 3.2 共享局面条件低秩分支

正式配置从第一步起默认启用一个跨层共享生成器，作用于最后 16 层 FFN down-projection。每层仍以该层自己的全局 token `c` 为上下文，并加入层嵌入：

```text
z = SiLU(proj(RMSNorm(c)) + embedding(layer))
y_board = W_down(h_board) + A(z) B(z) h_board / sqrt(rank)
```

默认 `rank=8`、生成器隐藏宽度 `128`。动态项只作用于 361 个棋盘 token，不修改注意力投影和全局 token。所有层共享 `RMSNorm`、上下文投影、层嵌入和 A/B 输出头，不为每层复制一套生成器。

A/B 原始输出经 `tanh` 有界化，B 头权重和 bias 为零初始化，因此初始前向严格等于关闭分支的基线，同时第一步 B 头即可取得梯度。分支 scale 恒为 1，不冻结也不 ramp。上下文梯度、学习率倍率和独立梯度裁剪上限均为 `0.1`。训练记录 A/B 饱和率、动态/静态分支 RMS 和超网络梯度范数。

### 3.3 稀疏 DenseFormer DWA

正式配置默认在第 4、8、12、16、20、24、28、32 层后执行 depth-weighted averaging，`dilation=4`、`period=4`。每个混合点只组合与当前深度同余的原始 block 输出（含 depth 0 输入）；当前层系数初始化为 1，其余为 0，所以初始前向严格等于未启用 DWA 的主干。

实现只保留未来混合点实际会引用的深度状态，避免无条件保存全部层输出而抵消 activation checkpoint 的显存收益。`configs/rtx4090l_baseline.toml` 同时关闭 DWA 和共享超网络，作为正式结构基线。

### 3.4 二维旋转位置编码

Q、K 的 64 个每头通道平均分给行、列两个旋转子空间：前 32 维编码行坐标，后 32 维编码列坐标。361 个棋盘 token 使用以天元为原点的 `-9..9` 行列坐标；全局 token 使用零坐标，因此旋转恒等。默认 `rope_base=100`、`rope_scale=1`，固定 FP32 频率按 `100^(-i/16)` 生成后再转换到计算 dtype。二维 RoPE 只作用于 Q、K，不改变 V，也不额外加入学习式绝对位置向量。

### 3.5 输出接口与损失

模型输出统一为：

- `policy_logits[B, 362]`：361 个交叉点由对应 token 读出，`pass` 由全局 token 读出；合法动作掩码在 softmax 前应用。
- `value[B, 1]`：当前行棋方视角的标量价值，由全局 token 读出。
- `ownership[B, 361]`：经 `tanh` 的逐点归属辅助预测。
- `score_margin[B, 1]`：包含贴目的当前方视角目差辅助预测。

默认损失权重为策略 `1.0`、价值 `1.0`、所有权 `0.15`、目差 `0.05`。所有权和目差都携带逐样本标签掩码；普通棋谱无法可靠产生某项标签时，该项不参与该样本的归一化损失，不能用伪造的零标签代替。辅助头不参与 MCTS 的第一版选择公式。

## 4. 训练与显存策略

- 参数和优化器状态保持稳定训练所需的精度，矩阵计算使用 BF16 autocast；首版优化器为 fused AdamW。
- 物理训练 batch 固定为 16，默认累积 16 个 micro-batch 后更新一次，因此有效 batch 为 256。
- 只在完整累积周期结束时执行梯度裁剪、优化器步骤、学习率调度和 `fast` 训练步计数；全局梯度范数上限为 `1.0`。
- 默认每层执行 `torch.utils.checkpoint.checkpoint`，显式设置 `use_reentrant=False` 和 `preserve_rng_state=False`。由于模型内部没有 dropout 或随机分支，重计算不依赖随机数恢复；以额外计算换取约 6.25 亿参数的单卡容量。
- 训练按 Transformer block 建立静态形状的 `torch.compile` 图，以限制 AOTAutograd 跨层缓冲生命周期；推理编译完整模型，并为 batch `1/2/4/8/16` 建立有限桶，避免任意动态形状反复编译。
- 注意力只通过 `torch.nn.functional.scaled_dot_product_attention` 实现，让 PyTorch 按设备和数据类型选择可用的 FlashAttention 或其他加速内核。
- 数据加载使用 pinned memory 和异步 H2D；训练清梯度复用既有 FP32 梯度缓冲，避免 625M 模型跨步反复释放和切分约 2.33 GiB 梯度造成 reserved 碎片。是否编译优化器必须以实际基准结果决定，不作为模型正确性的依赖。
- `slow` EMA 固定为 CPU FP32；训练 GPU 同时保留 `fast` 和一份独立的 BF16 publication 推理模型，不在 GPU 保存 FP32 EMA。

## 5. `fast` 与 `slow`

`slow` 的衰减以实际训练样本数定义，而不是依赖容易随梯度积累变化的固定 step 系数。默认半衰期为 `1,048,576` 个样本，每 16 个优化器步骤更新一次。设距上次 EMA 已处理 `delta_samples` 个样本，则：

```text
beta = 2 ** (-delta_samples / 1_048_576)
slow = beta * slow + (1 - beta) * fast
```

第一次创建 `slow` 时按名称完整复制 `fast` 到 CPU FP32。EMA 更新只发生在成功完成的优化器步骤之后；中断的梯度累积周期不增加样本计数。更新同步执行并校验全部 parameter/buffer 名称，buffer 直接同步；每次更新耗时写入 `ema_update_seconds`。首版不采用异步 D2H 双缓冲。每 256 个优化器步骤从 `slow` 发布一个不可变 BF16 模型快照，并分配单调递增的 `model_version`。

自博弈和 GTP 只加载独立的 GPU BF16 publication 副本。新版本只在完整自博弈阶段或一个明确的搜索任务边界替换；加载权重和更新 `model_version` 是同一个生命周期操作，推理请求还必须校验所请求版本。一次 MCTS 和一盘棋的所有叶节点使用同一版本。`fast` 不直接生成自博弈数据，也不存在 `anchor`、`candidate` 或 `champion`。

完整训练 checkpoint 至少保存：`fast`、FP32 `slow`、优化器、学习率调度、优化器步数、已处理样本数、EMA 上次更新时间、配置与哈希、数据游标、回放元数据和全部随机数状态。发布快照是由 `slow` 派生的 BF16 推理产物，不替代完整 checkpoint。

## 6. 批量 MCTS 与缓存

### 6.1 调度模型

一个推理进程只加载一份已发布模型，配套一个集中叶节点队列和一个共享的 16 线程 CPU 搜索池：

- GTP 模式下，16 个线程共同服务当前根搜索。
- 当前分阶段自博弈按盘顺序执行；每盘搜索由同一棵树的 16 个 CPU 线程共享。未来可以在不改变协议的情况下调度多盘，但不是首版行为。
- 队列按 `model_version` 分组，先去重相同状态，再组成最大 batch 16；GPU 返回后把结果分发给所有等待者。
- 推理统一使用 `torch.inference_mode()`、BF16 和编译后的有限 batch 桶。

每次并行下降对途经边增加 1 个虚拟损失。叶节点完成、取消或抛出异常时，都必须在同一清理路径撤销虚拟损失，避免永久污染访问统计。

### 6.2 树复用与评价缓存

真实落子后优先把对应子树提升为新根。不同树节点保留独立访问统计，不构造共享备份语义的 DAG；只有不可变神经评价使用有容量上限的 LRU 缓存。缓存命中必须保持规则语义：

- 键至少覆盖棋盘、当前方、规则、贴目、劫状态、超级劫历史语义和 `model_version`。
- 哈希只用于定位；命中后仍比较完整状态身份，不能依赖哈希碰撞概率宣称相等。
- 缓存保存无根噪声的原始策略 logits 与价值。Dirichlet 噪声只施加到当次自博弈根的先验副本，不能写回缓存。
- 模型版本变化后旧评价不得复用；可以按版本分区淘汰，不需要立即扫描整个缓存。

## 7. 会改变行为的搜索优化

### 7.1 根噪声与未访问节点

Dirichlet 噪声只在自博弈根启用，混合权重为 `0.25`，合法动作上的总浓度为 `10.83`，即单个合法动作参数为 `10.83 / legal_action_count`。GTP、固定局面回归和数据复盘不使用根噪声。

未访问子节点的初始 Q 使用 First Play Urgency：

```text
Q_fpu = clip(Q_parent - 0.2 * sqrt(visited_prior_mass), -1, 1)
```

`visited_prior_mass` 是当前节点已访问子边的原始先验概率之和。尚无已访问子边时，`Q_parent` 使用该节点的网络价值，且惩罚项为零。

### 7.2 动态搜索预算

自博弈在 256、512、768 次模拟后检查根访问分布，最迟在 1024 次停止。归一化熵以合法动作上的根访问比例计算，头两名差距为二者访问比例之差。

| 检查点 | 提前停止条件 |
| ---: | --- |
| 256 | 归一化熵不高于 `0.35`，且头两名差距至少 `0.30` |
| 512 | 领先动作与 256 次时相同，且头两名差距至少 `0.20` |
| 768 | 与 512 次相比领先动作连续两次不变，且头两名差距至少 `0.10` |
| 1024 | 无条件停止 |

每个样本记录实际搜索预算、各检查点统计和最终根访问次数。动态停止只改变计算预算，不改变访问分布到策略目标的归一化方式。

## 8. 回放与增强

在线自博弈经验回放默认最多保存最近 2,000,000 个局面。容量按局面计数，但插入和淘汰以整盘棋为单位；超过容量时从最旧棋局开始 FIFO 淘汰，不留下半盘记录。窗口内默认均匀抽取局面。

D4 的八种旋转/镜像在样本被抽取后在线随机应用。棋盘特征、合法动作掩码、策略目标和所有权目标必须使用同一个变换；`pass`、价值、目差和贴目不随几何变换改变。固定随机种子时，数据游标与增强随机状态必须可恢复。

## 9. 计划接口

实现阶段至少需要以下稳定边界：

- 严格 TOML 加载器返回不可变的 `ExperimentConfig`，包含模型、训练、搜索、回放和实验子配置。
- `ModelOutput` 包含策略 logits、价值及带可用性语义的辅助输出。
- `InferenceRequest` 包含特征、合法动作掩码、精确状态身份和 `model_version`；批处理器不得跨版本合批。
- `TrainingSample` 在现有字段上增加实际搜索预算、所有权/目差目标及各自标签掩码。
- `TrainingCheckpoint` 明确区分 `fast`、`slow` 和派生的发布快照，加载时禁止名称回退或隐式替换。

## 10. 验收要求

- 极小数据集可以过拟合；策略、价值和启用的辅助损失均按预期下降。
- Eager 与 compile、checkpoint 与非 checkpoint 的输出和梯度在规定容差内一致；不得要求逐位一致。
- 2D RoPE 行列映射、全局 token、D4 策略/所有权变换及 pass 不变性通过单元测试。
- 缓存不会跨规则、超级劫历史或模型版本污染；虚拟损失在成功、取消和异常路径都归零。
- 动态预算的四个边界、FPU 初值和根噪声启用范围具有确定性测试。
- checkpoint 恢复后，GPU `fast`、CPU FP32 `slow`、EMA 样本计数、BF16 publication 版本、数据游标和随机状态均连续。
- publication 后实际推理权重与 `model_version` 同时更新；整盘棋中不得换模型。
- 最终真实模型在训练图完成编译后，batch 16 的峰值保留显存不得超过 14.5 GiB。若超限，只增加 activation checkpoint 覆盖率，不缩小既定模型。

baseline 为 620,432,901 参数，默认共享超网络与 DWA 配置为 625,357,745 参数。RTX 4090 Laptop 三副本正式冒烟中，默认配置在 batch 16、累积 16、连续 16 个优化器步下峰值 allocated/reserved 为 13.062/14.246 GiB；候选扫描与完整结果见 [625M 扩容日志](devlog/2026-08-19-625m-model-scale-up.md)。

## 11. 论文取舍

- **CaiT：** class-attention 的目标是把 patch 表征集中到 class token；当前围棋模型需要全局特征在所有 block 内持续广播，不改用独立 class-attention 阶段。LayerScale 与现有 `1/sqrt(2L)` 深度缩放残差初始化作用重叠，本轮不叠加。
- **DeepNet / DeepNorm：** DeepNorm 依赖 Post-Norm 结构与配套参数初始化，不直接移植到当前 32 层 Pre-RMSNorm 主干。
- **ReZero：** 原版会移除归一化并用可学习零门控启动；本项目保留 RMSNorm，只吸收“新增残差从零开始”的思想，用 B 头零初始化实现。
- **DenseFormer：** 稀疏 DWA 进入默认实验配方，以恒等初始化维持起点等价，并保留完全关闭的基线。
- **HyperNetworks / HyperFormer / Hyperfan：** 采用跨层共享的条件生成器，避免 16 套独立头；Hyperfan 作为生成权重初始化风险依据，但零动态残差已经提供严格基线起点，因此不额外实现完整 Hyperfan 初始化。
- **SoViT：** 吸收联合扩大 width、depth 和 MLP dimension 的 shape-scaling 思路；最终 32×1280×3328 形状由本项目固定 362 token、SwiGLU 和 RTX 4090 Laptop 实测决定，不直接照搬图像模型参数。

论文 PDF、官方来源与 SHA-256 见 [`paper/README.md`](../paper/README.md)。这些结构当前只完成正确性、稳定性、吞吐和显存验证，不声称提升棋力。

## 参考

- [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [PyTorch activation checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PyTorch `torch.compile` 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [RoPE-ViT](https://arxiv.org/abs/2403.13298)
- [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565)
- [KataGo 自博弈配置参考](https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/selfplay1.cfg)
- [CaiT](https://arxiv.org/abs/2103.17239)
- [DeepNet](https://arxiv.org/abs/2203.00555)
- [ReZero](https://arxiv.org/abs/2003.04887)
- [DenseFormer](https://papers.nips.cc/paper_files/paper/2024/file/f67449c7ab72f441d3a713b046c6818c-Paper-Conference.pdf)
- [HyperNetworks](https://arxiv.org/abs/1609.09106)
- [HyperFormer](https://arxiv.org/abs/2106.04489)
- [Hyperfan](https://openreview.net/forum?id=H1lma24tPB)
- [PyTorch CPU/non-blocking transfer 教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
- [Getting ViT in Shape / SoViT](https://arxiv.org/abs/2305.13035)
