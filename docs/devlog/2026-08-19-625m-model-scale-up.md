# 2026-08-19：正式模型扩至 625M

## 目标

在 RTX 4090 Laptop 16 GB 上保持物理 batch 16、FP32 fused AdamW、CPU FP32 EMA 和 GPU BF16 publication 三副本生命周期，把空余显存转换为结构均衡的模型参数，同时保留 14.5 GiB 长跑验收线。

## 候选扫描

候选均开启共享 rank-8 超网络和稀疏 DWA。下表是单次编译前后向、一次稳态前后向和一次 EMA 的初步校准，不作为吞吐 benchmark：

| 形状 | checkpoint | 参数量 | compile | 单步 | EMA | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×1152×2944 | 隔层 | 501,280,817 | 158.046 s | 0.630 s | 0.399 s | 14.436 GiB |
| 32×1152×3072 | 隔层 | 515,568,689 | 158.926 s | 0.702 s | 0.406 s | 14.811 GiB |
| 32×1152×3072 | 每层 | 515,568,689 | 172.427 s | 0.698 s | 0.403 s | 12.104 GiB |
| 32×1280×3584 | 每层 | 657,079,217 | 163.311 s | 0.922 s | 1.037 s | 15.047 GiB |
| **32×1280×3328** | **每层** | **625,357,745** | **167.169 s** | **0.885 s** | **0.942 s** | **14.332 GiB** |

全层 activation checkpoint 在 515M 候选上回收约 2.71 GiB reserved，而该单次测量中没有观察到稳态步时恶化。657M 候选超过 15 GiB，不适合作为长期默认；625M 候选同时满足容量目标和既有 14.5 GiB 硬线。

## 正式配置

- 默认：32 层、`d_model=1280`、20 头、每头 64、SwiGLU `d_ff=3328`，后 16 层共享超网络，DWA `dilation=4`、`period=4`。
- baseline：相同主干，关闭共享超网络与 DWA，共 620,432,901 参数。
- 所有层 activation checkpoint；batch 16、累积 16、学习率 `2e-4`、warmup 2000 和 EMA/publication 频率不变。
- 正式 run path 改为 `runs/rtx4090l-625m` 与 `runs/rtx4090l-625m-baseline`，不读取旧 313M 状态；checkpoint schema 仍为 v2。

## 容量边界

- CPU FP32 EMA 参数约 2.33 GiB；GPU BF16 publication 约 1.16 GiB。
- fast、slow 和两份 FP32 Adam 状态组成的完整未压缩 checkpoint 约 9.3 GiB。
- `keep=2` 加 immutable/current publication 约占 21 GiB；原子保存期间至少保留约 32 GiB 空闲。本轮不修改保存机制。
- 验收机器的 Docker/WSL 可用内存约 45 GiB，D 盘可用空间约 1.4 TiB。

## 正式验收

正式冒烟同时驻留 GPU FP32 `fast`、CPU FP32 EMA 与 GPU BF16 publication，使用 fused AdamW、batch 16、累积 16 和全层 activation checkpoint。训练模型按 block 独立编译，推理模型保持整图编译；层索引用非持久 one-hot tensor selector 表示，使后 16 个超网络 block 复用编译图。训练复用固定 FP32 梯度缓冲，避免每步释放和重建约 2.33 GiB 梯度造成 reserved 碎片。

| 配置 | measured optimizer steps | compile | micro-batch 中位 | 完整 optimizer-step 中位 | EMA | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 43.758 s | 0.598 s | 9.699 s | 0.992 s（强制） | 11.719 GiB | 12.510 GiB |
| default | 16 | 53.169 s | 0.651 s | 10.501 s | 0.541 s（第 16 步自然触发） | 13.062 GiB | 14.246 GiB |

默认运行覆盖 256 个物理 micro-batch。最终损失、总梯度、超网络独立梯度、DWA 梯度、A/B 饱和率以及动态/静态分支范数均有限；GPU publication 推理也在同一生命周期内通过。默认和 baseline 的参数量分别精确为 625,357,745 与 620,432,901，均满足 14.5 GiB reserved 硬线。

更严格的累积验收也修正了初步扫描的口径：14.332 GiB 是单 micro-batch 校准值，不能代表长期梯度累积。完整模型整图编译会让共享超网络的跨层反向缓冲生命周期重叠；逐 block 编译把默认模型的正式 16 步峰值稳定在 14.246 GiB，且没有触发重编译上限或新增图断裂。表中 compile 时间来自同一容器内 baseline 后接 default 的顺序运行，存在编译缓存复用，只作本机工程记录。

验证结果：Ruff 与 `compileall` 通过；本机和 Docker 均为 43 项 pytest 全部通过；Docker 中 PyTorch 2.13.0+cu132、cuDNN 92000 与 RTX 4090 Laptop 基础 CUDA 冒烟通过。

## 产生的决策

- [D-026](../decisions/legacy.md#d-026默认模型扩至约-625-亿参数)

## 问题与下一步

- 当前结果不证明棋力提升，也不是训练吞吐的多次重复基准。
- 长期 checkpoint 加固、预训练和自博弈恢复策略留给后续独立工作。
