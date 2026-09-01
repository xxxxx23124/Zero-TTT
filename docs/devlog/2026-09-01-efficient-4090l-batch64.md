# 2026-09-01：本地高效模型与 batch-64 自博弈

## 目标

在 RTX 4090 Laptop 16 GB 上保持严格 FP32、compile、CPU EMA、HyperNet 和 DWA，把训练物理
batch 提高到 64，并关闭 activation checkpoint；训练和自博弈阶段均继续遵守 14.5 GiB 线。

## 完成内容

- 当前默认改为 12 层、`d_model=512`、8 头、`d_ff=1536`，后六层共享 rank-8 HyperNet，
  DWA 保持 `dilation=4`、`period=4`，共 43,371,150 参数。
- baseline 使用相同主干并关闭 HyperNet/DWA，共 41,189,893 参数。
- 训练改为 batch 64、累计 64、compile 开、activation checkpoint 关，有效 batch 仍为 4096。
- 自博弈改为 64 个 actor 和固定 batch-64 compiled evaluator。
- 原 625M 默认和 baseline 分别改名为 `rtx4090l_625m_future` 与
  `rtx4090l_625m_future_baseline`；其训练方案不变，自博弈改为 batch 64。

## 实验与结果

环境为 PyTorch 2.13.0+cu132、RTX 4090 Laptop，严格 FP32 且 TF32 关闭。候选均开启 HyperNet
与 DWA、batch 64、compile，并关闭 activation checkpoint：

| 形状 | 参数量 | 口径 | peak reserved | 稳态 micro-batch |
| --- | ---: | --- | ---: | ---: |
| 20×768×2048 | 145,217,945 | 2 个单累计 step | 31.709 GiB | 27.156 s |
| 12×512×1408 | 40,879,758 | 2 个单累计 step | 12.885 GiB | 0.467 s |
| 14×512×1408 | 47,304,846 | 2 个单累计 step | 14.807 GiB | 0.536 s |
| **12×512×1536** | **43,371,150** | **完整累计 64** | **13.174 GiB** | **0.500 s** |

14 层候选虽低于 15 GiB，但超过项目的 14.5 GiB 安全线；最终候选完整 optimizer step 为
32.008 秒，首次 warmup（含编译）为 7.369 秒，CPU EMA 为 0.048 秒。baseline 完整累计峰值为
12.445 GiB。数据只用于本机工程校准，不是多次重复吞吐 benchmark，也不证明棋力变化。

batch-64 compiled 推理结果：

| 模型 | peak reserved | 首次执行（含编译） |
| --- | ---: | ---: |
| 43.37M 默认 | 1.344 GiB | 16.078 s |
| 41.19M baseline | 1.184 GiB | 12.600 s |
| 625M future 默认 | 9.430 GiB | 47.926 s |
| 625M future baseline | 4.711 GiB | 36.304 s |

低预算并发闭环使用 64 局、每局最多 4 手和每手 2 simulations，64 局全部封存为一个 shard；
broker 出现 1 个完整 batch，真实槽位率为 57.8%。正式 64 simulations 的受控 benchmark 使用
相同 64 局和 4 手上限，共执行 16,384 simulations，耗时 56.445 秒，即 290.262 simulations/s；
164 个推理 batch 中有 103 个满 batch，真实槽位率为 88.9%，GPU peak allocated 为 1.233 GiB。
该时间包含首次推理编译，不代表长局稳态吞吐。

## 产生的决策

- [D-043](../decisions/README.md#d-043本地高效模型与-batch-64-自博弈)

## 问题

- 64 个自博弈线程会增加 CPU 与规则计算压力；需通过 batching fill 和 simulations/s 判断收益。
- 当前显存结果不替代长时真实数据训练、温度和恢复测试。

## 下一步

运行低 simulations 的 64 局并发闭环，再用正式 64 simulations 记录 batch fill、规则耗时和
simulations/s；长期训练继续观察显存碎片、CPU EMA 和数据读取吞吐。
