# 2026-08-19：共享超网络、DenseFormer 与 CPU EMA

## 目标

把共享局面条件超网络、稀疏 DenseFormer DWA 和 CPU FP32 EMA 接入正式配置，同时保留可完全关闭的基线，并修复 publication 权重与版本号可能不同步的生命周期问题。

## 完成内容

- 配置升级到 schema v2；正式配置默认开启后 12 层共享 rank-8 超网络和 `dilation=4`、`period=4` 的 DWA，新增全关闭的 `configs/rtx4090l_baseline.toml`。
- 12 个动态层共享唯一的上下文投影、层嵌入和 A/B heads。B 头零初始化，不冻结、不 ramp；上下文梯度、学习率倍率和独立裁剪均保持 `0.1`。
- DWA 在第 4、8、12、16、20、24 层后混合同余深度的原始输出，以当前层为恒等初始化，并只保留未来会被引用的状态。
- `slow` 改为 CPU FP32，同步按名称做样本数 EMA；GPU 分别驻留训练 `fast` 和独立的 BF16 publication 推理副本。
- publication 权重加载与 `model_version` 更新合并；推理请求校验版本，模型只在完整自博弈阶段之间替换。
- checkpoint/publication schema 升到 v2，不迁移旧结构；`StepMetrics` 增加超网络诊断和 EMA 更新时间。
- Compose 增加 `/workspace/src` 导入路径，挂载源码后可直接运行包、pytest 和冒烟脚本。
- `paper/` 登记并校验 CaiT、DeepNet、ReZero、DenseFormer、HyperNetworks、HyperFormer、Hyperfan 七篇 PDF。

## 实验与结果

结构参数量：

```text
rtx4090l_baseline  309,421,061
rtx4090l           313,519,264
```

本机单元测试与静态检查：

```text
python -m pytest -q                         42 passed in 4.79s
docker compose run --rm dev python -m pytest -q
                                               42 passed in 5.34s
python -m ruff check src tests scripts  通过
python -m compileall -q src scripts     通过
```

Docker GPU 正式结果（RTX 4090 Laptop 16 GB、PyTorch 2.13.0+cu132、batch 16）：

| 配置 | compile | 稳态步时 | EMA 更新时间 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 91.110 s | 0.492 s | 0.255 s | 7.501 GiB | 7.824 GiB |
| default | 119.930 s | 0.309 s | 0.247 s | 8.608 GiB | 9.379 GiB |

冒烟脚本同时驻留 GPU `fast`、CPU FP32 `slow` 和 GPU BF16 publication，执行 fused AdamW、推理/训练 compile、一次编译前后向、一次稳态前后向和一次同步 EMA。两份配置输出与梯度均有限，峰值 reserved 均低于 14.5 GiB。单次步时样本只用于证明流程可运行；default 比 baseline 快的这一次结果不能解释为结构本身更快，需要多次预热基准后才能比较吞吐。

## 产生的决策

- [D-024](../design_decisions.md#d-024共享超网络与稀疏-dwa-进入默认实验配方)
- [D-025](../design_decisions.md#d-025ema-固定驻留-cpu发布推理使用独立-gpu-副本)

## 问题

- 当前只验证结构正确性、数值、吞吐和显存，尚无棋力或长期稳定性证据。
- CPU EMA 首版是同步实现；若实测更新时间成为瓶颈，再单独设计带显式同步语义的异步 D2H 双缓冲。

## 下一步

做极小数据过拟合、固定局面报警与长期指标观察；需要吞吐结论时增加多次预热和重复采样的独立 benchmark。
