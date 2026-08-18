# 2026-08-18：核心闭环首版实现

## 目标

实现里程碑 1、4、5 所需的首版核心代码，并同时建立棋盘 Transformer 与默认关闭的里程碑 7 完整低秩超网络边界。SGF、外部数据、监督预训练、GTP 和原生搜索后端不在本次范围。

## 完成内容

- 建立 `src/zero_ttt` 包、严格版本化 TOML 配置、生产/tiny 配置与只允许选择命令和配置文件的 CLI。
- 实现 No-Suicide Tromp–Taylor 状态、位置超级劫、面积计分、25 个逐点特征、5 个全局特征和 D4 标签变换。
- 实现纯 CLS 全局 token、中心坐标轴向 2D RoPE、Pre-RMSNorm/QK-Norm/SDPA/SwiGLU 主干、四个输出头和辅助损失。
- 实现可关闭的逐局面完整 A/B 低秩超网络、B 头零初始化、上下文梯度缩放、独立学习率/梯度裁剪与冻结—ramp 日程。
- 实现 Python PUCT、集中推理队列、有限 batch 桶、精确请求去重、版本化 LRU 评价缓存、树复用、虚拟损失、根噪声、FPU 和动态预算。
- 实现按整盘事务保存与 FIFO 淘汰的 SQLite WAL 回放、SHA-256 校验、按局面均匀采样、状态重放和在线 D4 增强。
- 实现累积梯度、BF16/fused AdamW CUDA 路径、样本数 EMA、BF16 发布快照、完整/故障 checkpoint 和单 GPU“自博弈 → 训练 → EMA/发布”控制器。
- Docker 镜像增加项目可编辑安装和 pytest 依赖；增加正式模型关闭/启用超网络的 batch 16 编译显存脚本。

## 实验与结果

在当前 CPU 开发环境执行：

```text
PYTHONPATH=src python -m pytest -q
38 passed in 4.35s
```

测试覆盖规则、特征、D4、RoPE、非法策略屏蔽、checkpoint/compile 容差、超网络零初始残差与 ramp、MCTS 价值符号/预算/噪声/虚拟损失、缓存隔离/去重、16 线程无死锁、SQLite 校验/FIFO/重启、一次完整优化与 tiny 闭环恢复。

使用 PyTorch `meta` 设备核对正式结构参数量，不分配实际模型内存：

```text
hypernet=false  309421061
hypernet=true   325801477
```

`zero-ttt smoke --config configs/test.toml` 已完成本地前后向。当前 Windows 主机的 PyTorch 2.9/CUDA 12.8 安装能访问 RTX 4090 Laptop，但缺少 Inductor 所需的 Triton，因此正式脚本在 compile 阶段以 `TritonMissing` 明确失败，没有产生 compile 显存结果。随后在相同真实模型、BF16、batch 16、activation checkpoint、FP32 `slow` 和 fused AdamW 条件下关闭 compile，完成 eager 实测：

```text
hypernet=false  309421061 params  peak_reserved=6.6113 GiB
hypernet=true   325801477 params  peak_reserved=6.8086 GiB
```

这两项均低于 14.5 GiB，但不能替代 Linux Docker 内的最终 compile 验收。`scripts/model_smoke_test.py` 保持强制 compile，并已编码 14.5 GiB 上限，不会把 eager 结果误报为最终通过。

## 产生的决策

没有新增稳定决策；实现遵循 D-020 至 D-023。新增 I-003 记录未来外部数据只能通过统一 `GameSource → GameRecord → ReplayStore` 边界进入回放。

## 问题

- 尚未在目标 RTX 4090 Laptop Linux Docker 环境运行正式模型的关闭/启用超网络 batch 16 编译显存验收。
- 固定小数据过拟合、固定局面非阻塞报警和长期运行吞吐指标尚未实现完整生产版本。
- Python MCTS 的 16 线程结果受调度影响，不承诺逐次访问计数完全一致；单线程用于确定性回归。
- 外部数据、监督预训练、GTP、C++ 后端和训练/推理并发共享 GPU 均未实现。

## 下一步

先在目标 Docker GPU 环境运行单元测试和 `scripts/model_smoke_test.py`；若峰值超过 14.5 GiB，只提高 activation checkpoint 覆盖率。随后增加小数据过拟合与固定局面报警，再开始 I-003 数据源调查。
