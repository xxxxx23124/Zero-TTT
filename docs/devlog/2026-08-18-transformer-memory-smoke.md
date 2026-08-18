# 2026-08-18：Transformer 显存冒烟测试

## 目标

在正式实现前，确认约 3 亿参数的 24 层棋盘 Transformer、batch 16、FP32 `slow` 副本和后四层完整动态低秩分支在当前 16 GB GPU 上不存在明显的显存不可行性。

## 完成内容

通过一次性 Python 标准输入脚本，在现有 Docker GPU 开发镜像中构造结构近似模型：

- 24 层、`d_model=1024`、16 头、SwiGLU `d_ff=2816`；
- 后四层使用 rank 8、隐藏宽度 128 的逐局面完整 A/B 生成器；
- batch 16、序列长度 362、BF16 autocast；
- FP32 模型参数、FP32 `slow` 参数副本、fused AdamW、前向、反向、梯度裁剪和一步优化器更新；
- 使用 PyTorch SDPA；第一轮按两层一组做非重入 activation checkpoint。

测试在临时 Compose 容器中运行，没有修改仓库源码或保存模型产物。

## 实验与结果

环境为 NVIDIA GeForce RTX 4090 Laptop GPU 16 GB、PyTorch 2.13.0、CUDA 13.2。

| 项目 | 结果 |
| --- | ---: |
| 参数量 | 324.75M |
| 峰值 allocated | 6.18 GiB |
| 峰值 reserved | 6.34 GiB |
| 一步损失 | 0.169666，有限 |

另一次“仅隔层 checkpoint”的近似测试峰值 allocated 为 6.17 GiB、reserved 为 6.55 GiB，单步 GPU 计算约 0.91 秒。两次结果说明参数规模具备实现余量，但不能替代最终模型基准。

当前脚本没有包含完整输入编码、2D RoPE、QK-Norm、全部输出头和 `torch.compile`；初始化、编译缓存与真实数据管线也会改变峰值。因此正式验收仍要求最终训练图编译后，batch 16 的峰值保留显存不超过 14.5 GiB。

## 产生的决策

- 采用约 3.09 亿参数的基础 Transformer 配置。
- 保留 activation checkpoint 开关，默认隔层启用；如果最终模型超出显存线，只增加 checkpoint 覆盖率，不缩小模型。
- 低秩超网络默认关闭，启用时总规模约 3.25 亿参数。

相关规格见 `../model_search_design.md`，长期约束见 D-020 至 D-023。

## 问题

- 尚未测量最终 eager 与 compile 的速度、峰值显存和数值误差。
- 尚未测量真实多头损失与梯度积累 16 步的吞吐。
- 动态低秩分支本次只验证了内存和有限输出，没有验证所规划的稳定化初始化与训练日程。

## 下一步

实现最小模型接口后，使用正式配置重复 eager/compile、checkpoint 开关和完整输出头基准，并记录可复现命令与峰值。
