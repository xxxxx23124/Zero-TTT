# 论文清单

本目录保存 Zero-TTT 设计与实验直接参考的论文副本。论文中的实验结论不等于已经在围棋模型上复现；项目自己的结果记录在 `docs/devlog/`。

## 2026-08-19：深层 Transformer 与共享超网络

| 本地文件 | 官方来源 | SHA-256 |
| --- | --- | --- |
| `CaiT Going Deeper with Image Transformers.pdf` | [arXiv 2103.17239](https://arxiv.org/pdf/2103.17239) | `1b19d6662fa46bfaa45d180ae48d24b60b4ab201e0e056f5cc8b09074937f202` |
| `DeepNet Scaling Transformers to 1000 Layers.pdf` | [arXiv 2203.00555](https://arxiv.org/pdf/2203.00555) | `be1944f5a461dbb075d59f35594a7e8315ef10f73673a3bdabf759fad07e04a9` |
| `ReZero is All You Need.pdf` | [arXiv 2003.04887](https://arxiv.org/pdf/2003.04887) | `54eea1741fad2e7a20e52f55c64b34977f1206c5e3c8e14804ea5385c299c070` |
| `DenseFormer Enhancing Information Flow in Transformers.pdf` | [NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/f67449c7ab72f441d3a713b046c6818c-Paper-Conference.pdf) | `407104f0d63dbae08b732876060ddb8a3eb27e9ab5359c87e1b5ee79944207df` |
| `HyperNetworks.pdf` | [arXiv 1609.09106](https://arxiv.org/pdf/1609.09106) | `cc7057fe9b03ea8aa98d581eae1630bd6e03e25ac6057f7c53f45c50610cb4c5` |
| `HyperFormer Shared Hypernetworks for Transformers.pdf` | [arXiv 2106.04489](https://arxiv.org/pdf/2106.04489) | `39ac7c1b3a9f40afeebb4d9374edbec64060bee9b63195868fdb16ee406b3ab4` |
| `Hyperfan Principled Weight Initialization for Hypernetworks.pdf` | [arXiv 2312.08399](https://arxiv.org/pdf/2312.08399) | `9d5562dff89a87e79f36ae0172d01c4bda8260cd20cf1e62f0493c1ccbe96cc8` |
| `Getting ViT in Shape Scaling Laws for Compute-Optimal Model Design.pdf` | [arXiv 2305.13035](https://arxiv.org/pdf/2305.13035) | `360540d5b39e666adc1a72bf6f50c803b5494c05bf4e03b666bf479cc7ab903b` |

### 当前取舍

- 采用 DenseFormer 的恒等初始化稀疏 DWA，默认 dilation 4、period 4。
- 采用 HyperFormer 启发的跨层共享生成器，同时仍由每层自己的局面全局 token 提供上下文。
- 采用 SoViT 的联合 shape-scaling 思路，同时扩大宽度、深度和 FFN，而不是只堆单一维度；具体围棋形状仍以本项目 GPU 实测为准。
- 采用 ReZero 的“从零动态残差开始”思想，但保留 Pre-RMSNorm 和学习率 warmup。
- 不实现 CaiT class-attention、LayerScale 或 DeepNorm；它们与当前全局信息广播、深度缩放初始化或 Pre-RMSNorm 主干存在重叠和结构冲突。
