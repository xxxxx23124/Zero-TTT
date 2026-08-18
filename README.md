# Zero-TTT

Zero-TTT 是一个面向个人学习、实践和娱乐的围棋 AI 项目。

项目近期目标是在标准 19×19 棋盘上完成一套 AlphaZero 风格的训练与对弈闭环。当前已经有 Python 参考实现打通围棋规则、棋盘 Transformer、批量 MCTS、自博弈、SQLite 回放、`fast` 训练、`slow` EMA 发布和恢复；开源棋谱预训练与 GTP 是后续独立阶段。

这里不是严格的 AlphaZero 复现，也不以论文结论、排行榜成绩或最强棋力为目标。比起证明某个点子一定有效，项目更看重：能运行、能理解、能玩，并且方便随时加入一些有趣但未必成熟的实验。

## 当前状态

- 里程碑 1、4、5 的首版参考实现和里程碑 7 的默认关闭超网络分支已经进入 `src/zero_ttt/`。
- 当前工程主线是 19×19 围棋 AlphaZero 风格基线。
- 默认模型规格面向 RTX 4090 Laptop 16 GB：约 3.09 亿参数的 Transformer Encoder，训练物理 batch 为 16。
- tiny 配置的“自博弈 → 回放 → 优化 → EMA → 发布 → 恢复”自动化测试已经通过；正式模型的 Linux/CUDA 编译显存验收仍需在目标 GPU 容器中执行。
- 首个可玩成果计划提供 GTP v2 接口，可接入 Sabaki、q5Go 等棋盘软件。
- 原有的 TTT / 神经快记忆方向已经暂停并归档，但没有被放弃；未来条件成熟时仍可能重新探索。

## 路线概览

1. 已实现并测试 19×19 No-Suicide Tromp–Taylor 规则、状态与共享特征。
2. 已实现棋盘 Transformer、Python PUCT MCTS 和单 GPU 分阶段自博弈训练闭环。
3. 在目标 Linux GPU 容器完成正式 batch 16 的 compile/显存验收和小数据过拟合。
4. 通过统一 `GameSource → GameRecord → ReplayStore` 边界接入许可清晰的开源棋谱并做监督预训练。
5. 提供 GTP 引擎，完成实际人机对弈。
6. 试验默认关闭的完整低秩超网络及后续可插拔点子。

闭环不设置候选模型对冠军模型的竞技评估或晋升门槛。`fast` 权重接受梯度，`slow` 权重通过 EMA 跟随并用于自博弈、GTP 和发布；验证指标只用于暴露问题，不阻塞训练或触发自动回退。

具体阶段、交付物和验收条件见[实施计划](docs/implementation_plan.md)，模型、训练和搜索参数见[模型与搜索设计](docs/model_search_design.md)。未经验证、允许频繁修改的想法单独记录在[实验点子](docs/ideas.md)中。

## 本地验证

```bash
python -m pip install -e ".[dev]"
python -m pytest
zero-ttt smoke --config configs/test.toml
```

`loop`、`selfplay` 和 `train` 同样只接受一个版本化 TOML 文件，不提供逐项命令行或环境变量覆盖。例如 `zero-ttt selfplay --config configs/test.toml` 会在 `runs/test/` 生成可恢复状态。正式长跑使用 `configs/rtx4090l.toml`。

## 文档

- [文档索引](docs/README.md)
- [实施计划](docs/implementation_plan.md)
- [模型与搜索设计](docs/model_search_design.md)
- [设计决策](docs/design_decisions.md)
- [实验点子](docs/ideas.md)
- [开发日志](docs/devlog/README.md)
- [旧快记忆研究归档](docs/archive/fast-memory-2026/README.md)

## Docker GPU 开发环境

项目提供基于 PyTorch 2.13.0、CUDA 13.2 和 cuDNN 9 的 Linux 开发容器。镜像包含 NVCC、C/C++ 编译器和 Ninja，可用于后续编译自定义 CUDA 扩展。

宿主机需要：

- NVIDIA GPU 和兼容 CUDA 13.x 的驱动；
- 使用 WSL 2 后端且已启用 GPU 支持的 Docker Desktop；
- 足够的镜像空间（基础开发镜像压缩后约 11 GB）。

CUDA Toolkit 和 cuDNN 已包含在容器中，宿主机不需要另外安装它们。

### 构建和进入容器

```bash
docker compose build
docker compose run --rm dev
```

项目源码会挂载到容器的 `/workspace`。Hugging Face、pip、Torch 扩展和 uv 的缓存保存在 Docker 命名卷中，删除临时容器或重新构建镜像不会清空缓存。

### 验证 GPU 环境

```bash
docker compose run --rm dev nvidia-smi
docker compose run --rm dev python scripts/docker_smoke_test.py
docker compose run --rm dev python -m pytest
docker compose run --rm dev python scripts/model_smoke_test.py
```

第一项冒烟测试检查源码挂载、NVCC、PyTorch/CUDA/cuDNN 版本、GPU 型号与计算能力。模型冒烟脚本分别运行关闭/启用超网络的正式 batch 16 BF16 编译前后向，并强制检查峰值保留显存不超过 14.5 GiB。

### 维护命令

```bash
# 拉取基础镜像并重新构建
docker compose build --pull

# 不使用 Docker 构建缓存，完整重建
docker compose build --pull --no-cache

# 停止由 Compose 启动的服务，保留下载和编译缓存
docker compose down

# 停止服务并永久删除项目的 Docker 命名卷缓存
docker compose down --volumes
```

当前镜像只提供主项目的 PyTorch GPU 开发基座，不会自动安装 `third_party` 中各参考项目彼此不同的依赖。现有三个 TTT 相关子模块是长期方向的历史参考，不是当前 AlphaZero 主线的运行时依赖。
