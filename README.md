# Zero-TTT

Zero-TTT 是一个面向个人学习、实践和娱乐的围棋 AI 项目。

项目近期目标是在标准 19×19 棋盘上完成一套 AlphaZero 风格的训练与对弈闭环：先使用许可清晰的开源棋谱进行预训练，再逐步接入神经网络引导的 MCTS、自博弈训练、模型评估和可供棋盘软件调用的 GTP 引擎。棋谱既可以来自人类对局，也可以来自开源程序的自博弈；如果数据包含 MCTS 访问分布或根价值等额外信息，项目会优先利用这些监督信号。

这里不是严格的 AlphaZero 复现，也不以论文结论、排行榜成绩或最强棋力为目标。比起证明某个点子一定有效，项目更看重：能运行、能理解、能玩，并且方便随时加入一些有趣但未必成熟的实验。

## 当前状态

- 项目处于文档整理和实现准备阶段，尚无正式训练代码。
- 当前工程主线是 19×19 围棋 AlphaZero 风格基线。
- 首个可玩成果计划提供 GTP v2 接口，可接入 Sabaki、q5Go 等棋盘软件。
- 原有的 TTT / 神经快记忆方向已经暂停并归档，但没有被放弃；未来条件成熟时仍可能重新探索。

## 路线概览

1. 实现并验证 19×19 围棋规则、状态表示、SGF/自博弈数据读取。
2. 使用开源棋谱预训练策略—价值网络。
3. 完成 PUCT MCTS、批量神经网络推理和确定性测试。
4. 打通自博弈、经验回放、训练、候选模型评估和 checkpoint 恢复。
5. 提供 GTP 引擎，完成实际人机对弈。
6. 在不破坏标准基线的前提下加入“三速大脑”等可插拔实验。

具体阶段、交付物和验收条件见[实施计划](docs/implementation_plan.md)。未经验证、允许频繁修改的想法单独记录在[实验点子](docs/ideas.md)中。

## 文档

- [文档索引](docs/README.md)
- [实施计划](docs/implementation_plan.md)
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
```

冒烟测试会检查源码挂载、NVCC、PyTorch/CUDA/cuDNN 版本、GPU 型号与计算能力，并在 GPU 上执行一次矩阵乘法。

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
