# Zero-TTT

Zero-TTT 是一个仍处于研究设计阶段的个人项目，探索如何把 MCTS 产生的搜索经验压缩进一盘棋内持续存在的神经快记忆。

项目当前没有正式代码实现。稳定的算法逻辑与会频繁变化的工程计划被刻意分开维护。

## 文档

- [文档索引](docs/README.md)
- [研究设计（PDF）](docs/research_design.pdf)
- [研究设计（LaTeX）](docs/research_design.tex)
- [实施计划](docs/implementation_plan.md)
- [设计决策](docs/design_decisions.md)
- [开发日志入口](docs/devlog/README.md)

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

当前镜像只提供主项目的 PyTorch GPU 开发基座，不会自动安装 `third_party` 中 PyTorch 与 JAX 基线各自不同的依赖。
