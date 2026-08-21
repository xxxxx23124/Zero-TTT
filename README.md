# Zero-TTT

Zero-TTT 是一个仅在 Docker 中维护的 19×19 围棋学生模型训练研究项目。项目保留
Transformer、通用训练器、EMA、checkpoint 和本地棋规。已接受的目标架构由本项目运行
学生纯策略自博弈，并在未来可选本地 MCTS；未修改的官方 KataGo 只作为 Human-SL/强搜索
教师和独立 GTP 引擎。

## 当前可用

- 625M 与全关闭基线的策略—价值 Transformer 配置。
- Tromp–Taylor 棋规、特征编码、模型损失和通用 `BatchSource` 训练接口。
- 合成数据驱动的模型/训练冒烟测试。
- 固定到 v1.17.2 的 KataGo CUDA 镜像、Analysis Engine 与 GTP 服务入口。

尚未实现：目标 `Learner` 包、持久样本、棋谱导入、学生自博弈、分级教师、在线蒸馏、
局域网教师、本地 MCTS 和快权重。不会修改 KataGo 来加载或搜索 Zero-TTT checkpoint；
路线图中的“目标/未来”不代表现有功能。

## 初始化与验证

```bash
git submodule update --init --recursive
docker compose build dev
docker compose run --rm dev python -m pytest -q
docker compose run --rm dev python scripts/check_docs.py
docker compose run --rm dev zero-ttt config-check --config configs/test.toml
docker compose run --rm dev zero-ttt train-smoke --config configs/test.toml
```

项目不保证宿主机 Python、CUDA 或编译器环境可用。所有正式命令均以 Compose 服务为入口。

## KataGo

```bash
docker compose --profile katago build katago-version
docker compose --profile katago run --rm katago-version
```

Analysis/GTP 需要用户自行把权重放入 `models/katago/`。参见
[KataGo 集成说明](docs/integrations/katago.md)与[Docker 运维](docs/operations/docker.md)。

## 文档入口

- [文档索引](docs/README.md)
- [目标架构](docs/architecture/overview.md)
- [Learner 与流程边界](docs/architecture/learner-and-workflows.md)
- [离线模仿](docs/workflows/offline-imitation.md)
- [在线蒸馏](docs/workflows/online-distillation.md)
- [学生自博弈](docs/workflows/student-selfplay.md)
- [Human-SL 分级教师](docs/workflows/curriculum-teachers.md)
- [快权重研究](docs/research/fast-weights/overview.md)
- [论文索引](paper/README.md)

许可证：MIT。第三方源码和模型权重遵循各自的许可证。
