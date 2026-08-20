# Zero-TTT

Zero-TTT 是一个仅在 Docker 中维护的 19×19 围棋学生模型训练研究项目。项目保留
Transformer、通用训练器、EMA、checkpoint 和棋规；不再维护自己的 MCTS。官方 KataGo
作为独立的强教师和 GTP 引擎。

## 当前可用

- 625M 与全关闭基线的策略—价值 Transformer 配置。
- Tromp–Taylor 棋规、特征编码、模型损失和通用 `BatchSource` 训练接口。
- 合成数据驱动的模型/训练冒烟测试。
- 固定到 v1.17.2 的 KataGo CUDA 镜像、Analysis Engine 与 GTP 服务入口。

尚未实现：棋谱导入、学生自博弈、在线蒸馏、局域网教师、快权重，以及让 KataGo
直接搜索 Zero-TTT Transformer。路线图中的“未来”内容不代表现有功能。

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
- [离线模仿](docs/workflows/offline-imitation.md)
- [在线蒸馏](docs/workflows/online-distillation.md)
- [快权重研究](docs/research/fast-weights/overview.md)
- [论文索引](paper/README.md)

许可证：MIT。第三方源码和模型权重遵循各自的许可证。
