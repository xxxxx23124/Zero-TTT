# Zero-TTT

Zero-TTT 是一个仅在 Docker 中维护的 19×19 围棋学生模型训练研究项目。项目保留
Transformer、Learner、EMA、checkpoint 和本地 Tromp–Taylor 棋规；目标训练路线是
“监督冷启动 → OpenSpiel MCTS 自博弈 AlphaZero → 疑惑局面挑选 → 分级 KataGo 教师辅导”。

## 当前可用

- 625M 与全关闭基线的策略—价值 Transformer 配置。
- Tromp–Taylor 棋规、特征编码、模型损失和通用 `BatchSource` 训练接口。
- g170 SGF Importer、版本化 trajectory/annotation NPZ、SQLite catalog 与快照采样。
- `CatalogBatchSource`、样本尺度调度的 `Learner`、schema v6 checkpoint 和不可变 publication。
- 从不可变 publication 加载的固定 batch-16 evaluator、OpenSpiel PUCT 适配和可恢复 MCTS 自博弈采集。
- trajectory/shard/catalog v4、来源过滤 snapshot 与加权 `MixtureBatchSource`。
- Docker 交互式训练控制台、8 小时软停止、状态恢复和冷启动到 mixture 的 warm-start。
- 合成数据与 64 盘真实 g170 棋谱驱动的 Docker 垂直冒烟测试。
- 固定到 v1.17.2 的 KataGo CUDA 镜像、Analysis Engine 与 GTP 服务入口。
- 固定到 v2.0.1 指定提交的 OpenSpiel 源码，并在开发镜像中从锁定依赖构建 `pyspiel`。

尚未实现：无人值守 AlphaZero 自动长期循环、KataGo rich-NPZ 连接、分级教师、在线蒸馏、局域网教师、
数据窗口和快权重。KataGo 也不会加载 Zero-TTT checkpoint；路线图中的“目标/未来”不代表
现有功能。

## 初始化与验证

```bash
git submodule update --init --recursive
docker compose build dev
docker compose run --rm dev python -m pytest -q
docker compose run --rm dev python -m pytest -q tests/unit
docker compose run --rm dev python -m pytest -q tests/integration
docker compose run --rm dev python scripts/check_docs.py
docker compose run --rm dev zero-ttt config-check --config configs/test.toml
docker compose run --rm dev zero-ttt train-smoke --config configs/test.toml
```

配置好 `configs/console.toml` 的 cold-start snapshot 后，交互式控制台使用：

```bash
docker compose run --rm training-console
```

真实数据冒烟通过 `compose.data.yaml` 将外部目录只读挂载；命令见
[Docker 运维](docs/operations/docker.md)。

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
- [统一训练生命周期](docs/workflows/training-lifecycle.md)
- [序列化训练数据](docs/architecture/trajectory-storage.md)
- [内部格式版本](docs/architecture/versioning.md)
- [OpenSpiel MCTS 边界](docs/integrations/mcts-compatibility.md)
- [Learner 与流程边界](docs/architecture/learner-and-workflows.md)
- [Human-SL 分级教师](docs/workflows/curriculum-teachers.md)
- [快权重研究](docs/research/fast-weights/overview.md)
- [论文索引](paper/README.md)

许可证：MIT。第三方源码和模型权重遵循各自的许可证。
