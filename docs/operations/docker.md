# Docker 运维

Zero-TTT 只维护 Docker 工作流。宿主机只需要 Git、Docker Compose、兼容 NVIDIA 驱动和
Container Toolkit；不维护宿主机 Python、PyTorch、CUDA 或编译器兼容性。

## 开发与测试

```bash
git submodule update --init --recursive
docker compose build dev
docker compose run --rm dev python -m pytest -q
docker compose run --rm dev python scripts/check_docs.py
docker compose run --rm dev zero-ttt config-check --config configs/test.toml
docker compose run --rm dev zero-ttt model-smoke --config configs/test.toml
docker compose run --rm dev zero-ttt train-smoke --config configs/test.toml
```

`third_party/open_spiel` 当前只固定上游源码和许可证，尚未进入开发镜像，也没有运行时安装
命令。加入 adapter 时必须在 Dockerfile 中显式固定构建依赖并增加最小搜索冒烟。

生产显存测试使用 `scripts/model_smoke_test.py` 和 CUDA 正式配置，成本远高于 CPU 单元测试，
不应在每次文档修改后运行。

## KataGo

```bash
docker compose --profile katago build katago-version
docker compose --profile katago run --rm katago-version
docker compose --profile katago run --rm -T katago-analysis
docker compose --profile katago run --rm katago-gtp
```

后两个命令要求 `models/katago/` 中存在由 `KATAGO_MODEL_FILE` 指定的网络。配置目录和模型
目录都只读挂载；日志写入容器临时目录。

## 数据与清理

- `runs/`、trajectory/annotation shards、SQLite 数据库和 KataGo 网络不进入 Git。
- 当前重构不主动删除旧运行产物，但不再支持旧 replay/checkpoint schema。
- `docker compose down` 保留命名缓存卷；只有用户明确决定时才使用 `down --volumes`。
- 不把下载密钥、局域网凭据或未经核验的许可证信息写入镜像。

KataGo 网络管理见[集成说明](../integrations/katago.md)，未来数据布局见
[序列化训练数据](../architecture/trajectory-storage.md)。
