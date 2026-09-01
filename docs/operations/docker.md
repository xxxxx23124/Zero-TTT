# Docker 运维

Zero-TTT 只维护 Docker 工作流。宿主机只需要 Git、Docker Compose、兼容 NVIDIA 驱动和
Container Toolkit；不维护宿主机 Python、PyTorch、CUDA 或编译器兼容性。

## 开发与测试

```powershell
git submodule update --init --recursive
docker compose build dev
docker compose run --rm dev python -m pytest -q
docker compose run --rm dev python -m pytest -q tests/unit
docker compose run --rm dev python -m pytest -q tests/integration
docker compose run --rm dev python -m pytest -q tests/quality
docker compose run --rm dev python -m ruff check src tests scripts
docker compose run --rm dev python scripts/check_docs.py
docker compose run --rm dev zero-ttt config-check --config configs/test.toml
docker compose run --rm dev zero-ttt train-smoke --config configs/test.toml
```

正式 CUDA profile 位于 `configs/profiles/`。数据、训练任务与用户提供的模型均不进入 Git。

## Web 训练中心

默认数据根目录为 `D:\datasets\Zero-TTT`。如需更改，在当前 PowerShell 会话设置：

```powershell
$env:ZERO_TTT_DATA_ROOT = 'E:\Zero-TTT-data'
```

目录必须包含 `raw`、`staging`、`manifests`、`processed` 和 `catalog`。启动入口是：

```powershell
docker compose up --build training-ui
```

该命令同时启动 `training-agent` 与 `tensorboard`。页面仅绑定 `127.0.0.1:8080`，TensorBoard
仅绑定 `127.0.0.1:6006`，训练代理只在 Compose 内部网络开放。停止 Compose 时代理向可暂停作业
转发 SIGTERM；训练完成当前 optimizer step，数据导入封存当前 shard。

目录挂载如下：

| Windows | 容器 | 权限 |
| --- | --- | --- |
| `<data-root>\raw` | `/datasets/raw` | 只读 |
| `<data-root>\staging` | `/datasets/staging` | 读写 |
| `<data-root>\manifests` | `/datasets/manifests` | 读写 |
| `<data-root>\processed` | `/datasets/processed` | 读写 |
| `<data-root>\catalog` | `/datasets/catalog` | 读写 |
| `<repo>\runs` | `/runs` | 训练代理读写、TensorBoard 只读 |

pip、Torch、Hugging Face、编译扩展和 uv 缓存仍使用可删除命名卷。业务数据不使用命名卷，
`docker compose down` 后仍可从 Windows 直接查看。只有用户明确决定时才清理缓存卷。

页面工作流见[本地 Web 训练中心](training-console.md)。真实 g170 垂直冒烟可为 `dev` 叠加
`compose.data.yaml`，它使用相同的 Windows 子目录 bind mount：

```powershell
docker compose -f compose.yaml -f compose.data.yaml run --rm dev `
  python scripts/data_smoke_test.py
```

## 高级 CLI

底层 `manifest-create`、`data-import`、`data-verify`、`snapshot-create`、`selfplay-collect` 和
`offline-imitation` 仍保留给测试、自动化和高级排障。普通用户不需要执行这些命令。正式训练方案
路径应使用 `configs/profiles/rtx4090l.toml`；`offline-imitation` 还必须显式提供 `--run-dir`，
避免把运行目录重新塞回实验超参数配置。

当前写入格式为 record/shard/catalog v4、来源与 mixture manifest v2、实验配置和模型产物 v8、
Web run spec v1、控制状态 v2。读取器只接受精确当前版本，不提供旧格式迁移。

## KataGo

```powershell
docker compose --profile katago build katago-version
docker compose --profile katago run --rm katago-version
docker compose --profile katago run --rm -T katago-analysis
docker compose --profile katago run --rm katago-gtp
```

后两个命令要求 `models/katago/` 中存在由 `KATAGO_MODEL_FILE` 指定的网络。配置和模型只读挂载。
KataGo 网络管理见[集成说明](../integrations/katago.md)，数据格式见
[序列化训练数据](../../src/zero_ttt/data/trajectory-storage.md)。
