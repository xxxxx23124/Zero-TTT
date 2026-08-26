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

真实 g170 垂直冒烟只读挂载外部数据目录：

```powershell
$env:ZERO_TTT_DATASET_ROOT = 'D:\datasets\Zero-TTT'
docker compose -f compose.yaml -f compose.data.yaml run --rm dev `
  python scripts/data_smoke_test.py
```

该脚本只处理首个归档中稳定顺序的 64 个合格棋局；catalog、shard、checkpoint 和 publication
都写入容器临时目录。全量导入应使用同一挂载配合 `zero-ttt manifest-create` 与
`zero-ttt data-import`，并显式把输出目录挂载到可写卷。

完整 g170 导入与快照命令如下；这些命令只读 `/datasets/zero-ttt/raw`，输出写入
`/datasets/work` 命名卷：

```powershell
docker compose -f compose.yaml -f compose.data.yaml run --rm dev zero-ttt manifest-create `
  --dataset-id katago-g170 --source-type katago-g170-sgfs-zip `
  --license-id CC0-1.0 --license-url https://katagoarchive.org/g170/LICENSE.txt `
  --source-root /datasets/zero-ttt --glob 'raw/katago/g170/selfplay/*.zip' `
  --output /datasets/work/manifests/g170.json

docker compose -f compose.yaml -f compose.data.yaml run --rm dev zero-ttt manifest-check `
  --manifest /datasets/work/manifests/g170.json --source-root /datasets/zero-ttt

docker compose -f compose.yaml -f compose.data.yaml run --rm dev zero-ttt data-import `
  --manifest /datasets/work/manifests/g170.json --source-root /datasets/zero-ttt `
  --store-root /datasets/work/processed --catalog /datasets/work/catalog/catalog.sqlite

docker compose -f compose.yaml -f compose.data.yaml run --rm dev zero-ttt data-verify `
  --store-root /datasets/work/processed --catalog /datasets/work/catalog/catalog.sqlite

docker compose -f compose.yaml -f compose.data.yaml run --rm dev zero-ttt snapshot-create `
  --store-root /datasets/work/processed --catalog /datasets/work/catalog/catalog.sqlite `
  --seed 7 --split train --validation-fraction 0.1
```

`snapshot-create` 输出不可变 snapshot ID。把该 ID 传给 `offline-imitation --snapshot`；正式运行
前应在配置中把 `runtime.run_dir` 指向持久写入位置。本仓库不自动启动全量导入或长期训练。

当前数据格式为 record/shard/catalog v2，不迁移早期 v1 本地产物。旧 catalog 或 shard 会给出
明确的重建错误；先自行归档或删除旧输出目录，再从保留的 source manifest 重新执行 import 和
snapshot-create。工具不会自动删除运行产物。

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
