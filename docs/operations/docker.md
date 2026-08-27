# Docker 运维

Zero-TTT 只维护 Docker 工作流。宿主机只需要 Git、Docker Compose、兼容 NVIDIA 驱动和
Container Toolkit；不维护宿主机 Python、PyTorch、CUDA 或编译器兼容性。

## 开发与测试

```bash
git submodule update --init --recursive
docker compose build dev
docker compose run --rm dev python -m pytest -q
docker compose run --rm dev python -m pytest -q tests/unit
docker compose run --rm dev python -m pytest -q tests/integration
docker compose run --rm dev python -m pytest -q tests/quality
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

当前新写入格式为 record/shard/catalog v4，配置与 checkpoint/publication 为 v6；其余内部格式见
[版本矩阵](../architecture/versioning.md)。读取器只接受精确当前版本，旧 full checkpoint 不能
恢复，旧 publication 也不能用于自博弈，必须用当前 Learner 重新生成。

`third_party/open_spiel` 固定为 v2.0.1 指定提交；开发镜像从该源码和 Dockerfile 中锁定的
Abseil/JSON/pybind/DDS 提交构建 `pyspiel`。

MCTS 自博弈与 80/20 mixture 的手动分阶段示例：

```bash
zero-ttt selfplay-collect --config configs/rtx4090l.toml \
  --publication /runs/published/run/step_000000001000/model.pt \
  --store-root /datasets/work/processed --catalog /datasets/work/catalog/catalog.sqlite \
  --games 256

zero-ttt snapshot-create --store-root /datasets/work/processed \
  --catalog /datasets/work/catalog/catalog.sqlite --seed 7 \
  --validation-fraction 0 --source-kind selfplay --task-id TASK_ID

zero-ttt mixture-create --component SELFPLAY_SNAPSHOT=0.8 \
  --component COLDSTART_SNAPSHOT=0.2 --output /datasets/work/mixture.json

zero-ttt offline-imitation --config configs/rtx4090l.toml \
  --store-root /datasets/work/processed --catalog /datasets/work/catalog/catalog.sqlite \
  --mixture /datasets/work/mixture.json --steps 100
```

`mixture-create` 的 snapshot ID 必须是 catalog 输出的 64 位小写十六进制 SHA-256。

`selfplay-collect` 的 JSON 摘要包含真实与补齐 evaluation 数、满批比例、推理延迟、
simulations/s、棋规 CPU 总耗时及 GPU 峰值；RTX 4090 正式冒烟应留存该输出。

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
- 程序不主动删除旧运行产物，也不支持旧 replay、checkpoint、publication 或 manifest schema。
- 切换时保留原始 SGF/ZIP 与 KataGo 网络；确认挂载路径后，人工清理并重建 processed shard、
  catalog、snapshot、mixture、自博弈任务和 run 产物。
- `docker compose down` 保留命名缓存卷；只有用户明确决定时才使用 `down --volumes`。
- 不把下载密钥、局域网凭据或未经核验的许可证信息写入镜像。

KataGo 网络管理见[集成说明](../integrations/katago.md)，未来数据布局见
[序列化训练数据](../architecture/trajectory-storage.md)。
