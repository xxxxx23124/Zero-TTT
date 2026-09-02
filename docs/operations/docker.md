# Docker 运维

先设置业务数据根目录；默认值为 `D:/datasets/Zero-TTT`：

```powershell
$env:ZERO_TTT_DATA_ROOT = 'D:/datasets/Zero-TTT'
docker compose build
docker compose up -d control data-worker trainer-worker selfplay-worker ui tensorboard
docker compose ps
```

端口仅绑定本机：UI `8080`、Control API `8090`、TensorBoard `6006`。

权威质量门禁：

```powershell
docker compose run --rm dev python -m ruff check .
docker compose run --rm dev python -m ruff format --check .
docker compose run --rm dev pyright
docker compose run --rm dev python -m pytest -q
docker compose run --rm dev python scripts/generate_contracts.py --check
docker compose config --quiet
git diff --check
```

隔离的 Compose 端到端验收使用项目专属命名卷，不挂载正式业务目录：

```powershell
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev run --rm --no-deps dev python scripts/compose_e2e_test.py prepare /datasets
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e up -d --wait
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev run --rm --no-deps dev python scripts/compose_e2e_test.py run bootstrap
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e restart control ui
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev run --rm --no-deps dev python scripts/compose_e2e_test.py run alpha
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev down -v
```

GPU 验收分别执行驱动/严格 FP32、完整 optimizer step 和并发 MCTS 自博弈：

```powershell
docker compose run --rm dev python scripts/docker_smoke_test.py
docker compose run --rm dev python scripts/model_smoke_test.py --configs configs/profiles/rtx4090l.toml --default-optimizer-steps 1 --accumulation-steps 1 --disable-compile
docker compose run --rm dev python scripts/selfplay_gpu_smoke_test.py
```

Control 重启后会从 `control.sqlite` 恢复作业；过期租约重新排队。Data 与各 GPU Worker 启动
时会重用已提交的内容寻址产物，临时文件不会被当作完成结果。Trainer 与 Self-play 共同竞争
`gpu-exclusive` 租约，因此单 GPU 主机上不会并发运行。

业务目录：

```text
raw/                         用户输入，只读
work/                        可清理的作业临时文件
artifacts/data/              Data 唯一写
artifacts/models/            Trainer 唯一写
artifacts/selfplay/          Self-play 唯一写
state/control/control.sqlite Control 独占
state/data/data.sqlite       Data 独占
```
