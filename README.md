# Zero-TTT

Zero-TTT 是面向单机单 GPU 的围棋训练系统，采用单仓多包和独立服务部署：Control、Data、
Trainer、Self-play 与 NiceGUI UI 只通过版本化 HTTP 作业契约和不可变产物引用协作。

## 快速开始

```powershell
$env:ZERO_TTT_DATA_ROOT = 'D:/datasets/Zero-TTT'
docker compose build
docker compose up -d
```

- UI：<http://127.0.0.1:8080>
- Control OpenAPI：<http://127.0.0.1:8090/docs>
- TensorBoard：<http://127.0.0.1:6006>

业务数据不写入 Git 工作区。目录、服务所有权、三条有限流程和验证命令见
[当前文档](docs/README.md)。
