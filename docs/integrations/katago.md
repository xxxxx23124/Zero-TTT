# KataGo 集成

项目固定使用官方 KataGo `v1.17.2` 子模块。它当前是独立的强教师与 GTP 引擎，不是
Zero-TTT Python 包的搜索依赖。

## 初始化和版本检查

```bash
git submodule update --init --recursive
docker compose --profile katago build katago-version
docker compose --profile katago run --rm katago-version
```

源码固定在 `third_party/KataGo`；镜像通过 CUDA 后端编译，并在最终阶段只复制二进制。
升级 KataGo 必须单独更新 submodule、镜像标签、配置、manifest 和兼容性测试。

## 权重

1. 从可信来源取得与许可证允许用途相符的官方网络。
2. 保存到 `models/katago/`，计算 SHA-256。
3. 复制并填写 `manifest.example.toml`，但不要提交网络文件。
4. 将文件名传给 Compose：

```bash
KATAGO_MODEL_FILE=your-model.bin.gz docker compose --profile katago run --rm katago-gtp
```

PowerShell 可先执行 `$env:KATAGO_MODEL_FILE = "your-model.bin.gz"`。权重以只读目录挂载，
仓库不会自动下载，也不会把未经核验的许可信息写入 manifest。

## Analysis Engine

Analysis 服务读 stdin 的逐行 JSON，并将逐行 JSON 写到 stdout：

```bash
docker compose --profile katago run --rm -T katago-analysis
```

每个未来教师请求都必须显式提供 `rules: "tromp-taylor"`、19×19 棋盘、贴目、moves 和分析
预算。配置固定 `reportAnalysisWinratesAs = SIDETOMOVE`，下游仍需记录该视角。

## 边界

- `katago-version` 不需要模型权重或 GPU 推理。
- Analysis/GTP 需要用户权重和兼容 GPU；完整分析不是无权重 CI 的硬条件。
- 本轮没有 Python 客户端、进程守护、自动恢复或远程 worker。
- 官方 KataGo 不能直接读取 Zero-TTT Transformer checkpoint。

上游资料：[KataGo 仓库](https://github.com/lightvector/KataGo)、
[Analysis Engine](https://github.com/lightvector/KataGo/blob/v1.17.2/docs/Analysis_Engine.md)、
[规则说明](https://lightvector.github.io/KataGo/rules.html)。
