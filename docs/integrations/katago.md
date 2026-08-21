# KataGo 集成

项目固定使用未修改的官方 KataGo `v1.17.2` 子模块。它当前是独立的强教师与 GTP 引擎，
不是 Zero-TTT Python 包的搜索依赖，也不负责学生自博弈。

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

## Human-SL 分级教师

未来课程模式使用单独挂载的官方 Human-SL 模型和 `humanSLProfile`。每个阶段同时记录主分析
模型、Human-SL 模型、profile、配置指纹和实际 visits。仅设置 profile 不足以声明普通搜索
具有该等级；运行 manifest 必须显式保存 human policy 如何影响教学和对战搜索。

同一阶段的高预算教学和低预算对战使用相同 profile，教学 visits 必须更高。具体 rank 阶梯、
预算和权重参数等待标定，不写成默认棋力承诺。参见[分级教师](../workflows/curriculum-teachers.md)。

## 边界

- `katago-version` 不需要模型权重或 GPU 推理。
- Analysis/GTP 需要用户权重和兼容 GPU；完整分析不是无权重 CI 的硬条件。
- 本轮没有 Python 客户端、进程守护、自动恢复或远程 worker。
- 官方 KataGo 不能直接读取 Zero-TTT Transformer checkpoint。
- 项目不维护 KataGo C++ 补丁或外部 NN backend；学生搜索留在本地可选 MCTS。

上游资料：[KataGo 仓库](https://github.com/lightvector/KataGo)、
[Analysis Engine](https://github.com/lightvector/KataGo/blob/v1.17.2/docs/Analysis_Engine.md)、
[规则说明](https://lightvector.github.io/KataGo/rules.html)。
