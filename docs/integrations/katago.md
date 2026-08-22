# KataGo 集成

项目固定使用未修改的官方 KataGo `v1.17.2` 子模块。它是独立强教师与 GTP 引擎，不是
学生 MCTS 依赖，也不负责加载 Zero-TTT checkpoint。

## 初始化和版本检查

```bash
git submodule update --init --recursive
docker compose --profile katago build katago-version
docker compose --profile katago run --rm katago-version
```

源码固定在 `third_party/KataGo`；镜像通过 CUDA 后端编译，并在最终阶段只复制二进制。
升级 KataGo 必须单独更新 submodule、镜像标签、配置、manifest 和兼容性测试。

## 权重

1. 从可信来源取得许可证允许用途的官方网络。
2. 保存到 `models/katago/`，计算 SHA-256。
3. 复制并填写 `manifest.example.toml`，但不要提交网络文件。
4. 将文件名传给 Compose：

```bash
KATAGO_MODEL_FILE=your-model.bin.gz docker compose --profile katago run --rm katago-gtp
```

PowerShell 可先执行 `$env:KATAGO_MODEL_FILE = "your-model.bin.gz"`。权重只读挂载；仓库不
自动下载，也不替用户推断许可。

## Analysis Engine

```bash
docker compose --profile katago run --rm -T katago-analysis
```

未来教师请求显式提供 `rules: "tromp-taylor"`、19×19、贴目、moves 和预算。配置固定
`reportAnalysisWinratesAs = SIDETOMOVE`，下游仍记录视角。结果写独立 annotation shard，
不重写学生 trajectory。

## Human-SL 分级教师

课程模式记录主分析模型、Human-SL 模型、profile、配置指纹、实际 visits 和 human policy
影响参数。profile 只是可复现课程标签；高预算教学和对战配置分别记录，不直接宣称人类段位。
参见[分级教师](../workflows/curriculum-teachers.md)。

## 边界

- `katago-version` 不需要模型权重或 GPU 推理；Analysis/GTP 需要二者。
- 本轮没有 Python 客户端、进程守护、自动恢复或远程 worker。
- 项目不维护 KataGo C++ 补丁或外部 NN backend。
- 学生自博弈由本地 `GameState` 加 OpenSpiel Python MCTS 完成。

上游资料：[KataGo 仓库](https://github.com/lightvector/KataGo)、
[Analysis Engine](https://github.com/lightvector/KataGo/blob/v1.17.2/docs/Analysis_Engine.md)、
[规则说明](https://lightvector.github.io/KataGo/rules.html)。
