# 本地 Web 训练中心

Zero-TTT 的普通用户入口是本机 NiceGUI 页面。页面通过容器内部的 FastAPI 代理启动独立子进程，
不直接导入 Learner、数据实现，也不接触 Docker Socket。

## Windows 目录

默认数据根目录是 `D:\datasets\Zero-TTT`，可用 `ZERO_TTT_DATA_ROOT` 覆盖：

```text
D:\datasets\Zero-TTT\
├── raw\          # 原始 ZIP，只读挂载
├── staging\      # 可重建中间文件
├── manifests\    # 来源 manifest 与校验记录
├── processed\    # trajectory / annotation NPZ shard
└── catalog\      # SQLite catalog
```

训练任务位于仓库的 `runs/<run_id>/`。每个任务保存 `run.json`、冻结的 `experiment.toml`、
checkpoint、publication、TensorBoard、mixture 和控制状态。业务文件不写入 Docker 命名卷；
命名卷只保存 pip、Torch、Hugging Face 等可删除缓存。

## 启动

确认 Docker Desktop 已运行，然后在仓库根目录执行：

```powershell
docker compose up --build training-ui
```

浏览器打开 `http://127.0.0.1:8080`。TensorBoard 位于 `http://127.0.0.1:6006`，训练代理只在
Compose 内部网络开放。

## 首次数据准备

把 KataGo g170 ZIP 放入 `raw\katago\g170\selfplay\`，然后在网页依次执行：

1. 扫描并校验：建立包含路径、大小和 SHA-256 的来源 manifest；
2. 试导入 1000 局：验证 importer、shard 和 catalog 闭环；
3. 继续全量导入：按 game ID 去重，自动续接试导入结果；
4. 校验数据：恢复 orphan、核对已登记 shard，并写校验记录；
5. 创建训练 Snapshot：固定使用 external/train、seed 7 和 10% validation 切分。

扫描可在源文件边界安全暂停，导入可在原子 shard 边界安全暂停。下次全量导入会跳过已登记棋局。
只有全量导入完成且校验记录仍与 manifest/catalog 统计一致时，页面才允许创建 snapshot。

## 训练任务

创建任务时输入名称，并选择训练方案与 cold snapshot。任务建立后，这两个身份不可修改；更换方案
或 snapshot 必须创建新任务。方案 TOML 只保存模型、学习率、搜索、自博弈和 mixture 等稳定参数，
页面不提供修改这些深层超参数的接口。

单次运行时长是操作参数，不进入配置哈希。训练、安全暂停、MCTS 收集与 warm-start 沿用原有
原子 checkpoint/publication、sealed 自博弈任务和显式数据身份迁移规则。

## 恢复与并发

数据和训练共享一个全局作业锁，同一时间只有一个写操作。训练在 optimizer step 后暂停；采集在
完整 actor 轮次封存后暂停。页面轮询代理缓存，不反复加载完整 checkpoint。

旧 `configs/console.toml`、交互式菜单和旧 console state 不再读取。历史目录不会被自动删除；只有
带当前 `run.json` 与冻结配置的新任务目录会出现在页面。

内部边界见[控制编排说明](../../src/zero_ttt/console/README.md)，Docker 测试命令见
[Docker 运维](docker.md)。
