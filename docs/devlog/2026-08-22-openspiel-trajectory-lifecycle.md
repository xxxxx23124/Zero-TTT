# 2026-08-22：OpenSpiel、AlphaZero 与序列数据路线

## 目标

修正“纯策略自博弈可以作为默认提升路径”和“项目未来自行维护最小 MCTS”的旧判断，并把
冷启动、MCTS AlphaZero、主动教师辅导和序列优先持久数据统一为一条可渐进实现的路线。

## 完成内容

- 将 OpenSpiel v2.0.1 作为 `third_party/open_spiel` 子模块，精确固定到提交
  `112b77704631fc2ce7ad8e4581f6ca09798ce15a`。
- 规定只采用 Python `MCTSBot` 与自定义 Evaluator；本地 `GameState` 仍是棋规真源。
- 把默认采集改为 64 simulations 起步、可标定到约 100 的 MCTS 自博弈，访问数作为 policy。
- 建立“监督冷启动 → AlphaZero → 主动选点 → 分级教师”的统一生命周期和初始混合比例。
- 定义一盘一逻辑 trajectory、多盘一 immutable NPZ shard、annotation sidecar 与 SQLite 控制面。
- 明确棋力不是必填数据属性；先保存 agent/search 身份，未来以独立 rating snapshot 关联。
- 更新 README、架构、工作流、集成、快权重、路线图与决策记录；历史 devlog 保持原文。

## OpenSpiel 冒烟证据

方案制定阶段在 Python 3.12 Docker 环境验证了固定版本：19×19 动作空间为 362，自定义外部
Evaluator 可驱动 `MCTSBot`；以薄 `pyspiel.Game/State` 包装本地 `GameState` 的最小搜索可
运行。源码检查确认每次 `step` 新建根，`restart_at` 不维护树。

内置 Go 状态的多子自杀与 superko/重复局面终局行为不符合当前 Tromp–Taylor 契约，因此
决定只复用搜索控制流，不复用其 Go 棋规。该验证只证明适配可行，不代表生产 adapter、
GPU 聚批或自博弈采集器已经实现。

## 验证结果

- `python scripts/check_docs.py` 通过全部 Markdown 大小与本地链接检查。
- `python -m pytest tests/quality/test_docs.py -q`：`1 passed`；宿主机因沙箱无法写 `.pytest_cache`
  产生一个非测试失败 warning。
- `git diff --check` 通过；全局冲突扫描只命中历史 devlog/已标记被取代的决策和明确的消融说明。
- 子模块 HEAD、gitlink 与 tag 分别核对为指定完整提交和 `v2.0.1`。

没有运行 GPU 训练、棋力实验或新 adapter 冒烟；本页前述 OpenSpiel 证据来自方案制定阶段。

## 产生的决策

- D-036：OpenSpiel 搜索、本地棋规。
- D-037：MCTS 默认的统一训练生命周期。
- D-038：序列优先的持久数据。

D-036/D-037 取代 D-032 的纯策略优先与自维护 MCTS 方向；D-037 补充 D-035 的教师课程，
取消“学生毕业赛必须关闭 MCTS”的限制。

## 问题与下一步

本轮没有修改 Python API，也没有实现 adapter、持久 schema、教师服务或运行时安装。下一步
先做 trajectory shard/SQLite tiny 垂直切片和冷启动数据导入，再接 OpenSpiel adapter 与
64-simulation 自博弈；扩大采集前必须实现多棋局并发和统一 evaluator 聚批。
