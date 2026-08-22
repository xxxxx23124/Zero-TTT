# 局域网教师协议草案

状态：仅文档。训练机未来维护 SQLite 持久任务队列；装有官方模型的 2070 Super 12 GiB
笔记本教师机主动领取任务，未来可再加入 3060 台式机。能力差异通过 worker capability 与
实际预算记录，不假设两台机器吞吐相同。

## TeacherTask

```json
{
  "schema_version": 1,
  "task_id": "uuid",
  "game_id": "content-hash",
  "ply": 120,
  "rules": "tromp-taylor",
  "komi": 7.5,
  "initial_stones": [],
  "moves": [["B", "Q16"]],
  "human_sl_profile": "rank_5k",
  "max_visits": 4096,
  "requested_outputs": ["policy", "value", "score", "ownership"]
}
```

任务必须能独立重建局面。task ID 用于幂等提交；game ID 与 ply 必须命中已登记 trajectory。

## TeacherFingerprint 与结果

结果至少记录：

- KataGo 版本、模型文件 SHA-256、配置 SHA-256、后端和 worker capability；
- Human-SL 模型指纹、profile 及 human policy 搜索参数（适用时）；
- task ID、完成时间、实际 visits、终止原因和原始响应哈希；
- SIDETOMOVE 视角的 policy、value、score，以及可选 ownership 的 mask。

验证后结果写不可变 annotation shard，并以 `(game_id, ply, teacher_fingerprint)` 连接基础
trajectory。教师升级产生新指纹并共存，不覆盖旧标签。

## 队列语义

- SQLite 只保存任务、租约、结果索引、分片路径与校验，不保存大型训练数组。
- 状态为 `pending → leased → completed`；worker 周期续租，崩溃后任务可重领。
- 相同 task ID 只有同指纹、同内容时才幂等接受；否则进入人工检查。
- 超时、非法局面、OOM 和协议错误分别记录，达到重试上限后停止自动重试。
- 教师机主动拉取，不共享数据库文件，也不要求固定教师地址。

身份认证、传输加密、HTTP API、worker 和监控均延后实现。选点比例与去重规则见
[主动选点](../workflows/online-distillation.md)。
