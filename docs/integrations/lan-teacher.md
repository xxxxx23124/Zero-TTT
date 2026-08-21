# 局域网教师协议草案

状态：仅文档。训练机未来维护持久任务队列，装有官方强模型的教师机主动领取任务；训练机
不依赖教师机的固定地址，也不把任务仅保存在内存。

## TeacherTask

```json
{
  "schema_version": 1,
  "task_id": "uuid",
  "game_id": "content-hash",
  "position_index": 120,
  "board_size": 19,
  "rules": "tromp-taylor",
  "komi": 7.5,
  "initial_stones": [],
  "moves": [["B", "Q16"]],
  "human_sl_profile": "rank_5k",
  "max_visits": 4096,
  "requested_outputs": ["policy", "value", "score", "ownership"]
}
```

任务必须能独立重建局面。任务 ID 用于幂等提交；棋局内容哈希防止同一 ID 被不同内容复用。

## TeacherFingerprint 与结果

结果至少记录：

- KataGo 版本、模型文件 SHA-256、配置 SHA-256 和后端；
- Human-SL 模型指纹、profile 及 human policy 搜索参数（适用时）；
- task ID、完成时间、实际 visits 和搜索终止原因；
- SIDETOMOVE 视角的 policy 分布、value、score lead；
- ownership 可选，缺失时显式标记；
- 原始响应的内容哈希，便于审计和重新解析。

## 队列语义

- 状态为 `pending → leased → completed`，失败可回到 `pending`。
- worker 领取带期限租约并周期续租；崩溃后任务可被另一 worker 重领。
- 相同 task ID 的重复结果只有完全同指纹、同内容时才幂等接受。
- 教师模型或配置变化产生新指纹，不覆盖旧结果。
- profile、搜索预算或 human policy 参数变化同样产生不同任务身份。
- 超时、非法局面、OOM 和协议错误分别记录，达到重试上限后进入人工检查状态。

身份认证、传输加密、数据库、HTTP API、worker 和监控均延后实现。
