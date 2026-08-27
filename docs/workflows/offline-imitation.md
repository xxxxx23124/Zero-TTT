# 监督冷启动

状态：首版 SGF policy-supervision 垂直切片已实现。目标是先让 Transformer 形成棋盘空间、合法性、局部形状和基本胜负直觉，
为 MCTS 自博弈提供可用 prior/value；不是复制最强搜索，也不承诺达到某个人类段位。

## 数据路径

1. 建立带来源、许可、文件 SHA-256、规则和过滤统计的 manifest。
2. 导入许可清晰的历史棋谱；首选候选之一是 KataGo `g170` 的 CC0 自博弈棋谱。
3. 只接收本地规则可合法重放的 19×19 普通棋局；跨规则数据只训练 policy，完全兼容且结果
   可验证时才启用 value/score/ownership。
4. 在开局、中盘、终局分层抽样；加入合法着、终局、局部形状等规则相关任务。
5. 可用固定版本的直观/低阶段教师补充 policy、value、score 或 ownership，但不以最强搜索
   覆盖所有局面。
6. 封存不可变 NPZ trajectory/annotation 分片，再由 `BatchSource` 产生 `TrainBatch`。

普通棋谱只有实战落子时，policy 可以 one-hot，但必须标记来源。有教师分布时也保留教师
指纹、预算和有效 mask；缺失辅助标签不能填成假零值。

## 训练和验收

- 冷启动阶段 100% 使用监督语料；进入 AlphaZero 后初始保留 20% rehearsal。
- 训练/验证按完整棋局或来源切分，防止相邻状态泄漏。
- 棋盘对称增强对单局连续子序列只能选择一次 D4 变换。
- 以合法性、固定局面损失、校准和低预算 MCTS 稳定性判断是否“开眼”，不以来源声明的
  rank 直接充当模型棋力。

当前 CLI 提供 `manifest-create`、`manifest-check`、`data-import`、`data-verify`、
`snapshot-create` 和 `offline-imitation`。g170 rich NPZ 因旧 SGF 缺少可靠连接键暂不接入。

来源 rank、平台和评级体系原样保存，不自动换算成统一 Elo。完整强度元数据规则见
[序列化训练数据](../../src/zero_ttt/data/trajectory-storage.md)，当前 shard-local microbatch 算法见
[数据包说明](../../src/zero_ttt/data/README.md)。
