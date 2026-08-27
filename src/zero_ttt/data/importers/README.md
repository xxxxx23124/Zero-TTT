# 数据 Importer

Importer 是来源专属的适配层。它接收源文件与显式 `SourceManifest`，流式产出
`TrajectoryRecord`、可选 `AnnotationRecord` 和导入统计；它不写 NPZ/SQLite、不构造
`TrainBatch`，也不依赖 Learner。

`RecordImporter` 定义协议，`ImporterRegistry` 根据 manifest 中的显式格式身份选择实现。
新增来源应新增独立 Importer，而不是把来源猜测或分支堆入现有实现。

## KataGo g170 SGF

`KataGoSgfImporter` 只接受声明为 `katago-g170-sgfs-zip` 的资产，并流式读取 ZIP 中的
line-delimited SGF。处理顺序是：

1. 校验 manifest、资产 SHA-256 与容器结构；
2. 用 sgfmill 解析 typed property，并拒绝 collection、variation 或不支持的棋局头；
3. 把坐标和结果转换为本地域值，再由 `GameState` 重放验证全部 moves；
4. 生成带来源身份、训练范围、标签 mask 和内容哈希的逻辑记录；
5. 将单局格式问题记录为 `ImportEvent` rejection，继续处理同一可信资产中的其他棋局。

当前只支持空棋盘、黑方先行、19×19 且可由本地规则合法重放的普通棋局。KataGo rich NPZ、
Leela 和职业棋谱尚无 Importer；不能因为文件扩展名相似而复用 g170 语义。

Importer 产出的记录仍须由上层 `data.pipeline` 交给共同的 shard sink 和 Catalog。持久化与
采样规则见[数据包说明](../README.md)。
