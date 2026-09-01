# 内部格式版本

所有由 Zero-TTT 程序读取的持久格式都在 `zero_ttt.versioning` 集中登记。各格式独立演进，
但读取器只接受唯一的当前版本；项目不保留旧格式分支，不自动迁移或删除任何运行产物。

| 格式 | 当前版本 | 旧产物处理 |
| --- | ---: | --- |
| 实验配置 | v8 | 从当前 profile 创建新任务 |
| checkpoint、publication 与 publication 索引 | v8 | 新建 run 并重新发布 |
| trajectory / annotation record | v4 | 重新导入或采集 |
| NPZ shard | v4 | 重新生成 processed 数据 |
| SQLite catalog | v4 | 重建 catalog 与 snapshot |
| 来源 manifest | v2 | 从原始资产重新生成 |
| 训练 mixture manifest | v2 | 从当前 snapshot 重新创建 |
| 自博弈 task manifest | v2 | 重新采集任务 |
| Web 训练任务描述 | v1 | 从页面创建新任务 |
| 训练控制状态 | v2 | 核对 run 产物后仅重建控制状态 |

写入方必须从具名 `SchemaSpec.current` 取得版本，读取方必须调用同一 spec 的严格校验。
任一格式发生破坏性变化时只提升该格式，旧版和未知未来版得到同一种明确的重建错误。

KataGo、OpenSpiel 和 Python 包版本属于依赖/发布版本；规则、特征、采样及哈希域中的 `vN`
属于算法身份。它们不是持久格式兼容列表，不纳入 schema 注册表。

原始 SGF/ZIP、KataGo 网络等源资产可以保留。派生 shard、catalog、snapshot、mixture、
自博弈任务、checkpoint 和 publication 需要由使用者在确认路径后清理并重新生成；程序不会
替使用者执行破坏性清理。
