# Human-SL 分级教师

状态：已接受的目标设计，尚未实现。教师是 AlphaZero MCTS 数据之外的平滑辅导来源，不替代
学生自博弈，也不改变 Learner 的监督优化方式。

## 阶段身份

每个教师阶段的运行 manifest 必须固定：

- Human-SL `profile`、主分析模型与 Human-SL 模型指纹；
- 教学、对战的 visits、human policy 影响参数与完整搜索配置哈希；
- MCTS 自博弈、教师 annotation 和 rehearsal 的混合比例；
- 评测规则、贴目、开局集、随机种子、双方颜色与选着配置。

profile 是课程类别，不是精确人类段位。教学与对战可以使用不同预算，但都必须保存真实
配置；不把预算或 profile 压成单一 `strength` 标签。

## 训练循环

1. 固定学生 publication，以 MCTS 自博弈采集实际到达的完整 trajectories。
2. 主动选点后，当前教师高预算生成 sidecar annotations。
3. 初接入按 70% MCTS、10% 教师、20% rehearsal 混合；固定评测无回退后才逐步变为
   60%/25%/15%。
4. Learner 更新 EMA 并发布；旧阶段关键样本按策略保留或 pin。
5. 冻结新 publication 和教师配置运行对战；学生稳定超过教师后提升 profile/visits。
6. 新教师接管主要 policy 标签前，也要验证其可靠胜过当前学生。

## 评测判据

- 每个方向至少完成 400 盘有效对局，学生执黑、执白各 200 盘；无结果不计并补局。
- 同一批次固定双方 agent、publication、教师指纹、预算、开局、种子和选着配置。
- 双方可使用各自已记录的固定搜索配置；不再要求学生关闭 MCTS。
- 被检验的一方点胜率不低于 55%，且 95% Wilson 胜率下界高于 50%。
- 学生通过表示需要升级教师课程，不阻止正常 publication；候选教师通过才可成为主要教师。

结果保存逐局身份、颜色和胜负；汇总可形成独立 `RatingSnapshot`，但评级未知或评级池变化
不阻塞训练。任何配置变化都开启新评测批次，不能与旧批次拼接满足门槛。
