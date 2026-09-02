# 三条有限流程

流程模板每次执行一次后结束，不会自行提交下一轮。

## data-bootstrap

依次执行原始资产扫描与哈希、小规模试导入、校验、全量导入、完整校验、训练 snapshot 和
验证 snapshot。Data 单写者租约保证同一时间只有一个数据写作业。

## cold-start

Run 创建时冻结实验 Profile 和训练 snapshot。Trainer 在有限步数预算内训练，原子提交
checkpoint，并发布只包含慢权重和冻结模型配置的 publication。

## alpha-zero-round

Self-play 冻结一个 publication 并收集有限棋局；Data 验证 bundle 后准入并建立新的
self-play snapshot；Trainer 使用冻结的 cold/self-play mixture 恢复训练并发布新版本。

再次迭代必须由用户重新提交 `alpha-zero-round`，因此资源消耗和失败边界始终可见。

