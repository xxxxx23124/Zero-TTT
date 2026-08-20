# 论文清单

论文 PDF 仅作研究参考，不属于项目源码许可证。新增文件前记录来源；不要把“阅读过”写成
“项目已经实现”。

## 当前训练路线

- [DAgger](https://arxiv.org/abs/1011.0686) — 在线学生状态分布与专家重标注。
- [Expert Iteration](https://arxiv.org/abs/1705.08439) — 搜索改进策略再蒸馏回网络；仓库已有
  `Thinking Fast and Slow  with Deep Learning and Tree Search.pdf`。
- [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) — KataGo 的辅助目标
  和高效自博弈训练。
- [Policy Distillation](https://arxiv.org/abs/1511.06295) — 策略分布蒸馏的基础方法。
- [Online Knowledge Distillation with Diverse Peers](https://arxiv.org/abs/1912.00350) — 在线知识
  蒸馏参考；与本项目的搜索教师方案并不完全相同。

## 模型与未来快权重

- `Titans Learning to Memorize at Test Time.pdf`
- `Test-Time Training Done Right.pdf`
- `Learn at Test Time.pdf`
- `ATLAS Learning to Optimally Memorize the Context at Test Time.pdf`
- `In-Place Test-Time Training.pdf`
- `End-to-End Test-Time Training for Long Context.pdf`
- `HyperNetworks.pdf` 与三篇 Hypernetwork 设计参考。
- `DenseFormer Enhancing Information Flow in Transformers.pdf`
- `DeepNet Scaling Transformers to 1000 Layers.pdf`
- `ReZero is All You Need.pdf`、`CaiT Going Deeper with Image Transformers.pdf`。

## 取舍

当前实现只使用共享超网络、稀疏深度混合、训练/EMA/checkpoint 基础设施。DAgger 风格在线
聚合、Expert Iteration 教师循环和快权重均是未来工作。
