# MCTS + 神经快记忆研究归档（2026）

本目录保存 Zero-TTT 在 2026 年 8 月形成的第一版研究设计。该方案尝试把 MCTS 搜索经验压缩进一盘棋内持续存在的双私有神经快记忆，并通过局部元学习训练可学习的初始化模板。

这个方向没有被证明错误，也没有被项目放弃。它目前只是因为研究与工程复杂度过高而暂停，让项目先集中完成一个能够训练、能够对弈、方便个人学习和玩耍的 19×19 AlphaZero 风格基线。

## 归档内容

| 文件 | 内容 |
| --- | --- |
| [研究设计 PDF](research_design.pdf) | 当时整理的完整研究设计与数学语义 |
| [研究设计 LaTeX](research_design.tex) | PDF 的可编辑源码 |
| [旧实施计划](implementation_plan.md) | 围绕快记忆方案制定的六阶段实施路线 |
| [旧设计决策](design_decisions.md) | D-001 至 D-011 的原始决定与理由 |

## 归档规则

- 上述四个文件保持归档时的原貌，不跟随当前 AlphaZero 主线修改。
- 当前项目计划和决策分别以 [`../../implementation_plan.md`](../../implementation_plan.md) 与 [`../../design_decisions.md`](../../design_decisions.md) 为准。
- `third_party` 中的 TTT 参考子模块继续保留原路径，避免不必要地修改 Git 子模块结构。
- 将来重新启动快记忆方向时，应先基于已经可用的 AlphaZero 基线重新评估方案，并在当前设计决策中记录恢复范围。
