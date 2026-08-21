# Learner 与流程边界

状态：已接受的目标架构，尚未实现。当前代码仍使用 `zero_ttt.training.Trainer`；未来迁移不能
伪装成已经存在的公共 API。

## 组件职责

| 组件 | 唯一职责 |
| --- | --- |
| `model` | 定义纯 `nn.Module`、前向输出和可关闭的模型实验 |
| 目标 `learner` | 监督优化、EMA、训练 compile、梯度、checkpoint 和 publication |
| `inference` | 从不可变 publication 构造只读 `PositionEvaluator` |
| `data` | 持久样本契约、校验、混合和 `BatchSource` |
| 采集流程 | 离线棋谱、学生自博弈、KataGo 标注和课程调度 |
| KataGo adapter | JSON/坐标/视角与持久样本之间的转换 |

`Learner` 是小型门面，不是包含游戏、搜索、网络协议和调度的万能 Agent。目标操作沿用现有
Trainer 的能力：训练若干 step、保存、恢复和发布；输入仍只有 `BatchSource`/`TrainBatch`。
采集器不得持有训练模型、优化器或 EMA 的 Python 引用。

## 数据交接

采集器先写不可变分片及 manifest，再由 `BatchSource` 读取。每份可训练数据至少可追溯：

- 来源类型、规则、棋盘大小、贴目和局面/棋局标识；
- 学生 publication，或 KataGo 版本、模型与配置指纹；
- Human-SL profile、实际搜索预算和选着参数（适用时）；
- policy/value/score/ownership 的视角、有效 mask 和标签来源。

原始棋局、待标注局面和完成标注可以使用不同物理格式，但进入 Learner 前必须归一化为同一
`TrainBatch`。失败或指纹不一致的数据不能静默混入。

## 模型与显存所有权

只读 evaluator 从 BF16 slow publication 加载独立模型，不访问 Learner 内部地址。默认在同一
进程中同时常驻训练模型与 evaluator，但训练和采集按阶段顺序运行，不并发提交 CUDA 工作。

现有正式冒烟已经同时驻留 GPU FP32 fast、CPU FP32 EMA 和 GPU BF16 publication，并以训练
batch 16、累积 16 跑到峰值 `14.246 GiB reserved`。因此当前不增加优化器/模型换入换出；未来
任何生命周期改动仍须保持 14.5 GiB 验收线。

## 流程组合

运行级协调器只传递 publication 路径/版本、数据 manifest 和阶段结果。它可以顺序调用采集与
训练，但不复制 Learner、损失或 checkpoint 逻辑。不同采集流程可以独立文件实现；“并行”是
语义和数据来源的独立，不表示默认在同一 GPU 上并发执行。
