# `zero_ttt.inference`

`inference` 把不可变 publication 转换为只读 `PositionEvaluator`，并为并发搜索请求提供统一
聚批。张量与返回值语义见[公共契约](../../../docs/architecture/contracts.md)。

## Publication evaluator

`PublicationPositionEvaluator` 校验 publication 文件哈希和当前 artifact schema，从嵌入配置
重建 `PolicyValueTransformer`，严格加载 slow/EMA 权重，并将模型固定为 eval、无梯度状态。
CPU 与 CUDA 都使用 FP32 权重和前向，CUDA matmul/cuDNN TF32 均关闭；非 FP32 publication
或推理输入直接拒绝，不做自动转换。

`model_version` 同时包含 run、模型版本和 publication SHA-256。一次搜索或一盘棋必须冻结该
身份；publication 变化后不得复用旧 evaluator 或 cache。

后端使用固定 `inference_batch_size`。不足一批时复制最后一个真实样本补齐，只返回真实前缀；
补齐项不进入搜索结果。

## 并发聚批

`BatchedInferenceBroker` 是多棋局线程与 evaluator 之间的唯一队列：

- 在 `batch_wait_ms` 窗口内收集至固定 batch 上限；
- 按完整 `GameState.identity()` 合并同批重复请求，并维护有界 LRU cache；
- 编码局面后调用一次 evaluator，再把结果分发给所有等待者；
- 统计请求、cache hit、真实/补齐 evaluation、满批比例和推理延迟；
- 后端异常传递给本批所有等待者，关闭时终止并 join 工作线程。

broker 不解释 MCTS value 备份或选着；这些属于 [`search`](../search/README.md)。它也不拥有
publication 生命周期；自博弈应用服务只在进入采集阶段时创建并关闭它。
