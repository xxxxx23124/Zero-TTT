# KataGo 输入

首期只把 KataGo G170 棋谱作为原始数据源。用户将 ZIP 资产放入
`raw/katago/g170/selfplay/`；Data Service 以只读方式扫描、记录相对路径、大小与 SHA-256，
随后通过 `data-bootstrap` 导入。

用户提供的 KataGo 模型保留在仓库 `models/katago`，但当前没有教师 RPC 服务，也不属于
三条有限流程。未来增加教师能力时，应新增 Worker capability 和显式产物契约，不能让
Data、Trainer 或 Self-play 直接导入另一服务的实现。
