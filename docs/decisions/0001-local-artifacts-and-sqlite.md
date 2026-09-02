# ADR 0001：本地不可变产物与 SQLite 租约

状态：已采用。

当前部署是一台 Windows 主机、Docker Desktop 和一张 NVIDIA GPU。系统选择本地文件系统
作为不可变对象存储后端，并为 Control 与 Data 分别使用 SQLite WAL。大型数据不经 HTTP
复制，服务只传带哈希的引用。

这项决定避免为单机部署引入 Redis、NATS、S3 或 Kubernetes，同时仍保留清晰的所有权、
崩溃恢复和未来替换存储后端的契约边界。代价是调度吞吐量受单个 Control SQLite 写者限制，
且共享目录需要由 Compose 权限严格约束；对当前有限流程和单 GPU 资源规模足够。

