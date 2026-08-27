# KataGo 模型目录

把用户自行取得的 KataGo 网络权重放在这里。权重不会被 Git 跟踪，也不会在构建时下载。

复制 `manifest.example.toml`，填写来源、许可、SHA-256 和用途；运行服务时通过
`KATAGO_MODEL_FILE` 指定该目录中的文件名。详细命令见
[`../../docs/integrations/katago.md`](../../docs/integrations/katago.md)。

该 TOML 目前只作人工来源记录，不由 Zero-TTT 解析，因此不声明内部 schema 版本。
