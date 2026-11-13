---
title: 添加扩展字段
description: 如何向 vllm-sr System One 扩展集添加一个字段。
is_mtpe: false
translation:
  source_commit: "6b782822edead98397acc6401112e6ec1e45af96"
  source_file: "docs/community/system-one-schemas/adding-an-extension-field.md"
  outdated: false
---

# 添加扩展字段 {#adding-an-extension-field}

扩展规范 `config/schemas/systemone/vllm-sr-5.0.extensions.yaml` 对 vllm-sr 在标准之上增加的每个字段只定义一次：层级、schema、出现的版本、支持它的模型族，以及标准目标如何处理它。

1. **添加条目。** 给出 `level`（request、question、answer 等）、`since`、`families`、`schema` 和 `summary`，以及 `standard_target`：标准部分已承载其含义时为 `strip`，需要它的规则必须在加载时失败时为 `refuse`，只有 vllm-sr 服务器才会发送的字段为 `absent`。优先使用标准自己的形式：与标准字段重叠的字段会被拒绝，两份规范对同一字段定义不同时生成器会失败。
2. **重新生成。** `make systemone-schemas-generate` 写出 Go 字段表、模型运行时的字段表（`vllm_srun/api/systemone_fields.py`）、标准目标使用的去除清单，以及生成的扩展参考和兼容性矩阵。`make systemone-schemas-check` 在产物过期时失败。
3. **实现**：在读取它的地方实现，即支持它的模型运行时模型族，以及 Router 发送或读取它时的 Router codec；若是自上次发布以来的第一个新增字段，还要提升扩展集的小版本（`extension_version`）。
4. **测试**：通过两种 listener 和客户端 codec 测试：5.0 服务器接受它，标准目标按声明去除或拒绝它，相邻的未知字段仍被拒绝。
