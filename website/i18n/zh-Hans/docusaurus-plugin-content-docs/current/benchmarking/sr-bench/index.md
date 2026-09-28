---
title: sr-bench 1.0
description: 在可复用的冻结任务上比较 MoM 和单模型的能力、成本、延迟与 token 消耗。
translation:
  source_commit: "31fa0fdab6787b3b149894db259af13c8ce42f5a"
  source_file: "docs/benchmarking/sr-bench/index.md"
  outdated: true
---

# sr-bench 1.0

sr-bench 使用相同的冻结题目比较 MoM 入口与单模型。CLI 和 **Dashboard → Evaluation** 共用一个持久化服务、运行 ID、结果与报告。先用小规模开发集改进路由，再用不相交的保留集验收。

## 选择题量

下表是**每个目标的完整任务数**，不是模型调用次数。代码、Agent、裁判和用户模拟器可能产生多次调用；辅助调用费用单独报告。

| 基准 ID | 能力 | Smoke | Quick/dev | Standard/holdout |
| --- | --- | ---: | ---: | ---: |
| `mmlu-pro` | 14 个学科的知识 | 14 | 500 | 2,000 |
| `gpqa-diamond` | 科学推理 | 4 | 40 | 158 |
| `hle` | HLE 纯文本推理 | 4 | 40 | 200 |
| `livecodebench` | v6 累积编程题 | 2 | 30 | 150 |
| `scicode` | 科学编程完整主问题 | 1 | 3 | 20 |
| `terminal-bench-2.1` | 隔离终端任务 | 1 | 3 | 15 |
| `simpleqa-verified` | 事实准确性 | 5 | 100 | 500 |
| `arc-agi-2` | 公开评测谜题与精确网格输出 | 2 | 12 | 80 |
| `tau3` | τ³ 三个文本交互领域 | 3 | 12 | 60 |
| **总计** | | **36** | **740** | **3,183** |

用 `vllm-sr benchmark catalog` 检查当前安装版本。允许只选择某个能力切片；500 题 MMLU-Pro 开发集不是上游 12,032 题的全量结果。缺少任一基准时，没有完整 sr-bench 分数。

数据准备固定来源版本、内容哈希、任务 ID、种子和分层算法。Smoke 是 quick 的子集，standard 与 quick 不相交。SciCode 子问题保留在同一个任务中。公开题目不能宣称无污染；已看过标签的 GPQA 结果仍需注明复测。
