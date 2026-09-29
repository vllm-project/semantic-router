---
title: 模型性能评测
sidebar_label: 模型评测
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/training/model-performance-eval.md"
  outdated: false
---

# 模型性能评测 {#model-performance-evaluation}

在将候选生成模型分配给路由决策前先评测它们。仓库的 `model_eval` 脚本通过 OpenAI 兼容端点测量选择题准确率，绘制按类别的 MMLU-Pro 结果，并将这些结果转为规范配置脚手架。

此工作流回答哪个被评测模型在所选数据集和提示模式下表现最好。它不证明同一排名会在生产流量上成立。

## 工作流产出 {#what-the-workflow-produces}

| 步骤 | 输出 | 用途 |
|------|--------|-----|
| MMLU-Pro 评测 | 每题 CSV 加上 `analysis.json` 和 `summary.json` | 按类别和总体准确率比较模型 |
| ARC Challenge 评测 | 每题 CSV 加上总体分析 | 独立的选择题健全性检查 |
| 绘图 | 条形图或热力图 | 检查类别级差异 |
| 配置生成 | `config.eval.yaml` 脚手架 | 为评审播种 provider 绑定、Model Card 和领域分数 |

只有 MMLU-Pro 结果会进入 `result_to_config.py`，因为 ARC 输出不包含生成器使用的领域类别。

## 前置条件 {#prerequisites}

- 一个或多个提供待比较模型的 OpenAI 兼容端点
- 与通过 `--models` 传入的值匹配的已服务模型 ID
- 脚本使用的 Hugging Face 数据集所需的网络和数据集访问
- 足够的 provider 容量和预算，以便将每个所选问题发送给每个候选模型

从仓库根目录创建隔离环境：

```bash
cd src/training/model_eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

脚本会保存提示词、模型响应、正确性标签和计时数据。
选择对所评测内容有合适保留和访问控制的输出目录。

## 运行 MMLU-Pro {#run-mmlu-pro}

先用小样本验证模型 ID 和响应格式：

```bash
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --models phi4 qwen3-0.6B \
  --samples-per-category 10 \
  --output-dir results/mmlu-smoke
```

重要选项：

- `--models` 接受空格分隔的模型 ID 或一个逗号分隔值。省略时，脚本查询端点的 `/models` API。
- `--categories` 限制 MMLU-Pro 类别。
- `--samples-per-category` 控制每个所选类别的样本数；默认是 `5`，因此正式运行应显式设置。
- `--use-cot` 为思维链提示变体创建单独的 `_cot` 结果目录。
- `--concurrent-requests` 增加并行请求。从 `1` 开始，以免速率限制和排队悄然扭曲比较。
- `--temperature` 默认为 `0.0`，`--seed` 默认为 `42`。

每个模型和提示模式会得到如 `results/mmlu-smoke/phi4_direct/` 这样的目录，包含：

- `detailed_results.csv`
- `analysis.json`
- `summary.json`

准确率按成功请求计算。始终将 `successful_queries` 和 `failed_queries` 与准确率一起报告；排除失败而不披露，会让不可靠端点看起来更好。

## 运行 ARC Challenge {#run-arc-challenge}

将 ARC 用作第二个数据集，而不是领域分数来源：

```bash
python arc_challenge_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --models phi4 qwen3-0.6B \
  --samples 100 \
  --output-dir results/arc
```

`--samples` 是总样本数，默认为 `20`。其他生成、并发、模型和提示模式选项与 MMLU-Pro 脚本镜像。

## 绘制类别结果 {#plot-category-results}

绘图器递归读取 MMLU-Pro `analysis.json` 文件：

```bash
python plot_category_accuracies.py \
  --results-dir results/mmlu-smoke \
  --plot-type heatmap \
  --output-file results/mmlu-smoke/category-accuracy.png
```

使用 `--plot-type bar` 绘制分组条形图。`--sample-data` 仅渲染合成数据以预览图表布局；永远不要将该输出作为评测结果发布。

## 生成配置脚手架 {#generate-a-configuration-scaffold}

```bash
python result_to_config.py \
  --results-dir results/mmlu-smoke \
  --output-file config.eval.yaml \
  --backend-endpoint 127.0.0.1:8000 \
  --backend-protocol http \
  --provider-id vllm \
  --api-format openai
```

生成器创建带有以下内容的 v0.3 文档：

- 平均评测最高的模型作为 `providers.defaults.model`
- 每个被评测逻辑模型的一个 provider 绑定和 Model Card
- 每个观察到的 MMLU-Pro 类别的一个领域信号
- 每个类别的排名 `model_scores`
- 空的 `routing.decisions` 列表
- 响应缓存、工具、嵌入、prompt guard 和分类器的稀疏默认值

生成的文档具有此顶层形状。此处省略列表和模块体；检查 `config.eval.yaml` 以查看被评测模型、分数和类别信号。

```yaml
version: v0.3
listeners: []
providers:
  defaults:
    model: evaluated-model
  models: []
routing:
  modelCards: []
  signals:
    domains: []
  decisions: []
global:
  stores:
    response_cache: {}
  integrations:
    tools: {}
  model_catalog:
    embeddings: {}
    modules:
      prompt_guard: {}
      classifier: {}
```

同一基础模型的 Direct 和 CoT 结果目录会折叠为一个逻辑模型。对每个类别，生成器保留更高的观察准确率，并根据其内置类别映射设置 `use_reasoning`。评审该选择，而不是将其视为已学习的推理策略。

除非覆盖，生成的后端地址会应用到每个模型。用每个 provider 的真实端点拓扑、凭据、可靠性设置和定价替换它。

## 将脚手架转为路由策略 {#turn-the-scaffold-into-a-routing-policy}

`config.eval.yaml` 有意不完整：

- `listeners` 为空。
- `routing.decisions` 为空。
- provider 绑定使用命令行默认值，而不是部署发现。
- 评测类别可能与面向用户的决策不匹配。
- 稀疏的 `global` 分区可能与你的运行时或安全策略不匹配。

将由评测派生的 Model Card 和分数合并进完整配置。添加解释每个类别何时影响路由的决策，然后校验结果：

```bash
vllm-sr config validate --config config.yaml
```

不要整份替换生产配置。保留其 listeners、密钥、provider 专用端点、重试和健康策略、定价、服务和存储设置。

## 评测路由结果 {#evaluate-the-routing-outcome}

使用未用于选择模型、提示模式、类别映射或阈值的留出数据集或生产代表性回放。报告：

- 按类别和重要工作负载切片的质量
- 请求失败和排除的样本
- 添加决策后的所选模型分布
- 端到端延迟、token 用量和 provider 成本
- 与默认模型和最佳单模型基线的比较
- 数据集修订、模型修订、源提交、配置和命令

MMLU-Pro 生成器为被评测答案排名；它不测试完整 Router 数据路径。集成脚手架后运行端到端基准。可用套件见[基准测试](../benchmarking/overview)。

## 源 {#source}

- [`src/training/model_eval`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_eval)
- [模型训练概览](./training-overview)
- [当前模型目录](./model-catalog)
- [基于 ML 的模型选择](./ml-model-selection)
