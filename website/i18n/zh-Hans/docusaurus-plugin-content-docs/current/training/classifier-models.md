---
title: mmBERT-32K 分类器模型
sidebar_label: 分类器模型
translation:
  source_commit: "2de78e816d34699cf31245e2ad3b16d6b0a803df"
  source_file: "docs/training/classifier-models.md"
  outdated: false
---

# mmBERT-32K 分类器模型 {#mmbert-32k-classifier-models}

当前分类器家族将同一多语言、长上下文 ModernBERT 编码器适配到若干路由决策。共享基础使分词和编码器行为保持一致，而每个任务有自己的标签、数据准备、训练损失和输出头。

## 共享架构与训练模式 {#shared-architecture-and-training-pattern}

意图、越狱、反馈、模态和事实核查模型使用序列分类：

```text
request -> mmBERT-32K encoder -> pooled representation -> task classifier -> label
```

PII 模型使用 token 分类：

```text
request -> mmBERT-32K encoder -> one classifier output per token -> BIO entities
```

训练脚本对 ModernBERT 注意力和 MLP 投影应用 LoRA 更新，同时训练任务头。一次运行可以保留 adapter，或将其合并进基础权重。因此 `-lora` 和 `-merged` 产物共享同一逻辑架构和标签契约。

对于五个标准序列/token 工作流，仓库还提供便捷目标：

```bash
make train-mmbert32k-intent
make train-mmbert32k-jailbreak
make train-mmbert32k-feedback
make train-mmbert32k-factcheck
make train-mmbert32k-pii
```

覆盖目标默认值前，先用 `--help` 运行所选 Python 入口。数据集下载、输出检查点和缓存应放在 Git 之外。

## 意图分类器 {#intent-classifier}

意图模型预测 14 个学科领域之一：biology、business、chemistry、computer science、economics、engineering、health、history、law、math、other、philosophy、physics 或 psychology。

它是在 MMLU-Pro 题目加上改进 `other` 回退的补充样本上训练的 14 路序列分类器。标准目标使用 LoRA rank 32 和 alpha 64、5 个 epoch、batch 16，以及学习率 `2e-5`。评测报告准确率和加权 F1；部署评测还应检查每类召回率以及与 `other` 的混淆。

MMLU-Pro 只发布 `validation`（70 行）和 `test`（12032 行），因此训练池必须来自 `test`。训练器在采样任何内容前先保留分层的 20% `test`，将这些行排除在梯度路径之外，并将其行索引以及在其上测得的指标写入检查点旁的 `heldout_eval.json`。引用该数字：整个 `test` 划分上的准确率覆盖了模型训练过的行，因此不是留出证据。

产物：
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-merged)、
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-intent-classifier-lora)。

## 越狱检测器 {#jailbreak-detector}

越狱模型是 `benign` 对 `jailbreak` 的二分类序列分类器。训练结合 ToxicChat、Salad-Data 攻击样本，以及显式的短长攻击模式增强。流水线同时训练分类头和 LoRA adapter，并用留出分类指标选择检查点。

已发布 Model Card 记录 LoRA rank 48 和 alpha 96。当前标准目标默认为 rank 32、alpha 64、5 个 epoch、batch 16 和学习率 `2e-5`；复现已发布配置时覆盖 rank 和 alpha。分别评测良性假阳性和漏检攻击，并包含多语言、混淆、长上下文和间接提示切片；仅靠聚合准确率不是安全阈值。

产物：
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-merged)、
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-jailbreak-detector-lora)。

## 反馈检测器 {#feedback-detector}

反馈模型根据用户的跟进消息预测四种状态：

| 标签 | 含义 |
| --- | --- |
| `SAT` | 回答让用户满意 |
| `NEED_CLARIFICATION` | 用户需要澄清 |
| `WRONG_ANSWER` | 回答看起来不正确 |
| `WANT_DIFFERENT` | 用户想要不同的结果或方法 |

它使用加权交叉熵补偿类别不平衡。标准 LoRA 运行使用 rank 64 和 alpha 128、最多 10 个 epoch、batch 16、学习率 `2e-5`、按 macro F1 提前选择检查点，以及 512 token 训练上限。

产物：
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-merged)、
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-feedback-detector-lora)。

## 模态路由器 {#modality-router}

模态路由器预测下游响应应如何产生：

| 标签 | 路由 |
| --- | --- |
| `AR` | 自回归文本模型 |
| `DIFFUSION` | 图像生成模型 |
| `BOTH` | 文本解释加视觉输出 |

训练组装文本请求、图像生成提示词和混合模态样本。`BOTH` 类可以包含已评审的种子/模板样本，以及通过 OpenAI 兼容端点合成的可选样本。除非显式设置，训练器会根据数据集大小自动选择 LoRA rank，使用 focal loss 和类别权重，对严重少数类过采样，并按验证 F1 选择。已发布 Model Card 记录 rank 16、alpha 32、10 个 epoch、batch 32 和学习率 `2e-5`。当前脚本默认为 8 个 epoch，并为其默认的 6000 样本数据集选择 rank 16。

```bash
python src/training/model_classifier/modality_routing_classifier/\
modality_routing_bert_finetuning_lora.py --help
```

产物：
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-modality-router-merged)、
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-modality-router-lora)。

## 事实核查分类器 {#fact-check-classifier}

事实核查模型预测 `FACT_CHECK_NEEDED` 或 `NO_FACT_CHECK_NEEDED`。它是路由模型：决定请求是否应进入验证路径；它不判断主张是否为真。

正例是来自 QASPER 和 Natural Questions 等信息寻求问题。负例包括创意写作、代码和其他非信息寻求请求。构建器平衡两类，并创建分层的训练、验证和测试划分。标准目标使用 LoRA rank 32 和 alpha 64、5 个 epoch、batch 16 和学习率 `2e-5`；脚本按验证 F1 选择。

产物：
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-merged)、
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-factcheck-classifier-lora)。

## PII 检测器 {#pii-detector}

PII 模型是序列分类模式的例外。它使用 token 分类头和 BIO 编码：`B-TYPE` 标记实体的第一个 token，`I-TYPE` 继续它，`O` 标记非实体 token。已发布模型将 17 种实体类型暴露为 35 个标签（`O` 加上每种实体类型两个标签）。

已发布 Model Card 记录 Presidio 训练、LoRA rank 32、5 个 epoch、batch 16 和学习率 `1e-4`。当前标准目标用 70/30 的 AI4Privacy/Presidio 混合、字符跨度对齐到 mmBERT 子词 token、rank 48、alpha 96 和 8 个 epoch 扩展该方法。选择并报告实体级 F1，而不是被 `O` token 主导的 token 准确率。

产物：
[`merged`](https://huggingface.co/llm-semantic-router/mmbert32k-pii-detector-merged)、
[`LoRA`](https://huggingface.co/llm-semantic-router/mmbert32k-pii-detector-lora)。

## 校验产物契约 {#validate-the-artifact-contract}

发布或配置分类器前，验证以下全部内容：

- 分词器和基础模型修订与训练运行匹配；
- `id2label` 和 `label2id` 保留文档中的顺序；
- adapter 包含任务头，或合并模型包含完整模型权重；
- 运行时使用相同的截断和规范化规则；
- adapter 和合并 logits 在固定样本上一致；
- 留出指标和失败切片与产物一起存储。

训练入口和产物映射位于
[`src/training/model_classifier`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier)。
分层内容安全模型请继续阅读[训练安全分类器](./mmbert-safety-classifier)。
