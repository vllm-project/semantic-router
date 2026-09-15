---
title: 训练 mmBERT-32K 安全分类器
sidebar_label: 安全分类器
translation:
  source_commit: "e8c4109fd4151ad0c7c0163c8ead375bef882ddf"
  source_file: "docs/training/mmbert-safety-classifier.md"
  outdated: false
---

# 训练 mmBERT-32K 安全分类器 {#train-the-mmbert-32k-safety-classifiers}

安全工作流按顺序使用两个分类器：

```text
prompt -> Level 1: safe / unsafe
                    |
                    +-- safe   -> continue normal routing
                    +-- unsafe -> Level 2: one of nine hazard outputs
```

当二元策略决策已足够时使用 Level 1。当不安全请求必须按危害类型路由、记录或不同处理时，添加 Level 2。
第二个模型不打算在 Level 1 接受为安全的请求上运行。

## 已发布产物架构 {#published-artifact-architecture}

当前集合中的两个产物都是
[`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base)
上 `ModernBertForSequenceClassification` 的 PEFT LoRA adapter。它们将输入截断到 512 token，并适配四组注意力/MLP 投影：`attn.Wqkv`、`attn.Wo`、`mlp.Wi` 和 `mlp.Wo`。

| 任务 | 头 | 已发布产物形态 |
| --- | --- | --- |
| Level 1 | 两类序列头 | [`mmbert-safety-binary-merged`](https://huggingface.co/llm-semantic-router/mmbert-safety-binary-merged)，PEFT adapter |
| Level 2 | 九类序列头 | [`mmbert-safety-binary-hazard`](https://huggingface.co/llm-semantic-router/mmbert-safety-binary-hazard)，PEFT adapter |

Level 1 名称以 `-merged` 结尾，但其已发布文件包含 `adapter_model.safetensors` 和 `adapter_config.json`，而不是独立基础权重。用其 adapter 配置声明的基础模型加载它。

## 当前 32K 训练架构 {#current-32k-training-architecture}

已核对工作流在
[`mmbert-32k-yarn`](https://huggingface.co/llm-semantic-router/mmbert-32k-yarn)
上训练后继产物，同时保留相同的两个头、标签、数据策略、LoRA 目标和 512 token 安全输入上限。它可以为任一级别导出 adapter 和完整合并形态，并在发布前验证其 logits。

不要把已有的 `mmBERT-base` adapter 接到 32K 基础上。对已有检查点使用产物声明的基础；仅对当前训练契约产生的新运行使用 32K 基础。

## 标签 {#labels}

Level 1 使用 `safe` 和 `unsafe`。Level 2 保留以下九输出兼容契约：

| ID | 含义 |
| --- | --- |
| `S1_violent_crimes` | 暴力犯罪 |
| `S2_nonviolent_crimes` | 非暴力犯罪 |
| `S3_sex_crimes` | 性相关犯罪 |
| `S5_weapons_cbrne` | 武器和 CBRNE |
| `S6_self_harm` | 自残 |
| `S7_hate` | 仇恨 |
| `S8_specialized_advice` | 专业建议 |
| `S9_privacy` | 隐私 |
| `S13_misinformation` | 虚假信息 |

此顺序版本化为 `legacy-9-v1`。将字符串和数字顺序视为 API：更改任一者都需要迁移 Router 策略和已存储评测数据。

## 数据准备 {#data-preparation}

工作流使用来自 AEGIS 2.0 的提示词标签加上合成安全数据集。
响应和拒绝变体被排除。准备会为去重规范化文本，移除空或脱敏记录，让留出划分优先于训练数据，并丢弃标签冲突的重复组。

已核对数据契约创建：

- Level 1：每个二元标签 10000 条训练提示词；
- Level 2：每个危害标签 2000 条训练提示词，仅在某类不足时做确定性过采样。

验证和测试划分保持其自然 AEGIS 分布。对于映射到多个危害的提示词，第一个映射的源类别提供单个训练标签，而所有映射危害仍可用于更严格的评测。

分布式训练前先准备一次数据：

```bash
python -m src.training.model_classifier.safety_classifier.data prepare \
  --contract src/training/model_classifier/safety_classifier/configs/training-v1.json \
  --output-dir /artifacts/data
```

该命令验证固定的输入修订和文件校验和，并将物化划分写入 `/artifacts/data/level1` 和 `/artifacts/data/level2`。

## 训练方法 {#training-method}

两个任务都使用 LoRA rank 32、alpha 64、dropout 0.1、AdamW、线性调度、10% warmup、权重衰减 `0.01`、BF16、seed 42，以及 patience 为 3 的早停。已核对的 8 进程拓扑对全局 batch 64 使用每设备 batch 8，最多训练 10 个 epoch。

```bash
torchrun --standalone --nproc_per_node=8 \
  -m src.training.model_classifier.safety_classifier.train \
  --task level1 \
  --expected-world-size 8 \
  --data-dir /artifacts/data \
  --output-dir /artifacts/runs/level1

torchrun --standalone --nproc_per_node=8 \
  -m src.training.model_classifier.safety_classifier.train \
  --task level2 \
  --expected-world-size 8 \
  --data-dir /artifacts/data \
  --output-dir /artifacts/runs/level2
```

使用 `--max-steps 2` 做短加速器冒烟。覆盖已核对契约的运行仍可用于实验，但应将其解析配置与指标一起记录，而不是将其视为标准发布配方。

## 评测与导出 {#evaluate-and-export}

按 macro F1 选择检查点，并检查每类精确率和召回率。对于 Level 1，应分别报告假阴性和假阳性。对于 Level 2，包含混淆矩阵和严格多危害召回，这样高频类无法掩盖弱危害边界。

```bash
python -m src.training.model_classifier.safety_classifier.evaluate \
  --task level1 \
  --model /artifacts/runs/level1/adapter \
  --artifact-type adapter \
  --data /artifacts/data/level1/test.jsonl \
  --output-dir /artifacts/runs/level1/evaluation

python -m src.training.model_classifier.safety_classifier.export \
  --task level1 \
  --run-root /artifacts/runs/level1 \
  --merged-dir /artifacts/runs/level1-merged
```

对危害模型用 `--task level2` 重复。导出在固定样本上比较 adapter 和合并 logits，在配置容差内检查预测同一性，并写入校验和与标签元数据。

完整 CLI、环境和发布命令见[工作流 README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/safety_classifier)。
