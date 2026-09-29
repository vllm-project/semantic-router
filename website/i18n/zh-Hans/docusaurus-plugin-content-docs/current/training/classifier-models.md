---
title: 训练 Vela 分类器
sidebar_label: 分类器
translation:
  source_commit: "96399a94b9030d66f46c5d45f9a838defc091153"
  source_file: "docs/training/classifier-models.md"
  outdated: false
---

# 训练 Vela 分类器 {#train-vela-classifiers}

Vela 分类器将请求转换为路由信号。当应用需要更好地覆盖某种语言、领域或请求模式时，可以进行适配。各任务均使用共享 Vela Encoder 基座。

使用已发布模型，请先参阅 [Vela runtime 配置](../tutorials/global/vela-models.md)。

## 选择任务和标签 {#choose-the-task-and-labels}

| 模型 | 输出 | 示例用途 |
| --- | --- | --- |
| Domain | 14 个主题领域 | 将法律问题路由给专用模型 |
| Guard | `benign`、`jailbreak` | 检测覆盖指令的攻击 |
| Feedback | 四类反馈及 `NO_FEEDBACK` | 处理用户不满意的回答 |
| Modality | `AR`、`DIFFUSION`、`BOTH` | 选择文本、图像或组合输出 |
| FactCheck | `FACT_CHECK_NEEDED`、`NO_FACT_CHECK_NEEDED` | 选择答案核查路径 |
| PII | 17 类实体的 token 标签 | 定位需要脱敏的个人信息 |

内容风险检测见 [Safety 和 Hazard](./mmbert-safety-classifier)。Guard 负责提示词攻击，普通有害内容由 Safety 处理。

### Domain {#domain}

Domain 预测生物、商业、化学、计算机科学、经济、工程、健康、历史、法律、数学、其他、哲学、物理或心理学。加入不属于专业领域的请求，让模型学会使用 `other`。

### Feedback {#feedback}

| 标签 | 含义 |
| --- | --- |
| `SAT` | 用户对答案满意 |
| `NEED_CLARIFICATION` | 用户需要进一步解释 |
| `WRONG_ANSWER` | 用户指出答案错误 |
| `WANT_DIFFERENT` | 用户希望更换格式或方法 |
| `NO_FEEDBACK` | 消息没有表达反馈 |

输入是当前用户的后续消息。普通新问题应标为 `NO_FEEDBACK`；与答案无关的正面陈述不表示满意。如果缺少对话上下文导致无法可靠判断，应单独处理这些含糊回复。

### FactCheck 和 Modality {#factcheck-and-modality}

FactCheck 判断答案是否需要核查，不判断陈述真伪。数据应同时覆盖事实性问题、创作和非事实请求。

Modality 根据文本判断输出意图。`AR` 表示文本，`DIFFUSION` 表示图像，`BOTH` 表示组合响应。它是文本分类器，不检查上传的图片。

## 准备数据 {#prepare-your-data}

共享序列训练器要求 JSONL 行包含 `id`、`text`、`label` 和 `group_id`。相关样本应保留在同一分区；需要细分评测时，同时保留 `source`、`language`、`length_bucket` 和 `position`。独立的 `contract.json` 定义标签顺序。

[序列训练参考](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/sequence_repair)提供文件格式和来源准备流程。[应用配方](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_classifier/vela-applications.md)提供 Feedback 和 Guard 数据构建工具。按任务复核来源标签，特别是攻击引文、正常指令和中性后续消息。

## 训练序列分类器 {#train-a-sequence-classifier}

下载固定 revision 的 Vela Encoder 到 `/models/vela-base`，并将 `VELA_BASE_REVISION` 设置为该 revision。以下例子假设 FactCheck 契约、训练集和开发集已经准备好。

```bash
python -m src.training.model_classifier.sequence_repair.train \
  --method full --fresh-head \
  --base /models/vela-base \
  --base-id llm-semantic-router/Vela-1.0-Encoder-307M \
  --base-revision "${VELA_BASE_REVISION:?Set the downloaded revision}" \
  --contract /data/factcheck/contract.json \
  --train /data/factcheck/train.jsonl --dev /data/factcheck/dev.jsonl \
  --output /data/factcheck/run \
  --steps 600 --batch-size 4 --accumulate 4 \
  --max-length 32768 --microbatch-token-budget 32768 \
  --learning-rate 0.00001 --head-learning-rate 0.0001 \
  --eval-every 200 --evaluation-dtype float32 --selection source-macro-f1
```

`--fresh-head` 初始化分类 head，并与完整编码器一起训练。继续已有任务时，提供兼容的 Vela 任务 checkpoint 并省略该参数。根据数据选择步数和采样；示例参数仅作为起点。

输入预算包含特殊 tokens。超过预算的训练样本会被记录为拒绝，评测时拒绝溢出。增加预算时，应加入真实长样本。

### 继续训练并保留已有能力 {#continue-a-model-while-preserving-existing-behavior}

适配已发布的分类器时，从该任务的 checkpoint 开始，并省略 `--fresh-head`。可选的 `--trainable-last-layers` 只更新最后几层编码器和分类 head。将旧训练样本标记为 `retention_replay: true`，再使用 `--retention-targets`，可以在学习新标签时约束这些旧样本的预测变化。

按照[继续训练流程](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/sequence_repair#preserve-behavior-during-continued-training)生成目标并配置训练。替换已部署的 Guard 模型前，在独立开发集上同时检查攻击召回率和合法请求误报。

## PII 检测器 {#pii-detector}

PII 训练使用实体片段，而不是为整个请求指定一个标签。BIO 编码中，`B-TYPE` 表示实体开始，`I-TYPE` 表示继续，`O` 表示其他 token。

使用 [PII 训练流程](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/pii_model_fine_tuning_lora)中的 `train_repair.py`，对齐字符片段与 tokenizer 输出并训练 token 分类器。评测实体级 precision、recall 和 F1。大多数 token 为 `O`，因此 token accuracy 可能掩盖漏检。

## 评测并部署 {#evaluate-and-deploy}

在同一组独立请求上对比原始和训练后的模型，检查各类别错误、语言、长短输入，以及应不触发信号的请求。使用开发集选择阈值。

通过[序列导出工具](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/sequence_repair#freeze-then-evaluate-the-independent-test)导出选中的模型，包含训练权重、tokenizer 和任务标签映射。PII 使用专用导出流程。

最后配置[本地模型绑定](../installation/runtime/in-process.md)，通过[路由预览](../installation/runtime/lifecycle-diagnostics.md)发送代表性请求，同时检查实际信号、决策和置信度。

[产物索引](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_artifacts.json)保留早期 mmBERT adapter 和 merged 模型的入口。
