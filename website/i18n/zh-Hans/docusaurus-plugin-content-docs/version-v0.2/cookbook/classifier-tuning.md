---
title: 分类器调优
sidebar_label: 分类器阈值
translation:
  source_commit: "1d1439a"
  source_file: "docs/cookbook/classifier-tuning.md"
  outdated: false
---

# 分类器调优

本指南提供了在 vLLM Semantic Router 中调优分类阈值的快速配置方案。根据您的具体用例调整这些设置，以平衡 precision 和 recall。

## 类别分类器阈值

调整 domain classification 的置信度阈值：

```yaml
classifier:
  category_model:
    model_id: "models/lora_intent_classifier_bert-base-uncased_model"
    threshold: 0.6 # 默认值：0.6
    use_cpu: true
    category_mapping_path: "models/lora_intent_classifier_bert-base-uncased_model/category_mapping.json"
```

> 参见：[config.yaml#classifier.category_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L59-L64)。

| 阈值 | 行为                       |
| --------- | ------------------------------ |
| 0.5 - 0.6 | 更宽松，更高 recall |
| 0.7 - 0.8 | 平衡 precision/recall      |
| 0.9+      | 非常严格，更少匹配     |

## PII 检测阈值

配置 PII 检测器敏感度：

```yaml
classifier:
  pii_model:
    model_id: "models/lora_pii_detector_bert-base-uncased_model"
    threshold: 0.9 # 默认值：0.9（严格）
    use_cpu: true
    pii_mapping_path: "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"
```

> 参见：[config.yaml#classifier.pii_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L65-L70)。

:::tip
在生产环境中使用较高阈值（0.9+）以最小化 PII 误报。
:::

## Jailbreak 检测阈值

调优 Prompt Guard 敏感度：

```yaml
prompt_guard:
  enabled: true
  use_modernbert: true
  model_id: "models/jailbreak_classifier_modernbert-base_model"
  threshold: 0.7 # 默认值：0.7
  use_cpu: true
  jailbreak_mapping_path: "models/jailbreak_classifier_modernbert-base_model/jailbreak_type_mapping.json"
```

> 参见：[config.yaml#prompt_guard](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L35-L41)。

| 阈值 | 权衡                                 |
| --------- | ----------------------------------------- |
| 0.5 - 0.6 | 激进拦截，更多误报 |
| 0.7       | 平衡（推荐）                    |
| 0.8 - 0.9 | 宽松，更少拦截                  |

## 路由置信度阈值

微调智能路径选择：

```yaml
router:
  # 自动 LoRA 选择的高置信度阈值
  high_confidence_threshold: 0.99

  # 路径评估的基准分数
  lora_baseline_score: 0.8
  traditional_baseline_score: 0.7
  embedding_baseline_score: 0.75

  # 成功计算阈值
  success_confidence_threshold: 0.8

  # 默认置信度阈值
  default_confidence_threshold: 0.95
```

> 参见：[config.yaml#router](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L414-L458)。

## 语义缓存相似度阈值

按决策调整缓存匹配严格程度：

```yaml
decisions:
  - name: "health_decision"
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95 # 健康类非常严格

  - name: "general_decision"
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.75 # 通用类较宽松
```

> 参见：[config.yaml#decisions](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L120-L411)。

## BERT 模型阈值

配置 embedding 模型的 semantic matching 阈值：

```yaml
bert_model:
  model_id: models/all-MiniLM-L12-v2
  threshold: 0.6 # 语义相似度阈值
  use_cpu: true
```

> 参见：[config.yaml#bert_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L1-L4)。

## 调优指南

### 何时降低阈值

- 缺少有效分类（低 recall）
- Cache 命中率过低
- 用户报告查询未被正确路由

### 何时提高阈值

- 太多误报匹配
- 触发错误的类别
- PII/jailbreak 误报

### 调试分类

启用详细 log 以诊断阈值问题：

```yaml
observability:
  metrics:
    enabled: true
  tracing:
    enabled: true
    sampling:
      type: "always_on"
```

然后检查 log 中的分类置信度分数：

```
Classified query with confidence 0.72 to category 'math'
```
