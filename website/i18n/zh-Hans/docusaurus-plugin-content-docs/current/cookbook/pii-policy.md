---
title: PII 策略配置
sidebar_label: PII 策略
translation:
  source_commit: "1d1439a"
  source_file: "docs/cookbook/pii-policy.md"
  outdated: false
---

# PII 策略配置

本指南提供了 PII（个人身份信息）检测和策略执行的快速配置方案。根据您的合规要求使用这些模式来保护敏感数据。

## 按 Decision 启用 PII 检测

将 PII plugin 添加到特定 decision rule：

```yaml
decisions:
  - name: "health_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "health"
    modelRefs:
      - model: "qwen3"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed: [] # 阻止所有 PII
```

> 参见：[config.yaml#pii plugin](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L136-L139)。

## 允许特定 PII 类型

允许某些 PII 类型同时阻止其他类型：

```yaml
plugins:
  - type: "pii"
    configuration:
      enabled: true
      pii_types_allowed:
        - "LOCATION" # 允许位置提及
        - "DATE_TIME" # 允许日期和时间
        - "ORGANIZATION" # 允许公司名称
      # 所有其他类型（PERSON、EMAIL、PHONE 等）将被阻止
```

> 参见：[config.yaml#pii plugin](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L136-L139) 和 [config.go pii_types_allowed](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/config.go#L742)。

## 支持的 PII 类型

| PII 类型       | 描述             | 示例               |
| -------------- | ----------------------- | --------------------- |
| `PERSON`       | 人名         | "张三"          |
| `EMAIL`        | 电子邮件地址         | "user@example.com"    |
| `PHONE`        | 电话号码           | "+86-138-0000-0000"   |
| `LOCATION`     | 地理位置    | "北京"            |
| `DATE_TIME`    | 日期和时间         | "2024年1月15日"    |
| `ORGANIZATION` | 公司/组织名称       | "某某公司"           |
| `CREDIT_CARD`  | 信用卡号     | "4111-1111-1111-1111" |
| `SSN`          | 社会保障号码 | "123-45-6789"         |
| `IP_ADDRESS`   | IP 地址            | "192.168.1.1"         |

## 严格 PII 策略（阻止所有）

最大程度保护隐私：

```yaml
plugins:
  - type: "pii"
    configuration:
      enabled: true
      pii_types_allowed: [] # 空列表 = 阻止所有 PII
```

> 参见：[config.yaml#pii plugin](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L136-L139)。

## 宽松 PII 策略（仅 Warning）

Log PII 但不阻止：

```yaml
classifier:
  pii_model:
    threshold: 0.95 # 非常高的阈值
    # ...

decisions:
  - name: "internal_decision"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed:
            - "PERSON"
            - "EMAIL"
            - "PHONE"
            - "LOCATION"
            - "DATE_TIME"
            - "ORGANIZATION"
```

> 参见：[config.yaml#classifier.pii_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L65-L70) 和 [config.yaml#pii plugin](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L136-L139)。

## PII 模型配置

配置底层 PII 检测模型：

```yaml
classifier:
  pii_model:
    model_id: "models/lora_pii_detector_bert-base-uncased_model"
    use_modernbert: false
    threshold: 0.9 # 高阈值以减少误报
    use_cpu: true
    pii_mapping_path: "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"
```

> 参见：[config.yaml#classifier.pii_model](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L65-L70) 和 [pkg/utils/pii](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/utils/pii)。

## Domain 特定 PII 策略

不同 domain 可能需要不同的 PII 处理方式：

```yaml
decisions:
  # 健康类：非常严格的 PII 处理
  - name: "health_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "health"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed: [] # 不允许任何 PII

  # 商业类：允许组织名称
  - name: "business_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "business"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed:
            - "ORGANIZATION"
            - "LOCATION"

  # 通用类：更宽松
  - name: "general_decision"
    plugins:
      - type: "pii"
        configuration:
          enabled: true
          pii_types_allowed:
            - "LOCATION"
            - "DATE_TIME"
            - "ORGANIZATION"
```

## 调试 PII 检测

当 PII 被错误阻止时，检查 log：

```
PII policy violation for decision health_decision: denied PII types [PERSON, EMAIL]
```

修复方法：

1. 如果应该允许该 PII 类型，将其添加到 `pii_types_allowed`
2. 如果发生误报，提高 `classifier.pii_model.threshold`

> 参见代码：[pii/policy.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/utils/pii/policy.go)。
