---
title: PII 策略配置
sidebar_label: PII 策略
---

# PII 策略配置

本指南提供 PII（个人身份信息）检测和策略执行的快速配置示例。PII 检测是**一等公民信号** — 在 `signals.pii` 下定义命名的 pii 规则，然后在 `decisions` 中通过 `type: "pii"` 引用它们。

## 为决策启用 PII 检测

定义 PII 信号并在决策规则中引用它：

```yaml
signals:
  pii:
    - name: "pii_deny_all"
      threshold: 0.9
      description: "拦截所有 PII 类型"

decisions:
  - name: "block_pii"
    priority: 999
    rules:
      operator: "OR"
      conditions:
        - type: "pii"
          name: "pii_deny_all"
    plugins:
      - type: "fast_response"
        configuration:
          message: "请求被拦截：检测到个人信息。"
```

> 参见：[config/prompt-guard/hybrid.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/prompt-guard/hybrid.yaml)。

## 允许特定 PII 类型

使用 `pii_types_allowed` 允许某些 PII 类型，同时拦截其他所有类型：

```yaml
signals:
  pii:
    - name: "pii_allow_location_org"
      threshold: 0.5
      pii_types_allowed:
        - "GPE"          # 允许地理位置
        - "DATE_TIME"    # 允许日期和时间
        - "ORGANIZATION" # 允许公司名称
      description: "允许地点、日期、组织名称 — 拦截其他所有 PII"
```

信号仅在检测到**不在** `pii_types_allowed` 中的 PII 类型且超过阈值时触发。

> 参见：[pkg/config/config.go — PIIRule](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/config.go)。

## 支持的 PII 类型

| PII 类型       | 说明             | 示例               |
| -------------- | ---------------- | ------------------ |
| `PERSON`       | 人名             | "张三"             |
| `EMAIL_ADDRESS`| 电子邮件地址     | "user@example.com" |
| `PHONE_NUMBER` | 电话号码         | "+1-555-0123"      |
| `GPE`          | 地理位置         | "纽约"             |
| `DATE_TIME`    | 日期和时间       | "2024年1月15日"    |
| `ORGANIZATION` | 公司/组织名称    | "Acme Corp"        |
| `CREDIT_CARD`  | 信用卡号码       | "4111-1111-1111-1111" |
| `US_SSN`       | 社会安全号码     | "123-45-6789"      |
| `IP_ADDRESS`   | IP 地址          | "192.168.1.1"      |
| `STREET_ADDRESS`| 实际地址        | "123 Main St, NY"  |
| `US_DRIVER_LICENSE` | 美国驾照    | "D123456789"       |
| `IBAN_CODE`    | 银行账号         | "GB82 WEST 1234..."  |
| `DOMAIN_NAME`  | 域名/网站名称    | "example.com"      |
| `ZIP_CODE`     | 邮政编码         | "10001"            |

## 严格 PII 策略（拦截所有）

最大隐私保护 — 拦截每种检测到的 PII 类型：

```yaml
signals:
  pii:
    - name: "pii_deny_all"
      threshold: 0.5
      # pii_types_allowed 省略 → 所有检测到的 PII 类型都触发信号
      description: "拦截所有 PII"

decisions:
  - name: "block_pii"
    priority: 999
    rules:
      operator: "OR"
      conditions:
        - type: "pii"
          name: "pii_deny_all"
    plugins:
      - type: "fast_response"
        configuration:
          message: "请求被拦截：检测到个人信息。"
```

## 宽松 PII 策略（仅拦截敏感类型）

允许大多数常见 PII 类型，仅拦截高度敏感的类型：

```yaml
signals:
  pii:
    - name: "pii_block_sensitive_only"
      threshold: 0.95   # 非常高的阈值
      pii_types_allowed:
        - "PERSON"
        - "EMAIL_ADDRESS"
        - "PHONE_NUMBER"
        - "GPE"
        - "DATE_TIME"
        - "ORGANIZATION"
      description: "仅拦截高度敏感的 PII（身份证号、信用卡等）"
```

## PII 模型配置

配置底层 PII 检测模型：

```yaml
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: false
    threshold: 0.9   # 全局回退阈值（被各信号规则覆盖）
    use_cpu: true
  pii_mapping_path: "models/mom-pii-classifier/label_mapping.json"
```

> 参见：[pkg/utils/pii](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/utils/pii)。

## 领域特定的 PII 策略

不同领域可能需要不同的 PII 处理方式 — 结合 `pii` + `domain` 信号：

```yaml
signals:
  pii:
    - name: "pii_deny_all"
      threshold: 0.5
      description: "拦截所有 PII"
    - name: "pii_allow_email_phone"
      threshold: 0.5
      pii_types_allowed:
        - "EMAIL_ADDRESS"
        - "PHONE_NUMBER"
      description: "允许邮件/电话用于预约"
    - name: "pii_allow_org_location"
      threshold: 0.6
      pii_types_allowed:
        - "GPE"
        - "ORGANIZATION"
        - "DATE_TIME"
      description: "允许代码中常见的组织/地点名称"

  domains:
    - name: "health"
      mmlu_categories: ["health"]
    - name: "economics"
      mmlu_categories: ["economics"]
    - name: "computer_science"
      mmlu_categories: ["computer_science"]

decisions:
  # 医疗：拦截除邮件/电话外的所有 PII（允许预约）
  - name: "block_pii_health"
    priority: 998
    rules:
      operator: "AND"
      conditions:
        - type: "pii"
          name: "pii_allow_email_phone"
        - type: "domain"
          name: "health"
    plugins:
      - type: "fast_response"
        configuration:
          message: "请仅分享您的邮箱或电话号码。"

  # 金融：拦截所有 PII
  - name: "block_pii_finance"
    priority: 999
    rules:
      operator: "AND"
      conditions:
        - type: "pii"
          name: "pii_deny_all"
        - type: "domain"
          name: "economics"
    plugins:
      - type: "fast_response"
        configuration:
          message: "为保护您的安全，请勿在金融查询中分享个人信息。"

  # 代码：允许代码中常见的组织/地点名称
  - name: "block_pii_code"
    priority: 997
    rules:
      operator: "AND"
      conditions:
        - type: "pii"
          name: "pii_allow_org_location"
        - type: "domain"
          name: "computer_science"
    plugins:
      - type: "fast_response"
        configuration:
          message: "请求被拦截：代码中检测到敏感个人信息。"
```

## 调试 PII 检测

当 PII 被错误拦截时，检查日志：

```
PII signal fired: rule=pii_deny_all, detected_types=[PERSON, EMAIL_ADDRESS], threshold=0.5
```

修复方法：

1. 如果该 PII 类型应被允许，将其添加到信号规则的 `pii_types_allowed` 中
2. 如果存在误报，提高信号 `threshold`
3. 启用调试日志以获取详细的信号评估信息：

```yaml
logging:
  level: debug
  pii_detection: true
```

> 参见代码：[pii/policy.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/utils/pii/policy.go)。
