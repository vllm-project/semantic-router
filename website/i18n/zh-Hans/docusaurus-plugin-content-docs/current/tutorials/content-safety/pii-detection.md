---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/content-safety/pii-detection.md"
  outdated: false
---

# PII 检测

Semantic Router 内置了个人身份信息（PII）检测功能，用于保护用户查询中的敏感数据。系统使用经过微调的 BERT 模型来识别各种 PII 类型，并根据可配置的策略进行处理。

## 概述

PII 检测系统：

- **识别**用户查询中的常见 PII 类型
- **执行**每个决策的可配置 PII 策略
- **拦截**包含敏感信息的请求（基于信号规则）
- **集成**信号驱动决策，实现精细化控制
- **记录**策略违规，用于监控

## 支持的 PII 类型

系统可检测以下 PII 类型：

| PII 类型 | 说明 | 示例 |
|----------|------|------|
| `PERSON` | 人名 | "张三"、"John Smith" |
| `EMAIL_ADDRESS` | 电子邮件地址 | "user@example.com" |
| `PHONE_NUMBER` | 电话号码 | "+1-555-123-4567" |
| `US_SSN` | 美国社会安全号码 | "123-45-6789" |
| `STREET_ADDRESS` | 实际地址 | "123 Main St, New York, NY" |
| `GPE` | 地缘政治实体 | 国家、州、城市 |
| `ORGANIZATION` | 组织名称 | "Microsoft"、"OpenAI" |
| `CREDIT_CARD` | 信用卡号码 | "4111-1111-1111-1111" |
| `US_DRIVER_LICENSE` | 美国驾照 | "D123456789" |
| `IBAN_CODE` | 国际银行账号 | "GB82 WEST 1234 5698 7654 32" |
| `IP_ADDRESS` | IP 地址 | "192.168.1.1"、"2001:db8::1" |
| `DOMAIN_NAME` | 域名/网站名称 | "example.com"、"google.com" |
| `DATE_TIME` | 日期/时间信息 | "2024-01-15"、"1月15日" |
| `AGE` | 年龄信息 | "25岁"、"1990年出生" |
| `NRP` | 国籍/宗教/政治团体 | "美国人"、"基督徒" |
| `ZIP_CODE` | 邮政编码 | "10001"、"SW1A 1AA" |

## 配置

PII 检测现在是信号层中的**一等公民信号**。在 `signals.pii` 下定义命名的 pii 规则，然后在 `decisions` 中通过 `type: "pii"` 引用它们。

### 基础 PII 检测

```yaml
# router-config.yaml

# ── PII 分类器模型 ────────────────────────────────────────────────────────
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: false
    threshold: 0.9
    use_cpu: true
  pii_mapping_path: "models/mom-pii-classifier/label_mapping.json"

# ── 信号 ──────────────────────────────────────────────────────────────────
signals:
  pii:
    - name: "pii_deny_all"
      threshold: 0.9
      description: "拦截所有 PII 类型"

# ── 决策 ──────────────────────────────────────────────────────────────────
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
          message: "请求被拦截：检测到个人信息，请删除敏感数据后重试。"
```

### 允许列表：允许特定 PII 类型

使用 `pii_types_allowed` 允许某些 PII 类型，同时拦截其他所有类型：

```yaml
signals:
  pii:
    # 拦截所有 PII
    - name: "pii_deny_all"
      threshold: 0.5
      description: "拦截所有 PII"

    # 允许邮件和电话（例如用于预约）
    - name: "pii_allow_email_phone"
      threshold: 0.5
      pii_types_allowed:
        - "EMAIL_ADDRESS"
        - "PHONE_NUMBER"
      description: "允许邮件和电话，拦截身份证号/信用卡等"

    # 允许组织/地点名称（例如用于代码/技术内容）
    - name: "pii_allow_org_location"
      threshold: 0.6
      pii_types_allowed:
        - "GPE"
        - "ORGANIZATION"
        - "DATE_TIME"
      description: "允许地理位置、组织、日期 — 常见于代码和配置文件"
```

### 领域感知的 PII 策略

将 PII 信号与领域信号结合，实现上下文感知的数据保护：

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
      description: "允许邮件和电话用于预约"

  domains:
    - name: "economics"
      description: "金融和经济"
      mmlu_categories: ["economics"]
    - name: "health"
      description: "健康和医疗"
      mmlu_categories: ["health"]

decisions:
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

  # 医疗：允许邮件/电话用于预约，拦截其他 PII
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
          message: "请仅分享您的邮箱或电话号码，请勿包含其他个人信息。"
```

### 基于环境的 PII 策略（开发 vs 生产）

为不同环境应用不同的 PII 阈值和允许列表：

```yaml
signals:
  pii:
    # 开发环境：宽松 — 代码上下文中常有组织/地点引用
    - name: "pii_dev"
      threshold: 0.6
      pii_types_allowed:
        - "GPE"
        - "ORGANIZATION"
        - "DATE_TIME"
      description: "开发环境宽松 PII 策略"

    # 生产环境：严格 — 数据最小化，拦截所有 PII
    - name: "pii_prod"
      threshold: 0.9
      description: "生产环境严格 PII 策略"

decisions:
  - name: "block_pii_prod"
    priority: 999
    rules:
      operator: "OR"
      conditions:
        - type: "pii"
          name: "pii_prod"
    modelRefs:
      - model: "qwen14b-prod"
    plugins:
      - type: "fast_response"
        configuration:
          message: "请求被拦截：检测到个人信息。"
```

## 信号配置参考

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | ✅ | 信号名称，在决策条件中通过 `type: "pii"` 引用 |
| `threshold` | float (0.0–1.0) | ✅ | PII 实体检测的最低置信度分数 |
| `pii_types_allowed` | string[] | ❌ | **允许**（不拦截）的 PII 类型。为空时，所有检测到的 PII 类型都会触发信号 |
| `include_history` | bool | ❌ | 为 `true` 时分析所有对话消息（默认：`false`） |
| `description` | string | ❌ | 该规则的人类可读描述 |

**按使用场景的阈值指南**：

- **关键场景（医疗、金融、法律）**：`0.9–0.95` — 严格检测，误报更少
- **面向客户（支持、销售）**：`0.75–0.85` — 均衡检测
- **内部工具（代码、测试）**：`0.5–0.65` — 宽松，减少技术内容的误报
- **公开内容（文档、营销）**：`0.6–0.75` — 发布前的广泛检测

## PII 分类器模型配置

`classifier.pii_model` 部分配置所有 PII 信号使用的底层 ML 模型：

```yaml
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: false
    threshold: 0.9   # 全局回退阈值（被各信号规则覆盖）
    use_cpu: true
  pii_mapping_path: "models/mom-pii-classifier/label_mapping.json"
```

## PII 检测工作原理

1. **信号评估**：所有 `pii` 信号规则与其他信号（关键词、领域、jailbreak 等）**并行运行** — 对路由管道零额外延迟
2. **实体检测**：PII 分类器识别请求文本中的 PII 实体
3. **允许列表检查**：每个规则检查检测到的 PII 类型是否在其 `pii_types_allowed` 列表中
4. **信号触发**：如果超过阈值的被拒绝 PII 类型被检测到，信号触发
5. **决策匹配**：决策通过 `type: "pii"` 条件引用已触发的信号
6. **执行动作**：匹配的决策执行其插件（例如 `fast_response` 拦截请求）
7. **日志记录**：所有 PII 检测和策略决策均被记录，用于监控

## API 集成

PII 检测自动集成到路由流程中。当请求到达路由器时，系统：

1. 使用配置的分类器分析输入文本中的 PII
2. 并行评估所有 `pii` 信号规则
3. 对超过阈值的被拒绝 PII 类型触发信号
4. 路由到匹配已触发信号的决策

### 分类端点

您也可以直接通过分类 API 检查 PII 检测：

```bash
curl -X POST http://localhost:8080/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "我的邮箱是 john.doe@example.com，我住在纽约"
  }'
```

响应中包含 PII 信息和分类结果。

## 监控与指标

系统暴露 PII 相关指标：

```
# Prometheus 指标
pii_detections_total{type="EMAIL_ADDRESS"} 45
pii_detections_total{type="PERSON"} 23
pii_policy_violations_total 12
pii_requests_blocked_total 8
```

## 最佳实践

### 1. 使用优先级排序安全决策

将 PII 拦截决策设置为高优先级（例如 `999`），确保在路由决策之前评估：

```yaml
decisions:
  - name: "block_pii"
    priority: 999   # 在路由决策（优先级 < 999）之前评估
    rules:
      ...
```

### 2. 与 Jailbreak 检测结合使用

同时使用 `pii` 和 `jailbreak` 信号，实现全面的安全防护：

```yaml
signals:
  jailbreak:
    - name: "jailbreak_standard"
      threshold: 0.65
  pii:
    - name: "pii_deny_all"
      threshold: 0.5

decisions:
  - name: "block_jailbreak"
    priority: 1000
    rules:
      operator: "OR"
      conditions:
        - type: "jailbreak"
          name: "jailbreak_standard"
    plugins:
      - type: "fast_response"
        configuration:
          message: "请求被拦截：违反政策。"

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

### 3. 使用领域上下文实现更智能的策略

不同领域有不同的 PII 敏感度要求。结合 `pii` + `domain` 信号实现上下文感知策略，而非应用单一全局规则。

### 4. 为对话应用启用历史记录

对于多轮对话，启用 `include_history: true` 以检测跨多条消息分享的 PII：

```yaml
signals:
  pii:
    - name: "pii_full_history"
      threshold: 0.9
      include_history: true
      description: "检测完整对话历史中的 PII"
```

## 故障排查

### 误报率高

- 提高信号 `threshold`（例如从 `0.5` 提高到 `0.8`）
- 将常见误报类型添加到 `pii_types_allowed`
- 对于代码/技术内容，允许 `GPE`、`ORGANIZATION`、`DATE_TIME`

### 漏检 PII

- 降低信号 `threshold`
- 检查 PII 类型是否在[支持的类型表](#支持的-pii-类型)中
- 验证 PII 模型是否正确加载

### 调试模式

启用详细的 PII 日志：

```yaml
logging:
  level: debug
  pii_detection: true
```

这将记录所有 PII 检测决策和信号评估过程。
