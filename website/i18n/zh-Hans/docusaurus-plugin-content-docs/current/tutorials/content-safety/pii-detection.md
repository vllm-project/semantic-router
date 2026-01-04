---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/content-safety/pii-detection.md"
  outdated: false
---

# PII 检测 (PII Detection)

Semantic Router 提供了内置的个人身份信息 (PII) 检测功能，以保护用户查询中的敏感数据。系统使用经过微调的 BERT 模型，根据可配置的策略识别并处理各种类型的 PII。

## 概览

PII 检测系统：

- **识别** 用户查询中的常见 PII 类型
- **执行** 特定于模型的 PII 策略
- **拦截或掩码** 根据配置处理敏感信息
- **过滤** 基于 PII 合规性的候选模型
- **记录** 策略违规以供监控

## 支持的 PII 类型

系统可以检测以下 PII 类型：

| PII 类型 | 描述 | 示例 |
|----------|-------------|----------|
| `PERSON` | 人名 | "John Smith", "Mary Johnson" |
| `EMAIL_ADDRESS` | 电子邮件地址 | "user@example.com" |
| `PHONE_NUMBER` | 电话号码 | "+1-555-123-4567", "(555) 123-4567" |
| `US_SSN` | 美国社会安全号码 | "123-45-6789" |
| `STREET_ADDRESS` | 物理地址 | "123 Main St, New York, NY" |
| `GPE` | 地缘政治实体 | 国家、州、城市 |
| `ORGANIZATION` | 组织名称 | "Microsoft", "OpenAI" |
| `CREDIT_CARD` | 信用卡号 | "4111-1111-1111-1111" |
| `US_DRIVER_LICENSE` | 美国驾驶执照 | "D123456789" |
| `IBAN_CODE` | 国际银行账号 | "GB82 WEST 1234 5698 7654 32" |
| `IP_ADDRESS` | IP 地址 | "192.168.1.1", "2001:db8::1" |
| `DOMAIN_NAME` | 域名/网站名称 | "example.com", "google.com" |
| `DATE_TIME` | 日期/时间信息 | "2024-01-15", "January 15th" |
| `AGE` | 年龄信息 | "25 years old", "born in 1990" |
| `NRP` | 国籍/宗教/政治团体 | "American", "Christian", "Democrat" |
| `ZIP_CODE` | 邮政编码 | "10001", "SW1A 1AA" |

## 配置

### 基础 PII 检测

在您的配置中启用 PII 检测：

```yaml
# router-defaults.yaml
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: false
    threshold: 0.9                 # 全局检测阈值 (0.0-1.0)
    use_cpu: true
  pii_mapping_path: "models/mom-pii-classifier/label_mapping.json"
```

### 类别级 PII 检测

**v0.x 新功能**：在类别级别配置 PII 检测阈值，以便根据类别的具体要求和后果进行精细控制。

```yaml
# 全局 PII 配置 - 默认应用于所有类别
classifier:
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: false
    threshold: 0.9  # 全局默认阈值
    use_cpu: true
  pii_mapping_path: "models/mom-pii-classifier/label_mapping.json"

# 类别特定的 PII 设置
categories:
  # 医疗保健类别：针对关键 PII 的高阈值
  - name: healthcare
    description: "医疗保健和医学查询"
    pii_enabled: true       # 启用 PII 检测（默认：继承自全局）
    pii_threshold: 0.9      # 更高阈值，检测更严格
    model_scores:
      - model: secure-llm
        score: 0.9
        use_reasoning: false

  # 金融类别：针对金融 PII 的极高阈值
  - name: finance
    description: "金融查询"
    pii_enabled: true
    pii_threshold: 0.95     # 对 SSN、信用卡等非常严格
    model_scores:
      - model: secure-llm
        score: 0.9
        use_reasoning: false

  # 代码生成：较低阈值以减少误报
  - name: code_generation
    description: "代码和技术内容"
    pii_enabled: true
    pii_threshold: 0.5      # 较低阈值，避免将代码工件标记为 PII
    model_scores:
      - model: general-llm
        score: 0.9
        use_reasoning: true

  # 测试：禁用 PII 检测
  - name: testing
    description: "测试场景"
    pii_enabled: false      # 测试时禁用
    model_scores:
      - model: general-llm
        score: 0.6
        use_reasoning: false

  # 通用：使用全局设置
  - name: general
    description: "通用查询"
    # 未指定 pii_enabled 和 pii_threshold - 继承全局设置
    model_scores:
      - model: general-llm
        score: 0.5
        use_reasoning: false
```

**配置继承：**

- `pii_enabled`：如果未指定，继承自全局 PII 模型配置（如果配置了 `pii_model` 则启用）
- `pii_threshold`：如果未指定，继承自 `classifier.pii_model.threshold`

**各类别阈值指南：**

- **关键类别**（医疗、金融、法律）：0.9-0.95 - 严格检测，误报较少
- **面向客户**（支持、销售）：0.75-0.85 - 平衡检测
- **内部工具**（代码、测试）：0.5-0.65 - 宽松，以减少误报
- **公共内容**（文档、营销）：0.6-0.75 - 发布前进行更广泛的检测

### 模型特定的 PII 策略

为不同模型配置不同的 PII 策略：

```yaml
# vLLM 端点配置
vllm_endpoints:
  - name: secure-model
    address: "127.0.0.1"
    port: 8080
  - name: general-model
    address: "127.0.0.1"
    port: 8081

# 模型特定的配置
model_config:
  secure-llm:
    pii_policy:
      allow_by_default: false      # 默认拦截所有 PII
      pii_types:                   # 仅允许这些特定类型
        - "EMAIL_ADDRESS"
        - "GPE"
        - "ORGANIZATION"

  general-llm:
    pii_policy:
      allow_by_default: true       # 默认允许所有 PII
      pii_types: []                # 当 allow_by_default 为 true 时不使用
```

## PII 检测如何工作

PII 检测系统的工作流程如下：

1. **检测**：PII 分类器模型分析输入文本以识别 PII 类型
2. **策略检查**：系统检查目标模型是否允许检测到的 PII 类型
3. **路由决策**：过滤掉不允许检测到的 PII 类型的模型
4. **日志记录**：记录所有 PII 检测和策略决策以进行监控

## API 集成

PII 检测已自动集成到路由过程中。当向路由发送请求时，系统会：

1. 使用配置的分类器分析输入文本是否存在 PII
2. 检查候选模型的 PII 策略
3. 过滤掉不允许检测到的 PII 类型的模型
4. 路由到可以处理该 PII 的合适模型

**注意**：当前的实现在自动路由期间使用全局 PII 阈值。要使用类别特定的阈值，您可以：

- 在配置中为每个类别适当配置阈值
- 在代码中使用 `config.GetPIIThresholdForCategory(categoryName)` 获取类别特定的阈值
- 在具有类别上下文时，使用类别特定的阈值调用 `classifier.ClassifyPIIWithThreshold(text, threshold)`

### 分类端点

您还可以直接使用分类 API 检查 PII 检测：

```bash
curl -X POST http://localhost:8080/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "我的电子邮件是 john.doe@example.com，我住在纽约"
  }'
```

响应包含 PII 信息以及类别分类结果。

## 监控与指标

系统公开了与 PII 相关的指标：

```
# Prometheus 指标
pii_detections_total{type="EMAIL_ADDRESS"} 45
pii_detections_total{type="PERSON"} 23
pii_policy_violations_total{model="secure-model"} 12
pii_requests_blocked_total 8
pii_requests_masked_total 15
```

## 最佳实践

### 1. 阈值调整

- 从 `threshold: 0.7` 开始，以实现平衡的准确性
- 对于高安全性环境，增加到 `0.8-0.9`
- 降低到 `0.5-0.6` 以进行更广泛的检测
- **使用类别级阈值**，根据 PII 类型后果进行精细控制

#### 类别特定的阈值指南

不同类别具有不同的 PII 灵敏度要求：

**关键类别（医疗、金融、法律）：**

- 阈值：`0.9-0.95`
- 理由：需要高精度；医学/金融术语的误报代价高昂
- 示例 PII：SSN、信用卡、病历
- 阈值过低的风险：过多的误报会中断工作流

**面向客户的类别（支持、销售）：**

- 阈值：`0.75-0.85`
- 理由：在捕获 PII 和避免误报之间取得平衡
- 示例 PII：电子邮件、电话、姓名、地址
- 阈值过低的风险：中等的误报率

**内部工具（代码生成、开发）：**

- 阈值：`0.5-0.65`
- 理由：代码/技术内容经常触发误报；需要较低的阈值
- 示例 PII：变量名、看起来像 PII 的测试数据
- 阈值过高的风险：仍可能标记无害的代码工件

**公共内容（文档、营销）：**

- 阈值：`0.6-0.75`
- 理由：发布前进行更广泛的检测；可以接受审查更多误报
- 示例 PII：作者姓名、示例邮件、占位数据
- 阈值过高的风险：可能漏掉可能被发布的 PII

### 2. 策略设计

- 为敏感模型使用 `allow_by_default: false`
- 明确列出允许的 PII 类型，以确保清晰
- 为不同的用例考虑不同的策略
- **将类别级阈值与模型级策略结合使用**，实现纵深防御

### 3. 操作选择

- 在高安全性场景中使用 `block` (拦截)
- 当仍需处理时使用 `mask` (掩码)
- 为满足审计要求，使用带有日志记录的 `allow` (允许)

### 4. 模型过滤

- 配置 PII 策略以自动过滤候选模型
- 确保至少有一个模型可以处理每种 PII 场景
- 彻底测试策略组合

## 故障排除

### 常见问题

**误报过高**

- 降低检测阈值
- 针对边缘情况审查训练数据
- 考虑自定义模型微调

**漏掉 PII 检测**

- 增加检测灵敏度
- 检查 PII 类型是否受支持
- 验证模型是否已正确加载

**策略冲突**

- 确保至少有一个模型允许检测到的 PII 类型
- 检查 `allow_by_default` 设置
- 审查 `pii_types_allowed` 列表

### 调试模式

启用详细的 PII 日志记录：

```yaml
logging:
  level: debug
  pii_detection: true
```

这将记录所有 PII 检测决策和策略评估。
