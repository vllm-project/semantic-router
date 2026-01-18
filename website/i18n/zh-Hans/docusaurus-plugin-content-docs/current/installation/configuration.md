---
translation:
  source_commit: "bac2743"
  source_file: "docs/installation/configuration.md"
  outdated: true
sidebar_position: 4
---

# 配置

本指南涵盖了 Semantic Router 的配置选项。系统使用单个 YAML 配置文件来控制 **Signal-Driven Routing**、**Plugin Chain 处理**和**模型选择**。

## 架构概览

配置定义了三个主要层：

1. **Signal Extraction Layer（信号提取层）**：定义 7 种类型的信号（keyword、embedding、domain、fact_check、user_feedback、preference、language）
2. **Decision Engine（决策引擎）**：使用 AND/OR 运算符组合信号以做出路由决策
3. **Plugin Chain（插件链）**：配置用于缓存、安全和优化的插件

## 配置文件

配置文件位于 `config/config.yaml`。以下是基于实际实现的结构：

```yaml
# config/config.yaml - 实际配置结构

# 用于语义相似度的 BERT 模型
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

# 语义缓存
semantic_cache:
  backend_type: "memory"  # 选项: "memory" 或 "milvus"
  enabled: false
  similarity_threshold: 0.8  # 全局默认阈值
  max_entries: 1000
  ttl_seconds: 3600
  eviction_policy: "fifo"  # 选项: "fifo", "lru", "lfu"

# 工具自动选择
tools:
  enabled: false
  top_k: 3
  similarity_threshold: 0.2
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true

# Jailbreak 防护
prompt_guard:
  enabled: false  # 全局默认 - 可以针对每个类别覆盖
  use_modernbert: true
  model_id: "models/jailbreak_classifier_modernbert-base_model"
  threshold: 0.7
  use_cpu: true

# vLLM 端点 - 您的后端模型
vllm_endpoints:
  - name: "endpoint1"
    address: "192.168.1.100"  # 替换为您的服务器 IP 地址
    port: 11434
    models:
      - "your-model"           # 替换为您的模型
    weight: 1

# 模型配置
model_config:
  "your-model":
    pii_policy:
      allow_by_default: true
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]
    preferred_endpoints: ["endpoint1"]
  # 示例：具有自定义名称的 DeepSeek 模型
  "ds-v31-custom":
    reasoning_family: "deepseek"  # 使用 DeepSeek 推理语法
    preferred_endpoints: ["endpoint1"]
  # 示例：具有自定义名称的 Qwen3 模型
  "my-qwen3-model":
    reasoning_family: "qwen3"     # 使用 Qwen3 推理语法
    preferred_endpoints: ["endpoint2"]
  # 示例：不支持推理的模型
  "phi4":
    preferred_endpoints: ["endpoint1"]

# 分类模型
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7
    use_cpu: true

# 信号 - 信号提取配置
signals:
  # 基于关键词的信号（快速模式匹配）
  keywords:
    - name: "math_keywords"
      operator: "OR"
      keywords:
        - "calculate"
        - "equation"
        - "solve"
        - "derivative"
        - "integral"
      case_sensitive: false

    - name: "code_keywords"
      operator: "OR"
      keywords:
        - "function"
        - "class"
        - "debug"
        - "compile"
      case_sensitive: false

  # 基于嵌入的信号（语义相似度）
  embeddings:
    - name: "code_debug"
      threshold: 0.70
      candidates:
        - "how to debug the code"
        - "troubleshooting steps for my code"
      aggregation_method: "max"

    - name: "math_intent"
      threshold: 0.75
      candidates:
        - "solve mathematical problem"
        - "calculate the result"
      aggregation_method: "max"

  # 领域信号（MMLU 分类）
  domains:
    - name: "mathematics"
      description: "Mathematical and computational problems"
      mmlu_categories:
        - "abstract_algebra"
        - "college_mathematics"
        - "elementary_mathematics"

    - name: "computer_science"
      description: "Programming and computer science"
      mmlu_categories:
        - "computer_security"
        - "machine_learning"

  # 事实核查信号（检测验证需求）
  fact_check:
    - name: "needs_verification"
      description: "Queries requiring fact verification"

  # 用户反馈信号（满意度分析）
  user_feedbacks:
    - name: "correction_needed"
      description: "User indicates previous answer was wrong"

  # 偏好信号（基于 LLM 的匹配）
  preferences:
    - name: "complex_reasoning"
      description: "Requires deep reasoning and analysis"
      llm_endpoint: "http://localhost:11434"

# 类别 - 定义领域类别
categories:
- name: math
- name: computer science
- name: other

# 决策 - 结合信号以做出路由决策
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  rules:
    operator: "OR"  # 匹配任何条件
    conditions:
      - type: "keyword"
        name: "math_keywords"
      - type: "embedding"
        name: "math_intent"
      - type: "domain"
        name: "mathematics"
  modelRefs:
    - model: your-model
      use_reasoning: true  # 为数学问题启用推理
  # 可选：决策级插件
  plugins:
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.9  # 数学问题需要更高的阈值
    - type: "jailbreak"
      configuration:
        enabled: true
    - type: "pii"
      configuration:
        enabled: true
        threshold: 0.8
    - type: "system_prompt"
      configuration:
        enabled: true
        prompt: "You are a mathematics expert. Solve problems step by step."

- name: computer_science
  description: "Route computer science queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "keyword"
        name: "code_keywords"
      - type: "embedding"
        name: "code_debug"
      - type: "domain"
        name: "computer_science"
  modelRefs:
    - model: your-model
      use_reasoning: true  # 为代码启用推理
  plugins:
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.85
    - type: "system_prompt"
      configuration:
        enabled: true
        prompt: "You are a programming expert. Provide clear code examples."

- name: other
  description: "Route general queries"
  priority: 5
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "other"
  modelRefs:
    - model: your-model
      use_reasoning: false # 通用查询不使用推理
  plugins:
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.75  # 通用查询使用较低的阈值

default_model: your-model

# 推理家族配置 - 定义不同模型家族如何处理推理语法
reasoning_families:
  deepseek:
    type: "chat_template_kwargs"
    parameter: "thinking"
  
  qwen3:
    type: "chat_template_kwargs"
    parameter: "enable_thinking"
  
  gpt-oss:
    type: "reasoning_effort"
    parameter: "reasoning_effort"
  
  gpt:
    type: "reasoning_effort"
    parameter: "reasoning_effort"

# 全局默认推理努力等级
default_reasoning_effort: "medium"

```

在上面的 `model_config` 块中分配推理家族——每个模型使用 `reasoning_family`（参见示例中的 `ds-v31-custom` 和 `my-qwen3-model`）。不支持推理语法的模型只需省略该字段（例如 `phi4`）。

## 配置方案 (预设)

我们提供精心挑选的、版本化的预设，您可以直接使用或作为起点：

- 精度优化：https://github.com/vllm-project/semantic-router/blob/main/config/config.recipe-accuracy.yaml
- Token 效率优化：https://github.com/vllm-project/semantic-router/blob/main/config/config.recipe-token-efficiency.yaml
- 延迟优化：https://github.com/vllm-project/semantic-router/blob/main/config/config.recipe-latency.yaml
- 指南和用法：https://github.com/vllm-project/semantic-router/blob/main/config/RECIPES.md

快速使用：

- 本地：将方案复制到 config.yaml，然后运行
  - cp config/config.recipe-accuracy.yaml config/config.yaml
  - make run-router
- Helm/Argo：在您的 ConfigMap 中引用方案文件内容（示例在上述指南中）。

## 信号配置

信号是智能路由的基础。系统支持 7 种类型的信号，可以组合起来做出路由决策。

### 1. 关键词信号 - 快速模式匹配

```yaml
signals:
  keywords:
    - name: "math_keywords"
      operator: "OR"  # OR: 匹配任意关键词, AND: 匹配所有关键词
      keywords:
        - "calculate"
        - "equation"
        - "solve"
      case_sensitive: false
```

**用例：**

- 针对特定术语的确定性路由
- 合规性和安全性（PII 关键词、违禁术语）
- 需要 &lt;1ms 延迟的高吞吐量场景

### 2. 嵌入信号 - 语义理解

```yaml
signals:
  embeddings:
    - name: "code_debug"
      threshold: 0.70  # 相似度阈值 (0-1)
      candidates:
        - "how to debug the code"
        - "troubleshooting steps"
      aggregation_method: "max"  # max, avg, 或 min
```

**用例：**

- 对释义具有鲁棒性的意图检测
- 语义相似度匹配
- 处理多样化的用户措辞

### 3. 领域信号 - MMLU 分类

```yaml
signals:
  domains:
    - name: "mathematics"
      description: "Mathematical problems"
      mmlu_categories:
        - "abstract_algebra"
        - "college_mathematics"
```

**用例：**

- 学术和专业领域路由
- 领域专家模型选择
- 支持 14 个 MMLU 类别

### 4. 事实核查信号 - 验证需求检测

```yaml
signals:
  fact_check:
    - name: "needs_verification"
      description: "Queries requiring fact verification"
```

**用例：**

- 识别事实查询与创意/代码任务
- 路由到具有幻觉检测的模型
- 触发事实核查插件

### 5. 用户反馈信号 - 满意度分析

```yaml
signals:
  user_feedbacks:
    - name: "correction_needed"
      description: "User indicates previous answer was wrong"
```

**用例：**

- 处理后续更正（"that's wrong", "try again"）
- 检测满意度水平
- 路由到更强大的模型进行重试

### 6. 偏好信号 - 基于 LLM 的匹配

```yaml
signals:
  preferences:
    - name: "complex_reasoning"
      description: "Requires deep reasoning"
      llm_endpoint: "http://localhost:11434"
```

**用例：**

- 通过外部 LLM 进行复杂意图分析
- 细致的路由决策
- 当其他信号不足时

### 7. 语言信号 - 多语言检测

```yaml
signals:
  language:
    - name: "en"
      description: "English language queries"
    - name: "es"
      description: "Spanish language queries"
    - name: "zh"
      description: "Chinese language queries"
    - name: "ru"
      description: "Russian language queries"
    - name: "fr"
      description: "French language queries"
```

**用例：**

- 将查询路由到特定语言的模型
- 应用特定语言的策略
- 支持多语言应用
- 通过 whatlanggo 库支持 100 多种本地化语言

## 决策规则 - 信号融合

使用 AND/OR 运算符组合信号：

```yaml
decisions:
  - name: math
    description: "Route mathematical queries"
    priority: 10
    rules:
      operator: "OR"  # 匹配任意条件
      conditions:
        - type: "keyword"
          name: "math_keywords"
        - type: "embedding"
          name: "math_intent"
        - type: "domain"
          name: "mathematics"
    modelRefs:
      - model: math-specialist
        weight: 1.0
```

**策略：**

- **基于优先级**：首先评估优先级较高的决策
- **基于置信度**：选择置信度得分最高的决策
- **混合**：结合优先级和置信度

## 插件链配置

插件在链中处理请求/响应。每个决策都可以覆盖全局插件设置。

### 全局插件配置

```yaml
# 全局默认值
semantic_cache:
  enabled: true
  similarity_threshold: 0.8

prompt_guard:
  enabled: true
  threshold: 0.7

classifier:
  pii_model:
    enabled: true
    threshold: 0.8
```

### 决策级插件覆盖

```yaml
decisions:
  - name: math
    description: "Route mathematical queries"
    priority: 10
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.9  # 数学问题更高
      - type: "jailbreak"
        configuration:
          enabled: true
      - type: "pii"
        configuration:
          enabled: true
          threshold: 0.8
      - type: "system_prompt"
        configuration:
          enabled: true
          prompt: "You are a mathematics expert."
      - type: "header_mutation"
        configuration:
          enabled: true
          headers:
            X-Math-Mode: "enabled"
      - type: "hallucination"
        configuration:
          enabled: false  # 可选的实时检测
```

### 插件类型

| 插件 | 描述 | 配置 |
|--------|-------------|---------------|
| **semantic-cache** | 基于语义相似度的缓存 | `similarity_threshold`, `ttl_seconds` |
| **jailbreak** | 对抗性提示词检测 | `threshold`, `model_id` |
| **pii** | PII 检测和脱敏 | `threshold`, `pii_types_allowed` |
| **system_prompt** | 动态提示词注入 | `prompt` |
| **header_mutation** | HTTP Header 操控 | `headers` |
| **hallucination** | Token 级幻觉检测 | `enabled` |

## 关键配置部分

### 后端端点

配置您的 LLM 服务器：

```yaml
vllm_endpoints:
  - name: "my_endpoint"
    address: "127.0.0.1"  # 您的服务器 IP - 必须是 IP 地址格式
    port: 8000            # 您的服务器端口
    weight: 1             # 负载均衡权重

# 模型配置 - 将模型映射到端点
model_config:
  "llama2-7b":            # 模型名称 - 必须与 vLLM --served-model-name 匹配
    preferred_endpoints: ["my_endpoint"]
  "qwen3":               # 由同一端点服务的另一个模型
    preferred_endpoints: ["my_endpoint"]
```

### 示例：Llama / Qwen 后端配置

```yaml
vllm_endpoints:
  - name: "local-vllm"
    address: "127.0.0.1"
    port: 8000

model_config:
  "llama2-7b":
    preferred_endpoints: ["local-vllm"]
  "qwen3":
    preferred_endpoints: ["local-vllm"]
```

#### 地址格式要求

**重要**：`address` 字段必须包含有效的 IP 地址（IPv4 或 IPv6）。不支持域名和其他格式。

**✅ 支持的格式：**

```yaml
# IPv4 地址
address: "127.0.0.1"

# IPv6 地址
address: "2001:db8::1"
```

**❌ 不支持的格式：**

```yaml
# 域名
address: "localhost"        # ❌ 请使用 127.0.0.1 代替
address: "api.openai.com"   # ❌ 请使用 IP 地址代替

# 协议前缀
address: "http://127.0.0.1"   # ❌ 删除协议前缀

# 路径
address: "127.0.0.1/api"      # ❌ 删除路径，仅使用 IP

# 地址中的端口
address: "127.0.0.1:8080"     # ❌ 使用单独的 'port' 字段
```

#### 模型名称一致性

`model_config` 中的模型名称必须与启动 vLLM 服务器时使用的 `--served-model-name` 参数**完全匹配**：

```bash
# vLLM 服务器命令（示例）：
vllm serve meta-llama/Llama-2-7b-hf --served-model-name llama2-7b --port 8000
vllm serve Qwen/Qwen3-1.8B --served-model-name qwen3 --port 8000

# config.yaml 必须在 model_config 中引用模型：
model_config:
  "llama2-7b":  # ✅ 匹配 --served-model-name
    preferred_endpoints: ["your-endpoint"]
  "qwen3":      # ✅ 匹配 --served-model-name
    preferred_endpoints: ["your-endpoint"]
```

### 模型设置

配置模型特定的设置：

```yaml
model_config:
  "llama2-7b":
    pii_policy:
      allow_by_default: true    # 默认允许 PII
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]
    preferred_endpoints: ["my_endpoint"]  # 可选：指定可以为该模型提供服务的端点

  "gpt-4":
    pii_policy:
      allow_by_default: false
    # preferred_endpoints 省略 - 路由器将不设置端点头
    # 当外部负载均衡器处理端点选择时很有用
```

**关于 `preferred_endpoints` 的说明：**

- **可选字段**：如果省略，路由将不会设置 `x-vsr-destination-endpoint` 头
- **如果指定**：路由根据权重选择最佳端点并设置头
- **如果省略**：上游负载均衡器或服务网格处理端点选择
- **验证**：在类别中使用或作为 `default_model` 的模型必须配置 `preferred_endpoints`

### 定价（可选）

如果您希望路由计算请求成本并公开 Prometheus 成本指标，请在 `model_config` 的每个模型下添加每 100 万 token 的定价和货币。

```yaml
model_config:
  phi4:
    pricing:
      currency: USD
      prompt_per_1m: 0.07
      completion_per_1m: 0.35
  "mistral-small3.1":
    pricing:
      currency: USD
      prompt_per_1m: 0.1
      completion_per_1m: 0.3
  gemma3:27b:
    pricing:
      currency: USD
      prompt_per_1m: 0.067
      completion_per_1m: 0.267
```

- 成本公式：`(prompt_tokens * prompt_per_1m + completion_tokens * completion_per_1m) / 1_000_000` (使用给定货币)。
- 如果未配置，路由仍会报告 token 和延迟指标；成本视为 0。

### 分类模型

配置 BERT 分类模型：

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6            # 分类置信度阈值
    use_cpu: true             # 使用 CPU（不需要 GPU）
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    threshold: 0.7            # PII 检测阈值
    use_cpu: true
```

### 类别和路由

使用基于决策的路由系统定义如何处理不同的查询类型：

```yaml
# 类别定义用于分类的领域
categories:
- name: math
- name: computer science
- name: other

# 决策定义带有规则和模型选择的路由逻辑
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: your-model
      use_reasoning: true            # 为该模型在数学问题上启用推理

- name: computer science
  description: "Route computer science queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "computer science"
  modelRefs:
    - model: your-model
      use_reasoning: true            # 为代码启用推理

- name: other
  description: "Route general queries"
  priority: 5
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "other"
  modelRefs:
    - model: your-model
      use_reasoning: false           # 通用查询不进行推理

default_model: your-model          # 回退模型
```

### 模型特定的推理

`use_reasoning` 字段在每个决策的 modelRefs 中为每个模型配置，允许细粒度控制：

```yaml
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: gpt-oss-120b
      use_reasoning: true            # GPT-OSS-120b 支持数学推理
    - model: phi4
      use_reasoning: false           # phi4 不支持推理模式
    - model: deepseek-v31
      use_reasoning: true            # DeepSeek 支持数学推理
```

### 模型推理配置

配置不同模型如何处理推理模式语法。这允许您在不更改代码的情况下添加新模型：

```yaml
# 模型推理配置 - 定义不同模型如何处理推理语法
model_reasoning_configs:
  - name: "deepseek"
    patterns: ["deepseek", "ds-", "ds_", "ds:", "ds "]
    reasoning_syntax:
      type: "chat_template_kwargs"
      parameter: "thinking"

  - name: "qwen3"
    patterns: ["qwen3"]
    reasoning_syntax:
      type: "chat_template_kwargs"
      parameter: "enable_thinking"

  - name: "gpt-oss"
    patterns: ["gpt-oss", "gpt_oss"]
    reasoning_syntax:
      type: "reasoning_effort"
      parameter: "reasoning_effort"

  - name: "gpt"
    patterns: ["gpt"]
    reasoning_syntax:
      type: "reasoning_effort"
      parameter: "reasoning_effort"

# 全局默认推理努力等级（当未按类别指定时）
default_reasoning_effort: "medium"
```

#### 模型推理配置选项

**配置结构：**

- `name`：模型家族的唯一标识符
- `patterns`：与模型名称匹配的模式数组
- `reasoning_syntax.type`：模型期望如何指定推理模式
  - `"chat_template_kwargs"`：使用聊天模板参数（用于 DeepSeek、Qwen3 等模型）
  - `"reasoning_effort"`：使用与 OpenAI 兼容的 reasoning_effort 字段（用于 GPT 模型）
- `reasoning_syntax.parameter`：模型使用的具体参数名称

**模式匹配：**
系统支持简单的字符串模式和正则表达式，以实现灵活的模型匹配：

- **简单字符串匹配**：`"deepseek"` 匹配任何包含 "deepseek" 的模型
- **前缀模式**：`"ds-"` 匹配以 "ds-" 开头或完全是 "ds" 的模型
- **正则表达式**：`"^gpt-4.*"` 匹配以 "gpt-4" 开头的模型
- **通配符**：`"*"` 匹配所有模型（用于回退配置）
- **多个模式**：`["deepseek", "ds-", "^phi.*"]` 匹配这些模式中的任何一个

**正则表达式模式示例：**

```yaml
patterns:
  - "^gpt-4.*"        # Models starting with "gpt-4"
  - ".*-instruct$"    # Models ending with "-instruct"
  - "phi[0-9]+"       # Models like "phi3", "phi4", etc.
  - "^(llama|mistral)" # Models starting with "llama" or "mistral"
```

**添加新模型：**
要支持新的模型家族（例如 Claude），只需添加新配置：

```yaml
model_reasoning_configs:
  - name: "claude"
    patterns: ["claude"]
    reasoning_syntax:
      type: "chat_template_kwargs"
      parameter: "enable_reasoning"
```

**未知模型：**
不匹配任何已配置模式的模型在启用推理模式时将不会应用任何推理字段。这可以防止不支持推理语法的模型出现问题。

**默认推理努力等级：**
设置全局默认推理努力等级，当类别未指定其自己的努力等级时使用：

```yaml
default_reasoning_effort: "high"  # 选项: "low", "medium", "high"
```

**决策特定的推理努力等级：**
覆盖每个决策的默认努力等级：

```yaml
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  reasoning_effort: "high"        # 对复杂数学使用高努力等级
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: your-model
      use_reasoning: true           # 为该模型启用推理

- name: general
  description: "Route general queries"
  priority: 5
  reasoning_effort: "low"         # 对通用查询使用低努力等级
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "general"
  modelRefs:
    - model: your-model
      use_reasoning: true           # 为该模型启用推理
```

### 安全特性

配置 PII 检测和越狱保护：

```yaml
# PII 检测
classifier:
  pii_model:
    threshold: 0.7                 # 越高 = PII 检测越严格

# 越狱防护
prompt_guard:
  enabled: true                    # 启用越狱检测
  threshold: 0.7                   # 检测灵敏度
  use_cpu: true                    # 在 CPU 上运行

# 模型级 PII 策略
model_config:
  "your-model":
    pii_policy:
      allow_by_default: true       # 默认允许大多数内容
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]  # 允许的具体类型
```

### 可选特性

配置其他特性：

```yaml
# 语义缓存
semantic_cache:
  enabled: true                   # 全局启用语义缓存
  backend_type: "memory"          # 选项: "memory" 或 "milvus"
  similarity_threshold: 0.8       # 全局默认缓存命中阈值
  max_entries: 1000               # 最大缓存条目
  ttl_seconds: 3600               # 缓存过期时间
  eviction_policy: "fifo"         # 选项: "fifo", "lru", "lfu"

# 决策级缓存配置（新）
# 覆盖特定决策的全局缓存设置
categories:
  - name: health
  - name: general_chat
  - name: troubleshooting

decisions:
  - name: health
    description: "Route health queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "health"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95  # 非常严格 - 医疗准确性至关重要

  - name: general_chat
    description: "Route general chat queries"
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "general_chat"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "semantic-cache"
        configuration:
          similarity_threshold: 0.75  # 放宽以获得更好的缓存命中率

  - name: troubleshooting
    description: "Route troubleshooting queries"
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "troubleshooting"
    modelRefs:
      - model: your-model
        use_reasoning: false
    # 无缓存插件 - 使用全局默认值 (0.8)

# 工具自动选择
tools:
  enabled: true                    # 启用工具自动选择
  top_k: 3                        # 选择的工具数量
  similarity_threshold: 0.2        # 工具相关性阈值
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true         # 失败时返回空

# 用于相似度的 BERT 模型
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6                  # 相似度阈值
  use_cpu: true                   # 仅 CPU 推理

# 批量分类 API 配置
api:
  batch_classification:
    max_batch_size: 100            # 每个批量请求的最大文本数
    concurrency_threshold: 5       # 在此大小时切换到并发处理
    max_concurrency: 8             # 最大并发 goroutine
    
    # 监控指标配置
    metrics:
      enabled: true                # 启用 Prometheus 指标收集
      detailed_goroutine_tracking: true  # 跟踪单个 goroutine 生命周期
      high_resolution_timing: false      # 使用纳秒精度计时
      sample_rate: 1.0                   # 收集所有请求的指标 (1.0 = 100%)
      
      # 指标的批量大小范围标签（可选 - 使用合理的默认值）
      # 默认范围: "1", "2-5", "6-10", "11-20", "21-50", "50+"
      # 仅在需要自定义范围时指定：
      # batch_size_ranges:
      #   - {min: 1, max: 1, label: "1"}
      #   - {min: 2, max: 5, label: "2-5"}
      #   - {min: 6, max: 10, label: "6-10"}
      #   - {min: 11, max: 20, label: "11-20"}
      #   - {min: 21, max: 50, label: "21-50"}
      #   - {min: 51, max: -1, label: "50+"}  # -1 表示无上限
      
      # 直方图桶 - 从下面的预设中选择或自定义
      duration_buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30]
      size_buckets: [1, 2, 5, 10, 20, 50, 100, 200]
      
      # 快速配置的预设示例（复制上面的值）
      preset_examples:
        fast:
          duration: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
          size: [1, 2, 3, 5, 8, 10]
        standard:
          duration: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
          size: [1, 2, 5, 10, 20, 50, 100]
        slow:
          duration: [0.1, 0.5, 1, 5, 10, 30, 60, 120]
          size: [10, 50, 100, 500, 1000, 5000]
```

### 如何使用预设示例

配置包含用于快速设置的预设示例。以下是如何使用它们：

**步骤 1：选择您的场景**

- `fast` - 用于实时 API（微秒到毫秒响应时间）
- `standard` - 用于典型的 Web API（毫秒到秒响应时间）
- `slow` - 用于批处理或大量计算（秒到分钟）

**步骤 2：复制预设值**

```yaml
# 示例：切换到快速 API 配置
# 从 preset_examples.fast 复制并粘贴到实际配置：
duration_buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
size_buckets: [1, 2, 3, 5, 8, 10]
```

**步骤 3：重启服务**

```bash
pkill -f "router"
make run-router
```

### 默认批量大小范围

系统提供了适用于大多数用例的合理默认批量大小范围：

- **"1"** - 单个文本请求
- **"2-5"** - 小批量请求
- **"6-10"** - 中批量请求
- **"11-20"** - 大批量请求
- **"21-50"** - 超大批量请求
- **"50+"** - 最大批量请求

**除非您有特定要求，否则无需配置 `batch_size_ranges`。** 省略配置时会自动使用默认值。

### 按用例的配置示例

**实时聊天 API (fast 预设)**

```yaml
# 复制这些值到您的配置中以进行亚毫秒级监控
duration_buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
size_buckets: [1, 2, 3, 5, 8, 10]
# batch_size_ranges: 使用默认值（无需配置）
```

**电子商务 API (standard 预设)**

```yaml
# 复制这些值以用于典型的 Web API 响应时间
duration_buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
size_buckets: [1, 2, 5, 10, 20, 50, 100]
# batch_size_ranges: 使用默认值（无需配置）
```

**数据处理管道 (slow 预设)**

```yaml
# 复制这些值以用于重型计算工作负载
duration_buckets: [0.1, 0.5, 1, 5, 10, 30, 60, 120]
size_buckets: [10, 50, 100, 500, 1000, 5000]
# 用于大规模处理的自定义批量大小范围（覆盖默认值）
batch_size_ranges:
  - {min: 1, max: 50, label: "1-50"}
  - {min: 51, max: 200, label: "51-200"}
  - {min: 201, max: 1000, label: "201-1000"}
  - {min: 1001, max: -1, label: "1000+"}
```

**可用指标：**

- `batch_classification_requests_total` - 批量请求总数
- `batch_classification_duration_seconds` - 处理持续时间直方图
- `batch_classification_texts_total` - 处理的文本总数
- `batch_classification_errors_total` - 按类型划分的错误计数
- `batch_classification_concurrent_goroutines` - 活动 goroutine 计数
- `batch_classification_size_distribution` - 批量大小分布

访问指标地址：`http://localhost:9190/metrics`

## 类别级缓存配置

**新功能**：在类别级别配置语义缓存设置，以精细控制缓存行为。

### 为什么要使用类别级缓存设置？

不同的类别对语义变化的容忍度不同：

- **敏感类别**（健康、心理学、法律）：微小的词语变化可能有显著的意义差异。需要高相似度阈值 (0.92-0.95)。
- **通用类别**（聊天、故障排除）：对微小的措辞变化不太敏感。可以使用较低的阈值 (0.75-0.82) 以获得更好的缓存命中率。
- **隐私类别**：出于合规或安全原因，可能需要完全禁用缓存。

### 配置示例

#### 示例 1：针对不同决策的混合阈值

```yaml
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.8  # 全局默认值

categories:
  - name: health
  - name: psychology
  - name: general_chat
  - name: troubleshooting

decisions:
  - name: health
    description: "Route health queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "health"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a health expert..."
          mode: "replace"
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95  # 非常严格 - "headache" vs "severe headache" = 不同

  - name: psychology
    description: "Route psychology queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "psychology"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a psychology expert..."
          mode: "replace"
      - type: "semantic-cache"
        configuration:
          similarity_threshold: 0.92  # 严格 - 临床细微差别很重要

  - name: general_chat
    description: "Route general chat queries"
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "general_chat"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a helpful assistant..."
          mode: "replace"
      - type: "semantic-cache"
        configuration:
          similarity_threshold: 0.75  # 放宽 - "how's the weather" = "what's the weather"

  - name: troubleshooting
    description: "Route troubleshooting queries"
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "troubleshooting"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a tech support expert..."
          mode: "replace"
    # 无缓存插件 - 使用全局默认值 0.8
```

#### 示例 2：禁用敏感数据的缓存

```yaml
categories:
  - name: personal_data

decisions:
  - name: personal_data
    description: "Route personal data queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "personal_data"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "Handle personal information..."
          mode: "replace"
      - type: "semantic-cache"
        configuration:
          enabled: false  # 为了隐私完全禁用缓存
```

### 配置选项

**决策级插件字段：**

- `plugins[].type: "semantic-cache"` - 语义缓存插件配置
  - `configuration.enabled` (可选, boolean)：启用/禁用此决策的缓存。如果未指定，继承自全局 `semantic_cache.enabled`。
  - `configuration.similarity_threshold` (可选, float 0.0-1.0)：此决策的缓存命中的最小相似度分数。如果未指定，继承自全局 `semantic_cache.similarity_threshold`。

**回退层级：**

1. 决策特定的插件 `similarity_threshold`（如果设置）
2. 全局 `semantic_cache.similarity_threshold`（如果设置）
3. `bert_model.threshold`（最终回退）

### 最佳实践

**阈值选择：**

- **高精度 (0.92-0.95)**：健康、心理学、法律、金融
- **中等精度 (0.85-0.90)**：技术文档、教育
- **较低精度 (0.75-0.82)**：一般聊天、常见问题解答、故障排除

**隐私和合规性：**

- 对于处理以下内容的决策，禁用缓存（设置插件 `enabled: false`）：
  - 个人身份信息 (PII)
  - 金融数据
  - 健康记录
  - 敏感商业信息

**性能调优：**

- 从保守（较高）的阈值开始
- 监控每个决策的缓存命中率
- 降低命中率低的决策的阈值
- 提高有错误缓存命中的决策的阈值

## 常见配置示例

### 启用所有安全特性

```yaml
# 启用 PII 检测
classifier:
  pii_model:
    threshold: 0.8              # 严格 PII 检测

# 启用越狱保护
prompt_guard:
  enabled: true
  threshold: 0.7

# 配置模型 PII 策略
model_config:
  "your-model":
    pii_policy:
      allow_by_default: false   # 默认阻止所有 PII
      pii_types_allowed: []     # 不允许 PII
```

### 性能优化

```yaml
# 启用缓存
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.85    # 越高 = 更多缓存命中
  max_entries: 5000
  ttl_seconds: 7200             # 2 小时缓存
  eviction_policy: "fifo"       # 选项: "fifo", "lru", "lfu"

# 启用工具选择
tools:
  enabled: true
  top_k: 5                     # 选择更多工具
  similarity_threshold: 0.1    # 越低 = 选择更多工具
```

### 开发设置

```yaml
# 为测试禁用安全
prompt_guard:
  enabled: false

# 为一致结果禁用缓存
semantic_cache:
  enabled: false

# 降低分类阈值
classifier:
  category_model:
    threshold: 0.3             # 越低 = 更专业的路由
```

## 配置验证

### 测试您的配置

在开始之前验证您的配置：

```bash
# 测试配置语法
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# 使用您的配置测试路由
make build
make run-router
```

### 常见配置模式

**多模型：**

```yaml
vllm_endpoints:
  - name: "math_endpoint"
    address: "192.168.1.10"  # 数学服务器 IP
    port: 8000
    weight: 1
  - name: "general_endpoint"
    address: "192.168.1.20"  # 通用服务器 IP
    port: 8000
    weight: 1

categories:
- name: math
- name: other

decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: math-model
      use_reasoning: true           # 为数学启用推理

- name: other
  description: "Route general queries"
  priority: 5
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "other"
  modelRefs:
    - model: general-model
      use_reasoning: false          # 通用查询不进行推理
```

**负载均衡：**

```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: "192.168.1.30"  # 主服务器 IP
    port: 8000
    weight: 2              # 较高权重 = 更多流量
  - name: "endpoint2"
    address: "192.168.1.31"  # 辅助服务器 IP
    port: 8000
    weight: 1
```

## 最佳实践

### 安全配置

对于生产环境：

```yaml
# 启用所有安全特性
classifier:
  pii_model:
    threshold: 0.8              # 严格 PII 检测

prompt_guard:
  enabled: true                 # 启用越狱保护
  threshold: 0.7

model_config:
  "your-model":
    pii_policy:
      allow_by_default: false   # 默认阻止 PII
```

### 性能调优

对于高流量场景：

```yaml
# 启用缓存
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.85    # 越高 = 更多缓存命中
  max_entries: 10000
  ttl_seconds: 3600
  eviction_policy: "lru"        

# 优化分类
classifier:
  category_model:
    threshold: 0.7              # 平衡准确性与速度
```

### 开发与生产

**开发：**

```yaml
# 宽松的设置用于测试
classifier:
  category_model:
    threshold: 0.3              # 较低阈值用于测试
prompt_guard:
  enabled: false                # 开发时禁用
semantic_cache:
  enabled: false                # 为一致结果禁用
```

**生产：**

```yaml
# 严格的设置用于生产
classifier:
  category_model:
    threshold: 0.7              # 较高阈值用于准确性
prompt_guard:
  enabled: true                 # 启用安全
semantic_cache:
  enabled: true                 # 启用以获得性能
```

## 故障排除

### 常见问题

**无效的 YAML 语法：**

```bash
# 验证 YAML 语法
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

**缺少模型文件：**

```bash
# 检查模型是否已下载
ls -la models/
# 如果丢失，运行：make download-models
```

**端点连接性：**

```bash
# 测试您的后端服务器
curl -f http://your-server:8000/health
```

**配置未生效：**

```bash
# 更改配置后重启路由
make run-router
```

### 测试配置

```bash
# 使用不同查询进行测试
make test-auto-prompt-reasoning      # 数学查询
make test-auto-prompt-no-reasoning   # 通用查询
make test-pii                        # PII 检测
make test-prompt-guard               # 越狱保护
```

### 模型推理配置问题

**模型未获取推理字段：**

- 检查模型名称是否与 `model_reasoning_configs` 中的模式匹配
- 验证模式语法（精确匹配与前缀）
- 未知模型将不会应用推理字段（这是设计使然）

**应用了错误的推理语法：**

- 确保 `reasoning_syntax.type` 匹配您的模型期望的格式
- 检查 `reasoning_syntax.parameter` 名称是否正确
- DeepSeek 模型通常使用带有 `"thinking"` 的 `chat_template_kwargs`
- GPT 模型通常使用 `reasoning_effort`

**添加对新模型的支持：**

```yaml
# 添加新模型配置
model_reasoning_configs:
  - name: "my-new-model"
    patterns: ["my-model"]
    reasoning_syntax:
      type: "chat_template_kwargs"  # 或 "reasoning_effort"
      parameter: "custom_parameter"
```

**测试模型推理配置：**

```bash
# 使用您的特定模型测试推理
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ "model": "your-model-name", "messages": [{"role": "user", "content": "What is 2+2?"}] }'
```

## 配置生成

 Semantic Router 支持基于模型性能基准的自动配置生成。此工作流程使用 MMLU-Pro 评估结果来确定不同类别的最佳模型路由。

### 基准测试工作流程

1. **运行 MMLU-Pro 评估：**

    ```bash
    # 使用 MMLU-Pro 基准测试评估模型
    python src/training/model_eval/mmlu_pro_vllm_eval.py \
      --endpoint http://localhost:8000/v1 \
      --models phi4,gemma3:27b,mistral-small3.1 \
      --samples-per-category 5 \
      --use-cot \
      --concurrent-requests 4 \
      --output-dir results
    ```

2. **生成配置：**

    ```bash
    # 从基准测试结果生成 config.yaml
    python src/training/model_eval/result_to_config.py \
      --results-dir results \
      --output-file config/config.yaml \
      --similarity-threshold 0.80
    ```

### 生成的配置特性

生成的配置包括：

- **模型性能排名**：模型按每个类别的性能排名
- **推理设置**：自动配置每个类别的推理要求：
  - `use_reasoning`：是否使用逐步推理
  - `reasoning_effort`：所需的努力等级 (low/medium/high)
- **默认模型选择**：整体表现最佳的模型被设为默认值
- **安全和性能设置**：预配置的最佳值用于：
  - PII 检测阈值
  - 语义缓存设置
  - 工具选择参数

### 自定义生成的配置

可以通过以下方式自定义生成的 config.yaml：

1. 编辑 `result_to_config.py` 中的类别特定设置
2. 通过命令行参数调整阈值和参数
3. 手动修改生成的 config.yaml

### 示例工作流程

这是生成和测试配置的完整示例工作流程：

```bash
# 运行 MMLU-Pro 评估
# 选项 1：手动指定模型
python src/training/model_eval/mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --models phi4,gemma3:27b,mistral-small3.1 \
  --samples-per-category 5 \
  --use-cot \
  --concurrent-requests 4 \
  --output-dir results \
  --max-tokens 2048 \
  --temperature 0.0 \
  --seed 42

# 选项 2：从端点自动发现模型
python src/training/model_eval/mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --samples-per-category 5 \
  --use-cot \
  --concurrent-requests 4 \
  --output-dir results \
  --max-tokens 2048 \
  --temperature 0.0 \
  --seed 42

# 生成初始配置
python src/training/model_eval/result_to_config.py \
  --results-dir results \
  --output-file config/config.yaml \
  --similarity-threshold 0.80

# 测试生成的配置
make test
```

此工作流程确保您的配置：

- 基于实际模型性能
- 在部署前经过适当测试
- 经过版本控制以跟踪更改
- 针对您的特定用例进行了优化

## 下一步

- **[安装指南](installation.md)** - 设置说明
- **[快速入门指南](installation.md)** - 基本用法示例
- **[API 文档](../api/router.md)** - 完整 API 参考

配置系统旨在简单而强大。从基本配置开始，并根据需要逐步启用高级功能。
