---
title: Domain Routing
sidebar_label: Domain Routing
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/intelligent-route/domain-routing.md"
  outdated: false
---

# Domain Based Routing

本指南展示如何使用微调的 classification model 进行基于学术和专业 domain 的智能路由。Domain routing 使用带有 LoRA adapter 的专用 model（ModernBERT、Qwen3-Embedding、EmbeddingGemma）将查询 classify 到 math、physics、law、business 等 category。

## 核心优势

- **高效**：带有 LoRA adapter 的 fine-tuned model 提供快速推理（5-20ms）和高准确率
- **专业化**：多种 model 选项（ModernBERT 适用于英语，Qwen3 适用于多语言/长上下文，Gemma 适用于小内存占用）
- **多任务**：LoRA 使得通过共享 base model 运行多个 classification 任务（domain + PII + jailbreak 检测）成为可能
- **成本效益**：比基于 LLM 的 classification latency 更低，无 API 成本

## 解决什么问题？

通用 classification 方法难以处理 domain 特定术语和学术/专业 domain 之间的细微差异。Domain routing 提供：

- **准确的 domain 检测**：fine-tuned model 区分 math、physics、chemistry、law、business 等
- **多任务效率**：LoRA adapter 通过一次 base model 推理同时进行 domain classification、PII 检测和 jailbreak 检测
- **长上下文支持**：Qwen3-Embedding 支持高达 32K token（相比 ModernBERT 的 8K 限制）
- **多语言 routing**：Qwen3 在 100+ 种语言上 training，ModernBERT 针对英语优化
- **资源优化**：仅对受益的 domain（math、physics、chemistry）启用昂贵的 reasoning

## 适用场景

- **教育平台**：涵盖多样化学科 domain（STEM、人文、社会科学）
- **专业服务**：需要 domain 专业知识（法律、医疗、金融）
- **企业知识库**：跨越多个部门
- **研究辅助工具**：需要学术 domain awareness
- **多 domain 产品**：classification 准确性至关重要

## 配置

在 `config.yaml` 中配置 domain classifier：

```yaml
# 定义 domain signal
signals:
  domains:
    - name: "math"
      description: "数学和定量推理"
      mmlu_categories: ["math"]

    - name: "physics"
      description: "物理和物理科学"
      mmlu_categories: ["physics"]

    - name: "computer_science"
      description: "编程和计算机科学"
      mmlu_categories: ["computer_science"]

    - name: "business"
      description: "商业和管理"
      mmlu_categories: ["business"]

    - name: "health"
      description: "健康和医疗信息"
      mmlu_categories: ["health"]

    - name: "law"
      description: "法律和监管信息"
      mmlu_categories: ["law"]

    - name: "other"
      description: "通用查询"
      mmlu_categories: ["other"]

# 使用 domain signal 定义 decision
decisions:
  - name: math
    description: "路由数学查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "math"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位数学专家。提供带有清晰解释的逐步解决方案。"

  - name: physics
    description: "路由物理查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "physics"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位对物理定律有深刻理解的物理专家。用数学推导清晰地解释概念。"

  - name: computer_science
    description: "路由计算机科学查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "computer_science"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位精通算法和数据结构的计算机科学专家。提供清晰的代码示例。"

  - name: business
    description: "路由商业查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "business"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位资深商业顾问和战略顾问。"

  - name: health
    description: "路由健康查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "health"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位健康和医疗信息专家。"
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95

  - name: law
    description: "路由法律查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "law"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位知识渊博的法律专家。"

  - name: general_route
    description: "通用查询的默认回退路由"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "other"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.85
```

## 支持的 Domain

学术类：数学、物理、化学、生物、计算机科学、工程

专业类：商业、法律、经济、健康、心理学

通用类：哲学、历史、其他

## 功能特性

- **PII 检测**：自动检测和处理敏感信息
- **Semantic Cache**：cache 相似查询以加快响应
- **Reasoning 控制**：按 domain 启用/禁用 reasoning
- **自定义 threshold**：按 category 调整 cache 敏感度

## 请求示例

```bash
# Math 查询（启用 reasoning）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "求解：x^2 + 5x + 6 = 0"}]
  }'

# Business 查询（禁用 reasoning）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "什么是 SWOT 分析？"}]
  }'

# Health 查询（高 cache threshold）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "糖尿病的症状有哪些？"}]
  }'
```

## 真实用例

### 1. 使用 LoRA 的多任务分类（高效）

**问题**：每个请求都需要 domain classification + PII 检测 + jailbreak 检测
**解决方案**：LoRA adapter 通过一次 base model 推理运行所有 3 个 task，而非 3 个独立 model
**影响**：比运行 3 个完整 model 快 3 倍，每个 task 小于 1% 参数开销

### 2. 长文档分析（专业化 - Qwen3）

**问题**：研究论文和法律文档超过 ModernBERT 的 8K token 限制
**解决方案**：Qwen3-Embedding 支持高达 32K token，无需 truncation
**影响**：完整 document 准确 classification，truncation 不会丢失信息

### 3. 多语言教育平台（专业化 - Qwen3）

**问题**：学生用 100+ 种语言提问，ModernBERT 仅限于英语
**解决方案**：Qwen3-Embedding 在 100+ 种语言上 training，处理多语言 routing
**影响**：单一 model 服务全球用户，跨语言质量一致

### 4. 边缘部署（专业化 - Gemma）

**问题**：移动/IoT 设备无法运行大型 classification model
**解决方案**：EmbeddingGemma-300M 配合 Matryoshka embedding（128-768 维）
**影响**：Model 小 5 倍，在小于 100MB 内存的 edge 设备上运行

### 5. STEM 辅导平台（高效推理控制）

**问题**：math/physics 需要 reasoning，但 history/literature 不需要
**解决方案**：Domain classifier 将 STEM → reasoning model，人文 → fast model
**影响**：STEM 准确率提高 2 倍，非 STEM 查询节省 60% 成本

## Domain 特定优化

### STEM Domain（启用 Reasoning）

```yaml
decisions:
- name: math
  description: "路由数学查询"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: true  # 逐步解决方案
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "你是一位数学专家。提供逐步解决方案。"

- name: physics
  description: "路由物理查询"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "physics"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: true  # 推导和证明
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "你是一位物理专家。用数学推导清晰地解释概念。"

- name: chemistry
  description: "路由化学查询"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "chemistry"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: true  # 反应机理
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "你是一位化学专家。清晰地解释反应机理。"
```

### 专业 Domain（PII + Cache）

```yaml
decisions:
- name: health
  description: "路由健康查询"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "health"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: false
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "你是一位健康和医疗信息专家。"
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.95  # 非常严格

- name: law
  description: "路由法律查询"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "law"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: false
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "你是一位知识渊博的法律专家。"
```

### 通用 Domain（Fast + Cache）

```yaml
decisions:
- name: business
  description: "路由商业查询"
  priority: 100
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "business"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: false  # 快速响应
  plugins:
    - type: "system_prompt"
      configuration:
        system_prompt: "你是一位资深商业顾问和战略顾问。"

- name: general_route
  description: "通用查询的默认回退路由"
  priority: 50
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "other"
  modelRefs:
    - model: "openai/gpt-oss-120b"
      use_reasoning: false
  plugins:
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.75  # 宽松
```

## 性能特征

| Domain | Reasoning | Cache Threshold | Avg Latency | 用例 |
|--------|-----------|-----------------|-------------|----------|
| 数学 | ✅ | 0.85 | 2-5s | 逐步解决方案 |
| 物理 | ✅ | 0.85 | 2-5s | 推导 |
| 化学 | ✅ | 0.85 | 2-5s | 机理 |
| 健康 | ❌ | 0.95 | 500ms | 安全关键 |
| 法律 | ❌ | 0.85 | 500ms | 合规 |
| 商业 | ❌ | 0.80 | 300ms | 快速洞察 |
| 其他 | ❌ | 0.75 | 200ms | 通用查询 |

## 成本优化策略

1. **Reasoning Budget**：仅对 STEM（30% 的查询）启用 → 成本降低 60%
2. **Cache 策略**：敏感 domain 高 threshold → 70% hit rate
3. **Model 选择**：低价值 domain 降低 score → 使用更便宜的 model
4. **PII 检测**：仅对 health/law 启用 → 减少处理开销

## 参考

完整配置请参见 [bert_classification.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/in-tree/bert_classification.yaml)。
