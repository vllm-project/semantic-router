---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/intelligent-route/embedding-routing.md"
  outdated: true
---

# 基于嵌入的路由 (Embedding Based Routing)

本指南向您展示如何使用嵌入模型的语义相似度来路由请求。基于嵌入的路由根据含义而非精确的关键词将用户查询匹配到预定义类别，使其成为处理多样化措辞和快速演变类别的理想选择。

## 关键优势

- **可扩展性**：无需重新训练模型即可处理无限数量的类别
- **快速**：使用高效的嵌入模型（Qwen3, Gemma）实现 10-50ms 的推理
- **灵活**：通过更新关键词列表来添加/删除类别，无需重新训练模型
- **语义化**：捕捉超出精确关键词匹配的含义

## 它解决了什么问题？

当用户以不同的方式表达问题时，关键词匹配会失败。基于嵌入的路由解决了：

- **释义处理**："如何安装？" 即使没有完全相同的单词也能匹配 "安装指南"
- **意图检测**：根据语义而非表面模式进行路由
- **模糊匹配**：处理拼写错误、缩写和非正式语言
- **动态类别**：无需重新训练分类模型即可添加新类别
- **多语言支持**：嵌入可以捕捉跨语言的语义

## 何时使用

- 具有多样化查询措辞的**客户支持**
- 用户以多种不同方式询问同一事物的**产品咨询**
- 需要对错误描述进行语义理解的**技术支持**
- 需要频繁添加/更新类别的**快速演变类别**
- **适度的延迟容忍度**（为了更好的语义准确性，10-50ms 是可以接受的）

## 配置

在您的 `config.yaml` 中添加嵌入规则：

```yaml
# 定义嵌入信号
signals:
  embeddings:
    - name: "technical_support"
      threshold: 0.75
      candidates:
        - "如何配置系统"
        - "安装指南"
        - "故障排除步骤"
        - "错误信息解释"
      aggregation_method: "max"

    - name: "product_inquiry"
      threshold: 0.70
      candidates:
        - "产品特性和规格"
        - "定价信息"
        - "库存和供货情况"
      aggregation_method: "avg"

    - name: "account_management"
      threshold: 0.72
      candidates:
        - "重置密码"
        - "账号设置"
        - "订阅管理"
      aggregation_method: "max"

# 使用嵌入信号定义决策
decisions:
  - name: technical_support
    description: "路由技术支持查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "embedding"
          name: "technical_support"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位技术支持专家，在系统配置和故障排除方面拥有深厚的知识。"

  - name: product_inquiry
    description: "路由产品咨询查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "embedding"
          name: "product_inquiry"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位产品专家，对产品特性、定价和供货情况有全面的了解。"
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.85

  - name: account_management
    description: "路由账号管理查询"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "embedding"
          name: "account_management"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          system_prompt: "你是一位账号管理专家。请谨慎且安全地处理用户账号查询。"
```

## 嵌入模型

- **qwen3**: 高质量，1024 维，32K 上下文
- **gemma**: 平衡，768 维，8K 上下文，支持 Matryoshka (128/256/512/768)
- **auto**: 根据质量/延迟优先级自动选择

## 聚合方法 (Aggregation Methods)

- **max**: 使用最高的相似度分数
- **avg**: 使用所有候选词的平均相似度
- **any**: 如果任何一个候选词超过阈值则匹配

## 请求示例

```bash
# 技术支持查询
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "如何排除连接错误？"}]
  }'

# 产品咨询
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "有哪些定价方案？"}]
  }'
```

## 真实世界用例

### 1. 客户支持（可扩展类别）

**问题**：需要每周添加新的支持类别，而无需重新训练模型
**解决方案**：通过更新关键词列表来添加新类别，嵌入模型负责语义匹配
**影响**：部署新类别只需几分钟，而重新训练模型需要数周

### 2. 电子商务支持（快速语义匹配）

**问题**：“我的订单在哪里？”、“追踪包裹”、“物流状态” 都表达同一个意思
**解决方案**：Gemma 嵌入 (10-20ms) 将所有变体路由到订单追踪类别
**影响**：95% 的准确率，10-20ms 的延迟，可处理 5K+ 次查询/秒

### 3. SaaS 产品咨询（灵活路由）

**问题**：用户以 100 多种不同的方式询问定价
**解决方案**：语义相似度将所有变体匹配到 "定价信息" 关键词
**影响**：单个类别即可处理所有定价查询，无需显式规则

### 4. 创业公司迭代（快速更新类别）

**问题**：产品快速演变，需要每天调整类别
**解决方案**：在配置中更新嵌入关键词，无需重新训练模型
**影响**：更新类别只需几秒钟，而微调模型需要数天

### 5. 多语言平台（语义理解）

**问题**：英语、西班牙语、中文的相同问题需要相同的路由
**解决方案**：嵌入模型自动捕捉跨语言语义
**影响**：单个类别定义即可跨语言工作

## 模型选择策略

### 自动模式 (推荐)

```yaml
model: "auto"
quality_priority: 0.7  # 倾向于准确度
latency_priority: 0.3  # 接受一定延迟
```

- 根据优先级自动选择 Qwen3（高质量）或 Gemma（快速）
- 为每个请求平衡准确度与速度

### Qwen3 (高质量)

```yaml
model: "qwen3"
dimension: 1024
```

- 最适合：复杂查询、细微区别、高价值交互
- 延迟：每个查询约 30-50ms
- 用例：账号管理、金融查询

### Gemma (快速)

```yaml
model: "gemma"
dimension: 768  # 或使用 512, 256, 128 实现 Matryoshka
```

- 最适合：高吞吐量、简单分类、成本敏感
- 延迟：每个查询约 10-20ms
- 用例：产品咨询、一般支持

## 性能特征

| 模型 | 维度 | 延迟 | 准确度 | 内存 |
|-------|-----------|---------|----------|--------|
| Qwen3 | 1024 | 30-50ms | 最高 | 600MB |
| Gemma | 768 | 10-20ms | 高 | 300MB |
| Gemma | 512 | 8-15ms | 中 | 300MB |
| Gemma | 256 | 5-10ms | 较低 | 300MB |

## 参考

完整配置请参见 [embedding.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/in-tree/embedding.yaml)。
