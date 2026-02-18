# NVIDIA Dynamo 的语义智能层

## 1. 执行摘要

本提案概述了 **vLLM Semantic Router** 与 **NVIDIA Dynamo** 之间的全面集成策略，将语义智能与高性能分布式 Inference 相结合。该集成通过利用以下特性，创建了一个统一的 Inference 堆栈：

- **Semantic Router** 的智能请求分类（14 个领域类别）、领域感知的 System Prompt、融合路由（BERT 分类 + 关键词匹配 + 相似度搜索）、安全过滤、基于 Milvus 的 Semantic Cache
- **Dynamo** 的分离式服务 (Disaggregated Serving)、KV-aware 路由和多层内存管理

结果是一个具有系统级智能的生产级 LLM 服务平台，在**准确性**（通过优化的 Prompt 路由到正确的模型以获得最佳质量）和**效率**（最大化 GPU 利用率并最小化延迟）之间实现了最佳平衡，构建了一个整体智能的 Inference 系统。

**核心收益：**

- **系统级智能**：在整个 Inference 堆栈中优化平衡准确性和效率
- **显著降低成本**：通过智能模型选择结合基础设施优化实现
- **大幅改善延迟**：通过 Semantic Cache + KV Cache 管理以及自适应路由策略实现
- **增强 LLM 质量**：利用领域感知的 System Prompt 改善 Chain-of-Thought (CoT) 推理、Token 效率和 MoE 专家匹配
- **自适应路由智能**：通过融合路由实现：根据查询复杂度，从快速路径 (关键词) 到深度分析 (BERT)，在不牺牲准确性的情况下最大化效率
- **多信号决策**：结合 BERT 分类、关键词匹配和相似度搜索，实现稳健且准确的路由
- **全面的内容安全**：在 Inference 之前进行 PII 检测和 Jailbreak 防护
- **端到端可观测性**：跨语义层和基础设施层，用于持续系统优化

---

## 2. 动机：为什么为 Dynamo 引入 Semantic Router？

### 2.1 Dynamo 路由能力（现状）

NVIDIA Dynamo 提供了一个复杂的 **KV-aware 路由**，针对基础设施层面的效率进行了优化：

| 能力 | 描述 | 优化目标 |
|------------|-------------|---------------------|
| **KV Cache 感知路由** | 将请求路由到具有最高 KV Cache 命中率的工作节点 | TTFT，吞吐量 |
| **基于负载的路由** | 在工作节点之间平衡活动的解码块 | ITL，GPU 利用率 |
| **成本函数优化** | 最小化 `potential_prefill_blocks + potential_active_blocks` | 计算成本 |
| **基于温度的选择** | 概率性路由以防止工作节点饱和 | 负载分布 |
| **事件驱动追踪** | 通过工作节点事件实时获取缓存状态 | 路由准确性 |

**核心特征：**

- **专注于基础设施**：优化 GPU 内存和计算利用率
- **缓存感知**：利用现有的 KV Cache 来降低预填充 (Prefill) 成本
- **负载均衡**：在工作节点之间分配解码 (Decode) 工作负载
- **性能导向**：通过智能调度最小化 TTFT 和 ITL

### 2.2 Semantic Router 能力（系统智能层）

vLLM Semantic Router 提供了在请求理解层运行的**系统级智能**，通过在 **14 个领域类别**中进行智能决策，实现**准确性**与**效率**之间的最佳平衡：

| 能力 | 描述 | 智能焦点 |
|------------|-------------|---------------------|
| **意图分类** | 基于 BERT 的分类（14 个类别：数学、代码、商务、法律等） | 准确性：精确的领域理解 |
| **模型选择** | 为每个类别路由到表现最佳的模型 | 准确性：特定任务的质量优化 |
| **领域感知 System Prompt** | 自动注入类别特定的 System Prompt 用于 Prompt 工程 | 准确性：LLM CoT 质量、Token 效率、MoE 专家匹配 |
| **融合路由** | 多信号路由 (关键词 + 相似度 + BERT) | 效率：基于查询复杂度的自适应延迟 |
| **Semantic Cache** | 基于 Milvus 的向量缓存，相似度阈值 0.85+ | 效率：降低 Inference 成本 |
| **PII 检测** | Token 级分类 (PERSON, EMAIL, SSN 等) | 系统智能：隐私保护 |
| **Jailbreak 防护** | 针对 Prompt 注入攻击的二元分类 | 系统智能：安全执行 |
| **工具选择** | 相关工具的语义匹配，以减少 Prompt Token | 效率：上下文优化 |
| **推理 (Reasoning) 控制** | 为复杂查询自动启用 Reasoning 模式 | 准确性：质量感知的模式选择 |

**系统智能特征：**

- **整体智能**：跨 14 个领域类别理解查询意图、复杂度和安全影响
- **准确性-效率平衡**：根据查询复杂度动态选择路由策略 (关键词/相似度/BERT)，在最小化延迟的同时最大化准确性
- **质量优化**：根据特定任务的准确性要求选择模型和 Prompt
- **智能 Prompt 工程**：自动注入领域特定的 System Prompt，以优化 LLM 行为和输出质量
- **主动安全**：在到达 Inference 层之前拦截恶意或违反隐私的请求
- **成本智能**：对于简单查询避免使用昂贵的模型，同时确保复杂任务的质量
- **自适应路由**：多信号融合路由根据查询特征进行调整，以实现最佳的准确性-效率权衡

#### 2.2.1 具有 System Prompt 的 14 个领域分类

Semantic Router 将查询分为 **14 个专业类别**：数学、计算机科学、物理、化学、生物、工程、经济、商业、法律、心理学、哲学、历史、健康和其他。每个类别都有一个根据查询分类自动注入的优化 System Prompt。

**System Prompt 收益：**

1. **改进 Chain-of-Thought (CoT)**：领域特定的 Prompt 引导 LLM 使用适当的推理模式
   - 数学："提供逐步解决方案，清晰地展示你的工作过程"
   - 法律："提供准确的法律信息，同时清晰地陈述免责声明"
   - 商业："提供由成熟方法论支持的实用、可操作的建议"

2. **Token 效率**：优化的 Prompt 在保持质量的同时减少了不必要的冗余
   - 对于直接的类别（商业、历史）使用简短、集中的 Prompt
   - 对于需要特定方法论的复杂领域（数学、物理）使用详细的 Prompt

3. **MoE 专家匹配**：精心编写的 System Prompt 改善了 Mixture-of-Experts 模型中的专家选择
   - 领域特定的术语激活相关专家
   - 一致的 Prompt 结构提高专家路由准确性
   - 示例："你是一名数学专家" -> 激活 DeepSeek-V3 中专门研究数学的专家

4. **质量控制**：类别特定的免责声明和伦理准则
   - 医疗/法律：关于专业咨询的明确免责声明
   - 心理学：强调基于证据的方法
   - 健康：信息与医疗建议之间的清晰界限

**System Prompt 示例（数学类别）：**

```
You are a mathematics expert. Provide step-by-step solutions, show your
work clearly, and explain mathematical concepts in an understandable way.
```

**System Prompt 示例（商业类别）：**

```
You are a senior business consultant and strategic advisor with expertise
in corporate strategy, operations management, financial analysis, marketing,
and organizational development. Provide practical, actionable business advice
backed by proven methodologies and industry best practices. Consider market
dynamics, competitive landscape, and stakeholder interests in your recommendations.
```

#### 2.2.2 融合路由策略

Semantic Router 实现了一种**多信号融合路由**方法，结合了三种互补的路由方法（详见 [Prompt 分类路由提案](./prompt-classification-routing.md)）：

**1. 基于关键词 (Keyword) 的路由（快速路径）**

- 对技术特定术语（如 "kubernetes"、"SQL"、"React"）进行确定性路由
- **延迟**：极低（显著快于 BERT 分类）
- 支持布尔逻辑（AND/OR 运算符）
- 无需模型重新训练即可轻松更新
- **用例**：已知模式的精确术语匹配

**2. 基于相似度的路由（语义路径）**

- 用于语义概念检测的 Embedding 相似度
- 对改写具有鲁棒性（"逐步" ≈ "详细解释"）
- 可配置的相似度阈值（默认：0.75）
- **延迟**：低（快于完整的 BERT 分类）
- **用例**：超出精确术语的语义概念匹配

**3. BERT 分类（深度理解路径）**

- 使用 ModernBERT 进行 14 类别分类
- 对复杂查询具有最高准确性
- **延迟**：中等（全面分析）
- **用例**：全面的意图理解

**信号融合层：**

- **策略驱动决策**：结合具有可配置优先级的信号
- **路由逻辑**：
  1. 首先检查关键词规则（最快）
  2. 如果没有关键词匹配，检查相似度规则
  3. 如果没有相似度匹配，使用 BERT 分类（备选）
- **置信度评分**：每个信号提供置信度评分
- **覆盖机制**：高置信度信号可以覆盖低优先级信号
- **可观测性**：所有信号都被记录用于分析

**融合路由的系统智能收益：**

- **准确性-效率平衡**：根据查询复杂度动态选择路由策略——确定性模式的快速路径 (关键词) 可实现最小延迟，而复杂查询的深度分析 (BERT) 可确保最大准确性
- **自适应智能**：系统自动选择满足准确性要求的最高效信号，避免不必要的计算
- **灵活性**：无需重新训练模型即可轻松添加新的路由规则，实现持续的系统优化
- **鲁棒性**：多信号提供冗余和交叉验证，降低误分类风险并提高整体系统可靠性
- **整体优化**：在每个路由决策中同时考虑准确性和效率，使系统级智能最大化

---

### 2.3 差异化分析：互补优势

这两个系统在 Inference 堆栈的**不同层**运行，**重叠极小**：

#### Semantic Router：请求智能层

```
用户查询 → [语义理解] → 模型选择 → 请求增强
```

- **内容**：理解查询语义、意图和安全性
- **原因**：为任务路由到正确的模型
- **时机**：在请求到达基础设施之前
- **优化**：准确性、成本、安全

#### Dynamo 路由：基础设施效率层

```
增强请求 → [工作节点选择] → KV Cache 优化 → GPU 调度
```

- **内容**：优化工作节点选择和资源分配
- **原因**：最大化 GPU 利用率并最小化延迟
- **时机**：模型选择之后，执行期间
- **优化**：TTFT、ITL、吞吐量

#### 集成价值主张

| 维度 | 仅 Semantic Router | 仅 Dynamo 路由 | **集成系统** |
|-----------|----------------------|---------------------|----------------------|
| **模型选择** | ✅ 语义准确性（14 类别） | ❌ 无模型感知 | ✅ 任务的最佳模型 |
| **工作节点选择** | ❌ 无工作节点感知 | ✅ KV Cache 优化 | ✅ 模型的最佳工作节点 |
| **Prompt 工程** | ✅ 领域感知 System Prompt | ❌ 无 Prompt 优化 | ✅ 优化的 CoT 和 MoE 匹配 |
| **融合路由** | ✅ BERT + 关键词 + 相似度融合 | ❌ 仅 KV-aware | ✅ 多信号智能路由 |
| **缓存** | ✅ 语义相似度 (Milvus) | ✅ KV Cache 重用 | ✅✅ **双层缓存** |
| **安全** | ✅ PII + Jailbreak | ❌ 无安全层 | ✅ Inference 前过滤 |
| **成本优化** | ✅ 跨模型级别 | ✅ 基础设施级别 | ✅✅ **端到端优化** |
| **延迟** | 自适应（融合路由） | 低路由开销 | **并行执行** |

**具体示例：**

```
查询："请逐步解释费马大定理的证明"

┌─────────────────────────────────────────────────────────────────┐
│ Semantic Router 层                                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. 融合路由（3 信号分析）：                                      │
│    a) 关键词匹配："定理"、"证明" → 数学 (置信度: 0.8)            │
│    b) 相似度搜索：匹配 "数学证明" 概念 (相似度: 0.87)             │
│    c) BERT 分类："数学" 类别 (置信度: 0.92)                      │
│    → 最终决策："数学" (多信号共识)                                │
│ 2. 模型选择：deepseek-v31 (最适合数学推理)                       │
│ 3. System Prompt 注入：                                          │
│    "你是一名数学专家。请提供逐步解决方案，清晰地展示你的工作过程，   │
│     并以易于理解的方式解释数学概念。"                            │
│ 4. Reasoning 模式：已启用 (基于熵的决策)                         │
│ 5. 安全：通过 (无 PII，无 Jailbreak)                             │
│ 6. Semantic Cache：未命中 (新查询)                               │
│ 7. 增强请求：                                                    │
│    - model=deepseek-v31                                         │
│    - reasoning_effort=high                                      │
│    - system_prompt=<数学专家 prompt>                             │
│└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Dynamo 路由层                                                   │
├─────────────────────────────────────────────────────────────────┤
│ 1. 工作节点池：[worker-1, worker-2, worker-3] (deepseek-v31)     │
│ 2. KV Cache 分析：                                              │
│    - worker-1: 15 个缓存块 (数学证明上下文)                       │
│    - worker-2: 3 个缓存块                                        │
│    - worker-3: 0 个缓存块                                        │
│ 3. 成本计算：                                                    │
│    - worker-1: 85 prefill + 25 active = 110 (最佳)               │
│    - worker-2: 97 prefill + 20 active = 117                     │
│    - worker-3: 100 prefill + 18 active = 118                    │
│ 4. 选择：worker-1 (显著降低预填充成本)                           │
│└─────────────────────────────────────────────────────────────────┘

结果：
- 正确的模型 (deepseek-v31 用于数学推理)
- 正确的工作节点 (具有相关 KV Cache 的 worker-1)
- 正确的模式 (启用 Reasoning)
- 相较于随机工作节点选择，TTFT 显著加快
```

### 2.4 为什么集成至关重要：实现系统级智能

**挑战 1：缺乏智能的基础设施**

- Dynamo 优化了基础设施效率，但缺乏语义理解
- 无法区分 "2+2=?" 和 "证明费马大定理"
- 在不理解复杂度或质量要求的情况下，将两者都路由到相同的模型池
- 无法根据任务特征选择专业模型（数学 vs. 代码 vs. 创意）

**挑战 2：缺乏基础设施感知的智能**

- Semantic Router 提供了智能模型选择，但缺乏基础设施可见性
- 选择了正确的模型，但没有选择最优的工作节点
- 无法利用跨工作节点的 KV Cache 重用
- 对 GPU 利用率或工作节点负载没有感知，无法进行效率优化

**解决方案：通过分层集成实现整体系统智能**

```
系统智能层 (Semantic Router)
    ↓ [准确性：模型选择、质量优化、安全]
    ↓ [效率：Semantic Cache、自适应路由、成本控制]
基础设施优化层 (Dynamo)
    ↓ [效率：工作节点选择、KV Cache、GPU 调度]
    ↓ [准确性：一致的执行、可靠的服务]
执行层 (vLLM/SGLang/TRT-LLM)
```

**结果：** 一个整体智能的系统，在每一层都针对准确性（正确的模型、正确的 Prompt、正确的质量）和效率（正确的工作节点、正确的缓存、正确的资源利用）进行优化。

---

## 3. 目标与非目标

### 3.1 目标

**核心目标：**

1. **无缝集成**：Semantic Router 作为 Dynamo 路由之前的预处理层运行
2. **双层缓存**：Semantic Cache（请求级）+ KV Cache（Token 级）协同工作
3. **模型感知路由**：Dynamo 路由到由 Semantic Router 模型选择过滤后的工作节点池
4. **安全执行**：在请求到达 Dynamo 之前进行 PII 和 Jailbreak 检测
5. **统一可观测性**：单次追踪跨越语义和基础设施两层
6. **零停机时间**：热重载语义路由规则，无需重启 Dynamo

**次要目标：**

1. **性能**：组合延迟 < 50ms（语义 + 基础设施路由）
2. **可扩展性**：通过水平扩展支持 10K+ RPS
3. **灵活性**：支持多种部署模式（sidecar、网关、嵌入式）

### 3.2 非目标

1. **替换 Dynamo 路由**：Semantic Router 是增强而非替换 Dynamo 的 KV-aware 路由
2. **修改 Dynamo 核心**：通过标准 API 进行集成，无需更改 Dynamo 内部
3. **统一配置**：为语义层和基础设施层保持独立的配置
4. **同步耦合**：如果需要，系统可以独立运行

---

## 4. 提案详情

### 4.1 深度学习模型

Semantic Router 利用**四个专业的深度学习模型**进行智能请求处理。系统结合了针对不同任务优化的 **BERT** 和 **ModernBERT** 架构。

#### 4.1.1 相似度模型 (BERT Embeddings)

**用途：** 生成用于语义相似度比较的 Embedding

**模型：** `sentence-transformers/all-MiniLM-L12-v2`

**核心特性：**

- **架构**：基于 BERT (microsoft/MiniLM-L12-H384-uncased)
  - 12 层，384 个隐藏维度，12 个注意力头
  - 使用对比学习在 10 亿+ 句子对上进行微调
  - 基础模型：标准 BERT 架构（非 ModernBERT）
- **Embedding 维度**：384
- **用例**：
  - Semantic Cache 相似度匹配（阈值：0.8）
  - 通过语义搜索进行工具选择（阈值：0.2）
  - 基于相似度的语义概念路由
- **部署**：针对成本效率进行了 CPU 优化
- **模型大小**：3340 万参数（~120 MB）

**配置：**

```yaml
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true
```

**为什么选择 BERT（而非 ModernBERT）？**

- 成熟、经过充分测试且性能卓越的模型
- 通过对比学习针对句子 Embedding 进行了优化
- 模型尺寸更小（120 MB），加载速度更快
- ModernBERT（2024 年 12 月发布）用于下述分类任务

---

#### 4.1.2 分类模型（类别检测）

**用途：** 将查询分为 14 个领域类别

**模型：** `models/category_classifier_modernbert-base_model`

**核心特性：**

- **架构**：ModernBERT-base（2024 年 12 月发布）
  - 改进架构后的 BERT 现代替代品
  - 8192 Token 上下文长度（BERT 为 512）
  - 使用 RoPE 更好地处理长上下文
  - 使用 Flash Attention 2 实现更快推理
  - 在 MMLU-Pro 数据集上针对领域分类进行了微调
- **类别**：14 个领域（数学、计算机科学、物理、化学、生物、工程、经济、商业、法律、心理学、哲学、历史、健康、其他）
- **输出**：类别标签 + 置信度评分
- **阈值**：0.6（可配置）
- **训练数据**：包含领域特定示例的 MMLU-Pro 数据集
- **模型大小**：约 1.49 亿参数 (ModernBERT-base)

**配置：**

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"
```

**模型选择影响：**

- 决定路由到哪个 LLM（例如，数学用 DeepSeek-V3，商业用 Qwen3）
- 触发领域特定的 System Prompt 注入
- 控制 Reasoning 模式激活

---

#### 4.1.3 PII 检测模型（隐私保护）

**用途：** 在 Token 级别检测个人身份信息

**模型：** `models/pii_classifier_modernbert-base_presidio_token_model`

**核心特性：**

- **架构**：针对 Token 分类微调的 ModernBERT-base
  - Token 级序列标注（BIO 标注方案）
  - 在 Microsoft Presidio 数据集上微调
  - 针对隐私敏感实体检测进行了优化
- **检测到的 PII 类型**：17 种类型，包括：
  - **身份**：`PERSON`、`AGE`、`NRP`（国籍/宗教/政治）
  - **联系方式**：`EMAIL_ADDRESS`、`PHONE_NUMBER`、`STREET_ADDRESS`、`ZIP_CODE`
  - **财务**：`CREDIT_CARD`、`IBAN_CODE`、`US_SSN`、`US_DRIVER_LICENSE`
  - **技术**：`IP_ADDRESS`、`DOMAIN_NAME`
  - **组织**：`ORGANIZATION`、`GPE`（地缘政治实体）
  - **时间**：`DATE_TIME`
- **粒度**：Token 级分类（不仅仅是实体级）
- **阈值**：0.7（可配置）
- **操作**：拦截违反模型特定 PII 策略的请求
- **模型大小**：约 1.49 亿参数 (ModernBERT-base)

**配置：**

```yaml
classifier:
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7
    use_cpu: true
    pii_mapping_path: "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"
```

**策略执行：**

```yaml
model_config:
  public-model:
    pii_policy:
      allow_by_default: false
      pii_types_allowed: ["PERSON"]  # 仅允许人名
```

**响应头（被拦截时）：**

- `x-vsr-pii-violation: true`

---

#### 4.1.4 Jailbreak 检测模型（安全）

**用途：** 检测对抗性 Prompt 和 Jailbreak 尝试

**模型：** 从 `models/` 目录自动发现

**核心特性：**

- **架构**：具有自动选择功能的多种选项
  - **LoRA 模型（首选）**：在 BERT/RoBERTa/ModernBERT 基础上微调的适配器
    - `lora_jailbreak_classifier_bert_model` (优先级 1)
    - `lora_jailbreak_classifier_roberta_model` (优先级 2)
    - `lora_jailbreak_classifier_modernbert_model` (优先级 3)
  - **遗留模型（备选）**：`jailbreak_classifier_modernbert-base_model`
  - LoRA 模型以更小的尺寸（~10-20 MB 适配器）提供更好的准确性
- **模型发现**：根据架构优先级自动选择：BERT > RoBERTa > ModernBERT
- **检测类型**：
  - Prompt 注入攻击
  - 指令覆盖尝试
  - 对抗性 Prompt
  - 社会工程
- **阈值**：0.7（可配置）
- **操作**：拦截置信度高于阈值的请求
- **模型大小**：
  - LoRA：~10-20 MB（仅适配器）+ 基础模型
  - 遗留：约 1.49 亿参数 (ModernBERT-base)

**配置：**

```yaml
prompt_guard:
  enabled: true
  use_modernbert: true
  threshold: 0.7
  use_cpu: true
  # model_id 和 jailbreak_mapping_path 会被自动发现
```

**响应头（被拦截时）：**

- `x-vsr-jailbreak-blocked: true`
- `x-vsr-jailbreak-type: {type}` (例如 "prompt_injection")
- `x-vsr-jailbreak-confidence: {score}` (例如 "0.950")

---

#### 4.1.5 模型性能摘要

| 模型 | 用途 | 架构 | 参数量 | 阈值 | CPU/GPU |
|-------|---------|--------------|------------|-----------|---------|
| **相似度** | 语义匹配 | BERT (MiniLM-L12) | 33.4M | 0.6-0.8 | CPU |
| **分类** | 类别检测 | ModernBERT-base | 149M | 0.6 | CPU |
| **PII 检测** | 隐私保护 | ModernBERT-base | 149M | 0.7 | CPU |
| **Jailbreak** | 安全过滤 | ModernBERT-base/LoRA | 149M + 适配器 | 0.7 | CPU |

**架构对比：**

| 特性 | BERT (MiniLM) | ModernBERT |
|---------|---------------|------------|
| **发布日期** | 2020 | 2024 年 12 月 |
| **上下文长度** | 512 Tokens | 8192 Tokens |
| **位置编码** | 绝对位置 | RoPE (旋转) |
| **注意力机制** | 标准 | Flash Attention 2 |
| **用例** | Embeddings | 分类 |
| **模型大小** | 33.4M 参数 | 149M 参数 |

**优化策略：**

- **并行执行**：PII 和 Jailbreak 检测并行运行
- **提前退出**：缓存命中将跳过所有模型推理
- **基于关键词的路由**：确定性模式的快速路径
- **CPU 优化**：所有模型均针对 CPU 推理进行了优化，以降低成本
- **LoRA 适配器**：Jailbreak 模型使用轻量级适配器实现更快的加载

---

### 4.2 设计原则

1. **关注点分离**：语义智能与基础设施优化保持解耦
2. **API 驱动集成**：使用 Dynamo 的前端 API 和工作节点注册机制
3. **故障安全设计**：Semantic Router 故障时回退到 Dynamo 的默认路由
4. **可观测性优先**：每个决策（语义 + 基础设施）都会被追踪并记录
5. **Kubernetes 原生**：针对具有 CRD 和 operator 的云原生部署而设计

### 4.3 系统架构

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="系统架构概览" defaultZoom={10.5}>

{`graph TB
    Client[LLM 应用<br/>OpenAI SDK]

    subgraph Main["主要处理流程"]
        direction TB

        subgraph SIL["① vLLM Semantic Router 层"]
            direction TB
            Gateway[Envoy 网关 :8080]
            ExtProc[Semantic Router ExtProc :50051]

            subgraph SC["语义组件"]
                direction LR
                Classifier[BERT 分类器]
                PIIDetector[PII 检测器]
                JailbreakGuard[Jailbreak 卫士]
            end

            SemanticCache[Semantic Cache]
            ToolSelector[工具选择器]
        end

        subgraph DL["② NVIDIA Dynamo 层"]
            direction TB
            DynamoFrontend[Dynamo 前端 :8000]

            subgraph DR["路由与管理"]
                direction LR
                DynamoRouter[KV 路由]
                KVBM[KV 块管理器]
            end

            Planner[Planner - 动态扩缩容]
        end

        subgraph EL["③ 执行层 - 工作节点池"]
            direction TB

            subgraph MP1["模型池：deepseek-v31"]
                direction LR
                W1[Prefill 工作节点]
                W2[Decode 工作节点]
            end

            subgraph MP2["模型池：phi4"]
                direction LR
                W3[Prefill 工作节点]
                W4[Decode 工作节点]
            end

            subgraph MP3["模型池：qwen3"]
                W5[工作节点 - SGLang]
            end
        end
    end

    subgraph SL["存储层"]
        direction TB
        Milvus[(Milvus<br/>Semantic Cache)]
        SystemMem[(系统内存<br/>KV Offload)]
        NVMe[(NVMe<br/>冷缓存)]
    end

    Client -->|1. 请求| Gateway
    Gateway <-->|2. ExtProc| ExtProc
    ExtProc --> Classifier
    ExtProc --> PIIDetector
    ExtProc --> JailbreakGuard
    ExtProc --> SemanticCache
    ExtProc --> ToolSelector

    Gateway -->|3. 增强请求| DynamoFrontend
    DynamoFrontend --> DynamoRouter
    DynamoRouter <--> KVBM

    DynamoRouter -->|4. 工作节点选择| W1
    DynamoRouter -->|4. 工作节点选择| W2
    DynamoRouter -.-> W3
    DynamoRouter -.-> W4
    DynamoRouter -.-> W5

    Planner -.->|扩缩容| W1
    Planner -.->|扩缩容| W2
    Planner -.->|扩缩容| W3
    Planner -.->|扩缩容| W4
    Planner -.->|扩缩容| W5

    SemanticCache <--> Milvus
    KVBM <--> SystemMem
    KVBM <--> NVMe

    W1 -->|5. 响应| DynamoFrontend
    DynamoFrontend -->|6. 响应| Gateway
    Gateway -->|7. 响应| Client

    style ExtProc fill:#e1f5ff
    style DynamoRouter fill:#c8e6c9
    style SemanticCache fill:#fff9c4
    style KVBM fill:#fff9c4
    style SL fill:#f5f5f5`}
</ZoomableMermaid>

**架构分层：**

1. **语义智能层 (Semantic Router)**
   - 带有 ExtProc 用于请求拦截的 Envoy Gateway
   - 基于 BERT 的分类和安全过滤
   - 带有 Milvus 后端的 Semantic Cache
   - 带有路由元数据的请求增强

2. **基础设施优化层 (Dynamo)**
   - Dynamo Frontend 接收增强请求
   - KV 路由执行模型感知的工作节点选择
   - Planner 处理动态扩缩容
   - KVBM 管理多级 KV Cache

3. **执行层 (vLLM/SGLang/TRT-LLM)**
   - 模型特定的工作节点池
   - 分离式 Prefill/Decode 工作节点
   - 后端无关的执行

4. **存储层**
   - 用于 Semantic Cache 的 Milvus
   - 用于 KV Cache Offload 的系统内存
   - 用于冷 KV Cache 存储的 NVMe

### 4.4 请求流程

#### 4.4.1 端到端请求处理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段 1：语义智能 (Semantic Router)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 步骤 1：请求拦截                                                             │
│   - Envoy Gateway 接收 OpenAI API 请求                                      │
│   - 通过 ExtProc gRPC 调用 Semantic Router                                  │
│   - 从 messages 数组中提取查询                                              │
│                                                                              │
│ 步骤 2：安全过滤（并行执行）                                                 │
│   - PII 检测：扫描 PERSON、EMAIL、SSN 等                                     │
│   - Jailbreak 防护：针对 Prompt 注入的二元分类                               │
│   - 操作：如果检测到违反安全规定，则拦截                                     │
│   - 延迟：低                                                                 │
│                                                                              │
│ 步骤 3：Semantic Cache 查询                                                 │
│   - 为查询生成 BERT Embedding                                               │
│   - 在 Milvus 中搜索相似查询（阈值：0.85）                                   │
│   - 操作：如果命中，则返回缓存响应                                           │
│   - 延迟：极低（命中时），低（未命中时）                                     │
│                                                                              │
│ 步骤 4：融合路由（多信号分类）                                               │
│   - 信号 1：关键词匹配（快速路径）                                           │
│   - 信号 2：相似度搜索（语义概念）                                           │
│   - 信号 3：BERT 分类（深度理解）                                            │
│   - 基于熵的 Reasoning 决策                                           │
│   - 类别：数学、代码、Reasoning、创意等                                      │
│   - 延迟：自适应（关键词：极低，相似度：低，BERT：中等）                      │
│                                                                              │
│ 步骤 5：模型选择                                                             │
│   - 查询类别 → 模型得分映射                                                  │
│   - 为该类别选择表现最佳的模型                                               │
│   - 示例："数学" → deepseek-v31 (得分: 0.92)                                 │
│                                                                              │
│ 步骤 6：请求增强                                                             │
│   - 添加请求头：                                                             │
│     * X-VSR-Model: deepseek-v31                                             │
│     * X-VSR-Category: math                                                  │
│     * X-VSR-Reasoning: true                                                 │
│     * X-VSR-Reasoning-Effort: high                                          │
│     * X-VSR-Cache-Status: miss                                              │
│   - 修改请求体：                                                             │
│     * 将 "model" 字段更新为所选模型                                          │
│     * 如果适用，注入 Reasoning 参数                                          │
│     * 如果启用了工具选择，添加所选工具                                       │
│                                                                              │
│ 总延迟：低到中等（并行执行）                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段 2：基础设施优化 (Dynamo)                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 步骤 7：Dynamo Frontend 接收请求                                             │
│   - 解析 X-VSR-Model 请求头                                                  │
│   - 将工作节点池过滤为模型特定的工作节点                                     │
│   - 示例：仅考虑提供 deepseek-v31 服务的工作节点                             │
│                                                                              │
│ 步骤 8：KV-Aware 工作节点选择                                                │
│   - 查询 KVBM 以获取每个工作节点的缓存块                                     │
│   - 计算每个工作节点的成本：                                                 │
│     * potential_prefill_blocks = (input_tokens - overlap_blocks) / block_size│
│     * potential_active_blocks = current_active + new_request_blocks         │
│     * logit = kv_overlap_weight × prefill + active                          │
│   - 选择成本最低的工作节点                                                   │
│   - 延迟：低                                                                 │
│                                                                              │
│ 步骤 9：请求转发                                                             │
│   - 转发到所选的工作节点（Prefill 或 Decode）                                │
│   - 工作节点使用 vLLM/SGLang/TRT-LLM 处理请求                                │
│   - KVBM 追踪新的 KV Cache 块                                                │
│                                                                              │
│ 总延迟：低（路由开销）                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段 3：响应处理                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 步骤 10：工作节点响应                                                        │
│   - vLLM/SGLang 生成 Token                                                  │
│   - 将响应流传回 Dynamo Frontend                                             │
│                                                                              │
│ 步骤 11：Semantic Cache 更新                                                │
│   - Semantic Router 通过 ExtProc 接收响应                                    │
│   - 将查询 Embedding + 响应存储在 Milvus 中                                  │
│   - TTL：7200 秒（可配置）                                                   │
│                                                                              │
│ 步骤 12：响应客户端                                                          │
│   - Envoy Gateway 转发响应                                                   │
│   - 添加响应头：                                                             │
│     * X-VSR-Model-Used: deepseek-v31                                        │
│     * X-VSR-Cache-Hit: false                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.4.2 双层缓存策略

集成利用了**两个互补的缓存层**：

**第 1 层：Semantic Cache（请求级）**

- **粒度**：整个请求-响应对
- **匹配**：Embedding 相似度（余弦距离）
- **阈值**：0.85（可配置）
- **后端**：Milvus（向量数据库）
- **收益**：完全避免了类似查询的 Inference
- **示例**："2+2 等于多少？" ≈ "计算 2 加 2" (相似度: 0.91)

**第 2 层：KV Cache（Token 级）**

- **粒度**：Token 级 KV Cache 块
- **匹配**：精确前缀匹配
- **后端**：GPU HBM → 系统内存 → NVMe
- **收益**：降低部分重叠请求的 Prefill 成本
- **示例**："解释量子计算" → "解释量子计算的应用" (前缀重用)

**综合收益：**

```
场景 1：精确语义匹配
  查询："法国的首都是哪里？"
  Semantic Cache：命中 (与 "什么是法国的首都？" 具有高相似度)
  KV Cache：不适用 (跳过 Inference)
  延迟：极低 (仅缓存查询)
  成本降低：最大 (无 Inference)

场景 2：部分语义匹配 + KV 重用
  查询："详细解释费马大定理的证明"
  Semantic Cache：未命中 (新查询)
  KV Cache：命中 (与 "解释费马大定理" 有显著重叠)
  延迟：降低 (相较于无 KV 重用)
  成本降低：显著 (节省 Prefill 成本)

场景 3：新查询
  查询："为区块链设计一种分布式共识算法"
  Semantic Cache：未命中
  KV Cache：未命中
  延迟：标准 (完整 Inference)
  成本降低：无 (但被路由到最佳模型)
```

### 4.5 在 Kubernetes 中的集成

#### 4.5.1 部署架构

集成在 Kubernetes 中遵循**分层服务架构**，在语义智能与基础设施优化之间有明确的分离：

```
┌─────────────────────────────────────────────────────────────────────┐
│ Kubernetes 集群：llm-inference-stack                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 第 1 层：网关与语义智能                                      │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                             │    │
│  │  [Envoy Gateway]                                           │    │
│  │       ↓ (ExtProc gRPC)                                     │    │
│  │  [Semantic Router 服务]                                    │    │
│  │   - Pods: 3 个副本 (高可用)                                │    │
│  │   - 端口: 50051 (gRPC)                                     │    │
│  │   - 功能:                                                   │    │
│  │     * BERT 分类 (14 个类别)                                 │    │
│  │     * System Prompt 注入                                    │    │
│  │     * PII/Jailbreak 检测                                    │    │
│  │     * Semantic Cache 查询                                   │    │
│  │     * 模型选择                                              │    │
│  │   - 依赖项:                                                 │    │
│  │     * Milvus 服务 (Semantic Cache)                          │    │
│  │     * ConfigMap (路由规则)                                  │    │
│  │     * PVC (ML 模型)                                         │    │
│  │                                                             │    │
│  │  [Milvus 服务]                                             │    │
│  │   - 端口: 19530 (gRPC)                                     │    │
│  │   - 用于 Semantic Cache 的向量数据库                        │    │
│  │   - 存储: 用于持久化的 PVC                                  │    │
│  │                                                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓                                          │
│                   (带有请求头的 HTTP:                                │
│                    X-VSR-Model, X-VSR-Category 等)                  │
│                          ↓                                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 第 2 层：基础设施优化 (Dynamo)                                │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                             │    │
│  │  [Dynamo Frontend 服务]                                    │    │
│  │   - Pods: 2 个副本 (高可用)                                │    │
│  │   - 端口: 8000 (HTTP)                                      │    │
│  │   - 功能:                                                   │    │
│  │     * 解析 X-VSR-Model 请求头                               │    │
│  │     * 按模型过滤工作节点池                                  │    │
│  │     * KV-aware 工作节点选择                                 │    │
│  │     * 请求转发                                              │    │
│  │   - 组件:                                                   │    │
│  │     * KV 路由                                               │    │
│  │     * Planner (动态扩缩容)                                  │    │
│  │     * KVBM (KV Cache 管理器)                                │    │
│  │                                                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓                                          │
│                   (基于模型 + KV Cache 状态                          │
│                    选择工作节点)                                     │
│                          ↓                                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 第 3 层：执行 (vLLM/SGLang 工作节点)                        │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                             │    │
│  │  [模型池：deepseek-v31]                                     │    │
│  │   - StatefulSet: 多个副本                                   │    │
│  │   - 服务: vllm-deepseek-v31-svc                             │    │
│  │   - GPU: 每个 pod 多个 GPU                                  │    │
│  │   - 特性: 前缀缓存, fp8 KV Cache                             │    │
│  │                                                             │    │
│  │  [模型池：qwen3]                                            │    │
│  │   - StatefulSet: 多个副本                                   │    │
│  │   - 服务: vllm-qwen3-svc                                    │    │
│  │   - GPU: 每个 pod 多个 GPU                                  │    │
│  │                                                             │    │
│  │  [模型池：phi4]                                             │    │
│  │   - StatefulSet: 多个副本                                   │    │
│  │   - 服务: vllm-phi4-svc                                     │    │
│  │   - GPU: 每个 pod 单个/多个 GPU                             │    │
│  │                                                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**关键 Kubernetes 服务：**

1. **semantic-router-svc** (ClusterIP)
   - 在 50051 端口暴露 Semantic Router ExtProc
   - 被 Envoy Gateway 用于请求处理
   - 选择器：`app=semantic-router`

2. **dynamo-frontend-svc** (ClusterIP)
   - 在 8000 端口暴露 Dynamo Frontend
   - 接收来自 Envoy Gateway 的增强请求
   - 选择器：`app=dynamo-frontend`

3. **vllm-\{model\}-svc** (Headless Service)
   - 每个模型池一个服务
   - 实现 pod 之间的直接通信
   - 被 Dynamo 用于工作节点选择
   - 选择器：`app=vllm-worker, model=\{model-name\}`

4. **milvus-svc** (ClusterIP)
   - 在 19530 端口暴露 Milvus (gRPC)
   - 被 Semantic Router 用于 Semantic Cache
   - 用于 Embedding 相似度搜索的向量数据库
   - 选择器：`app=milvus`

#### 4.5.2 服务通信流程

**端到端请求路径：**

```
┌──────────────────────────────────────────────────────────────────────┐
│ 步骤 1：客户端请求                                                   │
├──────────────────────────────────────────────────────────────────────┤
│ POST /v1/chat/completions                                            │
│ Host: llm-gateway.example.com:8080                                   │
│ Content-Type: application/json                                       │
│                                                                       │
│ {                                                                    │
│   "messages": [                                                      │
│     {"role": "user", "content": "证明费马大定理"}                     │
│   ],                                                                 │
│   "model": "auto"                                                    │
│ }                                                                    │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 步骤 2：Envoy Gateway (端口 8080)                                    │
├──────────────────────────────────────────────────────────────────────┤
│ - 接收 HTTP 请求                                                    │
│ - 调用 ExtProc: semantic-router-svc:50051 (gRPC)                    │
│ - 将请求体 + 请求头发送到 Semantic Router                            │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 步骤 3：Semantic Router 服务 (ExtProc gRPC)                         │
├──────────────────────────────────────────────────────────────────────┤
│ 处理流水线：                                                         │
│                                                                       │
│ 3.1 融合路由（多信号分类）                                           │
│     - 输入："证明费马大定理"                                         │
│     - 关键词匹配：未匹配                                             │
│     - 相似度搜索：无强匹配                                           │
│     - BERT 分类：category="数学", 置信度=0.92                        │
│     - 决策：使用 BERT 结果（最高置信度）                             │
│                                                                       │
│ 3.2 System Prompt 选择                                               │
│     - 查询：categories["math"].system_prompt                         │
│     - Prompt："你是一名数学专家..."                                  │
│                                                                       │
│ 3.3 模型选择                                                         │
│     - 查询：categories["math"].model_scores                          │
│     - 选中：deepseek-v31 (得分: 0.92, reasoning: true)               │
│                                                                       │
│ 3.4 安全检查                                                         │
│     - PII 检测：通过 (无敏感数据)                                    │
│     - Jailbreak 防护：通过 (合法查询)                                │
│                                                                       │
│ 3.5 Semantic Cache 查询                                              │
│     - 查询 Milvus：Embedding 相似度搜索                              │
│     - 结果：未命中 (新查询)                                          │
│                                                                       │
│ 3.6 对 Envoy 的响应                                                  │
│     - 修改后的请求体：                                               │
│       * model: "auto" → "deepseek-v31" (已覆盖)                      │
│       * messages: [注入了 System Prompt]                             │
│     - 可观测性头（可选，添加到响应中）：                             │
│       * x-vsr-selected-category: math                               │
│       * x-vsr-selected-reasoning: on                                │
│       * x-vsr-selected-model: deepseek-v31                          │
│       * x-vsr-injected-system-prompt: true                          │
│└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 步骤 4：Envoy Gateway (转发)                                         │
├──────────────────────────────────────────────────────────────────────┤
│ - 接收来自 Semantic Router 的增强请求                                │
│ - 转发到：dynamo-frontend-svc:8000                                   │
│ - 请求体现已包含：model="deepseek-v31" (覆盖自 "auto")               │
│ - 保留可选的可观测性头                                               │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 步骤 5：Dynamo Frontend 服务 (端口 8000)                             │
├──────────────────────────────────────────────────────────────────────┤
│ 处理流水线：                                                         │
│                                                                       │
│ 5.1 请求体解析                                                       │
│     - 读取：request.model = "deepseek-v31"                           │
│     - Dynamo 对模型已被 VSR 更改一事并无感知                         │
│     - 将其视为对 deepseek-v31 的正常请求                            │
│                                                                       │
│ 5.2 工作节点池过滤                                                   │
│     - 查询 Kubernetes：vllm-deepseek-v31-svc (Headless)              │
│     - 可用工作节点：                                                 │
│       * vllm-deepseek-v31-0 (10.244.1.5:8000)                       │
│       * vllm-deepseek-v31-1 (10.244.1.6:8000)                       │
│       * vllm-deepseek-v31-2 (10.244.1.7:8000)                       │
│       * vllm-deepseek-v31-3 (10.244.1.8:8000)                       │
│                                                                       │
│ 5.3 KV-Aware 工作节点选择                                            │
│     - 查询 KVBM 以获取每个工作节点的缓存状态                          │
│     - 计算路由得分：                                                 │
│       score = kv_overlap × weight + active_blocks                   │
│     - 结果：                                                         │
│       * Worker-0: score=120 (高 KV 重叠)                             │
│       * Worker-1: score=85                                          │
│       * Worker-2: score=90                                          │
│       * Worker-3: score=75                                          │
│     - 选择：Worker-0 (10.244.1.5:8000)                               │
│                                                                       │
│ 5.4 请求转发                                                         │
│     - 转发到：http://10.244.1.5:8000/v1/chat/completions             │
│     - 请求体：model="deepseek-v31" (维持 VSR 处理后的状态)           │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 步骤 6：vLLM 工作节点 (deepseek-v31-0)                               │
├──────────────────────────────────────────────────────────────────────┤
│ 6.1 请求处理                                                         │
│     - 接收请求：model="deepseek-v31"                                 │
│     - System Prompt 已由 VSR 注入到 messages 中                      │
│     - 工作节点对 VSR 的参与并无感知                                  │
│                                                                       │
│ 6.2 Inference 执行                                                   │
│     - 模型：DeepSeek-V3                                              │
│     - Messages：[System Prompt + 用户查询]                           │
│     - 前缀缓存：已启用 (KV Cache 重用)                                │
│     - 生成包含逐步证明的响应                                         │
│                                                                       │
│ 6.3 响应生成                                                         │
│     - 返回：流式或非流式响应                                         │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 步骤 7：响应路径（反向）                                             │
├──────────────────────────────────────────────────────────────────────┤
│ 工作节点 → Dynamo Frontend → Envoy Gateway → 客户端                  │
│                                                                       │
│ - Envoy 添加可观测性头：X-Envoy-Upstream-Service-Time                │
│ - 客户端接收包含元数据的完整响应                                     │
└──────────────────────────────────────────────────────────────────────┘
```

**关键集成点：**

1. **透明模型覆盖（关键设计）**
   - 用户发送：`{"model": "auto", "messages": [...]}`
   - Semantic Router 修改请求体：`model: "auto" → "deepseek-v31"`
   - Dynamo 接收：`{"model": "deepseek-v31", "messages": [...]}`
   - **Dynamo 对 VSR 的参与完全无感知**
   - 无需为模型路由提供特殊请求头
   - 维持了标准 OpenAI API 的兼容性

2. **System Prompt 注入**
   - Semantic Router 将 System Prompt 注入到 messages 数组中
   - 示例：`messages: [{"role": "system", "content": "你是一名数学专家..."}, {"role": "user", "content": "..."}]`
   - 工作节点接收经过预增强的请求
   - Dynamo 或工作节点无需进行额外处理

3. **服务发现**
   - Envoy → Semantic Router: `semantic-router-svc.llm-inference-stack.svc.cluster.local:50051` (gRPC ExtProc)
   - Envoy → Dynamo: `dynamo-frontend-svc.llm-inference-stack.svc.cluster.local:8000` (HTTP)
   - Dynamo → 工作节点: `vllm-\{model\}-svc.llm-inference-stack.svc.cluster.local` (Headless Service)
   - Semantic Router → Milvus: `milvus-svc.llm-inference-stack.svc.cluster.local:19530` (gRPC)

4. **可观测性（可选头）**
   - `x-vsr-selected-category`：查询分类结果（如 "math"）
   - `x-vsr-selected-reasoning`：Reasoning 模式标志（如 "on" 或 "off"）
   - `x-vsr-selected-model`：VSR 选择的模型（如 "deepseek-v31"）
   - `x-vsr-injected-system-prompt`：是否注入了 System Prompt（如 "true" 或 "false"）
   - `x-vsr-cache-hit`：Semantic Cache 状态（缓存命中时值为 "true"）
   - 这些请求头**仅用于可观测性**，不被 Dynamo 用于路由
   - Dynamo 和工作节点可以忽略这些请求头
   - 请求头仅添加到未命中缓存的成功响应（HTTP 200-299）中

5. **分布式追踪**
   - 支持跨 VSR → Dynamo → 工作节点的整栈分布式追踪
   - 基于 OpenTelemetry 的检测
   - 单次追踪跨越所有层，并带有适当的上下文传播
   - 参考：[PR #322 - 分布式追踪支持](https://github.com/vllm-project/semantic-router/pull/322)
   - 实现端到端延迟分析和瓶颈识别

6. **缓存协调**
   - Semantic Cache (Milvus)：请求级，由 VSR 首先检查
   - KV Cache (Dynamo/vLLM)：Token 级，由 Dynamo 管理
   - 独立的层，无需协调
   - 如果 Semantic Cache 命中，请求将永远不会到达 Dynamo

#### 4.5.3 工作节点池管理

**通过 Kubernetes 服务发现工作节点：**

Dynamo Frontend 通过 Kubernetes Headless Service 发现工作节点，这些服务提供直接的 pod IP 地址：

1. **Headless Service 配置**
   - 服务类型：`ClusterIP: None` (headless)
   - 选择器：`app=vllm-worker, model=\{model-name\}`
   - DNS 返回所有 pod IP，而非负载均衡的 VIP

2. **工作节点注册流程**

   ```
   vLLM 工作节点 Pod 启动
   ↓
   工作节点通过 HTTP API 向 Dynamo Frontend 注册
   ↓
   Dynamo Frontend 追踪：
   - 工作节点 ID (pod 名称)
   - 模型名称 (deepseek-v31, qwen3, phi4)
   - Endpoint (pod IP:8000)
   - 能力 (Prefill, Decode, max_batch_size)
   - KV Cache 状态 (由 KVBM 追踪)
   ```

3. **模型池组织**
   - 每个模型都有专门的 StatefulSet + Headless Service
   - 示例：`vllm-deepseek-v31-svc` → 4 个服务于 DeepSeek-V3 的 pod
   - Dynamo 查询服务 DNS 以获取所有 pod IP
   - 根据来自 Semantic Router 的 `X-VSR-Model` 头过滤工作节点

4. **动态扩缩容**
   - 水平 Pod 自动扩缩器 (HPA) 根据 GPU 利用率调整副本数
   - 新 pod 在启动时向 Dynamo 自动注册
   - Dynamo 实时更新工作节点池

### 4.6 实施计划

#### 阶段 1：基础

**目标：**

- 建立 Semantic Router 与 Dynamo 之间的基础集成
- 在请求体中实现透明的模型覆盖
- 验证端到端请求流程

**任务：**

1. **Semantic Router 增强：**
   - 实现请求体修改：`model: "auto" → "selected-model"`
   - 将 System Prompt 注入添加到 messages 数组
   - 添加可选的可观测性头：
     - `x-vsr-selected-category`：分类结果
     - `x-vsr-selected-reasoning`：Reasoning 模式 ("on" 或 "off")
     - `x-vsr-selected-model`：所选模型名称
     - `x-vsr-injected-system-prompt`：System Prompt 注入状态 ("true" 或 "false")
     - `x-vsr-cache-hit`：缓存命中状态（仅在命中时）
   - 确保维持 OpenAI API 兼容性

2. **Dynamo Frontend（无需更改）：**
   - Dynamo 接收标准的 OpenAI API 请求
   - 模型字段已包含选定的模型名称
   - 无需感知 VSR 的参与
   - 现有的路由逻辑照常工作

3. **测试：**
   - 模型覆盖逻辑的单元测试
   - System Prompt 注入的集成测试
   - 验证 Dynamo 是否路由到正确的模型池
   - 1K RPS 的压力测试

**成功标准：**

- ✅ 请求根据覆盖的模型名称路由到正确的模型池
- ✅ System Prompt 正确注入到 messages 中
- ✅ Dynamo 透明运行，无需修改
- ✅ 延迟开销 < 10ms
- ✅ 不对现有部署产生破坏性更改

#### 阶段 2：双层缓存

**目标：**

- 将 Semantic Cache 与 KV Cache 集成
- 实施缓存协调策略
- 优化缓存命中率

**任务：**

1. **缓存集成：**
   - 在 Dynamo 路由之前添加 Semantic Cache 查询
   - 实现向 Dynamo 转发缓存未命中的情况
   - 添加缓存命中指标和响应头

2. **性能优化：**
   - 并行执行缓存查询和分类
   - Milvus 连接池化
   - 缓存预热策略

3. **测试：**
   - 缓存命中率基准测试
   - 延迟对比（缓存命中 vs. 未命中）
   - 缓存逐出策略验证

**成功标准：**

- ✅ 高 Semantic Cache 命中率（生产工作负载）
- ✅ 低缓存命中延迟
- ✅ 高综合缓存命中率（语义 + KV）

#### 阶段 3：可观测性与监控

**目标：**

- 跨 VSR → Dynamo → 工作节点的整栈分布式追踪
- 全面的指标和仪表板
- 告警和 SLO 监控

**任务：**

1. **分布式追踪 (OpenTelemetry)：**
   - 追踪上下文从 VSR 经由 Dynamo 传播到工作节点
   - Span 层级：
     - 根 span：Envoy Gateway
     - 子 span：Semantic Router (融合路由, 缓存, 安全)
       - 子 span：BERT 分类
       - 子 span：关键词匹配
       - 子 span：相似度搜索
       - 子 span：信号融合与决策
     - 子 span：Dynamo Frontend (路由, 工作节点选择)
     - 子 span：vLLM 工作节点 (Inference 执行)
   - 在请求头中自动注入追踪 ID
   - 支持 Jaeger、Tempo 和其他兼容 OTLP 的后端

2. **指标收集：**
   - Semantic Router 指标：
     - 融合路由性能：
       - BERT 分类延迟和准确性
       - 关键词匹配命中率和延迟
       - 相似度搜索延迟
       - 信号融合决策分布
     - Semantic Cache 命中率 (Milvus)
     - PII/Jailbreak 检测率
     - 按类别划分的模型选择分布
   - Dynamo 指标：
     - KV-aware 路由决策
     - 工作节点利用率
     - KV Cache 命中率
   - 按组件划分的端到端延迟分析

3. **仪表板：**
   - 集成堆栈的 Grafana 仪表板
   - 带有追踪瀑布图的请求流可视化
   - 成本和性能分析
   - 缓存效率指标（语义 + KV）

**成功标准：**

- ✅ 单次分布式追踪跨越所有层 (VSR → Dynamo → 工作节点)
- ✅ 极低的追踪采样开销
- ✅ 实时仪表板投入使用
- ✅ 追踪上下文在服务边界间正确传播

#### 阶段 4：生产加固

**目标：**

- 故障处理与韧性
- 性能优化
- 生产部署

**任务：**

1. **韧性：**
   - Semantic Router 故障时回退到 Dynamo
   - 缓存后端的断路器
   - 优雅降级策略

2. **性能：**
   - 延迟优化（目标：综合延迟 < 50ms）
   - 吞吐量测试（目标：10K RPS）
   - 资源利用率调优

3. **文档：**
   - 部署指南
   - 配置参考
   - 故障排除手册

**成功标准：**

- ✅ 高可用性
- ✅ 低 P99 延迟（路由开销）
- ✅ 10K+ RPS 持续吞吐量

---

## 6. 安全与隐私考量

### 6.1 PII 检测与拦截

**威胁模型：**

- 用户可能会在 Prompt 中无意间包含 PII
- PII 可能会被记录、缓存或发送到第三方模型
- 合规性要求 (GDPR, HIPAA, CCPA)

**缓解措施：**

- 使用 ModernBERT 分类器进行 Token 级 PII 检测
- 每个模型可配置的拦截策略
- PII 类型：PERSON, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, STREET_ADDRESS, IP_ADDRESS, IBAN_CODE, US_DRIVER_LICENSE 等
- 拦截时的响应头：`x-vsr-pii-violation: true`
- 所有 PII 检测的审计日志

**示例配置：**

```yaml
model_config:
  public-model:
    pii_policy:
      allow_by_default: false
      pii_types_allowed: ["PERSON"]  # 仅允许人名
```

### 6.2 Jailbreak 防护 (Prompt Guard)

**威胁模型：**

- 企图绕过安全护栏的对抗性 Prompt
- Prompt 注入攻击
- 社会工程尝试

**缓解措施：**

- 用于 Jailbreak 检测的 **Prompt Guard** 分类
- 基于阈值的拦截（可配置，例如 0.5）
- 基于 ModernBERT 的分类模型
- 带有置信度评分的 Jailbreak 类型检测
- 拦截时的响应头：
  - `x-vsr-jailbreak-blocked: true`
  - `x-vsr-jailbreak-type: {type}` (例如 "prompt_injection")
  - `x-vsr-jailbreak-confidence: {score}` (例如 "0.950")

**示例配置：**

```yaml
prompt_guard:
  enabled: true
  # model_id 从 models 目录自动发现：
  # - 遗留模型：models/jailbreak_classifier_modernbert-base_model
  # - LoRA 模型：models/lora_jailbreak_classifier_bert_model (首选)
  #               models/lora_jailbreak_classifier_roberta_model
  #               models/lora_jailbreak_classifier_modernbert_model
  threshold: 0.5
  use_cpu: false
  use_modernbert: true
  # jailbreak_mapping_path 从模型目录自动发现
```

**注：** Jailbreak 分类器使用自动发现功能在 `models/` 目录中寻找模型。为了获得更好的准确性，系统更倾向于使用 LoRA 模型 (BERT > RoBERTa > ModernBERT) 而非遗留的 ModernBERT 模型。

### 6.3 数据驻留与合规性

**考量因素：**

- Semantic Cache 可能会存储用户查询
- KV Cache 包含模型激活值
- 分布式追踪可能会记录请求内容

**最佳实践：**

1. **缓存加密**：对 Milvus 缓存进行静态加密和传输加密
2. **TTL 策略**：缓存数据的自动过期（默认：2 小时）
3. **数据局部性**：在合规批准的区域进行部署
4. **审计日志**：用于合规审计的全面日志
5. **删除权**：用于从缓存中清除用户数据的 API

---

## 7. 运维考量

### 7.1 监控与告警

**关键指标：**

| 指标 | 阈值 | 告警级别 |
|--------|-----------|----------------|
| Semantic Router 延迟 (P99) | 高 | 警告 |
| Dynamo 路由延迟 (P99) | 高 | 警告 |
| 综合延迟 (P99) | 极高 | 紧急 |
| Semantic Cache 命中率 | 低 | 警告 |
| KV Cache 命中率 | 低 | 警告 |
| 安全拦截率 | 高 | 警告 |
| 错误率 | 高 | 紧急 |
| GPU 利用率 | 过低或过高 | 警告 |

**仪表板：**

1. **请求流仪表板**：可视化请求流经各层的过程
2. **缓存性能仪表板**：命中率、延迟、逐出率
3. **安全仪表板**：PII 检测、Jailbreak 拦截、审计日志
4. **成本仪表板**：Token 使用情况、模型选择、单次查询成本

### 7.3 故障模式与恢复

**故障场景 1：Semantic Router 不可用**

- **检测**：健康检查失败、超时错误
- **影响**：无语义路由、安全过滤或缓存
- **恢复**：
  - Envoy Gateway 绕过 ExtProc（回退模式）
  - 请求直接转发到 Dynamo
  - Dynamo 执行默认路由
- **缓解**：部署 3 个以上副本并设置反亲和性

**故障场景 2：Milvus 缓存不可用**

- **检测**：连接错误、超时
- **影响**：无 Semantic Cache（缓存未命中）
- **恢复**：
  - Semantic Router 继续使用内存缓存
  - 所有请求转发到 Dynamo
  - 性能下降但无停机
- **缓解**：部署 Milvus 集群以实现高可用

**故障场景 3：Dynamo Frontend 不可用**

- **检测**：HTTP 503 错误、连接被拒绝
- **影响**：无法进行 Inference
- **恢复**：
  - Envoy Gateway 向客户端返回 503
  - Kubernetes 重启失败的 pod
  - 负载均衡器路由到健康的副本
- **缓解**：部署 2 个以上副本并设置就绪探针

**故障场景 4：工作节点池耗尽**

- **检测**：队列深度告警、高延迟
- **影响**：TTFT 和 ITL 增加
- **恢复**：
  - Dynamo Planner 自动扩缩工作节点
  - Semantic Router 可能会路由到备选模型
  - 请求排队直到容量可用
- **缓解**：自动扩缩容策略、过度配置

---

## 8. 未来增强

### 8.1 高级路由策略

**多目标优化：**

- 在路由决策中结合语义质量、延迟和成本
- 帕累托最优模型选择
- 用户指定的 SLO 偏好（快速 vs. 准确 vs. 廉价）

**自适应路由：**

- 从用户反馈中学习（点赞/点踩）
- 模型选择的 A/B 测试
- 用于路由策略的强化学习

### 8.2 跨层优化

**语义感知的 KV Cache 管理：**

- 为高价值类别优先保留 KV Cache
- 在 KV Cache 逐出决策中使用语义相似度
- 为相似查询实现跨请求的 KV Cache 共享

**预测性预取：**

- 预测对话中的下一个查询
- 为可能的后续请求预热 KV Cache
- 用于低延迟响应的推测性执行

### 8.3 多租户支持

**租户隔离：**

- 每个租户独立的 Semantic Cache 命名空间
- 每个租户的模型访问策略
- 每个租户的成本追踪和配额

**租户特定路由：**

- 为每个租户定制模型池
- 租户特定的安全策略
- 租户特定的 SLO

---

## 9. 参考文献

### 9.1 NVIDIA Dynamo 文档

- [Dynamo 架构概览](https://docs.nvidia.com/dynamo/latest/_sections/architecture.html)
- [Dynamo KV 路由](https://docs.nvidia.com/dynamo/latest/components/router/README.html)
- [Dynamo 分离式服务](https://docs.nvidia.com/dynamo/latest/_sections/disaggregated-serving.html)
- [Dynamo KVBM](https://docs.nvidia.com/dynamo/latest/components/kvbm/README.html)

### 9.2 vLLM Semantic Router 文档

- [Semantic Router 概览](https://vllm-semantic-router.com/docs/overview/semantic-router-overview/)
- [系统架构](https://vllm-semantic-router.com/docs/overview/architecture/system-architecture/)
- [Kubernetes 部署](https://vllm-semantic-router.com/docs/installation/k8s/ai-gateway)
- [分布式追踪支持 (PR #322)](https://github.com/vllm-project/semantic-router/pull/322)
- [基于 Milvus 的 Semantic Cache](https://vllm-semantic-router.com/docs/features/semantic-caching/)

### 9.3 相关研究

- **DistServe**：Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- **Mooncake**：KVCache-centric Disaggregated Architecture for LLM Serving
- **RouteLLM**：Learning to Route LLMs with Preference Data
- **DeepSeek-V3**：Technical Report on Mixture-of-Experts Architecture

### 9.4 集成提案

- [vLLM 生产堆栈集成 (#295)](https://github.com/vllm-project/semantic-router/issues/295)
- [Prompt 分类路由提案](https://vllm-semantic-router.com/docs/proposals/prompt-classification-routing/)

---

## 10. 附录

### 10.1 术语表

| 术语 | 定义 |
|------|------------|
| **BERT** | Bidirectional Encoder Representations from Transformers |
| **ExtProc** | Envoy 外部处理器 (用于请求处理的 gRPC 服务) |
| **融合路由** | 结合 BERT 分类、关键词匹配和相似度搜索的多信号路由 |
| **ITL** | Inter-Token Latency (生成的 Token 之间的时间间隔) |
| **KV Cache** | 存储 Transformer 注意力状态的键值缓存 |
| **KVBM** | KV 块管理器 (用于缓存管理的 Dynamo 组件) |
| **Milvus** | 用于 Semantic Cache 和相似度搜索的开源向量数据库 |
| **MoE** | Mixture-of-Experts (具有专业专家网络的用户模型架构) |
| **MoM** | Mixture-of-Models (根据任务路由到不同的模型) |
| **NIXL** | NVIDIA Inference Transfer Library |
| **OTLP** | OpenTelemetry Protocol (用于分布式追踪和指标) |
| **PII** | 个人身份信息 |
| **Prompt Guard** | 使用分类模型识别对抗性 Prompt 的 Jailbreak 检测系统 |
| **TTFT** | Time To First Token (生成第一个 Token 前的延迟) |

### 10.2 System Prompt 示例

**关键类别的领域感知 System Prompt：**

集成利用了 **14 个专业的 System Prompt**，这些 Prompt 根据查询分类自动注入。以下是代表性示例：

**1. 数学类别（重推理）**

```
You are a mathematics expert. Provide step-by-step solutions, show your
work clearly, and explain mathematical concepts in an understandable way.
```

- **用途**：鼓励结构化推理和清晰解释
- **模型**：DeepSeek-V3 (得分: 1.0, reasoning: 已启用)
- **MoE 影响**：激活数学推理专家

**2. 计算机科学类别（重代码）**

```
You are a computer science expert with knowledge of algorithms, data structures,
programming languages, and software engineering. Provide clear, practical solutions
with code examples when helpful.
```

- **用途**：在理论与实际代码示例之间取得平衡
- **模型**：Qwen3 (得分: 0.89, reasoning: 已禁用)
- **MoE 影响**：激活编程和算法专家

**3. 商业类别（行动导向）**

```
You are a senior business consultant and strategic advisor with expertise in
corporate strategy, operations management, financial analysis, marketing, and
organizational development. Provide practical, actionable business advice backed
by proven methodologies and industry best practices. Consider market dynamics,
competitive landscape, and stakeholder interests in your recommendations.
```

- **用途**：强调可操作的建议和商业背景
- **模型**：Phi-4 (得分: 0.88, reasoning: 已禁用)
- **MoE 影响**：激活商业策略和分析专家

**4. 法律类别（感知免责声明）**

```
You are a knowledgeable legal expert with comprehensive understanding of legal
principles, case law, statutory interpretation, and legal procedures. Provide
accurate legal information while clearly stating that your responses are for
informational purposes only and do not constitute legal advice.
```

- **用途**：在维持伦理界限的同时确保准确性
- **模型**：Phi-4 (得分: 0.75, reasoning: 已禁用)
- **MoE 影响**：通过适当的免责声明激活法律推理专家

**5. 健康类别（基于证据）**

```
You are a health and medical information expert with knowledge of anatomy,
physiology, diseases, treatments, preventive care, nutrition, and wellness.
Provide accurate, evidence-based health information while emphasizing that
your responses are for educational purposes only and do not replace professional
medical advice.
```

- **用途**：平衡信息量与医疗伦理
- **模型**：Phi-4 (得分: 0.76, reasoning: 已禁用)
- **MoE 影响**：通过安全护栏激活医学知识专家

**完整类别列表：**

- 数学、计算机科学、物理、化学、生物、工程
- 经济、商业、法律、心理学、哲学、历史、健康、其他

**System Prompt 收益：**

- **CoT 优化**：领域特定的推理模式提高输出质量
- **Token 效率**：集中的 Prompt 减少不必要的冗余（Token 减少 10-15%）
- **MoE 专家匹配**：专业术语激活相关专家（专家选择准确性提高 20-30%）
- **质量控制**：类别特定的免责声明确保符合伦理要求

### 10.3 API 示例

**带有 Semantic Router 请求头的请求：**

```bash
curl -X POST http://llm-gateway:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "证明 2 的平方根是无理数"
      }
    ]
  }'
```

**带有路由请求头的响应：**

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-selected-model: deepseek-v31
x-vsr-selected-category: math
x-vsr-selected-reasoning: on
x-vsr-injected-system-prompt: true
x-request-id: 7f3e9a2b4c5d6e8f

{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "deepseek-v31",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "为了证明 √2 是无理数，我们将使用反证法..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 250,
    "total_tokens": 265
  }
}
```

---

## 结论

本提案概述了 vLLM Semantic Router 与 NVIDIA Dynamo 之间的全面集成策略，将语义智能与基础设施优化相结合。分层架构确保了：

1. **语义正确性**：基于查询理解选择正确的模型
2. **基础设施效率**：最优的工作节点选择和 KV Cache 利用
3. **安全**：在 Inference 之前进行 PII 检测和 Jailbreak 防护
4. **性能**：双层缓存使延迟降低 40-60%
5. **成本优化**：通过智能路由降低 55% 的成本

该集成被设计为**非侵入性**、**模块化**且**生产就绪**的，具有清晰的实施阶段、全面的监控和鲁棒的故障处理机制。

**后续步骤：**

1. 审查并批准提案
2. 开始阶段 1 实施（基础）
3. 建立基准测试环境
4. 根据性能结果进行迭代
