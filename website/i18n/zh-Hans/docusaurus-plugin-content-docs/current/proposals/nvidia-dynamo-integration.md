---
translation:
  source_commit: "d60ca63"
  source_file: "docs/proposals/nvidia-dynamo-integration.md"
  outdated: false
---

# Semantic Intelligence Layer for NVIDIA Dynamo

## 1. Executive Summary

This proposal outlines a comprehensive integration strategy between **vLLM Semantic Router** and **NVIDIA Dynamo**, combining semantic intelligence with high-performance distributed inference. The integration creates a unified inference stack that leverages:

- **Semantic Router's** intelligent request classification (14 domain categories), domain-aware system prompts, fusion routing (BERT classification + keyword matching + similarity search), security filtering, Milvus-based semantic caching
- **Dynamo's** disaggregated serving, KV-aware routing, and multi-tier memory management

The result is a production-grade LLM serving platform with **system-level intelligence** that achieves optimal balance between **accuracy** (routing to the right model with optimized prompts for best quality) and **efficiency** (maximizing GPU utilization and minimizing latency), creating a holistically intelligent inference system.

**Key Benefits:**

- **System-level intelligence** that optimally balances accuracy and efficiency across the entire inference stack
- **Significant cost reduction** through intelligent model selection combined with infrastructure optimization
- **Substantial latency improvement** via semantic caching + KV cache management with adaptive routing strategies
- **Enhanced LLM quality** with domain-aware system prompts that improve Chain-of-Thought reasoning, token efficiency, and MoE expert matching
- **Adaptive routing intelligence** with fusion routing: fast path (keyword) to deep analysis (BERT) based on query complexity, maximizing efficiency without sacrificing accuracy
- **Multi-signal decision making** combining BERT classification, keyword matching, and similarity search for robust and accurate routing
- **Holistic content safety** with PII detection and jailbreak prevention before inference
- **End-to-end observability** across semantic and infrastructure layers for continuous system optimization

---

## 2. Motivation: Why Semantic Router for Dynamo?

### 2.1 Dynamo Router Capabilities (Current State)

NVIDIA Dynamo provides a sophisticated **KV-aware router** optimized for infrastructure-level efficiency:

| Capability | Description | Optimization Target |
|------------|-------------|---------------------|
| **KV Cache-Aware Routing** | Routes requests to workers with highest KV cache hit rate | TTFT, throughput |
| **Load-Based Routing** | Balances active decoding blocks across workers | ITL, GPU utilization |
| **Cost Function Optimization** | Minimizes `potential_prefill_blocks + potential_active_blocks` | Computational cost |
| **Temperature-Based Selection** | Probabilistic routing to prevent worker saturation | Load distribution |
| **Event-Driven Tracking** | Real-time cache state via worker events | Routing accuracy |

**Key Characteristics:**

- **Infrastructure-focused:** Optimizes GPU memory and compute utilization
- **Cache-aware:** Leverages existing KV caches to reduce prefill cost
- **Load-balanced:** Distributes decoding workload across workers
- **Performance-oriented:** Minimizes TTFT and ITL through smart scheduling

### 2.2 Semantic Router Capabilities (System Intelligence Layer)

vLLM Semantic Router provides **system-level intelligence** that operates at the request understanding layer, achieving optimal balance between **accuracy** and **efficiency** through intelligent decision-making across **14 domain categories**:

| Capability | Description | Intelligence Focus |
|------------|-------------|---------------------|
| **Intent Classification** | BERT-based categorization (14 categories: math, code, business, law, etc.) | Accuracy: Precise domain understanding |
| **Model Selection** | Routes to best-performing model per category | Accuracy: Task-specific quality optimization |
| **Domain-Aware System Prompts** | Auto-injects category-specific system prompts for prompt engineering | Accuracy: LLM CoT quality, token efficiency, MoE expert matching |
| **Fusion Routing** | Multi-signal routing (keyword + similarity + BERT) | Efficiency: Adaptive latency based on query complexity |
| **Semantic Caching** | Milvus-based vector cache with 0.85+ similarity threshold | Efficiency: Inference cost reduction |
| **PII Detection** | Token-level classification (PERSON, EMAIL, SSN, etc.) | System Intelligence: Privacy protection |
| **Jailbreak Prevention** | Binary classification for prompt injection attacks | System Intelligence: Security enforcement |
| **Tool Selection** | Semantic matching of relevant tools to reduce prompt tokens | Efficiency: Context optimization |
| **Reasoning Control** | Auto-enables reasoning mode for complex queries | Accuracy: Quality-aware mode selection |

**System Intelligence Characteristics:**

- **Holistic Intelligence:** Understands query intent, complexity, and security implications across 14 domain categories
- **Accuracy-Efficiency Balance:** Dynamically selects routing strategy (keyword/similarity/BERT) based on query complexity to maximize accuracy while minimizing latency
- **Quality Optimization:** Selects models and prompts based on task-specific accuracy requirements
- **Intelligent Prompt Engineering:** Auto-injects domain-specific system prompts to optimize LLM behavior and output quality
- **Proactive Security:** Blocks malicious or privacy-violating requests before reaching inference layer
- **Cost Intelligence:** Avoids expensive models for simple queries while ensuring quality for complex tasks
- **Adaptive Routing:** Multi-signal fusion routing adapts to query characteristics for optimal accuracy-efficiency tradeoff

#### 2.2.1 14 Domain Categories with System Prompts

Semantic Router classifies queries into **14 specialized categories**: math, computer science, physics, chemistry, biology, engineering, economics, business, law, psychology, philosophy, history, health, and other. Each category has an optimized system prompt automatically injected based on query classification.

**System Prompt Benefits:**

1. **Improved Chain-of-Thought (CoT):** Domain-specific prompts guide LLMs to use appropriate reasoning patterns
   - Math: "Provide step-by-step solutions, show your work clearly"
   - Law: "Provide accurate legal information while clearly stating disclaimers"
   - Business: "Provide practical, actionable advice backed by proven methodologies"

2. **Token Efficiency:** Optimized prompts reduce unnecessary verbosity while maintaining quality
   - Shorter, focused prompts for straightforward categories (business, history)
   - Detailed prompts for complex domains requiring specific methodologies (math, physics)

3. **MoE Expert Matching:** Well-crafted system prompts improve expert selection in Mixture-of-Experts models
   - Domain-specific terminology activates relevant experts
   - Consistent prompt structure improves expert routing accuracy
   - Example: "You are a mathematics expert" → activates math-specialized experts in DeepSeek-V3

4. **Quality Control:** Category-specific disclaimers and ethical guidelines
   - Medical/Legal: Explicit disclaimers about professional consultation
   - Psychology: Emphasis on evidence-based approaches
   - Health: Clear boundaries between information and medical advice

**Example System Prompt (Math Category):**

```
You are a mathematics expert. Provide step-by-step solutions, show your
work clearly, and explain mathematical concepts in an understandable way.
```

**Example System Prompt (Business Category):**

```
You are a senior business consultant and strategic advisor with expertise
in corporate strategy, operations management, financial analysis, marketing,
and organizational development. Provide practical, actionable business advice
backed by proven methodologies and industry best practices. Consider market
dynamics, competitive landscape, and stakeholder interests in your recommendations.
```

#### 2.2.2 Fusion Routing Strategy

Semantic Router implements a **multi-signal fusion routing** approach that combines three complementary routing methods (as detailed in the [Prompt Classification Routing proposal](./prompt-classification-routing.md)):

**1. Keyword-Based Routing (Fast Path)**

- Deterministic routing for technology-specific terms (e.g., "kubernetes", "SQL", "React")
- **Latency**: Minimal (significantly faster than BERT classification)
- Boolean logic support (AND/OR operators)
- Easy to update without model retraining
- **Use case**: Exact term matching for known patterns

**2. Similarity-Based Routing (Semantic Path)**

- Embedding similarity for semantic concept detection
- Robust to paraphrasing ("step-by-step" ≈ "explain thoroughly")
- Configurable similarity thresholds (default: 0.75)
- **Latency**: Low (faster than full BERT classification)
- **Use case**: Semantic concept matching beyond exact terms

**3. BERT Classification (Deep Understanding Path)**

- 14-category classification with ModernBERT
- Highest accuracy for complex queries
- **Latency**: Moderate (comprehensive analysis)
- **Use case**: Comprehensive intent understanding

**Signal Fusion Layer:**

- **Policy-driven decision making**: Combines signals with configurable priority
- **Routing logic**:
  1. Check keyword rules first (fastest)
  2. If no keyword match, check similarity rules
  3. If no similarity match, use BERT classification (fallback)
- **Confidence scoring**: Each signal provides confidence score
- **Override mechanism**: High-confidence signals can override lower-priority signals
- **Observability**: All signals logged for analysis

**System Intelligence Benefits of Fusion Routing:**

- **Accuracy-Efficiency Balance**: Dynamically selects routing strategy based on query complexity—fast path (keyword) for deterministic patterns achieves minimal latency, while deep analysis (BERT) for complex queries ensures maximum accuracy
- **Adaptive Intelligence**: System automatically chooses the most efficient signal that meets accuracy requirements, avoiding unnecessary computation
- **Flexibility**: Easy to add new routing rules without model retraining, enabling continuous system optimization
- **Robustness**: Multiple signals provide redundancy and cross-validation, reducing misclassification risk and improving overall system reliability
- **Holistic Optimization**: Considers both accuracy and efficiency in every routing decision, maximizing system-level intelligence

### 2.3 Differentiation Analysis: Complementary Strengths

The two systems operate at **different layers** of the inference stack with **minimal overlap**:

#### Semantic Router: Request Intelligence Layer

```
User Query → [Semantic Understanding] → Model Selection → Request Enrichment
```

- **What:** Understands query semantics, intent, and safety
- **Why:** Routes to the right model for the task
- **When:** Before request reaches infrastructure
- **Optimization:** Accuracy, cost, security

#### Dynamo Router: Infrastructure Efficiency Layer

```
Enriched Request → [Worker Selection] → KV Cache Optimization → GPU Scheduling
```

- **What:** Optimizes worker selection and resource allocation
- **Why:** Maximizes GPU utilization and minimizes latency
- **When:** After model selection, during execution
- **Optimization:** TTFT, ITL, throughput

#### Integration Value Proposition

| Dimension | Semantic Router Alone | Dynamo Router Alone | **Integrated System** |
|-----------|----------------------|---------------------|----------------------|
| **Model Selection** | ✅ Semantic accuracy (14 categories) | ❌ No model awareness | ✅ Best model for task |
| **Worker Selection** | ❌ No worker awareness | ✅ KV cache optimization | ✅ Optimal worker for model |
| **Prompt Engineering** | ✅ Domain-aware system prompts | ❌ No prompt optimization | ✅ Optimized CoT & MoE matching |
| **Fusion Routing** | ✅ BERT + keyword + similarity fusion | ❌ KV-aware only | ✅ Multi-signal intelligent routing |
| **Caching** | ✅ Semantic similarity (Milvus) | ✅ KV cache reuse | ✅✅ **Dual-layer caching** |
| **Security** | ✅ PII + jailbreak | ❌ No security layer | ✅ Pre-inference filtering |
| **Cost Optimization** | ✅ Cross-Model-level | ✅ Infrastructure-level | ✅✅ **End-to-end optimization** |
| **Latency** | Adaptive (fusion routing) | Low routing overhead | **Parallel execution** |

**Concrete Example:**

```
Query: "Explain the proof of Fermat's Last Theorem step-by-step"

┌─────────────────────────────────────────────────────────────────┐
│ Semantic Router Layer                                           │
├─────────────────────────────────────────────────────────────────┤
│ 1. Fusion Routing (3-signal analysis):                          │
│    a) Keyword Match: "theorem", "proof" → math (confidence: 0.8)│
│    b) Similarity Search: matches "mathematical proofs" concept  │
│       (similarity: 0.87)                                         │
│    c) BERT Classification: "math" category (confidence: 0.92)   │
│    → Final Decision: "math" (multi-signal consensus)            │
│ 2. Model Selection: deepseek-v31 (best for math reasoning)      │
│ 3. System Prompt Injection:                                     │
│    "You are a mathematics expert. Provide step-by-step          │
│     solutions, show your work clearly, and explain              │
│     mathematical concepts in an understandable way."            │
│ 4. Reasoning Mode: ENABLED (entropy-based decision)             │
│ 5. Security: PASS (no PII, no jailbreak)                        │
│ 6. Semantic Cache: MISS (novel query)                           │
│ 7. Enriched Request:                                            │
│    - model=deepseek-v31                                         │
│    - reasoning_effort=high                                      │
│    - system_prompt=<math expert prompt>                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Dynamo Router Layer                                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Worker Pool: [worker-1, worker-2, worker-3] (deepseek-v31)  │
│ 2. KV Cache Analysis:                                           │
│    - worker-1: 15 cached blocks (math proofs context)           │
│    - worker-2: 3 cached blocks                                  │
│    - worker-3: 0 cached blocks                                  │
│ 3. Cost Calculation:                                            │
│    - worker-1: 85 prefill + 25 active = 110 (BEST)             │
│    - worker-2: 97 prefill + 20 active = 117                     │
│    - worker-3: 100 prefill + 18 active = 118                    │
│ 4. Selection: worker-1 (significant prefill cost reduction)     │
└─────────────────────────────────────────────────────────────────┘

Result: 
- Right model (deepseek-v31 for math reasoning)
- Right worker (worker-1 with relevant KV cache)
- Right mode (reasoning enabled)
- Significantly faster TTFT vs. random worker selection
```

### 2.4 Why Integration Matters: Achieving System-Level Intelligence

**Challenge 1: Infrastructure without Intelligence**

- Dynamo optimizes infrastructure efficiency but lacks semantic understanding
- Cannot distinguish between "2+2=?" and "Prove Fermat's Last Theorem"
- Routes both to the same model pool without understanding complexity or quality requirements
- No ability to select specialized models (math vs. code vs. creative) based on task characteristics

**Challenge 2: Intelligence without Infrastructure Awareness**

- Semantic Router provides intelligent model selection but lacks infrastructure visibility
- Selects the right model but not the optimal worker
- Cannot leverage KV cache reuse across workers
- No awareness of GPU utilization or worker load for efficiency optimization

**Solution: Holistic System Intelligence through Layered Integration**

```
System Intelligence Layer (Semantic Router)
    ↓ [accuracy: model selection, quality optimization, security]
    ↓ [efficiency: semantic cache, adaptive routing, cost control]
Infrastructure Optimization Layer (Dynamo)
    ↓ [efficiency: worker selection, KV cache, GPU scheduling]
    ↓ [accuracy: consistent execution, reliable serving]
Execution Layer (vLLM/SGLang/TRT-LLM)
```

**Result:** A holistically intelligent system that optimizes for both accuracy (right model, right prompt, right quality) and efficiency (right worker, right cache, right resource utilization) at every layer.

---

## 3. Goals and Non-Goals

### 3.1 Goals

**Primary Goals:**

1. **Seamless Integration:** Semantic Router operates as a pre-processing layer before Dynamo's router
2. **Dual-Layer Caching:** Semantic cache (request-level) + KV cache (token-level) work in tandem
3. **Model-Aware Routing:** Dynamo routes to worker pools filtered by Semantic Router's model selection
4. **Security Enforcement:** PII and jailbreak detection before requests reach Dynamo
5. **Unified Observability:** Single trace spans both semantic and infrastructure layers
6. **Zero Downtime:** Hot-reload of semantic routing rules without Dynamo restart

**Secondary Goals:**

1. **Performance:** Combined latency < 50ms (semantic + infrastructure routing)
2. **Scalability:** Support 10K+ RPS with horizontal scaling
3. **Flexibility:** Support multiple deployment patterns (sidecar, gateway, embedded)

### 3.2 Non-Goals

1. **Replacing Dynamo Router:** Semantic Router augments, not replaces, Dynamo's KV-aware routing
2. **Modifying Dynamo Core:** Integration via standard APIs, no Dynamo internals changes required
3. **Unified Configuration:** Maintain separate configs for semantic and infrastructure layers
4. **Synchronous Coupling:** Systems can operate independently if needed

---

## 4. Proposal Details

### 4.1 Deep Learning Models

The Semantic Router leverages **four specialized deep learning models** for intelligent request processing. The system uses a combination of **BERT** and **ModernBERT** architectures optimized for different tasks.

#### 4.1.1 Similarity Model (BERT Embeddings)

**Purpose:** Generate embeddings for semantic similarity comparison

**Model:** `sentence-transformers/all-MiniLM-L12-v2`

**Key Features:**

- **Architecture:** BERT-based (microsoft/MiniLM-L12-H384-uncased)
  - 12 layers, 384 hidden dimensions, 12 attention heads
  - Fine-tuned on 1B+ sentence pairs using contrastive learning
  - Base model: Standard BERT architecture (not ModernBERT)
- **Embedding Dimension:** 384
- **Use Cases:**
  - Semantic cache similarity matching (threshold: 0.8)
  - Tool selection via semantic search (threshold: 0.2)
  - Similarity-based routing for semantic concepts
- **Deployment:** CPU-optimized for cost efficiency
- **Model Size:** 33.4M parameters (~120 MB)

**Configuration:**

```yaml
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true
```

**Why BERT (not ModernBERT)?**

- Mature, well-tested model with proven performance
- Optimized for sentence embeddings via contrastive learning
- Smaller model size (120 MB) for faster loading
- ModernBERT (released Dec 2024) is used for classification tasks below

---

#### 4.1.2 Classification Model (Category Detection)

**Purpose:** Classify queries into 14 domain categories

**Model:** `models/category_classifier_modernbert-base_model`

**Key Features:**

- **Architecture:** ModernBERT-base (released Dec 2024)
  - Modern replacement for BERT with improved architecture
  - 8192 token context length (vs. BERT's 512)
  - Rotary Position Embeddings (RoPE) for better long-context handling
  - Flash Attention 2 for faster inference
  - Fine-tuned on MMLU-Pro dataset for domain classification
- **Categories:** 14 domains (math, computer_science, physics, chemistry, biology, engineering, economics, business, law, psychology, philosophy, history, health, other)
- **Output:** Category label + confidence score
- **Threshold:** 0.6 (configurable)
- **Training Data:** MMLU-Pro dataset with domain-specific examples
- **Model Size:** ~149M parameters (ModernBERT-base)

**Configuration:**

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"
```

**Model Selection Impact:**

- Determines which LLM to route to (e.g., DeepSeek-V3 for math, Qwen3 for business)
- Triggers domain-specific system prompt injection
- Controls reasoning mode activation

---

#### 4.1.3 PII Detection Model (Privacy Protection)

**Purpose:** Detect personally identifiable information at token level

**Model:** `models/pii_classifier_modernbert-base_presidio_token_model`

**Key Features:**

- **Architecture:** ModernBERT-base fine-tuned for token classification
  - Token-level sequence labeling (BIO tagging scheme)
  - Fine-tuned on Microsoft Presidio dataset
  - Optimized for privacy-sensitive entity detection
- **PII Types Detected:** 17 types including:
  - **Identity:** `PERSON`, `AGE`, `NRP` (nationality/religious/political)
  - **Contact:** `EMAIL_ADDRESS`, `PHONE_NUMBER`, `STREET_ADDRESS`, `ZIP_CODE`
  - **Financial:** `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `US_DRIVER_LICENSE`
  - **Technical:** `IP_ADDRESS`, `DOMAIN_NAME`
  - **Organizational:** `ORGANIZATION`, `GPE` (geopolitical entity)
  - **Temporal:** `DATE_TIME`
- **Granularity:** Token-level classification (not just entity-level)
- **Threshold:** 0.7 (configurable)
- **Action:** Block requests violating model-specific PII policies
- **Model Size:** ~149M parameters (ModernBERT-base)

**Configuration:**

```yaml
classifier:
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7
    use_cpu: true
    pii_mapping_path: "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"
```

**Policy Enforcement:**

```yaml
model_config:
  public-model:
    pii_policy:
      allow_by_default: false
      pii_types_allowed: ["PERSON"]  # Only person names allowed
```

**Response Headers (when blocked):**

- `x-vsr-pii-violation: true`

---

#### 4.1.4 Jailbreak Detection Model (Security)

**Purpose:** Detect adversarial prompts and jailbreak attempts

**Model:** Auto-discovered from `models/` directory

**Key Features:**

- **Architecture:** Multiple options with automatic selection
  - **LoRA models (preferred):** Fine-tuned adapters on BERT/RoBERTa/ModernBERT base
    - `lora_jailbreak_classifier_bert_model` (Priority 1)
    - `lora_jailbreak_classifier_roberta_model` (Priority 2)
    - `lora_jailbreak_classifier_modernbert_model` (Priority 3)
  - **Legacy model (fallback):** `jailbreak_classifier_modernbert-base_model`
  - LoRA models offer better accuracy with smaller size (~10-20 MB adapters)
- **Model Discovery:** Automatic selection with architecture priority: BERT > RoBERTa > ModernBERT
- **Detection Types:**
  - Prompt injection attacks
  - Instruction override attempts
  - Adversarial prompts
  - Social engineering
- **Threshold:** 0.7 (configurable)
- **Action:** Block requests with confidence above threshold
- **Model Size:**
  - LoRA: ~10-20 MB (adapter only) + base model
  - Legacy: ~149M parameters (ModernBERT-base)

**Configuration:**

```yaml
prompt_guard:
  enabled: true
  use_modernbert: true
  threshold: 0.7
  use_cpu: true
  # model_id and jailbreak_mapping_path are auto-discovered
```

**Response Headers (when blocked):**

- `x-vsr-jailbreak-blocked: true`
- `x-vsr-jailbreak-type: {type}` (e.g., "prompt_injection")
- `x-vsr-jailbreak-confidence: {score}` (e.g., "0.950")

---

#### 4.1.5 Model Performance Summary

| Model | Purpose | Architecture | Parameters | Threshold | CPU/GPU |
|-------|---------|--------------|------------|-----------|---------|
| **Similarity** | Semantic matching | BERT (MiniLM-L12) | 33.4M | 0.6-0.8 | CPU |
| **Classification** | Category detection | ModernBERT-base | 149M | 0.6 | CPU |
| **PII Detection** | Privacy protection | ModernBERT-base | 149M | 0.7 | CPU |
| **Jailbreak** | Security filtering | ModernBERT-base/LoRA | 149M + adapters | 0.7 | CPU |

**Architecture Comparison:**

| Feature | BERT (MiniLM) | ModernBERT |
|---------|---------------|------------|
| **Release Date** | 2020 | December 2024 |
| **Context Length** | 512 tokens | 8192 tokens |
| **Position Encoding** | Absolute | RoPE (Rotary) |
| **Attention** | Standard | Flash Attention 2 |
| **Use Case** | Embeddings | Classification |
| **Model Size** | 33.4M params | 149M params |

**Optimization Strategies:**

- **Parallel Execution:** PII and Jailbreak detection run in parallel
- **Early Exit:** Cache hits bypass all model inference
- **Keyword Routing:** Fast path for deterministic patterns
- **CPU Optimization:** All models optimized for CPU inference to reduce cost
- **LoRA Adapters:** Jailbreak model uses lightweight adapters for faster loading

---

### 4.2 Design Principles

1. **Separation of Concerns:** Semantic intelligence and infrastructure optimization remain decoupled
2. **API-Driven Integration:** Use Dynamo's frontend API and worker registration mechanisms
3. **Fail-Safe Design:** Semantic Router failure falls back to Dynamo's default routing
4. **Observability-First:** Every decision (semantic + infrastructure) is traced and logged
5. **Kubernetes-Native:** Designed for cloud-native deployment with CRDs and operators

### 4.3 System Architecture

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="System Architecture Overview" defaultZoom={10.5}>

{`graph TB
    Client[LLM Application<br/>OpenAI SDK]

    subgraph Main["Main Processing Flow"]
        direction TB

        subgraph SIL["① vLLM Semantic Router Layer"]
            direction TB
            Gateway[Envoy Gateway :8080]
            ExtProc[Semantic Router ExtProc :50051]

            subgraph SC["Semantic Components"]
                direction LR
                Classifier[BERT Classifier]
                PIIDetector[PII Detector]
                JailbreakGuard[Jailbreak Guard]
            end

            SemanticCache[Semantic Cache]
            ToolSelector[Tool Selector]
        end

        subgraph DL["② NVIDIA Dynamo Layer"]
            direction TB
            DynamoFrontend[Dynamo Frontend :8000]

            subgraph DR["Routing & Management"]
                direction LR
                DynamoRouter[KV Router]
                KVBM[KV Block Manager]
            end

            Planner[Planner - Dynamic Scaling]
        end

        subgraph EL["③ Execution Layer - Worker Pools"]
            direction TB

            subgraph MP1["Model Pool: deepseek-v31"]
                direction LR
                W1[Prefill Worker]
                W2[Decode Worker]
            end

            subgraph MP2["Model Pool: phi4"]
                direction LR
                W3[Prefill Worker]
                W4[Decode Worker]
            end

            subgraph MP3["Model Pool: qwen3"]
                W5[Worker - SGLang]
            end
        end
    end

    subgraph SL["Storage Layer"]
        direction TB
        Milvus[(Milvus<br/>Semantic Cache)]
        SystemMem[(System Memory<br/>KV Offload)]
        NVMe[(NVMe<br/>Cold Cache)]
    end

    Client -->|1. Request| Gateway
    Gateway <-->|2. ExtProc| ExtProc
    ExtProc --> Classifier
    ExtProc --> PIIDetector
    ExtProc --> JailbreakGuard
    ExtProc --> SemanticCache
    ExtProc --> ToolSelector

    Gateway -->|3. Enriched Request| DynamoFrontend
    DynamoFrontend --> DynamoRouter
    DynamoRouter <--> KVBM

    DynamoRouter -->|4. Worker Selection| W1
    DynamoRouter -->|4. Worker Selection| W2
    DynamoRouter -.-> W3
    DynamoRouter -.-> W4
    DynamoRouter -.-> W5

    Planner -.->|Scaling| W1
    Planner -.->|Scaling| W2
    Planner -.->|Scaling| W3
    Planner -.->|Scaling| W4
    Planner -.->|Scaling| W5

    SemanticCache <--> Milvus
    KVBM <--> SystemMem
    KVBM <--> NVMe

    W1 -->|5. Response| DynamoFrontend
    DynamoFrontend -->|6. Response| Gateway
    Gateway -->|7. Response| Client

    style ExtProc fill:#e1f5ff
    style DynamoRouter fill:#c8e6c9
    style SemanticCache fill:#fff9c4
    style KVBM fill:#fff9c4
    style SL fill:#f5f5f5`}
</ZoomableMermaid>

**Architecture Layers:**

1. **Semantic Intelligence Layer (Semantic Router)**
   - Envoy Gateway with ExtProc for request interception
   - BERT-based classification and security filtering
   - Semantic caching with Milvus backend
   - Request enrichment with routing metadata

2. **Infrastructure Optimization Layer (Dynamo)**
   - Dynamo Frontend receives enriched requests
   - KV Router performs model-aware worker selection
   - Planner handles dynamic scaling
   - KVBM manages multi-tier KV cache

3. **Execution Layer (vLLM/SGLang/TRT-LLM)**
   - Model-specific worker pools
   - Disaggregated prefill/decode workers
   - Backend-agnostic execution

4. **Storage Layer**
   - Milvus for semantic cache
   - System memory for KV cache offload
   - NVMe for cold KV cache storage

### 4.4 Request Flow

#### 4.4.1 End-to-End Request Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Semantic Intelligence (Semantic Router)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 1: Request Interception                                                │
│   - Envoy Gateway receives OpenAI API request                               │
│   - ExtProc gRPC call to Semantic Router                                    │
│   - Extract query from messages array                                       │
│                                                                              │
│ Step 2: Security Filtering (Parallel Execution)                             │
│   - PII Detection: Scan for PERSON, EMAIL, SSN, etc.                        │
│   - Jailbreak Detection: Binary classification for prompt injection         │
│   - Action: BLOCK if security violation detected                            │
│   - Latency: Low                                                            │
│                                                                              │
│ Step 3: Semantic Cache Lookup                                               │
│   - Generate BERT embedding for query                                       │
│   - Search Milvus for similar queries (threshold: 0.85)                     │
│   - Action: Return cached response if HIT                                   │
│   - Latency: Very low (cache hit), Low (cache miss)                         │
│                                                                              │
│ Step 4: Fusion Routing (Multi-Signal Classification)                        │
│   - Signal 1: Keyword matching (fast path)                                  │
│   - Signal 2: Similarity search (semantic concepts)                         │
│   - Signal 3: BERT classification (deep understanding)                      │
│   - Entropy-based reasoning decision                                        │
│   - Category: math, code, reasoning, creative, etc.                         │
│   - Latency: Adaptive (keyword: minimal, similarity: low, BERT: moderate)   │
│                                                                              │
│ Step 5: Model Selection                                                     │
│   - Lookup category → model scores mapping                                  │
│   - Select best-performing model for category                               │
│   - Example: "math" → deepseek-v31 (score: 0.92)                            │
│                                                                              │
│ Step 6: Request Enrichment                                                  │
│   - Add headers:                                                            │
│     * X-VSR-Model: deepseek-v31                                             │
│     * X-VSR-Category: math                                                  │
│     * X-VSR-Reasoning: true                                                 │
│     * X-VSR-Reasoning-Effort: high                                          │
│     * X-VSR-Cache-Status: miss                                              │
│   - Modify request body:                                                    │
│     * Update "model" field to selected model                                │
│     * Inject reasoning parameters if applicable                             │
│     * Add selected tools if tool selection enabled                          │
│                                                                              │
│ Total Latency: Low to Moderate (parallel execution)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Infrastructure Optimization (Dynamo)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 7: Dynamo Frontend Receives Request                                    │
│   - Parse X-VSR-Model header                                                │
│   - Filter worker pool to model-specific workers                            │
│   - Example: Only consider workers serving deepseek-v31                     │
│                                                                              │
│ Step 8: KV-Aware Worker Selection                                           │
│   - Query KVBM for cached blocks per worker                                 │
│   - Calculate cost for each worker:                                         │
│     * potential_prefill_blocks = (input_tokens - overlap_blocks) / block_size│
│     * potential_active_blocks = current_active + new_request_blocks         │
│     * logit = kv_overlap_weight × prefill + active                          │
│   - Select worker with lowest cost                                          │
│   - Latency: Low                                                            │
│                                                                              │
│ Step 9: Request Forwarding                                                  │
│   - Forward to selected worker (prefill or decode)                          │
│   - Worker processes request with vLLM/SGLang/TRT-LLM                       │
│   - KVBM tracks new KV cache blocks                                         │
│                                                                              │
│ Total Latency: Low (routing overhead)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Response Processing                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 10: Worker Response                                                    │
│   - vLLM/SGLang generates tokens                                            │
│   - Stream response back to Dynamo Frontend                                 │
│                                                                              │
│ Step 11: Semantic Cache Update                                              │
│   - Semantic Router receives response via ExtProc                           │
│   - Store query embedding + response in Milvus                              │
│   - TTL: 7200 seconds (configurable)                                        │
│                                                                              │
│ Step 12: Response to Client                                                 │
│   - Envoy Gateway forwards response                                         │
│   - Add response headers:                                                   │
│     * X-VSR-Model-Used: deepseek-v31                                        │
│     * X-VSR-Cache-Hit: false                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.4.2 Dual-Layer Caching Strategy

The integration leverages **two complementary caching layers**:

**Layer 1: Semantic Cache (Request-Level)**

- **Granularity:** Entire request-response pairs
- **Matching:** Embedding similarity (cosine distance)
- **Threshold:** 0.85 (configurable)
- **Backend:** Milvus (vector database)
- **Benefit:** Avoids inference entirely for similar queries
- **Example:** "What is 2+2?" ≈ "Calculate 2 plus 2" (similarity: 0.91)

**Layer 2: KV Cache (Token-Level)**

- **Granularity:** Token-level KV cache blocks
- **Matching:** Exact prefix matching
- **Backend:** GPU HBM → System Memory → NVMe
- **Benefit:** Reduces prefill cost for partial overlaps
- **Example:** "Explain quantum computing" → "Explain quantum computing applications" (prefix reuse)

**Combined Benefit:**

```
Scenario 1: Exact Semantic Match
  Query: "What is the capital of France?"
  Semantic Cache: HIT (high similarity with "What's France's capital?")
  KV Cache: N/A (inference skipped)
  Latency: Very low (cache lookup only)
  Cost Reduction: Maximum (no inference)

Scenario 2: Partial Semantic Match + KV Reuse
  Query: "Explain the proof of Fermat's Last Theorem in detail"
  Semantic Cache: MISS (novel query)
  KV Cache: HIT (significant overlap with "Explain Fermat's Last Theorem")
  Latency: Reduced (vs. without KV reuse)
  Cost Reduction: Significant (prefill cost saved)

Scenario 3: Novel Query
  Query: "Design a distributed consensus algorithm for blockchain"
  Semantic Cache: MISS
  KV Cache: MISS
  Latency: Standard (full inference)
  Cost Reduction: None (but routed to best model)
```

### 4.5 Integration in Kubernetes

#### 4.5.1 Deployment Architecture

The integration follows a **layered service architecture** in Kubernetes, with clear separation between semantic intelligence and infrastructure optimization:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Kubernetes Cluster: llm-inference-stack                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Layer 1: Gateway & Semantic Intelligence                   │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                             │    │
│  │  [Envoy Gateway]                                           │    │
│  │       ↓ (ExtProc gRPC)                                     │    │
│  │  [Semantic Router Service]                                 │    │
│  │   - Pods: 3 replicas (HA)                                  │    │
│  │   - Port: 50051 (gRPC)                                     │    │
│  │   - Functions:                                             │    │
│  │     * BERT classification (14 categories)                  │    │
│  │     * System prompt injection                              │    │
│  │     * PII/Jailbreak detection                              │    │
│  │     * Semantic cache lookup                                │    │
│  │     * Model selection                                      │    │
│  │   - Dependencies:                                          │    │
│  │     * Milvus Service (semantic cache)                      │    │
│  │     * ConfigMap (routing rules)                            │    │
│  │     * PVC (ML models)                                      │    │
│  │                                                             │    │
│  │  [Milvus Service]                                          │    │
│  │   - Port: 19530 (gRPC)                                     │    │
│  │   - Vector database for semantic caching                   │    │
│  │   - Storage: PVC for persistence                           │    │
│  │                                                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓                                          │
│                   (HTTP with headers:                               │
│                    X-VSR-Model, X-VSR-Category, etc.)               │
│                          ↓                                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Layer 2: Infrastructure Optimization (Dynamo)              │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                             │    │
│  │  [Dynamo Frontend Service]                                 │    │
│  │   - Pods: 2 replicas (HA)                                  │    │
│  │   - Port: 8000 (HTTP)                                      │    │
│  │   - Functions:                                             │    │
│  │     * Parse X-VSR-Model header                             │    │
│  │     * Filter worker pool by model                          │    │
│  │     * KV-aware worker selection                            │    │
│  │     * Request forwarding                                   │    │
│  │   - Components:                                            │    │
│  │     * KV Router                                            │    │
│  │     * Planner (dynamic scaling)                            │    │
│  │     * KVBM (KV cache manager)                              │    │
│  │                                                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓                                          │
│                   (Worker selection based on                        │
│                    model + KV cache state)                          │
│                          ↓                                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Layer 3: Execution (vLLM/SGLang Workers)                   │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                             │    │
│  │  [Model Pool: deepseek-v31]                                │    │
│  │   - StatefulSet: Multiple replicas                        │    │
│  │   - Service: vllm-deepseek-v31-svc                         │    │
│  │   - GPU: Multi-GPU per pod                                 │    │
│  │   - Features: prefix caching, fp8 KV cache                 │    │
│  │                                                             │    │
│  │  [Model Pool: qwen3]                                       │    │
│  │   - StatefulSet: Multiple replicas                        │    │
│  │   - Service: vllm-qwen3-svc                                │    │
│  │   - GPU: Multi-GPU per pod                                 │    │
│  │                                                             │    │
│  │  [Model Pool: phi4]                                        │    │
│  │   - StatefulSet: Multiple replicas                        │    │
│  │   - Service: vllm-phi4-svc                                 │    │
│  │   - GPU: Single/Multi-GPU per pod                          │    │
│  │                                                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Kubernetes Services:**

1. **semantic-router-svc** (ClusterIP)
   - Exposes Semantic Router ExtProc on port 50051
   - Used by Envoy Gateway for request processing
   - Selector: `app=semantic-router`

2. **dynamo-frontend-svc** (ClusterIP)
   - Exposes Dynamo Frontend on port 8000
   - Receives enriched requests from Envoy Gateway
   - Selector: `app=dynamo-frontend`

3. **vllm-\{model\}-svc** (Headless Service)
   - One service per model pool
   - Enables direct pod-to-pod communication
   - Used by Dynamo for worker selection
   - Selector: `app=vllm-worker, model=\{model-name\}`

4. **milvus-svc** (ClusterIP)
   - Exposes Milvus on port 19530 (gRPC)
   - Used by Semantic Router for semantic caching
   - Vector database for embedding similarity search
   - Selector: `app=milvus`

#### 4.5.2 Service Communication Flow

**End-to-End Request Path:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: Client Request                                               │
├──────────────────────────────────────────────────────────────────────┤
│ POST /v1/chat/completions                                            │
│ Host: llm-gateway.example.com:8080                                   │
│ Content-Type: application/json                                       │
│                                                                       │
│ {                                                                    │
│   "messages": [                                                      │
│     {"role": "user", "content": "Prove Fermat's Last Theorem"}      │
│   ],                                                                 │
│   "model": "auto"                                                    │
│ }                                                                    │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Step 2: Envoy Gateway (Port 8080)                                   │
├──────────────────────────────────────────────────────────────────────┤
│ - Receives HTTP request                                             │
│ - Invokes ExtProc: semantic-router-svc:50051 (gRPC)                 │
│ - Sends request body + headers to Semantic Router                   │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Step 3: Semantic Router Service (ExtProc gRPC)                      │
├──────────────────────────────────────────────────────────────────────┤
│ Processing Pipeline:                                                │
│                                                                       │
│ 3.1 Fusion Routing (Multi-Signal Classification)                    │
│     - Input: "Prove Fermat's Last Theorem"                          │
│     - Keyword matching: No match                                    │
│     - Similarity search: No strong match                            │
│     - BERT classification: category="math", confidence=0.92         │
│     - Decision: Use BERT result (highest confidence)                │
│                                                                       │
│ 3.2 System Prompt Selection                                         │
│     - Lookup: categories["math"].system_prompt                      │
│     - Prompt: "You are a mathematics expert..."                     │
│                                                                       │
│ 3.3 Model Selection                                                 │
│     - Lookup: categories["math"].model_scores                       │
│     - Selected: deepseek-v31 (score: 0.92, reasoning: true)         │
│                                                                       │
│ 3.4 Security Checks                                                 │
│     - PII Detection: PASS (no sensitive data)                       │
│     - Jailbreak Detection: PASS (legitimate query)                  │
│                                                                       │
│ 3.5 Semantic Cache Lookup                                           │
│     - Query Milvus: embedding similarity search                     │
│     - Result: MISS (novel query)                                    │
│                                                                       │
│ 3.6 Response to Envoy                                               │
│     - Modified Request Body:                                        │
│       * model: "auto" → "deepseek-v31" (OVERRIDDEN)                │
│       * messages: [system prompt injected]                          │
│     - Observability Headers (optional, added to response):          │
│       * x-vsr-selected-category: math                               │
│       * x-vsr-selected-reasoning: on                                │
│       * x-vsr-selected-model: deepseek-v31                          │
│       * x-vsr-injected-system-prompt: true                          │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Step 4: Envoy Gateway (Forwarding)                                  │
├──────────────────────────────────────────────────────────────────────┤
│ - Receives enriched request from Semantic Router                    │
│ - Forwards to: dynamo-frontend-svc:8000                             │
│ - Request body now has: model="deepseek-v31" (overridden from "auto")│
│ - Optional observability headers preserved                          │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Step 5: Dynamo Frontend Service (Port 8000)                         │
├──────────────────────────────────────────────────────────────────────┤
│ Processing Pipeline:                                                │
│                                                                       │
│ 5.1 Request Body Parsing                                            │
│     - Read: request.model = "deepseek-v31"                          │
│     - Dynamo is UNAWARE that model was changed by VSR               │
│     - Treats it as a normal request for deepseek-v31                │
│                                                                       │
│ 5.2 Worker Pool Filtering                                           │
│     - Query Kubernetes: vllm-deepseek-v31-svc (Headless)            │
│     - Available Workers:                                            │
│       * vllm-deepseek-v31-0 (10.244.1.5:8000)                       │
│       * vllm-deepseek-v31-1 (10.244.1.6:8000)                       │
│       * vllm-deepseek-v31-2 (10.244.1.7:8000)                       │
│       * vllm-deepseek-v31-3 (10.244.1.8:8000)                       │
│                                                                       │
│ 5.3 KV-Aware Worker Selection                                       │
│     - Query KVBM for each worker's cache state                      │
│     - Calculate routing score:                                      │
│       score = kv_overlap × weight + active_blocks                   │
│     - Results:                                                       │
│       * Worker-0: score=120 (high KV overlap)                       │
│       * Worker-1: score=85                                          │
│       * Worker-2: score=90                                          │
│       * Worker-3: score=75                                          │
│     - Selected: Worker-0 (10.244.1.5:8000)                          │
│                                                                       │
│ 5.4 Request Forwarding                                              │
│     - Forward to: http://10.244.1.5:8000/v1/chat/completions        │
│     - Request body: model="deepseek-v31" (as-is from VSR)           │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Step 6: vLLM Worker (deepseek-v31-0)                                │
├──────────────────────────────────────────────────────────────────────┤
│ 6.1 Request Processing                                              │
│     - Receive request: model="deepseek-v31"                         │
│     - System prompt already injected in messages by VSR             │
│     - Worker is UNAWARE of VSR's involvement                        │
│                                                                       │
│ 6.2 Inference Execution                                             │
│     - Model: DeepSeek-V3                                            │
│     - Messages: [system prompt + user query]                        │
│     - Prefix Caching: Enabled (KV cache reuse)                      │
│     - Generate response with step-by-step proof                     │
│                                                                       │
│ 6.3 Response Generation                                             │
│     - Return: Streaming or non-streaming response                   │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Step 7: Response Path (Reverse)                                     │
├──────────────────────────────────────────────────────────────────────┤
│ Worker → Dynamo Frontend → Envoy Gateway → Client                   │
│                                                                       │
│ - Envoy adds observability headers: X-Envoy-Upstream-Service-Time   │
│ - Client receives complete response with metadata                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Key Integration Points:**

1. **Transparent Model Override (Critical Design)**
   - User sends: `{"model": "auto", "messages": [...]}`
   - Semantic Router modifies request body: `model: "auto" → "deepseek-v31"`
   - Dynamo receives: `{"model": "deepseek-v31", "messages": [...]}`
   - **Dynamo is completely unaware of VSR's involvement**
   - No special headers needed for model routing
   - Standard OpenAI API compatibility maintained

2. **System Prompt Injection**
   - Semantic Router injects system prompt into messages array
   - Example: `messages: [{"role": "system", "content": "You are a mathematics expert..."}, {"role": "user", "content": "..."}]`
   - Worker receives pre-enriched request
   - No additional processing needed by Dynamo or worker

3. **Service Discovery**
   - Envoy → Semantic Router: `semantic-router-svc.llm-inference-stack.svc.cluster.local:50051` (gRPC ExtProc)
   - Envoy → Dynamo: `dynamo-frontend-svc.llm-inference-stack.svc.cluster.local:8000` (HTTP)
   - Dynamo → Workers: `vllm-\{model\}-svc.llm-inference-stack.svc.cluster.local` (Headless Service)
   - Semantic Router → Milvus: `milvus-svc.llm-inference-stack.svc.cluster.local:19530` (gRPC)

4. **Observability (Optional Headers)**
   - `x-vsr-selected-category`: Query classification result (e.g., "math")
   - `x-vsr-selected-reasoning`: Reasoning mode flag (e.g., "on" or "off")
   - `x-vsr-selected-model`: Model selected by VSR (e.g., "deepseek-v31")
   - `x-vsr-injected-system-prompt`: Whether system prompt was injected (e.g., "true" or "false")
   - `x-vsr-cache-hit`: Semantic cache status (value: "true" when cache hit)
   - These headers are for **observability only**, not used by Dynamo for routing
   - Dynamo and workers can ignore these headers
   - Headers are only added to successful responses (HTTP 200-299) that did not hit cache

5. **Distributed Tracing**
   - Full-stack distributed tracing support across VSR → Dynamo → Workers
   - OpenTelemetry-based instrumentation
   - Single trace spans all layers with proper context propagation
   - Reference: [PR #322 - Distributed Tracing Support](https://github.com/vllm-project/semantic-router/pull/322)
   - Enables end-to-end latency analysis and bottleneck identification

6. **Cache Coordination**
   - Semantic cache (Milvus): Request-level, checked first by VSR
   - KV cache (Dynamo/vLLM): Token-level, managed by Dynamo
   - Independent layers, no coordination needed
   - If semantic cache hits, request never reaches Dynamo

#### 4.5.3 Worker Pool Management

**Worker Discovery via Kubernetes Services:**

Dynamo Frontend discovers workers through Kubernetes Headless Services, which provide direct pod IP addresses:

1. **Headless Service Configuration**
   - Service Type: `ClusterIP: None` (headless)
   - Selector: `app=vllm-worker, model=\{model-name\}`
   - DNS returns all pod IPs instead of load-balanced VIP

2. **Worker Registration Flow**

   ```
   vLLM Worker Pod Startup
   ↓
   Worker registers with Dynamo Frontend via HTTP API
   ↓
   Dynamo Frontend tracks:
   - Worker ID (pod name)
   - Model name (deepseek-v31, qwen3, phi4)
   - Endpoint (pod IP:8000)
   - Capabilities (prefill, decode, max_batch_size)
   - KV cache state (tracked by KVBM)
   ```

3. **Model Pool Organization**
   - Each model has dedicated StatefulSet + Headless Service
   - Example: `vllm-deepseek-v31-svc` → 4 pods serving DeepSeek-V3
   - Dynamo queries service DNS to get all pod IPs
   - Filters workers by `X-VSR-Model` header from Semantic Router

4. **Dynamic Scaling**
   - Horizontal Pod Autoscaler (HPA) adjusts replicas based on GPU utilization
   - New pods auto-register with Dynamo on startup
   - Dynamo updates worker pool in real-time

### 4.6 Implementation Plan

#### Phase 1: Foundation

**Objectives:**

- Establish basic integration between Semantic Router and Dynamo
- Implement transparent model override in request body
- Validate end-to-end request flow

**Tasks:**

1. **Semantic Router Enhancements:**
   - Implement request body modification: `model: "auto" → "selected-model"`
   - Add system prompt injection to messages array
   - Add optional observability headers:
     - `x-vsr-selected-category`: Classification result
     - `x-vsr-selected-reasoning`: Reasoning mode ("on" or "off")
     - `x-vsr-selected-model`: Selected model name
     - `x-vsr-injected-system-prompt`: System prompt injection status ("true" or "false")
     - `x-vsr-cache-hit`: Cache hit status (only when cache hit)
   - Ensure OpenAI API compatibility maintained

2. **Dynamo Frontend (No Changes Required):**
   - Dynamo receives standard OpenAI API requests
   - Model field already contains the selected model name
   - No awareness of VSR's involvement needed
   - Existing routing logic works as-is

3. **Testing:**
   - Unit tests for model override logic
   - Integration tests for system prompt injection
   - Verify Dynamo routes to correct model pools
   - Load tests with 1K RPS

**Success Criteria:**

- ✅ Requests routed to correct model pools based on overridden model name
- ✅ System prompts correctly injected into messages
- ✅ Dynamo operates transparently without modifications
- ✅ Latency overhead < 10ms
- ✅ No breaking changes to existing deployments

#### Phase 2: Dual-Layer Caching

**Objectives:**

- Integrate semantic cache with KV cache
- Implement cache coordination strategy
- Optimize cache hit rates

**Tasks:**

1. **Cache Integration:**
   - Add semantic cache lookup before Dynamo routing
   - Implement cache miss forwarding to Dynamo
   - Add cache hit metrics and headers

2. **Performance Optimization:**
   - Parallel cache lookup and classification
   - Milvus connection pooling
   - Cache warming strategies

3. **Testing:**
   - Cache hit rate benchmarks
   - Latency comparison (cache hit vs. miss)
   - Cache eviction policy validation

**Success Criteria:**

- ✅ High semantic cache hit rate (production workloads)
- ✅ Low cache hit latency
- ✅ High combined cache hit rate (semantic + KV)

#### Phase 3: Observability & Monitoring

**Objectives:**

- Full-stack distributed tracing across VSR → Dynamo → Workers
- Comprehensive metrics and dashboards
- Alerting and SLO monitoring

**Tasks:**

1. **Distributed Tracing (OpenTelemetry):**
   - Trace context propagation from VSR through Dynamo to workers
   - Span hierarchy:
     - Root span: Envoy Gateway
     - Child span: Semantic Router (fusion routing, cache, security)
       - Sub-span: BERT classification
       - Sub-span: Keyword matching
       - Sub-span: Similarity search
       - Sub-span: Signal fusion & decision
     - Child span: Dynamo Frontend (routing, worker selection)
     - Child span: vLLM Worker (inference execution)
   - Automatic trace ID injection in headers
   - Support for Jaeger, Tempo, and other OTLP-compatible backends

2. **Metrics Collection:**
   - Semantic Router metrics:
     - Fusion routing performance:
       - BERT classification latency and accuracy
       - Keyword matching hit rate and latency
       - Similarity search latency
       - Signal fusion decision distribution
     - Semantic cache hit rate (Milvus)
     - PII/Jailbreak detection rate
     - Model selection distribution by category
   - Dynamo metrics:
     - KV-aware routing decisions
     - Worker utilization
     - KV cache hit rate
   - End-to-end latency breakdown by component

3. **Dashboards:**
   - Grafana dashboard for integrated stack
   - Request flow visualization with trace waterfall
   - Cost and performance analytics
   - Cache efficiency metrics (semantic + KV)

**Success Criteria:**

- ✅ Single distributed trace spans all layers (VSR → Dynamo → Worker)
- ✅ Minimal trace sampling overhead
- ✅ Real-time dashboards operational
- ✅ Trace context properly propagated across service boundaries

#### Phase 4: Production Hardening

**Objectives:**

- Failure handling and resilience
- Performance optimization
- Production deployment

**Tasks:**

1. **Resilience:**
   - Semantic Router failure fallback to Dynamo
   - Circuit breaker for cache backend
   - Graceful degradation strategies

2. **Performance:**
   - Latency optimization (target: < 50ms combined)
   - Throughput testing (target: 10K RPS)
   - Resource utilization tuning

3. **Documentation:**
   - Deployment guide
   - Configuration reference
   - Troubleshooting runbook

**Success Criteria:**

- ✅ High availability
- ✅ Low P99 latency (routing overhead)
- ✅ 10K+ RPS sustained throughput

---

## 6. Security and Privacy Considerations

### 6.1 PII Detection and Blocking

**Threat Model:**

- Users may inadvertently include PII in prompts
- PII could be logged, cached, or sent to third-party models
- Compliance requirements (GDPR, HIPAA, CCPA)

**Mitigation:**

- Token-level PII detection using ModernBERT classifier
- Configurable blocking policies per model
- PII types: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, STREET_ADDRESS, IP_ADDRESS, IBAN_CODE, US_DRIVER_LICENSE, and more
- Response header when blocked: `x-vsr-pii-violation: true`
- Audit logging of all PII detections

**Example Configuration:**

```yaml
model_config:
  public-model:
    pii_policy:
      allow_by_default: false
      pii_types_allowed: ["PERSON"]  # Only person names allowed
```

### 6.2 Jailbreak Prevention (Prompt Guard)

**Threat Model:**

- Adversarial prompts attempting to bypass safety guardrails
- Prompt injection attacks
- Social engineering attempts

**Mitigation:**

- **Prompt Guard** classification for jailbreak detection
- Threshold-based blocking (configurable, e.g., 0.5)
- ModernBERT-based classification model
- Jailbreak type detection with confidence scoring
- Response headers when blocked:
  - `x-vsr-jailbreak-blocked: true`
  - `x-vsr-jailbreak-type: {type}` (e.g., "prompt_injection")
  - `x-vsr-jailbreak-confidence: {score}` (e.g., "0.950")

**Example Configuration:**

```yaml
prompt_guard:
  enabled: true
  # model_id is auto-discovered from models directory:
  # - Legacy: models/jailbreak_classifier_modernbert-base_model
  # - LoRA: models/lora_jailbreak_classifier_bert_model (preferred)
  #         models/lora_jailbreak_classifier_roberta_model
  #         models/lora_jailbreak_classifier_modernbert_model
  threshold: 0.5
  use_cpu: false
  use_modernbert: true
  # jailbreak_mapping_path is auto-discovered from model directory
```

**Note:** The jailbreak classifier uses auto-discovery to find models in the `models/` directory. The system prefers LoRA models (BERT > RoBERTa > ModernBERT) over legacy ModernBERT models for better accuracy.

### 6.3 Data Residency and Compliance

**Considerations:**

- Semantic cache may store user queries
- KV cache contains model activations
- Distributed tracing may log request content

**Best Practices:**

1. **Cache Encryption:** Encrypt Milvus cache at rest and in transit
2. **TTL Policies:** Automatic expiration of cached data (default: 2 hours)
3. **Data Locality:** Deploy in compliance-approved regions
4. **Audit Logging:** Comprehensive logs for compliance audits
5. **Right to Deletion:** API for purging user data from caches

---

## 7. Operational Considerations

### 7.1 Monitoring and Alerting

**Key Metrics:**

| Metric | Threshold | Alert Severity |
|--------|-----------|----------------|
| Semantic Router Latency (P99) | High | Warning |
| Dynamo Router Latency (P99) | High | Warning |
| Combined Latency (P99) | Very High | Critical |
| Semantic Cache Hit Rate | Low | Warning |
| KV Cache Hit Rate | Low | Warning |
| Security Block Rate | High | Warning |
| Error Rate | High | Critical |
| GPU Utilization | Too Low or Too High | Warning |

**Dashboards:**

1. **Request Flow Dashboard:** Visualize request journey through layers
2. **Cache Performance Dashboard:** Hit rates, latency, eviction rates
3. **Security Dashboard:** PII detections, jailbreak blocks, audit logs
4. **Cost Dashboard:** Token usage, model selection, cost per query

### 7.3 Failure Modes and Recovery

**Failure Scenario 1: Semantic Router Unavailable**

- **Detection:** Health check failures, timeout errors
- **Impact:** No semantic routing, security filtering, or caching
- **Recovery:**
  - Envoy Gateway bypasses ExtProc (fallback mode)
  - Requests forwarded directly to Dynamo
  - Dynamo performs default routing
- **Mitigation:** Deploy 3+ replicas with anti-affinity

**Failure Scenario 2: Milvus Cache Unavailable**

- **Detection:** Connection errors, timeout
- **Impact:** No semantic caching (cache misses)
- **Recovery:**
  - Semantic Router continues with in-memory cache
  - All requests forwarded to Dynamo
  - Performance degradation but no outage
- **Mitigation:** Milvus cluster deployment for HA

**Failure Scenario 3: Dynamo Frontend Unavailable**

- **Detection:** HTTP 503 errors, connection refused
- **Impact:** No inference possible
- **Recovery:**
  - Envoy Gateway returns 503 to clients
  - Kubernetes restarts failed pods
  - Load balancer routes to healthy replicas
- **Mitigation:** Deploy 2+ replicas with readiness probes

**Failure Scenario 4: Worker Pool Exhaustion**

- **Detection:** Queue depth alerts, high latency
- **Impact:** Increased TTFT and ITL
- **Recovery:**
  - Dynamo Planner auto-scales workers
  - Semantic Router may route to alternative models
  - Requests queued until capacity available
- **Mitigation:** Autoscaling policies, overprovisioning

## 8. Future Enhancements

### 8.1 Advanced Routing Strategies

**Multi-Objective Optimization:**

- Combine semantic quality, latency, and cost in routing decision
- Pareto-optimal model selection
- User-specified SLO preferences (fast vs. accurate vs. cheap)

**Adaptive Routing:**

- Learn from user feedback (thumbs up/down)
- A/B testing of model selections
- Reinforcement learning for routing policy

### 8.2 Cross-Layer Optimization

**Semantic-Aware KV Cache Management:**

- Prioritize KV cache retention for high-value categories
- Semantic similarity for KV cache eviction decisions
- Cross-request KV cache sharing for similar queries

**Predictive Prefetching:**

- Predict next query in conversation
- Pre-warm KV cache for likely follow-ups
- Speculative execution for low-latency responses

### 8.3 Multi-Tenant Support

**Tenant Isolation:**

- Per-tenant semantic cache namespaces
- Per-tenant model access policies
- Per-tenant cost tracking and quotas

**Tenant-Specific Routing:**

- Custom model pools per tenant
- Tenant-specific security policies
- Tenant-specific SLOs

---

## 9. References

### 9.1 NVIDIA Dynamo Documentation

- [Dynamo Architecture Overview](https://docs.nvidia.com/dynamo/latest/_sections/architecture.html)
- [Dynamo KV Router](https://docs.nvidia.com/dynamo/latest/components/router/README.html)
- [Dynamo Disaggregated Serving](https://docs.nvidia.com/dynamo/latest/_sections/disaggregated-serving.html)
- [Dynamo KVBM](https://docs.nvidia.com/dynamo/latest/components/kvbm/README.html)

### 9.2 vLLM Semantic Router Documentation

- [Semantic Router Overview](https://vllm-semantic-router.com/docs/overview/semantic-router-overview/)
- [System Architecture](https://vllm-semantic-router.com/docs/overview/architecture/system-architecture/)
- [Kubernetes Deployment](https://vllm-semantic-router.com/docs/installation/k8s/ai-gateway)
- [Distributed Tracing Support (PR #322)](https://github.com/vllm-project/semantic-router/pull/322)
- [Milvus-based Semantic Caching](https://vllm-semantic-router.com/docs/features/semantic-caching/)

### 9.3 Related Research

- **DistServe:** Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- **Mooncake:** KVCache-centric Disaggregated Architecture for LLM Serving
- **RouteLLM:** Learning to Route LLMs with Preference Data
- **DeepSeek-V3:** Technical Report on Mixture-of-Experts Architecture

### 9.4 Integration Proposals

- [vLLM Production Stack Integration (#295)](https://github.com/vllm-project/semantic-router/issues/295)
- [Prompt Classification Routing Proposal](https://vllm-semantic-router.com/docs/proposals/prompt-classification-routing/)

---

## 10. Appendix

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **BERT** | Bidirectional Encoder Representations from Transformers |
| **ExtProc** | Envoy External Processor (gRPC service for request processing) |
| **Fusion Routing** | Multi-signal routing combining BERT classification, keyword matching, and similarity search |
| **ITL** | Inter-Token Latency (time between generated tokens) |
| **KV Cache** | Key-Value cache storing transformer attention states |
| **KVBM** | KV Block Manager (Dynamo component for cache management) |
| **Milvus** | Open-source vector database for semantic caching and similarity search |
| **MoE** | Mixture-of-Experts (model architecture with specialized expert networks) |
| **MoM** | Mixture-of-Models (routing to different models based on task) |
| **NIXL** | NVIDIA Inference Transfer Library |
| **OTLP** | OpenTelemetry Protocol (for distributed tracing and metrics) |
| **PII** | Personally Identifiable Information |
| **Prompt Guard** | Jailbreak detection system using classification models to identify adversarial prompts |
| **TTFT** | Time To First Token (latency until first token generated) |

### 10.2 System Prompt Examples

**Domain-Aware System Prompts for Key Categories:**

The integration leverages **14 specialized system prompts** that are automatically injected based on query classification. Here are representative examples:

**1. Math Category (Reasoning-Heavy)**

```
You are a mathematics expert. Provide step-by-step solutions, show your
work clearly, and explain mathematical concepts in an understandable way.
```

- **Purpose**: Encourage structured reasoning and clear explanations
- **Model**: DeepSeek-V3 (score: 1.0, reasoning: enabled)
- **MoE Impact**: Activates mathematical reasoning experts

**2. Computer Science Category (Code-Focused)**

```
You are a computer science expert with knowledge of algorithms, data structures,
programming languages, and software engineering. Provide clear, practical solutions
with code examples when helpful.
```

- **Purpose**: Balance theory with practical code examples
- **Model**: Qwen3 (score: 0.89, reasoning: disabled)
- **MoE Impact**: Activates programming and algorithm experts

**3. Business Category (Action-Oriented)**

```
You are a senior business consultant and strategic advisor with expertise in
corporate strategy, operations management, financial analysis, marketing, and
organizational development. Provide practical, actionable business advice backed
by proven methodologies and industry best practices. Consider market dynamics,
competitive landscape, and stakeholder interests in your recommendations.
```

- **Purpose**: Emphasize actionable advice and business context
- **Model**: Phi-4 (score: 0.88, reasoning: disabled)
- **MoE Impact**: Activates business strategy and analysis experts

**4. Law Category (Disclaimer-Aware)**

```
You are a knowledgeable legal expert with comprehensive understanding of legal
principles, case law, statutory interpretation, and legal procedures. Provide
accurate legal information while clearly stating that your responses are for
informational purposes only and do not constitute legal advice.
```

- **Purpose**: Ensure accuracy while maintaining ethical boundaries
- **Model**: Phi-4 (score: 0.75, reasoning: disabled)
- **MoE Impact**: Activates legal reasoning experts with appropriate disclaimers

**5. Health Category (Evidence-Based)**

```
You are a health and medical information expert with knowledge of anatomy,
physiology, diseases, treatments, preventive care, nutrition, and wellness.
Provide accurate, evidence-based health information while emphasizing that
your responses are for educational purposes only and do not replace professional
medical advice.
```

- **Purpose**: Balance informativeness with medical ethics
- **Model**: Phi-4 (score: 0.76, reasoning: disabled)
- **MoE Impact**: Activates medical knowledge experts with safety guardrails

**Complete Category List:**

- math, computer science, physics, chemistry, biology, engineering
- economics, business, law, psychology, philosophy, history, health, other

**System Prompt Benefits:**

- **CoT Optimization**: Domain-specific reasoning patterns improve output quality
- **Token Efficiency**: Focused prompts reduce unnecessary verbosity (10-15% token reduction)
- **MoE Expert Matching**: Specialized terminology activates relevant experts (20-30% improvement in expert selection accuracy)
- **Quality Control**: Category-specific disclaimers ensure ethical compliance

### 10.3 API Examples

**Request with Semantic Router Headers:**

```bash
curl -X POST http://llm-gateway:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "Prove that the square root of 2 is irrational"
      }
    ]
  }'
```

**Response with Routing Headers:**

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
        "content": "To prove that √2 is irrational, we'll use proof by contradiction..."
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

## Conclusion

This proposal outlines a comprehensive integration strategy between vLLM Semantic Router and NVIDIA Dynamo that combines semantic intelligence with infrastructure optimization. The layered architecture ensures:

1. **Semantic Correctness:** Right model selection based on query understanding
2. **Infrastructure Efficiency:** Optimal worker selection and KV cache utilization
3. **Security:** PII detection and jailbreak prevention before inference
4. **Performance:** Dual-layer caching for 40-60% latency reduction
5. **Cost Optimization:** 55% cost reduction through intelligent routing

The integration is designed to be **non-invasive**, **modular**, and **production-ready**, with clear implementation phases, comprehensive monitoring, and robust failure handling.

**Next Steps:**

1. Review and approve proposal
2. Begin Phase 1 implementation (Foundation)
3. Establish benchmark environment
4. Iterate based on performance results
