# Config DSL & Visual Builder UX

## 1. Overview

The vLLM Semantic Router configuration system is extraordinarily complex:

| Metric | Value |
|:---|:---|
| Go struct lines (`config.go`) | **2,834** |
| Maximum nesting depth | **7 levels** |
| Distinct types | **90+** |
| Signal types | **11** |
| Plugin types | **11** |
| Algorithm variants | **15+** |

Writing raw YAML is error-prone and inaccessible to non-expert users. This document defines a **Config DSL** (domain-specific language) and a **Visual Builder UI** that expose the same power through three interaction modes, all sharing a single DSL AST as the source of truth.

### Design Goals

1. **Three modes, one truth** — Visual, DSL text, and Natural Language modes all read/write the same AST.
2. **Signal Compiler** — A Go→WASM compiler that runs in the browser, providing sub-millisecond compilation and validation.
3. **Lossless round-trip** — `DSL → YAML → DSL` and `Visual → DSL → Visual` are bijective.
4. **4:1 compression** — ~70 lines of DSL expand to ~300 lines of YAML.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                           │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────┐ │
│  │  Visual Mode   │  │  DSL Mode     │  │ Natural Language    │ │
│  │  (Forms/DnD)   │  │  (Monaco)     │  │ Mode (LLM)         │ │
│  └───────┬───────┘  └───────┬───────┘  └──────────┬──────────┘ │
│          │                  │                      │            │
│          └──────────────────┼──────────────────────┘            │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │    DSL AST      │  ◀── Single Source       │
│                    │  (Zustand Store)│      of Truth            │
│                    └────────┬────────┘                          │
│                             │                                   │
├─────────────────────────────┼───────────────────────────────────┤
│                    Compiler Layer                                │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │ Signal Compiler │                          │
│                    │  (Go → WASM)    │                          │
│                    └────────┬────────┘                          │
│                             │                                   │
├─────────────────────────────┼───────────────────────────────────┤
│                     Output Layer                                │
│          ┌──────────┼────────────┬────────────┐                 │
│          ▼          ▼            ▼             ▼                 │
│   ┌──────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐           │
│   │config.yaml│ │K8s CRD │ │Helm Values│ │config.dsl│           │
│   └──────────┘ └────────┘ └──────────┘ └──────────┘           │
│                                                                 │
│   ┌──────────────────────┐                                      │
│   │Validation Diagnostics│  (real-time, 3 severity levels)     │
│   └──────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘

```

### Data Flow

```
User Input ──▶ Mode Adapter ──▶ DSL AST ──▶ Signal Compiler (WASM) ──▶ config.yaml
                                   │                                       │
                                   ├──▶ DSL Preview (read-only)            │
                                   └──▶ Validation Diagnostics ◀───────────┘

```

---

## 3. DSL Language Definition

The DSL has exactly **5 core constructs**. Each construct maps to a panel in the Visual Builder and a section in the compiled YAML.

| Construct | Purpose | YAML Target |
|:---|:---|:---|
| `SIGNAL` | Declare what to detect in user queries | `keyword_rules`, `embedding_rules`, `categories`, `fact_check_rules`, `user_feedback_rules`, `preference_rules`, `language_rules`, `context_rules`, `complexity_rules`, `modality_rules`, `role_bindings` |
| `ROUTE` | Define routing decisions with boolean logic | `decisions[]` (rules tree, modelRefs, algorithm, plugins) |
| `PLUGIN` | Attach policies to routes (reusable templates) | `decisions[].plugins[]` |
| `BACKEND` | Configure infrastructure (cache, memory, storage, endpoints) | `semantic_cache`, `memory`, `response_api`, `vllm_endpoints`, `embedding_models`, `provider_profiles`, `image_gen_backends` |
| `GLOBAL` | Set defaults and global settings | Top-level `RouterConfig` fields (`default_model`, `strategy`, `observability`, `prompt_guard`, `authz`, `ratelimit`, etc.) |

### 3.1 Grammar (EBNF)

```ebnf
(* ===== Top Level ===== *)
program         = { statement } ;
statement       = signal_decl | route_decl | plugin_decl | backend_decl | global_decl ;

(* ===== SIGNAL: Declare detection signals ===== *)
signal_decl     = "SIGNAL" signal_type signal_name "{" { field_assign } "}" ;
signal_type     = "keyword" | "embedding" | "domain" | "fact_check"
                | "user_feedback" | "preference" | "language"
                | "context" | "complexity" | "modality" | "authz" ;
signal_name     = IDENTIFIER ;

(* ===== ROUTE: Define routing decisions ===== *)
route_decl      = "ROUTE" route_name [ route_opts ] "{"
                    "PRIORITY" INTEGER
                    "WHEN" bool_expr
                    "MODEL" model_list
                    [ "ALGORITHM" algo_spec ]
                    { "PLUGIN" plugin_ref }
                  "}" ;
route_name      = IDENTIFIER ;
route_opts      = "(" { route_opt } ")" ;
route_opt       = "description" "=" STRING ;

(* Boolean expression — infix notation with standard precedence *)
bool_expr       = bool_term { "OR" bool_term } ;
bool_term       = bool_factor { "AND" bool_factor } ;
bool_factor     = "NOT" bool_factor
                | "(" bool_expr ")"
                | signal_ref ;
signal_ref      = signal_type "(" signal_name ")" ;

(* Model references *)
model_list      = model_ref { "," model_ref } ;
model_ref       = STRING [ "(" model_opts ")" ] ;
model_opts      = model_opt { "," model_opt } ;
model_opt       = "reasoning" "=" BOOL
                | "effort" "=" STRING
                | "lora" "=" STRING
                | "param_size" "=" STRING ;

(* ===== ALGORITHM: Multi-model orchestration ===== *)
algo_spec       = algo_type [ "{" { field_assign } "}" ] ;
algo_type       = "confidence" | "ratings" | "remom"
                | "static" | "elo" | "router_dc" | "automix" | "hybrid"
                | "rl_driven" | "gmtrouter" | "latency_aware"
                | "knn" | "kmeans" | "svm" ;

(* ===== PLUGIN: Reusable policy templates ===== *)
plugin_decl     = "PLUGIN" plugin_name plugin_type "{" { field_assign } "}" ;
plugin_ref      = plugin_name [ "{" { field_assign } "}" ] ;
                  (* inline override or template reference *)
plugin_type     = "jailbreak" | "pii" | "semantic_cache" | "memory"
                | "system_prompt" | "header_mutation" | "hallucination"
                | "router_replay" | "rag" | "image_gen" ;

(* ===== BACKEND: Infrastructure configuration ===== *)
backend_decl    = "BACKEND" backend_type backend_name "{" { field_assign } "}" ;
backend_type    = "vllm_endpoint" | "provider_profile" | "embedding_model"
                | "semantic_cache" | "memory" | "response_api" | "vector_store"
                | "image_gen_backend" ;

(* ===== GLOBAL: Defaults and global settings ===== *)
global_decl     = "GLOBAL" "{" { field_assign } "}" ;

(* ===== Shared primitives ===== *)
field_assign    = IDENTIFIER ":" value ;
value           = STRING | INTEGER | FLOAT | BOOL | array | object ;
array           = "[" [ value { "," value } ] "]" ;
object          = "{" { field_assign } "}" ;

```

### 3.2 Signal Type Field Reference

Each signal type exposes specific fields in its body:

| Signal Type | Required Fields | Optional Fields |
|:---|:---|:---|
| `keyword` | `operator`, `keywords` | `method` (regex/bm25/ngram), `case_sensitive`, `fuzzy_match`, `fuzzy_threshold`, `bm25_threshold`, `ngram_threshold`, `ngram_arity` |
| `embedding` | `threshold`, `candidates` | `aggregation_method` (mean/max/any) |
| `domain` | `description` | `mmlu_categories`, `model_scores` |
| `fact_check` | `description` | — |
| `user_feedback` | `description` | — |
| `preference` | `description` | — |
| `language` | (none) | `description` |
| `context` | `min_tokens`, `max_tokens` | `description` |
| `complexity` | `threshold`, `hard`, `easy` | `description`, `composer` |
| `modality` | (none) | `description` |
| `authz` | `subjects`, `role` | `description` |

### 3.3 Complete DSL Example

```ruby
# =============================================================================
# SIGNALS — Declare what to detect in user queries
# =============================================================================

SIGNAL domain math {
  description: "Mathematics and quantitative reasoning"
  mmlu_categories: ["math"]
}

SIGNAL domain physics {
  description: "Physics and physical sciences"
  mmlu_categories: ["physics"]
}

SIGNAL domain computer_science {
  description: "Computer science and programming"
  mmlu_categories: ["computer_science"]
}

SIGNAL domain health {
  description: "Health and medical information queries"
  mmlu_categories: ["health"]
}

SIGNAL domain other {
  description: "General knowledge and miscellaneous topics"
  mmlu_categories: ["other"]
}

SIGNAL embedding ai_topics {
  threshold: 0.75
  candidates: ["machine learning", "neural network", "deep learning", "LLM"]
  aggregation_method: "max"
}

SIGNAL keyword urgent_request {
  operator: "any"
  keywords: ["urgent", "asap", "emergency"]
  method: "regex"
  case_sensitive: false
  fuzzy_match: true
  fuzzy_threshold: 2
}

SIGNAL context long_context {
  min_tokens: "4K"
  max_tokens: "32K"
  description: "Long-context requests requiring large window models"
}

SIGNAL complexity code_complexity {
  threshold: 0.1
  hard: { candidates: ["implement distributed system", "optimize compiler backend"] }
  easy: { candidates: ["print hello world", "simple for loop"] }
  description: "Code task complexity classification"
}

SIGNAL language zh {
  description: "Chinese language queries"
}

SIGNAL language en {
  description: "English language queries"
}

SIGNAL fact_check needs_fact_check {
  description: "Query requires external fact verification"
}

SIGNAL user_feedback wrong_answer {
  description: "User indicates the previous answer was incorrect"
}

SIGNAL modality DIFFUSION {
  description: "Image generation requests"
}

SIGNAL authz premium_binding {
  subjects: [
    { kind: "Group", name: "premium" },
    { kind: "User", name: "admin" }
  ]
  role: "premium_tier"
  description: "Premium users with access to large models"
}

# =============================================================================
# PLUGINS — Reusable policy templates
# =============================================================================

PLUGIN safe_pii pii {
  enabled: true
  pii_types_allowed: []
}

PLUGIN standard_jailbreak jailbreak {
  enabled: true
  threshold: 0.7
}

PLUGIN default_cache semantic_cache {
  enabled: true
  similarity_threshold: 0.80
}

# =============================================================================
# ROUTES — Define routing decisions
# =============================================================================

ROUTE math_decision (description = "Mathematics and quantitative reasoning") {
  PRIORITY 100

  WHEN domain("math")

  MODEL "qwen2.5:3b" (reasoning = true, effort = "high")

  PLUGIN system_prompt {
    system_prompt: "You are a mathematics expert. Provide step-by-step solutions."
  }
  PLUGIN safe_pii
}

ROUTE physics_decision (description = "Physics and physical sciences") {
  PRIORITY 100

  WHEN domain("physics")

  MODEL "qwen2.5:3b" (reasoning = true)

  PLUGIN system_prompt {
    system_prompt: "You are a physics expert with deep understanding of physical laws."
  }
  PLUGIN safe_pii
}

ROUTE health_decision (description = "Health and medical queries") {
  PRIORITY 100

  WHEN domain("health")

  MODEL "qwen2.5:3b" (reasoning = false)

  PLUGIN system_prompt {
    system_prompt: "You are a health expert. Provide evidence-based information."
  }
  PLUGIN semantic_cache {
    enabled: true
    similarity_threshold: 0.95
  }
  PLUGIN safe_pii
}

# Complex boolean condition with multiple signals
ROUTE urgent_ai_route (description = "Urgent AI-related requests get priority treatment") {
  PRIORITY 200

  WHEN keyword("urgent_request") AND embedding("ai_topics") AND NOT domain("other")

  MODEL "qwen3:70b" (reasoning = true, effort = "high", param_size = "70b"),
        "qwen2.5:3b" (reasoning = false, param_size = "3b")

  ALGORITHM confidence {
    confidence_method: "hybrid"
    threshold: 0.5
    hybrid_weights: { logprob_weight: 0.6, margin_weight: 0.4 }
    on_error: "skip"
  }

  PLUGIN safe_pii
  PLUGIN default_cache
  PLUGIN standard_jailbreak
}

# Multi-model ReMoM reasoning
ROUTE complex_reasoning (description = "Complex tasks requiring multi-model reasoning") {
  PRIORITY 150

  WHEN domain("math") AND complexity("code_complexity")

  MODEL "qwen3:70b", "deepseek-r1:32b"

  ALGORITHM remom {
    breadth_schedule: [8, 2]
    model_distribution: "weighted"
    temperature: 1.0
    include_reasoning: true
    on_error: "skip"
  }

  PLUGIN system_prompt {
    system_prompt: "Solve step by step with rigorous reasoning."
  }
}

# RAG-augmented decision
ROUTE knowledge_base_route (description = "Knowledge-grounded responses") {
  PRIORITY 120

  WHEN domain("computer_science") AND fact_check("needs_fact_check")

  MODEL "qwen2.5:3b" (reasoning = false)

  PLUGIN rag {
    enabled: true
    backend: "milvus"
    top_k: 5
    similarity_threshold: 0.7
    injection_mode: "tool_role"
    on_failure: "warn"
    backend_config: {
      collection: "knowledge_docs"
      reuse_cache_connection: true
      content_field: "content"
    }
  }
  PLUGIN hallucination {
    enabled: true
    use_nli: true
    hallucination_action: "body"
  }
  PLUGIN safe_pii
}

# Authz-gated premium route
ROUTE premium_route (description = "Premium users get access to large models") {
  PRIORITY 300

  WHEN authz("premium_tier")

  MODEL "gpt-4o", "claude-sonnet-4"

  ALGORITHM elo {
    initial_rating: 1500
    k_factor: 32
    category_weighted: true
  }

  PLUGIN safe_pii
  PLUGIN standard_jailbreak
}

# Feedback-aware re-routing
ROUTE wrong_answer_reroute (description = "Re-route when user indicates wrong answer") {
  PRIORITY 250

  WHEN user_feedback("wrong_answer") AND NOT domain("other")

  MODEL "qwen3:70b" (reasoning = true, effort = "high")

  PLUGIN system_prompt {
    system_prompt: "The user indicated the previous answer was incorrect. Re-examine carefully."
  }
}

# Fallback route
ROUTE general_decision (description = "General knowledge fallback") {
  PRIORITY 50

  WHEN domain("other")

  MODEL "qwen2.5:3b" (reasoning = false)

  PLUGIN system_prompt {
    system_prompt: "You are a helpful and knowledgeable assistant."
  }
  PLUGIN default_cache
  PLUGIN safe_pii
  PLUGIN memory {
    enabled: true
    retrieval_limit: 5
    similarity_threshold: 0.70
    auto_store: false
  }
}

# =============================================================================
# BACKENDS — Infrastructure configuration
# =============================================================================

BACKEND vllm_endpoint ollama {
  address: "127.0.0.1"
  port: 11434
  weight: 1
  type: "ollama"
}

BACKEND vllm_endpoint vllm_primary {
  address: "10.0.1.100"
  port: 8000
  weight: 3
  type: "vllm"
}

BACKEND provider_profile openai_prod {
  type: "openai"
  base_url: "https://api.openai.com/v1"
}

BACKEND provider_profile anthropic_prod {
  type: "anthropic"
  base_url: "https://api.anthropic.com"
  extra_headers: { "anthropic-version": "2023-06-01" }
}

BACKEND embedding_model ultra {
  mmbert_model_path: "models/mom-embedding-ultra"
  use_cpu: true
  hnsw_config: {
    model_type: "mmbert"
    preload_embeddings: true
    target_dimension: 768
    enable_soft_matching: true
    min_score_threshold: 0.5
  }
}

BACKEND semantic_cache main_cache {
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.8
  max_entries: 1000
  ttl_seconds: 3600
  eviction_policy: "fifo"
  use_hnsw: true
  hnsw_m: 16
  hnsw_ef_construction: 200
}

BACKEND memory agentic_memory {
  enabled: false
  auto_store: false
  milvus: {
    address: "localhost:19530"
    collection: "agentic_memory"
    dimension: 384
  }
  default_retrieval_limit: 5
  default_similarity_threshold: 0.70
}

BACKEND response_api main {
  enabled: true
  store_backend: "memory"
  ttl_seconds: 86400
  max_responses: 1000
}

# =============================================================================
# GLOBAL — Defaults and global settings
# =============================================================================

GLOBAL {
  default_model: "qwen2.5:3b"
  strategy: "priority"
  default_reasoning_effort: "low"

  reasoning_families: {
    deepseek: { type: "chat_template_kwargs", parameter: "thinking" }
    qwen3:    { type: "chat_template_kwargs", parameter: "enable_thinking" }
    gpt:      { type: "reasoning_effort",     parameter: "reasoning_effort" }
  }

  prompt_guard: {
    enabled: true
    threshold: 0.7
    use_mmbert_32k: true
    model_id: "models/mmbert32k-jailbreak-detector-merged"
  }

  hallucination_mitigation: {
    enabled: false
    fact_check_model: {
      model_id: "models/mmbert32k-factcheck-classifier-merged"
      threshold: 0.6
      use_mmbert_32k: true
    }
  }

  classifier: {
    category_model: {
      model_id: "models/mmbert32k-intent-classifier-merged"
      use_mmbert_32k: true
      threshold: 0.5
      category_mapping_path: "models/mmbert32k-intent-classifier-merged/category_mapping.json"
    }
    pii_model: {
      model_id: "models/mmbert32k-pii-detector-merged"
      use_mmbert_32k: true
      threshold: 0.9
      pii_mapping_path: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json"
    }
  }

  observability: {
    metrics: { enabled: true }
    tracing: {
      enabled: true
      provider: "opentelemetry"
      exporter: {
        type: "otlp"
        endpoint: "jaeger:4317"
        insecure: true
      }
      sampling: { type: "always_on", rate: 1.0 }
      resource: {
        service_name: "vllm-sr"
        service_version: "v0.1.0"
        deployment_environment: "development"
      }
    }
  }

  authz: {
    fail_open: false
    identity: {
      user_id_header: "x-authz-user-id"
      user_groups_header: "x-authz-user-groups"
    }
  }

  ratelimit: {
    fail_open: false
    providers: [
      {
        type: "local-limiter"
        rules: [
          { name: "free-rpm", match: { group: "free-tier" }, requests_per_unit: 10, unit: "minute" }
        ]
      }
    ]
  }

  looper: {
    endpoint: "http://localhost:8899/v1/chat/completions"
    timeout_seconds: 1200
  }

  model_selection: {
    enabled: true
    method: "knn"
  }
}

```

This ~280-line DSL compiles to ~800+ lines of equivalent YAML.

---

## 4. Visual Builder: Screen-by-Screen Design

### Screen 1: Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│  vLLM Semantic Router ─ Config Builder                    [▼ Visual]   │
│─────────────────────────────────────────────────────────────────────────│
│                                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  Signals   │  │   Routes   │  │  Backends  │  │  Health    │       │
│  │    15      │  │     8      │  │     6      │  │   ✅ OK    │       │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘       │
│                                                                         │
│  Quick Actions:                                                         │
│  [+ New Signal] [+ New Route] [🗣 Natural Language] [📥 Import YAML]   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Route Flow Graph                              │   │
│  │                                                                  │   │
│  │  User Query ──▶ [Signals] ──▶ [Decision Engine] ──▶ [Models]   │   │
│  │                    │              │                     │        │   │
│  │         keyword ───┤     math ────┤          qwen2.5 ──┤        │   │
│  │       embedding ───┤   physics ───┤           qwen3 ───┤        │   │
│  │          domain ───┤   urgent ────┤        gpt-4o ─────┤        │   │
│  │         context ───┤  premium ────┤                     │        │   │
│  │                    │              │                     │        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Mode: [Visual ●] [DSL] [Natural Language]                              │
└─────────────────────────────────────────────────────────────────────────┘

```

**Components:**

- Stats cards: signal count, route count, backend count, health status.
- Quick actions: create signal, route, NL mode, import YAML.
- Route Flow Graph: interactive React Flow diagram showing `User Query → Signals → Decisions → Models`.
- Mode toggle bar (persistent across all screens).

### Screen 2: Signal Editor

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Signal Editor                                          [+ Add Signal]  │
│─────────────────────────────────────────────────────────────────────────│
│                                                                         │
│  Signal Type: [▼ keyword ────────]     Name: [urgent_request_______]   │
│                                                                         │
│  ┌─ keyword fields ──────────────────────────────────────────────────┐ │
│  │  Method:    [▼ regex]  [bm25]  [ngram]                            │ │
│  │  Operator:  [▼ any]    [all]                                      │ │
│  │  Keywords:  [urgent] [asap] [emergency] [+ add]                   │ │
│  │  Case Sensitive: [ ] No                                           │ │
│  │  Fuzzy Match:    [✓] Yes   Threshold: [2___]                     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ── Signal List ─────────────────────────────────────────────────────  │
│  │ keyword    │ urgent_request  │ regex, 3 keywords         │ [Edit] │ │
│  │ embedding  │ ai_topics       │ threshold: 0.75, 4 cands  │ [Edit] │ │
│  │ domain     │ math            │ mmlu: [math]              │ [Edit] │ │
│  │ domain     │ physics         │ mmlu: [physics]           │ [Edit] │ │
│  │ context    │ long_context    │ 4K - 32K tokens           │ [Edit] │ │
│  │ complexity │ code_complexity │ threshold: 0.1            │ [Edit] │ │
│  │ authz      │ premium_binding │ role: premium_tier        │ [Edit] │ │
│  │ language   │ zh              │ Chinese                   │ [Edit] │ │
└─────────────────────────────────────────────────────────────────────────┘

```

**Dynamic fields by signal type:**

| Type | Fields |
|:---|:---|
| `keyword` | method (regex/bm25/ngram), operator, keywords list, case_sensitive, fuzzy_match, fuzzy_threshold, bm25_threshold, ngram_threshold, ngram_arity |
| `embedding` | candidates list, threshold slider (0–1), aggregation_method (mean/max/any) |
| `domain` | description, MMLU categories (multi-select), model_scores |
| `authz` | subjects list (kind: User/Group + name), role name |
| `language` | ISO code (auto-suggest from known codes), description |
| `context` | min_tokens, max_tokens (supports K/M suffixes) |
| `complexity` | threshold, hard candidates, easy candidates, composer (optional signal filter) |
| `fact_check` | description |
| `user_feedback` | description |
| `preference` | description |
| `modality` | description |

### Screen 3: Route Editor

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Route Editor: math_decision                         [Priority: 100]   │
│─────────────────────────────────────────────────────────────────────────│
│                                                                         │
│  ┌─ 1. Expression Builder ───────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  WHEN:  domain("math")  [▼ AND]  complexity("code_complexity")    │ │
│  │                                                                    │ │
│  │  [+ Add Condition]  [🔲 Full-Screen Expression Builder]           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ 2. Models ───────────────────────────────────────────────────────┐ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │ "qwen2.5:3b"  reasoning: ✓  effort: high  param_size: 3b  │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │  [+ Add Model]                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ 3. Algorithm (when 2+ models) ───────────────────────────────────┐ │
│  │  Type: [▼ confidence] method: hybrid  threshold: 0.5             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ 4. Plugins ──────────────────────────────────────────────────────┐ │
│  │  [✓] system_prompt  "You are a mathematics expert..."            │ │
│  │  [✓] pii            (template: safe_pii)                         │ │
│  │  [ ] jailbreak       —                                            │ │
│  │  [ ] semantic_cache   —                                           │ │
│  │  [ ] hallucination    —                                           │ │
│  │  [ ] memory           —                                           │ │
│  │  [ ] rag              —                                           │ │
│  │  [ ] router_replay    —                                           │ │
│  │  [ ] header_mutation   —                                          │ │
│  │  [ ] image_gen         —                                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

```

**Four sections:**

1. **Expression Builder** — visual signal nodes + logic operators (AND/OR/NOT). Links to full-screen canvas.
2. **Model List** — select models, configure reasoning, effort, LoRA, param_size.
3. **Algorithm Config** — appears only when 2+ models are added. Dropdown for algorithm type with dynamic parameter form.
4. **Plugin Toggle Panel** — toggle switches for each plugin type, with inline config or template reference.

### Screen 4: Expression Builder (Full-Screen Canvas)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Expression Builder: urgent_ai_route               [Done] [Cancel]     │
│─────────────────────────────────────────────────────────────────────────│
│                                                                         │
│       ┌───────────┐                                                    │
│       │  keyword   │                                                   │
│       │  urgent_   ├────────┐                                          │
│       │  request   │        │                                          │
│       └───────────┘        ▼                                           │
│                       ┌─────────┐      ┌───────────┐                   │
│       ┌───────────┐   │         │      │   NOT     │                   │
│       │ embedding  ├──▶│   AND   │◀─────│           │                   │
│       │ ai_topics  │   │         │      │  ┌──────┐ │                   │
│       └───────────┘   └────┬────┘      │  │domain│ │                   │
│                            │           │  │other │ │                   │
│                            ▼           │  └──────┘ │                   │
│                       [OUTPUT]         └───────────┘                   │
│                                                                         │
│  ── Palette ──────────────────────────────────────────────────────────  │
│  Signals:  [keyword ▪] [embedding ▪] [domain ▪] [context ▪] ...       │
│  Logic:    [AND ▪] [OR ▪] [NOT ▪]                                      │
│                                                                         │
│  Validation:                                                            │
│  ✅ All signal references are defined                                   │
│  ✅ NOT nodes have exactly 1 child                                      │
│  ✅ Expression is not empty                                             │
└─────────────────────────────────────────────────────────────────────────┘

```

**Interaction:**

- Drag signal nodes and logic operators from the palette onto the canvas.
- Connect nodes via wires (React Flow edges).
- Real-time validation: NOT must have exactly 1 child; signal names must exist; expression non-empty.
- Compiles to: `keyword("urgent_request") AND embedding("ai_topics") AND NOT domain("other")`

Which compiles to the YAML RuleNode tree:

```yaml
rules:
  operator: "AND"
  conditions:

    - type: "keyword"

      name: "urgent_request"

    - type: "embedding"

      name: "ai_topics"

    - operator: "NOT"

      conditions:

        - type: "domain"

          name: "other"

```

### Screen 5: Backend Configuration

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Backend Configuration                                                  │
│─────────────────────────────────────────────────────────────────────────│
│                                                                         │
│  [vLLM Endpoints] [Provider Profiles] [Embedding] [Cache] [Memory]     │
│                                                                         │
│  ── vLLM Endpoints ──────────────────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Name:    [ollama_________]       Type: [▼ ollama]               │  │
│  │ Address: [127.0.0.1______]       Port: [11434]                  │  │
│  │ Weight:  [1__]                   API Key: [________] (optional) │  │
│  │ Provider Profile: [▼ none]                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  [+ Add Endpoint]                                                       │
│                                                                         │
│  ── Embedding Models ────────────────────────────────────────────────  │
│  │ mmbert_model_path: models/mom-embedding-ultra                    │  │
│  │ use_cpu: ✓                                                       │  │
│  │ HNSW: model_type=mmbert  dimension=768  preload=✓               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

```

**Tab-based navigation:** Each backend type gets its own tab with type-specific fields.

### Screen 6: Global Settings

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Global Settings                                                        │
│─────────────────────────────────────────────────────────────────────────│
│                                                                         │
│  ── Routing ─────────────────────────────────────────────────────────  │
│  Default Model:     [▼ qwen2.5:3b]                                     │
│  Strategy:          [▼ priority]  [confidence]                         │
│  Reasoning Effort:  [▼ low]  [medium]  [high]                          │
│                                                                         │
│  ── Security ────────────────────────────────────────────────────────  │
│  Prompt Guard:           [✓] Enabled    Threshold: [0.7]               │
│  Hallucination:          [ ] Disabled                                   │
│  Authz Fail Open:        [ ] No (fail-closed)                          │
│  Rate Limit:             [✓] Enabled                                   │
│                                                                         │
│  ── Observability ───────────────────────────────────────────────────  │
│  Metrics:    [✓] Enabled                                               │
│  Tracing:    [✓] Enabled    Provider: [▼ opentelemetry]                │
│  Exporter:   [▼ otlp]      Endpoint: [jaeger:4317]                    │
│  Sampling:   [▼ always_on]  Rate: [1.0]                               │
│                                                                         │
│  ── Model Selection ─────────────────────────────────────────────────  │
│  Enabled: [✓]   Method: [▼ knn]                                       │
│  Looper Endpoint: [http://localhost:8899/v1/chat/completions]          │
└─────────────────────────────────────────────────────────────────────────┘

```

### Screen 7: Natural Language Mode

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🗣 Natural Language Mode                                               │
│─────────────────────────────────────────────────────────────────────────│
│                                                                         │
│  Describe what you want:                                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ I need a medical consultation platform that:                      │ │
│  │ - Routes health questions to a medical-specialized model          │ │
│  │ - Enables PII protection for all patient data                     │ │
│  │ - Uses semantic caching for common health queries                 │ │
│  │ - Falls back to a general assistant for non-medical topics        │ │
│  │ - Uses premium large models for authenticated premium users       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│  [Generate Config]                                                      │
│                                                                         │
│  ── Generated DSL ───────────────────────────────────────────────────  │
│  │ SIGNAL domain health {                                           │  │
│  │   description: "Health and medical queries"                      │  │
│  │   mmlu_categories: ["health"]                                    │  │
│  │ }                                                                │  │
│  │ ...                                                              │  │
│  │ ROUTE health_decision {                                          │  │
│  │   PRIORITY 100                                                   │  │
│  │   WHEN domain("health")                                          │  │
│  │   MODEL "medical-llm:7b" (reasoning = false)                    │  │
│  │   PLUGIN safe_pii                                                │  │
│  │   ...                                                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  [Accept & Edit in Visual Mode]  [Accept & Edit in DSL Mode]  [Retry]  │
└─────────────────────────────────────────────────────────────────────────┘

```

**Flow:**

1. User types natural language description.
2. Fine-tuned LLM generates DSL code.
3. DSL is parsed → AST (validated by WASM compiler).
4. User clicks "Accept & Edit" → populates Visual/DSL mode for fine-tuning.

**Data flow:** `NL → LLM API → DSL text → Parser → AST → Visual Builder`

---

## 5. DSL Text Editor (Monaco)

Built on **Monaco Editor** (VS Code core) with custom language support:

### 5.1 Features

| Feature | Implementation |
|:---|:---|
| **Syntax highlighting** | Custom Monarch tokenizer: keywords (blue), signal types (green), strings (orange), numbers (purple), operators (red) |
| **Auto-completion** | Context-aware: after `SIGNAL` → suggest types; after `WHEN` → suggest defined signal names; after `PLUGIN` → suggest plugin types and templates; after `ALGORITHM` → suggest algorithm types |
| **Error diagnostics** | Real-time red squiggly lines for parse errors; yellow for undefined references |
| **Hover info** | Hover signal name → show definition; hover field → show type and constraints |
| **Go to definition** | Ctrl+Click on signal reference in `WHEN` → jump to `SIGNAL` declaration |
| **Code folding** | Fold `SIGNAL`, `ROUTE`, `PLUGIN`, `BACKEND`, `GLOBAL` blocks |
| **Snippets** | `sig-kw` → keyword signal template; `sig-emb` → embedding template; `route` → route template; `plug-rag` → RAG plugin template |

### 5.2 Auto-Completion Triggers

```
SIGNAL |          → keyword, embedding, domain, fact_check, user_feedback,
                    preference, language, context, complexity, modality, authz

WHEN |            → domain("..."), keyword("..."), embedding("..."), ...
                    (lists all defined signal names by type)

WHEN ... | ...    → AND, OR
                    NOT (only at start of expression or after "(")

MODEL |           → (lists all models from BACKEND vllm_endpoint + provider_profiles)

ALGORITHM |       → confidence, ratings, remom, elo, router_dc, automix,
                    hybrid, rl_driven, gmtrouter, latency_aware, knn, kmeans, svm

PLUGIN |          → (lists all PLUGIN templates) + inline types:
                    jailbreak, pii, semantic_cache, memory, system_prompt,
                    header_mutation, hallucination, router_replay, rag, image_gen

BACKEND |         → vllm_endpoint, provider_profile, embedding_model,
                    semantic_cache, memory, response_api, vector_store, image_gen_backend

```

---

## 6. Mode Switching & Data Flow

All three modes share the **DSL AST** (Zustand store) as the single source of truth. Mode switches are **lossless**.

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  Visual Mode ◄──── serialize ────► DSL AST ◄──── parse ────► DSL Mode
│       │                              │                          │
│       │         ┌────────────────────┘                          │
│       │         │                                                │
│       │         ▼                                                │
│       │  Signal Compiler (WASM)                                  │
│       │         │                                                │
│       │         ├──► config.yaml (read-only preview)             │
│       │         ├──► Kubernetes CRD                              │
│       │         ├──► Helm Values                                 │
│       │         └──► Validation Diagnostics                      │
│       │                                                          │
│       └─────────────── NL Mode (LLM → DSL → AST) ──────────────┘
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

```

### Switch behavior:

| Transition | Action |
|:---|:---|
| Visual → DSL | `AST.serialize()` → DSL text displayed in Monaco |
| DSL → Visual | `Parser.parse(dslText)` → AST; if parse error, stay in DSL mode with diagnostics |
| NL → Visual/DSL | LLM generates DSL → `Parser.parse()` → AST → populate target mode |
| Any → YAML | `Compiler.compile(AST)` → read-only YAML preview (always available) |

---

## 7. Validation UX

Validation runs through the **WASM-compiled Signal Compiler** in real-time, producing three severity levels:

### Level 1: Syntax Errors (🔴 Red)

Parse failures detected by the DSL parser.

```
Error: Expected '{' after signal name, found 'threshold'
  at line 5, column 12

  SIGNAL keyword urgent_request threshold: 0.7
                                ^^^^^^^^^
  [Fix: Add '{' before field declarations]

```

### Level 2: Reference Errors (🟡 Yellow)

Undefined or type-mismatch references detected during AST linking.

```
Warning: Signal 'domain("mathematics")' is not defined
  at line 15 in ROUTE math_decision

  WHEN domain("mathematics")
               ^^^^^^^^^^^^^
  Did you mean: domain("math") ?
  [Fix: Change to "math"]  [Add: Create signal domain "mathematics"]

```

### Level 3: Constraint Violations (🟠 Orange)

Schema constraint violations detected during compilation.

```
Constraint: similarity_threshold must be between 0.0 and 1.0, got 1.5
  at line 22 in PLUGIN default_cache

  similarity_threshold: 1.5
                        ^^^
  [Fix: Set to 1.0]

```

### Validation Panel

```
┌─ Validation ─────────────────────────────────────────────────┐
│  🔴 2 errors  🟡 1 warning  🟠 0 constraints                │
│                                                               │
│  🔴 Line 5:  Expected '{' after signal name          [Fix]   │
│  🔴 Line 12: Unknown algorithm type "confdence"      [Fix]   │
│  🟡 Line 22: Signal "math2" is not defined           [Add]   │
└───────────────────────────────────────────────────────────────┘

```

**Quick Fix buttons** apply the fix and re-validate.

---

## 8. Export Options

The Signal Compiler's emit stage supports multiple output formats:

| Format | Use Case | Action |
|:---|:---|:---|
| **YAML** (`config.yaml`) | Standard deployment | Download / Copy / Apply Live |
| **Kubernetes CRD** | K8s GitOps | Download / Push to Git |
| **Helm Values** | Helm Chart deployment | Download / Copy |
| **DSL** (`config.dsl`) | Version control & sharing | Download / Copy / Push to Git |

### Export Dialog

```
┌─ Export Configuration ────────────────────────────────────────┐
│                                                                │
│  Format: [● YAML] [○ K8s CRD] [○ Helm Values] [○ DSL]       │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ # Generated by vLLM Semantic Router Config Builder       │ │
│  │ # Source: config.dsl (280 lines → 820 lines YAML)       │ │
│  │                                                          │ │
│  │ semantic_cache:                                          │ │
│  │   enabled: true                                          │ │
│  │   backend_type: "memory"                                 │ │
│  │   ...                                                    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  [📥 Download] [📋 Copy] [🔄 Apply Live] [🐙 Push to Git]   │
└────────────────────────────────────────────────────────────────┘

```

---

## 9. Technology Stack

| Component | Technology | Rationale |
|:---|:---|:---|
| **UI Framework** | React + Next.js | SSR + client interactivity |
| **Component Library** | shadcn/ui + Tailwind CSS | Modern, accessible, customizable |
| **Code Editor** | Monaco Editor | VS Code-grade editing experience |
| **Flow Diagrams** | React Flow | Production-grade node graph for expression builder |
| **State Management** | Zustand | Lightweight, AST as single store |
| **Compiler Core** | Go → WebAssembly | Sub-ms compilation in browser; shared with backend |
| **LLM Inference** | API call to fine-tuned model | Powers NL mode |

### WASM Compiler Architecture

The Signal Compiler is written in Go and compiled to WebAssembly:

```go
// main_wasm.go
package main

import (
    "syscall/js"
    "encoding/json"
)

func compile(this js.Value, args []js.Value) interface{} {
    dslSource := args[0].String()

    // 1. Lex
    tokens, lexErr := Lex(dslSource)
    if lexErr != nil {
        return errorResult(lexErr)
    }

    // 2. Parse → AST
    ast, parseErr := Parse(tokens)
    if parseErr != nil {
        return errorResult(parseErr)
    }

    // 3. Validate
    diagnostics := Validate(ast)

    // 4. Emit YAML
    yamlBytes, emitErr := EmitYAML(ast)
    if emitErr != nil {
        return errorResult(emitErr)
    }

    result := CompileResult{
        YAML:        string(yamlBytes),
        Diagnostics: diagnostics,
        AST:         ast,
    }
    jsonBytes, _ := json.Marshal(result)
    return string(jsonBytes)
}

func main() {
    js.Global().Set("signalCompile", js.FuncOf(compile))
    select {} // Keep alive
}

```

**Build command:**

```bash
GOOS=js GOARCH=wasm go build -o signal-compiler.wasm ./cmd/wasm/

```

**Browser usage:**

```javascript
const go = new Go();
const result = await WebAssembly.instantiateStreaming(
    fetch('/signal-compiler.wasm'), go.importObject
);
go.run(result.instance);

// Compile DSL to YAML (sub-millisecond)
const output = JSON.parse(window.signalCompile(dslSource));
// output.yaml, output.diagnostics, output.ast

```

**Key advantage:** The same Go compiler logic runs both in the browser (WASM) and on the server, ensuring **identical compilation behavior** in all environments.

---

## 10. User Journeys

### Journey 1: New User (Visual Mode → 5 min)

```
Dashboard → [+ New Signal] → Signal Editor (select "domain", fill form)
         → [+ New Route]  → Route Editor (select signal, add model, toggle plugins)
         → Export → Download config.yaml → Deploy

```

### Journey 2: Power User (DSL Mode → 2 min)

```
Dashboard → Switch to [DSL] → Type DSL with auto-complete
         → Real-time YAML preview in split pane
         → Export → Copy YAML → Deploy

```

### Journey 3: Explorer (NL Mode → 3 min)

```
Dashboard → [🗣 Natural Language] → Describe requirements
         → LLM generates DSL → Review
         → [Accept & Edit in Visual Mode] → Fine-tune
         → Export → Deploy

```

### Journey 4: Iterating User (Import → Edit → Export)

```
Dashboard → [📥 Import YAML] → YAML decompiled to DSL → AST populated
         → Switch freely between Visual / DSL / NL modes
         → Modify routes, add signals, change algorithms
         → Export as YAML / K8s CRD / Helm Values

```

Mode switches are **lossless**: Visual ↔ DSL ↔ NL all share the same AST.

---

## 11. DSL → YAML Compilation Rules

### Signal Mapping

| DSL Construct | YAML Target |
|:---|:---|
| `SIGNAL keyword X { ... }` | `keyword_rules[]: { name: X, ... }` |
| `SIGNAL embedding X { ... }` | `embedding_rules[]: { name: X, ... }` |
| `SIGNAL domain X { ... }` | `categories[]: { name: X, ... }` |
| `SIGNAL fact_check X { ... }` | `fact_check_rules[]: { name: X, ... }` |
| `SIGNAL user_feedback X { ... }` | `user_feedback_rules[]: { name: X, ... }` |
| `SIGNAL preference X { ... }` | `preference_rules[]: { name: X, ... }` |
| `SIGNAL language X { ... }` | `language_rules[]: { name: X, ... }` |
| `SIGNAL context X { ... }` | `context_rules[]: { name: X, ... }` |
| `SIGNAL complexity X { ... }` | `complexity_rules[]: { name: X, ... }` |
| `SIGNAL modality X { ... }` | `modality_rules[]: { name: X, ... }` |
| `SIGNAL authz X { ... }` | `role_bindings[]: { name: X, ... }` |

### Route Mapping

| DSL Element | YAML Target |
|:---|:---|
| `ROUTE name { ... }` | `decisions[]: { name: ... }` |
| `PRIORITY N` | `decisions[].priority: N` |
| `WHEN bool_expr` | `decisions[].rules: RuleNode` (recursive tree) |
| `MODEL "m" (opts)` | `decisions[].modelRefs[]: { model: m, ... }` |
| `ALGORITHM type { ... }` | `decisions[].algorithm: { type: ..., ... }` |
| `PLUGIN name { ... }` | `decisions[].plugins[]: { type: ..., configuration: ... }` |

### Boolean Expression Compilation

```
DSL:   keyword("urgent") AND (domain("math") OR embedding("ai")) AND NOT domain("other")

YAML:
rules:
  operator: "AND"
  conditions:

    - type: "keyword"

      name: "urgent"

    - operator: "OR"

      conditions:

        - type: "domain"

          name: "math"

        - type: "embedding"

          name: "ai"

    - operator: "NOT"

      conditions:

        - type: "domain"

          name: "other"

```

### Plugin Template Expansion

When a route references a named plugin template:

```ruby
# Template definition
PLUGIN safe_pii pii {
  enabled: true
  pii_types_allowed: []
}

# Route usage (reference only)
ROUTE math_decision {
  ...
  PLUGIN safe_pii
}

# Route usage (override)
ROUTE health_decision {
  ...
  PLUGIN safe_pii {
    pii_types_allowed: ["AGE", "GPE"]
  }
}

```

Compiles to:

```yaml
# math_decision
plugins:

  - type: "pii"

    configuration:
      enabled: true
      pii_types_allowed: []

# health_decision (merged override)
plugins:

  - type: "pii"

    configuration:
      enabled: true
      pii_types_allowed: ["AGE", "GPE"]

```

---

## 12. Relationship to Dataset Pipeline

```
                    ┌──────────────────────┐
                    │     DSL Grammar      │
                    │ (human & LLM shared) │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                            │
                 ▼                            ▼
        ┌────────────────┐          ┌────────────────┐
        │ Signal Compiler │          │ Signal Compiler │
        │  (WASM/Server)  │          │   (Go/Server)   │
        └────────┬───────┘          └────────┬───────┘
                 │                            │
                 ▼                            ▼
        ┌────────────────┐          ┌────────────────┐
        │ Visual Builder  │          │ Dataset Pipeline│
        │  (Frontend)     │          │  (Backend/Train)│
        └────────────────┘          └────────────────┘

```

- **DSL Grammar** defines the shared language for both human-authored and LLM-generated configurations.
- **Signal Compiler** validates both human input (Visual Builder) and LLM output (Dataset Pipeline).
- **Visual Builder** is the human-facing frontend.
- **Dataset Pipeline** is the training-data backend — it uses the same DSL grammar to generate synthetic training data for the NL→DSL fine-tuned model.

**Conclusion:** Building the unified DSL and Signal Compiler is the foundational step that unlocks both the Visual Builder UI and the Dataset Pipeline for LLM fine-tuning.

---

## 13. Step-by-Step Implementation Plan

> **Philosophy:** 每个 Step 都有可独立验证的交付物。后续 Step 依赖前序 Step 的产出物。严格按顺序执行。

### Overview: Step 依赖图

```
Step 1: Token + Lexer
   │
   ▼
Step 2: AST + Parser
   │
   ▼
Step 3: Compiler (AST → RouterConfig)
   │
   ├──────────────────────────┐
   ▼                          ▼
Step 4: Emitters           Step 5: 3-Level Validator
(YAML/CRD/Helm)              │
   │                          │
   ├──────────────────────────┘
   ▼
Step 6: Decompiler (YAML → DSL)
   │
   ▼
Step 7: CLI Integration (`sr dsl compile`)
   │
   ▼
Step 8: WASM Build
   │
   ▼
Step 9: Frontend — Zustand Store + WASM Bridge
   │
   ├──────────────────────────┐
   ▼                          ▼
Step 10: DSL Mode          Step 11: Visual Mode
(Monaco Editor)            (7 Screens)
   │                          │
   ├──────────────────────────┘
   ▼
Step 12: Mode Switching (Visual ↔ DSL ↔ NL)
   │
   ▼
Step 13: NL Mode + Dataset Pipeline + Fine-tuning
   │
   ▼
Step 14: E2E Testing + CI/CD

```

---

### Step 1: Token 定义 + Lexer

**目标：** 将 DSL 源码文本转换为 Token 流。

**产出物：**

```
src/semantic-router/pkg/dsl/
├── token.go          # Token 类型枚举 + Token 结构体
├── lexer.go          # Lexer 实现
└── lexer_test.go     # 单元测试

```

**`token.go` 需定义的 Token 类型：**

| Category | Tokens |
|:---|:---|
| 关键词 | `SIGNAL`, `ROUTE`, `PLUGIN`, `BACKEND`, `GLOBAL`, `PRIORITY`, `WHEN`, `MODEL`, `ALGORITHM` |
| 布尔操作符 | `AND`, `OR`, `NOT` |
| 信号类型 | `keyword`, `embedding`, `domain`, `fact_check`, `user_feedback`, `preference`, `language`, `context`, `complexity`, `modality`, `authz` |
| 插件类型 | `jailbreak`, `pii`, `semantic_cache`, `memory`, `system_prompt`, `header_mutation`, `hallucination`, `router_replay`, `rag`, `image_gen` |
| 算法类型 | `confidence`, `ratings`, `remom`, `static`, `elo`, `router_dc`, `automix`, `hybrid`, `rl_driven`, `gmtrouter`, `latency_aware`, `knn`, `kmeans`, `svm` |
| 后端类型 | `vllm_endpoint`, `provider_profile`, `embedding_model`, `semantic_cache`, `memory`, `response_api`, `vector_store`, `image_gen_backend` |
| 字面量 | `STRING` (`"..."`), `INTEGER`, `FLOAT`, `BOOL` (`true`/`false`) |
| 标点 | `LBRACE` `{`, `RBRACE` `}`, `LPAREN` `(`, `RPAREN` `)`, `LBRACKET` `[`, `RBRACKET` `]`, `COLON` `:`, `COMMA` `,`, `EQUALS` `=` |
| 标识符 | `IDENT` (用户自定义的信号名、路由名等) |
| 注释 | `COMMENT` (`# ...`) — Lexer 中跳过 |
| 终止 | `EOF` |

**Token 结构体：**

```go
type Token struct {
    Type    TokenType
    Literal string
    Line    int
    Column  int
}

```

**Lexer 关键逻辑：**

- 跳过空白和 `# comment` 行
- 识别字符串字面量 `"..."` （支持转义 `\"`）
- 数值识别：整数 vs 浮点 (`123` vs `0.75`)
- 关键词/标识符区分：先扫描为 `IDENT`，再查 keyword lookup table 转换
- 每个 Token 记录 `(line, column)` 位置，供后续错误报告使用

**验证标准：**

- [ ] 将 §3.3 完整 DSL 示例作为输入，Lexer 输出正确的 Token 流
- [ ] 错误位置准确：非法字符报告正确的行号和列号
- [ ] 100% 测试覆盖所有 Token 类型

**预估工期：** 2 天

---

### Step 2: AST 定义 + Parser

**目标：** 将 Token 流解析为类型安全的抽象语法树 (AST)。

**依赖：** Step 1 (token.go, lexer.go)

**产出物：**

```
src/semantic-router/pkg/dsl/
├── ast.go            # AST 节点类型定义
├── parser.go         # 递归下降解析器
└── parser_test.go    # 单元测试

```

**AST 节点体系：**

```go
// 顶层程序
type Program struct {
    Signals  []*SignalDecl
    Routes   []*RouteDecl
    Plugins  []*PluginDecl
    Backends []*BackendDecl
    Global   *GlobalDecl
}

// SIGNAL
type SignalDecl struct {
    Type   string            // "keyword", "embedding", "domain", ...
    Name   string
    Fields map[string]Value  // 通用 field_assign
    Pos    Position
}

// ROUTE
type RouteDecl struct {
    Name        string
    Description string         // 可选 route_opts
    Priority    int
    When        BoolExpr       // 布尔表达式树
    Models      []*ModelRef
    Algorithm   *AlgoSpec      // 可选
    Plugins     []*PluginRef
    Pos         Position
}

// 布尔表达式 (递归树)
type BoolExpr interface{ boolExpr() }
type BoolAnd struct { Left, Right BoolExpr }
type BoolOr  struct { Left, Right BoolExpr }
type BoolNot struct { Expr BoolExpr }
type SignalRef struct {
    Type string  // "keyword", "domain", ...
    Name string  // 信号名
    Pos  Position
}

// MODEL 引用
type ModelRef struct {
    Model     string
    Reasoning *bool
    Effort    string
    LoRA      string
    ParamSize string
}

// ALGORITHM
type AlgoSpec struct {
    Type   string
    Fields map[string]Value
}

// PLUGIN (模板声明)
type PluginDecl struct {
    Name   string
    Type   string
    Fields map[string]Value
    Pos    Position
}

// PLUGIN (路由内引用)
type PluginRef struct {
    Name   string            // 模板名 或 内联类型名
    Fields map[string]Value  // 可选覆盖字段
}

// BACKEND
type BackendDecl struct {
    Type   string
    Name   string
    Fields map[string]Value
    Pos    Position
}

// GLOBAL
type GlobalDecl struct {
    Fields map[string]Value
    Pos    Position
}

// 通用值类型
type Value interface{ value() }
type StringValue  struct { V string }
type IntValue     struct { V int }
type FloatValue   struct { V float64 }
type BoolValue    struct { V bool }
type ArrayValue   struct { Items []Value }
type ObjectValue  struct { Fields map[string]Value }

```

**Parser 设计（递归下降）：**

```
parseProgram()
  → loop: peek token type
    → SIGNAL  → parseSignalDecl()
    → ROUTE   → parseRouteDecl()
    → PLUGIN  → parsePluginDecl()
    → BACKEND → parseBackendDecl()
    → GLOBAL  → parseGlobalDecl()
    → EOF     → return Program

parseRouteDecl()
  → expect ROUTE, IDENT, optional "(" opts ")"
  → expect "{"
  → expect PRIORITY, INTEGER
  → expect WHEN → parseBoolExpr()
  → expect MODEL → parseModelList()
  → optional ALGORITHM → parseAlgoSpec()
  → loop: PLUGIN → parsePluginRef()
  → expect "}"

parseBoolExpr()   → parseBoolOr()
parseBoolOr()     → parseBoolAnd() { "OR" parseBoolAnd() }
parseBoolAnd()    → parseBoolFactor() { "AND" parseBoolFactor() }
parseBoolFactor() → "NOT" parseBoolFactor()
                  | "(" parseBoolExpr() ")"
                  | parseSignalRef()
parseSignalRef()  → signal_type "(" signal_name ")"

```

**错误恢复策略：**

- 遇到解析错误时，跳至下一个顶层关键词 (`SIGNAL`/`ROUTE`/`PLUGIN`/`BACKEND`/`GLOBAL`)
- 收集所有错误而非第一个就 panic
- 每个错误携带 `Position{Line, Column}` 信息

**验证标准：**

- [ ] 解析 §3.3 完整 DSL 示例生成正确 AST
- [ ] `AST.String()` 方法可将 AST 还原为 DSL 文本（为 Step 6 的 decompiler 做准备）
- [ ] 布尔表达式优先级正确：`a AND b OR c` = `(a AND b) OR c`
- [ ] 错误恢复：故意在示例中插入多个语法错误，解析器报告全部错误

**预估工期：** 3 天

---

### Step 3: Compiler（AST → RouterConfig）

**目标：** 将 DSL AST 编译为现有 Go `RouterConfig` 结构体（`pkg/config/config.go`）。

**依赖：** Step 2 (ast.go, parser.go)

**产出物：**

```
src/semantic-router/pkg/dsl/
├── compiler.go       # AST → RouterConfig
└── compiler_test.go  # 编译正确性测试

```

**编译规则实现要点：**

1. **Signal → Config 映射**（按 §11 Signal Mapping 表）：

   ```go
   func (c *Compiler) compileSignals(signals []*SignalDecl) {
       for _, s := range signals {
           switch s.Type {
           case "keyword":
               c.config.KeywordRules = append(c.config.KeywordRules, buildKeywordRule(s))
           case "embedding":
               c.config.EmbeddingRules = append(c.config.EmbeddingRules, buildEmbeddingRule(s))
           case "domain":
               c.config.Categories = append(c.config.Categories, buildCategory(s))
           // ... 11 种信号类型
           }
       }
   }

   ```

2. **Route → Decision 映射**（最复杂部分）：
   - `WHEN bool_expr` → 递归编译为 `RuleNode` 树
   - `MODEL list` → `[]ModelReference`，映射 reasoning/effort/lora/param_size
   - `ALGORITHM spec` → `AlgorithmConfig`，按算法类型映射到对应子配置
   - `PLUGIN refs` → `[]PluginConfig`，需实现模板展开（引用 PLUGIN 模板 + merge 覆盖字段）

3. **Plugin 模板展开**：

   ```go
   func (c *Compiler) resolvePlugin(ref *PluginRef) PluginConfig {
       if tmpl, ok := c.pluginTemplates[ref.Name]; ok {
           // 深拷贝模板，合并覆盖字段
           cfg := deepCopy(tmpl)
           mergeFields(cfg, ref.Fields)
           return cfg
       }
       // 内联插件声明
       return buildInlinePlugin(ref)
   }

   ```

4. **Backend → Config 映射**：直接映射 vllm_endpoints、provider_profiles、embedding_models 等

5. **Global → 顶层 Config 字段映射**：展开到 `RouterConfig` 的各个顶层字段

**验证标准：**

- [ ] 将 §3.3 DSL 示例编译为 `RouterConfig`，与手写 `config/config.yaml` 加载的结果做 `reflect.DeepEqual` 比对
- [ ] Plugin 模板展开正确：`safe_pii` 被多个 Route 引用，各自独立
- [ ] Plugin 覆盖正确：`health_decision` 中覆盖 `similarity_threshold` 生效
- [ ] 布尔表达式编译正确：`AND(keyword, OR(domain, embedding), NOT(domain))` 结构正确

**预估工期：** 4 天

---

### Step 4: Emitters（多格式输出）

**目标：** 将 `RouterConfig` 输出为 YAML / K8s CRD / Helm Values 三种格式。

**依赖：** Step 3 (compiler.go)

**产出物：**

```
src/semantic-router/pkg/dsl/
├── emitter_yaml.go   # RouterConfig → config.yaml
├── emitter_crd.go    # RouterConfig → K8s CRD YAML
├── emitter_helm.go   # RouterConfig → Helm values YAML
└── emitter_test.go   # 输出格式测试

```

**实现要点：**

| Emitter | 方法 | 输出格式 |
|:---|:---|:---|
| YAML | 使用 `gopkg.in/yaml.v3` 的 `yaml.Marshal(routerConfig)` | 标准 `config.yaml` |
| K8s CRD | 包装为 `apiVersion: semantic-router.io/v1alpha1` + `kind: RouterConfig` + `spec: routerConfig` | K8s 自定义资源 |
| Helm Values | 提取为扁平的 Helm `values.yaml` 结构，key 路径用 `.` 分隔 | Helm chart values |

**K8s CRD 输出示例：**

```yaml
apiVersion: semantic-router.io/v1alpha1
kind: RouterConfig
metadata:
  name: my-router
  namespace: default
spec:
  # ... RouterConfig fields ...

```

**验证标准：**

- [ ] YAML emitter 的输出可被 `config.LoadConfig()` 成功加载
- [ ] CRD emitter 的输出可通过 `kubectl apply --dry-run=client` 验证
- [ ] 输出的 YAML 与项目现有 `config/config.yaml` 格式一致（字段顺序、缩进）

**预估工期：** 2 天

---

### Step 5: 3-Level Validator

**目标：** 实现 §7 定义的三级验证机制。

**依赖：** Step 2 (AST), Step 3 (Compiler)

**产出物：**

```
src/semantic-router/pkg/dsl/
├── validator.go      # 三级验证
└── validator_test.go # 验证测试

```

**三级验证实现：**

```go
type Diagnostic struct {
    Level   DiagLevel  // Error, Warning, Constraint
    Message string
    Pos     Position
    Fix     *QuickFix  // 可选修复建议
}

type DiagLevel int
const (
    DiagError      DiagLevel = iota  // 🔴 Level 1: 语法错误
    DiagWarning                       // 🟡 Level 2: 引用错误
    DiagConstraint                    // 🟠 Level 3: 约束违规
)

```

| Level | 检查内容 | 实现方式 |
|:---|:---|:---|
| **Level 1 (🔴)** | Token 不合法、括号不匹配、缺少必需字段 | Parser 阶段的错误收集（Step 2 已实现） |
| **Level 2 (🟡)** | ROUTE 中引用了未定义的 SIGNAL/PLUGIN/BACKEND | AST 遍历，构建符号表，检查所有引用 |
| **Level 3 (🟠)** | 阈值范围 (0.0-1.0)、priority ≥ 0、算法类型合法、必需字段缺失 | 约束规则引擎 + 复用现有 `pkg/config/validator.go` 的 IP 校验等 |

**Level 2 引用检查详细规则：**

```go
func (v *Validator) checkReferences(prog *Program) {
    // 构建符号表
    signalNames := map[string]map[string]bool{}  // type → {name → true}
    pluginNames := map[string]bool{}
    backendNames := map[string]map[string]bool{}

    // 检查每个 Route 中的引用
    for _, route := range prog.Routes {
        // WHEN 表达式中的信号引用
        walkBoolExpr(route.When, func(ref *SignalRef) {
            if !signalNames[ref.Type][ref.Name] {
                v.addDiag(DiagWarning, ref.Pos,
                    fmt.Sprintf("Signal '%s(\"%s\")' is not defined", ref.Type, ref.Name),
                    suggestSimilar(ref.Name, signalNames[ref.Type]))
            }
        })
        // PLUGIN 引用
        for _, p := range route.Plugins {
            if !pluginNames[p.Name] && !isInlinePluginType(p.Name) {
                v.addDiag(DiagWarning, p.Pos, ...)
            }
        }
    }
}

```

**Level 3 约束规则示例：**

```go
var constraintRules = []ConstraintRule{
    {Field: "threshold",            Min: 0.0, Max: 1.0},
    {Field: "similarity_threshold", Min: 0.0, Max: 1.0},
    {Field: "priority",             Min: 0},
    {Field: "port",                 Min: 1, Max: 65535},
    {Field: "fuzzy_threshold",      Min: 0},
    {Field: "ngram_arity",          Min: 1},
}

```

**验证标准：**

- [ ] 未定义信号引用 → 黄色警告 + "Did you mean?" 建议
- [ ] 阈值 > 1.0 → 橙色约束违规
- [ ] 缺少 `PRIORITY` 字段 → 红色语法错误
- [ ] 所有验证结果的 `Position` 信息准确

**预估工期：** 3 天

---

### Step 6: Decompiler（YAML → DSL）

**目标：** 将现有 `config.yaml` 反编译为 DSL 文本，支持存量配置迁移。

**依赖：** Step 2 (AST), Step 3 (Compiler), Step 4 (Emitters)

**产出物：**

```
src/semantic-router/pkg/dsl/
├── decompiler.go     # RouterConfig → AST → DSL text
└── decompiler_test.go

```

**反编译流程：**

```
config.yaml → config.LoadConfig() → RouterConfig → Decompiler → AST → Serializer → DSL text

```

**关键反编译逻辑：**

1. **RuleNode 树 → 布尔表达式**：

   ```go
   func decompileRuleNode(node *RuleNode) BoolExpr {
       if node.Operator == "AND" {
           return foldBoolExpr(&BoolAnd{}, node.Conditions)
       }
       if node.Operator == "OR" {
           return foldBoolExpr(&BoolOr{}, node.Conditions)
       }
       if node.Operator == "NOT" {
           return &BoolNot{Expr: decompileRuleNode(node.Conditions[0])}
       }
       return &SignalRef{Type: node.Type, Name: node.Name}
   }

   ```

2. **Plugin 去重提取模板**：扫描所有 decisions 的 plugins，找出相同配置的插件，自动提取为 `PLUGIN` 模板

3. **AST → DSL 文本序列化**：保持 §3.3 中的格式（注释分隔、缩进、空行）

**验证标准（Round-Trip Test）：**

```
config.yaml → LoadConfig → RouterConfig → Decompiler → DSL → Parser → AST → Compiler → RouterConfig₂
assert RouterConfig == RouterConfig₂  (reflect.DeepEqual)

```

- [ ] 对项目现有 `config/config.yaml` (574行) 做 round-trip 测试
- [ ] 对 `src/vllm-sr/cli/templates/router-defaults.yaml` (298行) 做 round-trip 测试
- [ ] 自动提取的 Plugin 模板数量合理

**预估工期：** 3 天

---

### Step 7: CLI 集成

**目标：** 将 DSL 编译器集成到现有 CLI，提供 `sr dsl` 子命令。

**依赖：** Step 1-6（全部核心编译器）

**产出物：**

```
src/semantic-router/cmd/main.go    # 添加 dsl 子命令
src/semantic-router/pkg/dsl/
└── cli.go                         # CLI 命令实现

```

**子命令设计：**

```bash
# 编译 DSL → YAML
sr dsl compile config.dsl -o config.yaml

# 编译 DSL → K8s CRD
sr dsl compile config.dsl --format crd -o router-config.yaml

# 编译 DSL → Helm Values
sr dsl compile config.dsl --format helm -o values.yaml

# 反编译 YAML → DSL
sr dsl decompile config.yaml -o config.dsl

# 验证 DSL（不输出，仅检查）
sr dsl validate config.dsl

# 格式化 DSL
sr dsl fmt config.dsl

```

**验证标准：**

- [ ] `sr dsl compile` 能编译 §3.3 完整示例
- [ ] `sr dsl decompile` 能反编译 `config/config.yaml`
- [ ] `sr dsl validate` 对错误 DSL 输出有意义的三级诊断
- [ ] `sr dsl fmt` 输出格式化的 DSL（统一缩进、空行、注释位置）
- [ ] 退出码：0 = 成功，1 = 有错误

**预估工期：** 2 天

---

### Step 8: WASM Build

**目标：** 将 Go DSL 编译器编译为 WebAssembly，供浏览器端使用。

**依赖：** Step 1-5（Lexer/Parser/Compiler/Emitter/Validator）

**产出物：**

```
src/semantic-router/cmd/wasm/
├── main_wasm.go      # WASM 入口，注册 JS 函数
├── Makefile           # WASM 构建脚本
└── wasm_test.go       # Node.js 环境测试

```

**WASM 暴露的 JS API：**

```javascript
// 完整编译：DSL → { yaml, crd, helm, diagnostics, ast }
window.signalCompile(dslSource: string): string  // JSON 结果

// 增量验证：仅验证不编译（更快，用于实时编辑）
window.signalValidate(dslSource: string): string  // JSON diagnostics

// 反编译：YAML → DSL
window.signalDecompile(yamlSource: string): string  // DSL text

// 格式化
window.signalFormat(dslSource: string): string  // formatted DSL

```

**构建脚本：**

```makefile
# cmd/wasm/Makefile
WASM_OUT = ../../dashboard/frontend/public/signal-compiler.wasm

.PHONY: build
build:
    GOOS=js GOARCH=wasm go build -o $(WASM_OUT) -ldflags="-s -w" .
    @echo "WASM size: $$(du -h $(WASM_OUT) | cut -f1)"
    cp "$$(go env GOROOT)/misc/wasm/wasm_exec.js" ../../dashboard/frontend/public/

```

**性能目标：**

- WASM 二进制大小 < 5MB（使用 `-ldflags="-s -w"` 裁剪）
- 编译 280 行 DSL → YAML < 5ms
- 验证（无编译） < 1ms

**验证标准：**

- [ ] WASM 在 Node.js 18+ 环境可加载并执行
- [ ] `signalCompile()` 输出与 Go 原生编译结果完全一致
- [ ] 性能满足上述目标

**预估工期：** 2 天

---

### Step 9: Frontend — Zustand Store + WASM Bridge

**目标：** 在现有 Dashboard 前端中建立 DSL AST 状态管理层和 WASM 编译器桥接层。

**依赖：** Step 8 (WASM build)

**技术栈适配：** 现有 Dashboard 使用 React 18 + Vite + TypeScript + ReactFlow，需新增 `zustand` 依赖。

**产出物：**

```
dashboard/frontend/
├── src/lib/
│   ├── wasm.ts             # WASM 加载器 + 编译器 bridge
│   ├── store.ts            # Zustand store（AST 为单一数据源）
│   └── types.ts            # DSL AST TypeScript 类型定义
└── public/
    ├── signal-compiler.wasm   # Step 8 产物
    └── wasm_exec.js           # Go WASM runtime

```

**Zustand Store 设计：**

```typescript
interface DSLStore {
  // State
  ast: Program | null;
  dslText: string;
  yamlPreview: string;
  diagnostics: Diagnostic[];
  mode: 'visual' | 'dsl' | 'nl';

  // Actions
  setDSLText: (text: string) => void;        // DSL Mode 编辑时
  setAST: (ast: Program) => void;             // Visual Mode 编辑时
  compile: () => Promise<void>;               // 触发 WASM 编译
  validate: () => Promise<Diagnostic[]>;      // 触发 WASM 验证
  switchMode: (mode: 'visual' | 'dsl' | 'nl') => void;
  importYAML: (yaml: string) => Promise<void>;  // YAML → DSL
  exportAs: (format: 'yaml' | 'crd' | 'helm' | 'dsl') => string;
}

```

**WASM Bridge (`wasm.ts`)：**

```typescript
let compilerReady = false;

export async function initCompiler(): Promise<void> {
  const go = new (window as any).Go();
  const result = await WebAssembly.instantiateStreaming(
    fetch('/signal-compiler.wasm'), go.importObject
  );
  go.run(result.instance);
  compilerReady = true;
}

export function compile(dsl: string): CompileResult {
  if (!compilerReady) throw new Error('Compiler not loaded');
  return JSON.parse((window as any).signalCompile(dsl));
}

```

**验证标准：**

- [ ] WASM 在 Vite dev server 中可正常加载
- [ ] `store.setDSLText(dsl)` → 自动触发编译 → `yamlPreview` 更新
- [ ] `store.importYAML(yaml)` → decompile → `dslText` + `ast` 更新
- [ ] `store.diagnostics` 实时反映三级验证结果

**预估工期：** 3 天

---

### Step 10: DSL Mode（Monaco Editor）

**目标：** 实现 §5 定义的 DSL 文本编辑器。

**依赖：** Step 9 (Store + WASM Bridge)

**新增依赖：** `monaco-editor`, `@monaco-editor/react`

**产出物：**

```
dashboard/frontend/src/
├── components/dsl/
│   ├── DSLEditor.tsx         # Monaco Editor 包装器
│   └── monaco-lang.ts        # 自定义语言定义（语法高亮 + 补全 + 诊断）
└── pages/dsl/
    └── DSLPage.tsx            # DSL Mode 页面（编辑器 + YAML 预览分屏）

```

**Monaco 语言注册：**

```typescript
// monaco-lang.ts
export const DSL_LANGUAGE_ID = 'signal-dsl';

export const monarchTokenizer: monaco.languages.IMonarchLanguage = {
  keywords: ['SIGNAL', 'ROUTE', 'PLUGIN', 'BACKEND', 'GLOBAL',
             'PRIORITY', 'WHEN', 'MODEL', 'ALGORITHM'],
  operators: ['AND', 'OR', 'NOT'],
  signalTypes: ['keyword', 'embedding', 'domain', 'fact_check', ...],

  tokenizer: {
    root: [
      [/#.*$/, 'comment'],
      [/"[^"]*"/, 'string'],
      [/\d+\.\d+/, 'number.float'],
      [/\d+/, 'number'],
      [/true|false/, 'keyword.boolean'],
      [/[a-zA-Z_]\w*/, {
        cases: {
          '@keywords': 'keyword',
          '@operators': 'operator',
          '@signalTypes': 'type',
          '@default': 'identifier',
        }
      }],
    ],
  },
};

```

**补全 (CompletionItemProvider)：**

- `SIGNAL |` → 提示信号类型列表
- `WHEN |` → 提示已定义的信号引用 `domain("math")`, `keyword("urgent")`
- `PLUGIN |` → 提示已定义的插件模板 + 内联类型
- `ALGORITHM |` → 提示算法类型
- `MODEL |` → 提示后端中定义的模型

**实时诊断 (CodeActionProvider)：**

- 每次编辑触发 debounce(300ms) → `store.validate()` → WASM 验证
- 将 `Diagnostic[]` 转换为 Monaco markers (红/黄/橙 squiggly lines)
- Quick Fix actions 映射到 `CodeAction`

**验证标准：**

- [ ] DSL 语法高亮正确（关键词蓝色、信号类型绿色、字符串橙色、数字紫色、操作符红色）
- [ ] 自动补全在所有上下文生效
- [ ] 实时诊断在编辑后 300ms 内显示
- [ ] Go to Definition: Ctrl+Click 信号引用 → 跳转到信号声明

**预估工期：** 4 天

---

### Step 11: Visual Mode（7 个屏幕）

**目标：** 实现 §4 定义的全部 7 个可视化编辑器屏幕。

**依赖：** Step 9 (Store + WASM Bridge)

**产出物：**

```
dashboard/frontend/src/
├── components/builder/
│   ├── Dashboard.tsx              # Screen 1: 总览 + 路由流程图
│   ├── SignalEditor.tsx           # Screen 2: 信号编辑器（动态表单）
│   ├── SignalForm.tsx             # 按信号类型渲染不同表单
│   ├── RouteEditor.tsx            # Screen 3: 路由编辑器
│   ├── ExpressionBuilder.tsx      # Screen 4: 全屏表达式画布 (ReactFlow)
│   ├── BackendConfig.tsx          # Screen 5: 后端配置 (Tab 式)
│   ├── GlobalSettings.tsx         # Screen 6: 全局设置
│   ├── NLMode.tsx                 # Screen 7: 自然语言模式 (Step 13 完善)
│   ├── PluginToggle.tsx           # 插件开关面板（Route 内复用）
│   ├── ModelSelector.tsx          # 模型选择器（Route 内复用）
│   └── ValidationPanel.tsx        # 三级验证结果面板
└── pages/builder/
    └── BuilderPage.tsx            # Visual Mode 主页面 + 侧边导航

```

**实现分步：**

| 子步骤 | 屏幕 | 复杂度 | 依赖 |
|:---|:---|:---|:---|
| 11a | Dashboard (Screen 1) | 中 | ReactFlow (已有依赖) |
| 11b | Signal Editor (Screen 2) | 中 | 动态表单按 11 种信号类型切换 |
| 11c | Route Editor (Screen 3) | 高 | 依赖 11d (表达式) + ModelSelector + PluginToggle |
| 11d | Expression Builder (Screen 4) | 高 | ReactFlow 节点画布 + 自定义节点 |
| 11e | Backend Config (Screen 5) | 中 | Tab 组件 + 表单 |
| 11f | Global Settings (Screen 6) | 低 | 纯表单 |
| 11g | NL Mode (Screen 7) | 低 (占位) | 完整实现在 Step 13 |

**Expression Builder (Screen 4) 详细设计：**

```typescript
// 自定义 ReactFlow 节点类型
const nodeTypes = {
  signalNode: SignalNode,    // 圆角矩形，显示类型+名称
  andNode: LogicGateNode,    // AND 门
  orNode: LogicGateNode,     // OR 门
  notNode: LogicGateNode,    // NOT 门（单输入）
  outputNode: OutputNode,    // 最终输出
};

// AST ↔ ReactFlow 双向转换
function astToFlow(expr: BoolExpr): { nodes: Node[], edges: Edge[] }
function flowToAST(nodes: Node[], edges: Edge[]): BoolExpr

```

**验证标准：**

- [ ] Dashboard 展示信号/路由/后端计数 + ReactFlow 路由流程图
- [ ] Signal Editor 按类型动态渲染正确的表单字段
- [ ] Route Editor 的表达式构建器可拖拽信号+逻辑门，连线生成正确布尔表达式
- [ ] 所有屏幕的编辑操作实时更新 Zustand Store → 触发 WASM 编译 → YAML 预览更新

**预估工期：** 8 天

---

### Step 12: Mode Switching（三模式无损切换）

**目标：** 实现 §6 定义的三模式无损切换。

**依赖：** Step 10 (DSL Mode), Step 11 (Visual Mode)

**产出物：**

```
dashboard/frontend/src/
├── components/
│   └── ModeSwitcher.tsx       # 模式切换栏（Visual / DSL / NL）
└── lib/
    └── serializer.ts          # AST → DSL 文本序列化器

```

**切换逻辑：**

| 转换 | 实现 |
|:---|:---|
| Visual → DSL | `serializer.serialize(store.ast)` → 更新 `dslText` |
| DSL → Visual | `wasm.compile(store.dslText)` → 如果有 🔴 错误，阻止切换并显示诊断；否则更新 `ast` |
| NL → Visual/DSL | LLM 生成 DSL → compile → 同上 |
| Any → YAML Preview | 始终可用，`store.yamlPreview` 实时同步 |

**关键 UX 细节：**

- 模式切换栏固定在页面顶部，所有屏幕可见
- DSL → Visual 切换时如有解析错误，弹出诊断面板，不强制切换
- 切换时保留 undo/redo 历史栈

**验证标准：**

- [ ] Visual → DSL → Visual 无损往返
- [ ] DSL 中引入语法错误 → 切换到 Visual 被阻止 → 显示诊断
- [ ] NL 生成的 DSL → 切换到 Visual → 正确渲染
- [ ] YAML 预览在所有模式下实时同步

**预估工期：** 3 天

---

### Step 13: NL Mode + Dataset Pipeline + Fine-tuning

**目标：** 实现 §4 Screen 7 的自然语言模式，以及训练数据生成管线。

**依赖：** Step 12 (Mode Switching)

**产出物：**

```
# 前端
dashboard/frontend/src/components/builder/
└── NLMode.tsx                    # 完整 NL Mode 实现

# 后端 API
dashboard/backend/handlers/
└── nl_generate.go                # NL → LLM → DSL API 端点

# Dataset Pipeline
src/vllm-sr/dataset/
├── dsl_generator.py              # 合成 DSL 配置生成器
├── nl_dsl_pairs.py               # NL↔DSL 训练对生成器
└── fine_tune.py                  # LLM 微调脚本

```

**NL Mode 流程：**

```
User NL input → POST /api/nl/generate → Backend → LLM API → DSL text
  → WASM validate → 如果有错误，LLM 自动修正（最多 3 次重试）
  → 返回 DSL + diagnostics → 用户 Accept → 切换到 Visual/DSL Mode

```

**Dataset Pipeline 设计：**

1. `dsl_generator.py`: 随机组合信号/路由/插件/后端/全局，生成合法 DSL
2. `nl_dsl_pairs.py`: 为每个生成的 DSL 用 LLM 生成对应的自然语言描述
3. `fine_tune.py`: 使用 `(NL, DSL)` 对微调开源 LLM（如 Qwen 系列）

**验证标准：**

- [ ] NL 输入 "我需要一个数学问题路由到推理模型，带 PII 保护" → 生成包含 `SIGNAL domain math` + `ROUTE` + `PLUGIN pii` 的合法 DSL
- [ ] 生成的 DSL 通过 WASM 验证无 🔴 错误
- [ ] Accept 后正确切换到 Visual/DSL Mode

**预估工期：** 6 天

---

### Step 14: E2E Testing + CI/CD

**目标：** 端到端测试 + 持续集成。

**依赖：** 所有前序 Step

**产出物：**

```
# Go 编译器集成测试
src/semantic-router/pkg/dsl/
├── integration_test.go    # 大型 round-trip 测试
└── testdata/
    ├── full_config.dsl    # §3.3 完整示例
    ├── full_config.yaml   # 期望输出
    ├── minimal.dsl        # 最小配置
    ├── errors.dsl         # 含各种错误的 DSL（测试诊断）
    └── edge_cases.dsl     # 边界情况

# 前端 E2E 测试 (Playwright，项目已有依赖)
dashboard/frontend/e2e/
├── dsl-editor.spec.ts     # DSL Mode 编辑 + 补全 + 诊断
├── visual-builder.spec.ts # Visual Mode 各屏幕操作
├── mode-switch.spec.ts    # 三模式切换往返
├── import-export.spec.ts  # YAML 导入 + 多格式导出
└── expression.spec.ts     # 表达式画布拖拽

# CI 配置
tools/
└── dsl-ci.mk              # DSL 相关 CI 目标

```

**CI 检查项：**

```makefile
# tools/dsl-ci.mk
.PHONY: dsl-test dsl-wasm-test dsl-lint

dsl-test:
    cd src/semantic-router && go test ./pkg/dsl/... -v -race -coverprofile=coverage.out
    @echo "Coverage:"
    @go tool cover -func=coverage.out | tail -1

dsl-wasm-test:
    cd src/semantic-router/cmd/wasm && make build
    node --experimental-wasm-modules test_wasm.mjs

dsl-lint:
    cd src/semantic-router && go vet ./pkg/dsl/...
    cd src/semantic-router && golangci-lint run ./pkg/dsl/...

dsl-e2e:
    cd dashboard/frontend && npx playwright test e2e/

```

**验证标准：**

- [ ] Go 编译器测试覆盖率 > 85%
- [ ] Round-trip 测试通过：`config.yaml → DSL → config.yaml₂` 等价
- [ ] Playwright E2E：Visual Mode 创建信号+路由 → 切换 DSL → 切换 Visual → 导出 YAML → 内容正确
- [ ] CI 全绿

**预估工期：** 4 天

---

### 总览：工期 & 里程碑

| Step | 名称 | 工期 | 累计 | 里程碑 |
|:---|:---|:---|:---|:---|
| 1 | Token + Lexer | 2d | 2d | |
| 2 | AST + Parser | 3d | 5d | |
| 3 | Compiler | 4d | 9d | |
| 4 | Emitters | 2d | 11d | |
| 5 | Validator | 3d | 14d | |
| 6 | Decompiler | 3d | 17d | **🏁 M1: CLI 可用** — `sr dsl compile/decompile/validate` |
| 7 | CLI Integration | 2d | 19d | |
| 8 | WASM Build | 2d | 21d | **🏁 M2: WASM 可用** — 浏览器可调用编译器 |
| 9 | Zustand + WASM Bridge | 3d | 24d | |
| 10 | DSL Mode (Monaco) | 4d | 28d | **🏁 M3: DSL Editor** — 语法高亮+补全+实时验证 |
| 11 | Visual Mode (7 screens) | 8d | 36d | **🏁 M4: Visual Builder** — 完整可视化编辑 |
| 12 | Mode Switching | 3d | 39d | **🏁 M5: 三模式切换** — Visual ↔ DSL ↔ NL 无损 |
| 13 | NL Mode + Fine-tuning | 6d | 45d | **🏁 M6: NL Mode** — 自然语言生成配置 |
| 14 | E2E + CI/CD | 4d | 49d | **🏁 M7: Production Ready** |

**总预估：~49 工作日 (≈10 周)**

### 建议的团队分工

| 角色 | Step | 备注 |
|:---|:---|:---|
| **Go 后端工程师** | 1-7, 8 | 编译器核心 + CLI + WASM |
| **前端工程师** | 9-12, 14 (前端 E2E) | 状态管理 + Monaco + Visual Builder + Mode Switch |
| **ML 工程师** | 13 | NL Mode + Dataset Pipeline + Fine-tuning |
| **全栈/QA** | 14 | 集成测试 + CI/CD |

如果是单人开发，建议优先完成 **Step 1-7（CLI 可用）**，这是最核心的交付，后续所有功能都建立在此基础上。
