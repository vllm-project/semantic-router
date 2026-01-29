# Online Learning for Semantic Router Signal Fine-Tuning

## Executive Summary

This document describes an online learning system that leverages **Router Replay** data and **PromptWise**-style cost-aware contextual bandits to continuously fine-tune semantic router signals. The system learns optimal routing decisions from production traffic without requiring manual labeling or periodic batch retraining.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ONLINE LEARNING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   User Request  │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SEMANTIC ROUTER                                    │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────────┐   │
│  │  Signal    │───▶│  Decision  │───▶│   Model    │───▶│    Response    │   │
│  │ Extraction │    │   Engine   │    │  Selection │    │   Generation   │   │
│  └────────────┘    └──────┬─────┘    └──────┬─────┘    └───────┬────────┘   │
│                           │                 │                   │            │
│                           ▼                 ▼                   ▼            │
│                    ┌──────────────────────────────────────────────┐          │
│                    │            ROUTER REPLAY STORE               │          │
│                    │  • Request/Response Bodies                   │          │
│                    │  • Matched Signals & Scores                  │          │
│                    │  • Selected Model & Decision                 │          │
│                    │  • Response Status & Latency                 │          │
│                    └──────────────────────────┬───────────────────┘          │
└───────────────────────────────────────────────┼──────────────────────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
              ┌─────────────────────┐                    ┌─────────────────────┐
              │   REWARD EXTRACTOR  │                    │   FEEDBACK SIGNALS  │
              │  • User Satisfaction│                    │  • Explicit Ratings │
              │  • Response Quality │                    │  • Regeneration     │
              │  • Cost Efficiency  │                    │  • Follow-up Type   │
              └──────────┬──────────┘                    └──────────┬──────────┘
                         │                                          │
                         └────────────────────┬─────────────────────┘
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │         PROMPTWISE LEARNER              │
                         │  ┌─────────────────────────────────┐    │
                         │  │  Cost-Aware Contextual Bandit   │    │
                         │  │  • UCB Exploration Bonus        │    │
                         │  │  • Logistic Regression Model    │    │
                         │  │  • Per-Route Success Estimator  │    │
                         │  └─────────────────────────────────┘    │
                         └──────────────────┬──────────────────────┘
                                            │
                                            ▼
                         ┌─────────────────────────────────────────┐
                         │         SIGNAL OPTIMIZER                │
                         │  • Update Embedding Thresholds          │
                         │  • Adjust Keyword Weights               │
                         │  • Tune Domain Confidence Scores        │
                         │  • Recalibrate Model Rankings           │
                         └──────────────────┬──────────────────────┘
                                            │
                                            ▼
                         ┌─────────────────────────────────────────┐
                         │      ROUTER CONFIG UPDATER              │
                         │  (Hot-reload updated parameters)        │
                         └─────────────────────────────────────────┘
```

---

## 2. Core Concepts

### 2.1 The Learning Problem

Traditional semantic routers use **static thresholds** and **fixed model rankings**:

| Component | Static Approach | Problem |
|-----------|-----------------|---------|
| Embedding threshold | `0.75` | One size doesn't fit all categories |
| Keyword weights | Equal | Some keywords are more predictive |
| Model scores | Manual | Doesn't adapt to changing model quality |
| Domain routing | Rule-based | Misses nuanced query patterns |

**Online learning** addresses these by treating each routing decision as a **contextual multi-armed bandit** problem where:

- **Context** = Query features + matched signals
- **Arms** = Available routing decisions (model + category combinations)
- **Reward** = User satisfaction + response quality - cost

### 2.2 PromptWise Adaptation

We adapt PromptWise's key innovations:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPTWISE CORE IDEAS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. COST-AWARE SELECTION                                        │
│     ┌─────────────────────────────────────────────────────┐     │
│     │  Utility(route) = P(success) - λ × Cost(model)      │     │
│     │                                                     │     │
│     │  Select route with highest: P̂(success)/Cost        │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                 │
│  2. MULTI-ASSIGNMENT PER QUERY (Escalation)                     │
│     ┌─────────────────────────────────────────────────────┐     │
│     │  Query → Try cheap model → If fail → Try expensive  │     │
│     │                                                     │     │
│     │  Total reward = max(rewards) - Σ costs              │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                 │
│  3. UCB EXPLORATION BONUS                                       │
│     ┌─────────────────────────────────────────────────────┐     │
│     │  UCB(route) = P̂(success) + α × √(1/visits)         │     │
│     │                                                     │     │
│     │  Explores under-visited routes while exploiting     │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Reward Signal Design

### 3.1 Multi-Dimensional Reward

Unlike simple binary rewards, we construct a **composite reward** from replay data:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REWARD COMPOSITION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   R_total = w₁·R_quality + w₂·R_satisfaction + w₃·R_efficiency          │
│                                                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│   │   R_quality     │  │ R_satisfaction  │  │  R_efficiency   │         │
│   ├─────────────────┤  ├─────────────────┤  ├─────────────────┤         │
│   │ • HTTP 200      │  │ • No regenerate │  │ • Low latency   │         │
│   │ • No truncation │  │ • No follow-up  │  │ • Cache hit     │         │
│   │ • Complete resp │  │   complaint     │  │ • Cheap model   │         │
│   │ • Valid JSON    │  │ • Positive      │  │   succeeded     │         │
│   │                 │  │   feedback      │  │                 │         │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                         │
│   Default weights: w₁=0.4, w₂=0.4, w₃=0.2                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Reward Extraction from Replay

| Replay Field | Reward Signal | Extraction Method |
|--------------|---------------|-------------------|
| `response_status` | Quality | 200 → +1.0, 4xx → -0.5, 5xx → -1.0 |
| `from_cache` | Efficiency | Cache hit with success → +0.2 bonus |
| `streaming` | Quality | Complete stream → +0.1 |
| `response_body` | Quality | Non-empty, valid JSON → +0.2 |
| `signals.user_feedback` | Satisfaction | "satisfied" → +1.0, "wrong_answer" → -1.0 |
| `selected_model` pricing | Efficiency | Normalized cost penalty |
| Follow-up query | Satisfaction | Detected dissatisfaction → -0.5 |

### 3.3 Implicit Feedback Detection

```
┌─────────────────────────────────────────────────────────────────┐
│              IMPLICIT FEEDBACK FROM CONVERSATION                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Conversation Window Analysis (using replay chain):             │
│                                                                 │
│  Query₁ ──▶ Response₁ ──▶ Query₂ (follow-up)                    │
│                              │                                  │
│                              ▼                                  │
│              ┌───────────────────────────────┐                  │
│              │  Feedback Detector Signals    │                  │
│              ├───────────────────────────────┤                  │
│              │ • "need_clarification" → 0.0  │                  │
│              │ • "satisfied" → +1.0          │                  │
│              │ • "want_different" → -0.3     │                  │
│              │ • "wrong_answer" → -1.0       │                  │
│              └───────────────────────────────┘                  │
│                                                                 │
│  No follow-up within session = implicit satisfaction (+0.5)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Context Representation

### 4.1 Feature Vector Construction

Each replay record is transformed into a context vector for the bandit:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CONTEXT FEATURE VECTOR                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Context x ∈ ℝᵈ where d = d_embed + d_signals + d_meta                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  x = [ query_embedding | signal_features | meta_features ]   │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ query_embedding │  │ signal_features │  │  meta_features  │          │
│  │   (768 dims)    │  │   (32 dims)     │  │   (16 dims)     │          │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤          │
│  │ From embedding  │  │ • Keyword match │  │ • Query length  │          │
│  │ model (Qwen3/   │  │   count & score │  │ • Has code      │          │
│  │ ModernBERT)     │  │ • Embedding     │  │ • Has math      │          │
│  │                 │  │   similarity    │  │ • Is question   │          │
│  │                 │  │ • Domain conf.  │  │ • Time of day   │          │
│  │                 │  │ • Fact-check    │  │ • User history  │          │
│  │                 │  │   probability   │  │   (if tracked)  │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Route (Arm) Representation

Each routing decision is an "arm" in the bandit:

```
Route = (Decision Name, Model, Category)

Example arms:
├── ("code_generation", "deepseek-coder-33b", "coding")
├── ("code_generation", "gpt-4o", "coding")  
├── ("general_chat", "llama-3.1-70b", "general")
├── ("math_reasoning", "qwen2.5-math-72b", "math")
└── ("creative_writing", "claude-3.5-sonnet", "creative")
```

---

## 5. Online Learning Algorithm

### 5.1 Algorithm Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│            SEMANTIC ROUTER ONLINE LEARNING (SROL) ALGORITHM             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Initialize:                                                            │
│    • θ_r = initial weights for each route r                             │
│    • V_r = identity matrix (for confidence bounds)                      │
│    • τ_exp = exploration rounds per route (e.g., 10)                    │
│    • λ = cost penalty parameter                                         │
│    • α = exploration bonus coefficient                                  │
│                                                                         │
│  For each replay record (or batch):                                     │
│                                                                         │
│    1. FEATURE EXTRACTION                                                │
│       ┌─────────────────────────────────────────────────────┐           │
│       │  x_t = extract_context(replay_record)               │           │
│       │  r_t = extract_reward(replay_record, next_records)  │           │
│       │  route_t = (decision, model, category)              │           │
│       └─────────────────────────────────────────────────────┘           │
│                                                                         │
│    2. MODEL UPDATE (for observed route)                                 │
│       ┌─────────────────────────────────────────────────────┐           │
│       │  θ_route ← LogisticRegression.update(x_t, r_t)      │           │
│       │  V_route ← V_route + x_t × x_t^T                    │           │
│       └─────────────────────────────────────────────────────┘           │
│                                                                         │
│    3. COUNTERFACTUAL ESTIMATION (for unobserved routes)                 │
│       ┌─────────────────────────────────────────────────────┐           │
│       │  For similar historical contexts, impute rewards    │           │
│       │  for routes not taken (importance weighted)         │           │
│       └─────────────────────────────────────────────────────┘           │
│                                                                         │
│    4. SIGNAL THRESHOLD UPDATE                                           │
│       ┌─────────────────────────────────────────────────────┐           │
│       │  If sufficient samples, update:                     │           │
│       │    • embedding_threshold[category]                  │           │
│       │    • keyword_weights[decision]                      │           │
│       │    • model_scores[decision][model]                  │           │
│       └─────────────────────────────────────────────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 UCB-Style Route Selection

During inference (real-time routing), the learner suggests optimal routes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ROUTE SELECTION AT INFERENCE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  For query q with context x:                                            │
│                                                                         │
│    For each candidate route r:                                          │
│                                                                         │
│      ┌─────────────────────────────────────────────────────┐            │
│      │  q̂_r = σ(θ_r · x)           # Predicted success    │            │
│      │                                                     │            │
│      │  bonus_r = α · √(x^T V_r^{-1} x)  # Exploration    │            │
│      │                                                     │            │
│      │  utility_r = (q̂_r + bonus_r) - λ · cost_r          │            │
│      └─────────────────────────────────────────────────────┘            │
│                                                                         │
│    Select: r* = argmax_r { utility_r }                                  │
│                                                                         │
│    If utility_r* ≤ 0:                                                   │
│      → Use default route (too risky/expensive)                          │
│                                                                         │
│    Escalation (if enabled):                                             │
│      If r* fails and budget remains:                                    │
│        → Select next best route by utility_r / cost_r                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Signal Fine-Tuning Mechanism

### 6.1 What Gets Tuned

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TUNABLE SIGNAL PARAMETERS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  EMBEDDING RULES                                                │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  embedding_rules:                                       │    │    │
│  │  │    - name: "code_generation"                            │    │    │
│  │  │      threshold: 0.72  ◀── LEARNED (was 0.75)            │    │    │
│  │  │      candidates: [...]                                  │    │    │
│  │  │      aggregation_method: "max"  ◀── Can switch to mean  │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  MODEL SCORES PER CATEGORY                                      │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  categories:                                            │    │    │
│  │  │    - name: "coding"                                     │    │    │
│  │  │      model_scores:                                      │    │    │
│  │  │        - model: "deepseek-coder"                        │    │    │
│  │  │          score: 0.91  ◀── LEARNED (was 0.85)            │    │    │
│  │  │        - model: "gpt-4o"                                │    │    │
│  │  │          score: 0.88  ◀── LEARNED (was 0.95)            │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  KEYWORD WEIGHTS (implicit through learned θ)                   │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  Keyword "async" → coding route: weight 0.8             │    │    │
│  │  │  Keyword "explain" → general route: weight 0.6          │    │    │
│  │  │  Keyword "prove" → math route: weight 0.9               │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  DECISION PRIORITY (when multiple match)                        │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  decision: "code_generation"                            │    │    │
│  │  │    priority: 85  ◀── LEARNED (was 80)                   │    │    │
│  │  │  decision: "general_chat"                               │    │    │
│  │  │    priority: 50  ◀── LEARNED (was 60)                   │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Update Frequency & Safety

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SAFE UPDATE STRATEGY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    UPDATE TRIGGERS                              │    │
│  │                                                                 │    │
│  │  • Every N replay records (e.g., N=100)                         │    │
│  │  • Minimum time between updates (e.g., 5 minutes)               │    │
│  │  • Confidence threshold met (sufficient samples)                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    SAFETY GUARDRAILS                            │    │
│  │                                                                 │    │
│  │  1. Parameter Bounds                                            │    │
│  │     • Thresholds: [0.5, 0.95] (never too loose/strict)          │    │
│  │     • Model scores: [0.3, 1.0]                                  │    │
│  │     • Priorities: [0, 100]                                      │    │
│  │                                                                 │    │
│  │  2. Change Rate Limiting                                        │    │
│  │     • Max Δ per update: ±0.05 for thresholds                    │    │
│  │     • Exponential moving average smoothing                      │    │
│  │                                                                 │    │
│  │  3. Rollback Triggers                                           │    │
│  │     • If error rate increases >10% post-update                  │    │
│  │     • If latency increases >20% post-update                     │    │
│  │     • Automatic revert to previous parameters                   │    │
│  │                                                                 │    │
│  │  4. A/B Testing Mode                                            │    │
│  │     • 10% traffic uses learned parameters                       │    │
│  │     • 90% traffic uses stable parameters                        │    │
│  │     • Gradual rollout if improvements confirmed                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Flow & Processing Pipeline

### 7.1 Batch Processing Mode

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BATCH LEARNING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐       │
│   │ Replay  │      │ Reward  │      │ Bandit  │      │ Config  │       │
│   │  Store  │─────▶│ Compute │─────▶│  Train  │─────▶│ Update  │       │
│   │         │      │         │      │         │      │         │       │
│   └─────────┘      └─────────┘      └─────────┘      └─────────┘       │
│       │                                                    │            │
│       │              Every 15 minutes                      │            │
│       │                                                    ▼            │
│       │                                         ┌──────────────────┐    │
│       │                                         │  Hot Reload      │    │
│       │                                         │  Router Config   │    │
│       │                                         └──────────────────┘    │
│       │                                                                 │
│       └─── Retention: 7 days (for trend analysis)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

  Schedule:
  ┌────────────────────────────────────────────────────────────────────┐
  │  */15 * * * *  →  Process new replay records                       │
  │  0 */6 * * *   →  Full model retrain with all recent data          │
  │  0 0 * * 0     →  Weekly metrics report & parameter snapshot       │
  └────────────────────────────────────────────────────────────────────┘
```

### 7.2 Real-Time Learning Mode

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   STREAMING LEARNING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Replay Record                                                          │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     MESSAGE QUEUE                               │    │
│  │              (Redis Streams / Kafka / NATS)                     │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                     │
│         ▼                    ▼                    ▼                     │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │
│  │  Learner    │      │  Learner    │      │  Learner    │              │
│  │  Worker 1   │      │  Worker 2   │      │  Worker 3   │              │
│  │ (coding)    │      │ (general)   │      │ (math)      │              │
│  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘              │
│         │                    │                    │                     │
│         └────────────────────┼────────────────────┘                     │
│                              ▼                                          │
│                    ┌───────────────────┐                                │
│                    │  Parameter Store  │                                │
│                    │  (Redis/etcd)     │                                │
│                    └─────────┬─────────┘                                │
│                              │                                          │
│                              ▼                                          │
│                    ┌───────────────────┐                                │
│                    │   Router reads    │                                │
│                    │  updated params   │                                │
│                    │  (watch/poll)     │                                │
│                    └───────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Feedback Loop Integration

### 8.1 Conversation Chain Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│              FEEDBACK EXTRACTION FROM REPLAY CHAINS                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Session/Conversation Window:                                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  T=0:  User: "Write a Python function to sort a list"          │    │
│  │        → Route: code_generation, Model: deepseek-coder         │    │
│  │        → Replay ID: abc123                                      │    │
│  │                                                                 │    │
│  │  T=1:  Assistant: "def sort_list(lst): return sorted(lst)"     │    │
│  │        → Response captured in replay abc123                     │    │
│  │                                                                 │    │
│  │  T=2:  User: "That's not what I wanted, I need bubble sort"    │    │
│  │        → Feedback detector: "want_different"                    │    │
│  │        → NEGATIVE REWARD for replay abc123                      │    │
│  │                                                                 │    │
│  │  T=3:  User: "Never mind, use quicksort"                       │    │
│  │        → Route: code_generation, Model: deepseek-coder         │    │
│  │        → Replay ID: def456                                      │    │
│  │                                                                 │    │
│  │  T=4:  Assistant: "def quicksort(arr): ..."                    │    │
│  │                                                                 │    │
│  │  T=5:  User: "Perfect, thanks!"                                │    │
│  │        → Feedback detector: "satisfied"                         │    │
│  │        → POSITIVE REWARD for replay def456                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Linking Logic:                                                         │
│    • Same session/conversation_id within 10 minutes                     │
│    • Feedback message follows assistant response                        │
│    • Use Response API previous_response_id if available                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Explicit Feedback Collection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   OPTIONAL: EXPLICIT FEEDBACK UI                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Response Header:  X-VSR-Replay-ID: abc123                              │
│                                                                         │
│  Client can submit feedback via:                                        │
│                                                                         │
│    POST /v1/router_replay/abc123/feedback                               │
│    {                                                                    │
│      "rating": 4,           // 1-5 scale                                │
│      "correct": true,       // Was answer factually correct?            │
│      "helpful": true,       // Was it helpful?                          │
│      "model_fit": "good"    // Was this the right model?                │
│    }                                                                    │
│                                                                         │
│  This enriches the replay record and provides high-quality              │
│  reward signal for learning.                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Monitoring & Observability

### 9.1 Learning Metrics Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ONLINE LEARNING METRICS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  LEARNING PROGRESS                                               │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  • Samples processed: 125,432                              │  │   │
│  │  │  • Routes with sufficient data: 18/24                      │  │   │
│  │  │  • Avg prediction error (rolling): 0.18                    │  │   │
│  │  │  • Exploration rate: 12%                                   │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PARAMETER DRIFT                                                 │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  Route              | Threshold | Δ7d  | Trend             │  │   │
│  │  │  ─────────────────────────────────────────────────────────│  │   │
│  │  │  code_generation    | 0.72      | -0.03| ↘ (relaxing)      │  │   │
│  │  │  math_reasoning     | 0.81      | +0.02| ↗ (tightening)    │  │   │
│  │  │  creative_writing   | 0.68      | 0.00 | → (stable)        │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  COST-QUALITY TRADEOFF                                           │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  Avg cost per request: $0.0032 (↓12% from baseline)        │  │   │
│  │  │  Success rate: 94.2% (↑2.1% from baseline)                 │  │   │
│  │  │  Escalation rate: 8.3%                                     │  │   │
│  │  │  Cache hit rate: 23.1%                                     │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Alert Conditions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Success rate drop | >5% in 1 hour | Pause updates, alert |
| Cost increase | >20% in 1 hour | Review parameter changes |
| Exploration too high | >30% | Reduce α coefficient |
| Model score divergence | Variance >0.2 | Investigate data quality |
| Feedback detector accuracy | <70% | Retrain feedback model |

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1-2)

- [ ] Implement reward extractor from replay records
- [ ] Build context feature vector constructor  
- [ ] Set up parameter store (Redis/etcd)
- [ ] Create baseline metrics collection

### Phase 2: Learning Core (Week 3-4)

- [ ] Implement PromptWise-style bandit learner
- [ ] Add UCB exploration bonus calculation
- [ ] Build batch training pipeline
- [ ] Create A/B testing framework

### Phase 3: Signal Tuning (Week 5-6)

- [ ] Connect learner to embedding threshold updates
- [ ] Implement model score adjustment
- [ ] Add keyword weight learning
- [ ] Build hot-reload config updater

### Phase 4: Feedback Integration (Week 7-8)

- [ ] Implement conversation chain analysis
- [ ] Connect feedback detector signals
- [ ] Add explicit feedback API endpoint
- [ ] Build feedback quality monitoring

### Phase 5: Production Hardening (Week 9-10)

- [ ] Add safety guardrails and rollback
- [ ] Implement monitoring dashboard
- [ ] Performance optimization
- [ ] Documentation and runbooks

---

## 11. Expected Outcomes

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Routing accuracy | 85% | 92% | Correct model for task |
| Cost per request | $0.0042 | $0.0030 | Avg API cost |
| User satisfaction | 78% | 88% | Implicit + explicit feedback |
| Escalation rate | 15% | 8% | Cheap model success |
| Time to adapt | Manual | <1 hour | New model integration |

---

## References

1. PromptWise: Online Learning for Cost-Aware Prompt Assignment in Generative Models ([arXiv:2505.18901](https://arxiv.org/abs/2505.18901))
2. RouteLLM: Learning to Route LLMs with Preference Data ([arXiv:2406.18665](https://arxiv.org/abs/2406.18665))
3. AutoMix: Automatically Mixing Language Models ([arXiv:2310.12963](https://arxiv.org/abs/2310.12963))
4. Contextual Bandits with Linear Payoff Functions (Li et al., 2010)
