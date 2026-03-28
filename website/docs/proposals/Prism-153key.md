# [Proposal] PRISM — 153-key Legitimacy Layer for vLLM-SR Model Selection

**Issue:** #1422  
**Author:** Mossaab Souaissa — MSIA Systems  
**Milestone:** v0.3 — Themis  
**Reference:** https://doi.org/10.5281/zenodo.18750029  
**White Paper:** https://github.com/user-attachments/files/25750911/PRISM-Vllm-SR-whitepaper-COMPLET-EN.pdf

---

## 1. Context & Motivation

vLLM-SR answers: **which model is best suited for this request?**  
PRISM answers: **is the selected model legitimate to respond to this specific query?**

These are two different questions. They are complementary, not redundant.

### The lying model problem

Without a structural constraint, any model can respond to any query — even outside its training domain. This produces **confident hallucinations**: the model improvises answers with high confidence in a domain it does not master.

The current vLLM-SR pipeline has no post-selection legitimacy verification. PRISM adds this layer without modifying existing routing logic.

### Design principle: add without breaking

PRISM integrates as **optional extensions** of existing components:
 
- If `prism.enabled` is absent from a model's config → existing behavior **unchanged**
- If no `type: "prism-execution"` plugin block in a decision → Key 3 **skipped**
- If PRISM registry is not ready at request time → fallback to `"general"` → standard vLLM-SR routing

| PRISM component | Integration point | Type |
|----------------|------------------|------|
| Key 1 — QUALIFICATION | `model_config` in `config.yaml` | `prism.enabled: true` + auto-discovery at startup |
| Key 2 — CLASSIFICATION | `req_filter_classification.go` | New evaluation block — in-process `candle-binding` |
| Key 3 — EXECUTION | `req_filter_prism_execution.go` | New ExtProc filter — follows `req_filter_jailbreak.go` pattern |
| 153-Registry | `pkg/registry/prism_registry.go` | In-memory store, populated async at startup via Key 1 |

**Scope of this PR: hybrid mode only** (Key 1 + Key 2 + Key 3).  
`fine_filter` and `coarse_filter` are documented in §9 as future variants.

---

## 2. Architecture Overview

```
HTTP Request
     │
     ▼
ENVOY PROXY / API GATEWAY
     │
     ▼
processor_req_body.go
     │
     ├── PHASE 1 — Signal Extraction
     │       runClassification()              ← existing (keyword, embedding, domain...)
     │       runPrismClassification()         ← NEW Key 2
     │           writes: ctx.PrismDomain · ctx.PrismConfidence · ctx.PrismKeywords
     │           fallback: ctx.PrismDomain = "general" on any error or registry not ready
     │
     ├── PHASE 2 — Decision Engine
     │       evaluateDecisions()              ← existing, unchanged
     │           → candidates = modelRefs from matched decision
     │
     ├── PHASE 3 — Model Selection (pre-filter + re-route loop)
     │       filterCandidatesByPrism()        ← NEW — filter candidates by PrismDomain
     │       LOOP (max MaxRerouteAttempts+1):
     │           selectModelFromCandidates()  ← existing (Elo/AutoMix/MLP), unchanged
     │
     ├── PHASE 4 — Plugin Execution (inside loop)
     │       runSemanticCache()               ← existing
     │       runJailbreakFilter()             ← existing
     │       runPIIFilter()                   ← existing
     │       runPrismExecution()              ← NEW Key 3
     │           guard: decision.GetPluginConfig("prism-execution") required
     │           reads:  ctx.PrismDomain · ctx.PrismConfidence · ctx.SelectedModel
     │           writes: ctx.PrismRefused = true on failure
     │           loop exits if ctx.PrismRefused == false
     │           if all candidates refused → ctx.Blocked = true (HTTP response via buildResponse())
     │
     └── buildResponse() → ENVOY → CLIENT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STARTUP — once per server start (non-blocking)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NewOpenAIRouter(config)
  → NewPrismRegistry(config)          — returns immediately
      go qualifyAllModels()            — runs in background goroutine
          for each model with prism.enabled = true:
              send Key 1 QUALIFICATION prompt → parse JSON → register in 153-Registry
          registry.ready = true        — PRISM activates for all subsequent requests
```

---

## 3. Key 1 — QUALIFICATION (async auto-discovery at startup)

### 3.1 Principle

The human operator declares **only** `prism.enabled: true` in `model_config`. Nothing else.

At startup, `NewPrismRegistry()` launches a background goroutine that sends the Key 1 QUALIFICATION prompt to each enabled model, parses its self-declaration, and registers it in the 153-Registry. The router starts immediately — PRISM activates once the goroutine completes.

**During initialization:** `IsReady()` returns `false` → `runPrismClassification()` sets `ctx.PrismDomain = "general"` → standard vLLM-SR routing without PRISM.

### 3.2 config.yaml — minimal human declaration

```yaml
model_config:
  # Model without PRISM — existing behavior unchanged
  "gpt-4":
    preferred_endpoints: ["endpoint-cloud"]

  # Model with PRISM — human declares only enabled: true
  "slm-maintenance-v2":
    preferred_endpoints: ["endpoint1"]
    prism:
      enabled: true     # ← that's all the human needs to declare

  "slm-quality-v1":
    preferred_endpoints: ["endpoint1"]
    prism:
      enabled: true

  "slm-hr-v1":
    preferred_endpoints: ["endpoint2"]
    prism:
      enabled: true
```

### 3.3 Key 1 QUALIFICATION prompt (sent at startup to each model)

```
CRITICAL MISSION: SELF-EVALUATION OF YOUR EXPERTISE

YOU ARE A SPECIALIZED MODEL. This qualification determines your UNIQUE domain of expertise.

CONTEXT: You were created for a SPECIFIC domain.
Your training data, vocabulary, and capabilities are CONCENTRATED on ONE single domain.

ABSOLUTE RULE:
- Declare THE domain in which you are GENUINELY expert
- If you are a generalist: domain = "general"
- If no domain applies: domain = "unknown"

QUESTION FOR YOU:
What domain are you GENUINELY specialized in?
- What technical vocabulary do you master?
- What tasks can you accomplish with PRECISION?
- What are your technical LIMITS?

INSTRUCTIONS:
1. Examine your own vocabulary and knowledge
2. Declare your primary domain as a short slug (e.g. "industrial_maintenance")
3. List 2-5 tasks you can handle with confirmed expertise
4. List 1-3 domains you must systematically refuse
5. Score = depth of your expertise (0.00-1.00)

SCORING:
- 0.00-0.30: Superficial general knowledge
- 0.31-0.60: Specialized but basic vocabulary
- 0.61-0.90: Confirmed expertise with protocols and methods
- 0.91-1.00: Total mastery (standards, formulas, complex cases)

REQUIRED FORMAT (JSON on ONE line, no other text):
{"domain":"<your_domain_slug>","can_process":["task1","task2"],
 "must_refuse":["limit1","limit2"],"score":0.00}
```

**HTTP call:** `POST http://{VLLMEndpoint.Address}:{VLLMEndpoint.Port}/v1/chat/completions`  
**Auth:** `Authorization: Bearer {ModelParams.AccessKey}` (if AccessKey is set)  
**Timeout:** 30 seconds per model  
**On failure or domain = "general"/"unknown":** model excluded from registry with warning log

### 3.4 Go structs

```go
// pkg/config/config.go

// PrismModelConfig — minimal human declaration in config.yaml
type PrismModelConfig struct {
    Enabled bool `yaml:"enabled"`
}

// PrismThresholds — all scoring thresholds, fully configurable
type PrismThresholds struct {
    ConfidenceAutoAccept float64 `yaml:"confidence_auto_accept"` // default 0.90
    ConfidenceFlexible   float64 `yaml:"confidence_flexible"`    // default 0.70
    ConfidenceStrict     float64 `yaml:"confidence_strict"`      // default 0.40
    ExpertiseMinScore    float64 `yaml:"expertise_min_score"`    // default 0.61
    ScoreDeltaAccepted   float64 `yaml:"score_delta_accepted"`   // default 0.02
    ScoreDeltaRefused    float64 `yaml:"score_delta_refused"`    // default 0.03
}

// DefaultPrismThresholds — used when prism: block is absent from config.yaml
func DefaultPrismThresholds() PrismThresholds {
    return PrismThresholds{
        ConfidenceAutoAccept: 0.90,
        ConfidenceFlexible:   0.70,
        ConfidenceStrict:     0.40,
        ExpertiseMinScore:    0.61,
        ScoreDeltaAccepted:   0.02,
        ScoreDeltaRefused:    0.03,
    }
}

// PrismConfig — global PRISM configuration
type PrismConfig struct {
    Thresholds          PrismThresholds `yaml:"thresholds"`
    RerouteOnRefused    bool            `yaml:"reroute_on_refused"`
    MaxRerouteAttempts  int             `yaml:"max_reroute_attempts"`  // default 3
    UnregisteredPolicy  string          `yaml:"unregistered_policy"`   // "passthrough" | "refuse"
}

// Extend ModelParams — one optional field, nil = no PRISM for this model
type ModelParams struct {
    // ... existing fields unchanged ...
    PIIPolicy          PIIPolicy     `yaml:"pii_policy,omitempty"`
    ReasoningFamily    string        `yaml:"reasoning_family,omitempty"`
    PreferredEndpoints []string      `yaml:"preferred_endpoints,omitempty"`
    Pricing            ModelPricing  `yaml:"pricing,omitempty"`
    LoRAs              []LoRAAdapter `yaml:"loras,omitempty"`
    AccessKey          string        `yaml:"access_key,omitempty"`
    Prism              *PrismModelConfig `yaml:"prism,omitempty"` // ← NEW
}

// Extend RouterConfig — one optional field
type RouterConfig struct {
    // ... existing fields unchanged ...
    Prism *PrismConfig `yaml:"prism,omitempty"` // ← NEW
}
```

### 3.5 153-Registry — in-memory store with async init

```go
// pkg/registry/prism_registry.go — NEW FILE

type RegistryEntry struct {
    ModelName        string
    Domain           string
    DeclaredScore    float64  // from Key 1 — immutable
    EmpiricalScore   float64  // init = DeclaredScore, updated after each interaction
    CanProcess       []string
    MustRefuse       []string
    Status           string   // "active" | "suspended" | "banned"
    InteractionCount int64
    LastInteraction  string   // "ACCEPTED" | "REFUSED"
    DomainEmbedding  []float32 // pre-computed at registration — used by Key 2
}

type PrismRegistry struct {
    config      *config.RouterConfig
    ready       bool                        // false until qualifyAllModels() completes
    mu          sync.RWMutex
    byDomain    map[string][]*RegistryEntry // O(1) by domain — Key 2
    byModelName map[string]*RegistryEntry   // O(1) by model  — Key 3
}
// Both maps point to the same *RegistryEntry pointers — no data duplication

func NewPrismRegistry(config *config.RouterConfig) *PrismRegistry {
    r := &PrismRegistry{
        config:      config,
        ready:       false,
        byDomain:    make(map[string][]*RegistryEntry),
        byModelName: make(map[string]*RegistryEntry),
    }
    go r.qualifyAllModels() // non-blocking — router starts immediately
    return r
}

func (r *PrismRegistry) IsReady() bool                                     // thread-safe read
func (r *PrismRegistry) Register(entry *RegistryEntry)                     // called by qualifyAllModels
func (r *PrismRegistry) GetCandidatesForDomain(domain string) []*RegistryEntry
func (r *PrismRegistry) GetByModelName(modelName string) *RegistryEntry
func (r *PrismRegistry) UpdateEmpiricalScore(modelName string, accepted bool) // reads deltas from config.Prism.Thresholds
func (r *PrismRegistry) ListDomains() []string                             // used by Key 2 for embedding comparison
func FilterExcluded(candidates []string, excluded []string) []string       // package-level helper for re-route loop

// NewForTesting — creates empty registry for unit tests (do not use in production)
func NewForTesting(prismCfg *config.PrismConfig) *PrismRegistry
```

### 3.6 Scoring scale

| Score | Level | Meaning |
|-------|-------|---------|
| 0.00 - 0.30 | Superficial | General knowledge, no specialization |
| 0.31 - 0.60 | Basic | Specialized vocabulary, limited depth |
| 0.61 - 0.90 | Confirmed | Real expertise with protocols and methods |
| 0.91 - 1.00 | Mastery | Standards, formulas, complex cases |

### 3.7 Empirical score rules

```
init:    empirical_score = declared_score
ACCEPTED: empirical_score += Thresholds.ScoreDeltaAccepted (default +0.02)
REFUSED:  empirical_score -= Thresholds.ScoreDeltaRefused  (default -0.03)
clamp:   empirical_score ∈ [0.00, 1.00]

status:
  empirical_score >= 0.50  → "active"
  empirical_score  0.30-0.49 → "suspended" (excluded from candidates)
  empirical_score < 0.30   → "banned"    (removed from registry immediately)
```

---

## 4. Key 2 — CLASSIFICATION (new block in req_filter_classification.go)

### 4.1 Principle

Key 2 is a new evaluation block inside `req_filter_classification.go`.  
It runs **in-process** using `candle.GetEmbedding()` from the `candle-binding` package.  
Key 2 **never answers the question** — it classifies the intent only.  
Its result is written to `RequestContext` for use by Key 3 in Phase 4.

### 4.2 RequestContext extensions

```go
// pkg/extproc/router.go — extend RequestContext

type RequestContext struct {
    // ... existing fields unchanged (Headers, RequestID, SelectedModel,
    //     CacheHit, Blocked, BlockReason, ...) ...

    // PRISM fields — written by Key 2 in Phase 1, read by Key 3 in Phase 4
    PrismDomain          string    // domain from 153-Registry, or "general"
    PrismConfidence      float64   // cosine similarity score 0.00-1.00
    PrismKeywords        []string  // technical keywords extracted from query text
    PrismRefused         bool      // set by Key 3 on legitimacy failure
    PrismRefusedReason   string    // reason string from Key 3
    PrismRerouteAttempts int       // current re-route attempt count
    ExcludedModels       []string  // models excluded from re-route candidates
}
```

### 4.3 Embedding API — candle-binding (confirmed)

```go
// Package: github.com/vllm-project/semantic-router/candle-binding
// Import as: candle "github.com/vllm-project/semantic-router/candle-binding"
//
// CONFIRMED function — used directly, not via r.Classifier:
func GetEmbedding(text string, maxLength int) ([]float32, error)
// maxLength: 512 (standard BERT context)
```

### 4.4 Domain embedding cache

Domain embeddings are **pre-computed once** at the end of `qualifyAllModels()` and stored in each `RegistryEntry.DomainEmbedding`. This avoids recomputing domain embeddings on every request.

At runtime, Key 2 computes **1 embedding** (the query) and compares it against the cached domain embeddings — O(N) cosine comparisons where N = number of registered domains (typically 3-10 in industrial deployments).

### 4.5 Fallback rule — universal

```
Registry not ready (IsReady() = false)  → ctx.PrismDomain = "general"
candle.GetEmbedding() error             → ctx.PrismDomain = "general"
Best similarity < ConfidenceStrict      → ctx.PrismDomain = "general"
Unknown domain (not in registry)        → ctx.PrismDomain = "general"
ctx.PrismDomain = "general"             → filterCandidatesByPrism() skipped
                                        → Key 3 skipped (no domain to validate)
                                        → standard vLLM-SR routing
```

`"general"` is the universal fallback — graceful degradation, no error raised, no request blocked.

### 4.6 Confidence scale

| Confidence | Key 3 behavior |
|------------|---------------|
| < 0.40 (ConfidenceStrict) | Domain = "general" → Key 3 skipped |
| 0.40 - 0.69 | Key 3 strict: domain match + 2 keyword matches + empirical_score >= ExpertiseMinScore |
| 0.70 - 0.89 | Key 3 flexible: domain match + 1 keyword match |
| >= 0.90 | Key 3 auto-accept if domain matches Key 1 declaration |

### 4.7 Keywords extraction — independent from embedding

```go
// pkg/extproc/req_filter_prism_helpers.go

// extractKeywords — simple tokenizer, independent of candle-binding
// TODO v0.4: replace with NER-based extraction for better technical term detection
func extractKeywords(text string) []string {
    stopWords := map[string]bool{
        "the": true, "a": true, "an": true, "is": true, "are": true,
        "was": true, "what": true, "how": true, "why": true, "when": true,
        "for": true, "in": true, "on": true, "at": true, "to": true,
        "of": true, "and": true, "or": true, "but": true, "with": true,
    }
    words  := strings.Fields(strings.ToLower(text))
    result := make([]string, 0)
    seen   := make(map[string]bool)
    for _, w := range words {
        clean := strings.Trim(w, ".,;:!?\"'()")
        if len(clean) > 3 && !stopWords[clean] && !seen[clean] {
            result = append(result, clean)
            seen[clean] = true
        }
    }
    if len(result) > 10 { result = result[:10] }
    return result
}
```

---

## 5. Key 3 — EXECUTION (new ExtProc filter)

### 5.1 Principle

Key 3 is a new ExtProc filter method `runPrismExecution()` on `OpenAIRouter`, implemented in `pkg/extproc/req_filter_prism_execution.go`. It follows the **exact pattern** of `req_filter_jailbreak.go`.

It forces the selected model to confirm its legitimacy using the domain fixed by Key 1 at startup — not what the model claims at runtime.

### 5.2 Exact pattern to follow — req_filter_jailbreak.go

```go
// This is the exact pattern — Key 3 follows it verbatim
func (r *OpenAIRouter) runJailbreakFilter(ctx *RequestContext, decision *config.Decision) {
    if !decision.HasJailbreakPlugin() {
        return
    }
    result := r.JailbreakChecker.Check(ctx.RequestBody)
    if result.IsJailbreak {
        ctx.Blocked = true
        ctx.BlockReason = "jailbreak detected"
    }
}
```

### 5.3 Key 3 implementation

```go
// pkg/extproc/req_filter_prism_execution.go — NEW FILE

func (r *OpenAIRouter) runPrismExecution(ctx *RequestContext, decision *config.Decision) {
    // Guard 1: plugin must be declared in decision config
    if _, ok := decision.GetPluginConfig("prism-execution"); !ok {
        return
    }
    // Guard 2: skip if domain is general (no PRISM classification)
    if ctx.PrismDomain == "" || ctx.PrismDomain == "general" {
        return
    }
    // Guard 3: skip on cache hit — validation already cached
    if ctx.CacheHit {
        return
    }
    // Guard 4: skip if registry not initialized
    if r.PrismRegistry == nil || !r.PrismRegistry.IsReady() {
        return
    }

    entry := r.PrismRegistry.GetByModelName(ctx.SelectedModel)
    if entry == nil {
        r.handleUnregisteredModel(ctx)
        return
    }

    accepted := r.validatePrismLegitimacy(ctx, entry)

    if accepted {
        r.PrismRegistry.UpdateEmpiricalScore(ctx.SelectedModel, true)
        return
    }

    // REFUSED
    r.PrismRegistry.UpdateEmpiricalScore(ctx.SelectedModel, false)
    ctx.PrismRefused = true
    ctx.PrismRefusedReason = "domain legitimacy check failed"
}

// handleUnregisteredModel — behavior controlled by UnregisteredPolicy
func (r *OpenAIRouter) handleUnregisteredModel(ctx *RequestContext) {
    policy := "passthrough"
    if r.Config.Prism != nil && r.Config.Prism.UnregisteredPolicy == "refuse" {
        policy = "refuse"
    }
    if policy == "refuse" {
        ctx.PrismRefused = true
        ctx.PrismRefusedReason = "model not registered in PRISM 153-Registry"
    }
    // passthrough: log warning, continue normally
}

// validatePrismLegitimacy — 3-tier confidence-proportional validation
func (r *OpenAIRouter) validatePrismLegitimacy(ctx *RequestContext, entry *RegistryEntry) bool {
    // Domain must match (case-insensitive)
    if !strings.EqualFold(entry.Domain, ctx.PrismDomain) {
        return false
    }
    t := r.prismThresholds()
    switch {
    case ctx.PrismConfidence >= t.ConfidenceAutoAccept:
        return true
    case ctx.PrismConfidence >= t.ConfidenceFlexible:
        return r.countKeywordMatches(ctx.PrismKeywords, entry.CanProcess) >= 1
    case ctx.PrismConfidence >= t.ConfidenceStrict:
        return r.countKeywordMatches(ctx.PrismKeywords, entry.CanProcess) >= 2 &&
            entry.EmpiricalScore >= t.ExpertiseMinScore
    default:
        return false
    }
}

// countKeywordMatches — case-insensitive substring matching
func (r *OpenAIRouter) countKeywordMatches(keywords []string, canProcess []string) int {
    count := 0
    for _, kw := range keywords {
        for _, cp := range canProcess {
            if strings.Contains(strings.ToLower(cp), strings.ToLower(kw)) {
                count++
                break
            }
        }
    }
    return count
}
```

### 5.4 Shared helpers file

```go
// pkg/extproc/req_filter_prism_helpers.go — NEW FILE
// Single location for helpers shared by classification + execution files
// Avoids duplicate method compile error

func (r *OpenAIRouter) prismThresholds() config.PrismThresholds {
    if r.Config.Prism != nil {
        return r.Config.Prism.Thresholds
    }
    return config.DefaultPrismThresholds()
}

func (r *OpenAIRouter) getPrismMaxAttempts() int {
    if r.Config.Prism != nil && r.Config.Prism.MaxRerouteAttempts > 0 {
        return r.Config.Prism.MaxRerouteAttempts + 1 // 1 initial + N retries
    }
    return 4 // default: 1 initial + 3 retries
}
```

### 5.5 Re-routing loop — in processor_req_body.go

```go
// pkg/extproc/processor_req_body.go — modifications

// After filterCandidatesByPrism(), replace single selectModelFromCandidates() call
// with this loop:

for attempt := 0; attempt < r.getPrismMaxAttempts(); attempt++ {
    // TODO for @HuaminChen and @Xunzhuo: confirm exact signature before compilation
    // Variant 1: r.selectModelFromCandidates(ctx, candidates)
    // Variant 2: r.selectModelFromCandidates(ctx, candidates, decision)
    ctx.SelectedModel = r.selectModelFromCandidates(ctx, candidates, decision)

    r.runSemanticCache(ctx, decision)
    r.runJailbreakFilter(ctx, decision)
    r.runPIIFilter(ctx, decision)
    r.runPrismExecution(ctx, decision)

    if !ctx.PrismRefused {
        break // ACCEPTED or PRISM skipped — exit loop
    }

    // Key 3 REFUSED — exclude model and retry
    ctx.ExcludedModels = append(ctx.ExcludedModels, ctx.SelectedModel)
    ctx.PrismRefused = false
    candidates = registry.FilterExcluded(candidates, ctx.ExcludedModels)
    if len(candidates) == 0 {
        // All candidates refused — block request
        ctx.Blocked = true
        ctx.BlockReason = "PRISM: all candidates refused — no legitimate model available"
        break
    }
}
```

### 5.6 Global REFUSED — HTTP response

`ctx.Blocked = true` follows the **same pattern as jailbreak**. `buildResponse()` is not modified. HTTP status code is determined by existing `buildResponse()` logic.

---

## 6. filterCandidatesByPrism — pre-filter before model selection

```go
// pkg/extproc/processor_req_body.go — ADD before the re-route loop

func (r *OpenAIRouter) filterCandidatesByPrism(
    ctx *RequestContext,
    candidates []string,
) []string {
    if ctx.PrismDomain == "" || ctx.PrismDomain == "general" ||
        r.PrismRegistry == nil || !r.PrismRegistry.IsReady() {
        return candidates // no filter — standard routing
    }

    filtered := make([]string, 0, len(candidates))
    for _, model := range candidates {
        entry := r.PrismRegistry.GetByModelName(model)
        if entry != nil && strings.EqualFold(entry.Domain, ctx.PrismDomain) &&
            entry.Status == "active" {
            filtered = append(filtered, model)
        }
    }

    // Guarantee: if no PRISM-qualified candidate found → return all candidates
    // Standard vLLM-SR routing is NEVER blocked by PRISM
    if len(filtered) == 0 {
        return candidates
    }
    return filtered
}
```

**`selectModelFromCandidates()` is unchanged.** It receives the pre-filtered list and runs its existing algorithms (Elo, AutoMix, MLP, latency) without any knowledge of PRISM.

---

## 7. PRISM global configuration

```yaml
# config.yaml — PRISM global block (add after existing sections)

prism:
  thresholds:
    confidence_auto_accept: 0.90  # Key 3 auto-accept — no keyword check needed
    confidence_flexible:    0.70  # Key 3 flexible — 1 keyword match required
    confidence_strict:      0.40  # Key 3 strict  — 2 keyword matches + expertise score
    expertise_min_score:    0.61  # minimum empirical_score for strict tier
    score_delta_accepted:   0.02  # +delta per ACCEPTED interaction
    score_delta_refused:    0.03  # -delta per REFUSED interaction
  reroute_on_refused:       true
  max_reroute_attempts:     3
  unregistered_policy:      "passthrough"  # "passthrough" | "refuse"

# Note: default thresholds are tuned for industrial/specialized deployments.
# For general-purpose deployments, lower confidence_strict (e.g. 0.25)
# and expertise_min_score (e.g. 0.40).
```

---

## 8. Files to create / modify

### New files

```
pkg/registry/prism_registry.go               153-Registry — async init, Register,
                                             GetCandidates, UpdateScore, FilterExcluded,
                                             IsReady, NewForTesting
pkg/registry/prism_registry_test.go          unit tests
pkg/extproc/req_filter_prism_execution.go    Key 3 filter — runPrismExecution,
                                             handleUnregisteredModel,
                                             validatePrismLegitimacy, countKeywordMatches
pkg/extproc/req_filter_prism_execution_test.go unit tests
pkg/extproc/req_filter_prism_helpers.go      shared helpers — prismThresholds,
                                             getPrismMaxAttempts, extractQueryText,
                                             extractKeywords, cosineSimilarity
```

### Modified files

```
pkg/config/config.go          + PrismModelConfig struct
                              + PrismThresholds struct + DefaultPrismThresholds()
                              + PrismConfig struct (with UnregisteredPolicy)
                              + Prism field in ModelParams
                              + Prism field in RouterConfig

pkg/extproc/router.go         + 7 PRISM fields in RequestContext
                              + PrismRegistry field in OpenAIRouter
                              + registry.NewPrismRegistry(config) in NewOpenAIRouter()

pkg/extproc/req_filter_classification.go
                              + runPrismClassification() block
                                (uses candle.GetEmbedding, extractQueryText,
                                 extractKeywords, IsReady() guard)

pkg/extproc/processor_req_body.go
                              + runPrismClassification(ctx, body) after runClassification()
                              + filterCandidatesByPrism(ctx, candidates) before loop
                              + re-route loop replacing single selectModelFromCandidates call
                              + registry.FilterExcluded() import

config/config.yaml            + prism.enabled: true in example model_config entries
                              + global prism: block with all thresholds
                              + type: "prism-execution" plugin block in example decisions
```

---

## 9. Three integration modes (future variants)

| Mode | Keys active | Scope |
|------|------------|-------|
| `hybrid` | Key 1 + Key 2 + Key 3 | **This PR** — maximum legitimacy |
| `coarse_filter` | Key 1 + Key 2 | Future — pre-routing filter only, no post-validation |
| `fine_filter` | Key 1 + Key 3 | Future — post-routing validation only, no pre-filter |

---

## 10. Open questions for @HuaminChen and @Xunzhuo

1. **`selectModelFromCandidates` signature** — does it take `decision` as a third parameter `(ctx, candidates, decision)` or only two `(ctx, candidates)`? This is the **only pre-compilation check** — one line to verify in the existing `processor_req_body.go`.

2. **153-Registry storage** — in-memory (reset on restart) acceptable for v0.3, or should we target a persistent backend (Redis / SQLite) from the start?

3. **Key 2 domain embeddings** — domain embeddings are pre-computed at startup using `candle.GetEmbedding()` and cached in `RegistryEntry.DomainEmbedding`. Is this consistent with how the existing `candle-binding` package manages model state? Any threading concerns with concurrent CGo calls?

4. **Confidence thresholds** — default values (0.40 / 0.70 / 0.90) are tuned for industrial deployments. Should the repo ship with a `config/config.recipe-prism-general.yaml` preset with lower thresholds for general-purpose deployments?

5. **UnregisteredPolicy default** — current default is `"passthrough"` (model not in PRISM registry → routes normally). Should `"refuse"` be the default for stricter deployments? Or is `"passthrough"` the right default for progressive adoption?

---

## 11. Performance note

**Key 2 at high traffic:** `runPrismClassification()` calls `candle.GetEmbedding()` once per request (query embedding) and compares it against N pre-cached domain embeddings (cosine similarity). For typical industrial deployments with 3-10 domains, this is negligible. At high domain counts, consider adding an embedding similarity cache keyed on query hash.

This is a **future optimization** — not a blocker for v0.3.

---

## 12. References

- PRISM white paper: https://github.com/user-attachments/files/25750911/PRISM-Vllm-SR-whitepaper-COMPLET-EN.pdf
- Zenodo DOI: https://doi.org/10.5281/zenodo.18750029
- Issue #1422: https://github.com/vllm-project/semantic-router/issues/1422
- Draft PR #1425: https://github.com/vllm-project/semantic-router/pull/1425
