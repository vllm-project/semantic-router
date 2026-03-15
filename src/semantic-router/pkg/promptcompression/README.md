# promptcompression

Extractive NLP prompt compression for the semantic router.
Reduces token count before domain classification (UC1) and before vLLM inference (UC2) without requiring a model call.

## Two use cases

| | UC1 — pre-classification | UC2 — pre-inference |
|---|---|---|
| **When** | Before signal evaluation | After domain is known, before vLLM |
| **Goal** | Reduce classification latency | Reduce inference latency |
| **Domain known?** | No | Yes |
| **Config** | Single default pipeline | `PipelineSelector` maps domain → pipeline |
| **Entry point** | `Compress` or `CompressWithPipeline` | `compressMessagesInBody` in extproc |

---

## Plugin architecture

A **Pipeline** is a YAML-configured chain of four optimizer phases run in order:

```
text ──► [pre] ──► [scoring] ──► [adjust] ──► [select] ──► kept sentences
```

| Phase | Interface | Purpose |
|---|---|---|
| `pre` | `PreProcessor` | Mutate sentence list before scoring (e.g. dedup) |
| `scoring` | `ScoringOptimizer` | Return `[0,1]` score per sentence; combined as weighted sum |
| `adjust` | `AdjustOptimizer` | Multiply composite scores in-place (e.g. pattern boost) |
| `select` | `SelectionOptimizer` | Force-keep sentences regardless of score |

Sentences that fit within `max_tokens - output_headroom` are selected greedily by composite score, with `preserve_first_n` and `preserve_last_n` slots reserved first.

---

## Built-in optimizers

### Scoring (`scoring` phase)

| Name | What it does | Key param |
|---|---|---|
| `textrank` | Graph-based centrality (Mihalcea & Tarau, EMNLP 2004) | — |
| `position` | U-shaped attention curve (Liu et al., TACL 2024) | `depth` 0–1 |
| `tfidf` | Information density via TF-IDF | — |
| `novelty` | Inverse centrality — surfaces jailbreaks, PII, outliers | — |
| `age_decay` | Discounts older conversation turns; pairs with `CompressMessages` | `factor` (default 0.15) |

### Adjust (`adjust` phase)

| Name | What it does | Key params |
|---|---|---|
| `pattern_boost` | Multiplies score for sentences matching regexes (errors, file paths, function signatures, TODO markers) | `patterns: [{regex, multiplier}]` |
| `focus_keywords` | Boosts sentences containing user-supplied keywords | `keywords`, `boost` (default 2.0) |
| `role_weight` | Per-role score multiplier; active only when `CompressMessages` is used | `weights: {system, user, assistant, tool}` |

Default `role_weight` values mirror Claude Code's compaction policy:
`user ×2.0` (kept verbatim), `assistant ×0.6` (reasoning is droppable), `system ×1.5`, `tool ×1.0`.

### Pre-processor (`pre` phase)

| Name | What it does | Key param |
|---|---|---|
| `dedup` | Drops near-duplicate sentences by cosine similarity; keeps the more recent copy | `threshold` (default 0.95) |

### Selection (`select` phase)

| Name | What it does | Key params |
|---|---|---|
| `must_contain` | Force-keeps sentences matching substrings or regexes regardless of score | `substrings`, `patterns` |

---

## YAML schema

```yaml
max_tokens: 512         # hard token budget (0 = pass-through)
preserve_first_n: 3     # always keep first N sentences (primacy)
preserve_last_n: 2      # always keep last N sentences (recency)
output_headroom: 0      # tokens reserved for model output; subtracted from max_tokens

pipeline:
  pre:
    - name: dedup
      params:
        threshold: 0.95

  scoring:
    - name: textrank
      weight: 0.20
    - name: position
      weight: 0.40
      params:
        depth: 0.5
    - name: tfidf
      weight: 0.35
    - name: novelty
      weight: 0.05

  adjust:
    - name: pattern_boost          # uses default patterns when none supplied
    - name: role_weight
      params:
        weights:
          user: 2.0
          assistant: 0.6

  select:
    - name: must_contain
      params:
        substrings: ["CRITICAL:"]
        patterns:   ["\\bCVE-\\d{4}-\\d+\\b"]
```

Scorer weights are normalized automatically (any positive values work).

---

## Domain-aware pipeline selection (UC2)

`PipelineSelector` maps the router's `decisionName` (domain) to a `Pipeline`:

```yaml
# in your router config
prompt_compression:
  inference_enabled: true
  default_pipeline: config/prompt-compression/default.yaml
  domain_pipelines:
    coding:   config/prompt-compression/coding.yaml
    medical:  config/prompt-compression/medical.yaml
    security: config/prompt-compression/security.yaml
```

At runtime:
```go
pipeline := router.InferenceCompressor.Select(decisionName) // "coding" → coding.yaml
compressed, err := compressMessagesInBody(body, pipeline)
```

Falls back to `default_pipeline` for unknown domains.

---

## Usage in Go

```go
// Flat text (UC1 or ad-hoc)
p, _ := promptcompression.LoadPipeline("config/prompt-compression/default.yaml")
result := promptcompression.CompressWithPipeline(text, p)
fmt.Printf("%d → %d tokens (ratio %.2f)\n",
    result.OriginalTokens, result.CompressedTokens, result.Ratio)

// Multi-turn conversation (UC2, role-aware)
p, _ := promptcompression.LoadPipeline("config/prompt-compression/coding.yaml")
result := promptcompression.CompressMessages([]promptcompression.Message{
    {Role: "system",    Content: "You are a Go expert."},
    {Role: "user",      Content: "Why does this panic?"},
    {Role: "assistant", Content: "I'll read the stack trace..."},
    {Role: "tool",      Content: "goroutine 1: panic: index out of range..."},
    {Role: "user",      Content: "The crash is in auth.go:142."},
}, p)

// Register a custom optimizer (e.g. in a plugin package init())
promptcompression.RegisterAdjust("my_boost", func(params map[string]any) (promptcompression.AdjustOptimizer, error) {
    return &myBoostOptimizer{}, nil
})
```

---

## Adding a custom optimizer

1. Implement one of the four interfaces (`PreProcessor`, `ScoringOptimizer`, `AdjustOptimizer`, `SelectionOptimizer`).
2. Call the matching `Register*` function — typically in `init()`.
3. Reference the optimizer by name in any YAML pipeline config.

No other wiring required; the registry is global and thread-safe.
