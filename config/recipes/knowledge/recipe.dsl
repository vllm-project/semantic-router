# =============================================================================
# MODELS
# =============================================================================

MODEL cloud/frontier-72b {
}

MODEL local/small-7b {
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  name: "vllm-sr/auto"
  recipe: "default"
  assignments: [
    { decision: "escalate_72b", models: [{ model: "cloud/frontier-72b", weight: "1" }] },
    { decision: "keep_7b", models: [{ model: "local/small-7b", weight: "1" }] },
  ]
}

# =============================================================================
# RECIPE default
# =============================================================================

RECIPE default (description = "Default routing recipe.") {
  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword escalation_domain_override {
    operator: "OR"
    keywords: ["matrix", "determinant", "invertibility", "shareholder equity", "amortized time complexity", "reinforcement schedule"]
  }

  SIGNAL keyword local_domain_override {
    operator: "OR"
    keywords: ["blood cells", "transporting oxygen", "contract law", "legal consideration", "ideal transformer", "measured in joules", "share electron pairs"]
  }

  SIGNAL kb best_biology {
    kb: "mmlu_kb"
    target: { kind: "label", value: "biology" }
    match: "best"
  }

  SIGNAL kb best_business {
    kb: "mmlu_kb"
    target: { kind: "label", value: "business" }
    match: "best"
  }

  SIGNAL kb best_computing {
    kb: "mmlu_kb"
    target: { kind: "label", value: "computer science" }
    match: "best"
  }

  SIGNAL kb best_economics {
    kb: "mmlu_kb"
    target: { kind: "label", value: "economics" }
    match: "best"
  }

  SIGNAL kb best_engineering {
    kb: "mmlu_kb"
    target: { kind: "label", value: "engineering" }
    match: "best"
  }

  SIGNAL kb best_health {
    kb: "mmlu_kb"
    target: { kind: "label", value: "health" }
    match: "best"
  }

  SIGNAL kb best_history {
    kb: "mmlu_kb"
    target: { kind: "label", value: "history" }
    match: "best"
  }

  SIGNAL kb best_law {
    kb: "mmlu_kb"
    target: { kind: "label", value: "law" }
    match: "best"
  }

  SIGNAL kb best_math {
    kb: "mmlu_kb"
    target: { kind: "label", value: "math" }
    match: "best"
  }

  SIGNAL kb best_other {
    kb: "mmlu_kb"
    target: { kind: "label", value: "other" }
    match: "best"
  }

  SIGNAL kb best_philosophy {
    kb: "mmlu_kb"
    target: { kind: "label", value: "philosophy" }
    match: "best"
  }

  SIGNAL kb best_physics {
    kb: "mmlu_kb"
    target: { kind: "label", value: "physics" }
    match: "best"
  }

  SIGNAL kb best_psychology {
    kb: "mmlu_kb"
    target: { kind: "label", value: "psychology" }
    match: "best"
  }

  SIGNAL kb best_chemistry {
    kb: "mmlu_kb"
    target: { kind: "label", value: "chemistry" }
    match: "best"
  }

  PROJECTION score escalation_pressure {
    method: "weighted_sum"
    inputs: [{ type: "kb_metric", weight: 1, kb: "mmlu_kb", metric: "escalate_vs_keep", value_source: "score" }]
  }

  PROJECTION mapping escalation_band {
    source: "escalation_pressure"
    method: "threshold_bands"
    outputs: [{ name: "no_escalation", lt: 2 }, { name: "escalation_signal", gte: 2 }]
  }

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE escalate_72b (description = "Escalate high-uplift MMLU domains to the frontier 72B lane.") {
    PRIORITY 200
    TIER 1
    WHEN (kb("best_biology") OR kb("best_business") OR kb("best_computing") OR kb("best_economics") OR kb("best_history") OR kb("best_math") OR kb("best_other") OR kb("best_philosophy") OR kb("best_psychology") OR keyword("escalation_domain_override")) AND NOT keyword("local_domain_override")
  }

  ROUTE keep_7b (description = "Keep lower-uplift MMLU domains on the local 7B lane.") {
    PRIORITY 100
    TIER 2
    WHEN (projection("no_escalation") OR keyword("local_domain_override"))
  }

}
