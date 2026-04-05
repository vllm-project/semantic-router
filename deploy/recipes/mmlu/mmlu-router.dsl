# =============================================================================
# SIGNALS
# =============================================================================

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

SIGNAL kb best_computer_science {
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
# MODELS
# =============================================================================

MODEL cloud/frontier-72b {
  context_window_size: 262144
}

MODEL local/small-7b {
  context_window_size: 131072
}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE escalate_72b (description = "Escalate high-uplift MMLU domains to the frontier 72B lane.") {
  PRIORITY 200
  TIER 1
  WHEN (kb("best_biology") OR kb("best_business") OR kb("best_computer_science") OR kb("best_economics") OR kb("best_history") OR kb("best_math") OR kb("best_other") OR kb("best_philosophy") OR kb("best_psychology"))
  MODEL "cloud/frontier-72b" (reasoning = false)
}

ROUTE keep_7b (description = "Keep lower-uplift MMLU domains on the local 7B lane.") {
  PRIORITY 100
  TIER 2
  WHEN projection("no_escalation")
  MODEL "local/small-7b" (reasoning = false)
}
