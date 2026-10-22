# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL decision gb10_domain {
  description: "The domain Kai assigns to the request."
  question: { choices: [{ description: "Programming", key: "code" }, { description: "Mathematics", key: "math" }, { description: "Anything else", key: "other" }], instructions: "Which domain does this request belong to?", type: "choice" }
  timeout_ms: 30000
}

SIGNAL decision gb10_reasoning {
  description: "Kai's probability that the request needs multi-step reasoning."
  question: { instructions: "Does answering this request need multi-step reasoning?", type: "noul" }
  predicate: { gte: 0.5 }
  timeout_ms: 30000
}

SIGNAL decision gb10_difficulty {
  description: "Kai's expected difficulty level, from Trivial (0) through Hard (2)."
  question: { instructions: "How difficult is this request?", levels: ["Trivial", "Moderate", "Hard"], type: "score" }
  predicate: { gte: 1 }
  timeout_ms: 30000
}

# =============================================================================
# MODELS
# =============================================================================

MODEL external/generation {
  context_window_size: 32768
  description: "Explicit external generation backend for this recipe."
  capabilities: ["chat"]
  modality: "text"
}

MODEL local/generation {
  context_window_size: 32768
  description: "Explicit local generation backend for this recipe."
  capabilities: ["chat"]
  modality: "text"
}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE external_code (description = "Code requests that need multi-step reasoning (gb10_reasoning at least 0.5) and are moderately difficult or harder (gb10_difficulty at least 1) use the external generation backend. An unanswered Kai call fails the request.", on_unknown = "fail_request") {
  PRIORITY 200
  WHEN decision("gb10_domain", label: "code") AND decision("gb10_reasoning") AND decision("gb10_difficulty")
  MODEL "external/generation" (reasoning = false)
  ALGORITHM static
}

ROUTE local_generation (description = "Math, other, and code requests that are not both multi-step and at least moderately difficult use the local generation backend. An unanswered Kai call fails the request.", on_unknown = "fail_request") {
  PRIORITY 100
  WHEN (decision("gb10_domain", label: "math") OR decision("gb10_domain", label: "other") OR NOT decision("gb10_reasoning") OR NOT decision("gb10_difficulty"))
  MODEL "local/generation" (reasoning = false)
  ALGORITHM static
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vllm-sr/auto"]
  recipe: "default"
}
