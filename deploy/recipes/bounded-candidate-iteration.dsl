# =============================================================================
# BOUNDED CANDIDATE ITERATION RECIPE
# =============================================================================

SIGNAL keyword business_request {
  operator: "OR"
  keywords: ["business plan", "market analysis", "pricing strategy", "go to market"]
}

SIGNAL keyword complex_reasoning_request {
  operator: "OR"
  keywords: ["compare tradeoffs", "step by step", "deep analysis", "reason through"]
}

# =============================================================================
# MODELS
# =============================================================================

MODEL qwen3-8b {
  context_window_size: 32768
}

MODEL qwen3-32b {
  context_window_size: 131072
}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE iterate_existing_candidates (description = "Iterate over the route's declared candidate models.") {
  PRIORITY 100
  WHEN keyword("business_request")

  MODEL "qwen3-8b" (reasoning = false)
  MODEL "qwen3-32b" (reasoning = true, effort = "medium")

  FOR candidate IN decision.candidates {
    MODEL candidate
  }
}

ROUTE iterate_explicit_candidates (description = "Declare a bounded candidate list directly in the iteration source.") {
  PRIORITY 90
  WHEN keyword("complex_reasoning_request")

  FOR candidate IN ["qwen3-8b" (reasoning = false), "qwen3-32b" (reasoning = true, effort = "medium")] {
    MODEL candidate
  }
}
