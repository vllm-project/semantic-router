# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL context accuracy_long_context {
  description: "Long inputs stay on one capable worker to avoid multiplying context cost across a fan-out workflow."
  min_tokens: "16K"
  max_tokens: "1M"
}

SIGNAL conversation accuracy_has_images {
  description: "Request contains at least one image content part."
  feature: { source: { type: "image_content" }, type: "exists" }
}

SIGNAL decision accuracy_mode {
  description: "The way of working the request asks for."
  question: { choices: [{ description: "A single direct answer, explanation or summary, including when the user asks not to split the work, compare options or use several models", key: "direct" }, { description: "An investigation or plan with several workstreams, such as research, diagnosis or evidence gathering", key: "workstreams" }, { description: "Independent opinions, a debate or an adversarial review that ends in a verdict", key: "independent_review" }], instructions: "Which way of working does this request ask for?", type: "choice" }
}

SIGNAL decision accuracy_difficulty {
  description: "How much reasoning a strong expert needs, from 0 (none) to 4 (research level)."
  question: { instructions: "How much reasoning does a strong expert need to answer this request well?", levels: ["Answer immediately, with no thinking", "A little thought, one or two simple steps", "Multi-step reasoning or a careful check", "Deep expert reasoning on a hard problem", "Research-level work on an open or frontier problem"], type: "score" }
  predicate: { gte: 1.5 }
}

# =============================================================================
# MODELS
# =============================================================================

MODEL gemini31-worker {
  context_window_size: 1000000
  description: "OpenRouter worker for broad reasoning and coding."
  capabilities: ["chat", "code", "reasoning", "long-context"]
  tags: ["deployment:openrouter", "role:worker"]
  modality: "text"
}

MODEL gpt55-worker {
  context_window_size: 1048576
  description: "OpenRouter worker for long-context planning and frontier synthesis."
  capabilities: ["chat", "code", "reasoning", "long-context", "planning", "synthesis"]
  tags: ["deployment:openrouter", "role:worker", "role:planner"]
  modality: "text"
}

MODEL local/omni {
  capabilities: ["chat", "image_understanding", "multimodal", "omni", "text", "vision"]
  modality: "omni"
}

MODEL opus48-worker {
  context_window_size: 1000000
  description: "OpenRouter worker for long-horizon agentic work."
  capabilities: ["chat", "code", "reasoning", "long-context"]
  tags: ["deployment:openrouter", "role:worker"]
  modality: "text"
}

MODEL qwen-coordinator {
  context_window_size: 32768
  description: "Local planner and synthesis model for Router Flow dynamic plans."
  capabilities: ["chat", "planning", "synthesis", "code"]
  tags: ["deployment:self_hosted", "role:planner"]
  modality: "text"
}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE omni (description = "Understand image-bearing requests with the dedicated visual-language model.") {
  PRIORITY 200
  WHEN conversation("accuracy_has_images")
  MODEL "local/omni" (reasoning = false)
  ALGORITHM static
}

ROUTE accuracy_workflow (description = "Several workstreams (accuracy_mode: workstreams) that need multi-step reasoning (accuracy_difficulty 1.5 or more) run as a bounded parallel workflow.", on_unknown = "no_match") {
  PRIORITY 100
  WHEN decision("accuracy_mode", label: "workstreams", predicate: { gte: 0.6 }) AND decision("accuracy_difficulty")
  MODEL "opus48-worker" (reasoning = true, effort = "medium"),
        "gemini31-worker" (reasoning = true, effort = "medium"),
        "gpt55-worker" (reasoning = true, effort = "medium")
  ALGORITHM workflows {
    include_intermediate_responses: true
    max_completion_tokens: 8192
    max_parallel: 3
    max_steps: 4
    min_successful_responses: 2
    mode: "dynamic"
    on_error: "skip"
    planner: { max_completion_tokens: 2048, model: "gpt55-worker" }
    template: "micro_agent"
  }
}

ROUTE accuracy_long_context_direct (description = "Keep long-context work on one frontier model instead of paying the latency and token multiplier of fan-out.") {
  PRIORITY 95
  WHEN context("accuracy_long_context")
  MODEL "gpt55-worker" (reasoning = true, effort = "high")
  ALGORITHM static
}

ROUTE accuracy_deliberation (description = "Independent opinions or a verdict (accuracy_mode: independent_review) that need multi-step reasoning (accuracy_difficulty 1.5 or more) get independent frontier perspectives.", on_unknown = "no_match") {
  PRIORITY 90
  WHEN decision("accuracy_mode", label: "independent_review", predicate: { gte: 0.6 }) AND decision("accuracy_difficulty") AND NOT decision("accuracy_mode", label: "workstreams", predicate: { gte: 0.6 })
  MODEL "opus48-worker" (reasoning = true, effort = "medium"),
        "gemini31-worker" (reasoning = true, effort = "medium"),
        "gpt55-worker" (reasoning = true, effort = "medium"),
        "qwen-coordinator" (reasoning = false)
  ALGORITHM fusion {
    analysis_models: ["opus48-worker", "gemini31-worker", "gpt55-worker"]
    include_analysis: true
    include_intermediate_responses: true
    max_concurrent: 3
    min_successful_responses: 2
    model: "qwen-coordinator"
    on_error: "skip"
  }
}

ROUTE accuracy_direct (description = "Everything else, including a request for one direct answer and requests the decision model cannot answer, goes to one frontier worker that the decision model picks.") {
  PRIORITY 10
  MODEL "gpt55-worker" (reasoning = true, effort = "medium"),
        "opus48-worker" (reasoning = true, effort = "medium"),
        "gemini31-worker" (reasoning = true, effort = "medium")
  ALGORITHM decision {
    decision: { candidates: { gemini31-worker: "Broad reasoning, mathematics and everyday coding", gpt55-worker: "Long-context reading, planning and synthesis across many sources", opus48-worker: "Long-horizon agentic work and large code changes" }, instructions: "Which worker should answer this request?", timeout_ms: 30000 }
  }
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vllm-sr/auto"]
  recipe: "default"
}
