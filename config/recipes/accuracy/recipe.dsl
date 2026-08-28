# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL keyword accuracy_workflow_request {
  operator: "OR"
  keywords: ["investigate and produce an implementation plan", "research, compare, and synthesize", "diagnose the root cause and verify the fix", "break this into independent workstreams", "use tools to gather evidence", "调研、比较并综合", "分解为独立任务并验证", "根因分析并验证修复"]
}

SIGNAL keyword accuracy_deliberation_request {
  operator: "OR"
  keywords: ["debate both sides", "compare competing hypotheses", "adversarial review", "independent expert opinions", "challenge the proposed solution", "compare multiple approaches and reach a verdict", "比较多个假设并得出结论", "从正反两面论证", "对方案进行对抗性审查"]
}

SIGNAL keyword accuracy_direct_request {
  operator: "OR"
  keywords: ["answer the question directly", "do not debate", "do not compare", "do not create workstreams", "use a single model", "without multi-model fan-out", "直接回答问题", "不要比较多个方案", "不要创建并行工作流"]
}

SIGNAL context accuracy_long_context {
  description: "Long inputs stay on one capable worker to avoid multiplying context cost across a fan-out workflow."
  min_tokens: "16K"
  max_tokens: "1M"
}

SIGNAL conversation accuracy_has_images {
  description: "Request contains at least one image content part."
  feature: { source: { type: "image_content" }, type: "exists" }
}

# =============================================================================
# MODELS
# =============================================================================

MODEL gemini31-worker {
  context_window_size: 1000000
  description: "OpenRouter worker for broad reasoning and coding."
  capabilities: ["chat", "code", "reasoning", "long-context"]
  tags: ["deployment:openrouter", "role:worker"]
  quality_score: 0.93
  modality: "text"
}

MODEL gpt55-worker {
  context_window_size: 1048576
  description: "OpenRouter worker for frontier synthesis."
  capabilities: ["chat", "code", "reasoning", "long-context"]
  tags: ["deployment:openrouter", "role:worker"]
  quality_score: 0.94
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
  quality_score: 0.95
  modality: "text"
}

MODEL qwen-coordinator {
  context_window_size: 32768
  description: "Local planner and synthesis model for Router Flow dynamic plans."
  capabilities: ["chat", "planning", "synthesis", "code"]
  tags: ["deployment:self_hosted", "role:planner"]
  quality_score: 0.88
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

ROUTE accuracy_workflow (description = "Decompose evidence-gathering and tool-heavy tasks into a bounded parallel workflow.") {
  PRIORITY 100
  WHEN keyword("accuracy_workflow_request")
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
    planner: { max_completion_tokens: 2048, model: "qwen-coordinator" }
    template: "micro_agent"
  }
}

ROUTE accuracy_long_context_direct (description = "Keep long-context work on one frontier model instead of paying the latency and token multiplier of fan-out.") {
  PRIORITY 95
  WHEN context("accuracy_long_context")
  MODEL "gpt55-worker" (reasoning = true, effort = "high")
  ALGORITHM static
}

ROUTE accuracy_deliberation (description = "Use independent frontier perspectives for ambiguous, adversarial, or high-stakes judgments.") {
  PRIORITY 90
  WHEN keyword("accuracy_deliberation_request") AND NOT keyword("accuracy_direct_request")
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

ROUTE accuracy_direct (description = "Default to one frontier worker when orchestration has no explicit expected-quality benefit.") {
  PRIORITY 10
  MODEL "gpt55-worker" (reasoning = true, effort = "medium")
  ALGORITHM static
}
