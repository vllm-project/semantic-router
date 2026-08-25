# =============================================================================
# MODELS
# =============================================================================

MODEL gemini31-worker {
  capabilities: ["chat", "code", "reasoning", "long-context", "text"]
}

MODEL gpt55-worker {
  capabilities: ["chat", "code", "reasoning", "long-context", "text"]
}

MODEL local/omni {
  capabilities: ["chat", "image_understanding", "multimodal", "omni", "text", "vision"]
  modality: "omni"
}

MODEL opus48-worker {
  capabilities: ["chat", "code", "reasoning", "long-context", "text"]
}

MODEL qwen-coordinator {
  capabilities: ["chat", "planning", "synthesis", "code", "text"]
  reasoning: { type: "chat_template_kwargs" }
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  name: "vllm-sr/auto"
  recipe: "default"
  assignments: [
    { decision: "accuracy_deliberation", models: [{ model: "opus48-worker", weight: "1" }, { model: "gemini31-worker", weight: "1" }, { model: "gpt55-worker", weight: "1" }, { model: "qwen-coordinator", weight: "1" }] },
    { decision: "accuracy_direct", models: [{ model: "gpt55-worker", weight: "1" }] },
    { decision: "accuracy_long_context_direct", models: [{ model: "gpt55-worker", weight: "1" }] },
    { decision: "accuracy_workflow", models: [{ model: "opus48-worker", weight: "1" }, { model: "gemini31-worker", weight: "1" }, { model: "gpt55-worker", weight: "1" }] },
    { decision: "omni", models: [{ model: "local/omni", weight: "1" }] },
  ]
}

# =============================================================================
# RECIPE default
# =============================================================================

RECIPE default (description = "Default routing recipe.") {
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
  # ROUTES
  # =============================================================================

  ROUTE omni (description = "Understand image-bearing requests with the dedicated visual-language model.") {
    PRIORITY 200
    WHEN conversation("accuracy_has_images")
    ALGORITHM static
  }

  ROUTE accuracy_workflow (description = "Decompose evidence-gathering and tool-heavy tasks into a bounded parallel workflow.") {
    PRIORITY 100
    WHEN keyword("accuracy_workflow_request")
    ALGORITHM workflows {
      include_intermediate_responses: true
      max_completion_tokens: 8192
      max_parallel: 3
      max_steps: 4
      min_successful_responses: 2
      mode: "dynamic"
      on_error: "skip"
      planner: { max_completion_tokens: 2048 }
      template: "micro_agent"
    }
  }

  ROUTE accuracy_long_context_direct (description = "Keep long-context work on one frontier model instead of paying the latency and token multiplier of fan-out.") {
    PRIORITY 95
    WHEN context("accuracy_long_context")
    ALGORITHM static
  }

  ROUTE accuracy_deliberation (description = "Use independent frontier perspectives for ambiguous, adversarial, or high-stakes judgments.") {
    PRIORITY 90
    WHEN keyword("accuracy_deliberation_request") AND NOT keyword("accuracy_direct_request")
    ALGORITHM fusion {
      include_analysis: true
      include_intermediate_responses: true
      max_concurrent: 3
      min_successful_responses: 2
      on_error: "skip"
    }
  }

  ROUTE accuracy_direct (description = "Default to one frontier worker when orchestration has no explicit expected-quality benefit.") {
    PRIORITY 10
    ALGORITHM static
  }

}
