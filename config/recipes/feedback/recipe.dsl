# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL reask persistently_dissatisfied {
  description: "Current user turn repeats the last two user turns in a row."
  threshold: 0.8
  lookback_turns: 2
}

SIGNAL conversation feedback_has_images {
  description: "Request contains at least one image content part."
  feature: { source: { type: "image_content" }, type: "exists" }
}

SIGNAL decision wrong {
  description: "The latest message says an earlier answer or fix was wrong."
  question: { instructions: "Does the latest message say that an earlier answer or fix was wrong, incorrect or did not work?", type: "noul" }
  predicate: { gte: 0.5 }
}

SIGNAL decision clarify {
  description: "The latest message asks for an earlier answer again, more clearly."
  question: { instructions: "Does the latest message ask to explain an earlier answer again, more clearly, more simply or with an example?", type: "noul" }
  predicate: { gte: 0.6 }
}

SIGNAL decision code {
  description: "The request is about software."
  question: { instructions: "Is the request about software, such as code, a program error or exception, a failing test or a bug?", type: "noul" }
  predicate: { gte: 0.5 }
}

SIGNAL decision sources {
  description: "The user asks for sources, citations or a fact check."
  question: { instructions: "Does the user ask for sources, citations or a fact check of the answer?", type: "noul" }
  predicate: { gte: 0.6 }
}

# =============================================================================
# MODELS
# =============================================================================

MODEL google/gemini-3.1-pro {
  context_window_size: 262144
  description: "Mid-cost recovery lane for repeated dissatisfaction, debugging, and answer repair."
  capabilities: ["answer_repair", "debugging", "reasoning"]
  tags: ["tier:repair", "purpose:feedback_recovery"]
  modality: "text"
}

MODEL local/omni {
  capabilities: ["chat", "image_understanding", "multimodal", "omni", "text", "vision"]
  modality: "omni"
}

MODEL openai/gpt5.4 {
  context_window_size: 262144
  description: "Premium lane reserved for verified or high-stakes dissatisfaction recovery."
  capabilities: ["verified_reasoning", "evidence_synthesis", "high_stakes"]
  tags: ["tier:verified", "purpose:feedback_recovery"]
  modality: "text"
}

MODEL qwen/qwen3.5-rocm {
  context_window_size: 131072
  description: "Cheap default lane for standard follow-ups and lightweight clarification."
  capabilities: ["cheap_followup", "concise_rewrite", "general_chat"]
  tags: ["tier:cheap", "purpose:default_feedback"]
  modality: "text"
}

# =============================================================================
# PLUGINS
# =============================================================================

PLUGIN header_mutation header_mutation {}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE omni (description = "Image-bearing follow-ups go to the visual-language model.") {
  PRIORITY 260
  WHEN conversation("feedback_has_images")
  MODEL "local/omni" (reasoning = false)
  ALGORITHM static
}

ROUTE feedback_verified_recovery (description = "A wrong answer or a repeated question (wrong, persistently_dissatisfied) that asks for sources (sources), outside code, goes to the verified recovery lane.", on_unknown = "no_match") {
  PRIORITY 240
  WHEN (decision("wrong") OR reask("persistently_dissatisfied")) AND decision("sources") AND NOT decision("code")
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "verified-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_persistent_code_recovery (description = "A code question the user keeps repeating (code, persistently_dissatisfied) goes to the premium code-repair lane.", on_unknown = "no_match") {
  PRIORITY 230
  WHEN decision("code") AND reask("persistently_dissatisfied")
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "persistent-code-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_code_recovery (description = "A first report that code is still wrong (code, wrong) goes to the code-repair lane.", on_unknown = "no_match") {
  PRIORITY 220
  WHEN decision("code") AND decision("wrong") AND NOT reask("persistently_dissatisfied")
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "code-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_persistent_recovery (description = "Any other question the user keeps repeating (persistently_dissatisfied) goes to the premium general recovery lane.", on_unknown = "no_match") {
  PRIORITY 210
  WHEN reask("persistently_dissatisfied") AND NOT decision("code") AND NOT decision("sources")
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "persistent-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_general_recovery (description = "Any other report that an answer was wrong (wrong) goes to the general recovery lane.", on_unknown = "no_match") {
  PRIORITY 200
  WHEN decision("wrong") AND NOT decision("code") AND NOT decision("sources") AND NOT reask("persistently_dissatisfied")
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "general-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_need_clarification (description = "A request to explain an earlier answer again (clarify) that does not call it wrong stays on the low-cost lane.", on_unknown = "no_match") {
  PRIORITY 180
  WHEN decision("clarify") AND NOT decision("wrong") AND NOT reask("persistently_dissatisfied")
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN response_cache {
    enabled: true
    similarity_threshold: 0.86
    ttl_seconds: 1800
  }
}

ROUTE feedback_default (description = "Everything else, including requests the decision model cannot answer, stays on the low-cost default lane.") {
  PRIORITY 10
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vllm-sr/auto"]
  recipe: "default"
}
