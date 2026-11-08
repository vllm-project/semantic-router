# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL turn_relation persistently_dissatisfied {
  description: "Current user turn repeats the last two user turns in a row."
  labels: ["reask"]
  input: { previous_turns: 4 }
}

SIGNAL conversation two_prior_answers {
  description: "The conversation already holds two assistant answers."
  feature: { source: { role: "assistant", type: "message" }, type: "count" }
  predicate: { gte: 2 }
}

SIGNAL input_modality feedback_has_images {
  description: "Request contains at least one image content part."
  modality: "image"
}

SIGNAL systemone wrong {
  description: "The latest message says an earlier answer or fix was wrong."
  question: { instructions: "Does the latest message say that an earlier answer or fix was wrong, incorrect or did not work?", type: "noul" }
  predicate: { gte: 0.5 }
}

SIGNAL systemone clarify {
  description: "The latest message asks for an earlier answer again, more clearly."
  question: { instructions: "Does the latest message ask to explain an earlier answer again, more clearly, more simply or with an example?", type: "noul" }
  predicate: { gte: 0.6 }
}

SIGNAL systemone code {
  description: "The request is about software."
  question: { instructions: "Is the request about software, such as code, a program error or exception, a failing test or a bug?", type: "noul" }
  predicate: { gte: 0.5 }
}

SIGNAL systemone sources {
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
  WHEN input_modality("feedback_has_images")
  MODEL "local/omni" (reasoning = false)
  ALGORITHM static
}

ROUTE feedback_verified_recovery (description = "A wrong answer or a question repeated after two answers (wrong, persistently_dissatisfied, two_prior_answers) that asks for sources (sources), outside code, goes to the verified recovery lane.", on_unknown = "no_match") {
  PRIORITY 240
  WHEN (systemone("wrong") OR turn_relation("persistently_dissatisfied") AND conversation("two_prior_answers")) AND systemone("sources") AND NOT systemone("code")
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "verified-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_persistent_code_recovery (description = "A code question the user keeps repeating after two answers (code, persistently_dissatisfied, two_prior_answers) goes to the premium code-repair lane.", on_unknown = "no_match") {
  PRIORITY 230
  WHEN systemone("code") AND turn_relation("persistently_dissatisfied") AND conversation("two_prior_answers")
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "persistent-code-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_code_recovery (description = "A first report that code is still wrong (code, wrong) goes to the code-repair lane.", on_unknown = "no_match") {
  PRIORITY 220
  WHEN systemone("code") AND systemone("wrong") AND NOT (turn_relation("persistently_dissatisfied") AND conversation("two_prior_answers"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "code-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_persistent_recovery (description = "Any other question the user keeps repeating after two answers (persistently_dissatisfied, two_prior_answers) goes to the premium general recovery lane.", on_unknown = "no_match") {
  PRIORITY 210
  WHEN turn_relation("persistently_dissatisfied") AND conversation("two_prior_answers") AND NOT systemone("code") AND NOT systemone("sources")
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "persistent-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_general_recovery (description = "Any other report that an answer was wrong (wrong) goes to the general recovery lane.", on_unknown = "no_match") {
  PRIORITY 200
  WHEN systemone("wrong") AND NOT systemone("code") AND NOT systemone("sources") AND NOT (turn_relation("persistently_dissatisfied") AND conversation("two_prior_answers"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "general-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_need_clarification (description = "A request to explain an earlier answer again (clarify) that does not call it wrong stays on the low-cost lane.", on_unknown = "no_match") {
  PRIORITY 180
  WHEN systemone("clarify") AND NOT systemone("wrong") AND NOT (turn_relation("persistently_dissatisfied") AND conversation("two_prior_answers"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN response_cache {
    enabled: false
    semantic: { match: { recall: { min_similarity: 0.86, top_k: 3 } } }
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
