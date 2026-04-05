# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL domain "computer science" {
  description: "Coding, debugging, systems, APIs, and software engineering."
}

SIGNAL domain health {
  description: "Medical, clinical, or health-sensitive questions."
}

SIGNAL domain law {
  description: "Legal, policy, compliance, and regulatory topics."
}

SIGNAL domain other {
  description: "General-purpose follow-up traffic."
}

SIGNAL keyword verification_markers {
  operator: "OR"
  keywords: ["verify this", "fact check", "with sources", "cite sources", "cite the source", "give evidence", "provide evidence", "给出处", "给来源", "请核实", "请给证据"]
}

SIGNAL keyword code_error_markers {
  operator: "OR"
  keywords: ["traceback", "exception", "syntaxerror", "typeerror", "valueerror", "stack trace", "stacktrace", "notimplementederror", "assertionerror", "runtimeerror"]
}

SIGNAL keyword clarification_feedback_markers {
  operator: "OR"
  keywords: ["clarify", "explain more clearly", "simpler", "simpler answer", "one example", "walk me through it", "讲清楚一点", "简单一点", "举个例子"]
}

SIGNAL keyword frustration_feedback_markers {
  operator: "OR"
  keywords: ["wrong", "that's wrong", "this is wrong", "doesn't work", "does not work", "mistake", "invalid", "that failed", "try again", "fix this", "不对", "错了", "这不行", "重新来", "再试一次"]
}

SIGNAL fact_check needs_fact_check {
  description: "Request needs factual verification or sources."
}

SIGNAL user_feedback wrong_answer {
  description: "Explicit correction or dissatisfaction."
}

SIGNAL user_feedback need_clarification {
  description: "Explicit request to restate the answer more clearly."
}

SIGNAL reask likely_dissatisfied {
  description: "Current user turn closely repeats the immediately previous user turn."
  threshold: 0.8
  lookback_turns: 1
}

SIGNAL reask persistently_dissatisfied {
  description: "Current user turn repeats the last two user turns in a row."
  threshold: 0.8
  lookback_turns: 2
}

SIGNAL context short_context {
  min_tokens: "0"
  max_tokens: "999"
}

SIGNAL context medium_context {
  min_tokens: "1K"
  max_tokens: "7999"
}

SIGNAL context long_context {
  min_tokens: "8K"
  max_tokens: "256K"
}

PROJECTION score feedback_recovery_pressure {
  method: "weighted_sum"
  inputs: [{ type: "user_feedback", weight: 0.36, name: "wrong_answer" }, { type: "reask", weight: 0.24, name: "likely_dissatisfied", value_source: "confidence" }, { type: "reask", weight: 0.46, name: "persistently_dissatisfied", value_source: "confidence" }, { type: "keyword", weight: 0.18, name: "frustration_feedback_markers", value_source: "confidence" }, { type: "keyword", weight: 0.3, name: "code_error_markers", value_source: "confidence" }, { type: "fact_check", weight: 0.18, name: "needs_fact_check" }, { type: "domain", weight: 0.16, name: "health" }, { type: "domain", weight: 0.16, name: "law" }, { type: "context", weight: 0.06, name: "long_context" }]
}

PROJECTION score verification_pressure {
  method: "weighted_sum"
  inputs: [{ type: "fact_check", weight: 0.34, name: "needs_fact_check" }, { type: "keyword", weight: 0.28, name: "verification_markers", value_source: "confidence" }, { type: "domain", weight: 0.18, name: "health" }, { type: "domain", weight: 0.18, name: "law" }, { type: "user_feedback", weight: 0.08, name: "wrong_answer" }, { type: "reask", weight: 0.08, name: "persistently_dissatisfied", value_source: "confidence" }]
}

PROJECTION mapping feedback_recovery_band {
  source: "feedback_recovery_pressure"
  method: "threshold_bands"
  outputs: [{ name: "feedback_retry", gte: 0.24, lt: 0.55 }, { name: "feedback_escalate", gte: 0.55, lt: 0.85 }, { name: "feedback_verified_escalate", gte: 0.85 }]
}

PROJECTION mapping verification_band {
  source: "verification_pressure"
  method: "threshold_bands"
  outputs: [{ name: "feedback_needs_evidence", gte: 0.42 }]
}

# =============================================================================
# MODELS
# =============================================================================

MODEL google/gemini-3.1-pro {
  context_window_size: 262144
  description: "Mid-cost recovery lane for repeated dissatisfaction, debugging, and answer repair."
  capabilities: ["answer_repair", "debugging", "reasoning"]
  tags: ["tier:repair", "purpose:feedback_recovery"]
  quality_score: 0.88
  modality: "text"
}

MODEL openai/gpt5.4 {
  context_window_size: 262144
  description: "Premium lane reserved for verified or high-stakes dissatisfaction recovery."
  capabilities: ["verified_reasoning", "evidence_synthesis", "high_stakes"]
  tags: ["tier:verified", "purpose:feedback_recovery"]
  quality_score: 0.94
  modality: "text"
}

MODEL qwen/qwen3.5-rocm {
  context_window_size: 131072
  description: "Cheap default lane for standard follow-ups and lightweight clarification."
  capabilities: ["cheap_followup", "concise_rewrite", "general_chat"]
  tags: ["tier:cheap", "purpose:default_feedback"]
  quality_score: 0.72
  modality: "text"
}

# =============================================================================
# PLUGINS
# =============================================================================

PLUGIN header_mutation header_mutation {}

PLUGIN system_prompt system_prompt {}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE feedback_verified_recovery (description = "Premium recovery lane for explicit or persistent dissatisfaction on evidence-sensitive follow-ups.") {
  PRIORITY 240
  WHEN (projection("feedback_verified_escalate") OR reask("persistently_dissatisfied") OR user_feedback("wrong_answer")) AND (projection("feedback_needs_evidence") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR domain("health") OR domain("law")) AND NOT (keyword("code_error_markers") OR domain("computer science"))
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN system_prompt {
    enabled: true
    system_prompt: "You are handling a dissatisfied follow-up on a high-stakes or verification-sensitive question. Briefly acknowledge the miss, correct the answer directly, state uncertainty when needed, and cite evidence when the user asks for verification."
    mode: "insert"
  }
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "verified-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_persistent_code_recovery (description = "Premium code-repair lane for repeated same-question retries that already failed on a cheaper recovery pass.") {
  PRIORITY 230
  WHEN (keyword("code_error_markers") OR domain("computer science")) AND reask("persistently_dissatisfied")
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN system_prompt {
    enabled: true
    system_prompt: "The user has repeated the same failing coding question multiple times. Rebuild the diagnosis from first principles, identify the faulty assumption in the earlier answer, and return a corrected implementation plus a concrete validation step."
    mode: "insert"
  }
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "persistent-code-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_code_recovery (description = "Repair lane for code follow-ups that include execution failures or repeated dissatisfaction.") {
  PRIORITY 220
  WHEN (keyword("code_error_markers") OR domain("computer science")) AND (user_feedback("wrong_answer") OR reask("likely_dissatisfied") OR projection("feedback_escalate")) AND NOT reask("persistently_dissatisfied")
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN system_prompt {
    enabled: true
    system_prompt: "The previous answer likely failed in execution. Diagnose the failing assumption, explain the bug briefly, and return corrected code plus the smallest useful validation step."
    mode: "insert"
  }
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "code-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_persistent_recovery (description = "Premium general recovery lane for repeated same-question retries that have already failed on a cheaper recovery pass.") {
  PRIORITY 210
  WHEN reask("persistently_dissatisfied") AND NOT (projection("feedback_needs_evidence") OR keyword("code_error_markers") OR domain("computer science"))
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN system_prompt {
    enabled: true
    system_prompt: "The user has re-asked essentially the same question multiple times. Treat this as a hard recovery task: do not paraphrase the old answer, rebuild the answer from first principles, and make the improvement obvious."
    mode: "insert"
  }
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "persistent-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_general_recovery (description = "General recovery lane for repeated dissatisfaction that is neither verification-heavy nor code-specific.") {
  PRIORITY 200
  WHEN (projection("feedback_retry") OR projection("feedback_escalate") OR reask("likely_dissatisfied")) AND NOT (reask("persistently_dissatisfied") OR projection("feedback_needs_evidence") OR keyword("code_error_markers") OR domain("computer science"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN system_prompt {
    enabled: true
    system_prompt: "The user is likely dissatisfied with the previous answer. Do not repeat the earlier wording. Answer the same question again with a cleaner structure, tighter reasoning, and one concrete improvement over the last reply."
    mode: "insert"
  }
  PLUGIN header_mutation {
    add: [{ name: "X-Feedback-Lane", value: "general-recovery" }]
    update: [{ name: "X-Route-Source", value: "feedback-recipe" }]
  }
}

ROUTE feedback_need_clarification (description = "Keep explicit clarification follow-ups on the cheap lane unless dissatisfaction has clearly escalated.") {
  PRIORITY 180
  WHEN (user_feedback("need_clarification") OR keyword("clarification_feedback_markers")) AND (context("short_context") OR context("medium_context")) AND NOT (reask("persistently_dissatisfied") OR projection("feedback_verified_escalate") OR user_feedback("wrong_answer") OR keyword("verification_markers") OR keyword("code_error_markers"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN system_prompt {
    enabled: true
    system_prompt: "The user wants a clearer restatement, not a brand-new policy. Rewrite the answer simply, keep it concise, and give one small example when it helps."
    mode: "insert"
  }
  PLUGIN semantic_cache {
    enabled: true
    similarity_threshold: 0.86
    ttl_seconds: 1800
  }
}

