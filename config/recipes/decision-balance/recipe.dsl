# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL keyword brief_request {
  operator: "OR"
  keywords: ["answer briefly", "brief answer", "quick answer", "keep it short", "in one sentence", "one-word answer", "just the answer", "only the answer", "tl;dr", "简短回答", "简单回答", "一句话", "只要答案", "直接给答案"]
}

SIGNAL context beyond_qwen_window {
  description: "Long inputs that leave too little room on a 256K-token model."
  min_tokens: "200K"
}

SIGNAL conversation has_image {
  description: "The request carries at least one image."
  feature: { source: { type: "image_content" }, type: "exists" }
}

SIGNAL conversation has_tools {
  description: "The request declares at least one tool."
  feature: { source: { type: "tool_definition" }, type: "exists" }
}

SIGNAL conversation tool_loop {
  description: "The request continues an active tool loop."
  feature: { source: { type: "active_tool_loop" }, type: "exists" }
}

SIGNAL conversation has_prior_answer {
  description: "The conversation already holds an assistant answer."
  feature: { source: { role: "assistant", type: "message" }, type: "exists" }
}

SIGNAL jailbreak prompt_attack {
  threshold: 0.9
}

SIGNAL safety unsafe_request {
  model: ""
  labels: ["safe", "unsafe"]
  unsafe_labels: ["unsafe"]
  threshold: 0.9
}

SIGNAL decision task {
  description: "What kind of work the request asks for."
  question: { choices: [{ description: "Casual conversation, greetings, opinions or everyday advice", key: "chat" }, { description: "Writing, rewriting, translating or summarizing text", key: "writing" }, { description: "Writing, explaining, reviewing or debugging code", key: "code" }, { description: "A mathematics, science or engineering problem", key: "stem" }, { description: "Looking up a specific fact, name, date, number or quote", key: "facts" }, { description: "Comparing options, planning or reasoning through a question", key: "analysis" }, { description: "Multi-step work that uses tools, files, systems or the web", key: "agentic" }, { description: "Questions about a document, table or long text in the request", key: "document" }], instructions: "What kind of work does this request ask for?", type: "choice" }
}

SIGNAL decision difficulty {
  description: "How much reasoning a strong expert needs, from 0 (none) to 4 (research level)."
  question: { instructions: "How much reasoning does a strong expert need to answer this request well?", levels: ["Answer immediately, with no thinking", "A little thought, one or two simple steps", "Multi-step reasoning or a careful check", "Deep expert reasoning on a hard problem", "Research-level work on an open or frontier problem"], type: "score" }
  predicate: { gte: 2 }
}

SIGNAL decision precise_facts {
  description: "The answer depends on specific facts that most people would have to look up."
  question: { choices: [{ description: "Common knowledge, reasoning or creativity is enough", key: "false" }, { description: "Specific facts that most people would have to look up", key: "true" }], instructions: "Does a correct answer depend on recalling specific facts that are easy to misremember, such as exact dates, figures, names or citations that most people would have to look up?", type: "noul" }
  predicate: { gte: 0.8 }
}

SIGNAL decision needs {
  description: "What a good answer needs."
  question: { instructions: "What does a good answer to this request need?", labels: [{ description: "A derivation, proof or careful step-by-step check", key: "deliberation" }, { description: "Calling tools or functions, running code or browsing", key: "tools" }, { description: "Checking facts against sources or citing references", key: "verification" }, { description: "A long answer such as a full program, report or essay", key: "long_output" }, { description: "Original creative writing such as a story, poem or slogan", key: "creativity" }], type: "set" }
}

SIGNAL decision correction {
  description: "The user says the assistant's previous answer was wrong."
  question: { choices: [{ description: "No: the user continues, asks something new, or is satisfied", key: "false" }, { description: "Yes: the user says the previous answer was wrong", key: "true" }], instructions: "In the latest message, does the user say the assistant's previous answer was wrong and ask for a corrected answer?", type: "noul" }
  predicate: { gte: 0.6 }
}

PROJECTION score effort_score {
  method: "weighted_sum"
  inputs: [{ type: "decision", weight: 0.3, name: "difficulty", value_source: "raw" }, { type: "decision", weight: 0.35, name: "needs:deliberation", value_source: "raw" }, { type: "decision", weight: 0.1, name: "needs:verification", value_source: "raw" }, { type: "keyword", weight: -0.35, name: "brief_request" }]
}

PROJECTION mapping effort {
  source: "effort_score"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 10 }
  outputs: [{ name: "effort_off", lt: 0.4 }, { name: "effort_medium", gte: 0.4, lt: 0.8 }, { name: "effort_high", gte: 0.8, lt: 1.15 }, { name: "effort_max", gte: 1.15 }]
}

# =============================================================================
# MODELS
# =============================================================================

MODEL qwen/qwen3.8-27b {
  description: "Smallest and cheapest model; strong on documents and screenshots; one GPU per replica."
  tags: ["gpus:1", "role:default"]
}

MODEL qwen/qwen3.8-flash-next {
  description: "Fastest generation; strongest on code, competition mathematics and instruction following; two GPUs per replica."
  tags: ["gpus:2", "role:reasoning"]
}

MODEL zai/glm-5.3-flash {
  description: "Strongest model in the pool on agentic work, expert knowledge and exact facts; thinking cannot be turned off; four GPUs per replica."
  tags: ["gpus:4", "role:agentic", "role:facts"]
}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE guard (description = "Refuse prompt attacks and unsafe requests without calling a model.") {
  PRIORITY 1000
  WHEN (jailbreak("prompt_attack") OR safety("unsafe_request"))
  PLUGIN fast_response {
    message: "This request was declined by the router's safety guard. Please rephrase it."
  }
}

ROUTE long_context (description = "Inputs beyond the Qwen windows go to GLM-5.3-Flash, which reads up to 1M tokens.") {
  PRIORITY 900
  WHEN context("beyond_qwen_window")
  MODEL "zai/glm-5.3-flash" (reasoning = true, effort = "high")
  ALGORITHM static
}

ROUTE vision (description = "Image requests go to Qwen3.8-27B, strong on documents and screenshots and the cheapest model.") {
  PRIORITY 850
  WHEN conversation("has_image")
  MODEL "qwen/qwen3.8-27b" (reasoning = true, effort = "medium")
  ALGORITHM static
}

ROUTE recovery (description = "A user who says the last answer was wrong gets GLM-5.3-Flash at maximum effort, a different model family.", on_unknown = "no_match") {
  PRIORITY 800
  WHEN conversation("has_prior_answer") AND decision("correction")
  MODEL "zai/glm-5.3-flash" (reasoning = true, effort = "max")
  ALGORITHM static
}

ROUTE agentic (description = "Requests that call the declared tools go to GLM-5.3-Flash, the strongest agent in the pool.", on_unknown = "no_match") {
  PRIORITY 700
  WHEN (conversation("tool_loop") OR conversation("has_tools") AND (decision("task", label: "agentic") OR decision("needs", label: "tools", predicate: { gte: 0.5 })))
  MODEL "zai/glm-5.3-flash" (reasoning = true, effort = "high")
  ALGORITHM static
}

ROUTE facts (description = "Answers that depend on exact facts go to GLM-5.3-Flash, which hallucinates least and abstains when unsure.", on_unknown = "no_match") {
  PRIORITY 600
  WHEN decision("precise_facts") AND (decision("task", label: "facts") OR decision("task", label: "analysis") OR decision("task", label: "writing")) AND NOT projection("effort_max")
  MODEL "zai/glm-5.3-flash" (reasoning = true, effort = "low")
  ALGORITHM static
}

ROUTE frontier (description = "The hardest requests; the decision model chooses GLM-5.3-Flash max or Qwen3.8-Flash-Next xhigh.") {
  PRIORITY 500
  WHEN projection("effort_max")
  MODEL "zai/glm-5.3-flash" (reasoning = true, effort = "max"),
        "qwen/qwen3.8-flash-next" (reasoning = true, effort = "xhigh")
  ALGORITHM decision {
    decision: { candidates: { qwen/qwen3.8-flash-next: "Qwen3.8-Flash-Next: strongest on competition mathematics, code generation and precise instruction following", zai/glm-5.3-flash: "GLM-5.3-Flash: strongest on research-level science, expert knowledge and long agentic plans, and most reliable on exact facts" }, instructions: "Which model should answer this research-level request?", timeout_ms: 1000 }
  }
}

ROUTE hard (description = "Multi-step reasoning at extra-high effort; multi_factor balances quality, GPU cost, latency and load.") {
  PRIORITY 400
  WHEN projection("effort_high")
  MODEL "qwen/qwen3.8-flash-next" (reasoning = true, effort = "xhigh"),
        "qwen/qwen3.8-27b" (reasoning = true, effort = "xhigh")
  ALGORITHM multi_factor {
    latency_metric: "tpot"
    on_no_candidates: "cheapest"
    quality: { index: "vllm-sr/reasoning@1.0.0", on_missing: "exclude" }
    weights: { cost: 0.4, latency: 0.1, load: 0.25, quality: 0.25 }
  }
}

ROUTE standard (description = "Some reasoning at medium effort; multi_factor balances quality, GPU cost, latency and load.") {
  PRIORITY 300
  WHEN projection("effort_medium")
  MODEL "qwen/qwen3.8-flash-next" (reasoning = true, effort = "medium"),
        "qwen/qwen3.8-27b" (reasoning = true, effort = "medium")
  ALGORITHM multi_factor {
    latency_metric: "tpot"
    on_no_candidates: "cheapest"
    quality: { index: "decision-balance/operator-reasoning@1.0.0", on_missing: "exclude" }
    weights: { cost: 0.4, latency: 0.1, load: 0.25, quality: 0.25 }
  }
}

ROUTE fast (description = "Everything else goes to Qwen3.8-27B with thinking off.") {
  PRIORITY 100
  MODEL "qwen/qwen3.8-27b" (reasoning = false)
  ALGORITHM static
}
