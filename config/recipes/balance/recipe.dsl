# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL conversation has_image {
  description: "The request carries at least one image."
  feature: { source: { type: "image_content" }, type: "exists" }
}

SIGNAL conversation has_prior_answer {
  description: "The conversation already holds an assistant answer."
  feature: { source: { role: "assistant", type: "message" }, type: "exists" }
}

SIGNAL decision task {
  description: "The kind of work the request asks for; its most likely answer picks the work lane."
  question: { choices: [{ description: "Greetings, small talk, questions about the assistant itself, or a simple everyday explanation of how a household thing works", key: "chat" }, { description: "Creative writing or drafting: names, taglines, poems, stories, or personal messages and replies", key: "creative" }, { description: "Writing, debugging, testing or reviewing code, or fixing a software bug", key: "code" }, { description: "A mathematical proof or a formal derivation, such as proving a theorem", key: "proof" }, { description: "An engineering plan or design, such as an architecture, a migration or a runbook", key: "systems" }, { description: "A hard science problem that needs expert multi-step reasoning, such as analyzing physical or chemical mechanisms or designing experiments", key: "science" }, { description: "Legal or compliance analysis of contracts, regulations or obligations", key: "legal" }, { description: "Health, symptoms, medicine or treatment", key: "health" }, { description: "Philosophy or ethics: comparing moral frameworks or forms of reasoning such as deduction and induction, and arguing for a position", key: "argument" }, { description: "Explaining, comparing or verifying ideas in business, economics, history or psychology, such as pricing, markets, historical events, human behavior, biases and fallacies", key: "explain" }, { description: "A short factual question with a direct, well-known answer, such as a basic fact of science, nature, geography or history, even when the user asks to verify it or cite a source", key: "facts" }], instructions: "What kind of work does this request ask for?", type: "choice" }
  predicate: { gte: 0.4 }
}

SIGNAL decision verification {
  description: "The user asks for sources, citations or evidence."
  question: { choices: [{ description: "No: no verification or sources are requested", key: "false" }, { description: "Yes: the user asks for verification, sources or evidence", key: "true" }], instructions: "Does the user ask to verify the answer, or to back it with sources, citations or evidence?", type: "noul" }
  predicate: { gte: 0.5 }
}

SIGNAL decision correction {
  description: "The user says the assistant's previous answer was wrong."
  question: { choices: [{ description: "No: the user continues, asks something new, or is satisfied", key: "false" }, { description: "Yes: the user says the previous answer was wrong", key: "true" }], instructions: "In the latest message, does the user say the assistant's previous answer was wrong and ask for a corrected answer?", type: "noul" }
  predicate: { gte: 0.7 }
}

SIGNAL decision clarify {
  description: "The user asks for the previous answer again, more clearly."
  question: { choices: [{ description: "No: a new request, a correction or satisfaction", key: "false" }, { description: "Yes: the previous answer should be explained again more clearly", key: "true" }], instructions: "In the latest message, does the user ask the assistant to explain its previous answer again, more clearly, more simply or with an example?", type: "noul" }
  predicate: { gte: 0.6 }
}

# =============================================================================
# MODELS
# =============================================================================

MODEL anthropic/claude-opus-4.6 {
  context_window_size: 262144
  description: "PREMIUM tier alias reserved for legal and high-risk analysis."
  capabilities: ["legal_analysis", "policy_review", "high_risk_review"]
  tags: ["tier:premium", "cost:highest", "specialty:legal"]
  modality: "text"
}

MODEL google/gemini-2.5-flash-lite {
  context_window_size: 262144
  description: "MEDIUM tier alias for low-cost verified explanation and correction tasks."
  capabilities: ["verified_explanation", "source_backed_correction", "nuanced_explanation"]
  tags: ["tier:medium", "cost:low", "specialty:verified"]
  modality: "text"
}

MODEL google/gemini-3.1-pro {
  context_window_size: 262144
  description: "COMPLEX tier alias for systems design, hard STEM, health guidance, and deep general reasoning."
  capabilities: ["architecture", "stem_analysis", "long_context", "general_reasoning"]
  tags: ["tier:complex", "cost:upper_mid", "specialty:complex_generalist"]
  modality: "text"
}

MODEL local/omni {
  capabilities: ["chat", "image_understanding", "multimodal", "omni", "text", "vision"]
  modality: "omni"
}

MODEL openai/gpt5.4 {
  context_window_size: 262144
  description: "REASONING tier alias for narrow formal math proofs and derivations."
  capabilities: ["reasoning", "proofs", "formal_derivation"]
  tags: ["tier:reasoning", "cost:high", "specialty:formal_proof"]
  modality: "text"
}

MODEL qwen/qwen3.5-rocm {
  context_window_size: 262144
  description: "SIMPLE tier alias and free self-hosted default for fast QA, broad fallback, creative drafting, and most low-cost traffic."
  capabilities: ["fast_qa", "self_hosted", "concise_answers", "general_chat", "creative_drafting"]
  tags: ["tier:simple", "cost:free", "deployment:self_hosted", "traffic:default"]
  modality: "text"
}

# =============================================================================
# PLUGINS
# =============================================================================

PLUGIN router_replay router_replay {}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE omni (description = "Image requests go to the visual-language model.") {
  PRIORITY 300
  TIER 1
  WHEN conversation("has_image")
  MODEL "local/omni" (reasoning = false)
  ALGORITHM static
}

ROUTE premium_legal (description = "Legal and compliance analysis (task: legal) goes to the premium lane.", on_unknown = "no_match") {
  PRIORITY 260
  TIER 2
  WHEN decision("task", label: "legal")
  MODEL "anthropic/claude-opus-4.6" (reasoning = true, effort = "high"),
        "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE formal_math_proof (description = "Mathematical proofs and derivations (task: proof) go to the formal reasoning lane.", on_unknown = "no_match") {
  PRIORITY 252
  TIER 3
  WHEN decision("task", label: "proof")
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high"),
        "anthropic/claude-opus-4.6" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE reasoning_deep (description = "Comparing ideas and arguing for a position (task: argument) goes to the deep reasoning lane.", on_unknown = "no_match") {
  PRIORITY 250
  TIER 4
  WHEN decision("task", label: "argument")
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "high"),
        "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE complex_specialist (description = "Engineering plans and designs (task: systems) and advanced science (task: science) go to the specialist lane.", on_unknown = "no_match") {
  PRIORITY 242
  TIER 5
  WHEN (decision("task", label: "systems") OR decision("task", label: "science"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "high"),
        "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE feedback_wrong_answer_verified (description = "A follow-up that says the previous answer was wrong (correction, after an answer) goes to the recovery lane.", on_unknown = "no_match") {
  PRIORITY 232
  TIER 6
  WHEN conversation("has_prior_answer") AND decision("correction")
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false),
        "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_code_general (description = "Coding and debugging (task: code) go to the code lane.", on_unknown = "no_match") {
  PRIORITY 220
  TIER 7
  WHEN decision("task", label: "code")
  MODEL "qwen/qwen3.5-rocm" (reasoning = true, mode = "enabled"),
        "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_health (description = "Health questions (task: health) go to the conservative health lane.", on_unknown = "no_match") {
  PRIORITY 218
  TIER 8
  WHEN decision("task", label: "health")
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium"),
        "anthropic/claude-opus-4.6" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_explainer (description = "Explanations that ask for sources (task: explain, verification) go to the verified lane.", on_unknown = "no_match") {
  PRIORITY 214
  TIER 9
  WHEN decision("task", label: "explain") AND decision("verification")
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false),
        "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE feedback_need_clarification (description = "A follow-up that asks to explain the previous answer more clearly (clarify, after an answer) stays on the low-cost lane.", on_unknown = "no_match") {
  PRIORITY 212
  TIER 10
  WHEN conversation("has_prior_answer") AND decision("clarify") AND NOT decision("correction")
  MODEL "qwen/qwen3.5-rocm" (reasoning = false),
        "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_explainer (description = "Explanations without a source request (task: explain) go to the explainer lane.", on_unknown = "no_match") {
  PRIORITY 208
  TIER 11
  WHEN decision("task", label: "explain") AND NOT decision("verification")
  MODEL "qwen/qwen3.5-rocm" (reasoning = true, mode = "enabled"),
        "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_creative (description = "Creative writing and drafting (task: creative) go to the creative lane.", on_unknown = "no_match") {
  PRIORITY 200
  TIER 12
  WHEN decision("task", label: "creative")
  MODEL "qwen/qwen3.5-rocm" (reasoning = false),
        "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE fast_qa (description = "Short factual questions (task: facts) go to the fast lane.", on_unknown = "no_match") {
  PRIORITY 184
  TIER 13
  WHEN decision("task", label: "facts")
  MODEL "qwen/qwen3.5-rocm" (reasoning = false),
        "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE simple_general (description = "Greetings and everyday questions (task: chat) go to the simple lane.", on_unknown = "no_match") {
  PRIORITY 170
  TIER 14
  WHEN decision("task", label: "chat")
  MODEL "qwen/qwen3.5-rocm" (reasoning = false),
        "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE casual_chat (description = "Everything else, including requests whose answers are unclear or unknown, goes to the lowest-cost lane.") {
  PRIORITY 10
  TIER 15
  MODEL "qwen/qwen3.5-rocm" (reasoning = false),
        "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vllm-sr/auto"]
  recipe: "default"
}
