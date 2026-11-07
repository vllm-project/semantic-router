# =============================================================================
# ROUTING PROFILE
# =============================================================================

ROUTING {
  model_bindings: { embedding: "embedding-amd" }
}

# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL domain biology {
  description: "Observe biology using the published Vela model."
  labels: ["biology"]
  predicate: { gte: 0.5 }
}

SIGNAL domain business {
  description: "Observe business using the published Vela model."
  labels: ["business"]
  predicate: { gte: 0.5 }
}

SIGNAL domain chemistry {
  description: "Observe chemistry using the published Vela model."
  labels: ["chemistry"]
  predicate: { gte: 0.5 }
}

SIGNAL domain "computer science" {
  description: "Observe computer science using the published Vela model."
  labels: ["computer science"]
  predicate: { gte: 0.5 }
}

SIGNAL domain economics {
  description: "Observe economics using the published Vela model."
  labels: ["economics"]
  predicate: { gte: 0.5 }
}

SIGNAL domain engineering {
  description: "Observe engineering using the published Vela model."
  labels: ["engineering"]
  predicate: { gte: 0.5 }
}

SIGNAL domain health {
  description: "Observe health using the published Vela model."
  labels: ["health"]
  predicate: { gte: 0.5 }
}

SIGNAL domain history {
  description: "Observe history using the published Vela model."
  labels: ["history"]
  predicate: { gte: 0.5 }
}

SIGNAL domain law {
  description: "Observe law using the published Vela model."
  labels: ["law"]
  predicate: { gte: 0.5 }
}

SIGNAL domain math {
  description: "Observe math using the published Vela model."
  labels: ["math"]
  predicate: { gte: 0.5 }
}

SIGNAL domain other {
  description: "Observe other using the published Vela model."
  labels: ["other"]
  predicate: { gte: 0.5 }
}

SIGNAL domain philosophy {
  description: "Observe philosophy using the published Vela model."
  labels: ["philosophy"]
  predicate: { gte: 0.5 }
}

SIGNAL domain physics {
  description: "Observe physics using the published Vela model."
  labels: ["physics"]
  predicate: { gte: 0.5 }
}

SIGNAL domain psychology {
  description: "Observe psychology using the published Vela model."
  labels: ["psychology"]
  predicate: { gte: 0.5 }
}

SIGNAL keyword knowledge-query {
  operator: "OR"
  keywords: ["Search my documents"]
}

SIGNAL fact_check needs_fact_check {
  description: "Observe needs_fact_check using the published Vela model."
  predicate: { gte: 0.95 }
}

SIGNAL fact_check no_fact_check_needed {
  description: "Observe no_fact_check_needed using the published Vela model."
  labels: ["NO_FACT_CHECK_NEEDED"]
  predicate: { gt: 0.05 }
}

SIGNAL user_feedback wrong_answer {
  description: "Observe wrong_answer using the published Vela model."
  labels: ["WRONG_ANSWER"]
  predicate: { gte: 0.7 }
}

SIGNAL user_feedback need_clarification {
  description: "Observe need_clarification using the published Vela model."
  labels: ["NEED_CLARIFICATION"]
  predicate: { gte: 0.7 }
}

SIGNAL user_feedback want_different {
  description: "Observe want_different using the published Vela model."
  labels: ["WANT_DIFFERENT"]
  predicate: { gte: 0.7 }
}

SIGNAL user_feedback satisfied {
  description: "Observe satisfied using the published Vela model."
  labels: ["SAT"]
  predicate: { gte: 0.7 }
}

SIGNAL modality AR {
  labels: ["text"]
}

SIGNAL modality DIFFUSION {
  labels: ["image"]
}

SIGNAL modality BOTH {
  labels: ["text_image"]
}

SIGNAL jailbreak prompt-attack {
  predicate: { gte: 0.5 }
}

SIGNAL safety unsafe-content {
  predicate: { gte: 0.5 }
}

SIGNAL pii sensitive {
  predicate: { gte: 0.9 }
  input: { history: true }
}

SIGNAL systemone content-risk {
  question: { choices: [{ key: "violence" }, { key: "criminal_activity" }, { key: "sexual_content" }, { key: "child_exploitation" }, { key: "hate" }, { key: "harassment_abuse" }, { key: "regulated_substances" }, { key: "weapons" }, { key: "self_harm" }, { key: "privacy" }, { key: "specialized_advice" }, { key: "misinformation" }], instructions: "Which of these labels applies to the request?", type: "choice" }
}

SIGNAL systemone semantic-code {
  question: { instructions: "Is the request about any of these: Debug this Python program and fix its error.?", type: "noul" }
}

# =============================================================================
# MODELS
# =============================================================================

MODEL vela-default {
  capabilities: ["text"]
  modality: "ar"
}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE knowledge (description = "Retrieve indexed documents and keep the ones the decision model judges relevant before generation.", on_unknown = "fail_request") {
  PRIORITY 200
  WHEN keyword("knowledge-query")
  MODEL "vela-default" (reasoning = false)
  ALGORITHM static
  PLUGIN context {
    enabled: true
    sources: [{ backend: "vectorstore", backend_config: { vector_store_id: "vs_your_documents" }, match: { judge: { min_score: 0.5, preset: "relevance", top_k: 2 }, recall: { method: "embedding", min_similarity: 0, top_k: 3 } }, max_context_length: 4096, on_failure: "block", type: "rag" }]
  }
}

ROUTE observe (description = "Inspect Vela signals and send the request to the connected backend.", on_unknown = "fail_request") {
  PRIORITY 100
  WHEN (jailbreak("prompt-attack") OR safety("unsafe-content") OR pii("sensitive") OR systemone("content-risk", label: "violence") OR systemone("content-risk", label: "criminal_activity") OR systemone("content-risk", label: "sexual_content") OR systemone("content-risk", label: "child_exploitation") OR systemone("content-risk", label: "hate") OR systemone("content-risk", label: "harassment_abuse") OR systemone("content-risk", label: "regulated_substances") OR systemone("content-risk", label: "weapons") OR systemone("content-risk", label: "self_harm") OR systemone("content-risk", label: "privacy") OR systemone("content-risk", label: "specialized_advice") OR systemone("content-risk", label: "misinformation") OR fact_check("needs_fact_check") OR user_feedback("wrong_answer") OR user_feedback("need_clarification") OR user_feedback("want_different") OR user_feedback("satisfied") OR modality("AR") OR systemone("semantic-code") OR domain("biology") OR domain("business") OR domain("chemistry") OR domain("computer science") OR domain("economics") OR domain("engineering") OR domain("health") OR domain("history") OR domain("law") OR domain("math") OR domain("other") OR domain("philosophy") OR domain("physics") OR domain("psychology"))
  MODEL "vela-default" (reasoning = false)
  ALGORITHM static
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vela-auto"]
  recipe: "default"
}
