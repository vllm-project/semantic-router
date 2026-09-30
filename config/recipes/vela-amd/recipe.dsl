# =============================================================================
# ROUTING PROFILE
# =============================================================================

ROUTING {
  model_bindings: { classifier.content-risk: { adapter: "modernbert", contract: "label_scores.v1", deployment: "hazard-amd", operating_point: { path: "operating_point.json", sha256: "e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5" } }, domain_classifier: { adapter: "modernbert", contract: "label_distribution.v1", deployment: "domain-amd", mapping_path: "models/Vela-1.0-Encoder-307M-Domain/category_mapping.json" }, embedding: { adapter: "mmbert", contract: "embedding.v1", deployment: "embedding-amd", head: "onnx/model_fa.onnx" }, fact_check_classifier: { adapter: "modernbert", contract: "label_distribution.v1", deployment: "factcheck-amd" }, feedback_detector: { adapter: "modernbert", contract: "label_distribution.v1", deployment: "feedback-amd" }, modality_detector: { adapter: "modernbert", contract: "label_distribution.v1", deployment: "modality-amd" }, pii_classifier: { adapter: "modernbert", contract: "token_spans.v1", deployment: "pii-amd", mapping_path: "models/Vela-1.0-Encoder-307M-PII/pii_mapping.json" }, prompt_guard: { adapter: "modernbert", contract: "label_distribution.v1", deployment: "guard-amd", head: "onnx/model_rocm_8k.onnx", mapping_path: "models/Vela-1.0-Encoder-307M-Guard/jailbreak_type_mapping.json" }, rag.reranker: { adapter: "vela_reranker", contract: "relevance_scores.v1", deployment: "reranker-amd", head: "onnx/model_fa.onnx", pair_scorer: { dimension: 768, layer: 22 } }, safety.unsafe-content: { adapter: "modernbert", contract: "label_distribution.v1", deployment: "safety-amd" } }
}

# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL domain biology {
  description: "Observe biology using the published Vela model."
  mmlu_categories: ["biology"]
}

SIGNAL domain business {
  description: "Observe business using the published Vela model."
  mmlu_categories: ["business"]
}

SIGNAL domain chemistry {
  description: "Observe chemistry using the published Vela model."
  mmlu_categories: ["chemistry"]
}

SIGNAL domain "computer science" {
  description: "Observe computer science using the published Vela model."
  mmlu_categories: ["computer science"]
}

SIGNAL domain economics {
  description: "Observe economics using the published Vela model."
  mmlu_categories: ["economics"]
}

SIGNAL domain engineering {
  description: "Observe engineering using the published Vela model."
  mmlu_categories: ["engineering"]
}

SIGNAL domain health {
  description: "Observe health using the published Vela model."
  mmlu_categories: ["health"]
}

SIGNAL domain history {
  description: "Observe history using the published Vela model."
  mmlu_categories: ["history"]
}

SIGNAL domain law {
  description: "Observe law using the published Vela model."
  mmlu_categories: ["law"]
}

SIGNAL domain math {
  description: "Observe math using the published Vela model."
  mmlu_categories: ["math"]
}

SIGNAL domain other {
  description: "Observe other using the published Vela model."
  mmlu_categories: ["other"]
}

SIGNAL domain philosophy {
  description: "Observe philosophy using the published Vela model."
  mmlu_categories: ["philosophy"]
}

SIGNAL domain physics {
  description: "Observe physics using the published Vela model."
  mmlu_categories: ["physics"]
}

SIGNAL domain psychology {
  description: "Observe psychology using the published Vela model."
  mmlu_categories: ["psychology"]
}

SIGNAL keyword knowledge-query {
  operator: "OR"
  keywords: ["Search my documents"]
}

SIGNAL embedding semantic-code {
  threshold: 0.65
  candidates: ["Debug this Python program and fix its error."]
  aggregation_method: "max"
}

SIGNAL fact_check needs_fact_check {
  description: "Observe needs_fact_check using the published Vela model."
}

SIGNAL fact_check no_fact_check_needed {
  description: "Observe no_fact_check_needed using the published Vela model."
}

SIGNAL user_feedback wrong_answer {
  description: "Observe wrong_answer using the published Vela model."
}

SIGNAL user_feedback need_clarification {
  description: "Observe need_clarification using the published Vela model."
}

SIGNAL user_feedback want_different {
  description: "Observe want_different using the published Vela model."
}

SIGNAL user_feedback satisfied {
  description: "Observe satisfied using the published Vela model."
}

SIGNAL modality AR {
}

SIGNAL modality DIFFUSION {
}

SIGNAL modality BOTH {
}

SIGNAL jailbreak prompt-attack {
  threshold: 0.5
}

SIGNAL safety unsafe-content {
  model: ""
  labels: ["safe", "unsafe"]
  unsafe_labels: ["unsafe"]
  threshold: 0.5
}

SIGNAL pii sensitive {
  threshold: 0.9
  include_history: true
}

SIGNAL classifier content-risk {
  type: "local"
  labels: ["violence", "criminal_activity", "sexual_content", "child_exploitation", "hate", "harassment_abuse", "regulated_substances", "weapons", "self_harm", "privacy", "specialized_advice", "misinformation"]
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

ROUTE knowledge (description = "Retrieve indexed documents and rerank the candidates before generation.", on_unknown = "fail_request") {
  PRIORITY 200
  WHEN keyword("knowledge-query")
  MODEL "vela-default" (reasoning = false)
  ALGORITHM static
  PLUGIN rag {
    enabled: true
    backend: "vectorstore"
    similarity_threshold: 0
    top_k: 3
    max_context_length: 4096
    injection_mode: "tool_role"
    rerank: { top_k: 2 }
    backend_config: { vector_store_id: "vs_your_documents" }
    on_failure: "block"
  }
}

ROUTE observe (description = "Inspect Vela signals and send the request to the connected backend.", on_unknown = "fail_request") {
  PRIORITY 100
  WHEN (jailbreak("prompt-attack") OR safety("unsafe-content") OR pii("sensitive") OR classifier("content-risk", label: "violence") OR classifier("content-risk", label: "criminal_activity") OR classifier("content-risk", label: "sexual_content") OR classifier("content-risk", label: "child_exploitation") OR classifier("content-risk", label: "hate") OR classifier("content-risk", label: "harassment_abuse") OR classifier("content-risk", label: "regulated_substances") OR classifier("content-risk", label: "weapons") OR classifier("content-risk", label: "self_harm") OR classifier("content-risk", label: "privacy") OR classifier("content-risk", label: "specialized_advice") OR classifier("content-risk", label: "misinformation") OR fact_check("needs_fact_check") OR user_feedback("wrong_answer") OR user_feedback("need_clarification") OR user_feedback("want_different") OR user_feedback("satisfied") OR modality("AR") OR embedding("semantic-code") OR domain("biology") OR domain("business") OR domain("chemistry") OR domain("computer science") OR domain("economics") OR domain("engineering") OR domain("health") OR domain("history") OR domain("law") OR domain("math") OR domain("other") OR domain("philosophy") OR domain("physics") OR domain("psychology"))
  MODEL "vela-default" (reasoning = false)
  ALGORITHM static
}
