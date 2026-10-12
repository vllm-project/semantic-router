# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL keyword ai_pc_local_only {
  operator: "OR"
  keywords: ["keep this local", "local only", "on this device only", "do not send this to the cloud", "do not use an external model", "仅在本地处理", "不要发送到云端", "不要使用外部模型"]
}

SIGNAL keyword ai_pc_external_ok {
  operator: "OR"
  keywords: ["you may use a cloud model", "an external model is fine", "cloud processing is allowed", "可以使用云端模型", "允许使用外部模型"]
}

SIGNAL context ai_pc_exceeds_local_window {
  description: "Input leaves too little room for a reply in the local model's window."
  min_tokens: "6K"
  max_tokens: "1M"
}

SIGNAL conversation ai_pc_has_images {
  description: "Request contains at least one image content part."
  feature: { source: { type: "image_content" }, type: "exists" }
}

# =============================================================================
# MODELS
# =============================================================================

MODEL cloud/external-chat {
  context_window_size: 1048576
  description: "External text model, used only when the request explicitly allows it."
  capabilities: ["chat", "code", "reasoning", "long-context", "text"]
  tags: ["deployment:openrouter", "role:external"]
  modality: "text"
}

MODEL local/ai-pc {
  context_window_size: 8192
  description: "Small text model served on the AI PC's CPU through a local Ollama server."
  capabilities: ["chat", "text"]
  tags: ["deployment:self_hosted", "role:local"]
  modality: "text"
}

# =============================================================================
# PLUGINS
# =============================================================================

PLUGIN fast_response fast_response {}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE unsupported_image_input (description = "Decline image input: this profile has no qualified local vision path.") {
  PRIORITY 300
  WHEN conversation("ai_pc_has_images")
  PLUGIN fast_response {
    message: "Image input is not supported by the AMD AI PC profile. Send text only, or use a recipe with a vision backend."
  }
}

ROUTE local_window_exceeded (description = "Decline input that leaves the local model too little room to reply, instead of sending it out silently.") {
  PRIORITY 250
  WHEN context("ai_pc_exceeds_local_window") AND (keyword("ai_pc_local_only") OR NOT keyword("ai_pc_external_ok"))
  PLUGIN fast_response {
    message: "This input is too large for the local model's context window. Shorten it, or explicitly allow an external model in the request."
  }
}

ROUTE local_only (description = "Keep requests marked local-only on the local model.") {
  PRIORITY 200
  WHEN keyword("ai_pc_local_only") AND NOT context("ai_pc_exceeds_local_window")
  MODEL "local/ai-pc" (reasoning = false)
  ALGORITHM static
}

ROUTE external_opt_in (description = "Use the external model only when the request allows it and does not ask to stay local.") {
  PRIORITY 150
  WHEN keyword("ai_pc_external_ok") AND NOT keyword("ai_pc_local_only")
  MODEL "cloud/external-chat" (reasoning = true, effort = "medium")
  ALGORITHM static
}

ROUTE local_default (description = "Process every other request locally.") {
  PRIORITY 10
  MODEL "local/ai-pc" (reasoning = false)
  ALGORITHM static
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vllm-sr/auto"]
  recipe: "default"
}
