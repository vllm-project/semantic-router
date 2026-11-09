# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL conversation knowledge_has_images {
  description: "Request contains at least one image content part."
  feature: { source: { type: "image_content" }, type: "exists" }
}

SIGNAL decision knowledge_subject {
  description: "The subject a knowledge question is about."
  question: { choices: [{ description: "Biology, cells, organisms and genetics", key: "biology" }, { description: "Business, accounting, finance and management", key: "business" }, { description: "Computer science, algorithms and programming", key: "computer_science" }, { description: "Economics, markets and macroeconomics", key: "economics" }, { description: "Engineering, circuits, machines and materials", key: "engineering" }, { description: "Health, medicine, anatomy and nutrition", key: "health" }, { description: "History, wars, empires and historical events", key: "history" }, { description: "Law, contracts, courts and legal rules", key: "law" }, { description: "Mathematics, algebra, matrices and proofs", key: "math" }, { description: "Philosophy, ethics and logic", key: "philosophy" }, { description: "Physics, energy, forces and units", key: "physics" }, { description: "Chemistry, reactions, bonds and solutions", key: "chemistry" }, { description: "Psychology, behavior and learning", key: "psychology" }, { description: "General knowledge, geography, culture and everything else", key: "other" }], instructions: "Which subject is this question about?", type: "choice" }
  predicate: { gte: 0.4 }
}

# =============================================================================
# MODELS
# =============================================================================

MODEL cloud/frontier-72b {
  context_window_size: 262144
}

MODEL local/omni {
  capabilities: ["chat", "image_understanding", "multimodal", "omni", "text", "vision"]
  modality: "omni"
}

MODEL local/small-7b {
  context_window_size: 131072
}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE omni (description = "Understand image-bearing knowledge requests with the dedicated visual-language model.") {
  PRIORITY 300
  WHEN conversation("knowledge_has_images")
  MODEL "local/omni" (reasoning = false)
  ALGORITHM static
}

ROUTE escalate_72b (description = "Subjects where the larger model gains most (knowledge_subject: biology, business, computer science, economics, history, math, philosophy, psychology or other) escalate to the frontier 72B lane.", on_unknown = "no_match") {
  PRIORITY 200
  TIER 1
  WHEN (decision("knowledge_subject", label: "biology") OR decision("knowledge_subject", label: "business") OR decision("knowledge_subject", label: "computer_science") OR decision("knowledge_subject", label: "economics") OR decision("knowledge_subject", label: "history") OR decision("knowledge_subject", label: "math") OR decision("knowledge_subject", label: "other") OR decision("knowledge_subject", label: "philosophy") OR decision("knowledge_subject", label: "psychology"))
  MODEL "cloud/frontier-72b" (reasoning = false)
}

ROUTE keep_7b (description = "Lower-uplift subjects (engineering, health, law, physics, chemistry) and questions whose subject is unclear or unknown stay on the local 7B lane.") {
  PRIORITY 100
  TIER 2
  MODEL "local/small-7b" (reasoning = false)
}

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vllm-sr/auto"]
  recipe: "default"
}
