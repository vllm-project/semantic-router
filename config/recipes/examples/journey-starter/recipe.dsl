# Journey starter routing DSL.
# Extend signals and decisions using config/fragments/ as reference.

DECISION "default-route" {
  priority: 100
  rules: []
  modelRefs: [{ model: "journey-starter-model", use_reasoning: false }]
  algorithm: static
