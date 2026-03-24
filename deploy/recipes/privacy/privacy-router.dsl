# =============================================================================
# PRIVACY ROUTER
# =============================================================================
#
# This recipe keeps sensitive or suspicious traffic on a local model, only
# sends clearly non-sensitive high-reasoning work to a cloud frontier model,
# and records every routing decision for audit. It intentionally avoids the
# preference signal family so routing stays policy-driven rather than
# preference-inferred.

# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL keyword local_only_markers {
  operator: "OR"
  keywords: ["local processing only", "local model only", "on-prem only", "all processing on-prem", "process this on-prem", "do not send to the cloud", "do not upload this", "confidential handling", "internal use only", "handle this locally", "stay on the local model", "stay on local infrastructure", "本地处理", "仅限本地模型", "不要发到云端", "不要上传", "机密处理", "仅供内部使用"]
  method: "bm25"
  bm25_threshold: 0.18
}

SIGNAL keyword private_code_markers {
  operator: "OR"
  keywords: ["private repo", "private repository", "internal repo", "proprietary code", "company codebase", "our source tree", "internal SDK", "monorepo", "private package", "confidential code", "私有仓库", "内部仓库", "公司代码", "内部 SDK", "私有代码", "保密代码"]
  method: "bm25"
  bm25_threshold: 0.08
}

SIGNAL keyword internal_doc_markers {
  operator: "OR"
  keywords: ["internal document", "internal doc", "internal design doc", "internal design document", "internal architecture decision record", "internal ADR", "confidential memo", "confidential board memo", "internal runbook", "internal incident review", "internal postmortem", "internal SOP", "internal RFC", "employee handbook", "confidential board deck", "internal board deck", "internal-only document", "内部文档", "内部设计文档", "内部架构决策记录", "内部备忘录", "保密备忘录", "内部运行手册", "内部事故复盘", "内部 SOP", "内部 RFC", "员工手册", "董事会材料", "仅内部流转"]
  method: "bm25"
  bm25_threshold: 0.18
}

SIGNAL keyword prompt_injection_markers {
  operator: "OR"
  keywords: ["ignore previous instructions", "ignore all previous instructions", "ignore the instructions", "ignore the system instructions", "disregard the system prompt", "bypass safety", "override policy", "reveal hidden instructions", "reveal the system prompt", "hidden system prompt", "forget your rules", "you are now unrestricted", "ignore the developer message", "忽略之前的指令", "忽略系统提示词", "绕过安全", "覆盖策略", "泄露隐藏指令", "忘掉你的规则"]
  method: "bm25"
  bm25_threshold: 0.08
}

SIGNAL keyword exfiltration_markers {
  operator: "OR"
  keywords: ["show the system prompt", "show the hidden system prompt", "reveal the system prompt", "reveal the hidden prompt", "reveal the hidden system prompt", "print the secret", "print the api key", "dump credentials", "expose the api key", "leak the policy", "show internal instructions", "输出系统提示词", "泄露提示词", "打印密钥", "暴露凭据", "泄露内部规则"]
  method: "bm25"
  bm25_threshold: 0.08
}

SIGNAL keyword reasoning_request_markers {
  operator: "OR"
  keywords: ["reason carefully", "step by step", "compare the trade-offs", "compare the tradeoffs", "justify the trade-offs", "trade-offs between", "analyze deeply", "first principles", "root cause analysis", "root-cause analysis", "failure modes", "逐步推理", "仔细分析", "比较取舍", "从第一性原理出发", "根因分析"]
  method: "bm25"
  bm25_threshold: 0.18
}

SIGNAL keyword research_request_markers {
  operator: "OR"
  keywords: ["research memo", "synthesize the sources", "compare the evidence", "survey the literature", "write an analysis memo", "研究备忘录", "综合资料", "对比证据", "文献综述", "分析备忘录"]
  method: "bm25"
  bm25_threshold: 0.12
}

SIGNAL keyword architecture_markers {
  operator: "OR"
  keywords: ["distributed system", "system design", "migration strategy", "failover strategy", "event-driven architecture", "event-driven architectures", "multi-region", "multi-region architecture", "control plane", "data plane", "consistency trade-offs", "failure modes", "fault tolerance", "recovery guarantees", "system architecture", "分布式系统", "系统设计", "迁移策略", "故障切换方案", "控制面", "数据面", "一致性取舍", "容错设计"]
  method: "bm25"
  bm25_threshold: 0.12
}

SIGNAL keyword multi_step_markers {
  operator: "OR"
  keywords: ["phased plan", "execution plan", "checklist", "runbook", "rollback plan", "implementation steps", "分阶段方案", "实施计划", "检查清单", "运行手册", "回滚方案", "实施步骤"]
  method: "bm25"
  bm25_threshold: 0.1
}

SIGNAL embedding private_code_request {
  threshold: 0.79
  candidates: ["Review source code from a private repository and keep all processing on-prem.", "Debug proprietary backend code from our internal monorepo without sending it to a cloud model.", "Refactor a confidential microservice implementation from our company codebase.", "Analyze an internal SDK source file and suggest code changes locally.", "帮我分析公司私有仓库里的源代码，不要发到云端。"]
  aggregation_method: "max"
}

SIGNAL embedding pii_request {
  threshold: 0.82
  candidates: ["Review an HR spreadsheet containing employee phone numbers, home addresses, and government identifiers.", "Analyze a CRM export with names, phone numbers, street addresses, and account numbers.", "Process payroll records that include passport numbers, bank accounts, and home addresses.", "Summarize a file that contains employee personal identifiers and contact details locally.", "帮我处理包含手机号、家庭住址和身份证号的员工信息，并且只在本地分析。"]
  aggregation_method: "max"
}

SIGNAL embedding internal_document_request {
  threshold: 0.9
  candidates: ["Summarize this internal design document and highlight the risks.", "Extract action items from a confidential board memo.", "Summarize this internal runbook, extract the action items, and keep it on the local model.", "Analyze a private incident postmortem for follow-up actions.", "帮我总结这份内部设计文档，只能在本地模型处理，不要发到云端。"]
  aggregation_method: "max"
}

SIGNAL embedding frontier_reasoning_request {
  threshold: 0.88
  candidates: ["Compare several distributed system architectures and recommend the best one with explicit trade-offs and failure modes.", "Produce a rigorous root-cause analysis for a multi-service production outage.", "Evaluate multi-region failover strategies and justify the recovery guarantees.", "Synthesize multiple technical constraints into an architecture recommendation with explicit alternatives.", "从第一性原理比较多种系统方案，并给出带取舍的架构建议。"]
  aggregation_method: "max"
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

SIGNAL structure many_questions {
  description: "Prompts with several explicit questions, usually indicating multi-part reasoning."
  feature: { source: { pattern: "[?？]", type: "regex" }, type: "count" }
  predicate: { gte: 4 }
}

SIGNAL structure ordered_workflow {
  description: "Prompts with ordered workflow markers."
  feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["先", "再"], ["首先", "然后"]], type: "sequence" }, type: "sequence" }
}

SIGNAL structure override_directive_dense {
  description: "Override-oriented instruction language is dense relative to prompt length."
  feature: { source: { keywords: ["ignore", "override", "bypass", "reveal", "forget", "system prompt", "developer message", "忽略", "覆盖", "绕过", "泄露", "提示词"], type: "keyword_set" }, type: "density" }
  predicate: { gt: 0.06 }
}

SIGNAL structure fenced_instruction_blob {
  description: "Prompt contains fenced or tagged instruction wrappers often used in injection attempts."
  feature: { source: { pattern: "(?i)(<system>|<assistant>|begin system prompt|```system|###\\s*system)", type: "regex" }, type: "exists" }
}

SIGNAL complexity frontier_reasoning {
  threshold: 0.12
  description: "General reasoning boundary between local handling and frontier-cloud escalation."
  hard: { candidates: ["compare several approaches and justify the trade-offs", "build a rigorous argument from first principles", "perform a deep root-cause analysis across multiple systems", "synthesize many constraints into one recommendation", "write a detailed strategy memo with alternatives"] }
  easy: { candidates: ["give a short definition", "answer briefly", "rewrite this paragraph", "explain the error in one paragraph", "summarize this simply"] }
}

SIGNAL complexity code_reasoning {
  threshold: 0.12
  description: "Coding difficulty boundary for simple local help versus frontier-scale engineering analysis."
  hard: { candidates: ["design a distributed service with rollback and failure handling", "debug a production race condition across multiple services", "plan a large-scale refactor with compatibility constraints", "analyze deep performance trade-offs across a codebase"] }
  easy: { candidates: ["explain what this function does", "fix a small bug", "rename variables for clarity", "write a helper function", "summarize the stack trace"] }
}

SIGNAL jailbreak jailbreak_strict {
  method: "classifier"
  threshold: 0.45
  description: "Strict jailbreak classifier for routing suspicious prompts into local containment."
}

SIGNAL pii pii_strict {
  threshold: 0.85
  description: "Detect personally identifiable information that should remain on local infrastructure."
}

PROJECTION score security_risk_score {
  method: "weighted_sum"
  inputs: [{ type: "jailbreak", name: "jailbreak_strict", weight: 0.82 }, { type: "keyword", name: "prompt_injection_markers", weight: 0.48, value_source: "confidence" }, { type: "keyword", name: "exfiltration_markers", weight: 0.5, value_source: "confidence" }, { type: "structure", name: "override_directive_dense", weight: 0.6 }, { type: "structure", name: "fenced_instruction_blob", weight: 0.35 }]
}

PROJECTION score privacy_risk_score {
  method: "weighted_sum"
  inputs: [{ type: "pii", name: "pii_strict", weight: 0.92 }, { type: "keyword", name: "local_only_markers", weight: 0.15, value_source: "confidence" }, { type: "keyword", name: "private_code_markers", weight: 0.5, value_source: "confidence" }, { type: "keyword", name: "internal_doc_markers", weight: 0.2, value_source: "confidence" }, { type: "embedding", name: "private_code_request", weight: 0.32, value_source: "confidence" }, { type: "embedding", name: "pii_request", weight: 0.4, value_source: "confidence" }, { type: "embedding", name: "internal_document_request", weight: 0.4, value_source: "confidence" }, { type: "context", name: "long_context", weight: 0.06 }]
}

PROJECTION score reasoning_pressure {
  method: "weighted_sum"
  inputs: [{ type: "keyword", name: "reasoning_request_markers", weight: 0.28, value_source: "confidence" }, { type: "keyword", name: "research_request_markers", weight: 0.12, value_source: "confidence" }, { type: "keyword", name: "architecture_markers", weight: 0.28, value_source: "confidence" }, { type: "keyword", name: "multi_step_markers", weight: 0.1, value_source: "confidence" }, { type: "embedding", name: "frontier_reasoning_request", weight: 0.48, value_source: "confidence" }, { type: "context", name: "medium_context", weight: 0.04 }, { type: "context", name: "long_context", weight: 0.16 }, { type: "structure", name: "many_questions", weight: 0.08 }, { type: "structure", name: "ordered_workflow", weight: 0.1 }, { type: "complexity", name: "frontier_reasoning:medium", weight: 0.08 }, { type: "complexity", name: "frontier_reasoning:hard", weight: 0.3 }, { type: "complexity", name: "code_reasoning:medium", weight: 0.02 }, { type: "complexity", name: "code_reasoning:hard", weight: 0.18 }]
}

PROJECTION mapping security_policy_band {
  source: "security_risk_score"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 10 }
  outputs: [{ name: "policy_security_standard", lt: 0.35 }, { name: "policy_security_local_only", gte: 0.35 }]
}

PROJECTION mapping privacy_policy_band {
  source: "privacy_risk_score"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 10 }
  outputs: [{ name: "policy_privacy_cloud_allowed", lt: 0.35 }, { name: "policy_privacy_local_only", gte: 0.35 }]
}

PROJECTION mapping reasoning_policy_band {
  source: "reasoning_pressure"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 10 }
  outputs: [{ name: "policy_local_reasoning", lt: 0.5 }, { name: "policy_frontier_reasoning", gte: 0.5 }]
}

# =============================================================================
# MODELS
# =============================================================================

MODEL local/private-qwen {
  context_window_size: 131072
  description: "Low-cost self-hosted lane for privacy-sensitive, suspicious, and standard local traffic."
  capabilities: ["self_hosted", "privacy_locality", "security_containment", "code", "internal_docs"]
  tags: ["deployment:self_hosted", "policy:local_first", "policy:privacy_first", "cost:free"]
  quality_score: 0.74
  modality: "text"
}

MODEL cloud/frontier-reasoning {
  context_window_size: 262144
  description: "High-cost cloud frontier lane reserved for non-sensitive deep reasoning and synthesis."
  capabilities: ["frontier_reasoning", "deep_synthesis", "architecture_review", "long_context"]
  tags: ["deployment:cloud", "policy:non_sensitive_only", "tier:frontier", "cost:high"]
  quality_score: 0.92
  modality: "text"
}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE local_security_containment (description = "Keep suspicious or jailbreak-like prompts on the local safety lane.") {
  PRIORITY 300
  TIER 1
  WHEN projection("policy_security_local_only")
  MODEL "local/private-qwen" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 50000
    capture_request_body: false
    capture_response_body: false
    max_body_bytes: 2048
  }
}

ROUTE local_privacy_policy (description = "Route PII, private code, and internal documents to the local model according to the user policy bands.") {
  PRIORITY 250
  TIER 2
  WHEN projection("policy_privacy_local_only") AND NOT projection("policy_security_local_only")
  MODEL "local/private-qwen" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 50000
    capture_request_body: false
    capture_response_body: false
    max_body_bytes: 2048
  }
}

ROUTE cloud_frontier_reasoning (description = "Send only non-sensitive, non-suspicious high-reasoning traffic to the cloud frontier model.") {
  PRIORITY 200
  TIER 3
  WHEN projection("policy_frontier_reasoning") AND projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard")
  MODEL "cloud/frontier-reasoning" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 50000
    capture_request_body: false
    capture_response_body: false
    max_body_bytes: 2048
  }
}

ROUTE local_standard (description = "Default local route for non-sensitive tasks that do not justify cloud escalation.") {
  PRIORITY 100
  TIER 4
  WHEN projection("policy_local_reasoning") AND projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard")
  MODEL "local/private-qwen" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 50000
    capture_request_body: false
    capture_response_body: false
    max_body_bytes: 2048
  }
}
