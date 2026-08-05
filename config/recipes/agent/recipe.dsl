# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL domain "computer science" {
  description: "Programming, software systems, debugging, APIs, and infrastructure."
}

SIGNAL domain law {
  description: "Legal, compliance, policy, and regulatory topics."
}

SIGNAL domain health {
  description: "Health, medicine, clinical guidance, and patient-facing information."
}

SIGNAL domain math {
  description: "Mathematics, statistics, and quantitative reasoning."
}

SIGNAL domain physics {
  description: "Physics, chemistry, biology, engineering, and technical research."
}

SIGNAL domain business {
  description: "Business, product, operations, and management topics."
}

SIGNAL domain other {
  description: "General knowledge and miscellaneous topics."
}

SIGNAL keyword prompt_injection_markers {
  operator: "OR"
  keywords: ["\\bignore (all )?(the )?previous instructions\\b", "\\bignore the system instructions\\b", "\\bdisregard the system prompt\\b", "\\bbypass (every )?(safety|security) (rule|restriction)s?\\b", "\\boverride (the )?policy\\b", "\\breveal (the )?(hidden instructions|system prompt)\\b", "\\bforget (all )?(of )?your rules\\b", "忽略(所有)?之前的指令", "绕过(所有)?安全(限制|规则)", "(泄露|透露)系统提示词"]
  method: "regex"
}

SIGNAL keyword secret_markers {
  operator: "OR"
  keywords: ["\\bapi[_ -]?key\\b", "\\bsecret[_ -]?key\\b", "\\baccess[_ -]?token\\b", "\\boauth[_ -]?token\\b", "\\bid[_ -]?token\\b", "\\bprivate[_ -]?key\\b", "\\bssh[_ -]?key\\b", "\\bcredential(s)?\\b", "\\bpassword\\b", "\\bconnection[_ -]?string\\b", "\\bwebhook[_ -]?secret\\b", "\\bsigning[_ -]?secret\\b", "\\bbearer\\s+[A-Za-z0-9._~+/-]+=*\\b", "\\bsk-[A-Za-z0-9_-]{8,}\\b", "密钥", "凭据", "访问令牌", "私钥", "密码"]
  method: "regex"
}

SIGNAL keyword personal_data_markers {
  operator: "OR"
  keywords: ["\\b\\d{3}-\\d{2}-\\d{4}\\b", "\\bssn\\b", "\\bsocial security( number)?\\b", "\\bcredit card\\b", "\\bcard number\\b", "\\bbank account\\b", "\\biban\\b", "\\bpassport number\\b", "\\bdriver'?s? license\\b", "\\bnational id\\b", "\\bgovernment id\\b", "\\bmedical record\\b", "\\bpatient record\\b", "\\bdate of birth\\b", "\\bdob\\b", "\\bhome address\\b", "\\bphone number\\b", "\\bemail address\\b", "\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", "customer data", "employee data", "payroll record", "hr spreadsheet", "crm export", "客户数据", "员工数据", "身份证号", "手机号", "护照号", "银行卡号", "家庭住址"]
  method: "regex"
}

SIGNAL keyword private_code_markers {
  operator: "OR"
  keywords: ["\\bprivate repo(sitory)?\\b", "\\binternal repo(sitory)?\\b", "\\bproprietary(?: internal)? (?:source )?code\\b", "\\bcompany codebase\\b", "\\bour codebase\\b", "\\binternal sdk\\b", "\\bprivate package\\b", "\\bconfidential code\\b", "\\binternal monorepo\\b", "\\bproprietary source\\b", "私有仓库", "内部仓库", "公司代码", "私有代码", "保密代码"]
  method: "regex"
}

SIGNAL keyword internal_doc_markers {
  operator: "OR"
  keywords: ["\\binternal document\\b", "\\binternal doc\\b", "\\binternal design doc(ument)?\\b", "\\binternal ADR\\b", "\\binternal RFC\\b", "\\binternal runbook\\b", "\\binternal incident review\\b", "\\binternal postmortem\\b", "\\bconfidential memo\\b", "\\bconfidential board memo\\b", "\\bconfidential board deck\\b", "\\binternal board deck\\b", "\\binternal-only document\\b", "\\bemployee handbook\\b", "内部文档", "内部设计文档", "内部备忘录", "保密备忘录", "内部运行手册", "内部事故复盘", "仅内部流转"]
  method: "regex"
}

SIGNAL keyword local_only_markers {
  operator: "OR"
  keywords: ["\\blocal processing only\\b", "\\blocal model only\\b", "\\blocal only\\b", "\\blocally only\\b", "\\bon-prem(?:ises)? only\\b", "\\bkeep (this|processing) local\\b", "\\bdo not (send (this )?to the cloud|upload this)\\b", "\\bconfidential handling\\b", "\\binternal use only\\b", "\\bhandle this locally\\b", "\\buse (the )?local model\\b", "\\brun locally\\b", "\\bstay on the local model\\b", "\\bprivate infrastructure\\b", "仅本地处理", "不要发送到云端"]
  method: "regex"
}

SIGNAL keyword simple_request_markers {
  operator: "OR"
  keywords: ["\\bbriefly\\b", "\\bquick answer\\b", "\\bone sentence\\b", "\\bsimple explanation\\b", "\\bsummarize briefly\\b", "\\bwhat is\\b", "\\bwho is\\b", "\\bdefine\\b", "简单解释", "简短回答"]
  method: "regex"
}

SIGNAL keyword agentic_request_markers {
  operator: "OR"
  keywords: ["\\bagent(ic)?\\b", "\\btool (loop|call)\\b", "\\buse a tool\\b", "\\brun a command\\b", "\\b(read-only )?shell command\\b", "\\bplan and execute\\b", "\\brun until fixed\\b", "\\biterate until it passes\\b", "\\bcheckpoints?\\b", "\\brollback (plan|criteria)\\b", "\\bvalidation loop\\b", "\\bmulti-step workflow\\b", "分阶段实施", "回滚步骤"]
  method: "regex"
}

SIGNAL keyword code_request_markers {
  operator: "OR"
  keywords: ["\\bimplement\\b", "\\brefactor\\b", "\\bdebug\\b", "\\bstack trace\\b", "\\bfailing tests?\\b", "\\bpull request\\b", "\\brepositor(y|ies)\\b", "\\bcodebase\\b", "\\blint\\b", "\\bkubernetes controller\\b", "\\bdistributed system\\b", "\\brate limiter\\b"]
  method: "regex"
}

SIGNAL keyword legal_risk_markers {
  operator: "OR"
  keywords: ["\\bcontract risk\\b", "\\bcompliance\\b", "\\bindemnity\\b", "\\blimitation[- ]of[- ]liability\\b", "\\bprivacy policy\\b", "\\bregulatory\\b", "\\blegal memo\\b", "\\bdata transfer agreement\\b"]
  method: "regex"
}

SIGNAL keyword health_markers {
  operator: "OR"
  keywords: ["\\bsymptoms?\\b", "\\bdiagnos(is|e)\\b", "\\bmedical\\b", "\\bmedication\\b", "\\btreatment\\b", "\\bclinical\\b", "\\bdoctor\\b", "\\bblood pressure\\b"]
  method: "regex"
}

SIGNAL keyword research_request_markers {
  operator: "OR"
  keywords: ["\\bcite sources\\b", "\\bwith evidence\\b", "\\bcompare the evidence\\b", "\\bliterature review\\b", "\\bcompare studies\\b", "\\bsource-backed\\b", "\\breferences\\b", "\\bsurvey the literature\\b", "\\bresearch (approaches|methods?|synthesis)\\b"]
  method: "regex"
}

SIGNAL keyword reasoning_request_markers {
  operator: "OR"
  keywords: ["\\bcompare (the )?tradeoffs\\b", "\\broot cause\\b", "\\barchitecture\\b", "\\bdesign review\\b", "\\breason (carefully|step by step)\\b", "\\bevaluate options\\b", "\\bsynthesize\\b", "\\bfirst principles\\b"]
  method: "regex"
}

SIGNAL embedding fast_qa {
  threshold: 0.72
  candidates: ["What is the capital of France?", "Briefly explain what an API is.", "What does CPU stand for?", "Give me a quick summary of this paragraph.", "用一句话解释这个概念。"]
  aggregation_method: "max"
}

SIGNAL embedding coding_workflows {
  threshold: 0.75
  candidates: ["Debug this Python stack trace and explain the likely bug.", "Refactor this TypeScript module while preserving behavior.", "Implement the feature and update the tests.", "Fix the CI failure and rerun the relevant gate."]
  aggregation_method: "max"
}

SIGNAL embedding architecture_design {
  threshold: 0.78
  candidates: ["Design a distributed rate limiter and explain failure modes.", "Plan a migration from a monolith to event-driven microservices.", "Propose a consistency strategy for a global payments platform."]
  aggregation_method: "max"
}

SIGNAL embedding agentic_workflows {
  threshold: 0.76
  candidates: ["Create a migration plan with checkpoints, rollback, and validation.", "Troubleshoot the production issue and iterate until it is fixed.", "Break this project into tasks, dependencies, and acceptance checks.", "Design an execution workflow with milestones, owners, and guardrails.", "给我一个分阶段实施方案，并包含校验与回滚步骤。"]
  aggregation_method: "max"
}

SIGNAL embedding legal_analysis {
  threshold: 0.78
  candidates: ["Analyze indemnity, limitation of liability, and governing-law risks in this contract.", "Draft a legal-risk memo for a cross-border data transfer policy.", "Compare regulatory exposure under two compliance strategies."]
  aggregation_method: "max"
}

SIGNAL embedding health_guidance {
  threshold: 0.77
  candidates: ["Explain common causes of chest pain and when someone should seek urgent care.", "Compare evidence-backed treatment options for type 2 diabetes management.", "Summarize safe, clinically grounded steps for lowering high blood pressure."]
  aggregation_method: "max"
}

SIGNAL embedding research_synthesis {
  threshold: 0.77
  candidates: ["Compare several studies, cite the evidence, and explain the trade-offs.", "Write a source-backed memo that synthesizes multiple references.", "Survey the literature and recommend a position with supporting evidence."]
  aggregation_method: "max"
}

SIGNAL embedding business_analysis {
  threshold: 0.75
  candidates: ["Compare two pricing strategies for a B2B SaaS product.", "Analyze CAC, LTV, and retention trade-offs for a subscription business.", "Draft a market-entry strategy for a new AI tooling startup."]
  aggregation_method: "max"
}

SIGNAL embedding private_code_request {
  threshold: 0.79
  candidates: ["Review source code from a private repository and suggest a fix.", "Debug proprietary backend code from our internal monorepo.", "Refactor a confidential microservice from our company codebase.", "Analyze an internal SDK source file and explain the risks.", "帮我分析公司私有仓库里的源代码。"]
  aggregation_method: "max"
}

SIGNAL embedding pii_request {
  threshold: 0.82
  candidates: ["Review an HR spreadsheet containing employee phone numbers, home addresses, and government identifiers.", "Analyze a CRM export with names, phone numbers, street addresses, and account numbers.", "Process payroll records that include passport numbers, bank accounts, and home addresses.", "Summarize a file containing employee personal identifiers and contact details.", "帮我处理包含手机号、家庭住址和身份证号的员工信息。"]
  aggregation_method: "max"
}

SIGNAL embedding internal_document_request {
  threshold: 0.86
  candidates: ["Summarize this internal design document and highlight the risks.", "Extract action items from a confidential board memo.", "Analyze a private incident postmortem for follow-up actions.", "Summarize this internal runbook and identify owners.", "帮我总结这份内部设计文档并提炼风险。"]
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

SIGNAL structure ordered_workflow {
  feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["plan", "execute", "verify"], ["先", "再"]], type: "sequence" }, type: "sequence" }
}

SIGNAL structure numbered_steps {
  feature: { source: { pattern: "(?m)^\\s*\\d+\\.\\s+", type: "regex" }, type: "exists" }
}

SIGNAL structure constraint_dense {
  feature: { source: { keywords: ["under", "at most", "at least", "within", "no more than", "must", "should", "不超过", "至少"], type: "keyword_set" }, type: "density" }
  predicate: { gt: 0.08 }
}

SIGNAL conversation multi_turn_user {
  feature: { source: { role: "user", type: "message" }, type: "count" }
  predicate: { gte: 2 }
}

SIGNAL conversation has_tools {
  feature: { source: { type: "tool_definition" }, type: "count" }
  predicate: { gte: 1 }
}

SIGNAL conversation active_tool_use {
  feature: { source: { type: "assistant_tool_call" }, type: "count" }
  predicate: { gte: 1 }
}

SIGNAL conversation completed_tool_cycle {
  feature: { source: { type: "assistant_tool_cycle" }, type: "count" }
  predicate: { gte: 1 }
}

SIGNAL conversation current_tool_loop {
  feature: { source: { type: "active_tool_loop" }, type: "exists" }
}

SIGNAL conversation heavy_non_user {
  feature: { source: { role: "non_user", type: "message" }, type: "count" }
  predicate: { gt: 4 }
}

SIGNAL complexity general_reasoning {
  threshold: 0.14
  hard: { candidates: ["compare several approaches and justify the trade-offs", "synthesize constraints into a plan", "root-cause analysis for a complex failure", "derive the answer from first principles"] }
  easy: { candidates: ["brief definition", "quick summary", "simple explanation", "short direct answer"] }
}

SIGNAL complexity code_task {
  threshold: 0.12
  hard: { candidates: ["design a distributed system with failure handling", "debug a race condition in production", "refactor a large service boundary safely"] }
  easy: { candidates: ["explain what this function does", "fix a small bug in this code snippet", "write a helper function"] }
}

SIGNAL complexity agentic_delivery {
  threshold: 0.12
  hard: { candidates: ["create a migration plan with checkpoints, rollback, and validation", "troubleshoot this production issue until the root cause is fixed", "design an execution workflow with milestones, owners, and guardrails", "automate the process and include verification after each phase"] }
  easy: { candidates: ["give me a short checklist", "suggest the next step only", "write a small implementation plan"] }
}

SIGNAL complexity evidence_synthesis {
  threshold: 0.12
  hard: { candidates: ["compare several sources and recommend the strongest evidence-backed position", "write a source-backed memo that synthesizes multiple references", "survey the literature and explain conflicting evidence with citations"] }
  easy: { candidates: ["give a quick overview of the topic", "summarize one article briefly", "explain the term in simple language"] }
}

SIGNAL pii pii_strict {
  threshold: 0.85
}

PROJECTION score security_risk_score {
  method: "weighted_sum"
  inputs: [{ type: "keyword", weight: 0.6, name: "prompt_injection_markers", value_source: "confidence" }, { type: "keyword", weight: 0.2, name: "local_only_markers", value_source: "confidence" }]
}

PROJECTION score privacy_risk_score {
  method: "weighted_sum"
  inputs: [{ type: "pii", weight: 0.92, name: "pii_strict" }, { type: "keyword", weight: 0.12, name: "local_only_markers", value_source: "confidence" }, { type: "keyword", weight: 0.5, name: "private_code_markers", value_source: "confidence" }, { type: "keyword", weight: 0.46, name: "internal_doc_markers", value_source: "confidence" }, { type: "keyword", weight: 0.72, name: "secret_markers", value_source: "confidence" }, { type: "keyword", weight: 0.72, name: "personal_data_markers", value_source: "confidence" }, { type: "embedding", weight: 0.22, name: "private_code_request", value_source: "confidence" }, { type: "embedding", weight: 0.24, name: "pii_request", value_source: "confidence" }, { type: "embedding", weight: 0.22, name: "internal_document_request", value_source: "confidence" }, { type: "context", weight: 0.04, name: "long_context" }]
}

PROJECTION score difficulty_score {
  method: "weighted_sum"
  inputs: [{ type: "keyword", weight: -0.24, name: "simple_request_markers", value_source: "confidence" }, { type: "embedding", weight: -0.22, name: "fast_qa", value_source: "confidence" }, { type: "context", weight: -0.1, name: "short_context" }, { type: "context", weight: 0.06, name: "medium_context" }, { type: "context", weight: 0.18, name: "long_context" }, { type: "keyword", weight: 0.18, name: "reasoning_request_markers", value_source: "confidence" }, { type: "keyword", weight: 0.1, name: "code_request_markers", value_source: "confidence" }, { type: "keyword", weight: 0.14, name: "research_request_markers", value_source: "confidence" }, { type: "keyword", weight: 0.16, name: "agentic_request_markers", value_source: "confidence" }, { type: "embedding", weight: 0.12, name: "coding_workflows", value_source: "confidence" }, { type: "embedding", weight: 0.18, name: "architecture_design", value_source: "confidence" }, { type: "embedding", weight: 0.2, name: "agentic_workflows", value_source: "confidence" }, { type: "embedding", weight: 0.16, name: "research_synthesis", value_source: "confidence" }, { type: "structure", weight: 0.1, name: "ordered_workflow" }, { type: "structure", weight: 0.06, name: "numbered_steps" }, { type: "structure", weight: 0.05, name: "constraint_dense" }, { type: "complexity", weight: 0.08, name: "general_reasoning:medium" }, { type: "complexity", weight: 0.2, name: "general_reasoning:hard" }, { type: "complexity", weight: 0.16, name: "code_task:hard" }, { type: "complexity", weight: 0.22, name: "agentic_delivery:hard" }, { type: "complexity", weight: 0.18, name: "evidence_synthesis:hard" }]
}

PROJECTION score agentic_pressure {
  method: "weighted_sum"
  inputs: [{ type: "conversation", weight: 0.34, name: "current_tool_loop" }, { type: "conversation", weight: 0.04, name: "active_tool_use" }, { type: "conversation", weight: 0.04, name: "completed_tool_cycle" }, { type: "conversation", weight: 0.06, name: "has_tools" }, { type: "conversation", weight: 0.04, name: "multi_turn_user" }, { type: "conversation", weight: 0.04, name: "heavy_non_user" }, { type: "keyword", weight: 0.18, name: "agentic_request_markers", value_source: "confidence" }, { type: "embedding", weight: 0.06, name: "agentic_workflows", value_source: "confidence" }, { type: "structure", weight: 0.1, name: "ordered_workflow" }, { type: "complexity", weight: 0.18, name: "agentic_delivery:hard" }]
}

PROJECTION mapping security_policy_band {
  source: "security_risk_score"
  method: "threshold_bands"
  outputs: [{ name: "policy_security_standard", lt: 0.3 }, { name: "policy_security_local_only", gte: 0.3 }]
}

PROJECTION mapping privacy_policy_band {
  source: "privacy_risk_score"
  method: "threshold_bands"
  outputs: [{ name: "policy_privacy_cloud_allowed", lt: 0.35 }, { name: "policy_privacy_local_only", gte: 0.35 }]
}

PROJECTION mapping difficulty_band {
  source: "difficulty_score"
  method: "threshold_bands"
  outputs: [{ name: "balance_simple", lt: 0.2 }, { name: "balance_medium", gte: 0.2, lt: 0.45 }, { name: "balance_complex", gte: 0.45, lt: 0.7 }, { name: "balance_reasoning", gte: 0.7 }]
}

PROJECTION mapping agentic_band {
  source: "agentic_pressure"
  method: "threshold_bands"
  outputs: [{ name: "non_agentic", lt: 0.3 }, { name: "agentic_workflow", gte: 0.3 }]
}

# =============================================================================
# MODELS
# =============================================================================

MODEL anthropic/claude-opus-4.6 {
  context_window_size: 262144
  description: "Simulated high-care domain lane for non-private legal, compliance, and health analysis."
  capabilities: ["chat", "legal_analysis", "health_guidance", "high_stakes"]
  tags: ["deployment:vllm_alias", "tier:premium", "domain:high_care"]
  quality_score: 0.96
  modality: "text"
}

MODEL google/gemini-2.5-flash-lite {
  context_window_size: 262144
  description: "Simulated low-cost domain lane for medium difficulty explanations, follow-ups, and lightweight coding."
  capabilities: ["chat", "low_cost", "explanation", "coding"]
  tags: ["deployment:vllm_alias", "tier:medium"]
  quality_score: 0.84
  modality: "text"
}

MODEL google/gemini-3.1-pro {
  context_window_size: 262144
  description: "Simulated complex-domain lane for architecture, STEM, research synthesis, and difficult coding."
  capabilities: ["chat", "reasoning", "architecture", "research", "code"]
  tags: ["deployment:vllm_alias", "tier:complex", "domain:technical"]
  quality_score: 0.9
  modality: "text"
}

MODEL openai/gpt5.4 {
  context_window_size: 262144
  description: "Simulated frontier reasoning lane for hard non-private planning and long-horizon agent sessions."
  capabilities: ["chat", "frontier_reasoning", "planning", "synthesis"]
  tags: ["deployment:vllm_alias", "tier:frontier"]
  quality_score: 0.95
  modality: "text"
}

MODEL qwen/qwen3.6-rocm {
  context_window_size: 262144
  description: "Local AMD vLLM alias backed by Qwen3.6 for privacy-sensitive work, safety containment, simple traffic, and low-cost fallbacks."
  capabilities: ["chat", "privacy_locality", "private_code", "simple_qa", "tool_use"]
  tags: ["deployment:self_hosted", "tier:simple", "policy:privacy_first"]
  quality_score: 0.78
  modality: "text"
}

# =============================================================================
# PLUGINS
# =============================================================================

PLUGIN router_replay router_replay {}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE local_security_containment (description = "Keep prompt-injection or jailbreak-like prompts on the local AMD model.") {
  PRIORITY 340
  TIER 1
  WHEN projection("policy_security_local_only")
  MODEL "qwen/qwen3.6-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 2048
    max_tool_trace_steps: 100
  }
}

ROUTE local_privacy_policy (description = "Route private code, internal documents, credentials, and PII-like requests to the local AMD model.") {
  PRIORITY 320
  TIER 1
  WHEN projection("policy_privacy_local_only") AND NOT projection("policy_security_local_only")
  MODEL "qwen/qwen3.6-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 2048
    max_tool_trace_steps: 100
  }
}

ROUTE simple_math_fast_path (description = "Keep simple math and short factual STEM questions on the simple local alias.") {
  PRIORITY 300
  TIER 2
  WHEN projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard") AND domain("math") AND (projection("balance_simple") OR keyword("simple_request_markers"))
  MODEL "qwen/qwen3.6-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 2048
    max_tool_trace_steps: 100
  }
}

ROUTE domain_legal_health (description = "Non-private legal, compliance, and health requests route to the high-care domain alias.") {
  PRIORITY 260
  TIER 3
  WHEN projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard") AND (domain("law") OR domain("health") OR keyword("legal_risk_markers") OR keyword("health_markers") OR embedding("legal_analysis") OR embedding("health_guidance"))
  MODEL "anthropic/claude-opus-4.6" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    max_body_bytes: 4096
  }
}

ROUTE domain_code_complex (description = "Non-private complex coding, architecture, and software-system requests route to stronger technical aliases.") {
  PRIORITY 255
  TIER 3
  WHEN projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard") AND (keyword("code_request_markers") OR domain("computer science") AND (embedding("coding_workflows") OR embedding("architecture_design"))) AND (projection("balance_complex") OR projection("balance_reasoning") OR embedding("architecture_design") OR complexity("general_reasoning:hard") OR complexity("code_task:hard")) AND NOT projection("balance_simple") AND NOT embedding("fast_qa") AND NOT keyword("simple_request_markers") AND NOT keyword("research_request_markers")
  MODEL "google/gemini-3.1-pro" (reasoning = false),
        "openai/gpt5.4" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    max_body_bytes: 4096
  }
}

ROUTE domain_code (description = "Non-private medium coding, repository, and software-system requests route to the code-capable domain alias.") {
  PRIORITY 250
  TIER 3
  WHEN projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard") AND (keyword("code_request_markers") OR domain("computer science") AND embedding("coding_workflows")) AND NOT (projection("balance_complex") OR projection("balance_reasoning") OR keyword("reasoning_request_markers") OR embedding("architecture_design") OR complexity("general_reasoning:hard") OR complexity("code_task:hard")) AND NOT keyword("research_request_markers")
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false),
        "google/gemini-3.1-pro" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    max_body_bytes: 4096
  }
}

ROUTE domain_stem_research (description = "Non-private math, science, research, and evidence-heavy requests route to the STEM/research alias.") {
  PRIORITY 240
  TIER 3
  WHEN projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard") AND (domain("math") OR domain("physics") OR keyword("research_request_markers")) AND NOT (projection("balance_simple") OR domain("business") OR embedding("business_analysis") OR embedding("fast_qa") OR keyword("simple_request_markers"))
  MODEL "google/gemini-3.1-pro" (reasoning = false),
        "openai/gpt5.4" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    max_body_bytes: 4096
  }
}

ROUTE domain_business (description = "Non-private business and product-analysis requests route to the medium domain alias.") {
  PRIORITY 230
  TIER 3
  WHEN projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard") AND (domain("business") OR embedding("business_analysis")) AND NOT (projection("balance_complex") OR projection("balance_reasoning") OR keyword("reasoning_request_markers") OR embedding("architecture_design"))
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    max_body_bytes: 4096
  }
}

ROUTE complex_general (description = "Non-private complex or reasoning-heavy work routes to stronger aliases.") {
  PRIORITY 190
  TIER 4
  WHEN projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard") AND (projection("balance_complex") OR projection("balance_reasoning") OR keyword("reasoning_request_markers") OR keyword("agentic_request_markers") OR projection("agentic_workflow")) AND NOT (keyword("code_request_markers") OR domain("math") OR domain("physics") OR keyword("research_request_markers") OR domain("computer science") AND embedding("coding_workflows"))
  MODEL "google/gemini-3.1-pro" (reasoning = false),
        "openai/gpt5.4" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 10000
    max_body_bytes: 4096
  }
}

ROUTE medium_general (description = "Non-private medium-difficulty explanations and follow-ups route to the medium alias.") {
  PRIORITY 150
  TIER 5
  WHEN projection("policy_privacy_cloud_allowed") AND projection("policy_security_standard") AND projection("balance_medium")
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false)
}

ROUTE simple_general (description = "Simple or otherwise unmatched traffic uses the local fallback.") {
  PRIORITY 80
  TIER 6
  MODEL "qwen/qwen3.6-rocm" (reasoning = false)
}
