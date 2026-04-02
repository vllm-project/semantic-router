# =============================================================================
# SIGNALS
# =============================================================================

SIGNAL domain "computer science" {
  description: "Programming, software systems, debugging, APIs, and infrastructure."
}

SIGNAL domain math {
  description: "Mathematics, statistics, and quantitative reasoning."
}

SIGNAL domain physics {
  description: "Physics and physical sciences."
}

SIGNAL domain chemistry {
  description: "Chemistry and chemical sciences."
}

SIGNAL domain biology {
  description: "Biology and life sciences."
}

SIGNAL domain engineering {
  description: "Engineering and technical problem solving."
}

SIGNAL domain health {
  description: "Health, medicine, clinical guidance, and patient-facing information."
}

SIGNAL domain business {
  description: "Business, product, operations, and management topics."
}

SIGNAL domain economics {
  description: "Economics, pricing, incentives, and market dynamics."
}

SIGNAL domain law {
  description: "Legal, compliance, policy, and regulatory topics."
}

SIGNAL domain psychology {
  description: "Psychology, behavior, and mental models."
}

SIGNAL domain philosophy {
  description: "Philosophy, ethics, and abstract argumentation."
}

SIGNAL domain history {
  description: "Historical explanation, comparison, and context."
}

SIGNAL domain other {
  description: "General knowledge and miscellaneous topics."
}

SIGNAL keyword correction_feedback_markers {
  operator: "OR"
  keywords: ["that's wrong", "this is wrong", "that's incorrect", "that is incorrect", "incorrect", "wrong answer", "please correct", "correct the explanation", "correct this answer", "try again", "answer again", "re-answer", "fix your answer", "you got this wrong", "you got this wrong earlier", "错了", "不对", "回答错了", "重新回答", "再答一次"]
}

SIGNAL keyword clarification_feedback_markers {
  operator: "OR"
  keywords: ["explain that more clearly", "clarify your answer", "give one simple example", "restate that more simply", "that was confusing", "restate it more simply", "walk me through that again", "use one simple example", "讲清楚一点", "说得更清楚", "简单一点", "举个例子", "解释得更明白"]
}

SIGNAL keyword verification_markers {
  operator: "OR"
  keywords: ["verify this", "verify the claim", "verify with a source", "verify with sources", "verify with a source whether", "cite the source", "cite a reliable source", "cite sources", "with sources", "with a source", "with citations", "answer with citations", "with reliable sources", "with reputable sources", "cite reliable historical sources", "reliable historical sources", "reputable historical sources", "reliable medical sources", "is this true", "fact check this", "verify with evidence", "核实一下", "给出处", "请给出处", "这是真的吗", "请核验", "请给来源", "请核实并给出处", "请核实并给来源", "请核验并给来源"]
}

SIGNAL keyword reference_heavy_markers {
  operator: "OR"
  keywords: ["cite sources", "according to", "reference the paper", "compare the literature", "use case law", "relevant regulation", "cite the rfc", "support the answer with sources", "support the answer with reputable sources", "support the correction with sources", "historical sources", "reliable historical sources", "reputable historical sources", "medical sources", "verify with a source whether", "请核实并给来源", "请核实并给出处", "引用资料", "根据文献", "参考法规"]
}

SIGNAL keyword legal_risk_markers {
  operator: "OR"
  keywords: ["contract clause", "liability analysis", "compliance memo", "regulatory risk", "legal exposure", "indemnity", "jurisdiction", "lawsuit", "合同条款", "合规分析", "法律风险", "责任认定"]
  method: "bm25"
  bm25_threshold: 0.12
}

SIGNAL keyword simple_request_markers {
  operator: "OR"
  keywords: ["quick answer", "answer briefly", "keep it short", "one sentence", "simple explanation", "tl;dr", "briefly explain", "concise answer", "简短回答", "简单解释", "用一句话", "直接回答"]
}

SIGNAL keyword creative_request_markers {
  operator: "OR"
  keywords: ["brainstorm", "creative writing", "write a story", "write a poem", "imagine", "slogan", "tagline", "campaign idea", "make it more vivid", "rewrite creatively", "想点子", "头脑风暴", "写一个故事", "写一首诗"]
}

SIGNAL keyword multi_step_markers {
  operator: "OR"
  keywords: ["step by step", "phased plan", "implementation plan", "migration plan", "checklist", "roadmap", "runbook", "rollback plan", "rollback steps", "rollback criteria", "checkpoints", "owners", "dependencies", "verification gates", "validation steps", "validation after each phase", "分步骤", "实施计划", "路线图", "检查清单"]
}

SIGNAL keyword agentic_request_markers {
  operator: "OR"
  keywords: ["create a migration plan", "propose an execution plan", "refactor this system", "troubleshoot the issue", "implement the workflow", "automate the process", "break this into tasks", "create a runbook", "define rollback criteria", "include verification gates", "phase the rollout", "制定迁移计划", "排查这个问题", "实施这个方案", "自动化这个流程"]
}

SIGNAL keyword architecture_markers {
  operator: "OR"
  keywords: ["distributed system", "microservices", "rate limiter", "high availability", "consistency model", "sharding strategy", "event driven architecture", "reliability architecture", "system design", "fault tolerance", "observability architecture", "service boundaries", "storage boundaries", "cache strategy", "multi-region", "control plane", "data plane", "tradeoffs", "trade-offs"]
  method: "bm25"
  bm25_threshold: 0.12
}

SIGNAL keyword implementation_markers {
  operator: "OR"
  keywords: ["build the system", "create the service", "implement the solution", "design the workflow", "develop the pipeline", "deploy the stack", "configure the service", "write the implementation", "构建这个系统", "创建这个服务", "实现这个方案", "设计这个流程", "搭建这个管道", "部署这个系统"]
  method: "bm25"
  bm25_threshold: 0.12
}

SIGNAL keyword code_request_markers {
  operator: "OR"
  keywords: ["code", "function", "class", "stack trace", "api", "sql", "python", "typescript", "debug", "refactor", "bug", "endpoint", "algorithm", "recursion"]
  method: "bm25"
  bm25_threshold: 0.12
}

SIGNAL keyword history_topic_markers {
  operator: "OR"
  keywords: ["roman republic", "roman republic collapse", "ming dynasty", "ming dynasty fell", "meiji restoration", "empire", "dynasty", "revolution", "treaty", "monarchy", "republic", "historical collapse", "历史", "王朝", "帝国", "革命"]
  method: "bm25"
  bm25_threshold: 0.08
}

SIGNAL keyword research_request_markers {
  operator: "OR"
  keywords: ["literature review", "compare the evidence", "survey the field", "synthesize the sources", "summarize the research", "source-backed analysis", "policy memo", "文献综述", "对比证据", "综合资料", "总结研究现状", "基于来源分析"]
  method: "bm25"
  bm25_threshold: 0.12
}

SIGNAL keyword reasoning_request_markers {
  operator: "OR"
  keywords: ["think step by step", "reason carefully", "prove rigorously", "derive the formula", "compare trade-offs", "analyze the root cause", "evaluate multiple approaches", "formal proof", "逐步推理", "严格证明", "推导公式", "权衡利弊", "深入分析"]
  method: "bm25"
  bm25_threshold: 0.18
}

SIGNAL keyword urgency_markers {
  operator: "OR"
  keywords: ["urgent", "urgently", "asap", "right now", "immediately", "as soon as possible", "马上", "立刻", "立即", "尽快", "赶紧", "现在就"]
}

SIGNAL embedding fast_qa_en {
  threshold: 0.72
  candidates: ["Who are you?", "What does CPU stand for?", "What is the capital of France?", "Briefly explain what an API is.", "What is the boiling point of water?", "What is 2 + 2?", "Does light travel faster than sound?"]
  aggregation_method: "max"
}

SIGNAL embedding fast_qa_zh {
  threshold: 0.72
  candidates: ["你是谁？", "CPU 是什么意思？", "法国的首都是哪里？", "什么是 API？请简单解释。", "水的沸点是多少？", "一年有几个月？", "太阳系最大的行星是什么？"]
  aggregation_method: "max"
}

SIGNAL embedding creative_tasks {
  threshold: 0.74
  candidates: ["Brainstorm a launch campaign for a new tea brand.", "Write a short poem about late-night coding.", "Rewrite this paragraph to sound more cinematic.", "Create three slogan options for a climate startup.", "帮我想一个更有画面感的品牌故事。"]
  aggregation_method: "max"
}

SIGNAL embedding business_analysis {
  threshold: 0.75
  candidates: ["Compare two pricing strategies for a B2B SaaS product.", "Analyze CAC, LTV, and retention trade-offs for a subscription business.", "Draft a market-entry strategy for a new AI tooling startup.", "Evaluate org design options for a fast-growing product team.", "Compare enterprise SaaS churn benchmarks and explain the trade-offs."]
  aggregation_method: "max"
}

SIGNAL embedding history_explainer {
  threshold: 0.75
  candidates: ["Explain why the Roman Republic collapsed.", "Explain why the Ming dynasty fell.", "Compare the causes of the Roman Empire's decline with later empires.", "Explain how industrialization changed political power in Europe.", "Analyze the historical consequences of the Treaty of Versailles.", "总结明治维新对日本国家能力的长期影响。"]
  aggregation_method: "max"
}

SIGNAL embedding psychology_support {
  threshold: 0.75
  candidates: ["Explain why people procrastinate and what interventions usually help.", "Explain confirmation bias and what strategies help reduce it.", "Explain cognitive biases that affect negotiation outcomes.", "Compare attachment styles and how they affect adult relationships.", "Analyze burnout patterns in high-pressure knowledge work.", "帮我解释拖延背后的常见心理机制。"]
  aggregation_method: "max"
}

SIGNAL embedding health_guidance {
  threshold: 0.77
  candidates: ["Explain common causes of chest pain and when someone should seek urgent care.", "Compare evidence-backed treatment options for type 2 diabetes management.", "Summarize safe, clinically grounded steps for lowering high blood pressure.", "解释常见呼吸道感染症状的区别，以及何时应尽快就医。"]
  aggregation_method: "max"
}

SIGNAL embedding code_general {
  threshold: 0.75
  candidates: ["Debug this Python stack trace and explain the likely bug.", "Refactor this TypeScript function to reduce duplication.", "Write a SQL query to aggregate weekly active users.", "Explain the difference between sync and async execution."]
  aggregation_method: "max"
}

SIGNAL embedding complex_stem {
  threshold: 0.77
  candidates: ["Compare modeling approaches for turbulent fluid simulation.", "Explain the trade-offs between battery chemistries for grid storage.", "Analyze a protein-folding pipeline and its computational bottlenecks.", "Design an anomaly-detection method for sensor networks."]
  aggregation_method: "max"
}

SIGNAL embedding architecture_design {
  threshold: 0.78
  candidates: ["Design a distributed rate limiter and explain failure modes.", "Design a multi-region feature-flag service with storage boundaries, cache strategy, and consistency trade-offs.", "Plan a migration from a monolith to event-driven microservices.", "Propose a consistency strategy for a global payments platform.", "Design a multi-tenant observability architecture with low latency."]
  aggregation_method: "max"
}

SIGNAL embedding agentic_workflows {
  threshold: 0.78
  candidates: ["Create a migration plan with rollback, validation, and checkpoints.", "Plan a zero-downtime migration with checkpoints, owners, rollback steps, and validation after each phase.", "Troubleshoot the production issue and iterate until it is fixed.", "Break this system redesign into phases, owners, and verification steps.", "Design an execution workflow with concrete milestones and guardrails.", "给我一个分阶段实施方案，并包含校验与回滚步骤。"]
  aggregation_method: "max"
}

SIGNAL embedding research_synthesis {
  threshold: 0.77
  candidates: ["Compare several studies, cite the evidence, and explain the trade-offs.", "Write a source-backed memo that synthesizes multiple references.", "Survey the literature and recommend a position with supporting evidence.", "Compare conflicting historical or policy interpretations with references.", "请综合多份资料并给出带依据的分析结论。"]
  aggregation_method: "max"
}

SIGNAL embedding general_chat_fallback {
  threshold: 0.72
  candidates: ["Explain this in plain language for a general audience.", "Give me a practical answer without assuming special expertise.", "Help me understand the basics before we go deeper.", "用通俗的话解释清楚这个问题。", "先给我一个适合普通用户的直接回答。"]
  aggregation_method: "max"
}

SIGNAL embedding reasoning_general_en {
  threshold: 0.78
  candidates: ["Compare multiple approaches and recommend the best one with trade-offs.", "Analyze the root cause step by step and justify the conclusion.", "Build a rigorous argument from first principles.", "Synthesize several constraints into a single recommendation."]
  aggregation_method: "max"
}

SIGNAL embedding reasoning_general_zh {
  threshold: 0.78
  candidates: ["请从多个角度严格分析并给出结论。", "请逐步推理，比较几种方案的取舍。", "请从第一性原理出发建立完整论证。", "请综合多个约束给出系统性建议。"]
  aggregation_method: "max"
}

SIGNAL embedding premium_legal_analysis {
  threshold: 0.8
  candidates: ["Analyze indemnity, limitation of liability, and governing-law risks in this contract.", "Assess the legal risk in this agreement by analyzing indemnification, limitation of liability, and compliance duties.", "Draft a legal-risk memo for a cross-border data transfer policy.", "Compare regulatory exposure under two compliance strategies.", "评估该合同条款中的责任分配、赔偿和合规风险。"]
  aggregation_method: "max"
}

SIGNAL fact_check needs_fact_check {
  description: "Narrow factual-verification route for requests that explicitly ask for evidence or source checking."
}

SIGNAL user_feedback wrong_answer {
  description: "Explicit correction or dissatisfaction with the previous answer."
}

SIGNAL user_feedback need_clarification {
  description: "Explicit request to restate the answer more clearly."
}

SIGNAL reask likely_dissatisfied {
  description: "Current user turn closely repeats the immediately previous user turn."
  threshold: 0.8
  lookback_turns: 1
}

SIGNAL language en {
  description: "English language queries"
}

SIGNAL language zh {
  description: "Chinese language queries"
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
  description: "Prompts with ordered workflow markers that imply phased execution."
  feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["先", "再"], ["首先", "然后"]], type: "sequence" }, type: "sequence" }
}

SIGNAL structure numbered_steps {
  description: "Prompts that contain numbered list items such as \"1. ...\""
  feature: { source: { pattern: "(?m)^\\s*\\d+\\.\\s+", type: "regex" }, type: "exists" }
}

SIGNAL structure first_then_flow {
  description: "Prompts that express an ordered workflow."
  feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["首先", "然后"], ["先", "再"]], type: "sequence" }, type: "sequence" }
}

SIGNAL structure constraint_dense {
  description: "Constraint language is dense relative to multilingual text units."
  feature: { source: { keywords: ["under", "at most", "at least", "within", "no more than", "不超过", "至少", "最多"], type: "keyword_set" }, type: "density" }
  predicate: { gt: 0.08 }
}

SIGNAL structure format_directive_dense {
  description: "Output-format directives are dense relative to multilingual text units."
  feature: { source: { keywords: ["table", "bullet", "json", "markdown", "表格", "列表", "JSON"], type: "keyword_set" }, type: "density" }
  predicate: { gt: 0.08 }
}

SIGNAL structure low_question_density {
  description: "Prompts with very low question density relative to multilingual text units."
  feature: { source: { pattern: "[?？]", type: "regex" }, type: "density" }
  predicate: { lt: 0.05 }
}

SIGNAL structure exclamation_emphasis {
  description: "Repeated exclamation marks that usually indicate elevated urgency."
  feature: { source: { pattern: "[!！]", type: "regex" }, type: "count" }
  predicate: { gte: 2 }
}

SIGNAL complexity general_reasoning {
  threshold: 0.14
  description: "General difficulty boundary for simple answers versus synthesis-heavy reasoning."
  hard: { candidates: ["compare several approaches and justify the trade-offs", "build a rigorous step-by-step argument", "synthesize constraints into a plan", "root-cause analysis for a complex failure", "derive the answer from first principles", "严格论证并比较取舍"] }
  easy: { candidates: ["brief definition", "quick summary", "simple explanation", "short direct answer", "rewrite this sentence", "简单解释一下"] }
}

SIGNAL complexity code_task {
  threshold: 0.12
  description: "Coding and software-engineering difficulty for cheap code help versus hard systems work."
  composer: { operator: "AND", conditions: [{ type: "domain", name: "computer science" }] }
  hard: { candidates: ["design a distributed system with failure handling", "debug a race condition in production", "optimize a database query plan at scale", "refactor a large service boundary safely", "migrate a monolith to microservices"] }
  easy: { candidates: ["explain what this function does", "fix a small bug in this code snippet", "write a helper function", "convert this loop to a list comprehension", "explain the stack trace briefly"] }
}

SIGNAL complexity math_task {
  threshold: 0.12
  description: "Math difficulty for simple calculations versus formal proofs and derivations."
  composer: { operator: "AND", conditions: [{ type: "domain", name: "math" }] }
  hard: { candidates: ["prove the theorem rigorously", "derive the equation step by step", "analyze asymptotic behavior formally", "solve a differential equation with proof", "prove this by contradiction"] }
  easy: { candidates: ["what is 2 plus 2", "solve this simple linear equation", "calculate a percentage", "basic geometry area question", "simple arithmetic word problem"] }
}

SIGNAL complexity legal_risk {
  threshold: 0.12
  description: "Legal risk boundary for informational law questions versus premium-risk analysis."
  composer: { operator: "AND", conditions: [{ type: "domain", name: "law" }] }
  hard: { candidates: ["analyze indemnity, liability, and jurisdiction risk", "compare two compliance strategies and regulatory exposure", "draft a legal-risk memo for cross-border operations", "interpret contract clauses with risk trade-offs"] }
  easy: { candidates: ["what does NDA mean", "define arbitration clause", "explain what compliance means", "brief overview of a privacy policy"] }
}

SIGNAL complexity agentic_delivery {
  threshold: 0.12
  description: "Workflow and execution difficulty boundary for agentic requests."
  hard: { candidates: ["create a migration plan with checkpoints, rollback, and validation", "troubleshoot this production issue until the root cause is fixed", "design an execution workflow with milestones, owners, and guardrails", "automate the process and include verification after each phase", "break this project into tasks, dependencies, and acceptance checks"] }
  easy: { candidates: ["give me a short checklist", "provide a simple one-step setup guide", "write a small implementation plan", "suggest the next step only", "give me a brief task list"] }
}

SIGNAL complexity evidence_synthesis {
  threshold: 0.12
  description: "Evidence-heavy research boundary for source-backed synthesis versus lighter overview tasks."
  composer: { operator: "OR", conditions: [{ type: "domain", name: "business" }, { type: "domain", name: "economics" }, { type: "domain", name: "health" }, { type: "domain", name: "history" }, { type: "domain", name: "law" }, { type: "domain", name: "philosophy" }, { type: "domain", name: "psychology" }] }
  hard: { candidates: ["compare several sources and recommend the strongest evidence-backed position", "write a source-backed memo that synthesizes multiple references", "survey the literature and explain conflicting evidence with citations", "compare policy or historical interpretations and justify the conclusion", "synthesize several studies into one decision recommendation"] }
  easy: { candidates: ["give a quick overview of the topic", "summarize one article briefly", "explain the term in simple language", "list the main idea without detailed sourcing", "provide a short summary only"] }
}

PROJECTION partition balance_domain_partition {
  semantics: "softmax_exclusive"
  temperature: 0.1
  members: ["biology", "business", "chemistry", "computer science", "economics", "engineering", "health", "history", "law", "math", "other", "philosophy", "physics", "psychology"]
  default: "other"
}

PROJECTION partition balance_intent_partition {
  semantics: "softmax_exclusive"
  temperature: 0.18
  members: ["agentic_workflows", "architecture_design", "business_analysis", "code_general", "complex_stem", "creative_tasks", "fast_qa_en", "fast_qa_zh", "general_chat_fallback", "health_guidance", "history_explainer", "premium_legal_analysis", "psychology_support", "reasoning_general_en", "reasoning_general_zh", "research_synthesis"]
  default: "general_chat_fallback"
}

PROJECTION score difficulty_score {
  method: "weighted_sum"
  inputs: [{ type: "keyword", weight: -0.26, name: "simple_request_markers" }, { type: "embedding", weight: -0.18, name: "fast_qa_en", value_source: "confidence" }, { type: "embedding", weight: -0.18, name: "fast_qa_zh", value_source: "confidence" }, { type: "context", weight: -0.1, name: "short_context" }, { type: "context", weight: 0.03, name: "medium_context" }, { type: "context", weight: 0.18, name: "long_context" }, { type: "structure", weight: 0.12, name: "ordered_workflow" }, { type: "structure", weight: 0.08, name: "numbered_steps" }, { type: "structure", weight: 0.1, name: "first_then_flow" }, { type: "structure", weight: 0.06, name: "constraint_dense" }, { type: "structure", weight: 0.04, name: "format_directive_dense" }, { type: "structure", weight: -0.05, name: "low_question_density" }, { type: "keyword", weight: 0.2, name: "reasoning_request_markers", value_source: "confidence" }, { type: "keyword", weight: 0.14, name: "multi_step_markers", value_source: "confidence" }, { type: "keyword", weight: 0.12, name: "code_request_markers", value_source: "confidence" }, { type: "keyword", weight: 0.12, name: "architecture_markers", value_source: "confidence" }, { type: "keyword", weight: 0.11, name: "research_request_markers", value_source: "confidence" }, { type: "keyword", weight: 0.16, name: "agentic_request_markers", value_source: "confidence" }, { type: "keyword", weight: 0.08, name: "implementation_markers", value_source: "confidence" }, { type: "embedding", weight: 0.18, name: "reasoning_general_en", value_source: "confidence" }, { type: "embedding", weight: 0.18, name: "reasoning_general_zh", value_source: "confidence" }, { type: "embedding", weight: 0.2, name: "agentic_workflows", value_source: "confidence" }, { type: "embedding", weight: 0.16, name: "architecture_design", value_source: "confidence" }, { type: "embedding", weight: 0.14, name: "complex_stem", value_source: "confidence" }, { type: "embedding", weight: 0.14, name: "research_synthesis", value_source: "confidence" }, { type: "embedding", weight: 0.16, name: "premium_legal_analysis", value_source: "confidence" }, { type: "embedding", weight: 0.08, name: "business_analysis", value_source: "confidence" }, { type: "embedding", weight: 0.05, name: "history_explainer", value_source: "confidence" }, { type: "embedding", weight: 0.05, name: "psychology_support", value_source: "confidence" }, { type: "complexity", weight: 0.08, name: "general_reasoning:medium" }, { type: "complexity", weight: 0.2, name: "general_reasoning:hard" }, { type: "complexity", weight: 0.08, name: "code_task:medium" }, { type: "complexity", weight: 0.18, name: "code_task:hard" }, { type: "complexity", weight: 0.1, name: "math_task:medium" }, { type: "complexity", weight: 0.22, name: "math_task:hard" }, { type: "complexity", weight: 0.18, name: "legal_risk:hard" }, { type: "complexity", weight: 0.1, name: "agentic_delivery:medium" }, { type: "complexity", weight: 0.22, name: "agentic_delivery:hard" }, { type: "complexity", weight: 0.08, name: "evidence_synthesis:medium" }, { type: "complexity", weight: 0.18, name: "evidence_synthesis:hard" }]
}

PROJECTION score verification_pressure {
  method: "weighted_sum"
  inputs: [{ type: "fact_check", weight: 0.28, name: "needs_fact_check" }, { type: "keyword", weight: 0.22, name: "verification_markers", value_source: "confidence" }, { type: "keyword", weight: 0.18, name: "reference_heavy_markers", value_source: "confidence" }, { type: "keyword", weight: 0.1, name: "research_request_markers", value_source: "confidence" }, { type: "keyword", weight: 0.08, name: "legal_risk_markers", value_source: "confidence" }, { type: "domain", weight: 0.12, name: "health" }, { type: "domain", weight: 0.14, name: "law" }, { type: "domain", weight: 0.05, name: "business" }, { type: "domain", weight: 0.05, name: "history" }, { type: "user_feedback", weight: 0.1, name: "wrong_answer" }, { type: "keyword", weight: 0.06, name: "correction_feedback_markers", value_source: "confidence" }, { type: "context", weight: 0.04, name: "long_context" }]
}

PROJECTION score feedback_correction_pressure {
  method: "weighted_sum"
  inputs: [{ type: "user_feedback", weight: 0.14, name: "wrong_answer" }, { type: "keyword", weight: 0.34, name: "correction_feedback_markers", value_source: "confidence" }, { type: "reask", weight: 0.1, name: "likely_dissatisfied", value_source: "confidence" }, { type: "fact_check", weight: 0.12, name: "needs_fact_check" }, { type: "keyword", weight: 0.18, name: "verification_markers", value_source: "confidence" }, { type: "keyword", weight: 0.16, name: "reference_heavy_markers", value_source: "confidence" }, { type: "complexity", weight: 0.08, name: "evidence_synthesis:medium" }, { type: "complexity", weight: 0.16, name: "evidence_synthesis:hard" }, { type: "context", weight: 0.04, name: "short_context" }, { type: "context", weight: 0.02, name: "medium_context" }, { type: "keyword", weight: -0.22, name: "code_request_markers", value_source: "confidence" }, { type: "keyword", weight: -0.16, name: "implementation_markers", value_source: "confidence" }]
}

PROJECTION score feedback_clarification_pressure {
  method: "weighted_sum"
  inputs: [{ type: "user_feedback", weight: 0.14, name: "need_clarification" }, { type: "keyword", weight: 0.34, name: "clarification_feedback_markers", value_source: "confidence" }, { type: "reask", weight: 0.24, name: "likely_dissatisfied", value_source: "confidence" }, { type: "context", weight: 0.08, name: "short_context" }, { type: "context", weight: 0.04, name: "medium_context" }, { type: "user_feedback", weight: -0.14, name: "wrong_answer" }, { type: "keyword", weight: -0.18, name: "correction_feedback_markers", value_source: "confidence" }, { type: "keyword", weight: -0.16, name: "verification_markers", value_source: "confidence" }, { type: "keyword", weight: -0.14, name: "reference_heavy_markers", value_source: "confidence" }, { type: "fact_check", weight: -0.14, name: "needs_fact_check" }, { type: "keyword", weight: -0.2, name: "code_request_markers", value_source: "confidence" }, { type: "keyword", weight: -0.14, name: "implementation_markers", value_source: "confidence" }, { type: "keyword", weight: -0.16, name: "simple_request_markers", value_source: "confidence" }, { type: "embedding", weight: -0.18, name: "fast_qa_en", value_source: "confidence" }, { type: "embedding", weight: -0.18, name: "fast_qa_zh", value_source: "confidence" }]
}

PROJECTION score urgency_pressure {
  method: "weighted_sum"
  inputs: [{ type: "keyword", weight: 0.24, name: "urgency_markers", value_source: "confidence" }, { type: "structure", weight: 0.16, name: "exclamation_emphasis", value_source: "confidence" }]
}

PROJECTION mapping difficulty_band {
  source: "difficulty_score"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 10 }
  outputs: [{ name: "balance_simple", lt: 0.18 }, { name: "balance_medium", gte: 0.18, lt: 0.48 }, { name: "balance_complex", gte: 0.48, lt: 0.82 }, { name: "balance_reasoning", gte: 0.82 }]
}

PROJECTION mapping verification_band {
  source: "verification_pressure"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 12 }
  outputs: [{ name: "verification_standard", lt: 0.35 }, { name: "verification_required", gte: 0.35 }]
}

PROJECTION mapping feedback_correction_band {
  source: "feedback_correction_pressure"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 12 }
  outputs: [{ name: "feedback_correction_standard", lt: 0.34 }, { name: "feedback_correction_verified", gte: 0.34 }]
}

PROJECTION mapping feedback_clarification_band {
  source: "feedback_clarification_pressure"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 12 }
  outputs: [{ name: "feedback_clarification_standard", lt: 0.26 }, { name: "feedback_clarification_overlay", gte: 0.26 }]
}

PROJECTION mapping urgency_band {
  source: "urgency_pressure"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 12 }
  outputs: [{ name: "urgency_standard", lt: 0.24 }, { name: "urgency_elevated", gte: 0.24 }]
}

# =============================================================================
# MODELS
# =============================================================================

MODEL anthropic/claude-opus-4.6 {
  context_window_size: 262144
  description: "PREMIUM tier alias reserved for legal and high-risk analysis."
  capabilities: ["legal_analysis", "policy_review", "high_risk_review"]
  tags: ["tier:premium", "cost:highest", "specialty:legal"]
  quality_score: 0.94
  modality: "text"
}

MODEL google/gemini-2.5-flash-lite {
  context_window_size: 262144
  description: "MEDIUM tier alias for low-cost verified explanation and correction tasks."
  capabilities: ["verified_explanation", "source_backed_correction", "nuanced_explanation"]
  tags: ["tier:medium", "cost:low", "specialty:verified"]
  quality_score: 0.68
  modality: "text"
}

MODEL google/gemini-3.1-pro {
  context_window_size: 262144
  description: "COMPLEX tier alias for systems design, hard STEM, health guidance, and deep general reasoning."
  capabilities: ["architecture", "stem_analysis", "long_context", "general_reasoning"]
  tags: ["tier:complex", "cost:upper_mid", "specialty:complex_generalist"]
  quality_score: 0.82
  modality: "text"
}

MODEL openai/gpt5.4 {
  context_window_size: 262144
  description: "REASONING tier alias for narrow formal math proofs and derivations."
  capabilities: ["reasoning", "proofs", "formal_derivation"]
  tags: ["tier:reasoning", "cost:high", "specialty:formal_proof"]
  quality_score: 0.9
  modality: "text"
}

MODEL qwen/qwen3.5-rocm {
  context_window_size: 262144
  description: "SIMPLE tier alias and free self-hosted default for fast QA, broad fallback, creative drafting, and most low-cost traffic."
  capabilities: ["fast_qa", "self_hosted", "concise_answers", "general_chat", "creative_drafting"]
  tags: ["tier:simple", "cost:free", "deployment:self_hosted", "traffic:default"]
  quality_score: 0.58
  modality: "text"
}

# =============================================================================
# PLUGINS
# =============================================================================

PLUGIN router_replay router_replay {}

# =============================================================================
# ROUTES
# =============================================================================

ROUTE premium_legal (description = "Premium-only route for high-value legal and compliance analysis.") {
  PRIORITY 260
  TIER 1
  WHEN (domain("law") OR keyword("legal_risk_markers") OR embedding("premium_legal_analysis")) AND (embedding("premium_legal_analysis") OR projection("verification_required") OR complexity("legal_risk:medium") OR complexity("legal_risk:hard"))
  MODEL "anthropic/claude-opus-4.6" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE formal_math_proof (description = "Narrow premium reasoning lane for formal math proofs and derivations.") {
  PRIORITY 252
  TIER 2
  WHEN domain("math") AND keyword("reasoning_request_markers") AND NOT (projection("verification_required") OR user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR fact_check("needs_fact_check") OR keyword("architecture_markers") OR keyword("agentic_request_markers") OR keyword("code_request_markers") OR keyword("implementation_markers") OR embedding("architecture_design") OR embedding("complex_stem") OR embedding("agentic_workflows") OR embedding("premium_legal_analysis"))
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE reasoning_deep (description = "Deep philosophy and first-principles reasoning outside the narrow formal-math overlay.") {
  PRIORITY 250
  TIER 3
  WHEN (domain("math") AND NOT keyword("reasoning_request_markers") AND (projection("balance_reasoning") OR complexity("math_task:medium")) OR (domain("philosophy") AND (embedding("reasoning_general_en") OR embedding("reasoning_general_zh") OR embedding("research_synthesis"))) OR (embedding("reasoning_general_en") OR embedding("reasoning_general_zh") OR embedding("research_synthesis") OR keyword("research_request_markers")) AND (projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning"))) AND NOT (domain("law") OR domain("health") OR projection("verification_required") OR user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR fact_check("needs_fact_check") OR keyword("architecture_markers") OR keyword("agentic_request_markers") OR keyword("code_request_markers") OR keyword("implementation_markers") OR embedding("architecture_design") OR embedding("complex_stem") OR embedding("agentic_workflows") OR embedding("premium_legal_analysis"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE complex_specialist (description = "High-structure execution plans, systems design, and specialist STEM synthesis.") {
  PRIORITY 242
  TIER 4
  WHEN ((embedding("agentic_workflows") OR keyword("agentic_request_markers")) AND (keyword("multi_step_markers") OR structure("ordered_workflow") OR structure("numbered_steps") OR structure("first_then_flow") OR structure("constraint_dense") OR structure("format_directive_dense")) OR domain("computer science") AND (embedding("architecture_design") OR keyword("architecture_markers")) OR (domain("physics") OR domain("chemistry") OR domain("biology") OR domain("engineering") OR domain("computer science")) AND embedding("complex_stem")) AND (projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning")) AND NOT (embedding("fast_qa_en") OR embedding("fast_qa_zh") OR keyword("simple_request_markers") OR keyword("creative_request_markers") OR embedding("creative_tasks"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE feedback_wrong_answer_verified (description = "Explicit correction requests on evidence-sensitive follow-ups.") {
  PRIORITY 232
  TIER 5
  WHEN projection("feedback_correction_verified") AND (user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied")) AND NOT keyword("code_request_markers")
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_code_general (description = "Low-medium cost coding, debugging, refactoring, and technical Q&A.") {
  PRIORITY 220
  TIER 6
  WHEN (keyword("code_request_markers") OR keyword("implementation_markers") OR embedding("code_general")) AND (projection("balance_medium") OR projection("balance_complex") OR projection("balance_simple") AND (projection("urgency_elevated") OR structure("exclamation_emphasis"))) AND NOT (keyword("agentic_request_markers") OR keyword("architecture_markers") OR keyword("creative_request_markers") OR embedding("creative_tasks") OR embedding("architecture_design") OR embedding("agentic_workflows"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_health (description = "Conservative route for evidence-sensitive health and medical guidance.") {
  PRIORITY 218
  TIER 7
  WHEN domain("health") AND (projection("verification_required") OR keyword("verification_markers") OR fact_check("needs_fact_check") OR keyword("reference_heavy_markers") OR complexity("evidence_synthesis:hard")) AND (embedding("health_guidance") OR projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning")) AND NOT (user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_explainer (description = "Evidence-sensitive business, economics, history, and psychology explanation.") {
  PRIORITY 214
  TIER 8
  WHEN (domain("business") OR domain("economics") OR domain("history") OR domain("psychology") OR embedding("business_analysis") OR embedding("history_explainer") OR embedding("psychology_support") OR keyword("history_topic_markers")) AND (projection("verification_required") OR complexity("evidence_synthesis:hard") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR fact_check("needs_fact_check")) AND NOT (embedding("fast_qa_en") OR embedding("fast_qa_zh") OR keyword("simple_request_markers") OR domain("health") OR domain("law") OR user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied"))
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE feedback_need_clarification (description = "Cheap clarification lane for explicit restatements and single-turn re-asks.") {
  PRIORITY 212
  TIER 9
  WHEN projection("feedback_clarification_overlay") AND NOT (projection("feedback_correction_verified") OR projection("verification_required") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR fact_check("needs_fact_check") OR keyword("code_request_markers"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_explainer (description = "Low-cost business, history, and psychology explanation when verification pressure is absent.") {
  PRIORITY 208
  TIER 10
  WHEN (domain("business") OR domain("economics") OR domain("history") OR domain("psychology") OR embedding("business_analysis") OR embedding("history_explainer") OR embedding("psychology_support") OR keyword("history_topic_markers")) AND (projection("balance_medium") OR projection("balance_complex") OR projection("balance_simple") AND (context("medium_context") OR keyword("history_topic_markers") OR complexity("evidence_synthesis:medium"))) AND NOT (projection("verification_required") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR complexity("evidence_synthesis:hard") OR domain("health") OR domain("law") OR embedding("fast_qa_en") OR embedding("fast_qa_zh") OR keyword("simple_request_markers") OR keyword("reasoning_request_markers") OR keyword("research_request_markers") OR keyword("creative_request_markers") OR embedding("reasoning_general_en") OR embedding("reasoning_general_zh") OR embedding("research_synthesis") OR embedding("creative_tasks"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_creative (description = "Low-cost creative writing, copywriting, and interpersonal drafting.") {
  PRIORITY 200
  TIER 11
  WHEN (keyword("creative_request_markers") OR embedding("creative_tasks")) AND (projection("balance_simple") OR projection("balance_medium")) AND NOT (embedding("fast_qa_en") OR embedding("fast_qa_zh") OR embedding("health_guidance") OR embedding("code_general") OR embedding("architecture_design") OR embedding("agentic_workflows") OR embedding("premium_legal_analysis") OR projection("verification_required") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR keyword("reference_heavy_markers"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE fast_qa (description = "Short English or Chinese factual questions, including explicit verification asks, that should stay on the cheap lane.") {
  PRIORITY 184
  TIER 12
  WHEN (embedding("fast_qa_en") AND language("en") OR embedding("fast_qa_zh") AND language("zh") OR keyword("simple_request_markers")) AND context("short_context") AND ((projection("balance_simple") OR projection("balance_medium")) AND (projection("verification_required") OR keyword("verification_markers") OR keyword("reference_heavy_markers")) AND NOT (domain("health") OR domain("law") OR keyword("code_request_markers") OR keyword("implementation_markers") OR projection("urgency_elevated") OR user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied")) OR projection("balance_simple") AND NOT (projection("verification_required") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR keyword("code_request_markers") OR keyword("implementation_markers") OR projection("urgency_elevated") OR user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied") OR projection("feedback_clarification_overlay")))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE simple_general (description = "Lowest-cost fallback for everyday traffic and non-specialized requests.") {
  PRIORITY 170
  TIER 13
  WHEN (context("short_context") AND (projection("balance_simple") OR projection("balance_medium")) AND (embedding("general_chat_fallback") OR structure("low_question_density")) AND NOT (keyword("simple_request_markers") OR projection("verification_required") OR user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied") OR projection("feedback_clarification_overlay") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR keyword("code_request_markers") OR keyword("architecture_markers") OR keyword("agentic_request_markers") OR keyword("creative_request_markers")) OR context("medium_context") AND domain("other") AND (projection("balance_simple") OR projection("balance_medium")) AND NOT (projection("verification_required") OR user_feedback("wrong_answer") OR keyword("correction_feedback_markers") OR reask("likely_dissatisfied") OR projection("feedback_clarification_overlay") OR keyword("verification_markers") OR keyword("reference_heavy_markers")))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE casual_chat (description = "Absolute final fallback that guarantees a routing decision when no earlier balance lane matches.") {
  PRIORITY 10
  TIER 14
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}
