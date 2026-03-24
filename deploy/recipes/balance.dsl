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

SIGNAL keyword jailbreak_attempt {
  operator: "OR"
  keywords: ["ignore previous instructions", "disregard all rules", "bypass safety", "jailbreak", "pretend you are", "act as if", "forget your guidelines"]
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

SIGNAL keyword output_format_markers {
  operator: "OR"
  keywords: ["json", "yaml", "csv", "markdown table", "bullet points", "format as", "return a schema", "用表格", "用 json", "按列表输出"]
}

SIGNAL keyword constraint_markers {
  operator: "OR"
  keywords: ["must", "only", "exactly", "at least", "no more than", "cannot", "avoid", "do not", "必须", "只能", "不超过", "不要"]
}

SIGNAL keyword negation_markers {
  operator: "OR"
  keywords: ["except", "unless", "without", "instead of", "do not include", "not this but that", "除非", "不包含", "不要用", "而不是"]
}

SIGNAL keyword emotion_positive_markers {
  operator: "OR"
  keywords: ["excited", "thrilled", "happy", "great news", "amazing news", "so glad", "太好了", "太棒了", "开心", "高兴", "兴奋", "激动"]
}

SIGNAL keyword emotion_negative_markers {
  operator: "OR"
  keywords: ["frustrated", "upset", "angry", "anxious", "worried", "stressed", "overwhelmed", "panicking", "this is ridiculous", "焦虑", "着急", "崩溃", "烦死了", "生气", "太离谱了"]
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
  description: "Conservative follow-up signal for explicit correction or dissatisfaction."
}

SIGNAL user_feedback need_clarification {
  description: "Conservative follow-up signal for explicit requests to restate more clearly."
}

SIGNAL preference concise_answers {
  description: "Users who prefer short and direct replies."
  examples: ["keep it concise", "answer briefly", "one paragraph only", "bullet points only", "简洁一点", "简短回答"]
  threshold: 0.7
}

SIGNAL preference creative_collaboration {
  description: "Users who want exploratory or imaginative help."
  examples: ["brainstorm with me", "make it more imaginative", "give me multiple creative options", "make the writing more vivid", "帮我头脑风暴", "多给几个创意方向"]
  threshold: 0.7
}

SIGNAL preference coding_partner {
  description: "Users who want implementation, debugging, or refactoring help."
  examples: ["write the code", "debug this issue", "refactor this module", "explain the stack trace", "帮我写代码", "帮我排查 bug"]
  threshold: 0.7
}

SIGNAL preference structured_delivery {
  description: "Users who want output in a constrained structure."
  examples: ["return json", "use a markdown table", "list the steps clearly", "format the answer as yaml", "用表格输出", "按步骤列出"]
  threshold: 0.7
}

SIGNAL preference agentic_execution {
  description: "Users who want a plan, workflow, or execution-oriented answer."
  examples: ["propose an execution plan", "create a migration checklist", "give me a troubleshooting workflow", "break this into tasks", "给我实施计划", "给我排查流程"]
  threshold: 0.7
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

SIGNAL structure many_questions {
  description: "Prompts with several explicit questions, usually indicating multi-part answers."
  feature: { source: { pattern: "[?？]", type: "regex" }, type: "count" }
  predicate: { gte: 4 }
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
  description: "Repeated exclamation marks that usually indicate elevated emotional intensity or urgency."
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

SIGNAL jailbreak jailbreak_standard {
  method: "classifier"
  threshold: 0.65
  description: "BERT classifier - standard sensitivity, single-turn"
}

SIGNAL jailbreak jailbreak_strict {
  method: "classifier"
  threshold: 0.4
  description: "BERT classifier - strict sensitivity, full history"
}

SIGNAL pii pii_deny_all {
  threshold: 0.9
  description: "Block all detected PII types."
}

SIGNAL pii pii_relaxed {
  threshold: 0.75
  pii_types_allowed: ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION", "DATE_TIME", "URL", "ORG"]
  description: "Block high-sensitivity PII while allowing common contact entities."
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
  inputs: [{ type: "keyword", name: "simple_request_markers", weight: -0.28 }, { type: "embedding", name: "fast_qa_en", weight: -0.16, value_source: "confidence" }, { type: "embedding", name: "fast_qa_zh", weight: -0.16, value_source: "confidence" }, { type: "context", name: "short_context", weight: -0.1 }, { type: "context", name: "medium_context", weight: 0.03 }, { type: "context", name: "long_context", weight: 0.18 }, { type: "structure", name: "many_questions", weight: 0.06 }, { type: "structure", name: "ordered_workflow", weight: 0.11 }, { type: "structure", name: "numbered_steps", weight: 0.08 }, { type: "structure", name: "first_then_flow", weight: 0.1 }, { type: "structure", name: "constraint_dense", weight: 0.06 }, { type: "structure", name: "format_directive_dense", weight: 0.04 }, { type: "structure", name: "low_question_density", weight: -0.05 }, { type: "keyword", name: "reasoning_request_markers", weight: 0.22, value_source: "confidence" }, { type: "keyword", name: "multi_step_markers", weight: 0.14, value_source: "confidence" }, { type: "keyword", name: "code_request_markers", weight: 0.12, value_source: "confidence" }, { type: "keyword", name: "architecture_markers", weight: 0.1, value_source: "confidence" }, { type: "keyword", name: "research_request_markers", weight: 0.1, value_source: "confidence" }, { type: "keyword", name: "constraint_markers", weight: 0.08, value_source: "confidence" }, { type: "keyword", name: "output_format_markers", weight: 0.05, value_source: "confidence" }, { type: "keyword", name: "negation_markers", weight: 0.03, value_source: "confidence" }, { type: "keyword", name: "agentic_request_markers", weight: 0.14, value_source: "confidence" }, { type: "keyword", name: "implementation_markers", weight: 0.1, value_source: "confidence" }, { type: "embedding", name: "reasoning_general_en", weight: 0.18, value_source: "confidence" }, { type: "embedding", name: "reasoning_general_zh", weight: 0.18, value_source: "confidence" }, { type: "embedding", name: "agentic_workflows", weight: 0.18, value_source: "confidence" }, { type: "embedding", name: "architecture_design", weight: 0.15, value_source: "confidence" }, { type: "embedding", name: "complex_stem", weight: 0.14, value_source: "confidence" }, { type: "embedding", name: "research_synthesis", weight: 0.12, value_source: "confidence" }, { type: "embedding", name: "premium_legal_analysis", weight: 0.16, value_source: "confidence" }, { type: "complexity", name: "general_reasoning:medium", weight: 0.1 }, { type: "complexity", name: "general_reasoning:hard", weight: 0.22 }, { type: "complexity", name: "code_task:medium", weight: 0.08 }, { type: "complexity", name: "code_task:hard", weight: 0.18 }, { type: "complexity", name: "math_task:hard", weight: 0.2 }, { type: "complexity", name: "legal_risk:hard", weight: 0.16 }, { type: "complexity", name: "agentic_delivery:medium", weight: 0.1 }, { type: "complexity", name: "agentic_delivery:hard", weight: 0.22 }, { type: "complexity", name: "evidence_synthesis:medium", weight: 0.1 }, { type: "complexity", name: "evidence_synthesis:hard", weight: 0.18 }]
}

PROJECTION score verification_pressure {
  method: "weighted_sum"
  inputs: [{ type: "fact_check", name: "needs_fact_check", weight: 0.28 }, { type: "keyword", name: "verification_markers", weight: 0.2, value_source: "confidence" }, { type: "keyword", name: "reference_heavy_markers", weight: 0.16, value_source: "confidence" }, { type: "keyword", name: "research_request_markers", weight: 0.1, value_source: "confidence" }, { type: "keyword", name: "legal_risk_markers", weight: 0.1, value_source: "confidence" }, { type: "domain", name: "health", weight: 0.1 }, { type: "domain", name: "law", weight: 0.12 }, { type: "domain", name: "business", weight: 0.06 }, { type: "domain", name: "history", weight: 0.06 }, { type: "user_feedback", name: "wrong_answer", weight: 0.12 }, { type: "keyword", name: "correction_feedback_markers", weight: 0.06, value_source: "confidence" }, { type: "context", name: "long_context", weight: 0.04 }]
}

PROJECTION score emotion_valence {
  method: "weighted_sum"
  inputs: [{ type: "keyword", name: "emotion_positive_markers", weight: 0.24, value_source: "confidence" }, { type: "keyword", name: "emotion_negative_markers", weight: -0.26, value_source: "confidence" }]
}

PROJECTION score urgency_pressure {
  method: "weighted_sum"
  inputs: [{ type: "keyword", name: "urgency_markers", weight: 0.22, value_source: "confidence" }, { type: "structure", name: "exclamation_emphasis", weight: 0.16, value_source: "confidence" }, { type: "keyword", name: "emotion_negative_markers", weight: 0.1, value_source: "confidence" }]
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

PROJECTION mapping emotion_band {
  source: "emotion_valence"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 10 }
  outputs: [{ name: "emotion_negative", lte: -0.18 }, { name: "emotion_positive", gte: 0.18 }]
}

PROJECTION mapping urgency_band {
  source: "urgency_pressure"
  method: "threshold_bands"
  calibration: { method: "sigmoid_distance", slope: 12 }
  outputs: [{ name: "urgency_standard", lt: 0.26 }, { name: "urgency_elevated", gte: 0.26 }]
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
  description: "MEDIUM tier alias for low-cost expressive tasks when the free self-hosted route is not the best fit."
  capabilities: ["creative_writing", "copywriting", "nuanced_explanation"]
  tags: ["tier:medium", "cost:low", "specialty:expressive"]
  quality_score: 0.68
  modality: "text"
}

MODEL google/gemini-3.1-pro {
  context_window_size: 262144
  description: "COMPLEX tier alias for systems design, hard STEM, and long-context synthesis."
  capabilities: ["architecture", "stem_analysis", "long_context"]
  tags: ["tier:complex", "cost:upper_mid", "specialty:systems"]
  quality_score: 0.82
  modality: "text"
}

MODEL openai/gpt5.4 {
  context_window_size: 262144
  description: "REASONING tier alias for proofs, philosophy, and deep multi-step reasoning."
  capabilities: ["reasoning", "proofs", "argumentation"]
  tags: ["tier:reasoning", "cost:high", "specialty:multi_step"]
  quality_score: 0.9
  modality: "text"
}

MODEL qwen/qwen3.5-rocm {
  context_window_size: 262144
  description: "SIMPLE tier alias and free self-hosted default for fast QA, broad fallback, and low-cost medium traffic."
  capabilities: ["fast_qa", "self_hosted", "concise_answers", "general_chat"]
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

ROUTE reasoning_math (description = "Dedicated reasoning tier for proofs, derivations, and hard math.") {
  PRIORITY 250
  TIER 2
  WHEN domain("math") AND (projection("balance_reasoning") OR complexity("math_task:medium") OR complexity("math_task:hard"))
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE reasoning_philosophy (description = "Dedicated reasoning tier for philosophy, ethics, and abstract argumentation.") {
  PRIORITY 245
  TIER 3
  WHEN domain("philosophy") AND (projection("balance_simple") OR projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning")) AND (keyword("reasoning_request_markers") OR embedding("reasoning_general_en") OR embedding("reasoning_general_zh") OR embedding("research_synthesis")) AND NOT (keyword("agentic_request_markers") OR keyword("code_request_markers") OR keyword("implementation_markers"))
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE complex_agentic (description = "High-structure execution plans, migrations, and workflow orchestration with multi-step constraints.") {
  PRIORITY 243
  TIER 4
  WHEN (embedding("agentic_workflows") OR keyword("agentic_request_markers")) AND (keyword("multi_step_markers") OR structure("ordered_workflow") OR structure("numbered_steps") OR structure("first_then_flow") OR structure("constraint_dense") OR structure("format_directive_dense")) AND (projection("balance_simple") OR projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning")) AND NOT keyword("architecture_markers")
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE complex_architecture (description = "Complex systems, architecture, and large-scope technical design.") {
  PRIORITY 240
  TIER 5
  WHEN (domain("computer science") OR domain("engineering")) AND (embedding("architecture_design") OR keyword("architecture_markers")) AND (projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning")) AND NOT (keyword("multi_step_markers") OR structure("ordered_workflow") OR structure("numbered_steps") OR structure("first_then_flow"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE complex_stem (description = "Complex STEM analysis outside the dedicated math reasoning route.") {
  PRIORITY 235
  TIER 6
  WHEN (domain("physics") OR domain("chemistry") OR domain("biology") OR domain("engineering")) AND (embedding("complex_stem") OR embedding("research_synthesis") OR keyword("reasoning_request_markers") OR keyword("research_request_markers") OR projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning")) AND NOT (embedding("fast_qa_en") OR embedding("fast_qa_zh") OR keyword("simple_request_markers"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE feedback_wrong_answer_verified (description = "Narrow recovery path for explicit corrections on evidence-sensitive or high-stakes follow-ups.") {
  PRIORITY 232
  TIER 7
  WHEN user_feedback("wrong_answer") AND keyword("correction_feedback_markers") AND (projection("verification_required") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR complexity("evidence_synthesis:medium") OR complexity("evidence_synthesis:hard")) AND (context("short_context") OR context("medium_context"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
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
  TIER 8
  WHEN (domain("computer science") OR keyword("code_request_markers") OR keyword("implementation_markers") OR embedding("code_general")) AND (projection("balance_medium") OR projection("balance_complex") OR projection("balance_simple") AND projection("urgency_elevated")) AND NOT (keyword("agentic_request_markers") OR keyword("architecture_markers") OR keyword("creative_request_markers") OR preference("agentic_execution") OR preference("creative_collaboration") OR user_feedback("wrong_answer") OR user_feedback("need_clarification"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_business (description = "Mid-tier business and economics analysis without premium escalation.") {
  PRIORITY 215
  TIER 10
  WHEN (domain("business") OR domain("economics")) AND embedding("business_analysis") AND (projection("balance_medium") OR projection("balance_complex")) AND NOT (projection("verification_required") OR complexity("evidence_synthesis:hard") OR keyword("verification_markers") OR keyword("reference_heavy_markers"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_business (description = "Conservative factual overlay for evidence-sensitive business and economics requests.") {
  PRIORITY 216
  TIER 9
  WHEN (domain("business") OR domain("economics")) AND (projection("verification_required") OR complexity("evidence_synthesis:hard") OR keyword("verification_markers") OR keyword("reference_heavy_markers")) AND (projection("balance_medium") OR projection("balance_complex") OR embedding("business_analysis"))
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_health (description = "Conservative route for evidence-sensitive health and medical guidance.") {
  PRIORITY 214
  TIER 11
  WHEN domain("health") AND (embedding("health_guidance") OR projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning"))
  MODEL "google/gemini-3.1-pro" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_history (description = "Mid-tier historical explanation and comparison with better writing quality.") {
  PRIORITY 210
  TIER 13
  WHEN (domain("history") OR embedding("history_explainer") OR keyword("history_topic_markers")) AND (projection("balance_simple") OR projection("balance_medium") OR projection("balance_complex")) AND NOT (projection("verification_required") OR complexity("evidence_synthesis:hard") OR keyword("verification_markers") OR keyword("reference_heavy_markers"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = true, effort = "medium")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_history (description = "Conservative factual overlay for source-sensitive history explanation.") {
  PRIORITY 211
  TIER 12
  WHEN (domain("history") OR embedding("history_explainer") OR keyword("history_topic_markers")) AND (projection("verification_required") OR complexity("evidence_synthesis:hard") OR keyword("verification_markers") OR keyword("reference_heavy_markers")) AND (embedding("history_explainer") OR projection("balance_medium") OR projection("balance_complex"))
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_psychology (description = "Mid-tier psychology and behavior queries with nuanced explanation.") {
  PRIORITY 205
  TIER 14
  WHEN domain("psychology") AND embedding("psychology_support") AND (projection("balance_simple") OR projection("balance_medium") OR projection("balance_complex"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = true, effort = "low")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE engaged_general (description = "General and psychology-adjacent prompts with visible emotion or urgency that merit a sturdier mid-tier fallback.") {
  PRIORITY 202
  TIER 15
  WHEN (projection("emotion_positive") OR projection("emotion_negative") OR projection("urgency_elevated")) AND (domain("other") OR domain("psychology") OR embedding("general_chat_fallback") OR embedding("psychology_support")) AND (context("short_context") OR context("medium_context")) AND NOT (embedding("fast_qa_en") OR embedding("fast_qa_zh") OR embedding("creative_tasks") OR embedding("business_analysis") OR embedding("health_guidance") OR embedding("history_explainer") OR embedding("code_general") OR embedding("complex_stem") OR embedding("architecture_design") OR embedding("agentic_workflows") OR embedding("premium_legal_analysis") OR projection("verification_required") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR keyword("code_request_markers") OR keyword("implementation_markers") OR keyword("creative_request_markers"))
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE medium_creative (description = "Mid-tier creative, copywriting, and ideation requests.") {
  PRIORITY 200
  TIER 16
  WHEN (keyword("creative_request_markers") OR embedding("creative_tasks")) AND (projection("balance_simple") OR projection("balance_medium")) AND NOT (embedding("fast_qa_en") OR embedding("fast_qa_zh") OR embedding("business_analysis") OR embedding("health_guidance") OR embedding("history_explainer") OR embedding("code_general") OR embedding("complex_stem") OR embedding("architecture_design") OR embedding("agentic_workflows") OR embedding("premium_legal_analysis") OR projection("verification_required") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR keyword("reference_heavy_markers"))
  MODEL "google/gemini-2.5-flash-lite" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE reasoning_general (description = "General reasoning escalation for non-domain-specific deep analysis.") {
  PRIORITY 190
  TIER 16
  WHEN (embedding("reasoning_general_en") OR embedding("reasoning_general_zh") OR embedding("research_synthesis") OR keyword("reasoning_request_markers") OR keyword("multi_step_markers") OR keyword("research_request_markers")) AND (projection("balance_medium") OR projection("balance_complex") OR projection("balance_reasoning")) AND NOT (embedding("premium_legal_analysis") OR embedding("architecture_design") OR embedding("complex_stem") OR embedding("business_analysis") OR embedding("health_guidance") OR embedding("history_explainer") OR embedding("fast_qa_en") OR embedding("fast_qa_zh") OR domain("math") OR domain("philosophy") OR keyword("architecture_markers") OR keyword("agentic_request_markers") OR keyword("code_request_markers") OR keyword("implementation_markers") OR preference("agentic_execution"))
  MODEL "openai/gpt5.4" (reasoning = true, effort = "high")
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE feedback_need_clarification (description = "Narrow follow-up lane for explicit clarification requests that should stay on the cheap model.") {
  PRIORITY 185
  TIER 17
  WHEN keyword("clarification_feedback_markers") AND (context("short_context") OR context("medium_context"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_fast_qa_zh (description = "Conservative factual overlay for Chinese short-context questions that explicitly ask for verification.") {
  PRIORITY 181
  TIER 18
  WHEN (embedding("fast_qa_zh") OR keyword("simple_request_markers")) AND language("zh") AND context("short_context") AND (projection("balance_simple") OR projection("balance_medium")) AND (projection("verification_required") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR fact_check("needs_fact_check")) AND NOT (keyword("code_request_markers") OR keyword("implementation_markers"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE simple_fast_qa_zh (description = "Cheapest short-context Chinese factual or definitional answers.") {
  PRIORITY 180
  TIER 20
  WHEN (embedding("fast_qa_zh") OR keyword("simple_request_markers")) AND language("zh") AND context("short_context") AND projection("balance_simple") AND NOT (projection("verification_required") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR keyword("code_request_markers") OR keyword("implementation_markers") OR projection("urgency_elevated"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE simple_fast_qa_en (description = "Cheapest short-context English factual or definitional answers.") {
  PRIORITY 175
  TIER 22
  WHEN (embedding("fast_qa_en") OR keyword("simple_request_markers")) AND language("en") AND context("short_context") AND projection("balance_simple") AND NOT (projection("verification_required") OR fact_check("needs_fact_check") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR keyword("code_request_markers") OR keyword("implementation_markers") OR projection("urgency_elevated"))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

ROUTE verified_fast_qa_en (description = "Conservative factual overlay for English short-context questions that explicitly ask for verification.") {
  PRIORITY 176
  TIER 21
  WHEN (embedding("fast_qa_en") OR keyword("simple_request_markers")) AND language("en") AND context("short_context") AND (projection("balance_simple") OR projection("balance_medium")) AND (projection("verification_required") OR keyword("verification_markers") OR keyword("reference_heavy_markers") OR fact_check("needs_fact_check")) AND NOT (keyword("code_request_markers") OR keyword("implementation_markers"))
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
  TIER 23
  WHEN (context("short_context") AND projection("balance_simple") AND (embedding("general_chat_fallback") OR structure("low_question_density")) AND NOT (user_feedback("wrong_answer") OR keyword("clarification_feedback_markers") OR keyword("simple_request_markers") OR embedding("fast_qa_en") OR embedding("fast_qa_zh") OR fact_check("needs_fact_check") OR projection("verification_required")) OR context("medium_context") AND domain("other") AND (projection("balance_simple") OR projection("balance_medium")) AND NOT (embedding("fast_qa_en") OR embedding("fast_qa_zh")))
  MODEL "qwen/qwen3.5-rocm" (reasoning = false)
  PLUGIN router_replay {
    enabled: true
    max_records: 100000
    capture_request_body: true
    capture_response_body: true
    max_body_bytes: 4096
  }
}

