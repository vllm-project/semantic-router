# =============================================================================
# ROUTING PROFILE
# =============================================================================

ROUTING {
  strategy: priority
}

# =============================================================================
# SIGNALS
# =============================================================================

# =============================================================================
# MODELS
# =============================================================================

MODEL local/deepseek-v4-flash-analyst {
  context_window_size: 262144
  description: "Latest MIT-licensed sparse analyst tier, retained behind a stable judge after correctness calibration."
  capabilities: ["independent_analysis", "long_context", "tool_use", "reasoning_diversity"]
  tags: ["tier:experimental_frontier", "cost:high", "deployment:self_hosted", "physical:deepseek-v4-flash-0731"]
  quality_score: 0.88
  modality: "text"
}

MODEL local/gemma4-26b-balanced {
  context_window_size: 32768
  description: "Fast architecture-diverse MoE tier for balanced general reasoning."
  capabilities: ["reasoning", "multilingual", "structured_output", "long_context"]
  tags: ["tier:balanced", "cost:medium", "latency:fastest_measured", "deployment:self_hosted", "physical:gemma4-26b-a4b"]
  quality_score: 0.89
  modality: "text"
}

MODEL local/qwen3.5-122b-frontier {
  context_window_size: 262144
  description: "Large local MoE tier for accuracy-first synthesis and review."
  capabilities: ["legal_analysis", "high_risk_review", "deep_synthesis"]
  tags: ["tier:premium", "cost:highest", "deployment:self_hosted", "physical:qwen3.5-122b-a10b-fp8"]
  quality_score: 0.94
  modality: "text"
}

MODEL local/qwen3.5-9b-economy {
  context_window_size: 32768
  description: "Small dense local tier for low-cost and short interactive workloads."
  capabilities: ["fast_qa", "explanation", "general_chat"]
  tags: ["tier:economy", "cost:lowest", "latency:fastest", "deployment:self_hosted", "physical:qwen3.5-9b"]
  quality_score: 0.68
  modality: "text"
}

MODEL local/qwen3.5-9b-economy-replica {
  context_window_size: 32768
  description: "Independent replica of the economy tier for latency and load-aware selection."
  capabilities: ["fast_qa", "explanation", "general_chat"]
  tags: ["tier:economy_replica", "cost:lowest", "latency:fastest", "deployment:self_hosted", "physical:qwen3.5-9b"]
  quality_score: 0.68
  modality: "text"
}

MODEL local/qwen3.5-9b-private {
  context_window_size: 32768
  description: "Isolated alias of the local 9B tier for privacy-policy routes."
  capabilities: ["privacy_locality", "sensitive_data", "general_chat"]
  tags: ["tier:private", "cost:low", "deployment:self_hosted", "physical:qwen3.5-9b"]
  quality_score: 0.68
  modality: "text"
}

MODEL local/qwen3.6-27b-coder {
  context_window_size: 32768
  description: "Dense Qwen3.6 tier for coding, structured output, and verification."
  capabilities: ["reasoning", "coding", "structured_output", "tool_use"]
  tags: ["tier:coder", "cost:upper_mid", "deployment:self_hosted", "physical:qwen3.6-27b"]
  quality_score: 0.9
  modality: "text"
}

MODEL local/qwen3.6-35b-flash {
  context_window_size: 32768
  description: "Low-latency alias of Qwen3.6-35B-A3B-FP8 for fast reasoning."
  capabilities: ["fast_qa", "coding", "reasoning"]
  tags: ["tier:flash", "cost:medium", "latency:fast", "deployment:self_hosted", "physical:qwen3.6-35b-a3b-fp8"]
  quality_score: 0.84
  modality: "text"
}

# =============================================================================
# ROUTES
# =============================================================================

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vllm-sr/mom-balanced-v1"]
  recipe: "balanced"
}

ENTRYPOINT {
  model_names: ["vllm-sr/mom-flash-v1"]
  recipe: "speed-first"
}

ENTRYPOINT {
  model_names: ["vllm-sr/mom-economy-v1"]
  recipe: "cost-first"
}

ENTRYPOINT {
  model_names: ["vllm-sr/mom-frontier-v1"]
  recipe: "accuracy-first"
}

ENTRYPOINT {
  model_names: ["vllm-sr/mom-private-v1"]
  recipe: "privacy-first"
}

# =============================================================================
# RECIPE balanced
# =============================================================================

RECIPE balanced (description = "Mixture-of-Models · Balanced — adaptive quality, speed, and efficiency for everyday routing.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword unified_balance_simple_markers {
    operator: "OR"
    keywords: ["quick answer", "answer briefly", "one sentence", "concise summary", "简单回答", "简要说明", "respuesta breve", "réponse brève", "簡潔に答えて", "kurz antworten"]
    method: "regex"
  }

  SIGNAL keyword unified_balance_reasoning_markers {
    operator: "OR"
    keywords: ["analyze the tradeoffs", "analyze the trade-offs", "from first principles", "root cause", "step by step", "system design", "consistency tradeoffs", "competing production failure", "分析取舍", "第一性原理", "根因分析", "逐步推理", "取舍", "根因", "analizar las ventajas y desventajas", "causa raíz", "analyser les compromis", "cause racine", "根本原因を分析", "段階的に推論", "トレードオフ", "根本原因", "トレードオフを分析", "根本原因と", "kompromisse analysieren"]
    method: "regex"
  }

  SIGNAL keyword unified_balance_verification_markers {
    operator: "OR"
    keywords: ["verify the answer", "cite sources", "fact-check", "check the evidence", "provide evidence", "核实答案", "引用来源", "检查证据", "verificar la respuesta", "citer les sources", "答えを検証", "quellen zitieren"]
    method: "regex"
  }

  SIGNAL keyword unified_balance_correction_markers {
    operator: "OR"
    keywords: ["that's wrong", "wrong answer", "please correct the answer", "try again", "回答错了", "请纠正答案", "重新回答", "la respuesta es incorrecta", "corrige la respuesta", "la réponse est incorrecte", "corrige la réponse", "回答が間違っています", "bitte korrigiere die antwort"]
    method: "regex"
  }

  SIGNAL fact_check needs_fact_check {
    description: "Detect claims that benefit from evidence-backed verification."
  }

  SIGNAL user_feedback wrong_answer {
    description: "Detect explicit correction or dissatisfaction with the previous answer."
  }

  SIGNAL reask unified_balance_reask {
    description: "Detect an immediate semantic repeat after an unsatisfactory answer."
    threshold: 0.8
    lookback_turns: 1
  }

  SIGNAL preference unified_balance_terse_preference {
    description: "Detect a durable preference for concise, direct answers beyond exact trigger phrases."
    examples: ["Keep the answer concise and direct.", "请保持回答简洁直接。", "Responde de forma breve y directa.", "Réponds de manière brève et directe.", "簡潔で直接的に答えてください。"]
    threshold: 0.72
  }

  SIGNAL language zh {
    description: "Chinese-language requests."
    threshold: 0.5
  }

  SIGNAL language es {
    description: "Spanish-language requests."
    threshold: 0.5
  }

  SIGNAL language fr {
    description: "French-language requests."
    threshold: 0.5
  }

  SIGNAL language ja {
    description: "Japanese-language requests."
    threshold: 0.5
  }

  SIGNAL language de {
    description: "German-language requests."
    threshold: 0.5
  }

  SIGNAL context unified_balance_long_context {
    description: "Long inputs that justify stronger synthesis."
    min_tokens: "12K"
    max_tokens: "262K"
  }

  SIGNAL structure unified_balance_constraint_dense {
    description: "Requests containing a dense set of explicit constraints."
    feature: { source: { keywords: ["at most", "at least", "must", "without", "no more than", "不超过", "至少", "必须"], type: "keyword_set" }, type: "density" }
    predicate: { gt: 0.08 }
  }

  SIGNAL conversation unified_balance_multi_turn {
    description: "Multi-turn conversations benefit from a small synthesis allowance."
    feature: { source: { role: "user", type: "message" }, type: "count" }
    predicate: { gte: 2 }
  }

  SIGNAL complexity unified_balance_difficulty {
    threshold: 0.08
    description: "Semantic boundary between direct requests and synthesis-heavy work."
    hard: { candidates: ["Analyze a production failure from several competing root causes.", "Design a distributed system and justify its consistency tradeoffs.", "Synthesize conflicting evidence into a defensible recommendation."] }
    easy: { candidates: ["Give a short definition of a common term.", "Summarize one paragraph in a single sentence.", "Explain a basic concept with one example."] }
  }

  PROJECTION score unified_balance_effort_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: -0.35, name: "unified_balance_simple_markers", value_source: "confidence" }, { type: "preference", weight: -0.1, name: "unified_balance_terse_preference", value_source: "confidence" }, { type: "keyword", weight: 0.46, name: "unified_balance_reasoning_markers", value_source: "confidence" }, { type: "keyword", weight: 0.45, name: "unified_balance_verification_markers", value_source: "confidence" }, { type: "fact_check", weight: 0.45, name: "needs_fact_check" }, { type: "user_feedback", weight: 0.35, name: "wrong_answer" }, { type: "keyword", weight: 0.35, name: "unified_balance_correction_markers", value_source: "confidence" }, { type: "reask", weight: 0.25, name: "unified_balance_reask", value_source: "confidence" }, { type: "context", weight: 0.18, name: "unified_balance_long_context" }, { type: "structure", weight: 0.45, name: "unified_balance_constraint_dense" }, { type: "complexity", weight: 0.4, name: "unified_balance_difficulty:hard" }, { type: "complexity", weight: -0.2, name: "unified_balance_difficulty:easy" }, { type: "conversation", weight: 0.08, name: "unified_balance_multi_turn" }, { type: "language", weight: 0.06, name: "zh" }, { type: "language", weight: 0.06, name: "es" }, { type: "language", weight: 0.06, name: "fr" }, { type: "language", weight: 0.06, name: "ja" }, { type: "language", weight: 0.06, name: "de" }]
  }

  PROJECTION mapping unified_balance_effort_band {
    source: "unified_balance_effort_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 10 }
    outputs: [{ name: "unified_balance_standard", lt: 0.32 }, { name: "unified_balance_deliberate", gte: 0.32 }]
  }

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE unified_balance_recovery (description = "Recover from explicit dissatisfaction with a stronger reasoning pool and corrective prompt.") {
    PRIORITY 300
    TIER 1
    WHEN (user_feedback("wrong_answer") OR keyword("unified_balance_correction_markers") OR reask("unified_balance_reask"))
    MODEL "local/qwen3.6-27b-coder" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/gemma4-26b-balanced" (reasoning = false),
          "local/qwen3.5-122b-frontier" (reasoning = false)
    ALGORITHM multi_factor {
      latency_percentile: 95
      on_no_candidates: "first"
      weights: { cost: 0.1, latency: 0.1, load: 0.15, quality: 0.65 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "Rebuild the answer from first principles, correct the earlier miss directly, and make the improvement explicit."
      mode: "insert"
    }
  }

  ROUTE unified_balance_deliberate_route (description = "Increase quality for complex, constrained, long-context, or verification-heavy work.") {
    PRIORITY 200
    TIER 2
    WHEN projection("unified_balance_deliberate")
    MODEL "local/qwen3.5-9b-economy" (reasoning = false),
          "local/qwen3.6-27b-coder" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/qwen3.5-122b-frontier" (reasoning = false)
    ALGORITHM multi_factor {
      latency_percentile: 95
      on_no_candidates: "cheapest"
      weights: { cost: 0.15, latency: 0.15, load: 0.15, quality: 0.55 }
    }
  }

  ROUTE unified_balance_route (description = "Balance quality, observed latency, configured cost, and current load.") {
    PRIORITY 100
    TIER 3
    MODEL "local/qwen3.5-9b-economy" (reasoning = false),
          "local/qwen3.6-27b-coder" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/gemma4-26b-balanced" (reasoning = false),
          "local/qwen3.5-122b-frontier" (reasoning = false)
    ALGORITHM multi_factor {
      latency_percentile: 95
      on_no_candidates: "cheapest"
      weights: { cost: 0.25, latency: 0.2, load: 0.15, quality: 0.4 }
    }
  }

}

# =============================================================================
# RECIPE speed-first
# =============================================================================

RECIPE speed-first (description = "Prefer the lowest observed latency while preserving a bounded lane for heavy requests.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword unified_speed_heavy_markers {
    operator: "OR"
    keywords: ["deep analysis", "detailed architecture", "comprehensive review", "multi-step plan", "深入分析", "详细架构", "全面审查", "多步骤计划", "análisis profundo", "architecture détaillée", "詳細なアーキテクチャ", "gründliche analyse"]
    method: "regex"
  }

  SIGNAL context unified_speed_long_context {
    description: "Long requests where first-token and generation latency should be measured separately."
    min_tokens: "16K"
    max_tokens: "262K"
  }

  SIGNAL structure unified_speed_ordered_workflow {
    description: "Prompts that require an ordered workflow."
    feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["首先", "然后"], ["先", "再"]], type: "sequence" }, type: "sequence" }
  }

  PROJECTION score unified_speed_work_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.6, name: "unified_speed_heavy_markers", value_source: "confidence" }, { type: "context", weight: 0.35, name: "unified_speed_long_context" }, { type: "structure", weight: 0.25, name: "unified_speed_ordered_workflow" }]
  }

  PROJECTION mapping unified_speed_work_band {
    source: "unified_speed_work_score"
    method: "threshold_bands"
    outputs: [{ name: "unified_speed_interactive", lt: 0.35 }, { name: "unified_speed_heavy", gte: 0.35 }]
  }

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE unified_speed_heavy_route (description = "Use live TTFT and TPOT percentiles across efficient models for heavier requests.") {
    PRIORITY 200
    TIER 1
    WHEN projection("unified_speed_heavy")
    MODEL "local/qwen3.5-9b-economy" (reasoning = false),
          "local/qwen3.5-9b-economy-replica" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/gemma4-26b-balanced" (reasoning = false),
          "local/qwen3.6-27b-coder" (reasoning = false)
    ALGORITHM latency_aware {
      description: "Minimize observed generation and first-token latency."
      tpot_percentile: 90
      ttft_percentile: 90
    }
  }

  ROUTE unified_speed_first_route (description = "Choose the fastest healthy candidate from live p90 latency and load.") {
    PRIORITY 100
    TIER 2
    MODEL "local/qwen3.5-9b-economy" (reasoning = false),
          "local/qwen3.5-9b-economy-replica" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/gemma4-26b-balanced" (reasoning = false)
    ALGORITHM multi_factor {
      latency_percentile: 90
      on_no_candidates: "first"
      weights: { latency: 0.85, load: 0.15 }
    }
    PLUGIN semantic_cache {
      enabled: true
      similarity_threshold: 0.9
      ttl_seconds: 900
    }
  }

}

# =============================================================================
# RECIPE cost-first
# =============================================================================

RECIPE cost-first (description = "Keep every request local and spend additional compute only when the request justifies reasoning.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword unified_cost_reasoning_markers {
    operator: "OR"
    keywords: ["reason step by step", "analyze the tradeoffs", "root cause", "design a system", "prove that", "逐步推理", "分析取舍", "根因分析", "设计系统", "证明", "razonar paso a paso", "razona paso a paso", "concevoir un système", "conçois un système", "raisonner étape par étape", "raisonne étape par étape", "段階的に推論", "system entwerfen"]
    method: "regex"
  }

  SIGNAL context unified_cost_long_context {
    description: "Long requests that benefit from local reasoning."
    min_tokens: "16K"
    max_tokens: "262K"
  }

  SIGNAL structure unified_cost_ordered_workflow {
    description: "Multi-stage requests that benefit from local reasoning."
    feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["首先", "然后"], ["先", "再"]], type: "sequence" }, type: "sequence" }
  }

  PROJECTION score unified_cost_compute_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.6, name: "unified_cost_reasoning_markers", value_source: "confidence" }, { type: "context", weight: 0.35, name: "unified_cost_long_context" }, { type: "structure", weight: 0.3, name: "unified_cost_ordered_workflow" }]
  }

  PROJECTION mapping unified_cost_compute_band {
    source: "unified_cost_compute_score"
    method: "threshold_bands"
    outputs: [{ name: "unified_cost_direct", lt: 0.35 }, { name: "unified_cost_reasoning", gte: 0.35 }]
  }

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE unified_cost_local_reasoning (description = "Enable bounded reasoning on the self-hosted model for genuinely demanding requests.") {
    PRIORITY 200
    TIER 1
    WHEN projection("unified_cost_reasoning")
    MODEL "local/qwen3.5-9b-economy" (reasoning = true, effort = "medium")
    ALGORITHM static
  }

  ROUTE unified_cost_first_route (description = "Deterministic local route with semantic reuse for ordinary requests.") {
    PRIORITY 100
    TIER 2
    MODEL "local/qwen3.5-9b-economy" (reasoning = false)
    ALGORITHM static
    PLUGIN semantic_cache {
      enabled: true
      similarity_threshold: 0.88
      ttl_seconds: 3600
    }
  }

}

# =============================================================================
# RECIPE accuracy-first
# =============================================================================

RECIPE accuracy-first (description = "Escalate from a frontier direct answer to ReMoM, Fusion, or Router Flow when the task benefits from orchestration.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword unified_frontier_workflow_markers {
    operator: "OR"
    keywords: ["investigate and implement", "plan and execute", "multi-agent workflow", "delegate to agents", "coordinate multiple agents", "调查并实现", "规划并执行", "多智能体工作流", "协调多个智能体", "investigar e implementar", "planificar y ejecutar", "enquêter et implémenter", "planifier et exécuter", "調査して実装", "計画して実行", "untersuchen und implementieren"]
    method: "regex"
  }

  SIGNAL keyword unified_frontier_fusion_markers {
    operator: "OR"
    keywords: ["independent analyses", "compare multiple expert opinions", "cross-check the answer", "panel of experts", "resolve disagreements", "独立分析", "比较多个专家意见", "交叉验证答案", "解决分歧", "análisis independientes", "comparar opiniones de expertos", "analyses indépendantes", "comparer plusieurs avis d'experts", "独立した分析", "複数の専門家の意見を比較", "unabhängige analysen"]
    method: "regex"
  }

  SIGNAL keyword unified_frontier_deep_markers {
    operator: "OR"
    keywords: ["explore several approaches", "search multiple reasoning paths", "from first principles", "rigorous proof", "derive step by step", "hard reasoning problem", "difficult result", "multiple possible derivations", "探索多种方法", "多条推理路径", "第一性原理", "严格证明", "逐步推导", "explorar varios enfoques", "demostración rigurosa", "explorer plusieurs approches", "preuve rigoureuse", "複数のアプローチを探索", "厳密な証明", "mehrere ansätze untersuchen", "denkwege untersuchen"]
    method: "regex"
  }

  SIGNAL keyword unified_frontier_verification_markers {
    operator: "OR"
    keywords: ["verify with evidence", "verify the answer", "cite reliable sources", "check every factual claim", "用证据核实", "核实每个事实", "verificar con evidencia", "vérifier avec des preuves", "証拠で検証", "mit belegen überprüfen", "\\b(is|was|were)\\b.{0,80}\\b(accurate|true|correct)\\b", "\\b(when was|who invented|what is the population|how tall is)\\b"]
    method: "regex"
  }

  SIGNAL embedding unified_frontier_workflow_intent {
    threshold: 0.78
    candidates: ["Investigate the repository, implement the fix, run tests, and iterate until it works.", "Coordinate specialized agents to research, code, review, and validate a complete solution.", "制定执行计划，完成实现、评审和验证。", "Investiga el repositorio, implementa la solución y valida todos los cambios.", "リポジトリを調査し、修正を実装して検証してください。"]
    aggregation_method: "max"
  }

  SIGNAL embedding unified_frontier_fusion_intent {
    threshold: 0.79
    candidates: ["Ask independent experts to solve the problem and synthesize the most reliable answer.", "Compare conflicting analyses, identify disagreements, and produce a verified conclusion.", "汇总多个独立观点并解决其中的分歧。", "Compara análisis independientes, resuelve desacuerdos y sintetiza la respuesta.", "複数の独立した分析を比較し、相違点を解決してください。"]
    aggregation_method: "max"
  }

  SIGNAL embedding unified_frontier_deep_intent {
    threshold: 0.78
    candidates: ["Explore several reasoning paths and synthesize the strongest rigorous solution.", "Build a careful derivation from first principles for a difficult problem.", "从多个推理路径探索难题并综合最强答案。", "Explora varias rutas de razonamiento y sintetiza la solución más rigurosa.", "複数の推論経路を探索し、最も厳密な解答を統合してください。"]
    aggregation_method: "max"
  }

  SIGNAL fact_check needs_fact_check {
    description: "Detect factual requests that benefit from independent verification."
  }

  SIGNAL language zh {
    description: "Chinese-language requests."
    threshold: 0.5
  }

  SIGNAL language es {
    description: "Spanish-language requests."
    threshold: 0.5
  }

  SIGNAL language fr {
    description: "French-language requests."
    threshold: 0.5
  }

  SIGNAL language ja {
    description: "Japanese-language requests."
    threshold: 0.5
  }

  SIGNAL language de {
    description: "German-language requests."
    threshold: 0.5
  }

  SIGNAL context unified_frontier_long_context {
    description: "Long-context tasks that benefit from multi-pass synthesis."
    min_tokens: "16K"
    max_tokens: "262K"
  }

  SIGNAL structure unified_frontier_ordered_workflow {
    description: "Prompts that explicitly describe a multi-stage workflow."
    feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["首先", "然后"], ["先", "再"]], type: "sequence" }, type: "sequence" }
  }

  SIGNAL structure unified_frontier_constraint_dense {
    description: "Prompts with dense correctness or output constraints."
    feature: { source: { keywords: ["must", "exactly", "at least", "at most", "verify", "必须", "严格", "至少", "不超过"], type: "keyword_set" }, type: "density" }
    predicate: { gt: 0.08 }
  }

  SIGNAL structure unified_frontier_direct_reference {
    description: "Detect requests that quote orchestration vocabulary only to define, translate, or briefly explain it; this independently suppresses workflow and fusion escalation even when their keyword signals match."
    feature: { source: { pattern: "(?i)(\\b(define|translate|explain)\\b.{0,48}\\b(phrase|term|expression)\\b|(définis|traduis|explique).{0,48}(expression|terme)|(define|traduce|explica).{0,48}(frase|expresión|término)|(definiere|übersetze|erkläre).{0,48}(ausdruck|begriff)|(定义|翻译|解释).{0,16}(短语|词语|“|「)|(説明|翻訳).{0,16}(表現|語句|「))", type: "regex" }, type: "exists" }
  }

  SIGNAL conversation unified_frontier_tooling_available {
    description: "Tool-rich requests are candidates for Router Flow decomposition."
    feature: { source: { type: "tool_definition" }, type: "count" }
    predicate: { gte: 2 }
  }

  SIGNAL conversation unified_frontier_active_tool_loop {
    description: "Continue an already active tool loop through Router Flow."
    feature: { source: { type: "active_tool_loop" }, type: "exists" }
  }

  SIGNAL complexity unified_frontier_complexity {
    threshold: 0.15
    description: "Semantic boundary for tasks that merit multi-round reasoning."
    hard: { candidates: ["Prove a difficult result by exploring multiple possible derivations.", "Diagnose a complex distributed-system failure with incomplete evidence.", "Synthesize competing scientific explanations into a rigorous conclusion."] }
    easy: { candidates: ["Explain a familiar concept in plain language.", "Summarize a short paragraph.", "Answer a direct factual question."] }
  }

  PROJECTION score unified_frontier_workflow_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.7, name: "unified_frontier_workflow_markers", value_source: "confidence" }, { type: "embedding", weight: 0.55, name: "unified_frontier_workflow_intent", value_source: "confidence" }, { type: "conversation", weight: 0.1, name: "unified_frontier_tooling_available" }, { type: "conversation", weight: 0.1, name: "unified_frontier_active_tool_loop" }, { type: "structure", weight: 0.25, name: "unified_frontier_ordered_workflow" }, { type: "structure", weight: -0.9, name: "unified_frontier_direct_reference" }]
  }

  PROJECTION score unified_frontier_fusion_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.7, name: "unified_frontier_fusion_markers", value_source: "confidence" }, { type: "embedding", weight: 0.55, name: "unified_frontier_fusion_intent", value_source: "confidence" }, { type: "fact_check", weight: 0.15, name: "needs_fact_check" }, { type: "structure", weight: -0.9, name: "unified_frontier_direct_reference" }]
  }

  PROJECTION score unified_frontier_deliberation_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.45, name: "unified_frontier_deep_markers", value_source: "confidence" }, { type: "embedding", weight: 0.35, name: "unified_frontier_deep_intent", value_source: "confidence" }, { type: "fact_check", weight: 0.2, name: "needs_fact_check" }, { type: "context", weight: 0.2, name: "unified_frontier_long_context" }, { type: "structure", weight: 0.15, name: "unified_frontier_ordered_workflow" }, { type: "structure", weight: 0.1, name: "unified_frontier_constraint_dense" }, { type: "complexity", weight: 0.35, name: "unified_frontier_complexity:hard" }, { type: "structure", weight: -0.65, name: "unified_frontier_direct_reference" }, { type: "language", weight: 0.05, name: "zh" }, { type: "language", weight: 0.05, name: "es" }, { type: "language", weight: 0.05, name: "fr" }, { type: "language", weight: 0.05, name: "ja" }, { type: "language", weight: 0.05, name: "de" }]
  }

  PROJECTION mapping unified_frontier_workflow_band {
    source: "unified_frontier_workflow_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 12 }
    outputs: [{ name: "unified_frontier_not_workflow", lt: 0.5 }, { name: "unified_frontier_use_workflow", gte: 0.5 }]
  }

  PROJECTION mapping unified_frontier_fusion_band {
    source: "unified_frontier_fusion_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 12 }
    outputs: [{ name: "unified_frontier_not_fusion", lt: 0.5 }, { name: "unified_frontier_use_fusion", gte: 0.5 }]
  }

  PROJECTION mapping unified_frontier_deliberation_band {
    source: "unified_frontier_deliberation_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 10 }
    outputs: [{ name: "unified_frontier_direct", lt: 0.35 }, { name: "unified_frontier_deliberate", gte: 0.35 }]
  }

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE unified_frontier_workflow (description = "Use Router Flow for explicit investigate-plan-execute tasks with separable roles.") {
    PRIORITY 400
    TIER 1
    WHEN projection("unified_frontier_use_workflow") AND (keyword("unified_frontier_workflow_markers") OR structure("unified_frontier_ordered_workflow"))
    MODEL "local/qwen3.6-27b-coder" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/qwen3.5-122b-frontier" (reasoning = false)
    ALGORITHM workflows {
      include_intermediate_responses: false
      max_parallel: 3
      max_steps: 4
      min_successful_responses: 2
      mode: "dynamic"
      on_error: "skip"
      planner: { model: "local/qwen3.6-27b-coder" }
      round_timeout_seconds: 120
      template: "micro_agent"
    }
  }

  ROUTE unified_frontier_tool_result_synthesis (description = "Synthesize an existing client tool result directly without spawning another tool or multi-agent loop.") {
    PRIORITY 375
    TIER 2
    WHEN conversation("unified_frontier_active_tool_loop") AND NOT keyword("unified_frontier_workflow_markers")
    MODEL "local/qwen3.5-122b-frontier" (reasoning = true, effort = "medium")
    ALGORITHM static
    PLUGIN tools {
      enabled: true
      mode: "none"
    }
  }

  ROUTE unified_frontier_fusion (description = "Use independent expert answers plus a frontier judge when disagreement resolution matters.") {
    PRIORITY 350
    TIER 3
    WHEN projection("unified_frontier_use_fusion") AND keyword("unified_frontier_fusion_markers")
    MODEL "local/qwen3.6-27b-coder" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/gemma4-26b-balanced" (reasoning = false),
          "local/deepseek-v4-flash-analyst" (reasoning = false),
          "local/qwen3.5-122b-frontier" (reasoning = false)
    ALGORITHM fusion {
      analysis_models: ["local/qwen3.6-27b-coder", "local/gemma4-26b-balanced", "local/deepseek-v4-flash-analyst", "local/qwen3.5-122b-frontier"]
      include_analysis: false
      include_intermediate_responses: false
      judge_prompt_version: "fusion-v1"
      max_concurrent: 4
      min_successful_responses: 2
      model: "local/qwen3.5-122b-frontier"
      on_error: "skip"
      round_timeout_seconds: 180
      temperature: 0.2
    }
  }

  ROUTE unified_frontier_verified_answer (description = "Escalate evidence-sensitive factual answers from an efficient model to frontier models only when confidence is insufficient.") {
    PRIORITY 325
    TIER 4
    WHEN (keyword("unified_frontier_verification_markers") OR fact_check("needs_fact_check"))
    MODEL "local/qwen3.5-9b-economy" (reasoning = false),
          "local/qwen3.6-27b-coder" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/gemma4-26b-balanced" (reasoning = false),
          "local/qwen3.5-122b-frontier" (reasoning = false)
    ALGORITHM confidence {
      confidence_method: "hybrid"
      escalation_order: "small_to_large"
      on_error: "skip"
      threshold: 0.72
    }
  }

  ROUTE unified_frontier_remom (description = "Use bounded multi-round search and synthesis for deep reasoning tasks.") {
    PRIORITY 300
    TIER 5
    WHEN projection("unified_frontier_deliberate")
    MODEL "local/qwen3.6-27b-coder" (reasoning = true, effort = "high"),
          "local/deepseek-v4-flash-analyst" (reasoning = false),
          "local/qwen3.5-122b-frontier" (reasoning = true, effort = "high")
    ALGORITHM remom {
      breadth_schedule: [3, 2]
      compaction_strategy: "last_n_tokens"
      compaction_tokens: 8000
      include_reasoning: true
      max_concurrent: 3
      max_responses_per_round: 3
      min_successful_responses: 2
      model_distribution: "round_robin"
      on_error: "skip"
      round_timeout_seconds: 180
      synthesis_model: "local/qwen3.5-122b-frontier"
      temperature: 0.6
    }
  }

  ROUTE unified_frontier_direct (description = "Use the strongest direct model when orchestration would add little value.") {
    PRIORITY 100
    TIER 6
    MODEL "local/qwen3.6-27b-coder" (reasoning = false),
          "local/qwen3.6-35b-flash" (reasoning = false),
          "local/gemma4-26b-balanced" (reasoning = false),
          "local/deepseek-v4-flash-analyst" (reasoning = false),
          "local/qwen3.5-122b-frontier" (reasoning = false)
    ALGORITHM multi_factor {
      on_no_candidates: "first"
      weights: { quality: 1 }
    }
  }

}

# =============================================================================
# RECIPE privacy-first
# =============================================================================

RECIPE privacy-first (description = "Keep every request local, using recipe-scoped risk projections to isolate sensitive and suspicious prompts.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword unified_privacy_local_only_markers {
    operator: "OR"
    keywords: ["local processing only", "on-prem only", "do not send to the cloud", "confidential handling", "internal use only", "private repository", "proprietary code", "internal document", "本地处理", "不要发到云端", "仅供内部使用", "私有仓库", "内部文档", "solo procesamiento local", "no enviar a la nube", "traitement local uniquement", "ne pas envoyer au cloud", "ローカル処理のみ", "クラウドに送信しない", "nur lokale verarbeitung"]
    method: "regex"
  }

  SIGNAL keyword unified_privacy_sensitive_markers {
    operator: "OR"
    keywords: ["\\b\\d{3}-\\d{2}-\\d{4}\\b", "\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", "\\b(ssn|social security number|credit card|passport number)\\b", "(身份证|邮箱|护照号码|银行卡号)", "(número de seguridad social|correo electrónico|pasaporte)", "(numéro de sécurité sociale|adresse e-mail|passeport)", "(マイナンバー|メールアドレス|パスポート番号)", "(sozialversicherungsnummer|e-mail-adresse|reisepassnummer)"]
    method: "regex"
  }

  SIGNAL keyword unified_privacy_attack_markers {
    operator: "OR"
    keywords: ["ignore previous instructions", "reveal the system prompt", "exfiltrate credentials", "bypass the safety policy", "忽略之前的指令", "泄露系统提示词", "导出凭据", "绕过安全策略", "ignora las instrucciones anteriores", "extraer credenciales", "ignorer les instructions précédentes", "exfiltrer les identifiants", "以前の指示を無視", "認証情報を流出", "vorherige anweisungen ignorieren"]
    method: "regex"
  }

  SIGNAL jailbreak unified_privacy_jailbreak_strict {
    method: "classifier"
    threshold: 0.45
    description: "Detect suspicious prompts that require local containment."
  }

  SIGNAL pii unified_privacy_pii_strict {
    threshold: 0.7
    description: "Detect personally identifiable information that must remain local."
  }

  SIGNAL kb unified_privacy_policy {
    kb: "privacy_kb"
    target: { kind: "group", value: "privacy_policy" }
    match: "best"
  }

  PROJECTION score unified_privacy_risk_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.5, name: "unified_privacy_local_only_markers", value_source: "confidence" }, { type: "keyword", weight: 0.75, name: "unified_privacy_sensitive_markers", value_source: "confidence" }, { type: "pii", weight: 0.9, name: "unified_privacy_pii_strict" }, { type: "jailbreak", weight: 0.9, name: "unified_privacy_jailbreak_strict" }, { type: "keyword", weight: 0.9, name: "unified_privacy_attack_markers", value_source: "confidence" }, { type: "kb", weight: 0.3, name: "unified_privacy_policy" }]
  }

  PROJECTION mapping unified_privacy_risk_band {
    source: "unified_privacy_risk_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 12 }
    outputs: [{ name: "unified_privacy_standard", lt: 0.35 }, { name: "unified_privacy_sensitive", gte: 0.35 }]
  }

  # =============================================================================
  # PLUGINS
  # =============================================================================

  PLUGIN tools tools {}

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE unified_privacy_security_containment (description = "Disable tool access for suspicious prompts while keeping inference local.") {
    PRIORITY 300
    TIER 1
    WHEN (jailbreak("unified_privacy_jailbreak_strict") OR keyword("unified_privacy_attack_markers"))
    MODEL "local/qwen3.5-9b-private" (reasoning = false)
    ALGORITHM static
    PLUGIN tools {
      enabled: true
      mode: "none"
    }
  }

  ROUTE unified_privacy_sensitive_route (description = "Keep PII, private-domain, and explicit local-only work on the local model with no tools.") {
    PRIORITY 200
    TIER 2
    WHEN projection("unified_privacy_sensitive")
    MODEL "local/qwen3.5-9b-private" (reasoning = false)
    ALGORITHM static
    PLUGIN tools {
      enabled: true
      mode: "none"
    }
  }

  ROUTE unified_privacy_local_default (description = "Route non-sensitive traffic locally as the privacy-first default.") {
    PRIORITY 100
    TIER 3
    MODEL "local/qwen3.5-9b-private" (reasoning = false)
  }

}
