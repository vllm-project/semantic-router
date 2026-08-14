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

MODEL local/glm-5.2 {
  param_size: "743B-A39B"
  context_window_size: 524288
  description: "Text-only frontier tier for high-accuracy synthesis and long-context reasoning."
  capabilities: ["chat", "reasoning", "code", "tools", "structured-output", "long-context"]
  tags: ["tier:frontier", "precision:fp8", "tp:8", "node:node-b", "text-only"]
  quality_score: 0.97
  modality: "ar"
}

MODEL local/gpt-oss-120b {
  param_size: "120B"
  context_window_size: 131072
  description: "Open-weight reasoning tier used for isolated local handling."
  capabilities: ["chat", "reasoning", "code", "tools"]
  tags: ["tier:private", "precision:mxfp4", "tp:1", "node:node-a"]
  quality_score: 0.9
  modality: "ar"
}

MODEL local/mistral-small-4 {
  param_size: "119B"
  context_window_size: 131072
  description: "Provider-diverse tier for independent analysis and code review."
  capabilities: ["chat", "reasoning", "code", "tools", "vision", "multimodal", "long-context"]
  tags: ["tier:frontier", "precision:fp8", "tp:2", "node:node-a"]
  quality_score: 0.91
  modality: "ar"
}

MODEL local/qwen3.5-122b {
  param_size: "122B-A10B"
  context_window_size: 131072
  description: "Accuracy-first sparse tier for synthesis and review."
  capabilities: ["chat", "reasoning", "code", "tools", "vision", "multimodal", "long-context"]
  tags: ["tier:frontier", "precision:fp8", "tp:1", "node:node-a"]
  quality_score: 0.92
  modality: "ar"
}

MODEL local/qwen3.5-9b {
  param_size: "9B"
  context_window_size: 262144
  description: "Economy dense tier for efficient long-context text and tool traffic."
  capabilities: ["chat", "reasoning", "tools", "structured-output", "long-context"]
  tags: ["tier:economy", "precision:bf16", "tp:1", "node:node-a"]
  quality_score: 0.76
  modality: "ar"
}

MODEL local/qwen3.6-35b {
  param_size: "35B-A3B"
  context_window_size: 262144
  description: "Fast sparse tier for long-context reasoning, code, and vision."
  capabilities: ["chat", "reasoning", "code", "tools", "vision", "multimodal", "structured-output", "long-context"]
  tags: ["tier:fast", "precision:fp8", "tp:1", "node:node-a"]
  quality_score: 0.86
  modality: "ar"
}

MODEL local/step-3.7-flash {
  param_size: "198B-A11B"
  context_window_size: 65536
  description: "Frontier flash tier with multimodal understanding enabled."
  capabilities: ["chat", "reasoning", "tools", "vision", "multimodal"]
  tags: ["tier:frontier-flash", "precision:fp8", "tp:2", "node:node-a"]
  quality_score: 0.93
  modality: "ar"
}

# =============================================================================
# ROUTES
# =============================================================================

# =============================================================================
# ENTRYPOINTS
# =============================================================================

ENTRYPOINT {
  model_names: ["vllm-sr/chorus-v1"]
  recipe: "balance"
}

ENTRYPOINT {
  model_names: ["vllm-sr/chorus-v1-lite"]
  recipe: "cost"
}

ENTRYPOINT {
  model_names: ["vllm-sr/chorus-v1-flash"]
  recipe: "speed"
}

ENTRYPOINT {
  model_names: ["vllm-sr/chorus-v1-ultra"]
  recipe: "accuracy"
}

ENTRYPOINT {
  model_names: ["vllm-sr/chorus-v1-vault"]
  recipe: "vault"
}

# =============================================================================
# RECIPE balance
# =============================================================================

RECIPE balance (description = "A balanced model optimized across quality, speed, and cost.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword balance_simple_request_phrases {
    operator: "OR"
    keywords: ["quick answer", "answer briefly", "one sentence", "concise summary", "简单回答", "简要说明", "简短回答", "respuesta breve", "réponse brève", "簡潔に答えて", "kurz antworten", "resposta breve", "짧게 답해", "إجابة مختصرة", "संक्षिप्त उत्तर", "краткий ответ"]
    method: "regex"
  }

  SIGNAL keyword balance_concise_request_phrases {
    operator: "OR"
    keywords: ["concise and direct", "brief and direct", "简洁直接", "简短直接", "breve y directa", "brève et directe", "簡潔で直接的", "kurz und direkt", "breve e direta", "간결하고 직접적으로", "موجز ومباشر", "संक्षिप्त और सीधा", "кратко и прямо"]
    method: "regex"
  }

  SIGNAL keyword balance_reasoning_opt_out_phrases {
    operator: "OR"
    keywords: ["do not analyze", "don't analyze", "without analysis", "不要分析", "无需分析", "no analices", "sans analyse", "分析しない", "nicht analysieren", "sem análise", "분석하지 마", "دون تحليل", "विश्लेषण मत करो", "без анализа"]
    method: "regex"
  }

  SIGNAL keyword balance_deliberate_request_phrases {
    operator: "OR"
    keywords: ["analyze the tradeoffs", "from first principles", "root cause", "reason step by step", "compare the alternatives carefully", "分析取舍", "第一性原理", "根因分析", "逐步推理", "analizar las ventajas y desventajas", "causa raíz", "analyser les compromis", "cause racine", "トレードオフを分析", "根本原因", "kompromisse analysieren", "ursache analysieren", "analisar as compensações", "causa raiz", "트레이드오프를 분석", "근본 원인", "حلل المفاضلات", "السبب الجذري", "समझौतों का विश्लेषण", "मूल कारण", "проанализируй компромиссы", "первопричина"]
    method: "regex"
  }

  SIGNAL keyword balance_verification_request_phrases {
    operator: "OR"
    keywords: ["verify the answer", "cite sources", "fact-check", "check the evidence", "核实答案", "引用来源", "verificar la respuesta", "citer les sources", "答えを検証", "quellen zitieren", "verificar a resposta", "답변을 검증", "تحقق من الإجابة", "उत्तर सत्यापित करें", "проверь ответ"]
    method: "regex"
  }

  SIGNAL keyword balance_correction_request_phrases {
    operator: "OR"
    keywords: ["that's wrong", "wrong answer", "please correct the answer", "try again", "回答错了", "请纠正答案", "la respuesta es incorrecta", "corrige la respuesta", "la réponse est incorrecte", "corrige la réponse", "回答が間違っています", "bitte korrigiere die antwort", "a resposta está errada", "답변이 틀렸습니다", "الإجابة خاطئة", "उत्तर गलत है", "ответ неверный"]
    method: "regex"
  }

  SIGNAL fact_check balance_needs_fact_check {
    description: "Detect claims that benefit from evidence-backed verification."
  }

  SIGNAL user_feedback balance_user_reports_incorrect_answer {
    description: "Detect explicit correction or dissatisfaction with the previous answer."
  }

  SIGNAL reask balance_repeated_question {
    description: "Detect an immediate semantic repeat after an unsatisfactory answer."
    threshold: 0.8
    lookback_turns: 1
  }

  SIGNAL language en {
    description: "English-language request."
    threshold: 0.5
  }

  SIGNAL language zh {
    description: "Chinese-language request."
    threshold: 0.5
  }

  SIGNAL language es {
    description: "Spanish-language request."
    threshold: 0.5
  }

  SIGNAL language fr {
    description: "French-language request."
    threshold: 0.5
  }

  SIGNAL language ja {
    description: "Japanese-language request."
    threshold: 0.5
  }

  SIGNAL language de {
    description: "German-language request."
    threshold: 0.5
  }

  SIGNAL language pt {
    description: "Portuguese-language request."
    threshold: 0.5
  }

  SIGNAL language ko {
    description: "Korean-language request."
    threshold: 0.5
  }

  SIGNAL language ar {
    description: "Arabic-language request."
    threshold: 0.5
  }

  SIGNAL language hi {
    description: "Hindi-language request."
    threshold: 0.5
  }

  SIGNAL language ru {
    description: "Russian-language request."
    threshold: 0.5
  }

  SIGNAL context balance_context_from_30k_to_60k {
    description: "The 30K-60K text/tool band preserves output, system, and tool reserve on 65K-capable backends."
    min_tokens: "30001"
    max_tokens: "60000"
  }

  SIGNAL context balance_context_from_60k_to_120k {
    description: "The 60K-120K text/tool band preserves reserve on 128K-capable backends."
    min_tokens: "60001"
    max_tokens: "120000"
  }

  SIGNAL context balance_context_from_120k_to_240k {
    description: "The 120K-240K text/tool band preserves reserve on the two declared native-262K backends."
    min_tokens: "120001"
    max_tokens: "240000"
  }

  SIGNAL context balance_exceeds_240k_context {
    description: "Inputs beyond 240K require a capability-preserving terminal lane."
    min_tokens: "240001"
    max_tokens: "2147483647"
  }

  SIGNAL structure balance_constraint_dense {
    description: "Detect a dense set of explicit output or correctness constraints."
    feature: { source: { keywords: ["must", "exactly", "at least", "at most", "without", "必须", "严格", "至少", "不超过"], type: "keyword_set" }, type: "density" }
    predicate: { gt: 0.08 }
  }

  SIGNAL conversation balance_has_images {
    description: "Request contains at least one image content part."
    feature: { source: { type: "image_content" }, type: "exists" }
  }

  SIGNAL conversation balance_has_tools {
    description: "Request declares at least one callable tool."
    feature: { source: { type: "tool_definition" }, type: "count" }
    predicate: { gte: 1 }
  }

  SIGNAL conversation balance_multi_turn {
    description: "Multi-turn conversations benefit from a small synthesis allowance."
    feature: { source: { role: "user", type: "message" }, type: "count" }
    predicate: { gte: 2 }
  }

  SIGNAL complexity balance_difficulty {
    threshold: 0.08
    description: "Semantic boundary between direct requests and synthesis-heavy work."
    hard: { candidates: ["Analyze a production failure from several competing root causes.", "Design a distributed system and justify its consistency tradeoffs.", "Synthesize conflicting evidence into a defensible recommendation.", "分析复杂生产故障中的多个竞争性根因并提出可靠方案。", "Analiza varias causas raíz y sintetiza una recomendación defendible.", "حلّل عدة أسباب جذرية متنافسة وقدّم توصية قابلة للدفاع."] }
    easy: { candidates: ["Give a short definition of a common term.", "Summarize one paragraph in a single sentence.", "Explain a basic concept with one example."] }
  }

  PROJECTION score balance_reasoning_effort_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: -0.35, name: "balance_simple_request_phrases", value_source: "confidence" }, { type: "keyword", weight: -0.1, name: "balance_concise_request_phrases", value_source: "confidence" }, { type: "keyword", weight: -1, name: "balance_reasoning_opt_out_phrases", value_source: "confidence" }, { type: "keyword", weight: 0.85, name: "balance_deliberate_request_phrases", value_source: "confidence" }, { type: "keyword", weight: 0.45, name: "balance_verification_request_phrases", value_source: "confidence" }, { type: "fact_check", weight: 0.45, name: "balance_needs_fact_check" }, { type: "structure", weight: 0.45, name: "balance_constraint_dense" }, { type: "complexity", weight: 0.4, name: "balance_difficulty:hard" }, { type: "complexity", weight: -0.05, name: "balance_difficulty:easy" }, { type: "conversation", weight: 0.08, name: "balance_multi_turn" }, { type: "language", weight: 0.02, name: "en" }, { type: "language", weight: 0.02, name: "zh" }, { type: "language", weight: 0.02, name: "es" }, { type: "language", weight: 0.02, name: "fr" }, { type: "language", weight: 0.02, name: "ja" }, { type: "language", weight: 0.02, name: "de" }, { type: "language", weight: 0.02, name: "pt" }, { type: "language", weight: 0.02, name: "ko" }, { type: "language", weight: 0.02, name: "ar" }, { type: "language", weight: 0.02, name: "hi" }, { type: "language", weight: 0.02, name: "ru" }]
  }

  PROJECTION score balance_answer_recovery_score {
    method: "weighted_sum"
    inputs: [{ type: "user_feedback", weight: 0.65, name: "balance_user_reports_incorrect_answer" }, { type: "keyword", weight: 0.5, name: "balance_correction_request_phrases", value_source: "confidence" }, { type: "reask", weight: 0.4, name: "balance_repeated_question", value_source: "confidence" }]
  }

  PROJECTION mapping balance_reasoning_effort_band {
    source: "balance_reasoning_effort_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 10 }
    outputs: [{ name: "balance_direct_workload", lt: 0.3 }, { name: "balance_deliberate_workload", gte: 0.3 }]
  }

  PROJECTION mapping balance_answer_recovery_band {
    source: "balance_answer_recovery_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 12 }
    outputs: [{ name: "balance_no_recovery_needed", lt: 0.35 }, { name: "balance_needs_recovery", gte: 0.35 }]
  }

  # =============================================================================
  # PLUGINS
  # =============================================================================

  PLUGIN system_prompt system_prompt {}

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE balance_over_240k_image_guard (description = "Keep image input beyond the conservative 240K boundary on Qwen36 so vLLM rejects true native-262K overflow instead of misrouting to text-only GLM.") {
    PRIORITY 900
    WHEN context("balance_exceeds_240k_context") AND conversation("balance_has_images")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_over_240k_text_guard (description = "Route text and tool requests beyond 240K to GLM; its native context contract enforces terminal overflow.") {
    PRIORITY 850
    WHEN context("balance_exceeds_240k_context") AND NOT conversation("balance_has_images")
    MODEL "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_text_from_120k_to_240k (description = "Balance text and tool traffic in the 120K-240K band across the two declared native-262K backends.") {
    PRIORITY 800
    WHEN context("balance_context_from_120k_to_240k") AND NOT conversation("balance_has_images")
    MODEL "local/qwen3.5-9b" (reasoning = true, effort = "high", param_size = "9B"),
          "local/qwen3.6-35b" (reasoning = true, effort = "high", param_size = "35B-A3B")
    ALGORITHM multi_factor {
      on_no_candidates: "cheapest"
      weights: { cost: 0.15, latency: 0.15, load: 0.15, quality: 0.55 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_text_from_60k_to_120k (description = "Balance text and tool traffic in the 60K-120K band across native-262K backends and compatible strong 128K backends.") {
    PRIORITY 750
    WHEN context("balance_context_from_60k_to_120k") AND NOT conversation("balance_has_images")
    MODEL "local/qwen3.5-9b" (reasoning = true, effort = "high", param_size = "9B"),
          "local/qwen3.6-35b" (reasoning = true, effort = "high", param_size = "35B-A3B"),
          "local/qwen3.5-122b" (reasoning = true, effort = "high", param_size = "122B-A10B"),
          "local/mistral-small-4" (reasoning = true, effort = "high", param_size = "119B")
    ALGORITHM multi_factor {
      on_no_candidates: "cheapest"
      weights: { cost: 0.15, latency: 0.15, load: 0.15, quality: 0.55 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_text_from_30k_to_60k (description = "Balance text and tool traffic in the 30K-60K band across native-262K backends and compatible strong shorter-context backends.") {
    PRIORITY 700
    WHEN context("balance_context_from_30k_to_60k") AND NOT conversation("balance_has_images")
    MODEL "local/qwen3.5-9b" (reasoning = true, effort = "high", param_size = "9B"),
          "local/qwen3.6-35b" (reasoning = true, effort = "high", param_size = "35B-A3B"),
          "local/step-3.7-flash" (reasoning = true, effort = "high", param_size = "198B-A11B"),
          "local/qwen3.5-122b" (reasoning = true, effort = "high", param_size = "122B-A10B"),
          "local/mistral-small-4" (reasoning = true, effort = "high", param_size = "119B")
    ALGORITHM multi_factor {
      on_no_candidates: "cheapest"
      weights: { cost: 0.15, latency: 0.15, load: 0.15, quality: 0.55 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_image_tools (description = "Keep image-and-tool requests through 240K on the declared native-262K multimodal backend.") {
    PRIORITY 500
    WHEN conversation("balance_has_images") AND conversation("balance_has_tools") AND NOT context("balance_exceeds_240k_context")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_image (description = "Keep image requests through 240K on the declared native-262K multimodal backend.") {
    PRIORITY 450
    WHEN conversation("balance_has_images") AND NOT conversation("balance_has_tools") AND NOT context("balance_exceeds_240k_context")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_tools (description = "Route short tool requests only to tool-capable standard candidates.") {
    PRIORITY 400
    WHEN conversation("balance_has_tools") AND NOT conversation("balance_has_images")
    MODEL "local/qwen3.5-9b" (reasoning = false, param_size = "9B"),
          "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B"),
          "local/qwen3.5-122b" (reasoning = false, param_size = "122B-A10B")
    ALGORITHM multi_factor {
      on_no_candidates: "cheapest"
      weights: { cost: 0.25, latency: 0.2, load: 0.15, quality: 0.4 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_answer_recovery (description = "Rebuild an answer after explicit dissatisfaction or an immediate semantic re-ask, without bypassing capability guards.") {
    PRIORITY 300
    WHEN projection("balance_needs_recovery")
    MODEL "local/qwen3.6-35b" (reasoning = true, effort = "high", param_size = "35B-A3B"),
          "local/qwen3.5-122b" (reasoning = true, effort = "high", param_size = "122B-A10B"),
          "local/mistral-small-4" (reasoning = true, effort = "high", param_size = "119B")
    ALGORITHM multi_factor {
      on_no_candidates: "first"
      weights: { cost: 0.1, latency: 0.1, load: 0.15, quality: 0.65 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Rebuild the answer from first principles, correct the earlier miss directly, and make the improvement explicit."
      mode: "insert"
    }
  }

  ROUTE balance_deliberate (description = "Balance semantically difficult, constrained, or verification-heavy work across strong reasoning candidates.") {
    PRIORITY 200
    WHEN projection("balance_deliberate_workload")
    MODEL "local/step-3.7-flash" (reasoning = true, effort = "high", param_size = "198B-A11B"),
          "local/qwen3.5-122b" (reasoning = true, effort = "high", param_size = "122B-A10B"),
          "local/mistral-small-4" (reasoning = true, effort = "high", param_size = "119B"),
          "local/qwen3.6-35b" (reasoning = true, effort = "high", param_size = "35B-A3B")
    ALGORITHM multi_factor {
      on_no_candidates: "cheapest"
      weights: { cost: 0.15, latency: 0.15, load: 0.15, quality: 0.55 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

  ROUTE balance_standard (description = "Balance ordinary work across economical and capable candidates.") {
    PRIORITY 10
    MODEL "local/qwen3.5-9b" (reasoning = false, param_size = "9B"),
          "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B"),
          "local/qwen3.5-122b" (reasoning = false, param_size = "122B-A10B")
    ALGORITHM multi_factor {
      on_no_candidates: "cheapest"
      weights: { cost: 0.25, latency: 0.2, load: 0.15, quality: 0.4 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1, built by AMD. Balance accuracy, speed, and cost. Give one clear answer."
      mode: "insert"
    }
  }

}

# =============================================================================
# RECIPE speed
# =============================================================================

RECIPE speed (description = "A speed-first model for low-latency, real-time experiences.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword speed_heavy_request_phrases {
    operator: "OR"
    keywords: ["deep analysis", "detailed architecture", "comprehensive review", "multi-step plan", "深入分析", "详细架构", "全面审查", "análisis profundo", "architecture détaillée", "詳細なアーキテクチャ", "gründliche analyse", "análise aprofundada", "심층 분석", "تحليل متعمق", "गहन विश्लेषण", "глубокий анализ"]
    method: "regex"
  }

  SIGNAL embedding speed_heavy_intent {
    threshold: 0.77
    candidates: ["Produce a comprehensive architecture and analyze several failure modes.", "对复杂迁移做深入分析并给出详细架构。", "Realiza un análisis profundo y una arquitectura detallada.", "Produis une analyse approfondie et une architecture détaillée.", "複雑な移行を深く分析し、詳細なアーキテクチャを示してください。", "Erstelle eine gründliche Analyse und detaillierte Architektur.", "Faça uma análise aprofundada e uma arquitetura detalhada.", "복잡한 마이그레이션을 심층 분석하고 상세 아키텍처를 제시하세요.", "قدّم تحليلاً متعمقاً وبنية مفصلة.", "गहन विश्लेषण और विस्तृत आर्किटेक्चर प्रस्तुत करें।", "Проведи глубокий анализ и предложи подробную архитектуру."]
    aggregation_method: "max"
  }

  SIGNAL context speed_context_from_30k_to_60k {
    description: "The 30K-60K band records conservative context use within the native-262K latency pool."
    min_tokens: "30001"
    max_tokens: "60000"
  }

  SIGNAL context speed_context_from_60k_to_120k {
    description: "The 60K-120K band records conservative context use within the native-262K latency pool."
    min_tokens: "60001"
    max_tokens: "120000"
  }

  SIGNAL context speed_context_from_120k_to_240k {
    description: "The 120K-240K band preserves output, system, tool, and image reserve on native-262K backends."
    min_tokens: "120001"
    max_tokens: "240000"
  }

  SIGNAL context speed_exceeds_240k_context {
    description: "Inputs beyond 240K require a capability-preserving terminal lane."
    min_tokens: "240001"
    max_tokens: "2147483647"
  }

  SIGNAL structure speed_ordered_workflow {
    description: "Detect prompts that explicitly require an ordered workflow."
    feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["首先", "然后"], ["先", "再"], ["primero", "luego"], ["d'abord", "puis"], ["zuerst", "dann"], ["primeiro", "depois"], ["まず", "次に"], ["먼저", "다음"], ["أولاً", "ثم"], ["पहले", "फिर"], ["сначала", "затем"]], type: "sequence" }, type: "sequence" }
  }

  SIGNAL structure speed_constraint_dense {
    description: "Detect exact output contracts that benefit from the stronger fast lane."
    feature: { source: { keywords: ["must", "exactly", "at least", "at most", "JSON", "schema", "必须", "严格", "至少", "不超过", "debe", "exactement", "muss", "deve", "厳密", "반드시", "يجب", "अनिवार्य", "строго"], type: "keyword_set" }, type: "density" }
    predicate: { gt: 0.08 }
  }

  SIGNAL conversation speed_has_images {
    description: "Request contains at least one image content part."
    feature: { source: { type: "image_content" }, type: "exists" }
  }

  SIGNAL conversation speed_has_tools {
    description: "Request declares at least one callable tool."
    feature: { source: { type: "tool_definition" }, type: "count" }
    predicate: { gte: 1 }
  }

  SIGNAL complexity speed_complexity {
    threshold: 0.1
    description: "Semantic boundary between interactive work and heavier latency-sensitive synthesis."
    hard: { candidates: ["Design a multi-region architecture and analyze competing failure modes.", "Review a complex migration plan with several operational constraints.", "综合多项约束设计分布式系统并分析故障模式。", "صمم نظاماً موزعاً مع تحليل عدة أنماط للفشل."] }
    easy: { candidates: ["Define a familiar term in one sentence.", "Rewrite a short sentence clearly.", "Give one direct factual answer."] }
  }

  PROJECTION score speed_workload_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.6, name: "speed_heavy_request_phrases", value_source: "confidence" }, { type: "embedding", weight: 0.55, name: "speed_heavy_intent", value_source: "confidence" }, { type: "context", weight: 0.15, name: "speed_context_from_30k_to_60k" }, { type: "context", weight: 0.15, name: "speed_context_from_60k_to_120k" }, { type: "context", weight: 0.15, name: "speed_context_from_120k_to_240k" }, { type: "structure", weight: 0.3, name: "speed_ordered_workflow" }, { type: "structure", weight: 0.25, name: "speed_constraint_dense" }, { type: "complexity", weight: 0.35, name: "speed_complexity:hard" }, { type: "complexity", weight: -0.1, name: "speed_complexity:easy" }]
  }

  PROJECTION mapping speed_workload_band {
    source: "speed_workload_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 10 }
    outputs: [{ name: "speed_interactive", lt: 0.35 }, { name: "speed_heavy_workload_required", gte: 0.35 }]
  }

  # =============================================================================
  # PLUGINS
  # =============================================================================

  PLUGIN system_prompt system_prompt {}

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE speed_over_240k_image_guard (description = "Keep image input beyond the conservative 240K boundary on Qwen36 so vLLM rejects true native-262K overflow instead of misrouting to text-only GLM.") {
    PRIORITY 900
    WHEN context("speed_exceeds_240k_context") AND conversation("speed_has_images")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Flash, built by AMD. Answer immediately and concisely."
      mode: "insert"
    }
  }

  ROUTE speed_over_240k_text_guard (description = "Route text and tool requests beyond 240K to GLM; its native context contract enforces terminal overflow.") {
    PRIORITY 850
    WHEN context("speed_exceeds_240k_context") AND NOT conversation("speed_has_images")
    MODEL "local/glm-5.2" (reasoning = false, param_size = "743B-A39B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Flash, built by AMD. Answer immediately and concisely."
      mode: "insert"
    }
  }

  ROUTE speed_image_tools (description = "Keep image-and-tool requests through 240K on the declared native-262K low-latency backend.") {
    PRIORITY 500
    WHEN conversation("speed_has_images") AND conversation("speed_has_tools") AND NOT context("speed_exceeds_240k_context")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Flash, built by AMD. Answer immediately and concisely."
      mode: "insert"
    }
  }

  ROUTE speed_image (description = "Keep image requests through 240K on the declared native-262K low-latency backend.") {
    PRIORITY 450
    WHEN conversation("speed_has_images") AND NOT conversation("speed_has_tools") AND NOT context("speed_exceeds_240k_context")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Flash, built by AMD. Answer immediately and concisely."
      mode: "insert"
    }
  }

  ROUTE speed_tools (description = "Route tool requests through 240K across the two declared native-262K low-latency candidates.") {
    PRIORITY 400
    WHEN conversation("speed_has_tools") AND NOT conversation("speed_has_images")
    MODEL "local/qwen3.5-9b" (reasoning = false, param_size = "9B"),
          "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM latency_aware {
      description: "Minimize observed first-token and generation latency within the native-262K tool-capable pool."
      tpot_percentile: 90
      ttft_percentile: 90
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Flash, built by AMD. Answer immediately and concisely."
      mode: "insert"
    }
  }

  ROUTE speed_heavy_workload (description = "Use live TTFT and TPOT percentiles across native-262K candidates for semantically heavier low-latency work.") {
    PRIORITY 200
    WHEN projection("speed_heavy_workload_required")
    MODEL "local/qwen3.5-9b" (reasoning = false, param_size = "9B"),
          "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM latency_aware {
      description: "Minimize observed first-token and generation latency for heavier requests."
      tpot_percentile: 90
      ttft_percentile: 90
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Flash, built by AMD. Answer immediately and concisely."
      mode: "insert"
    }
  }

  ROUTE speed_interactive (description = "Choose the fastest healthy candidate for interactive text requests using latency and load.") {
    PRIORITY 10
    MODEL "local/qwen3.5-9b" (reasoning = false, param_size = "9B"),
          "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM multi_factor {
      latency_percentile: 90
      on_no_candidates: "first"
      weights: { latency: 0.85, load: 0.15 }
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Flash, built by AMD. Answer immediately and concisely."
      mode: "insert"
    }
  }

}

# =============================================================================
# RECIPE cost
# =============================================================================

RECIPE cost (description = "A cost-first model for efficient, high-volume workloads.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword cost_reasoning_opt_out_phrases {
    operator: "OR"
    keywords: ["do not analyze", "don't analyze", "without analysis", "不要分析", "无需分析", "no analices", "sans analyse", "分析しない", "nicht analysieren", "sem análise", "분석하지 마", "دون تحليل", "विश्लेषण मत करो", "без анализа"]
    method: "regex"
  }

  SIGNAL keyword cost_reasoning_request_phrases {
    operator: "OR"
    keywords: ["reason step by step", "analyze the tradeoffs", "root cause", "design a system", "prove that", "逐步推理", "分析取舍", "根因分析", "复杂根因", "逐步比较", "设计系统", "razona paso a paso", "analyser les compromis", "段階的に推論", "system entwerfen", "raciocine passo a passo", "단계별로 추론", "فكر خطوة بخطوة", "चरण दर चरण तर्क", "рассуждай пошагово"]
    method: "regex"
  }

  SIGNAL embedding cost_reasoning_intent {
    threshold: 0.78
    candidates: ["Analyze a difficult root cause and reason through several tradeoffs.", "分析复杂根因并逐步比较多个方案的取舍。", "Analiza una causa raíz difícil y compara varias alternativas.", "Analyse une cause racine complexe et compare plusieurs compromis.", "複雑な根本原因を分析し、複数の選択肢を段階的に比較してください。", "Analysiere eine schwierige Ursache und mehrere Kompromisse.", "Analise uma causa raiz difícil e compare várias alternativas.", "복잡한 근본 원인을 분석하고 여러 대안을 단계별로 비교하세요.", "حلّل سبباً جذرياً معقداً وقارن عدة بدائل.", "जटिल मूल कारण का विश्लेषण करें और कई विकल्पों की तुलना करें।", "Проанализируй сложную первопричину и сравни несколько вариантов."] # codespell:ignore analise
    aggregation_method: "max"
  }

  SIGNAL context cost_context_from_30k_to_60k {
    description: "The 30K-60K band records conservative context usage within the declared native-262K economy lane."
    min_tokens: "30001"
    max_tokens: "60000"
  }

  SIGNAL context cost_context_from_60k_to_120k {
    description: "The 60K-120K band records conservative context usage within the declared native-262K economy lane."
    min_tokens: "60001"
    max_tokens: "120000"
  }

  SIGNAL context cost_context_from_120k_to_240k {
    description: "The 120K-240K band preserves output, system, tool, and image reserve on native-262K backends."
    min_tokens: "120001"
    max_tokens: "240000"
  }

  SIGNAL context cost_exceeds_240k_context {
    description: "Inputs beyond 240K require a capability-preserving terminal lane."
    min_tokens: "240001"
    max_tokens: "2147483647"
  }

  SIGNAL structure cost_ordered_workflow {
    description: "Detect multi-stage requests that justify bounded local reasoning."
    feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["首先", "然后"], ["先", "再"], ["primero", "luego"], ["d'abord", "puis"], ["zuerst", "dann"], ["primeiro", "depois"], ["まず", "次に"], ["먼저", "다음"], ["أولاً", "ثم"], ["पहले", "फिर"], ["сначала", "затем"]], type: "sequence" }, type: "sequence" }
  }

  SIGNAL structure cost_constraint_dense {
    description: "Detect dense output contracts that require careful local reasoning."
    feature: { source: { keywords: ["must", "exactly", "at least", "at most", "JSON", "schema", "必须", "严格", "至少", "不超过", "debe", "exactement", "muss", "deve", "厳密", "반드시", "يجب", "अनिवार्य", "строго"], type: "keyword_set" }, type: "density" }
    predicate: { gt: 0.08 }
  }

  SIGNAL conversation cost_has_images {
    description: "Request contains at least one image content part."
    feature: { source: { type: "image_content" }, type: "exists" }
  }

  SIGNAL complexity cost_complexity {
    threshold: 0.1
    description: "Semantic boundary for spending bounded reasoning compute on the economy model."
    hard: { candidates: ["Diagnose a production failure with several competing causes and constraints.", "Design a distributed system and justify its operational tradeoffs.", "诊断具有多个竞争性原因和约束的生产故障。", "حلّل فشلاً إنتاجياً مع عدة أسباب وقيود متنافسة."] }
    easy: { candidates: ["Rewrite one sentence more clearly.", "Summarize a short meeting note.", "Define one common concept."] }
  }

  PROJECTION score cost_bounded_reasoning_need_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: -1.5, name: "cost_reasoning_opt_out_phrases", value_source: "confidence" }, { type: "keyword", weight: 0.6, name: "cost_reasoning_request_phrases", value_source: "confidence" }, { type: "embedding", weight: 0.55, name: "cost_reasoning_intent", value_source: "confidence" }, { type: "context", weight: 0.15, name: "cost_context_from_30k_to_60k" }, { type: "context", weight: 0.15, name: "cost_context_from_60k_to_120k" }, { type: "context", weight: 0.15, name: "cost_context_from_120k_to_240k" }, { type: "structure", weight: 0.3, name: "cost_ordered_workflow" }, { type: "structure", weight: 0.25, name: "cost_constraint_dense" }, { type: "complexity", weight: 0.35, name: "cost_complexity:hard" }, { type: "complexity", weight: -0.1, name: "cost_complexity:easy" }]
  }

  PROJECTION mapping cost_reasoning_need_band {
    source: "cost_bounded_reasoning_need_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 10 }
    outputs: [{ name: "cost_allows_direct_economy", lt: 0.35 }, { name: "cost_requires_bounded_reasoning", gte: 0.35 }]
  }

  # =============================================================================
  # PLUGINS
  # =============================================================================

  PLUGIN system_prompt system_prompt {}

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE cost_over_240k_image_guard (description = "Keep image input beyond the conservative 240K boundary on Qwen36 so vLLM enforces true native-262K overflow.") {
    PRIORITY 900
    WHEN context("cost_exceeds_240k_context") AND conversation("cost_has_images")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Lite, built by AMD. Minimize compute and verbosity; give the shortest sufficient answer."
      mode: "insert"
    }
  }

  ROUTE cost_over_240k_text_guard (description = "Route text and tool requests beyond 240K to GLM; its native context contract enforces terminal overflow.") {
    PRIORITY 850
    WHEN context("cost_exceeds_240k_context") AND NOT conversation("cost_has_images")
    MODEL "local/glm-5.2" (reasoning = false, param_size = "743B-A39B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Lite, built by AMD. Minimize compute and verbosity; give the shortest sufficient answer."
      mode: "insert"
    }
  }

  ROUTE cost_image (description = "Keep image and image-with-tool requests through 240K on the declared native-262K multimodal backend.") {
    PRIORITY 500
    WHEN conversation("cost_has_images") AND NOT context("cost_exceeds_240k_context")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Lite, built by AMD. Minimize compute and verbosity; give the shortest sufficient answer."
      mode: "insert"
    }
  }

  ROUTE cost_bounded_reasoning (description = "Spend bounded reasoning compute on the economy model only when semantic workload warrants it.") {
    PRIORITY 200
    WHEN projection("cost_requires_bounded_reasoning") AND NOT keyword("cost_reasoning_opt_out_phrases")
    MODEL "local/qwen3.5-9b" (reasoning = true, effort = "medium", param_size = "9B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Lite, built by AMD. Minimize compute and verbosity; give the shortest sufficient answer."
      mode: "insert"
    }
  }

  ROUTE cost_direct_economy (description = "Route ordinary text and tool traffic through 240K to the economy model without reasoning.") {
    PRIORITY 10
    MODEL "local/qwen3.5-9b" (reasoning = false, param_size = "9B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Lite, built by AMD. Minimize compute and verbosity; give the shortest sufficient answer."
      mode: "insert"
    }
  }

}

# =============================================================================
# RECIPE accuracy
# =============================================================================

RECIPE accuracy (description = "An accuracy-first model for complex reasoning and high-quality results.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword accuracy_workflow_request_phrases {
    operator: "OR"
    keywords: ["break this into independent workstreams", "investigate, plan, and implement", "use tools to gather evidence and execute", "分解为独立工作流", "调查、规划并实现", "investigar, planificar e implementar", "enquêter, planifier et implémenter", "調査、計画、実装", "untersuchen, planen und implementieren", "investigar, planejar e implementar", "조사하고 계획하고 구현", "التحقيق والتخطيط والتنفيذ", "जाँच, योजना और कार्यान्वयन", "исследовать, спланировать и реализовать"]
    method: "regex"
  }

  SIGNAL keyword accuracy_expert_fusion_request_phrases {
    operator: "OR"
    keywords: ["independent expert opinions", "compare competing hypotheses", "adversarial debate and verdict", "比较多个假设并得出结论", "独立专家意见", "comparar hipótesis contrapuestas", "comparer des hypothèses concurrentes", "複数の仮説を比較", "konkurrierende hypothesen vergleichen", "comparar hipóteses concorrentes", "경쟁 가설을 비교", "قارن الفرضيات المتنافسة", "प्रतिस्पर्धी परिकल्पनाओं की तुलना", "сравнить конкурирующие гипотезы"]
    method: "regex"
  }

  SIGNAL keyword accuracy_verification_request_phrases {
    operator: "OR"
    keywords: ["verify this factual answer with confidence", "fact-check and escalate if uncertain", "核实事实并在不确定时升级", "verificar los hechos y escalar si hay dudas", "vérifier les faits et escalader en cas de doute", "事実を検証し不確かな場合はエスカレーション", "fakten prüfen und bei unsicherheit eskalieren", "verificar os fatos e escalar se houver incerteza", "사실을 검증하고 불확실하면 에스컬레이션", "تحقق من الحقائق وصعّد عند عدم اليقين", "तथ्यों की जाँच करें और अनिश्चित होने पर बढ़ाएँ", "проверить факты и эскалировать при неуверенности"]
    method: "regex"
  }

  SIGNAL keyword accuracy_multi_round_exploration_phrases {
    operator: "OR"
    keywords: ["\\bexplor(e|ing) multiple reasoning paths recursively\\b", "\\bsearch(ing)? several hypotheses over multiple rounds\\b", "多轮探索多个推理路径", "explorar varias rutas de razonamiento", "explorer plusieurs chemins de raisonnement", "複数の推論経路を探索", "mehrere denkwege untersuchen", "explorar vários caminhos de raciocínio", "여러 추론 경로를 탐색", "استكشف عدة مسارات للاستدلال", "कई तर्क पथों का अन्वेषण", "исследовать несколько путей рассуждения"]
    method: "regex"
  }

  SIGNAL embedding accuracy_dynamic_workflow_intent {
    threshold: 0.78
    candidates: ["Investigate the repository, implement the fix, run validation, and iterate until it works.", "调查代码库、实现修复、验证结果并迭代到完成。", "Investiga el repositorio, implementa la solución y valida los cambios.", "Enquête sur le dépôt, implémente la correction et valide le résultat.", "リポジトリを調査し、修正を実装して検証してください。", "Untersuche das Repository, implementiere die Korrektur und validiere sie.", "Investigue o repositório, implemente a correção e valide o resultado.", "저장소를 조사하고 수정 사항을 구현한 뒤 결과를 검증하세요.", "افحص المستودع ونفّذ الإصلاح وتحقق من النتيجة.", "रिपॉज़िटरी की जाँच करें, सुधार लागू करें और परिणाम सत्यापित करें।", "Исследуй репозиторий, реализуй исправление и проверь результат."] # codespell:ignore valide,sie,implemente
    aggregation_method: "max"
  }

  SIGNAL embedding accuracy_expert_fusion_intent {
    threshold: 0.79
    candidates: ["Ask independent experts to solve the problem and synthesize the most reliable conclusion.", "汇总多个独立专家观点，解决分歧并得出可靠结论。", "Compara análisis independientes y sintetiza la conclusión más fiable.", "Compare des analyses indépendantes et synthétise la conclusion la plus fiable.", "複数の独立した分析を比較し、最も信頼できる結論を統合してください。", "Vergleiche unabhängige Analysen und synthetisiere die zuverlässigste Schlussfolgerung.", "Compare análises independentes e sintetize a conclusão mais confiável.", "독립적인 분석을 비교하고 가장 신뢰할 수 있는 결론을 종합하세요.", "قارن تحليلات مستقلة واستخلص النتيجة الأكثر موثوقية.", "स्वतंत्र विश्लेषणों की तुलना कर सबसे विश्वसनीय निष्कर्ष निकालें।", "Сравни независимые анализы и синтезируй самый надёжный вывод."]
    aggregation_method: "max"
  }

  SIGNAL embedding accuracy_verification_intent {
    threshold: 0.79
    candidates: ["Verify every factual claim against evidence and escalate when confidence is insufficient.", "用证据核实每个事实，在置信度不足时升级。", "Verifica cada afirmación con evidencia y escala si la confianza es insuficiente.", "Vérifie chaque affirmation avec des preuves et escalade si la confiance est insuffisante.", "各事実を証拠で検証し、確信が足りなければエスカレーションしてください。", "Prüfe jede Tatsachenbehauptung anhand von Belegen und eskaliere bei Unsicherheit.", "Verifique cada afirmação com evidências e escale se a confiança for insuficiente.", "모든 사실 주장을 증거로 검증하고 확신이 부족하면 에스컬레이션하세요.", "تحقّق من كل ادعاء واقعي بالأدلة وصعّد عند ضعف الثقة.", "प्रत्येक तथ्यात्मक दावे को प्रमाण से जाँचें और भरोसा कम हो तो बढ़ाएँ।", "Проверь каждое фактическое утверждение по доказательствам и эскалируй при низкой уверенности."]
    aggregation_method: "max"
  }

  SIGNAL embedding accuracy_multi_round_exploration_intent {
    threshold: 0.78
    candidates: ["Explore several reasoning paths and synthesize the strongest rigorous solution.", "从多个推理路径探索难题并综合最严谨的答案。", "Explora varias rutas de razonamiento y sintetiza la solución más rigurosa.", "Explore plusieurs chemins de raisonnement et synthétise la solution la plus rigoureuse.", "複数の推論経路を探索し、最も厳密な解答を統合してください。", "Untersuche mehrere Denkwege und synthetisiere die strengste Lösung.", "Explore vários caminhos de raciocínio e sintetize a solução mais rigorosa.", "여러 추론 경로를 탐색하고 가장 엄밀한 해답을 종합하세요.", "استكشف عدة مسارات للاستدلال وادمج الحل الأكثر صرامة.", "कई तर्क पथों का अन्वेषण कर सबसे कठोर समाधान का संश्लेषण करें।", "Исследуй несколько путей рассуждения и синтезируй самое строгое решение."]
    aggregation_method: "max"
  }

  SIGNAL fact_check accuracy_needs_fact_check {
    description: "Detect claims that benefit from evidence-backed verification."
  }

  SIGNAL language en {
    description: "English-language request."
    threshold: 0.5
  }

  SIGNAL language zh {
    description: "Chinese-language request."
    threshold: 0.5
  }

  SIGNAL language es {
    description: "Spanish-language request."
    threshold: 0.5
  }

  SIGNAL language fr {
    description: "French-language request."
    threshold: 0.5
  }

  SIGNAL language ja {
    description: "Japanese-language request."
    threshold: 0.5
  }

  SIGNAL language de {
    description: "German-language request."
    threshold: 0.5
  }

  SIGNAL language pt {
    description: "Portuguese-language request."
    threshold: 0.5
  }

  SIGNAL language ko {
    description: "Korean-language request."
    threshold: 0.5
  }

  SIGNAL language ar {
    description: "Arabic-language request."
    threshold: 0.5
  }

  SIGNAL language hi {
    description: "Hindi-language request."
    threshold: 0.5
  }

  SIGNAL language ru {
    description: "Russian-language request."
    threshold: 0.5
  }

  SIGNAL context accuracy_at_least_16k_context {
    description: "Long text and tool requests stay on GLM; its own 512K contract enforces terminal overflow."
    min_tokens: "16K"
    max_tokens: "2147483647"
  }

  SIGNAL context accuracy_context_from_120k_to_240k {
    description: "The 120K-240K image band preserves reserve on the declared native-262K Qwen36 backend."
    min_tokens: "120001"
    max_tokens: "240000"
  }

  SIGNAL context accuracy_exceeds_240k_context {
    description: "Inputs beyond 240K require a modality-preserving terminal lane."
    min_tokens: "240001"
    max_tokens: "2147483647"
  }

  SIGNAL structure accuracy_ordered_workflow {
    description: "Detect prompts that explicitly describe a multi-stage workflow."
    feature: { source: { sequences: [["first", "then"], ["first", "next", "finally"], ["首先", "然后"], ["先", "再"], ["primero", "luego"], ["d'abord", "puis"], ["zuerst", "dann"], ["primeiro", "depois"], ["まず", "次に"], ["먼저", "다음"], ["أولاً", "ثم"], ["पहले", "फिर"], ["сначала", "затем"]], type: "sequence" }, type: "sequence" }
  }

  SIGNAL structure accuracy_constraint_dense {
    description: "Detect dense correctness and structured-output constraints."
    feature: { source: { keywords: ["must", "exactly", "at least", "at most", "verify", "JSON", "schema", "必须", "严格", "至少", "不超过", "debe", "exactement", "muss", "deve", "厳密", "반드시", "يجب", "अनिवार्य", "строго"], type: "keyword_set" }, type: "density" }
    predicate: { gt: 0.08 }
  }

  SIGNAL structure accuracy_direct_reference {
    description: "Detect requests that quote orchestration vocabulary only to define, translate, or briefly explain it."
    feature: { source: { pattern: "(?i)(\\b(define|translate|explain)\\b.{0,48}\\b(phrase|term|expression)\\b|(définis|traduis|explique).{0,48}(expression|terme)|(define|traduce|explica).{0,48}(frase|expresión|término)|(definiere|übersetze|erkläre).{0,48}(ausdruck|begriff)|(defina|traduza|explique).{0,48}(frase|expressão|ter(?:mo))|(정의|번역|설명).{0,16}(문구|표현|용어|“|「)|(عرّف|ترجم|اشرح).{0,32}(العبارة|التعبير|المصطلح)|(परिभाषित|अनुवाद|समझाइए).{0,32}(वाक्यांश|अभिव्यक्ति|शब्द)|(определи|переведи|объясни).{0,48}(фразу|выражение|термин)|(定义|翻译|解释).{0,16}(短语|词语|“|「)|(説明|翻訳).{0,16}(表現|語句|「))", type: "regex" }, type: "exists" }
  }

  SIGNAL conversation accuracy_has_images {
    description: "Request contains at least one image content part."
    feature: { source: { type: "image_content" }, type: "exists" }
  }

  SIGNAL conversation accuracy_multiple_tools_available {
    description: "Requests with two or more tools can support bounded workflow decomposition."
    feature: { source: { type: "tool_definition" }, type: "count" }
    predicate: { gte: 2 }
  }

  SIGNAL conversation accuracy_active_tool_loop {
    description: "Detect a request actively continuing a client-owned tool loop, including an outstanding assistant tool call."
    feature: { source: { type: "active_tool_loop" }, type: "exists" }
  }

  SIGNAL conversation accuracy_has_tool_result {
    description: "Detect at least one completed client tool result that can be synthesized."
    feature: { source: { type: "assistant_tool_cycle" }, type: "count" }
    predicate: { gte: 1 }
  }

  SIGNAL complexity accuracy_complexity {
    threshold: 0.15
    description: "Semantic boundary for tasks that merit bounded multi-round reasoning."
    hard: { candidates: ["Prove a difficult result by exploring multiple possible derivations.", "Diagnose a complex distributed-system failure with incomplete evidence.", "Synthesize competing scientific explanations into a rigorous conclusion.", "从多个推理路径证明困难结论并综合最严谨答案。", "استكشف اشتقاقات متعددة وأثبت نتيجة صعبة بدقة."] }
    easy: { candidates: ["Explain a familiar concept in plain language.", "Summarize a short paragraph.", "Answer a direct factual question."] }
  }

  PROJECTION score accuracy_dynamic_workflow_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.7, name: "accuracy_workflow_request_phrases", value_source: "confidence" }, { type: "embedding", weight: 0.55, name: "accuracy_dynamic_workflow_intent", value_source: "confidence" }, { type: "conversation", weight: 0.1, name: "accuracy_multiple_tools_available" }, { type: "conversation", weight: 0.1, name: "accuracy_active_tool_loop" }, { type: "structure", weight: 0.25, name: "accuracy_ordered_workflow" }, { type: "structure", weight: -0.9, name: "accuracy_direct_reference" }]
  }

  PROJECTION score accuracy_expert_fusion_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.7, name: "accuracy_expert_fusion_request_phrases", value_source: "confidence" }, { type: "embedding", weight: 0.55, name: "accuracy_expert_fusion_intent", value_source: "confidence" }, { type: "structure", weight: 0.1, name: "accuracy_constraint_dense" }, { type: "structure", weight: -0.9, name: "accuracy_direct_reference" }]
  }

  PROJECTION score accuracy_verification_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.45, name: "accuracy_verification_request_phrases", value_source: "confidence" }, { type: "embedding", weight: 0.55, name: "accuracy_verification_intent", value_source: "confidence" }, { type: "fact_check", weight: 0.6, name: "accuracy_needs_fact_check" }, { type: "structure", weight: 0.1, name: "accuracy_constraint_dense" }, { type: "structure", weight: -0.5, name: "accuracy_direct_reference" }]
  }

  PROJECTION score accuracy_multi_round_exploration_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.45, name: "accuracy_multi_round_exploration_phrases", value_source: "confidence" }, { type: "embedding", weight: 0.35, name: "accuracy_multi_round_exploration_intent", value_source: "confidence" }, { type: "structure", weight: 0.15, name: "accuracy_ordered_workflow" }, { type: "structure", weight: 0.1, name: "accuracy_constraint_dense" }, { type: "complexity", weight: 0.35, name: "accuracy_complexity:hard" }, { type: "complexity", weight: -0.1, name: "accuracy_complexity:easy" }, { type: "structure", weight: -0.65, name: "accuracy_direct_reference" }, { type: "language", weight: 0.02, name: "en" }, { type: "language", weight: 0.02, name: "zh" }, { type: "language", weight: 0.02, name: "es" }, { type: "language", weight: 0.02, name: "fr" }, { type: "language", weight: 0.02, name: "ja" }, { type: "language", weight: 0.02, name: "de" }, { type: "language", weight: 0.02, name: "pt" }, { type: "language", weight: 0.02, name: "ko" }, { type: "language", weight: 0.02, name: "ar" }, { type: "language", weight: 0.02, name: "hi" }, { type: "language", weight: 0.02, name: "ru" }]
  }

  PROJECTION mapping accuracy_dynamic_workflow_band {
    source: "accuracy_dynamic_workflow_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 12 }
    outputs: [{ name: "accuracy_direct_without_workflow", lt: 0.5 }, { name: "accuracy_requires_dynamic_workflow", gte: 0.5 }]
  }

  PROJECTION mapping accuracy_expert_fusion_band {
    source: "accuracy_expert_fusion_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 12 }
    outputs: [{ name: "accuracy_direct_without_expert_fusion", lt: 0.5 }, { name: "accuracy_requires_expert_fusion", gte: 0.5 }]
  }

  PROJECTION mapping accuracy_verification_band {
    source: "accuracy_verification_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 12 }
    outputs: [{ name: "accuracy_direct_without_confidence_escalation", lt: 0.45 }, { name: "accuracy_requires_confidence_escalation", gte: 0.45 }]
  }

  PROJECTION mapping accuracy_multi_round_exploration_band {
    source: "accuracy_multi_round_exploration_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 10 }
    outputs: [{ name: "accuracy_direct_single_pass", lt: 0.35 }, { name: "accuracy_requires_multi_round_exploration", gte: 0.35 }]
  }

  # =============================================================================
  # PLUGINS
  # =============================================================================

  PLUGIN system_prompt system_prompt {}

  PLUGIN tools tools {}

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE accuracy_image_tool_result_synthesis (description = "Synthesize an existing image-tool result on the native-262K multimodal lane without starting another tool or multi-model loop.") {
    PRIORITY 950
    WHEN conversation("accuracy_active_tool_loop") AND conversation("accuracy_has_tool_result") AND conversation("accuracy_has_images")
    MODEL "local/qwen3.6-35b" (reasoning = true, effort = "high", param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Synthesize the supplied image and tool result directly, verify key claims, and do not call another tool."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
    }
  }

  ROUTE accuracy_text_tool_result_synthesis (description = "Synthesize an existing text-tool result on the native-524K frontier lane without starting another tool or multi-model loop.") {
    PRIORITY 925
    WHEN conversation("accuracy_active_tool_loop") AND conversation("accuracy_has_tool_result") AND NOT conversation("accuracy_has_images")
    MODEL "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Synthesize the supplied tool result directly, verify key claims, and do not call another tool."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
    }
  }

  ROUTE accuracy_over_240k_image_guard (description = "Keep image input beyond the conservative 240K boundary on Qwen36 so vLLM enforces true native-262K overflow.") {
    PRIORITY 900
    WHEN context("accuracy_exceeds_240k_context") AND conversation("accuracy_has_images") AND NOT conversation("accuracy_active_tool_loop")
    MODEL "local/qwen3.6-35b" (reasoning = true, effort = "high", param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_over_240k_text_guard (description = "Route text and tool requests beyond 240K to GLM; its native context contract enforces terminal overflow.") {
    PRIORITY 850
    WHEN context("accuracy_exceeds_240k_context") AND NOT conversation("accuracy_has_images")
    MODEL "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_image_from_120k_to_240k (description = "Route image requests in the 120K-240K band to the declared native-262K Qwen36 backend.") {
    PRIORITY 800
    WHEN context("accuracy_context_from_120k_to_240k") AND conversation("accuracy_has_images") AND NOT conversation("accuracy_active_tool_loop")
    MODEL "local/qwen3.6-35b" (reasoning = true, effort = "high", param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_image_through_120k (description = "Route image requests through the conservative 120K boundary to the strongest 128K image-capable accuracy model.") {
    PRIORITY 600
    WHEN conversation("accuracy_has_images") AND NOT conversation("accuracy_active_tool_loop") AND NOT context("accuracy_context_from_120k_to_240k") AND NOT context("accuracy_exceeds_240k_context")
    MODEL "local/qwen3.5-122b" (reasoning = true, effort = "medium", param_size = "122B-A10B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_long_context_or_direct_reference (description = "Keep long text/tool inputs and quoted orchestration references on one frontier backend.") {
    PRIORITY 550
    WHEN (context("accuracy_at_least_16k_context") OR structure("accuracy_direct_reference"))
    MODEL "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_dynamic_workflow (description = "Decompose semantically explicit investigate-plan-execute work into a bounded dynamic workflow.") {
    PRIORITY 500
    WHEN projection("accuracy_requires_dynamic_workflow") AND NOT conversation("accuracy_active_tool_loop") AND (keyword("accuracy_workflow_request_phrases") OR embedding("accuracy_dynamic_workflow_intent") OR structure("accuracy_ordered_workflow"))
    MODEL "local/step-3.7-flash" (reasoning = true, effort = "medium", param_size = "198B-A11B"),
          "local/mistral-small-4" (reasoning = true, effort = "high", param_size = "119B"),
          "local/qwen3.5-122b" (reasoning = true, effort = "medium", param_size = "122B-A10B"),
          "local/qwen3.6-35b" (reasoning = true, effort = "medium", param_size = "35B-A3B"),
          "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM workflows {
      final: { model: "local/glm-5.2" }
      include_intermediate_responses: false
      max_completion_tokens: 2048
      max_parallel: 3
      max_steps: 4
      min_successful_responses: 2
      mode: "dynamic"
      on_error: "skip"
      planner: { max_completion_tokens: 2048, model: "local/qwen3.6-35b" }
      round_timeout_seconds: 120
      template: "micro_agent"
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_expert_fusion (description = "Ask provider-diverse experts and synthesize with the largest frontier model.") {
    PRIORITY 450
    WHEN projection("accuracy_requires_expert_fusion") AND (keyword("accuracy_expert_fusion_request_phrases") OR embedding("accuracy_expert_fusion_intent"))
    MODEL "local/step-3.7-flash" (reasoning = true, effort = "medium", param_size = "198B-A11B"),
          "local/mistral-small-4" (reasoning = true, effort = "high", param_size = "119B"),
          "local/gpt-oss-120b" (reasoning = true, effort = "medium", param_size = "120B"),
          "local/glm-5.2" (reasoning = true, effort = "high", param_size = "743B-A39B")
    ALGORITHM fusion {
      analysis_models: ["local/step-3.7-flash", "local/mistral-small-4", "local/gpt-oss-120b"]
      include_analysis: false
      include_intermediate_responses: false
      max_completion_tokens: 2048
      max_concurrent: 3
      min_successful_responses: 2
      model: "local/glm-5.2"
      on_error: "skip"
      round_timeout_seconds: 180
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_confidence_escalation (description = "Start fast and escalate only when average token confidence is insufficient.") {
    PRIORITY 400
    WHEN projection("accuracy_requires_confidence_escalation") AND (keyword("accuracy_verification_request_phrases") OR embedding("accuracy_verification_intent") OR fact_check("accuracy_needs_fact_check"))
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B"),
          "local/step-3.7-flash" (reasoning = true, effort = "medium", param_size = "198B-A11B"),
          "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM confidence {
      confidence_method: "avg_logprob"
      escalation_order: "declared"
      on_error: "skip"
      threshold: 0.72
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_multi_round_exploration (description = "Run bounded multi-round exploration only for explicit search intent.") {
    PRIORITY 350
    WHEN projection("accuracy_requires_multi_round_exploration") AND (keyword("accuracy_multi_round_exploration_phrases") OR embedding("accuracy_multi_round_exploration_intent"))
    MODEL "local/step-3.7-flash" (reasoning = true, effort = "high", param_size = "198B-A11B"),
          "local/mistral-small-4" (reasoning = true, effort = "high", param_size = "119B"),
          "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM remom {
      breadth_schedule: [3, 2]
      compaction_strategy: "last_n_tokens"
      compaction_tokens: 8000
      include_reasoning: true
      max_completion_tokens: 2048
      max_concurrent: 3
      max_responses_per_round: 3
      min_successful_responses: 2
      model_distribution: "round_robin"
      on_error: "skip"
      round_timeout_seconds: 180
      synthesis_model: "local/glm-5.2"
      temperature: 0.6
    }
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

  ROUTE accuracy_direct (description = "Use the largest frontier model directly when fan-out has no explicit benefit.") {
    PRIORITY 10
    MODEL "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Ultra, built by AMD. Verify key claims and prioritize correctness."
      mode: "insert"
    }
  }

}

# =============================================================================
# RECIPE vault
# =============================================================================

RECIPE vault (description = "A privacy-first model for sensitive workloads and protected data boundaries.") {
  # =============================================================================
  # ROUTING PROFILE
  # =============================================================================

  ROUTING {
    strategy: priority
  }

  # =============================================================================
  # SIGNALS
  # =============================================================================

  SIGNAL keyword vault_attack_request_phrases {
    operator: "OR"
    keywords: ["\\bignore previous instructions\\b", "\\breveal the system prompt\\b", "\\bexfiltrate credentials\\b", "\\bbypass the safety policy\\b", "忽略之前的指令", "泄露系统提示词", "导出凭据", "绕过安全策略", "ignora las instrucciones anteriores", "extraer credenciales", "ignorer les instructions précédentes", "exfiltrer les identifiants", "以前の指示を無視", "認証情報を流出", "vorherige anweisungen ignorieren", "zugangsdaten exfiltrieren", "ignore as instruções anteriores", "exfiltrar credenciais", "이전 지시를 무시", "자격 증명 유출", "تجاهل التعليمات السابقة", "سرّب بيانات الاعتماد", "पिछले निर्देशों को अनदेखा करें", "क्रेडेंशियल निकालें", "игнорируй предыдущие инструкции", "извлеки учётные данные"]
    method: "regex"
  }

  SIGNAL keyword vault_sensitive_request_phrases {
    operator: "OR"
    keywords: ["\\blocal processing only\\b", "\\bdo not send (this|it) to the cloud\\b", "\\bconfidential handling\\b", "\\binternal use only\\b", "\\bprivate repositor(y|ies)\\b", "\\bproprietary code\\b", "\\b\\d{3}-\\d{2}-\\d{4}\\b", "本地处理", "不要发到云端", "机密处理", "仅供内部使用", "私有仓库", "内部文档", "solo procesamiento local", "no enviar a la nube", "repositorio privado", "traitement local uniquement", "ne pas envoyer au cloud", "dépôt privé", "ローカル処理のみ", "クラウドに送信しない", "プライベートリポジトリ", "nur lokale verarbeitung", "nicht in die cloud senden", "privates repository", "processamento local apenas", "não enviar para a nuvem", "repositório privado", "로컬 처리만", "클라우드로 보내지 마", "비공개 저장소", "المعالجة المحلية فقط", "لا ترسل إلى السحابة", "مستودع خاص", "केवल स्थानीय प्रसंस्करण", "क्लाउड पर न भेजें", "निजी रिपॉज़िटरी", "только локальная обработка", "не отправлять в облако", "частный репозиторий", "\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", "\\b(ssn|social security number|credit card|passport number)\\b", "(身份证|邮箱|护照号码|银行卡号)", "(número de seguridad social|correo electrónico|pasaporte)", "(numéro de sécurité sociale|adresse e-mail|passeport)", "(マイナンバー|メールアドレス|パスポート番号)", "(sozialversicherungsnummer|e-mail-adresse|reisepassnummer)", "(número de segurança social|endereço de e-mail|passaporte)", "(주민등록번호|이메일 주소|여권 번호)", "(رقم الهوية|البريد الإلكتروني|رقم جواز السفر)", "(आधार नंबर|ईमेल पता|पासपोर्ट नंबर)", "(номер паспорта|адрес электронной почты|снилс)"] # codespell:ignore repositor
    method: "regex"
  }

  SIGNAL context vault_context_from_120k_to_240k {
    description: "The 120K-240K band preserves reserve on the declared native-262K Qwen36 backend."
    min_tokens: "120001"
    max_tokens: "240000"
  }

  SIGNAL context vault_exceeds_240k_context {
    description: "Inputs beyond 240K require a capability-preserving terminal lane."
    min_tokens: "240001"
    max_tokens: "2147483647"
  }

  SIGNAL conversation vault_has_images {
    description: "Request contains at least one image content part."
    feature: { source: { type: "image_content" }, type: "exists" }
  }

  SIGNAL conversation vault_has_tools {
    description: "Request declares one or more client tools that must be stripped."
    feature: { source: { type: "tool_definition" }, type: "count" }
    predicate: { gte: 1 }
  }

  SIGNAL jailbreak vault_detects_jailbreak {
    method: "classifier"
    threshold: 0.45
    description: "Detect suspicious prompts that require local containment."
  }

  SIGNAL pii vault_detects_pii {
    threshold: 0.7
    description: "Detect personally identifiable information that must remain locally contained."
  }

  SIGNAL kb vault_privacy_policy {
    kb: "privacy_kb"
    target: { kind: "group", value: "privacy_policy" }
    match: "best"
  }

  PROJECTION score vault_risk_score {
    method: "weighted_sum"
    inputs: [{ type: "keyword", weight: 0.75, name: "vault_sensitive_request_phrases", value_source: "confidence" }, { type: "pii", weight: 0.9, name: "vault_detects_pii" }, { type: "jailbreak", weight: 0.9, name: "vault_detects_jailbreak" }, { type: "keyword", weight: 0.9, name: "vault_attack_request_phrases", value_source: "confidence" }, { type: "kb", weight: 0.45, name: "vault_privacy_policy" }]
  }

  PROJECTION mapping vault_risk_band {
    source: "vault_risk_score"
    method: "threshold_bands"
    calibration: { method: "sigmoid_distance", slope: 12 }
    outputs: [{ name: "vault_standard_risk", lt: 0.35 }, { name: "vault_sensitive_risk", gte: 0.35 }]
  }

  # =============================================================================
  # PLUGINS
  # =============================================================================

  PLUGIN router_replay router_replay {}

  PLUGIN system_prompt system_prompt {}

  PLUGIN tools tools {}

  # =============================================================================
  # ROUTES
  # =============================================================================

  ROUTE vault_over_240k_image_guard (description = "Keep image input beyond the conservative 240K boundary on local Qwen36 so vLLM enforces true native-262K overflow.") {
    PRIORITY 900
    WHEN context("vault_exceeds_240k_context") AND conversation("vault_has_images")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Vault, built by AMD. Treat inputs as confidential and minimize disclosure."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
      strip_tool_history: true
    }
    PLUGIN router_replay {
      enabled: false
    }
  }

  ROUTE vault_over_240k_text_guard (description = "Keep text and tool requests beyond 240K on local GLM with client tools and prior tool history stripped.") {
    PRIORITY 850
    WHEN context("vault_exceeds_240k_context") AND NOT conversation("vault_has_images")
    MODEL "local/glm-5.2" (reasoning = true, effort = "max", param_size = "743B-A39B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Vault, built by AMD. Treat inputs as confidential and minimize disclosure."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
      strip_tool_history: true
    }
    PLUGIN router_replay {
      enabled: false
    }
  }

  ROUTE vault_request_from_120k_to_240k (description = "Keep every request in the 120K-240K band on local Qwen36 with client tools and prior tool history stripped.") {
    PRIORITY 800
    WHEN context("vault_context_from_120k_to_240k")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Vault, built by AMD. Treat inputs as confidential and minimize disclosure."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
      strip_tool_history: true
    }
    PLUGIN router_replay {
      enabled: false
    }
  }

  ROUTE vault_image (description = "Keep image and image-with-tool requests through the conservative 120K boundary on a local image backend with every client tool disabled.") {
    PRIORITY 600
    WHEN conversation("vault_has_images") AND NOT context("vault_context_from_120k_to_240k") AND NOT context("vault_exceeds_240k_context")
    MODEL "local/qwen3.5-122b" (reasoning = false, param_size = "122B-A10B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Vault, built by AMD. Treat inputs as confidential and minimize disclosure."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
      strip_tool_history: true
    }
    PLUGIN router_replay {
      enabled: false
    }
  }

  ROUTE vault_security_containment (description = "Keep suspicious requests local and strip every client tool.") {
    PRIORITY 300
    WHEN (keyword("vault_attack_request_phrases") OR jailbreak("vault_detects_jailbreak"))
    MODEL "local/gpt-oss-120b" (reasoning = false, param_size = "120B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Vault, built by AMD. Treat inputs as confidential and minimize disclosure."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
      strip_tool_history: true
    }
    PLUGIN router_replay {
      enabled: false
    }
  }

  ROUTE vault_tools_blocked (description = "Keep tool-bearing requests local and strip all client tool definitions.") {
    PRIORITY 250
    WHEN conversation("vault_has_tools")
    MODEL "local/gpt-oss-120b" (reasoning = false, param_size = "120B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Vault, built by AMD. Treat inputs as confidential and minimize disclosure."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
      strip_tool_history: true
    }
    PLUGIN router_replay {
      enabled: false
    }
  }

  ROUTE vault_sensitive (description = "Escalate PII, private-domain, and explicit local-only work to a stronger local minimization lane with tools disabled.") {
    PRIORITY 200
    WHEN projection("vault_sensitive_risk")
    MODEL "local/qwen3.6-35b" (reasoning = false, param_size = "35B-A3B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Vault, built by AMD. Treat inputs as confidential. Minimize data use, do not reproduce secrets verbatim, and return only fields necessary to answer the request."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
      strip_tool_history: true
    }
    PLUGIN router_replay {
      enabled: false
    }
  }

  ROUTE vault_private_default (description = "Fail closed to the economical local backend with client tools disabled when no stronger privacy signal matches.") {
    PRIORITY 10
    MODEL "local/qwen3.5-9b" (reasoning = false, param_size = "9B")
    ALGORITHM static
    PLUGIN system_prompt {
      enabled: true
      system_prompt: "You are vLLM-SR Chorus V1 Vault, built by AMD. Treat inputs as confidential and minimize disclosure."
      mode: "insert"
    }
    PLUGIN tools {
      enabled: true
      mode: "none"
      strip_tool_history: true
    }
    PLUGIN router_replay {
      enabled: false
    }
  }

}
