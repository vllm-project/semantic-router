// topology/utils/topologyParser.ts - Config to Topology Parser

import {
  ConfigData,
  ParsedTopology,
  SignalConfig,
  DecisionConfig,
  GlobalPluginConfig,
  ModelConfig,
  SignalType,
  RuleCombination,
  AlgorithmConfig,
  PluginConfig,
  ModelRefConfig,
} from '../types'
import { SIGNAL_LATENCY } from '../constants'

/**
 * Parse raw config data into structured topology data
 */
export function parseConfigToTopology(config: ConfigData): ParsedTopology {
  const globalPlugins = extractGlobalPlugins(config)
  const signals = extractSignals(config)
  const decisions = extractDecisions(config)
  const models = extractModels(config)
  const strategy = 'priority' // Default strategy
  const defaultModel = config.providers?.default_model

  return { globalPlugins, signals, decisions, models, strategy, defaultModel }
}

/**
 * Extract global plugins (Jailbreak, PII, Cache)
 */
function extractGlobalPlugins(config: ConfigData): GlobalPluginConfig[] {
  const plugins: GlobalPluginConfig[] = []

  // 1. Prompt Guard (Jailbreak Detection)
  if (config.prompt_guard) {
    plugins.push({
      type: 'prompt_guard',
      enabled: config.prompt_guard.enabled ?? false,
      modelId: config.prompt_guard.model_id || 'vLLM-SR-Jailbreak',
      threshold: config.prompt_guard.threshold,
      config: {
        use_modernbert: config.prompt_guard.use_modernbert,
        use_vllm: config.prompt_guard.use_vllm,
      },
    })
  }

  // 2. PII Detection
  // Note: Global PII only loads the model. Actual detection requires decision-level pii plugin.
  if (config.classifier?.pii_model) {
    plugins.push({
      type: 'pii_detection',
      enabled: !!config.classifier.pii_model.model_id,
      modelId: config.classifier.pii_model.model_id || 'vLLM-SR-PII',
      threshold: config.classifier.pii_model.threshold,
      config: {
        // Mark as "model loaded" not "active detection"
        mode: 'model_loaded',
        description: 'Model loaded. Enable per-decision via pii plugin.',
      },
    })
  }

  // 3. Semantic Cache (Global)
  if (config.semantic_cache) {
    plugins.push({
      type: 'semantic_cache',
      enabled: config.semantic_cache.enabled ?? false,
      config: {
        backend_type: config.semantic_cache.backend_type,
        similarity_threshold: config.semantic_cache.similarity_threshold,
        ttl_seconds: config.semantic_cache.ttl_seconds,
      },
    })
  }

  return plugins
}

/**
 * Extract all signal definitions from config
 * Supports both Go format (keyword_rules, embedding_rules, etc.)
 * and Python CLI format (signals.keywords, signals.embeddings, etc.)
 */
function extractSignals(config: ConfigData): SignalConfig[] {
  const signals: SignalConfig[] = []
  const addedSignals = new Set<string>() // Track added signals to avoid duplicates

  // Helper to add signal if not already added
  const addSignal = (signal: SignalConfig) => {
    const key = `${signal.type}:${signal.name}`
    if (!addedSignals.has(key)) {
      addedSignals.add(key)
      signals.push(signal)
    }
  }

  // 1. Keyword Rules → keyword signals
  // From keyword_rules (Go/Router format)
  config.keyword_rules?.forEach(rule => {
    addSignal({
      type: 'keyword',
      name: rule.name,
      latency: SIGNAL_LATENCY.keyword,
      config: {
        operator: rule.operator,
        keywords: rule.keywords,
        case_sensitive: rule.case_sensitive ?? false,
      },
    })
  })
  // From signals.keywords (Python CLI format)
  config.signals?.keywords?.forEach(rule => {
    addSignal({
      type: 'keyword',
      name: rule.name,
      latency: SIGNAL_LATENCY.keyword,
      config: {
        operator: rule.operator,
        keywords: rule.keywords,
        case_sensitive: rule.case_sensitive ?? false,
      },
    })
  })

  // 2. Embedding Rules → embedding signals
  // From embedding_rules (Go/Router format)
  config.embedding_rules?.forEach(rule => {
    addSignal({
      type: 'embedding',
      name: rule.name,
      latency: SIGNAL_LATENCY.embedding,
      config: {
        threshold: rule.threshold,
        candidates: rule.candidates,
        aggregation_method: rule.aggregation_method || 'max',
      },
    })
  })
  // From signals.embeddings (Python CLI format)
  config.signals?.embeddings?.forEach(rule => {
    addSignal({
      type: 'embedding',
      name: rule.name,
      latency: SIGNAL_LATENCY.embedding,
      config: {
        threshold: rule.threshold,
        candidates: rule.candidates,
        aggregation_method: rule.aggregation_method || 'max',
      },
    })
  })

  // 3. Categories/Domains → domain signals
  // From signals.domains (Python CLI format)
  config.signals?.domains?.forEach(domain => {
    addSignal({
      type: 'domain',
      name: domain.name,
      description: domain.description,
      latency: SIGNAL_LATENCY.domain,
      config: {
        mmlu_categories: domain.mmlu_categories,
      },
    })
  })
  // From categories (Go/Router format)
  config.categories?.forEach(cat => {
    // Only add if it has mmlu_categories (domain signal)
    if (cat.mmlu_categories) {
      addSignal({
        type: 'domain',
        name: cat.name,
        description: cat.description,
        latency: SIGNAL_LATENCY.domain,
        config: {
          mmlu_categories: cat.mmlu_categories,
        },
      })
    }
  })

  // 4. Fact Check Rules
  // From fact_check_rules (Go/Router format)
  config.fact_check_rules?.forEach(rule => {
    addSignal({
      type: 'fact_check',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.fact_check,
      config: {},
    })
  })
  // From signals.fact_check (Python CLI format)
  config.signals?.fact_check?.forEach(rule => {
    addSignal({
      type: 'fact_check',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.fact_check,
      config: {},
    })
  })

  // 5. User Feedback Rules
  // From user_feedback_rules (Go/Router format)
  config.user_feedback_rules?.forEach(rule => {
    addSignal({
      type: 'user_feedback',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.user_feedback,
      config: {},
    })
  })
  // From signals.user_feedbacks (Python CLI format)
  config.signals?.user_feedbacks?.forEach(rule => {
    addSignal({
      type: 'user_feedback',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.user_feedback,
      config: {},
    })
  })

  // 6. Preference Rules
  // From preference_rules (Go/Router format)
  config.preference_rules?.forEach(rule => {
    addSignal({
      type: 'preference',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.preference,
      config: {},
    })
  })
  // From signals.preferences (Python CLI format)
  config.signals?.preferences?.forEach(rule => {
    addSignal({
      type: 'preference',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.preference,
      config: {},
    })
  })

  // 7. Language Rules
  // From language_rules (Go/Router format)
  config.language_rules?.forEach(rule => {
    addSignal({
      type: 'language',
      name: rule.name,
      latency: SIGNAL_LATENCY.language,
      config: {},
    })
  })
  // From signals.language (Python CLI format)
  config.signals?.language?.forEach(rule => {
    addSignal({
      type: 'language',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.language,
      config: {},
    })
  })

  // 8. Latency Rules
  // From latency_rules (Go/Router format)
  config.latency_rules?.forEach(rule => {
    addSignal({
      type: 'latency',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.latency,
      config: {
        tpot_percentile: rule.tpot_percentile,
        ttft_percentile: rule.ttft_percentile,
      },
    })
  })
  // From signals.latency (Python CLI format)
  config.signals?.latency?.forEach(rule => {
    addSignal({
      type: 'latency',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.latency,
      config: {
        tpot_percentile: rule.tpot_percentile,
        ttft_percentile: rule.ttft_percentile,
      },
    })
  })

  // 9. Context Rules
  // From context_rules (Go/Router format)
  config.context_rules?.forEach(rule => {
    addSignal({
      type: 'context',
      name: rule.name,
      latency: SIGNAL_LATENCY.context,
      config: {
        min_tokens: rule.min_tokens,
        max_tokens: rule.max_tokens,
      },
    })
  })
  // From signals.context (Python CLI format)
  config.signals?.context?.forEach(rule => {
    addSignal({
      type: 'context',
      name: rule.name,
      latency: SIGNAL_LATENCY.context,
      config: {
        min_tokens: rule.min_tokens,
        max_tokens: rule.max_tokens,
      },
    })
  })

  // 10. Complexity Rules
  // From complexity_rules (Go/Router format)
  config.complexity_rules?.forEach(rule => {
    addSignal({
      type: 'complexity',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.complexity,
      config: {
        threshold: rule.threshold,
        hard: rule.hard,
        easy: rule.easy,
      },
    })
  })
  // From signals.complexity (Python CLI format)
  config.signals?.complexity?.forEach(rule => {
    addSignal({
      type: 'complexity',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.complexity,
      config: {
        threshold: rule.threshold,
        hard: rule.hard,
        easy: rule.easy,
      },
    })
  })

  return signals
}

/**
 * Extract decisions from config
 */
function extractDecisions(config: ConfigData): DecisionConfig[] {
  const decisions: DecisionConfig[] = []

  // Python CLI format: decisions array
  if (config.decisions && config.decisions.length > 0) {
    config.decisions.forEach(decision => {
      const rules: RuleCombination = {
        operator: (decision.rules?.operator as 'AND' | 'OR') || 'AND',
        conditions: (decision.rules?.conditions || []).map(cond => ({
          type: cond.type as SignalType,
          name: cond.name,
        })),
      }

      const algorithm: AlgorithmConfig | undefined = decision.algorithm
        ? {
            type: decision.algorithm.type as AlgorithmConfig['type'],
            confidence: decision.algorithm.confidence,
            concurrent: decision.algorithm.concurrent,
          }
        : undefined

      const plugins: PluginConfig[] = (decision.plugins || []).map(p => ({
        type: p.type as PluginConfig['type'],
        enabled: p.enabled ?? true,
        configuration: p.configuration,
      }))

      // Find reasoning_family from providers.models
      const modelRefs: ModelRefConfig[] = (decision.modelRefs || []).map(ref => {
        const modelConfig = config.providers?.models?.find(m => m.name === ref.model)
        return {
          model: ref.model,
          use_reasoning: ref.use_reasoning,
          reasoning_effort: ref.reasoning_effort,
          lora_name: ref.lora_name,
          reasoning_family: modelConfig?.reasoning_family,
        }
      })

      decisions.push({
        name: decision.name,
        description: decision.description,
        priority: decision.priority || 0,
        rules,
        modelRefs,
        algorithm,
        plugins,
      })
    })
  }
  // Legacy format: categories array
  else if (config.categories && config.categories.length > 0) {
    config.categories.forEach((cat, index) => {
      const modelScores = normalizeModelScores(cat.model_scores)
      const modelRefs: ModelRefConfig[] = modelScores.map(ms => {
        const modelConfig = config.model_config?.[ms.model]
        return {
          model: ms.model,
          use_reasoning: ms.use_reasoning,
          reasoning_family: modelConfig?.reasoning_family,
        }
      })

      // Create implicit domain rule for category
      const rules: RuleCombination = {
        operator: 'OR',
        conditions: [
          {
            type: 'domain',
            name: cat.name,
          },
        ],
      }

      decisions.push({
        name: cat.name,
        description: cat.description,
        priority: index + 1,
        rules,
        modelRefs,
      })
    })
  }

  // Sort by priority (descending)
  return decisions.sort((a, b) => b.priority - a.priority)
}

/**
 * Extract models from config
 */
function extractModels(config: ConfigData): ModelConfig[] {
  const models: ModelConfig[] = []

  // From providers.models
  config.providers?.models?.forEach(model => {
    models.push({
      name: model.name,
      reasoning_family: model.reasoning_family,
    })
  })

  // From model_config (Legacy)
  if (config.model_config) {
    Object.entries(config.model_config).forEach(([name, cfg]) => {
      if (!models.find(m => m.name === name)) {
        models.push({
          name,
          reasoning_family: cfg.reasoning_family,
        })
      }
    })
  }

  return models
}

/**
 * Normalize model_scores from object to array (Legacy format uses object)
 */
interface NormalizedModelScore {
  model: string
  score: number
  use_reasoning?: boolean
}

function normalizeModelScores(
  modelScores: Array<{ model: string; score: number; use_reasoning?: boolean }> | Record<string, number> | undefined
): NormalizedModelScore[] {
  if (!modelScores) return []
  if (Array.isArray(modelScores)) return modelScores
  // Object format (Legacy) - convert to array
  return Object.entries(modelScores).map(([model, score]) => ({
    model,
    score: typeof score === 'number' ? score : 0,
    use_reasoning: false,
  }))
}

/**
 * Group signals by type
 */
export function groupSignalsByType(signals: SignalConfig[]): Record<SignalType, SignalConfig[]> {
  const groups: Record<SignalType, SignalConfig[]> = {
    keyword: [],
    embedding: [],
    domain: [],
    fact_check: [],
    user_feedback: [],
    preference: [],
    language: [],
    latency: [],
    context: [],
    complexity: [],
  }

  signals.forEach(signal => {
    if (groups[signal.type]) {
      groups[signal.type].push(signal)
    }
  })

  return groups
}

/**
 * Check if config is Python CLI format
 */
export function isPythonCLIFormat(config: ConfigData): boolean {
  return !!(config.decisions && config.decisions.length > 0)
}
