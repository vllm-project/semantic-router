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
  RuleNode,
  RawRuleNode,
  RawRuleCombination,
  AlgorithmConfig,
  PluginConfig,
  ModelRefConfig,
  KBSignalConfig,
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
  const strategy = config.global?.router?.strategy || 'priority'
  const defaultModel = config.providers?.defaults?.default_model

  return { globalPlugins, signals, decisions, models, strategy, defaultModel }
}

/**
 * Extract global plugins (Jailbreak, PII, Cache)
 */
function extractGlobalPlugins(config: ConfigData): GlobalPluginConfig[] {
  const plugins: GlobalPluginConfig[] = []
  const promptGuard = config.global?.model_catalog?.modules?.prompt_guard || config.prompt_guard
  const piiModel = config.global?.model_catalog?.modules?.classifier?.pii || config.classifier?.pii_model
  const semanticCache = config.global?.stores?.semantic_cache || config.semantic_cache
  const promptGuardModel = promptGuard?.model_id || promptGuard?.model_ref
  const piiModelRef = piiModel?.model_id || piiModel?.model_ref

  // 1. Prompt Guard (Jailbreak Detection)
  if (promptGuard) {
    plugins.push({
      type: 'prompt_guard',
      enabled: promptGuard.enabled ?? !!promptGuardModel,
      modelId: promptGuardModel || 'vLLM-SR-Jailbreak',
      threshold: promptGuard.threshold,
      config: {
        use_modernbert: promptGuard.use_modernbert,
        use_vllm: promptGuard.use_vllm,
      },
    })
  }

  // 2. PII Detection
  // Note: Global PII only loads the model. Actual detection requires decision-level pii plugin.
  if (piiModel) {
    plugins.push({
      type: 'pii_detection',
      enabled: piiModel.enabled ?? !!piiModelRef,
      modelId: piiModelRef || 'vLLM-SR-PII',
      threshold: piiModel.threshold,
      config: {
        // Mark as "model loaded" not "active detection"
        mode: 'model_loaded',
        description: 'Model loaded. Enable per-decision via pii plugin.',
      },
    })
  }

  // 3. Semantic Cache (Global)
  if (semanticCache) {
    plugins.push({
      type: 'semantic_cache',
      enabled: semanticCache.enabled ?? false,
      config: {
        backend_type: semanticCache.backend_type,
        similarity_threshold: semanticCache.similarity_threshold,
        ttl_seconds: semanticCache.ttl_seconds,
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
  const routingSignals = config.routing?.signals ?? config.signals

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
  routingSignals?.keywords?.forEach(rule => {
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
  routingSignals?.embeddings?.forEach(rule => {
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
  routingSignals?.domains?.forEach(domain => {
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
  routingSignals?.fact_check?.forEach(rule => {
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
  routingSignals?.user_feedbacks?.forEach(rule => {
    addSignal({
      type: 'user_feedback',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.user_feedback,
      config: {},
    })
  })

  // 6. Reask Rules
  // From reask_rules (Go/Router format)
  config.reask_rules?.forEach(rule => {
    addSignal({
      type: 'reask',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.reask,
      config: {
        threshold: rule.threshold,
        lookback_turns: rule.lookback_turns,
      },
    })
  })
  // From signals.reasks (Python CLI format)
  routingSignals?.reasks?.forEach(rule => {
    addSignal({
      type: 'reask',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.reask,
      config: {
        threshold: rule.threshold,
        lookback_turns: rule.lookback_turns,
      },
    })
  })

  // 7. Preference Rules
  // From preference_rules (Go/Router format)
  config.preference_rules?.forEach(rule => {
    addSignal({
      type: 'preference',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.preference,
      config: {
        examples: rule.examples,
        threshold: rule.threshold,
      },
    })
  })
  // From signals.preferences (Python CLI format)
  routingSignals?.preferences?.forEach(rule => {
    addSignal({
      type: 'preference',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.preference,
      config: {
        examples: rule.examples,
        threshold: rule.threshold,
      },
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
  routingSignals?.language?.forEach(rule => {
    addSignal({
      type: 'language',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.language,
      config: {},
    })
  })

  // 8. Context Rules
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
  routingSignals?.context?.forEach(rule => {
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

  // 9. Structure Rules
  // From structure_rules (Go/Router format)
  config.structure_rules?.forEach(rule => {
    addSignal({
      type: 'structure',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.structure,
      config: {
        feature: rule.feature,
        predicate: rule.predicate,
      },
    })
  })
  // From signals.structure (Python CLI format)
  routingSignals?.structure?.forEach(rule => {
    addSignal({
      type: 'structure',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.structure,
      config: {
        feature: rule.feature,
        predicate: rule.predicate,
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
  routingSignals?.complexity?.forEach(rule => {
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

  // 11. Modality Rules
  // From modality_rules (Go/Router format)
  config.modality_rules?.forEach(rule => {
    addSignal({
      type: 'modality',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.modality,
      config: {},
    })
  })
  // From signals.modality (Python CLI format)
  routingSignals?.modality?.forEach(rule => {
    addSignal({
      type: 'modality',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.modality,
      config: {},
    })
  })

  // 12. Session Rules
  routingSignals?.session?.forEach(rule => {
    addSignal({
      type: 'session',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.session,
      config: {
        fact: rule.fact,
        predicate: rule.predicate,
        intent_or_domain: rule.intent_or_domain,
        previous_model: rule.previous_model,
        candidate_model: rule.candidate_model,
      },
    })
  })

  // 13. Authz / RBAC Role Bindings
  // From role_bindings (Go/Router format)
  config.role_bindings?.forEach(rule => {
    addSignal({
      type: 'authz',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.authz,
      config: {
        role: rule.role,
      },
    })
  })
  // From signals.role_bindings (Python CLI format)
  routingSignals?.role_bindings?.forEach(rule => {
    addSignal({
      type: 'authz',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.authz,
      config: {
        role: rule.role,
      },
    })
  })

  // 13. Jailbreak Rules
  // From jailbreak (Go/Router format - top-level due to yaml:",inline")
  config.jailbreak?.forEach(rule => {
    addSignal({
      type: 'jailbreak',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.jailbreak,
      config: {
        threshold: rule.threshold,
        include_history: rule.include_history,
      },
    })
  })
  // From signals.jailbreak (Python CLI format)
  routingSignals?.jailbreak?.forEach(rule => {
    addSignal({
      type: 'jailbreak',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.jailbreak,
      config: {
        threshold: rule.threshold,
        include_history: rule.include_history,
      },
    })
  })

  // 14. PII Rules
  // From pii (Go/Router format - top-level due to yaml:",inline")
  config.pii?.forEach(rule => {
    addSignal({
      type: 'pii',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.pii,
      config: {
        threshold: rule.threshold,
        pii_types_allowed: rule.pii_types_allowed,
        include_history: rule.include_history,
      },
    })
  })
  // From signals.pii (Python CLI format)
  routingSignals?.pii?.forEach(rule => {
    addSignal({
      type: 'pii',
      name: rule.name,
      description: rule.description,
      latency: SIGNAL_LATENCY.pii,
      config: {
        threshold: rule.threshold,
        pii_types_allowed: rule.pii_types_allowed,
        include_history: rule.include_history,
      },
    })
  })

  // 15. Knowledge-base Rules
  routingSignals?.kb?.forEach(rule => {
    addSignal({
      type: 'kb',
      name: rule.name,
      description: rule.description || `KB bind ${rule.kb} ${rule.target.kind}=${rule.target.value}`,
      latency: SIGNAL_LATENCY.kb,
      config: {
        kb: rule.kb,
        target: rule.target,
        match: rule.match,
      } satisfies KBSignalConfig,
    })
  })

  extractProjectionSignals(config).forEach(addSignal)

  return signals
}

function extractProjectionSignals(config: ConfigData): SignalConfig[] {
  const projectionSignals: SignalConfig[] = []
  const projections = config.routing?.projections ?? config.projections
  const scoreInputsByName = new Map(
    (projections?.scores ?? []).map((score) => [
      score.name,
      (score.inputs ?? [])
        .filter((input): input is NonNullable<typeof input> => Boolean(input?.type && input?.name))
        .map((input) => ({
          type: input.type,
          name: input.name,
        })),
    ]),
  )

  projections?.mappings?.forEach(mapping => {
    mapping.outputs?.forEach(output => {
      projectionSignals.push({
        type: 'projection',
        name: output.name,
        description: `Projection output from ${mapping.name}`,
        latency: SIGNAL_LATENCY.projection,
        config: {
          source: mapping.source,
          method: mapping.method || 'threshold_bands',
          mapping: mapping.name,
          upstreamSignals: scoreInputsByName.get(mapping.source) ?? [],
        },
      })
    })
  })

  return projectionSignals
}

/**
 * Extract decisions from config
 */
function extractDecisions(config: ConfigData): DecisionConfig[] {
  const decisions: DecisionConfig[] = []
  const routingDecisions = config.routing?.decisions ?? config.decisions

  // Python CLI format: decisions array
  if (routingDecisions && routingDecisions.length > 0) {
    routingDecisions.forEach(decision => {
      const rules = parseRuleCombination(decision.rules)

      const algorithm: AlgorithmConfig | undefined = decision.algorithm
        ? {
          type: decision.algorithm.type as AlgorithmConfig['type'],
          confidence: decision.algorithm.confidence,
          concurrent: decision.algorithm.concurrent,
          latency_aware: decision.algorithm.latency_aware,
          session_aware: decision.algorithm.session_aware,
        }
        : undefined

      const plugins: PluginConfig[] = (decision.plugins || []).map(p => ({
        type: p.type as PluginConfig['type'],
        enabled: p.enabled ?? true,
        configuration: p.configuration,
      }))

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

function normalizeRuleOperator(operator?: string): RuleCombination['operator'] {
  if (operator === 'OR' || operator === 'NOT') {
    return operator
  }

  return 'AND'
}

function parseRuleNode(node: RawRuleNode): RuleNode | null {
  if (Array.isArray(node.conditions)) {
    return {
      operator: normalizeRuleOperator(node.operator),
      conditions: node.conditions
        .map((condition) => parseRuleNode(condition))
        .filter((condition): condition is RuleNode => condition !== null),
    }
  }

  if (node.type && node.name) {
    return {
      type: node.type as SignalType,
      name: node.name,
    }
  }

  return null
}

function parseRuleCombination(rules?: RawRuleCombination): RuleCombination {
  const conditions = (rules?.conditions || [])
    .map((condition: RawRuleNode) => parseRuleNode(condition))
    .filter((condition: RuleNode | null): condition is RuleNode => condition !== null)

  return {
    operator: normalizeRuleOperator(rules?.operator),
    conditions,
  }
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

  for (const card of config.routing?.modelCards || []) {
    if (!models.find((model) => model.name === card.name)) {
      models.push({
        name: card.name,
        reasoning_family: undefined,
      })
    }
  }

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
    reask: [],
    preference: [],
    language: [],
    context: [],
    structure: [],
    complexity: [],
    modality: [],
    authz: [],
    jailbreak: [],
    pii: [],
    kb: [],
    projection: [],
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
