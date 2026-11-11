// topology/utils/topologyParser.ts - Config to Topology Parser

import type {
  AlgorithmConfig,
  ConfigData,
  DecisionConfig,
  GlobalPluginConfig,
  ModelConfig,
  ModelRefConfig,
  ParsedTopology,
  PluginConfig,
  RawRuleCombination,
  RawRuleNode,
  RuleCombination,
  RuleNode,
  SignalConfig,
  SignalType,
} from '../types'
import { SIGNAL_TYPES } from '../constants'
import { extractSignals } from './topologySignalParser'

/**
 * Parse raw config data into structured topology data
 */
export function parseConfigToTopology(config: ConfigData): ParsedTopology {
  const globalPlugins = extractGlobalPlugins(config)
  const signals = extractSignals(config)
  const decisions = extractDecisions(config)
  const models = extractModels(config)
  const strategy = config.routing?.strategy || config.global?.router?.strategy || 'priority'
  const defaultModel = config.providers?.defaults?.model

  return { globalPlugins, signals, decisions, models, strategy, defaultModel }
}

/**
 * Extract global plugins (Cache)
 */
function extractGlobalPlugins(config: ConfigData): GlobalPluginConfig[] {
  const plugins: GlobalPluginConfig[] = []
  const responseCache = config.global?.stores?.response_cache

  // Semantic Cache (Global)
  if (responseCache) {
    plugins.push({
      type: 'response_cache',
      enabled: responseCache.enabled ?? false,
      config: {
        backend_type: responseCache.backend_type,
        similarity_threshold: responseCache.similarity_threshold,
        ttl_seconds: responseCache.ttl_seconds,
      },
    })
  }

  return plugins
}

/**
 * Extract decisions from config
 */
function extractDecisions(config: ConfigData): DecisionConfig[] {
  const decisions: DecisionConfig[] = []
  const routingDecisions = config.routing?.decisions ?? config.decisions

  if (routingDecisions && routingDecisions.length > 0) {
    routingDecisions.forEach((decision) => {
      const rules = parseRuleCombination(decision.rules)
      const algorithm = extractDecisionAlgorithm(decision.algorithm)

      const plugins: PluginConfig[] = (decision.plugins || []).map((p) => ({
        type: p.type as PluginConfig['type'],
        enabled: p.enabled ?? true,
        configuration: p.configuration,
      }))

      const modelRefs: ModelRefConfig[] = (decision.modelRefs || []).map((ref) => {
        const modelConfig = config.providers?.models?.find((m) => m.name === ref.model)
        return {
          model: ref.model,
          use_reasoning: ref.use_reasoning,
          reasoning_mode: ref.reasoning_mode,
          reasoning_effort: ref.reasoning_effort,
          lora_name: ref.lora_name,
          reasoning_family: modelConfig?.reasoning?.family,
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

  // Sort by priority (descending)
  return decisions.sort((a, b) => b.priority - a.priority)
}

function extractDecisionAlgorithm(
  algorithm: NonNullable<ConfigData['decisions']>[number]['algorithm'] | undefined,
): AlgorithmConfig | undefined {
  if (!algorithm) {
    return undefined
  }

  return {
    type: algorithm.type as AlgorithmConfig['type'],
    minimum_candidates: algorithm.minimum_candidates,
    budget: algorithm.budget,
    on_error: algorithm.on_error,
    multi_factor: algorithm.multi_factor,
    systemone: algorithm.systemone,
    escalate: algorithm.escalate,
    fusion: algorithm.fusion,
    remom: algorithm.remom,
    ratings: algorithm.ratings,
    workflows: algorithm.workflows,
  }
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
  // A single condition can be the whole rule tree: `rules: {type, name}`.
  const rootCondition = rules && !rules.operator && !rules.conditions ? parseRuleNode(rules) : null
  if (rootCondition) {
    return { operator: 'AND', conditions: [rootCondition] }
  }

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
  config.providers?.models?.forEach((model) => {
    models.push({
      name: model.name,
      reasoning_family: model.reasoning?.family,
    })
  })

  return models
}

/**
 * Group signals by type
 */
export function groupSignalsByType(signals: SignalConfig[]): Record<SignalType, SignalConfig[]> {
  const groups = Object.fromEntries(
    SIGNAL_TYPES.map((signalType) => [signalType, [] as SignalConfig[]]),
  ) as Record<SignalType, SignalConfig[]>

  signals.forEach((signal) => {
    if (groups[signal.type]) {
      groups[signal.type].push(signal)
    }
  })

  return groups
}
