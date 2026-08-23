// topology/utils/topologyParser.ts - Config to Topology Parser

import type {
  AlgorithmConfig,
  DecisionConfig,
  GlobalPluginConfig,
  ManagedTopologyConfig,
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
import { extractSignals } from './topologySignalParser'

/**
 * Parse raw config data into structured topology data
 */
export function parseConfigToTopology(config: ManagedTopologyConfig): ParsedTopology {
  const globalPlugins: GlobalPluginConfig[] = []
  const signals = extractSignals(config)
  const decisions = extractDecisions(config)
  const models = extractModels(config)
  const strategy = config.document.strategy === 'confidence' ? 'confidence' : 'priority'

  return { globalPlugins, signals, decisions, models, strategy }
}

/**
 * Extract decisions from config
 */
function extractDecisions(config: ManagedTopologyConfig): DecisionConfig[] {
  const decisions: DecisionConfig[] = []
  const routingDecisions = config.document.decisions

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
        const modelConfig = config.models.find((model) => model.name === ref.model)
        return {
          model: ref.model,
          use_reasoning: ref.use_reasoning,
          reasoning_effort: ref.reasoning_effort,
          lora_name: ref.lora_name,
          reasoning_family: modelConfig?.card.reasoning?.type,
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
  algorithm:
    | NonNullable<ManagedTopologyConfig['document']['decisions']>[number]['algorithm']
    | undefined,
): AlgorithmConfig | undefined {
  if (!algorithm) {
    return undefined
  }

  return {
    type: algorithm.type as AlgorithmConfig['type'],
    confidence: algorithm.confidence,
    concurrent: algorithm.concurrent,
    latency_aware: algorithm.latency_aware,
    ratings: algorithm.ratings,
    remom: algorithm.remom,
    fusion: algorithm.fusion,
    workflows: algorithm.workflows,
    router_dc: algorithm.router_dc,
    automix: algorithm.automix,
    autoMix: algorithm.autoMix ?? algorithm.automix,
    hybrid: algorithm.hybrid,
    knn: algorithm.knn,
    kmeans: algorithm.kmeans,
    svm: algorithm.svm,
    mlp: algorithm.mlp,
    multi_factor: algorithm.multi_factor,
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
function extractModels(config: ManagedTopologyConfig): ModelConfig[] {
  return config.models.map((model) => ({
    name: model.name,
    reasoning_family: model.card.reasoning?.type,
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
    conversation: [],
    event: [],
    projection: [],
  }

  signals.forEach((signal) => {
    if (groups[signal.type]) {
      groups[signal.type].push(signal)
    }
  })

  return groups
}
