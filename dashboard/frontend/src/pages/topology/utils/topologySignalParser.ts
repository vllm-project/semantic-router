// topology/utils/topologySignalParser.ts - Signal extraction helpers

import type { KBSignalConfig, ManagedTopologyConfig, SignalConfig } from '../types'
import { SIGNAL_LATENCY } from '../constants'

type AddSignal = (signal: SignalConfig) => void
type SignalBuilder<T> = (rule: T) => SignalConfig | null

function addRuleSignals<T>(
  addSignal: AddSignal,
  rules: readonly T[] | undefined,
  buildSignal: SignalBuilder<T>,
) {
  rules?.forEach((rule) => {
    const signal = buildSignal(rule)
    if (signal) {
      addSignal(signal)
    }
  })
}

/**
 * Extract signal definitions from one native Recipe document.
 */
export function extractSignals(config: ManagedTopologyConfig): SignalConfig[] {
  const signals: SignalConfig[] = []
  const addedSignals = new Set<string>()
  const routingSignals = config.document.signals

  const addSignal: AddSignal = (signal) => {
    const key = `${signal.type}:${signal.name}`
    if (!addedSignals.has(key)) {
      addedSignals.add(key)
      signals.push(signal)
    }
  }

  addRuleSignals(addSignal, routingSignals?.keywords, (rule) => ({
    type: 'keyword',
    name: rule.name,
    latency: SIGNAL_LATENCY.keyword,
    config: {
      operator: rule.operator,
      keywords: rule.keywords,
      case_sensitive: rule.case_sensitive ?? false,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.embeddings, (rule) => ({
    type: 'embedding',
    name: rule.name,
    latency: SIGNAL_LATENCY.embedding,
    config: {
      threshold: rule.threshold,
      candidates: rule.candidates,
      aggregation_method: rule.aggregation_method || 'max',
    },
  }))

  addRuleSignals(addSignal, routingSignals?.domains, (domain) => ({
    type: 'domain',
    name: domain.name,
    description: domain.description,
    latency: SIGNAL_LATENCY.domain,
    config: {
      mmlu_categories: domain.mmlu_categories,
    },
  }))
  addRuleSignals(addSignal, routingSignals?.fact_check, (rule) => ({
    type: 'fact_check',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.fact_check,
    config: {},
  }))

  addRuleSignals(addSignal, routingSignals?.user_feedbacks, (rule) => ({
    type: 'user_feedback',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.user_feedback,
    config: {},
  }))

  addRuleSignals(addSignal, routingSignals?.reasks, (rule) => ({
    type: 'reask',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.reask,
    config: {
      threshold: rule.threshold,
      lookback_turns: rule.lookback_turns,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.preferences, (rule) => ({
    type: 'preference',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.preference,
    config: {
      examples: rule.examples,
      threshold: rule.threshold,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.language, (rule) => ({
    type: 'language',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.language,
    config: {},
  }))

  addRuleSignals(addSignal, routingSignals?.context, (rule) => ({
    type: 'context',
    name: rule.name,
    latency: SIGNAL_LATENCY.context,
    config: {
      min_tokens: rule.min_tokens,
      max_tokens: rule.max_tokens,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.structure, (rule) => ({
    type: 'structure',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.structure,
    config: {
      feature: rule.feature,
      predicate: rule.predicate,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.complexity, (rule) => ({
    type: 'complexity',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.complexity,
    config: {
      threshold: rule.threshold,
      hard: rule.hard,
      easy: rule.easy,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.modality, (rule) => ({
    type: 'modality',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.modality,
    config: {},
  }))

  addRuleSignals(addSignal, routingSignals?.role_bindings, (rule) => ({
    type: 'authz',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.authz,
    config: {
      role: rule.role,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.jailbreak, (rule) => ({
    type: 'jailbreak',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.jailbreak,
    config: {
      threshold: rule.threshold,
      include_history: rule.include_history,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.pii, (rule) => ({
    type: 'pii',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.pii,
    config: {
      threshold: rule.threshold,
      pii_types_allowed: rule.pii_types_allowed,
      include_history: rule.include_history,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.kb, (rule) => ({
    type: 'kb',
    name: rule.name,
    description: rule.description || `KB bind ${rule.kb} ${rule.target.kind}=${rule.target.value}`,
    latency: SIGNAL_LATENCY.kb,
    config: {
      kb: rule.kb,
      target: rule.target,
      match: rule.match,
    } satisfies KBSignalConfig,
  }))

  addRuleSignals(addSignal, routingSignals?.conversation, (rule) => ({
    type: 'conversation',
    name: rule.name,
    description: rule.description,
    latency: SIGNAL_LATENCY.conversation,
    config: {
      feature: rule.feature,
      predicate: rule.predicate,
    },
  }))

  addRuleSignals(addSignal, routingSignals?.events, (rule) => ({
    type: 'event',
    name: rule.name,
    latency: SIGNAL_LATENCY.event,
    config: {
      event_types: rule.event_types,
      severities: rule.severities,
      action_codes: rule.action_codes,
      temporal: rule.temporal,
    },
  }))

  extractProjectionSignals(config).forEach(addSignal)

  return signals
}

function extractProjectionSignals(config: ManagedTopologyConfig): SignalConfig[] {
  const projectionSignals: SignalConfig[] = []
  const projections = config.document.projections
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

  projections?.mappings?.forEach((mapping) => {
    mapping.outputs?.forEach((output) => {
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
