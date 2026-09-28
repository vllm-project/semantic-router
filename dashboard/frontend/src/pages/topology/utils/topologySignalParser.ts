// topology/utils/topologySignalParser.ts - Signal extraction helpers

import { ROUTER_CONFIG_EXTENSION } from '../../../generated/routerConfigContract'
import type { ConfigData, SignalConfig, SignalType } from '../types'
import { SIGNAL_LATENCY } from '../constants'

interface RawSignalRecord extends Record<string, unknown> {
  name: string
  description?: string
}

interface LegacySignalSource {
  type: Exclude<SignalType, 'projection'>
  value: unknown
}

function signalRecords(value: unknown): RawSignalRecord[] {
  if (!Array.isArray(value)) return []
  return value.filter(
    (candidate): candidate is RawSignalRecord =>
      Boolean(candidate) &&
      typeof candidate === 'object' &&
      typeof (candidate as { name?: unknown }).name === 'string',
  )
}

function signalFromRecord(
  type: Exclude<SignalType, 'projection'>,
  record: RawSignalRecord,
): SignalConfig {
  const { name, description, ...fields } = record
  return {
    type,
    name,
    description: typeof description === 'string' ? description : undefined,
    latency: SIGNAL_LATENCY[type],
    config: fields,
  }
}

function legacySignalSources(config: ConfigData): LegacySignalSource[] {
  return [
    { type: 'keyword', value: config.keyword_rules },
    { type: 'embedding', value: config.embedding_rules },
    { type: 'domain', value: config.categories },
    { type: 'fact_check', value: config.fact_check_rules },
    { type: 'user_feedback', value: config.user_feedback_rules },
    { type: 'reask', value: config.reask_rules },
    { type: 'preference', value: config.preference_rules },
    { type: 'language', value: config.language_rules },
    { type: 'context', value: config.context_rules },
    { type: 'structure', value: config.structure_rules },
    { type: 'complexity', value: config.complexity_rules },
    { type: 'modality', value: config.modality_rules },
    { type: 'authz', value: config.role_bindings },
    { type: 'jailbreak', value: config.jailbreak },
    { type: 'hallucination', value: config.hallucination },
    { type: 'pii', value: config.pii },
    { type: 'kb', value: config.kb },
    { type: 'conversation', value: config.conversation },
    { type: 'event', value: config.events },
  ]
}

/**
 * Extract every canonical signal through the generated Router catalog. Legacy
 * root collections remain an explicit compatibility seam and never define the
 * canonical inventory.
 */
export function extractSignals(config: ConfigData): SignalConfig[] {
  const signals: SignalConfig[] = []
  const addedSignals = new Set<string>()
  const addSignal = (signal: SignalConfig) => {
    const key = `${signal.type}:${signal.name}`
    if (addedSignals.has(key)) return
    addedSignals.add(key)
    signals.push(signal)
  }

  const routingSignals = config.routing?.signals ?? config.signals
  const canonicalCollections = (routingSignals ?? {}) as Record<string, unknown>
  ROUTER_CONFIG_EXTENSION.signals.forEach((surface) => {
    signalRecords(canonicalCollections[surface.collection]).forEach((record) =>
      addSignal(signalFromRecord(surface.type, record)),
    )
  })

  legacySignalSources(config).forEach((source) => {
    signalRecords(source.value).forEach((record) =>
      addSignal(signalFromRecord(source.type, record)),
    )
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
