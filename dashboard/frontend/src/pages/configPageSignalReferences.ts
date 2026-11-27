import { SIGNAL_CATALOG } from './configPageSignalCatalog'
import type {
  ConfigData,
  DecisionConfig,
  RecipeRoutingConfig,
  RoutingConfig,
  SignalType,
} from './configPageSupport'

const SIGNAL_CONFIG_TYPES = Object.fromEntries(
  SIGNAL_CATALOG.map(({ label, type }) => [label, type]),
) as Record<SignalType, string>

type Routing = RecipeRoutingConfig | RoutingConfig | undefined

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function sameSignal(record: Record<string, unknown>, type: string, name: string): boolean {
  if (record.type !== type || typeof record.name !== 'string') return false
  return record.name === name || record.name.startsWith(`${name}:`)
}

function countReferences(value: unknown, type: string, name: string): number {
  if (Array.isArray(value)) {
    return value.reduce((total, item) => total + countReferences(item, type, name), 0)
  }
  if (!isRecord(value)) return 0
  const ownMatch = sameSignal(value, type, name) ? 1 : 0
  return (
    ownMatch +
    Object.values(value).reduce<number>(
      (total, item) => total + countReferences(item, type, name),
      0,
    )
  )
}

export interface SignalUsage {
  /** Decisions whose rules read the signal. */
  decisions: DecisionConfig[]
  /** Decisions whose response_guard plugin reads it. */
  guards: DecisionConfig[]
  /** Decisions whose masking plugin masks what it finds. */
  masking: DecisionConfig[]
  /** Projection scores with the signal as an input, and partitions it is a member of. */
  projections: Array<{ kind: 'score' | 'partition'; name: string; weight?: number }>
  total: number
}

function pluginReads(
  decision: DecisionConfig,
  pluginType: string,
  type: string,
  name: string,
): boolean {
  return (decision.plugins ?? []).some((plugin) => {
    if (plugin.type !== pluginType) return false
    const configuration = plugin.configuration
    if (pluginType === 'masking' && type === 'pii' && configuration?.signal === name) return true
    return countReferences(configuration, type, name) > 0
  })
}

/** Everything in one routing profile that reads a signal, by canonical type and name. */
export function signalUsageInRoutingProfile(
  routing: Routing,
  type: string,
  name: string,
): SignalUsage {
  const decisions = routing?.decisions ?? []
  const ruleReaders = decisions.filter(
    (decision) => countReferences(decision.rules, type, name) > 0,
  )
  const guards = decisions.filter((decision) => pluginReads(decision, 'response_guard', type, name))
  const masking = decisions.filter((decision) => pluginReads(decision, 'masking', type, name))
  const projections: SignalUsage['projections'] = []
  for (const score of routing?.projections?.scores ?? []) {
    const input = (score.inputs ?? []).find((candidate) =>
      sameSignal(candidate as unknown as Record<string, unknown>, type, name),
    )
    if (input) projections.push({ kind: 'score', name: score.name, weight: input.weight })
  }
  if (type === 'domain') {
    for (const partition of routing?.projections?.partitions ?? []) {
      if ((partition.members ?? []).includes(name))
        projections.push({ kind: 'partition', name: partition.name })
    }
  }
  return {
    decisions: ruleReaders,
    guards,
    masking,
    projections,
    total: ruleReaders.length + guards.length + masking.length + projections.length,
  }
}

export function getSignalReferenceCountInRoutingProfile(
  routing: Routing,
  signalType: SignalType,
  signalName: string,
): number {
  if (!routing) return 0
  const type = SIGNAL_CONFIG_TYPES[signalType]
  return (
    countReferences(routing.decisions, type, signalName) +
    countReferences(routing.projections?.scores, type, signalName)
  )
}

export function getSignalReferenceCount(
  config: ConfigData | null,
  signalType: SignalType,
  signalName: string,
): number {
  if (!config) return 0

  return (
    getSignalReferenceCountInRoutingProfile(config.routing, signalType, signalName) +
    (config.recipes ?? []).reduce(
      (total, recipe) =>
        total + getSignalReferenceCountInRoutingProfile(recipe.routing, signalType, signalName),
      0,
    )
  )
}
