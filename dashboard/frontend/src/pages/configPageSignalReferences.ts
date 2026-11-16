import { SIGNAL_CATALOG } from './configPageSignalCatalog'
import type {
  ConfigData,
  RecipeRoutingConfig,
  RoutingConfig,
  SignalType,
} from './configPageSupport'

const SIGNAL_CONFIG_TYPES = Object.fromEntries(
  SIGNAL_CATALOG.map(({ label, type }) => [label, type]),
) as Record<SignalType, string>

function countReferences(value: unknown, type: string, name: string): number {
  if (Array.isArray(value)) {
    return value.reduce((total, item) => total + countReferences(item, type, name), 0)
  }
  if (!value || typeof value !== 'object') return 0

  const record = value as Record<string, unknown>
  const ownMatch = record.type === type && record.name === name ? 1 : 0
  return (
    ownMatch +
    Object.values(record).reduce<number>(
      (total, item) => total + countReferences(item, type, name),
      0,
    )
  )
}

export function getSignalReferenceCountInRoutingProfile(
  routing: RecipeRoutingConfig | RoutingConfig | undefined,
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
