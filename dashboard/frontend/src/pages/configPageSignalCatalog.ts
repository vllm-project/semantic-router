import {
  ROUTER_CONFIG_EXTENSION,
  type SignalType as CanonicalSignalType,
} from '../generated/routerConfigContract'
import type { ConfigSignals, SignalType } from './configPageSupport'

export interface SignalCatalogEntry {
  type: CanonicalSignalType
  label: SignalType
  collection: keyof ConfigSignals
  schemaRef: string
  referenceSuffixes: readonly string[]
}

export const SIGNAL_CATALOG: SignalCatalogEntry[] = ROUTER_CONFIG_EXTENSION.signals.map(
  (surface) => ({
    type: surface.type,
    label: surface.display_name,
    collection: surface.collection as keyof ConfigSignals,
    schemaRef: surface.schema_ref,
    referenceSuffixes: 'reference_suffixes' in surface ? (surface.reference_suffixes ?? []) : [],
  }),
)

export const signalCatalogByType = (type: CanonicalSignalType): SignalCatalogEntry => {
  const entry = SIGNAL_CATALOG.find((surface) => surface.type === type)
  if (!entry) throw new Error(`Unsupported signal type: ${type}`)
  return entry
}

export const signalCatalogByLabel = (label: SignalType): SignalCatalogEntry => {
  const entry = SIGNAL_CATALOG.find((surface) => surface.label === label)
  if (!entry) throw new Error(`Unsupported signal label: ${label}`)
  return entry
}
