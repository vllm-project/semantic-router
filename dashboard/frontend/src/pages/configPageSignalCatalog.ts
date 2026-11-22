import type { LucideIcon } from 'lucide-react'
import {
  Ban,
  Blend,
  Bot,
  BrainCircuit,
  FileImage,
  Fingerprint,
  Gauge,
  Ghost,
  Globe,
  Image,
  KeyRound,
  Languages,
  MessageSquareHeart,
  MessagesSquare,
  Radar,
  Repeat2,
  Ruler,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  Tags,
  TextSearch,
} from 'lucide-react'

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
  /** Decisions can reference it in their rules; response-only signals are read by response guards. */
  decisionReferenceable: boolean
  /** Rule conditions on it may name one of the labels it reports. */
  labelled: boolean
}

export const SIGNAL_CATALOG: SignalCatalogEntry[] = ROUTER_CONFIG_EXTENSION.signals.map(
  (surface) => ({
    type: surface.type,
    label: surface.display_name,
    collection: surface.collection as keyof ConfigSignals,
    schemaRef: surface.schema_ref,
    decisionReferenceable: surface.decision_referenceable,
    labelled: 'reference_qualifier' in surface && surface.reference_qualifier === 'label',
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

export const findSignalCatalogEntry = (type: string): SignalCatalogEntry | undefined =>
  SIGNAL_CATALOG.find((surface) => surface.type === type)

const SIGNAL_ICONS: Record<CanonicalSignalType, LucideIcon> = {
  keyword: TextSearch,
  domain: Globe,
  fact_check: ScanSearch,
  user_feedback: MessageSquareHeart,
  language: Languages,
  context: Ruler,
  complexity: Gauge,
  modality: Image,
  authz: KeyRound,
  jailbreak: ShieldAlert,
  safety: ShieldCheck,
  hallucination: Ghost,
  refusal: Ban,
  turn_relation: Repeat2,
  pii: Fingerprint,
  conversation: MessagesSquare,
  metadata: Tags,
  input_modality: FileImage,
  agent: Bot,
  systemone: BrainCircuit,
}

/** Icon for a signal type, or a projection reference, in rules, tables and chips. */
export function signalTypeIcon(type: string): LucideIcon {
  if (type === 'projection') return Blend
  return SIGNAL_ICONS[type as CanonicalSignalType] ?? Radar
}

/** Display name for a signal type, or a projection reference. */
export function signalTypeLabel(type: string): string {
  if (type === 'projection') return 'Projection'
  return findSignalCatalogEntry(type)?.label ?? type
}
