import type { Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'

import styles from './ConfigPage.module.css'
import metaStyles from './ConfigPageMetaRoutingSection.module.css'
import type { OpenEditModal } from './configPageRouterSectionSupport'
import type {
  ConfigData,
  MetaRefinementAction,
  MetaRequiredSignalFamily,
  MetaRoutingConfig,
  MetaSignalFamilyDisagreement,
  MetaTriggerPolicy,
} from './configPageSupport'

export interface ConfigPageMetaRoutingSectionProps {
  config: ConfigData | null
  isReadonly: boolean
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
}

export interface RequiredFamilyRow extends MetaRequiredSignalFamily {
  id: string
}

export interface FamilyDisagreementRow extends MetaSignalFamilyDisagreement {
  id: string
}

export interface AllowedActionRow extends MetaRefinementAction {
  id: string
}

export interface RequiredFamilyFormState {
  type: string
  min_confidence?: number
  min_matches?: number
}

export interface FamilyDisagreementFormState {
  cheap: string
  expensive: string
}

export interface AllowedActionFormState {
  type: MetaRefinementAction['type']
  signal_families: string[]
}

export const META_MODE_OPTIONS: MetaRoutingConfig['mode'][] = ['observe', 'shadow', 'active']
export const META_ACTION_OPTIONS: MetaRefinementAction['type'][] = ['disable_compression', 'rerun_signal_families']
export const META_SIGNAL_FAMILY_OPTIONS = [
  'keyword',
  'embedding',
  'domain',
  'fact_check',
  'user_feedback',
  'preference',
  'language',
  'context',
  'structure',
  'complexity',
  'modality',
  'authz',
  'jailbreak',
  'pii',
] as const

export const buildDefaultMetaRoutingConfig = (): MetaRoutingConfig => ({
  mode: 'observe',
  max_passes: 2,
  trigger_policy: {
    partition_conflict: true,
    required_families: [],
    family_disagreements: [],
  },
  allowed_actions: [],
})

export const cloneMetaRoutingConfig = (value?: MetaRoutingConfig): MetaRoutingConfig =>
  value
    ? JSON.parse(JSON.stringify(value)) as MetaRoutingConfig
    : buildDefaultMetaRoutingConfig()

export const asOptionalNumber = (value: unknown): number | undefined => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return undefined
}

export const normalizeTriggerPolicy = (value?: MetaTriggerPolicy): MetaTriggerPolicy | undefined => {
  if (!value) {
    return undefined
  }

  const normalized: MetaTriggerPolicy = {}
  const decisionMarginBelow = asOptionalNumber(value.decision_margin_below)
  if (decisionMarginBelow !== undefined) {
    normalized.decision_margin_below = decisionMarginBelow
  }
  const projectionBoundaryWithin = asOptionalNumber(value.projection_boundary_within)
  if (projectionBoundaryWithin !== undefined) {
    normalized.projection_boundary_within = projectionBoundaryWithin
  }
  if (value.partition_conflict === true) {
    normalized.partition_conflict = true
  }

  const requiredFamilies = (value.required_families || [])
    .map((family) => ({
      type: family.type.trim(),
      min_confidence: asOptionalNumber(family.min_confidence),
      min_matches: asOptionalNumber(family.min_matches),
    }))
    .filter((family) => family.type)
  if (requiredFamilies.length > 0) {
    normalized.required_families = requiredFamilies
  }

  const familyDisagreements = (value.family_disagreements || [])
    .map((family) => ({
      cheap: family.cheap.trim(),
      expensive: family.expensive.trim(),
    }))
    .filter((family) => family.cheap && family.expensive)
  if (familyDisagreements.length > 0) {
    normalized.family_disagreements = familyDisagreements
  }

  return Object.keys(normalized).length > 0 ? normalized : undefined
}

export const normalizeAllowedActions = (value?: MetaRefinementAction[]): MetaRefinementAction[] | undefined => {
  const actions = (value || [])
    .map((action) => ({
      type: action.type,
      signal_families: action.type === 'rerun_signal_families'
        ? (action.signal_families || [])
          .map((family) => family.trim())
          .filter(Boolean)
        : undefined,
    }))
    .filter((action) =>
      action.type === 'disable_compression' ||
      (action.type === 'rerun_signal_families' && (action.signal_families?.length || 0) > 0)
    )
  return actions.length > 0 ? actions : undefined
}

export const normalizeMetaRoutingConfig = (draft: MetaRoutingConfig): MetaRoutingConfig => {
  const normalized: MetaRoutingConfig = {
    mode: draft.mode,
  }

  const maxPasses = asOptionalNumber(draft.max_passes)
  if (maxPasses !== undefined) {
    normalized.max_passes = Math.max(1, Math.round(maxPasses))
  }

  const triggerPolicy = normalizeTriggerPolicy(draft.trigger_policy)
  if (triggerPolicy) {
    normalized.trigger_policy = triggerPolicy
  }

  const allowedActions = normalizeAllowedActions(draft.allowed_actions)
  if (allowedActions) {
    normalized.allowed_actions = allowedActions
  }

  return normalized
}

export const metaRoutingConfigFingerprint = (value: MetaRoutingConfig | null) =>
  JSON.stringify(value ?? null)

export const buildConfiguredFamilyOptions = (config: ConfigData | null) => {
  const configured = new Set<string>()
  const signalConfig = config?.signals
  if (signalConfig?.keywords?.length) configured.add('keyword')
  if (signalConfig?.embeddings?.length) configured.add('embedding')
  if (signalConfig?.domains?.length) configured.add('domain')
  if (signalConfig?.fact_check?.length) configured.add('fact_check')
  if (signalConfig?.user_feedbacks?.length) configured.add('user_feedback')
  if (signalConfig?.preferences?.length) configured.add('preference')
  if (signalConfig?.language?.length) configured.add('language')
  if (signalConfig?.context?.length) configured.add('context')
  if (signalConfig?.structure?.length) configured.add('structure')
  if (signalConfig?.complexity?.length) configured.add('complexity')
  if (signalConfig?.modality?.length) configured.add('modality')
  if (signalConfig?.role_bindings?.length) configured.add('authz')
  if (signalConfig?.jailbreak?.length) configured.add('jailbreak')
  if (signalConfig?.pii?.length) configured.add('pii')
  for (const family of META_SIGNAL_FAMILY_OPTIONS) {
    configured.add(family)
  }
  return Array.from(configured).sort()
}

export const buildRequiredFamilyFields = (
  configuredFamilyOptions: string[],
): FieldConfig<RequiredFamilyFormState>[] => [
  {
    name: 'type',
    label: 'Signal family',
    type: 'select',
    required: true,
    options: configuredFamilyOptions,
  },
  {
    name: 'min_confidence',
    label: 'Minimum confidence',
    type: 'number',
    min: 0,
    max: 1,
    step: 0.01,
    placeholder: '0.65',
  },
  {
    name: 'min_matches',
    label: 'Minimum matches',
    type: 'number',
    min: 1,
    step: 1,
    placeholder: '1',
  },
]

export const buildDisagreementFields = (
  configuredFamilyOptions: string[],
): FieldConfig<FamilyDisagreementFormState>[] => [
  {
    name: 'cheap',
    label: 'Cheap family',
    type: 'select',
    required: true,
    options: configuredFamilyOptions,
  },
  {
    name: 'expensive',
    label: 'Expensive family',
    type: 'select',
    required: true,
    options: configuredFamilyOptions,
  },
]

export const buildActionFields = (
  configuredFamilyOptions: string[],
): FieldConfig<AllowedActionFormState>[] => [
  {
    name: 'type',
    label: 'Action type',
    type: 'select',
    required: true,
    options: META_ACTION_OPTIONS,
  },
  {
    name: 'signal_families',
    label: 'Signal families',
    type: 'multiselect',
    options: configuredFamilyOptions,
    shouldHide: (data) => data.type !== 'rerun_signal_families',
    description: 'Families that can be selectively rerun when the meta assessment plans a refinement.',
  },
]

export const createRequiredFamilyColumns = (): Column<RequiredFamilyRow>[] => [
  {
    key: 'type',
    header: 'Signal Family',
    sortable: true,
    render: (row) => <span className={`${styles.tableMetaBadge} ${styles.tableMetaBadgeMono}`}>{row.type}</span>,
  },
  {
    key: 'min_confidence',
    header: 'Min Confidence',
    width: '160px',
    render: (row) => row.min_confidence !== undefined ? row.min_confidence.toFixed(2) : 'Optional',
  },
  {
    key: 'min_matches',
    header: 'Min Matches',
    width: '160px',
    render: (row) => row.min_matches !== undefined ? row.min_matches : 'Optional',
  },
]

export const createDisagreementColumns = (): Column<FamilyDisagreementRow>[] => [
  {
    key: 'cheap',
    header: 'Cheap Family',
    render: (row) => <span className={`${styles.tableMetaBadge} ${styles.tableMetaBadgeMono}`}>{row.cheap}</span>,
  },
  {
    key: 'expensive',
    header: 'Expensive Family',
    render: (row) => <span className={`${styles.tableMetaBadge} ${styles.tableMetaBadgeMono}`}>{row.expensive}</span>,
  },
]

export const createAllowedActionColumns = (): Column<AllowedActionRow>[] => [
  {
    key: 'type',
    header: 'Action',
    sortable: true,
    render: (row) => <span className={`${styles.tableMetaBadge} ${styles.tableMetaBadgeMono}`}>{row.type}</span>,
  },
  {
    key: 'signal_families',
    header: 'Signal Families',
    render: (row) => row.signal_families?.length
      ? (
        <div className={metaStyles.pillRow}>
          {row.signal_families.map((family) => (
            <span key={`${row.id}-${family}`} className={metaStyles.pill}>{family}</span>
          ))}
        </div>
      )
      : <span className={metaStyles.helpText}>Applies without signal family scoping.</span>,
  },
]
