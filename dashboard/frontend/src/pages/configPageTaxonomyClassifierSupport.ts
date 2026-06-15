export interface TaxonomySignalBinding {
  kind: 'label' | 'group'
  value: string
}

export interface TaxonomySignalReference {
  name: string
  target: TaxonomySignalBinding
  match?: 'best' | 'threshold'
}

export interface TaxonomyClassifierTier {
  name: string
}

export interface TaxonomyClassifierCategory {
  name: string
  description?: string
  exemplars: string[]
}

export interface TaxonomyClassifierMetric {
  name: string
  type: string
  positive_group?: string
  negative_group?: string
}

export interface TaxonomyClassifierRecord {
  name: string
  type: string
  builtin: boolean
  managed: boolean
  editable: boolean
  threshold: number
  label_thresholds?: Record<string, number>
  description?: string
  source: {
    path: string
    manifest?: string
  }
  labels: TaxonomyClassifierCategory[]
  groups?: Record<string, string[]>
  metrics?: TaxonomyClassifierMetric[]
  signal_references: TaxonomySignalReference[]
  bind_options: {
    labels: string[]
    groups: string[]
    metrics: string[]
  }
  load_error?: string
}

export interface TaxonomyClassifierListResponse {
  items: TaxonomyClassifierRecord[]
}

export interface TaxonomyClassifierDraft {
  name: string
  threshold: number
  description: string
  labels: TaxonomyClassifierCategory[]
  groups: Array<{
    name: string
    labels: string
  }>
  metrics: TaxonomyClassifierMetric[]
  label_thresholds: Array<{
    label: string
    threshold: number
  }>
}

export interface TaxonomyTierDraft {
  name: string
}

export interface TaxonomyCategoryDraft {
  name: string
  description: string
  exemplars: string
}

export function emptyTaxonomyClassifierDraft(): TaxonomyClassifierDraft {
  return {
    name: '',
    threshold: 0.55,
    description: '',
    labels: [
      {
        name: '',
        description: '',
        exemplars: [''],
      },
    ],
    groups: [],
    metrics: [],
    label_thresholds: [],
  }
}

export function emptyTaxonomyTierDraft(): TaxonomyTierDraft {
  return {
    name: '',
  }
}

export function emptyTaxonomyCategoryDraft(): TaxonomyCategoryDraft {
  return {
    name: '',
    description: '',
    exemplars: '',
  }
}

export function classifierDraftFromRecord(record: TaxonomyClassifierRecord): TaxonomyClassifierDraft {
  return {
    name: record.name,
    threshold: record.threshold,
    description: record.description ?? '',
    labels: record.labels.map((label) => ({
      name: label.name,
      description: label.description ?? '',
      exemplars: label.exemplars.length > 0 ? [...label.exemplars] : [''],
    })),
    groups: Object.entries(record.groups ?? {}).map(([name, labels]) => ({
      name,
      labels: labels.join(', '),
    })),
    metrics: (record.metrics ?? []).map((metric) => ({
      name: metric.name,
      type: metric.type,
      positive_group: metric.positive_group,
      negative_group: metric.negative_group,
    })),
    label_thresholds: Object.entries(record.label_thresholds ?? {}).map(([label, threshold]) => ({
      label,
      threshold,
    })),
  }
}

export function tierDraftFromTier(tier: TaxonomyClassifierTier): TaxonomyTierDraft {
  return {
    name: tier.name,
  }
}

export function categoryDraftFromCategory(category: TaxonomyClassifierCategory): TaxonomyCategoryDraft {
  return {
    name: category.name,
    description: category.description ?? '',
    exemplars: category.exemplars.join('\n'),
  }
}

export function payloadFromDraft(draft: TaxonomyClassifierDraft) {
  const groups = draft.groups.reduce<Record<string, string[]>>((acc, group) => {
    const name = group.name.trim()
    if (!name) {
      return acc
    }

    const labels = group.labels
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)

    if (labels.length > 0) {
      acc[name] = labels
    }

    return acc
  }, {})

  const labelThresholds = draft.label_thresholds.reduce<Record<string, number>>((acc, entry) => {
    const label = entry.label.trim()
    if (!label) {
      return acc
    }
    acc[label] = entry.threshold
    return acc
  }, {})

  return {
    name: draft.name.trim(),
    threshold: draft.threshold,
    description: draft.description.trim(),
    labels: draft.labels.map((label) => ({
      name: label.name.trim(),
      description: label.description?.trim() || '',
      exemplars: label.exemplars
        .map((exemplar) => exemplar.trim())
        .filter(Boolean),
    })),
    groups,
    metrics: draft.metrics
      .map((metric) => ({
        name: metric.name.trim(),
        type: metric.type.trim(),
        positive_group: metric.positive_group?.trim() || '',
        negative_group: metric.negative_group?.trim() || '',
      }))
      .filter((metric) => metric.name.length > 0 && metric.type.length > 0),
    label_thresholds: Object.keys(labelThresholds).length > 0 ? labelThresholds : undefined,
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string')
}

export function normalizeTaxonomySignalReference(raw: unknown): TaxonomySignalReference {
  const record = asRecord(raw)
  const targetRecord = asRecord(record.target)
  const match = asString(record.match || record.Match)
  return {
    name: asString(record.name),
    target: {
      kind: (asString(targetRecord.kind || targetRecord.Kind) as TaxonomySignalBinding['kind']) || 'group',
      value: asString(targetRecord.value || targetRecord.Value),
    },
    match: match === 'best' || match === 'threshold' ? match : undefined,
  }
}

export function normalizeTaxonomyClassifierRecord(raw: unknown): TaxonomyClassifierRecord {
  const record = asRecord(raw)
  const source = asRecord(record.source)
  const bindOptions = asRecord(record.bind_options)
  const labels = Array.isArray(record.labels) ? record.labels : []
  const signalReferences = Array.isArray(record.signal_references) ? record.signal_references : []
  const groups = asRecord(record.groups)
  const metrics = Array.isArray(record.metrics) ? record.metrics : []
  const labelThresholds = asRecord(record.label_thresholds)

  return {
    name: asString(record.name),
    type: asString(record.type),
    builtin: asBoolean(record.builtin),
    managed: asBoolean(record.managed),
    editable: asBoolean(record.editable),
    threshold: asNumber(record.threshold),
    label_thresholds: Object.keys(labelThresholds).length > 0
      ? Object.fromEntries(
          Object.entries(labelThresholds)
            .filter(([, value]) => typeof value === 'number')
            .map(([key, value]) => [key, value as number])
        )
      : undefined,
    description: asString(record.description) || undefined,
    source: {
      path: asString(source.path || source.Path),
      manifest: asString(source.manifest || source.Manifest) || undefined,
    },
    labels: labels.map((label) => {
      const labelRecord = asRecord(label)
      return {
        name: asString(labelRecord.name),
        description: asString(labelRecord.description) || undefined,
        exemplars: asStringArray(labelRecord.exemplars),
      }
    }),
    groups: Object.keys(groups).length > 0
      ? Object.fromEntries(
          Object.entries(groups).map(([key, value]) => [key, asStringArray(value)])
        )
      : undefined,
    metrics: metrics.map((metric) => {
      const metricRecord = asRecord(metric)
      return {
        name: asString(metricRecord.name),
        type: asString(metricRecord.type),
        positive_group: asString(metricRecord.positive_group) || undefined,
        negative_group: asString(metricRecord.negative_group) || undefined,
      }
    }),
    signal_references: signalReferences.map((reference) => normalizeTaxonomySignalReference(reference)),
    bind_options: {
      labels: asStringArray(bindOptions.labels),
      groups: asStringArray(bindOptions.groups),
      metrics: asStringArray(bindOptions.metrics),
    },
    load_error: asString(record.load_error) || undefined,
  }
}

export function normalizeTaxonomyClassifierListResponse(raw: unknown): TaxonomyClassifierListResponse {
  const record = asRecord(raw)
  const items = Array.isArray(record.items) ? record.items : []
  return {
    items: items.map((item) => normalizeTaxonomyClassifierRecord(item)),
  }
}

export function formatSignalReference(reference: TaxonomySignalReference): string {
  const targetKind = reference.target.kind || 'unknown'
  const targetValue = reference.target.value || 'unknown'
  const match = reference.match ? ` (${reference.match})` : ''
  return `${reference.name} -> ${targetKind}:${targetValue}${match}`
}

export function countSignalsForTier(record: TaxonomyClassifierRecord, groupName: string): number {
  return record.signal_references.filter(
    (reference) => reference.target.kind === 'group' && reference.target.value === groupName
  ).length
}

export function countSignalsForCategory(record: TaxonomyClassifierRecord, labelName: string): number {
  return record.signal_references.filter(
    (reference) => reference.target.kind === 'label' && reference.target.value === labelName
  ).length
}

export function countMetricsForGroup(record: TaxonomyClassifierRecord, groupName: string): number {
  return (record.metrics ?? []).filter(
    (metric) => metric.positive_group === groupName || metric.negative_group === groupName
  ).length
}

function rewriteGroupsOnLabelRename(
  groups: TaxonomyClassifierDraft['groups'],
  originalName: string,
  nextName: string
) {
  return groups.map((group) => ({
    ...group,
    labels: group.labels
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
      .map((item) => (item === originalName ? nextName : item))
      .join(', '),
  }))
}

function rewriteGroupsOnLabelDelete(
  groups: TaxonomyClassifierDraft['groups'],
  labelName: string
) {
  return groups
    .map((group) => ({
      ...group,
      labels: group.labels
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item && item !== labelName)
        .join(', '),
    }))
    .filter((group) => group.name.trim())
}

function rewriteThresholdsOnLabelRename(
  thresholds: TaxonomyClassifierDraft['label_thresholds'],
  originalName: string,
  nextName: string
) {
  return thresholds.map((entry) =>
    entry.label === originalName ? { ...entry, label: nextName } : entry
  )
}

function rewriteThresholdsOnLabelDelete(
  thresholds: TaxonomyClassifierDraft['label_thresholds'],
  labelName: string
) {
  return thresholds.filter((entry) => entry.label !== labelName)
}

function rewriteMetricsOnGroupRename(
  metrics: TaxonomyClassifierDraft['metrics'],
  originalName: string,
  nextName: string
) {
  return metrics.map((metric) => ({
    ...metric,
    positive_group: metric.positive_group === originalName ? nextName : metric.positive_group,
    negative_group: metric.negative_group === originalName ? nextName : metric.negative_group,
  }))
}

export function renameTierInDraft(
  draft: TaxonomyClassifierDraft,
  originalName: string,
  nextTier: TaxonomyTierDraft
): TaxonomyClassifierDraft {
  const nextName = nextTier.name.trim()
  return {
    ...draft,
    groups: draft.groups.map((group) =>
      group.name === originalName
        ? { ...group, name: nextName }
        : group
    ),
    metrics: rewriteMetricsOnGroupRename(draft.metrics, originalName, nextName),
  }
}

export function addTierToDraft(
  draft: TaxonomyClassifierDraft,
  nextTier: TaxonomyTierDraft
): TaxonomyClassifierDraft {
  return {
    ...draft,
    groups: [
      ...draft.groups,
      {
        name: nextTier.name.trim(),
        labels: '',
      },
    ],
  }
}

export function removeTierFromDraft(
  draft: TaxonomyClassifierDraft,
  tierName: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    groups: draft.groups.filter((group) => group.name !== tierName),
  }
}

export function renameCategoryInDraft(
  draft: TaxonomyClassifierDraft,
  originalName: string,
  nextCategory: TaxonomyCategoryDraft
): TaxonomyClassifierDraft {
  const nextName = nextCategory.name.trim()
  return {
    ...draft,
    labels: draft.labels.map((label) =>
      label.name === originalName
        ? {
            name: nextName,
            description: nextCategory.description.trim(),
            exemplars: nextCategory.exemplars
              .split('\n')
              .map((item) => item.trim())
              .filter(Boolean),
          }
        : label
    ),
    groups: rewriteGroupsOnLabelRename(draft.groups, originalName, nextName),
    label_thresholds: rewriteThresholdsOnLabelRename(draft.label_thresholds, originalName, nextName),
  }
}

export function addCategoryToDraft(
  draft: TaxonomyClassifierDraft,
  nextCategory: TaxonomyCategoryDraft
): TaxonomyClassifierDraft {
  return {
    ...draft,
    labels: [
      ...draft.labels,
      {
        name: nextCategory.name.trim(),
        description: nextCategory.description.trim(),
        exemplars: nextCategory.exemplars
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean),
      },
    ],
  }
}

export function removeCategoryFromDraft(
  draft: TaxonomyClassifierDraft,
  categoryName: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    labels: draft.labels.filter((label) => label.name !== categoryName),
    groups: rewriteGroupsOnLabelDelete(draft.groups, categoryName),
    label_thresholds: rewriteThresholdsOnLabelDelete(draft.label_thresholds, categoryName),
  }
}
