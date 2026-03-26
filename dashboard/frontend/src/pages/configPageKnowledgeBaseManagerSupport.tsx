import type { Column } from '../components/DataTable'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  countMetricsForGroup,
  countSignalsForCategory,
  countSignalsForTier,
  type TaxonomyClassifierCategory,
  type TaxonomyClassifierDraft,
  type TaxonomyClassifierRecord,
} from './configPageTaxonomyClassifierSupport'

export type KnowledgeBaseManagerView = 'bases' | 'groups' | 'labels'

export interface KnowledgeBaseRow extends TaxonomyClassifierRecord {
  label_count: number
  group_count: number
  metric_count: number
  signal_count: number
}

export interface GroupRow {
  name: string
  labels: string[]
  label_count: number
  signal_count: number
  metric_count: number
}

export interface LabelRow extends TaxonomyClassifierCategory {
  exemplar_count: number
  signal_count: number
  threshold_value?: number
}

export interface KnowledgeBaseGroupDraft {
  name: string
  labels: string[]
}

export interface KnowledgeBaseLabelDraft {
  name: string
  description: string
  exemplars: string
}

export function renameGroupInDraft(
  draft: TaxonomyClassifierDraft,
  originalName: string,
  nextGroup: KnowledgeBaseGroupDraft
): TaxonomyClassifierDraft {
  const nextName = nextGroup.name.trim()
  return {
    ...draft,
    groups: draft.groups.map((group) =>
      group.name === originalName
        ? { name: nextName, labels: nextGroup.labels.join(', ') }
        : group
    ),
    metrics: draft.metrics.map((metric) => ({
      ...metric,
      positive_group: metric.positive_group === originalName ? nextName : metric.positive_group,
      negative_group: metric.negative_group === originalName ? nextName : metric.negative_group,
    })),
  }
}

export function addGroupToDraft(
  draft: TaxonomyClassifierDraft,
  nextGroup: KnowledgeBaseGroupDraft
): TaxonomyClassifierDraft {
  return {
    ...draft,
    groups: [
      ...draft.groups,
      {
        name: nextGroup.name.trim(),
        labels: nextGroup.labels.join(', '),
      },
    ],
  }
}

export function removeGroupFromDraft(
  draft: TaxonomyClassifierDraft,
  groupName: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    groups: draft.groups.filter((group) => group.name !== groupName),
  }
}

export function renameLabelInDraft(
  draft: TaxonomyClassifierDraft,
  originalName: string,
  nextLabel: KnowledgeBaseLabelDraft
): TaxonomyClassifierDraft {
  const nextName = nextLabel.name.trim()
  return {
    ...draft,
    labels: draft.labels.map((label) =>
      label.name === originalName
        ? {
            name: nextName,
            description: nextLabel.description.trim(),
            exemplars: nextLabel.exemplars
              .split('\n')
              .map((item) => item.trim())
              .filter(Boolean),
          }
        : label
    ),
    groups: draft.groups.map((group) => ({
      ...group,
      labels: group.labels
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => (item === originalName ? nextName : item))
        .join(', '),
    })),
    label_thresholds: draft.label_thresholds.map((entry) =>
      entry.label === originalName ? { ...entry, label: nextName } : entry
    ),
  }
}

export function addLabelToDraft(
  draft: TaxonomyClassifierDraft,
  nextLabel: KnowledgeBaseLabelDraft
): TaxonomyClassifierDraft {
  return {
    ...draft,
    labels: [
      ...draft.labels,
      {
        name: nextLabel.name.trim(),
        description: nextLabel.description.trim(),
        exemplars: nextLabel.exemplars
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean),
      },
    ],
  }
}

export function removeLabelFromDraft(
  draft: TaxonomyClassifierDraft,
  labelName: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    labels: draft.labels.filter((label) => label.name !== labelName),
    groups: draft.groups.map((group) => ({
      ...group,
      labels: group.labels
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item && item !== labelName)
        .join(', '),
    })),
    label_thresholds: draft.label_thresholds.filter((entry) => entry.label !== labelName),
  }
}

export function buildKnowledgeBaseRows(
  knowledgeBases: TaxonomyClassifierRecord[],
  query: string
): KnowledgeBaseRow[] {
  const normalizedQuery = query.trim().toLowerCase()
  return knowledgeBases
    .filter((knowledgeBase) => {
      if (!normalizedQuery) {
        return true
      }
      return (
        knowledgeBase.name.toLowerCase().includes(normalizedQuery) ||
        (knowledgeBase.description ?? '').toLowerCase().includes(normalizedQuery) ||
        knowledgeBase.source.path.toLowerCase().includes(normalizedQuery)
      )
    })
    .map((knowledgeBase) => ({
      ...knowledgeBase,
      label_count: knowledgeBase.labels.length,
      group_count: Object.keys(knowledgeBase.groups ?? {}).length,
      metric_count: knowledgeBase.metrics?.length ?? 0,
      signal_count: knowledgeBase.signal_references.length,
    }))
}

export function buildGroupRows(
  selectedKnowledgeBase: TaxonomyClassifierRecord | null,
  query: string
): GroupRow[] {
  if (!selectedKnowledgeBase) {
    return []
  }
  const normalizedQuery = query.trim().toLowerCase()
  return Object.entries(selectedKnowledgeBase.groups ?? {})
    .map(([name, labels]) => ({
      name,
      labels,
      label_count: labels.length,
      signal_count: countSignalsForTier(selectedKnowledgeBase, name),
      metric_count: countMetricsForGroup(selectedKnowledgeBase, name),
    }))
    .filter((group) => {
      if (!normalizedQuery) {
        return true
      }
      return (
        group.name.toLowerCase().includes(normalizedQuery) ||
        group.labels.some((label) => label.toLowerCase().includes(normalizedQuery))
      )
    })
}

export function buildLabelRows(
  selectedKnowledgeBase: TaxonomyClassifierRecord | null,
  query: string
): LabelRow[] {
  if (!selectedKnowledgeBase) {
    return []
  }
  const normalizedQuery = query.trim().toLowerCase()
  return selectedKnowledgeBase.labels
    .filter((label) => {
      if (!normalizedQuery) {
        return true
      }
      return (
        label.name.toLowerCase().includes(normalizedQuery) ||
        (label.description ?? '').toLowerCase().includes(normalizedQuery)
      )
    })
    .map((label) => ({
      ...label,
      exemplar_count: label.exemplars.length,
      signal_count: countSignalsForCategory(selectedKnowledgeBase, label.name),
      threshold_value: selectedKnowledgeBase.label_thresholds?.[label.name],
    }))
}

export function buildKnowledgeBaseCounts(knowledgeBases: TaxonomyClassifierRecord[]) {
  return {
    total: knowledgeBases.length,
    builtin: knowledgeBases.filter((knowledgeBase) => knowledgeBase.builtin).length,
    custom: knowledgeBases.filter((knowledgeBase) => !knowledgeBase.builtin).length,
  }
}

interface KnowledgeBaseColumnArgs {
  selectedKnowledgeBaseName?: string
  isReadonly: boolean
  onSelect: (name: string) => void
  onEdit: (row: TaxonomyClassifierRecord) => void
  onDelete: (row: TaxonomyClassifierRecord) => void
}

export function buildKnowledgeBaseColumns({
  selectedKnowledgeBaseName,
  isReadonly,
  onSelect,
  onEdit,
  onDelete,
}: KnowledgeBaseColumnArgs): Column<KnowledgeBaseRow>[] {
  return [
    {
      key: 'name',
      header: 'Knowledge Base',
      sortable: true,
      render: (row) => (
        <div className={styles.primaryCell}>
          <div className={styles.primaryCellTitleRow}>
            <span className={styles.primaryCellTitle}>{row.name}</span>
            {row.builtin ? <span className={styles.tableBadge}>Built-in</span> : <span className={styles.mutedBadge}>Custom</span>}
            {selectedKnowledgeBaseName === row.name ? <span className={styles.selectedBadge}>Selected</span> : null}
          </div>
          <span className={styles.primaryCellMeta}>{row.description || 'No description provided.'}</span>
        </div>
      ),
    },
    {
      key: 'label_count',
      header: 'Labels',
      sortable: true,
      align: 'center',
    },
    {
      key: 'group_count',
      header: 'Groups',
      sortable: true,
      align: 'center',
    },
    {
      key: 'metric_count',
      header: 'Metrics',
      sortable: true,
      align: 'center',
    },
    {
      key: 'signal_count',
      header: 'Signals',
      sortable: true,
      align: 'center',
    },
    {
      key: 'source',
      header: 'Source',
      render: (row) => <code className={styles.inlineCode}>{row.source.path}</code>,
    },
    {
      key: 'actions',
      header: 'Actions',
      align: 'right',
      render: (row) => (
        <div className={styles.tableActionGroup}>
          <button type="button" className={styles.secondaryButton} onClick={() => onSelect(row.name)}>
            {selectedKnowledgeBaseName === row.name ? 'Open' : 'View'}
          </button>
          {row.editable && !isReadonly ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => onEdit(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void onDelete(row)}>
                Delete
              </button>
            </>
          ) : null}
        </div>
      ),
    },
  ]
}

interface GroupColumnArgs {
  isReadonly: boolean
  editable?: boolean
  onEdit: (row: GroupRow) => void
  onDelete: (row: GroupRow) => void
}

export function buildGroupColumns({
  isReadonly,
  editable,
  onEdit,
  onDelete,
}: GroupColumnArgs): Column<GroupRow>[] {
  return [
    {
      key: 'name',
      header: 'Group',
      sortable: true,
      render: (row) => (
        <div className={styles.primaryCell}>
          <span className={styles.primaryCellTitle}>{row.name}</span>
          <span className={styles.primaryCellMeta}>
            {row.labels.length > 0 ? row.labels.join(', ') : 'No labels assigned.'}
          </span>
        </div>
      ),
    },
    {
      key: 'label_count',
      header: 'Labels',
      sortable: true,
      align: 'center',
    },
    {
      key: 'signal_count',
      header: 'Signals',
      sortable: true,
      align: 'center',
    },
    {
      key: 'metric_count',
      header: 'Metrics',
      sortable: true,
      align: 'center',
    },
    {
      key: 'actions',
      header: 'Actions',
      align: 'right',
      render: (row) => (
        <div className={styles.tableActionGroup}>
          {!isReadonly && editable ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => onEdit(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void onDelete(row)}>
                Delete
              </button>
            </>
          ) : (
            <span className={styles.readOnlyHint}>Read-only</span>
          )}
        </div>
      ),
    },
  ]
}

interface LabelColumnArgs {
  isReadonly: boolean
  editable?: boolean
  onEdit: (row: LabelRow) => void
  onDelete: (row: LabelRow) => void
}

export function buildLabelColumns({
  isReadonly,
  editable,
  onEdit,
  onDelete,
}: LabelColumnArgs): Column<LabelRow>[] {
  return [
    {
      key: 'name',
      header: 'Label',
      sortable: true,
      render: (row) => (
        <div className={styles.primaryCell}>
          <span className={styles.primaryCellTitle}>{row.name}</span>
          <span className={styles.primaryCellMeta}>{row.description || 'No label description provided.'}</span>
        </div>
      ),
    },
    {
      key: 'threshold_value',
      header: 'Threshold',
      sortable: true,
      align: 'center',
      render: (row) => (
        <span className={styles.tableBadge}>
          {typeof row.threshold_value === 'number' ? row.threshold_value : 'Default'}
        </span>
      ),
    },
    {
      key: 'exemplar_count',
      header: 'Exemplars',
      sortable: true,
      align: 'center',
    },
    {
      key: 'signal_count',
      header: 'Signals',
      sortable: true,
      align: 'center',
    },
    {
      key: 'actions',
      header: 'Actions',
      align: 'right',
      render: (row) => (
        <div className={styles.tableActionGroup}>
          {!isReadonly && editable ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => onEdit(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void onDelete(row)}>
                Delete
              </button>
            </>
          ) : (
            <span className={styles.readOnlyHint}>Read-only</span>
          )}
        </div>
      ),
    },
  ]
}
