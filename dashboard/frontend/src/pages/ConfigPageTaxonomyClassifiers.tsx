import { useCallback, useEffect, useMemo, useState } from 'react'
import type { FieldConfig } from '../components/EditModal'
import { DataTable, type Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import pageStyles from './ConfigPage.module.css'
import ConfigPageTaxonomyClassifierEditor from './ConfigPageTaxonomyClassifierEditor'
import ConfigPageTaxonomyClassifierDetail from './ConfigPageTaxonomyClassifierDetail'
import type { OpenEditModal } from './configPageRouterSectionSupport'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  classifierDraftFromRecord,
  countMetricsForGroup,
  countSignalsForCategory,
  countSignalsForTier,
  emptyTaxonomyClassifierDraft,
  normalizeTaxonomyClassifierListResponse,
  payloadFromDraft,
  type TaxonomyClassifierCategory,
  type TaxonomyClassifierDraft,
  type TaxonomyClassifierRecord,
} from './configPageTaxonomyClassifierSupport'

export type KnowledgeBaseManagerView = 'knowledge-bases' | 'groups' | 'labels' | 'exemplars'

interface ConfigPageTaxonomyClassifiersProps {
  isReadonly: boolean
  openEditModal: OpenEditModal
  activeView?: KnowledgeBaseManagerView
}

interface KnowledgeBaseRow extends TaxonomyClassifierRecord {
  modeLabel: string
  label_count: number
  group_count: number
  metric_count: number
  signal_count: number
  exemplar_count: number
}

interface GroupRow {
  name: string
  labels: string[]
  label_count: number
  signal_count: number
  metric_count: number
}

interface LabelRow extends TaxonomyClassifierCategory {
  exemplar_count: number
  signal_count: number
  threshold_value?: number
}

interface ExemplarRow {
  key: string
  label: string
  exemplar: string
  exemplarIndex: number
}

interface KnowledgeBaseGroupDraft {
  name: string
  labels: string[]
}

interface KnowledgeBaseLabelDraft {
  name: string
  description: string
  exemplars: string
}

interface KnowledgeBaseExemplarDraft {
  label: string
  exemplar: string
}

function buildKnowledgeBaseModeLabel(knowledgeBase: TaxonomyClassifierRecord): string {
  if (knowledgeBase.builtin) {
    return 'Built-in'
  }
  if (knowledgeBase.editable) {
    return 'Managed'
  }
  return 'External'
}

function addExemplarToDraft(
  draft: TaxonomyClassifierDraft,
  labelName: string,
  exemplar: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    labels: draft.labels.map((label) =>
      label.name === labelName
        ? { ...label, exemplars: [...label.exemplars, exemplar] }
        : label
    ),
  }
}

function updateExemplarInDraft(
  draft: TaxonomyClassifierDraft,
  originalLabel: string,
  originalIndex: number,
  nextLabel: string,
  nextExemplar: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    labels: draft.labels.map((label) => {
      if (label.name === originalLabel && label.name === nextLabel) {
        return {
          ...label,
          exemplars: label.exemplars.map((exemplar, index) =>
            index === originalIndex ? nextExemplar : exemplar
          ),
        }
      }
      if (label.name === originalLabel) {
        return {
          ...label,
          exemplars: label.exemplars.filter((_, index) => index !== originalIndex),
        }
      }
      if (label.name === nextLabel) {
        return {
          ...label,
          exemplars: [...label.exemplars, nextExemplar],
        }
      }
      return label
    }),
  }
}

function removeExemplarFromDraft(
  draft: TaxonomyClassifierDraft,
  labelName: string,
  exemplarIndex: number
): TaxonomyClassifierDraft {
  return {
    ...draft,
    labels: draft.labels.map((label) =>
      label.name === labelName
        ? {
            ...label,
            exemplars: label.exemplars.filter((_, index) => index !== exemplarIndex),
          }
        : label
    ),
  }
}

function renameGroupInDraft(
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

function addGroupToDraft(
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

function removeGroupFromDraft(
  draft: TaxonomyClassifierDraft,
  groupName: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    groups: draft.groups.filter((group) => group.name !== groupName),
  }
}

function renameLabelInDraft(
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

function addLabelToDraft(
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

function removeLabelFromDraft(
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

export default function ConfigPageTaxonomyClassifiers({
  isReadonly,
  openEditModal,
  activeView = 'knowledge-bases',
}: ConfigPageTaxonomyClassifiersProps) {
  const [knowledgeBases, setKnowledgeBases] = useState<TaxonomyClassifierRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedKnowledgeBaseName, setSelectedKnowledgeBaseName] = useState('')
  const [knowledgeBaseSearch, setKnowledgeBaseSearch] = useState('')
  const [groupSearch, setGroupSearch] = useState('')
  const [labelSearch, setLabelSearch] = useState('')
  const [exemplarSearch, setExemplarSearch] = useState('')

  const loadKnowledgeBases = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/router/config/kbs')
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }
      const payload = normalizeTaxonomyClassifierListResponse(await response.json())
      const items = payload.items || []
      setKnowledgeBases(items)
      setSelectedKnowledgeBaseName((current) => {
        if (current && items.some((knowledgeBase) => knowledgeBase.name === current)) {
          return current
        }
        return items.find((knowledgeBase) => knowledgeBase.editable)?.name ?? items[0]?.name ?? ''
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load knowledge bases')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadKnowledgeBases()
  }, [loadKnowledgeBases])

  const knowledgeBaseRows = useMemo<KnowledgeBaseRow[]>(() => {
    return knowledgeBases
      .filter((knowledgeBase) => {
        const query = knowledgeBaseSearch.trim().toLowerCase()
        if (!query) {
          return true
        }
        return (
          knowledgeBase.name.toLowerCase().includes(query) ||
          (knowledgeBase.description ?? '').toLowerCase().includes(query) ||
          knowledgeBase.source.path.toLowerCase().includes(query)
        )
      })
      .map((knowledgeBase) => ({
        ...knowledgeBase,
        modeLabel: buildKnowledgeBaseModeLabel(knowledgeBase),
        label_count: knowledgeBase.labels.length,
        group_count: Object.keys(knowledgeBase.groups ?? {}).length,
        metric_count: knowledgeBase.metrics?.length ?? 0,
        signal_count: knowledgeBase.signal_references.length,
        exemplar_count: knowledgeBase.labels.reduce((count, label) => count + label.exemplars.length, 0),
      }))
  }, [knowledgeBaseSearch, knowledgeBases])

  const selectedKnowledgeBase = useMemo(
    () =>
      knowledgeBases.find((knowledgeBase) => knowledgeBase.name === selectedKnowledgeBaseName) ??
      knowledgeBaseRows[0] ??
      null,
    [knowledgeBaseRows, knowledgeBases, selectedKnowledgeBaseName]
  )

  const groupRows = useMemo<GroupRow[]>(() => {
    if (!selectedKnowledgeBase) {
      return []
    }
    const query = groupSearch.trim().toLowerCase()
    return Object.entries(selectedKnowledgeBase.groups ?? {})
      .map(([name, labels]) => ({
        name,
        labels,
        label_count: labels.length,
        signal_count: countSignalsForTier(selectedKnowledgeBase, name),
        metric_count: countMetricsForGroup(selectedKnowledgeBase, name),
      }))
      .filter((group) => {
        if (!query) {
          return true
        }
        return (
          group.name.toLowerCase().includes(query) ||
          group.labels.some((label) => label.toLowerCase().includes(query))
        )
      })
  }, [groupSearch, selectedKnowledgeBase])

  const labelRows = useMemo<LabelRow[]>(() => {
    if (!selectedKnowledgeBase) {
      return []
    }
    const query = labelSearch.trim().toLowerCase()
    return selectedKnowledgeBase.labels
      .filter((label) => {
        if (!query) {
          return true
        }
        return (
          label.name.toLowerCase().includes(query) ||
          (label.description ?? '').toLowerCase().includes(query)
        )
      })
      .map((label) => ({
        ...label,
        exemplar_count: label.exemplars.length,
        signal_count: countSignalsForCategory(selectedKnowledgeBase, label.name),
        threshold_value: selectedKnowledgeBase.label_thresholds?.[label.name],
      }))
  }, [labelSearch, selectedKnowledgeBase])

  const exemplarRows = useMemo<ExemplarRow[]>(() => {
    if (!selectedKnowledgeBase) {
      return []
    }
    const query = exemplarSearch.trim().toLowerCase()
    return selectedKnowledgeBase.labels.flatMap((label) =>
      label.exemplars
        .map((exemplar, exemplarIndex) => ({
          key: `${label.name}:${exemplarIndex}`,
          label: label.name,
          exemplar,
          exemplarIndex,
        }))
        .filter((row) => {
          if (!query) {
            return true
          }
          return (
            row.label.toLowerCase().includes(query) ||
            row.exemplar.toLowerCase().includes(query)
          )
        })
    )
  }, [exemplarSearch, selectedKnowledgeBase])

  const counts = useMemo(
    () => ({
      total: knowledgeBases.length,
      builtin: knowledgeBases.filter((knowledgeBase) => knowledgeBase.builtin).length,
      managed: knowledgeBases.filter((knowledgeBase) => !knowledgeBase.builtin).length,
    }),
    [knowledgeBases]
  )

  const knowledgeBaseEditorField = useCallback(
    (disableName: boolean): FieldConfig[] => [
      {
        name: 'draft',
        label: 'Knowledge Base',
        type: 'custom',
        customRender: (value, onChange) => (
          <ConfigPageTaxonomyClassifierEditor
            value={value}
            onChange={(nextValue) => onChange(nextValue)}
            disableName={disableName}
          />
        ),
      },
    ],
    []
  )

  const persistKnowledgeBase = useCallback(
    async (
      endpoint: string,
      method: 'POST' | 'PUT',
      draft: TaxonomyClassifierDraft,
      nextSelection?: string
    ) => {
      const response = await fetch(endpoint, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payloadFromDraft(draft)),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }
      if (nextSelection) {
        setSelectedKnowledgeBaseName(nextSelection)
      }
      await loadKnowledgeBases()
    },
    [loadKnowledgeBases]
  )

  const persistSelectedKnowledgeBase = useCallback(
    async (mutate: (draft: TaxonomyClassifierDraft) => TaxonomyClassifierDraft) => {
      if (!selectedKnowledgeBase || !selectedKnowledgeBase.editable) {
        throw new Error('Select a knowledge base to edit.')
      }
      const currentDraft = classifierDraftFromRecord(selectedKnowledgeBase)
      const nextDraft = mutate(currentDraft)
      await persistKnowledgeBase(
        `/api/router/config/kbs/${selectedKnowledgeBase.name}`,
        'PUT',
        nextDraft,
        selectedKnowledgeBase.name
      )
    },
    [persistKnowledgeBase, selectedKnowledgeBase]
  )

  const openCreateModal = () => {
    openEditModal<{ draft: TaxonomyClassifierDraft }>(
      'Add Knowledge Base',
      { draft: emptyTaxonomyClassifierDraft() },
      knowledgeBaseEditorField(false),
      async (data) => {
        const nextName = data.draft.name.trim()
        await persistKnowledgeBase('/api/router/config/kbs', 'POST', data.draft, nextName)
      },
      'add'
    )
  }

  const openEditKnowledgeBaseModal = (knowledgeBase: TaxonomyClassifierRecord) => {
    openEditModal<{ draft: TaxonomyClassifierDraft }>(
      `Edit ${knowledgeBase.name}`,
      { draft: classifierDraftFromRecord(knowledgeBase) },
      knowledgeBaseEditorField(true),
      async (data) => {
        await persistKnowledgeBase(`/api/router/config/kbs/${knowledgeBase.name}`, 'PUT', data.draft, knowledgeBase.name)
      },
      'edit'
    )
  }

  const handleDeleteKnowledgeBase = async (knowledgeBase: TaxonomyClassifierRecord) => {
    if (!window.confirm(`Delete knowledge base "${knowledgeBase.name}"?`)) {
      return
    }

    setError(null)
    try {
      const response = await fetch(`/api/router/config/kbs/${knowledgeBase.name}`, {
        method: 'DELETE',
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }
      if (selectedKnowledgeBaseName === knowledgeBase.name) {
        setSelectedKnowledgeBaseName('')
      }
      await loadKnowledgeBases()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete knowledge base')
    }
  }

  const openAddGroupModal = () => {
    if (!selectedKnowledgeBase) {
      return
    }
    openEditModal<KnowledgeBaseGroupDraft>(
      `Add Group · ${selectedKnowledgeBase.name}`,
      { name: '', labels: [] },
      [
        { name: 'name', label: 'Group Name', type: 'text', required: true, placeholder: 'private' },
        {
          name: 'labels',
          label: 'Labels',
          type: 'multiselect',
          options: selectedKnowledgeBase.bind_options.labels,
          description: 'Select the labels that belong to this group.',
        },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (groupRows.some((group) => group.name === nextName)) {
          throw new Error(`Group "${nextName}" already exists.`)
        }
        await persistSelectedKnowledgeBase((draft) => addGroupToDraft(draft, data))
      },
      'add'
    )
  }

  const openEditGroupModal = (group: GroupRow) => {
    if (!selectedKnowledgeBase) {
      return
    }
    openEditModal<KnowledgeBaseGroupDraft>(
      `Edit Group · ${group.name}`,
      { name: group.name, labels: group.labels },
      [
        { name: 'name', label: 'Group Name', type: 'text', required: true, placeholder: 'private' },
        {
          name: 'labels',
          label: 'Labels',
          type: 'multiselect',
          options: selectedKnowledgeBase.bind_options.labels,
          description: 'Select the labels that belong to this group.',
        },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (nextName !== group.name && groupRows.some((existingGroup) => existingGroup.name === nextName)) {
          throw new Error(`Group "${nextName}" already exists.`)
        }
        if (nextName !== group.name && countSignalsForTier(selectedKnowledgeBase, group.name) > 0) {
          throw new Error(`Group "${group.name}" is referenced by KB signals. Update those signals before renaming it.`)
        }
        await persistSelectedKnowledgeBase((draft) => renameGroupInDraft(draft, group.name, data))
      },
      'edit'
    )
  }

  const handleDeleteGroup = async (group: GroupRow) => {
    if (!selectedKnowledgeBase) {
      return
    }
    if (group.signal_count > 0) {
      setError(`Group "${group.name}" is referenced by KB signals. Remove those signals first.`)
      return
    }
    if (group.metric_count > 0) {
      setError(`Group "${group.name}" is referenced by metrics. Update those metrics first.`)
      return
    }
    if (!window.confirm(`Delete group "${group.name}" from ${selectedKnowledgeBase.name}?`)) {
      return
    }
    setError(null)
    try {
      await persistSelectedKnowledgeBase((draft) => removeGroupFromDraft(draft, group.name))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete group')
    }
  }

  const openAddLabelModal = () => {
    if (!selectedKnowledgeBase) {
      return
    }
    openEditModal<KnowledgeBaseLabelDraft>(
      `Add Label · ${selectedKnowledgeBase.name}`,
      { name: '', description: '', exemplars: '' },
      [
        { name: 'name', label: 'Label Name', type: 'text', required: true, placeholder: 'prompt_injection' },
        { name: 'description', label: 'Description', type: 'textarea', placeholder: 'What this label captures.' },
        { name: 'exemplars', label: 'Exemplars', type: 'textarea', required: true, placeholder: 'One exemplar per line' },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (selectedKnowledgeBase.labels.some((label) => label.name === nextName)) {
          throw new Error(`Label "${nextName}" already exists.`)
        }
        await persistSelectedKnowledgeBase((draft) => addLabelToDraft(draft, data))
      },
      'add'
    )
  }

  const openEditLabelModal = (label: TaxonomyClassifierCategory) => {
    if (!selectedKnowledgeBase) {
      return
    }
    openEditModal<KnowledgeBaseLabelDraft>(
      `Edit Label · ${label.name}`,
      {
        name: label.name,
        description: label.description ?? '',
        exemplars: label.exemplars.join('\n'),
      },
      [
        { name: 'name', label: 'Label Name', type: 'text', required: true, placeholder: 'prompt_injection' },
        { name: 'description', label: 'Description', type: 'textarea', placeholder: 'What this label captures.' },
        { name: 'exemplars', label: 'Exemplars', type: 'textarea', required: true, placeholder: 'One exemplar per line' },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (
          nextName !== label.name &&
          selectedKnowledgeBase.labels.some((existingLabel) => existingLabel.name === nextName)
        ) {
          throw new Error(`Label "${nextName}" already exists.`)
        }
        if (nextName !== label.name && countSignalsForCategory(selectedKnowledgeBase, label.name) > 0) {
          throw new Error(`Label "${label.name}" is referenced by KB signals. Update those signals before renaming it.`)
        }
        await persistSelectedKnowledgeBase((draft) => renameLabelInDraft(draft, label.name, data))
      },
      'edit'
    )
  }

  const handleDeleteLabel = async (label: TaxonomyClassifierCategory) => {
    if (!selectedKnowledgeBase) {
      return
    }
    if (countSignalsForCategory(selectedKnowledgeBase, label.name) > 0) {
      setError(`Label "${label.name}" is referenced by KB signals. Remove those signals first.`)
      return
    }
    if (!window.confirm(`Delete label "${label.name}" from ${selectedKnowledgeBase.name}?`)) {
      return
    }
    setError(null)
    try {
      await persistSelectedKnowledgeBase((draft) => removeLabelFromDraft(draft, label.name))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete label')
    }
  }

  const openAddExemplarModal = () => {
    if (!selectedKnowledgeBase) {
      return
    }
    if (selectedKnowledgeBase.labels.length === 0) {
      setError('Add at least one label before creating exemplars.')
      return
    }
    openEditModal<KnowledgeBaseExemplarDraft>(
      `Add Exemplar · ${selectedKnowledgeBase.name}`,
      {
        label: selectedKnowledgeBase.labels[0]?.name ?? '',
        exemplar: '',
      },
      [
        { name: 'label', label: 'Label', type: 'select', required: true, options: selectedKnowledgeBase.labels.map((label) => label.name) },
        { name: 'exemplar', label: 'Exemplar', type: 'textarea', required: true, placeholder: 'Representative prompt for this label' },
      ],
      async (data) => {
        const nextLabel = data.label.trim()
        const nextExemplar = data.exemplar.trim()
        if (!nextLabel || !nextExemplar) {
          throw new Error('Label and exemplar are required.')
        }
        await persistSelectedKnowledgeBase((draft) => addExemplarToDraft(draft, nextLabel, nextExemplar))
      },
      'add'
    )
  }

  const openEditExemplarModal = (row: ExemplarRow) => {
    if (!selectedKnowledgeBase) {
      return
    }
    openEditModal<KnowledgeBaseExemplarDraft>(
      `Edit Exemplar · ${row.label}`,
      {
        label: row.label,
        exemplar: row.exemplar,
      },
      [
        { name: 'label', label: 'Label', type: 'select', required: true, options: selectedKnowledgeBase.labels.map((label) => label.name) },
        { name: 'exemplar', label: 'Exemplar', type: 'textarea', required: true, placeholder: 'Representative prompt for this label' },
      ],
      async (data) => {
        const nextLabel = data.label.trim()
        const nextExemplar = data.exemplar.trim()
        if (!nextLabel || !nextExemplar) {
          throw new Error('Label and exemplar are required.')
        }
        await persistSelectedKnowledgeBase((draft) =>
          updateExemplarInDraft(draft, row.label, row.exemplarIndex, nextLabel, nextExemplar)
        )
      },
      'edit'
    )
  }

  const handleDeleteExemplar = async (row: ExemplarRow) => {
    if (!selectedKnowledgeBase) {
      return
    }
    if (!window.confirm(`Delete this exemplar from "${row.label}"?`)) {
      return
    }
    setError(null)
    try {
      await persistSelectedKnowledgeBase((draft) => removeExemplarFromDraft(draft, row.label, row.exemplarIndex))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete exemplar')
    }
  }

  const knowledgeBaseColumns = useMemo<Column<KnowledgeBaseRow>[]>(() => [
    {
      key: 'name',
      header: 'Knowledge Base',
      sortable: true,
      render: (row) => (
        <div className={styles.primaryCell}>
          <div className={styles.primaryCellTitleRow}>
            <span className={styles.primaryCellTitle}>{row.name}</span>
            {selectedKnowledgeBase?.name === row.name ? <span className={styles.selectedBadge}>Selected</span> : null}
          </div>
          <span className={styles.primaryCellMeta}>{row.description || 'No description provided.'}</span>
        </div>
      ),
    },
    {
      key: 'modeLabel',
      header: 'Mode',
      sortable: true,
      render: (row) => <span className={styles.tableBadge}>{row.modeLabel}</span>,
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
          <button type="button" className={styles.secondaryButton} onClick={() => setSelectedKnowledgeBaseName(row.name)}>
            {selectedKnowledgeBase?.name === row.name ? 'Open' : 'View'}
          </button>
          {row.editable && !isReadonly ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => openEditKnowledgeBaseModal(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void handleDeleteKnowledgeBase(row)}>
                Delete
              </button>
            </>
          ) : null}
        </div>
      ),
    },
  ], [handleDeleteKnowledgeBase, isReadonly, openEditKnowledgeBaseModal, selectedKnowledgeBase?.name])

  const groupColumns = useMemo<Column<GroupRow>[]>(() => [
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
          {!isReadonly && selectedKnowledgeBase?.editable ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => openEditGroupModal(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void handleDeleteGroup(row)}>
                Delete
              </button>
            </>
          ) : (
            <span className={styles.readOnlyHint}>Read-only</span>
          )}
        </div>
      ),
    },
  ], [handleDeleteGroup, isReadonly, openEditGroupModal, selectedKnowledgeBase?.editable])

  const labelColumns = useMemo<Column<LabelRow>[]>(() => [
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
          {!isReadonly && selectedKnowledgeBase?.editable ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => openEditLabelModal(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void handleDeleteLabel(row)}>
                Delete
              </button>
            </>
          ) : (
            <span className={styles.readOnlyHint}>Read-only</span>
          )}
        </div>
      ),
    },
  ], [handleDeleteLabel, isReadonly, openEditLabelModal, selectedKnowledgeBase?.editable])

  const exemplarColumns = useMemo<Column<ExemplarRow>[]>(() => [
    {
      key: 'label',
      header: 'Label',
      sortable: true,
      render: (row) => <span className={styles.tableBadge}>{row.label}</span>,
    },
    {
      key: 'exemplar',
      header: 'Exemplar',
      render: (row) => (
        <div className={styles.primaryCell}>
          <span className={styles.primaryCellMeta}>{row.exemplar}</span>
        </div>
      ),
    },
    {
      key: 'actions',
      header: 'Actions',
      align: 'right',
      render: (row) => (
        <div className={styles.tableActionGroup}>
          {!isReadonly && selectedKnowledgeBase?.editable ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => openEditExemplarModal(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void handleDeleteExemplar(row)}>
                Delete
              </button>
            </>
          ) : (
            <span className={styles.readOnlyHint}>Read-only</span>
          )}
        </div>
      ),
    },
  ], [handleDeleteExemplar, isReadonly, openEditExemplarModal, selectedKnowledgeBase?.editable])

  return (
    <section id="knowledge-bases" className={styles.section}>
      <div className={styles.summaryGrid}>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Total Knowledge Bases</span>
          <strong className={styles.summaryValue}>{counts.total}</strong>
          <span className={styles.summaryHint}>All embedding-backed KB packages currently discoverable by the router.</span>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Built-In</span>
          <strong className={styles.summaryValue}>{counts.builtin}</strong>
          <span className={styles.summaryHint}>Router-shipped KBs that stay editable through the same CRUD surface.</span>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Managed</span>
          <strong className={styles.summaryValue}>{counts.managed}</strong>
          <span className={styles.summaryHint}>Config-managed KB packages currently tracked in `global.model_catalog.kbs[]`.</span>
        </article>
      </div>

      {loading ? <div className={styles.notice}>Loading knowledge base catalog...</div> : null}
      {error ? <div className={styles.error}>{error}</div> : null}

      <div className={pageStyles.sectionTableBlock}>
        <TableHeader
          title={activeView === 'knowledge-bases' ? 'Knowledge Base Catalog' : 'Knowledge Base Context'}
          count={knowledgeBaseRows.length}
          searchPlaceholder="Search KB name, description, or source"
          searchValue={knowledgeBaseSearch}
          onSearchChange={setKnowledgeBaseSearch}
          onSecondaryAction={() => {
            void loadKnowledgeBases()
          }}
          secondaryActionText="Refresh"
          onAdd={!isReadonly ? openCreateModal : undefined}
          addButtonText="Add Knowledge Base"
          variant="embedded"
        />
        <DataTable
          columns={knowledgeBaseColumns}
          data={knowledgeBaseRows}
          keyExtractor={(row) => row.name}
          emptyMessage="No knowledge bases found."
          className={pageStyles.managerTable}
        />
      </div>

      <ConfigPageTaxonomyClassifierDetail selectedClassifier={selectedKnowledgeBase} />

      {activeView === 'groups' ? (
        <div className={pageStyles.sectionTableBlock}>
          <TableHeader
            title={selectedKnowledgeBase ? `Groups · ${selectedKnowledgeBase.name}` : 'Groups'}
            count={groupRows.length}
            searchPlaceholder="Search groups or assigned labels"
            searchValue={groupSearch}
            onSearchChange={setGroupSearch}
            onAdd={!isReadonly && selectedKnowledgeBase?.editable ? openAddGroupModal : undefined}
            addButtonText="Add Group"
            variant="embedded"
          />
          <DataTable
            columns={groupColumns}
            data={groupRows}
            keyExtractor={(row) => row.name}
            emptyMessage={selectedKnowledgeBase ? 'No groups defined for this knowledge base.' : 'Select a knowledge base first.'}
            className={pageStyles.managerTable}
          />
        </div>
      ) : null}

      {activeView === 'labels' ? (
        <div className={pageStyles.sectionTableBlock}>
          <TableHeader
            title={selectedKnowledgeBase ? `Labels · ${selectedKnowledgeBase.name}` : 'Labels'}
            count={labelRows.length}
            searchPlaceholder="Search labels"
            searchValue={labelSearch}
            onSearchChange={setLabelSearch}
            onAdd={!isReadonly && selectedKnowledgeBase?.editable ? openAddLabelModal : undefined}
            addButtonText="Add Label"
            variant="embedded"
          />
          <DataTable
            columns={labelColumns}
            data={labelRows}
            keyExtractor={(row) => row.name}
            emptyMessage={selectedKnowledgeBase ? 'No labels defined for this knowledge base.' : 'Select a knowledge base first.'}
            className={pageStyles.managerTable}
          />
        </div>
      ) : null}

      {activeView === 'exemplars' ? (
        <div className={pageStyles.sectionTableBlock}>
          <TableHeader
            title={selectedKnowledgeBase ? `Exemplars · ${selectedKnowledgeBase.name}` : 'Exemplars'}
            count={exemplarRows.length}
            searchPlaceholder="Search exemplar text or label"
            searchValue={exemplarSearch}
            onSearchChange={setExemplarSearch}
            onAdd={!isReadonly && selectedKnowledgeBase?.editable ? openAddExemplarModal : undefined}
            addButtonText="Add Exemplar"
            variant="embedded"
          />
          <DataTable
            columns={exemplarColumns}
            data={exemplarRows}
            keyExtractor={(row) => row.key}
            emptyMessage={selectedKnowledgeBase ? 'No exemplars defined for this knowledge base.' : 'Select a knowledge base first.'}
            className={pageStyles.managerTable}
          />
        </div>
      ) : null}
    </section>
  )
}
