import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { FieldConfig } from '../components/EditModal'
import { DataTable } from '../components/DataTable'
import ConfirmDialog from '../components/ConfirmDialog'
import { StringListEditor } from '../components/StringListEditor'
import TableHeader from '../components/TableHeader'
import pageStyles from './ConfigPage.module.css'
import ConfigPageTaxonomyClassifierEditor from './ConfigPageTaxonomyClassifierEditor'
import ConfigPageTaxonomyClassifierDetail from './ConfigPageTaxonomyClassifierDetail'
import ConfigPageKnowledgeBasePicker from './ConfigPageKnowledgeBasePicker'
import type { OpenEditModal } from './configPageRouterSectionSupport'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  classifierDraftFromRecord,
  emptyTaxonomyClassifierDraft,
  normalizeTaxonomyClassifierListResponse,
  payloadFromDraft,
  type TaxonomyClassifierCategory,
  type TaxonomyClassifierDraft,
  type TaxonomyClassifierRecord,
} from './configPageTaxonomyClassifierSupport'
import {
  addGroupToDraft,
  addLabelToDraft,
  buildGroupColumns,
  buildGroupRows,
  buildKnowledgeBaseColumns,
  buildKnowledgeBaseCounts,
  buildKnowledgeBaseRows,
  buildLabelColumns,
  buildLabelRows,
  type GroupRow,
  type KnowledgeBaseGroupDraft,
  type KnowledgeBaseLabelDraft,
  type KnowledgeBaseManagerView,
  removeGroupFromDraft,
  removeLabelFromDraft,
  renameGroupInDraft,
  renameLabelInDraft,
} from './configPageKnowledgeBaseManagerSupport'
import { buildTaxonomySummaryCards } from './configPageTaxonomySummarySupport'

interface ConfigPageTaxonomyClassifiersProps {
  isReadonly: boolean
  openEditModal: OpenEditModal
  activeView?: KnowledgeBaseManagerView
}

type TaxonomyDeleteTarget =
  | { kind: 'knowledge-base'; knowledgeBase: TaxonomyClassifierRecord }
  | { kind: 'group'; group: GroupRow; knowledgeBaseName: string }
  | { kind: 'label'; label: TaxonomyClassifierCategory; knowledgeBaseName: string }

export default function ConfigPageTaxonomyClassifiers({
  isReadonly,
  openEditModal,
  activeView = 'bases',
}: ConfigPageTaxonomyClassifiersProps) {
  const navigate = useNavigate()
  const [knowledgeBases, setKnowledgeBases] = useState<TaxonomyClassifierRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedKnowledgeBaseName, setSelectedKnowledgeBaseName] = useState('')
  const [knowledgeBaseSearch, setKnowledgeBaseSearch] = useState('')
  const [groupSearch, setGroupSearch] = useState('')
  const [labelSearch, setLabelSearch] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<TaxonomyDeleteTarget | null>(null)
  const [deletePending, setDeletePending] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

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

  const knowledgeBaseRows = useMemo(
    () => buildKnowledgeBaseRows(knowledgeBases, knowledgeBaseSearch),
    [knowledgeBases, knowledgeBaseSearch],
  )

  const selectedKnowledgeBase = useMemo(
    () =>
      knowledgeBases.find((knowledgeBase) => knowledgeBase.name === selectedKnowledgeBaseName) ??
      knowledgeBaseRows[0] ??
      null,
    [knowledgeBaseRows, knowledgeBases, selectedKnowledgeBaseName],
  )

  const groupRows = useMemo(
    () => buildGroupRows(selectedKnowledgeBase, groupSearch),
    [groupSearch, selectedKnowledgeBase],
  )

  const labelRows = useMemo(
    () => buildLabelRows(selectedKnowledgeBase, labelSearch),
    [labelSearch, selectedKnowledgeBase],
  )

  const counts = useMemo(() => buildKnowledgeBaseCounts(knowledgeBases), [knowledgeBases])

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
    [],
  )

  const persistKnowledgeBase = useCallback(
    async (
      endpoint: string,
      method: 'POST' | 'PUT',
      draft: TaxonomyClassifierDraft,
      nextSelection?: string,
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
    [loadKnowledgeBases],
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
        selectedKnowledgeBase.name,
      )
    },
    [persistKnowledgeBase, selectedKnowledgeBase],
  )

  const openCreateModal = useCallback(() => {
    openEditModal<{ draft: TaxonomyClassifierDraft }>(
      'Add Knowledge Base',
      { draft: emptyTaxonomyClassifierDraft() },
      knowledgeBaseEditorField(false),
      async (data) => {
        const nextName = data.draft.name.trim()
        await persistKnowledgeBase('/api/router/config/kbs', 'POST', data.draft, nextName)
      },
      'add',
    )
  }, [knowledgeBaseEditorField, openEditModal, persistKnowledgeBase])

  const openEditKnowledgeBaseModal = useCallback(
    (knowledgeBase: TaxonomyClassifierRecord) => {
      openEditModal<{ draft: TaxonomyClassifierDraft }>(
        `Edit ${knowledgeBase.name}`,
        { draft: classifierDraftFromRecord(knowledgeBase) },
        knowledgeBaseEditorField(true),
        async (data) => {
          await persistKnowledgeBase(
            `/api/router/config/kbs/${knowledgeBase.name}`,
            'PUT',
            data.draft,
            knowledgeBase.name,
          )
        },
        'edit',
      )
    },
    [knowledgeBaseEditorField, openEditModal, persistKnowledgeBase],
  )

  const handleDeleteKnowledgeBase = useCallback((knowledgeBase: TaxonomyClassifierRecord) => {
    setDeleteError(null)
    setDeleteTarget({ kind: 'knowledge-base', knowledgeBase })
  }, [])

  const openAddGroupModal = useCallback(() => {
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
      'add',
    )
  }, [groupRows, openEditModal, persistSelectedKnowledgeBase, selectedKnowledgeBase])

  const openEditGroupModal = useCallback(
    (group: GroupRow) => {
      if (!selectedKnowledgeBase) {
        return
      }
      openEditModal<KnowledgeBaseGroupDraft>(
        `Edit Group · ${group.name}`,
        { name: group.name, labels: group.labels },
        [
          {
            name: 'name',
            label: 'Group Name',
            type: 'text',
            required: true,
            placeholder: 'private',
          },
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
          if (
            nextName !== group.name &&
            groupRows.some((existingGroup) => existingGroup.name === nextName)
          ) {
            throw new Error(`Group "${nextName}" already exists.`)
          }
          if (nextName !== group.name && group.signal_count > 0) {
            throw new Error(
              `Group "${group.name}" is referenced by KB signals. Update those signals before renaming it.`,
            )
          }
          await persistSelectedKnowledgeBase((draft) => renameGroupInDraft(draft, group.name, data))
        },
        'edit',
      )
    },
    [groupRows, openEditModal, persistSelectedKnowledgeBase, selectedKnowledgeBase],
  )

  const handleDeleteGroup = useCallback(
    (group: GroupRow) => {
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
      setDeleteError(null)
      setDeleteTarget({
        kind: 'group',
        group,
        knowledgeBaseName: selectedKnowledgeBase.name,
      })
    },
    [selectedKnowledgeBase],
  )

  const openAddLabelModal = useCallback(() => {
    if (!selectedKnowledgeBase) {
      return
    }
    openEditModal<KnowledgeBaseLabelDraft>(
      `Add Label · ${selectedKnowledgeBase.name}`,
      { name: '', description: '', exemplars: [] },
      [
        {
          name: 'name',
          label: 'Label Name',
          type: 'text',
          required: true,
          placeholder: 'prompt_injection',
        },
        {
          name: 'description',
          label: 'Description',
          type: 'textarea',
          placeholder: 'What this label captures.',
        },
        {
          name: 'exemplars',
          label: 'Exemplars',
          type: 'custom',
          required: true,
          customRender: (value, onChange) => (
            <StringListEditor
              value={
                Array.isArray(value)
                  ? value.filter((item): item is string => typeof item === 'string')
                  : []
              }
              onChange={onChange}
              addLabel="Add exemplar"
              emptyLabel="Add at least one representative exemplar."
              itemLabel="Exemplar"
              placeholder="A representative request for this label"
            />
          ),
        },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (selectedKnowledgeBase.labels.some((label) => label.name === nextName)) {
          throw new Error(`Label "${nextName}" already exists.`)
        }
        await persistSelectedKnowledgeBase((draft) => addLabelToDraft(draft, data))
      },
      'add',
    )
  }, [openEditModal, persistSelectedKnowledgeBase, selectedKnowledgeBase])

  const openEditLabelModal = useCallback(
    (label: TaxonomyClassifierCategory) => {
      if (!selectedKnowledgeBase) {
        return
      }
      openEditModal<KnowledgeBaseLabelDraft>(
        `Edit Label · ${label.name}`,
        {
          name: label.name,
          description: label.description ?? '',
          exemplars: [...label.exemplars],
        },
        [
          {
            name: 'name',
            label: 'Label Name',
            type: 'text',
            required: true,
            placeholder: 'prompt_injection',
          },
          {
            name: 'description',
            label: 'Description',
            type: 'textarea',
            placeholder: 'What this label captures.',
          },
          {
            name: 'exemplars',
            label: 'Exemplars',
            type: 'custom',
            required: true,
            customRender: (value, onChange) => (
              <StringListEditor
                value={
                  Array.isArray(value)
                    ? value.filter((item): item is string => typeof item === 'string')
                    : []
                }
                onChange={onChange}
                addLabel="Add exemplar"
                emptyLabel="Add at least one representative exemplar."
                itemLabel="Exemplar"
                placeholder="A representative request for this label"
              />
            ),
          },
        ],
        async (data) => {
          const nextName = data.name.trim()
          if (
            nextName !== label.name &&
            selectedKnowledgeBase.labels.some((existingLabel) => existingLabel.name === nextName)
          ) {
            throw new Error(`Label "${nextName}" already exists.`)
          }
          if (
            nextName !== label.name &&
            labelRows.some((entry) => entry.name === label.name && entry.signal_count > 0)
          ) {
            throw new Error(
              `Label "${label.name}" is referenced by KB signals. Update those signals before renaming it.`,
            )
          }
          await persistSelectedKnowledgeBase((draft) => renameLabelInDraft(draft, label.name, data))
        },
        'edit',
      )
    },
    [labelRows, openEditModal, persistSelectedKnowledgeBase, selectedKnowledgeBase],
  )

  const handleDeleteLabel = useCallback(
    (label: TaxonomyClassifierCategory) => {
      if (!selectedKnowledgeBase) {
        return
      }
      const signalCount = labelRows.find((entry) => entry.name === label.name)?.signal_count ?? 0
      if (signalCount > 0) {
        setError(`Label "${label.name}" is referenced by KB signals. Remove those signals first.`)
        return
      }
      setDeleteError(null)
      setDeleteTarget({
        kind: 'label',
        label,
        knowledgeBaseName: selectedKnowledgeBase.name,
      })
    },
    [labelRows, selectedKnowledgeBase],
  )

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget || deletePending) return

    setDeletePending(true)
    setDeleteError(null)
    setError(null)
    try {
      if (deleteTarget.kind === 'knowledge-base') {
        const { knowledgeBase } = deleteTarget
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
      } else if (deleteTarget.kind === 'group') {
        await persistSelectedKnowledgeBase((draft) =>
          removeGroupFromDraft(draft, deleteTarget.group.name),
        )
      } else {
        await persistSelectedKnowledgeBase((draft) =>
          removeLabelFromDraft(draft, deleteTarget.label.name),
        )
      }
      setDeleteTarget(null)
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete item')
    } finally {
      setDeletePending(false)
    }
  }, [
    deletePending,
    deleteTarget,
    loadKnowledgeBases,
    persistSelectedKnowledgeBase,
    selectedKnowledgeBaseName,
  ])

  const knowledgeBaseColumns = useMemo(
    () =>
      buildKnowledgeBaseColumns({
        selectedKnowledgeBaseName: selectedKnowledgeBase?.name,
        isReadonly,
        onSelect: setSelectedKnowledgeBaseName,
        onOpenMap: (name) => navigate(`/knowledge-bases/${encodeURIComponent(name)}/map`),
        onEdit: openEditKnowledgeBaseModal,
        onDelete: handleDeleteKnowledgeBase,
      }),
    [
      handleDeleteKnowledgeBase,
      isReadonly,
      navigate,
      openEditKnowledgeBaseModal,
      selectedKnowledgeBase?.name,
    ],
  )

  const groupColumns = useMemo(
    () =>
      buildGroupColumns({
        isReadonly,
        editable: selectedKnowledgeBase?.editable,
        onEdit: openEditGroupModal,
        onDelete: handleDeleteGroup,
      }),
    [handleDeleteGroup, isReadonly, openEditGroupModal, selectedKnowledgeBase?.editable],
  )

  const labelColumns = useMemo(
    () =>
      buildLabelColumns({
        isReadonly,
        editable: selectedKnowledgeBase?.editable,
        onEdit: openEditLabelModal,
        onDelete: handleDeleteLabel,
      }),
    [handleDeleteLabel, isReadonly, openEditLabelModal, selectedKnowledgeBase?.editable],
  )

  const groupOverview = useMemo(
    () => ({
      total: groupRows.length,
      referenced: groupRows.filter((group) => group.signal_count > 0).length,
      metricBacked: groupRows.filter((group) => group.metric_count > 0).length,
    }),
    [groupRows],
  )

  const labelOverview = useMemo(
    () => ({
      total: labelRows.length,
      referenced: labelRows.filter((label) => label.signal_count > 0).length,
      overrides: labelRows.filter((label) => typeof label.threshold_value === 'number').length,
    }),
    [labelRows],
  )

  const activeSummaryCards = useMemo(
    () =>
      buildTaxonomySummaryCards({
        activeView,
        counts: { total: counts.total, builtin: counts.builtin, custom: counts.custom },
        groupOverview,
        labelOverview,
      }),
    [activeView, counts.builtin, counts.custom, counts.total, groupOverview, labelOverview],
  )

  const deleteTargetName =
    deleteTarget?.kind === 'knowledge-base'
      ? deleteTarget.knowledgeBase.name
      : deleteTarget?.kind === 'group'
        ? deleteTarget.group.name
        : (deleteTarget?.label.name ?? '')
  const deleteTargetKnowledgeBaseName =
    deleteTarget?.kind === 'group' || deleteTarget?.kind === 'label'
      ? deleteTarget.knowledgeBaseName
      : ''

  return (
    <section id="knowledge-bases" className={styles.section}>
      <div className={styles.summaryGrid}>
        {activeSummaryCards.map((card) => (
          <article key={card.label} className={styles.summaryCard}>
            <span className={styles.summaryLabel}>{card.label}</span>
            <strong className={styles.summaryValue}>{card.value}</strong>
            <span className={styles.summaryHint}>{card.hint}</span>
          </article>
        ))}
      </div>

      {loading ? <div className={styles.notice}>Loading knowledge base catalog...</div> : null}
      {error ? <div className={styles.error}>{error}</div> : null}

      {activeView === 'bases' ? (
        <>
          <div className={pageStyles.sectionTableBlock}>
            <TableHeader
              title="Knowledge Catalog"
              count={knowledgeBaseRows.length}
              searchPlaceholder="Search knowledge base name, description, or source"
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
              pagination={{
                pageSize: 10,
                pageSizeOptions: [10, 25, 50],
                itemLabel: 'knowledge bases',
                resetKey: knowledgeBaseSearch,
              }}
            />
          </div>

          <ConfigPageTaxonomyClassifierDetail
            selectedClassifier={selectedKnowledgeBase}
            onOpenMap={(name) => navigate(`/knowledge-bases/${encodeURIComponent(name)}/map`)}
          />
        </>
      ) : (
        <div className={pageStyles.sectionTableBlock}>
          <ConfigPageKnowledgeBasePicker
            knowledgeBases={knowledgeBases}
            selectedKnowledgeBase={selectedKnowledgeBase}
            onSelect={setSelectedKnowledgeBaseName}
            onRefresh={() => {
              void loadKnowledgeBases()
            }}
          />
        </div>
      )}

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
            emptyMessage={
              selectedKnowledgeBase
                ? 'No groups defined for this knowledge base.'
                : 'Select a knowledge base first.'
            }
            className={pageStyles.managerTable}
            pagination={{
              pageSize: 10,
              pageSizeOptions: [10, 25, 50],
              itemLabel: 'groups',
              resetKey: `${selectedKnowledgeBaseName}:${groupSearch}`,
            }}
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
            emptyMessage={
              selectedKnowledgeBase
                ? 'No labels defined for this knowledge base.'
                : 'Select a knowledge base first.'
            }
            className={pageStyles.managerTable}
            pagination={{
              pageSize: 10,
              pageSizeOptions: [10, 25, 50],
              itemLabel: 'labels',
              resetKey: `${selectedKnowledgeBaseName}:${labelSearch}`,
            }}
          />
        </div>
      ) : null}

      <ConfirmDialog
        isOpen={Boolean(deleteTarget)}
        title={
          deleteTarget?.kind === 'knowledge-base'
            ? 'Delete knowledge base'
            : deleteTarget?.kind === 'group'
              ? 'Delete group'
              : 'Delete label'
        }
        description={
          deleteTarget?.kind === 'knowledge-base' ? (
            <>
              Delete <strong>{deleteTargetName}</strong> and its managed groups, labels, and
              exemplars?
            </>
          ) : (
            <>
              Delete <strong>{deleteTargetName}</strong> from{' '}
              <strong>{deleteTargetKnowledgeBaseName}</strong>?
            </>
          )
        }
        details={deleteError ? <span role="alert">{deleteError}</span> : undefined}
        confirmLabel="Delete"
        confirmationText={deleteTarget?.kind === 'knowledge-base' ? deleteTargetName : undefined}
        pending={deletePending}
        onCancel={() => {
          if (!deletePending) {
            setDeleteTarget(null)
            setDeleteError(null)
          }
        }}
        onConfirm={confirmDelete}
      />
    </section>
  )
}
