import React from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import { DataTable, type Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import { formatRoutingMetadataValue } from '../components/routingMetadataDisplay'
import RoutingScopeSelector from '../components/RoutingScopeSelector'
import TableHeader from '../components/TableHeader'
import type { ViewSection } from '../components/ViewModal'
import type { SignalType as CanonicalSignalType } from '../lib/dslSchemaCatalogs'
import { getSignalFieldSchema } from '../lib/dslSchemas'
import { hasFlatSignals } from '../types/config'
import styles from './ConfigPage.module.css'
import signalStyles from './ConfigPageSignalsSection.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import { requiredSchemaFieldErrors } from './configPageSchemaValidation'
import { cloneConfigData } from './configPageCanonicalization'
import { SIGNAL_CATALOG, signalCatalogByType } from './configPageSignalCatalog'
import { getSignalReferenceCountInRoutingProfile } from './configPageSignalReferences'
import { useRoutingScopeManager } from './configPageRoutingScopeSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import type { ConfigData, ConfigSignals, SignalType } from './configPageSupport'

interface ConfigPageSignalsSectionProps {
  config: ConfigData | null
  isPythonCLI: boolean
  isReadonly: boolean
  signalsSearch: string
  onSignalsSearchChange: (value: string) => void
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
}

interface SignalRecord extends Record<string, unknown> {
  name: string
}

interface ManagedSignal {
  name: string
  type: SignalType
  canonicalType: CanonicalSignalType
  collection: keyof ConfigSignals
  summary: string
  rawData: SignalRecord
}

interface SignalDefinition {
  type: CanonicalSignalType
  fields: Record<string, unknown>
}

interface SignalFormState {
  name: string
  definition: SignalDefinition
}

function asSignalRecords(value: unknown): SignalRecord[] {
  if (!Array.isArray(value)) return []
  return value.filter(
    (entry): entry is SignalRecord =>
      Boolean(entry) &&
      typeof entry === 'object' &&
      typeof (entry as SignalRecord).name === 'string',
  )
}

function readSignalCollection(signals: ConfigSignals | undefined, key: keyof ConfigSignals) {
  return asSignalRecords(signals?.[key])
}

function writeSignalCollection(
  signals: ConfigSignals,
  key: keyof ConfigSignals,
  records: SignalRecord[],
) {
  ;(signals as unknown as Record<string, SignalRecord[]>)[key] = records
}

function withoutName(record: SignalRecord): Record<string, unknown> {
  const fields: Record<string, unknown> = { ...record }
  delete fields.name
  return fields
}

function summarizeSignal(record: SignalRecord): string {
  if (typeof record.description === 'string' && record.description.trim()) return record.description
  if (typeof record.threshold === 'number') return `Threshold ${record.threshold}`
  for (const [key, value] of Object.entries(record)) {
    if (key === 'name' || !Array.isArray(value)) continue
    return `${value.length} ${key.replace(/_/g, ' ')}`
  }
  const configured = Object.keys(record).filter((key) => key !== 'name').length
  return configured > 0 ? `${configured} configured fields` : 'Uses default behavior'
}

function buildLegacySignals(config: ConfigData | null): ConfigSignals | undefined {
  if (!hasFlatSignals(config)) return undefined
  return {
    keywords: config?.keyword_rules,
    embeddings: config?.embedding_rules,
    domains: (config?.categories || []).map((category) => ({
      name: category.name,
      description: category.description || '',
      mmlu_categories: category.mmlu_categories,
      model_scores: Array.isArray(category.model_scores) ? category.model_scores : undefined,
    })),
    fact_check: config?.fact_check_rules,
    user_feedbacks: config?.user_feedback_rules,
    reasks: config?.reask_rules,
    preferences: config?.preference_rules,
    language: config?.language_rules,
    context: config?.context_rules,
    structure: config?.structure_rules,
    complexity: config?.complexity_rules,
    jailbreak: config?.jailbreak,
    hallucination: config?.hallucination,
    pii: config?.pii,
  }
}

function SignalDefinitionEditor({
  value,
  onChange,
}: {
  value: SignalDefinition
  onChange: (value: SignalDefinition) => void
}) {
  return (
    <div className={signalStyles.definitionEditor}>
      <label className={signalStyles.definitionType}>
        <span>Signal type</span>
        <select
          value={value.type}
          onChange={(event) =>
            onChange({ type: event.target.value as CanonicalSignalType, fields: {} })
          }
        >
          {SIGNAL_CATALOG.map((candidate) => (
            <option key={candidate.type} value={candidate.type}>
              {candidate.label}
            </option>
          ))}
        </select>
      </label>
      <ConfigPageSchemaFieldsEditor
        schema={getSignalFieldSchema(value.type)}
        value={value.fields}
        onChange={(fields) => onChange({ ...value, fields })}
      />
    </div>
  )
}

export default function ConfigPageSignalsSection({
  config,
  isPythonCLI,
  isReadonly,
  signalsSearch,
  onSignalsSearchChange,
  saveConfig,
  openEditModal,
  openViewModal,
}: ConfigPageSignalsSectionProps) {
  const [selectedSignalKeys, setSelectedSignalKeys] = React.useState<Set<string>>(new Set())
  const [signalsPendingDelete, setSignalsPendingDelete] = React.useState<ManagedSignal[]>([])
  const [deletePending, setDeletePending] = React.useState(false)
  const [deleteError, setDeleteError] = React.useState<string | null>(null)
  const [actionError, setActionError] = React.useState<string | null>(null)
  const {
    applyScopedConfig,
    routingScopes,
    scopedConfig,
    selectedScope,
    selectedScopeId,
    setSelectedScopeId,
  } = useRoutingScopeManager(config)

  React.useEffect(() => {
    setSelectedSignalKeys(new Set())
    setSignalsPendingDelete([])
    setActionError(null)
  }, [selectedScopeId])

  const effectiveSignals = scopedConfig?.signals || buildLegacySignals(config)
  const allSignals = SIGNAL_CATALOG.flatMap<ManagedSignal>((entry) =>
    readSignalCollection(effectiveSignals, entry.collection).map((rawData) => ({
      name: rawData.name,
      type: entry.label,
      canonicalType: entry.type,
      collection: entry.collection,
      summary: summarizeSignal(rawData),
      rawData,
    })),
  )
  const search = signalsSearch.trim().toLowerCase()
  const filteredSignals = allSignals.filter((signal) =>
    [signal.name, signal.type, signal.canonicalType, signal.summary]
      .join(' ')
      .toLowerCase()
      .includes(search),
  )
  const signalKey = (signal: ManagedSignal) => `${signal.canonicalType}-${signal.name}`
  const signalReferenceCount = (signal: ManagedSignal) =>
    getSignalReferenceCountInRoutingProfile(
      (selectedScope ?? routingScopes[0])?.routing as ConfigData['routing'],
      signal.type,
      signal.name,
    )

  const signalColumns: Column<ManagedSignal>[] = [
    {
      key: 'name',
      header: 'Name',
      sortable: true,
      render: (signal) => (
        <span style={{ fontWeight: 600 }}>
          {formatRoutingMetadataValue(`x-vsr-matched-${signal.canonicalType}`, signal.name)}
        </span>
      ),
    },
    {
      key: 'type',
      header: 'Type',
      width: '170px',
      sortable: true,
      render: (signal) => <span className={styles.tableMetaBadge}>{signal.type}</span>,
    },
    { key: 'summary', header: 'Summary', render: (signal) => signal.summary },
  ]

  const openSignalEditor = (mode: 'add' | 'edit', signal?: ManagedSignal) => {
    const initialData: SignalFormState = {
      name: signal?.name || '',
      definition: {
        type: signal?.canonicalType || 'keyword',
        fields: signal ? withoutName(signal.rawData) : {},
      },
    }
    const fields: FieldConfig<SignalFormState>[] = [
      { name: 'name', label: 'Name', type: 'text', required: true, placeholder: 'my_signal' },
      {
        name: 'definition',
        label: 'Configuration',
        type: 'custom',
        required: true,
        description: 'Fields are driven by the same canonical signal schema used by Builder.',
        customRender: (value, onChange) => (
          <SignalDefinitionEditor
            value={(value as SignalDefinition | undefined) || initialData.definition}
            onChange={(next) => onChange(next)}
          />
        ),
      },
    ]

    openEditModal<SignalFormState>(
      mode === 'add' ? 'Add Signal' : `Edit Signal: ${signal?.name}`,
      initialData,
      fields,
      async (formData) => {
        if (!config || !scopedConfig) throw new Error('Routing profile not loaded yet.')
        if (!isPythonCLI) throw new Error('Editing signals requires the canonical config format.')
        const name = formData.name.trim()
        if (!name) throw new Error('Name is required.')
        const definition = formData.definition
        const catalog = signalCatalogByType(definition.type)
        const schemaErrors = requiredSchemaFieldErrors(
          getSignalFieldSchema(definition.type),
          definition.fields,
        )
        if (schemaErrors.length > 0) throw new Error(schemaErrors[0])

        const next = cloneConfigData(scopedConfig)
        next.signals ||= {}
        if (signal) {
          writeSignalCollection(
            next.signals,
            signal.collection,
            readSignalCollection(next.signals, signal.collection).filter(
              (entry) => entry.name !== signal.name,
            ),
          )
        }
        const target = readSignalCollection(next.signals, catalog.collection)
        if (target.some((entry) => entry.name === name)) {
          throw new Error(`${catalog.label} signal “${name}” already exists.`)
        }
        writeSignalCollection(next.signals, catalog.collection, [
          ...target,
          { name, ...definition.fields },
        ])
        await saveConfig(applyScopedConfig(next))
      },
      mode,
    )
  }

  const handleViewSignal = (signal: ManagedSignal) => {
    const sections: ViewSection[] = [
      {
        title: 'Basic Information',
        fields: [
          { label: 'Name', value: signal.name },
          { label: 'Type', value: `${signal.type} (${signal.canonicalType})` },
          { label: 'Summary', value: signal.summary, fullWidth: true },
        ],
      },
      {
        title: 'Complete Configuration',
        fields: [
          {
            label: 'Fields',
            value: (
              <ConfigPageSchemaFieldsEditor
                schema={getSignalFieldSchema(signal.canonicalType)}
                value={withoutName(signal.rawData)}
                readOnly
              />
            ),
            fullWidth: true,
          },
        ],
      },
    ]
    openViewModal(`Signal: ${signal.name}`, sections, () => openSignalEditor('edit', signal))
  }

  const removeManagedSignal = (next: ConfigData, signal: ManagedSignal) => {
    next.signals ||= {}
    writeSignalCollection(
      next.signals,
      signal.collection,
      readSignalCollection(next.signals, signal.collection).filter(
        (entry) => entry.name !== signal.name,
      ),
    )
  }

  const handleDeleteSignal = (signal: ManagedSignal) => {
    const references = signalReferenceCount(signal)
    if (references > 0) {
      setActionError(
        `Signal “${signal.name}” has ${references} active reference${references === 1 ? '' : 's'}. Update them before deleting it.`,
      )
      return
    }
    setActionError(null)
    setDeleteError(null)
    setSignalsPendingDelete([signal])
  }

  const handleBulkDeleteSignals = () => {
    const selected = allSignals.filter((signal) => selectedSignalKeys.has(signalKey(signal)))
    if (selected.some((signal) => signalReferenceCount(signal) > 0)) {
      setActionError(
        'One or more selected signals are referenced. Refresh the selection and retry.',
      )
      return
    }
    setActionError(null)
    setDeleteError(null)
    setSignalsPendingDelete(selected)
  }

  const confirmDeleteSignals = async () => {
    if (!scopedConfig || signalsPendingDelete.length === 0 || deletePending) return
    setDeletePending(true)
    setDeleteError(null)
    try {
      const next = cloneConfigData(scopedConfig)
      signalsPendingDelete.forEach((signal) => removeManagedSignal(next, signal))
      await saveConfig(applyScopedConfig(next))
      setSelectedSignalKeys(new Set())
      setSignalsPendingDelete([])
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : 'Failed to delete signals.')
    } finally {
      setDeletePending(false)
    }
  }

  return (
    <ConfigPageManagerLayout
      title="Signals"
      description="Manage every canonical signal type and its complete routing configuration."
    >
      <div className={styles.sectionPanel}>
        {actionError ? (
          <div className={styles.error} role="alert">
            {actionError}
          </div>
        ) : null}
        <div className={styles.sectionTableBlock}>
          <RoutingScopeSelector
            scopes={routingScopes}
            value={selectedScopeId}
            onChange={setSelectedScopeId}
          />
          <TableHeader
            title="Signals"
            count={filteredSignals.length}
            searchPlaceholder="Search signals..."
            searchValue={signalsSearch}
            onSearchChange={onSignalsSearchChange}
            onAdd={() => openSignalEditor('add')}
            addButtonText="Add Signal"
            disabled={isReadonly || !isPythonCLI}
            variant="embedded"
          />
          {selectedSignalKeys.size > 0 ? (
            <div className={signalStyles.bulkBar} role="status">
              <div className={signalStyles.bulkCopy}>
                <strong>{selectedSignalKeys.size} selected</strong>
                <span className={signalStyles.bulkHint}>Referenced signals cannot be deleted.</span>
              </div>
              <div className={signalStyles.bulkActions}>
                <button
                  type="button"
                  className={signalStyles.clearButton}
                  onClick={() => setSelectedSignalKeys(new Set())}
                >
                  Clear
                </button>
                <button
                  type="button"
                  className={signalStyles.deleteButton}
                  onClick={handleBulkDeleteSignals}
                >
                  Delete selected
                </button>
              </div>
            </div>
          ) : null}
          <DataTable
            columns={signalColumns}
            data={filteredSignals}
            keyExtractor={signalKey}
            onView={handleViewSignal}
            onEdit={(signal) => openSignalEditor('edit', signal)}
            onDelete={handleDeleteSignal}
            emptyMessage={search ? 'No signals match your search' : 'No signals configured'}
            className={styles.managerTable}
            readonly={isReadonly || !isPythonCLI}
            pagination={{
              pageSize: 25,
              pageSizeOptions: [10, 25, 50],
              itemLabel: 'signals',
              resetKey: `${selectedScopeId}:${signalsSearch}`,
            }}
            selection={
              !isReadonly && isPythonCLI
                ? {
                    selectedKeys: selectedSignalKeys,
                    onChange: setSelectedSignalKeys,
                    isRowDisabled: (signal) => signalReferenceCount(signal) > 0,
                    label: 'signal',
                  }
                : undefined
            }
          />
        </div>
      </div>
      <ConfirmDialog
        isOpen={signalsPendingDelete.length > 0}
        title={signalsPendingDelete.length === 1 ? 'Delete signal' : 'Delete signals'}
        description={`Delete ${signalsPendingDelete.length === 1 ? `“${signalsPendingDelete[0]?.name}”` : `${signalsPendingDelete.length} selected signals`} from this routing profile?`}
        details={deleteError ? <span role="alert">{deleteError}</span> : undefined}
        confirmLabel={signalsPendingDelete.length === 1 ? 'Delete signal' : 'Delete signals'}
        pending={deletePending}
        onCancel={() => {
          if (!deletePending) {
            setSignalsPendingDelete([])
            setDeleteError(null)
          }
        }}
        onConfirm={confirmDeleteSignals}
      />
    </ConfigPageManagerLayout>
  )
}
