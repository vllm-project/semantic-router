import type { ReactNode } from 'react'

import { DataTable, type Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import configStyles from './ConfigPage.module.css'
import styles from './ConfigPageModelsSection.module.css'
import ConfigPageModelLiveVerification from './ConfigPageModelLiveVerification'
import ModelProviderLogo from './ModelProviderLogo'
import type { NormalizedModel } from './configPageSupport'
import type { ModelEndpointFilter, ModelRoleFilter } from './configPageModelInventory'
import {
  modelLiveVerificationState,
  type ModelLiveVerificationState,
} from './useModelLiveVerification'
import { findModelProviderPreset } from './modelProviderCatalog'

function ModelProviderMark({ model }: { model: NormalizedModel }) {
  const backend = model.backend_refs?.[0]
  const preset = findModelProviderPreset({
    backendName: backend?.name,
    baseUrl: backend?.base_url,
    apiFormat: model.api_format ?? backend?.provider,
  })
  return (
    <span className={styles.modelProviderMark} aria-hidden="true">
      <ModelProviderLogo provider={preset} size="small" />
    </span>
  )
}

interface ConfigPageModelInventoryPanelProps {
  models: NormalizedModel[]
  filteredModels: NormalizedModel[]
  defaultModel: string
  modelReferenceCounts: ReadonlyMap<string, number>
  modelsSearch: string
  onModelsSearchChange: (value: string) => void
  reasoningFamilyFilter: string
  onReasoningFamilyFilterChange: (value: string) => void
  reasoningFamilyOptions: string[]
  endpointFilter: ModelEndpointFilter
  onEndpointFilterChange: (value: ModelEndpointFilter) => void
  roleFilter: ModelRoleFilter
  onRoleFilterChange: (value: ModelRoleFilter) => void
  filtersActive: boolean
  onClearFilters: () => void
  isReadonly: boolean
  selectedModelKeys: ReadonlySet<string>
  onSelectedModelKeysChange: (value: Set<string>) => void
  onClearSelection: () => void
  onDeleteSelected: () => void
  operationError: string | null
  onDismissOperationError: () => void
  onAddModel: () => void
  onViewModel: (model: NormalizedModel) => void
  onEditModel: (model: NormalizedModel) => void
  onDeleteModel: (model: NormalizedModel) => void
  expandedModels: ReadonlySet<string>
  onToggleExpand: (model: NormalizedModel) => void
  renderExpandedRow: (model: NormalizedModel) => ReactNode
  getDeleteBlocker: (modelName: string) => string | null
  liveVerificationStates: ReadonlyMap<string, ModelLiveVerificationState>
  onVerifyModel: (modelName: string) => void
  canVerifyModels: boolean
}

export default function ConfigPageModelInventoryPanel({
  models,
  filteredModels,
  defaultModel,
  modelReferenceCounts,
  modelsSearch,
  onModelsSearchChange,
  reasoningFamilyFilter,
  onReasoningFamilyFilterChange,
  reasoningFamilyOptions,
  endpointFilter,
  onEndpointFilterChange,
  roleFilter,
  onRoleFilterChange,
  filtersActive,
  onClearFilters,
  isReadonly,
  selectedModelKeys,
  onSelectedModelKeysChange,
  onClearSelection,
  onDeleteSelected,
  operationError,
  onDismissOperationError,
  onAddModel,
  onViewModel,
  onEditModel,
  onDeleteModel,
  expandedModels,
  onToggleExpand,
  renderExpandedRow,
  getDeleteBlocker,
  liveVerificationStates,
  onVerifyModel,
  canVerifyModels,
}: ConfigPageModelInventoryPanelProps) {
  const columns: Column<NormalizedModel>[] = [
    {
      key: 'name',
      header: 'Model Name',
      width: '300px',
      minWidth: '260px',
      sortable: true,
      render: (row) => (
        <div className={styles.modelIdentity}>
          <ModelProviderMark model={row} />
          <div className={styles.modelIdentityCopy}>
            <div className={styles.modelIdentityPrimary}>
              <span className={styles.modelName} title={row.name}>
                {row.name}
              </span>
              {row.name === defaultModel ? (
                <span className={styles.defaultBadge}>Default</span>
              ) : null}
            </div>
            <span className={styles.modelPhysicalId} title={row.provider_model_id || row.name}>
              {row.provider_model_id || row.name}
            </span>
          </div>
        </div>
      ),
    },
    {
      key: 'references',
      header: 'Routing Use',
      width: '120px',
      minWidth: '112px',
      align: 'center',
      render: (row) => {
        const references = modelReferenceCounts.get(row.name) ?? 0
        return references > 0 ? (
          <span className={styles.referenceBadge}>
            {references} {references === 1 ? 'decision' : 'decisions'}
          </span>
        ) : (
          <span className={styles.unusedLabel}>Unused</span>
        )
      },
    },
    {
      key: 'reasoning_family',
      header: 'Reasoning Family',
      width: '150px',
      minWidth: '136px',
      sortable: true,
      render: (row) =>
        row.reasoning_family ? (
          <span className={configStyles.tableMetaBadge}>{row.reasoning_family}</span>
        ) : (
          <span style={{ color: 'var(--color-text-secondary)' }}>N/A</span>
        ),
    },
    {
      key: 'endpoints',
      header: 'Endpoints',
      width: '112px',
      minWidth: '104px',
      align: 'center',
      render: (row) => {
        const count = row.endpoints?.length || 0
        return (
          <span style={{ color: count > 0 ? 'var(--color-text)' : 'var(--color-text-secondary)' }}>
            {count} {count === 1 ? 'endpoint' : 'endpoints'}
          </span>
        )
      },
    },
    {
      key: 'live_verification',
      header: 'Live',
      width: '164px',
      minWidth: '156px',
      render: (row) => (
        <ConfigPageModelLiveVerification
          model={row.name}
          hasBackend={Boolean(row.endpoints?.some((endpoint) => endpoint.endpoint.trim()))}
          allowed={canVerifyModels}
          state={modelLiveVerificationState(liveVerificationStates, row.name)}
          onVerify={() => onVerifyModel(row.name)}
        />
      ),
    },
    {
      key: 'pricing',
      header: 'Pricing',
      width: '132px',
      minWidth: '124px',
      render: (row) => {
        if (!row.pricing) return <span style={{ color: 'var(--color-text-secondary)' }}>N/A</span>
        const currency = row.pricing.currency || 'USD'
        const prompt = row.pricing.prompt_per_1m?.toFixed(2) || '0.00'
        return (
          <span
            title={`${prompt} ${currency} per 1M input tokens`}
            style={{
              fontSize: '0.8125rem',
              fontFamily: 'var(--font-mono)',
              whiteSpace: 'nowrap',
            }}
          >
            {currency === 'USD' ? '$' : `${currency} `}
            {prompt} / 1M
          </span>
        )
      },
    },
  ]
  const visibleColumns = canVerifyModels
    ? columns
    : columns.filter((column) => column.key !== 'live_verification')

  return (
    <>
      <TableHeader
        title="Models"
        count={models.length}
        searchPlaceholder="Search name, ID, family, tag, or capability..."
        searchValue={modelsSearch}
        onSearchChange={onModelsSearchChange}
        onAdd={onAddModel}
        addButtonText="Add Model"
        disabled={isReadonly}
        variant="embedded"
      />

      <div className={styles.inventoryToolbar} aria-label="Model inventory filters">
        <label className={styles.filterControl}>
          <span>Reasoning family</span>
          <select
            value={reasoningFamilyFilter}
            onChange={(event) => onReasoningFamilyFilterChange(event.target.value)}
          >
            <option value="all">All families</option>
            <option value="__unassigned__">Unassigned</option>
            {reasoningFamilyOptions.map((family) => (
              <option key={family} value={family}>
                {family}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.filterControl}>
          <span>Endpoint state</span>
          <select
            value={endpointFilter}
            onChange={(event) => onEndpointFilterChange(event.target.value as ModelEndpointFilter)}
          >
            <option value="all">All endpoints</option>
            <option value="configured">Configured</option>
            <option value="missing">Missing</option>
          </select>
        </label>
        <label className={styles.filterControl}>
          <span>Routing role</span>
          <select
            value={roleFilter}
            onChange={(event) => onRoleFilterChange(event.target.value as ModelRoleFilter)}
          >
            <option value="all">All roles</option>
            <option value="default">Default model</option>
            <option value="standard">Standard model</option>
          </select>
        </label>
        <div className={styles.inventorySummary} aria-live="polite">
          <strong>{filteredModels.length}</strong>
          <span>of {models.length} models</span>
        </div>
        {filtersActive ? (
          <button type="button" className={styles.clearFiltersButton} onClick={onClearFilters}>
            Clear filters
          </button>
        ) : null}
      </div>

      {!isReadonly && selectedModelKeys.size > 0 ? (
        <div className={styles.bulkBar} role="status">
          <div className={styles.bulkCopy}>
            <strong>{selectedModelKeys.size} selected</strong>
            <span>Selection is preserved while paging and filtering.</span>
          </div>
          <div className={styles.bulkActions}>
            <button
              type="button"
              className={styles.clearSelectionButton}
              onClick={onClearSelection}
            >
              Clear selection
            </button>
            <button type="button" className={styles.bulkDeleteButton} onClick={onDeleteSelected}>
              Delete selected
            </button>
          </div>
        </div>
      ) : null}

      {operationError ? (
        <div className={styles.operationError} role="alert">
          <span>{operationError}</span>
          <button
            type="button"
            onClick={onDismissOperationError}
            aria-label="Dismiss model operation error"
          >
            Dismiss
          </button>
        </div>
      ) : null}

      <DataTable
        columns={visibleColumns}
        data={filteredModels}
        keyExtractor={(row) => row.name}
        onView={onViewModel}
        onEdit={onEditModel}
        onDelete={onDeleteModel}
        expandable
        renderExpandedRow={renderExpandedRow}
        isRowExpanded={(row) => expandedModels.has(row.name)}
        onToggleExpand={onToggleExpand}
        emptyMessage={
          filtersActive ? 'No models match the current search and filters' : 'No models configured'
        }
        className={configStyles.managerTable}
        readonly={isReadonly}
        pagination={{
          pageSize: 25,
          pageSizeOptions: [25, 50, 100],
          itemLabel: 'models',
          resetKey: `${modelsSearch}|${reasoningFamilyFilter}|${endpointFilter}|${roleFilter}`,
        }}
        selection={
          isReadonly
            ? undefined
            : {
                selectedKeys: selectedModelKeys,
                onChange: onSelectedModelKeysChange,
                isRowDisabled: (row) => Boolean(getDeleteBlocker(row.name)),
                label: 'model',
              }
        }
      />
    </>
  )
}
