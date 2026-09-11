import { DataTable } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import ConfigPageConnectModelsDialog from './ConfigPageConnectModelsDialog'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageModelEndpoints from './ConfigPageModelEndpoints'
import ConfigPageModelInventoryPanel from './ConfigPageModelInventoryPanel'
import styles from './ConfigPage.module.css'
import { evaluationRecordColumns, evaluationRecordKey } from './configPageEvaluationRecordsSupport'
import { reasoningFamilyColumns } from './configPageReasoningFamilySupport'
import type { ConfigPageModelsSectionController } from './useConfigPageModelsSectionController'
import type { ConfigPageModelsSectionProps } from './configPageModelsSectionTypes'

interface ModelsSectionViewProps {
  props: ConfigPageModelsSectionProps
  controller: ConfigPageModelsSectionController
}

function ModelInventoryBlock({ props, controller }: ModelsSectionViewProps) {
  const { filters, deletion, forms, liveVerification } = controller
  return (
    <div className={styles.sectionTableBlock}>
      <ConfigPageModelInventoryPanel
        models={props.models}
        filteredModels={filters.models}
        defaultModel={props.defaultModel}
        modelReferenceCounts={filters.referenceCounts}
        modelsSearch={props.modelsSearch}
        onModelsSearchChange={props.onModelsSearchChange}
        reasoningFamilyFilter={filters.reasoningFamily}
        onReasoningFamilyFilterChange={filters.setReasoningFamily}
        reasoningFamilyOptions={filters.reasoningFamilyOptions}
        endpointFilter={filters.endpointState}
        onEndpointFilterChange={filters.setEndpointState}
        roleFilter={filters.role}
        onRoleFilterChange={filters.setRole}
        filtersActive={filters.active}
        onClearFilters={filters.clear}
        isReadonly={props.isReadonly}
        selectedModelKeys={deletion.selectedKeys}
        onSelectedModelKeysChange={deletion.setSelectedKeys}
        onClearSelection={() => deletion.setSelectedKeys(new Set())}
        onDeleteSelected={() => {
          deletion.setError(null)
          deletion.setModelNames([...deletion.selectedKeys])
        }}
        operationError={deletion.error}
        onDismissOperationError={() => deletion.setError(null)}
        onAddModel={() => controller.connect.setOpen(true)}
        onViewModel={forms.view}
        onEditModel={forms.edit}
        onDeleteModel={deletion.requestOne}
        expandedModels={props.expandedModels}
        onToggleExpand={controller.toggleExpand}
        renderExpandedRow={(model) => (
          <ConfigPageModelEndpoints model={model} redactEndpoints={props.isReadonly} />
        )}
        getDeleteBlocker={deletion.blocker}
        liveVerificationStates={liveVerification.states}
        onVerifyModel={(modelName) => void liveVerification.verify(modelName)}
        canVerifyModels={props.canVerifyModels}
      />
    </div>
  )
}

function ReasoningFamiliesBlock({ controller }: ModelsSectionViewProps) {
  const { reasoning, connect } = controller
  return (
    <div className={styles.sectionTableBlock}>
      {connect.catalogError ? (
        <p role="status" className={styles.inlineInfo}>
          Using the bundled catalog because the live catalog API is unavailable.
        </p>
      ) : null}
      <TableHeader
        title="Built-in Reasoning Families"
        count={reasoning.rows.length}
        searchPlaceholder="Search family, type, or parameter..."
        searchValue={reasoning.search}
        onSearchChange={reasoning.setSearch}
        variant="embedded"
      />
      <DataTable
        columns={reasoningFamilyColumns}
        data={reasoning.rows}
        keyExtractor={(row) => row.name}
        onView={(row) => reasoning.view(row.name)}
        emptyMessage="No built-in reasoning families are available"
        className={styles.managerTable}
        readonly
        pagination={{
          pageSize: 25,
          pageSizeOptions: [25, 50, 100],
          itemLabel: 'families',
          resetKey: reasoning.search,
        }}
      />
    </div>
  )
}

function EvaluationRecordsBlock({ props, controller }: ModelsSectionViewProps) {
  const records = controller.evaluations.records
  return (
    <div className={styles.sectionTableBlock}>
      <TableHeader
        title="Evaluation Records"
        count={records.length}
        onAdd={props.isReadonly ? undefined : controller.evaluations.manage}
        addButtonText="Manage records"
        variant="embedded"
      />
      <DataTable
        columns={evaluationRecordColumns}
        data={records}
        keyExtractor={evaluationRecordKey}
        emptyMessage="No operator evaluation records configured; built-in evidence remains available automatically."
        className={styles.managerTable}
        readonly
        pagination={{ pageSize: 25, pageSizeOptions: [25, 50, 100], itemLabel: 'records' }}
      />
    </div>
  )
}

function ConnectModelsDialog({ props, controller }: ModelsSectionViewProps) {
  const { connect, forms } = controller
  return (
    <ConfigPageConnectModelsDialog
      isOpen={connect.open}
      existingModelNames={[
        ...props.models.map((model) => model.name),
        ...(props.config?.entrypoints ?? []).flatMap((entrypoint) => entrypoint.model_names),
      ]}
      reasoningFamilies={Object.keys(props.reasoningFamilies)}
      catalog={connect.catalog}
      onClose={() => connect.setOpen(false)}
      onImport={forms.connect}
      onManualSetup={forms.add}
    />
  )
}

export default function ConfigPageModelsSectionView(viewProps: ModelsSectionViewProps) {
  return (
    <>
      <ConfigPageManagerLayout
        title="Models"
        description="Manage provider models, model-linked evaluation evidence, reasoning families, and endpoints available to routing decisions."
      >
        <div className={styles.sectionPanel}>
          <ModelInventoryBlock {...viewProps} />
          <EvaluationRecordsBlock {...viewProps} />
          <ReasoningFamiliesBlock {...viewProps} />
        </div>
      </ConfigPageManagerLayout>
      <ConnectModelsDialog {...viewProps} />
    </>
  )
}
