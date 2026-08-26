import { useEffect, useMemo, useState } from 'react'
import styles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfirmDialog from '../components/ConfirmDialog'
import RoutingScopeSelector from '../components/RoutingScopeSelector'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'
import type {
  ConfigData,
  ConfigProjections,
  ProjectionMapping,
  ProjectionPartition,
  ProjectionScore,
} from './configPageSupport'
import { cloneConfigData } from './configPageCanonicalization'
import {
  assertProjectionCalibration,
  assertProjectionInputs,
  assertProjectionMappingOutputCount,
  assertProjectionMembers,
  assertProjectionOutputs,
  assertProjectionPartitionSettings,
  normalizeProjectionCalibration,
  normalizeProjectionInputs,
  normalizeProjectionMembers,
  normalizeProjectionOutputs,
} from './configPageProjectionFormSupport'
import {
  ProjectionCalibrationEditor,
  ProjectionInputsEditor,
  ProjectionMembersEditor,
  ProjectionOutputsEditor,
} from './configPageProjectionStructuredEditors'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { useRoutingScopeManager } from './configPageRoutingScopeSupport'
import ConfigPageRoutingScopeState from './ConfigPageRoutingScopeState'
import {
  matchesProjectionMapping,
  matchesProjectionPartition,
  matchesProjectionScore,
} from './configPageProjectionSearch'
import {
  cloneProjections,
  EMPTY_MAPPINGS,
  EMPTY_PARTITIONS,
  EMPTY_PROJECTIONS,
  EMPTY_SCORES,
  ensureProjectionConfig,
  type ProjectionDeleteTarget,
  type ProjectionMappingFormState,
  type ProjectionPartitionFormState,
  type ProjectionScoreFormState,
} from './configPageProjectionTableSupport'

interface ConfigPageProjectionsSectionProps {
  isReadonly: boolean
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
}

export default function ConfigPageProjectionsSection({
  isReadonly,
  openEditModal,
  openViewModal,
}: ConfigPageProjectionsSectionProps) {
  const [search, setSearch] = useState('')
  const [projectionPendingDelete, setProjectionPendingDelete] =
    useState<ProjectionDeleteTarget | null>(null)
  const [projectionDeletePending, setProjectionDeletePending] = useState(false)
  const [projectionDeleteError, setProjectionDeleteError] = useState<string | null>(null)
  const {
    error: routingScopeError,
    loading: routingScopeLoading,
    reload: reloadRoutingScopes,
    saveScopedConfig,
    routingScopes,
    scopedConfig,
    selectedScope,
    selectedScopeId,
    setSelectedScopeId,
    selectedRecipe,
  } = useRoutingScopeManager()
  const scopeReadonly = isReadonly || Boolean(selectedRecipe?.immutable)
  useEffect(() => {
    setProjectionPendingDelete(null)
    setProjectionDeleteError(null)
  }, [selectedScopeId])
  const projections = useMemo<ConfigProjections>(
    () => scopedConfig?.projections || EMPTY_PROJECTIONS,
    [scopedConfig?.projections],
  )
  const partitions = projections.partitions || EMPTY_PARTITIONS
  const scores = projections.scores || EMPTY_SCORES
  const mappings = projections.mappings || EMPTY_MAPPINGS
  const scoreOptions = scores.map((score) => score.name)

  const filteredPartitions = useMemo(
    () => partitions.filter((partition) => matchesProjectionPartition(partition, search)),
    [partitions, search],
  )

  const filteredScores = useMemo(
    () => scores.filter((score) => matchesProjectionScore(score, search)),
    [scores, search],
  )

  const filteredMappings = useMemo(
    () => mappings.filter((mapping) => matchesProjectionMapping(mapping, search)),
    [mappings, search],
  )

  const partitionFields: FieldConfig<ProjectionPartitionFormState>[] = [
    { name: 'name', label: 'Name', type: 'text', required: true },
    {
      name: 'semantics',
      label: 'Semantics',
      type: 'select',
      required: true,
      options: ['exclusive', 'softmax_exclusive'],
    },
    {
      name: 'members',
      label: 'Members',
      type: 'custom',
      required: true,
      description: 'Declared domain or embedding signals coordinated by this partition.',
      customRender: (value, onChange) => (
        <ProjectionMembersEditor value={value} onChange={onChange} />
      ),
    },
    {
      name: 'temperature',
      label: 'Temperature',
      type: 'number',
      step: 0.01,
      shouldHide: (data) => data.semantics !== 'softmax_exclusive',
    },
    {
      name: 'default',
      label: 'Default',
      type: 'text',
      description: 'Fallback member name used when no member wins.',
    },
  ]

  const scoreFields: FieldConfig<ProjectionScoreFormState>[] = [
    { name: 'name', label: 'Name', type: 'text', required: true },
    {
      name: 'method',
      label: 'Method',
      type: 'select',
      required: true,
      options: ['weighted_sum'],
    },
    {
      name: 'inputs',
      label: 'Inputs',
      type: 'custom',
      required: true,
      description: 'Weighted signal, knowledge-base metric, or earlier projection contributions.',
      customRender: (value, onChange) => (
        <ProjectionInputsEditor value={value} onChange={onChange} />
      ),
    },
  ]

  const mappingFields: FieldConfig<ProjectionMappingFormState>[] = [
    { name: 'name', label: 'Name', type: 'text', required: true },
    {
      name: 'source',
      label: 'Source Score',
      type: 'select',
      required: true,
      options: scoreOptions.length > 0 ? scoreOptions : [''],
      description: 'Projection score whose output will be mapped into named routing bands.',
    },
    {
      name: 'method',
      label: 'Method',
      type: 'select',
      required: true,
      options: ['threshold_bands', 'multi_emit'],
    },
    {
      name: 'calibration',
      label: 'Calibration',
      type: 'custom',
      description: 'Optional confidence calibration applied to output bands.',
      customRender: (value, onChange) => (
        <ProjectionCalibrationEditor value={value} onChange={onChange} />
      ),
    },
    {
      name: 'outputs',
      label: 'Outputs',
      type: 'custom',
      required: true,
      description: 'Named routing bands and their lower or upper threshold bounds.',
      customRender: (value, onChange) => (
        <ProjectionOutputsEditor value={value} onChange={onChange} />
      ),
    },
  ]

  const withClonedConfig = async (mutate: (next: ConfigData) => void) => {
    if (!scopedConfig) return
    const next = cloneConfigData(scopedConfig)
    mutate(next)
    await saveScopedConfig(next)
  }

  const handleAddPartition = () => {
    openEditModal<ProjectionPartitionFormState>(
      'Add Projection Partition',
      {
        name: '',
        semantics: 'exclusive',
        members: [],
        temperature: undefined,
        default: '',
      },
      partitionFields,
      async (data) => {
        const members = normalizeProjectionMembers(data.members)
        const defaultMember = data.default?.trim() ?? ''
        assertProjectionMembers(members, defaultMember)
        assertProjectionPartitionSettings(data.semantics, data.temperature)
        const nextPartition: ProjectionPartition = {
          name: data.name.trim(),
          semantics: data.semantics,
          members,
          temperature: data.temperature,
          default: defaultMember,
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.partitions = [...projectionConfig.partitions!, nextPartition]
        })
      },
      'add',
    )
  }

  const handleEditPartition = (partition: ProjectionPartition) => {
    openEditModal<ProjectionPartitionFormState>(
      `Edit Partition: ${partition.name}`,
      {
        name: partition.name,
        semantics: partition.semantics,
        members: partition.members || [],
        temperature: partition.temperature,
        default: partition.default || '',
      },
      partitionFields,
      async (data) => {
        const members = normalizeProjectionMembers(data.members)
        const defaultMember = data.default?.trim() ?? ''
        assertProjectionMembers(members, defaultMember)
        assertProjectionPartitionSettings(data.semantics, data.temperature)
        const nextPartition: ProjectionPartition = {
          name: data.name.trim(),
          semantics: data.semantics,
          members,
          temperature: data.temperature,
          default: defaultMember,
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.partitions = cloneProjections(next).partitions?.map((entry) =>
            entry.name === partition.name ? nextPartition : entry,
          )
        })
      },
    )
  }

  const handleDeletePartition = (partition: ProjectionPartition) => {
    setProjectionDeleteError(null)
    setProjectionPendingDelete({ kind: 'partition', name: partition.name })
  }

  const handleViewPartition = (partition: ProjectionPartition) => {
    const sections: ViewSection[] = [
      {
        title: 'Partition',
        fields: [
          { label: 'Name', value: partition.name },
          { label: 'Semantics', value: partition.semantics },
          { label: 'Temperature', value: partition.temperature ?? 'N/A' },
          { label: 'Default', value: partition.default || 'N/A' },
          {
            label: 'Members',
            value: <ProjectionMembersEditor value={partition.members || []} readOnly />,
            fullWidth: true,
          },
        ],
      },
    ]
    openViewModal(
      `Projection Partition: ${partition.name}`,
      sections,
      () => handleEditPartition(partition),
      scopeReadonly
        ? []
        : [
            {
              label: 'Delete partition',
              tone: 'destructive',
              onClick: () => handleDeletePartition(partition),
            },
          ],
    )
  }

  const handleAddScore = () => {
    openEditModal<ProjectionScoreFormState>(
      'Add Projection Score',
      {
        name: '',
        method: 'weighted_sum',
        inputs: [],
      },
      scoreFields,
      async (data) => {
        const inputs = normalizeProjectionInputs(data.inputs)
        assertProjectionInputs(inputs)
        const nextScore: ProjectionScore = {
          name: data.name.trim(),
          method: data.method,
          inputs,
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.scores = [...projectionConfig.scores!, nextScore]
        })
      },
      'add',
    )
  }

  const handleEditScore = (score: ProjectionScore) => {
    openEditModal<ProjectionScoreFormState>(
      `Edit Score: ${score.name}`,
      {
        name: score.name,
        method: score.method,
        inputs: score.inputs || [],
      },
      scoreFields,
      async (data) => {
        const inputs = normalizeProjectionInputs(data.inputs)
        assertProjectionInputs(inputs)
        const nextScore: ProjectionScore = {
          name: data.name.trim(),
          method: data.method,
          inputs,
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.scores = cloneProjections(next).scores?.map((entry) =>
            entry.name === score.name ? nextScore : entry,
          )
        })
      },
    )
  }

  const handleDeleteScore = (score: ProjectionScore) => {
    setProjectionDeleteError(null)
    setProjectionPendingDelete({ kind: 'score', name: score.name })
  }

  const handleViewScore = (score: ProjectionScore) => {
    const sections: ViewSection[] = [
      {
        title: 'Projection Score',
        fields: [
          { label: 'Name', value: score.name },
          { label: 'Method', value: score.method },
          {
            label: 'Inputs',
            value: <ProjectionInputsEditor value={score.inputs || []} readOnly />,
            fullWidth: true,
          },
        ],
      },
    ]
    openViewModal(
      `Projection Score: ${score.name}`,
      sections,
      () => handleEditScore(score),
      scopeReadonly
        ? []
        : [
            {
              label: 'Delete score',
              tone: 'destructive',
              onClick: () => handleDeleteScore(score),
            },
          ],
    )
  }

  const handleAddMapping = () => {
    openEditModal<ProjectionMappingFormState>(
      'Add Projection Mapping',
      {
        name: '',
        source: scoreOptions[0] || '',
        method: 'threshold_bands',
        calibration: undefined,
        outputs: [],
      },
      mappingFields,
      async (data) => {
        const calibration = normalizeProjectionCalibration(data.calibration)
        const outputs = normalizeProjectionOutputs(data.outputs)
        assertProjectionCalibration(calibration)
        assertProjectionOutputs(outputs)
        assertProjectionMappingOutputCount(data.method, outputs)
        const nextMapping: ProjectionMapping = {
          name: data.name.trim(),
          source: data.source,
          method: data.method,
          calibration,
          outputs,
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.mappings = [...projectionConfig.mappings!, nextMapping]
        })
      },
      'add',
    )
  }

  const handleEditMapping = (mapping: ProjectionMapping) => {
    openEditModal<ProjectionMappingFormState>(
      `Edit Mapping: ${mapping.name}`,
      {
        name: mapping.name,
        source: mapping.source,
        method: mapping.method,
        calibration: mapping.calibration,
        outputs: mapping.outputs || [],
      },
      mappingFields,
      async (data) => {
        const calibration = normalizeProjectionCalibration(data.calibration)
        const outputs = normalizeProjectionOutputs(data.outputs)
        assertProjectionCalibration(calibration)
        assertProjectionOutputs(outputs)
        assertProjectionMappingOutputCount(data.method, outputs)
        const nextMapping: ProjectionMapping = {
          name: data.name.trim(),
          source: data.source,
          method: data.method,
          calibration,
          outputs,
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.mappings = cloneProjections(next).mappings?.map((entry) =>
            entry.name === mapping.name ? nextMapping : entry,
          )
        })
      },
    )
  }

  const handleDeleteMapping = (mapping: ProjectionMapping) => {
    setProjectionDeleteError(null)
    setProjectionPendingDelete({ kind: 'mapping', name: mapping.name })
  }

  const confirmDeleteProjection = async () => {
    if (!projectionPendingDelete) return
    if (!scopedConfig) {
      setProjectionDeleteError('Recipe configuration is not available.')
      return
    }

    setProjectionDeletePending(true)
    setProjectionDeleteError(null)
    try {
      const target = projectionPendingDelete
      await withClonedConfig((next) => {
        const projectionConfig = ensureProjectionConfig(next)
        const cloned = cloneProjections(next)
        if (target.kind === 'partition') {
          projectionConfig.partitions = cloned.partitions?.filter(
            (entry) => entry.name !== target.name,
          )
        } else if (target.kind === 'score') {
          projectionConfig.scores = cloned.scores?.filter((entry) => entry.name !== target.name)
        } else {
          projectionConfig.mappings = cloned.mappings?.filter((entry) => entry.name !== target.name)
        }
      })
      setProjectionPendingDelete(null)
    } catch (error) {
      setProjectionDeleteError(
        error instanceof Error ? error.message : 'Failed to delete projection.',
      )
    } finally {
      setProjectionDeletePending(false)
    }
  }

  const handleViewMapping = (mapping: ProjectionMapping) => {
    const sections: ViewSection[] = [
      {
        title: 'Projection Mapping',
        fields: [
          { label: 'Name', value: mapping.name },
          { label: 'Source', value: mapping.source },
          { label: 'Method', value: mapping.method },
          {
            label: 'Calibration',
            value: mapping.calibration ? (
              <ProjectionCalibrationEditor value={mapping.calibration} readOnly />
            ) : (
              'N/A'
            ),
            fullWidth: true,
          },
          {
            label: 'Outputs',
            value: <ProjectionOutputsEditor value={mapping.outputs || []} readOnly />,
            fullWidth: true,
          },
        ],
      },
    ]
    openViewModal(
      `Projection Mapping: ${mapping.name}`,
      sections,
      () => handleEditMapping(mapping),
      scopeReadonly
        ? []
        : [
            {
              label: 'Delete mapping',
              tone: 'destructive',
              onClick: () => handleDeleteMapping(mapping),
            },
          ],
    )
  }

  const partitionColumns: Column<ProjectionPartition>[] = [
    {
      key: 'name',
      header: 'Name',
      sortable: true,
      render: (row) => <span style={{ fontWeight: 600 }}>{row.name}</span>,
    },
    {
      key: 'semantics',
      header: 'Semantics',
      width: '180px',
      sortable: true,
      render: (row) => <span className={styles.tableMetaBadge}>{row.semantics}</span>,
    },
    {
      key: 'members',
      header: 'Members',
      width: '140px',
      render: (row) => <span>{row.members?.length || 0} members</span>,
    },
    {
      key: 'default',
      header: 'Default',
      width: '180px',
      render: (row) => row.default || 'N/A',
    },
  ]

  const scoreColumns: Column<ProjectionScore>[] = [
    {
      key: 'name',
      header: 'Name',
      sortable: true,
      render: (row) => <span style={{ fontWeight: 600 }}>{row.name}</span>,
    },
    {
      key: 'method',
      header: 'Method',
      width: '180px',
      sortable: true,
      render: (row) => <span className={styles.tableMetaBadge}>{row.method}</span>,
    },
    {
      key: 'inputs',
      header: 'Inputs',
      width: '140px',
      render: (row) => <span>{row.inputs?.length || 0} inputs</span>,
    },
    {
      key: 'sources',
      header: 'Sources',
      render: (row) =>
        row.inputs
          ?.map((input) => `${input.type}:${input.name}`)
          .slice(0, 3)
          .join(', ') || 'N/A',
    },
  ]

  const mappingColumns: Column<ProjectionMapping>[] = [
    {
      key: 'name',
      header: 'Name',
      sortable: true,
      render: (row) => <span style={{ fontWeight: 600 }}>{row.name}</span>,
    },
    {
      key: 'source',
      header: 'Source',
      width: '180px',
      sortable: true,
      render: (row) => <span className={styles.tableMetaBadge}>{row.source}</span>,
    },
    {
      key: 'method',
      header: 'Method',
      width: '180px',
      sortable: true,
      render: (row) => row.method,
    },
    {
      key: 'outputs',
      header: 'Outputs',
      render: (row) => row.outputs?.map((output) => output.name).join(', ') || 'N/A',
    },
  ]

  if (routingScopeLoading || routingScopeError || !selectedRecipe) {
    return (
      <ConfigPageManagerLayout
        title="Projections"
        description="Turn signals into clear routing bands."
      >
        <ConfigPageRoutingScopeState
          error={routingScopeError}
          loading={routingScopeLoading}
          onRetry={() => void reloadRoutingScopes()}
        />
      </ConfigPageManagerLayout>
    )
  }

  return (
    <ConfigPageManagerLayout
      title="Projections"
      description="Turn signals into clear routing bands."
      scope={selectedScope?.label ?? 'Routing profile'}
    >
      <div className={styles.sectionPanel}>
        <div className={`${styles.sectionTableBlock} ${styles.projectionStack}`}>
          <RoutingScopeSelector
            scopes={routingScopes}
            value={selectedScopeId}
            onChange={setSelectedScopeId}
          />
          <p className={styles.productHint}>
            Trace projection outcomes in{' '}
            <a
              href="https://vllm-sr.ai/docs/tutorials/projection/traces"
              target="_blank"
              rel="noreferrer"
            >
              Insights
            </a>
            .
          </p>
          <TableHeader
            title="Projection Surfaces"
            count={filteredPartitions.length + filteredScores.length + filteredMappings.length}
            searchPlaceholder="Search partitions, scores, or mappings..."
            searchValue={search}
            onSearchChange={setSearch}
            disabled={scopeReadonly}
          />

          <section>
            <TableHeader
              title="Partitions"
              count={filteredPartitions.length}
              onAdd={handleAddPartition}
              addButtonText="Add Partition"
              disabled={scopeReadonly}
              variant="embedded"
            />
            <DataTable
              columns={partitionColumns}
              data={filteredPartitions}
              keyExtractor={(row) => row.name}
              onView={handleViewPartition}
              openOnRowClick
              emptyMessage="No projection partitions configured."
              readonly={scopeReadonly}
              pagination={{
                pageSize: 25,
                pageSizeOptions: [25, 50, 100],
                itemLabel: 'partitions',
                resetKey: search,
              }}
            />
          </section>

          <section>
            <TableHeader
              title="Scores"
              count={filteredScores.length}
              onAdd={handleAddScore}
              addButtonText="Add Score"
              disabled={scopeReadonly}
              variant="embedded"
            />
            <DataTable
              columns={scoreColumns}
              data={filteredScores}
              keyExtractor={(row) => row.name}
              onView={handleViewScore}
              openOnRowClick
              emptyMessage="No projection scores configured."
              readonly={scopeReadonly}
              pagination={{
                pageSize: 25,
                pageSizeOptions: [25, 50, 100],
                itemLabel: 'scores',
                resetKey: search,
              }}
            />
          </section>

          <section>
            <TableHeader
              title="Mappings"
              count={filteredMappings.length}
              onAdd={handleAddMapping}
              addButtonText="Add Mapping"
              disabled={scopeReadonly}
              variant="embedded"
            />
            <DataTable
              columns={mappingColumns}
              data={filteredMappings}
              keyExtractor={(row) => row.name}
              onView={handleViewMapping}
              openOnRowClick
              emptyMessage="No projection mappings configured."
              readonly={scopeReadonly}
              pagination={{
                pageSize: 25,
                pageSizeOptions: [25, 50, 100],
                itemLabel: 'mappings',
                resetKey: search,
              }}
            />
          </section>
        </div>
      </div>

      <ConfirmDialog
        isOpen={projectionPendingDelete !== null}
        title={`Delete projection ${projectionPendingDelete?.kind || ''} “${projectionPendingDelete?.name || ''}”?`}
        description="Remove this projection surface from the active routing configuration. Any dependent projection or decision references must be updated separately."
        eyebrow="Destructive configuration change"
        confirmLabel={`Delete ${projectionPendingDelete?.kind || 'projection'}`}
        pending={projectionDeletePending}
        details={
          projectionDeleteError ? <span role="alert">{projectionDeleteError}</span> : undefined
        }
        onCancel={() => {
          if (projectionDeletePending) return
          setProjectionPendingDelete(null)
          setProjectionDeleteError(null)
        }}
        onConfirm={confirmDeleteProjection}
      />
    </ConfigPageManagerLayout>
  )
}
