import { useMemo, useState } from 'react'
import styles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'
import type {
  ConfigData,
  ConfigProjections,
  ProjectionMapping,
  ProjectionMappingOutput,
  ProjectionPartition,
  ProjectionScore,
  ProjectionScoreInput,
} from './configPageSupport'
import { cloneConfigData } from './configPageCanonicalization'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'

interface ConfigPageProjectionsSectionProps {
  config: ConfigData | null
  isReadonly: boolean
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
}

interface ProjectionPartitionFormState {
  name: string
  semantics: string
  members: string[]
  temperature?: number
  default?: string
}

interface ProjectionScoreFormState {
  name: string
  method: string
  inputs: ProjectionScoreInput[]
}

interface ProjectionMappingFormState {
  name: string
  source: string
  method: string
  calibration?: ProjectionMapping['calibration']
  outputs: ProjectionMappingOutput[]
}

const toPrettyJSON = (value: unknown) => (
  <pre
    style={{
      margin: 0,
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-word',
      fontSize: '0.8125rem',
      lineHeight: 1.5,
      fontFamily: 'var(--font-mono)',
    }}
  >
    {JSON.stringify(value, null, 2)}
  </pre>
)

const cloneProjections = (cfg: ConfigData): ConfigProjections => ({
  partitions: [...(cfg.projections?.partitions || [])],
  scores: [...(cfg.projections?.scores || [])],
  mappings: [...(cfg.projections?.mappings || [])],
})

const ensureProjectionConfig = (cfg: ConfigData) => {
  if (!cfg.projections) {
    cfg.projections = { partitions: [], scores: [], mappings: [] }
  }
  if (!cfg.projections.partitions) cfg.projections.partitions = []
  if (!cfg.projections.scores) cfg.projections.scores = []
  if (!cfg.projections.mappings) cfg.projections.mappings = []
  return cfg.projections
}

const parseJSONField = <T,>(value: unknown, label: string): T => {
  if (typeof value === 'string') {
    try {
      return JSON.parse(value) as T
    } catch {
      throw new Error(`${label} must be valid JSON`)
    }
  }
  return value as T
}

export default function ConfigPageProjectionsSection({
  config,
  isReadonly,
  saveConfig,
  openEditModal,
  openViewModal,
}: ConfigPageProjectionsSectionProps) {
  const [search, setSearch] = useState('')
  const projections = useMemo<ConfigProjections>(
    () => config?.projections || { partitions: [], scores: [], mappings: [] },
    [config?.projections]
  )
  const partitions = projections.partitions || []
  const scores = projections.scores || []
  const mappings = projections.mappings || []
  const scoreOptions = scores.map((score) => score.name)

  const filteredPartitions = useMemo(
    () =>
      partitions.filter((partition) =>
        [partition.name, partition.semantics, partition.default || '', ...(partition.members || [])]
          .join(' ')
          .toLowerCase()
          .includes(search.toLowerCase())
      ),
    [partitions, search]
  )

  const filteredScores = useMemo(
    () =>
      scores.filter((score) =>
        [
          score.name,
          score.method,
          ...(score.inputs || []).flatMap((input) => [input.type, input.name, input.value_source || '']),
        ]
          .join(' ')
          .toLowerCase()
          .includes(search.toLowerCase())
      ),
    [scores, search]
  )

  const filteredMappings = useMemo(
    () =>
      mappings.filter((mapping) =>
        [
          mapping.name,
          mapping.source,
          mapping.method,
          mapping.calibration?.method || '',
          ...(mapping.outputs || []).map((output) => output.name),
        ]
          .join(' ')
          .toLowerCase()
          .includes(search.toLowerCase())
      ),
    [mappings, search]
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
      type: 'json',
      required: true,
      description: 'JSON string array of declared domain or embedding signal names.',
      placeholder: '["law", "business", "other"]',
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
      type: 'json',
      required: true,
      description:
        'JSON array of { type, name, weight, value_source?, match?, miss? } objects. Supported value_source: "binary" (default), "confidence", "raw". Use type "projection" with value_source "score" to reference earlier projection scores, or value_source "confidence" to reference mapping output confidences.',
      placeholder:
        '[{"type":"embedding","name":"technical_support","weight":0.18,"value_source":"confidence"}]',
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
      options: ['threshold_bands'],
    },
    {
      name: 'calibration',
      label: 'Calibration',
      type: 'json',
      description: 'Optional JSON object like {"method":"sigmoid_distance","slope":6}.',
      placeholder: '{"method":"sigmoid_distance","slope":6}',
    },
    {
      name: 'outputs',
      label: 'Outputs',
      type: 'json',
      required: true,
      description: 'JSON array of { name, lt?, lte?, gt?, gte? } threshold objects.',
      placeholder: '[{"name":"support_fast","lt":0.25},{"name":"support_escalated","gte":0.25}]',
    },
  ]

  const withClonedConfig = async (mutate: (next: ConfigData) => void) => {
    if (!config) return
    const next = cloneConfigData(config)
    mutate(next)
    await saveConfig(next)
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
        const nextPartition: ProjectionPartition = {
          name: data.name.trim(),
          semantics: data.semantics,
          members: parseJSONField<string[]>(data.members, 'Members'),
          temperature: data.temperature,
          default: data.default?.trim() ?? '',
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.partitions = [...projectionConfig.partitions!, nextPartition]
        })
      },
      'add'
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
        const nextPartition: ProjectionPartition = {
          name: data.name.trim(),
          semantics: data.semantics,
          members: parseJSONField<string[]>(data.members, 'Members'),
          temperature: data.temperature,
          default: data.default?.trim() ?? '',
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.partitions = cloneProjections(next).partitions?.map((entry) =>
            entry.name === partition.name ? nextPartition : entry
          )
        })
      }
    )
  }

  const handleDeletePartition = async (partition: ProjectionPartition) => {
    if (!window.confirm(`Delete projection partition "${partition.name}"?`)) return
    await withClonedConfig((next) => {
      const projectionConfig = ensureProjectionConfig(next)
      projectionConfig.partitions = cloneProjections(next).partitions?.filter((entry) => entry.name !== partition.name)
    })
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
          { label: 'Members', value: toPrettyJSON(partition.members || []), fullWidth: true },
        ],
      },
    ]
    openViewModal(`Projection Partition: ${partition.name}`, sections, () => handleEditPartition(partition))
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
        const nextScore: ProjectionScore = {
          name: data.name.trim(),
          method: data.method,
          inputs: parseJSONField<ProjectionScoreInput[]>(data.inputs, 'Inputs'),
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.scores = [...projectionConfig.scores!, nextScore]
        })
      },
      'add'
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
        const nextScore: ProjectionScore = {
          name: data.name.trim(),
          method: data.method,
          inputs: parseJSONField<ProjectionScoreInput[]>(data.inputs, 'Inputs'),
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.scores = cloneProjections(next).scores?.map((entry) =>
            entry.name === score.name ? nextScore : entry
          )
        })
      }
    )
  }

  const handleDeleteScore = async (score: ProjectionScore) => {
    if (!window.confirm(`Delete projection score "${score.name}"?`)) return
    await withClonedConfig((next) => {
      const projectionConfig = ensureProjectionConfig(next)
      projectionConfig.scores = cloneProjections(next).scores?.filter((entry) => entry.name !== score.name)
    })
  }

  const handleViewScore = (score: ProjectionScore) => {
    const sections: ViewSection[] = [
      {
        title: 'Projection Score',
        fields: [
          { label: 'Name', value: score.name },
          { label: 'Method', value: score.method },
          { label: 'Inputs', value: toPrettyJSON(score.inputs || []), fullWidth: true },
        ],
      },
    ]
    openViewModal(`Projection Score: ${score.name}`, sections, () => handleEditScore(score))
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
        const calibration = parseJSONField<ProjectionMapping['calibration'] | string | undefined>(
          data.calibration,
          'Calibration'
        )
        const nextMapping: ProjectionMapping = {
          name: data.name.trim(),
          source: data.source,
          method: data.method,
          calibration: typeof calibration === 'string' ? undefined : calibration,
          outputs: parseJSONField<ProjectionMappingOutput[]>(data.outputs, 'Outputs'),
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.mappings = [...projectionConfig.mappings!, nextMapping]
        })
      },
      'add'
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
        const calibration = parseJSONField<ProjectionMapping['calibration'] | string | undefined>(
          data.calibration,
          'Calibration'
        )
        const nextMapping: ProjectionMapping = {
          name: data.name.trim(),
          source: data.source,
          method: data.method,
          calibration: typeof calibration === 'string' ? undefined : calibration,
          outputs: parseJSONField<ProjectionMappingOutput[]>(data.outputs, 'Outputs'),
        }
        await withClonedConfig((next) => {
          const projectionConfig = ensureProjectionConfig(next)
          projectionConfig.mappings = cloneProjections(next).mappings?.map((entry) =>
            entry.name === mapping.name ? nextMapping : entry
          )
        })
      }
    )
  }

  const handleDeleteMapping = async (mapping: ProjectionMapping) => {
    if (!window.confirm(`Delete projection mapping "${mapping.name}"?`)) return
    await withClonedConfig((next) => {
      const projectionConfig = ensureProjectionConfig(next)
      projectionConfig.mappings = cloneProjections(next).mappings?.filter((entry) => entry.name !== mapping.name)
    })
  }

  const handleViewMapping = (mapping: ProjectionMapping) => {
    const sections: ViewSection[] = [
      {
        title: 'Projection Mapping',
        fields: [
          { label: 'Name', value: mapping.name },
          { label: 'Source', value: mapping.source },
          { label: 'Method', value: mapping.method },
          { label: 'Calibration', value: mapping.calibration ? toPrettyJSON(mapping.calibration) : 'N/A', fullWidth: true },
          { label: 'Outputs', value: toPrettyJSON(mapping.outputs || []), fullWidth: true },
        ],
      },
    ]
    openViewModal(`Projection Mapping: ${mapping.name}`, sections, () => handleEditMapping(mapping))
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
      render: (row) => row.inputs?.map((input) => `${input.type}:${input.name}`).slice(0, 3).join(', ') || 'N/A',
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

  return (
    <ConfigPageManagerLayout
      title="Projections"
      description="Coordinate mutually exclusive signal partitions, derive weighted scores, and map them into named routing bands that decisions can reference."
      pills={[
        { label: 'Models', active: false },
        { label: 'Signals', active: false },
        { label: 'Projections', active: true },
        { label: 'Decisions', active: false },
      ]}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <TableHeader
          title="Projection Surfaces"
          count={partitions.length + scores.length + mappings.length}
          searchPlaceholder="Search partitions, scores, or mappings..."
          searchValue={search}
          onSearchChange={setSearch}
          disabled={isReadonly}
        />

        <section>
          <TableHeader
            title="Partitions"
            count={filteredPartitions.length}
            onAdd={handleAddPartition}
            addButtonText="Add Partition"
            disabled={isReadonly}
            variant="embedded"
          />
          <DataTable
            columns={partitionColumns}
            data={filteredPartitions}
            keyExtractor={(row) => row.name}
            onView={handleViewPartition}
            onEdit={handleEditPartition}
            onDelete={handleDeletePartition}
            emptyMessage="No projection partitions configured."
            readonly={isReadonly}
          />
        </section>

        <section>
          <TableHeader
            title="Scores"
            count={filteredScores.length}
            onAdd={handleAddScore}
            addButtonText="Add Score"
            disabled={isReadonly}
            variant="embedded"
          />
          <DataTable
            columns={scoreColumns}
            data={filteredScores}
            keyExtractor={(row) => row.name}
            onView={handleViewScore}
            onEdit={handleEditScore}
            onDelete={handleDeleteScore}
            emptyMessage="No projection scores configured."
            readonly={isReadonly}
          />
        </section>

        <section>
          <TableHeader
            title="Mappings"
            count={filteredMappings.length}
            onAdd={handleAddMapping}
            addButtonText="Add Mapping"
            disabled={isReadonly}
            variant="embedded"
          />
          <DataTable
            columns={mappingColumns}
            data={filteredMappings}
            keyExtractor={(row) => row.name}
            onView={handleViewMapping}
            onEdit={handleEditMapping}
            onDelete={handleDeleteMapping}
            emptyMessage="No projection mappings configured."
            readonly={isReadonly}
          />
        </section>
      </div>
    </ConfigPageManagerLayout>
  )
}
