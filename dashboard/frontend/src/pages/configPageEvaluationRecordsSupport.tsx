import type { Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'
import ConfigPageEvaluationDetails from './ConfigPageEvaluationDetails'
import type { EffectiveEvaluationGroup } from './configPageEffectiveEvaluations'
import { cloneConfigData } from './configPageCanonicalization'
import { normalizeEvaluationRecords } from './configPageModelFormSupport'
import { validateEvaluationRecords } from './configPageModelInventory'
import { EvaluationRecordsEditor } from './configPageModelStructuredEditors'
import type { ConfigData } from './configPageSupport'

export const evaluationRecordsFormData = (config: ConfigData | null) => ({
  records: config?.evaluation?.records ?? [],
})

export const evaluationRecordsFields: FieldConfig[] = [
  {
    name: 'records',
    label: 'Evaluation Records',
    type: 'custom',
    description:
      'Measurements reference canonical Model Card identities. Built-in evidence is supplied automatically.',
    customRender: (value, onChange) => (
      <EvaluationRecordsEditor value={value} onChange={onChange} />
    ),
  },
]

export function buildEvaluationRecordsConfig(config: ConfigData, value: unknown): ConfigData {
  validateEvaluationRecords(value)
  const next = cloneConfigData(config)
  const records = normalizeEvaluationRecords(value)
  const definitions = next.evaluation
  if (records.length > 0 || definitions?.benchmarks?.length || definitions?.indices?.length) {
    next.evaluation = { ...definitions, records: records.length > 0 ? records : undefined }
  } else {
    delete next.evaluation
  }
  return next
}

export const evaluationModelColumns: Column<EffectiveEvaluationGroup>[] = [
  { key: 'modelName', header: 'Model', minWidth: '190px', sortable: true },
  { key: 'recordCount', header: 'Evidence', render: (group) => group.records.length },
  { key: 'availableCount', header: 'Available', sortable: true },
  {
    key: 'sources',
    header: 'Sources',
    render: (group) => `${group.builtInCount} built-in · ${group.configuredCount} configured`,
  },
]

export const evaluationModelViewSections = (group: EffectiveEvaluationGroup): ViewSection[] => [
  {
    title: 'Evaluation coverage',
    fields: [
      { label: 'Model', value: group.modelName },
      { label: 'Model card identity', value: group.catalogId },
      { label: 'Evidence records', value: group.records.length },
      { label: 'Benchmarks', value: group.benchmarkCount },
      { label: 'Built-in records', value: group.builtInCount },
      { label: 'Configured records', value: group.configuredCount },
      {
        label: 'Evidence policy',
        value:
          'Built-in and configured evidence are combined. Configured records do not replace built-in results. Available describes a recorded result, not guaranteed coverage in every routing index.',
        fullWidth: true,
      },
    ],
  },
  {
    fields: [
      {
        label: 'Benchmark evidence',
        value: <ConfigPageEvaluationDetails key={group.modelName} group={group} />,
        fullWidth: true,
      },
    ],
  },
]
