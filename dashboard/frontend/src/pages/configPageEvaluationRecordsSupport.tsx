import type { Column } from '../components/DataTable'
import type { FieldConfig } from '../components/EditModal'
import { cloneConfigData } from './configPageCanonicalization'
import { normalizeEvaluationRecords } from './configPageModelFormSupport'
import { validateEvaluationRecords } from './configPageModelInventory'
import { EvaluationRecordsEditor } from './configPageModelStructuredEditors'
import type { ConfigData, EvaluationRecordConfig } from './configPageSupport'

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

const metricSummary = (record: EvaluationRecordConfig): string =>
  Object.entries(record.metrics)
    .map(([metric, value]) => `${metric} ${value}`)
    .join(', ')

export const evaluationRecordColumns: Column<EvaluationRecordConfig>[] = [
  { key: 'model', header: 'Model Card', minWidth: '190px', sortable: true },
  { key: 'benchmark', header: 'Benchmark', minWidth: '230px', sortable: true },
  {
    key: 'reasoning_effort',
    header: 'Effort',
    minWidth: '90px',
    render: (record) => record.reasoning_effort || 'default',
    sortable: true,
  },
  {
    key: 'metrics',
    header: 'Metrics',
    minWidth: '150px',
    render: metricSummary,
  },
  {
    key: 'benchmark_profile',
    header: 'Profile',
    minWidth: '130px',
    render: (record) => record.benchmark_profile || 'benchmark default',
  },
]

export const evaluationRecordKey = (record: EvaluationRecordConfig): string =>
  [
    record.model,
    record.benchmark,
    record.benchmark_profile ?? '',
    record.reasoning_effort ?? '',
    JSON.stringify(record.metrics),
  ].join('\u0000')
