import type { FieldConfig } from '../components/EditModal'
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
