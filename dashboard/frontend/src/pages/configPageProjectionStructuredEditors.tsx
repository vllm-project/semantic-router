import { ObjectListEditor, type ObjectEditorField } from '../components/ObjectListEditor'
import { StringListEditor } from '../components/StringListEditor'
import type {
  ProjectionMappingCalibration,
  ProjectionMappingOutput,
  ProjectionScoreInput,
} from './configPageSupport'
import {
  normalizeProjectionCalibration,
  normalizeProjectionInputs,
  normalizeProjectionMembers,
  normalizeProjectionOutputs,
  projectionInputErrors,
  projectionOutputErrors,
} from './configPageProjectionFormSupport'

interface StructuredProjectionFieldProps {
  value: unknown
  onChange?: (value: unknown) => void
  disabled?: boolean
  readOnly?: boolean
}

const projectionInputTypes = [
  'keyword',
  'embedding',
  'domain',
  'fact_check',
  'user_feedback',
  'reask',
  'preference',
  'language',
  'context',
  'structure',
  'complexity',
  'modality',
  'authz',
  'jailbreak',
  'pii',
  'kb',
  'conversation',
  'event',
  'kb_metric',
  'projection',
] as const

const inputFields: ObjectEditorField<ProjectionScoreInput>[] = [
  {
    key: 'type',
    label: 'Input type',
    type: 'select',
    options: projectionInputTypes,
    required: true,
  },
  {
    key: 'name',
    label: 'Signal or projection name',
    placeholder: 'technical_support',
    required: true,
    shouldHide: (item) => item.type === 'kb_metric',
  },
  {
    key: 'kb',
    label: 'Knowledge base',
    placeholder: 'privacy_kb',
    required: true,
    shouldHide: (item) => item.type !== 'kb_metric',
  },
  {
    key: 'metric',
    label: 'Knowledge-base metric',
    placeholder: 'private_vs_public',
    required: true,
    shouldHide: (item) => item.type !== 'kb_metric',
  },
  { key: 'weight', label: 'Weight', type: 'number', step: 0.01, required: true },
  {
    key: 'value_source',
    label: 'Value source',
    type: 'select',
    options: ['binary', 'confidence', 'raw', 'score'],
  },
  { key: 'match', label: 'Match value', type: 'number', step: 0.01 },
  { key: 'miss', label: 'Miss value', type: 'number', step: 0.01 },
]

const calibrationFields: ObjectEditorField<ProjectionMappingCalibration>[] = [
  { key: 'method', label: 'Method', type: 'select', options: ['sigmoid_distance'] },
  { key: 'slope', label: 'Slope', type: 'number', step: 0.1, placeholder: '6' },
]

const outputFields: ObjectEditorField<ProjectionMappingOutput>[] = [
  {
    key: 'name',
    label: 'Output name',
    placeholder: 'support_fast',
    required: true,
    fullWidth: true,
  },
  { key: 'gt', label: 'Greater than', type: 'number', step: 0.01 },
  { key: 'gte', label: 'Greater than or equal', type: 'number', step: 0.01 },
  { key: 'lt', label: 'Less than', type: 'number', step: 0.01 },
  { key: 'lte', label: 'Less than or equal', type: 'number', step: 0.01 },
]

export function ProjectionMembersEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredProjectionFieldProps) {
  const members = Array.isArray(value)
    ? value.filter((member): member is string => typeof member === 'string')
    : normalizeProjectionMembers(value)
  return (
    <StringListEditor
      value={members}
      onChange={(nextValue) => onChange?.(nextValue)}
      addLabel="Add member"
      emptyLabel="No domain or embedding signals in this partition."
      itemLabel="Member"
      placeholder="technical_support"
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ProjectionInputsEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredProjectionFieldProps) {
  const inputs = normalizeProjectionInputs(value)
  return (
    <ObjectListEditor
      value={inputs}
      onChange={(nextValue) => onChange?.(nextValue)}
      fields={inputFields}
      createItem={() => ({ type: 'embedding', name: '', weight: 1, value_source: 'binary' })}
      addLabel="Add input"
      emptyLabel="No weighted inputs configured."
      itemLabel={(item, index) => item.name?.trim() || item.kb?.trim() || `Input ${index + 1}`}
      itemDescription={(item) =>
        `${item.type || 'Type required'} · weight ${Number.isFinite(item.weight) ? item.weight : 'required'}`
      }
      validateItem={projectionInputErrors}
      minItems={1}
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ProjectionCalibrationEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredProjectionFieldProps) {
  const calibration = normalizeProjectionCalibration(value)
  return (
    <ObjectListEditor
      value={calibration ? [calibration] : []}
      onChange={(nextValue) => onChange?.(nextValue[0])}
      fields={calibrationFields}
      createItem={() => ({ method: 'sigmoid_distance', slope: 6 })}
      addLabel="Enable calibration"
      emptyLabel="No confidence calibration configured."
      itemLabel={() => 'Confidence calibration'}
      itemDescription={(item) => item.method || 'sigmoid_distance'}
      maxItems={1}
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ProjectionOutputsEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredProjectionFieldProps) {
  const outputs = normalizeProjectionOutputs(value)
  return (
    <ObjectListEditor
      value={outputs}
      onChange={(nextValue) => onChange?.(nextValue)}
      fields={outputFields}
      createItem={(index) => ({ name: `output-${index + 1}`, gte: 0 })}
      addLabel="Add output band"
      emptyLabel="No routing output bands configured."
      itemLabel={(item, index) => item.name?.trim() || `Output ${index + 1}`}
      itemDescription={(item) => {
        const bounds = ['gt', 'gte', 'lt', 'lte'].flatMap((key) =>
          typeof item[key as keyof ProjectionMappingOutput] === 'number'
            ? [`${key} ${item[key as keyof ProjectionMappingOutput]}`]
            : [],
        )
        return bounds.join(' · ') || 'Threshold required'
      }}
      validateItem={projectionOutputErrors}
      minItems={1}
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}
