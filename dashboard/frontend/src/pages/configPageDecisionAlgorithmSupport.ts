import {
  ALGORITHM_TYPES,
  getAlgorithmFieldSchema,
  isExecutorAlgorithm,
  type AlgorithmType,
} from '../lib/dslAlgorithmSchemas'

type AlgorithmRecord = Record<string, unknown>

// Settings written beside `type`; everything else nests under the type key.
const SHARED_FIELDS = ['minimum_candidates', 'budget', 'on_error'] as const

function asRecord(value: unknown): AlgorithmRecord {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as AlgorithmRecord)
    : {}
}

export function algorithmType(value: unknown): AlgorithmType {
  const type = asRecord(value).type
  return ALGORITHM_TYPES.includes(type as AlgorithmType) ? (type as AlgorithmType) : 'static'
}

function editsPayloadAsOneField(type: AlgorithmType): boolean {
  return getAlgorithmFieldSchema(type).some(({ key }) => key === type)
}

export function algorithmFields(value: unknown): AlgorithmRecord {
  const algorithm = asRecord(value)
  const type = algorithmType(algorithm)
  const fields: AlgorithmRecord = editsPayloadAsOneField(type)
    ? algorithm[type] === undefined
      ? {}
      : { [type]: algorithm[type] }
    : { ...asRecord(algorithm[type]) }
  for (const key of SHARED_FIELDS) {
    if (algorithm[key] !== undefined) fields[key] = algorithm[key]
  }
  return fields
}

export function mergeAlgorithmFields(
  existing: unknown,
  type: AlgorithmType,
  fields: AlgorithmRecord,
): AlgorithmRecord {
  const next: AlgorithmRecord = { ...asRecord(existing), type }
  for (const algorithmTypeName of ALGORITHM_TYPES) delete next[algorithmTypeName]
  const settings: AlgorithmRecord = { ...fields }
  for (const key of SHARED_FIELDS) {
    delete next[key]
    delete settings[key]
    const value = fields[key]
    if (value === undefined || value === '') continue
    if (key === 'minimum_candidates' || isExecutorAlgorithm(type)) next[key] = value
  }
  const payload = editsPayloadAsOneField(type) ? asRecord(settings[type]) : settings
  if (Object.keys(payload).length > 0) next[type] = payload
  return next
}
