import { ALGORITHM_TYPES, type AlgorithmType } from '../lib/dslAlgorithmSchemas'

type AlgorithmRecord = Record<string, unknown>

function asRecord(value: unknown): AlgorithmRecord {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as AlgorithmRecord)
    : {}
}

export function algorithmType(value: unknown): AlgorithmType {
  const type = asRecord(value).type
  return ALGORITHM_TYPES.includes(type as AlgorithmType) ? (type as AlgorithmType) : 'static'
}

export function algorithmFields(value: unknown): AlgorithmRecord {
  const algorithm = asRecord(value)
  const type = algorithmType(algorithm)
  const nested = asRecord(algorithm[type])
  const fields: AlgorithmRecord = { ...nested }
  if (algorithm.minimum_candidates !== undefined) {
    fields.minimum_candidates = algorithm.minimum_candidates
  }
  if (type === 'prompt') {
    fields.prompt = algorithm.prompt
    fields.on_error = algorithm.on_error
  }
  return fields
}

export function mergeAlgorithmFields(
  existing: unknown,
  type: AlgorithmType,
  fields: AlgorithmRecord,
): AlgorithmRecord {
  const previous = asRecord(existing)
  const next: AlgorithmRecord = { ...previous, type }
  for (const algorithmTypeName of ALGORITHM_TYPES) delete next[algorithmTypeName]

  if (fields.minimum_candidates === undefined) delete next.minimum_candidates
  else next.minimum_candidates = fields.minimum_candidates

  const specific: AlgorithmRecord = { ...fields }
  delete specific.minimum_candidates
  if (type === 'prompt') {
    next.prompt = specific.prompt
    if (specific.on_error === undefined || specific.on_error === '') delete next.on_error
    else next.on_error = specific.on_error
  } else {
    delete next.on_error
    delete specific.prompt
    if (Object.keys(specific).length > 0) next[type] = specific
  }
  return next
}
