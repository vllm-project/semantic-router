import {
  EVALUATION_GATE_DISPOSITIONS,
  EVALUATION_GATE_VERDICTS,
  EVALUATION_SCHEMA_VERSION,
  EVALUATION_SUMMARY_VERDICTS,
  EVALUATION_TRACK_IDS,
} from '../types/evaluationPlane'

export type EvaluationRecord = Record<string, unknown>

export const EVALUATION_TRACK_ID_SET = new Set<string>(EVALUATION_TRACK_IDS)
export const EVALUATION_MODE_SET = new Set(['replay', 'live'])
export const EVALUATION_EVIDENCE_LEVEL_SET = new Set(['E0', 'E1', 'E2', 'E3', 'E4', 'E5'])
export const EVALUATION_GATE_DISPOSITION_SET = new Set<string>(EVALUATION_GATE_DISPOSITIONS)
export const EVALUATION_SUMMARY_VERDICT_SET = new Set<string>(EVALUATION_SUMMARY_VERDICTS)
export const EVALUATION_GATE_VERDICT_SET = new Set<string>(EVALUATION_GATE_VERDICTS)

export function isEvaluationRecord(value: unknown): value is EvaluationRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

export function hasOnlyEvaluationFields(
  value: EvaluationRecord,
  fields: readonly string[],
): boolean {
  return Object.keys(value).every((key) => fields.includes(key))
}

export function isNonEmptyText(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

export function isPortableEvaluationID(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)
}

export function isOptionalText(value: unknown): boolean {
  return value === undefined || typeof value === 'string'
}

export function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function isNonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0
}

export function isKnownValue(value: unknown, values: Set<string>): value is string {
  return typeof value === 'string' && values.has(value)
}

export function isTextArray(value: unknown, allowEmpty = true): value is string[] {
  return Array.isArray(value) && (allowEmpty || value.length > 0) && value.every(isNonEmptyText)
}

export function isKnownValueArray(value: unknown, values: Set<string>, allowEmpty = true): boolean {
  return (
    Array.isArray(value) &&
    (allowEmpty || value.length > 0) &&
    value.every((item) => isKnownValue(item, values))
  )
}

export function isStringRecord(value: unknown): value is Record<string, string> {
  return isEvaluationRecord(value) && Object.values(value).every((item) => typeof item === 'string')
}

export function assertCurrentEvaluationContract(
  payload: unknown,
  resource: string,
): asserts payload is EvaluationRecord {
  if (!isEvaluationRecord(payload) || payload.schema_version !== EVALUATION_SCHEMA_VERSION) {
    throw new Error(`${resource} did not match the ${EVALUATION_SCHEMA_VERSION} contract.`)
  }
}
