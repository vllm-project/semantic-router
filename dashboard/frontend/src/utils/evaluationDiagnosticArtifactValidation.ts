export class InvalidEvaluationDiagnosticArtifactError extends Error {
  constructor(
    readonly artifactName: string,
    detail: string,
  ) {
    super(`${artifactName}: ${detail}`)
    this.name = 'InvalidEvaluationDiagnosticArtifactError'
  }
}

export function invalid(artifactName: string, detail: string): never {
  throw new InvalidEvaluationDiagnosticArtifactError(artifactName, detail)
}

export function recordWithExactKeys(
  value: unknown,
  keys: readonly string[],
  artifactName: string,
  path: string,
): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    invalid(artifactName, `${path} must be an object`)
  }
  const record = value as Record<string, unknown>
  const actualKeys = Object.keys(record)
  if (
    actualKeys.length !== keys.length ||
    keys.some((key) => !Object.prototype.hasOwnProperty.call(record, key))
  ) {
    invalid(artifactName, `${path} has an unexpected structure`)
  }
  return record
}

export function nonNegativeFiniteNumber(
  value: unknown,
  artifactName: string,
  path: string,
  minimum = 0,
): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum) {
    invalid(artifactName, `${path} must be a finite number greater than or equal to ${minimum}`)
  }
  return value
}

export function positiveFiniteNumber(value: unknown, artifactName: string, path: string): number {
  const result = nonNegativeFiniteNumber(value, artifactName, path)
  if (result === 0) invalid(artifactName, `${path} must be greater than zero`)
  return result
}

export function boundedInteger(
  value: unknown,
  artifactName: string,
  path: string,
  minimum = 0,
  maximum = Number.MAX_SAFE_INTEGER,
): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    invalid(artifactName, `${path} must be an integer between ${minimum} and ${maximum}`)
  }
  return value as number
}

export function booleanValue(value: unknown, artifactName: string, path: string): boolean {
  if (typeof value !== 'boolean') invalid(artifactName, `${path} must be boolean`)
  return value
}

export function approximatelyEqual(left: number, right: number): boolean {
  if (left === right) return true
  return Math.abs(left - right) <= 1e-10 * Math.max(1, Math.abs(left), Math.abs(right))
}
