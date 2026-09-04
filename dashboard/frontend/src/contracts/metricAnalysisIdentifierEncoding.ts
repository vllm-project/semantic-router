export interface IdentifierEncodingVector {
  readonly raw: string
  readonly encoded: string
}

export interface IdentifierEncoding {
  readonly scheme: string
  readonly raw_pattern: string
  readonly direct_pattern: string
  readonly reserved_prefix: string
  readonly encoded_pattern: string
  readonly vectors: readonly IdentifierEncodingVector[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function assertCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`metric analysis catalog: ${message}`)
}

function assertExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  context: string,
) {
  const actual = Object.keys(value).sort()
  const wanted = [...expected].sort()
  assertCondition(
    actual.length === wanted.length && actual.every((key, index) => key === wanted[index]),
    `${context} fields are invalid`,
  )
}

function assertTrimmedText(value: unknown, context: string): asserts value is string {
  assertCondition(
    typeof value === 'string' && value.length > 0 && value.trim() === value,
    `${context} is invalid`,
  )
}

export function validateEncoding(value: unknown): asserts value is IdentifierEncoding {
  assertCondition(isRecord(value), 'identifier encoding must be an object')
  assertExactKeys(
    value,
    ['direct_pattern', 'encoded_pattern', 'raw_pattern', 'reserved_prefix', 'scheme', 'vectors'],
    'identifier encoding',
  )
  assertCondition(
    value.scheme === 'portable-segment-base64url.v1' && value.reserved_prefix === 'u-',
    'identifier encoding version is invalid',
  )
  for (const field of ['raw_pattern', 'direct_pattern', 'encoded_pattern'] as const) {
    assertTrimmedText(value[field], `identifier encoding ${field}`)
    assertCondition(
      value[field].startsWith('^') && value[field].endsWith('$'),
      `identifier encoding ${field} is not anchored`,
    )
    new RegExp(value[field])
  }
  assertCondition(
    Array.isArray(value.vectors) && value.vectors.length > 0,
    'identifier encoding vectors are missing',
  )
  for (const vector of value.vectors) {
    assertCondition(isRecord(vector), 'identifier encoding vector must be an object')
    assertExactKeys(vector, ['encoded', 'raw'], 'identifier encoding vector')
    assertTrimmedText(vector.raw, 'identifier encoding vector raw id')
    assertTrimmedText(vector.encoded, 'identifier encoding vector encoded id')
  }
}

export function encodeSubjectID(rawID: string, encoding: IdentifierEncoding): string {
  if (typeof rawID !== 'string' || !new RegExp(encoding.raw_pattern).test(rawID)) {
    throw new Error('metric subject id is not a portable raw identifier')
  }
  if (
    !rawID.startsWith(encoding.reserved_prefix) &&
    new RegExp(encoding.direct_pattern).test(rawID)
  ) {
    return rawID
  }
  const payload = btoa(rawID).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  const encoded = `${encoding.reserved_prefix}${payload}`
  if (!new RegExp(encoding.encoded_pattern).test(encoded)) {
    throw new Error('metric subject id exceeds the encoded segment contract')
  }
  return encoded
}

export function decodeSubjectID(encodedID: string, encoding: IdentifierEncoding): string {
  if (typeof encodedID !== 'string') {
    throw new Error('metric subject segment is not canonical')
  }
  if (!encodedID.startsWith(encoding.reserved_prefix)) {
    if (!new RegExp(encoding.direct_pattern).test(encodedID)) {
      throw new Error('metric subject segment is not canonical')
    }
    return encodedID
  }
  if (!new RegExp(encoding.encoded_pattern).test(encodedID)) {
    throw new Error('metric subject segment is not canonical base64url')
  }
  const payload = encodedID
    .slice(encoding.reserved_prefix.length)
    .replace(/-/g, '+')
    .replace(/_/g, '/')
  let raw: string
  try {
    raw = atob(payload + '='.repeat((4 - (payload.length % 4)) % 4))
  } catch {
    throw new Error('metric subject segment is not canonical base64url')
  }
  if (encodeSubjectID(raw, encoding) !== encodedID) {
    throw new Error('metric subject segment has a non-canonical encoding')
  }
  return raw
}
