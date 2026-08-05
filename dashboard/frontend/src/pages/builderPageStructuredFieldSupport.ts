import type { FieldSchema } from '@/lib/dslMutations'
import type { DSLFieldObject, DSLFieldValue } from '@/types/dsl'

export function parseLegacyStructuredValue(value: unknown): unknown {
  if (typeof value !== 'string') return value
  const trimmed = value.trim()
  if (!trimmed) return value
  try {
    return JSON.parse(trimmed)
  } catch {
    return value
  }
}

export function normalizeStructuredObject(value: unknown): DSLFieldObject {
  const parsed = parseLegacyStructuredValue(value)
  return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
    ? (parsed as DSLFieldObject)
    : {}
}

export function normalizeStructuredObjectList(value: unknown): DSLFieldObject[] {
  const parsed = parseLegacyStructuredValue(value)
  if (!Array.isArray(parsed)) return []
  return parsed.filter(
    (item): item is DSLFieldObject => Boolean(item && typeof item === 'object' && !Array.isArray(item)),
  )
}

export function normalizeStructuredStringList(value: unknown): string[] {
  const parsed = parseLegacyStructuredValue(value)
  return Array.isArray(parsed)
    ? parsed.filter((item): item is string => typeof item === 'string')
    : []
}

export function normalizeStructuredStringMatrix(value: unknown): string[][] {
  const parsed = parseLegacyStructuredValue(value)
  if (!Array.isArray(parsed)) return []
  return parsed.map((row) =>
    Array.isArray(row) ? row.filter((item): item is string => typeof item === 'string') : [],
  )
}

export function updateStructuredObjectField(
  value: DSLFieldObject,
  key: string,
  nextValue: DSLFieldValue | undefined,
): DSLFieldObject {
  const next = { ...value }
  if (nextValue === undefined) delete next[key]
  else next[key] = nextValue
  return next
}

export function updateStructuredObjectListItem(
  value: readonly DSLFieldObject[],
  index: number,
  nextItem: DSLFieldObject,
): DSLFieldObject[] {
  return value.map((item, itemIndex) => (itemIndex === index ? nextItem : item))
}

export function removeStructuredObjectListItem(
  value: readonly DSLFieldObject[],
  index: number,
): DSLFieldObject[] {
  return value.filter((_, itemIndex) => itemIndex !== index)
}

export function structuredItemLabel(
  schema: FieldSchema,
  item: DSLFieldObject,
  index: number,
): string {
  const base = schema.itemLabel || 'Item'
  const candidate = schema.itemLabelKey ? item[schema.itemLabelKey] : undefined
  return typeof candidate === 'string' && candidate.trim()
    ? `${base}: ${candidate.trim()}`
    : `${base} ${index + 1}`
}

export function requiredStructuredFieldErrors(
  fields: readonly FieldSchema[],
  value: DSLFieldObject,
): string[] {
  return fields.flatMap((field) => {
    if (!field.required) return []
    const fieldValue = value[field.key]
    const missing =
      fieldValue === undefined ||
      fieldValue === null ||
      fieldValue === '' ||
      (Array.isArray(fieldValue) && fieldValue.length === 0)
    return missing ? [`${field.label} is required.`] : []
  })
}

export function supportsSharedObjectList(fields: readonly FieldSchema[]): boolean {
  return fields.every((field) =>
    ['string', 'number', 'select', 'key-value'].includes(field.type),
  )
}
