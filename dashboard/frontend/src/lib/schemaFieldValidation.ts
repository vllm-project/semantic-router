import type { FieldSchema } from './dslSchemaTypes'

function isMissing(value: unknown): boolean {
  return (
    value === undefined ||
    value === null ||
    value === '' ||
    (Array.isArray(value) && value.length === 0)
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

/**
 * Required-field problems of a value against a field schema, nested objects
 * and list items included. The config managers and the Builder share it, so a
 * form reports the same problems wherever it is edited.
 */
export function requiredFieldErrors(
  schema: readonly FieldSchema[],
  value: Record<string, unknown>,
  prefix = '',
): string[] {
  return schema.flatMap((field) => {
    const fieldValue = value[field.key]
    const label = prefix ? `${prefix} > ${field.label}` : field.label
    const errors = field.required && isMissing(fieldValue) ? [`${label} is required.`] : []
    if (field.type === 'object' && field.fields && isRecord(fieldValue)) {
      errors.push(...requiredFieldErrors(field.fields, fieldValue, label))
    }
    if (field.type === 'object[]' && field.fields && Array.isArray(fieldValue)) {
      const nested = field.fields
      fieldValue.forEach((entry, index) => {
        if (isRecord(entry))
          errors.push(...requiredFieldErrors(nested, entry, `${label} ${index + 1}`))
      })
    }
    return errors
  })
}
