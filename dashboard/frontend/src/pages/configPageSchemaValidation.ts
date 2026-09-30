import type { FieldSchema } from '../lib/dslSchemas'

export function requiredSchemaFieldErrors(
  schema: FieldSchema[],
  value: Record<string, unknown>,
  prefix = '',
): string[] {
  return schema.flatMap((field) => {
    const fieldValue = value[field.key]
    const label = prefix ? `${prefix} > ${field.label}` : field.label
    const missing =
      fieldValue === undefined ||
      fieldValue === null ||
      fieldValue === '' ||
      (Array.isArray(fieldValue) && fieldValue.length === 0)
    const errors = field.required && missing ? [`${label} is required.`] : []

    if (field.type === 'object' && field.fields && fieldValue && typeof fieldValue === 'object') {
      errors.push(
        ...requiredSchemaFieldErrors(field.fields, fieldValue as Record<string, unknown>, label),
      )
    }
    if (field.type === 'object[]' && field.fields && Array.isArray(fieldValue)) {
      const nestedFields = field.fields
      fieldValue.forEach((entry, index) => {
        if (entry && typeof entry === 'object') {
          errors.push(
            ...requiredSchemaFieldErrors(
              nestedFields,
              entry as Record<string, unknown>,
              `${label} ${index + 1}`,
            ),
          )
        }
      })
    }
    return errors
  })
}
