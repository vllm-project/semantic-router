import type { EditFormData, FieldConfig } from './EditModal'

export type EditDialogSize = 'md' | 'lg' | 'xl'

export const FULL_WIDTH_FIELD_TYPES = new Set<FieldConfig['type']>([
  'custom',
  'multiselect',
  'textarea',
])

export function visibleFieldSections(fields: FieldConfig[], formData: EditFormData) {
  const sections = new Map<string, FieldConfig[]>()
  for (const field of fields) {
    if (field.shouldHide?.(formData)) continue
    const section = field.section?.trim() || ''
    const entries = sections.get(section) ?? []
    entries.push(field)
    sections.set(section, entries)
  }
  return [...sections.entries()]
}

/** Sizes the dialog from the whole field list so toggling dependent fields never resizes it. */
export function editDialogSize(fields: FieldConfig[]): EditDialogSize {
  const sectionCount = new Set(fields.map((field) => field.section?.trim() || '')).size
  if (fields.length > 8 || sectionCount > 2 || fields.some((field) => field.type === 'custom')) {
    return 'xl'
  }
  if (fields.length > 3 || fields.some((field) => FULL_WIDTH_FIELD_TYPES.has(field.type))) {
    return 'lg'
  }
  return 'md'
}

/** Percentage fields are edited as 0-100 and stored as 0-1. */
export function convertPercentageFields(
  data: EditFormData,
  fields: FieldConfig[],
  direction: 'toDisplay' | 'toStored',
): EditFormData {
  const converted: EditFormData = { ...data }
  for (const field of fields) {
    if (field.type !== 'percentage') continue
    const rawValue = converted[field.name]
    const numericValue =
      typeof rawValue === 'number'
        ? rawValue
        : typeof rawValue === 'string' && rawValue.trim() !== ''
          ? Number(rawValue)
          : NaN
    if (Number.isFinite(numericValue)) {
      converted[field.name] =
        direction === 'toDisplay' ? Math.round(numericValue * 100) : numericValue / 100
    }
  }
  return converted
}

/** Keeps a configured value visible even when the field's option list no longer offers it. */
export function selectFieldOptions(
  options: readonly string[] | undefined,
  current: string,
): string[] {
  const list = [...(options ?? [])]
  if (current !== '' && !list.includes(current)) list.unshift(current)
  return list
}
