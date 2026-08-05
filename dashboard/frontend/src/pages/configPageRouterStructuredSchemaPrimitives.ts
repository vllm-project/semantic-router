export type RouterStructuredKind =
  | 'string'
  | 'password'
  | 'number'
  | 'boolean'
  | 'select'
  | 'string-list'
  | 'number-list'
  | 'string-map'
  | 'object'
  | 'object-list'

export interface RouterStructuredSchema {
  kind: RouterStructuredKind
  label: string
  description?: string
  required?: boolean
  placeholder?: string
  options?: readonly string[]
  min?: number
  max?: number
  step?: number
  fields?: Record<string, RouterStructuredSchema>
  item?: RouterStructuredSchema
  itemLabelKey?: string
  addLabel?: string
  emptyLabel?: string
  defaultValue?: unknown
}

export interface RouterStructuredFieldDefinition {
  label: string
  description: string
  schema: RouterStructuredSchema
}

export const text = (
  label: string,
  options: Omit<RouterStructuredSchema, 'kind' | 'label'> = {},
): RouterStructuredSchema => ({ kind: 'string', label, ...options })

export const password = (label: string): RouterStructuredSchema => ({
  kind: 'password',
  label,
})

export const number = (
  label: string,
  options: Omit<RouterStructuredSchema, 'kind' | 'label'> = {},
): RouterStructuredSchema => ({ kind: 'number', label, ...options })

export const boolean = (label: string): RouterStructuredSchema => ({ kind: 'boolean', label })

export const select = (
  label: string,
  options: readonly string[],
  required = false,
): RouterStructuredSchema => ({
  kind: 'select',
  label,
  options,
  required,
})

export const stringList = (label: string, placeholder?: string): RouterStructuredSchema => ({
  kind: 'string-list',
  label,
  placeholder,
  defaultValue: [],
})

export const numberList = (label: string): RouterStructuredSchema => ({
  kind: 'number-list',
  label,
  defaultValue: [],
})

export const stringMap = (label: string): RouterStructuredSchema => ({
  kind: 'string-map',
  label,
  defaultValue: {},
})

export const object = (
  label: string,
  fields: Record<string, RouterStructuredSchema>,
): RouterStructuredSchema => ({ kind: 'object', label, fields, defaultValue: {} })

export const objectList = (
  label: string,
  fields: Record<string, RouterStructuredSchema>,
  itemLabelKey: string,
): RouterStructuredSchema => ({
  kind: 'object-list',
  label,
  item: object(label, fields),
  itemLabelKey,
  defaultValue: [],
  addLabel: `Add ${label.toLocaleLowerCase()}`,
  emptyLabel: `No ${label.toLocaleLowerCase()} configured.`,
})
