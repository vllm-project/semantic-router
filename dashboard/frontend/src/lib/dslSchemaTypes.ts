export interface FieldSchema {
  key: string
  label: string
  type:
    | 'string'
    | 'number'
    | 'boolean'
    | 'string[]'
    | 'number[]'
    | 'string[][]'
    | 'select'
    | 'object'
    | 'object[]'
    | 'key-value'
    | 'rule'
  options?: string[]
  required?: boolean
  placeholder?: string
  description?: string
  min?: number
  fields?: FieldSchema[]
  addLabel?: string
  emptyLabel?: string
  itemLabel?: string
  itemLabelKey?: string
  keyLabel?: string
  valueLabel?: string
}
