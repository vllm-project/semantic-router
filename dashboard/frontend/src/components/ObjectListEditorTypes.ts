export interface StructuredEditorStateProps {
  disabled?: boolean
  readOnly?: boolean
}

export type ObjectEditorFieldType = 'text' | 'number' | 'select' | 'password' | 'key-value'

export interface ObjectEditorField<TItem extends object> {
  key: Extract<keyof TItem, string>
  label: string
  type?: ObjectEditorFieldType
  placeholder?: string
  options?: readonly string[]
  required?: boolean
  min?: number
  max?: number
  step?: number
  fullWidth?: boolean
  helpText?: string
  emptyValueLabel?: string
  keyLabel?: string
  keyPlaceholder?: string
  valueLabel?: string
  valuePlaceholder?: string
  shouldHide?: (item: TItem) => boolean
  /** Keep an explicit empty string instead of treating it as an omitted field. */
  preserveEmpty?: boolean
}

export interface ObjectListEditorProps<TItem extends object> extends StructuredEditorStateProps {
  value: readonly TItem[]
  onChange: (value: TItem[]) => void
  fields: readonly ObjectEditorField<TItem>[]
  createItem: (index: number) => TItem
  addLabel?: string
  emptyLabel?: string
  itemLabel?: (item: TItem, index: number) => string
  itemDescription?: (item: TItem, index: number) => string | undefined
  validateItem?: (item: TItem, index: number) => string[]
  minItems?: number
  maxItems?: number
}
