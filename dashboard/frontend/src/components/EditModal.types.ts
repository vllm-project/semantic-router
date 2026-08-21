import type { ReactNode } from 'react'

export type EditFormData = Record<string, unknown>

type BivariantCallback<T extends (...args: never[]) => unknown> = {
  bivarianceHack: T
}['bivarianceHack']

export interface FieldConfig<TForm extends object = EditFormData> {
  name: string
  label: string
  type:
    | 'text'
    | 'number'
    | 'boolean'
    | 'select'
    | 'multiselect'
    | 'textarea'
    | 'percentage'
    | 'custom'
  required?: boolean
  options?: string[]
  placeholder?: string
  description?: string
  min?: number
  max?: number
  step?: number
  shouldHide?: BivariantCallback<(data: TForm) => boolean>
  customRender?: BivariantCallback<
    (value: unknown, onChange: (value: unknown) => void) => ReactNode
  >
}
