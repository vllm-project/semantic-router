import { ChevronDown } from 'lucide-react'

import { Button, Input, cn } from '../ui'
import { KeyValueEditor } from './KeyValueEditor'
import NativeSelect from './NativeSelect'
import type { ObjectEditorField } from './ObjectListEditorTypes'

interface ObjectEditorFieldControlProps<TItem extends object> {
  field: ObjectEditorField<TItem>
  fieldId: string
  value: unknown
  disabled: boolean
  readOnly: boolean
  onChange: (value: unknown) => void
}

const stringValue = (value: unknown): string => (typeof value === 'string' ? value : '')

const numberValue = (value: unknown): number | '' =>
  typeof value === 'number' && Number.isFinite(value) ? value : ''

const isMissing = (value: unknown): boolean => value === undefined || value === null || value === ''

function ObjectEditorFieldControl<TItem extends object>({
  field,
  fieldId,
  value,
  disabled,
  readOnly,
  onChange,
}: ObjectEditorFieldControlProps<TItem>) {
  if (field.type === 'key-value') {
    return (
      <KeyValueEditor
        value={
          value && typeof value === 'object' && !Array.isArray(value)
            ? (value as Record<string, string>)
            : {}
        }
        onChange={onChange}
        emptyLabel={field.emptyValueLabel}
        keyLabel={field.keyLabel ?? 'Key'}
        keyPlaceholder={field.keyPlaceholder}
        valueLabel={field.valueLabel ?? 'Value'}
        valuePlaceholder={field.valuePlaceholder}
        disabled={disabled}
        readOnly={readOnly}
      />
    )
  }
  if (field.type === 'select') {
    return (
      <NativeSelect
        id={fieldId}
        size="sm"
        value={stringValue(value)}
        onChange={(event) => onChange(event.target.value || undefined)}
        disabled={disabled || readOnly}
        aria-invalid={field.required && isMissing(value) ? 'true' : undefined}
      >
        <option value="">Not set</option>
        {field.options?.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </NativeSelect>
    )
  }
  if (field.type === 'number') {
    return (
      <Input
        id={fieldId}
        className="h-8 tabular read-only:bg-surface-subtle read-only:text-muted-foreground"
        type="number"
        value={numberValue(value)}
        onChange={(event) =>
          onChange(event.target.value === '' ? undefined : Number(event.target.value))
        }
        placeholder={field.placeholder}
        min={field.min}
        max={field.max}
        step={field.step ?? 'any'}
        disabled={disabled}
        readOnly={readOnly}
      />
    )
  }
  return (
    <Input
      id={fieldId}
      className="h-8 read-only:bg-surface-subtle read-only:text-muted-foreground"
      type={field.type === 'password' ? 'password' : 'text'}
      value={readOnly && field.type === 'password' && value ? '••••••••' : stringValue(value)}
      onChange={(event) => onChange(event.target.value || (field.preserveEmpty ? '' : undefined))}
      placeholder={field.placeholder}
      disabled={disabled}
      readOnly={readOnly}
      aria-invalid={field.required && isMissing(value) ? 'true' : undefined}
    />
  )
}

interface ObjectEditorFieldsProps<TItem extends object> {
  editorId: string
  item: TItem
  index: number
  fields: readonly ObjectEditorField<TItem>[]
  disabled: boolean
  readOnly: boolean
  onChange: (field: ObjectEditorField<TItem>, value: unknown) => void
}

function ObjectEditorFields<TItem extends object>({
  editorId,
  item,
  index,
  fields,
  disabled,
  readOnly,
  onChange,
}: ObjectEditorFieldsProps<TItem>) {
  const record = item as Record<string, unknown>
  return (
    <div className="grid grid-cols-1 gap-x-3 gap-y-3.5 border-t border-border px-3.5 py-3.5 sm:grid-cols-2">
      {fields.map((field) => {
        if (field.shouldHide?.(item)) return null
        const value = record[field.key]
        if (readOnly && (value === undefined || value === '' || value === null)) return null
        const fieldId = `${editorId}-${index}-${field.key}`
        return (
          <div
            key={field.key}
            className={cn('flex min-w-0 flex-col gap-1.5', field.fullWidth && 'sm:col-span-full')}
          >
            <label
              className="text-xs font-medium text-foreground/90"
              htmlFor={field.type === 'key-value' ? undefined : fieldId}
            >
              {field.label}
              {field.required ? <span className="text-destructive-foreground"> *</span> : null}
            </label>
            {field.helpText ? (
              <p className="-mt-0.5 text-xs leading-relaxed text-muted-foreground">
                {field.helpText}
              </p>
            ) : null}
            <ObjectEditorFieldControl
              field={field}
              fieldId={fieldId}
              value={value}
              disabled={disabled}
              readOnly={readOnly}
              onChange={(nextValue) => onChange(field, nextValue)}
            />
          </div>
        )
      })}
    </div>
  )
}

interface ObjectListEditorItemProps<TItem extends object> {
  editorId: string
  item: TItem
  index: number
  fields: readonly ObjectEditorField<TItem>[]
  label: string
  description?: string
  errors: string[]
  expanded: boolean
  removable: boolean
  disabled: boolean
  readOnly: boolean
  onToggle: () => void
  onRemove: () => void
  onChange: (field: ObjectEditorField<TItem>, value: unknown) => void
}

export function ObjectListEditorItem<TItem extends object>({
  editorId,
  item,
  index,
  fields,
  label,
  description,
  errors,
  expanded,
  removable,
  disabled,
  readOnly,
  onToggle,
  onRemove,
  onChange,
}: ObjectListEditorItemProps<TItem>) {
  const invalid = errors.length > 0
  return (
    <section
      className={cn(
        'overflow-hidden rounded-lg border bg-surface shadow-xs transition-colors',
        invalid ? 'border-destructive/40' : 'border-border',
      )}
      aria-labelledby={`${editorId}-item-${index}`}
    >
      <div className="flex items-center justify-between gap-3 px-3.5 py-2.5">
        <div className="flex min-w-0 items-center gap-3">
          <span className="grid h-6 min-w-6 place-items-center rounded-md bg-surface-sunken px-1 font-mono text-[11px] font-medium tabular text-muted-foreground">
            {String(index + 1).padStart(2, '0')}
          </span>
          <div className="min-w-0">
            <h4
              id={`${editorId}-item-${index}`}
              className="truncate text-[13px] font-medium text-foreground"
            >
              {label}
            </h4>
            {description ? (
              <p className="truncate text-xs text-muted-foreground">{description}</p>
            ) : null}
          </div>
        </div>
        {!readOnly ? (
          <div className="flex shrink-0 items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={onToggle}
              disabled={disabled}
              aria-expanded={expanded}
            >
              {expanded ? 'Done' : 'Edit'}
              <ChevronDown
                className={cn('transition-transform duration-200', expanded && 'rotate-180')}
              />
            </Button>
            {removable ? (
              <Button
                type="button"
                variant="destructive-ghost"
                size="xs"
                onClick={onRemove}
                disabled={disabled}
              >
                Remove
              </Button>
            ) : null}
          </div>
        ) : null}
      </div>
      {invalid ? (
        <ul
          className="mx-3.5 mb-3 flex list-disc flex-col gap-0.5 rounded-md bg-destructive-soft py-2 pl-7 pr-3 text-xs text-destructive-foreground"
          aria-live="polite"
        >
          {errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      ) : null}
      {expanded ? (
        <ObjectEditorFields
          editorId={editorId}
          item={item}
          index={index}
          fields={fields}
          disabled={disabled}
          readOnly={readOnly}
          onChange={onChange}
        />
      ) : null}
    </section>
  )
}
