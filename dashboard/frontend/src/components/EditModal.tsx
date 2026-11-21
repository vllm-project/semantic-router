import React, { useEffect, useId, useRef, useState } from 'react'
import { X } from 'lucide-react'

import {
  Alert,
  Button,
  Checkbox,
  Dialog,
  DialogBody,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Input,
  Label,
  StatusDot,
  Switch,
  Textarea,
  cn,
  useRetainedValue,
} from '../ui'
import ConfirmDialog from './ConfirmDialog'
import NativeSelect from './NativeSelect'
import { focusFirstFormControl, keepOpenForToastInteraction } from './dialogFocusSupport'
import {
  FULL_WIDTH_FIELD_TYPES,
  convertPercentageFields,
  editDialogSize,
  selectFieldOptions,
  visibleFieldSections,
} from './editModalSupport'

export type EditFormData = Record<string, unknown>
type BivariantCallback<T extends (...args: never[]) => unknown> = {
  bivarianceHack: T
}['bivarianceHack']

interface EditModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (data: EditFormData) => Promise<void>
  title: string
  data: EditFormData | null
  fields: FieldConfig[]
  mode?: 'edit' | 'add'
}

export interface FieldConfig<TForm extends object = EditFormData> {
  name: string
  label: string
  section?: string
  fullWidth?: boolean
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
    (value: unknown, onChange: (value: unknown) => void, data?: TForm) => React.ReactNode
  >
}

const EditModal: React.FC<EditModalProps> = ({
  isOpen,
  onClose,
  onSave,
  title,
  data,
  fields,
  mode = 'edit',
}) => {
  const [formData, setFormData] = useState<EditFormData>({})
  const [initialFormData, setInitialFormData] = useState<EditFormData>({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showDiscardConfirmation, setShowDiscardConfirmation] = useState(false)
  const formId = useId()
  const contentRef = useRef<HTMLDivElement>(null)
  const isDirty = JSON.stringify(formData) !== JSON.stringify(initialFormData)
  // Callers clear title and fields on close; keep them while the dialog animates out.
  const shownTitle = useRetainedValue(isOpen ? title : null) ?? title
  const shownFields = useRetainedValue(isOpen ? fields : null) ?? fields
  const size = editDialogSize(shownFields)

  const requestClose = () => {
    if (saving) return
    if (isDirty) {
      setShowDiscardConfirmation(true)
      return
    }
    onClose()
  }

  const readField = (fieldName: string): unknown => formData[fieldName]
  const readString = (fieldName: string): string => {
    const value = readField(fieldName)
    return typeof value === 'string' ? value : ''
  }
  const readNumberInput = (fieldName: string): string | number => {
    const value = readField(fieldName)
    if (typeof value === 'number') return Number.isNaN(value) ? '' : value
    return typeof value === 'string' ? value : ''
  }
  const readBoolean = (fieldName: string): boolean => readField(fieldName) === true
  const readStringArray = (fieldName: string): string[] => {
    const value = readField(fieldName)
    return Array.isArray(value)
      ? value.filter((item): item is string => typeof item === 'string')
      : []
  }

  useEffect(() => {
    if (isOpen) {
      const convertedData = convertPercentageFields(data || {}, fields, 'toDisplay')
      setFormData(convertedData)
      setInitialFormData(convertedData)
      setError(null)
      setShowDiscardConfirmation(false)
    }
  }, [isOpen, data, fields])

  const handleChange = (fieldName: string, value: unknown) => {
    setFormData((prev) => ({
      ...prev,
      [fieldName]: value,
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    setError(null)

    try {
      await onSave(convertPercentageFields(formData, fields, 'toStored'))
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  const renderControl = (field: FieldConfig, fieldId: string, describedBy: string | undefined) => {
    switch (field.type) {
      case 'text':
        return (
          <Input
            id={fieldId}
            value={readString(field.name)}
            onChange={(e) => handleChange(field.name, e.target.value)}
            placeholder={field.placeholder}
            required={field.required}
            aria-describedby={describedBy}
          />
        )
      case 'number':
        return (
          <Input
            id={fieldId}
            type="number"
            inputMode="decimal"
            step={field.step !== undefined ? field.step : 'any'}
            min={field.min}
            max={field.max}
            className="tabular"
            value={readNumberInput(field.name)}
            onChange={(e) => handleChange(field.name, parseFloat(e.target.value))}
            placeholder={field.placeholder}
            required={field.required}
            aria-describedby={describedBy}
          />
        )
      case 'percentage':
        return (
          <div className="relative">
            <Input
              id={fieldId}
              type="number"
              inputMode="decimal"
              step={field.step !== undefined ? field.step : 1}
              min={0}
              max={100}
              className="tabular pr-8"
              value={readField(field.name) !== undefined ? readNumberInput(field.name) : ''}
              onChange={(e) => {
                const val = e.target.value
                handleChange(field.name, val === '' ? '' : parseFloat(val))
              }}
              placeholder={field.placeholder}
              required={field.required}
              aria-describedby={describedBy}
            />
            <span
              aria-hidden="true"
              className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[13px] text-subtle-foreground"
            >
              %
            </span>
          </div>
        )
      case 'boolean':
        return (
          <div className="flex h-9 items-center gap-2.5">
            <Switch
              id={fieldId}
              checked={readBoolean(field.name)}
              onCheckedChange={(checked) => handleChange(field.name, checked)}
              aria-describedby={describedBy}
            />
            <span className="text-[13px] text-muted-foreground" aria-hidden="true">
              {readBoolean(field.name) ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        )
      case 'select': {
        const current = readString(field.name)
        const options = selectFieldOptions(field.options, current)
        return (
          <NativeSelect
            id={fieldId}
            value={current}
            onChange={(e) => handleChange(field.name, e.target.value)}
            required={field.required}
            aria-describedby={describedBy}
          >
            {options.includes('') ? null : (
              <option value="" disabled>
                {field.placeholder ?? 'Select an option'}
              </option>
            )}
            {options.map((option) => (
              <option key={option} value={option}>
                {option || '(None)'}
              </option>
            ))}
          </NativeSelect>
        )
      }
      case 'multiselect': {
        const selected = readStringArray(field.name)
        return (
          <div
            role="group"
            aria-labelledby={`${fieldId}-label`}
            aria-describedby={describedBy}
            className="grid max-h-72 grid-cols-1 gap-1.5 overflow-y-auto sm:grid-cols-2 lg:grid-cols-3"
          >
            {field.options?.map((option) => {
              const checked = selected.includes(option)
              return (
                <label
                  key={option}
                  className={cn(
                    'flex min-w-0 cursor-pointer items-center gap-2.5 rounded-md border px-2.5 py-2 text-[13px] transition-colors',
                    checked
                      ? 'border-primary/35 bg-primary-soft text-foreground'
                      : 'border-border bg-surface text-foreground/90 hover:border-border-strong hover:bg-surface-subtle',
                  )}
                >
                  <Checkbox
                    checked={checked}
                    onCheckedChange={(next) => {
                      const currentValues = readStringArray(field.name)
                      const newValues =
                        next === true
                          ? [...currentValues, option]
                          : currentValues.filter((v: string) => v !== option)
                      handleChange(field.name, newValues)
                    }}
                  />
                  <span className="min-w-0 truncate" title={option}>
                    {option}
                  </span>
                </label>
              )
            })}
          </div>
        )
      }
      case 'textarea':
        return (
          <Textarea
            id={fieldId}
            value={readString(field.name)}
            onChange={(e) => handleChange(field.name, e.target.value)}
            placeholder={field.placeholder}
            required={field.required}
            rows={4}
            aria-describedby={describedBy}
          />
        )
      case 'custom':
        return field.customRender ? (
          <div role="group" aria-labelledby={`${fieldId}-label`} aria-describedby={describedBy}>
            {field.customRender(
              readField(field.name),
              (value) => handleChange(field.name, value),
              formData,
            )}
          </div>
        ) : null
      default:
        return null
    }
  }

  const sections = visibleFieldSections(shownFields, formData)

  return (
    <>
      <Dialog
        open={isOpen}
        onOpenChange={(open) => {
          if (!open) requestClose()
        }}
      >
        <DialogContent
          ref={contentRef}
          size={size}
          hideClose
          aria-modal="true"
          aria-describedby={undefined}
          aria-busy={saving}
          className="flex flex-col"
          onOpenAutoFocus={(event) => focusFirstFormControl(event, contentRef.current)}
          onInteractOutside={keepOpenForToastInteraction}
        >
          <DialogHeader className="flex-row items-start justify-between gap-4 border-b border-border pb-4">
            <DialogTitle className="min-w-0 pt-0.5 leading-6 [overflow-wrap:anywhere]">
              {shownTitle}
            </DialogTitle>
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              className="-mr-2 shrink-0"
              aria-label="Close editor"
              onClick={requestClose}
              disabled={saving}
            >
              <X />
            </Button>
          </DialogHeader>

          <form onSubmit={handleSubmit} className="flex min-h-0 flex-1 flex-col">
            {error ? (
              <div role="alert" className="px-6 pt-4">
                <Alert tone="danger" title="Changes were not saved">
                  {error}
                </Alert>
              </div>
            ) : null}

            <DialogBody className="flex min-h-0 flex-1 flex-col gap-6 py-5">
              {sections.map(([sectionTitle, sectionFields]) => (
                <section
                  key={sectionTitle || 'fields'}
                  className="flex flex-col gap-4 border-t border-border pt-5 first:border-t-0 first:pt-0"
                >
                  {sectionTitle ? (
                    <h3 className="text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
                      {sectionTitle}
                    </h3>
                  ) : null}
                  <div
                    className={cn(
                      'grid grid-cols-1 gap-x-4 gap-y-5',
                      size !== 'md' && 'sm:grid-cols-2',
                    )}
                  >
                    {sectionFields.map((field) => {
                      const fieldId = `${formId}-${field.name.replace(/[^a-zA-Z0-9_-]/g, '-')}`
                      const descriptionId = `${fieldId}-description`
                      const isGroupedField = field.type === 'multiselect' || field.type === 'custom'
                      const fullWidth = field.fullWidth ?? FULL_WIDTH_FIELD_TYPES.has(field.type)
                      const describedBy = field.description ? descriptionId : undefined
                      const description = field.description ? (
                        <p
                          id={descriptionId}
                          className={cn(
                            'text-xs leading-relaxed text-muted-foreground',
                            fullWidth ? '-mt-1' : 'order-last',
                          )}
                        >
                          {field.description}
                        </p>
                      ) : null

                      return (
                        <div
                          key={field.name}
                          className={cn(
                            'flex min-w-0 flex-col gap-2',
                            fullWidth && 'sm:col-span-full',
                          )}
                        >
                          <div className="flex min-h-5 items-baseline justify-between gap-3">
                            <Label
                              id={`${fieldId}-label`}
                              htmlFor={isGroupedField ? undefined : fieldId}
                              className="gap-1 leading-5"
                            >
                              {field.label}
                              {field.required ? (
                                <span className="text-destructive-foreground" aria-hidden="true">
                                  *
                                </span>
                              ) : null}
                            </Label>
                            {field.type === 'multiselect' && field.options?.length ? (
                              <span className="shrink-0 text-xs tabular text-subtle-foreground">
                                {readStringArray(field.name).length} of {field.options.length}
                              </span>
                            ) : null}
                          </div>
                          {description}
                          {renderControl(field, fieldId, describedBy)}
                        </div>
                      )
                    })}
                  </div>
                </section>
              ))}
            </DialogBody>

            <DialogFooter>
              {isDirty ? (
                <span className="mr-auto flex items-center gap-2 text-xs text-muted-foreground">
                  <StatusDot tone="warning" />
                  Unsaved changes
                </span>
              ) : null}
              <Button type="button" variant="ghost" onClick={requestClose} disabled={saving}>
                Cancel
              </Button>
              <Button type="submit" loading={saving}>
                {saving ? 'Saving…' : mode === 'add' ? 'Add' : 'Save'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        isOpen={showDiscardConfirmation}
        title="Discard unsaved changes?"
        description="Your edits have not been saved. Closing the editor will lose them."
        confirmLabel="Discard changes"
        cancelLabel="Keep editing"
        tone="warning"
        onCancel={() => setShowDiscardConfirmation(false)}
        onConfirm={() => {
          setShowDiscardConfirmation(false)
          onClose()
        }}
      />
    </>
  )
}

export default EditModal
