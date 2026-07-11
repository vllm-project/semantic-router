import React, { useEffect, useId, useRef, useState } from 'react'
import styles from './EditModal.module.css'

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

const focusableSelector = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

export interface FieldConfig<TForm extends object = EditFormData> {
  name: string
  label: string
  type: 'text' | 'number' | 'boolean' | 'select' | 'multiselect' | 'textarea' | 'json' | 'percentage' | 'custom'
  required?: boolean
  options?: string[]
  placeholder?: string
  description?: string
  min?: number
  max?: number
  step?: number
  shouldHide?: BivariantCallback<(data: TForm) => boolean>
  customRender?: BivariantCallback<(value: unknown, onChange: (value: unknown) => void) => React.ReactNode>
}

const EditModal: React.FC<EditModalProps> = ({
  isOpen,
  onClose,
  onSave,
  title,
  data,
  fields,
  mode = 'edit'
}) => {
  const [formData, setFormData] = useState<EditFormData>({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement>(null)

  const readField = (fieldName: string): unknown => formData[fieldName]
  const readString = (fieldName: string): string => {
    const value = readField(fieldName)
    return typeof value === 'string' ? value : ''
  }
  const readNumberInput = (fieldName: string): string | number => {
    const value = readField(fieldName)
    return typeof value === 'number' || typeof value === 'string' ? value : ''
  }
  const readBoolean = (fieldName: string): boolean => readField(fieldName) === true
  const readStringArray = (fieldName: string): string[] => {
    const value = readField(fieldName)
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
  }

  useEffect(() => {
    if (isOpen) {
      // Convert percentage fields from 0-1 to 0-100 for display
      const convertedData: EditFormData = { ...(data || {}) }
      fields.forEach(field => {
        if (field.type !== 'percentage') {
          return
        }
        const rawValue = convertedData[field.name]
        const numericValue = typeof rawValue === 'number'
          ? rawValue
          : typeof rawValue === 'string' && rawValue.trim() !== ''
            ? Number(rawValue)
            : NaN
        if (Number.isFinite(numericValue)) {
          convertedData[field.name] = Math.round(numericValue * 100)
        }
      })
      setFormData(convertedData)
      setError(null)
    }
  }, [isOpen, data, fields])

  useEffect(() => {
    if (!isOpen) return

    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
    const previousBodyOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const frame = window.requestAnimationFrame(() => {
      const firstControl = dialogRef.current?.querySelector<HTMLElement>(focusableSelector)
      ;(firstControl ?? dialogRef.current)?.focus()
    })

    return () => {
      window.cancelAnimationFrame(frame)
      document.body.style.overflow = previousBodyOverflow
      previouslyFocused?.focus()
    }
  }, [isOpen])

  const handleDialogKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      onClose()
      return
    }

    if (event.key !== 'Tab' || !dialogRef.current) return

    const controls = Array.from(
      dialogRef.current.querySelectorAll<HTMLElement>(focusableSelector),
    ).filter((element) => element.getAttribute('aria-hidden') !== 'true')

    if (controls.length === 0) {
      event.preventDefault()
      dialogRef.current.focus()
      return
    }

    const firstControl = controls[0]
    const lastControl = controls[controls.length - 1]
    const activeElement = document.activeElement

    if (
      event.shiftKey
      && (activeElement === firstControl || !dialogRef.current.contains(activeElement))
    ) {
      event.preventDefault()
      lastControl.focus()
    } else if (!event.shiftKey && activeElement === lastControl) {
      event.preventDefault()
      firstControl.focus()
    }
  }

  const handleChange = (fieldName: string, value: unknown) => {
    setFormData((prev) => ({
      ...prev,
      [fieldName]: value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    setError(null)

    try {
      // Convert percentage fields from 0-100 back to 0-1 before saving
      const convertedData: EditFormData = { ...formData }
      fields.forEach(field => {
        if (field.type !== 'percentage') {
          return
        }
        const rawValue = convertedData[field.name]
        const numericValue = typeof rawValue === 'number'
          ? rawValue
          : typeof rawValue === 'string' && rawValue.trim() !== ''
            ? Number(rawValue)
            : NaN
        if (Number.isFinite(numericValue)) {
          convertedData[field.name] = numericValue / 100
        }
      })
      await onSave(convertedData)
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div
        ref={dialogRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={handleDialogKeyDown}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.header}>
          <h2 id={titleId} className={styles.title}>{title}</h2>
          <button
            className={styles.closeButton}
            type="button"
            aria-label="Close editor"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && (
            <div className={styles.error}>
              {error}
            </div>
          )}

          <div className={styles.fields}>
            {fields.map((field) => {
              if (field.shouldHide?.(formData)) return null
              const fieldId = `${titleId}-${field.name.replace(/[^a-zA-Z0-9_-]/g, '-')}`
              const descriptionId = `${fieldId}-description`
              const isGroupedField = field.type === 'multiselect' || field.type === 'custom'

              return (
              <div key={field.name} className={styles.field}>
                <label
                  id={`${fieldId}-label`}
                  className={styles.label}
                  htmlFor={isGroupedField ? undefined : fieldId}
                >
                  {field.label}
                  {field.required && <span className={styles.required}>*</span>}
                </label>
                {field.description && (
                  <p id={descriptionId} className={styles.description}>{field.description}</p>
                )}

                {field.type === 'text' && (
                  <input
                    id={fieldId}
                    type="text"
                    className={styles.input}
                    value={readString(field.name)}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    placeholder={field.placeholder}
                    required={field.required}
                    aria-describedby={field.description ? descriptionId : undefined}
                  />
                )}

                {field.type === 'number' && (
                  <input
                    id={fieldId}
                    type="number"
                    step={field.step !== undefined ? field.step : "any"}
                    min={field.min}
                    max={field.max}
                    className={styles.input}
                    value={readNumberInput(field.name)}
                    onChange={(e) => handleChange(field.name, parseFloat(e.target.value))}
                    placeholder={field.placeholder}
                    required={field.required}
                    aria-describedby={field.description ? descriptionId : undefined}
                  />
                )}

                {field.type === 'percentage' && (
                  <div style={{ position: 'relative' }}>
                    <input
                      id={fieldId}
                      type="number"
                      step={field.step !== undefined ? field.step : 1}
                      min={0}
                      max={100}
                      className={styles.input}
                      value={readField(field.name) !== undefined ? readNumberInput(field.name) : ''}
                      onChange={(e) => {
                        const val = e.target.value
                        handleChange(field.name, val === '' ? '' : parseFloat(val))
                      }}
                      placeholder={field.placeholder}
                      required={field.required}
                      aria-describedby={field.description ? descriptionId : undefined}
                      style={{ paddingRight: '2.5rem' }}
                    />
                    <span style={{
                      position: 'absolute',
                      right: '0.75rem',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      color: 'var(--color-text-secondary)',
                      fontSize: '0.875rem',
                      pointerEvents: 'none'
                    }}>
                      %
                    </span>
                  </div>
                )}

                {field.type === 'boolean' && (
                  <label className={styles.checkbox}>
                    <input
                      id={fieldId}
                      type="checkbox"
                      checked={readBoolean(field.name)}
                      onChange={(e) => handleChange(field.name, e.target.checked)}
                    />
                    <span>Enable</span>
                  </label>
                )}

                {field.type === 'select' && (
                  <select
                    id={fieldId}
                    className={styles.select}
                    value={readString(field.name)}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    required={field.required}
                    aria-describedby={field.description ? descriptionId : undefined}
                  >
                    {field.options?.map((option) => (
                      <option key={option} value={option}>
                        {option || '(None)'}
                      </option>
                    ))}
                  </select>
                )}

                {field.type === 'multiselect' && (
                  <div
                    className={styles.multiselect}
                    role="group"
                    aria-labelledby={`${fieldId}-label`}
                    aria-describedby={field.description ? descriptionId : undefined}
                  >
                    {field.options?.map((option) => (
                      <label key={option} className={styles.multiselectOption}>
                        <input
                          type="checkbox"
                          checked={readStringArray(field.name).includes(option)}
                          onChange={(e) => {
                            const currentValues = readStringArray(field.name)
                            const newValues = e.target.checked
                              ? [...currentValues, option]
                              : currentValues.filter((v: string) => v !== option)
                            handleChange(field.name, newValues)
                          }}
                        />
                        <span>{option}</span>
                      </label>
                    ))}
                  </div>
                )}

                {field.type === 'textarea' && (
                  <textarea
                    id={fieldId}
                    className={styles.textarea}
                    value={readString(field.name)}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    placeholder={field.placeholder}
                    required={field.required}
                    rows={4}
                    aria-describedby={field.description ? descriptionId : undefined}
                  />
                )}

                {field.type === 'json' && (
                  <textarea
                    id={fieldId}
                    className={styles.textarea}
                    value={
                      typeof readField(field.name) === 'object' && readField(field.name) !== null
                        ? JSON.stringify(readField(field.name), null, 2)
                        : readString(field.name)
                    }
                    onChange={(e) => {
                      try {
                        const parsed = JSON.parse(e.target.value)
                        handleChange(field.name, parsed)
                      } catch {
                        handleChange(field.name, e.target.value)
                      }
                    }}
                    placeholder={field.placeholder}
                    required={field.required}
                    rows={6}
                    aria-describedby={field.description ? descriptionId : undefined}
                  />
                )}

                {field.type === 'custom' && field.customRender && (
                  <div
                    role="group"
                    aria-labelledby={`${fieldId}-label`}
                    aria-describedby={field.description ? descriptionId : undefined}
                  >
                    {field.customRender(readField(field.name), (value) => handleChange(field.name, value))}
                  </div>
                )}
              </div>
              )
            })}
          </div>

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.cancelButton}
              onClick={onClose}
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="submit"
              className={styles.saveButton}
              disabled={saving}
            >
              {saving ? 'Saving...' : mode === 'add' ? 'Add' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default EditModal
