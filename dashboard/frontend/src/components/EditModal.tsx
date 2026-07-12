import React, { useEffect, useId, useState } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import ConfirmDialog from './ConfirmDialog'
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

export interface FieldConfig<TForm extends object = EditFormData> {
  name: string
  label: string
  type: 'text' | 'number' | 'boolean' | 'select' | 'multiselect' | 'textarea' | 'percentage' | 'custom'
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
  const [initialFormData, setInitialFormData] = useState<EditFormData>({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showDiscardConfirmation, setShowDiscardConfirmation] = useState(false)
  const titleId = useId()
  const isDirty = JSON.stringify(formData) !== JSON.stringify(initialFormData)

  const requestClose = () => {
    if (saving) return
    if (isDirty) {
      setShowDiscardConfirmation(true)
      return
    }
    onClose()
  }

  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen,
    onClose: requestClose,
    dismissible: !saving,
  })

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
      setInitialFormData(convertedData)
      setError(null)
      setShowDiscardConfirmation(false)
    }
  }, [isOpen, data, fields])

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
    <>
      <div
        className={styles.overlay}
        role="presentation"
        onMouseDown={saving ? undefined : requestClose}
      >
        <div
          ref={dialogRef}
          className={styles.modal}
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          aria-busy={saving}
          tabIndex={-1}
          onMouseDown={(event) => event.stopPropagation()}
        >
          <div className={styles.header}>
            <h2 id={titleId} className={styles.title}>{title}</h2>
            <button
              className={styles.closeButton}
              type="button"
              aria-label="Close editor"
              onClick={requestClose}
              disabled={saving}
            >
              ×
            </button>
          </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && (
            <div className={styles.error} role="alert">
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
              onClick={requestClose}
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

      <ConfirmDialog
        isOpen={showDiscardConfirmation}
        title="Discard unsaved changes?"
        description="Your edits have not been saved. Closing the editor will lose them."
        eyebrow="Unsaved changes"
        confirmLabel="Discard changes"
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
