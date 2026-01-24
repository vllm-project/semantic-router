import React, { useState, useEffect } from 'react'
import styles from './EditModal.module.css'

type BivariantCallback<TArgs extends unknown[], R> = {
  bivarianceHack(...args: TArgs): R
}['bivarianceHack']

export type FieldValue =
  | string
  | number
  | boolean
  | string[]
  | number[]
  | unknown[]
  | Record<string, unknown>
  | null
  | undefined


type NaiveFieldValue = string | number | null | undefined
export type FormData = Record<string, FieldValue>

interface EditModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (data: FormData) => Promise<void>
  title: string
  data: FormData
  fields: FieldConfig[]
  mode?: 'edit' | 'add'
}

export interface FieldConfig {
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
  shouldHide?: BivariantCallback<[FormData], boolean>
  customRender?: BivariantCallback<[FieldValue, (value: FieldValue) => void], React.ReactNode>
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
  const [formData, setFormData] = useState<FormData>({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen) {
      // Convert percentage fields from 0-1 to 0-100 for display
      const initialData = (data && typeof data === 'object') ? (data as Record<string, unknown>) : {}
      const convertedData: FormData = { ...initialData }
      fields.forEach(field => {
        const value = convertedData[field.name]
        if (field.type === 'percentage' && typeof value === 'number') {
          convertedData[field.name] = Math.round(value * 100)
        }
      })
      setFormData(convertedData)
      setError(null)
    }
  }, [isOpen, data, fields])

  const handleChange = (fieldName: string, value: FieldValue): void => {
    setFormData((prev: FormData) => ({
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
      const convertedData: FormData = { ...formData }
      fields.forEach(field => {
        const value = convertedData[field.name]
        if (field.type === 'percentage' && typeof value === 'number') {
          convertedData[field.name] = value / 100
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
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>{title}</h2>
          <button className={styles.closeButton} onClick={onClose}>✕</button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && (
            <div className={styles.error}>
              <span className={styles.errorIcon}>⚠️</span>
              {error}
            </div>
          )}

          <div className={styles.fields}>
            {fields.map((field) => {
              if (field.shouldHide?.(formData)) return null
              const fieldValue = formData[field.name]

              return (
                <div key={field.name} className={styles.field}>
                  <label className={styles.label}>
                    {field.label}
                    {field.required && <span className={styles.required}>*</span>}
                  </label>
                  {field.description && (
                    <p className={styles.description}>{field.description}</p>
                  )}

                  {field.type === 'text' && (
                    <input
                      type="text"
                      className={styles.input}
                      value={fieldValue as NaiveFieldValue || ''}
                      onChange={(e) => handleChange(field.name, e.target.value)}
                      placeholder={field.placeholder}
                      required={field.required}
                    />
                  )}

                  {field.type === 'number' && (
                    <input
                      type="number"
                      step={field.step !== undefined ? field.step : "any"}
                      min={field.min}
                      max={field.max}
                      className={styles.input}
                      value={fieldValue as NaiveFieldValue || ''}
                      onChange={(e) => handleChange(field.name, parseFloat(e.target.value))}
                      placeholder={field.placeholder}
                      required={field.required}
                    />
                  )}

                  {field.type === 'percentage' && (
                    <div style={{ position: 'relative' }}>
                      <input
                        type="number"
                        step={field.step !== undefined ? field.step : 1}
                        min={0}
                        max={100}
                        className={styles.input}
                        value={fieldValue as NaiveFieldValue || ''}
                        onChange={(e) => {
                          const val = e.target.value
                          handleChange(field.name, val === '' ? '' : parseFloat(val))
                        }}
                        placeholder={field.placeholder}
                        required={field.required}
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
                        type="checkbox"
                        checked={Boolean(fieldValue)}
                        onChange={(e) => handleChange(field.name, e.target.checked)}
                      />
                      <span>Enable</span>
                    </label>
                  )}

                  {field.type === 'select' && (
                    <select
                      className={styles.select}
                      value={fieldValue as NaiveFieldValue || ''}
                      onChange={(e) => handleChange(field.name, e.target.value)}
                      required={field.required}
                    >
                      {field.options?.map((option) => (
                        <option key={option} value={option}>
                          {option || '(None)'}
                        </option>
                      ))}
                    </select>
                  )}

                  {field.type === 'multiselect' && (
                    <div className={styles.multiselect}>
                      {field.options?.map((option) => (
                        <label key={option} className={styles.multiselectOption}>
                          <input
                            type="checkbox"
                            checked={(fieldValue as NaiveFieldValue[] || []).includes(option)}
                            onChange={(e) => {
                              const currentValues = fieldValue as NaiveFieldValue[] || []
                              const newValues = e.target.checked
                                ? [...currentValues, option]
                                : currentValues.filter((v) => v !== option)
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
                      className={styles.textarea}
                      value={fieldValue as NaiveFieldValue || ''}
                      onChange={(e) => handleChange(field.name, e.target.value)}
                      placeholder={field.placeholder}
                      required={field.required}
                      rows={4}
                    />
                  )}

                  {field.type === 'json' && (
                    <textarea
                      className={styles.textarea}
                      value={
                        typeof fieldValue === 'object' && fieldValue !== null
                          ? JSON.stringify(fieldValue, null, 2)
                          : fieldValue as NaiveFieldValue || ''
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
                    />
                  )}

                  {field.type === 'custom' && field.customRender && (
                    <div>
                      {field.customRender(fieldValue, (value) => handleChange(field.name, value))}
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

