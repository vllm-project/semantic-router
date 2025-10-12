import React, { useState, useEffect } from 'react'
import styles from './EditModal.module.css'

interface EditModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (data: any) => Promise<void>
  title: string
  data: any
  fields: FieldConfig[]
  mode?: 'edit' | 'add'
}

export interface FieldConfig {
  name: string
  label: string
  type: 'text' | 'number' | 'boolean' | 'select' | 'multiselect' | 'textarea' | 'json'
  required?: boolean
  options?: string[]
  placeholder?: string
  description?: string
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
  const [formData, setFormData] = useState<any>({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen) {
      setFormData(data || {})
      setError(null)
    }
  }, [isOpen, data])

  const handleChange = (fieldName: string, value: any) => {
    setFormData((prev: any) => ({
      ...prev,
      [fieldName]: value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    setError(null)

    try {
      await onSave(formData)
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
            {fields.map((field) => (
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
                    value={formData[field.name] || ''}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    placeholder={field.placeholder}
                    required={field.required}
                  />
                )}

                {field.type === 'number' && (
                  <input
                    type="number"
                    step="any"
                    className={styles.input}
                    value={formData[field.name] || ''}
                    onChange={(e) => handleChange(field.name, parseFloat(e.target.value))}
                    placeholder={field.placeholder}
                    required={field.required}
                  />
                )}

                {field.type === 'boolean' && (
                  <label className={styles.checkbox}>
                    <input
                      type="checkbox"
                      checked={formData[field.name] || false}
                      onChange={(e) => handleChange(field.name, e.target.checked)}
                    />
                    <span>Enable</span>
                  </label>
                )}

                {field.type === 'select' && (
                  <select
                    className={styles.select}
                    value={formData[field.name] || ''}
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
                          checked={(formData[field.name] || []).includes(option)}
                          onChange={(e) => {
                            const currentValues = formData[field.name] || []
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
                    className={styles.textarea}
                    value={formData[field.name] || ''}
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
                      typeof formData[field.name] === 'object'
                        ? JSON.stringify(formData[field.name], null, 2)
                        : formData[field.name] || ''
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
              </div>
            ))}
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

