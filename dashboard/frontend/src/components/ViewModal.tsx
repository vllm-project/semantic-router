import React from 'react'
import styles from './ViewModal.module.css'

export interface ViewField {
  label: string
  value: React.ReactNode
  fullWidth?: boolean
}

export interface ViewSection {
  title?: string
  fields: ViewField[]
}

interface ViewModalProps {
  isOpen: boolean
  onClose: () => void
  onEdit?: () => void
  title: string
  sections: ViewSection[]
  onPrimaryAction?: () => void
  primaryActionLabel?: string
  primaryActionDisabled?: boolean
  primaryActionLoading?: boolean
}

const ViewModal: React.FC<ViewModalProps> = ({
  isOpen,
  onClose,
  onEdit,
  title,
  sections,
  onPrimaryAction,
  primaryActionLabel,
  primaryActionDisabled,
  primaryActionLoading
}) => {
  if (!isOpen) return null

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>{title}</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div className={styles.content}>
          {sections.map((section, sectionIndex) => (
            <div key={sectionIndex} className={styles.section}>
              {section.title && (
                <h3 className={styles.sectionTitle}>{section.title}</h3>
              )}
              <div className={styles.fieldsGrid}>
                {section.fields.map((field, fieldIndex) => (
                  <div
                    key={fieldIndex}
                    className={`${styles.field} ${field.fullWidth ? styles.fullWidth : ''}`}
                  >
                    <div className={styles.fieldLabel}>{field.label}</div>
                    <div className={styles.fieldValue}>{field.value}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className={styles.footer}>
          <button className={styles.closeFooterButton} onClick={onClose}>
            Close
          </button>
          {primaryActionLabel && onPrimaryAction && (
            <button
              className={styles.primaryFooterButton}
              onClick={onPrimaryAction}
              disabled={primaryActionDisabled}
            >
              {primaryActionLoading ? 'Working...' : primaryActionLabel}
            </button>
          )}
          {onEdit && (
            <button className={styles.editFooterButton} onClick={onEdit}>
              Edit
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default ViewModal

