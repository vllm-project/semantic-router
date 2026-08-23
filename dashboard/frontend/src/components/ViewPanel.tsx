import React from 'react'

import ProductIcon, { type ProductIconName } from './ProductIcon'
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

export interface ViewPanelAction {
  label: string
  icon?: ProductIconName
  onClick: () => void
  tone?: 'secondary' | 'primary' | 'destructive'
  disabled?: boolean
  availableWhenReadonly?: boolean
}

interface ViewPanelProps {
  title: string
  sections: ViewSection[]
  onClose?: () => void
  closeLabel?: string
  onEdit?: () => void
  actions?: ViewPanelAction[]
  variant?: 'modal' | 'page'
}

const ViewPanel: React.FC<ViewPanelProps> = ({
  title,
  sections,
  onClose,
  closeLabel = 'Close',
  onEdit,
  actions = [],
  variant = 'modal',
}) => {
  const panelClassName = [styles.modal, variant === 'page' ? styles.modalStandalone : '']
    .filter(Boolean)
    .join(' ')
  const contentClassName = [styles.content, variant === 'page' ? styles.contentStandalone : '']
    .filter(Boolean)
    .join(' ')
  const hasFooter = actions.length > 0 || Boolean(onClose) || Boolean(onEdit)

  return (
    <div className={panelClassName}>
      <div className={styles.header}>
        <div className={styles.headerIdentity}>
          <div className={styles.headerLogo} aria-hidden="true">
            <img src="/vllm.png" alt="" />
          </div>
          <div>
            <span className={styles.eyebrow}>Details</span>
            <h2 className={styles.title}>{title}</h2>
          </div>
        </div>
        {onClose ? (
          <button
            className={styles.closeButton}
            onClick={onClose}
            type="button"
            aria-label={closeLabel}
          >
            <ProductIcon name="close" />
          </button>
        ) : null}
      </div>

      <div className={contentClassName}>
        {sections.map((section, sectionIndex) => (
          <div key={sectionIndex} className={styles.section}>
            {section.title ? <h3 className={styles.sectionTitle}>{section.title}</h3> : null}
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

      {hasFooter ? (
        <div className={styles.footer}>
          {actions.map((action) => (
            <button
              key={action.label}
              className={
                action.tone === 'primary'
                  ? styles.primaryFooterButton
                  : action.tone === 'destructive'
                    ? styles.destructiveFooterButton
                    : styles.closeFooterButton
              }
              onClick={action.onClick}
              type="button"
              disabled={action.disabled}
            >
              {action.icon ? <ProductIcon name={action.icon} /> : null}
              {action.label}
            </button>
          ))}
          {onClose ? (
            <button className={styles.closeFooterButton} onClick={onClose} type="button">
              <ProductIcon name="check" />
              {closeLabel}
            </button>
          ) : null}
          {onEdit ? (
            <button className={styles.editFooterButton} onClick={onEdit} type="button">
              <ProductIcon name="edit" />
              Edit
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

export default ViewPanel
