import type { ReactNode } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import ProductIcon, { type ProductIconName } from './ProductIcon'
import styles from './ProductMomentDialog.module.css'

export interface ProductMomentAction {
  label: string
  icon: ProductIconName
  tone: 'primary' | 'secondary'
  onClick: () => void
  initialFocus?: boolean
}

interface ProductMomentDialogProps {
  titleId: string
  eyebrow: string
  title: string
  description: string
  children?: ReactNode
  actions: ProductMomentAction[]
  onClose?: () => void
}

export default function ProductMomentDialog({
  titleId,
  eyebrow,
  title,
  description,
  children,
  actions,
  onClose,
}: ProductMomentDialogProps) {
  const dismissible = Boolean(onClose)
  const dialogRef = useAccessibleDialog<HTMLElement>({
    isOpen: true,
    onClose: onClose ?? (() => undefined),
    dismissible,
  })

  return (
    <div
      className={styles.backdrop}
      onMouseDown={(event) => {
        if (dismissible && event.target === event.currentTarget) onClose?.()
      }}
    >
      <section
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        {onClose ? (
          <button type="button" className={styles.close} onClick={onClose} aria-label="Close">
            <ProductIcon name="close" />
          </button>
        ) : null}
        <div className={styles.logo} aria-hidden="true">
          <img src="/vllm.png" alt="" />
        </div>
        <span className={styles.eyebrow}>{eyebrow}</span>
        <h2 id={titleId}>{title}</h2>
        <p>{description}</p>
        {children ? <div className={styles.content}>{children}</div> : null}
        <div className={styles.actions} data-action-count={actions.length}>
          {actions.map((action) => (
            <button
              type="button"
              key={action.label}
              className={action.tone === 'primary' ? styles.primary : styles.secondary}
              onClick={action.onClick}
              data-dialog-initial-focus={action.initialFocus ? '' : undefined}
            >
              {action.tone === 'secondary' ? <ProductIcon name={action.icon} /> : null}
              {action.label}
              {action.tone === 'primary' ? <ProductIcon name={action.icon} /> : null}
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
