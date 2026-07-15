import { type ReactNode, useId } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './OpenClawPage.module.css'

interface OpenClawDialogProps {
  busy?: boolean
  children: ReactNode
  footer?: ReactNode
  isOpen: boolean
  title: string
  wide?: boolean
  onClose: () => void
}

export function OpenClawDialog({
  busy = false,
  children,
  footer,
  isOpen,
  title,
  wide = false,
  onClose,
}: OpenClawDialogProps) {
  const titleId = useId()
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen,
    onClose,
    dismissible: !busy,
  })

  if (!isOpen) return null

  return (
    <div
      className={styles.ocModalOverlay}
      role="presentation"
      onMouseDown={busy ? undefined : onClose}
    >
      <div
        ref={dialogRef}
        className={`${styles.ocModal} ${wide ? styles.ocModalWide : ''}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-busy={busy}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={styles.ocModalHeader}>
          <h3 id={titleId} className={styles.ocModalTitle}>
            {title}
          </h3>
          <button
            type="button"
            className={styles.ocModalClose}
            onClick={onClose}
            disabled={busy}
            aria-label={`Close ${title}`}
          >
            ×
          </button>
        </div>
        <div className={styles.ocModalBody}>{children}</div>
        {footer ? <div className={styles.ocModalFooter}>{footer}</div> : null}
      </div>
    </div>
  )
}
