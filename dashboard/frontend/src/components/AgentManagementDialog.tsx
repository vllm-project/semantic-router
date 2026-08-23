import type { ReactNode } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import ProductIcon from './ProductIcon'
import styles from './AgentManagementPanel.module.css'

interface AgentManagementDialogProps {
  children: ReactNode
  eyebrow: string
  title: string
  description?: string
  busy?: boolean
  onClose: () => void
}

export default function AgentManagementDialog({
  children,
  eyebrow,
  title,
  description,
  busy = false,
  onClose,
}: AgentManagementDialogProps) {
  const dialogRef = useAccessibleDialog<HTMLElement>({
    isOpen: true,
    onClose,
    dismissible: !busy,
  })

  return (
    <div
      className={styles.dialogBackdrop}
      onMouseDown={(event) => {
        if (!busy && event.target === event.currentTarget) onClose()
      }}
    >
      <section
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="agent-management-dialog-title"
        tabIndex={-1}
      >
        <header className={styles.dialogHeader}>
          <div className={styles.dialogLogo}>
            <img src="/vllm.png" alt="" />
          </div>
          <div>
            <span>{eyebrow}</span>
            <h2 id="agent-management-dialog-title">{title}</h2>
            {description ? <p>{description}</p> : null}
          </div>
          <button type="button" onClick={onClose} disabled={busy} aria-label="Close dialog">
            <ProductIcon name="close" />
          </button>
        </header>
        {children}
      </section>
    </div>
  )
}
