import { type FormEvent, type ReactNode, useEffect, useId, useState } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './ConfirmDialog.module.css'

interface ConfirmDialogProps {
  isOpen: boolean
  title: string
  description: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  eyebrow?: string
  details?: ReactNode
  pending?: boolean
  tone?: 'danger' | 'warning' | 'neutral'
  confirmationText?: string
  onCancel: () => void
  onConfirm: () => void | Promise<void>
}

export default function ConfirmDialog({
  isOpen,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  eyebrow = 'Confirm action',
  details,
  pending = false,
  tone = 'danger',
  confirmationText,
  onCancel,
  onConfirm,
}: ConfirmDialogProps) {
  const titleId = useId()
  const descriptionId = useId()
  const [confirmation, setConfirmation] = useState('')
  const dialogRef = useAccessibleDialog<HTMLElement>({
    isOpen,
    onClose: onCancel,
    dismissible: !pending,
  })
  const confirmationReady = !confirmationText || confirmation === confirmationText

  useEffect(() => {
    if (isOpen) setConfirmation('')
  }, [isOpen, confirmationText])

  if (!isOpen) return null

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (pending || !confirmationReady) return
    void onConfirm()
  }

  return (
    <div
      className={styles.overlay}
      role="presentation"
      onMouseDown={pending ? undefined : onCancel}
    >
      <section
        ref={dialogRef}
        className={styles.dialog}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={`${styles.signal} ${styles[tone]}`} aria-hidden="true">
          {tone === 'danger' ? '!' : tone === 'warning' ? '•' : 'i'}
        </div>
        <div className={styles.copy}>
          <span className={styles.eyebrow}>{eyebrow}</span>
          <h2 id={titleId}>{title}</h2>
          <div id={descriptionId} className={styles.description}>
            {description}
          </div>
        </div>

        {details ? <div className={styles.details}>{details}</div> : null}

        <form onSubmit={handleSubmit}>
          {confirmationText ? (
            <label className={styles.confirmation}>
              Type <strong>{confirmationText}</strong> to confirm
              <input
                type="text"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                autoComplete="off"
                data-dialog-initial-focus
              />
            </label>
          ) : null}

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.cancelButton}
              onClick={onCancel}
              disabled={pending}
              data-dialog-initial-focus={!confirmationText ? true : undefined}
            >
              {cancelLabel}
            </button>
            <button
              type="submit"
              className={`${styles.confirmButton} ${styles[tone]}`}
              disabled={pending || !confirmationReady}
            >
              {pending ? 'Working…' : confirmLabel}
            </button>
          </div>
        </form>
      </section>
    </div>
  )
}
