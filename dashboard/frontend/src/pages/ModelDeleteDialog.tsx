import { useEffect, useId, useState } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './ModelDeleteDialog.module.css'

interface ModelDeleteDialogProps {
  modelNames: string[]
  pending: boolean
  onCancel: () => void
  onConfirm: () => void
}

export default function ModelDeleteDialog({
  modelNames,
  pending,
  onCancel,
  onConfirm,
}: ModelDeleteDialogProps) {
  const titleId = useId()
  const descriptionId = useId()
  const [confirmation, setConfirmation] = useState('')
  const requiredConfirmation = modelNames.length > 1 ? `DELETE ${modelNames.length}` : ''
  const confirmationReady = modelNames.length === 1 || confirmation === requiredConfirmation
  const isOpen = modelNames.length > 0
  const dismissible = !pending && confirmation.length === 0
  const dialogRef = useAccessibleDialog<HTMLElement>({
    isOpen,
    onClose: onCancel,
    dismissible,
  })

  useEffect(() => {
    setConfirmation('')
  }, [modelNames])

  if (modelNames.length === 0) {
    return null
  }

  return (
    <div
      className={styles.overlay}
      role="presentation"
      onMouseDown={dismissible ? onCancel : undefined}
    >
      <section
        ref={dialogRef}
        className={styles.dialog}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        aria-busy={pending}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={styles.signal} aria-hidden="true">!</div>
        <div className={styles.copy}>
          <span className={styles.eyebrow}>Destructive configuration change</span>
          <h2 id={titleId}>Delete {modelNames.length} {modelNames.length === 1 ? 'model' : 'models'}?</h2>
          <p id={descriptionId}>
            The selected models are not defaults and are not referenced by routing decisions.
            This operation writes the configuration once and cannot be undone from the dashboard.
          </p>
        </div>

        <div className={styles.modelList}>
          {modelNames.slice(0, 6).map((name) => <code key={name}>{name}</code>)}
          {modelNames.length > 6 ? <span>+{modelNames.length - 6} more</span> : null}
        </div>

        {modelNames.length > 1 ? (
          <label className={styles.confirmation}>
            Type <strong>{requiredConfirmation}</strong> to confirm
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
            data-dialog-initial-focus={modelNames.length === 1 ? true : undefined}
          >
            Cancel
          </button>
          <button
            type="button"
            className={styles.deleteButton}
            onClick={onConfirm}
            disabled={pending || !confirmationReady}
          >
            {pending ? 'Deleting…' : `Delete ${modelNames.length === 1 ? 'model' : 'models'}`}
          </button>
        </div>
      </section>
    </div>
  )
}
