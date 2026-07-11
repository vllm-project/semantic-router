import { useEffect, useId, useRef, useState } from 'react'

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
  const dialogRef = useRef<HTMLElement>(null)
  const onCancelRef = useRef(onCancel)
  const pendingRef = useRef(pending)
  const [confirmation, setConfirmation] = useState('')
  const requiredConfirmation = modelNames.length > 1 ? `DELETE ${modelNames.length}` : ''
  const confirmationReady = modelNames.length === 1 || confirmation === requiredConfirmation
  const isOpen = modelNames.length > 0

  onCancelRef.current = onCancel
  pendingRef.current = pending

  useEffect(() => {
    setConfirmation('')
  }, [modelNames])

  useEffect(() => {
    if (!isOpen) return

    const previousActiveElement = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const focusableSelector = [
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[href]',
      '[tabindex]:not([tabindex="-1"])',
    ].join(',')

    const focusableElements = () => Array.from(
      dialogRef.current?.querySelectorAll<HTMLElement>(focusableSelector) ?? [],
    ).filter((element) => element.getAttribute('aria-hidden') !== 'true')

    window.requestAnimationFrame(() => {
      const elements = focusableElements()
      ;(dialogRef.current?.querySelector('input') as HTMLElement | null ?? elements[0] ?? dialogRef.current)?.focus()
    })

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !pendingRef.current) {
        event.preventDefault()
        onCancelRef.current()
        return
      }

      if (event.key !== 'Tab') return
      const elements = focusableElements()
      if (elements.length === 0) {
        event.preventDefault()
        dialogRef.current?.focus()
        return
      }

      const first = elements[0]
      const last = elements[elements.length - 1]
      const activeElement = document.activeElement
      if (event.shiftKey && (activeElement === first || !dialogRef.current?.contains(activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (activeElement === last || !dialogRef.current?.contains(activeElement))) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = previousOverflow
      previousActiveElement?.focus()
    }
  }, [isOpen])

  if (modelNames.length === 0) {
    return null
  }

  return (
    <div className={styles.overlay} role="presentation" onMouseDown={pending ? undefined : onCancel}>
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
              autoFocus
            />
          </label>
        ) : null}

        <div className={styles.actions}>
          <button type="button" className={styles.cancelButton} onClick={onCancel} disabled={pending}>
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
