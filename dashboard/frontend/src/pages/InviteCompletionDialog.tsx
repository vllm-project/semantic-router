import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './InviteCompletionDialog.module.css'

interface InviteCompletionDialogProps {
  firstName: string
  onCreateKey: () => void
  onExplore: () => void
}

export default function InviteCompletionDialog({
  firstName,
  onCreateKey,
  onExplore,
}: InviteCompletionDialogProps) {
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose: onExplore,
  })

  return (
    <div className={styles.backdrop} onMouseDown={onExplore}>
      <section
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="invite-complete-title"
        aria-describedby="invite-complete-description"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={styles.icon} aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <circle cx="8.5" cy="15.5" r="3.5" />
            <path d="M11.6 13.8 21 4.4M17.4 8l2.3 2.3M14.8 10.6l2.2 2.2" />
          </svg>
        </div>
        <span className={styles.eyebrow}>Welcome aboard</span>
        <h2 id="invite-complete-title">You’re ready, {firstName}.</h2>
        <p id="invite-complete-description">
          Create your first API key and make the workspace yours.
        </p>
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.primaryAction}
            onClick={onCreateKey}
            data-dialog-initial-focus
          >
            Create my API key
          </button>
          <button type="button" className={styles.secondaryAction} onClick={onExplore}>
            Explore first
          </button>
        </div>
      </section>
    </div>
  )
}
