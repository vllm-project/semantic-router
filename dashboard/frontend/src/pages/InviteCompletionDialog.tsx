import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './InviteCompletionDialog.module.css'

interface InviteCompletionDialogProps {
  firstName: string
  busy: boolean
  error: string
  onCreateKey: () => void
  onExplore: () => void
}

export default function InviteCompletionDialog({
  firstName,
  busy,
  error,
  onCreateKey,
  onExplore,
}: InviteCompletionDialogProps) {
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose: onExplore,
    dismissible: !busy,
  })

  return (
    <div className={styles.backdrop} onMouseDown={() => !busy && onExplore()}>
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
        <div className={styles.logo} aria-hidden="true">
          <img src="/vllm.png" alt="" />
        </div>
        <span className={styles.eyebrow}>Welcome aboard</span>
        <h2 id="invite-complete-title">You’re in, {firstName}.</h2>
        <p id="invite-complete-description">Your workspace and Team access are ready.</p>
        {error ? (
          <div className={styles.error} role="alert">
            {error}
          </div>
        ) : null}
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.primaryAction}
            onClick={onCreateKey}
            disabled={busy}
            data-dialog-initial-focus
          >
            {busy ? 'Preparing your key…' : 'Continue with API key'}
          </button>
          <button
            type="button"
            className={styles.secondaryAction}
            onClick={onExplore}
            disabled={busy}
          >
            Not now
          </button>
        </div>
      </section>
    </div>
  )
}
