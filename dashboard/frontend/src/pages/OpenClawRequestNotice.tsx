import styles from './OpenClawPage.module.css'

interface OpenClawRequestNoticeProps {
  message: string
  title: string
  retryLabel?: string
  tone?: 'error' | 'warning'
  onDismiss?: () => void
  onRetry?: () => void
}

export function OpenClawRequestNotice({
  message,
  title,
  retryLabel = 'Retry',
  tone = 'error',
  onDismiss,
  onRetry,
}: OpenClawRequestNoticeProps) {
  return (
    <div
      className={`${styles.enterpriseRequestNotice} ${styles[`enterpriseRequestNotice_${tone}`]}`}
      role={tone === 'error' ? 'alert' : 'status'}
    >
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
      <div className={styles.enterpriseNoticeActions}>
        {onRetry ? (
          <button type="button" onClick={onRetry}>
            {retryLabel}
          </button>
        ) : null}
        {onDismiss ? (
          <button type="button" onClick={onDismiss}>
            Dismiss
          </button>
        ) : null}
      </div>
    </div>
  )
}
