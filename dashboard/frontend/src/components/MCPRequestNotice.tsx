import styles from './MCPConfigPanel.module.css'

interface MCPRequestNoticeProps {
  message: string
  retryLabel: string
  title: string
  onDismiss?: () => void
  onRetry: () => void
}

export function MCPRequestNotice({
  message,
  retryLabel,
  title,
  onDismiss,
  onRetry,
}: MCPRequestNoticeProps) {
  return (
    <div className={styles.requestNotice} role="alert">
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
      <div className={styles.noticeActions}>
        <button type="button" className={styles.retryBtn} onClick={onRetry}>
          {retryLabel}
        </button>
        {onDismiss ? (
          <button type="button" className={styles.dismissBtn} onClick={onDismiss}>
            Dismiss
          </button>
        ) : null}
      </div>
    </div>
  )
}
