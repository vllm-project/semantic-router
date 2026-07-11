import styles from './ChatComponent.module.css'
import type { PlaygroundRoutingModelStatus } from './usePlaygroundRoutingModel'

interface PlaygroundRoutingStatusProps {
  model: string
  onRetry: () => void
  status: PlaygroundRoutingModelStatus
}

const STATUS_LABELS: Record<PlaygroundRoutingModelStatus, string> = {
  discovering: 'Discovering route',
  ready: 'Live routing',
  error: 'Route unavailable',
}

export default function PlaygroundRoutingStatus({ model, onRetry, status }: PlaygroundRoutingStatusProps) {
  return (
    <div className={styles.routingStatus} data-testid="playground-routing-status">
      <div className={styles.routingStatusIdentity}>
        <span className={styles.routingStatusMark} aria-hidden="true" />
        <span className={styles.routingStatusTitle}>Semantic Router</span>
        <span className={styles.routingStatusDivider} aria-hidden="true" />
        <span className={styles.routingStatusLabel}>{STATUS_LABELS[status]}</span>
      </div>
      {status === 'error' ? (
        <button type="button" className={styles.routingStatusRetry} onClick={onRetry}>
          Retry discovery
        </button>
      ) : (
        <code className={styles.routingStatusModel}>{model}</code>
      )}
    </div>
  )
}
