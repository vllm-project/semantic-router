import styles from './ChatComponent.module.css'
import type { PlaygroundRoutingModelStatus } from './usePlaygroundRoutingModel'

interface ChatComponentErrorsProps {
  onDismissError: () => void
  onRetryRoutingModelDiscovery: () => void
  routingModelStatus: PlaygroundRoutingModelStatus
  visibleError: string | null
}

export default function ChatComponentErrors({
  onDismissError,
  onRetryRoutingModelDiscovery,
  routingModelStatus,
  visibleError,
}: ChatComponentErrorsProps) {
  if (routingModelStatus === 'error' && !visibleError) {
    return (
      <div className={styles.error} role="alert">
        <span className={styles.errorIcon}>⚠️</span>
        <span>The automatic routing model is unavailable.</span>
        <button type="button" className={styles.errorAction} onClick={onRetryRoutingModelDiscovery}>
          Retry discovery
        </button>
      </div>
    )
  }

  if (!visibleError) return null

  return (
    <div className={styles.error}>
      <span className={styles.errorIcon}>⚠️</span>
      <span>{visibleError}</span>
      <button className={styles.errorDismiss} onClick={onDismissError}>
        ×
      </button>
    </div>
  )
}
