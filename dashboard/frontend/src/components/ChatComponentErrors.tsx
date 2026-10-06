import styles from './ChatComponentErrors.module.css'
import type { PlaygroundErrorPresentation } from './playgroundErrorPresentation'
import type { PlaygroundRoutingModelStatus } from './usePlaygroundRoutingModel'

interface ChatComponentErrorsProps {
  overlay: boolean
  onDismissError: () => void
  onRetryRoutingModelDiscovery: () => void
  routingModelStatus: PlaygroundRoutingModelStatus
  visibleError: PlaygroundErrorPresentation | null
}

export default function ChatComponentErrors({
  overlay,
  onDismissError,
  onRetryRoutingModelDiscovery,
  routingModelStatus,
  visibleError,
}: ChatComponentErrorsProps) {
  const routingModelUnavailable = routingModelStatus === 'error' && !visibleError
  if (!routingModelUnavailable && !visibleError) return null

  return (
    <div
      className={[styles.region, overlay ? styles.regionOverlay : ''].filter(Boolean).join(' ')}
      data-testid="playground-error-region"
    >
      <div className={styles.alert} role="alert">
        <span className={styles.icon} aria-hidden="true">
          !
        </span>
        <div className={styles.copy}>
          <span className={styles.message}>
            {routingModelUnavailable
              ? 'The automatic routing model is unavailable.'
              : visibleError?.message}
          </span>
          {!routingModelUnavailable && visibleError?.technicalDetails ? (
            <details className={styles.details} data-playground-technical-details="true">
              <summary>Technical details</summary>
              <pre>{visibleError.technicalDetails}</pre>
            </details>
          ) : null}
        </div>
        {routingModelUnavailable ? (
          <button
            type="button"
            className={styles.action}
            onClick={onRetryRoutingModelDiscovery}
          >
            Retry discovery
          </button>
        ) : (
          <button
            type="button"
            className={styles.dismiss}
            aria-label="Dismiss error"
            onClick={onDismissError}
          >
            ×
          </button>
        )}
      </div>
    </div>
  )
}
