import styles from './InsightsEmptyState.module.css'

interface InsightsEmptyStateProps {
  replayUnavailable: boolean
  hasError: boolean
}

export default function InsightsEmptyState({
  replayUnavailable,
  hasError,
}: InsightsEmptyStateProps) {
  const title = hasError
    ? 'Insights couldn’t load'
    : replayUnavailable
      ? 'Waiting for routing insights'
      : 'No records in this view'
  const description = hasError
    ? 'Check the Router connection, then refresh to try again.'
    : replayUnavailable
      ? 'Routing activity appears here when replay capture is available and requests reach the router.'
      : 'No replay records are available for this view. Check your filters or send a request through the router.'

  return (
    <section className={styles.emptyState} aria-labelledby="insights-empty-title">
      <div className={styles.icon} aria-hidden="true">
        <svg
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M4 19h16M5 14V9m7 5V5m7 9v-3" />
          <circle cx="5" cy="6" r="1" />
          <circle cx="19" cy="8" r="1" />
        </svg>
      </div>
      <h3 id="insights-empty-title" className={styles.title}>
        {title}
      </h3>
      <p className={styles.description}>{description}</p>
      <a
        className={styles.guideLink}
        href="https://vllm-sr.ai/docs/api/router#router-replay"
        target="_blank"
        rel="noopener noreferrer"
      >
        Read the replay guide
        <svg
          aria-hidden="true"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M7 17 17 7M7 7h10v10" />
        </svg>
      </a>
      <details className={styles.privacy}>
        <summary className={styles.privacySummary}>
          <svg
            aria-hidden="true"
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6l-8-3Z" />
            <path d="m9 12 2 2 4-4" />
          </svg>
          Capture &amp; privacy
          <svg
            className={styles.chevron}
            aria-hidden="true"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="m9 5 7 7-7 7" />
          </svg>
        </summary>
        <div className={styles.privacyBody}>
          <p>
            Replay capture depends on global settings, decision settings, and the recipe’s privacy
            policy.
          </p>
          <p>
            <code>routing.data_policy.replay: false</code> prevents a recipe from producing replay
            records, even when global or decision capture is enabled.
          </p>
        </div>
      </details>
    </section>
  )
}
