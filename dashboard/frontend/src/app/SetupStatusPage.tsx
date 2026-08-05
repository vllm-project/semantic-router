import React from 'react'
import styles from './AppStatus.module.css'

export interface SetupStatusPageProps {
  title: string
  description: string
  actionLabel: string
  onAction: () => void
  variant?: 'loading' | 'error'
}

/** Full-screen status card used during auth/setup loading and errors. */
const SetupStatusPage: React.FC<SetupStatusPageProps> = ({
  title,
  description,
  actionLabel,
  onAction,
  variant = 'error',
}) => (
  <div className={styles.viewport}>
    <div className={styles.card} role="status" aria-live="polite">
      <div className={styles.signalRow}>
        <span className={`${styles.signal} ${variant === 'loading' ? styles.loadingSignal : ''}`} aria-hidden="true" />
        <span>{variant === 'loading' ? 'Control plane startup' : 'Control plane attention'}</span>
      </div>
      <h1 className={styles.title}>{title}</h1>
      <p className={styles.description}>{description}</p>
      {variant === 'error' ? (
        <button type="button" className={styles.action} onClick={onAction}>
          {actionLabel}
        </button>
      ) : null}
    </div>
  </div>
)

export default SetupStatusPage
