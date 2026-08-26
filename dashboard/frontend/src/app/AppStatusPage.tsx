import React from 'react'
import ProductLoadingState from '../components/ProductLoadingState'
import styles from './AppStatus.module.css'

export interface AppStatusPageProps {
  title: string
  description: string
  actionLabel: string
  onAction: () => void
  variant?: 'loading' | 'error'
}

/** Full-screen status card used while the authenticated app is loading. */
const AppStatusPage: React.FC<AppStatusPageProps> = ({
  title,
  description,
  actionLabel,
  onAction,
  variant = 'error',
}) =>
  variant === 'loading' ? (
    <ProductLoadingState label={title} />
  ) : (
    <div className={styles.viewport}>
      <div className={styles.card} role="status" aria-live="polite">
        <div className={styles.signalRow}>
          <span className={styles.signal} aria-hidden="true" />
          <span>Control plane attention</span>
        </div>
        <h1 className={styles.title}>{title}</h1>
        <p className={styles.description}>{description}</p>
        <button type="button" className={styles.action} onClick={onAction}>
          {actionLabel}
        </button>
      </div>
    </div>
  )

export default AppStatusPage
