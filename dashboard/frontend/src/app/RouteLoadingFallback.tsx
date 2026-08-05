import React from 'react'
import styles from './AppStatus.module.css'

const RouteLoadingFallback: React.FC = () => (
  <div aria-live="polite" role="status" className={styles.routeFallback}>
    <div className={styles.routeStatus}>
      <span>Loading control plane</span>
      <span className={styles.routeLine} aria-hidden="true" />
    </div>
  </div>
)

export default RouteLoadingFallback
