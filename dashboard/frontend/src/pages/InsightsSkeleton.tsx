import styles from './InsightsPage.module.css'

/** The summary's shape while its first answer loads, so the page does not jump when it arrives. */
export default function InsightsSkeleton() {
  return (
    <div className={styles.skeleton} role="status" aria-live="polite" aria-label="Loading insights">
      <div className={styles.skeletonGrid}>
        {Array.from({ length: 4 }, (_, index) => (
          <div key={index} className={styles.skeletonCard}>
            <span className={`${styles.skeletonLine} ${styles.skeletonLabel}`} />
            <span className={`${styles.skeletonLine} ${styles.skeletonValue}`} />
          </div>
        ))}
      </div>
      <div className={styles.skeletonRows}>
        {Array.from({ length: 6 }, (_, index) => (
          <span key={index} className={styles.skeletonLine} />
        ))}
      </div>
    </div>
  )
}
