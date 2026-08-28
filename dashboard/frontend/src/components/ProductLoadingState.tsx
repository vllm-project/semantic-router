import { ThinkingOrb } from 'thinking-orbs'

import styles from './ProductLoadingState.module.css'

interface ProductLoadingStateProps {
  label?: string
  compact?: boolean
}

export default function ProductLoadingState({
  label = 'Loading',
  compact = false,
}: ProductLoadingStateProps) {
  return (
    <div
      className={`${styles.loading} ${compact ? styles.compact : ''}`}
      role="status"
      aria-live="polite"
    >
      <div className={styles.surface}>
        <ThinkingOrb
          className={styles.orb}
          state="working"
          size={compact ? 20 : 64}
          theme="dark"
          aria-label={label}
        />
        <span>{label}</span>
      </div>
    </div>
  )
}
