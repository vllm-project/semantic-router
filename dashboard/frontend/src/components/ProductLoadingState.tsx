import styles from './ProductLoadingState.module.css'

interface ProductLoadingStateProps {
  label?: string
  compact?: boolean
  fill?: boolean
  className?: string
}

/**
 * One page-level loading treatment for every Dashboard surface. Keep local
 * action progress on the action itself; use this state while a page's primary
 * content is unavailable.
 */
export default function ProductLoadingState({
  label = 'Loading',
  compact = false,
  fill = false,
  className = '',
}: ProductLoadingStateProps) {
  return (
    <div
      className={`${styles.root} ${compact ? styles.compact : ''} ${fill ? styles.fill : ''} ${className}`.trim()}
      role="status"
      aria-live="polite"
      aria-label={label}
      data-testid="product-loading-state"
    >
      <span className={styles.mark} aria-hidden="true">
        <img src="/vllm-sr-logo.white.png" alt="" />
        <i />
      </span>
      <span className={styles.srOnly}>{label}</span>
    </div>
  )
}
