import ProductLoadingState from '../ProductLoadingState'
import ProductIcon from '../ProductIcon'
import styles from './SrBench.module.css'
import controls from './BenchControls.module.css'

export default function RunOptionsStatus({
  loading,
  error,
  next,
  loaded,
  scanLimited,
  label,
  onRetry,
  onMore,
}: {
  loading: boolean
  error: string
  next: string | null
  loaded: boolean
  scanLimited: boolean
  label: string
  onRetry: () => void
  onMore: () => void
}) {
  return (
    <>
      {scanLimited && (
        <p className={styles.notice}>
          Some saved evidence exceeded the verification limit. Only verified combinations are shown.
        </p>
      )}
      {loading && <ProductLoadingState compact label={`Loading ${label}…`} />}
      {error && (
        <div role="alert" className={styles.error}>
          <p>{error}</p>
          <button className={controls.compactButton} disabled={loading} onClick={onRetry}>
            <ProductIcon name="refresh" /> Retry {label}
          </button>
        </div>
      )}
      {loaded && next && !error && (
        <button className={controls.compactButton} disabled={loading} onClick={onMore}>
          Load more {label} <ProductIcon name="chevron-down" />
        </button>
      )}
    </>
  )
}
