import styles from './DslEditorPage.module.css'

interface DslEditorStatusBarProps {
  isValid: boolean
  errorCount: number
  signalCount: number
  routeCount: number
  lineCount: number
}

export function DslEditorStatusBar({
  isValid,
  errorCount,
  signalCount,
  routeCount,
  lineCount,
}: DslEditorStatusBarProps) {
  return (
    <div className={styles.statusBar}>
      <div
        className={`${styles.statusItem} ${isValid ? styles.statusValid : styles.statusInvalid}`}
      >
        <svg
          className={styles.statusCheckmark}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          aria-hidden="true"
        >
          <path
            d={isValid ? 'M3 8.5l3 3 7-7' : 'M4 4l8 8M12 4l-8 8'}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        {isValid ? 'Config valid' : `${errorCount} error${errorCount === 1 ? '' : 's'}`}
      </div>
      <div className={styles.statusItem}>Signals: {signalCount}</div>
      <div className={styles.statusItem}>Routes: {routeCount}</div>
      <div className={styles.statusItem}>Lines: {lineCount}</div>
    </div>
  )
}
