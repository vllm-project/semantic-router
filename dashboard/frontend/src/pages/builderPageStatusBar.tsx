import type { EditorMode } from '@/types/dsl'

import styles from './BuilderPage.module.css'

interface BuilderStatusBarProps {
  isValid: boolean
  errorCount: number
  recipeName: string
  revision: number
  immutable: boolean
  signalCount: number
  routeCount: number
  pluginCount: number
  lineCount: number
  mode: EditorMode
}

export function BuilderStatusBar({
  isValid,
  errorCount,
  recipeName,
  revision,
  immutable,
  signalCount,
  routeCount,
  pluginCount,
  lineCount,
  mode,
}: BuilderStatusBarProps) {
  return (
    <div className={styles.statusBar} role="status">
      <div
        className={`${styles.statusItem} ${isValid ? styles.statusValid : styles.statusInvalid}`}
      >
        {isValid ? 'Valid' : `${errorCount} error${errorCount === 1 ? '' : 's'}`}
      </div>
      <div className={styles.statusItem}>{recipeName || 'No Recipe'}</div>
      {revision > 0 ? <div className={styles.statusItem}>Revision {revision}</div> : null}
      {immutable ? <div className={styles.statusItem}>Built-in</div> : null}
      <div className={styles.statusItem}>Signals {signalCount}</div>
      <div className={styles.statusItem}>Decisions {routeCount}</div>
      <div className={styles.statusItem}>Plugins {pluginCount}</div>
      {mode === 'dsl' ? <div className={styles.statusItem}>{lineCount} lines</div> : null}
      <div className={styles.statusItem}>{mode === 'dsl' ? 'DSL' : 'Visual'}</div>
    </div>
  )
}
