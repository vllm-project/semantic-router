import { useState } from 'react'
import type { InstanceMode, useInstanceDeployment } from './useInstanceDeployment'
import styles from './DecisionModelPage.module.css'

export default function InstanceModePanel({
  instance,
  writable,
}: {
  instance: ReturnType<typeof useInstanceDeployment>
  writable: boolean
}) {
  const [selection, setSelection] = useState<InstanceMode | null>(null)
  const active = instance.status?.observed_mode
  const selected = selection ?? (active === 'engine' ? 'engine' : 'router')
  const disabled = !writable || !instance.status?.can_switch || instance.busy
  return (
    <section className={styles.panel} aria-labelledby="instance-mode-title">
      <span className={styles.eyebrow}>Serving capabilities</span>
      <h2 id="instance-mode-title">Instance mode</h2>
      <p className={styles.muted}>
        Both modes serve System One through the same frontend. Router mode also enables your signal,
        decision and backend routing configuration.
      </p>
      <div className={styles.modeOptions} role="group" aria-label="Instance mode">
        {(['engine', 'router'] as const).map((mode) => (
          <button
            key={mode}
            type="button"
            aria-pressed={selected === mode}
            disabled={disabled}
            onClick={() => setSelection(mode)}
          >
            <span className={styles.modeOptionTitle}>
              <strong>{mode === 'engine' ? 'Engine' : 'Router'}</strong>
              {active === mode && <small>Active</small>}
            </span>
            <span>
              {mode === 'engine'
                ? 'Run decision models and native questions without Chat backends.'
                : 'Use decision models in routing, and serve native questions alongside Chat.'}
            </span>
          </button>
        ))}
      </div>
      <div className={styles.deployBar}>
        <p className={styles.muted}>
          Switching keeps the frontend, unchanged model workers and saved routing configuration.
        </p>
        <button
          type="button"
          className={styles.primary}
          disabled={disabled || selected === active || active === 'unknown' || !active}
          onClick={() =>
            void instance
              .deploy(selected)
              .then(() => setSelection(null))
              .catch(() => {})
          }
        >
          {instance.busy ? 'Applying…' : 'Apply mode'}
        </button>
      </div>
      {(instance.error || instance.status?.unavailable_reason) && (
        <p className={styles.notice} role="status">
          {instance.error || instance.status?.unavailable_reason}
        </p>
      )}
    </section>
  )
}
