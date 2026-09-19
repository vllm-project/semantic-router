import { useState } from 'react'
import { benchApi } from './api'
import type { Run } from './types'
import styles from './SrBench.module.css'

export default function ReplayComposer({
  runs,
  canRun,
  onCreated,
}: {
  runs: Run[]
  canRun: boolean
  onCreated: (run: Run) => void
}) {
  const [baseline, setBaseline] = useState('')
  const [preview, setPreview] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const completed = runs.filter((run) => run.status === 'completed')
  async function replay() {
    setPending(true)
    setError('')
    try {
      onCreated(await benchApi.replay(baseline, preview))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Replay failed.')
    } finally {
      setPending(false)
    }
  }
  return (
    <section className={styles.panel}>
      <h2>Replay a routing change</h2>
      <p>
        Combine saved single-model answers with a completed route preview on the same frozen cases.
        This creates diagnostic estimates with no model generation. Live evaluation is required to
        validate the candidate.
      </p>
      <div className={styles.formGrid}>
        <label>
          Saved single-model baseline
          <select
            value={baseline}
            disabled={pending}
            onChange={(event) => setBaseline(event.target.value)}
          >
            <option value="">Select saved answers</option>
            {completed
              .filter(
                (run) =>
                  run.manifest.mode === 'live' &&
                  run.manifest.targets.some((target) => target.kind === 'single'),
              )
              .map((run) => (
                <option key={run.id} value={run.id}>
                  {run.manifest.name} · {run.id}
                </option>
              ))}
          </select>
        </label>
        <label>
          Routing preview
          <select
            value={preview}
            disabled={pending}
            onChange={(event) => setPreview(event.target.value)}
          >
            <option value="">Select preview</option>
            {completed
              .filter((run) => run.manifest.mode === 'preview')
              .map((run) => (
                <option key={run.id} value={run.id}>
                  {run.manifest.name} · {run.id}
                </option>
              ))}
          </select>
        </label>
      </div>
      <p className={styles.muted}>
        Replay supports direct single-model routes. Plugin, agent and compound execution changes
        require a live run.
      </p>
      {error && (
        <p role="alert" className={styles.error}>
          {error}
        </p>
      )}
      <button disabled={!canRun || pending || !baseline || !preview} onClick={() => void replay()}>
        {pending ? 'Replaying saved answers…' : 'Create diagnostic replay'}
      </button>
    </section>
  )
}
