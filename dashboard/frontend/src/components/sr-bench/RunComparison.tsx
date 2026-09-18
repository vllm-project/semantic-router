import { useState } from 'react'
import { benchApi } from './api'
import { money, number, percent } from './model'
import type { Comparison, Run } from './types'
import styles from './SrBench.module.css'

export default function RunComparison({ runs }: { runs: Run[] }) {
  const [baseline, setBaseline] = useState('')
  const [candidate, setCandidate] = useState('')
  const [result, setResult] = useState<Comparison | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const complete = runs.filter((run) => run.status === 'completed' && run.manifest.mode === 'live')
  async function compare() {
    setPending(true)
    setError('')
    setResult(null)
    try {
      setResult(await benchApi.compare(baseline, candidate))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not compare these runs.')
    } finally {
      setPending(false)
    }
  }
  return (
    <section className={styles.panel}>
      <h2>Compare iterations</h2>
      <p>
        Compare completed live runs on the same frozen cases against the strongest observed single
        model. Paired uncertainty and matching denominators determine whether a quality or cost
        change is supported.
      </p>
      <div className={styles.formGrid}>
        <label>
          Baseline run
          <select
            value={baseline} disabled={pending}
            onChange={(event) => {
              setBaseline(event.target.value)
              setResult(null)
            }}
          >
            <option value="">Select baseline</option>
            {complete
              .filter((run) => run.manifest.targets.some((target) => target.kind === 'single'))
              .map((run) => (
                <option key={run.id} value={run.id}>
                  {run.manifest.name} · {run.id}
                </option>
              ))}
          </select>
        </label>
        <label>
          Candidate run
          <select
            value={candidate} disabled={pending}
            onChange={(event) => {
              setCandidate(event.target.value)
              setResult(null)
            }}
          >
            <option value="">Select candidate</option>
            {complete.map((run) => (
              <option key={run.id} value={run.id}>
                {run.manifest.name} · {run.id}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className={styles.actions}>
        <button
          className={styles.primary}
          disabled={pending || !baseline || !candidate}
          onClick={() => void compare()}
        >
          {pending ? 'Comparing…' : 'Compare runs'}
        </button>
      </div>
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {result && (
        <div>
          <h3>Paired comparison</h3>
          <p className={styles.notice}>{result.baseline_selection}</p>
          {result.comparisons.map((row) => (
            <section className={styles.comparisonRow} key={row.candidate_target_id}>
              <h4>
                {row.candidate_target_id} vs {row.baseline_target_id}
              </h4>
              <div className={styles.metricGrid}>
                <div>
                  <span>Macro accuracy difference</span>
                  <strong>{percent(row.quality_delta)}</strong>
                </div>
                <div>
                  <span>Cost saving</span>
                  <strong>
                    {row.cost_saving_percent === null
                      ? '—'
                      : `${number(row.cost_saving_percent, 2)}%`}
                  </strong>
                </div>
                <div>
                  <span>Paired cases</span>
                  <strong>{number(row.paired_cases)}</strong>
                </div>
                <div>
                  <span>Wins / losses / ties</span>
                  <strong>
                    {number(row.wins)} / {number(row.losses)} / {number(row.ties)}
                  </strong>
                </div>
              </div>
              <p>
                95% interval: {row.quality_delta_ci95.map((value) => percent(value)).join(' to ')}
              </p>
              <p className={styles.muted}>
                Model cost: {money(row.candidate_cost_usd)} candidate ·{' '}
                {money(row.baseline_cost_usd)} baseline. Unknown costs cannot support a saving
                claim.
              </p>
            </section>
          ))}
          <details className={styles.details}>
            <summary>Full comparison, intervals and exclusions</summary>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </details>
        </div>
      )}
    </section>
  )
}
