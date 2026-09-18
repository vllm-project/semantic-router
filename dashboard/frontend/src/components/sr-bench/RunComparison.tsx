import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { benchApi } from './api'
import { money, number, percent, seconds, tokenTotal } from './model'
import type { Comparison, Report, Run } from './types'
import styles from './SrBench.module.css'

const stages = ['Current Balance', 'Optimization 1', 'Optimization 2']
interface IterationEvidence {
  id: string
  stage: string
  comparison?: Comparison
  report?: Report
  error?: string
}

export default function RunComparison({ runs }: { runs: Run[] }) {
  const [search, setSearch] = useSearchParams()
  const savedBaseline = search.get('baseline') ?? ''
  const savedIterations = [0, 1, 2].map((index) => search.get(`iteration${index}`) ?? '').join('|')
  const [baseline, setBaseline] = useState(savedBaseline)
  const [candidates, setCandidates] = useState(savedIterations.split('|'))
  const [results, setResults] = useState<IterationEvidence[]>([])
  const [baselineReport, setBaselineReport] = useState<Report | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)
  const complete = runs.filter((run) => run.status === 'completed' && run.manifest.mode === 'live')
  const completeIDs = complete
    .map((run) => run.id)
    .sort()
    .join('|')

  useEffect(() => {
    setBaseline(savedBaseline)
    setCandidates(savedIterations.split('|'))
  }, [savedBaseline, savedIterations])

  useEffect(() => {
    let cancelled = false
    const selected = savedIterations
      .split('|')
      .flatMap((id, index) => (id ? [{ id, stage: stages[index] }] : []))
    if (!savedBaseline || !selected.length || !completeIDs) {
      setPending(false)
      setResults([])
      setBaselineReport(null)
      return
    }
    async function load() {
      setPending(true)
      setError('')
      setResults([])
      setBaselineReport(null)
      const baselineResult = await benchApi.report(savedBaseline).then(
        (report) => ({ report }),
        (cause: unknown) => ({
          error: cause instanceof Error ? cause.message : 'Baseline report is unavailable.',
        }),
      )
      if (cancelled) return
      if ('report' in baselineResult) setBaselineReport(baselineResult.report)
      else setError(baselineResult.error)
      const evidence = await Promise.all(
        selected.map(async (item): Promise<IterationEvidence> => {
          try {
            const [comparison, report] = await Promise.all([
              benchApi.compare(savedBaseline, item.id),
              benchApi.report(item.id),
            ])
            return { ...item, comparison, report }
          } catch (cause) {
            return {
              ...item,
              error: cause instanceof Error ? cause.message : 'Comparison unavailable.',
            }
          }
        }),
      )
      if (!cancelled) {
        setResults(evidence)
        setPending(false)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [savedBaseline, savedIterations, completeIDs, revision])

  function compare() {
    const next = new URLSearchParams({ view: 'compare', baseline })
    candidates.forEach((id, index) => {
      if (id) next.set(`iteration${index}`, id)
    })
    if (next.toString() === search.toString()) setRevision((value) => value + 1)
    else setSearch(next)
  }
  const selectedIDs = candidates.filter(Boolean)
  const duplicate = new Set(selectedIDs).size !== selectedIDs.length
  const reviewedBaseline = runs.find((run) => run.id === savedBaseline)

  return (
    <section className={styles.panel}>
      <h2>Compare iterations</h2>
      <p>
        Follow current Balance through two optimization rounds against your strongest observed
        single model. Each comparison requires the same frozen cases, sampling, prices and limits.
      </p>
      <div className={styles.formGrid}>
        <label>
          Baseline run
          <select
            value={baseline}
            disabled={pending}
            onChange={(event) => setBaseline(event.target.value)}
          >
            <option value="">Select single-model baseline</option>
            {complete
              .filter((run) => run.manifest.targets.some((target) => target.kind === 'single'))
              .map((run) => (
                <option key={run.id} value={run.id}>
                  {run.manifest.name} · {run.id}
                </option>
              ))}
          </select>
        </label>
        {stages.map((stage, index) => (
          <label key={stage}>
            {stage} run
            <select
              value={candidates[index]}
              disabled={pending}
              onChange={(event) =>
                setCandidates((previous) =>
                  previous.map((value, i) => (i === index ? event.target.value : value)),
                )
              }
            >
              <option value="">
                {index ? 'Optional: select completed iteration' : 'Select current Balance run'}
              </option>
              {complete.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.manifest.name} · {run.id}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      {duplicate && <p className={styles.error}>Choose a different run for each iteration.</p>}
      <div className={styles.actions}>
        <button
          className={styles.primary}
          disabled={pending || !baseline || !selectedIDs.length || duplicate}
          onClick={compare}
        >
          {pending ? 'Comparing…' : 'Compare runs'}
        </button>
        <span className={styles.muted}>
          Selections are saved in this page URL. Bookmark it to reopen the same evidence.
        </span>
      </div>
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {baselineReport && (
        <>
          <h3>Single-model baseline</h3>
          <p>{reviewedBaseline?.manifest.name ?? savedBaseline}</p>
          <div className={styles.tableScroll}>
            <table>
              <thead>
                <tr>
                  <th>Single model</th>
                  <th>Macro accuracy</th>
                  <th>Correct / denominator</th>
                  <th>Model cost</th>
                  <th>Tokens</th>
                  <th>Latency p50 / p95</th>
                </tr>
              </thead>
              <tbody>
                {baselineReport.summary.targets
                  .filter((target) =>
                    reviewedBaseline?.manifest.targets.some(
                      (item) => item.id === target.id && item.kind === 'single',
                    ),
                  )
                  .map((target) => (
                    <tr key={target.id}>
                      <th scope="row">{target.id}</th>
                      <td>{percent(target.macro_accuracy)}</td>
                      <td>
                        {number(target.correct)} / {number(target.total)}
                      </td>
                      <td>{money(target.cost_usd)}</td>
                      <td>{number(tokenTotal(target.tokens))}</td>
                      <td>
                        {seconds(target.latency_p50_s)} / {seconds(target.latency_p95_s)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!!results.length && (
        <>
          <h3>Balance optimization trajectory</h3>
          <p className={styles.mobileHint}>Swipe the tables horizontally to inspect all metrics.</p>
          <p className={styles.muted}>
            Quality differences are percentage points. Intervals that include zero do not establish
            an improvement. Unknown cost cannot support a saving claim. Development-set gains
            require a separate holdout check.
          </p>
          <div className={styles.tableScroll}>
            <table>
              <thead>
                <tr>
                  <th>Iteration / target</th>
                  <th>Macro accuracy</th>
                  <th>Correct / denominator</th>
                  <th>Quality Δ vs best single</th>
                  <th>95% paired interval</th>
                  <th>Model cost / saving</th>
                  <th>Tokens</th>
                  <th>Latency p50 / p95</th>
                  <th>Wall time</th>
                </tr>
              </thead>
              <tbody>
                {results.flatMap(
                  (item) =>
                    item.comparison?.comparisons.map((row) => {
                      const metrics = item.report?.summary.targets.find(
                        (target) => target.id === row.candidate_target_id,
                      )
                      const candidateRun = runs.find((run) => run.id === item.id)
                      return (
                        <tr key={`${item.id}:${row.candidate_target_id}`}>
                          <th scope="row">
                            {item.stage}
                            <small>
                              <a href={`?view=runs&run=${encodeURIComponent(item.id)}`}>
                                {candidateRun?.manifest.name ?? item.id}
                              </a>{' '}
                              · {row.candidate_target_id}
                            </small>
                          </th>
                          <td>{percent(metrics?.macro_accuracy)}</td>
                          <td>
                            {number(metrics?.correct)} / {number(metrics?.total)}
                          </td>
                          <td>
                            {number(row.quality_delta * 100, 2)} pp
                            <small>vs {row.baseline_target_id}</small>
                          </td>
                          <td>
                            {row.quality_delta_ci95
                              .map((value) => `${number(value * 100, 2)} pp`)
                              .join(' to ')}
                            <small>{number(row.paired_cases)} paired cases</small>
                          </td>
                          <td>
                            {money(row.candidate_cost_usd)}
                            <small>
                              {row.cost_saving_percent === null
                                ? 'Saving unknown'
                                : `${number(row.cost_saving_percent, 2)}% saving`}
                            </small>
                          </td>
                          <td>{number(tokenTotal(metrics?.tokens))}</td>
                          <td>
                            {seconds(metrics?.latency_p50_s)} / {seconds(metrics?.latency_p95_s)}
                          </td>
                          <td>{seconds(item.report?.summary.wall_time_s)}</td>
                        </tr>
                      )
                    }) ?? [],
                )}
              </tbody>
            </table>
          </div>
          {results.map((item) => (
            <section className={styles.comparisonRow} key={item.id}>
              <h4>
                {item.stage} · {runs.find((run) => run.id === item.id)?.manifest.name ?? item.id}
              </h4>
              {item.error ? (
                <p className={styles.error} role="alert">
                  Comparison withheld: {item.error}
                </p>
              ) : (
                <>
                  <p className={styles.muted}>{item.comparison?.baseline_selection}</p>
                  {(item.comparison?.baseline_tied_best_target_ids?.length ?? 0) > 1 && (
                    <p className={styles.notice}>
                      Tied best single models:{' '}
                      {item.comparison?.baseline_tied_best_target_ids?.join(', ')}.{' '}
                      {item.comparison?.baseline_tie_policy}
                    </p>
                  )}
                  {item.comparison?.baseline_cost_comparison_eligible === false && (
                    <p className={styles.notice}>
                      Cost saving withheld:{' '}
                      {item.comparison.baseline_cost_comparison_reason ??
                        'Baseline cost evidence is incomplete.'}
                    </p>
                  )}
                  <dl className={styles.identity}>
                    {runs
                      .find((run) => run.id === item.id)
                      ?.manifest.targets.filter((target) => target.kind === 'mom')
                      .map((target) => (
                        <div key={target.id}>
                          <dt>{target.id} frozen configuration</dt>
                          <dd>
                            <code>
                              {target.config_hash ?? 'Configuration identity unavailable'}
                            </code>
                          </dd>
                        </div>
                      ))}
                  </dl>
                  <details className={styles.details}>
                    <summary>Full comparison, provenance and exclusions</summary>
                    <pre>{JSON.stringify(item.comparison, null, 2)}</pre>
                    <pre>{JSON.stringify(item.report?.provenance, null, 2)}</pre>
                  </details>
                </>
              )}
            </section>
          ))}
        </>
      )}
    </section>
  )
}
