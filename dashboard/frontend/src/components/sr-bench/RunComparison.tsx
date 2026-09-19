import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { benchApi } from './api'
import { targetLabel, targetName } from './targetPresentation'
import { changeDirection, changeSummary, formatSignedChange } from './comparisonMetrics'
import AccountingCorrection from './AccountingCorrection'
import { money, number, percent, seconds, tokenTotal } from './model'
import type { Comparison, Report, Run } from './types'
import styles from './SrBench.module.css'
import ProductLoadingState from '../ProductLoadingState'
import ProductIcon from '../ProductIcon'
import ComparisonSetup from './ComparisonSetup'
import BenchPagination from './BenchPagination'
import { IterationChart, QualityCostChart, type QualityCostPoint } from './EvaluationCharts'

const changeClass = (value: number | null | undefined) =>
  ({
    positive: styles.changePositive,
    negative: styles.changeNegative,
    neutral: styles.changeNeutral,
    unknown: styles.changeUnknown,
  })[changeDirection(value)]
function ChangeValue({
  value,
  unit,
  metric,
}: {
  value: number | null | undefined
  unit: string
  metric?: 'quality' | 'cost'
}) {
  return (
    <span className={changeClass(value)} data-direction={changeDirection(value)}>
      {formatSignedChange(value)}
      {changeDirection(value) !== 'unknown' ? unit : ''}
      {metric && (
        <small className={styles.changeCaption}>
          {value === 0 ? 'No change' : changeSummary(value, metric)}
        </small>
      )}
    </span>
  )
}
function ChangeInterval({ values }: { values: [number, number] }) {
  return (
    <>
      {values.map((value, index) => (
        <span key={index}>
          {index ? ' to ' : ''}
          <ChangeValue value={value * 100} unit=" pp" />
        </span>
      ))}
    </>
  )
}

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
  const savedIterations = search.getAll('candidate').join('|')
  const [baseline, setBaseline] = useState(savedBaseline)
  const [candidates, setCandidates] = useState(savedIterations.split('|').filter(Boolean))
  const [savedResults, setResults] = useState<IterationEvidence[]>([])
  const [savedBaselineReport, setBaselineReport] = useState<Report | null>(null)
  const [evidenceKey, setEvidenceKey] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)
  const [evidencePage, setEvidencePage] = useState(0)
  const draftChanged =
    baseline !== savedBaseline ||
    [...candidates].sort().join('|') !== savedIterations.split('|').filter(Boolean).sort().join('|')
  const selectionKey = `${savedBaseline}|${savedIterations}`
  const evidenceMatches = !draftChanged && evidenceKey === selectionKey
  const results = evidenceMatches ? savedResults : []
  const baselineReport = evidenceMatches ? savedBaselineReport : null

  useEffect(() => {
    setBaseline(savedBaseline)
    setCandidates(savedIterations.split('|').filter(Boolean))
  }, [savedBaseline, savedIterations])

  useEffect(() => {
    let cancelled = false
    const selected = savedIterations
      .split('|')
      .flatMap((id, index) => (id ? [{ id, stage: `Run ${index + 1}` }] : []))
    if (!savedBaseline || !selected.length) {
      setPending(false)
      setResults([])
      setBaselineReport(null)
      return
    }
    async function load() {
      setEvidenceKey(`${savedBaseline}|${savedIterations}`)
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
  }, [savedBaseline, savedIterations, revision])

  function compare() {
    setEvidencePage(0)
    const next = new URLSearchParams({ view: 'compare', baseline })
    const ordered = [...candidates].sort((a, b) => {
      const created = (id: string) => runs.find((run) => run.id === id)?.created_at ?? ''
      return created(a).localeCompare(created(b)) || a.localeCompare(b)
    })
    ordered.forEach((id) => {
      if (id) next.append('candidate', id)
    })
    if (next.toString() === search.toString()) setRevision((value) => value + 1)
    else setSearch(next)
  }
  function download(format: 'json' | 'csv') {
    const payload = {
      baseline_run_id: savedBaseline,
      baseline_report: baselineReport,
      comparisons: results,
      exported_at: new Date().toISOString(),
      qualified,
      cost_note:
        'Costs apply frozen per-token prices to recorded usage, not invoice or hardware measurements.',
    }
    const csv = [
      [
        'run',
        'target',
        'baseline',
        'paired_cases',
        'quality_delta',
        'ci95_low',
        'ci95_high',
        'model_cost_usd',
        'cost_saving_percent',
      ],
      ...results.flatMap(
        (item) =>
          item.comparison?.comparisons.map((row) => [
            item.id,
            row.candidate_target_id,
            row.baseline_target_id,
            row.paired_cases,
            row.quality_delta,
            ...row.quality_delta_ci95,
            row.candidate_cost_usd,
            row.cost_saving_percent,
          ]) ?? [],
      ),
    ]
      .map((row) => row.map((value) => `"${String(value ?? '').replace(/"/g, '""')}"`).join(','))
      .join('\n')
    const url = URL.createObjectURL(
      new Blob([format === 'json' ? JSON.stringify(payload, null, 2) : csv], {
        type: format === 'json' ? 'application/json' : 'text/csv',
      }),
    )
    const link = document.createElement('a')
    link.href = url
    link.download = `sr-bench-comparison.${format}`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const reviewedBaseline = runs.find((run) => run.id === savedBaseline)
  const baselineComparison = results[0]?.comparison
  const baselineTargets = baselineComparison?.baseline_targets ?? []

  const qualified =
    !!baselineReport &&
    results.length > 0 &&
    results.every(
      (item) => !item.error && (item.comparison?.comparisons.length ?? 0) > 0 && !!item.report,
    )
  const baselineFailures = baselineTargets.reduce((sum, target) => sum + (target.failed ?? 0), 0)
  const candidateFailures = results.reduce(
    (sum, item) =>
      sum +
      (item.comparison?.candidate_targets.reduce(
        (total, target) => total + (target.failed ?? 0),
        0,
      ) ?? 0),
    0,
  )
  const points: QualityCostPoint[] = qualified
    ? [
        ...baselineTargets
          .filter((target) =>
            reviewedBaseline?.manifest.targets.some(
              (item) => item.id === target.id && item.kind === 'single',
            ),
          )
          .flatMap((target) =>
            typeof target.macro_accuracy === 'number' && typeof target.cost_usd === 'number'
              ? [
                  {
                    name: targetName(reviewedBaseline?.manifest, target.id),
                    quality: target.macro_accuracy * 100,
                    cost: target.cost_usd,
                    kind: 'single' as const,
                  },
                ]
              : [],
          ),
        ...results.flatMap(
          (item) =>
            item.comparison?.comparisons.flatMap((row) => {
              const metric = item.comparison?.candidate_targets.find(
                (target) => target.id === row.candidate_target_id,
              )
              return typeof metric?.macro_accuracy === 'number' &&
                typeof row.candidate_cost_usd === 'number'
                ? [
                    {
                      name: `${item.stage} · ${targetName(runs.find((run) => run.id === item.id)?.manifest, row.candidate_target_id)}`,
                      quality: metric.macro_accuracy * 100,
                      cost: row.candidate_cost_usd,
                      kind:
                        runs
                          .find((run) => run.id === item.id)
                          ?.manifest.targets.find((target) => target.id === row.candidate_target_id)
                          ?.kind ?? ('mom' as const),
                    },
                  ]
                : []
            }) ?? [],
        ),
      ]
    : []
  const trajectory =
    qualified && results.every((item) => item.comparison?.comparisons.length === 1)
      ? results.map((item) => {
          const row = item.comparison!.comparisons[0]
          const metric = item.comparison?.candidate_targets.find(
            (target) => target.id === row.candidate_target_id,
          )
          return {
            stage: item.stage,
            quality:
              typeof metric?.macro_accuracy === 'number' ? metric.macro_accuracy * 100 : null,
            saving: row.cost_saving_percent,
          }
        })
      : []
  const currentEvidencePage = Math.min(evidencePage, Math.max(0, Math.ceil(results.length / 6) - 1))
  const baselineMetric = baselineTargets.find(
    (target) => target.id === results[0]?.comparison?.comparisons[0]?.baseline_target_id,
  )
  return (
    <section className={styles.panel}>
      <h2>Compare iterations</h2>
      <p>Choose a single-model baseline and compare compatible saved results.</p>
      {!!results.length && (
        <p className={styles.muted}>
          Costs apply frozen per-token prices to recorded usage; they are not invoice or
          hardware-cost measurements.
        </p>
      )}
      <ComparisonSetup
        baseline={baseline}
        candidates={candidates}
        pending={pending}
        onBaseline={setBaseline}
        onCandidates={setCandidates}
        onCompare={compare}
      />
      {draftChanged && savedResults.length > 0 && (
        <p className={styles.notice}>
          Selection changed. Choose Compare runs to update the displayed evidence.
        </p>
      )}
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {pending && <ProductLoadingState compact label="Comparing saved evaluation evidence…" />}
      {qualified && (baselineFailures > 0 || candidateFailures > 0) && (
        <p className={styles.notice}>
          {number(baselineFailures)} failed baseline results · {number(candidateFailures)} failed
          candidate results. Scores use all planned cases; explicit failed outcomes count as
          incorrect. Run statuses remain unchanged, and unknown costs cannot support a saving claim.
        </p>
      )}
      {qualified && (
        <div className={styles.chartGrid}>
          <QualityCostChart points={points} />
          {trajectory.length > 1 && (
            <IterationChart
              points={trajectory}
              baselineQuality={
                typeof baselineMetric?.macro_accuracy === 'number'
                  ? baselineMetric.macro_accuracy * 100
                  : null
              }
            />
          )}
        </div>
      )}
      {results.length > 0 && !qualified && !pending && (
        <p className={styles.notice}>
          A continuous trend is withheld until every selected iteration has a complete, compatible
          comparison.
        </p>
      )}
      {results.length > 0 && (
        <div className={styles.iterationCards}>
          {results.slice(currentEvidencePage * 6, currentEvidencePage * 6 + 6).map((item) => (
            <article className={styles.iterationCard} key={item.id}>
              <p className={styles.eyebrow}>{item.stage}</p>
              <h3>{runs.find((run) => run.id === item.id)?.manifest.name ?? item.id}</h3>
              {item.comparison && !item.comparison.candidate_quality_complete && (
                <p className={styles.muted}>
                  {item.comparison.candidate_status} ·{' '}
                  {number(
                    item.comparison.candidate_targets.reduce(
                      (sum, target) => sum + (target.failed ?? 0),
                      0,
                    ),
                  )}{' '}
                  failed results
                </p>
              )}
              {item.error ? (
                <p className={styles.error}>Comparison withheld: {item.error}</p>
              ) : (
                item.comparison?.comparisons.map((row) => (
                  <div key={row.candidate_target_id}>
                    <div className={styles.iterationNumbers}>
                      <div>
                        <span>Quality Δ</span>
                        <strong>
                          <ChangeValue
                            value={row.quality_delta * 100}
                            unit=" pp"
                            metric="quality"
                          />
                        </strong>
                      </div>
                      <div>
                        <span>Cost saving</span>
                        <strong>
                          <ChangeValue value={row.cost_saving_percent} unit="%" metric="cost" />
                        </strong>
                      </div>
                    </div>
                    <p className={styles.muted}>
                      95% paired interval <ChangeInterval values={row.quality_delta_ci95} /> ·{' '}
                      {number(row.paired_cases)} cases
                    </p>
                    <p className={styles.muted}>
                      {row.quality_delta_ci95[0] <= 0 && row.quality_delta_ci95[1] >= 0
                        ? 'Interval includes zero; improvement is not established.'
                        : 'See the comparison assumptions before making a quality claim.'}
                    </p>
                  </div>
                ))
              )}
              {item.report?.provenance.accounting_correction && (
                <p className={styles.notice}>
                  {item.report.provenance.accounting_correction.qualified
                    ? 'Accounting verified'
                    : 'Partial accounting'}{' '}
                  · {number(item.report.provenance.accounting_correction.corrected_call_count)}{' '}
                  corrected calls. Original receipts preserved; full correction evidence below.
                </p>
              )}
              <a href={`?view=runs&run=${encodeURIComponent(item.id)}`}>
                Open run <ProductIcon name="arrow-right" />
              </a>
            </article>
          ))}
        </div>
      )}
      <BenchPagination
        label="Comparison results"
        total={results.length}
        page={currentEvidencePage}
        pageSize={6}
        onChange={setEvidencePage}
      />
      {baselineReport && (
        <>
          <AccountingCorrection report={baselineReport} />
          <details className={styles.details}>
            <summary>Single-model baseline metrics</summary>
            <p>
              {reviewedBaseline?.manifest.name ?? savedBaseline} ·{' '}
              {baselineComparison?.baseline_status}
            </p>
            <div className={styles.tableScroll}>
              <table>
                <thead>
                  <tr>
                    <th>Single model</th>
                    <th>Macro accuracy</th>
                    <th>Correct / denominator</th>
                    <th>Saved outcomes</th>
                    <th>Observed model cost</th>
                    <th>Cache-neutral estimate</th>
                    <th>Tokens</th>
                    <th>Latency p50 / p95</th>
                  </tr>
                </thead>
                <tbody>
                  {baselineTargets
                    .filter((target) =>
                      reviewedBaseline?.manifest.targets.some(
                        (item) => item.id === target.id && item.kind === 'single',
                      ),
                    )
                    .map((target) => (
                      <tr key={target.id}>
                        <th scope="row">{targetName(reviewedBaseline?.manifest, target.id)}</th>
                        <td>{percent(target.macro_accuracy)}</td>
                        <td>
                          {number(target.correct)} / {number(target.total)}
                        </td>
                        <td>
                          {number(target.completed)} completed · {number(target.failed)} failed
                        </td>
                        <td>{money(target.cost_usd)}</td>
                        <td title={target.cache_neutral_cost_basis}>
                          {money(target.cache_neutral_cost_usd)}
                        </td>
                        <td>{number(tokenTotal(target.tokens))}</td>
                        <td>
                          {seconds(target.latency_p50_s)} / {seconds(target.latency_p95_s)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
      {!!results.length && (
        <>
          <div className={styles.sectionHeading}>
            <h3>Comparison evidence</h3>
            <div className={styles.actions}>
              <button onClick={() => download('csv')}>Export comparison CSV</button>
              <button onClick={() => download('json')}>Export comparison JSON</button>
            </div>
          </div>
          <p className={styles.mobileHint}>Swipe the tables horizontally to inspect all metrics.</p>
          <p className={styles.muted}>
            Quality differences are percentage points. Intervals that include zero do not establish
            an improvement. Unknown cost cannot support a saving claim. Development-set gains
            require a separate holdout check.
          </p>
          {!!baselineComparison?.baseline_selection_qualification &&
            (baselineFailures > 0 || candidateFailures > 0) && (
              <p className={styles.muted}>{baselineComparison.baseline_selection_qualification}</p>
            )}
          <p className={styles.muted}>
            Cache-neutral estimates use the same selected single-model baseline and reprice every
            prompt token at the frozen fresh-input rate plus output. This counterfactual excludes
            cache discounts and premiums; it is neither billed spend nor a measured cache-free run.
          </p>
          <details className={styles.details}>
            <summary>Detailed iteration metrics and uncertainty</summary>
            <div className={styles.tableScroll}>
              <table>
                <thead>
                  <tr>
                    <th>Iteration / target</th>
                    <th>Macro accuracy</th>
                    <th>Correct / denominator</th>
                    <th>Quality Δ vs best single</th>
                    <th>95% paired interval</th>
                    <th>Observed cost / saving</th>
                    <th>Cache-neutral estimate / saving</th>
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
                                {item.report?.provenance.accounting_correction && (
                                  <p className={styles.notice}>
                                    {item.report.provenance.accounting_correction.qualified
                                      ? 'Accounting verified'
                                      : 'Partial accounting'}{' '}
                                    ·{' '}
                                    {number(
                                      item.report.provenance.accounting_correction
                                        .corrected_call_count,
                                    )}{' '}
                                    corrected calls. Original receipts preserved; full correction
                                    evidence below.
                                  </p>
                                )}
                                <a href={`?view=runs&run=${encodeURIComponent(item.id)}`}>
                                  {candidateRun?.manifest.name ?? item.id}
                                </a>{' '}
                                ·{' '}
                                {targetName(
                                  runs.find((run) => run.id === item.id)?.manifest,
                                  row.candidate_target_id,
                                )}
                              </small>
                            </th>
                            <td>{percent(metrics?.macro_accuracy)}</td>
                            <td>
                              {number(metrics?.correct)} / {number(metrics?.total)}
                            </td>
                            <td>
                              <ChangeValue
                                value={row.quality_delta * 100}
                                unit=" pp"
                                metric="quality"
                              />
                              <small>
                                vs {targetName(reviewedBaseline?.manifest, row.baseline_target_id)}
                              </small>
                            </td>
                            <td>
                              <ChangeInterval values={row.quality_delta_ci95} />
                              {row.quality_delta_ci95_method && (
                                <small>
                                  {row.quality_delta_ci95_method === 'weighted-paired-hoeffding'
                                    ? 'Conservative weighted Hoeffding'
                                    : row.quality_delta_ci95_method}
                                </small>
                              )}
                              <small>{number(row.paired_cases)} paired cases</small>
                              {(row.quality_delta_ci95_qualification ||
                                row.quality_delta_bootstrap_ci95) && (
                                <details>
                                  <summary>Uncertainty details</summary>
                                  {row.quality_delta_ci95_qualification && (
                                    <p>{row.quality_delta_ci95_qualification}</p>
                                  )}
                                  {row.quality_delta_bootstrap_ci95 && (
                                    <p>
                                      Bootstrap diagnostic (95%):{' '}
                                      <ChangeInterval values={row.quality_delta_bootstrap_ci95} />.
                                      Use the conservative interval above for quality claims;
                                      resampling identical paired outcomes can produce a zero-width
                                      diagnostic interval.
                                    </p>
                                  )}
                                </details>
                              )}
                            </td>
                            <td>
                              {money(row.candidate_cost_usd)}
                              <small>
                                <ChangeValue
                                  value={row.cost_saving_percent}
                                  unit="% saving"
                                  metric="cost"
                                />
                              </small>
                            </td>
                            <td title={row.cache_neutral_cost_basis}>
                              {money(row.cache_neutral_candidate_cost_usd)}
                              <small>
                                <ChangeValue
                                  value={row.cache_neutral_cost_saving_percent}
                                  unit="% estimated saving"
                                  metric="cost"
                                />
                              </small>
                              <small>Baseline {money(row.cache_neutral_baseline_cost_usd)}</small>
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
          </details>
          <details className={styles.details}>
            <summary>Technical comparison evidence</summary>
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
                    <AccountingCorrection report={item.report ?? null} />
                    <details className={styles.details}>
                      <summary>Baseline selection policy</summary>
                      <p className={styles.muted}>{item.comparison?.baseline_selection}</p>
                      {(item.comparison?.baseline_tied_best_target_ids?.length ?? 0) > 1 && (
                        <p className={styles.notice}>
                          Tied best single models:{' '}
                          {item.comparison?.baseline_tied_best_target_ids
                            ?.map((id) => targetName(reviewedBaseline?.manifest, id))
                            .join(', ')}
                          . {item.comparison?.baseline_tie_policy}
                        </p>
                      )}
                    </details>
                    {item.comparison?.baseline_cost_comparison_eligible === false && (
                      <p className={styles.notice}>
                        Cost saving withheld:{' '}
                        {item.comparison.baseline_cost_comparison_reason ??
                          'Baseline cost evidence is incomplete.'}
                      </p>
                    )}
                    <details className={styles.details}>
                      <summary>Frozen configuration identity</summary>
                      <dl className={styles.identity}>
                        {runs
                          .find((run) => run.id === item.id)
                          ?.manifest.targets.filter((target) => target.kind === 'mom')
                          .map((target) => (
                            <div key={target.id}>
                              <dt>{targetLabel(target)} frozen configuration</dt>
                              <dd>
                                <code>
                                  {target.config_hash ?? 'Configuration identity unavailable'}
                                </code>
                              </dd>
                            </div>
                          ))}
                      </dl>
                    </details>
                    <details className={styles.details}>
                      <summary>Full comparison, provenance and exclusions</summary>
                      <pre>{JSON.stringify(item.comparison, null, 2)}</pre>
                      <pre>{JSON.stringify(item.report?.provenance, null, 2)}</pre>
                    </details>
                  </>
                )}
              </section>
            ))}
          </details>
        </>
      )}
    </section>
  )
}
