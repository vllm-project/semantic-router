import { useState } from 'react'
import { benchApi, SR_BENCH_API } from './api'
import { active, reportDistribution, money, number, percent, seconds, tokenTotal } from './model'
import type { CaseResult, Run, TargetMetrics } from './types'
import { useRunEvidence } from './useRunEvidence'
import RunArtifacts from './RunArtifacts'
import CallEvidence from './CallEvidence'
import AccountingCorrection from './AccountingCorrection'
import PreviewEvidence from './PreviewEvidence'
import RunRecovery from './RunRecovery'
import RunLineage from './RunLineage'
import RecipeEvidence from './RecipeEvidence'
import { RunStatus } from './RunList'
import styles from './SrBench.module.css'

function MetricsTable({
  targets,
  preview,
  summary = false,
}: {
  targets: TargetMetrics[]
  preview: boolean
  summary?: boolean
}) {
  return (
    <div className={styles.tableScroll}>
      <table>
        <thead>
          <tr>
            <th>Target</th>
            <th>{summary ? 'Macro accuracy' : 'Accuracy'}</th>
            {summary && <th>sr-bench score</th>}
            <th>Correct / denominator</th>
            <th>Failures</th>
            <th>Observed model cost</th>
            <th>Cache-neutral estimate</th>
            <th>Tokens</th>
            <th>Latency p50 / p95</th>
            <th>TTFT p50</th>
          </tr>
        </thead>
        <tbody>
          {targets.map((target) => (
            <tr key={target.id}>
              <th scope="row">{target.id}</th>
              <td>
                {preview
                  ? 'Preview only'
                  : percent(summary ? target.macro_accuracy : target.accuracy)}
                {!preview && !summary && Array.isArray(target.accuracy_ci95) && (
                  <small>
                    95% CI {target.accuracy_ci95.map((value) => percent(value)).join(' – ')}
                  </small>
                )}
              </td>
              {summary && <td>{preview ? '—' : percent(target.sr_bench_score)}</td>}
              <td>{preview ? '—' : `${number(target.correct)} / ${number(target.total)}`}</td>
              <td>{number(target.failed)}</td>
              <td>{money(target.cost_usd)}</td>
              <td title={target.cache_neutral_cost_basis}>
                {money(target.cache_neutral_cost_usd)}
              </td>
              <td>{number(tokenTotal(target.tokens))}</td>
              <td>
                {seconds(target.latency_p50_s)} / {seconds(target.latency_p95_s)}
              </td>
              <td>{seconds(target.ttft_p50_s)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Distribution({ title, entries }: { title: string; entries: Array<[string, number]> }) {
  const maximum = Math.max(1, ...entries.map(([, count]) => count))
  return (
    <section>
      <h3>{title}</h3>
      {entries.length ? (
        <ul className={styles.distribution}>
          {entries.map(([label, count]) => (
            <li key={label}>
              <span>{label}</span>
              <div className={styles.barTrack}>
                <div style={{ width: `${(count / maximum) * 100}%` }} />
              </div>
              <strong>{number(count)}</strong>
            </li>
          ))}
        </ul>
      ) : (
        <p className={styles.muted}>No routing trace recorded for these results.</p>
      )}
    </section>
  )
}

export default function RunDetails({
  id,
  canRun,
  onChanged,
  onRecovered,
}: {
  id: string
  canRun: boolean
  onChanged: () => void
  onRecovered: (run: Run) => void
}) {
  const [pending, setPending] = useState(false)
  const [filter, setFilter] = useState('')
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState<CaseResult | null>(null)
  const [revision, setRevision] = useState(0)
  const [actionError, setActionError] = useState('')
  const [eventLimit, setEventLimit] = useState(100)
  const {
    run,
    readAt,
    report,
    results,
    events,
    calls,
    error: readError,
    resultsPage,
    callsPage,
    loadMoreResults,
    loadMoreCalls,
  } = useRunEvidence(id, revision)
  const error = actionError || readError

  function refreshEvidence() {
    setPage(0)
    setSelected(null)
    setRevision((value) => value + 1)
  }

  async function cancel() {
    setPending(true)
    setActionError('')
    try {
      await benchApi.cancel(id)
      refreshEvidence()
      onChanged()
    } catch (cause) {
      setActionError(
        cause instanceof Error
          ? cause.message
          : 'Cancellation failed. Refresh this run before taking another action.',
      )
    } finally {
      setPending(false)
    }
  }

  const visible = results.filter((result) =>
    `${result.case_id} ${result.target_id} ${result.benchmark} ${result.status}`
      .toLowerCase()
      .includes(filter.toLowerCase()),
  )
  const progress = run?.progress
  const percentage = progress?.total
    ? Math.min(100, ((progress.completed + progress.failed) / progress.total) * 100)
    : 0
  const metrics = report?.summary.targets ?? run?.summary?.targets ?? []
  const benchmarkIDs = [...new Set(report?.benchmarks.map((row) => String(row.benchmark)) ?? [])]
  return (
    <section className={styles.panel} aria-labelledby="run-detail-title">
      <div className={styles.sectionHeading}>
        <div>
          <h2 id="run-detail-title">{run?.manifest.name ?? 'Evaluation run'}</h2>
          <code>{id}</code>
        </div>
        <div className={styles.actions}>
          <button onClick={refreshEvidence}>Refresh evidence</button>
          {run && active(run.status) && (
            <button
              className={styles.danger}
              disabled={!canRun || pending}
              onClick={() => void cancel()}
            >
              {pending ? 'Stopping…' : 'Cancel evaluation'}
            </button>
          )}
        </div>
      </div>
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {!run ? (
        <p role="status">Loading run…</p>
      ) : (
        <>
          <div className={styles.metricGrid}>
            <div>
              <span>Status</span>
              <strong>
                <RunStatus status={run.status} />
              </strong>
            </div>
            <div>
              <span>Completed</span>
              <strong>
                {number(progress?.completed)} / {number(progress?.total)}
              </strong>
            </div>
            <div>
              <span>Failed</span>
              <strong>{number(progress?.failed)}</strong>
            </div>
            <div>
              <span>Elapsed wall time</span>
              <strong>{seconds(report?.summary.wall_time_s ?? run.summary?.wall_time_s)}</strong>
            </div>
          </div>
          <progress
            className={styles.progress}
            max={100}
            value={percentage}
            aria-label="Evaluation progress"
          />
          <p className={styles.muted}>
            Last synchronized {readAt ?? '—'} · Last persisted update{' '}
            <time dateTime={run.updated_at}>{new Date(run.updated_at).toLocaleString()}</time>.
            Closing this page does not stop the worker.
          </p>
          <nav className={styles.detailNav} aria-label="Run sections">
            <a href="#run-targets">Results</a>
            {run.manifest.targets.some((target) => target.kind === 'mom') && (
              <a href="#run-recipe">Recipe</a>
            )}
            <a href="#run-cases">Cases and calls</a>
            <a href="#run-events">Events</a>
            <a href="#run-provenance">Provenance</a>
          </nav>
          {run.status !== 'completed' && (
            <p className={styles.notice}>
              This run is {run.status}. Partial results are not a completed evaluation.
            </p>
          )}
          {(run.error || report?.failure) && (
            <div className={styles.error} role="alert">
              <p>{run.error || report?.failure?.reason}</p>
              {!run.error && report?.failure && (
                <p>
                  Target {report.failure.target_id} · case {report.failure.case_id}
                  {report.failure.inferred_from_saved_results ? ' · from saved case evidence' : ''}
                </p>
              )}
            </div>
          )}
          <RunLineage report={report} />
          {run.manifest.mode === 'preview' && (
            <p className={styles.notice}>
              Route preview only. This run provides routing diagnostics, not a capability score.
            </p>
          )}
          {run.manifest.mode === 'replay' && (
            <section className={styles.notice}>
              <h3>Diagnostic replay estimates</h3>
              <p>
                These estimates reuse saved answers. No new inference occurred. They are not a
                measured capability score, latency or cost-saving result.
              </p>
              <div className={styles.tableScroll}>
                <table>
                  <thead>
                    <tr>
                      <th>Target</th>
                      <th>Estimated macro accuracy</th>
                      <th>Estimated cost</th>
                      <th>Estimated latency p50</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map((target) => (
                      <tr key={target.id}>
                        <th>{target.id}</th>
                        <td>{percent(target.estimated_macro_accuracy)}</td>
                        <td>{money(target.estimated_cost_usd)}</td>
                        <td>{seconds(target.estimated_latency_p50_s)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
          <AccountingCorrection report={report} />
          <h3 id="run-targets">Target comparison</h3>
          {metrics.length ? (
            <MetricsTable targets={metrics} preview={run.manifest.mode === 'preview'} summary />
          ) : (
            <p className={styles.muted}>Summary metrics will appear as results are persisted.</p>
          )}
          <p className={styles.muted}>
            “—” means unrecorded or unknown, not zero. Cost is reported model usage; simulator and
            judge overhead remain separate in the report. Wall time is elapsed run time, not the sum
            of request durations.
          </p>
          <p className={styles.muted}>
            Cache-neutral estimates reprice every prompt token at the frozen fresh-input rate plus
            output. This counterfactual excludes cache discounts and premiums; it is neither billed
            spend nor a measured cache-free run.
          </p>
          {!!report?.benchmarks.length && (
            <section>
              <h3>Benchmark results</h3>
              <div className={styles.benchmarkResults}>
                {benchmarkIDs.map((benchmark) => (
                  <div key={benchmark}>
                    <h4>{benchmark}</h4>
                    <MetricsTable
                      targets={
                        report.benchmarks.filter(
                          (row) => row.benchmark === benchmark,
                        ) as TargetMetrics[]
                      }
                      preview={run.manifest.mode === 'preview'}
                    />
                  </div>
                ))}
              </div>
            </section>
          )}
          <details className={styles.details}>
            <summary>Token buckets and evaluation overhead</summary>
            <div className={styles.tableScroll}>
              <table>
                <thead>
                  <tr>
                    <th>Target</th>
                    <th>Input</th>
                    <th>Cache read</th>
                    <th>Cache write</th>
                    <th>Output</th>
                    <th>Subject calls</th>
                    <th>Simulator / judge cost</th>
                    <th>Total spend</th>
                    <th>Sum request time</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((target) => {
                    const usage = typeof target.tokens === 'object' ? target.tokens : null
                    return (
                      <tr key={target.id}>
                        <th>{target.id}</th>
                        {[
                          'input_tokens',
                          'cached_input_tokens',
                          'cache_write_tokens',
                          'output_tokens',
                        ].map((key) => (
                          <td key={key}>{number(usage?.[key])}</td>
                        ))}
                        <td>{number(target.call_count)}</td>
                        <td>{money(target.evaluation_cost_usd)}</td>
                        <td>{money(target.total_spend_usd)}</td>
                        <td>{seconds(target.request_time_sum_s)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </details>
          <RecipeEvidence report={report} targets={run.manifest.targets} />
          <div className={styles.twoColumns}>
            <Distribution
              title="Selected models"
              entries={reportDistribution(metrics, 'selected_models')}
            />
            <Distribution
              title="Matched decisions"
              entries={reportDistribution(metrics, 'decisions')}
            />
          </div>
          {run.manifest.mode !== 'live' && (
            <>
              {metrics.some(
                (target) => (target.selection_statuses?.execution_required ?? 0) > 0,
              ) && (
                <p className={styles.notice}>
                  Some model selections require live execution. Preview can identify the matched
                  decision, but does not determine the model for these cases.
                </p>
              )}
              <div className={styles.twoColumns}>
                <Distribution
                  title="Selection status"
                  entries={reportDistribution(metrics, 'selection_statuses')}
                />
                <Distribution
                  title="Selection explanation"
                  entries={reportDistribution(metrics, 'selection_reasons')}
                />
              </div>
            </>
          )}
          {!!report?.limitations.length && (
            <aside className={styles.notice}>
              <h3>Interpretation and limitations</h3>
              <ul>
                {report.limitations.map((limitation, i) => (
                  <li key={i}>{limitation}</li>
                ))}
              </ul>
            </aside>
          )}
          <div className={styles.sectionHeading}>
            <h3 id="run-cases">Case results</h3>
            <label className={styles.inlineLabel}>
              Filter loaded results
              <input
                type="search"
                value={filter}
                onChange={(event) => {
                  setFilter(event.target.value)
                  setPage(0)
                }}
                placeholder="Target, case, benchmark or status"
              />
            </label>
          </div>
          <p className={styles.muted}>
            Loaded {number(results.length)} of {number(resultsPage.total)} persisted results.
            Filtering applies to loaded results. Detail pages are snapshots; refresh evidence to
            reload them. Scores and costs above use the full report.
          </p>
          {report?.provenance.accounting_correction && (
            <p className={styles.notice}>
              Original receipt accounting is shown in case and call details. See the report summary
              for reconciled tokens and costs; original evidence has not been overwritten.
            </p>
          )}
          <div className={styles.tableScroll}>
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Benchmark</th>
                  <th>Target</th>
                  <th>Status</th>
                  <th>Score</th>
                  <th>{report?.provenance.accounting_correction ? 'Cost (original)' : 'Cost'}</th>
                  <th>Latency / TTFT</th>
                </tr>
              </thead>
              <tbody>
                {visible.slice(page * 25, (page + 1) * 25).map((result, i) => (
                  <tr key={`${result.case_id}-${result.target_id}-${i}`}>
                    <td>
                      <button className={styles.linkButton} onClick={() => setSelected(result)}>
                        {result.case_id}
                      </button>
                    </td>
                    <td>{result.benchmark}</td>
                    <td>{result.target_id}</td>
                    <td>{result.status}</td>
                    <td>{number(result.score, 3)}</td>
                    <td>{money(result.cost_usd)}</td>
                    <td>
                      {seconds(result.latency_s)} / {seconds(result.ttft_s)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!visible.length && <p className={styles.muted}>No matching persisted results.</p>}
          <div className={styles.actions}>
            <button disabled={page === 0} onClick={() => setPage((value) => value - 1)}>
              Previous results
            </button>
            <span>
              Showing {number(visible.length ? page * 25 + 1 : 0)}–
              {number(Math.min((page + 1) * 25, visible.length))} of {number(visible.length)} loaded
              matches · page {page + 1}
            </span>
            <button
              disabled={(page + 1) * 25 >= visible.length}
              onClick={() => setPage((value) => value + 1)}
            >
              Next results
            </button>
          </div>
          {resultsPage.error && (
            <p className={styles.error} role="alert">
              {resultsPage.error}
            </p>
          )}
          {resultsPage.nextCursor !== null && (
            <button disabled={resultsPage.loading} onClick={() => void loadMoreResults()}>
              {resultsPage.loading ? 'Loading results…' : 'Load more results'}
            </button>
          )}
          {selected && (
            <div className={styles.caseDetail}>
              <div className={styles.sectionHeading}>
                <h3>
                  {selected.case_id} · {selected.target_id}
                </h3>
                <button onClick={() => setSelected(null)}>Close case</button>
              </div>
              {selected.error && <p className={styles.error}>{selected.error}</p>}
              {run.manifest.mode === 'preview' ? (
                <PreviewEvidence routing={selected.details?.routing} />
              ) : (
                <>
                  <h4>Final answer</h4>
                  <pre>{selected.answer || 'No final answer recorded.'}</pre>
                </>
              )}
              <details>
                <summary>Usage, trace and grading evidence</summary>
                <pre>{JSON.stringify(selected, null, 2)}</pre>
              </details>
            </div>
          )}
          <CallEvidence
            id={id}
            calls={calls}
            page={callsPage}
            loadMore={loadMoreCalls}
            accountingReconciled={!!report?.provenance.accounting_correction}
          />
          <details className={styles.details}>
            <summary id="run-events">Run events ({events.length})</summary>
            {events.length > eventLimit && (
              <button onClick={() => setEventLimit((value) => value + 100)}>
                Show 100 earlier events
              </button>
            )}
            <p className={styles.muted}>
              Showing the latest {Math.min(eventLimit, events.length)} saved events.
            </p>
            <ol className={styles.events}>
              {events.slice(-eventLimit).map((event, i) => (
                <li key={event.seq ?? event.sequence ?? i}>
                  <strong>{event.kind ?? event.type ?? event.event ?? 'Event'}</strong>{' '}
                  <time>{event.at ?? event.timestamp}</time>
                  <pre>{JSON.stringify(event, null, 2)}</pre>
                </li>
              ))}
            </ol>
          </details>
          <details className={styles.details}>
            <summary id="run-provenance">Frozen manifest and provenance</summary>
            <pre>{JSON.stringify(run.manifest, null, 2)}</pre>
            <pre>{JSON.stringify(report?.provenance ?? {}, null, 2)}</pre>
          </details>
          {run.manifest.mode === 'live' && !active(run.status) && run.status !== 'completed' && (
            <RunRecovery run={run} canRun={canRun} onRecovered={onRecovered} />
          )}
          {run.status === 'completed' && run.manifest.mode === 'live' && (
            <RunArtifacts id={id} canRun={canRun} />
          )}
          <div className={styles.actions}>
            <a
              href={`${SR_BENCH_API}/runs/${encodeURIComponent(id)}/report`}
              target="_blank"
              rel="noreferrer"
            >
              Open report JSON ↗
            </a>
            <a
              href={`${SR_BENCH_API}/runs/${encodeURIComponent(id)}/results`}
              target="_blank"
              rel="noreferrer"
            >
              Open first results page ↗
            </a>
            <a
              href={`${SR_BENCH_API}/runs/${encodeURIComponent(id)}/events?after=0`}
              target="_blank"
              rel="noreferrer"
            >
              Open event ledger ↗
            </a>
          </div>
        </>
      )}
    </section>
  )
}
