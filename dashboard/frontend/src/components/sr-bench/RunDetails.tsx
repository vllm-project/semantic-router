import { useState } from 'react'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'
import { QualityCostChart, RoutingBars } from './EvaluationCharts'
import { benchApi, SR_BENCH_API } from './api'
import { active, reportDistribution, money, number, percent, seconds, tokenTotal } from './model'
import type { CaseResult, Manifest, Run, TargetMetrics } from './types'
import { targetName } from './targetPresentation'
import { canReuseBaseline } from './baselineReuse'
import { useRunEvidence } from './useRunEvidence'
import RunArtifacts from './RunArtifacts'
import CallEvidence from './CallEvidence'
import RunEvents from './RunEvents'
import BenchPagination from './BenchPagination'
import AccountingCorrection from './AccountingCorrection'
import OutputDiagnostics from './OutputDiagnostics'
import PreviewEvidence from './PreviewEvidence'
import RunRecovery from './RunRecovery'
import RunLineage from './RunLineage'
import RecipeEvidence from './RecipeEvidence'
import { RunStatus } from './RunList'
import styles from './SrBench.module.css'
import controls from './BenchControls.module.css'

function MetricsTable({
  targets,
  manifest,
  preview,
  summary = false,
}: {
  targets: TargetMetrics[]
  manifest: Manifest
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
            <th>Correct / denominator</th>
            <th>Failures</th>
            <th>Observed model cost</th>
            <th>Tokens</th>
            <th>Latency p50 / p95</th>
          </tr>
        </thead>
        <tbody>
          {targets.map((target) => (
            <tr key={target.id}>
              <th scope="row">{targetName(manifest, target.id)}</th>
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
              <td>{preview ? '—' : `${number(target.correct)} / ${number(target.total)}`}</td>
              <td>{number(target.failed)}</td>
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
  )
}

function questionText(run: Run, caseID: string): string {
  const source = run.manifest.cases?.find(
    (value) => value && typeof value === 'object' && (value as { id?: string }).id === caseID,
  ) as { messages?: Array<{ role?: string; content?: unknown }> } | undefined
  return (
    source?.messages
      ?.filter((message) => message.role === 'user')
      .map((message) =>
        typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
      )
      .join('\n\n') || 'Question text is unavailable in this saved manifest.'
  )
}

export default function RunDetails({
  id,
  actorID,
  canRun,
  section = 'results',
  onSectionChange,
  onChanged,
  onRecovered,
  onCandidate,
}: {
  id: string
  actorID: string
  canRun: boolean
  section?: string
  onSectionChange: (section: string) => void
  onChanged: () => void
  onRecovered: (run: Run) => void
  onCandidate: (id: string, mode: 'live' | 'preview') => void
}) {
  const [pending, setPending] = useState(false)
  const [filter, setFilter] = useState('')
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState<CaseResult | null>(null)
  const [revision, setRevision] = useState(0)
  const [actionError, setActionError] = useState('')
  const {
    run,
    readAt,
    report,
    reportRead,
    results,
    events,
    eventsPage,
    loadMoreEvents,
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
    `${result.case_id} ${result.target_id} ${targetName(run?.manifest, result.target_id)} ${result.benchmark} ${result.status}`
      .toLowerCase()
      .includes(filter.toLowerCase()),
  )
  const currentPage = Math.min(page, Math.max(0, Math.ceil(visible.length / 25) - 1))
  const progress = run?.progress
  const percentage = progress?.total
    ? Math.min(100, ((progress.completed + progress.failed) / progress.total) * 100)
    : 0
  const metrics = report?.summary.targets ?? run?.summary?.targets ?? []
  const totalTokens =
    metrics.length && metrics.every((target) => tokenTotal(target.tokens) !== null)
      ? metrics.reduce((sum, target) => sum + tokenTotal(target.tokens)!, 0)
      : null
  const chartPoints =
    run?.status === 'completed' && run.manifest.mode === 'live'
      ? metrics.flatMap((target) =>
          typeof target.macro_accuracy === 'number' && typeof target.cost_usd === 'number'
            ? [
                {
                  name: targetName(run.manifest, target.id),
                  quality: target.macro_accuracy * 100,
                  cost: target.cost_usd,
                  kind:
                    run.manifest.targets.find((item) => item.id === target.id)?.kind ??
                    ('single' as const),
                },
              ]
            : [],
        )
      : []
  const benchmarkIDs = [...new Set(report?.benchmarks.map((row) => String(row.benchmark)) ?? [])]
  return (
    <section className={styles.panel} aria-labelledby="run-detail-title">
      <div className={styles.sectionHeading}>
        <div>
          <h2 id="run-detail-title">{run?.manifest.name ?? 'Evaluation run'}</h2>
          <p className={styles.runSubtitle}>
            {run && (
              <>
                <RunStatus status={run.status} /> {run.manifest.profile} · {run.manifest.mode} ·{' '}
                {run.manifest.targets.length} targets
              </>
            )}
          </p>
        </div>
        <div className={styles.actions}>
          <button onClick={refreshEvidence}>
            <ProductIcon name="refresh" />
            Refresh evidence
          </button>
          {run && active(run.status) && (
            <button
              className={styles.danger}
              disabled={!canRun || pending}
              onClick={() => void cancel()}
            >
              {pending ? 'Stopping…' : 'Cancel evaluation'}
            </button>
          )}
          {canRun && canReuseBaseline(run) && (
            <>
              <button
                className={controls.compactButton}
                onClick={() => onCandidate(run.id, 'preview')}
              >
                <ProductIcon name="decision" /> Preview candidate
              </button>
              <button
                className={controls.compactButton}
                onClick={() => onCandidate(run.id, 'live')}
              >
                <ProductIcon name="evaluation" /> Evaluate candidate
              </button>
            </>
          )}
        </div>
      </div>
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {!run ? (
        readError ? (
          <p role="status">Run details are unavailable. Refresh evidence to retry.</p>
        ) : (
          <ProductLoadingState compact label="Loading run…" />
        )
      ) : (
        <>
          <div className={styles.metricGrid}>
            <div>
              <span>Completed cells</span>
              <strong>
                {number(progress?.completed)} <small>/ {number(progress?.total)}</small>
              </strong>
              <small>{number(progress?.failed)} failed</small>
            </div>
            <div>
              <span>Total recorded cost</span>
              <strong>{money(report?.summary.total_spend_usd)}</strong>
              <small>Includes evaluation overhead</small>
            </div>
            <div>
              <span>Model tokens</span>
              <strong>{number(totalTokens)}</strong>
              <small>All four usage buckets</small>
            </div>
            <div>
              <span>Elapsed wall time</span>
              <strong>{seconds(report?.summary.wall_time_s ?? run.summary?.wall_time_s)}</strong>
              <small>Saved run duration</small>
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
          <nav className={styles.detailNav} role="tablist" aria-label="Run sections">
            {(
              [
                ['results', 'Results', 'chart'],
                ['questions', 'Questions', 'list'],
                ['calls', 'Calls', 'trace'],
                ['evidence', 'Evidence', 'audit'],
                ...(run.manifest.targets.some((target) => target.kind === 'mom')
                  ? [['recipe', 'Recipe', 'mixture']]
                  : []),
              ] as Array<[string, string, import('../ProductIcon').ProductIconName]>
            ).map(([value, label, icon]) => (
              <button
                key={value}
                role="tab"
                id={`run-tab-${value}`}
                tabIndex={section === value ? 0 : -1}
                onKeyDown={(event) => {
                  if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return
                  event.preventDefault()
                  const tabs = Array.from(
                    event.currentTarget
                      .closest('[role=tablist]')!
                      .querySelectorAll<HTMLButtonElement>('[role=tab]'),
                  )
                  const index = tabs.indexOf(event.currentTarget)
                  const next =
                    event.key === 'Home'
                      ? 0
                      : event.key === 'End'
                        ? tabs.length - 1
                        : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) %
                          tabs.length
                  tabs[next].focus()
                  tabs[next].click()
                }}
                aria-selected={section === value}
                aria-controls={`run-panel-${value}`}
                onClick={() => onSectionChange(value)}
              >
                <ProductIcon name={icon} />
                {label}
              </button>
            ))}
          </nav>
          {run.status !== 'completed' && (
            <p className={styles.notice}>
              This run is {run.status}. {number(run.progress.failed)} failed results remain part of
              the saved evidence. Comparisons require an explicit terminal outcome for every planned
              result; missing or ungraded results are not eligible.
            </p>
          )}
          {(run.error || report?.failure) && (
            <div className={styles.error} role="alert">
              <p>{run.error || report?.failure?.reason}</p>
              {!run.error && report?.failure && (
                <p>
                  Model {targetName(run.manifest, report.failure.target_id)} · case{' '}
                  {report.failure.case_id}
                  {report.failure.inferred_from_saved_results ? ' · from saved case evidence' : ''}
                </p>
              )}
            </div>
          )}
          <div
            role="tabpanel"
            id="run-panel-results"
            aria-labelledby="run-tab-results"
            hidden={section !== 'results'}
          >
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
                          <th>{targetName(run.manifest, target.id)}</th>
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
            {section === 'results' &&
              report &&
              run.status === 'completed' &&
              run.manifest.mode === 'live' && <QualityCostChart points={chartPoints} />}
            <h3 id="run-targets">Target comparison</h3>
            <p className={styles.muted}>
              Costs apply frozen per-token prices to recorded usage; they are not invoice or
              hardware-cost measurements.
            </p>
            {metrics.length ? (
              <MetricsTable
                manifest={run.manifest}
                targets={metrics}
                preview={run.manifest.mode === 'preview'}
                summary
              />
            ) : reportRead.loading ? (
              <ProductLoadingState compact label="Loading report metrics…" />
            ) : reportRead.error ? (
              <p className={styles.error}>
                Report metrics are unavailable. Refresh evidence to retry.
              </p>
            ) : (
              <p className={styles.muted}>Summary metrics will appear as results are persisted.</p>
            )}
            <p className={styles.muted}>
              Unknown usage and cost remain “—”; incomplete evidence cannot establish a cost saving.
            </p>
            {run.manifest.mode === 'live' && report && (
              <OutputDiagnostics targets={report.summary.targets} manifest={run.manifest} />
            )}
            <details className={styles.details}>
              <summary>Cost and timing interpretation</summary>
              <p className={styles.muted}>
                “—” means unrecorded or unknown, not zero. Cost is reported model usage; simulator
                and judge overhead remain separate in the report. Wall time is elapsed run time, not
                the sum of request durations.
              </p>
              <p className={styles.muted}>
                Cache-neutral estimates reprice every prompt token at the frozen fresh-input rate
                plus output. This counterfactual excludes cache discounts and premiums; it is
                neither billed spend nor a measured cache-free run.
              </p>
            </details>
            {!!report?.benchmarks.length && (
              <details className={styles.details}>
                <summary>Benchmark results</summary>
                <div className={styles.benchmarkResults}>
                  {benchmarkIDs.map((benchmark) => (
                    <div key={benchmark}>
                      <h4>{benchmark}</h4>
                      <MetricsTable
                        manifest={run.manifest}
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
              </details>
            )}
            {(report || metrics.length > 0) && (
              <div className={styles.twoColumns}>
                <RoutingBars
                  title="Selected models"
                  entries={reportDistribution(metrics, 'selected_models', run.manifest)}
                />
                <RoutingBars
                  title="Matched decisions"
                  entries={reportDistribution(metrics, 'decisions', run.manifest)}
                />
              </div>
            )}
            {run.manifest.mode !== 'live' && (report || metrics.length > 0) && (
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
                  <RoutingBars
                    title="Selection status"
                    entries={reportDistribution(metrics, 'selection_statuses', run.manifest)}
                  />
                  <RoutingBars
                    title="Selection explanation"
                    entries={reportDistribution(metrics, 'selection_reasons', run.manifest)}
                  />
                </div>
              </>
            )}
            {!!report?.limitations.length && (
              <aside className={styles.notice}>
                <p>{report.limitations[0]}</p>
              </aside>
            )}
          </div>
          <div
            role="tabpanel"
            id="run-panel-questions"
            aria-labelledby="run-tab-questions"
            hidden={section !== 'questions'}
          >
            <div hidden={!!selected}>
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
              {resultsPage.total !== null && (
                <p className={styles.muted}>
                  Loaded {number(results.length)} of {number(resultsPage.total)} persisted results.
                  Filtering applies to loaded results. Detail pages are snapshots; refresh evidence
                  to reload them. Scores and costs above use the full report.
                </p>
              )}
              {resultsPage.loading && resultsPage.total === null && (
                <ProductLoadingState compact label="Loading persisted case results…" />
              )}
              {report?.provenance.accounting_correction && (
                <p className={styles.notice}>
                  Original receipt accounting is shown in case and call details. See the report
                  summary for reconciled tokens and costs; original evidence has not been
                  overwritten.
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
                      <th>
                        {report?.provenance.accounting_correction ? 'Cost (original)' : 'Cost'}
                      </th>
                      <th>Latency / TTFT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.slice(currentPage * 25, (currentPage + 1) * 25).map((result, i) => (
                      <tr key={`${result.case_id}-${result.target_id}-${i}`}>
                        <td>
                          <button className={styles.linkButton} onClick={() => setSelected(result)}>
                            {result.case_id}
                          </button>
                        </td>
                        <td>{result.benchmark}</td>
                        <td>{targetName(run.manifest, result.target_id)}</td>
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
              {!visible.length &&
                !resultsPage.loading &&
                !resultsPage.error &&
                resultsPage.total !== null && (
                  <p className={styles.muted}>
                    {resultsPage.total === 0
                      ? 'No persisted case results yet.'
                      : 'No matching persisted results.'}
                  </p>
                )}
              <BenchPagination
                label="Results"
                total={visible.length}
                page={currentPage}
                pageSize={25}
                onChange={setPage}
              />
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
            </div>
            {selected && (
              <div className={styles.recordDetail}>
                <div className={styles.sectionHeading}>
                  <h3>
                    {selected.case_id} · {targetName(run.manifest, selected.target_id)}
                  </h3>
                  <button onClick={() => setSelected(null)}>
                    <ProductIcon name="arrow-left" />
                    Back to questions
                  </button>
                </div>
                <h4>Question</h4>
                <pre>{questionText(run, selected.case_id)}</pre>
                {selected.error && <p className={styles.error}>{selected.error}</p>}
                {run.manifest.mode === 'preview' ? (
                  <PreviewEvidence routing={selected.details?.routing} />
                ) : (
                  <>
                    <h4>
                      {['mmlu-pro', 'gpqa-diamond'].includes(selected.benchmark)
                        ? 'Parsed answer'
                        : 'Recorded answer'}
                    </h4>
                    <pre>
                      {selected.answer == null || selected.answer === ''
                        ? ['mmlu-pro', 'gpqa-diamond'].includes(selected.benchmark)
                          ? 'No answer parsed.'
                          : 'No answer recorded for this case.'
                        : typeof selected.answer === 'string'
                          ? selected.answer
                          : JSON.stringify(selected.answer, null, 2)}
                    </pre>
                    {selected.details?.quality_failure === 'output_limit' ? (
                      <p className={styles.notice}>
                        The output limit was reached. This case counts as incorrect under the frozen
                        protocol.
                      </p>
                    ) : selected.details?.strict_format === false ? (
                      <p className={styles.notice}>
                        The final text did not match the required answer format. Inspect the
                        original call for the full response.
                      </p>
                    ) : null}
                  </>
                )}
                <details>
                  <summary>Usage, trace and grading evidence</summary>
                  <pre>{JSON.stringify(selected, null, 2)}</pre>
                </details>
              </div>
            )}
          </div>
          <div
            role="tabpanel"
            id="run-panel-calls"
            aria-labelledby="run-tab-calls"
            hidden={section !== 'calls'}
          >
            <CallEvidence
              id={id}
              calls={calls}
              manifest={run.manifest}
              page={callsPage}
              loadMore={loadMoreCalls}
              accountingReconciled={!!report?.provenance.accounting_correction}
            />
          </div>
          <div
            role="tabpanel"
            id="run-panel-evidence"
            aria-labelledby="run-tab-evidence"
            hidden={section !== 'evidence'}
          >
            <h3>Run evidence</h3>
            <p className={styles.muted}>
              Run ID <code>{id}</code>. Original receipts and frozen configuration remain available
              here.
            </p>
            <RunLineage report={report} />
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
                      <th>Cache-neutral estimate</th>
                      <th>TTFT p50</th>
                      <th>Sum request time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map((target) => {
                      const usage = typeof target.tokens === 'object' ? target.tokens : null
                      return (
                        <tr key={target.id}>
                          <th>{targetName(run.manifest, target.id)}</th>
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
                          <td>{money(target.cache_neutral_cost_usd)}</td>
                          <td>{seconds(target.ttft_p50_s)}</td>
                          <td>{seconds(target.request_time_sum_s)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </details>
            <RunEvents
              manifest={run.manifest}
              events={events}
              page={eventsPage}
              loadMore={loadMoreEvents}
            />
            <details className={styles.details}>
              <summary id="run-provenance">Frozen manifest and provenance</summary>
              <pre>{JSON.stringify(run.manifest, null, 2)}</pre>
              <pre>{JSON.stringify(report?.provenance ?? {}, null, 2)}</pre>
            </details>
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
          </div>
          <div
            role="tabpanel"
            id="run-panel-recipe"
            aria-labelledby="run-tab-recipe"
            hidden={section !== 'recipe'}
          >
            {report ? (
              <RecipeEvidence report={report} targets={run.manifest.targets} />
            ) : reportRead.error ? (
              <p className={styles.error}>
                Recipe evidence is unavailable. Refresh evidence to retry.
              </p>
            ) : (
              <ProductLoadingState compact label="Loading recipe evidence…" />
            )}
          </div>
          {run.manifest.mode === 'live' && !active(run.status) && run.status !== 'completed' && (
            <RunRecovery
              key={`${actorID}:${run.id}`}
              run={run}
              actorID={actorID}
              canRun={canRun}
              onRecovered={onRecovered}
            />
          )}
        </>
      )}
    </section>
  )
}
