import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ArrowRight, FileDown } from 'lucide-react'
import { Button, Checkbox, Label, cn } from '../../ui'
import { benchApi } from './api'
import { targetLabel, targetName } from './targetPresentation'
import { changeDirection, changeSummary, formatSignedChange } from './comparisonMetrics'
import AccountingCorrection from './AccountingCorrection'
import { money, number, percent, seconds, tokenTotal } from './model'
import type { Comparison, Report, Run } from './types'
import ComparisonSetup from './ComparisonSetup'
import useExperimentComparison from './useExperimentComparison'
import { comparisonSelectionReady } from './experimentComparison'
import BenchPagination from './BenchPagination'
import { IterationChart, QualityCostChart, type QualityCostPoint } from './EvaluationCharts'
import { ActionRow, BenchLoading, Disclosure, Notice, TableFrame } from './benchUi'
import { benchTable, benchText, identityClass } from './benchStyles'

const changeClass = (value: number | null | undefined) =>
  ({
    positive: 'text-success-foreground',
    negative: 'text-destructive-foreground',
    neutral: 'text-foreground',
    unknown: 'text-muted-foreground',
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
        <small className="mt-0.5 block text-xs font-normal text-current opacity-85">
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

function ComparisonCostBreakdown({ row }: { row: Comparison['comparisons'][number] }) {
  return (
    <Disclosure summary="Cost breakdown" bare className="my-2">
      <p className={benchText.muted}>
        Evaluation includes judges, simulators and other auxiliary calls. Unknown costs are not
        treated as zero.
      </p>
      <TableFrame className="my-2">
        <table className={benchTable} aria-label="Cost breakdown">
          <thead>
            <tr>
              <th>Cost</th>
              <th>Best single model</th>
              <th>This run</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Subject model</th>
              <td>{money(row.baseline_subject_cost_usd)}</td>
              <td>{money(row.candidate_subject_cost_usd)}</td>
            </tr>
            <tr>
              <th scope="row">Evaluation</th>
              <td>{money(row.baseline_evaluation_cost_usd)}</td>
              <td>{money(row.candidate_evaluation_cost_usd)}</td>
            </tr>
            <tr>
              <th scope="row">Total</th>
              <td>{money(row.baseline_total_cost_usd)}</td>
              <td>{money(row.candidate_total_cost_usd)}</td>
            </tr>
          </tbody>
        </table>
      </TableFrame>
      <p className="text-xs text-muted-foreground">
        Subject model saving: <ChangeValue value={row.subject_cost_saving_percent} unit="%" />
      </p>
    </Disclosure>
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
  const experiment = search.get('experiment') ?? undefined
  const includeOutside = search.get('comparison_scope') === 'all'
  const [baseline, setBaseline] = useState(savedBaseline)
  const [candidates, setCandidates] = useState(savedIterations.split('|').filter(Boolean))
  const [savedResults, setResults] = useState<IterationEvidence[]>([])
  const [savedBaselineReport, setBaselineReport] = useState<Report | null>(null)
  const [evidenceKey, setEvidenceKey] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)
  const [evidencePage, setEvidencePage] = useState(0)
  const options = useExperimentComparison(baseline, includeOutside ? undefined : experiment)
  const verifiedSelection =
    comparisonSelectionReady(
      baseline,
      candidates,
      options.baselines.items,
      options.choices.items,
    ) &&
    !options.membershipLoading &&
    !options.membershipError &&
    !options.baselines.loading &&
    !options.baselines.error &&
    !options.choices.loading &&
    !options.choices.error
  const draftChanged =
    baseline !== savedBaseline ||
    [...candidates].sort().join('|') !== savedIterations.split('|').filter(Boolean).sort().join('|')
  const selectionKey = `${experiment ?? ''}:${includeOutside}:${savedBaseline}|${savedIterations}`
  const evidenceMatches = verifiedSelection && !draftChanged && evidenceKey === selectionKey
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
    if (!savedBaseline || !selected.length || draftChanged || !verifiedSelection) {
      setPending(false)
      setError('')
      setResults([])
      setBaselineReport(null)
      return
    }
    async function load() {
      setEvidenceKey(selectionKey)
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
  }, [savedBaseline, savedIterations, revision, draftChanged, verifiedSelection, selectionKey])

  function compare() {
    if (!verifiedSelection) return
    setEvidencePage(0)
    const next = new URLSearchParams({ view: 'compare', baseline })
    if (experiment) next.set('experiment', experiment)
    if (experiment && includeOutside) next.set('comparison_scope', 'all')
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
        'baseline_subject_cost_usd',
        'candidate_subject_cost_usd',
        'baseline_evaluation_cost_usd',
        'candidate_evaluation_cost_usd',
        'baseline_total_cost_usd',
        'candidate_total_cost_usd',
        'subject_cost_saving_percent',
        'total_cost_saving_percent',
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
            row.baseline_subject_cost_usd,
            row.candidate_subject_cost_usd,
            row.baseline_evaluation_cost_usd,
            row.candidate_evaluation_cost_usd,
            row.baseline_total_cost_usd,
            row.candidate_total_cost_usd,
            row.subject_cost_saving_percent,
            row.total_cost_saving_percent,
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
            typeof target.macro_accuracy === 'number' && typeof target.total_spend_usd === 'number'
              ? [
                  {
                    name: targetName(reviewedBaseline?.manifest, target.id),
                    quality: target.macro_accuracy * 100,
                    cost: target.total_spend_usd,
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
                typeof row.candidate_total_cost_usd === 'number'
                ? [
                    {
                      name: `${item.stage} · ${targetName(runs.find((run) => run.id === item.id)?.manifest, row.candidate_target_id)}`,
                      quality: metric.macro_accuracy * 100,
                      cost: row.candidate_total_cost_usd,
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
            saving: row.total_cost_saving_percent,
          }
        })
      : []
  const currentEvidencePage = Math.min(evidencePage, Math.max(0, Math.ceil(results.length / 6) - 1))
  const baselineMetric = baselineTargets.find(
    (target) => target.id === results[0]?.comparison?.comparisons[0]?.baseline_target_id,
  )
  return (
    <div className="min-w-0">
      {!!results.length && (
        <p className={`${benchText.muted} max-w-3xl`}>
          Total cost includes model answers and evaluation calls, priced from recorded usage at
          frozen rates. These are estimates, not invoices or hardware costs.
        </p>
      )}
      {experiment && (
        <Label className="mt-3 w-fit gap-2 font-normal text-foreground">
          <Checkbox
            checked={includeOutside}
            onCheckedChange={(checked) => {
              const next = new URLSearchParams({ view: 'compare', experiment })
              if (checked === true) next.set('comparison_scope', 'all')
              setBaseline('')
              setCandidates([])
              setSearch(next)
            }}
          />
          Include runs outside this experiment
        </Label>
      )}
      <ComparisonSetup
        baseline={baseline}
        candidates={candidates}
        pending={pending}
        options={options}
        onBaseline={setBaseline}
        onCandidates={setCandidates}
        onCompare={compare}
      />
      {draftChanged && savedResults.length > 0 && (
        <Notice tone="info" as="p">
          Selection changed. Choose Compare runs to update the displayed evidence.
        </Notice>
      )}
      {error && (
        <Notice tone="danger" as="p" role="alert">
          {error}
        </Notice>
      )}
      {pending && <BenchLoading label="Comparing saved evaluation evidence…" lines={3} />}
      {qualified && (baselineFailures > 0 || candidateFailures > 0) && (
        <Notice tone="warning" as="p">
          {number(baselineFailures)} failed baseline results · {number(candidateFailures)} failed
          candidate results. Scores use all planned cases; explicit failed outcomes count as
          incorrect. Run statuses remain unchanged, and unknown costs cannot support a saving claim.
        </Notice>
      )}
      {qualified && (
        <div className="my-4 grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-2 [&>section]:my-0">
          <QualityCostChart points={points} costBasis="total" />
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
        <Notice tone="warning" as="p">
          A continuous trend is withheld until every selected iteration has a complete, compatible
          comparison.
        </Notice>
      )}
      {results.length > 0 && (
        <div className="my-4 grid grid-cols-[repeat(auto-fit,minmax(min(280px,100%),1fr))] gap-3">
          {results.slice(currentEvidencePage * 6, currentEvidencePage * 6 + 6).map((item) => (
            <article
              key={item.id}
              className="flex min-w-0 flex-col rounded-xl border border-border bg-surface p-4 shadow-xs"
            >
              <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
                {item.stage}
              </p>
              <h3 className={`${benchText.h3} mb-3 mt-1 break-words`}>
                {runs.find((run) => run.id === item.id)?.manifest.name ?? item.id}
              </h3>
              {item.comparison && !item.comparison.candidate_quality_complete && (
                <p className={`${benchText.muted} mb-2`}>
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
                <Notice tone="danger" as="p" className="my-0">
                  Comparison withheld: {item.error}
                </Notice>
              ) : (
                item.comparison?.comparisons.map((row) => (
                  <div key={row.candidate_target_id} className="min-w-0">
                    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border">
                      <div className="flex flex-col gap-0.5 bg-card px-3 py-2.5">
                        <span className="text-xs text-muted-foreground">Quality Δ</span>
                        <strong className="text-lg font-semibold leading-tight tabular-nums">
                          <ChangeValue
                            value={row.quality_delta * 100}
                            unit=" pp"
                            metric="quality"
                          />
                        </strong>
                      </div>
                      <div className="flex flex-col gap-0.5 bg-card px-3 py-2.5">
                        <span className="text-xs text-muted-foreground">Total cost saving</span>
                        <strong className="text-lg font-semibold leading-tight tabular-nums">
                          <ChangeValue
                            value={row.total_cost_saving_percent}
                            unit="%"
                            metric="cost"
                          />
                        </strong>
                      </div>
                    </div>
                    {row.total_cost_comparison_reason && (
                      <p className={`${benchText.muted} mt-2`}>
                        {row.total_cost_comparison_reason}
                      </p>
                    )}
                    <ComparisonCostBreakdown row={row} />
                    <p className={benchText.muted}>
                      95% paired interval <ChangeInterval values={row.quality_delta_ci95} /> ·{' '}
                      {number(row.paired_cases)} cases
                    </p>
                    <p className={`${benchText.muted} mt-1`}>
                      {row.quality_delta_ci95[0] <= 0 && row.quality_delta_ci95[1] >= 0
                        ? 'Interval includes zero; improvement is not established.'
                        : 'See the comparison assumptions before making a quality claim.'}
                    </p>
                  </div>
                ))
              )}
              {item.report?.provenance.accounting_correction && (
                <Notice tone="info" as="p" className="mb-0">
                  {item.report.provenance.accounting_correction.qualified
                    ? 'Accounting verified'
                    : 'Partial accounting'}{' '}
                  · {number(item.report.provenance.accounting_correction.corrected_call_count)}{' '}
                  corrected calls. Original receipts preserved; full correction evidence below.
                </Notice>
              )}
              <div className="mt-auto pt-3">
                <Button variant="ghost" size="xs" className="-ml-2" asChild>
                  <a href={`?view=runs&run=${encodeURIComponent(item.id)}`}>
                    Open run <ArrowRight aria-hidden="true" />
                  </a>
                </Button>
              </div>
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
          <Disclosure summary="Single-model baseline metrics">
            <p className={benchText.muted}>
              {reviewedBaseline?.manifest.name ?? savedBaseline} ·{' '}
              {baselineComparison?.baseline_status}
            </p>
            <TableFrame>
              <table className={benchTable}>
                <thead>
                  <tr>
                    <th>Single model</th>
                    <th>Macro accuracy</th>
                    <th>Correct / denominator</th>
                    <th>Saved outcomes</th>
                    <th>Subject model cost</th>
                    <th>Evaluation cost</th>
                    <th>Total cost</th>
                    <th>Cache-neutral subject estimate</th>
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
                        <td>{money(target.evaluation_cost_usd)}</td>
                        <td>{money(target.total_spend_usd)}</td>
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
            </TableFrame>
          </Disclosure>
        </>
      )}
      {!!results.length && (
        <>
          <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
            <h3 className={benchText.h3}>Comparison evidence</h3>
            <ActionRow>
              <Button variant="outline" size="sm" onClick={() => download('csv')}>
                <FileDown aria-hidden="true" />
                Export comparison CSV
              </Button>
              <Button variant="outline" size="sm" onClick={() => download('json')}>
                <FileDown aria-hidden="true" />
                Export comparison JSON
              </Button>
            </ActionRow>
          </div>
          <p className={`${benchText.muted} mt-2 md:hidden`}>
            Swipe the tables horizontally to inspect all metrics.
          </p>
          <p className={`${benchText.muted} mt-2 max-w-3xl`}>
            Quality differences are percentage points. Intervals that include zero do not establish
            an improvement. Unknown cost cannot support a saving claim. Development-set gains
            require a separate holdout check.
          </p>
          {!!baselineComparison?.baseline_selection_qualification &&
            (baselineFailures > 0 || candidateFailures > 0) && (
              <p className={`${benchText.muted} mt-2 max-w-3xl`}>
                {baselineComparison.baseline_selection_qualification}
              </p>
            )}
          <p className={`${benchText.muted} mt-2 max-w-3xl`}>
            Cache-neutral estimates cover subject model calls only, using the same selected
            baseline. They reprice prompt tokens at the frozen fresh-input rate plus output,
            excluding evaluation calls and cache discounts or premiums. They are not measured spend.
          </p>
          <Disclosure summary="Detailed iteration metrics and uncertainty">
            <TableFrame className="my-0">
              <table className={benchTable}>
                <thead>
                  <tr>
                    <th>Iteration / target</th>
                    <th>Macro accuracy</th>
                    <th>Correct / denominator</th>
                    <th>Quality Δ vs best single</th>
                    <th>95% paired interval</th>
                    <th>Total cost / saving</th>
                    <th>Cache-neutral subject estimate / saving</th>
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
                                  <span className="block text-warning-foreground">
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
                                  </span>
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
                                <Disclosure
                                  summary="Uncertainty details"
                                  bare
                                  className="my-1 whitespace-normal"
                                  bodyClassName="max-w-[22rem] text-xs text-muted-foreground"
                                >
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
                                </Disclosure>
                              )}
                            </td>
                            <td>
                              {money(row.candidate_total_cost_usd)}
                              <small>
                                <ChangeValue
                                  value={row.total_cost_saving_percent}
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
            </TableFrame>
          </Disclosure>
          <Disclosure summary="Technical comparison evidence">
            <div className="divide-y divide-border">
              {results.map((item) => (
                <section className="min-w-0 py-3 first:pt-0 last:pb-0" key={item.id}>
                  <h4 className={benchText.h4}>
                    {item.stage} ·{' '}
                    {runs.find((run) => run.id === item.id)?.manifest.name ?? item.id}
                  </h4>
                  {item.error ? (
                    <Notice tone="danger" as="p" role="alert">
                      Comparison withheld: {item.error}
                    </Notice>
                  ) : (
                    <>
                      <AccountingCorrection report={item.report ?? null} />
                      <Disclosure summary="Baseline selection policy" bare className="my-1">
                        <p className={benchText.muted}>{item.comparison?.baseline_selection}</p>
                        {(item.comparison?.baseline_tied_best_target_ids?.length ?? 0) > 1 && (
                          <Notice tone="info" as="p">
                            Tied best single models:{' '}
                            {item.comparison?.baseline_tied_best_target_ids
                              ?.map((id) => targetName(reviewedBaseline?.manifest, id))
                              .join(', ')}
                            . {item.comparison?.baseline_tie_policy}
                          </Notice>
                        )}
                      </Disclosure>
                      {item.comparison?.baseline_cost_comparison_eligible === false && (
                        <Notice tone="warning" as="p">
                          Cost saving withheld:{' '}
                          {item.comparison.baseline_cost_comparison_reason ??
                            'Baseline cost evidence is incomplete.'}
                        </Notice>
                      )}
                      <Disclosure summary="Frozen configuration identity" bare className="my-1">
                        <dl
                          className={cn(identityClass, 'grid-cols-1 [&>div]:grid [&>div]:gap-0.5')}
                        >
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
                      </Disclosure>
                    </>
                  )}
                </section>
              ))}
            </div>
          </Disclosure>
        </>
      )}
    </div>
  )
}
