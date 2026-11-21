import type { ReactNode } from 'react'
import { Button, cn } from '../../ui'
import { QualityCostChart, RoutingBars } from './EvaluationCharts'
import {
  active,
  hasScoredOutcomes,
  money,
  number,
  percent,
  reportDistribution,
  seconds,
  tokenTotal,
} from './model'
import type { Manifest, Report, Run, TargetMetrics } from './types'
import { targetName } from './targetPresentation'
import { benchmarkTitle } from './datasetPresentation'
import AccountingCorrection from './AccountingCorrection'
import OutputDiagnostics from './OutputDiagnostics'
import DatasetExposure from './DatasetExposure'
import { BenchLoading, Disclosure, Notice, TableFrame } from './benchUi'
import { benchTable, benchText } from './benchStyles'

export function MetricsTable({
  targets,
  manifest,
  preview,
  running,
  summary = false,
}: {
  targets: TargetMetrics[]
  manifest: Manifest
  preview: boolean
  running: boolean
  summary?: boolean
}) {
  return (
    <TableFrame>
      <table className={benchTable}>
        <thead>
          <tr>
            <th>Target</th>
            <th>{summary ? 'Macro accuracy' : 'Accuracy'}</th>
            <th>Correct / denominator</th>
            <th>Scored / planned</th>
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
                  : hasScoredOutcomes(target)
                    ? percent(summary ? target.macro_accuracy : target.accuracy)
                    : running
                      ? 'Pending'
                      : 'Incomplete'}
                {!preview &&
                  !summary &&
                  hasScoredOutcomes(target) &&
                  Array.isArray(target.accuracy_ci95) && (
                    <small>
                      95% CI {target.accuracy_ci95.map((value) => percent(value)).join(' – ')}
                    </small>
                  )}
              </td>
              <td>{preview ? '—' : `${number(target.correct)} / ${number(target.total)}`}</td>
              <td>{preview ? '—' : `${number(target.scored)} / ${number(target.total)}`}</td>
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
    </TableFrame>
  )
}

/** Targets as rows and benchmarks as columns; the best scored value per column is emphasized. */
function BenchmarkMatrix({
  run,
  report,
  benchmarks,
}: {
  run: Run
  report: Report
  benchmarks: string[]
}) {
  const running = active(run.status)
  const rows = report.benchmarks as Array<TargetMetrics & { benchmark: string }>
  const targetIDs = [...new Set(rows.map((row) => row.id))]
  const cell = (target: string, benchmark: string) =>
    rows.find((row) => row.id === target && String(row.benchmark) === benchmark)
  const best = Object.fromEntries(
    benchmarks.map((benchmark) => [
      benchmark,
      Math.max(
        ...targetIDs.map((target) => {
          const row = cell(target, benchmark)
          return row && hasScoredOutcomes(row) && typeof row.accuracy === 'number'
            ? row.accuracy
            : -1
        }),
      ),
    ]),
  )
  return (
    <TableFrame>
      <table className={benchTable}>
        <thead>
          <tr>
            <th>Target</th>
            {benchmarks.map((benchmark) => (
              <th key={benchmark} className="text-right">
                {benchmarkTitle(benchmark)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {targetIDs.map((target) => (
            <tr key={target}>
              <th scope="row">{targetName(run.manifest, target)}</th>
              {benchmarks.map((benchmark) => {
                const row = cell(target, benchmark)
                const scored = !!row && hasScoredOutcomes(row) && typeof row.accuracy === 'number'
                return (
                  <td
                    key={benchmark}
                    className={cn(
                      'text-right font-mono tabular-nums',
                      scored && targetIDs.length > 1 && row!.accuracy === best[benchmark]
                        ? 'font-semibold text-foreground'
                        : 'text-muted-foreground',
                    )}
                  >
                    {!row
                      ? '—'
                      : scored
                        ? percent(row.accuracy)
                        : running
                          ? 'Pending'
                          : 'Incomplete'}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </TableFrame>
  )
}

export default function RunResults({
  run,
  report,
  reportRead,
  visible,
  recovery,
  onRetry,
}: {
  run: Run
  report: Report | null
  reportRead: { loading: boolean; error: string }
  visible: boolean
  recovery?: ReactNode
  onRetry?: () => void
}) {
  const metrics = report?.summary.targets ?? run.summary?.targets ?? []
  const chartPoints =
    run.status === 'completed' && run.manifest.mode === 'live'
      ? metrics.flatMap((target) =>
          hasScoredOutcomes(target) &&
          typeof target.macro_accuracy === 'number' &&
          typeof target.cost_usd === 'number'
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
    <div className="flex min-w-0 flex-col gap-1">
      {recovery}
      {run.manifest.mode === 'preview' && (
        <Notice tone="info" as="p">
          Route preview only. This run provides routing diagnostics, not a capability score.
        </Notice>
      )}
      {run.manifest.mode === 'replay' && (
        <Notice tone="info" as="section">
          <h3>Diagnostic replay estimates</h3>
          <p>
            These estimates reuse saved answers. No new inference occurred. They are not a measured
            capability score, latency or cost-saving result.
          </p>
          <TableFrame>
            <table className={benchTable}>
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
          </TableFrame>
        </Notice>
      )}
      <AccountingCorrection report={report} />
      <h3 id="run-targets" className={`${benchText.h3} mt-3`}>
        Target comparison
      </h3>
      <p className={benchText.muted}>
        Costs apply frozen per-token prices to recorded usage; they are not invoice or hardware-cost
        measurements.
      </p>
      {metrics.length ? (
        <MetricsTable
          manifest={run.manifest}
          targets={metrics}
          preview={run.manifest.mode === 'preview'}
          running={active(run.status)}
          summary
        />
      ) : reportRead.loading ? (
        <BenchLoading label="Loading report metrics…" lines={3} />
      ) : reportRead.error ? (
        <Notice tone="danger" as="div">
          <p>Report metrics are unavailable.</p>
          {onRetry ? (
            <div>
              <Button variant="outline" size="xs" onClick={onRetry}>
                Try again
              </Button>
            </div>
          ) : null}
        </Notice>
      ) : (
        <p className={`${benchText.muted} my-3`}>
          Summary metrics will appear as results are persisted.
        </p>
      )}
      <p className={benchText.muted}>
        Scores wait for every planned outcome; failures count as incorrect. Unknown usage and cost
        remain “—”; incomplete evidence cannot establish a cost saving.
      </p>
      {report && benchmarkIDs.length > 0 && run.manifest.mode !== 'preview' && (
        <>
          <h3 className={`${benchText.h3} mt-4`}>Accuracy by benchmark</h3>
          <BenchmarkMatrix run={run} report={report} benchmarks={benchmarkIDs} />
          <Disclosure summary="Benchmark results">
            <div className="flex flex-col gap-2">
              {benchmarkIDs.map((benchmark) => (
                <div key={benchmark} className="min-w-0">
                  <h4 className={`${benchText.h4} font-mono text-xs`}>{benchmark}</h4>
                  <MetricsTable
                    manifest={run.manifest}
                    targets={
                      report.benchmarks.filter(
                        (row) => row.benchmark === benchmark,
                      ) as TargetMetrics[]
                    }
                    preview={false}
                    running={active(run.status)}
                  />
                </div>
              ))}
            </div>
          </Disclosure>
        </>
      )}
      <DatasetExposure
        benchmarks={report?.provenance.dataset?.benchmarks ?? run.manifest.dataset?.benchmarks}
        preparation={report?.provenance.dataset?.preparation ?? run.manifest.dataset?.preparation}
      />
      {visible && report && run.status === 'completed' && run.manifest.mode === 'live' && (
        <QualityCostChart points={chartPoints} costBasis="subject" />
      )}
      {run.manifest.mode === 'live' && report && (
        <OutputDiagnostics targets={report.summary.targets} manifest={run.manifest} />
      )}
      {(report || metrics.length > 0) && (
        <div className="mt-4 grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-2">
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
          {metrics.some((target) => (target.selection_statuses?.execution_required ?? 0) > 0) && (
            <Notice tone="warning" as="p">
              Some model selections require live execution. Preview can identify the matched
              decision, but does not determine the model for these cases.
            </Notice>
          )}
          <div className="mt-4 grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-2">
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
      <Disclosure summary="Token buckets and evaluation overhead">
        <TableFrame className="my-0">
          <table className={benchTable}>
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
        </TableFrame>
      </Disclosure>
      <Disclosure summary="Cost and timing interpretation">
        <p className={benchText.muted}>
          “—” means unrecorded or unknown, not zero. Cost is reported model usage; simulator and
          judge overhead remain separate in the report. Wall time is elapsed run time, not the sum
          of request durations.
        </p>
        <p className={`${benchText.muted} mt-2`}>
          Cache-neutral estimates reprice every prompt token at the frozen fresh-input rate plus
          output. This counterfactual excludes cache discounts and premiums; it is neither billed
          spend nor a measured cache-free run.
        </p>
      </Disclosure>
      {!!report?.limitations.length && (
        <Notice tone="neutral" as="aside">
          <p>{report.limitations[0]}</p>
        </Notice>
      )}
    </div>
  )
}
