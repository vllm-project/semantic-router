import { cn } from '../../ui'
import { number, percent } from './model'
import { targetName } from './targetPresentation'
import type { Manifest, TargetMetrics } from './types'
import { Disclosure } from './benchUi'
import { benchText } from './benchStyles'

const finishReasonLabels: Record<string, string> = {
  stop: 'Stop',
  length: 'Output limit',
  tool_calls: 'Tool calls',
  function_call: 'Function call',
}

function Rate({ count, total }: { count: number; total: number }) {
  return (
    <>
      <strong className="font-semibold">{number(count)}</strong> / {number(total)}
      {total > 0 && (
        <span className="ml-1.5 text-xs font-normal text-muted-foreground">
          ({percent(count / total)})
        </span>
      )}
    </>
  )
}

const metricLabel = 'text-xs text-muted-foreground'
const metricNote = 'block text-xs font-normal text-muted-foreground'

export default function OutputDiagnostics({
  targets,
  manifest,
}: {
  targets: TargetMetrics[]
  manifest: Manifest
}) {
  return (
    <section className="my-5 min-w-0" aria-label="Output diagnostics">
      <h3 className={benchText.h3}>Output diagnostics</h3>
      <p className={benchText.muted}>
        Full-report counts. Format compliance is separate from answer correctness.
      </p>
      <ul className="mt-3 divide-y divide-border border-y border-border">
        {targets.map((target) => {
          const diagnostics = target.output_diagnostics
          const model = targetName(manifest, target.id)
          return (
            <li
              key={target.id}
              className="grid min-w-0 grid-cols-1 items-start gap-3 py-3 sm:grid-cols-[minmax(150px,1fr)_minmax(300px,2fr)] sm:gap-4"
              aria-label={model}
            >
              <div className="grid gap-1 break-words text-[13px]">
                <strong className="font-medium text-foreground">{model}</strong>
                {diagnostics && diagnostics.result_cases < diagnostics.planned_cases && (
                  <span className="text-xs text-muted-foreground">
                    {number(diagnostics.result_cases)} / {number(diagnostics.planned_cases)} case
                    results recorded
                  </span>
                )}
              </div>
              {diagnostics ? (
                <dl className="m-0 grid grid-cols-2 gap-3 [&_dd]:mt-0.5 [&_dd]:tabular-nums">
                  <div>
                    <dt className={metricLabel}>Recorded output-limit cases</dt>
                    <dd
                      className={cn(
                        'text-[13px] text-foreground',
                        diagnostics.output_limit_cases > 0 && '[&>strong]:text-warning-foreground',
                      )}
                    >
                      <Rate
                        count={diagnostics.output_limit_cases}
                        total={diagnostics.planned_cases}
                      />
                      <small className={metricNote}>planned cases</small>
                    </dd>
                  </div>
                  <div>
                    <dt className={metricLabel}>Answer-format failures</dt>
                    <dd
                      className={cn(
                        'text-[13px] text-foreground',
                        diagnostics.strict_format.failed_cases > 0 &&
                          '[&>strong]:text-warning-foreground',
                      )}
                    >
                      {diagnostics.strict_format.checked_cases > 0 ? (
                        <>
                          <Rate
                            count={diagnostics.strict_format.failed_cases}
                            total={diagnostics.strict_format.checked_cases}
                          />
                          <small className={metricNote}>format-checked cases</small>
                        </>
                      ) : (
                        <span className={metricNote}>Not assessed</span>
                      )}
                      {diagnostics.strict_format.unassessed_cases > 0 && (
                        <small className={metricNote}>
                          {number(diagnostics.strict_format.unassessed_cases)} planned{' '}
                          {diagnostics.strict_format.unassessed_cases === 1 ? 'case' : 'cases'} not
                          assessed
                        </small>
                      )}
                    </dd>
                  </div>
                </dl>
              ) : (
                <p className={benchText.muted}>Output diagnostics are unavailable.</p>
              )}
            </li>
          )
        })}
      </ul>
      <Disclosure summary="How subject calls ended" bare className="mt-2">
        <p className={benchText.muted}>
          Counts are subject model calls, which can exceed the number of cases. Stop does not mean
          correct; tool calls can continue a task. Missing finish reasons remain unknown.
        </p>
        <ul className="mt-3 grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-3 text-[13px]">
          {targets.map((target) => {
            const calls = target.output_diagnostics?.subject_calls
            return (
              <li
                key={target.id}
                className="min-w-0 break-words rounded-lg border border-border bg-surface px-3 py-2.5"
              >
                <strong className="font-medium text-foreground">
                  {targetName(manifest, target.id)}
                </strong>
                {calls ? (
                  <>
                    <span className={metricNote}>{number(calls.total)} recorded subject calls</span>
                    {calls.total > 0 && (
                      <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs [&_dd]:tabular-nums [&_dt]:text-muted-foreground [&>div]:flex [&>div]:gap-1.5">
                        {Object.entries(calls.finish_reasons).map(([reason, count]) => (
                          <div key={reason}>
                            <dt title={reason}>{finishReasonLabels[reason] ?? reason}</dt>
                            <dd>
                              {number(count)} / {number(calls.total)}
                            </dd>
                          </div>
                        ))}
                        <div>
                          <dt>Unknown</dt>
                          <dd>
                            {number(calls.unknown_finish_reason)} / {number(calls.total)}
                          </dd>
                        </div>
                      </dl>
                    )}
                  </>
                ) : (
                  <span className={metricNote}>Not available</span>
                )}
              </li>
            )
          })}
        </ul>
      </Disclosure>
    </section>
  )
}
