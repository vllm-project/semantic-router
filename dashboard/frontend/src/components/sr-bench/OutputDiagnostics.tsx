import ProductIcon from '../ProductIcon'
import { number, percent } from './model'
import { targetName } from './targetPresentation'
import type { Manifest, TargetMetrics } from './types'
import styles from './OutputDiagnostics.module.css'

const finishReasonLabels: Record<string, string> = {
  stop: 'Stop',
  length: 'Output limit',
  tool_calls: 'Tool calls',
  function_call: 'Function call',
}

function Rate({ count, total }: { count: number; total: number }) {
  return (
    <>
      <strong>{number(count)}</strong> / {number(total)}
      {total > 0 && <span className={styles.rate}>({percent(count / total)})</span>}
    </>
  )
}

export default function OutputDiagnostics({
  targets,
  manifest,
}: {
  targets: TargetMetrics[]
  manifest: Manifest
}) {
  return (
    <section className={styles.section} aria-label="Output diagnostics">
      <h3>Output diagnostics</h3>
      <p className={styles.description}>
        Full-report counts. Format compliance is separate from answer correctness.
      </p>
      <ul className={styles.rows}>
        {targets.map((target) => {
          const diagnostics = target.output_diagnostics
          const model = targetName(manifest, target.id)
          return (
            <li key={target.id} className={styles.row} aria-label={model}>
              <div className={styles.model}>
                <strong>{model}</strong>
                {diagnostics && diagnostics.result_cases < diagnostics.planned_cases && (
                  <span>
                    {number(diagnostics.result_cases)} / {number(diagnostics.planned_cases)} case
                    results recorded
                  </span>
                )}
              </div>
              {diagnostics ? (
                <dl className={styles.metrics}>
                  <div>
                    <dt>Recorded output-limit cases</dt>
                    <dd
                      className={diagnostics.output_limit_cases > 0 ? styles.affected : undefined}
                    >
                      <Rate
                        count={diagnostics.output_limit_cases}
                        total={diagnostics.planned_cases}
                      />
                      <small>planned cases</small>
                    </dd>
                  </div>
                  <div>
                    <dt>Answer-format failures</dt>
                    <dd
                      className={
                        diagnostics.strict_format.failed_cases > 0 ? styles.affected : undefined
                      }
                    >
                      {diagnostics.strict_format.checked_cases > 0 ? (
                        <>
                          <Rate
                            count={diagnostics.strict_format.failed_cases}
                            total={diagnostics.strict_format.checked_cases}
                          />
                          <small>format-checked cases</small>
                        </>
                      ) : (
                        <span className={styles.unassessed}>Not assessed</span>
                      )}
                      {diagnostics.strict_format.unassessed_cases > 0 && (
                        <small>
                          {number(diagnostics.strict_format.unassessed_cases)} planned{' '}
                          {diagnostics.strict_format.unassessed_cases === 1 ? 'case' : 'cases'} not
                          assessed
                        </small>
                      )}
                    </dd>
                  </div>
                </dl>
              ) : (
                <p className={styles.description}>Output diagnostics are unavailable.</p>
              )}
            </li>
          )
        })}
      </ul>
      <details className={styles.finishReasons}>
        <summary>
          <ProductIcon name="chevron-right" width={14} height={14} />
          How subject calls ended
        </summary>
        <p className={styles.description}>
          Counts are subject model calls, which can exceed the number of cases. Stop does not mean
          correct; tool calls can continue a task. Missing finish reasons remain unknown.
        </p>
        <ul className={styles.callRows}>
          {targets.map((target) => {
            const calls = target.output_diagnostics?.subject_calls
            return (
              <li key={target.id}>
                <strong>{targetName(manifest, target.id)}</strong>
                {calls ? (
                  <>
                    <span className={styles.callTotal}>
                      {number(calls.total)} recorded subject calls
                    </span>
                    {calls.total > 0 && (
                      <dl className={styles.reasonCounts}>
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
                  <span className={styles.unassessed}>Not available</span>
                )}
              </li>
            )
          })}
        </ul>
      </details>
    </section>
  )
}
