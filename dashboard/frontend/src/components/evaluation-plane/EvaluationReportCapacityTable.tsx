import type { EvaluationCapacityLevel } from '../../types/evaluationCapacityReport'
import EvaluationDisclosure from './EvaluationDisclosure'
import { formatMetric } from './evaluationPresentation'
import { EvaluationTag } from './EvaluationPrimitives'
import styles from './EvaluationReportDiagnostics.module.css'
import tableStyles from './EvaluationReportTable.module.css'

function capacityServiceChecks(level: EvaluationCapacityLevel) {
  return [
    ['Warmup requests', level.warmup_passed],
    ['Response-time target', level.latency_slo_passed],
    ['Independent-window coverage', level.cluster_coverage_passed],
    ['Error-rate stability', level.error_rate_stability_passed],
    ['Error-rate target', level.error_slo_passed],
    ['Throughput target', level.throughput_slo_passed],
    ['Scaling efficiency', level.scaling_slo_passed],
    ['Throughput stability', level.throughput_stability_passed],
    ['Response-time stability', level.latency_stability_passed],
  ] as const
}

function CapacityServiceChecks({ level }: { level: EvaluationCapacityLevel }) {
  return (
    <ul
      className={styles.capacityChecks}
      aria-label={`Service checks at concurrency ${level.concurrency}`}
    >
      {capacityServiceChecks(level).map(([label, passed]) => (
        <li
          key={String(label)}
          data-passed={passed}
          aria-label={`${label}: ${passed ? 'passed' : 'failed'}`}
        >
          <span aria-hidden="true">{passed ? '✓' : '×'}</span>
          <span>{label}</span>
        </li>
      ))}
    </ul>
  )
}

function CapacityRepetitions({ level }: { level: EvaluationCapacityLevel }) {
  return (
    <EvaluationDisclosure
      className={styles.repetitionDisclosure}
      summary={`${level.repetitions.length} independent windows`}
      summaryClassName={styles.repetitionDisclosureSummary}
    >
      <ol>
        {level.repetitions.map((repetition) => (
          <li key={repetition.repetition}>
            r{repetition.repetition}: {repetition.successes}/{repetition.requests} ok ·{' '}
            {(repetition.error_rate * 100).toFixed(2)}% errors /{' '}
            {(repetition.error_rate_upper_bound * 100).toFixed(2)}% upper bound ·{' '}
            {formatMetric({ value: repetition.throughput_rps, unit: 'requests/s' })} · p95{' '}
            {formatMetric({ value: repetition.latency_p95_ms, unit: 'ms' })}
          </li>
        ))}
      </ol>
    </EvaluationDisclosure>
  )
}

function CapacityLevelPerformanceCells({ level }: { level: EvaluationCapacityLevel }) {
  return (
    <>
      <td>
        <EvaluationTag tone={level.qualified ? 'positive' : 'warning'}>
          {level.qualified ? 'Within target' : 'Outside target'}
        </EvaluationTag>
      </td>
      <td>
        {level.warmup_requests} requests · {level.warmup_errors} errors ·{' '}
        {formatMetric({ value: level.warmup_elapsed_seconds, unit: 's' })}
      </td>
      <td>
        {level.successes}/{level.measurement_requests} successful
      </td>
      <td>
        {level.errors} · {(level.error_rate * 100).toFixed(2)}% cluster mean /{' '}
        {(level.error_rate_upper_bound * 100).toFixed(2)}% worst-cluster upper bound ·{' '}
        {(level.error_rate_cluster_range * 100).toFixed(2)}% spread across{' '}
        {level.measurement_cluster_count} windows
      </td>
      <td>
        {formatMetric({ value: level.throughput_rps, unit: 'requests/s' })} /{' '}
        {(level.throughput_cv * 100).toFixed(1)}%
      </td>
      <td>
        {level.throughput_scaling_efficiency === null
          ? 'Baseline'
          : `${(level.throughput_scaling_efficiency * 100).toFixed(1)}%`}
      </td>
      <td>
        {formatMetric({ value: level.latency_p50_ms, unit: 'ms' })} /{' '}
        {formatMetric({ value: level.latency_p95_ms, unit: 'ms' })} /{' '}
        {formatMetric({ value: level.latency_p99_ms, unit: 'ms' })} /{' '}
        {(level.latency_p95_cv * 100).toFixed(1)}%
      </td>
    </>
  )
}

function CapacityLevelEvidenceCells({ level }: { level: EvaluationCapacityLevel }) {
  return (
    <>
      <td>
        <CapacityServiceChecks level={level} />
      </td>
      <td>
        <CapacityRepetitions level={level} />
      </td>
      <td>
        {level.input_tokens} / {level.output_tokens}
      </td>
      <td>{formatMetric({ value: level.elapsed_seconds, unit: 's' })}</td>
      <td>{formatMetric({ value: level.runtime_cost_usd, unit: 'usd' })}</td>
    </>
  )
}

function CapacityLevelRow({ level }: { level: EvaluationCapacityLevel }) {
  return (
    <tr>
      <th scope="row">{level.concurrency}</th>
      <CapacityLevelPerformanceCells level={level} />
      <CapacityLevelEvidenceCells level={level} />
    </tr>
  )
}

function CapacityTableHeader() {
  return (
    <thead>
      <tr>
        <th scope="col">Concurrency</th>
        <th scope="col">Envelope</th>
        <th scope="col">Warmup</th>
        <th scope="col">Measurement</th>
        <th scope="col">Errors / upper confidence estimate</th>
        <th scope="col">Throughput / variation</th>
        <th scope="col">Scaling</th>
        <th scope="col">Latency p50 / p95 / p99 / p95 variation</th>
        <th scope="col">Service checks</th>
        <th scope="col">Repetitions</th>
        <th scope="col">Tokens in / out</th>
        <th scope="col">Duration</th>
        <th scope="col">Runtime cost</th>
      </tr>
    </thead>
  )
}

export default function EvaluationReportCapacityTable({
  levels,
}: {
  levels: EvaluationCapacityLevel[]
}) {
  return (
    <div
      className={tableStyles.tableScroll}
      role="region"
      tabIndex={0}
      aria-label="Scrollable capacity envelope observations"
    >
      <table className={tableStyles.table}>
        <caption>Capacity observations and service-objective decisions by concurrency</caption>
        <CapacityTableHeader />
        <tbody>
          {levels.map((level) => (
            <CapacityLevelRow key={level.concurrency} level={level} />
          ))}
        </tbody>
      </table>
    </div>
  )
}
