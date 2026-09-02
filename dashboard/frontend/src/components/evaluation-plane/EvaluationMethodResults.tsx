import type { EvaluationMethodReport } from '../../types/evaluationMethodReport'
import type { EvaluationReport } from '../../types/evaluationReport'
import EvaluationDisclosure, { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import { EvaluationTag } from './EvaluationPrimitives'
import { evaluationResultScopeLabel } from './evaluationPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import styles from './EvaluationMethodResults.module.css'
import disclosureStyles from './EvaluationReportDisclosures.module.css'
import layoutStyles from './EvaluationReportLayout.module.css'
import tableStyles from './EvaluationReportTable.module.css'

const READINESS_COPY = {
  'native-qualified': 'Runnable and gradeable live',
  'exploratory-import': 'Exploratory import only',
  'data-required': 'Required data is missing',
  blocked: 'Not supported by this benchmark',
} as const

const PARITY_COPY: Record<EvaluationMethodReport['method']['native_parity'], string> = {
  native: 'Native benchmark method',
  source_qualified: 'Pinned-source normalized import · native parity not verified',
  none: 'vLLM Semantic Router method',
}

function formatMetric(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(4)
}

function methodDisplayName(method: EvaluationMethodReport['method']): string {
  const trackLabels = method.applicable_tracks.flatMap((track) =>
    track in TRACK_PRESENTATION
      ? [TRACK_PRESENTATION[track as keyof typeof TRACK_PRESENTATION].label]
      : [],
  )
  return trackLabels.length ? `${trackLabels.join(' + ')} benchmark analysis` : 'Benchmark analysis'
}

function methodActionLabels(result: EvaluationMethodReport): Map<string, string> {
  return new Map(
    [...new Set(result.raw_shared_domain_curve.map((point) => point.action.id))].map(
      (actionID, index) => [actionID, `Configuration ${index + 1}`],
    ),
  )
}

function MethodResultHeader({
  method,
  methodLabel,
  actionCount,
}: {
  method: EvaluationMethodReport['method']
  methodLabel: string
  actionCount: number
}) {
  const ready = method.status === 'native-qualified'
  return (
    <header className={styles.methodHeader}>
      <div>
        <strong>{methodLabel}</strong>
        <p>
          Measures outcomes across {actionCount}{' '}
          {actionCount === 1 ? 'configuration' : 'configurations'} and the recorded budget range.
          Missing observations block the result.
        </p>
      </div>
      <EvaluationTag tone={ready ? 'info' : 'warning'}>
        {READINESS_COPY[method.status]}
      </EvaluationTag>
    </header>
  )
}

function MethodFacts({ method }: { method: EvaluationMethodReport['method'] }) {
  return (
    <dl className={styles.facts}>
      <div>
        <dt>Evaluation scope</dt>
        <dd>{evaluationResultScopeLabel(method.evidence_ceiling)}</dd>
      </div>
      <div>
        <dt>Benchmark fidelity</dt>
        <dd>{PARITY_COPY[method.native_parity]}</dd>
      </div>
      <div>
        <dt>Evaluates</dt>
        <dd>
          {method.applicable_tracks
            .map((track) =>
              track in TRACK_PRESENTATION
                ? TRACK_PRESENTATION[track as keyof typeof TRACK_PRESENTATION].label
                : 'Additional evaluation area',
            )
            .join(' · ')}
        </dd>
      </div>
      <div>
        <dt>Required inputs</dt>
        <dd>
          {method.required_artifact_ids.length}{' '}
          {method.required_artifact_ids.length === 1 ? 'data file' : 'data files'}
        </dd>
      </div>
      <div>
        <dt>Outputs</dt>
        <dd>
          {method.produced_metric_ids.length}{' '}
          {method.produced_metric_ids.length === 1 ? 'reported score' : 'reported scores'}
        </dd>
      </div>
    </dl>
  )
}

function MethodSummary({ result }: { result: EvaluationMethodReport }) {
  return (
    <div className={styles.summary} aria-label="Score and budget summary">
      <span>
        <small>Score across budget</small>
        <strong>{formatMetric(result.audc)}</strong>
      </span>
      <span>
        <small>Normalized score</small>
        <strong>{formatMetric(result.nauc)}</strong>
      </span>
      <span>
        <small>Peak</small>
        <strong>{formatMetric(result.peak)}</strong>
      </span>
      <span>
        <small>End-of-budget score</small>
        <strong>{formatMetric(result.qnc)}</strong>
      </span>
    </div>
  )
}

function MethodScoreTable({
  result,
  methodLabel,
  actionLabels,
}: {
  result: EvaluationMethodReport
  methodLabel: string
  actionLabels: Map<string, string>
}) {
  return (
    <div
      className={tableStyles.tableScroll}
      role="region"
      tabIndex={0}
      aria-label={`${methodLabel} score by action and budget`}
    >
      <table className={tableStyles.table}>
        <caption>Benchmark score by action and budget for {methodLabel}</caption>
        <thead>
          <tr>
            <th scope="col">Action</th>
            <th scope="col">Budget</th>
            <th scope="col">Mean score</th>
            <th scope="col">Cases</th>
          </tr>
        </thead>
        <tbody>
          {result.raw_shared_domain_curve.map((point) => (
            <tr key={`${point.action.id}:${point.budget}`}>
              <th scope="row">{actionLabels.get(point.action.id)}</th>
              <td>{point.budget}</td>
              <td>{formatMetric(point.mean_score)}</td>
              <td>{point.case_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MethodTechnicalDetails({ result }: { result: EvaluationMethodReport }) {
  const { method } = result
  return (
    <EvaluationTechnicalDisclosure
      className={disclosureStyles.technicalNotes}
      summary="Technical details"
      summaryClassName={disclosureStyles.technicalNotesSummary}
    >
      <p>
        Method ID <code>{method.id}</code>
      </p>
      <p>
        Required artifact IDs <code>{method.required_artifact_ids.join(', ') || 'None'}</code>
      </p>
      <p>
        Metric IDs <code>{method.produced_metric_ids.join(', ') || 'None'}</code>
      </p>
      <p>
        Action IDs <code>{result.action_refs.map((action) => action.id).join(', ') || 'None'}</code>
      </p>
      <p>
        Analysis plan <code>{result.analysis_plan.id}</code> · unit{' '}
        <code>{result.analysis_plan.analysis_unit}</code> · grouping{' '}
        <code>{result.analysis_plan.cluster_unit}</code>
      </p>
    </EvaluationTechnicalDisclosure>
  )
}

function MethodResult({ result }: { result: EvaluationMethodReport }) {
  const { method } = result
  const methodLabel = methodDisplayName(method)
  const actionLabels = methodActionLabels(result)
  return (
    <article className={styles.method} aria-label={`${methodLabel} method result`}>
      <MethodResultHeader
        method={method}
        methodLabel={methodLabel}
        actionCount={actionLabels.size}
      />
      <MethodFacts method={method} />
      <MethodSummary result={result} />
      <MethodScoreTable result={result} methodLabel={methodLabel} actionLabels={actionLabels} />
      <MethodTechnicalDetails result={result} />
    </article>
  )
}

export default function EvaluationMethodResults({ report }: { report: EvaluationReport }) {
  return (
    <EvaluationDisclosure
      className={disclosureStyles.disclosure}
      data-evaluation-report-disclosure="true"
      summary={
        <>
          Benchmark analyses <span>{report.method_reports.length}</span>
        </>
      }
      summaryClassName={disclosureStyles.disclosureSummary}
    >
      <div className={disclosureStyles.disclosureBody}>
        <div className={layoutStyles.sectionHeader}>
          <div>
            <span className={layoutStyles.eyebrow}>Benchmark methods</span>
            <h3>Benchmark-specific analysis</h3>
            <p>
              Recomputed from raw run outputs to preserve each benchmark's evaluation method and
              make its scope explicit.
            </p>
          </div>
          <span>{report.method_reports.length} analyses</span>
        </div>
        {report.method_reports.length ? (
          report.method_reports.map((result) => (
            <MethodResult key={result.method.id} result={result} />
          ))
        ) : (
          <p className={layoutStyles.empty}>
            This run did not include benchmark-specific analysis.
          </p>
        )}
      </div>
    </EvaluationDisclosure>
  )
}
