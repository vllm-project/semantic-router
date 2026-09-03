import type { EvaluationReport } from '../../types/evaluationReport'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import overviewStyles from './EvaluationOverview.module.css'
import { evaluationMetricLabel } from './evaluationMetricPresentation'
import type { EvaluationOverviewModel } from './evaluationOverview'
import { evaluationResultScopeLabel, formatMetric } from './evaluationPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import { EvaluationActionButton, RunStatusBadge } from './EvaluationPrimitives'
import styles from './EvaluationPlane.module.css'

interface EvaluationLatestEvidenceProps {
  model: EvaluationOverviewModel
  latestReport: EvaluationReport | null
  reportLoading: boolean
  reportError: string | null
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  onRetryReport: () => void
  onLoadMoreRuns: () => void
  onOpenReport: (runID: string) => void
}

function LatestEvidenceHeader({
  model,
  latestReport,
  reportLoading,
  runLedgerAvailable,
  runLedgerComplete,
  onOpenReport,
}: Pick<
  EvaluationLatestEvidenceProps,
  | 'model'
  | 'latestReport'
  | 'reportLoading'
  | 'runLedgerAvailable'
  | 'runLedgerComplete'
  | 'onOpenReport'
>) {
  return (
    <header className={styles.surfaceHeader}>
      <div>
        <span className={styles.eyebrow}>
          {runLedgerComplete ? 'Latest completed report' : 'Latest available report'}
        </span>
        <h2 id="latest-evidence-title">{model.latestEvidenceName || 'No completed report yet'}</h2>
        <p>
          {!runLedgerAvailable
            ? 'Run history must load before the newest completed report can be selected.'
            : reportLoading && !latestReport
              ? 'Loading verified headline results.'
              : 'Headline results are verified by the evaluation service. Open the full report for every measured outcome.'}
        </p>
      </div>
      {latestReport ? (
        <EvaluationActionButton
          type="button"
          variant="quiet"
          onClick={() => onOpenReport(latestReport.run.id)}
        >
          Open full report
        </EvaluationActionButton>
      ) : null}
    </header>
  )
}

function LatestEvidenceError({
  reportError,
  onRetryReport,
}: Pick<EvaluationLatestEvidenceProps, 'reportError' | 'onRetryReport'>) {
  if (!reportError) return null
  return (
    <div className={styles.inlineError} role="alert">
      <div>
        <strong>Latest report could not be refreshed.</strong>
        <span>Retry to load the newest verified headline results.</span>
        <EvaluationIssueDetails
          issues={[{ label: 'Latest report request', message: reportError }]}
        />
      </div>
      <EvaluationActionButton type="button" compact onClick={onRetryReport}>
        Retry
      </EvaluationActionButton>
    </div>
  )
}

function LatestEvidenceHeadlines({
  model,
  latestReport,
}: Pick<EvaluationLatestEvidenceProps, 'model'> & { latestReport: EvaluationReport }) {
  if (!model.headlines.length) {
    return (
      <div className={styles.scopeNotice}>
        <strong>
          {model.isDiagnostic
            ? 'Diagnostic run — no release recommendation'
            : 'No measured headline applies to this run'}
        </strong>
        <span>
          {model.isDiagnostic
            ? 'This run validates the evaluation setup and execution path. Open the full report to review its observations and incomplete release checks.'
            : 'Open the full report for measured outcomes and evaluation coverage.'}
        </span>
      </div>
    )
  }
  return (
    <dl className={overviewStyles.headlineStrip}>
      {model.headlines.map((metric) => (
        <div key={`${metric.track_id || 'system'}-${metric.id}`}>
          <dt>{evaluationMetricLabel(metric)}</dt>
          <dd>{formatMetric(metric)}</dd>
          <span>
            {evaluationResultScopeLabel(latestReport.run.evidence_level)} ·{' '}
            {metric.track_id ? TRACK_PRESENTATION[metric.track_id].label : 'System'}
          </span>
        </div>
      ))}
    </dl>
  )
}

function LatestEvidenceEmpty({
  model,
  runLedgerAvailable,
  hasMoreRuns,
  loadingMoreRuns,
  onLoadMoreRuns,
}: Pick<
  EvaluationLatestEvidenceProps,
  'model' | 'runLedgerAvailable' | 'hasMoreRuns' | 'loadingMoreRuns' | 'onLoadMoreRuns'
>) {
  return (
    <div className={styles.emptyState}>
      <p>
        {!runLedgerAvailable
          ? 'Retry run history to discover completed reports.'
          : hasMoreRuns
            ? 'No completed report is present in the loaded runs. Load older runs to continue searching.'
            : 'Complete a run to establish a report.'}
      </p>
      {runLedgerAvailable && hasMoreRuns ? (
        <EvaluationActionButton
          type="button"
          compact
          disabled={loadingMoreRuns}
          onClick={onLoadMoreRuns}
        >
          {loadingMoreRuns ? 'Loading older runs…' : 'Load older runs'}
        </EvaluationActionButton>
      ) : null}
      {model.latestRun ? <RunStatusBadge status={model.latestRun.status} /> : null}
    </div>
  )
}

export default function EvaluationLatestEvidence(props: EvaluationLatestEvidenceProps) {
  const { reportLoading, reportError, latestReport } = props
  return (
    <section className={styles.surface} aria-labelledby="latest-evidence-title">
      <LatestEvidenceHeader {...props} />
      {reportLoading ? <p className={styles.emptyCopy}>Loading report summary…</p> : null}
      <LatestEvidenceError {...props} />
      {!reportLoading && !reportError && latestReport ? (
        <LatestEvidenceHeadlines model={props.model} latestReport={latestReport} />
      ) : null}
      {!reportLoading && !reportError && !latestReport ? <LatestEvidenceEmpty {...props} /> : null}
    </section>
  )
}
