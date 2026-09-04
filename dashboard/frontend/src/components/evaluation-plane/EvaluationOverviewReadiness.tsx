import type { EvaluationView } from '../../pages/evaluationRoute'
import type { EvaluationRun } from '../../types/evaluationPlane'
import { formatDateTime } from '../../utils/dateTime'
import styles from './EvaluationOverview.module.css'
import type { EvaluationOverviewModel } from './evaluationOverview'
import { EvaluationActionButton } from './EvaluationPrimitives'
import planeStyles from './EvaluationPlane.module.css'

interface EvaluationOverviewReadinessProps {
  model: EvaluationOverviewModel
  runs: EvaluationRun[]
  totalRuns: number
  hasMoreRuns: boolean
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  reportLoading: boolean
  onNavigate: (view: EvaluationView) => void
}

export default function EvaluationOverviewReadiness({
  model,
  runs,
  totalRuns,
  hasMoreRuns,
  runLedgerAvailable,
  runLedgerComplete,
  reportLoading,
  onNavigate,
}: EvaluationOverviewReadinessProps) {
  return (
    <>
      <section className={styles.readiness} aria-labelledby="evaluation-readiness-title">
        <div className={styles.readinessCopy}>
          <span className={planeStyles.eyebrow}>Latest decision</span>
          <h2 id="evaluation-readiness-title">
            {model.latestEvidenceName || 'Establish the first evaluation baseline'}
          </h2>
          <p>
            {model.hasLatestReport
              ? model.isDiagnostic
                ? 'This run is useful for exploration, but it is not ready to support a release decision. Run a qualified benchmark or live evaluation before changing production.'
                : 'Review blocked checks and measured outcomes before changing the production recipe or model pool.'
              : !runLedgerAvailable
                ? 'Run history is unavailable. Retry before selecting a baseline or drawing conclusions from earlier results.'
                : reportLoading && model.hasRequestedReportRun
                  ? 'Loading the newest completed report. No decision is shown until the result is ready.'
                  : 'Create a replay or live evaluation to measure the change before it reaches production.'}
          </p>
        </div>
        <div className={styles.readinessActions}>
          <EvaluationActionButton type="button" variant="primary" onClick={() => onNavigate('new')}>
            New experiment
          </EvaluationActionButton>
          <EvaluationActionButton type="button" onClick={() => onNavigate('runs')}>
            Inspect runs
          </EvaluationActionButton>
        </div>
      </section>

      <dl className={styles.statusStrip} aria-label="Evaluation status">
        <div>
          <dt>
            {hasMoreRuns
              ? 'Loaded runs'
              : !runLedgerAvailable
                ? 'Run history'
                : runLedgerComplete
                  ? 'Runs'
                  : 'Visible runs'}
          </dt>
          <dd>
            {!runLedgerAvailable ? '—' : hasMoreRuns ? `${runs.length}/${totalRuns}` : runs.length}
          </dd>
          <span>
            {!runLedgerAvailable
              ? 'Not loaded'
              : runLedgerComplete
                ? `${model.running} active loaded`
                : 'History incomplete'}
          </span>
        </div>
        <div>
          <dt>{hasMoreRuns ? 'Completed loaded' : 'Completed'}</dt>
          <dd>{runLedgerAvailable ? model.completed : '—'}</dd>
          <span>
            {!runLedgerAvailable
              ? 'Run history unavailable'
              : model.latestRun
                ? formatDateTime(model.latestRun.created_at)
                : 'No history yet'}
          </span>
        </div>
        <div>
          <dt>{hasMoreRuns ? 'Failures loaded' : 'Failures'}</dt>
          <dd>{runLedgerAvailable ? model.failed : '—'}</dd>
          <span>{hasMoreRuns ? 'Among loaded runs' : 'Execution failures only'}</span>
        </div>
        <div>
          <dt>Release blockers</dt>
          <dd>{model.hasLatestReport ? model.requiredBlockers : '—'}</dd>
          <span>Failed or incomplete checks</span>
        </div>
      </dl>
    </>
  )
}
