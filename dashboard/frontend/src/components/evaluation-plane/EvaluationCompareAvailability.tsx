import type { EvaluationCompareModel } from './evaluationCompareModel'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { EvaluationActionButton } from './EvaluationPrimitives'
import styles from './EvaluationCompare.module.css'
import heroStyles from './EvaluationReportHero.module.css'
import reportStyles from './EvaluationReportLayout.module.css'

interface EvaluationCompareAvailabilityProps {
  model: EvaluationCompareModel
  runsLoaded: number
  totalRuns: number
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  resourcesLoading: boolean
  resourcesError: string | null
  onLoadMoreRuns: () => void
  onRetryResources: () => void
  onCreateRun?: () => void
}

export default function EvaluationCompareAvailability({
  model,
  runsLoaded,
  totalRuns,
  runLedgerAvailable,
  runLedgerComplete,
  hasMoreRuns,
  loadingMoreRuns,
  resourcesLoading,
  resourcesError,
  onLoadMoreRuns,
  onRetryResources,
  onCreateRun,
}: EvaluationCompareAvailabilityProps) {
  const showEmpty =
    runLedgerAvailable &&
    runLedgerComplete &&
    !resourcesLoading &&
    !resourcesError &&
    model.candidates.length === 0
  return (
    <>
      {!runLedgerAvailable ? (
        <div className={heroStyles.error} role="alert">
          Run history is unavailable. Retry before selecting or comparing results.
        </div>
      ) : null}
      {runLedgerAvailable && !runLedgerComplete ? (
        <div className={heroStyles.error} role="alert">
          Some saved runs could not be read. Baseline selection and comparison are paused until the
          affected results are repaired.
        </div>
      ) : null}
      {resourcesError ? (
        <div className={heroStyles.error} role="alert">
          <span>Run details could not be loaded. Retry before comparing results.</span>
          <EvaluationIssueDetails issues={[{ label: 'Run details', message: resourcesError }]} />
          <EvaluationActionButton type="button" compact onClick={onRetryResources}>
            Retry run identities
          </EvaluationActionButton>
        </div>
      ) : null}
      {runLedgerAvailable && runLedgerComplete && hasMoreRuns ? (
        <div className={styles.selectionScope} role="status">
          <span>
            Candidate selection covers {runsLoaded} of {totalRuns} loaded runs.
          </span>
          <EvaluationActionButton
            type="button"
            compact
            disabled={loadingMoreRuns}
            onClick={onLoadMoreRuns}
          >
            {loadingMoreRuns ? 'Loading older runs…' : 'Load older candidates'}
          </EvaluationActionButton>
        </div>
      ) : null}
      {showEmpty ? (
        <div className={reportStyles.emptyState}>
          <div>
            <strong>No comparable candidate exists.</strong>
            <p>Create a candidate from a completed baseline to compare matching workloads.</p>
          </div>
          {onCreateRun ? (
            <EvaluationActionButton type="button" variant="primary" onClick={onCreateRun}>
              Create candidate run
            </EvaluationActionButton>
          ) : null}
        </div>
      ) : null}
    </>
  )
}
