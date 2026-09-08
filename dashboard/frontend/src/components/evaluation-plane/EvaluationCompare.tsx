import type { EvaluationRun } from '../../types/evaluationPlane'
import type { EvaluationComparison } from '../../types/evaluationComparison'
import EvaluationCompareAvailability from './EvaluationCompareAvailability'
import EvaluationCompareCohort from './EvaluationCompareCohort'
import { buildEvaluationCompareModel } from './evaluationCompareModel'
import EvaluationCompareSelection from './EvaluationCompareSelection'
import EvaluationComparisonResults from './EvaluationComparisonResults'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import heroStyles from './EvaluationReportHero.module.css'
import reportStyles from './EvaluationReportLayout.module.css'

interface EvaluationCompareProps {
  runs: EvaluationRun[]
  baselineID: string
  candidateID: string
  comparison: EvaluationComparison | null
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  totalRuns: number
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  resourcesLoading: boolean
  resourcesError: string | null
  loading: boolean
  error: string | null
  onPairChange: (candidateID: string, baselineID: string) => void
  onCompare: () => void
  onLoadMoreRuns: () => void
  onRetryResources: () => void
  onCreateRun?: () => void
}

export default function EvaluationCompare(props: EvaluationCompareProps) {
  const model = buildEvaluationCompareModel(props)
  const chooseCandidate = (id: string) => {
    const next = model.completed.get(id)
    props.onPairChange(id, next?.baseline_run_id || '')
  }
  const showPrompt =
    props.runLedgerAvailable &&
    props.runLedgerComplete &&
    !props.resourcesLoading &&
    !props.resourcesError &&
    !props.comparison &&
    !props.error &&
    model.candidates.length > 0

  return (
    <div className={reportStyles.report} aria-busy={props.loading}>
      <EvaluationCompareSelection
        model={model}
        candidateID={props.candidateID}
        runLedgerAvailable={props.runLedgerAvailable}
        runLedgerComplete={props.runLedgerComplete}
        resourcesLoading={props.resourcesLoading}
        loading={props.loading}
        onChooseCandidate={chooseCandidate}
        onCompare={props.onCompare}
      />
      <EvaluationCompareAvailability
        model={model}
        runsLoaded={props.runs.length}
        totalRuns={props.totalRuns}
        runLedgerAvailable={props.runLedgerAvailable}
        runLedgerComplete={props.runLedgerComplete}
        hasMoreRuns={props.hasMoreRuns}
        loadingMoreRuns={props.loadingMoreRuns}
        resourcesLoading={props.resourcesLoading}
        resourcesError={props.resourcesError}
        onLoadMoreRuns={props.onLoadMoreRuns}
        onRetryResources={props.onRetryResources}
        onCreateRun={props.onCreateRun}
      />
      <EvaluationCompareCohort model={model} runLedgerComplete={props.runLedgerComplete} />
      {props.error ? (
        <div className={heroStyles.error} role="alert">
          <span>The comparison could not be calculated. Retry after checking both runs.</span>
          <EvaluationIssueDetails
            issues={[{ label: 'Comparison service', message: props.error }]}
          />
        </div>
      ) : null}
      {showPrompt ? (
        <div className={reportStyles.empty}>
          Choose a candidate, then calculate its paired comparison.
        </div>
      ) : null}
      {props.runLedgerComplete && props.comparison && model.comparisonVerdict ? (
        <EvaluationComparisonResults
          comparison={props.comparison}
          verdict={model.comparisonVerdict}
          evidenceLevel={model.candidate?.evidence_level}
        />
      ) : null}
    </div>
  )
}
