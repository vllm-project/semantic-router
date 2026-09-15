import type { EvaluationCompareModel } from './evaluationCompareModel'
import {
  changeProfileLabel,
  runCohortTargetLabel,
  runWorkloadLabel,
} from './evaluationRunPresentation'
import styles from './EvaluationCompare.module.css'
import heroStyles from './EvaluationReportHero.module.css'

export default function EvaluationCompareCohort({
  model,
  runLedgerComplete,
}: {
  model: EvaluationCompareModel
  runLedgerComplete: boolean
}) {
  return (
    <>
      {runLedgerComplete && model.baseline && model.candidate ? (
        <>
          <dl className={styles.comparabilityStrip} aria-label="Comparison settings">
            <div>
              <dt>Profile</dt>
              <dd>{changeProfileLabel(model.candidate.change_profile)}</dd>
            </div>
            <div>
              <dt>Mixture</dt>
              <dd>{runCohortTargetLabel(model.candidate)}</dd>
            </div>
            <div>
              <dt>Workload</dt>
              <dd>{runWorkloadLabel(model.candidate)}</dd>
            </div>
            <div>
              <dt>Repeatability</dt>
              <dd>{model.candidate.seed}</dd>
            </div>
          </dl>
          {model.routingRecipeAggregateBoundary ? (
            <p className={styles.routingComparisonBoundary} role="note">
              <strong>Routing details stay in each run report.</strong> Open the two reports to
              review decision coverage, calibration, top-choice coverage, and the quality gap to the
              best model.
            </p>
          ) : null}
        </>
      ) : null}
      {runLedgerComplete && model.lineageMismatch ? (
        <div className={heroStyles.error} role="alert">
          The selected candidate is not pinned to this baseline.
        </div>
      ) : null}
      {runLedgerComplete && model.mismatches.length ? (
        <div className={heroStyles.error} role="alert">
          These runs cannot be compared yet. Align their {model.mismatches.join(', ')}.
        </div>
      ) : null}
    </>
  )
}
