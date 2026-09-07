import type { EvaluationCompareModel } from './evaluationCompareModel'
import { EvaluationActionButton } from './EvaluationPrimitives'
import { comparisonRunOptionLabels } from './evaluationRunPresentation'
import styles from './EvaluationCompare.module.css'
import reportStyles from './EvaluationReportLayout.module.css'

interface EvaluationCompareSelectionProps {
  model: EvaluationCompareModel
  candidateID: string
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  resourcesLoading: boolean
  loading: boolean
  onChooseCandidate: (id: string) => void
  onCompare: () => void
}

export default function EvaluationCompareSelection({
  model,
  candidateID,
  runLedgerAvailable,
  runLedgerComplete,
  resourcesLoading,
  loading,
  onChooseCandidate,
  onCompare,
}: EvaluationCompareSelectionProps) {
  return (
    <section className={styles.compareHero}>
      <div className={styles.compareIntro}>
        <span className={reportStyles.eyebrow}>Compare runs</span>
        <h2>Compare a candidate with its baseline</h2>
        <p>
          Choose a candidate created from a baseline. We only compare compatible workloads and
          configurations so the difference is meaningful.
        </p>
      </div>
      {resourcesLoading ? (
        <p className={styles.compareLoading} role="status">
          Loading compatible runs…
        </p>
      ) : model.candidates.length > 0 ? (
        <div className={styles.compareControls}>
          <div className={styles.compareFields}>
            <label>
              <span>Candidate run</span>
              <select
                aria-label="Comparison candidate"
                value={candidateID}
                disabled={
                  !runLedgerAvailable ||
                  !runLedgerComplete ||
                  loading ||
                  resourcesLoading ||
                  model.candidates.length === 0
                }
                onChange={(event) => onChooseCandidate(event.target.value)}
              >
                <option value="">Choose a compatible candidate</option>
                {model.candidates.map((run) => (
                  <option key={run.id} value={run.id}>
                    {model.candidateLabels.get(run.id)}
                  </option>
                ))}
              </select>
            </label>
            <div className={styles.baselineField} aria-live="polite">
              <span>Baseline run</span>
              <strong>
                {model.baseline
                  ? comparisonRunOptionLabels([model.baseline]).get(model.baseline.id)
                  : 'Selected with candidate'}
              </strong>
            </div>
          </div>
          <div className={styles.compareFooter}>
            <p>The matching baseline is selected automatically from the candidate run.</p>
            <EvaluationActionButton
              type="button"
              variant="primary"
              disabled={model.invalidPair || loading}
              onClick={onCompare}
            >
              {resourcesLoading ? 'Loading runs…' : loading ? 'Comparing…' : 'Compare results'}
            </EvaluationActionButton>
          </div>
        </div>
      ) : null}
    </section>
  )
}
