import { EVALUATION_RUN_LIMITS } from './evaluationExperiment'
import type { EvaluationExperimentFormModel } from './useEvaluationExperimentForm'
import EvaluationExperimentSectionHeading from './EvaluationExperimentSectionHeading'
import styles from './EvaluationExperimentFields.module.css'
import sectionStyles from './EvaluationExperimentSection.module.css'

interface EvaluationExperimentBudgetProps {
  canAutoStart: boolean
  form: EvaluationExperimentFormModel
}

export default function EvaluationExperimentBudget({
  canAutoStart,
  form,
}: EvaluationExperimentBudgetProps) {
  return (
    <section className={sectionStyles.formSection}>
      <EvaluationExperimentSectionHeading
        index={form.capacitySLOActive ? '06' : '05'}
        title="Budget and reproducibility"
        description="Bound execution and keep repeated runs comparable."
      />
      <div className={styles.numericGrid}>
        <label>
          Maximum cases
          <input
            type="number"
            min={1}
            max={EVALUATION_RUN_LIMITS.sampleLimit}
            step={1}
            value={form.sampleLimit}
            disabled={form.baselineLocked}
            onChange={(event) => form.setSampleLimit(Number(event.target.value))}
          />
        </label>
        <label>
          <span>Parallel requests</span>
          <input
            type="number"
            min={form.capacitySLOActive ? 2 : 1}
            max={EVALUATION_RUN_LIMITS.concurrency}
            step={1}
            value={form.concurrency}
            disabled={form.baselineLocked}
            onChange={(event) => form.setConcurrency(Number(event.target.value))}
          />
          {form.capacitySLOActive ? (
            <small>At least two levels are required to measure scaling and saturation.</small>
          ) : null}
        </label>
        <label>
          Repeatability key
          <input
            type="number"
            min={0}
            max={EVALUATION_RUN_LIMITS.seed}
            step={1}
            value={form.seed}
            disabled={form.baselineLocked}
            onChange={(event) => form.setSeed(Number(event.target.value))}
          />
          <small>Keep this value unchanged when comparing two runs.</small>
        </label>
      </div>
      <label className={styles.autoStart}>
        <input
          type="checkbox"
          checked={form.autoStart}
          disabled={!canAutoStart}
          onChange={(event) => form.setAutoStart(event.target.checked)}
        />
        <span>
          <strong>Start immediately</strong>
          <small>
            {canAutoStart
              ? 'Create the snapshot and enqueue execution.'
              : "You don't have permission to start evaluations."}
          </small>
        </span>
      </label>
    </section>
  )
}
