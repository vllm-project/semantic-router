import type {
  EvaluationCatalog,
  EvaluationCatalogTarget,
  EvaluationRun,
} from '../../types/evaluationPlane'
import { EVALUATION_RUN_LIMITS } from './evaluationExperiment'
import { baselineCohortIssue } from './evaluationExperimentValidation'
import EvaluationExperimentMixture from './EvaluationExperimentMixture'
import EvaluationExperimentSectionHeading from './EvaluationExperimentSectionHeading'
import { EvaluationActionButton } from './EvaluationPrimitives'
import { runOptionLabels } from './evaluationRunPresentation'
import { targetOptionLabels } from './evaluationTargetPresentation'
import styles from './EvaluationExperimentFields.module.css'
import sectionStyles from './EvaluationExperimentSection.module.css'
import type { EvaluationExperimentFormModel } from './useEvaluationExperimentForm'

interface EvaluationExperimentIdentityProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  totalRuns: number
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  pending: boolean
  onLoadMoreRuns: () => void
  form: EvaluationExperimentFormModel
}

function ExperimentDescriptionFields({ form }: { form: EvaluationExperimentFormModel }) {
  return (
    <>
      <label className={styles.fieldWide}>
        Experiment name
        <input
          value={form.name}
          onChange={(event) => form.setName(event.target.value)}
          placeholder="Recipe v3 vs production baseline"
          maxLength={EVALUATION_RUN_LIMITS.name}
          required
        />
      </label>
      <label className={styles.fieldWide}>
        Description
        <textarea
          value={form.description}
          onChange={(event) => form.setDescription(event.target.value)}
          placeholder="Hypothesis, expected trade-offs, and the release decision this should inform."
          rows={3}
          maxLength={EVALUATION_RUN_LIMITS.description}
        />
      </label>
    </>
  )
}

function ExperimentModeField({ form }: { form: EvaluationExperimentFormModel }) {
  return (
    <fieldset className={styles.choiceGroup}>
      <legend>Run type</legend>
      <div className={styles.choiceOptions}>
        {(['replay', 'live'] as const).map((option) => (
          <label key={option} className={styles.choiceCard}>
            <input
              type="radio"
              name="evaluation-mode"
              value={option}
              aria-label={
                option === 'replay'
                  ? 'Replay: deterministic and reproducible.'
                  : 'Live: evaluate a registered Mixture.'
              }
              checked={form.mode === option}
              disabled={form.baselineLocked}
              onChange={() => form.setMode(option)}
            />
            <strong>{option === 'replay' ? 'Replay' : 'Live'}</strong>
          </label>
        ))}
      </div>
      <small className={styles.choiceHelp}>
        {form.mode === 'replay'
          ? 'Deterministic and reproducible.'
          : 'Evaluate the configured routing recipe and model pool.'}
      </small>
    </fieldset>
  )
}

function ExperimentTargetField({
  catalog,
  form,
  selectedTarget,
}: {
  catalog: EvaluationCatalog
  form: EvaluationExperimentFormModel
  selectedTarget: EvaluationCatalogTarget | undefined
}) {
  const modeTargets = catalog.targets.filter((target) => target.modes.includes(form.mode))
  const targetLabels = targetOptionLabels(modeTargets)
  const live = form.mode === 'live'
  return (
    <label>
      {live ? 'Mixture to evaluate' : 'Evaluation source'}
      <select
        value={form.targetID}
        disabled={form.baselineLocked}
        onChange={(event) => form.setTargetID(event.target.value)}
        required
      >
        <option value="">{live ? 'Select Mixture' : 'Select source'}</option>
        {modeTargets.map((target) => (
          <option key={target.id} value={target.id} disabled={target.healthy === false}>
            {targetLabels.get(target.id)}
            {target.healthy === false ? ' · not ready' : ''}
          </option>
        ))}
      </select>
      <small>
        {selectedTarget?.description ||
          (live
            ? 'Only available Mixtures can be evaluated.'
            : 'Choose an available source for reproducible evaluation.')}
      </small>
    </label>
  )
}

function baselineGuidance(props: EvaluationExperimentIdentityProps): string {
  if (!props.runLedgerAvailable) {
    return 'Run history is unavailable. Retry before selecting a baseline.'
  }
  if (!props.runLedgerComplete) {
    return 'Baseline selection is paused until unreadable saved runs are repaired.'
  }
  if (props.form.baselineLocked) {
    return 'The comparison setup is copied and locked: change type, run type, Mixture, benchmarks, evaluation areas, sample size, parallel requests, performance goals, and repeatability key.'
  }
  return 'Selecting a baseline copies and locks the same comparison setup.'
}

function ExperimentBaselineField(props: EvaluationExperimentIdentityProps) {
  const { form, catalog } = props
  const baselineLabels = runOptionLabels(form.completedRuns)
  return (
    <div className={styles.fieldControl}>
      <label>
        Baseline run
        <select
          value={form.baselineRunID}
          disabled={props.pending || !props.runLedgerAvailable || !props.runLedgerComplete}
          onChange={(event) => form.selectBaseline(event.target.value)}
        >
          <option value="">No baseline</option>
          {form.completedRuns.map((run) => {
            const issue = baselineCohortIssue(catalog, run)
            return (
              <option key={run.id} value={run.id} disabled={Boolean(issue)}>
                {baselineLabels.get(run.id)}
                {issue ? ' · not eligible' : ''}
              </option>
            )
          })}
        </select>
      </label>
      <small role={form.baselineLocked ? 'status' : undefined}>{baselineGuidance(props)}</small>
      {props.runLedgerAvailable &&
      props.runLedgerComplete &&
      props.hasMoreRuns &&
      !form.baselineLocked ? (
        <EvaluationActionButton
          type="button"
          compact
          disabled={props.loadingMoreRuns}
          onClick={props.onLoadMoreRuns}
        >
          {props.loadingMoreRuns
            ? 'Loading older runs…'
            : `Load older baselines · ${props.runs.length}/${props.totalRuns}`}
        </EvaluationActionButton>
      ) : null}
    </div>
  )
}

export default function EvaluationExperimentIdentity(props: EvaluationExperimentIdentityProps) {
  const selectedTarget = props.catalog.targets.find((target) => target.id === props.form.targetID)
  return (
    <section className={sectionStyles.formSection}>
      <EvaluationExperimentSectionHeading
        index="01"
        title="Experiment setup"
        description="Name what changed, then choose a reproducible replay or a live Mixture evaluation."
      />
      <div className={styles.fieldGrid}>
        <ExperimentDescriptionFields form={props.form} />
        <ExperimentModeField form={props.form} />
        <ExperimentTargetField
          catalog={props.catalog}
          form={props.form}
          selectedTarget={selectedTarget}
        />
        <ExperimentBaselineField {...props} />
        <EvaluationExperimentMixture target={selectedTarget} form={props.form} />
      </div>
    </section>
  )
}
