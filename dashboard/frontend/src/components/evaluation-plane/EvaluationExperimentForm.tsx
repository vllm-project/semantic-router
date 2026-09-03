import type {
  EvaluationCatalog,
  EvaluationCatalogTarget,
  EvaluationExperimentIntent,
  EvaluationRun,
} from '../../types/evaluationPlane'
import EvaluationExperimentBenchmarkScope from './EvaluationExperimentBenchmarkScope'
import EvaluationExperimentBudget from './EvaluationExperimentBudget'
import EvaluationExperimentCapacitySLO from './EvaluationExperimentCapacitySLO'
import EvaluationExperimentGateScope from './EvaluationExperimentGateScope'
import EvaluationExperimentIdentity from './EvaluationExperimentIdentity'
import { evaluationResultScopeLabel } from './evaluationPresentation'
import { EvaluationActionButton, EvaluationTag } from './EvaluationPrimitives'
import { changeProfileLabel } from './evaluationRunPresentation'
import { targetPresentationLabel } from './evaluationTargetPresentation'
import styles from './EvaluationForm.module.css'
import useEvaluationExperimentForm, {
  type EvaluationExperimentFormModel,
} from './useEvaluationExperimentForm'

interface EvaluationExperimentFormProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  totalRuns: number
  canCreate: boolean
  canAutoStart: boolean
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  pending: boolean
  initialEntrypoint?: string | null
  onLoadMoreRuns: () => void
  onSubmit: (intent: EvaluationExperimentIntent) => Promise<boolean>
}

function requestedLiveTarget(
  catalog: EvaluationCatalog,
  initialEntrypoint: string | null | undefined,
): EvaluationCatalogTarget | undefined {
  if (!initialEntrypoint) return undefined
  return catalog.targets.find(
    (target) =>
      target.modes.includes('live') &&
      (target.mixture?.entrypoint_model === initialEntrypoint ||
        target.mixture?.aliases.includes(initialEntrypoint)),
  )
}

function ReadOnlyExperimentState() {
  return (
    <section className={styles.permissionState}>
      <span>Read-only evaluation access</span>
      <h2>Experiment creation is not available for this session.</h2>
      <p>
        You can still inspect completed results, reports, configuration details, and comparisons.
      </p>
    </section>
  )
}

function ExperimentIntro({ form }: { form: EvaluationExperimentFormModel }) {
  return (
    <div className={styles.intro}>
      <div>
        <span className={styles.eyebrow}>Reproducible evaluation</span>
        <h2>New evaluation experiment</h2>
        <p>
          Choose a registered Mixture and benchmark scope. Each run preserves the exact workload,
          configuration, and evaluation source behind its results.
        </p>
      </div>
      <div className={styles.introBadges}>
        <EvaluationTag tone="info">
          {form.catalogEvidenceClass
            ? `Evaluation scope · ${evaluationResultScopeLabel(form.catalogEvidenceClass)}`
            : 'Choose benchmarks to set the scope'}
        </EvaluationTag>
      </div>
    </div>
  )
}

function ExperimentValidationStates({
  form,
  initialEntrypoint,
  requestedTarget,
}: {
  form: EvaluationExperimentFormModel
  initialEntrypoint?: string | null
  requestedTarget: EvaluationCatalogTarget | undefined
}) {
  return (
    <>
      {form.validationError ? (
        <div ref={form.errorRef} className={styles.error} role="alert" tabIndex={-1}>
          {form.validationError}
        </div>
      ) : null}
      {initialEntrypoint && !requestedTarget && !form.targetID ? (
        <div className={styles.deepLinkWarning} role="alert">
          <div>
            <strong>Requested Mixture is not registered for evaluation</strong>
            <span>
              <code>{initialEntrypoint}</code> does not have a saved test setup or an available live
              destination. Refresh its configuration, or choose another live Mixture.
            </span>
          </div>
        </div>
      ) : null}
    </>
  )
}

function ExperimentActions({
  form,
  selectedTarget,
  pending,
}: {
  form: EvaluationExperimentFormModel
  selectedTarget: EvaluationCatalogTarget | undefined
  pending: boolean
}) {
  return (
    <div className={styles.actions}>
      <span>
        {form.suiteIDs.length} {form.suiteIDs.length === 1 ? 'benchmark' : 'benchmarks'} ·{' '}
        {form.trackIDs.length} {form.trackIDs.length === 1 ? 'area' : 'areas'} · change type{' '}
        {form.changeProfile ? changeProfileLabel(form.changeProfile) : 'not selected'} · source{' '}
        {selectedTarget ? targetPresentationLabel(selectedTarget) : 'not selected'}
        {form.capacitySLOActive ? ' · performance goals included' : ''}
      </span>
      <EvaluationActionButton type="submit" variant="primary" disabled={pending}>
        {pending ? 'Creating…' : form.autoStart ? 'Create and start' : 'Create draft'}
      </EvaluationActionButton>
    </div>
  )
}

function ExperimentFields({
  props,
  form,
  selectedTarget,
}: {
  props: EvaluationExperimentFormProps
  form: EvaluationExperimentFormModel
  selectedTarget: EvaluationCatalogTarget | undefined
}) {
  return (
    <fieldset
      disabled={props.pending}
      aria-busy={props.pending}
      aria-label="Evaluation experiment fields"
      className={styles.formFields}
    >
      <EvaluationExperimentIdentity
        catalog={props.catalog}
        runs={props.runs}
        totalRuns={props.totalRuns}
        runLedgerAvailable={props.runLedgerAvailable}
        runLedgerComplete={props.runLedgerComplete}
        hasMoreRuns={props.hasMoreRuns}
        loadingMoreRuns={props.loadingMoreRuns}
        pending={props.pending}
        onLoadMoreRuns={props.onLoadMoreRuns}
        form={form}
      />
      <EvaluationExperimentGateScope catalog={props.catalog} form={form} />
      <EvaluationExperimentBenchmarkScope catalog={props.catalog} form={form} />
      <EvaluationExperimentCapacitySLO form={form} />
      <EvaluationExperimentBudget canAutoStart={props.canAutoStart} form={form} />
      <ExperimentActions form={form} selectedTarget={selectedTarget} pending={props.pending} />
    </fieldset>
  )
}

export default function EvaluationExperimentForm(props: EvaluationExperimentFormProps) {
  const requestedTarget = requestedLiveTarget(props.catalog, props.initialEntrypoint)
  const form = useEvaluationExperimentForm({
    catalog: props.catalog,
    runs: props.runs,
    canAutoStart: props.canAutoStart,
    runLedgerAvailable: props.runLedgerAvailable,
    runLedgerComplete: props.runLedgerComplete,
    pending: props.pending,
    initialTargetID: requestedTarget?.id,
    preserveMissingLiveTarget: Boolean(props.initialEntrypoint && !requestedTarget),
    onSubmit: props.onSubmit,
  })
  if (!props.canCreate) return <ReadOnlyExperimentState />
  const selectedTarget = props.catalog.targets.find((target) => target.id === form.targetID)
  return (
    <form className={styles.form} onSubmit={form.submit} aria-busy={props.pending}>
      <ExperimentIntro form={form} />
      <ExperimentValidationStates
        form={form}
        initialEntrypoint={props.initialEntrypoint}
        requestedTarget={requestedTarget}
      />
      <ExperimentFields props={props} form={form} selectedTarget={selectedTarget} />
    </form>
  )
}
