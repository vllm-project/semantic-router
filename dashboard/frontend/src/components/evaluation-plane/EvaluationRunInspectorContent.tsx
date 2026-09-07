import type { EvaluationRun } from '../../types/evaluationPlane'
import { formatDurationBetween } from '../../utils/dateTime'
import ProductLoadingState from '../ProductLoadingState'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import { EvaluationActionButton, RunStatusBadge, TrackChips } from './EvaluationPrimitives'
import EvaluationRunTimeline from './EvaluationRunTimeline'
import RunInspectorActions from './EvaluationRunInspectorActions'
import type {
  EvaluationRunInspectorProps,
  LoadedRunInspectorProps,
} from './EvaluationRunInspector.types'
import { evaluationResultScopeLabel } from './evaluationPresentation'
import { runEvaluationTargetLabel, runWorkloadLabel } from './evaluationRunPresentation'
import planeStyles from './EvaluationPlane.module.css'
import styles from './EvaluationRunInspector.module.css'

function RunInspectorHeader({
  run,
  loading,
  controlledPairRefreshing,
}: Pick<LoadedRunInspectorProps, 'run' | 'loading' | 'controlledPairRefreshing'>) {
  return (
    <header className={styles.inspectorHeader}>
      <div>
        <span className={planeStyles.eyebrow}>Run details</span>
        <h3>{run.name}</h3>
      </div>
      <div className={styles.inspectorState}>
        {loading ? (
          <span className={styles.inspectorRefreshing} role="status">
            Refreshing details…
          </span>
        ) : null}
        {controlledPairRefreshing ? (
          <span className={styles.inspectorRefreshing} role="status">
            Refreshing comparison actions…
          </span>
        ) : null}
        <RunStatusBadge status={run.status} />
      </div>
    </header>
  )
}

function RunInspectorNotices({
  run,
  error,
  controlledPairError,
  onRetry,
  onRetryControlledPair,
}: Pick<
  LoadedRunInspectorProps,
  'run' | 'error' | 'controlledPairError' | 'onRetry' | 'onRetryControlledPair'
>) {
  return (
    <>
      {error ? (
        <div className={styles.inspectorNotice} role="alert">
          <div className={styles.inspectorNoticeCopy}>
            <span>The latest refresh failed. Showing the last saved run details.</span>
            <EvaluationIssueDetails issues={[{ label: 'Run detail refresh', message: error }]} />
          </div>
          <EvaluationActionButton type="button" compact variant="quiet" onClick={onRetry}>
            Retry details
          </EvaluationActionButton>
        </div>
      ) : null}
      {run.controlled_pair && controlledPairError ? (
        <div className={styles.inspectorNotice} role="alert">
          <div className={styles.inspectorNoticeCopy}>
            <span>
              Comparison actions could not be loaded. Existing run evidence remains available.
            </span>
            <EvaluationIssueDetails
              issues={[{ label: 'Comparison actions', message: controlledPairError }]}
            />
          </div>
          <EvaluationActionButton
            type="button"
            compact
            variant="quiet"
            onClick={onRetryControlledPair}
          >
            Retry comparison actions
          </EvaluationActionButton>
        </div>
      ) : null}
    </>
  )
}

function RunInspectorMetadata({ run }: { run: EvaluationRun }) {
  return (
    <dl className={styles.definitionGrid}>
      <div>
        <dt>{run.mixture ? 'Mixture entrypoint' : 'Evaluation target'}</dt>
        <dd>{runEvaluationTargetLabel(run)}</dd>
      </div>
      <div>
        <dt>Evaluation scope</dt>
        <dd>{evaluationResultScopeLabel(run.evidence_level)}</dd>
      </div>
      {run.mixture ? (
        <>
          <div>
            <dt>Routing recipe</dt>
            <dd>{run.mixture.recipe_name}</dd>
          </div>
          <div>
            <dt>Model pool</dt>
            <dd>
              {run.mixture.model_arms.length} models · {run.mixture.decisions.length} decisions
            </dd>
          </div>
        </>
      ) : null}
      <div>
        <dt>Workload</dt>
        <dd>{runWorkloadLabel(run)}</dd>
      </div>
      <div>
        <dt>Repeatability key</dt>
        <dd>{run.seed}</dd>
      </div>
      <div>
        <dt>Duration</dt>
        <dd>{formatDurationBetween(run.started_at, run.completed_at)}</dd>
      </div>
      <div>
        <dt>Baseline</dt>
        <dd>{run.baseline_run_id ? 'Linked baseline' : 'None'}</dd>
      </div>
      <div className={styles.definitionWide}>
        <dt>Benchmarks</dt>
        <dd>{run.suite_ids.length ? `${run.suite_ids.length} selected` : 'None'}</dd>
      </div>
    </dl>
  )
}

function RunInspectorTechnicalDetails({ run }: { run: EvaluationRun }) {
  return (
    <EvaluationTechnicalDisclosure
      className={styles.runTechnicalDetails}
      summary="Technical details"
      summaryClassName={styles.runTechnicalSummary}
    >
      <dl className={styles.runTechnicalGrid}>
        <div>
          <dt>Run ID</dt>
          <dd>
            <code>{run.id}</code>
          </dd>
        </div>
        <div>
          <dt>Evaluation target ID</dt>
          <dd>
            <code>{run.target_id}</code>
          </dd>
        </div>
        <div>
          <dt>Baseline run ID</dt>
          <dd>
            <code>{run.baseline_run_id || 'None'}</code>
          </dd>
        </div>
        <div>
          <dt>Benchmark suite IDs</dt>
          <dd>
            <code>{run.suite_ids.join(', ') || 'None'}</code>
          </dd>
        </div>
        {run.progress.message ? (
          <div>
            <dt>Recorded progress</dt>
            <dd>{run.progress.message}</dd>
          </div>
        ) : null}
      </dl>
    </EvaluationTechnicalDisclosure>
  )
}

function RunExecutionError({ run }: { run: EvaluationRun }) {
  if (!run.error) return null
  return (
    <div className={planeStyles.errorBanner} role="alert">
      <strong>This run stopped before a report was published.</strong>
      <EvaluationIssueDetails issues={[{ label: 'Run execution', message: run.error }]} />
    </div>
  )
}

export function LoadedRunInspector(props: LoadedRunInspectorProps) {
  const { run } = props
  const controlledPair = run.controlled_pair
  const controlledPairState =
    controlledPair && props.controlledPairExecution?.id === controlledPair.pair_id
      ? props.controlledPairExecution
      : null
  const pairCapabilities = props.controlledPairError
    ? null
    : (controlledPairState?.capabilities ?? null)
  return (
    <>
      <RunInspectorHeader {...props} />
      <RunInspectorNotices {...props} />
      <TrackChips trackIDs={run.track_ids} />
      <RunInspectorMetadata run={run} />
      <RunInspectorTechnicalDetails run={run} />
      <RunExecutionError run={run} />
      <RunInspectorActions {...props} pairCapabilities={pairCapabilities} />
      {run.status !== 'completed' && ['failed', 'cancelled'].includes(run.status) ? (
        <p className={planeStyles.scopeNotice}>
          A completed report was not published. Review the execution timeline and available
          technical details before retrying.
        </p>
      ) : null}
      <EvaluationRunTimeline
        run={run}
        events={props.events}
        connected={props.eventsConnected}
        error={props.eventsError}
        onReconnect={props.onReconnectEvents}
      />
    </>
  )
}

export function EmptyRunInspector({
  selectedRunID,
  loading,
  error,
  onRetry,
}: Pick<EvaluationRunInspectorProps, 'selectedRunID' | 'loading' | 'error' | 'onRetry'>) {
  if (loading) {
    return (
      <div className={styles.inspectorEmpty}>
        <ProductLoadingState label="Loading evaluation run" compact />
      </div>
    )
  }
  if (error) {
    return (
      <div className={styles.inspectorEmpty} role="alert">
        <strong>Run could not be loaded</strong>
        <p>Retry to load the selected run and its saved evidence.</p>
        <EvaluationIssueDetails issues={[{ label: 'Run request', message: error }]} />
        <EvaluationActionButton type="button" onClick={onRetry}>
          Retry run
        </EvaluationActionButton>
      </div>
    )
  }
  return (
    <div className={styles.inspectorEmpty}>
      <strong>{selectedRunID ? 'Run is not loaded' : 'Select a run'}</strong>
      <p>
        {selectedRunID
          ? 'Retry the explicit run URL or load older pages to inspect it.'
          : 'Its setup, available actions, and execution timeline appear here.'}
      </p>
      {selectedRunID ? (
        <EvaluationActionButton type="button" onClick={onRetry}>
          Retry run
        </EvaluationActionButton>
      ) : null}
    </div>
  )
}
