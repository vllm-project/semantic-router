import type { EvaluationControlledPairExecution } from '../../types/evaluationControlledPair'
import type { EvaluationChangeProfileId, EvaluationRun } from '../../types/evaluationPlane'
import type { useEvaluationControlledPair } from '../../hooks/useEvaluationControlledPair'
import EvaluationIssueDetails, { type EvaluationIssueDetail } from './EvaluationIssueDetails'
import { EvaluationActionButton } from './EvaluationPrimitives'
import { RUN_STATUS_LABELS } from './evaluationPresentation'
import { runOptionLabels } from './evaluationRunPresentation'
import styles from './EvaluationCampaignControlledPair.module.css'

type ControlledPairWorkflow = ReturnType<typeof useEvaluationControlledPair>

export interface EvaluationCampaignControlledPairViewProps {
  slotGateID: string
  baselineSourceID: string
  candidateSourceID: string
  baselineOptions: EvaluationRun[]
  candidateOptions: EvaluationRun[]
  canCreate: boolean
  disabled: boolean
  busy: boolean
  activePairID: string | null
  resumablePair: { id: string; profileID: EvaluationChangeProfileId } | null
  sourceReady: boolean
  selectionRationale: string
  pair: ControlledPairWorkflow
  onBaselineSourceChange: (runID: string) => void
  onCandidateSourceChange: (runID: string) => void
  onClearSavedPair: () => void
  onResumePair: () => void
}

function runProgress(run: EvaluationRun): string {
  return `${Math.round(run.progress.percent)}% · ${RUN_STATUS_LABELS[run.status]}`
}

function runProgressGuidance(run: EvaluationRun): string {
  switch (run.status) {
    case 'pending':
      return 'Waiting to start.'
    case 'running':
      return run.progress.total > 0
        ? `${run.progress.completed} of ${run.progress.total} evaluation steps complete.`
        : 'Evaluation is in progress.'
    case 'sealing':
      return 'Finalizing verified results.'
    case 'completed':
      return 'Results are ready.'
    case 'failed':
      return 'Stopped before completing.'
    case 'cancelled':
      return 'Cancelled before completing.'
  }
}

function pairProductError(
  execution: EvaluationControlledPairExecution | null,
  activePairID: string | null,
): string {
  if (
    execution &&
    [execution.baseline_run.status, execution.candidate_run.status].some((status) =>
      ['failed', 'cancelled'].includes(status),
    )
  ) {
    return 'One comparison run stopped before completing. Review the technical details, then retry.'
  }
  if (
    execution?.state === 'terminal' &&
    execution.baseline_run.status === 'completed' &&
    execution.candidate_run.status === 'completed'
  ) {
    return 'The completed comparison could not be attached to this decision. Retry the comparison.'
  }
  if (activePairID) {
    return 'The saved comparison could not be refreshed. Retry it or clear the saved comparison.'
  }
  return 'The controlled comparison could not continue. Review the technical details, then retry.'
}

function runTechnicalIssues(
  execution: EvaluationControlledPairExecution | null,
): EvaluationIssueDetail[] {
  if (!execution) return []
  return [
    ...(execution.baseline_run.progress.message
      ? [
          {
            label: 'Baseline progress response',
            message: execution.baseline_run.progress.message,
          },
        ]
      : []),
    ...(execution.baseline_run.error
      ? [{ label: 'Baseline execution response', message: execution.baseline_run.error }]
      : []),
    ...(execution.candidate_run.progress.message
      ? [
          {
            label: 'Candidate progress response',
            message: execution.candidate_run.progress.message,
          },
        ]
      : []),
    ...(execution.candidate_run.error
      ? [{ label: 'Candidate execution response', message: execution.candidate_run.error }]
      : []),
  ]
}

function ControlledPairSourceSelectors({
  baselineSourceID,
  candidateSourceID,
  baselineOptions,
  candidateOptions,
  disabled,
  busy,
  pair,
  onBaselineSourceChange,
  onCandidateSourceChange,
}: EvaluationCampaignControlledPairViewProps) {
  const baselineLabels = runOptionLabels(baselineOptions)
  const candidateLabels = runOptionLabels(candidateOptions)
  const selectionDisabled = disabled || busy || pair.status === 'ready'
  return (
    <div className={styles.sourceGrid}>
      <label>
        Baseline run
        <select
          aria-label="Controlled comparison baseline run"
          value={baselineSourceID}
          disabled={selectionDisabled}
          onChange={(event) => onBaselineSourceChange(event.target.value)}
        >
          <option value="">Select completed live baseline</option>
          {baselineOptions.map((run) => (
            <option key={run.id} value={run.id}>
              {baselineLabels.get(run.id)}
            </option>
          ))}
        </select>
      </label>
      <label>
        Candidate run
        <select
          aria-label="Controlled comparison candidate run"
          value={candidateSourceID}
          disabled={selectionDisabled || !baselineSourceID}
          onChange={(event) => onCandidateSourceChange(event.target.value)}
        >
          <option value="">Select matching live candidate</option>
          {candidateOptions.map((run) => (
            <option key={run.id} value={run.id}>
              {candidateLabels.get(run.id)}
            </option>
          ))}
        </select>
      </label>
    </div>
  )
}

function ControlledPairProgress({ pair }: Pick<EvaluationCampaignControlledPairViewProps, 'pair'>) {
  if (!pair.execution) return null
  return (
    <>
      <dl className={styles.progress} aria-label="Controlled comparison progress">
        <div>
          <dt>Baseline run</dt>
          <dd>{runProgress(pair.execution.baseline_run)}</dd>
          <small>{runProgressGuidance(pair.execution.baseline_run)}</small>
        </div>
        <div>
          <dt>Candidate run</dt>
          <dd>{runProgress(pair.execution.candidate_run)}</dd>
          <small>{runProgressGuidance(pair.execution.candidate_run)}</small>
        </div>
      </dl>
      <EvaluationIssueDetails
        className={styles.progressDetails}
        issues={runTechnicalIssues(pair.execution)}
      />
    </>
  )
}

function ControlledPairError({
  pair,
  activePairID,
  canCreate,
  busy,
  onClearSavedPair,
}: Pick<
  EvaluationCampaignControlledPairViewProps,
  'pair' | 'activePairID' | 'canCreate' | 'busy' | 'onClearSavedPair'
>) {
  if (!pair.error) return null
  return (
    <div className={styles.error} role="alert">
      <div className={styles.errorCopy}>
        <span>{pairProductError(pair.execution, activePairID)}</span>
        <EvaluationIssueDetails
          issues={[{ label: 'Comparison workflow response', message: pair.error }]}
        />
      </div>
      <div className={styles.errorActions}>
        <EvaluationActionButton
          type="button"
          compact
          disabled={!canCreate || busy}
          onClick={pair.retry}
        >
          Retry comparison
        </EvaluationActionButton>
        {activePairID ? (
          <EvaluationActionButton
            type="button"
            compact
            variant="quiet"
            disabled={busy}
            onClick={onClearSavedPair}
          >
            Clear saved comparison
          </EvaluationActionButton>
        ) : null}
      </div>
    </div>
  )
}

function ControlledPairLaunchAction(props: EvaluationCampaignControlledPairViewProps) {
  const { pair, activePairID, resumablePair, busy } = props
  if (
    pair.status === 'ready' ||
    pair.error ||
    (pair.status === 'idle' && !activePairID && resumablePair)
  ) {
    return null
  }
  const actionLabel =
    pair.status === 'creating'
      ? 'Starting comparison…'
      : pair.status === 'recovering'
        ? 'Recovering comparison…'
        : pair.status === 'assigning'
          ? 'Attaching completed comparison…'
          : pair.status === 'running'
            ? 'Comparison running…'
            : 'Launch comparison'
  return (
    <div className={styles.pairAction}>
      <span>
        {pair.status === 'assigning'
          ? 'Both runs completed. Refreshing run history before attaching the comparison.'
          : busy
            ? 'Both runs must finish before the comparison can be finalized.'
            : props.selectionRationale}
      </span>
      <EvaluationActionButton
        type="button"
        compact
        variant="primary"
        disabled={!props.canCreate || props.disabled || busy || !props.sourceReady}
        onClick={() => void pair.create(props.baselineSourceID, props.candidateSourceID)}
      >
        {actionLabel}
      </EvaluationActionButton>
    </div>
  )
}

function ControlledPairResumeAction(props: EvaluationCampaignControlledPairViewProps) {
  if (props.activePairID || props.pair.status !== 'idle' || !props.resumablePair) return null
  return (
    <div className={styles.pairAction} role="status">
      <span>Run history contains one active comparison that is not linked here.</span>
      <EvaluationActionButton type="button" compact variant="quiet" onClick={props.onResumePair}>
        Resume comparison
      </EvaluationActionButton>
    </div>
  )
}

export default function EvaluationCampaignControlledPairView(
  props: EvaluationCampaignControlledPairViewProps,
) {
  return (
    <section
      className={styles.pairStep}
      aria-labelledby="campaign-controlled-pair-title"
      aria-busy={props.busy}
      data-check-id={props.slotGateID}
    >
      <div className={styles.pairIntro}>
        <h4 id="campaign-controlled-pair-title">Controlled live comparison</h4>
        <p>
          Choose two completed live runs. The server launches a fresh order-balanced comparison and
          keeps credentials, workload, execution order, and evaluation settings consistent.
        </p>
      </div>
      <div className={styles.pairWorkspace}>
        <ControlledPairSourceSelectors {...props} />
        <ControlledPairProgress pair={props.pair} />
        <ControlledPairError {...props} />
        {props.pair.status === 'ready' ? (
          <div className={styles.ready} role="status">
            Fresh baseline and candidate runs completed and are attached to the value comparison.
          </div>
        ) : null}
        <ControlledPairLaunchAction {...props} />
        <ControlledPairResumeAction {...props} />
      </div>
    </section>
  )
}
