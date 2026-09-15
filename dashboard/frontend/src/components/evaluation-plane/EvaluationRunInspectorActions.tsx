import type { EvaluationControlledPairCapabilities } from '../../types/evaluationControlledPair'
import { EvaluationActionButton } from './EvaluationPrimitives'
import type { LoadedRunInspectorProps } from './EvaluationRunInspector.types'
import styles from './EvaluationRunInspector.module.css'

function RunLifecycleActions({
  run,
  canRun,
  mutationKey,
  onStart,
  onCancel,
  onOpenReport,
}: Pick<
  LoadedRunInspectorProps,
  'run' | 'canRun' | 'mutationKey' | 'onStart' | 'onCancel' | 'onOpenReport'
>) {
  if (run.controlled_pair) {
    return run.status === 'completed' ? (
      <EvaluationActionButton
        type="button"
        variant="primary"
        compact
        aria-label={`Open report for ${run.name}`}
        onClick={() => onOpenReport(run)}
      >
        Open report
      </EvaluationActionButton>
    ) : null
  }
  return (
    <>
      {run.status === 'pending' && canRun ? (
        <EvaluationActionButton
          type="button"
          variant="primary"
          compact
          disabled={mutationKey !== null}
          aria-label={`Start ${run.name}`}
          onClick={() => onStart(run)}
        >
          {mutationKey === `start:${run.id}` ? 'Starting…' : 'Start'}
        </EvaluationActionButton>
      ) : null}
      {run.status === 'running' && canRun ? (
        <EvaluationActionButton
          type="button"
          compact
          disabled={mutationKey !== null}
          aria-label={`Cancel ${run.name}`}
          onClick={() => onCancel(run)}
        >
          Cancel
        </EvaluationActionButton>
      ) : null}
      {run.status === 'completed' ? (
        <EvaluationActionButton
          type="button"
          variant="primary"
          compact
          aria-label={`Open report for ${run.name}`}
          onClick={() => onOpenReport(run)}
        >
          Open report
        </EvaluationActionButton>
      ) : null}
    </>
  )
}

function ControlledPairActions({
  run,
  capabilities,
  controlledPairLoading,
  controlledPairRefreshing,
  canRun,
  canDelete,
  mutationKey,
  onCancel,
  onDelete,
}: Pick<
  LoadedRunInspectorProps,
  | 'run'
  | 'controlledPairLoading'
  | 'controlledPairRefreshing'
  | 'canRun'
  | 'canDelete'
  | 'mutationKey'
  | 'onCancel'
  | 'onDelete'
> & { capabilities: EvaluationControlledPairCapabilities | null }) {
  const controlledPair = run.controlled_pair
  if (!controlledPair) return null
  const mutationPending = mutationKey !== null
  return (
    <>
      {capabilities?.can_cancel && canRun ? (
        <EvaluationActionButton
          type="button"
          compact
          disabled={mutationPending || controlledPairRefreshing}
          aria-label="Cancel controlled comparison"
          onClick={() => onCancel(run)}
        >
          {mutationKey === `cancel-pair:${controlledPair.pair_id}`
            ? 'Cancelling comparison…'
            : 'Cancel comparison'}
        </EvaluationActionButton>
      ) : null}
      {capabilities?.can_delete && canDelete ? (
        <EvaluationActionButton
          type="button"
          variant="danger"
          compact
          disabled={mutationPending || controlledPairRefreshing}
          aria-label="Delete controlled comparison"
          onClick={() => onDelete(run)}
        >
          {mutationKey === `delete-pair:${controlledPair.pair_id}`
            ? 'Deleting comparison…'
            : 'Delete comparison'}
        </EvaluationActionButton>
      ) : null}
      {controlledPairLoading ? (
        <span className={styles.pairControlStatus} role="status">
          Loading comparison actions…
        </span>
      ) : null}
    </>
  )
}

function StandaloneDeleteAction({
  run,
  canDelete,
  mutationKey,
  onDelete,
}: Pick<LoadedRunInspectorProps, 'run' | 'canDelete' | 'mutationKey' | 'onDelete'>) {
  if (run.controlled_pair || run.status === 'running' || run.status === 'sealing' || !canDelete) {
    return null
  }
  return (
    <EvaluationActionButton
      type="button"
      variant="danger"
      compact
      disabled={mutationKey !== null}
      aria-label={`Delete ${run.name}`}
      onClick={() => onDelete(run)}
    >
      Delete
    </EvaluationActionButton>
  )
}

export default function RunInspectorActions({
  pairCapabilities,
  ...props
}: LoadedRunInspectorProps & { pairCapabilities: EvaluationControlledPairCapabilities | null }) {
  return (
    <div
      className={styles.inspectorActions}
      role="group"
      data-testid="evaluation-run-actions"
      aria-label={`Actions for ${props.run.name}`}
    >
      <RunLifecycleActions {...props} />
      <ControlledPairActions {...props} capabilities={pairCapabilities} />
      <StandaloneDeleteAction {...props} />
    </div>
  )
}
