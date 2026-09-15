import type {
  EvaluationControlledPairReadyHandler,
  EvaluationControlledPairWorkflow,
} from './evaluationControlledPairHookSupport'
import {
  useControlledPairPolling,
  useControlledPairRouteReconciliation,
  useControlledPairSession,
  useCreateControlledPair,
  useDeliverControlledPairReady,
  useReconcileControlledPair,
  useResetControlledPair,
  useRetryControlledPair,
} from './useEvaluationControlledPairWorkflow'

export function useEvaluationControlledPair(
  onReady: EvaluationControlledPairReadyHandler,
  workflow: EvaluationControlledPairWorkflow,
) {
  const { state, session } = useControlledPairSession(onReady, workflow)
  const deliverReady = useDeliverControlledPairReady(session)
  const create = useCreateControlledPair(session, deliverReady)
  const reconcile = useReconcileControlledPair(session, deliverReady)
  useControlledPairRouteReconciliation(session, state, workflow.activePairID, reconcile)
  useControlledPairPolling(session, state, deliverReady)
  const retry = useRetryControlledPair(
    session,
    state,
    workflow.activePairID,
    create,
    deliverReady,
    reconcile,
  )
  const reset = useResetControlledPair(session)

  return { ...state, create, retry, reset }
}
