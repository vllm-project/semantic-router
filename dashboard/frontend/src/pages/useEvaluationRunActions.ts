import { useCallback, useEffect, useState } from 'react'

import type { EvaluationControlledPairExecution } from '../types/evaluationControlledPair'
import type { EvaluationRun } from '../types/evaluationPlane'
import { removeEvaluationRun, type EvaluationRoute } from './evaluationRoute'

interface EvaluationRunActionOperations {
  cancelRun: (runID: string) => Promise<EvaluationRun | null>
  cancelControlledPair: (pairID: string) => Promise<EvaluationControlledPairExecution | null>
  deleteRun: (runID: string) => Promise<boolean>
  deleteControlledPair: (pairID: string) => Promise<boolean>
  clearMutationError: () => void
}

interface EvaluationRunActionsOptions {
  canRun: boolean
  canWrite: boolean
  route: EvaluationRoute
  setRoute: (route: EvaluationRoute, replace?: boolean) => void
  operations: EvaluationRunActionOperations
  adoptControlledPair: (execution: EvaluationControlledPairExecution) => void
}

export function useEvaluationRunActions(options: EvaluationRunActionsOptions) {
  const { canRun, canWrite, route, setRoute, operations, adoptControlledPair } = options
  const [cancelTarget, setCancelTarget] = useState<EvaluationRun | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<EvaluationRun | null>(null)
  const [cancelReturnFocusMode, setCancelReturnFocusMode] = useState<'fallback' | 'always'>(
    'fallback',
  )
  const [deleteReturnFocusMode, setDeleteReturnFocusMode] = useState<'fallback' | 'always'>(
    'fallback',
  )

  useEffect(() => {
    if (!canRun) setCancelTarget(null)
    if (!canWrite) setDeleteTarget(null)
  }, [canRun, canWrite])

  const requestCancel = useCallback(
    (run: EvaluationRun) => {
      operations.clearMutationError()
      setCancelReturnFocusMode('fallback')
      setCancelTarget(run)
    },
    [operations],
  )
  const requestDelete = useCallback(
    (run: EvaluationRun) => {
      operations.clearMutationError()
      setDeleteReturnFocusMode('fallback')
      setDeleteTarget(run)
    },
    [operations],
  )
  const closeCancel = useCallback(() => {
    setCancelReturnFocusMode('fallback')
    setCancelTarget(null)
    operations.clearMutationError()
  }, [operations])
  const closeDelete = useCallback(() => {
    setDeleteReturnFocusMode('fallback')
    setDeleteTarget(null)
    operations.clearMutationError()
  }, [operations])

  const confirmCancel = useCallback(async () => {
    if (!cancelTarget || !canRun) return
    const pairExecution = cancelTarget.controlled_pair
      ? await operations.cancelControlledPair(cancelTarget.controlled_pair.pair_id)
      : null
    const succeeded = cancelTarget.controlled_pair
      ? Boolean(pairExecution)
      : Boolean(await operations.cancelRun(cancelTarget.id))
    if (!succeeded) return

    if (pairExecution) adoptControlledPair(pairExecution)
    setCancelReturnFocusMode('always')
    setCancelTarget(null)
    setRoute(
      {
        view: 'runs',
        runID: cancelTarget.id,
        controlledPairID: route.controlledPairID,
        controlledPairProfileID: route.controlledPairProfileID,
      },
      true,
    )
  }, [adoptControlledPair, canRun, cancelTarget, operations, route, setRoute])

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget || !canWrite) return
    const deleted = deleteTarget.controlled_pair
      ? await operations.deleteControlledPair(deleteTarget.controlled_pair.pair_id)
      : await operations.deleteRun(deleteTarget.id)
    if (!deleted) return

    const nextRoute = removeEvaluationRun(route, deleteTarget.id)
    setDeleteReturnFocusMode('always')
    setDeleteTarget(null)
    setRoute(
      deleteTarget.controlled_pair?.pair_id === route.controlledPairID
        ? { ...nextRoute, controlledPairID: null, controlledPairProfileID: null }
        : nextRoute,
      true,
    )
  }, [canWrite, deleteTarget, operations, route, setRoute])

  return {
    cancelTarget,
    deleteTarget,
    cancelReturnFocusMode,
    deleteReturnFocusMode,
    requestCancel,
    requestDelete,
    closeCancel,
    closeDelete,
    confirmCancel,
    confirmDelete,
  }
}
