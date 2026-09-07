import { useCallback } from 'react'

import type { CreateEvaluationRunPayload, EvaluationRun } from '../types/evaluationPlane'
import type { EvaluationControlledPairExecution } from '../types/evaluationControlledPair'
import {
  cancelEvaluationControlledPair,
  cancelEvaluationRun,
  createEvaluationRun,
  deleteEvaluationControlledPair,
  deleteEvaluationRun,
  startEvaluationRun,
} from '../utils/evaluationPlaneApi'
import type { EvaluationPlaneHookState } from './evaluationPlaneHookState'
import { evaluationErrorMessage, sortEvaluationRuns } from './evaluationPlaneHookSupport'

type MutationRunner = <T>(
  key: string,
  operation: () => Promise<T>,
  fallback: string,
  onSuccess: (value: T) => void | Promise<void>,
) => Promise<T | null>

function useEvaluationMutationRunner(state: EvaluationPlaneHookState): MutationRunner {
  const { mutationLock, setMutationError, setMutationKey, setMutationPending } = state
  return useCallback(
    async <T>(
      key: string,
      operation: () => Promise<T>,
      fallback: string,
      onSuccess: (value: T) => void | Promise<void>,
    ) => {
      if (mutationLock.current) return null
      mutationLock.current = true
      setMutationPending(true)
      setMutationKey(key)
      setMutationError(null)
      try {
        const value = await operation()
        const completion = onSuccess(value)
        if (completion !== undefined) await completion
        return value
      } catch (mutationFailure) {
        setMutationError(evaluationErrorMessage(mutationFailure, fallback))
        return null
      } finally {
        mutationLock.current = false
        setMutationPending(false)
        setMutationKey(null)
      }
    },
    [mutationLock, setMutationError, setMutationKey, setMutationPending],
  )
}

function useReplaceEvaluationRuns(state: EvaluationPlaneHookState) {
  const { setRuns } = state
  return useCallback(
    (nextRuns: EvaluationRun[]) => {
      const nextIDs = new Set(nextRuns.map((run) => run.id))
      setRuns((current) =>
        sortEvaluationRuns([
          ...nextRuns,
          ...current.filter((candidate) => !nextIDs.has(candidate.id)),
        ]),
      )
    },
    [setRuns],
  )
}

function useResetRunRequestsAfterMutation(state: EvaluationPlaneHookState) {
  const {
    loadingMoreRequest,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setLoadingAllRuns,
    setLoadingMoreRuns,
    setRefreshing,
  } = state
  return useCallback(() => {
    runsRequestVersion.current += 1
    runsController.current?.abort()
    runsRefreshPromise.current = null
    loadingMoreRequest.current = false
    setRefreshing(false)
    setLoadingMoreRuns(false)
    setLoadingAllRuns(false)
  }, [
    loadingMoreRequest,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setLoadingAllRuns,
    setLoadingMoreRuns,
    setRefreshing,
  ])
}

function useEvaluationRunMutations(
  state: EvaluationPlaneHookState,
  runMutation: MutationRunner,
  replaceRuns: (runs: EvaluationRun[]) => void,
  resetRunRequests: () => void,
) {
  const { catalog, setLastUpdatedAt, setMutationError, setRunPage } = state
  const mutateRun = useCallback(
    (key: string, operation: () => Promise<EvaluationRun>, fallback: string) =>
      runMutation(key, operation, fallback, (run) => {
        resetRunRequests()
        replaceRuns([run])
        setLastUpdatedAt(new Date())
      }),
    [replaceRuns, resetRunRequests, runMutation, setLastUpdatedAt],
  )
  const createRun = useCallback(
    async (request: CreateEvaluationRunPayload) => {
      if (!catalog) {
        setMutationError('The evaluation catalog is not available yet.')
        return null
      }
      const created = await mutateRun(
        'create',
        () => createEvaluationRun(request, catalog),
        'Failed to create the evaluation run.',
      )
      if (created) setRunPage((current) => ({ ...current, totalRuns: current.totalRuns + 1 }))
      return created
    },
    [catalog, mutateRun, setMutationError, setRunPage],
  )
  const startRun = useCallback(
    (id: string) =>
      mutateRun(`start:${id}`, () => startEvaluationRun(id), 'Failed to start the evaluation run.'),
    [mutateRun],
  )
  const cancelRun = useCallback(
    (id: string) =>
      mutateRun(
        `cancel:${id}`,
        () => cancelEvaluationRun(id),
        'Failed to cancel the evaluation run.',
      ),
    [mutateRun],
  )
  return { createRun, startRun, cancelRun }
}

function useControlledPairMutation(
  state: EvaluationPlaneHookState,
  runMutation: MutationRunner,
  replaceRuns: (runs: EvaluationRun[]) => void,
  resetRunRequests: () => void,
) {
  const { setLastUpdatedAt } = state
  const mutateControlledPair = useCallback(
    (key: string, operation: () => Promise<EvaluationControlledPairExecution>, fallback: string) =>
      runMutation(key, operation, fallback, (execution) => {
        resetRunRequests()
        replaceRuns([execution.baseline_run, execution.candidate_run])
        setLastUpdatedAt(new Date())
      }),
    [replaceRuns, resetRunRequests, runMutation, setLastUpdatedAt],
  )
  return useCallback(
    (id: string) =>
      mutateControlledPair(
        `cancel-pair:${id}`,
        () => cancelEvaluationControlledPair(id),
        'Failed to cancel the controlled comparison.',
      ),
    [mutateControlledPair],
  )
}

function useEvaluationDeleteMutations(
  state: EvaluationPlaneHookState,
  runMutation: MutationRunner,
  refreshRuns: () => Promise<boolean>,
  resetRunRequests: () => void,
) {
  const { setLastUpdatedAt, setRunPage, setRuns } = state
  const deleteRun = useCallback(
    async (id: string) => {
      const deleted = await runMutation(
        `delete:${id}`,
        async () => {
          await deleteEvaluationRun(id)
          return true
        },
        'Failed to delete the evaluation run.',
        () => {
          resetRunRequests()
          setRuns((current) => current.filter((run) => run.id !== id))
          setRunPage((current) => ({ ...current, totalRuns: Math.max(0, current.totalRuns - 1) }))
          setLastUpdatedAt(new Date())
        },
      )
      return deleted === true
    },
    [resetRunRequests, runMutation, setLastUpdatedAt, setRunPage, setRuns],
  )
  const deleteControlledPair = useCallback(
    async (id: string) => {
      const deleted = await runMutation(
        `delete-pair:${id}`,
        async () => {
          await deleteEvaluationControlledPair(id)
          return true
        },
        'Failed to delete the controlled comparison.',
        async () => {
          resetRunRequests()
          setRuns((current) => current.filter((run) => run.controlled_pair?.pair_id !== id))
          setRunPage((current) => ({ ...current, totalRuns: Math.max(0, current.totalRuns - 2) }))
          setLastUpdatedAt(new Date())
          await refreshRuns()
        },
      )
      return deleted === true
    },
    [refreshRuns, resetRunRequests, runMutation, setLastUpdatedAt, setRunPage, setRuns],
  )
  return { deleteRun, deleteControlledPair }
}

export function useEvaluationPlaneMutations(
  state: EvaluationPlaneHookState,
  refreshRuns: () => Promise<boolean>,
) {
  const runMutation = useEvaluationMutationRunner(state)
  const replaceRuns = useReplaceEvaluationRuns(state)
  const resetRunRequests = useResetRunRequestsAfterMutation(state)
  const runActions = useEvaluationRunMutations(state, runMutation, replaceRuns, resetRunRequests)
  const cancelControlledPair = useControlledPairMutation(
    state,
    runMutation,
    replaceRuns,
    resetRunRequests,
  )
  const deleteActions = useEvaluationDeleteMutations(
    state,
    runMutation,
    refreshRuns,
    resetRunRequests,
  )
  return { ...runActions, cancelControlledPair, ...deleteActions }
}
