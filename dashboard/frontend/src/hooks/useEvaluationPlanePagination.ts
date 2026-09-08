import { useCallback } from 'react'

import type { EvaluationRun } from '../types/evaluationPlane'
import { listEvaluationRuns } from '../utils/evaluationPlaneApi'
import type { EvaluationPlaneHookState } from './evaluationPlaneHookState'
import { evaluationErrorMessage, mergeEvaluationRuns } from './evaluationPlaneHookSupport'

type EvaluationRunLedger = Awaited<ReturnType<typeof listEvaluationRuns>>

function useApplyCompleteRunLedger(state: EvaluationPlaneHookState) {
  const {
    loadedPageCount,
    setLastUpdatedAt,
    setRunLedgerComplete,
    setRunLedgerWarnings,
    setRunPage,
    setRunPollingPaused,
    setRuns,
    setRunsError,
    setRunsLoaded,
  } = state
  return useCallback(
    (ledger: EvaluationRunLedger, merged: EvaluationRun[], pageCount: number) => {
      setRuns(merged)
      setRunsLoaded(true)
      setRunLedgerComplete(ledger.ledger_complete)
      setRunLedgerWarnings(ledger.warnings)
      setRunPage({
        nextCursor: null,
        totalRuns: ledger.total_runs,
        warningCount: ledger.warning_count,
      })
      setRunsError(
        merged.length === ledger.total_runs
          ? null
          : 'The run ledger changed while it was loading. Refresh before building a campaign.',
      )
      setLastUpdatedAt(new Date())
      loadedPageCount.current += pageCount
      setRunPollingPaused(true)
    },
    [
      loadedPageCount,
      setLastUpdatedAt,
      setRunLedgerComplete,
      setRunLedgerWarnings,
      setRunPage,
      setRunPollingPaused,
      setRuns,
      setRunsError,
      setRunsLoaded,
    ],
  )
}

export function useLoadMoreEvaluationRuns(
  state: EvaluationPlaneHookState,
  applyRunLedger: (ledger: EvaluationRunLedger, append: boolean) => void,
) {
  const {
    loadingMoreRequest,
    loadingMoreRuns,
    refreshing,
    runPage,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setLoadingMoreRuns,
    setRunsError,
  } = state
  return useCallback(async () => {
    const cursor = runPage.nextCursor
    if (!cursor || refreshing || loadingMoreRuns) return
    const version = ++runsRequestVersion.current
    runsController.current?.abort()
    runsRefreshPromise.current = null
    const controller = new AbortController()
    runsController.current = controller
    loadingMoreRequest.current = true
    setLoadingMoreRuns(true)
    try {
      const ledger = await listEvaluationRuns({ cursor, signal: controller.signal })
      if (controller.signal.aborted || version !== runsRequestVersion.current) return
      applyRunLedger(ledger, true)
    } catch (loadError) {
      if (controller.signal.aborted || version !== runsRequestVersion.current) return
      setRunsError(evaluationErrorMessage(loadError, 'Failed to load more evaluation runs.'))
    } finally {
      if (version === runsRequestVersion.current) {
        loadingMoreRequest.current = false
        setLoadingMoreRuns(false)
      }
    }
  }, [
    applyRunLedger,
    loadingMoreRequest,
    loadingMoreRuns,
    refreshing,
    runPage.nextCursor,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setLoadingMoreRuns,
    setRunsError,
  ])
}

export function useLoadAllEvaluationRuns(state: EvaluationPlaneHookState) {
  const applyCompleteRunLedger = useApplyCompleteRunLedger(state)
  const {
    loadingMoreRequest,
    loadingMoreRuns,
    refreshing,
    runPage,
    runs,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setLoadingAllRuns,
    setLoadingMoreRuns,
    setRunsError,
  } = state
  return useCallback(async () => {
    let cursor = runPage.nextCursor
    if (!cursor || refreshing || loadingMoreRuns) return
    const version = ++runsRequestVersion.current
    runsController.current?.abort()
    runsRefreshPromise.current = null
    const controller = new AbortController()
    runsController.current = controller
    loadingMoreRequest.current = true
    setLoadingMoreRuns(true)
    setLoadingAllRuns(true)
    const pages: EvaluationRun[] = []
    let finalLedger: EvaluationRunLedger | null = null
    let pageCount = 0
    try {
      while (cursor) {
        const ledger = await listEvaluationRuns({ cursor, signal: controller.signal })
        if (controller.signal.aborted || version !== runsRequestVersion.current) return
        pages.push(...ledger.runs)
        finalLedger = ledger
        cursor = ledger.next_cursor || ''
        pageCount += 1
      }
      if (!finalLedger) return
      applyCompleteRunLedger(finalLedger, mergeEvaluationRuns(runs, pages), pageCount)
    } catch (loadError) {
      if (controller.signal.aborted || version !== runsRequestVersion.current) return
      setRunsError(
        evaluationErrorMessage(loadError, 'Failed to load the complete evaluation run ledger.'),
      )
    } finally {
      if (version === runsRequestVersion.current) {
        loadingMoreRequest.current = false
        setLoadingMoreRuns(false)
        setLoadingAllRuns(false)
      }
    }
  }, [
    applyCompleteRunLedger,
    loadingMoreRequest,
    loadingMoreRuns,
    refreshing,
    runPage.nextCursor,
    runs,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setLoadingAllRuns,
    setLoadingMoreRuns,
    setRunsError,
  ])
}
