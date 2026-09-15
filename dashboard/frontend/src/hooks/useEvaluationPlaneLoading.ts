import { useCallback, useEffect } from 'react'

import { getEvaluationCatalog, listEvaluationRuns } from '../utils/evaluationPlaneApi'
import type { EvaluationPlaneHookState } from './evaluationPlaneHookState'
import {
  evaluationErrorMessage,
  mergeEvaluationRuns,
  sortEvaluationRuns,
} from './evaluationPlaneHookSupport'

type EvaluationRunLedger = Awaited<ReturnType<typeof listEvaluationRuns>>

function abortActiveEvaluationRequests(
  catalogController: EvaluationPlaneHookState['catalogController'],
  runsController: EvaluationPlaneHookState['runsController'],
): void {
  catalogController.current?.abort()
  runsController.current?.abort()
}

export function useApplyEvaluationRunLedger(state: EvaluationPlaneHookState) {
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
    (ledger: EvaluationRunLedger, append: boolean) => {
      setRuns((current) =>
        append ? mergeEvaluationRuns(current, ledger.runs) : sortEvaluationRuns(ledger.runs),
      )
      setRunsLoaded(true)
      setRunLedgerComplete(ledger.ledger_complete)
      setRunLedgerWarnings(ledger.warnings)
      setRunPage({
        nextCursor: ledger.next_cursor || null,
        totalRuns: ledger.total_runs,
        warningCount: ledger.warning_count,
      })
      setRunsError(null)
      setLastUpdatedAt(new Date())
      loadedPageCount.current = append ? loadedPageCount.current + 1 : 1
      setRunPollingPaused(append)
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

export function useEvaluationPlaneRefresh(
  state: EvaluationPlaneHookState,
  applyRunLedger: (ledger: EvaluationRunLedger, append: boolean) => void,
) {
  const {
    catalogController,
    catalogRequestVersion,
    loadingMoreRequest,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setCatalog,
    setCatalogError,
    setLoadPending,
    setLoadingAllRuns,
    setLoadingMoreRuns,
    setRefreshing,
    setRunsError,
  } = state
  return useCallback(
    async (showLoading = false) => {
      const catalogVersion = ++catalogRequestVersion.current
      const runsVersion = ++runsRequestVersion.current
      abortActiveEvaluationRequests(catalogController, runsController)
      runsRefreshPromise.current = null
      loadingMoreRequest.current = false
      setLoadingMoreRuns(false)
      setLoadingAllRuns(false)
      const nextCatalogController = new AbortController()
      const nextRunsController = new AbortController()
      catalogController.current = nextCatalogController
      runsController.current = nextRunsController
      if (showLoading) setLoadPending({ catalog: true, runs: true })
      else setRefreshing(true)
      const catalogRequest = getEvaluationCatalog(nextCatalogController.signal)
        .then((nextCatalog) => {
          if (
            nextCatalogController.signal.aborted ||
            catalogVersion !== catalogRequestVersion.current
          )
            return
          setCatalog(nextCatalog)
          setCatalogError(null)
        })
        .catch((reason: unknown) => {
          if (
            nextCatalogController.signal.aborted ||
            catalogVersion !== catalogRequestVersion.current
          )
            return
          setCatalogError(evaluationErrorMessage(reason, 'Failed to load the evaluation catalog.'))
        })
        .finally(() => {
          if (catalogVersion === catalogRequestVersion.current) {
            setLoadPending((current) => ({ ...current, catalog: false }))
          }
        })
      const runsRequest = listEvaluationRuns({ signal: nextRunsController.signal })
        .then((ledger) => {
          if (nextRunsController.signal.aborted || runsVersion !== runsRequestVersion.current)
            return
          applyRunLedger(ledger, false)
        })
        .catch((reason: unknown) => {
          if (nextRunsController.signal.aborted || runsVersion !== runsRequestVersion.current)
            return
          setRunsError(evaluationErrorMessage(reason, 'Failed to load evaluation runs.'))
        })
        .finally(() => {
          if (runsVersion === runsRequestVersion.current) {
            setLoadPending((current) => ({ ...current, runs: false }))
          }
        })
      await Promise.allSettled([catalogRequest, runsRequest])
      if (!nextRunsController.signal.aborted && runsVersion === runsRequestVersion.current) {
        setRefreshing(false)
      }
    },
    [
      applyRunLedger,
      catalogController,
      catalogRequestVersion,
      loadingMoreRequest,
      runsController,
      runsRefreshPromise,
      runsRequestVersion,
      setCatalog,
      setCatalogError,
      setLoadPending,
      setLoadingAllRuns,
      setLoadingMoreRuns,
      setRefreshing,
      setRunsError,
    ],
  )
}

export function useEvaluationRunsRefresh(
  state: EvaluationPlaneHookState,
  applyRunLedger: (ledger: EvaluationRunLedger, append: boolean) => void,
) {
  const {
    loadingMoreRequest,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setLoadPending,
    setLoadingAllRuns,
    setLoadingMoreRuns,
    setRefreshing,
    setRunsError,
  } = state
  return useCallback(() => {
    if (runsRefreshPromise.current) return runsRefreshPromise.current
    const version = ++runsRequestVersion.current
    runsController.current?.abort()
    loadingMoreRequest.current = false
    setLoadingMoreRuns(false)
    setLoadingAllRuns(false)
    const controller = new AbortController()
    runsController.current = controller
    setRefreshing(true)
    const pending = (async () => {
      try {
        const ledger = await listEvaluationRuns({ signal: controller.signal })
        if (controller.signal.aborted || version !== runsRequestVersion.current) return false
        applyRunLedger(ledger, false)
        return true
      } catch (refreshError) {
        if (controller.signal.aborted || version !== runsRequestVersion.current) return false
        setRunsError(evaluationErrorMessage(refreshError, 'Failed to refresh evaluation runs.'))
        return false
      } finally {
        if (version === runsRequestVersion.current) {
          setLoadPending((current) => ({ ...current, runs: false }))
          setRefreshing(false)
        }
      }
    })()
    runsRefreshPromise.current = pending
    void pending.finally(() => {
      if (runsRefreshPromise.current === pending) runsRefreshPromise.current = null
    })
    return pending
  }, [
    applyRunLedger,
    loadingMoreRequest,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
    setLoadPending,
    setLoadingAllRuns,
    setLoadingMoreRuns,
    setRefreshing,
    setRunsError,
  ])
}

export function useEvaluationPlanePolling(
  state: EvaluationPlaneHookState,
  refresh: (showLoading?: boolean) => Promise<void>,
  refreshRuns: () => Promise<boolean>,
): void {
  const {
    catalogController,
    catalogRequestVersion,
    loadedPageCount,
    loadingMoreRequest,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
  } = state
  useEffect(() => {
    void refresh(true)
    const pollIfVisible = () => {
      if (!document.hidden && !loadingMoreRequest.current && loadedPageCount.current <= 1) {
        void refreshRuns()
      }
    }
    const interval = window.setInterval(pollIfVisible, 5_000)
    document.addEventListener('visibilitychange', pollIfVisible)
    return () => {
      catalogRequestVersion.current += 1
      runsRequestVersion.current += 1
      abortActiveEvaluationRequests(catalogController, runsController)
      runsRefreshPromise.current = null
      loadingMoreRequest.current = false
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', pollIfVisible)
    }
  }, [
    catalogController,
    catalogRequestVersion,
    loadedPageCount,
    loadingMoreRequest,
    refresh,
    refreshRuns,
    runsController,
    runsRefreshPromise,
    runsRequestVersion,
  ])
}
