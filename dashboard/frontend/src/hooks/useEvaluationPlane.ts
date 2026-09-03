import { useEvaluationPlaneHookState } from './evaluationPlaneHookState'
import {
  useApplyEvaluationRunLedger,
  useEvaluationPlanePolling,
  useEvaluationPlaneRefresh,
  useEvaluationRunsRefresh,
} from './useEvaluationPlaneLoading'
import { useEvaluationPlaneMutations } from './useEvaluationPlaneMutations'
import { useLoadAllEvaluationRuns, useLoadMoreEvaluationRuns } from './useEvaluationPlanePagination'

export function useEvaluationPlane() {
  const state = useEvaluationPlaneHookState()
  const applyRunLedger = useApplyEvaluationRunLedger(state)
  const refresh = useEvaluationPlaneRefresh(state, applyRunLedger)
  const refreshRuns = useEvaluationRunsRefresh(state, applyRunLedger)
  const loadMoreRuns = useLoadMoreEvaluationRuns(state, applyRunLedger)
  const loadAllRuns = useLoadAllEvaluationRuns(state)
  useEvaluationPlanePolling(state, refresh, refreshRuns)
  const mutations = useEvaluationPlaneMutations(state, refreshRuns)

  return {
    catalog: state.catalog,
    runs: state.runs,
    runsLoaded: state.runsLoaded,
    runLedgerComplete: state.runLedgerComplete,
    runLedgerWarnings: state.runLedgerWarnings,
    runLedgerWarningCount: state.runPage.warningCount,
    totalRuns: state.runPage.totalRuns,
    hasMoreRuns: Boolean(state.runPage.nextCursor),
    loading: state.loadPending.catalog || state.loadPending.runs,
    refreshing: state.refreshing,
    loadingMoreRuns: state.loadingMoreRuns,
    loadingAllRuns: state.loadingAllRuns,
    runPollingPaused: state.runPollingPaused,
    error: state.catalogError || state.runsError,
    catalogError: state.catalogError,
    runsError: state.runsError,
    lastUpdatedAt: state.lastUpdatedAt,
    mutationPending: state.mutationPending,
    mutationKey: state.mutationKey,
    mutationError: state.mutationError,
    clearMutationError: () => state.setMutationError(null),
    refresh: () => refresh(true),
    refreshRuns,
    loadMoreRuns,
    loadAllRuns,
    ...mutations,
  }
}
