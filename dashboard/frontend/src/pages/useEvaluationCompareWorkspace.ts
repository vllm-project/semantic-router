import { useCallback, useEffect, useMemo, useState } from 'react'

import { useEvaluationComparison } from '../hooks/useEvaluationComparison'
import { useEvaluationRun } from '../hooks/useEvaluationRun'
import type { EvaluationRun } from '../types/evaluationPlane'
import type {
  CompareRoute,
  EvaluationCompareWorkspaceProps,
} from './evaluationCompareWorkspaceTypes'

export function useEvaluationCompareWorkspaceRoute({
  route,
  defaultPair,
  hasMoreRuns,
  loadingAllRuns,
  loadingMoreRuns,
  onLoadAllRuns,
  onRouteChange,
}: EvaluationCompareWorkspaceProps) {
  const [releaseDecisionOpen, setReleaseDecisionOpen] = useState(Boolean(route.campaignID))
  const hasRequestedPair = Boolean(route.baselineRunID || route.candidateRunID)
  const baselineRunID = hasRequestedPair ? route.baselineRunID || '' : defaultPair?.baselineID || ''
  const candidateRunID = hasRequestedPair
    ? route.candidateRunID || ''
    : defaultPair?.candidateID || ''
  const updateRoute = useCallback(
    (
      baselineID: string | null,
      candidateID: string | null,
      campaignID: string | null,
      controlledPairID = route.controlledPairID,
      controlledPairProfileID = route.controlledPairProfileID,
    ) => {
      const nextRoute: CompareRoute = {
        view: 'compare',
        baselineRunID: baselineID,
        candidateRunID: candidateID,
        campaignID,
        controlledPairID,
        controlledPairProfileID,
      }
      onRouteChange(nextRoute)
    },
    [onRouteChange, route.controlledPairID, route.controlledPairProfileID],
  )

  useEffect(() => {
    if (route.campaignID && hasMoreRuns && !loadingAllRuns && !loadingMoreRuns) onLoadAllRuns()
  }, [hasMoreRuns, loadingAllRuns, loadingMoreRuns, onLoadAllRuns, route.campaignID])
  useEffect(() => {
    if (route.campaignID) setReleaseDecisionOpen(true)
  }, [route.campaignID])

  return {
    releaseDecisionOpen,
    setReleaseDecisionOpen,
    baselineRunID,
    candidateRunID,
    updateRoute,
  }
}

export function useEvaluationCompareWorkspaceRuns(
  runs: EvaluationRun[],
  baselineRunID: string,
  candidateRunID: string,
  runLedgerComplete: boolean,
) {
  const loadedBaseline = runs.find((run) => run.id === baselineRunID) || null
  const loadedCandidate = runs.find((run) => run.id === candidateRunID) || null
  const baselineState = useEvaluationRun(baselineRunID || null, loadedBaseline)
  const candidateState = useEvaluationRun(candidateRunID || null, loadedCandidate)
  const comparisonState = useEvaluationComparison(baselineRunID, candidateRunID, runLedgerComplete)
  const comparisonRuns = useMemo(() => {
    const byID = new Map(runs.map((run) => [run.id, run]))
    if (baselineState.run) byID.set(baselineState.run.id, baselineState.run)
    if (candidateState.run) byID.set(candidateState.run.id, candidateState.run)
    return [...byID.values()]
  }, [baselineState.run, candidateState.run, runs])
  return { baselineState, candidateState, comparisonState, comparisonRuns }
}
