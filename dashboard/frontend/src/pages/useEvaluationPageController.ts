import { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { useEvaluationControlledPairResource } from '../hooks/useEvaluationControlledPairResource'
import { useEvaluationPlane } from '../hooks/useEvaluationPlane'
import { useEvaluationReport } from '../hooks/useEvaluationReport'
import { useEvaluationRun } from '../hooks/useEvaluationRun'
import { useEvaluationRunEvents } from '../hooks/useEvaluationRunEvents'
import type { EvaluationExperimentIntent, EvaluationRun } from '../types/evaluationPlane'
import { canRunEvaluation, canWriteEvaluation } from '../utils/accessControl'
import { defaultComparisonPair } from '../utils/evaluationComparisonCohort'
import {
  parseEvaluationRoute,
  serializeEvaluationRoute,
  type EvaluationRoute,
  type EvaluationView,
} from './evaluationRoute'
import { useEvaluationRunActions } from './useEvaluationRunActions'

type EvaluationPlane = ReturnType<typeof useEvaluationPlane>

function useEvaluationPageRouting(plane: EvaluationPlane) {
  const [searchParams, setSearchParams] = useSearchParams()
  const route = useMemo(() => parseEvaluationRoute(searchParams), [searchParams])
  const setRoute = useCallback(
    (nextRoute: EvaluationRoute, replace = false) => {
      setSearchParams(serializeEvaluationRoute(nextRoute), { replace })
    },
    [setSearchParams],
  )
  const latestCompletedID = useMemo(
    () => plane.runs.find((run) => run.status === 'completed')?.id || null,
    [plane.runs],
  )
  const defaultPair = useMemo(
    () => (plane.runLedgerComplete ? defaultComparisonPair(plane.runs) : null),
    [plane.runLedgerComplete, plane.runs],
  )
  const navigate = useCallback(
    (view: EvaluationView) => {
      const workflow = {
        controlledPairID: route.controlledPairID,
        controlledPairProfileID: route.controlledPairProfileID,
      }
      const destinations: Partial<Record<EvaluationView, EvaluationRoute>> = {
        new: { view: 'new', entrypoint: null, ...workflow },
        runs: { view: 'runs', runID: null, ...workflow },
        reports: { view: 'reports', reportRunID: latestCompletedID, ...workflow },
        compare: {
          view: 'compare',
          baselineRunID: defaultPair?.baselineID || null,
          candidateRunID: defaultPair?.candidateID || null,
          campaignID: null,
          ...workflow,
        },
      }
      setRoute(destinations[view] || { view: 'overview', ...workflow })
    },
    [defaultPair, latestCompletedID, route, setRoute],
  )
  return { route, activeView: route.view, setRoute, latestCompletedID, defaultPair, navigate }
}

function useEvaluationPageResources(
  plane: EvaluationPlane,
  routing: ReturnType<typeof useEvaluationPageRouting>,
) {
  const { route, activeView, latestCompletedID } = routing
  const selectedRunID = route.view === 'runs' ? route.runID || plane.runs[0]?.id || null : null
  const requestedReportRunID = route.view === 'reports' ? route.reportRunID : null
  const reportRunID = activeView === 'reports' ? requestedReportRunID || latestCompletedID : null
  const latestReportState = useEvaluationReport(
    activeView === 'overview' ? latestCompletedID : null,
  )
  const reportState = useEvaluationReport(reportRunID)
  const loadedSelectedRun = plane.runs.find((run) => run.id === selectedRunID) || null
  const selectedRunState = useEvaluationRun(selectedRunID, loadedSelectedRun)
  const selectedPairState = useEvaluationControlledPairResource(
    selectedRunState.run?.controlled_pair?.pair_id || null,
  )
  const refreshSelectedRunResource = selectedRunState.refresh
  const refreshSelectedPairResource = selectedPairState.refresh
  const refreshRunLedger = plane.refreshRuns
  const refreshSelectedRun = useCallback(() => {
    void refreshSelectedRunResource()
    void refreshSelectedPairResource()
    void refreshRunLedger()
  }, [refreshRunLedger, refreshSelectedPairResource, refreshSelectedRunResource])
  const eventState = useEvaluationRunEvents(selectedRunState.run, refreshSelectedRun)
  return {
    selectedRunID,
    reportRunID,
    latestReportState,
    reportState,
    selectedRunState,
    selectedPairState,
    refreshSelectedRun,
    eventState,
  }
}

function useEvaluationPageMutations(
  plane: EvaluationPlane,
  routing: ReturnType<typeof useEvaluationPageRouting>,
  resources: ReturnType<typeof useEvaluationPageResources>,
  canWrite: boolean,
  canRun: boolean,
) {
  const { route, setRoute } = routing
  const createRun = useCallback(
    async (intent: EvaluationExperimentIntent) => {
      if (!canWrite || (intent.autoStart && !canRun)) return false
      const { autoStart, ...request } = intent
      const pendingRun = await plane.createRun(request)
      if (!pendingRun) return false
      setRoute({
        view: 'runs',
        runID: pendingRun.id,
        controlledPairID: route.controlledPairID,
        controlledPairProfileID: route.controlledPairProfileID,
      })
      if (!autoStart) return true
      return Boolean(await plane.startRun(pendingRun.id))
    },
    [canRun, canWrite, plane, route, setRoute],
  )
  const openReport = useCallback(
    (run: EvaluationRun) => {
      if (run.status !== 'completed') return
      setRoute({
        view: 'reports',
        reportRunID: run.id,
        controlledPairID: route.controlledPairID,
        controlledPairProfileID: route.controlledPairProfileID,
      })
    },
    [route, setRoute],
  )
  const runActions = useEvaluationRunActions({
    canRun,
    canWrite,
    route,
    setRoute,
    operations: plane,
    adoptControlledPair: resources.selectedPairState.adopt,
  })
  return { createRun, openReport, runActions }
}

export function useEvaluationPageController() {
  const { user } = useAuth()
  const { serverReadonly, isLoading: readonlyLoading } = useReadonly()
  const mutationsAllowed = !readonlyLoading && !serverReadonly
  const canWrite = mutationsAllowed && canWriteEvaluation(user)
  const canRun = mutationsAllowed && canRunEvaluation(user)
  const plane = useEvaluationPlane()
  const routing = useEvaluationPageRouting(plane)
  const resources = useEvaluationPageResources(plane, routing)
  const mutations = useEvaluationPageMutations(plane, routing, resources, canWrite, canRun)
  return {
    readonlyLoading,
    serverReadonly,
    canWrite,
    canRun,
    plane,
    routing,
    resources,
    mutations,
  }
}

export type EvaluationPageController = ReturnType<typeof useEvaluationPageController>
