import type { EvaluationChangeProfileId } from '../types/evaluationPlane'
import { isPortableEvaluationID } from '../utils/evaluationContractValidation'
import { isCanonicalEvaluationRunID } from '../utils/evaluationRunContract'

export type EvaluationView = 'overview' | 'new' | 'runs' | 'reports' | 'compare'

interface EvaluationWorkflowRoute {
  controlledPairID: string | null
  controlledPairProfileID: EvaluationChangeProfileId | null
}

export type EvaluationRoute = (
  | { view: 'overview' }
  | { view: 'new'; entrypoint: string | null }
  | { view: 'runs'; runID: string | null }
  | { view: 'reports'; reportRunID: string | null }
  | {
      view: 'compare'
      baselineRunID: string | null
      candidateRunID: string | null
      campaignID: string | null
    }
) &
  EvaluationWorkflowRoute

const VIEWS = new Set<EvaluationView>(['overview', 'new', 'runs', 'reports', 'compare'])

function value(params: URLSearchParams, key: string): string | null {
  return params.get(key)?.trim() || null
}

export function parseEvaluationRoute(params: URLSearchParams): EvaluationRoute {
  const requestedView = value(params, 'view')
  const view =
    requestedView && VIEWS.has(requestedView as EvaluationView)
      ? (requestedView as EvaluationView)
      : 'overview'
  const requestedPairID = value(params, 'controlled_pair')
  const requestedPairProfileID = value(params, 'controlled_pair_profile')
  const controlledPairRouteValid = Boolean(
    requestedPairID &&
      isCanonicalEvaluationRunID(requestedPairID) &&
      requestedPairProfileID &&
      isPortableEvaluationID(requestedPairProfileID),
  )
  const controlledPairID = controlledPairRouteValid ? requestedPairID : null
  const controlledPairProfileID = controlledPairRouteValid
    ? (requestedPairProfileID as EvaluationChangeProfileId)
    : null

  switch (view) {
    case 'new':
      return {
        view,
        entrypoint: value(params, 'entrypoint'),
        controlledPairID,
        controlledPairProfileID,
      }
    case 'runs':
      return { view, runID: value(params, 'run'), controlledPairID, controlledPairProfileID }
    case 'reports':
      return {
        view,
        reportRunID: value(params, 'report'),
        controlledPairID,
        controlledPairProfileID,
      }
    case 'compare':
      return {
        view,
        baselineRunID: value(params, 'baseline'),
        candidateRunID: value(params, 'candidate'),
        campaignID: value(params, 'campaign'),
        controlledPairID,
        controlledPairProfileID,
      }
    default:
      return { view: 'overview', controlledPairID, controlledPairProfileID }
  }
}

export function serializeEvaluationRoute(route: EvaluationRoute): URLSearchParams {
  const params = new URLSearchParams()
  if (route.controlledPairID && route.controlledPairProfileID) {
    params.set('controlled_pair', route.controlledPairID)
    params.set('controlled_pair_profile', route.controlledPairProfileID)
  }
  if (route.view === 'overview') return params
  params.set('view', route.view)
  if (route.view === 'new' && route.entrypoint) params.set('entrypoint', route.entrypoint)
  if (route.view === 'runs' && route.runID) params.set('run', route.runID)
  if (route.view === 'reports' && route.reportRunID) params.set('report', route.reportRunID)
  if (route.view === 'compare') {
    if (route.baselineRunID) params.set('baseline', route.baselineRunID)
    if (route.candidateRunID) params.set('candidate', route.candidateRunID)
    if (route.campaignID) params.set('campaign', route.campaignID)
  }
  return params
}

export function removeEvaluationRun(route: EvaluationRoute, runID: string): EvaluationRoute {
  switch (route.view) {
    case 'runs':
      return route.runID === runID ? { ...route, runID: null } : route
    case 'reports':
      return route.reportRunID === runID ? { ...route, reportRunID: null } : route
    case 'compare':
      return route.baselineRunID === runID || route.candidateRunID === runID
        ? {
            view: 'compare',
            baselineRunID: null,
            candidateRunID: null,
            campaignID: route.campaignID,
            controlledPairID: route.controlledPairID,
            controlledPairProfileID: route.controlledPairProfileID,
          }
        : route
    default:
      return route
  }
}
