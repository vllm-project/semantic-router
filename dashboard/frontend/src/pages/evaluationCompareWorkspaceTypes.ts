import type { EvaluationCatalog, EvaluationRun } from '../types/evaluationPlane'
import type { EvaluationRoute } from './evaluationRoute'

export type CompareRoute = Extract<EvaluationRoute, { view: 'compare' }>

export interface ComparisonPair {
  baselineID: string | null
  candidateID: string | null
}

export interface EvaluationCompareWorkspaceProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  totalRuns: number
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  loadingAllRuns: boolean
  canCreateCampaign: boolean
  route: CompareRoute
  defaultPair: ComparisonPair | null
  onRouteChange: (route: CompareRoute) => void
  onLoadMoreRuns: () => void
  onLoadAllRuns: () => void
  onRefreshRuns: () => boolean | Promise<boolean>
  onCreateRun: () => void
}
