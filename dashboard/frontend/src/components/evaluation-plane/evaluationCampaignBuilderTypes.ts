import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaign,
} from '../../types/evaluationCampaign'
import type {
  EvaluationCatalog,
  EvaluationChangeProfileId,
  EvaluationRun,
} from '../../types/evaluationPlane'
import type { EvaluationCampaignBuilderModel } from './useEvaluationCampaignBuilder'

export interface EvaluationCampaignBuilderProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  totalRuns: number
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  allRunsLoaded: boolean
  loadingAllRuns: boolean
  canCreate: boolean
  createPending: boolean
  createError: string | null
  activeControlledPairID: string | null
  model: EvaluationCampaignBuilderModel
  onLoadAllRuns: () => void
  onRefreshRuns: () => boolean | Promise<boolean>
  onControlledPairIdentityChange: (
    pairID: string | null,
    profileID: EvaluationChangeProfileId | null,
  ) => void
  onCreate: (request: CreateEvaluationCampaignPayload) => Promise<EvaluationCampaign | null>
  onClearCreateError: () => void
}
