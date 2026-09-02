import type {
  EvaluationCatalog,
  EvaluationChangeProfileId,
  EvaluationRun,
} from '../../types/evaluationPlane'
import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaign as EvaluationCampaignResource,
} from '../../types/evaluationCampaign'
import EvaluationCampaignBuilder from './EvaluationCampaignBuilder'
import EvaluationCampaignDecision from './EvaluationCampaignDecision'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { EvaluationActionButton } from './EvaluationPrimitives'
import useEvaluationCampaignBuilder from './useEvaluationCampaignBuilder'
import styles from './EvaluationCampaign.module.css'

interface EvaluationCampaignProps {
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
  campaign: EvaluationCampaignResource | null
  campaignLoading: boolean
  campaignError: string | null
  activeControlledPairID: string | null
  activeControlledPairProfileID: EvaluationChangeProfileId | null
  onLoadAllRuns: () => void
  onRefreshRuns: () => boolean | Promise<boolean>
  onControlledPairIdentityChange: (
    pairID: string | null,
    profileID: EvaluationChangeProfileId | null,
  ) => void
  onCreate: (request: CreateEvaluationCampaignPayload) => Promise<EvaluationCampaignResource | null>
  onClearCreateError: () => void
  onRetryCampaign: () => void
  onClearCampaign: () => void
}

export default function EvaluationCampaign(props: EvaluationCampaignProps) {
  const builder = useEvaluationCampaignBuilder({
    catalog: props.catalog,
    runs: props.runs,
    runLedgerAvailable: props.runLedgerAvailable,
    runLedgerComplete: props.runLedgerComplete,
    allRunsLoaded: props.allRunsLoaded,
    lockedChangeProfile: props.activeControlledPairProfileID,
    onClearCreateError: props.onClearCreateError,
  })

  const startAnother = () => {
    builder.reset()
    props.onClearCreateError()
    props.onClearCampaign()
  }

  return (
    <div className={styles.campaign} aria-busy={props.createPending || props.campaignLoading}>
      {props.campaignLoading ? (
        <div className={styles.inlineNotice} role="status">
          <div>
            <strong>Loading release decision</strong>
            <span>Every selected result is being verified again before the decision is shown.</span>
          </div>
        </div>
      ) : null}
      {props.campaignError ? (
        <div className={styles.inlineError} role="alert">
          <div>
            <strong>Release decision could not be loaded</strong>
            <span>Retry to load the saved decision and its verified evidence.</span>
            <EvaluationIssueDetails
              issues={[{ label: 'Release decision request', message: props.campaignError }]}
            />
          </div>
          <EvaluationActionButton
            type="button"
            compact
            disabled={props.campaignLoading}
            onClick={props.onRetryCampaign}
          >
            {props.campaignLoading ? 'Retrying decision…' : 'Retry decision'}
          </EvaluationActionButton>
        </div>
      ) : null}
      {props.campaign ? (
        <EvaluationCampaignDecision
          campaign={props.campaign}
          runs={props.runs}
          onStartAnother={startAnother}
        />
      ) : null}
      {!props.campaign && !props.campaignLoading && !props.campaignError ? (
        <EvaluationCampaignBuilder
          catalog={props.catalog}
          runs={props.runs}
          totalRuns={props.totalRuns}
          runLedgerAvailable={props.runLedgerAvailable}
          runLedgerComplete={props.runLedgerComplete}
          allRunsLoaded={props.allRunsLoaded}
          loadingAllRuns={props.loadingAllRuns}
          canCreate={props.canCreate}
          createPending={props.createPending}
          createError={props.createError}
          activeControlledPairID={props.activeControlledPairID}
          model={builder}
          onLoadAllRuns={props.onLoadAllRuns}
          onRefreshRuns={props.onRefreshRuns}
          onControlledPairIdentityChange={props.onControlledPairIdentityChange}
          onCreate={props.onCreate}
          onClearCreateError={props.onClearCreateError}
        />
      ) : null}
    </div>
  )
}
