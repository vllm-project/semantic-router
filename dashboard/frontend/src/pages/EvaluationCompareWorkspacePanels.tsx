import EvaluationCampaign from '../components/evaluation-plane/EvaluationCampaign'
import EvaluationCompare from '../components/evaluation-plane/EvaluationCompare'
import type {
  useCreateEvaluationCampaign,
  useEvaluationCampaign,
} from '../hooks/useEvaluationCampaign'
import type { EvaluationCompareWorkspaceProps } from './evaluationCompareWorkspaceTypes'
import type {
  useEvaluationCompareWorkspaceRoute,
  useEvaluationCompareWorkspaceRuns,
} from './useEvaluationCompareWorkspace'

type RouteState = ReturnType<typeof useEvaluationCompareWorkspaceRoute>
type RunsState = ReturnType<typeof useEvaluationCompareWorkspaceRuns>
type CampaignState = ReturnType<typeof useEvaluationCampaign>
type CampaignCreateState = ReturnType<typeof useCreateEvaluationCampaign>

interface CampaignPanelProps {
  workspace: EvaluationCompareWorkspaceProps
  routeState: RouteState
  campaignState: CampaignState
  campaignCreateState: CampaignCreateState
}

export function EvaluationCampaignWorkspacePanel({
  workspace,
  routeState,
  campaignState,
  campaignCreateState,
}: CampaignPanelProps) {
  const { route } = workspace
  const campaign =
    campaignState.campaign ||
    (campaignCreateState.campaign?.id === route.campaignID ? campaignCreateState.campaign : null)
  const updateCurrentRoute = (campaignID: string | null) =>
    routeState.updateRoute(
      routeState.baselineRunID || null,
      routeState.candidateRunID || null,
      campaignID,
    )
  return (
    <EvaluationCampaign
      catalog={workspace.catalog}
      runs={workspace.runs}
      totalRuns={workspace.totalRuns}
      runLedgerAvailable={workspace.runLedgerAvailable}
      runLedgerComplete={workspace.runLedgerComplete}
      allRunsLoaded={
        workspace.runLedgerAvailable &&
        !workspace.hasMoreRuns &&
        workspace.runs.length === workspace.totalRuns
      }
      loadingAllRuns={workspace.loadingAllRuns}
      canCreate={workspace.canCreateCampaign}
      createPending={campaignCreateState.pending}
      createError={campaignCreateState.error}
      campaign={campaign}
      campaignLoading={campaignState.loading && !campaign}
      campaignError={campaignState.error}
      activeControlledPairID={route.controlledPairID}
      activeControlledPairProfileID={route.controlledPairProfileID}
      onLoadAllRuns={() =>
        workspace.hasMoreRuns ? workspace.onLoadAllRuns() : workspace.onRefreshRuns()
      }
      onRefreshRuns={workspace.onRefreshRuns}
      onControlledPairIdentityChange={(pairID, profileID) =>
        routeState.updateRoute(
          routeState.baselineRunID || null,
          routeState.candidateRunID || null,
          route.campaignID,
          pairID,
          profileID,
        )
      }
      onCreate={async (request) => {
        const created = await campaignCreateState.create(request)
        if (created) updateCurrentRoute(created.id)
        return created
      }}
      onClearCreateError={campaignCreateState.clearError}
      onRetryCampaign={() => void campaignState.refresh()}
      onClearCampaign={() => {
        campaignCreateState.reset()
        updateCurrentRoute(null)
      }}
    />
  )
}

export function EvaluationRunComparisonPanel({
  workspace,
  routeState,
  runsState,
}: {
  workspace: EvaluationCompareWorkspaceProps
  routeState: RouteState
  runsState: RunsState
}) {
  return (
    <EvaluationCompare
      runs={runsState.comparisonRuns}
      baselineID={routeState.baselineRunID}
      candidateID={routeState.candidateRunID}
      comparison={runsState.comparisonState.comparison}
      runLedgerAvailable={workspace.runLedgerAvailable}
      runLedgerComplete={workspace.runLedgerComplete}
      totalRuns={workspace.totalRuns}
      hasMoreRuns={workspace.hasMoreRuns}
      loadingMoreRuns={workspace.loadingMoreRuns}
      resourcesLoading={runsState.baselineState.loading || runsState.candidateState.loading}
      resourcesError={runsState.baselineState.error || runsState.candidateState.error}
      loading={runsState.comparisonState.loading}
      error={runsState.comparisonState.error}
      onPairChange={(candidate, baseline) =>
        routeState.updateRoute(baseline, candidate, workspace.route.campaignID)
      }
      onCompare={() => void runsState.comparisonState.compare()}
      onLoadMoreRuns={workspace.onLoadMoreRuns}
      onRetryResources={() => {
        void runsState.baselineState.refresh()
        void runsState.candidateState.refresh()
      }}
      onCreateRun={workspace.onCreateRun}
    />
  )
}
