import EvaluationExperimentForm from '../components/evaluation-plane/EvaluationExperimentForm'
import EvaluationOverview from '../components/evaluation-plane/EvaluationOverview'
import EvaluationReports from '../components/evaluation-plane/EvaluationReports'
import EvaluationRuns from '../components/evaluation-plane/EvaluationRuns'
import EvaluationCompareWorkspace from './EvaluationCompareWorkspace'
import type { EvaluationPageController } from './useEvaluationPageController'

function EvaluationOverviewView({ controller }: { controller: EvaluationPageController }) {
  const { plane, routing, resources } = controller
  if (!plane.catalog) return null
  return (
    <EvaluationOverview
      catalog={plane.catalog}
      runs={plane.runs}
      totalRuns={plane.totalRuns}
      hasMoreRuns={plane.hasMoreRuns}
      loadingMoreRuns={plane.loadingMoreRuns}
      runLedgerAvailable={plane.runsLoaded}
      runLedgerComplete={plane.runLedgerComplete}
      latestReport={resources.latestReportState.report}
      requestedReportRunID={routing.latestCompletedID}
      reportLoading={resources.latestReportState.loading}
      reportError={resources.latestReportState.error}
      onRetryReport={() => void resources.latestReportState.refresh()}
      onLoadMoreRuns={() => void plane.loadMoreRuns()}
      onNavigate={routing.navigate}
      onOpenReport={(id) =>
        routing.setRoute({
          view: 'reports',
          reportRunID: id,
          controlledPairID: routing.route.controlledPairID,
          controlledPairProfileID: routing.route.controlledPairProfileID,
        })
      }
    />
  )
}

function EvaluationNewView({ controller }: { controller: EvaluationPageController }) {
  const { plane, routing, mutations, canWrite, canRun } = controller
  if (!plane.catalog) return null
  return (
    <EvaluationExperimentForm
      catalog={plane.catalog}
      runs={plane.runs}
      totalRuns={plane.totalRuns}
      canCreate={canWrite}
      canAutoStart={canWrite && canRun}
      runLedgerAvailable={plane.runsLoaded}
      runLedgerComplete={plane.runLedgerComplete}
      hasMoreRuns={plane.hasMoreRuns}
      loadingMoreRuns={plane.loadingMoreRuns}
      pending={plane.mutationPending}
      initialEntrypoint={routing.route.view === 'new' ? routing.route.entrypoint : null}
      onLoadMoreRuns={() => void plane.loadMoreRuns()}
      onSubmit={mutations.createRun}
    />
  )
}

function EvaluationRunsView({ controller }: { controller: EvaluationPageController }) {
  const { plane, routing, resources, mutations, canWrite, canRun } = controller
  const { selectedRunState, selectedPairState, eventState } = resources
  return (
    <EvaluationRuns
      runs={plane.runs}
      selectedRunID={resources.selectedRunID}
      selectedRun={selectedRunState.run}
      selectedRunLoading={selectedRunState.loading}
      selectedRunError={selectedRunState.error}
      onRetrySelectedRun={() => void selectedRunState.refresh()}
      selectedPair={selectedPairState.execution}
      selectedPairLoading={selectedPairState.loading}
      selectedPairRefreshing={selectedPairState.refreshing}
      selectedPairError={selectedPairState.error}
      onRetrySelectedPair={() => void selectedPairState.refresh()}
      events={eventState.events}
      eventsConnected={eventState.connected}
      eventsError={eventState.error}
      onReconnectEvents={eventState.retry}
      canRun={canRun}
      canDelete={canWrite}
      refreshing={plane.refreshing}
      loadingMore={plane.loadingMoreRuns}
      runLedgerAvailable={plane.runsLoaded}
      autoRefreshPaused={plane.runPollingPaused}
      totalRuns={plane.totalRuns}
      hasMoreRuns={plane.hasMoreRuns}
      lastUpdatedAt={plane.lastUpdatedAt}
      mutationKey={plane.mutationKey}
      onSelect={(run) =>
        routing.setRoute(
          {
            view: 'runs',
            runID: run.id,
            controlledPairID: routing.route.controlledPairID,
            controlledPairProfileID: routing.route.controlledPairProfileID,
          },
          true,
        )
      }
      onStart={(run) => void plane.startRun(run.id)}
      onCancel={mutations.runActions.requestCancel}
      onDelete={mutations.runActions.requestDelete}
      onOpenReport={mutations.openReport}
      onRefresh={resources.refreshSelectedRun}
      onLoadMore={() => void plane.loadMoreRuns()}
    />
  )
}

function EvaluationReportsView({ controller }: { controller: EvaluationPageController }) {
  const { plane, routing, resources } = controller
  return (
    <EvaluationReports
      runs={plane.runs}
      selectedRunID={resources.reportRunID || ''}
      report={resources.reportState.report}
      loading={resources.reportState.loading}
      runLedgerAvailable={plane.runsLoaded}
      totalRuns={plane.totalRuns}
      hasMoreRuns={plane.hasMoreRuns}
      loadingMoreRuns={plane.loadingMoreRuns}
      error={resources.reportState.error}
      onSelect={(id) =>
        routing.setRoute(
          {
            view: 'reports',
            reportRunID: id,
            controlledPairID: routing.route.controlledPairID,
            controlledPairProfileID: routing.route.controlledPairProfileID,
          },
          true,
        )
      }
      onRetry={() => void resources.reportState.refresh()}
      onLoadMoreRuns={() => void plane.loadMoreRuns()}
    />
  )
}

function EvaluationCompareView({ controller }: { controller: EvaluationPageController }) {
  const { plane, routing, canWrite } = controller
  if (!plane.catalog || routing.route.view !== 'compare') return null
  return (
    <EvaluationCompareWorkspace
      catalog={plane.catalog}
      runs={plane.runs}
      totalRuns={plane.totalRuns}
      runLedgerAvailable={plane.runsLoaded}
      runLedgerComplete={plane.runLedgerComplete}
      hasMoreRuns={plane.hasMoreRuns}
      loadingMoreRuns={plane.loadingMoreRuns}
      loadingAllRuns={plane.loadingAllRuns}
      canCreateCampaign={canWrite}
      route={routing.route}
      defaultPair={routing.defaultPair}
      onRouteChange={(nextRoute) => routing.setRoute(nextRoute, true)}
      onLoadMoreRuns={() => void plane.loadMoreRuns()}
      onLoadAllRuns={() => void plane.loadAllRuns()}
      onRefreshRuns={() => plane.refreshRuns()}
      onCreateRun={() => routing.navigate('new')}
    />
  )
}

export default function EvaluationActiveView({
  controller,
}: {
  controller: EvaluationPageController
}) {
  switch (controller.routing.activeView) {
    case 'overview':
      return <EvaluationOverviewView controller={controller} />
    case 'new':
      return <EvaluationNewView controller={controller} />
    case 'runs':
      return <EvaluationRunsView controller={controller} />
    case 'reports':
      return <EvaluationReportsView controller={controller} />
    case 'compare':
      return <EvaluationCompareView controller={controller} />
  }
}
