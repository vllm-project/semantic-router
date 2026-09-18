import { useEffect, useRef, type RefObject } from 'react'

import DashboardManagerLayout from '../components/DashboardManagerLayout'
import ProductLoadingState from '../components/ProductLoadingState'
import EvaluationIssueDetails from '../components/evaluation-plane/EvaluationIssueDetails'
import EvaluationNavigation from '../components/evaluation-plane/EvaluationNavigation'
import { EvaluationActionButton } from '../components/evaluation-plane/EvaluationPrimitives'
import styles from './EvaluationPage.module.css'
import EvaluationPageStatus from './EvaluationPageStatus'
import EvaluationActiveView from './EvaluationPageViews'
import EvaluationRunActionDialogs from './EvaluationRunActionDialogs'
import type { EvaluationPageController } from './useEvaluationPageController'

function EvaluationStatus({ controller }: { controller: EvaluationPageController }) {
  const { plane, mutations, readonlyLoading, serverReadonly } = controller
  return (
    <EvaluationPageStatus
      readonlyLoading={readonlyLoading}
      serverReadonly={serverReadonly}
      hasCatalog={plane.catalog !== null}
      catalogError={plane.catalogError}
      runsError={plane.runsError}
      runsLoaded={plane.runsLoaded}
      refreshing={plane.refreshing}
      runLedgerComplete={plane.runLedgerComplete}
      runLedgerWarningCount={plane.runLedgerWarningCount}
      runLedgerWarnings={plane.runLedgerWarnings}
      mutationError={
        mutations.runActions.cancelTarget || mutations.runActions.deleteTarget
          ? null
          : plane.mutationError
      }
      onRefresh={plane.refresh}
      onClearMutationError={plane.clearMutationError}
    />
  )
}

function EvaluationPanel({
  controller,
  panelRef,
}: {
  controller: EvaluationPageController
  panelRef: RefObject<HTMLElement>
}) {
  const { plane, routing } = controller
  return (
    <section
      id="evaluation-panel"
      ref={panelRef}
      role="tabpanel"
      aria-labelledby={`evaluation-tab-${routing.activeView}`}
      tabIndex={-1}
      className={styles.panelRegion}
    >
      {plane.loading ? (
        <div className={styles.loading}>
          <ProductLoadingState label="Loading evaluation" />
        </div>
      ) : null}
      {!plane.loading && plane.catalogError && !plane.catalog ? (
        <div className={styles.loadError} role="alert">
          <h2>Evaluation setup isn’t ready</h2>
          <p>
            The benchmark catalog could not be loaded. Check that the evaluation service has
            finished starting, then retry.
          </p>
          <EvaluationIssueDetails
            issues={[{ label: 'Benchmark catalog', message: plane.catalogError }]}
          />
          <EvaluationActionButton type="button" onClick={plane.refresh}>
            Retry
          </EvaluationActionButton>
        </div>
      ) : null}
      {!plane.loading && plane.catalog ? <EvaluationActiveView controller={controller} /> : null}
    </section>
  )
}

export default function EvaluationPageContent({
  controller,
}: {
  controller: EvaluationPageController
}) {
  const { plane, routing, mutations } = controller
  const panelRef = useRef<HTMLElement>(null)
  const previousView = useRef(routing.activeView)
  useEffect(() => {
    if (previousView.current === routing.activeView) return
    previousView.current = routing.activeView
    panelRef.current?.focus({ preventScroll: true })
  }, [routing.activeView])

  return (
    <div className={styles.pageShell}>
      <DashboardManagerLayout
        eyebrow="Decision quality"
        title="Evaluation"
        description="Understand how a Mixture routes, which models add value, and whether a change is reliable, efficient, and safe to ship."
      >
        <div className={styles.evaluationScope} data-testid="evaluation-scope">
          <EvaluationStatus controller={controller} />
          <EvaluationNavigation active={routing.activeView} onChange={routing.navigate} />
          <EvaluationPanel controller={controller} panelRef={panelRef} />
          <EvaluationRunActionDialogs
            cancelTarget={mutations.runActions.cancelTarget}
            deleteTarget={mutations.runActions.deleteTarget}
            mutationKey={plane.mutationKey}
            error={plane.mutationError}
            returnFocusRef={panelRef}
            cancelReturnFocusMode={mutations.runActions.cancelReturnFocusMode}
            deleteReturnFocusMode={mutations.runActions.deleteReturnFocusMode}
            onCloseCancel={mutations.runActions.closeCancel}
            onCloseDelete={mutations.runActions.closeDelete}
            onConfirmCancel={mutations.runActions.confirmCancel}
            onConfirmDelete={mutations.runActions.confirmDelete}
          />
        </div>
      </DashboardManagerLayout>
    </div>
  )
}
