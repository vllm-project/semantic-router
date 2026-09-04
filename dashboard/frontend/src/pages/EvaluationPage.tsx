import { useCallback, useEffect, useMemo, useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import DashboardSurfaceHero from '../components/DashboardSurfaceHero'
import ProductLoadingState from '../components/ProductLoadingState'
import EvaluationCompare from '../components/evaluation-plane/EvaluationCompare'
import EvaluationExperimentForm from '../components/evaluation-plane/EvaluationExperimentForm'
import EvaluationNavigation, {
  type EvaluationView,
} from '../components/evaluation-plane/EvaluationNavigation'
import EvaluationOverview from '../components/evaluation-plane/EvaluationOverview'
import EvaluationReports from '../components/evaluation-plane/EvaluationReports'
import EvaluationRuns from '../components/evaluation-plane/EvaluationRuns'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import {
  useEvaluationComparison,
  useEvaluationPlane,
  useEvaluationReport,
  useEvaluationRunEvents,
} from '../hooks/useEvaluationPlane'
import type { CreateEvaluationRunRequest, EvaluationRun } from '../types/evaluationPlane'
import { canRunEvaluation, canWriteEvaluation } from '../utils/accessControl'
import styles from './EvaluationPage.module.css'

export function EvaluationPage() {
  const { user } = useAuth()
  const { serverReadonly, isLoading: readonlyLoading } = useReadonly()
  const mutationsAllowed = !readonlyLoading && !serverReadonly
  const canWrite = mutationsAllowed && canWriteEvaluation(user)
  const canRun = mutationsAllowed && canRunEvaluation(user)
  const plane = useEvaluationPlane()
  const [activeView, setActiveView] = useState<EvaluationView>('overview')
  const [selectedRunID, setSelectedRunID] = useState<string | null>(null)
  const [reportRunID, setReportRunID] = useState('')
  const [baselineRunID, setBaselineRunID] = useState('')
  const [candidateRunID, setCandidateRunID] = useState('')
  const [cancelTarget, setCancelTarget] = useState<EvaluationRun | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<EvaluationRun | null>(null)

  const completedRuns = useMemo(
    () => plane.runs.filter((run) => run.status === 'completed'),
    [plane.runs],
  )
  const latestCompletedID = completedRuns[0]?.id || null
  const latestReportState = useEvaluationReport(
    activeView === 'overview' ? latestCompletedID : null,
  )
  const reportState = useEvaluationReport(reportRunID || null)
  const comparisonState = useEvaluationComparison(baselineRunID, candidateRunID)
  const selectedRun = plane.runs.find((run) => run.id === selectedRunID) || null
  const eventState = useEvaluationRunEvents(selectedRun, plane.refreshRuns)

  useEffect(() => {
    if (activeView === 'reports' && !reportRunID && latestCompletedID) {
      setReportRunID(latestCompletedID)
    }
  }, [activeView, latestCompletedID, reportRunID])

  useEffect(() => {
    if (!candidateRunID && completedRuns[0]) setCandidateRunID(completedRuns[0].id)
    if (!baselineRunID && completedRuns[1]) setBaselineRunID(completedRuns[1].id)
  }, [baselineRunID, candidateRunID, completedRuns])

  useEffect(() => {
    if (selectedRunID && !plane.runs.some((run) => run.id === selectedRunID)) {
      setSelectedRunID(null)
    }
  }, [plane.runs, selectedRunID])

  useEffect(() => {
    if (!canRun) setCancelTarget(null)
    if (!canWrite) setDeleteTarget(null)
  }, [canRun, canWrite])

  const createRun = useCallback(
    async (request: CreateEvaluationRunRequest) => {
      if (!canWrite || (request.auto_start && !canRun)) return false
      const pendingRun = await plane.createRun({ ...request, auto_start: false })
      if (!pendingRun) return false
      setSelectedRunID(pendingRun.id)
      setActiveView('runs')
      if (request.auto_start) {
        const startedRun = await plane.startRun(pendingRun.id)
        if (startedRun) setSelectedRunID(startedRun.id)
      }
      return true
    },
    [canRun, canWrite, plane],
  )

  const openReport = useCallback((run: EvaluationRun) => {
    setReportRunID(run.id)
    setActiveView('reports')
  }, [])

  const confirmCancel = useCallback(async () => {
    if (!cancelTarget || !canRun) return
    const run = await plane.cancelRun(cancelTarget.id)
    if (run) {
      setCancelTarget(null)
      setSelectedRunID(run.id)
    }
  }, [canRun, cancelTarget, plane])

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget || !canWrite) return
    if (await plane.deleteRun(deleteTarget.id)) {
      setDeleteTarget(null)
      if (reportRunID === deleteTarget.id) setReportRunID('')
    }
  }, [canWrite, deleteTarget, plane, reportRunID])

  return (
    <div className={styles.container}>
      <DashboardSurfaceHero
        eyebrow="Evaluation plane"
        title="Evaluation"
        description="Measure routing recipes, model pools, and end-to-end intelligence with reproducible evidence and promotion gates."
        meta={[
          { label: 'Tracks', value: '8' },
          { label: 'Modes', value: 'Replay · Live' },
          { label: 'Evidence', value: 'E0–E5' },
        ]}
      />

      {!readonlyLoading && serverReadonly ? (
        <div className={styles.readonlyBanner} role="status">
          Evaluation evidence remains readable, but the server-wide read-only policy disables run
          creation, execution, cancellation, and deletion.
        </div>
      ) : null}
      {plane.mutationError ? (
        <div className={styles.errorBanner} role="alert">
          <span>{plane.mutationError}</span>
          <button type="button" onClick={plane.clearMutationError}>
            Dismiss
          </button>
        </div>
      ) : null}

      <EvaluationNavigation active={activeView} onChange={setActiveView} />

      <main
        id={`evaluation-panel-${activeView}`}
        role="tabpanel"
        aria-labelledby={`evaluation-tab-${activeView}`}
      >
        {plane.loading ? (
          <div className={styles.loading}>
            <ProductLoadingState label="Loading evaluation plane" />
          </div>
        ) : null}
        {!plane.loading && plane.error && !plane.catalog ? (
          <div className={styles.loadError} role="alert">
            <h2>Evaluation plane unavailable</h2>
            <p>{plane.error}</p>
            <button type="button" onClick={plane.refresh}>
              Retry
            </button>
          </div>
        ) : null}
        {!plane.loading && plane.catalog ? (
          <>
            {activeView === 'overview' ? (
              <EvaluationOverview
                catalog={plane.catalog}
                runs={plane.runs}
                latestReport={latestReportState.report}
                reportLoading={latestReportState.loading}
                onNavigate={setActiveView}
              />
            ) : null}
            {activeView === 'new' ? (
              <EvaluationExperimentForm
                catalog={plane.catalog}
                runs={plane.runs}
                canCreate={canWrite}
                canAutoStart={canWrite && canRun}
                pending={plane.mutationPending}
                onSubmit={createRun}
              />
            ) : null}
            {activeView === 'runs' ? (
              <EvaluationRuns
                runs={plane.runs}
                selectedRunID={selectedRunID}
                events={eventState.events}
                eventsConnected={eventState.connected}
                eventsError={eventState.error}
                canRun={canRun}
                canDelete={canWrite}
                pending={plane.mutationPending}
                onSelect={(run) => setSelectedRunID(run.id)}
                onStart={(run) => void plane.startRun(run.id)}
                onCancel={setCancelTarget}
                onDelete={setDeleteTarget}
                onOpenReport={openReport}
                onRefresh={() => void plane.refreshRuns()}
              />
            ) : null}
            {activeView === 'reports' ? (
              <EvaluationReports
                runs={plane.runs}
                selectedRunID={reportRunID}
                report={reportState.report}
                loading={reportState.loading}
                error={reportState.error}
                onSelect={setReportRunID}
                onRetry={() => void reportState.refresh()}
              />
            ) : null}
            {activeView === 'compare' ? (
              <EvaluationCompare
                runs={plane.runs}
                baselineID={baselineRunID}
                candidateID={candidateRunID}
                comparison={comparisonState.comparison}
                loading={comparisonState.loading}
                error={comparisonState.error}
                onBaselineChange={setBaselineRunID}
                onCandidateChange={setCandidateRunID}
                onCompare={() => void comparisonState.compare()}
              />
            ) : null}
          </>
        ) : null}
      </main>

      <ConfirmDialog
        isOpen={cancelTarget !== null}
        title={`Cancel ${cancelTarget?.name || 'this run'}?`}
        description="Execution will stop after the active step. Partial evidence remains explicit and unavailable gates will not count as passed."
        eyebrow="Evaluation execution"
        confirmLabel="Cancel run"
        tone="warning"
        pending={plane.mutationPending}
        details={cancelTarget ? <code>{cancelTarget.id}</code> : null}
        onCancel={() => setCancelTarget(null)}
        onConfirm={confirmCancel}
      />
      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title={`Delete ${deleteTarget?.name || 'this run'}?`}
        description="The run snapshot, report index, and dashboard history will be removed. Preserve any required artifacts first."
        eyebrow="Evaluation evidence"
        confirmLabel="Delete run"
        pending={plane.mutationPending}
        details={deleteTarget ? <code>{deleteTarget.id}</code> : null}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={confirmDelete}
      />
    </div>
  )
}

export default EvaluationPage
