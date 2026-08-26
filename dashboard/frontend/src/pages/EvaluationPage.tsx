import { useState, useCallback, useEffect } from 'react'
import type { EvaluationTask, CreateTaskRequest } from '../types/evaluation'
import { useTasks, useTaskMutations, useResults } from '../hooks/useEvaluation'
import {
  TaskList,
  TaskCreationForm,
  ProgressTracker,
  ReportViewer,
  HistoricalResults,
} from '../components/evaluation'
import ConfirmDialog from '../components/ConfirmDialog'
import ProductIcon, { type ProductIconName } from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canRunEvaluation, canWriteEvaluation } from '../utils/accessControl'
import styles from './EvaluationPage.module.css'

type TabType = 'tasks' | 'create' | 'progress' | 'report' | 'history'

interface TabState {
  active: TabType
  selectedTaskId: string | null
}

export function EvaluationPage() {
  const { user } = useAuth()
  const { serverReadonly, isLoading: readonlyLoading } = useReadonly()
  const evaluationMutationsAllowed = !readonlyLoading && !serverReadonly
  const canWrite = evaluationMutationsAllowed && canWriteEvaluation(user)
  const canRun = evaluationMutationsAllowed && canRunEvaluation(user)
  const { tasks, loading: tasksLoading, error: tasksError, refresh: refreshTasks } = useTasks(true)
  const {
    loading: mutationLoading,
    error: mutationError,
    createTask,
    runTask,
    cancelTask,
    deleteTask,
    clearError,
  } = useTaskMutations()

  const [tabState, setTabState] = useState<TabState>({ active: 'tasks', selectedTaskId: null })
  const [deleteTarget, setDeleteTarget] = useState<EvaluationTask | null>(null)
  const [cancelTarget, setCancelTarget] = useState<EvaluationTask | null>(null)

  // Fetch results when viewing a task's report
  const {
    results: selectedResults,
    loading: resultsLoading,
    error: resultsError,
    refresh: refreshResults,
  } = useResults(tabState.active === 'report' ? tabState.selectedTaskId : null)

  useEffect(() => {
    if (!canWrite && tabState.active === 'create') {
      setTabState({ active: 'tasks', selectedTaskId: null })
    }
    if (!canWrite) setDeleteTarget(null)
    if (!canRun) setCancelTarget(null)
  }, [canRun, canWrite, tabState.active])

  const handleViewTask = useCallback((task: EvaluationTask) => {
    if (task.status === 'pending' || task.status === 'running') {
      setTabState({ active: 'progress', selectedTaskId: task.id })
    } else if (
      task.status === 'completed' ||
      task.status === 'failed' ||
      task.status === 'cancelled'
    ) {
      setTabState({ active: 'report', selectedTaskId: task.id })
    }
  }, [])

  const handleRunTask = useCallback(
    async (task: EvaluationTask) => {
      if (!canRun) return
      const success = await runTask(task.id)
      if (success) {
        setTabState({ active: 'progress', selectedTaskId: task.id })
        void refreshTasks()
      }
    },
    [canRun, runTask, refreshTasks],
  )

  const handleCancelTask = useCallback(
    (task: EvaluationTask) => {
      if (!canRun) return
      setCancelTarget(task)
    },
    [canRun],
  )

  const confirmCancelTask = useCallback(async () => {
    if (!cancelTarget || !canRun) return
    const cancelled = await cancelTask(cancelTarget.id)
    if (cancelled) {
      setCancelTarget(null)
      void refreshTasks()
      setTabState({ active: 'tasks', selectedTaskId: null })
    }
  }, [canRun, cancelTarget, cancelTask, refreshTasks])

  const handleDeleteTask = useCallback(
    (task: EvaluationTask) => {
      if (!canWrite) return
      setDeleteTarget(task)
    },
    [canWrite],
  )

  const confirmDeleteTask = useCallback(async () => {
    if (!deleteTarget || !canWrite) return
    const deleted = await deleteTask(deleteTarget.id)
    if (deleted) {
      setDeleteTarget(null)
      void refreshTasks()
    }
  }, [canWrite, deleteTarget, deleteTask, refreshTasks])

  const handleCreateTask = useCallback(
    async (request: CreateTaskRequest) => {
      if (!canWrite) return
      const task = await createTask(request)
      if (task) {
        void refreshTasks()
        setTabState({ active: 'tasks', selectedTaskId: task.id })
      }
    },
    [canWrite, createTask, refreshTasks],
  )

  const handleCancelCreate = useCallback(() => {
    setTabState({ active: 'tasks', selectedTaskId: null })
  }, [])

  const handleProgressComplete = useCallback(() => {
    void refreshTasks()
    if (tabState.selectedTaskId) {
      setTabState({ active: 'report', selectedTaskId: tabState.selectedTaskId })
    }
  }, [refreshTasks, tabState.selectedTaskId])

  const handleBackFromReport = useCallback(() => {
    setTabState({ active: 'tasks', selectedTaskId: null })
  }, [])

  const handleViewHistoricalResults = useCallback((task: EvaluationTask) => {
    setTabState({ active: 'report', selectedTaskId: task.id })
  }, [])

  const tabs: Array<{ id: 'tasks' | 'create' | 'history'; label: string; icon: ProductIconName }> =
    [
      { id: 'tasks', label: 'Tasks', icon: 'evaluation' },
      ...(canWrite ? [{ id: 'create' as const, label: 'Create', icon: 'plus' as const }] : []),
      { id: 'history', label: 'History', icon: 'chart' },
    ]

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h1>Evaluation</h1>
          <p>
            Evaluate the Mixture-of-Models across multiple dimensions at Signal and System Level.
          </p>
        </div>
      </div>

      {mutationError && (
        <div className={styles.errorBanner}>
          <span>{mutationError}</span>
          <button type="button" onClick={clearError}>
            <ProductIcon name="close" /> Dismiss
          </button>
        </div>
      )}

      {!readonlyLoading && serverReadonly ? (
        <div className={styles.readonlyBanner} role="status">
          Evaluation history remains available, but the server-wide read-only policy disables task
          creation, deletion, runs, and cancellation.
        </div>
      ) : null}

      {tabState.active === 'progress' && tabState.selectedTaskId && (
        <div className={styles.progressView}>
          <button
            type="button"
            className={styles.backButton}
            onClick={() => setTabState({ active: 'tasks', selectedTaskId: null })}
          >
            <ProductIcon name="arrow-left" /> Back to Tasks
          </button>
          <ProgressTracker
            taskId={tabState.selectedTaskId}
            onComplete={handleProgressComplete}
            onCancel={
              canRun
                ? () => {
                    const task = tasks.find((t) => t.id === tabState.selectedTaskId)
                    if (task) handleCancelTask(task)
                  }
                : undefined
            }
          />
        </div>
      )}

      {tabState.active === 'report' &&
        (selectedResults ? (
          <ReportViewer results={selectedResults} onBack={handleBackFromReport} />
        ) : (
          <div className={styles.progressView}>
            <button type="button" className={styles.backButton} onClick={handleBackFromReport}>
              <ProductIcon name="arrow-left" /> Back to Tasks
            </button>
            {resultsLoading ? (
              <ProductLoadingState compact label="Loading evaluation results" />
            ) : (
              <div
                className={`${styles.asyncState} ${resultsError ? styles.asyncStateError : ''}`}
                role={resultsError ? 'alert' : 'status'}
              >
                <h2>{resultsError ? 'Results are unavailable' : 'No results available'}</h2>
                <p>
                  {resultsError || 'No result payload was returned for this evaluation.'}
                </p>
                {resultsError ? (
                  <button type="button" onClick={() => void refreshResults()}>
                    <ProductIcon name="refresh" /> Retry
                  </button>
                ) : null}
              </div>
            )}
          </div>
        ))}

      {tabState.active !== 'progress' && tabState.active !== 'report' && (
        <>
          <div className={styles.tabs} role="tablist" aria-label="Evaluation views">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                id={`evaluation-tab-${tab.id}`}
                type="button"
                role="tab"
                aria-selected={tabState.active === tab.id}
                aria-controls={`evaluation-panel-${tab.id}`}
                className={`${styles.tab} ${tabState.active === tab.id ? styles.activeTab : ''}`}
                onClick={() => setTabState({ active: tab.id, selectedTaskId: null })}
              >
                <ProductIcon className={styles.tabIcon} name={tab.icon} aria-hidden="true" />
                <span className={styles.tabLabel}>{tab.label}</span>
              </button>
            ))}
          </div>

          <div
            id={`evaluation-panel-${tabState.active}`}
            className={styles.tabContent}
            role="tabpanel"
            aria-labelledby={`evaluation-tab-${tabState.active}`}
          >
            {tabState.active === 'tasks' && (
              <TaskList
                tasks={tasks}
                loading={tasksLoading || mutationLoading}
                error={tasksError}
                onView={handleViewTask}
                onRun={handleRunTask}
                onCancel={handleCancelTask}
                onDelete={handleDeleteTask}
                onRefresh={refreshTasks}
                canRunTasks={canRun}
                canDeleteTasks={canWrite}
                canCreateTasks={canWrite}
              />
            )}

            {tabState.active === 'create' && canWrite && (
              <TaskCreationForm
                onSubmit={handleCreateTask}
                onCancel={handleCancelCreate}
                loading={mutationLoading}
              />
            )}

            {tabState.active === 'history' && (
              <HistoricalResults
                tasks={tasks}
                loading={tasksLoading}
                error={tasksError}
                onRefresh={refreshTasks}
                onViewResults={handleViewHistoricalResults}
              />
            )}
          </div>
        </>
      )}

      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title={`Delete ${deleteTarget?.name || 'this evaluation'}?`}
        description="The task definition and its dashboard history will be removed. Export any results you still need before continuing."
        eyebrow="Evaluation lifecycle"
        confirmLabel="Delete evaluation"
        pending={mutationLoading}
        details={deleteTarget ? <code>{deleteTarget.id}</code> : null}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={confirmDeleteTask}
      />
      <ConfirmDialog
        isOpen={cancelTarget !== null}
        title={`Cancel ${cancelTarget?.name || 'this evaluation'}?`}
        description="The running evaluation will stop at its current step. Partial result records may remain available."
        eyebrow="Evaluation run"
        confirmLabel="Cancel evaluation"
        pending={mutationLoading}
        tone="warning"
        details={cancelTarget ? <code>{cancelTarget.id}</code> : null}
        onCancel={() => setCancelTarget(null)}
        onConfirm={confirmCancelTask}
      />
    </div>
  )
}

export default EvaluationPage
