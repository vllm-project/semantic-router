import { useState, useCallback } from 'react';
import type { EvaluationTask, CreateTaskRequest } from '../types/evaluation';
import { useTasks, useTaskMutations, useResults } from '../hooks/useEvaluation';
import {
  TaskList,
  TaskCreationForm,
  ProgressTracker,
  ReportViewer,
  HistoricalResults,
} from '../components/evaluation';
import styles from './EvaluationPage.module.css';

type TabType = 'tasks' | 'create' | 'progress' | 'report' | 'history';

interface TabState {
  active: TabType;
  selectedTaskId: string | null;
}

export function EvaluationPage() {
  const { tasks, loading: tasksLoading, refresh: refreshTasks } = useTasks(true);
  const { loading: mutationLoading, error: mutationError, createTask, runTask, cancelTask, deleteTask, clearError } = useTaskMutations();

  const [tabState, setTabState] = useState<TabState>({ active: 'tasks', selectedTaskId: null });

  // Fetch results when viewing a task's report
  const { results: selectedResults } = useResults(
    tabState.active === 'report' ? tabState.selectedTaskId : null
  );

  const handleViewTask = useCallback((task: EvaluationTask) => {
    if (task.status === 'running') {
      setTabState({ active: 'progress', selectedTaskId: task.id });
    } else if (task.status === 'completed' || task.status === 'failed') {
      setTabState({ active: 'report', selectedTaskId: task.id });
    } else {
      // For pending tasks, just select them
      setTabState({ active: 'tasks', selectedTaskId: task.id });
    }
  }, []);

  const handleRunTask = useCallback(async (task: EvaluationTask) => {
    const success = await runTask(task.id);
    if (success) {
      setTabState({ active: 'progress', selectedTaskId: task.id });
      refreshTasks();
    }
  }, [runTask, refreshTasks]);

  const handleCancelTask = useCallback(async (task: EvaluationTask) => {
    await cancelTask(task.id);
    refreshTasks();
    setTabState({ active: 'tasks', selectedTaskId: null });
  }, [cancelTask, refreshTasks]);

  const handleDeleteTask = useCallback(async (task: EvaluationTask) => {
    if (window.confirm(`Are you sure you want to delete "${task.name}"?`)) {
      await deleteTask(task.id);
      refreshTasks();
    }
  }, [deleteTask, refreshTasks]);

  const handleCreateTask = useCallback(async (request: CreateTaskRequest) => {
    const task = await createTask(request);
    if (task) {
      refreshTasks();
      setTabState({ active: 'tasks', selectedTaskId: task.id });
    }
  }, [createTask, refreshTasks]);

  const handleCancelCreate = useCallback(() => {
    setTabState({ active: 'tasks', selectedTaskId: null });
  }, []);

  const handleProgressComplete = useCallback(() => {
    refreshTasks();
    if (tabState.selectedTaskId) {
      setTabState({ active: 'report', selectedTaskId: tabState.selectedTaskId });
    }
  }, [refreshTasks, tabState.selectedTaskId]);

  const handleBackFromReport = useCallback(() => {
    setTabState({ active: 'tasks', selectedTaskId: null });
  }, []);

  const handleViewHistoricalResults = useCallback((task: EvaluationTask) => {
    setTabState({ active: 'report', selectedTaskId: task.id });
  }, []);

  const tabs = [
    { id: 'tasks' as const, label: 'Tasks', icon: '📋' },
    { id: 'create' as const, label: 'Create', icon: '➕' },
    { id: 'history' as const, label: 'History', icon: '📊' },
  ];

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h1>MoM Evaluation</h1>
          <p>Assess the Mixture-of-Models system performance across multiple dimensions.</p>
        </div>
      </div>

      {mutationError && (
        <div className={styles.errorBanner}>
          <span>{mutationError}</span>
          <button onClick={clearError}>Dismiss</button>
        </div>
      )}

      {tabState.active === 'progress' && tabState.selectedTaskId && (
        <div className={styles.progressView}>
          <button
            className={styles.backButton}
            onClick={() => setTabState({ active: 'tasks', selectedTaskId: null })}
          >
            Back to Tasks
          </button>
          <ProgressTracker
            taskId={tabState.selectedTaskId}
            onComplete={handleProgressComplete}
            onCancel={() => {
              const task = tasks.find(t => t.id === tabState.selectedTaskId);
              if (task) handleCancelTask(task);
            }}
          />
        </div>
      )}

      {tabState.active === 'report' && selectedResults && (
        <ReportViewer
          results={selectedResults}
          onBack={handleBackFromReport}
        />
      )}

      {tabState.active !== 'progress' && tabState.active !== 'report' && (
        <>
          <div className={styles.tabs}>
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={`${styles.tab} ${tabState.active === tab.id ? styles.activeTab : ''}`}
                onClick={() => setTabState({ active: tab.id, selectedTaskId: null })}
              >
                <span className={styles.tabIcon}>{tab.icon}</span>
                <span className={styles.tabLabel}>{tab.label}</span>
              </button>
            ))}
          </div>

          <div className={styles.tabContent}>
            {tabState.active === 'tasks' && (
              <TaskList
                tasks={tasks}
                loading={tasksLoading || mutationLoading}
                onView={handleViewTask}
                onRun={handleRunTask}
                onCancel={handleCancelTask}
                onDelete={handleDeleteTask}
                onRefresh={refreshTasks}
              />
            )}

            {tabState.active === 'create' && (
              <TaskCreationForm
                onSubmit={handleCreateTask}
                onCancel={handleCancelCreate}
                loading={mutationLoading}
              />
            )}

            {tabState.active === 'history' && (
              <HistoricalResults
                tasks={tasks}
                onViewResults={handleViewHistoricalResults}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default EvaluationPage;
