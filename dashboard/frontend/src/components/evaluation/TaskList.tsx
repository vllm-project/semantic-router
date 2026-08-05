import { useCallback, useDeferredValue, useEffect, useMemo, useState } from 'react'

import { getPageWindow, paginateRows } from '../dataTableSupport'
import type { EvaluationLevel, EvaluationStatus, EvaluationTask } from '../../types/evaluation'
import { STATUS_INFO, LEVEL_INFO, formatDate, formatDuration } from '../../types/evaluation'
import EvaluationPagination from './EvaluationPagination'
import {
  EVALUATION_TASK_PAGE_SIZE,
  filterAndSortEvaluationTasks,
  formatEvaluationResultCount,
  type EvaluationTaskSort,
} from './evaluationListSupport'
import styles from './TaskList.module.css'

interface TaskListProps {
  tasks: EvaluationTask[]
  loading: boolean
  error?: string | null
  onView: (task: EvaluationTask) => void
  onRun: (task: EvaluationTask) => void
  onCancel: (task: EvaluationTask) => void
  onDelete: (task: EvaluationTask) => void
  onRefresh: () => void
  canRunTasks?: boolean
  canDeleteTasks?: boolean
  canCreateTasks?: boolean
}

export function TaskList({
  tasks,
  loading,
  error,
  onView,
  onRun,
  onCancel,
  onDelete,
  onRefresh,
  canRunTasks = true,
  canDeleteTasks = true,
  canCreateTasks = true,
}: TaskListProps) {
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<EvaluationStatus | 'all'>('all')
  const [levelFilter, setLevelFilter] = useState<EvaluationLevel | 'all'>('all')
  const [sortBy, setSortBy] = useState<EvaluationTaskSort>('created-desc')
  const [currentPage, setCurrentPage] = useState(1)
  const deferredSearch = useDeferredValue(search)

  const getStatusBadge = useCallback((status: EvaluationTask['status']) => {
    const info = STATUS_INFO[status]
    return (
      <span
        className={styles.statusBadge}
        style={{ color: info.color, backgroundColor: info.bgColor }}
      >
        {info.label}
      </span>
    )
  }, [])

  const filteredTasks = useMemo(
    () =>
      filterAndSortEvaluationTasks(tasks, {
        search: deferredSearch,
        status: statusFilter,
        level: levelFilter,
        sort: sortBy,
      }),
    [deferredSearch, levelFilter, sortBy, statusFilter, tasks],
  )
  const pageWindow = getPageWindow(filteredTasks.length, currentPage, EVALUATION_TASK_PAGE_SIZE)
  const visibleTasks = paginateRows(filteredTasks, pageWindow)

  useEffect(() => setCurrentPage(1), [deferredSearch, levelFilter, sortBy, statusFilter])
  useEffect(() => {
    if (currentPage !== pageWindow.page) setCurrentPage(pageWindow.page)
  }, [currentPage, pageWindow.page])

  const canRun = (task: EvaluationTask) => task.status === 'pending' || task.status === 'failed'
  const canCancel = (task: EvaluationTask) => task.status === 'running'

  if (loading && tasks.length === 0) {
    return (
      <div className={styles.loading} role="status">
        <div className={styles.spinner} />
        <span>Loading evaluation tasks…</span>
      </div>
    )
  }

  if (error && tasks.length === 0) {
    return (
      <div className={styles.errorState} role="alert">
        <h3>Evaluation tasks are unavailable</h3>
        <p>{error}</p>
        <button type="button" onClick={onRefresh}>Retry</button>
      </div>
    )
  }

  if (tasks.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>📋</div>
        <h3>No Evaluation Tasks</h3>
        <p>
          {canCreateTasks
            ? 'Create an evaluation task to start measuring the model system.'
            : 'No evaluation tasks are available in this workspace.'}
        </p>
      </div>
    )
  }

  return (
    <div className={styles.container} aria-busy={loading}>
      <div className={styles.header}>
        <div className={styles.headingGroup}>
          <h3>Evaluation Tasks</h3>
          <span className={styles.resultCount}>
            {formatEvaluationResultCount(filteredTasks.length, tasks.length, 'tasks')}
          </span>
        </div>
        <button type="button" className={styles.refreshButton} onClick={onRefresh} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      <div className={styles.controls}>
        <input
          type="search"
          className={styles.searchInput}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search task, ID, model, dimension…"
          aria-label="Search evaluation tasks"
        />
        <select
          className={styles.filterSelect}
          value={statusFilter}
          onChange={(event) => setStatusFilter(event.target.value as EvaluationStatus | 'all')}
          aria-label="Filter tasks by status"
        >
          <option value="all">All statuses</option>
          {Object.entries(STATUS_INFO).map(([status, info]) => (
            <option key={status} value={status}>{info.label}</option>
          ))}
        </select>
        <select
          className={styles.filterSelect}
          value={levelFilter}
          onChange={(event) => setLevelFilter(event.target.value as EvaluationLevel | 'all')}
          aria-label="Filter tasks by level"
        >
          <option value="all">All levels</option>
          {Object.entries(LEVEL_INFO).map(([level, info]) => (
            <option key={level} value={level}>{info.label}</option>
          ))}
        </select>
        <select
          className={styles.filterSelect}
          value={sortBy}
          onChange={(event) => setSortBy(event.target.value as EvaluationTaskSort)}
          aria-label="Sort evaluation tasks"
        >
          <option value="created-desc">Newest first</option>
          <option value="created-asc">Oldest first</option>
          <option value="name-asc">Name A–Z</option>
          <option value="progress-desc">Progress high–low</option>
        </select>
        <span className={styles.clientPagingNote}>Client view · {EVALUATION_TASK_PAGE_SIZE} rows/page</span>
      </div>

      {error ? <div className={styles.inlineError} role="alert">Refresh failed: {error}</div> : null}

      {visibleTasks.length === 0 ? (
        <div className={styles.noMatches}>
          <h4>No tasks match this view</h4>
          <p>Adjust the search or filters to broaden the result set.</p>
        </div>
      ) : (
        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Level</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Created</th>
                <th>Duration</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {visibleTasks.map((task) => (
                <tr key={task.id}>
                  <td>
                    <div className={styles.taskName}>
                      <span className={styles.name}>{task.name}</span>
                      <span className={styles.taskId}>{task.id}</span>
                      {task.description ? <span className={styles.description}>{task.description}</span> : null}
                    </div>
                  </td>
                  <td>
                    <span className={styles.levelBadge} style={{ color: LEVEL_INFO[task.config.level].color }}>
                      {LEVEL_INFO[task.config.level].label}
                    </span>
                  </td>
                  <td>{getStatusBadge(task.status)}</td>
                  <td>
                    <div className={styles.progress}>
                      <div
                        className={styles.progressBar}
                        style={{ width: `${Math.min(100, Math.max(0, task.progress_percent))}%` }}
                      />
                      <span className={styles.progressText}>{task.progress_percent}%</span>
                    </div>
                    {task.current_step ? <span className={styles.currentStep}>{task.current_step}</span> : null}
                  </td>
                  <td className={styles.date}>{formatDate(task.created_at)}</td>
                  <td className={styles.duration}>{formatDuration(task.started_at, task.completed_at)}</td>
                  <td>
                    <div className={styles.actions}>
                      <button type="button" className={styles.actionButton} onClick={() => onView(task)}>
                        View
                      </button>
                      {canRunTasks && canRun(task) ? (
                        <button
                          type="button"
                          className={`${styles.actionButton} ${styles.runButton}`}
                          onClick={() => onRun(task)}
                          disabled={loading}
                        >
                          Run
                        </button>
                      ) : null}
                      {canRunTasks && canCancel(task) ? (
                        <button
                          type="button"
                          className={`${styles.actionButton} ${styles.cancelButton}`}
                          onClick={() => onCancel(task)}
                          disabled={loading}
                        >
                          Cancel
                        </button>
                      ) : null}
                      {canDeleteTasks ? (
                        <button
                          type="button"
                          className={`${styles.actionButton} ${styles.deleteButton}`}
                          onClick={() => onDelete(task)}
                          disabled={loading || task.status === 'running'}
                        >
                          Delete
                        </button>
                      ) : null}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <EvaluationPagination
        page={pageWindow.page}
        totalPages={pageWindow.totalPages}
        start={pageWindow.start}
        end={pageWindow.end}
        total={filteredTasks.length}
        itemLabel="tasks"
        onPageChange={setCurrentPage}
      />
    </div>
  )
}

export default TaskList
