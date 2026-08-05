import { useDeferredValue, useEffect, useMemo, useState } from 'react'

import { getPageWindow, paginateRows } from '../dataTableSupport'
import type { EvaluationLevel, EvaluationStatus, EvaluationTask } from '../../types/evaluation'
import { LEVEL_INFO, STATUS_INFO, formatDate, formatDuration } from '../../types/evaluation'
import EvaluationPagination from './EvaluationPagination'
import {
  EVALUATION_HISTORY_PAGE_SIZE,
  filterAndSortEvaluationTasks,
  formatEvaluationResultCount,
  isHistoricalEvaluationTask,
  type EvaluationTaskSort,
} from './evaluationListSupport'
import styles from './HistoricalResults.module.css'

interface HistoricalResultsProps {
  tasks: EvaluationTask[]
  loading?: boolean
  error?: string | null
  onRefresh?: () => void
  onViewResults: (task: EvaluationTask) => void
  onCompare?: (tasks: EvaluationTask[]) => void
}

export function HistoricalResults({
  tasks,
  loading = false,
  error,
  onRefresh,
  onViewResults,
  onCompare,
}: HistoricalResultsProps) {
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set())
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<EvaluationTaskSort>('created-desc')
  const [filterStatus, setFilterStatus] = useState<EvaluationStatus | 'all'>('all')
  const [levelFilter, setLevelFilter] = useState<EvaluationLevel | 'all'>('all')
  const [currentPage, setCurrentPage] = useState(1)
  const deferredSearch = useDeferredValue(search)

  const historicalTasks = useMemo(() => tasks.filter(isHistoricalEvaluationTask), [tasks])
  const filteredTasks = useMemo(
    () =>
      filterAndSortEvaluationTasks(
        tasks,
        {
          search: deferredSearch,
          status: filterStatus,
          level: levelFilter,
          sort: sortBy,
        },
        { historicalOnly: true },
      ),
    [deferredSearch, filterStatus, levelFilter, sortBy, tasks],
  )
  const pageWindow = getPageWindow(filteredTasks.length, currentPage, EVALUATION_HISTORY_PAGE_SIZE)
  const visibleTasks = paginateRows(filteredTasks, pageWindow)

  useEffect(() => setCurrentPage(1), [deferredSearch, filterStatus, levelFilter, sortBy])
  useEffect(() => {
    if (currentPage !== pageWindow.page) setCurrentPage(pageWindow.page)
  }, [currentPage, pageWindow.page])
  useEffect(() => {
    if (!onCompare) return
    const available = new Set(historicalTasks.map((task) => task.id))
    setSelectedTasks((current) => {
      const retained = [...current].filter((taskID) => available.has(taskID))
      return retained.length === current.size ? current : new Set(retained)
    })
  }, [historicalTasks, onCompare])

  const toggleTask = (taskId: string) => {
    setSelectedTasks((previous) => {
      const next = new Set(previous)
      if (next.has(taskId)) next.delete(taskId)
      else next.add(taskId)
      return next
    })
  }

  const handleCompare = () => {
    onCompare?.(historicalTasks.filter((task) => selectedTasks.has(task.id)))
  }

  if (loading && historicalTasks.length === 0) {
    return <div className={styles.loading} role="status">Loading evaluation history…</div>
  }

  if (error && historicalTasks.length === 0) {
    return (
      <div className={styles.errorState} role="alert">
        <h3>Evaluation history is unavailable</h3>
        <p>{error}</p>
        {onRefresh ? <button type="button" onClick={onRefresh}>Retry</button> : null}
      </div>
    )
  }

  if (historicalTasks.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>📊</div>
        <h3>No Historical Results</h3>
        <p>Complete an evaluation to see results here.</p>
      </div>
    )
  }

  return (
    <div className={styles.container} aria-busy={loading}>
      <div className={styles.toolbar}>
        <div className={styles.filters}>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className={styles.searchInput}
            placeholder="Search history, ID, dataset…"
            aria-label="Search evaluation history"
          />
          <select
            value={filterStatus}
            onChange={(event) => setFilterStatus(event.target.value as EvaluationStatus | 'all')}
            className={styles.select}
            aria-label="Filter history by status"
          >
            <option value="all">All terminal statuses</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <select
            value={levelFilter}
            onChange={(event) => setLevelFilter(event.target.value as EvaluationLevel | 'all')}
            className={styles.select}
            aria-label="Filter history by level"
          >
            <option value="all">All levels</option>
            {Object.entries(LEVEL_INFO).map(([level, info]) => (
              <option key={level} value={level}>{info.label}</option>
            ))}
          </select>
          <select
            value={sortBy}
            onChange={(event) => setSortBy(event.target.value as EvaluationTaskSort)}
            className={styles.select}
            aria-label="Sort evaluation history"
          >
            <option value="created-desc">Newest first</option>
            <option value="created-asc">Oldest first</option>
            <option value="name-asc">Name A–Z</option>
          </select>
        </div>
        <div className={styles.toolbarMeta}>
          <span className={styles.resultCount}>
            {formatEvaluationResultCount(filteredTasks.length, historicalTasks.length, 'runs')}
          </span>
          <span className={styles.clientPagingNote}>Client view · {EVALUATION_HISTORY_PAGE_SIZE}/page</span>
          {onRefresh ? (
            <button type="button" className={styles.refreshButton} onClick={onRefresh} disabled={loading}>
              {loading ? 'Refreshing…' : 'Refresh'}
            </button>
          ) : null}
          {onCompare && selectedTasks.size >= 2 ? (
            <button type="button" className={styles.compareButton} onClick={handleCompare}>
              Compare Selected ({selectedTasks.size})
            </button>
          ) : null}
        </div>
      </div>

      {error ? <div className={styles.inlineError} role="alert">Refresh failed: {error}</div> : null}

      {visibleTasks.length === 0 ? (
        <div className={styles.noMatches}>
          <h3>No historical runs match this view</h3>
          <p>Adjust the search or filters to broaden the result set.</p>
        </div>
      ) : (
        <div className={styles.list}>
          {visibleTasks.map((task) => {
            const statusInfo = STATUS_INFO[task.status]
            const isSelected = selectedTasks.has(task.id)

            return (
              <article key={task.id} className={`${styles.card} ${isSelected ? styles.selected : ''}`}>
                {onCompare ? (
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleTask(task.id)}
                    className={styles.checkbox}
                    aria-label={`Select ${task.name} for comparison`}
                  />
                ) : null}
                <div className={styles.cardContent}>
                  <div className={styles.cardHeader}>
                    <h4>{task.name}</h4>
                    <span
                      className={styles.statusBadge}
                      style={{ color: statusInfo.color, backgroundColor: statusInfo.bgColor }}
                    >
                      {statusInfo.label}
                    </span>
                  </div>
                  <span className={styles.taskId}>{task.id}</span>
                  {task.description ? <p className={styles.description}>{task.description}</p> : null}
                  <div className={styles.cardMeta}>
                    <span>Finished: {formatDate(task.completed_at || task.created_at)}</span>
                    <span>Duration: {formatDuration(task.started_at, task.completed_at)}</span>
                    <span>{LEVEL_INFO[task.config.level].label}</span>
                    <span>Dimensions: {task.config.dimensions.length}</span>
                  </div>
                </div>
                <button type="button" className={styles.viewButton} onClick={() => onViewResults(task)}>
                  View Results
                </button>
              </article>
            )
          })}
        </div>
      )}

      <EvaluationPagination
        page={pageWindow.page}
        totalPages={pageWindow.totalPages}
        start={pageWindow.start}
        end={pageWindow.end}
        total={filteredTasks.length}
        itemLabel="runs"
        onPageChange={setCurrentPage}
      />
    </div>
  )
}

export default HistoricalResults
