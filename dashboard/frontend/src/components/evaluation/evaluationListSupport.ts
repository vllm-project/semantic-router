import type {
  EvaluationLevel,
  EvaluationResult,
  EvaluationStatus,
  EvaluationTask,
} from '../../types/evaluation'
import { getMetricValue } from '../../types/evaluation'

export const EVALUATION_TASK_PAGE_SIZE = 25
export const EVALUATION_HISTORY_PAGE_SIZE = 20
export const EVALUATION_RESULT_PAGE_SIZE = 8

export type EvaluationTaskSort =
  | 'created-desc'
  | 'created-asc'
  | 'name-asc'
  | 'progress-desc'

export interface EvaluationTaskFilters {
  search: string
  status: EvaluationStatus | 'all'
  level: EvaluationLevel | 'all'
  sort: EvaluationTaskSort
}

export type EvaluationResultSort = 'dataset-asc' | 'dimension-asc' | 'score-desc'

export interface EvaluationResultFilters {
  search: string
  dimension: EvaluationResult['dimension'] | 'all'
  sort: EvaluationResultSort
}

function normalizedSearch(value: string): string {
  return value.trim().toLocaleLowerCase()
}

function taskSearchText(task: EvaluationTask): string {
  const datasets = Object.entries(task.config.datasets).flatMap(([dimension, names]) => [
    dimension,
    ...names,
  ])
  return [
    task.id,
    task.name,
    task.description,
    task.status,
    task.config.level,
    task.config.model,
    task.config.endpoint,
    task.current_step,
    task.error_message,
    ...task.config.dimensions,
    ...datasets,
  ]
    .filter((value): value is string => typeof value === 'string')
    .join(' ')
    .toLocaleLowerCase()
}

export function isHistoricalEvaluationTask(task: EvaluationTask): boolean {
  return task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled'
}

export function filterAndSortEvaluationTasks(
  tasks: readonly EvaluationTask[],
  filters: EvaluationTaskFilters,
  options: { historicalOnly?: boolean } = {},
): EvaluationTask[] {
  const query = normalizedSearch(filters.search)
  const filtered = tasks.filter((task) => {
    if (options.historicalOnly && !isHistoricalEvaluationTask(task)) return false
    if (filters.status !== 'all' && task.status !== filters.status) return false
    if (filters.level !== 'all' && task.config.level !== filters.level) return false
    return !query || taskSearchText(task).includes(query)
  })

  return [...filtered].sort((left, right) => {
    switch (filters.sort) {
      case 'created-asc':
        return Date.parse(left.created_at) - Date.parse(right.created_at)
      case 'name-asc':
        return left.name.localeCompare(right.name, undefined, { sensitivity: 'base' })
      case 'progress-desc':
        return right.progress_percent - left.progress_percent || Date.parse(right.created_at) - Date.parse(left.created_at)
      case 'created-desc':
      default:
        return Date.parse(right.completed_at || right.created_at) - Date.parse(left.completed_at || left.created_at)
    }
  })
}

function resultSearchText(result: EvaluationResult): string {
  const metadata = result.metrics.metadata
  const metadataText =
    metadata && typeof metadata === 'object'
      ? Object.values(metadata)
          .filter((value): value is string | number =>
            typeof value === 'string' || typeof value === 'number',
          )
          .join(' ')
      : ''
  return [result.id, result.dimension, result.dataset_name, metadataText]
    .join(' ')
    .toLocaleLowerCase()
}

export function getEvaluationResultScore(result: EvaluationResult): number | null {
  return getMetricValue(result.metrics, 'accuracy') ?? getMetricValue(result.metrics, 'f1_score')
}

export function filterAndSortEvaluationResults(
  results: readonly EvaluationResult[],
  filters: EvaluationResultFilters,
): EvaluationResult[] {
  const query = normalizedSearch(filters.search)
  const filtered = results.filter((result) => {
    if (filters.dimension !== 'all' && result.dimension !== filters.dimension) return false
    return !query || resultSearchText(result).includes(query)
  })

  return [...filtered].sort((left, right) => {
    switch (filters.sort) {
      case 'dimension-asc':
        return left.dimension.localeCompare(right.dimension) || left.dataset_name.localeCompare(right.dataset_name)
      case 'score-desc':
        return (getEvaluationResultScore(right) ?? -1) - (getEvaluationResultScore(left) ?? -1)
      case 'dataset-asc':
      default:
        return left.dataset_name.localeCompare(right.dataset_name, undefined, { sensitivity: 'base' })
    }
  })
}

export function formatEvaluationResultCount(filtered: number, total: number, label: string): string {
  return filtered === total ? `${total} ${label}` : `${filtered} of ${total} ${label}`
}
