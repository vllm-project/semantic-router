import { describe, expect, it } from 'vitest'

import type { EvaluationResult, EvaluationTask } from '../../types/evaluation'
import {
  EVALUATION_HISTORY_PAGE_SIZE,
  EVALUATION_RESULT_PAGE_SIZE,
  EVALUATION_TASK_PAGE_SIZE,
  filterAndSortEvaluationResults,
  filterAndSortEvaluationTasks,
  formatEvaluationResultCount,
} from './evaluationListSupport'

const tasks: EvaluationTask[] = Array.from({ length: 500 }, (_, index) => ({
  id: `eval-${index + 1}`,
  name: index === 419 ? 'Routing quality sentinel' : `Evaluation ${index + 1}`,
  description: index % 2 === 0 ? 'signal routing' : 'system accuracy',
  status: index % 5 === 0 ? 'failed' : index % 3 === 0 ? 'running' : 'completed',
  created_at: new Date(Date.UTC(2026, 0, index + 1)).toISOString(),
  progress_percent: index % 100,
  config: {
    level: index % 2 === 0 ? 'router' : 'mom',
    dimensions: index % 2 === 0 ? ['domain'] : ['accuracy'],
    datasets: {},
    max_samples: 50,
    endpoint: 'http://router',
    model: 'auto',
    concurrent: 1,
    samples_per_cat: 10,
  },
}))

describe('evaluation list support', () => {
  it('filters and sorts a 500-task client catalog without mutating it', () => {
    const originalFirst = tasks[0]
    const filtered = filterAndSortEvaluationTasks(tasks, {
      search: 'sentinel',
      status: 'all',
      level: 'mom',
      sort: 'created-desc',
    })

    expect(filtered.map((task) => task.id)).toEqual(['eval-420'])
    expect(tasks[0]).toBe(originalFirst)
  })

  it('keeps history terminal-only and exposes explicit bounded page sizes', () => {
    const history = filterAndSortEvaluationTasks(
      tasks,
      { search: '', status: 'all', level: 'all', sort: 'name-asc' },
      { historicalOnly: true },
    )

    expect(
      history.every((task) =>
        task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled',
      ),
    ).toBe(true)
    expect(EVALUATION_TASK_PAGE_SIZE).toBe(25)
    expect(EVALUATION_HISTORY_PAGE_SIZE).toBe(20)
    expect(EVALUATION_RESULT_PAGE_SIZE).toBe(8)
  })

  it('filters and ranks result cards by dimension, dataset, and score', () => {
    const results: EvaluationResult[] = [
      { id: 'r1', task_id: 't1', dimension: 'domain', dataset_name: 'Alpha', metrics: { accuracy: 0.8 } },
      { id: 'r2', task_id: 't1', dimension: 'accuracy', dataset_name: 'MMLU Pro', metrics: { accuracy: 0.92 } },
      { id: 'r3', task_id: 't1', dimension: 'accuracy', dataset_name: 'General', metrics: { accuracy: 0.7 } },
    ]

    expect(
      filterAndSortEvaluationResults(results, {
        search: 'mmlu',
        dimension: 'accuracy',
        sort: 'score-desc',
      }).map((result) => result.id),
    ).toEqual(['r2'])
    expect(formatEvaluationResultCount(1, 3, 'results')).toBe('1 of 3 results')
  })
})
