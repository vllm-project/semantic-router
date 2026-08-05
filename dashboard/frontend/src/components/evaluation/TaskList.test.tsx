import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationTask } from '../../types/evaluation'
import { TaskList } from './TaskList'

const task: EvaluationTask = {
  id: 'eval-1',
  name: 'Routing quality',
  description: '',
  status: 'pending',
  progress_percent: 0,
  created_at: new Date().toISOString(),
  config: {
    level: 'router',
    dimensions: [],
    datasets: {},
    max_samples: 10,
    endpoint: '/v1/chat/completions',
    model: 'auto',
    concurrent: 1,
    samples_per_cat: 1,
  },
}

describe('TaskList permissions', () => {
  it('keeps read-only evaluation views free of mutation actions', () => {
    const markup = renderToStaticMarkup(
      createElement(TaskList, {
        tasks: [task],
        loading: false,
        onView: () => undefined,
        onRun: () => undefined,
        onCancel: () => undefined,
        onDelete: () => undefined,
        onRefresh: () => undefined,
        canRunTasks: false,
        canDeleteTasks: false,
      }),
    )

    expect(markup).toContain('View')
    expect(markup).not.toMatch(/>Run</)
    expect(markup).not.toMatch(/>Cancel</)
    expect(markup).not.toMatch(/>Delete</)
  })

  it('renders only the first bounded page for a large task catalog', () => {
    const tasks = Array.from({ length: 60 }, (_, index) => ({
      ...task,
      id: `eval-${index + 1}`,
      name: `Evaluation ${index + 1}`,
      created_at: new Date(Date.UTC(2026, 0, index + 1)).toISOString(),
    }))
    const markup = renderToStaticMarkup(
      createElement(TaskList, {
        tasks,
        loading: false,
        onView: () => undefined,
        onRun: () => undefined,
        onCancel: () => undefined,
        onDelete: () => undefined,
        onRefresh: () => undefined,
      }),
    )

    expect(markup).toContain('eval-60')
    expect(markup).toContain('eval-36')
    expect(markup).not.toContain('eval-35')
    expect(markup).toContain('1–25 of 60 tasks')
    expect(markup).toContain('60 tasks')
  })

  it('renders a retryable error instead of an ambiguous empty state', () => {
    const markup = renderToStaticMarkup(
      createElement(TaskList, {
        tasks: [],
        loading: false,
        error: 'service unavailable',
        onView: () => undefined,
        onRun: () => undefined,
        onCancel: () => undefined,
        onDelete: () => undefined,
        onRefresh: () => undefined,
      }),
    )

    expect(markup).toContain('Evaluation tasks are unavailable')
    expect(markup).toContain('service unavailable')
    expect(markup).toMatch(/>Retry</)
  })
})
