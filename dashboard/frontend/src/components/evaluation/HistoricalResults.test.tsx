import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationTask } from '../../types/evaluation'
import { HistoricalResults } from './HistoricalResults'

const tasks: EvaluationTask[] = Array.from({ length: 30 }, (_, index) => ({
  id: `history-${index + 1}`,
  name: `Historical run ${index + 1}`,
  description: '',
  status: index % 5 === 0 ? 'failed' : 'completed',
  progress_percent: 100,
  created_at: new Date(Date.UTC(2026, 0, index + 1)).toISOString(),
  completed_at: new Date(Date.UTC(2026, 0, index + 1, 1)).toISOString(),
  config: {
    level: 'router',
    dimensions: ['domain'],
    datasets: { domain: ['mmlu'] },
    max_samples: 10,
    endpoint: '/api/v1/eval',
    model: 'auto',
    concurrent: 1,
    samples_per_cat: 1,
  },
}))

describe('HistoricalResults', () => {
  it('bounds historical card rendering to one client page', () => {
    const markup = renderToStaticMarkup(
      createElement(HistoricalResults, {
        tasks,
        onViewResults: () => undefined,
        onRefresh: () => undefined,
      }),
    )

    expect(markup).toContain('history-30')
    expect(markup).toContain('history-11')
    expect(markup).not.toContain('history-10')
    expect(markup).toContain('1–20 of 30 runs')
    expect(markup).toContain('30 runs')
  })

  it('distinguishes loading, error, and empty history states', () => {
    const loading = renderToStaticMarkup(
      createElement(HistoricalResults, {
        tasks: [],
        loading: true,
        onViewResults: () => undefined,
      }),
    )
    const error = renderToStaticMarkup(
      createElement(HistoricalResults, {
        tasks: [],
        error: 'history offline',
        onViewResults: () => undefined,
      }),
    )
    const empty = renderToStaticMarkup(
      createElement(HistoricalResults, { tasks: [], onViewResults: () => undefined }),
    )

    expect(loading).toContain('Loading evaluation history')
    expect(error).toContain('history offline')
    expect(empty).toContain('No Historical Results')
  })
})
