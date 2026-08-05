import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationResult, TaskResults } from '../../types/evaluation'
import { ReportViewer } from './ReportViewer'

const results: EvaluationResult[] = Array.from({ length: 12 }, (_, index) => ({
  id: `result-${index + 1}`,
  task_id: 'task-1',
  dimension: 'domain',
  dataset_name: `Dataset ${String(index + 1).padStart(2, '0')}`,
  metrics: {
    accuracy: 0.7 + index / 100,
    details: Array.from({ length: 1_000 }, () => ({
      query: 'VERY-LARGE-DETAIL',
      expected: 'domain',
      actual: 'domain',
      status: 'correct' as const,
    })),
  },
}))

const report: TaskResults = {
  task: {
    id: 'task-1',
    name: 'Routing report',
    description: '',
    status: 'completed',
    created_at: '2026-07-01T00:00:00.000Z',
    started_at: '2026-07-01T00:00:01.000Z',
    completed_at: '2026-07-01T00:00:02.000Z',
    progress_percent: 100,
    config: {
      level: 'router',
      dimensions: ['domain'],
      datasets: { domain: [] },
      max_samples: 1_000,
      endpoint: '/api/v1/eval',
      model: 'auto',
      concurrent: 1,
      samples_per_cat: 10,
    },
  },
  results,
}

describe('ReportViewer', () => {
  it('bounds result cards and keeps large test-case payloads collapsed', () => {
    const markup = renderToStaticMarkup(createElement(ReportViewer, { results: report }))

    expect(markup).toContain('Dataset 01')
    expect(markup).toContain('Dataset 08')
    expect(markup).not.toContain('Dataset 09')
    expect(markup).toContain('1–8 of 12 results')
    expect(markup).not.toContain('VERY-LARGE-DETAIL')
    expect(markup).toContain('Test Cases (1000)')
  })

  it('renders a specific empty result state', () => {
    const markup = renderToStaticMarkup(
      createElement(ReportViewer, { results: { ...report, results: [] } }),
    )
    expect(markup).toContain('No evaluation results were recorded')
  })
})
