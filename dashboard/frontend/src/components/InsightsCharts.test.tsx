import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { InsightsAggregateResponse } from '../pages/insightsPageTypes'
import InsightsCharts from './InsightsCharts'

const aggregate: InsightsAggregateResponse = {
  object: 'management.usage.aggregate',
  record_count: 12,
  lifecycle: { completed: 12, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 2.5,
    baseline_spend: 10,
    actual_spend: 7.5,
    currency: 'USD',
    cost_record_count: 12,
    excluded_record_count: 0,
  },
  model_selection: [],
  decision_distribution: [],
  signal_distribution: [],
  token_volume: {
    input_tokens: 800,
    output_tokens: 200,
    total_tokens: 1000,
    excluded_record_count: 0,
  },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: [],
  available_decisions: [],
  available_models: [],
}

describe('InsightsCharts summary', () => {
  it('keeps the savings rate, baseline, and actual spend visible with total savings', () => {
    const markup = renderToStaticMarkup(createElement(InsightsCharts, { aggregate }))

    expect(markup).toContain('Total Saved')
    expect(markup).toContain('25.0% saved')
    expect(markup).toContain('$7.50 actual')
    expect(markup).toContain('$10.00 baseline')
  })

  it('keeps a percentage visible before priced requests arrive', () => {
    const markup = renderToStaticMarkup(
      createElement(InsightsCharts, {
        aggregate: {
          ...aggregate,
          summary: {
            ...aggregate.summary,
            total_saved: 0,
            baseline_spend: 0,
            actual_spend: 0,
            cost_record_count: 0,
          },
        },
      }),
    )

    expect(markup).toContain('0.0% saved')
  })
})
