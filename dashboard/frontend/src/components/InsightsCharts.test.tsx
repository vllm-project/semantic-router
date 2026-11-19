import { renderToStaticMarkup } from 'react-dom/server'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'

import { TooltipProvider } from '../ui'
import InsightsCharts from './InsightsCharts'
import type { InsightsAggregateResponse } from '../pages/insightsPageTypes'

const aggregate: InsightsAggregateResponse = {
  object: 'router_replay.aggregate',
  record_count: 2,
  lifecycle: { completed: 2, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0,
    baseline_spend: 0,
    actual_spend: 0,
    cost_record_count: 0,
    excluded_record_count: 2,
  },
  model_selection: [],
  decision_distribution: [],
  signal_distribution: [],
  token_volume: {
    input_tokens: 1500,
    output_tokens: 500,
    total_tokens: 2000,
    excluded_record_count: 0,
  },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: [],
  available_decisions: [],
  available_models: [],
}

const render = (node: ReactNode) => renderToStaticMarkup(<TooltipProvider>{node}</TooltipProvider>)

const compared = {
  ...aggregate.summary,
  currency: 'USD',
  cost_record_count: 2,
  excluded_record_count: 0,
  total_saved: 0.25,
  baseline_spend: 1,
  actual_spend: 0.75,
}

describe('Insights breakdown', () => {
  it('lays the headline figures out as one grid, without explanation paragraphs', () => {
    const html = render(<InsightsCharts aggregate={{ ...aggregate, summary: compared }} />)
    for (const label of [
      'Requests',
      'Input tokens',
      'Output tokens',
      'Prompt cache',
      'Model cost',
      'Vs reference',
    ]) {
      expect(html).toContain(`>${label}<`)
    }
    expect(html).toContain('aria-label="Breakdown figures"')
    expect(html).toContain('1.5K')
    expect(html).toContain('$0.75')
    expect(html).toContain('-$0.25')
    expect(html).toContain('25.0% below reference')
    expect(html).toContain('Cost-only estimate against a reference model’s list price.')
    expect(html).not.toContain('How estimates work')
    expect(html).not.toContain('<details')
  })

  it('shows a dash with its reason, never N/A, where nothing was compared', () => {
    const html = render(<InsightsCharts aggregate={aggregate} />)
    expect(html).toContain(
      'aria-label="Not available: No request in range has both a priced model cost',
    )
    expect(html).not.toContain('N/A')
  })

  it('reads a higher routed cost as above the reference', () => {
    const summary = { ...compared, total_saved: -0.5, actual_spend: 1.5 }
    const html = render(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('+$0.50')
    expect(html).toContain('50.0% above reference')
  })

  it('takes the figures of the currency with the most spend', () => {
    const summary = {
      ...aggregate.summary,
      by_currency: [
        { ...compared, currency: 'EUR', actual_spend: 0.1, total_saved: 0.1, baseline_spend: 0.2 },
        { ...compared, currency: 'USD' },
      ],
    }
    const html = render(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('$0.75')
    expect(html).not.toContain('€0.10')
  })

  it('distributes requests by agent harness, each bar with its logo, API clients last', () => {
    const html = render(
      <InsightsCharts
        aggregate={aggregate}
        harnessCounts={{ api_client: 50, openclaw: 5, claude_code: 30, cursor: 15 }}
      />,
    )
    expect(html).toContain('aria-label="Requests by agent harness"')
    const order = ['claude_code', 'cursor', 'openclaw', 'api_client'].map((id) =>
      html.indexOf(`data-harness="${id}"`),
    )
    expect(order.every((at) => at > 0)).toBe(true)
    expect([...order].sort((a, b) => a - b)).toEqual(order)
    expect(html).toContain('Claude Code')
    expect(html).toContain('OpenClaw')
    expect(html).toContain('API client')
    expect(html).toContain('50.0%')
    expect(html).toContain('<img')
  })

  it('ranks models and decisions with their shares of the filtered records', () => {
    const html = render(
      <InsightsCharts
        aggregate={{
          ...aggregate,
          model_selection: [{ name: 'deepseek', value: 3 }],
          decision_distribution: [{ name: 'agentic', value: 1 }],
          record_count: 4,
        }}
      />,
    )
    expect(html).toContain('aria-label="Models"')
    expect(html).toContain('aria-label="Decisions"')
    expect(html).toContain('75.0%')
    expect(html).toContain('25.0%')
  })
})
