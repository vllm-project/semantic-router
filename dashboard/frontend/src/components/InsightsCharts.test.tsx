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

// The card a title heads, from its section to the next.
const card = (html: string, title: string) => {
  const start = html.indexOf(`aria-label="${title}"`)
  const end = html.indexOf('</section>', start)
  return html.slice(start, end)
}

describe('Insights breakdown', () => {
  it('shows only the Agents, Models and Decisions cards: no second grid of figures, no footnote', () => {
    const html = render(
      <InsightsCharts
        aggregate={{
          ...aggregate,
          summary: { ...aggregate.summary, currency: 'USD', cost_record_count: 2, actual_spend: 1 },
        }}
        harnessCounts={{ codex: 2 }}
      />,
    )
    for (const title of ['Agents', 'Models', 'Decisions']) {
      expect(html).toContain(`aria-label="${title}"`)
    }
    for (const gone of [
      'Breakdown figures',
      'Input tokens',
      'Output tokens',
      'Prompt cache',
      'Model cost',
      'Price units',
      'Cost-only estimate',
      '<details',
    ]) {
      expect(html).not.toContain(gone)
    }
    expect(html).not.toContain('N/A')
  })

  it('heads every card with an icon and lays every row out alike: mark, name, count, share, one bar', () => {
    const html = render(
      <InsightsCharts
        aggregate={{
          ...aggregate,
          record_count: 4,
          model_selection: [{ name: 'deepseek/deepseek-v4.1-flash', value: 3 }],
          decision_distribution: [{ name: 'agentic', value: 1 }],
        }}
        harnessCounts={{ claude_code: 3, api_client: 1 }}
      />,
    )
    for (const title of ['Agents', 'Models', 'Decisions']) {
      const section = card(html, title)
      expect(section).toMatch(/<h4[^>]*><svg/)
      expect(section).toContain('h-[3px]')
      expect(section).toMatch(/grid-cols-\[18px_minmax\(0,1fr\)_auto_3\.25rem\]/)
    }
    // the agents' logos stand bare, as the models' and decisions' marks do
    expect(card(html, 'Agents')).not.toContain('rounded-lg border border-border bg-surface')
    expect(card(html, 'Models')).toContain('75.0%')
    expect(card(html, 'Decisions')).toContain('25.0%')
    expect(card(html, 'Decisions')).toContain('Agentic')
  })

  it('ranks the harnesses by their requests with the plain API client last', () => {
    const html = render(
      <InsightsCharts
        aggregate={aggregate}
        harnessCounts={{ api_client: 50, openclaw: 5, claude_code: 30, codex: 40, cursor: 15 }}
      />,
    )
    expect(html).toContain('aria-label="Requests by agent harness"')
    const order = ['codex', 'claude_code', 'cursor', 'openclaw', 'api_client'].map((id) =>
      html.indexOf(`data-harness="${id}"`),
    )
    expect(order.every((at) => at > 0)).toBe(true)
    expect([...order].sort((a, b) => a - b)).toEqual(order)
    expect(html).toContain('Claude Code')
    expect(html).toContain('API client')
    expect(html).toContain('<img')
    expect(html).toContain('5 agents')
  })

  it('folds rows past the first seven behind a "Show more" button', () => {
    const decisions = Array.from({ length: 9 }, (_, index) => ({
      name: `decision_${index}`,
      value: 10 - index,
    }))
    const html = render(
      <InsightsCharts
        aggregate={{ ...aggregate, record_count: 54, decision_distribution: decisions }}
      />,
    )
    const section = card(html, 'Decisions')
    expect(section.match(/data-decision=/g)).toHaveLength(7)
    expect(section).toContain('Show 2 more')
    expect(section).toContain('aria-expanded="false"')
  })

  it('waits for the harness counts with a skeleton that keeps the three columns', () => {
    const html = render(<InsightsCharts aggregate={aggregate} harnessCounts={null} />)
    expect(card(html, 'Agents')).toContain('aria-busy="true"')
    expect(html).toContain('lg:grid-cols-3')
  })
})
