import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
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
    input_tokens: 100,
    output_tokens: 100,
    total_tokens: 200,
    excluded_record_count: 0,
  },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: [],
  available_decisions: [],
  available_models: [],
}

describe('Insights cost charts', () => {
  it('distinguishes tiny savings from an exact zero model cost', () => {
    const summary = {
      ...aggregate.summary,
      currency: 'USD',
      cost_record_count: 1,
      excluded_record_count: 0,
      total_saved: 0.000001,
      baseline_spend: 0.000001,
      actual_spend: 0,
    }
    const html = renderToStaticMarkup(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('&lt;$0.0001')
    expect(html).toContain('>$0.00<')
  })

  it('keeps missing pricing unknown and states the estimate and lifecycle scope', () => {
    const html = renderToStaticMarkup(<InsightsCharts aggregate={aggregate} />)
    expect(html).toContain('Model cost')
    expect(html).toContain('Reference cost')
    expect(html).toContain('not a quality comparison, not GPU bills or provider invoices.')
    expect(html).toContain('<summary>How estimates work</summary>')
    expect(html).not.toMatch(/<details[^>]*\sopen(?:[\s=>]|$)/)
    expect(html.indexOf('<details')).toBeLessThan(html.indexOf('Each request is priced twice'))
    expect(html.indexOf('</details>')).toBeGreaterThan(html.indexOf('Older records retain'))
    expect(html).not.toContain('N/A')
    expect(html).toContain(
      'title="No request in range has both a priced model cost and a priced reference model',
    )
    expect(html).toContain('not completed or lack token usage, model pricing, or reference data')
    expect(html).toContain('The total still counts the priced calls of a request that failed.')
    expect(html).toContain('recipe’s complete model pool across all decisions')
    expect(html).toContain('Historical requests without captured prices remain')
    expect(html).toContain('Price not recorded')
    expect(html).toContain('current model rates do not backfill them')
    expect(html).not.toContain('Actual Spend')
    expect(html).not.toContain('$0')
  })

  it('renders each currency instead of a mixed-currency total', () => {
    const summary = {
      ...aggregate.summary,
      cost_record_count: 2,
      excluded_record_count: 0,
      by_currency: [
        {
          currency: 'EUR',
          actual_spend: 1,
          baseline_spend: 2,
          total_saved: 1,
          cost_record_count: 1,
        },
        {
          currency: 'USD',
          actual_spend: 10,
          baseline_spend: 20,
          total_saved: 10,
          cost_record_count: 1,
        },
      ],
    }
    const html = renderToStaticMarkup(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('>EUR<')
    expect(html).toContain('>USD<')
    expect(html).toContain('no exchange-rate conversion is applied')
    expect(html).toContain('$10.00')
    expect(html).not.toContain('$11.00')
  })

  it('shows prices that name no currency as plain amounts in price units', () => {
    const summary = {
      total_saved: 579.0035,
      baseline_spend: 1740.2211,
      actual_spend: 1161.2176,
      currency: 'XXX',
      cost_record_count: 610147,
      excluded_record_count: 57906,
    }
    const html = renderToStaticMarkup(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('>-579.00<')
    expect(html).toContain('>1,740.22<')
    expect(html).toContain('>1,161.22<')
    expect(html).toContain('Price units')
    expect(html).not.toMatch(/XXX(\s|&nbsp;|\u00a0)?[0-9]/)
    expect(html).toContain('57,906 filtered records left out of the comparison')
  })

  it('ranks each distribution with its share of the filtered records', () => {
    const html = renderToStaticMarkup(
      <InsightsCharts
        aggregate={{
          ...aggregate,
          record_count: 4000,
          model_selection: [
            { name: 'model-b', value: 1000 },
            { name: 'model-a', value: 3000 },
          ],
          decision_distribution: [{ name: 'agents::code', value: 4000 }],
        }}
      />,
    )
    expect(html.indexOf('model-a')).toBeLessThan(html.indexOf('model-b'))
    expect(html).toContain('3,000')
    expect(html).toContain('75.0%')
    expect(html).toContain('100.0%')
    expect(html).toContain('4,000 records')
  })

  it('continues rendering a single-currency flat summary', () => {
    const summary = {
      total_saved: 1,
      baseline_spend: 2,
      actual_spend: 1,
      currency: 'USD',
      cost_record_count: 2,
      excluded_record_count: 0,
    }
    const html = renderToStaticMarkup(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('$1.00')
    expect(html).toContain('50.0%')
    expect(html).not.toContain('>N/A<')
  })
})
