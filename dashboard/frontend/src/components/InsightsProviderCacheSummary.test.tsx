import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import InsightsProviderCacheSummary from './InsightsProviderCacheSummary'
import type { InsightsAggregateProviderCache } from '../pages/insightsPageTypes'

const summary: InsightsAggregateProviderCache = {
  no_cache_spend: 0.00044,
  actual_spend: 0.0007,
  net_savings: -0.00026,
  currency: 'USD',
  priced_records: 2,
  excluded_records: 3,
  by_currency: [
    {
      currency: 'USD',
      no_cache_spend: 0.00044,
      actual_spend: 0.0007,
      net_savings: -0.00026,
      priced_records: 2,
    },
  ],
  by_decision: [
    {
      name: 'agents::writes',
      currency: 'USD',
      no_cache_spend: 0.00022,
      actual_spend: 0.00057,
      net_savings: -0.00035,
      priced_records: 1,
    },
    {
      name: 'agents::reads',
      currency: 'USD',
      no_cache_spend: 0.00022,
      actual_spend: 0.00013,
      net_savings: 0.00009,
      priced_records: 1,
    },
  ],
  by_selected_model: [
    {
      name: 'model-a',
      currency: 'USD',
      no_cache_spend: 0.00044,
      actual_spend: 0.0007,
      net_savings: -0.00026,
      priced_records: 2,
    },
  ],
}

const render = (value?: InsightsAggregateProviderCache) =>
  renderToStaticMarkup(<InsightsProviderCacheSummary summary={value} />)

describe('Insights provider prompt-cache summary', () => {
  it('shows a negative net saving as negative instead of clamping requests at zero', () => {
    const html = render(summary)
    expect(html).toMatch(/Net Cache Saving<\/span><strong[^>]*>-\$0\.0003</)
    expect(html).toMatch(/Net Saved %<\/span><strong[^>]*>-59\.1%</)
    expect(html).toContain('Cache writes cost more than cache reads saved across these requests.')
    expect(html).toContain('No request is clamped at zero.')
    expect(html).toContain('2 settled requests')
  })

  it('splits the net saving by decision and selected model', () => {
    const html = render(summary)
    expect(html).toContain('Net Saving by Decision')
    expect(html).toContain('agents::writes')
    expect(html).toContain('-$0.0004')
    expect(html).toContain('agents::reads')
    expect(html).toContain('&lt;$0.0001')
    expect(html).toContain('Net Saving by Selected Model')
    expect(html).toContain('model-a')
  })

  it('states how many filtered records are excluded and why', () => {
    expect(render(summary)).toContain(
      '3 filtered records excluded: not completed, served from the Router cache, or without complete provider usage and prices.',
    )
    expect(render({ ...summary, excluded_records: 0 })).not.toContain('excluded:')
  })

  it('renders each currency separately', () => {
    const html = render({
      ...summary,
      currency: undefined,
      no_cache_spend: 0,
      actual_spend: 0,
      net_savings: 0,
      by_currency: [
        { currency: 'EUR', no_cache_spend: 4, actual_spend: 5, net_savings: -1, priced_records: 1 },
        { currency: 'USD', no_cache_spend: 2, actual_spend: 1, net_savings: 1, priced_records: 1 },
      ],
    })
    expect(html).toContain('EUR provider cache')
    expect(html).toContain('USD provider cache')
    expect(html).toContain('-€1.00')
    expect(html).toContain('50.0%')
    expect(html).not.toContain('>$0.0000<')
  })

  it('renders nothing without settled provider-cache records', () => {
    expect(render(undefined)).toBe('')
    expect(render({ ...summary, priced_records: 0, by_currency: [] })).toBe('')
  })
})
