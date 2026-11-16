import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import InsightsCharts from '../components/InsightsCharts'
import { buildInsightsRecordSections, createInsightsTableColumns } from './insightsPageSupport'
import type {
  InsightsAggregateCostCoverage,
  InsightsAggregateResponse,
  InsightsRecord,
} from './insightsPageTypes'

const priced: InsightsRecord = {
  id: 'priced',
  timestamp: '2026-11-01T00:00:00Z',
  signals: {},
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 0,
  lifecycle_state: 'completed',
  selected_model: 'premium',
  total_tokens: 100,
  currency: 'USD',
  actual_cost: 0.004,
  baseline_cost: 0.001,
  cost_savings: -0.003,
  baseline_model: 'small',
  cost_accounting: {
    pricing_source: 'catalog_list_price',
    caller_login: true,
    baseline_source: 'configured_model',
  },
}

const usageAndCost = (record: InsightsRecord) =>
  buildInsightsRecordSections(record, { isReadonly: false }).find(
    (section) => section.title === 'Usage & Cost',
  )!

const field = (record: InsightsRecord, label: string) =>
  usageAndCost(record).fields.find((item) => item.label === label)?.value

describe('Insights cost accounting labels', () => {
  it('marks list prices and caller logins as estimates and names the configured baseline', () => {
    expect(field(priced, 'Cost basis')).toBe(
      'Recorded tokens × the built-in catalog’s list price (an estimate: no pricing is configured) at capture time; not a GPU bill or provider invoice. It ran on the client’s own login: this is the API-price equivalent, not what the plan charges.',
    )
    expect(field(priced, 'Baseline basis')).toContain('the configured baseline model (small)')
    expect(field(priced, 'Baseline basis')).toContain(
      'a negative saving means the baseline was cheaper',
    )
    const savings = createInsightsTableColumns().find((column) => column.key === 'cost_savings')!
    const html = renderToStaticMarkup(<>{savings.render!(priced)}</>)
    expect(html).toContain('-$0.0030')
    expect(html).not.toContain('costValuePositive')
  })

  it('explains a missing price, a missing baseline and a Looper request', () => {
    const unpriced: InsightsRecord = {
      ...priced,
      actual_cost: undefined,
      baseline_cost: undefined,
      cost_savings: undefined,
      currency: undefined,
      baseline_model: undefined,
      cost_accounting: { unpriced_reason: 'pricing_not_configured', caller_login: true },
    }
    expect(field(unpriced, 'Estimated model cost')).toBe('No price')
    expect(field(unpriced, 'Cost basis')).toContain('No price is configured for premium.')
    expect(field(unpriced, 'Cost basis')).toContain('It ran on the client’s own login')
    expect(field(unpriced, 'Cost basis')).toContain('providers.models[].pricing')

    const noBaseline: InsightsRecord = {
      ...priced,
      baseline_cost: undefined,
      cost_savings: undefined,
      baseline_model: undefined,
      cost_accounting: {
        baseline_source: 'configured_model',
        baseline_unavailable: 'currency_mismatch',
      },
    }
    expect(field(noBaseline, 'Baseline basis')).toContain(
      'No baseline: the configured baseline model is priced in another currency.',
    )

    const looper: InsightsRecord = {
      ...unpriced,
      cost_accounting: {
        router_calls: [
          {
            kind: 'looper',
            calls: 3,
            cost: 0.02,
            currency: 'USD',
            unpriced_calls: 1,
            unpriced_reasons: { pricing_not_configured: 1 },
          },
        ],
      },
    }
    expect(field(looper, 'Router calls')).toBe(
      '3 Looper calls, 0 input / 0 output tokens, 2 priced at $0.02, 1 unpriced (1 no price configured)',
    )
    expect(field(looper, 'Estimated model cost')).toBe('In router calls')
  })
})

const coverage: InsightsAggregateCostCoverage = {
  priced_records: 2,
  partly_priced_records: 1,
  unpriced_records: 2,
  unpriced_reasons: { pricing_not_configured: 2, not_completed: 1 },
  looper_hop_records: 4,
  list_price_records: 1,
  caller_login_records: 1,
  router_calls: 3,
  unpriced_router_calls: 1,
  baseline_sources: { configured_model: 1, recipe_most_expensive: 1 },
}

const aggregate: InsightsAggregateResponse = {
  object: 'router_replay.aggregate',
  record_count: 9,
  lifecycle: { completed: 8, failed: 0, aborted: 0, in_progress: 1, unknown: 0 },
  summary: {
    total_saved: -1,
    baseline_spend: 1,
    actual_spend: 2,
    router_call_spend: 0.5,
    total_spend: 3.5,
    currency: 'USD',
    cost_record_count: 2,
    excluded_record_count: 7,
    coverage,
  },
  model_selection: [],
  decision_distribution: [],
  signal_distribution: [],
  token_volume: { input_tokens: 0, output_tokens: 0, total_tokens: 0, excluded_record_count: 0 },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: [],
  available_decisions: [],
  available_models: [],
}

describe('Insights cost coverage', () => {
  it('shows priced and unpriced counts and router calls beside the cost cards', () => {
    const html = renderToStaticMarkup(<InsightsCharts aggregate={aggregate} />)
    expect(html).toContain('Priced: 2 of 5 requests, 1 partly, 2 without a price.')
    expect(html).toContain('Left out: 2 no price configured, 1 not completed.')
    expect(html).toContain('Router calls: 3 calls, 1 unpriced')
    expect(html).toContain(
      'Savings compare 2 requests with the configured baseline model (1), the recipe pool’s most expensive model (1).',
    )
    expect(html).toContain('1 request priced at a catalog list price: an estimate')
    expect(html).toContain('API-price equivalent, not what the plan charges')
    expect(html).toContain('4 Looper hop records not counted')
    expect(html).toContain('Estimated Router Calls')
    expect(html).toContain('$3.50')
    expect(html).toContain('summaryValueNegative')
    expect(html).not.toContain('No cost is shown')
  })

  it('says plainly why no cost shows and how to set it', () => {
    const unpriced = {
      ...coverage,
      priced_records: 0,
      partly_priced_records: 0,
      router_calls: 0,
      baseline_sources: undefined,
      list_price_records: 0,
    }
    const summary = {
      total_saved: 0,
      baseline_spend: 0,
      actual_spend: 0,
      cost_record_count: 0,
      excluded_record_count: 5,
      coverage: unpriced,
    }
    const html = renderToStaticMarkup(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('No cost is shown because the models these requests used have no price.')
    expect(html).toContain('auth.credential: caller')
    expect(html).toContain('providers.models[].pricing')
    expect(html).not.toContain('Estimated Router Calls')
  })

  it('says savings are net of the Router calls and keeps the gross figure', () => {
    const summary = {
      total_saved: 3.25,
      gross_saved: 4,
      netted_router_call_spend: 0.75,
      baseline_spend: 6,
      actual_spend: 2,
      router_call_spend: 0.875,
      total_spend: 4.875,
      currency: 'USD',
      cost_record_count: 2,
      excluded_record_count: 2,
      coverage: {
        ...coverage,
        baseline_sources: { configured_model: 2 },
        netted_router_calls: { systemone: 1, decision_model: 2, classify: 1 },
        net_unavailable: { no_usage: 1 },
      },
    }
    const html = renderToStaticMarkup(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('Estimated Gross Savings')
    expect(html).toContain('$4.00')
    expect(html).toContain('Router Calls Netted')
    expect(html).toContain('$0.75')
    expect(html).toContain('$3.25')
    expect(html).toContain(
      'Savings are net of the Router’s own calls for these requests: 2 Decision model calls, 1 classify call, 1 System One call.',
    )
    expect(html).toContain(
      '1 request left out of savings for a Router call without a price (1 no token usage reported).',
    )
  })

  it('leaves savings unknown where nothing was compared with a baseline', () => {
    const summary = {
      total_saved: 0,
      baseline_spend: 0,
      actual_spend: 0,
      router_call_spend: 0.5,
      total_spend: 0.5,
      currency: 'USD',
      cost_record_count: 0,
      excluded_record_count: 1,
      coverage: { ...coverage, baseline_sources: undefined },
    }
    const html = renderToStaticMarkup(<InsightsCharts aggregate={{ ...aggregate, summary }} />)
    expect(html).toContain('Estimated Total Cost')
    expect(html).toContain('$0.50')
    expect(html).not.toContain('$0.0000')
  })
})

describe('Router decision costs', () => {
  it('shows model identity, tokens, spend and unpriced calls in the cost section', () => {
    const record: InsightsRecord = {
      ...priced,
      cost_accounting: {
        router_calls: [
          {
            kind: 'decision_model',
            model: 'primary',
            calls: 1,
            prompt_tokens: 120,
            completion_tokens: 3,
            cost: 0.002,
            currency: 'USD',
          },
          {
            kind: 'systemone',
            model: 'remote',
            calls: 1,
            prompt_tokens: 10,
            unpriced_calls: 1,
            unpriced_reasons: { pricing_not_configured: 1 },
          },
        ],
      },
    }
    expect(field(record, 'Router calls')).toContain(
      '1 Decision model call (primary), 120 input / 3 output tokens, 1 priced at $0.0020',
    )
    expect(field(record, 'Router calls')).toContain(
      '1 System One call (remote), 10 input / 0 output tokens, 1 unpriced',
    )
    expect(field(priced, 'Router calls')).toBeUndefined()
    const savings = createInsightsTableColumns().find((column) => column.key === 'cost_savings')!
    expect(renderToStaticMarkup(<>{savings.render!(record)}</>)).toContain(
      'Before its Router calls',
    )
    expect(renderToStaticMarkup(<>{savings.render!(priced)}</>)).not.toContain('Router calls')
  })
})
