import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import InsightsCharts from '../components/InsightsCharts'
import { TooltipProvider } from '../ui'
import { buildCostBaselineModel, InsightsConfigContext } from './insightsConfigFacts'
import {
  describeEstimateMethod,
  describeFreeRecords,
  describeRecordSaving,
  formatSaving,
  savingsReference,
  savingVersus,
} from './insightsCostAccounting'
import { RequestsSummary } from './InsightsRequestsView'
import { ModelLabelsContext, modelText, type ModelLabels } from './insightsModelLabelSupport'
import { buildInsightsRecordSections, renderInsightsSavingsCell } from './insightsPageSupport'
import type {
  InsightsAggregateCostCoverage,
  InsightsAggregateResponse,
  InsightsRecord,
} from './insightsPageTypes'

const savingsColumn = (labels: ModelLabels = {}) => ({
  render: (record: InsightsRecord) => renderInsightsSavingsCell(record, labels),
})

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

const usageAndCost = (record: InsightsRecord, modelLabels?: Record<string, string>) =>
  buildInsightsRecordSections(record, { isReadonly: false, modelLabels }).find(
    (section) => section.title === 'Usage & Cost',
  )!

const field = (record: InsightsRecord, label: string, modelLabels?: Record<string, string>) =>
  usageAndCost(record, modelLabels).fields.find((item) => item.label === label)?.value

describe('Insights cost accounting labels', () => {
  it('marks list prices and caller logins as estimates and names the configured baseline', () => {
    expect(field(priced, 'Cost basis')).toBe(
      'Recorded tokens × the built-in catalog’s list price (an estimate: no pricing is configured) at capture time; not a GPU bill or provider invoice. It ran on the client’s own login: this is the API-price equivalent, not what the plan charges.',
    )
    expect(field(priced, 'Baseline basis')).toContain('the configured baseline model, small.')
    expect(field(priced, 'Baseline basis')).toContain(
      'a negative saving means the baseline was cheaper',
    )
    const savings = savingsColumn()
    const html = renderToStaticMarkup(<>{savings.render(priced)}</>)
    expect(html).toContain('-$0.0030')
    expect(html).not.toContain('text-success-foreground')
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

describe('Insights cost text names models by their labels', () => {
  const labels = { premium: 'Qwen3.6 35B A3B', small: 'qwen36', 'qwen-chat': 'qwen36' }

  it('keeps the configured name beside the label', () => {
    expect(modelText(labels, 'premium')).toBe('Qwen3.6 35B A3B (premium)')
    expect(modelText(labels, 'vllm-sr/auto')).toBe('vllm-sr/auto')
  })

  it('in the missing price, the baseline, the Router calls and the savings cell', () => {
    const unpriced: InsightsRecord = {
      ...priced,
      actual_cost: undefined,
      currency: undefined,
      cost_accounting: {
        unpriced_reason: 'pricing_not_configured',
        router_calls: [{ kind: 'decision_model', model: 'qwen-chat', calls: 2 }],
      },
    }
    expect(field(unpriced, 'Cost basis', labels)).toContain(
      'No price is configured for Qwen3.6 35B A3B (premium).',
    )
    expect(field(priced, 'Baseline basis', labels)).toContain(
      'the configured baseline model, qwen36 (small).',
    )
    expect(field(unpriced, 'Router calls', labels)).toContain(
      '2 Decision model calls on qwen36 (qwen-chat)',
    )
    const savings = savingsColumn(labels)
    const html = renderToStaticMarkup(
      <ModelLabelsContext.Provider value={labels}>
        {savings.render(priced)}
      </ModelLabelsContext.Provider>,
    )
    expect(html).toContain('Baseline (configured rates): <span')
    expect(html).toContain('title="Configured as small">qwen36</span>')
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

// The cost figures sit in the Requests summary row; the Breakdown below it holds no second copy.
describe('Insights cost coverage', () => {
  const render = (summary: InsightsAggregateResponse['summary'] = aggregate.summary) =>
    renderToStaticMarkup(
      <TooltipProvider>
        <RequestsSummary
          aggregate={{ ...aggregate, summary }}
          loading={false}
          error={null}
          onRetry={() => undefined}
          breakdown={false}
          onBreakdown={() => undefined}
          clientFiltered={false}
          harnessCounts={null}
        />
      </TooltipProvider>,
    )

  it('shows the spend and the saving as figures, the Router calls on their own line, the method behind a mark', () => {
    const html = render()
    expect(html).toContain('$3.50')
    // two rules compared these requests, so the label names neither
    expect(html).toContain('Saved vs reference')
    expect(html).toContain('−$1.00')
    expect(html).toContain('Cost only · 100.0% more than the reference')
    expect(html).toContain('Router’s own calls: $0.50')
    expect(html).toContain('aria-label="How the estimate is made"')
    expect(html).not.toContain('Cost vs reference')
    expect(html).not.toContain('Priced: 2 of 5 requests')
    expect(html).not.toContain('text-destructive-foreground')
  })

  it('counts requests refused before any model answered apart, as free', () => {
    const refused = {
      ...aggregate,
      summary: { ...aggregate.summary, coverage: { ...coverage, refused_records: 3 } },
    }
    expect(describeFreeRecords(refused.summary.coverage)).toBe(
      ' 3 requests refused before any model answered cost nothing.',
    )
  })

  it('counts requests no provider answered apart, as free', () => {
    const down = {
      ...aggregate,
      summary: { ...aggregate.summary, coverage: { ...coverage, unanswered_records: 56 } },
    }
    expect(describeFreeRecords(down.summary.coverage)).toBe(
      ' 56 requests no provider answered cost nothing.',
    )
  })

  it('opens the Breakdown on the cards alone', () => {
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <InsightsCharts aggregate={aggregate} />
      </TooltipProvider>,
    )
    expect(html).not.toContain('$3.50')
    expect(html).not.toContain('above reference')
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
    const html = render(summary)
    expect(html).toContain('No cost is shown because the models these requests used have no price.')
    expect(html).toContain('providers.models[].pricing')
    expect(html).not.toContain('N/A')
  })

  it('shows the saving before the Router calls, which read on their own line', () => {
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
      coverage: { ...coverage, baseline_sources: { configured_model: 2 } },
    }
    const html = render(summary)
    expect(html).toContain('Saved vs configured model')
    expect(html).toContain('$4.00')
    expect(html).not.toContain('$3.25')
    expect(html).toContain('Cost only · 66.7% less than the configured model')
    expect(html).toContain('Router’s own calls: $0.88')

    // the configuration names the model, by its label
    const named = renderToStaticMarkup(
      <InsightsConfigContext.Provider value={{ modelProviders: {}, costBaselineModel: 'small' }}>
        <ModelLabelsContext.Provider value={{ small: 'qwen36' }}>
          <TooltipProvider>
            <RequestsSummary
              aggregate={{ ...aggregate, summary }}
              loading={false}
              error={null}
              onRetry={() => undefined}
              breakdown={false}
              onBreakdown={() => undefined}
              clientFiltered={false}
              harnessCounts={null}
            />
          </TooltipProvider>
        </ModelLabelsContext.Provider>
      </InsightsConfigContext.Provider>,
    )
    expect(named).toContain('Saved vs qwen36')
    expect(named).toContain('Cost only · 66.7% less than qwen36')
  })

  it('reads a saving against the priciest model of each pool, positive where it saved', () => {
    const summary = {
      total_saved: 92.34,
      gross_saved: 92.34,
      baseline_spend: 261.98,
      actual_spend: 169.64,
      router_call_spend: 0,
      total_spend: 169.84,
      currency: 'USD',
      cost_record_count: 39856,
      excluded_record_count: 2683,
      coverage: {
        ...coverage,
        baseline_sources: { recipe_most_expensive: 39851, selected_model: 5 },
      },
    }
    const html = render(summary)
    expect(html).toContain('Saved vs priciest model')
    expect(html).toContain('$92.34')
    expect(html).not.toContain('−$92.34')
    expect(html).toContain('Cost only · 35.2% less than the priciest model')
    expect(html).toContain('Router’s own calls: $0.00')
  })

  it('leaves the difference unknown where nothing was compared with a reference', () => {
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
    const html = render(summary)
    expect(html).toContain(
      'aria-label="Not available: No request in range has both a priced model cost',
    )
    expect(html).not.toContain('$0.0000')
  })
})

describe('what a saving compares with', () => {
  const POOL_HINT =
    'Cost only: each request priced at the most expensive model in its pool for the same tokens, at this deployment’s own price table (catalog list prices when the pricing source says so); not a quality comparison.'

  it('names the priciest model of each pool, which direct requests do not change', () => {
    const reference = savingsReference({
      baseline_sources: { recipe_most_expensive: 40, selected_model: 2 },
    })
    expect(reference).toEqual({
      label: 'Saved vs priciest model',
      short: 'Priciest model',
      noun: 'the priciest model',
      hint: POOL_HINT,
      mixed: undefined,
    })
    // with nothing compared yet, the configuration decides
    expect(savingsReference(undefined).label).toBe('Saved vs priciest model')
    expect(savingsReference({ baseline_sources: {} }, 'small', { small: 'qwen36' }).label).toBe(
      'Saved vs qwen36',
    )
  })

  it('names the configured model, by its label', () => {
    const reference = savingsReference({ baseline_sources: { configured_model: 3 } }, 'small', {
      small: 'qwen36',
    })
    expect(reference.label).toBe('Saved vs qwen36')
    expect(reference.hint).toBe(
      'Cost only: each request priced at qwen36 for the same tokens, at this deployment’s own price table (catalog list prices when the pricing source says so); not a quality comparison.',
    )
    expect(savingsReference({ baseline_sources: { configured_model: 3 } }).label).toBe(
      'Saved vs configured model',
    )
  })

  it('says so when the requests of a range compared with different rules', () => {
    const reference = savingsReference(
      { baseline_sources: { recipe_most_expensive: 1200, configured_model: 26308, unrecorded: 4 } },
      'deepseek/deepseek-v4.1-flash',
    )
    expect(reference.label).toBe('Saved vs reference')
    expect(reference.mixed).toBe(
      'References differ in this range: 1,200 requests vs the priciest model in its pool, 26,308 requests vs deepseek/deepseek-v4.1-flash, 4 requests vs an unrecorded rule (older records).',
    )
    expect(
      describeEstimateMethod(
        {
          coverage: {
            ...coverage,
            baseline_sources: { recipe_most_expensive: 1, configured_model: 1 },
          },
          excluded_record_count: 0,
        },
        5,
      ),
    ).toContain('References differ in this range')
  })

  it('reads the method with the reference first and the Router calls apart', () => {
    const method = describeEstimateMethod(
      {
        coverage: { ...coverage, baseline_sources: { recipe_most_expensive: 2 } },
        excluded_record_count: 7,
      },
      9,
    )
    expect(method.startsWith(POOL_HINT)).toBe(true)
    expect(method).toContain('the Router’s own calls are shown on their own line and not taken off')
    expect(method).toContain('Priced: 2 of 9 requests, 1 partly.')
    expect(method).toContain('7 requests left out of the comparison.')
  })

  it('prints a saving plain and a loss with a minus sign', () => {
    expect(formatSaving(92.34, 'USD')).toBe('$92.34')
    expect(formatSaving(-7.97, 'USD')).toBe('−$7.97')
    expect(formatSaving(0, 'USD')).toBe('$0.00')
    expect(formatSaving(-1, undefined)).toBe('—')
  })

  it('reads the configured baseline model from global.plugins or a decision', () => {
    const replay = (model?: string) => ({
      type: 'router_replay',
      configuration: model ? { cost_baseline_model: model } : {},
    })
    expect(buildCostBaselineModel({ global: { plugins: [replay()] } })).toBeUndefined()
    expect(buildCostBaselineModel({ global: { plugins: [replay('small')] } })).toBe('small')
    expect(
      buildCostBaselineModel({
        global: { plugins: [replay()] },
        recipes: [
          {
            decisions: [{ plugins: [replay(' small ')] }, { plugins: [{ type: 'fast_response' }] }],
          },
        ],
      }),
    ).toBe('small')
    // two models leave it unnamed
    expect(
      buildCostBaselineModel({
        global: { plugins: [replay('small')] },
        recipes: [{ decisions: [{ plugins: [replay('premium')] }] }],
      }),
    ).toBeUndefined()
  })

  it('names what a record compared with', () => {
    const pool: InsightsRecord = {
      ...priced,
      cost_savings: 0.00144495,
      baseline_model: 'zai/glm-5.3-flash',
      cost_accounting: { baseline_source: 'recipe_most_expensive' },
    }
    expect(describeRecordSaving(pool)).toBe(
      'Saved $0.0014 vs zai/glm-5.3-flash, the priciest model in its pool for these tokens (cost only)',
    )
    expect(savingVersus(pool, { 'zai/glm-5.3-flash': 'GLM-5.3-Flash' })).toBe(
      'vs. GLM-5.3-Flash, the priciest in its pool',
    )
    expect(describeRecordSaving(priced)).toBe('Cost $0.0030 more than small (cost only)')
    expect(savingVersus(priced)).toBe('vs. small')
    const direct: InsightsRecord = {
      ...priced,
      cost_savings: 0,
      baseline_model: 'premium',
      cost_accounting: { baseline_source: 'selected_model' },
    }
    expect(describeRecordSaving(direct)).toBe(
      'A direct request, compared with its own model, premium (cost only)',
    )
    expect(savingVersus({ ...priced, baseline_model: undefined })).toBe('No baseline recorded')
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
      '1 Decision model call on primary, 120 input / 3 output tokens, 1 priced at $0.0020',
    )
    expect(field(record, 'Router calls')).toContain(
      '1 System One call on remote, 10 input / 0 output tokens, 1 unpriced',
    )
    expect(field(priced, 'Router calls')).toBeUndefined()
    // The record's own savings are gross, as the table's column says.
    expect(field(record, 'Estimated savings before its Router calls')).toBe('-$0.0030')
    expect(field(record, 'Estimated savings')).toBeUndefined()
    expect(field(priced, 'Estimated savings')).toBeDefined()
    const savings = savingsColumn()
    expect(renderToStaticMarkup(<>{savings.render(record)}</>)).toContain('Before its Router calls')
    expect(renderToStaticMarkup(<>{savings.render(priced)}</>)).not.toContain('Router calls')
  })
})

describe('spend budget on a record', () => {
  it('shows what the budget had left and what the call reserved and was charged', () => {
    const settled: InsightsRecord = {
      ...priced,
      cost_accounting: {
        ...priced.cost_accounting,
        budget: {
          scope: 'run',
          currency: 'USD',
          limit: 0.03,
          remaining: 0.024,
          estimate: 0.011,
          charged: 0.006,
          settlement: 'usage',
        },
      },
    }
    expect(field(settled, 'Spend budget')).toBe(
      'run: $0.024 of $0.03 left; reserved $0.011, charged $0.0060 from its usage',
    )
  })

  it('says a refused call did not fit', () => {
    const refused: InsightsRecord = {
      ...priced,
      lifecycle_state: 'failed',
      terminal_reason: 'budget_exceeded',
      cost_accounting: {
        budget: {
          scope: 'user/daily',
          currency: 'USD',
          limit: 0.05,
          remaining: 0.008,
          estimate: 0.011,
        },
      },
    }
    expect(field(refused, 'Spend budget')).toBe(
      'user/daily: $0.0080 of $0.05 left; refused: the call was estimated at $0.011',
    )
    const exhausted: InsightsRecord = {
      ...refused,
      cost_accounting: {
        budget: { scope: 'run', currency: 'USD', limit: 0.03, remaining: 0, estimate: 0 },
      },
    }
    expect(field(exhausted, 'Spend budget')).toBe(
      'run: $0.0000 of $0.03 left; refused: nothing was left',
    )
    expect(field({ ...priced }, 'Spend budget')).toBeUndefined()
  })
})
