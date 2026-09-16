import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import {
  buildInsightsRecordSections,
  buildInsightsSummary,
  createInsightsTableColumns,
  getInsightsCostUnavailableReason,
  hasCompleteCostData,
} from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'

const complete: InsightsRecord = {
  id: 'cost-record',
  timestamp: '2026-09-01T00:00:00Z',
  signals: {},
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 0,
  lifecycle_state: 'completed',
  total_tokens: 100,
  currency: 'USD',
  actual_cost: 0.001,
  baseline_cost: 0.004,
  cost_savings: 0.003,
  baseline_model: 'baseline',
}

describe('Insights configured-rate estimates', () => {
  it('distinguishes incomplete requests, missing usage, pricing and baseline', () => {
    expect(getInsightsCostUnavailableReason({ ...complete, lifecycle_state: 'in_progress' })).toBe(
      'Request not completed',
    )
    expect(getInsightsCostUnavailableReason({ ...complete, lifecycle_state: 'failed' })).toBe(
      'Request not completed',
    )
    expect(getInsightsCostUnavailableReason({ ...complete, total_tokens: undefined })).toBe(
      'Token usage unavailable',
    )
    expect(getInsightsCostUnavailableReason({ ...complete, actual_cost: undefined })).toBe(
      'Model pricing estimate unavailable',
    )
    expect(getInsightsCostUnavailableReason({ ...complete, baseline_cost: undefined })).toBe(
      'Baseline estimate unavailable',
    )
    expect(getInsightsCostUnavailableReason(complete)).toBeUndefined()
  })

  it('retains explicit free pricing but never replaces unknown prices with zero', () => {
    const free = { ...complete, actual_cost: 0, baseline_cost: 0, cost_savings: 0 }
    const unknown = {
      ...complete,
      actual_cost: undefined,
      baseline_cost: undefined,
      cost_savings: undefined,
    }
    expect(hasCompleteCostData(free)).toBe(true)
    expect(hasCompleteCostData(unknown)).toBe(false)
    const summary = buildInsightsSummary([free, unknown])
    expect(summary).toMatchObject({
      currency: 'USD',
      actualSpend: 0,
      costRecordCount: 1,
      excludedRecordCount: 1,
    })
  })

  it('preserves currency groups without adding amounts or excluding valid records', () => {
    const eur = { ...complete, id: 'eur', currency: 'EUR', actual_cost: 0.002 }
    const summary = buildInsightsSummary([complete, eur])
    expect(summary.currency).toBeUndefined()
    expect(summary.actualSpend).toBe(0)
    expect(summary.costRecordCount).toBe(2)
    expect(summary.excludedRecordCount).toBe(0)
    expect(summary.byCurrency).toMatchObject([
      { currency: 'EUR', actualSpend: 0.002, costRecordCount: 1 },
      { currency: 'USD', actualSpend: 0.001, costRecordCount: 1 },
    ])
  })

  it('labels configured-rate costs and exposes the baseline basis in details', () => {
    const columns = createInsightsTableColumns()
    expect(columns.find((c) => c.key === 'actual_cost')?.header).toBe('Estimated Cost')
    const section = buildInsightsRecordSections(complete, { isReadonly: false }).find(
      (s) => s.title === 'Usage & Cost',
    )
    expect(section?.fields).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: 'Cost basis',
          value: expect.stringContaining('configured model rates'),
        }),
        expect.objectContaining({
          label: 'Baseline basis',
          value: expect.stringContaining('same currency'),
        }),
        expect.objectContaining({ label: 'Estimated model cost' }),
      ]),
    )
    const row = { ...complete, actual_cost: undefined }
    const value = columns.find((c) => c.key === 'actual_cost')!.render!(row)
    const markup = renderToStaticMarkup(<>{value}</>)
    expect(markup).toContain('N/A')
    expect(markup).toContain('Model pricing estimate unavailable')
    expect(markup).not.toContain('$0')
  })
})
