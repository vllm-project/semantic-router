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
import { formatInsightsCost } from '../utils/insightsCost'
import InsightsRecordSection from './InsightsRecordSection'

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
  it('shows the estimate caveat with the metrics and keeps the longer explanation closed', () => {
    const section = buildInsightsRecordSections(complete, { isReadonly: false }).find(
      (item) => item.title === 'Usage & Cost',
    )!
    const html = renderToStaticMarkup(<InsightsRecordSection section={section} sectionIndex={0} />)
    const disclosure = html.indexOf('<details')
    expect(disclosure).toBeGreaterThan(0)
    expect(html.indexOf('not a GPU bill or provider invoice')).toBeLessThan(disclosure)
    expect(html.indexOf('Estimated model cost')).toBeLessThan(disclosure)
    expect(html.indexOf('highest estimate in the recipe')).toBeGreaterThan(disclosure)
    expect(html.indexOf('historical records are not repriced')).toBeGreaterThan(disclosure)
    expect(html).toContain('How these estimates are calculated')
    expect(html).not.toContain('open=""')
  })

  it('distinguishes tiny amounts, exact zero and unavailable data in each currency', () => {
    expect(formatInsightsCost(0.000001, 'USD')).toBe('<$0.0001')
    expect(formatInsightsCost(0.000075, 'USD')).toBe('<$0.0001')
    expect(formatInsightsCost(0.0001, 'USD')).toBe('$0.0001')
    expect(formatInsightsCost(-0.000001, 'USD')).toBe('>-$0.0001')
    expect(formatInsightsCost(0.000001, 'EUR')).toBe('<€0.0001')
    expect(formatInsightsCost(0, 'USD')).toBe('$0.0000')
    expect(formatInsightsCost(undefined, 'USD')).toBe('N/A')
    expect(formatInsightsCost(Number.NaN, 'USD')).toBe('N/A')
  })

  it('preserves tiny amounts in record cells and details', () => {
    const record = { ...complete, actual_cost: 0.000001 }
    const column = createInsightsTableColumns().find((item) => item.key === 'actual_cost')!
    expect(renderToStaticMarkup(<>{column.render!(record)}</>)).toContain('&lt;$0.0001')
    const usage = buildInsightsRecordSections(record, { isReadonly: false }).find(
      (section) => section.title === 'Usage & Cost',
    )
    expect(usage?.fields).toContainEqual({ label: 'Estimated model cost', value: '<$0.0001' })
  })

  it.each([
    ['baseline', 'Baseline model selected'],
    ['another-model', 'Equal estimated cost'],
  ])('explains exact zero savings when %s was selected', (selected_model, reason) => {
    const record = {
      ...complete,
      selected_model,
      actual_cost: complete.baseline_cost,
      cost_savings: 0,
    }
    const before = { ...record }
    const column = createInsightsTableColumns().find((item) => item.key === 'cost_savings')!
    const html = renderToStaticMarkup(<>{column.render!(record)}</>)
    expect(html).toContain('No savings')
    expect(html).toContain(reason)
    expect(html).toContain('Baseline (configured rates):')
    expect(html).not.toContain('$0.0000')
    const usage = buildInsightsRecordSections(record, { isReadonly: false }).find(
      (section) => section.title === 'Usage & Cost',
    )!
    expect(usage.fields).toContainEqual({
      label: 'Estimated savings',
      value: `No savings — ${reason}`,
    })
    expect(record).toEqual(before)
  })

  it('never describes tiny nonzero savings as no savings', () => {
    const record = { ...complete, selected_model: 'cheaper', cost_savings: 0.000001 }
    const column = createInsightsTableColumns().find((item) => item.key === 'cost_savings')!
    const html = renderToStaticMarkup(<>{column.render!(record)}</>)
    expect(html).toContain('&lt;$0.0001')
    expect(html).not.toContain('No savings')
    const usage = buildInsightsRecordSections(record, { isReadonly: false }).find(
      (section) => section.title === 'Usage & Cost',
    )!
    expect(usage.fields).toContainEqual({ label: 'Estimated savings', value: '<$0.0001' })
  })

  it('links current model configuration without presenting it as historical pricing', () => {
    const usage = buildInsightsRecordSections(complete, { isReadonly: false }).find(
      (section) => section.title === 'Usage & Cost',
    )!
    const field = usage.fields.find((item) => item.label === 'Current pricing')!
    const html = renderToStaticMarkup(<>{field.value}</>)
    expect(html).toContain('href="/config/models"')
    expect(html).toContain('View current configured model rates')
    expect(html).toContain('Current rates may differ from this record')
    expect(html).toContain('historical records are not repriced')
    expect(usage.fields).toContainEqual({
      label: 'Cost basis',
      value: expect.stringContaining('not a GPU bill or provider invoice'),
    })
  })

  it('distinguishes incomplete requests, missing usage, pricing and baseline', () => {
    expect(getInsightsCostUnavailableReason({ ...complete, lifecycle_state: 'in_progress' })).toBe(
      'Request not completed',
    )
    expect(getInsightsCostUnavailableReason({ ...complete, lifecycle_state: 'failed' })).toBe(
      'Request not completed',
    )
    expect(getInsightsCostUnavailableReason({ ...complete, total_tokens: undefined })).toBe(
      'Token usage was not recorded for this request',
    )
    expect(getInsightsCostUnavailableReason({ ...complete, actual_cost: undefined })).toBe(
      'No pricing estimate was recorded with this request. Historical records are not repriced using current model rates.',
    )
    expect(getInsightsCostUnavailableReason({ ...complete, baseline_cost: undefined })).toBe(
      'Baseline estimate was not recorded for this request',
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
    expect(markup).toContain('Price not recorded')
    expect(markup).toContain('Historical records are not repriced')
    expect(markup).not.toContain('$0')
  })

  it('explains historical missing prices in both cost cells without mutating the record', () => {
    const historical = {
      ...complete,
      actual_cost: undefined,
      baseline_cost: undefined,
      cost_savings: undefined,
      currency: undefined,
      baseline_model: undefined,
    }
    const before = { ...historical }
    const columns = createInsightsTableColumns()
    for (const key of ['actual_cost', 'cost_savings']) {
      const cell = columns.find((column) => column.key === key)!
      const markup = renderToStaticMarkup(<>{cell.render!(historical)}</>)
      expect(markup).toContain('Price not recorded')
      expect(markup).not.toContain('No savings')
      expect(markup).toContain('current model rates')
      expect(markup).not.toContain('>N/A<')
      expect(markup).not.toContain('$0')
    }
    expect(historical).toEqual(before)
    expect(buildInsightsSummary([historical]).costRecordCount).toBe(0)
  })

  it('explains missing prices and usage directly in record detail fields', () => {
    const historical = {
      ...complete,
      actual_cost: undefined,
      baseline_cost: undefined,
      cost_savings: undefined,
      currency: undefined,
      baseline_model: undefined,
    }
    for (const [record, label] of [
      [historical, 'Price not recorded'],
      [{ ...historical, total_tokens: undefined }, 'Usage not recorded'],
      [{ ...historical, lifecycle_state: 'failed' }, 'Not completed'],
    ] as const) {
      const usage = buildInsightsRecordSections(record, { isReadonly: false }).find(
        (section) => section.title === 'Usage & Cost',
      )!
      for (const field of [
        'Estimated model cost',
        'Estimated baseline cost',
        'Estimated savings',
      ]) {
        expect(usage.fields).toContainEqual({ label: field, value: label })
      }
      expect(usage.fields).toContainEqual({
        label: 'Baseline model',
        value: 'Baseline not recorded',
      })
      expect(usage.fields).toContainEqual({ label: 'Prompt tokens', value: 'Not recorded' })
      expect(usage.fields).not.toContainEqual(expect.objectContaining({ value: 'N/A' }))
    }
  })

  it('keeps a recorded free cost visible when only the baseline is missing', () => {
    const partial = {
      ...complete,
      actual_cost: 0,
      baseline_cost: undefined,
      baseline_model: undefined,
      cost_savings: undefined,
    }
    const usage = buildInsightsRecordSections(partial, { isReadonly: false }).find(
      (section) => section.title === 'Usage & Cost',
    )!
    expect(usage.fields).toContainEqual({ label: 'Estimated model cost', value: '$0.0000' })
    expect(usage.fields).toContainEqual({
      label: 'Estimated baseline cost',
      value: 'Baseline not recorded',
    })
    expect(usage.fields).toContainEqual({
      label: 'Estimated savings',
      value: 'Baseline not recorded',
    })
  })
})
