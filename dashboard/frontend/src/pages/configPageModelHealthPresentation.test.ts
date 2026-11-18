import { describe, expect, it } from 'vitest'

import type { ModelHealth } from '../utils/modelHealthApi'
import {
  checkedAgo,
  formatLatency,
  healthSortValue,
  median,
  modelHealthHint,
  modelHealthTotals,
  visibleP95,
} from './configPageModelHealthPresentation'
import { isPermanentHealthFailure, mergeModelHealth } from './useBackendModelHealth'
import { parseBackendTraffic, trafficUnavailableReason } from './useBackendModelTraffic'
import { ModelHealthApiError } from '../utils/modelHealthApi'

const entry = (model: string, status: ModelHealth['status'], p95Ms?: number): ModelHealth => ({
  model,
  status,
  p95Ms,
  recentMs: p95Ms ? [p95Ms] : [],
  endpoints: [],
})

describe('model health presentation', () => {
  it('formats answer times the way the preview does', () => {
    expect(formatLatency(640.4)).toBe('640ms')
    expect(formatLatency(1380)).toBe('1.38s')
  })

  it('shows a p95 only while the model answers', () => {
    expect(visibleP95(entry('a', 'healthy', 640))).toBe(640)
    expect(visibleP95(entry('a', 'degraded', 3400))).toBe(3400)
    expect(visibleP95(entry('a', 'down', 900))).toBeUndefined()
    expect(visibleP95(undefined)).toBeUndefined()
  })

  it('sorts healthy before degraded, down and not checked, faster first', () => {
    const order = [
      entry('down', 'down'),
      entry('slow', 'healthy', 1200),
      undefined,
      entry('degraded', 'degraded', 300),
      entry('fast', 'healthy', 200),
    ]
      .map((health) => ({ health, value: healthSortValue(health) }))
      .sort((left, right) => left.value - right.value)
      .map(({ health }) => health?.model ?? 'unchecked')
    expect(order).toEqual(['fast', 'slow', 'degraded', 'down', 'unchecked'])
  })

  it('totals models by status and summarizes what is not healthy', () => {
    const byModel = new Map([
      ['a', entry('a', 'healthy', 100)],
      ['b', entry('b', 'degraded', 100)],
      ['c', entry('c', 'down')],
    ])
    const totals = modelHealthTotals(['a', 'b', 'c', 'd'], byModel)
    expect(totals).toEqual({ healthy: 1, degraded: 1, down: 1, unchecked: 1 })
    expect(modelHealthHint(totals)).toBe('1 degraded · 1 down · 1 not checked')
    expect(modelHealthHint({ healthy: 3, degraded: 0, down: 0, unchecked: 0 })).toBe(
      'Every model answers',
    )
  })

  it('takes the median of an odd or even set, and nothing of none', () => {
    expect(median([900, 100, 500])).toBe(500)
    expect(median([100, 300])).toBe(200)
    expect(median([])).toBeUndefined()
  })

  it('says how old a check is', () => {
    const now = Date.parse('2026-11-18T10:00:00Z')
    expect(checkedAgo('2026-11-18T09:59:58Z', now)).toBe('just now')
    expect(checkedAgo('2026-11-18T09:59:30Z', now)).toBe('30 s ago')
    expect(checkedAgo('2026-11-18T09:56:00Z', now)).toBe('4 min ago')
    expect(checkedAgo(undefined, now)).toBeNull()
  })
})

describe('backend model health state', () => {
  const snapshot = (models: ModelHealth[]) => ({
    checkedAt: '2026-11-18T10:00:00Z',
    intervalSeconds: 30,
    models,
  })

  it('replaces every model with a full snapshot and updates only the checked ones otherwise', () => {
    const current = new Map([
      ['a', entry('a', 'healthy', 100)],
      ['removed', entry('removed', 'down')],
    ])
    expect([...mergeModelHealth(current, snapshot([entry('a', 'down')]), false).keys()]).toEqual([
      'a',
    ])
    const partial = mergeModelHealth(current, snapshot([entry('a', 'down')]), true)
    expect(partial.get('a')?.status).toBe('down')
    expect(partial.has('removed')).toBe(true)
  })

  it('stops polling only when the Dashboard will keep refusing', () => {
    expect(isPermanentHealthFailure(new ModelHealthApiError('missing', 404))).toBe(true)
    expect(isPermanentHealthFailure(new ModelHealthApiError('forbidden', 403))).toBe(true)
    expect(isPermanentHealthFailure(new ModelHealthApiError('config', 503))).toBe(false)
    expect(isPermanentHealthFailure(new Error('network'))).toBe(false)
  })
})

describe('backend model traffic', () => {
  it('turns a Router Replay aggregate into shares, spend and savings', () => {
    const traffic = parseBackendTraffic({
      record_count: 200,
      model_selection: [
        { name: 'qwen3-8b', value: 150 },
        { name: 'gpt-5', value: 50 },
      ],
      summary: { total_spend: 4.25, currency: 'USD', baseline_spend: 10, total_saved: 5.75 },
    })
    expect(traffic?.requests).toBe(200)
    expect(traffic?.shares.get('qwen3-8b')).toBe(75)
    expect(traffic?.complete).toBe(true)
    expect(traffic?.spend).toEqual({ amount: 4.25, currency: 'USD', currencies: 1 })
    expect(traffic?.savedPercent).toBeCloseTo(57.5)
  })

  it('knows the shares are partial when Replay lists only the busiest ten', () => {
    const traffic = parseBackendTraffic({
      record_count: 1000,
      model_selection: Array.from({ length: 10 }, (_, index) => ({
        name: `m${index}`,
        value: 90,
      })),
      summary: {
        by_currency: [
          { currency: 'EUR', total_spend: 2 },
          { currency: 'USD', total_spend: 9 },
        ],
      },
    })
    expect(traffic?.complete).toBe(false)
    expect(traffic?.spend).toEqual({ amount: 9, currency: 'USD', currencies: 2 })
    expect(traffic?.savedPercent).toBeNull()
  })

  it('rejects answers that are not an aggregate and names why Replay is unavailable', () => {
    expect(parseBackendTraffic({ data: [] })).toBeNull()
    expect(parseBackendTraffic(null)).toBeNull()
    expect(trafficUnavailableReason(404)).toBe('replay_off')
    expect(trafficUnavailableReason(403)).toBe('forbidden')
    expect(trafficUnavailableReason(502)).toBe('router_unreachable')
  })
})
