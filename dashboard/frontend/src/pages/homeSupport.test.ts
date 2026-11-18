import { describe, expect, it } from 'vitest'

import {
  carriedP95,
  formatUpdatedAgo,
  greeting,
  latencyStats,
  percentile,
  readHomeRange,
  spendSummary,
  timeTicks,
  topShares,
  trafficFromSeries,
  trafficFromWindow,
} from './homeSupport'
import type { InsightsAggregateResponse, InsightsRecord } from './insightsPageTypes'

const HOUR = 3_600_000
const now = Date.parse('2026-11-18T12:00:00Z')

const record = (hoursAgo: number, extra: Partial<InsightsRecord> = {}): InsightsRecord =>
  ({
    id: `r-${hoursAgo}`,
    timestamp: new Date(now - hoursAgo * HOUR).toISOString(),
    turn_index: 0,
    decision_tier: 0,
    decision_priority: 0,
    signals: {},
    lifecycle_state: 'completed',
    ...extra,
  }) as InsightsRecord

describe('home traffic helpers', () => {
  it('reads the range and greets by the hour', () => {
    expect(readHomeRange('7d')).toBe('7d')
    expect(readHomeRange('all')).toBe('24h')
    expect(readHomeRange(null)).toBe('24h')
    expect(greeting(9)).toBe('Good morning')
    expect(greeting(15)).toBe('Good afternoon')
    expect(greeting(21)).toBe('Good evening')
  })

  it('takes nearest-rank percentiles of completed durations', () => {
    expect(percentile([], 95)).toBeUndefined()
    expect(percentile([5, 1, 3], 50)).toBe(3)
    expect(
      latencyStats([
        record(1, { duration_ms: 100 }),
        record(2, { duration_ms: 300 }),
        record(3, { duration_ms: 900, lifecycle_state: 'in_progress' }),
      ]),
    ).toEqual({ p50: 100, p95: 300, sampled: 2 })
  })

  it('buckets the newest records over the range when they hold all of it', () => {
    const records = [
      record(0.5, { duration_ms: 200, actual_cost: 0.01, baseline_cost: 0.05, currency: 'USD' }),
      record(0.6, { duration_ms: 400, actual_cost: 0.02, from_cache: true }),
      record(5.5, { duration_ms: 100, lifecycle_state: 'failed' }),
      record(30),
    ]
    const view = trafficFromWindow(records, '24h', records.length, now)
    expect(view?.coversRange).toBe(true)
    expect(view?.points).toHaveLength(24)
    expect(view?.points[23]).toMatchObject({ requests: 2, cached: 1, p95: 400 })
    expect(view?.points[23].cost).toBeCloseTo(0.03)
    expect(view?.points[23].baseline).toBeCloseTo(0.07)
    expect(view?.points[18]).toMatchObject({ requests: 1, failed: 1 })
    expect(view?.points.reduce((sum, point) => sum + point.requests, 0)).toBe(3)
    expect(view?.hasBaseline).toBe(true)
    expect(carriedP95(view?.points ?? []).slice(17, 20)).toEqual([0, 100, 100])
    expect(carriedP95(view?.points ?? [])[23]).toBe(400)

    expect(trafficFromWindow([], '1h', 0, now)).toBeNull()
  })

  it('spans only the newest records when the range holds more', () => {
    const records = [record(0.5), record(2)]
    const view = trafficFromWindow(records, '7d', 300, now)
    expect(view?.coversRange).toBe(false)
    expect(view?.sampled).toBe(2)
    expect(view?.points).toHaveLength(28)
    expect(view?.points[0].start).toBe(now - 2 * HOUR)
    expect(view?.points[0].requests).toBe(1)
    expect(view?.points.reduce((sum, point) => sum + point.requests, 0)).toBe(2)
  })

  it('reads the aggregate series and folds its tail sliver into the last interval', () => {
    const bucket = (hour: number, requests: number, extra: object = {}) => ({
      start: new Date(now - (24 - hour) * HOUR).toISOString(),
      requests,
      cached: 0,
      failed: 0,
      cost: [],
      ...extra,
    })
    const usd = (spend: number, saved: number, compared: number) => ({
      currency: 'USD',
      total_spend: spend,
      actual_spend: spend,
      total_saved: saved,
      baseline_spend: spend + saved,
      cost_record_count: compared,
    })
    const series = {
      since: new Date(now - 24 * HOUR).toISOString(),
      interval_seconds: 3600,
      latency_p50_ms: 120,
      latency_p95_ms: 900,
      buckets: [
        ...Array.from({ length: 23 }, (_, hour) => bucket(hour, 0)),
        bucket(23, 10, { latency_p95_ms: 800, cost: [usd(0.5, 0.25, 4)] }),
        bucket(24, 1, { latency_p95_ms: 5000, cost: [usd(0.1, 0, 0)] }),
      ],
    }
    const view = trafficFromSeries(series, '24h', 'USD')
    expect(view.points).toHaveLength(24)
    expect(view.points[23]).toMatchObject({ requests: 11, p95: 800 })
    expect(view.points[23].cost).toBeCloseTo(0.6)
    expect(view.points[23].baseline).toBeCloseTo(0.85)
    expect(view.points[0]).toMatchObject({ requests: 0, cost: 0, baseline: 0 })
    expect(view.latency).toEqual({ p50: 120, p95: 900 })
    expect(view.hasBaseline).toBe(true)
    expect(view.coversRange).toBe(true)
  })

  it('labels a time axis at round local times', () => {
    const start = new Date(2026, 10, 17, 7, 53).getTime()
    const day = timeTicks(start, start + 23 * HOUR)
    expect(day.stepMs).toBe(4 * HOUR)
    expect(day.ticks.map((tick) => new Date(tick).getHours())).toEqual([8, 12, 16, 20, 0, 4])
    expect(day.ticks.every((tick) => new Date(tick).getMinutes() === 0)).toBe(true)

    const hour = timeTicks(start, start + 55 * 60_000)
    expect(hour.stepMs).toBe(10 * 60_000)
    expect(hour.ticks.map((tick) => new Date(tick).getMinutes())).toEqual([0, 10, 20, 30, 40])

    const week = timeTicks(start, start + 27 * 6 * HOUR)
    expect(week.ticks.every((tick) => new Date(tick).getHours() === 0)).toBe(true)
    expect(week.ticks).toHaveLength(7)
  })

  it('says how long ago the page refreshed', () => {
    expect(formatUpdatedAgo(now - 2_000, now)).toBe('Updated just now')
    expect(formatUpdatedAgo(now - 42_000, now)).toBe('Updated 42 s ago')
    expect(formatUpdatedAgo(now - 5 * 60_000, now)).toBe('Updated 5 min ago')
  })

  it('summarizes the lead currency and leaves savings unknown without a baseline', () => {
    const aggregate = {
      summary: {
        total_saved: 1.5,
        baseline_spend: 4,
        actual_spend: 2.5,
        total_spend: 2.6,
        currency: 'USD',
        cost_record_count: 3,
        excluded_record_count: 0,
      },
    } as InsightsAggregateResponse
    expect(spendSummary(aggregate)).toEqual({
      currency: 'USD',
      spend: 2.6,
      saved: 1.5,
      baseline: 4,
      otherCurrencies: 0,
    })
    expect(
      spendSummary({
        summary: { ...aggregate.summary, cost_record_count: 0, total_spend: undefined },
      } as InsightsAggregateResponse),
    ).toMatchObject({ spend: 2.5, saved: undefined, baseline: undefined })
  })

  it('ranks a distribution with each entry’s share', () => {
    expect(
      topShares(
        [
          { name: 'b', value: 1 },
          { name: 'a', value: 3 },
          { name: '', value: 9 },
        ],
        1,
      ),
    ).toEqual([{ name: 'a', value: 3, share: 75 }])
  })
})
