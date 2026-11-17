import { describe, expect, it } from 'vitest'

import {
  bucketTraffic,
  greeting,
  latencyStats,
  percentile,
  readHomeRange,
  spendSummary,
  topShares,
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

  it('buckets the window over the range only when it holds every record of it', () => {
    const records = [
      record(0.5, { duration_ms: 200, actual_cost: 0.01 }),
      record(0.6, { duration_ms: 400, actual_cost: 0.02 }),
      record(5.5, { duration_ms: 100 }),
      record(30),
    ]
    const series = bucketTraffic(records, '24h', records.length, now)
    expect(series?.requests).toHaveLength(24)
    expect(series?.requests[23]).toBe(2)
    expect(series?.requests[18]).toBe(1)
    expect(series?.requests.reduce((sum, value) => sum + value, 0)).toBe(3)
    expect(series?.cost[23]).toBeCloseTo(0.03)
    expect(series?.p95[18]).toBe(100)
    expect(series?.p95[19]).toBe(100)
    expect(series?.p95[23]).toBe(400)

    expect(bucketTraffic(records, '24h', 500, now)).toBeNull()
    expect(bucketTraffic([], '1h', 0, now)).toBeNull()
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
