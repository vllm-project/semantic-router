import { describe, expect, it } from 'vitest'

import { formatRequestCount, parsePublicModelTraffic } from './usePublicModelTraffic'

describe('public model traffic', () => {
  it('reads requests, p95, cost and the hourly series of an aggregate', () => {
    const entry = parsePublicModelTraffic({
      record_count: 48200,
      summary: { total_spend: 18.42, currency: 'USD' },
      traffic: {
        latency_p95_ms: 1240,
        buckets: [
          { start: '2026-11-17T10:00:00Z', requests: 1800 },
          { start: '2026-11-17T11:00:00Z', requests: 2100 },
        ],
      },
    })
    expect(entry).toEqual({
      requests: 48200,
      p95Ms: 1240,
      cost: { amount: 18.42, currency: 'USD' },
      hourly: [1800, 2100],
      hours: ['2026-11-17T10:00:00Z', '2026-11-17T11:00:00Z'],
    })
  })

  it('needs a traffic series and leaves out what Replay did not price', () => {
    expect(parsePublicModelTraffic({ record_count: 3 })).toBeNull()
    expect(
      parsePublicModelTraffic({ record_count: 0, summary: {}, traffic: { buckets: [] } }),
    ).toMatchObject({ requests: 0, cost: null, p95Ms: undefined })
  })

  it('shortens large request counts the way the cards show them', () => {
    expect(formatRequestCount(9120)).toBe('9,120')
    expect(formatRequestCount(48200)).toBe('48.2k')
    expect(formatRequestCount(2_400_000)).toBe('2.4M')
  })
})
