import { describe, expect, it } from 'vitest'

import {
  carriedP95,
  formatUpdatedAgo,
  greeting,
  hasTrafficSeries,
  homeAggregate,
  homeQuery,
  homeSince,
  homeSnapshotKey,
  homeTrafficView,
  latencyStats,
  needsTrafficWindow,
  parseHomeSnapshot,
  percentile,
  readHomeRange,
  runTitles,
  runTitlesSince,
  serializeHomeSnapshot,
  spendSummary,
  timeTicks,
  topShares,
  trafficFromSeries,
  trafficFromWindow,
  untitledSessions,
  unpricedHint,
  unpricedNote,
  type HomeAggregate,
  type HomeSnapshot,
} from './homeSupport'
import type { InsightsAggregateResponse, InsightsRecord } from './insightsPageTypes'
import type { RunSummary } from './insightsRunTypes'

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
  })

  it('draws an empty range when the newest records hold none', () => {
    const view = trafficFromWindow([], '1h', 0, now)
    expect(view.coversRange).toBe(true)
    expect(view.points).toHaveLength(12)
    expect(view.points[0].start).toBe(now - HOUR)
    expect(view.points.every((point) => point.requests === 0)).toBe(true)
    expect(view.latency).toEqual({ p50: undefined, p95: undefined })
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
      otherCurrencies: [],
      noPrice: 0,
    })
    expect(
      spendSummary({
        summary: { ...aggregate.summary, cost_record_count: 0, total_spend: undefined },
      } as InsightsAggregateResponse),
    ).toMatchObject({ spend: 2.5, saved: undefined, baseline: undefined })
  })

  it('reads money in one currency: a real one before plain units, the rest counted, never added', () => {
    const group = (currency: string, spend: number, calls = spend > 0 ? 1 : 0) => ({
      currency,
      total_saved: 0,
      baseline_spend: 0,
      actual_spend: spend,
      total_spend: spend,
      cost_record_count: calls,
    })
    const aggregate = (...groups: ReturnType<typeof group>[]) =>
      ({
        summary: {
          ...groups[0],
          excluded_record_count: 0,
          by_currency: groups,
          coverage: { unpriced_records: 93 },
        },
      }) as InsightsAggregateResponse
    // plain units lead only where nothing else priced a call
    expect(spendSummary(aggregate(group('USD', 0), group('XXX', 0.29)))).toMatchObject({
      currency: 'XXX',
      spend: 0.29,
      otherCurrencies: [],
    })
    // USD leads over a larger plain-unit spend: the older prices are counted, not added
    const mixed = spendSummary(aggregate(group('USD', 28.27, 11465), group('XXX', 144.2, 56397)))
    expect(mixed).toMatchObject({
      currency: 'USD',
      spend: 28.27,
      otherCurrencies: [{ currency: 'XXX', calls: 56397 }],
      noPrice: 93,
    })
    expect(unpricedHint(mixed)).toBe('56,490 calls unpriced')
    expect(unpricedNote(mixed)).toBe(
      'The total is in USD only. Not added: 56,397 calls priced in plain units, by rates set without a currency; 93 without a price (no usage or not completed).',
    )
    expect(
      spendSummary(aggregate(group('USD', 1), group('EUR', 3), group('XXX', 0))),
    ).toMatchObject({ currency: 'EUR', otherCurrencies: [{ currency: 'USD', calls: 1 }] })
    expect(unpricedHint({ otherCurrencies: [], noPrice: 0 })).toBeUndefined()
  })

  it('charts the aggregate series when the router has one, else the newest records', () => {
    const window = trafficFromWindow([record(1)], '24h', 1, now)
    const aggregate = {
      record_count: 1,
      lifecycle: { completed: 1, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
      summary: {
        total_saved: 0,
        baseline_spend: 0,
        actual_spend: 0,
        cost_record_count: 0,
        excluded_record_count: 0,
      },
      model_selection: [],
      decision_distribution: [],
    } as HomeAggregate
    expect(homeTrafficView(aggregate, window, '24h')).toBe(window)
    expect(homeTrafficView(null, null, '24h')).toBeNull()
    expect(needsTrafficWindow(aggregate)).toBe(true)
    expect(needsTrafficWindow({ ...aggregate, record_count: 0 })).toBe(false)
    expect(needsTrafficWindow(null)).toBe(false)

    const series = {
      since: new Date(now - 24 * HOUR).toISOString(),
      interval_seconds: 3600,
      buckets: [
        { start: new Date(now - HOUR).toISOString(), requests: 1, cached: 0, failed: 0, cost: [] },
      ],
    }
    const withSeries = { ...aggregate, traffic: series }
    expect(hasTrafficSeries(withSeries)).toBe(true)
    expect(needsTrafficWindow(withSeries)).toBe(false)
    const points = homeTrafficView(withSeries, window, '24h')?.points ?? []
    expect(points[points.length - 1]?.requests).toBe(1)
    expect(hasTrafficSeries({ ...aggregate, traffic: { ...series, buckets: [] } })).toBe(false)
  })

  it('keeps only what Home shows of the aggregate', () => {
    const full = {
      object: 'router_replay.aggregate',
      record_count: 4,
      lifecycle: { completed: 4, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
      summary: {
        total_saved: 0,
        baseline_spend: 0,
        actual_spend: 0,
        cost_record_count: 0,
        excluded_record_count: 0,
      },
      model_selection: [{ name: 'm', value: 4 }],
      decision_distribution: [{ name: 'd', value: 4 }],
      signal_distribution: [{ name: 's', value: 4 }],
      available_models: ['m'],
    } as unknown as InsightsAggregateResponse
    expect(homeAggregate(full)).toEqual({
      record_count: 4,
      lifecycle: full.lifecycle,
      summary: full.summary,
      model_selection: full.model_selection,
      decision_distribution: full.decision_distribution,
    })
  })

  it('queries a range from the start of its minute', () => {
    const at = Date.parse('2026-11-18T12:00:42.500Z')
    expect(homeSince('1h', at)).toBe('2026-11-18T11:00:00.000Z')
    expect(homeSince('7d', at)).toBe('2026-11-11T12:00:00.000Z')
    expect(homeQuery({ since: homeSince('24h', at), interval: '1h' })).toBe(
      '?since=2026-11-17T12%3A00%3A00.000Z&interval=1h',
    )
  })

  it('asks run titles only for sessions nothing names yet, from far enough back', () => {
    const rows = [
      record(0.1, { session_id: 'a' }),
      record(0.2, { session_id: 'a' }),
      record(0.3, { session_id: 'b', session_name: 'Fix the login flow' }),
      record(0.4, { session_id: 'c' }),
      record(0.5),
    ]
    expect(untitledSessions(rows, {})).toEqual(['a', 'c'])
    expect(untitledSessions(rows, { c: 'Ship the release' })).toEqual(['a'])
    expect(runTitlesSince(rows, now)).toBe(new Date(now - 24 * HOUR).toISOString())
    expect(runTitlesSince([record(50)], now + 30_000)).toBe(new Date(now - 50 * HOUR).toISOString())
    expect(
      runTitles([
        { session_id: 'a', session_name: 'Refactor the router' },
        { session_id: 'c', session_name: '  ' },
        { session_id: 'd' },
      ] as RunSummary[]),
    ).toEqual({ a: 'Refactor the router' })
  })

  it('keeps a snapshot for a day, for its account and range', () => {
    const snapshot: HomeSnapshot = {
      savedAt: now,
      aggregate: null,
      rows: [record(1)],
      window: trafficFromWindow([record(1)], '1h', 1, now),
      titles: { a: 'Title' },
    }
    const text = serializeHomeSnapshot(snapshot)
    expect(parseHomeSnapshot(text, now + HOUR)).toEqual(snapshot)
    expect(parseHomeSnapshot(text, now + 25 * HOUR)).toBeNull()
    expect(parseHomeSnapshot(JSON.stringify({ ...snapshot, v: 0 }), now)).toBeNull()
    expect(parseHomeSnapshot(JSON.stringify({ ...JSON.parse(text), rows: {} }), now)).toBeNull()
    expect(parseHomeSnapshot('{', now)).toBeNull()
    expect(parseHomeSnapshot(null, now)).toBeNull()
    const counted = { ...snapshot, harnessCounts: { claude_code: 3, api_client: 1 } }
    expect(parseHomeSnapshot(serializeHomeSnapshot(counted), now)).toEqual(counted)
    expect(homeSnapshotKey('user-1', '7d')).not.toBe(homeSnapshotKey('user-2', '7d'))
    expect(homeSnapshotKey('user-1', '7d')).not.toBe(homeSnapshotKey('user-1', '24h'))
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
