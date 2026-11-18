import { describe, expect, it } from 'vitest'

import { homeSnapshotKey, serializeHomeSnapshot, type HomeAggregate } from './homeSupport'
import { createHomeTraffic, type FetchJSON } from './homeTraffic'
import { InsightsRequestError } from './insightsPageApi'
import type { InsightsRecord } from './insightsPageTypes'

const NOW = Date.parse('2026-11-18T12:00:00Z')
const HOUR = 3_600_000
const minutesAgo = (minutes: number) => new Date(NOW - minutes * 60_000).toISOString()

class MemoryStorage implements Storage {
  private items = new Map<string, string>()
  get length() {
    return this.items.size
  }
  clear() {
    this.items.clear()
  }
  getItem(key: string) {
    return this.items.get(key) ?? null
  }
  key(index: number) {
    return [...this.items.keys()][index] ?? null
  }
  removeItem(key: string) {
    this.items.delete(key)
  }
  setItem(key: string, value: string) {
    this.items.set(key, value)
  }
}

const record = (id: string, minutes: number, extra: Partial<InsightsRecord> = {}) =>
  ({
    id,
    timestamp: minutesAgo(minutes),
    turn_index: 0,
    decision_tier: 0,
    decision_priority: 0,
    signals: {},
    lifecycle_state: 'completed',
    duration_ms: 100 * minutes,
    ...extra,
  }) as InsightsRecord

const RECORDS = [
  record('r1', 1, { session_id: 'claude' }),
  record('r2', 2, { session_id: 'codex', session_name: 'Ship the release' }),
  record('r3', 3),
]

const aggregate = (count: number, series: boolean) => ({
  object: 'router_replay.aggregate',
  record_count: count,
  lifecycle: { completed: count, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0,
    baseline_spend: 0,
    actual_spend: 0,
    cost_record_count: 0,
    excluded_record_count: 0,
  },
  model_selection: [],
  decision_distribution: [],
  signal_distribution: [],
  ...(series
    ? {
        traffic: {
          since: new Date(NOW - 24 * HOUR).toISOString(),
          interval_seconds: 3600,
          buckets: [{ start: minutesAgo(60), requests: count, cached: 0, failed: 0, cost: [] }],
        },
      }
    : {}),
})

/** A replay API that answers at once, or fails a read `fail` picks. */
function replayApi({
  series = true,
  records = RECORDS,
  fail,
}: { series?: boolean; records?: InsightsRecord[]; fail?: (url: URL) => Error | undefined } = {}) {
  const urls: URL[] = []
  const fetchJSON: FetchJSON = async <T>(url: string) => {
    const parsed = new URL(url, 'https://dashboard.example')
    urls.push(parsed)
    const failure = fail?.(parsed)
    if (failure) throw failure
    if (parsed.pathname.endsWith('/aggregate')) return aggregate(records.length, series) as T
    if (parsed.pathname.endsWith('/runs'))
      return {
        object: 'router_replay.runs',
        data: [{ session_id: 'claude', session_name: 'Fix login' }],
      } as T
    const limit = Number(parsed.searchParams.get('limit'))
    return {
      object: 'router_replay.list',
      data: records.slice(0, limit),
      total: records.length,
    } as T
  }
  const reads = (kind: 'aggregate' | 'rows' | 'window' | 'runs') =>
    urls.filter((url) =>
      kind === 'aggregate'
        ? url.pathname.endsWith('/aggregate')
        : kind === 'runs'
          ? url.pathname.endsWith('/runs')
          : url.pathname.endsWith('/replays') &&
            url.searchParams.get('limit') === (kind === 'rows' ? '7' : '100'),
    )
  return { fetchJSON, urls, reads }
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

function source(api: { fetchJSON: FetchJSON }, storage: Storage = new MemoryStorage()) {
  return createHomeTraffic({ scope: 'user-1', storage, fetchJSON: api.fetchJSON, now: () => NOW })
}

describe('Home traffic', () => {
  it('reads the aggregate and the newest records apart, and charts the series without a window', async () => {
    const api = replayApi()
    const traffic = source(api)
    traffic.select('24h')
    expect(traffic.getState('24h')).toMatchObject({ aggregateLoading: true, rowsLoading: true })
    await settle()

    const state = traffic.getState('24h')
    expect(state).toMatchObject({ aggregateLoading: false, rowsLoading: false, window: null })
    expect(state.aggregate?.traffic?.buckets).toHaveLength(1)
    expect(state.rows.map((row) => row.id)).toEqual(['r1', 'r2', 'r3'])
    const [read] = api.reads('aggregate')
    expect(read.searchParams.get('since')).toBe(new Date(NOW - 24 * HOUR).toISOString())
    expect(read.searchParams.get('interval')).toBe('1h')
    expect(api.reads('rows')[0].searchParams.get('showDetails')).toBe('false')
    expect(api.reads('window')).toHaveLength(0)
  })

  it('charts a router without a series from one read of the newest 100 records per range', async () => {
    const api = replayApi({ series: false })
    const traffic = source(api)
    traffic.select('24h')
    await settle()
    expect(traffic.getState('24h').window).toMatchObject({ coversRange: true, sampled: 3 })
    expect(traffic.getState('24h').window?.latency).toEqual({ p50: 200, p95: 300 })

    traffic.refresh()
    await settle()
    traffic.select('7d')
    await settle()
    traffic.select('24h')
    await settle()
    expect(api.reads('aggregate')).toHaveLength(4)
    expect(api.reads('window').map((url) => url.searchParams.get('since'))).toEqual([
      new Date(NOW - 24 * HOUR).toISOString(),
      new Date(NOW - 7 * 24 * HOUR).toISOString(),
    ])
  })

  it('skips the window for an empty range', async () => {
    const api = replayApi({ series: false, records: [] })
    const traffic = source(api)
    traffic.select('1h')
    await settle()
    expect(traffic.getState('1h')).toMatchObject({ rows: [], window: null, rowsLoading: false })
    expect(api.reads('window')).toHaveLength(0)
    expect(api.reads('runs')).toHaveLength(0)
  })

  it('shows the last snapshot at once, per account, while it reads the range again', () => {
    const storage = new MemoryStorage()
    const cached = aggregate(3, false) as HomeAggregate
    storage.setItem(
      homeSnapshotKey('user-1', '24h'),
      serializeHomeSnapshot({
        savedAt: NOW - 60_000,
        aggregate: cached,
        rows: [RECORDS[0]],
        window: null,
        titles: { claude: 'Fix login' },
      }),
    )
    const urls: string[] = []
    const pending: FetchJSON = (url) => {
      urls.push(url)
      return new Promise<never>(() => undefined)
    }
    const traffic = createHomeTraffic({
      scope: 'user-1',
      storage,
      fetchJSON: pending,
      now: () => NOW,
    })
    expect(traffic.getState('24h')).toMatchObject({
      aggregate: { record_count: 3 },
      rows: [{ id: 'r1' }],
      titles: { claude: 'Fix login' },
    })

    traffic.select('24h')
    expect(traffic.getState('24h')).toMatchObject({ aggregateLoading: true, rows: [{ id: 'r1' }] })
    // The cached aggregate has no series, so the window reads beside the aggregate.
    expect(urls.filter((url) => url.includes('limit=100'))).toHaveLength(1)
    expect(urls.filter((url) => url.includes('/runs'))).toHaveLength(0)

    const other = createHomeTraffic({
      scope: 'user-2',
      storage,
      fetchJSON: pending,
      now: () => NOW,
    })
    expect(other.getState('24h').aggregate).toBeNull()
  })

  it('keeps each fresh answer for the next visit', async () => {
    const storage = new MemoryStorage()
    const traffic = source(replayApi({ series: false }), storage)
    traffic.select('24h')
    await settle()
    const next = source({ fetchJSON: () => new Promise<never>(() => undefined) }, storage)
    expect(next.getState('24h')).toMatchObject({
      aggregate: { record_count: 3 },
      window: { sampled: 3 },
      titles: { claude: 'Fix login' },
    })
    expect(next.getState('24h').rows).toHaveLength(3)
  })

  it('reads titles last and once, only for listed sessions the records leave unnamed', async () => {
    const api = replayApi()
    const traffic = source(api)
    traffic.select('24h')
    await settle()
    expect(traffic.getState('24h').titles).toEqual({ claude: 'Fix login' })
    const [runs] = api.reads('runs')
    expect(runs.searchParams.get('limit')).toBe('20')
    expect(runs.searchParams.get('since')).toBe(new Date(NOW - 24 * HOUR).toISOString())
    expect(api.urls.indexOf(runs)).toBeGreaterThan(api.urls.indexOf(api.reads('rows')[0]))

    traffic.refresh()
    await settle()
    expect(api.reads('runs')).toHaveLength(1)

    const named = replayApi({ records: [RECORDS[1], RECORDS[2]] })
    source(named).select('24h')
    await settle()
    expect(named.reads('runs')).toHaveLength(0)
  })

  it('explains a replay API that is off and forgets the range snapshot', async () => {
    const storage = new MemoryStorage()
    source(replayApi(), storage).select('24h')
    await settle()
    expect(storage.getItem(homeSnapshotKey('user-1', '24h'))).not.toBeNull()

    const off = replayApi({ fail: () => new InsightsRequestError('traffic', 404, 'Not Found') })
    const traffic = source(off, storage)
    traffic.select('24h')
    await settle()
    expect(traffic.getState('24h')).toMatchObject({
      unavailable: 'replay_off',
      aggregate: null,
      rows: [],
      error: null,
    })
    expect(storage.getItem(homeSnapshotKey('user-1', '24h'))).toBeNull()
  })

  it('keeps the last good traffic through a failed refresh and retries a failed window', async () => {
    let failAggregate = false
    let failWindow = true
    const api = replayApi({
      series: false,
      fail: (url) =>
        failAggregate && url.pathname.endsWith('/aggregate')
          ? new Error('The request timed out. Please retry.')
          : failWindow && url.searchParams.get('limit') === '100'
            ? new Error('Failed to fetch the newest requests: 500 Internal Server Error')
            : undefined,
    })
    const traffic = source(api)
    traffic.select('24h')
    await settle()
    expect(traffic.getState('24h')).toMatchObject({
      window: null,
      windowError: 'Failed to fetch the newest requests: 500 Internal Server Error',
    })

    failWindow = false
    failAggregate = true
    traffic.refresh()
    await settle()
    expect(traffic.getState('24h')).toMatchObject({
      aggregate: { record_count: 3 },
      error: 'The request timed out. Please retry.',
      window: { sampled: 3 },
      windowError: null,
    })
    expect(api.reads('window')).toHaveLength(2)
  })

  it('stops its reads when disposed and starts again on the next select', async () => {
    const signals: AbortSignal[] = []
    const fetchJSON: FetchJSON = (_url, _label, signal) => {
      signals.push(signal)
      return new Promise<never>((_resolve, reject) =>
        signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError'))),
      )
    }
    const traffic = createHomeTraffic({ scope: 'user-1', storage: new MemoryStorage(), fetchJSON })
    traffic.select('24h')
    expect(signals).toHaveLength(2)
    traffic.dispose()
    expect(signals.every((signal) => signal.aborted)).toBe(true)
    await settle()
    expect(traffic.getState('24h')).toMatchObject({ aggregateLoading: true, error: null })

    traffic.select('24h')
    expect(signals).toHaveLength(4)
    expect(signals.slice(2).some((signal) => signal.aborted)).toBe(false)
  })
})
