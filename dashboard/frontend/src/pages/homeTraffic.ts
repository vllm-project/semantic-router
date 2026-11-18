import { useEffect, useMemo, useSyncExternalStore } from 'react'

import { withRequestTimeout } from '../utils/boundedRequest'
import {
  HOME_RECENT_LIMIT,
  HOME_RUN_LIMIT,
  HOME_WINDOW_LIMIT,
  TRAFFIC_INTERVAL,
  homeAggregate,
  homeQuery,
  homeSince,
  homeSnapshotKey,
  needsTrafficWindow,
  parseHomeSnapshot,
  runTitles,
  runTitlesSince,
  serializeHomeSnapshot,
  trafficFromWindow,
  untitledSessions,
  type HomeAggregate,
  type HomeRange,
  type TrafficView,
} from './homeSupport'
import { insightsUnavailableReason, type InsightsUnavailableReason } from './insightsPageApi'
import { fetchAbortableInsightsJSON, isAbortError } from './insightsPageRequestSupport'
import type {
  InsightsAggregateResponse,
  InsightsListResponse,
  InsightsRecord,
} from './insightsPageTypes'
import type { RunListResponse } from './insightsRunTypes'

const REPLAYS = '/api/router/api/v1/observability/replays'
const TIMEOUT_MS = 25_000

export interface HomeTrafficState {
  aggregate: HomeAggregate | null
  /** The newest records of the range, newest first. */
  rows: InsightsRecord[]
  /** The newest records' traffic, for a router whose aggregate has no series. */
  window: TrafficView | null
  /** Agent session titles by session id, for listed records that carry none. */
  titles: Record<string, string>
  /** The aggregate is being read, or has not answered yet. */
  aggregateLoading: boolean
  rowsLoading: boolean
  /** Why the aggregate did not load, when the replay API gave no reason Home explains. */
  error: string | null
  rowsError: string | null
  windowError: string | null
  unavailable?: InsightsUnavailableReason
}

export type FetchJSON = <T>(url: string, label: string, signal: AbortSignal) => Promise<T>

type Read = 'aggregate' | 'rows' | 'window' | 'titles'

const fetchReplayJSON: FetchJSON = (url, label, signal) =>
  withRequestTimeout(
    (request) => fetchAbortableInsightsJSON(url, label, request),
    signal,
    TIMEOUT_MS,
  )

function tabStorage(): Storage | undefined {
  try {
    return typeof window === 'undefined' ? undefined : window.sessionStorage
  } catch {
    return undefined
  }
}

const message = (error: unknown) => (error instanceof Error ? error.message : 'Unknown error')

/**
 * Home's traffic per range, from Router Replay. The aggregate and the newest
 * records load independently; when the aggregate has no traffic series, one read
 * of the newest records per range draws the chart instead, and session titles
 * load last, only for listed sessions the records leave unnamed. Each range's
 * last good answer is kept for the tab's session and shown at once while a fresh
 * read runs.
 */
export function createHomeTraffic({
  scope,
  storage = tabStorage(),
  fetchJSON = fetchReplayJSON,
  now = Date.now,
}: {
  /** Keeps one account's snapshot from another's in a shared tab. */
  scope: string
  storage?: Storage
  fetchJSON?: FetchJSON
  now?: () => number
}) {
  const states = new Map<HomeRange, HomeTrafficState>()
  const listeners = new Set<() => void>()
  const answered = new Set<string>()
  const windowRead = new Set<HomeRange>()
  const titlesAsked = new Set<string>()
  let controller = new AbortController()
  let reading = new Set<string>()
  let active: HomeRange | null = null

  const loading = (range: HomeRange, kind: Read) =>
    reading.has(`${range}:${kind}`) || !answered.has(`${range}:${kind}`)

  const flags = (range: HomeRange) => ({
    aggregateLoading: loading(range, 'aggregate'),
    rowsLoading: loading(range, 'rows'),
  })

  function get(range: HomeRange): HomeTrafficState {
    const known = states.get(range)
    if (known) return known
    let saved = null
    try {
      saved = parseHomeSnapshot(storage?.getItem(homeSnapshotKey(scope, range)) ?? null, now())
    } catch {
      saved = null
    }
    const state: HomeTrafficState = {
      aggregate: saved?.aggregate ?? null,
      rows: saved?.rows ?? [],
      window: saved?.window ?? null,
      titles: saved?.titles ?? {},
      ...flags(range),
      error: null,
      rowsError: null,
      windowError: null,
    }
    states.set(range, state)
    return state
  }

  function persist(range: HomeRange, state: HomeTrafficState) {
    const key = homeSnapshotKey(scope, range)
    try {
      if (state.unavailable) storage?.removeItem(key)
      else if (state.aggregate || state.rows.length)
        storage?.setItem(
          key,
          serializeHomeSnapshot({
            savedAt: now(),
            aggregate: state.aggregate,
            rows: state.rows,
            window: state.window,
            titles: state.titles,
          }),
        )
    } catch {
      // A full or blocked storage only costs the next visit its head start.
    }
  }

  function publish(range: HomeRange, patch: Partial<HomeTrafficState>, save = false) {
    const previous = get(range)
    const next: HomeTrafficState = { ...previous, ...patch, ...flags(range) }
    const keys = Object.keys(next) as (keyof HomeTrafficState)[]
    if (keys.every((key) => next[key] === previous[key])) return
    states.set(range, next)
    if (save) persist(range, next)
    listeners.forEach((listener) => listener())
  }

  function read<T>(
    range: HomeRange,
    kind: Read,
    url: string,
    label: string,
    on: { done: (data: T) => void; failed: (error: unknown) => void; aborted?: () => void },
  ): boolean {
    const key = `${range}:${kind}`
    if (reading.has(key)) return false
    const current = reading
    const signal = controller.signal
    current.add(key)
    publish(range, {})
    fetchJSON<T>(url, label, signal)
      .then((data) => {
        current.delete(key)
        answered.add(key)
        on.done(data)
      })
      .catch((error: unknown) => {
        current.delete(key)
        if (signal.aborted || isAbortError(error)) {
          on.aborted?.()
          publish(range, {})
          return
        }
        answered.add(key)
        on.failed(error)
      })
    return true
  }

  function readAggregate(range: HomeRange) {
    const query = homeQuery({ since: homeSince(range, now()), interval: TRAFFIC_INTERVAL[range] })
    read<InsightsAggregateResponse>(range, 'aggregate', `${REPLAYS}/aggregate${query}`, 'traffic', {
      done: (response) => {
        publish(
          range,
          { aggregate: homeAggregate(response), error: null, unavailable: undefined },
          true,
        )
        readWindow(range)
      },
      failed: (error) => {
        const unavailable = insightsUnavailableReason(error)
        if (!unavailable) return publish(range, { error: message(error) })
        publish(
          range,
          { aggregate: null, rows: [], window: null, error: null, rowsError: null, unavailable },
          true,
        )
      },
    })
  }

  function readRows(range: HomeRange) {
    const query = homeQuery({
      since: homeSince(range, now()),
      limit: HOME_RECENT_LIMIT,
      offset: 0,
      showDetails: 'false',
    })
    read<InsightsListResponse>(range, 'rows', `${REPLAYS}${query}`, 'recent requests', {
      done: (list) => {
        publish(range, { rows: list.data ?? [], rowsError: null }, true)
        readTitles(range)
      },
      failed: (error) => {
        const unavailable = insightsUnavailableReason(error)
        publish(range, unavailable ? { rows: [], unavailable } : { rowsError: message(error) })
      },
    })
  }

  /** Once per range: the newest records chart a range whose aggregate has no series. */
  function readWindow(range: HomeRange) {
    if (range !== active || windowRead.has(range) || !needsTrafficWindow(get(range).aggregate))
      return
    const query = homeQuery({
      since: homeSince(range, now()),
      limit: HOME_WINDOW_LIMIT,
      offset: 0,
      showDetails: 'false',
    })
    const started = read<InsightsListResponse>(
      range,
      'window',
      `${REPLAYS}${query}`,
      'the newest requests',
      {
        done: (list) => {
          const records = list.data ?? []
          const total = typeof list.total === 'number' ? list.total : records.length
          publish(
            range,
            { window: trafficFromWindow(records, range, total, now()), windowError: null },
            true,
          )
        },
        failed: (error) => {
          windowRead.delete(range)
          publish(range, { windowError: message(error) })
        },
        aborted: () => windowRead.delete(range),
      },
    )
    if (started) windowRead.add(range)
  }

  /** The titles of listed sessions the records leave unnamed, each asked once. */
  function readTitles(range: HomeRange) {
    if (range !== active) return
    const { rows, titles } = get(range)
    const asked = untitledSessions(rows, titles)
      .map((session) => `${range}:${session}`)
      .filter((key) => !titlesAsked.has(key))
    if (!asked.length) return
    const query = homeQuery({
      since: runTitlesSince(rows, now()),
      limit: HOME_RUN_LIMIT,
      offset: 0,
    })
    const started = read<RunListResponse>(
      range,
      'titles',
      `${REPLAYS}/runs${query}`,
      'agent runs',
      {
        done: (list) =>
          publish(range, { titles: { ...get(range).titles, ...runTitles(list.data ?? []) } }, true),
        failed: () => undefined,
        aborted: () => asked.forEach((key) => titlesAsked.delete(key)),
      },
    )
    if (started) asked.forEach((key) => titlesAsked.add(key))
  }

  function refresh() {
    if (!active) return
    readAggregate(active)
    readRows(active)
    // A cached aggregate without a series already says the chart needs the newest records.
    readWindow(active)
  }

  return {
    getState: get,
    subscribe(listener: () => void) {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    /** Shows a range and reads it fresh. */
    select(range: HomeRange) {
      if (active === range) return
      active = range
      refresh()
    },
    refresh,
    /** Stops every read; a later select starts again. */
    dispose() {
      controller.abort()
      controller = new AbortController()
      reading = new Set()
      active = null
    },
  }
}

export type HomeTraffic = HomeTrafficState & { refresh: () => void }

export function useHomeTraffic(range: HomeRange, scope: string): HomeTraffic {
  const source = useMemo(() => createHomeTraffic({ scope }), [scope])
  const getState = () => source.getState(range)
  const state = useSyncExternalStore(source.subscribe, getState, getState)
  useEffect(() => {
    source.select(range)
  }, [source, range])
  useEffect(() => () => source.dispose(), [source])
  return useMemo(() => ({ ...state, refresh: source.refresh }), [state, source])
}
