import { useCallback, useEffect, useMemo, useState } from 'react'

import { normalizeHarnessId } from './insightsHarness'
import { useLivePolling, useReplayRead, useReplayWindow, useWindowQuery } from './insightsListData'
import type { InsightsUnavailableReason } from './insightsPageApi'
import { fetchAbortableInsightsJSON } from './insightsPageRequestSupport'
import { isRequestRange, requestRangeSince, type RequestRange } from './insightsRequestsData'
import type { RunListResponse, RunSummary } from './insightsRunTypes'

const RUNS = '/api/router/api/v1/observability/replays/runs'
/** Runs the harness and status filters read: the latest active under the range and search. */
export const RUN_WINDOW_SIZE = 300
/** Each read summarizes every run in the range, so a live list reads again once a minute. */
const RUNS_INTERVAL_MS = 60_000
const MINUTE = 60_000

/** The value a harness filter matches for a run: its harness, or "api" for none. */
export const API_RUN_FILTER = 'api'

export function runHarnessValue(run: Pick<RunSummary, 'harness'>): string {
  return normalizeHarnessId(run.harness) || API_RUN_FILTER
}

/** The runs list's filters, as the page URL keeps them. */
export interface RunFilters {
  search: string
  range: RequestRange
  harness: string[]
  status: string[]
  /** One session's full id, as the Requests page's session filter names it. */
  session?: string
}

const csv = (value: string | null) =>
  (value ?? '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

/** The runs a URL asks for; the range defaults to the last day, as the request list's does. */
export function readRunFilters(params: URLSearchParams): RunFilters {
  const range = params.get('range')
  return {
    search: params.get('q') ?? '',
    range: isRequestRange(range) ? range : '24h',
    harness: csv(params.get('harness')),
    status: csv(params.get('status')),
  }
}

/** The URL with some run filters changed, back at the first page; the default range stays out. */
export function writeRunFilters(
  params: URLSearchParams,
  patch: Partial<RunFilters>,
): URLSearchParams {
  const next = new URLSearchParams(params)
  for (const [key, value] of Object.entries(patch) as [keyof RunFilters, unknown][]) {
    const name = key === 'search' ? 'q' : key
    const text = Array.isArray(value) ? value.join(',') : String(value ?? '').trim()
    if (!text || (key === 'range' && text === '24h')) next.delete(name)
    else next.set(name, text)
  }
  next.delete('page')
  return next
}

/** The runs API filters by search and time; harness and status are read in the browser. */
export function hasRunClientFilters(filters: RunFilters): boolean {
  return filters.harness.length > 0 || filters.status.length > 0
}

export function matchesRunFilters(run: RunSummary, filters: RunFilters): boolean {
  return (
    (filters.harness.length === 0 || filters.harness.includes(runHarnessValue(run))) &&
    (filters.status.length === 0 || filters.status.includes(run.outcome.status))
  )
}

/** The runs query: the search or session, and with a range the runs that made a call since it began. */
export function runsQuery(
  filters: Pick<RunFilters, 'search' | 'range' | 'session'>,
  minute: number,
): string {
  const query = new URLSearchParams()
  const search = filters.search.trim()
  if (search) query.set('search', search)
  if (filters.session) query.set('session_id', filters.session)
  const since = requestRangeSince(filters.range, minute)
  if (since) query.set('since', since)
  return query.toString()
}

function pageUrl(query: string, limit: number, offset: number): string {
  const params = new URLSearchParams(query)
  params.set('limit', String(limit))
  params.set('offset', String(offset))
  return `${RUNS}?${params.toString()}`
}

export interface RunsData {
  rows: RunSummary[]
  total: number
  /** The latest active runs under the search and range, for the browser's filters and counts. */
  window: RunSummary[]
  windowTotal: number
  windowLoaded: boolean
  loading: boolean
  /** The rows answer the filters before the last change while the new ones load. */
  stale: boolean
  error: string | null
  unavailable?: InsightsUnavailableReason
  refresh: () => void
}

const runKey = (run: RunSummary) => run.session_id

/**
 * The agent runs list: one server page of the runs with calls in the range,
 * polled once a minute while live, and, when withWindow or a harness or
 * status filter asks for it, the latest active runs those filters and their
 * counts read in the browser.
 */
export function useRunsData({
  filters,
  page,
  pageSize,
  live,
  held,
  withWindow,
  paused = false,
}: {
  filters: RunFilters
  page: number
  pageSize: number
  live: boolean
  held: boolean
  withWindow: boolean
  /** Reads nothing yet, as while a session filter is matched with its full id. */
  paused?: boolean
}): RunsData {
  const [minute, setMinute] = useState(() => Math.floor(Date.now() / MINUTE))
  const query = runsQuery(filters, minute)
  const scope = `${filters.search.trim()}|${filters.session ?? ''}|${filters.range}`
  const clientMode = hasRunClientFilters(filters)

  const listRead = useReplayRead<RunListResponse>(
    clientMode || paused ? null : pageUrl(query, pageSize, (page - 1) * pageSize),
    'agent runs',
    `${scope}|${page}|${pageSize}`,
  )
  const [windowQuery, releaseWindow] = useWindowQuery(
    paused ? 'off' : clientMode ? 'live' : withWindow ? 'counts' : 'off',
    scope,
    query,
  )
  const windowUrl = useMemo(
    () =>
      windowQuery === null
        ? null
        : (limit: number, offset: number) => pageUrl(windowQuery, limit, offset),
    [windowQuery],
  )
  const windowRead = useReplayWindow<RunSummary>(
    windowUrl,
    'agent runs',
    RUN_WINDOW_SIZE,
    runKey,
    scope,
  )

  useLivePolling(live && !held, RUNS_INTERVAL_MS, () => {
    const current = Math.floor(Date.now() / MINUTE)
    if (filters.range !== 'all' && current !== minute) {
      setMinute(current)
      return
    }
    if (clientMode) windowRead.reload()
    else listRead.reload()
  })

  const { reload: reloadList } = listRead
  const { reload: reloadWindow } = windowRead
  const refresh = useCallback(() => {
    const current = Math.floor(Date.now() / MINUTE)
    if (filters.range !== 'all' && current !== minute) {
      releaseWindow(true)
      setMinute(current)
      return
    }
    reloadList()
    if (!releaseWindow()) reloadWindow()
  }, [filters.range, minute, releaseWindow, reloadList, reloadWindow])

  const filtered = useMemo(
    () => (clientMode ? windowRead.rows.filter((run) => matchesRunFilters(run, filters)) : []),
    [clientMode, filters, windowRead.rows],
  )
  const list = listRead.data
  return {
    rows: clientMode ? filtered.slice((page - 1) * pageSize, page * pageSize) : (list?.data ?? []),
    total: clientMode ? filtered.length : (list?.total ?? 0),
    window: windowRead.rows,
    windowTotal: windowRead.total,
    windowLoaded: windowRead.loaded,
    loading:
      paused ||
      (clientMode
        ? !windowRead.loaded && !windowRead.error && !windowRead.unavailable
        : listRead.loading && (list === null || listRead.stale)),
    stale: clientMode ? false : listRead.stale && list !== null,
    error: clientMode ? windowRead.error : listRead.error,
    unavailable: clientMode ? windowRead.unavailable : listRead.unavailable,
    refresh,
  }
}

/**
 * One session's run summary, for its title and agents. The runs API reads
 * only that session's records, by an index, so this stays cheap on any history.
 */
export function useSessionRun(sessionId: string | undefined): RunSummary | undefined {
  const [state, setState] = useState<{ session: string; run?: RunSummary }>()
  useEffect(() => {
    if (!sessionId) return undefined
    const abort = new AbortController()
    const query = new URLSearchParams({ session_id: sessionId, limit: '1', offset: '0' })
    fetchAbortableInsightsJSON<RunListResponse>(
      `${RUNS}?${query.toString()}`,
      'agent run',
      abort.signal,
    )
      .then((list) =>
        setState({
          session: sessionId,
          run: (list.data ?? []).find((run) => run.session_id === sessionId),
        }),
      )
      .catch(() => undefined)
    return () => abort.abort()
  }, [sessionId])
  return sessionId && state?.session === sessionId ? state.run : undefined
}
