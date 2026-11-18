import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  REQUEST_CLASSES,
  requestClassOf,
  type AgentRuleIndex,
  type RequestClass,
} from './insightsAgentIdentity'
import { recordHarnessId } from './insightsHarness'
import {
  DEFAULT_PAGE_SIZE,
  useLivePolling,
  useReplayRead,
  useReplayWindow,
  useWindowQuery,
} from './insightsListData'
import type { InsightsUnavailableReason } from './insightsPageApi'
import { buildReplayQueryString } from './insightsPageFilters'
import { fetchAbortableInsightsJSON } from './insightsPageRequestSupport'
import type {
  InsightsAggregateResponse,
  InsightsFilterType,
  InsightsListResponse,
  InsightsRecord,
} from './insightsPageTypes'
import type { RunListResponse, RunSummary } from './insightsRunTypes'

export type InsightsTab = 'requests' | 'runs' | 'reviews'

export const INSIGHTS_TABS: { value: InsightsTab; label: string }[] = [
  { value: 'requests', label: 'Requests' },
  { value: 'runs', label: 'Agent runs' },
  { value: 'reviews', label: 'Reviews' },
]

/** The tab a URL opens: `tab`, or `view` as links from before the redesign name it. */
export function readInsightsTab(params: URLSearchParams): InsightsTab {
  const value = params.get('tab') ?? params.get('view')
  return value === 'runs' || value === 'reviews' ? value : 'requests'
}

export const REQUESTS_PAGE_SIZE = DEFAULT_PAGE_SIZE
/** Records the client-side filters and their counts read: the newest under the server filters. */
export const REQUEST_WINDOW_SIZE = 300
const LIVE_INTERVAL_MS = 5_000
/** The aggregate scans every record the filters select, so a live list reads it once a minute. */
const AGGREGATE_INTERVAL_MS = 60_000
const WINDOW_INTERVAL_MS = 60_000
const REPLAYS = '/api/router/api/v1/observability/replays'

export type RequestRange = 'all' | '1h' | '24h' | '7d'
const RANGE_MS: Record<Exclude<RequestRange, 'all'>, number> = {
  '1h': 3_600_000,
  '24h': 86_400_000,
  '7d': 604_800_000,
}

/** The ranges a list offers, shortest first. */
export const REQUEST_RANGES: { value: RequestRange; label: string }[] = [
  { value: '1h', label: '1h' },
  { value: '24h', label: '24h' },
  { value: '7d', label: '7d' },
  { value: 'all', label: 'All' },
]

/** A range as a sentence reads it: "in the last 24 hours". */
export const RANGE_TEXT: Record<RequestRange, string> = {
  '1h': 'in the last hour',
  '24h': 'in the last 24 hours',
  '7d': 'in the last 7 days',
  all: 'in all time',
}

export function isRequestRange(value: string | null): value is RequestRange {
  return value === 'all' || value === '1h' || value === '24h' || value === '7d'
}

/**
 * The range a list opens with: a session or a Codex turn reads in full, since
 * either is already narrow; anything else reads its last day.
 */
export function defaultRequestRange(scope: { session?: string; turn?: string }): RequestRange {
  return scope.session || scope.turn ? 'all' : '24h'
}

/** The request list's filters, as the page URL keeps them. */
export interface RequestFilters {
  search: string
  recipe: string
  decision: string
  model: string
  cache: InsightsFilterType
  session: string
  turn: string
  range: RequestRange
  harness: string[]
  classes: RequestClass[]
  hideHousekeeping: boolean
}

const list = (value: string | null) =>
  (value ?? '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

export function readRequestFilters(params: URLSearchParams): RequestFilters {
  const cache = params.get('cache')
  const range = params.get('range')
  const session = params.get('session') ?? ''
  const turn = params.get('turn') ?? ''
  return {
    search: params.get('q') ?? '',
    recipe: params.get('recipe') ?? '',
    decision: params.get('decision') ?? '',
    model: params.get('model') ?? '',
    cache: cache === 'cached' || cache === 'streamed' ? cache : 'all',
    session,
    turn,
    range: isRequestRange(range) ? range : defaultRequestRange({ session, turn }),
    harness: list(params.get('harness')),
    classes: list(params.get('class')).filter((value): value is RequestClass =>
      (REQUEST_CLASSES as readonly string[]).includes(value),
    ),
    hideHousekeeping: params.get('housekeeping') === 'hide',
  }
}

const PARAM_OF: Record<keyof RequestFilters, string> = {
  search: 'q',
  recipe: 'recipe',
  decision: 'decision',
  model: 'model',
  cache: 'cache',
  session: 'session',
  turn: 'turn',
  range: 'range',
  harness: 'harness',
  classes: 'class',
  hideHousekeeping: 'housekeeping',
}

/**
 * The URL with some filters changed; a change returns the list to its first
 * page, and a range equal to the default for the other filters stays out of it.
 */
export function writeRequestFilters(
  params: URLSearchParams,
  patch: Partial<RequestFilters>,
): URLSearchParams {
  const next = new URLSearchParams(params)
  for (const [key, value] of Object.entries(patch) as [keyof RequestFilters, unknown][]) {
    const name = PARAM_OF[key]
    const text = Array.isArray(value)
      ? value.join(',')
      : typeof value === 'boolean'
        ? value
          ? 'hide'
          : ''
        : String(value ?? '')
    if (!text || (key !== 'range' && text === 'all')) next.delete(name)
    else next.set(name, text)
  }
  const range = next.get('range')
  if (
    range !== null &&
    (!isRequestRange(range) ||
      range ===
        defaultRequestRange({ session: next.get('session') ?? '', turn: next.get('turn') ?? '' }))
  ) {
    next.delete('range')
  }
  next.delete('page')
  return next
}

export function hasClientFilters(filters: RequestFilters): boolean {
  return filters.harness.length > 0 || filters.classes.length > 0 || filters.hideHousekeeping
}

/** Whether any filter but the time range narrows the list. */
export function hasAnyFilter(filters: RequestFilters): boolean {
  return Boolean(
    filters.search ||
      filters.recipe ||
      filters.decision ||
      filters.model ||
      filters.cache !== 'all' ||
      filters.session ||
      filters.turn ||
      hasClientFilters(filters),
  )
}

/** The value a "harness" filter matches: the harness id, or "api" for none. */
export const API_CLIENT_FILTER = 'api'

export function recordClientFilterValue(record: InsightsRecord): string {
  return recordHarnessId(record) || API_CLIENT_FILTER
}

export function matchesClientFilters(
  record: InsightsRecord,
  filters: RequestFilters,
  rules?: AgentRuleIndex,
): boolean {
  if (filters.harness.length > 0 && !filters.harness.includes(recordClientFilterValue(record))) {
    return false
  }
  const requestClass = requestClassOf(record, rules)
  if (filters.classes.length > 0 && !filters.classes.includes(requestClass)) return false
  return !(filters.hideHousekeeping && requestClass === 'housekeeping')
}

const MINUTE = 60_000

/** The start of a range, to the minute, so a live list keeps one query a minute. */
export function requestRangeSince(range: RequestRange, minute: number): string | undefined {
  return range === 'all' ? undefined : new Date(minute * MINUTE - RANGE_MS[range]).toISOString()
}

function withInterval(query: string, interval: string | undefined): string {
  if (!interval) return query
  const params = new URLSearchParams(query)
  params.set('interval', interval)
  return `?${params.toString()}`
}

function withPage(query: string, limit: number, offset: number): string {
  const params = new URLSearchParams(query)
  params.set('limit', String(limit))
  params.set('offset', String(offset))
  params.set('showDetails', 'false')
  return `?${params.toString()}`
}

/** The server-side part of the filters as the replay API's query string. */
export function serverQuery(filters: RequestFilters, since: string | undefined): string {
  return buildReplayQueryString({
    searchTerm: filters.search,
    filter: filters.cache,
    recipeFilter: filters.recipe || 'all',
    decisionFilter: filters.decision || 'all',
    modelFilter: filters.model || 'all',
    turn: filters.turn,
    session: filters.session,
    since,
  })
}

export interface RequestsData {
  rows: InsightsRecord[]
  total: number
  aggregate: InsightsAggregateResponse | null
  /** The newest records under the server filters, which the client filters and counts read. */
  window: InsightsRecord[]
  windowTotal: number
  /** The window holds the newest records of the current filters. */
  windowLoaded: boolean
  windowLoading: boolean
  /** The first rows of the current filters have not arrived. */
  loading: boolean
  /** The rows shown answer the filters before the last change while the new ones load. */
  stale: boolean
  aggregateLoading: boolean
  /** The list's error, else the aggregate's. */
  error: string | null
  listError: string | null
  aggregateError: string | null
  unavailable?: InsightsUnavailableReason
  refresh: () => void
}

const recordId = (record: InsightsRecord) => record.id

/**
 * The request list, each part loaded on its own so the rows never wait: one
 * server page, polled while live; the aggregate of every record the server
 * filters select, read again once a minute; and, when withWindow or a client
 * filter asks for it, a window of the newest records for the harness and
 * request-class filters and counts the replay API does not answer. Only a
 * client filter keeps the window live; for counts alone it loads once per
 * set of filters.
 */
export function useRequestsData({
  filters,
  page,
  pageSize = REQUESTS_PAGE_SIZE,
  live,
  held,
  rules,
  interval,
  withWindow = true,
}: {
  filters: RequestFilters
  page: number
  pageSize?: number
  live: boolean
  held: boolean
  rules?: AgentRuleIndex
  /** With a range, asks the aggregate for its traffic in intervals of this length, such as 1h. */
  interval?: string
  withWindow?: boolean
}): RequestsData {
  const [minute, setMinute] = useState(() => Math.floor(Date.now() / MINUTE))
  const baseQuery = serverQuery(filters, requestRangeSince(filters.range, minute))
  const scope = `${serverQuery(filters, undefined)}|${filters.range}`
  const clientMode = hasClientFilters(filters)

  const listRead = useReplayRead<InsightsListResponse>(
    clientMode ? null : `${REPLAYS}${withPage(baseQuery, pageSize, (page - 1) * pageSize)}`,
    'insight records',
    `${scope}|${page}|${pageSize}`,
  )
  const aggregateRead = useReplayRead<InsightsAggregateResponse>(
    `${REPLAYS}/aggregate${withInterval(baseQuery, filters.range === 'all' ? undefined : interval)}`,
    'insight aggregates',
    `${scope}|${interval ?? ''}`,
  )

  const [windowQuery, releaseWindow] = useWindowQuery(
    clientMode ? 'live' : withWindow ? 'counts' : 'off',
    scope,
    baseQuery,
  )
  const windowUrl = useMemo(
    () =>
      windowQuery === null
        ? null
        : (limit: number, offset: number) => `${REPLAYS}${withPage(windowQuery, limit, offset)}`,
    [windowQuery],
  )
  const windowRead = useReplayWindow<InsightsRecord>(
    windowUrl,
    'recent records',
    REQUEST_WINDOW_SIZE,
    recordId,
    scope,
  )

  const nextMinute = () => {
    const current = Math.floor(Date.now() / MINUTE)
    if (filters.range === 'all' || current === minute) return false
    setMinute(current)
    return true
  }

  useLivePolling(live && !held, LIVE_INTERVAL_MS, () => {
    // a new minute moves the range, and every part reads again with it
    if (nextMinute()) return
    const now = Date.now()
    if (!clientMode) listRead.reload()
    if (now - aggregateRead.requestedAt >= AGGREGATE_INTERVAL_MS) aggregateRead.reload()
    if (clientMode && now - windowRead.requestedAt >= WINDOW_INTERVAL_MS) windowRead.reload()
  })

  const { reload: reloadList } = listRead
  const { reload: reloadAggregate } = aggregateRead
  const { reload: reloadWindow } = windowRead
  const refresh = useCallback(() => {
    const current = Math.floor(Date.now() / MINUTE)
    if (filters.range !== 'all' && current !== minute) {
      releaseWindow(true)
      setMinute(current)
      return
    }
    reloadList()
    reloadAggregate()
    if (!releaseWindow()) reloadWindow()
  }, [filters.range, minute, releaseWindow, reloadAggregate, reloadList, reloadWindow])

  const filtered = useMemo(
    () =>
      clientMode
        ? windowRead.rows.filter((record) => matchesClientFilters(record, filters, rules))
        : [],
    [clientMode, filters, rules, windowRead.rows],
  )

  const listResponse = listRead.data
  const listError = clientMode ? windowRead.error : listRead.error
  const aggregateError = aggregateRead.error
  return {
    rows: clientMode
      ? filtered.slice((page - 1) * pageSize, page * pageSize)
      : (listResponse?.data ?? []),
    total: clientMode
      ? filtered.length
      : listResponse
        ? typeof listResponse.total === 'number'
          ? listResponse.total
          : listResponse.count
        : 0,
    aggregate: aggregateRead.stale ? null : aggregateRead.data,
    window: windowRead.rows,
    windowTotal: windowRead.total,
    windowLoaded: windowRead.loaded,
    windowLoading: windowRead.loading,
    loading: clientMode
      ? !windowRead.loaded && !windowRead.error && !windowRead.unavailable
      : listRead.loading && (listResponse === null || listRead.stale),
    stale: clientMode ? false : listRead.stale && listResponse !== null,
    aggregateLoading: aggregateRead.loading,
    error: listError ?? aggregateError,
    listError,
    aggregateError,
    unavailable:
      (clientMode ? windowRead.unavailable : listRead.unavailable) ?? aggregateRead.unavailable,
    refresh,
  }
}

/**
 * The recent runs by session: their titles and harnesses label the rows of
 * their requests. since bounds the runs to those with calls after it, which
 * spares the router a scan of its whole history.
 */
export function useRecentRuns(
  enabled: boolean,
  refreshKey: string,
  since?: string,
): Map<string, RunSummary> {
  const [runs, setRuns] = useState<Map<string, RunSummary>>(new Map())
  useEffect(() => {
    if (!enabled) return undefined
    const abort = new AbortController()
    const query = new URLSearchParams({ limit: '100', offset: '0' })
    if (since) query.set('since', since)
    fetchAbortableInsightsJSON<RunListResponse>(
      `${REPLAYS}/runs?${query.toString()}`,
      'agent runs',
      abort.signal,
    )
      .then((response) =>
        setRuns(new Map((response.data ?? []).map((run) => [run.session_id, run]))),
      )
      .catch(() => undefined)
    return () => abort.abort()
  }, [enabled, refreshKey, since])
  return runs
}

/**
 * Whether Replay holds any record, asked only when enabled: when a range is
 * quiet it tells a router that never recorded anything from a quiet day.
 */
export function useAnyRecord(enabled: boolean): boolean | undefined {
  const [any, setAny] = useState<boolean>()
  useEffect(() => {
    if (!enabled) return undefined
    const abort = new AbortController()
    fetchAbortableInsightsJSON<InsightsListResponse>(
      `${REPLAYS}?limit=1&offset=0&showDetails=false`,
      'insight records',
      abort.signal,
    )
      .then((response) =>
        setAny((typeof response.total === 'number' ? response.total : response.count) > 0),
      )
      .catch(() => undefined)
    return () => abort.abort()
  }, [enabled])
  return any
}

/** How many records of the window each value of a facet holds, most first. */
export function countFacet(records: InsightsRecord[], valueOf: (record: InsightsRecord) => string) {
  const counts = new Map<string, number>()
  for (const record of records) {
    const value = valueOf(record)
    counts.set(value, (counts.get(value) ?? 0) + 1)
  }
  return counts
}
