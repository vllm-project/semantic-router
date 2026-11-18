import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  REQUEST_CLASSES,
  requestClassOf,
  type AgentRuleIndex,
  type RequestClass,
} from './insightsAgentIdentity'
import { recordHarnessId } from './insightsHarness'
import { insightsUnavailableReason, type InsightsUnavailableReason } from './insightsPageApi'
import { buildReplayQueryString } from './insightsPageFilters'
import { fetchAbortableInsightsJSON, isAbortError } from './insightsPageRequestSupport'
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

export const REQUESTS_PAGE_SIZE = 25
/** Records the client-side filters and their counts read: the newest under the server filters. */
export const REQUEST_WINDOW_SIZE = 300
const WINDOW_PAGE = 100
const LIVE_INTERVAL_MS = 5_000
const WINDOW_INTERVAL_MS = 60_000
const REPLAYS = '/api/router/api/v1/observability/replays'

export type RequestRange = 'all' | '1h' | '24h' | '7d'
const RANGE_MS: Record<Exclude<RequestRange, 'all'>, number> = {
  '1h': 3_600_000,
  '24h': 86_400_000,
  '7d': 604_800_000,
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
  return {
    search: params.get('q') ?? '',
    recipe: params.get('recipe') ?? '',
    decision: params.get('decision') ?? '',
    model: params.get('model') ?? '',
    cache: cache === 'cached' || cache === 'streamed' ? cache : 'all',
    session: params.get('session') ?? '',
    turn: params.get('turn') ?? '',
    range: range === '1h' || range === '24h' || range === '7d' ? range : 'all',
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

/** The URL with some filters changed; a change returns the list to its first page. */
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
    if (!text || text === 'all') next.delete(name)
    else next.set(name, text)
  }
  next.delete('page')
  return next
}

export function hasClientFilters(filters: RequestFilters): boolean {
  return filters.harness.length > 0 || filters.classes.length > 0 || filters.hideHousekeeping
}

export function hasAnyFilter(filters: RequestFilters): boolean {
  return Boolean(
    filters.search ||
      filters.recipe ||
      filters.decision ||
      filters.model ||
      filters.cache !== 'all' ||
      filters.session ||
      filters.turn ||
      filters.range !== 'all' ||
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
function rangeSince(range: RequestRange, minute: number): string | undefined {
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
  loading: boolean
  error: string | null
  unavailable?: InsightsUnavailableReason
  refresh: () => void
}

interface PageState {
  rows: InsightsRecord[]
  total: number
  aggregate: InsightsAggregateResponse | null
  loading: boolean
  error: string | null
  unavailable?: InsightsUnavailableReason
}

/**
 * The request list: one server page and the aggregate of every record the
 * server filters select, polled while live; and a bounded window of the newest
 * records, refreshed less often, for the harness and request-class filters and
 * counts the replay API does not answer.
 */
export function useRequestsData({
  filters,
  page,
  live,
  held,
  rules,
  interval,
}: {
  filters: RequestFilters
  page: number
  live: boolean
  held: boolean
  rules?: AgentRuleIndex
  /** With a range, asks the aggregate for its traffic in intervals of this length, such as 1h. */
  interval?: string
}): RequestsData {
  const [minute, setMinute] = useState(() => Math.floor(Date.now() / MINUTE))
  const baseQuery = serverQuery(filters, rangeSince(filters.range, minute))
  const clientMode = hasClientFilters(filters)
  const pageQuery = withPage(baseQuery, REQUESTS_PAGE_SIZE, (page - 1) * REQUESTS_PAGE_SIZE)
  const aggregateQuery = withInterval(baseQuery, filters.range === 'all' ? undefined : interval)

  const [state, setState] = useState<PageState>({
    rows: [],
    total: 0,
    aggregate: null,
    loading: true,
    error: null,
  })
  const [windowState, setWindowState] = useState<{ records: InsightsRecord[]; total: number }>({
    records: [],
    total: 0,
  })
  const pageAbort = useRef<AbortController | null>(null)
  const windowAbort = useRef<AbortController | null>(null)
  const sequence = useRef(0)

  const loadPage = useCallback(async () => {
    const current = ++sequence.current
    pageAbort.current?.abort()
    const abort = new AbortController()
    pageAbort.current = abort
    setState((previous) => ({ ...previous, loading: true }))
    try {
      const [listResponse, aggregate] = await Promise.all([
        clientMode
          ? Promise.resolve(null)
          : fetchAbortableInsightsJSON<InsightsListResponse>(
              `${REPLAYS}${pageQuery}`,
              'insight records',
              abort.signal,
            ),
        fetchAbortableInsightsJSON<InsightsAggregateResponse>(
          `${REPLAYS}/aggregate${aggregateQuery}`,
          'insight aggregates',
          abort.signal,
        ),
      ])
      if (current !== sequence.current) return
      setState({
        rows: listResponse?.data ?? [],
        total: listResponse
          ? typeof listResponse.total === 'number'
            ? listResponse.total
            : listResponse.count
          : 0,
        aggregate,
        loading: false,
        error: null,
      })
    } catch (error) {
      if (isAbortError(error) || current !== sequence.current) return
      const unavailable = insightsUnavailableReason(error)
      setState({
        rows: [],
        total: 0,
        aggregate: null,
        loading: false,
        error: unavailable ? null : error instanceof Error ? error.message : 'Unknown error',
        unavailable,
      })
    }
  }, [aggregateQuery, clientMode, pageQuery])

  const loadWindow = useCallback(async () => {
    windowAbort.current?.abort()
    const abort = new AbortController()
    windowAbort.current = abort
    const records: InsightsRecord[] = []
    let total = 0
    try {
      for (let offset = 0; offset < REQUEST_WINDOW_SIZE; offset += WINDOW_PAGE) {
        const response = await fetchAbortableInsightsJSON<InsightsListResponse>(
          `${REPLAYS}${withPage(baseQuery, WINDOW_PAGE, offset)}`,
          'recent records',
          abort.signal,
        )
        const data = response.data ?? []
        records.push(...data)
        total = typeof response.total === 'number' ? response.total : records.length
        if (!response.has_more || data.length === 0) break
      }
      // a record written while paging shifts the newest-first pages; the id keeps each once
      const seen = new Set<string>()
      setWindowState({
        records: records.filter((record) => !seen.has(record.id) && Boolean(seen.add(record.id))),
        total,
      })
    } catch (error) {
      if (!isAbortError(error)) setWindowState({ records: [], total: 0 })
    }
  }, [baseQuery])

  useEffect(() => {
    void loadPage()
  }, [loadPage])

  useEffect(() => {
    void loadWindow()
  }, [loadWindow])

  useEffect(
    () => () => {
      sequence.current += 1
      pageAbort.current?.abort()
      windowAbort.current?.abort()
    },
    [],
  )

  const polling = live && !held
  useEffect(() => {
    if (!polling) return undefined
    let ticks = 0
    const interval = window.setInterval(() => {
      if (document.hidden) return
      ticks += 1
      const current = Math.floor(Date.now() / MINUTE)
      if (filters.range !== 'all' && current !== minute) {
        setMinute(current)
        return
      }
      void loadPage()
      if (ticks % Math.round(WINDOW_INTERVAL_MS / LIVE_INTERVAL_MS) === 0) void loadWindow()
    }, LIVE_INTERVAL_MS)
    return () => window.clearInterval(interval)
  }, [filters.range, loadPage, loadWindow, minute, polling])

  const refresh = useCallback(() => {
    const current = Math.floor(Date.now() / MINUTE)
    if (filters.range !== 'all' && current !== minute) {
      setMinute(current)
      return
    }
    void loadPage()
    void loadWindow()
  }, [filters.range, loadPage, loadWindow, minute])

  const filtered = useMemo(
    () =>
      clientMode
        ? windowState.records.filter((record) => matchesClientFilters(record, filters, rules))
        : [],
    [clientMode, filters, rules, windowState.records],
  )

  return {
    rows: clientMode
      ? filtered.slice((page - 1) * REQUESTS_PAGE_SIZE, page * REQUESTS_PAGE_SIZE)
      : state.rows,
    total: clientMode ? filtered.length : state.total,
    aggregate: state.aggregate,
    window: windowState.records,
    windowTotal: windowState.total,
    loading: state.loading,
    error: state.error,
    unavailable: state.unavailable,
    refresh,
  }
}

/** The recent runs by session: their titles and harnesses label the rows of their requests. */
export function useRecentRuns(enabled: boolean, refreshKey: string): Map<string, RunSummary> {
  const [runs, setRuns] = useState<Map<string, RunSummary>>(new Map())
  useEffect(() => {
    if (!enabled) return undefined
    const abort = new AbortController()
    fetchAbortableInsightsJSON<RunListResponse>(
      `${REPLAYS}/runs?limit=100&offset=0`,
      'agent runs',
      abort.signal,
    )
      .then((response) =>
        setRuns(new Map((response.data ?? []).map((run) => [run.session_id, run]))),
      )
      .catch(() => undefined)
    return () => abort.abort()
  }, [enabled, refreshKey])
  return runs
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
