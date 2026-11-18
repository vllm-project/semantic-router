import { useCallback, useEffect, useRef, useState } from 'react'

import { insightsUnavailableReason, type InsightsUnavailableReason } from './insightsPageApi'
import { fetchAbortableInsightsJSON, isAbortError } from './insightsPageRequestSupport'

/** The page sizes the lists offer; the replay API returns at most 100 rows a page. */
export const PAGE_SIZES = [25, 50, 100] as const
export const DEFAULT_PAGE_SIZE = 25
/** Rows one window request reads: the replay API's page limit. */
export const WINDOW_PAGE = 100

/** The page size a URL asks for with `size`, else the default. */
export function readPageSize(params: URLSearchParams): number {
  const size = Number(params.get('size'))
  return (PAGE_SIZES as readonly number[]).includes(size) ? size : DEFAULT_PAGE_SIZE
}

/** The page a URL opens at with `page`, from 1. */
export function readPage(params: URLSearchParams): number {
  return Math.max(1, Math.floor(Number(params.get('page'))) || 1)
}

/** The URL at another page, or at another page size from its first page. */
export function writePage(params: URLSearchParams, page: number, size?: number): URLSearchParams {
  const next = new URLSearchParams(params)
  if (size !== undefined) {
    if (size === DEFAULT_PAGE_SIZE) next.delete('size')
    else next.set('size', String(size))
    next.delete('page')
    return next
  }
  if (page <= 1) next.delete('page')
  else next.set('page', String(page))
  return next
}

/** One page of rows a table pages itself, the page clamped to the pages there are. */
export function pageOf<T>(rows: readonly T[], page: number, size: number) {
  const pages = Math.max(1, Math.ceil(rows.length / size))
  const current = Math.min(Math.max(1, page), pages)
  return { rows: rows.slice((current - 1) * size, current * size), page: current, pages }
}

/** The page of a client-paged table that holds the row at index. */
export function pageHolding(index: number, size: number): number {
  return index < 0 ? 1 : Math.floor(index / size) + 1
}

/**
 * Calls tick every intervalMs while enabled and the tab is visible, and once
 * when the tab comes back into view.
 */
export function useLivePolling(enabled: boolean, intervalMs: number, tick: () => void) {
  const latest = useRef(tick)
  latest.current = tick
  useEffect(() => {
    if (!enabled) return undefined
    const run = () => {
      if (!document.hidden) latest.current()
    }
    const timer = window.setInterval(run, intervalMs)
    document.addEventListener('visibilitychange', run)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', run)
    }
  }, [enabled, intervalMs])
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown error'
}

export interface ReplayRead<T> {
  /** The latest answer; it may answer an earlier scope while the current one loads. */
  data: T | null
  /** The data answers an earlier scope, such as the filters before the last change. */
  stale: boolean
  loading: boolean
  error: string | null
  unavailable?: InsightsUnavailableReason
  /** When the last load started, in ms since the epoch, to pace polling. */
  requestedAt: number
  /** Loads again, unless a load of the same url is still running. */
  reload: () => void
}

interface ReadState<T> {
  data: T | null
  scope: string | null
  loading: boolean
  error: string | null
  unavailable?: InsightsUnavailableReason
}

/**
 * One replay read, loaded whenever url changes and on reload. scope names what
 * the answer is about apart from moving time bounds, so a minute's shift keeps
 * the old answer current while a filter change marks it stale.
 */
export function useReplayRead<T>(url: string | null, label: string, scope: string): ReplayRead<T> {
  const [state, setState] = useState<ReadState<T>>({
    data: null,
    scope: null,
    loading: url !== null,
    error: null,
  })
  const abortRef = useRef<AbortController | null>(null)
  const inFlight = useRef<string | null>(null)
  const requestedAt = useRef(0)

  const load = useCallback(
    async (force: boolean) => {
      if (!force && url !== null && inFlight.current === url) return
      abortRef.current?.abort()
      inFlight.current = url
      if (url === null) {
        setState((current) => (current.loading ? { ...current, loading: false } : current))
        return
      }
      const abort = new AbortController()
      abortRef.current = abort
      requestedAt.current = Date.now()
      setState((current) => (current.loading ? current : { ...current, loading: true }))
      try {
        const data = await fetchAbortableInsightsJSON<T>(url, label, abort.signal)
        if (abort.signal.aborted) return
        setState({ data, scope, loading: false, error: null })
      } catch (error) {
        if (abort.signal.aborted || isAbortError(error)) return
        const unavailable = insightsUnavailableReason(error)
        setState((current) => ({
          data: current.scope === scope ? current.data : null,
          scope: current.scope === scope ? scope : null,
          loading: false,
          error: unavailable ? null : messageOf(error),
          unavailable,
        }))
      } finally {
        if (abortRef.current === abort) inFlight.current = null
      }
    },
    [label, scope, url],
  )

  useEffect(() => {
    void load(true)
    return () => abortRef.current?.abort()
  }, [load])

  const reload = useCallback(() => void load(false), [load])
  return {
    data: state.data,
    stale: state.scope !== scope,
    loading: state.loading,
    error: state.error,
    unavailable: state.unavailable,
    requestedAt: requestedAt.current,
    reload,
  }
}

/** A list answer of the replay API: records and runs page the same way. */
export interface ReplayPage<T> {
  data?: T[] | null
  total?: number
  count?: number
  has_more?: boolean
}

/**
 * The newest rows under a query, up to size: the first page, then the rest at
 * once. A row written while paging shifts the newest-first pages, so keyOf
 * keeps each row once.
 */
export async function fetchReplayWindow<T>(
  pageUrl: (limit: number, offset: number) => string,
  label: string,
  size: number,
  keyOf: (row: T) => string,
  signal: AbortSignal,
): Promise<{ rows: T[]; total: number }> {
  const first = await fetchAbortableInsightsJSON<ReplayPage<T>>(
    pageUrl(Math.min(WINDOW_PAGE, size), 0),
    label,
    signal,
  )
  const rows = [...(first.data ?? [])]
  const total = typeof first.total === 'number' ? first.total : rows.length
  if (first.has_more && rows.length > 0) {
    const offsets: number[] = []
    for (let offset = WINDOW_PAGE; offset < Math.min(size, total); offset += WINDOW_PAGE) {
      offsets.push(offset)
    }
    const pages = await Promise.all(
      offsets.map((offset) =>
        fetchAbortableInsightsJSON<ReplayPage<T>>(
          pageUrl(Math.min(WINDOW_PAGE, size - offset), offset),
          label,
          signal,
        ),
      ),
    )
    for (const page of pages) rows.push(...(page.data ?? []))
  }
  const seen = new Set<string>()
  return {
    rows: rows.filter((row) => {
      const key = keyOf(row)
      if (seen.has(key)) return false
      seen.add(key)
      return true
    }),
    total,
  }
}

/**
 * What a window reads: nothing when off; the current query while live; for
 * counts alone the query it first read for the scope, so the minute a live
 * list moves on does not read it again. release lets go of that time bound,
 * on its own render unless forced by a change that renders anyway, and says
 * whether the window will read again.
 */
export function useWindowQuery(
  mode: 'off' | 'counts' | 'live',
  scope: string,
  query: string,
): [string | null, (force?: boolean) => boolean] {
  const frozen = useRef<{ scope: string; query: string } | null>(null)
  const [, rerender] = useState(0)
  let current: string | null = null
  if (mode === 'live') {
    frozen.current = { scope, query }
    current = query
  } else if (mode === 'counts') {
    if (frozen.current?.scope !== scope) frozen.current = { scope, query }
    current = frozen.current.query
  }
  const release = useCallback(
    (force = false) => {
      if (mode !== 'counts' || !frozen.current) return false
      if (!force && frozen.current.query === query) return false
      frozen.current = null
      if (!force) rerender((count) => count + 1)
      return true
    },
    [mode, query],
  )
  return [current, release]
}

export interface ReplayWindow<T> {
  rows: T[]
  total: number
  /** The rows answer the current scope. */
  loaded: boolean
  loading: boolean
  error: string | null
  unavailable?: InsightsUnavailableReason
  requestedAt: number
  reload: () => void
}

/** The newest rows under a query for the filters and counts the API does not answer. */
export function useReplayWindow<T>(
  pageUrl: ((limit: number, offset: number) => string) | null,
  label: string,
  size: number,
  keyOf: (row: T) => string,
  scope: string,
): ReplayWindow<T> {
  const [state, setState] = useState<{
    rows: T[]
    total: number
    scope: string | null
    loading: boolean
    error: string | null
    unavailable?: InsightsUnavailableReason
  }>({ rows: [], total: 0, scope: null, loading: pageUrl !== null, error: null })
  const abortRef = useRef<AbortController | null>(null)
  const running = useRef(false)
  const requestedAt = useRef(0)

  const load = useCallback(
    async (force: boolean) => {
      if (!force && running.current) return
      abortRef.current?.abort()
      if (!pageUrl) {
        running.current = false
        setState((current) => (current.loading ? { ...current, loading: false } : current))
        return
      }
      const abort = new AbortController()
      abortRef.current = abort
      running.current = true
      requestedAt.current = Date.now()
      setState((current) => (current.loading ? current : { ...current, loading: true }))
      try {
        const { rows, total } = await fetchReplayWindow(pageUrl, label, size, keyOf, abort.signal)
        if (abort.signal.aborted) return
        setState({ rows, total, scope, loading: false, error: null })
      } catch (error) {
        if (abort.signal.aborted || isAbortError(error)) return
        const unavailable = insightsUnavailableReason(error)
        setState((current) => ({
          rows: current.scope === scope ? current.rows : [],
          total: current.scope === scope ? current.total : 0,
          scope: current.scope === scope ? scope : null,
          loading: false,
          error: unavailable ? null : messageOf(error),
          unavailable,
        }))
      } finally {
        if (abortRef.current === abort) running.current = false
      }
    },
    [keyOf, label, pageUrl, scope, size],
  )

  useEffect(() => {
    void load(true)
    return () => abortRef.current?.abort()
  }, [load])

  const reload = useCallback(() => void load(false), [load])
  const loaded = state.scope === scope
  return {
    rows: loaded ? state.rows : [],
    total: loaded ? state.total : 0,
    loaded,
    loading: state.loading,
    error: state.error,
    unavailable: state.unavailable,
    requestedAt: requestedAt.current,
    reload,
  }
}
