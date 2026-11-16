import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'

import configStyles from './ConfigPage.module.css'
import styles from './InsightsPage.module.css'
import { isInsightsReplayUnavailableError } from './insightsPageApi'
import { fetchAbortableInsightsJSON, isAbortError } from './insightsPageRequestSupport'
import type {
  InsightsAggregateResponse,
  InsightsListResponse,
  InsightsRecord,
} from './insightsPageTypes'
import {
  buildReviewQueryString,
  EMPTY_REVIEW_FILTERS,
  type ReviewFilters,
} from './insightsReviewSupport'
import InsightsReviewsTable from './InsightsReviewsTable'

const reviewPageSize = 25
const sessionDebounceMs = 300
const InsightsReviewCharts = lazy(() => import('../components/InsightsReviewCharts'))

interface ReviewData {
  records: InsightsRecord[]
  total: number
  aggregate: InsightsAggregateResponse | null
  error: string | null
  loading: boolean
}

// Loads one page of reviews and the summary of every review the filters
// select, dropping answers to superseded requests.
function useReviewData(filters: ReviewFilters, page: number, autoRefresh: boolean) {
  const [data, setData] = useState<ReviewData>({
    records: [],
    total: 0,
    aggregate: null,
    error: null,
    loading: true,
  })
  const abortRef = useRef<AbortController | null>(null)
  const refresh = useCallback(async () => {
    abortRef.current?.abort()
    const abort = new AbortController()
    abortRef.current = abort
    setData((current) => ({ ...current, loading: true }))
    const base = '/api/router/api/v1/observability/replays'
    try {
      const [list, aggregate] = await Promise.all([
        fetchAbortableInsightsJSON<InsightsListResponse>(
          `${base}${buildReviewQueryString(filters, { limit: reviewPageSize, offset: (page - 1) * reviewPageSize })}`,
          'approval reviews',
          abort.signal,
        ),
        fetchAbortableInsightsJSON<InsightsAggregateResponse>(
          `${base}/aggregate${buildReviewQueryString(filters)}`,
          'review summary',
          abort.signal,
        ),
      ])
      setData({
        records: list.data || [],
        total: typeof list.total === 'number' ? list.total : list.count,
        aggregate,
        error: null,
        loading: false,
      })
    } catch (err) {
      if (isAbortError(err)) return
      const unavailable = isInsightsReplayUnavailableError(err)
      const message = err instanceof Error ? err.message : 'Unknown error'
      setData({
        records: [],
        total: 0,
        aggregate: null,
        error: unavailable ? null : message,
        loading: false,
      })
    }
  }, [filters, page])

  useEffect(() => {
    void refresh()
    const interval = autoRefresh ? window.setInterval(() => void refresh(), 5000) : undefined
    return () => {
      window.clearInterval(interval)
      abortRef.current?.abort()
    }
  }, [autoRefresh, refresh])

  return { ...data, refresh }
}

export default function InsightsReviewsView({ initialSession = '' }: { initialSession?: string }) {
  const [filters, setFilters] = useState<ReviewFilters>({
    ...EMPTY_REVIEW_FILTERS,
    session: initialSession,
  })
  const [sessionInput, setSessionInput] = useState(initialSession)
  const [page, setPage] = useState(1)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const { records, total, aggregate, error, refresh } = useReviewData(filters, page, autoRefresh)
  const totalPages = Math.max(1, Math.ceil(total / reviewPageSize))

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setFilters((current) =>
        current.session === sessionInput ? current : { ...current, session: sessionInput },
      )
      setPage(1)
    }, sessionDebounceMs)
    return () => window.clearTimeout(timer)
  }, [sessionInput])

  const onFilterChange = useCallback((key: 'result' | 'risk' | 'model', value: string) => {
    setFilters((current) => ({ ...current, [key]: value }))
    setPage(1)
  }, [])

  return (
    <>
      {error ? (
        <div className={styles.error}>
          <span>{error}</span>
        </div>
      ) : null}
      {aggregate?.reviews ? (
        <Suspense fallback={null}>
          <InsightsReviewCharts reviews={aggregate.reviews} />
        </Suspense>
      ) : null}
      <div className={configStyles.sectionPanel}>
        <section className={configStyles.sectionTableBlock}>
          <div className={styles.toolbar}>
            <div>
              <h2 className={styles.sectionTitle}>Reviews</h2>
              <p className={styles.sectionSubtitle}>
                Each approval review an agent asked for: the planned action, the verdict, and what
                it cost. Open a row for the rationale and the reviewed session.
              </p>
            </div>
            <div className={styles.toolbarActions}>
              <label className={styles.toggle}>
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(event) => setAutoRefresh(event.target.checked)}
                />
                <span>Auto-refresh</span>
              </label>
              <button type="button" onClick={() => void refresh()} className={styles.refreshButton}>
                Refresh
              </button>
            </div>
          </div>
          <InsightsReviewsTable
            records={records}
            total={total}
            filters={filters}
            sessionInput={sessionInput}
            modelOptions={aggregate?.available_review_models ?? []}
            hasReviews={total > 0 || Boolean(aggregate?.available_review_models?.length)}
            onSessionInput={setSessionInput}
            onFilterChange={onFilterChange}
          />
        </section>
      </div>
      {total > reviewPageSize ? (
        <div className={styles.pagination}>
          <button
            type="button"
            className={styles.paginationButton}
            onClick={() => setPage((current) => Math.max(1, current - 1))}
            disabled={page === 1}
          >
            Previous
          </button>
          <span className={styles.paginationInfo}>
            Page {page} of {totalPages} ({total} reviews)
          </span>
          <button
            type="button"
            className={styles.paginationButton}
            onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
            disabled={page === totalPages}
          >
            Next
          </button>
        </div>
      ) : null}
    </>
  )
}
