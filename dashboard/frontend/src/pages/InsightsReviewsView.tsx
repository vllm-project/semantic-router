import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'

import { Card, Skeleton } from '../ui'

import { InsightsLoadError, InsightsUnavailable, LiveToggle, Pager } from './InsightsDisplay'
import { insightsUnavailableReason, type InsightsUnavailableReason } from './insightsPageApi'
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
  unavailable?: InsightsUnavailableReason
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
      const unavailable = insightsUnavailableReason(err)
      setData({
        records: [],
        total: 0,
        aggregate: null,
        error: unavailable ? null : err instanceof Error ? err.message : 'Unknown error',
        unavailable,
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
  const { records, total, aggregate, error, unavailable, loading, refresh } = useReviewData(
    filters,
    page,
    autoRefresh,
  )

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

  if (unavailable) {
    return (
      <Card className="border-dashed shadow-none">
        <InsightsUnavailable reason={unavailable} noun="reviews" onRetry={() => void refresh()} />
      </Card>
    )
  }

  return (
    <>
      {error ? (
        <InsightsLoadError
          title="Reviews could not load"
          message={error}
          onRetry={() => void refresh()}
        />
      ) : null}
      {aggregate?.reviews ? (
        <Suspense fallback={<Skeleton className="mb-4 h-40 w-full rounded-xl" />}>
          <InsightsReviewCharts reviews={aggregate.reviews} />
        </Suspense>
      ) : null}
      <InsightsReviewsTable
        records={records}
        total={total}
        filters={filters}
        sessionInput={sessionInput}
        modelOptions={aggregate?.available_review_models ?? []}
        hasReviews={total > 0 || Boolean(aggregate?.available_review_models?.length) || loading}
        loading={loading}
        onSessionInput={setSessionInput}
        onFilterChange={onFilterChange}
        actions={
          <>
            <LiveToggle
              live={autoRefresh}
              onLiveChange={(next) => {
                setAutoRefresh(next)
                if (next) void refresh()
              }}
            />
          </>
        }
      />
      <Pager page={page} pageSize={reviewPageSize} total={total} noun="review" onPage={setPage} />
    </>
  )
}
