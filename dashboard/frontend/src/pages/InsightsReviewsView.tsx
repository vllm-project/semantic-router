import { lazy, Suspense, useCallback, useEffect, useState } from 'react'

import { Card, Skeleton } from '../ui'

import { InsightsLoadError, InsightsUnavailable, LiveToggle, Pager } from './InsightsDisplay'
import { DEFAULT_PAGE_SIZE, useLivePolling, useReplayRead } from './insightsListData'
import type { InsightsAggregateResponse, InsightsListResponse } from './insightsPageTypes'
import {
  buildReviewQueryString,
  EMPTY_REVIEW_FILTERS,
  type ReviewFilters,
} from './insightsReviewSupport'
import InsightsReviewsTable from './InsightsReviewsTable'

const sessionDebounceMs = 300
const LIVE_INTERVAL_MS = 5_000
const SUMMARY_INTERVAL_MS = 60_000
const REPLAYS = '/api/router/api/v1/observability/replays'
const InsightsReviewCharts = lazy(() => import('../components/InsightsReviewCharts'))

// One page of reviews and the summary of every review the filters select,
// each loaded on its own; while live the page refreshes every few seconds and
// the summary once a minute.
function useReviewData(
  filters: ReviewFilters,
  page: number,
  pageSize: number,
  live: boolean,
  held: boolean,
) {
  const scope = buildReviewQueryString(filters)
  const list = useReplayRead<InsightsListResponse>(
    `${REPLAYS}${buildReviewQueryString(filters, { limit: pageSize, offset: (page - 1) * pageSize })}`,
    'approval reviews',
    `${scope}|${page}|${pageSize}`,
  )
  const summary = useReplayRead<InsightsAggregateResponse>(
    `${REPLAYS}/aggregate${scope}`,
    'review summary',
    scope,
  )
  useLivePolling(live && !held, LIVE_INTERVAL_MS, () => {
    list.reload()
    if (Date.now() - summary.requestedAt >= SUMMARY_INTERVAL_MS) summary.reload()
  })
  const { reload: reloadList } = list
  const { reload: reloadSummary } = summary
  const refresh = useCallback(() => {
    reloadList()
    reloadSummary()
  }, [reloadList, reloadSummary])
  const data = list.data
  return {
    records: data?.data ?? [],
    total: data ? (typeof data.total === 'number' ? data.total : data.count) : 0,
    aggregate: summary.stale ? null : summary.data,
    error: list.error ?? summary.error,
    unavailable: list.unavailable ?? summary.unavailable,
    loading: list.loading && (data === null || list.stale),
    refresh,
  }
}

export default function InsightsReviewsView({ initialSession = '' }: { initialSession?: string }) {
  const [filters, setFilters] = useState<ReviewFilters>({
    ...EMPTY_REVIEW_FILTERS,
    session: initialSession,
  })
  const [sessionInput, setSessionInput] = useState(initialSession)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const [live, setLive] = useState(true)
  const [held, setHeld] = useState(false)
  const { records, total, aggregate, error, unavailable, loading, refresh } = useReviewData(
    filters,
    page,
    pageSize,
    live,
    held,
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
        <InsightsUnavailable reason={unavailable} noun="reviews" onRetry={refresh} />
      </Card>
    )
  }

  return (
    <>
      {error ? (
        <InsightsLoadError title="Reviews could not load" message={error} onRetry={refresh} />
      ) : null}
      {aggregate?.reviews ? (
        <Suspense fallback={<Skeleton className="mb-4 h-40 w-full rounded-xl" />}>
          <InsightsReviewCharts reviews={aggregate.reviews} />
        </Suspense>
      ) : null}
      <div onPointerEnter={() => setHeld(true)} onPointerLeave={() => setHeld(false)}>
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
            <LiveToggle
              live={live}
              held={held}
              onLiveChange={(next) => {
                setLive(next)
                if (next) refresh()
              }}
            />
          }
        />
      </div>
      <Pager
        page={page}
        pageSize={pageSize}
        total={total}
        noun="review"
        onPage={setPage}
        onPageSize={(next) => {
          setPageSize(next)
          setPage(1)
        }}
      />
    </>
  )
}
