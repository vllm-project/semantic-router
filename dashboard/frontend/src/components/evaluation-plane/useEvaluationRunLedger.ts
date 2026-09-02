import { useDeferredValue, useEffect, useMemo, useState } from 'react'

import type {
  EvaluationRun,
  EvaluationRunStatus,
  EvaluationTrackId,
} from '../../types/evaluationPlane'
import { filterEvaluationRuns } from './evaluationRunLedger'

const PAGE_SIZE = 10

export default function useEvaluationRunLedger(runs: EvaluationRun[]) {
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState<EvaluationRunStatus | 'all'>('all')
  const [track, setTrack] = useState<EvaluationTrackId | 'all'>('all')
  const [page, setPage] = useState(1)
  const deferredSearch = useDeferredValue(search.trim().toLowerCase())
  const filteredRuns = useMemo(
    () => filterEvaluationRuns(runs, { query: deferredSearch, status, track }),
    [deferredSearch, runs, status, track],
  )
  const pages = Math.max(1, Math.ceil(filteredRuns.length / PAGE_SIZE))
  const visibleRuns = filteredRuns.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  useEffect(() => setPage(1), [deferredSearch, status, track])
  useEffect(() => {
    if (page > pages) setPage(pages)
  }, [page, pages])

  return {
    search,
    status,
    track,
    page,
    pages,
    filteredRuns,
    visibleRuns,
    filtersActive: Boolean(search || status !== 'all' || track !== 'all'),
    setSearch,
    setStatus,
    setTrack,
    setPage,
    resetFilters: () => {
      setSearch('')
      setStatus('all')
      setTrack('all')
    },
  }
}

export type EvaluationRunLedgerModel = ReturnType<typeof useEvaluationRunLedger>
