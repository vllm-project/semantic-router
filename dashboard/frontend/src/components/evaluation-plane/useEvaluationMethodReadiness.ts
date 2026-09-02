import { useMemo, useState } from 'react'

import type { EvaluationCatalog, EvaluationTrackId } from '../../types/evaluationPlane'
import {
  buildEvaluationMethodReadiness,
  countEvaluationMethodReadiness,
  type EvaluationMethodReadinessStatus,
  filterEvaluationMethodReadiness,
} from './evaluationMethodReadinessModel'

export default function useEvaluationMethodReadiness(catalog: EvaluationCatalog) {
  const [query, setQuery] = useState('')
  const [track, setTrack] = useState<EvaluationTrackId | 'all'>('all')
  const [status, setStatus] = useState<EvaluationMethodReadinessStatus | 'all'>('all')
  const methods = useMemo(() => buildEvaluationMethodReadiness(catalog), [catalog])
  const visibleMethods = useMemo(
    () => filterEvaluationMethodReadiness(methods, { query, track, status }),
    [methods, query, status, track],
  )
  const counts = useMemo(() => countEvaluationMethodReadiness(methods), [methods])
  return {
    query,
    track,
    status,
    methods,
    visibleMethods,
    counts,
    setQuery,
    setTrack,
    setStatus,
  }
}
