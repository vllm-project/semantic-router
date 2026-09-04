import type {
  EvaluationRun,
  EvaluationRunStatus,
  EvaluationTrackId,
} from '../../types/evaluationPlane'

export interface EvaluationRunLedgerFilter {
  query: string
  status: EvaluationRunStatus | 'all'
  track: EvaluationTrackId | 'all'
}

export function filterEvaluationRuns(
  runs: EvaluationRun[],
  { query, status, track }: EvaluationRunLedgerFilter,
) {
  const normalizedQuery = query.trim().toLowerCase()
  return runs.filter((run) => {
    if (status !== 'all' && run.status !== status) return false
    if (track !== 'all' && !run.track_ids.includes(track)) return false
    if (!normalizedQuery) return true
    return [
      run.id,
      run.name,
      run.description,
      run.target_id,
      run.change_profile,
      run.mixture?.entrypoint_model || '',
      run.mixture?.recipe_name || '',
      ...(run.mixture?.aliases || []),
      ...(run.mixture?.model_arms.map((arm) => arm.model) || []),
      ...run.track_ids,
    ]
      .join(' ')
      .toLowerCase()
      .includes(normalizedQuery)
  })
}
