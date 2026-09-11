import type { EvaluationRun } from '../types/evaluationPlane'

export function evaluationErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

export function sortEvaluationRuns(runs: EvaluationRun[]): EvaluationRun[] {
  return [...runs].sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
}

export function mergeEvaluationRuns(
  current: EvaluationRun[],
  nextPage: EvaluationRun[],
): EvaluationRun[] {
  return sortEvaluationRuns([
    ...new Map([...current, ...nextPage].map((run) => [run.id, run])).values(),
  ])
}
