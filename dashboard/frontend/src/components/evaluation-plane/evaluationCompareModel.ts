import type { EvaluationRun } from '../../types/evaluationPlane'
import type { EvaluationComparison } from '../../types/evaluationComparison'
import {
  comparisonCohortMismatches,
  eligibleComparisonCandidates,
} from '../../utils/evaluationComparisonCohort'
import { effectiveGateVerdict } from './evaluationPresentation'
import { comparisonRunOptionLabels } from './evaluationRunPresentation'

export interface EvaluationCompareModel {
  completed: Map<string, EvaluationRun>
  candidates: EvaluationRun[]
  candidateLabels: Map<string, string>
  candidate?: EvaluationRun
  baseline?: EvaluationRun
  mismatches: string[]
  lineageMismatch: boolean
  routingRecipeAggregateBoundary: boolean
  invalidPair: boolean
  comparisonVerdict: ReturnType<typeof effectiveGateVerdict> | null
}

interface EvaluationCompareModelInput {
  runs: EvaluationRun[]
  baselineID: string
  candidateID: string
  comparison: EvaluationComparison | null
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  resourcesLoading: boolean
}

export function buildEvaluationCompareModel({
  runs,
  baselineID,
  candidateID,
  comparison,
  runLedgerAvailable,
  runLedgerComplete,
  resourcesLoading,
}: EvaluationCompareModelInput): EvaluationCompareModel {
  const completed = new Map(
    runs.filter((run) => run.status === 'completed').map((run) => [run.id, run]),
  )
  const candidates =
    runLedgerAvailable && runLedgerComplete ? eligibleComparisonCandidates(runs) : []
  const candidate = runLedgerAvailable && runLedgerComplete ? completed.get(candidateID) : undefined
  const baseline = runLedgerAvailable && runLedgerComplete ? completed.get(baselineID) : undefined
  const mismatches = baseline && candidate ? comparisonCohortMismatches(baseline, candidate) : []
  const lineageMismatch = Boolean(candidate && candidate.baseline_run_id !== baselineID)
  const routingRecipeAggregateBoundary = Boolean(
    baseline?.mixture &&
      candidate?.mixture &&
      baseline.mode === 'live' &&
      candidate.mode === 'live' &&
      baseline.track_ids.includes('routing') &&
      candidate.track_ids.includes('routing'),
  )
  return {
    completed,
    candidates,
    candidateLabels: comparisonRunOptionLabels(candidates),
    candidate,
    baseline,
    mismatches,
    lineageMismatch,
    routingRecipeAggregateBoundary,
    invalidPair:
      !runLedgerAvailable ||
      !runLedgerComplete ||
      !baseline ||
      !candidate ||
      resourcesLoading ||
      lineageMismatch ||
      mismatches.length > 0,
    comparisonVerdict: comparison
      ? effectiveGateVerdict(comparison.verdict, comparison.gates)
      : null,
  }
}
