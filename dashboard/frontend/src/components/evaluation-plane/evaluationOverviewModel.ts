import type { EvaluationRun } from '../../types/evaluationPlane'
import type { EvaluationReport } from '../../types/evaluationReport'
import { effectiveGateVerdict, selectHeadlineMetrics } from './evaluationPresentation'

interface EvaluationOverviewModelInput {
  runs: EvaluationRun[]
  latestReport: EvaluationReport | null
  requestedReportRunID: string | null
}

export function buildEvaluationOverviewModel({
  runs,
  latestReport,
  requestedReportRunID,
}: EvaluationOverviewModelInput) {
  const latestRun = runs[0] || null
  const latestCompletedRun = runs.find((run) => run.status === 'completed') || null
  const requestedReportRun = requestedReportRunID
    ? runs.find((run) => run.id === requestedReportRunID) || null
    : null
  const requiredGates = latestReport
    ? latestReport.gates.filter((gate) => gate.disposition === 'required')
    : []

  return {
    running: runs.filter((run) => run.status === 'running' || run.status === 'sealing').length,
    completed: runs.filter((run) => run.status === 'completed').length,
    failed: runs.filter((run) => run.status === 'failed').length,
    latestRun,
    latestEvidenceName:
      latestReport?.run.name || requestedReportRun?.name || latestCompletedRun?.name,
    hasLatestReport: latestReport !== null,
    hasRequestedReportRun: requestedReportRun !== null,
    isDiagnostic: latestReport?.run.evidence_level === 'E0',
    latestVerdict: latestReport
      ? effectiveGateVerdict(latestReport.summary.verdict, latestReport.gates)
      : null,
    headlines: latestReport ? selectHeadlineMetrics(latestReport) : [],
    requiredBlockers: requiredGates.filter(
      (gate) => gate.verdict === 'fail' || gate.verdict === 'unavailable',
    ).length,
  }
}

export type EvaluationOverviewModel = ReturnType<typeof buildEvaluationOverviewModel>
