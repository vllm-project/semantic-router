import { useEffect, useState } from 'react'

import type { EvaluationCapacityProfile } from '../types/evaluationCapacityReport'
import type { EvaluationFailureSummary, EvaluationReport } from '../types/evaluationReport'
import type {
  EvaluationDiagnosticArtifactIssue,
  EvaluationReportDiagnosticsState,
} from '../types/evaluationReportDiagnostics'
import { evaluationDiagnosticArtifactIssue } from '../utils/evaluationDiagnosticArtifacts'
import { decodeEvaluationCapacityProfile } from '../utils/evaluationCapacityProfileContract'
import { decodeEvaluationFailureSummary } from '../utils/evaluationFailureSummaryContract'
import { getEvaluationArtifactJSON } from '../utils/evaluationPlaneApi'

function artifactID(report: EvaluationReport | null, name: string): string | null {
  if (!report) return null
  return (
    report.artifacts.find((artifact) => artifact.name.toLowerCase() === name.toLowerCase())?.id ||
    null
  )
}

interface DiagnosticArtifactResult<T> {
  value: T | null
  issue: EvaluationDiagnosticArtifactIssue | null
}

async function loadDiagnosticArtifact<T>(
  runID: string,
  artifactID: string,
  artifactName: string,
  decode: (value: unknown) => T,
  signal: AbortSignal,
): Promise<DiagnosticArtifactResult<T>> {
  try {
    const value = await getEvaluationArtifactJSON<unknown>(runID, artifactID, signal)
    return { value: decode(value), issue: null }
  } catch (reason) {
    return { value: null, issue: evaluationDiagnosticArtifactIssue(artifactName, reason) }
  }
}

export default function useEvaluationReportDiagnostics(
  report: EvaluationReport | null,
): EvaluationReportDiagnosticsState {
  const [state, setState] = useState<EvaluationReportDiagnosticsState>({
    failureSummary: null,
    capacityProfile: null,
    failureSummaryIssue: null,
    capacityProfileIssue: null,
    loading: false,
  })

  useEffect(() => {
    if (!report) {
      setState({
        failureSummary: null,
        capacityProfile: null,
        failureSummaryIssue: null,
        capacityProfileIssue: null,
        loading: false,
      })
      return
    }
    const failureID = artifactID(report, 'failure-summary.json')
    const capacityID = artifactID(report, 'capacity-profile.json')
    if (!failureID && !capacityID) {
      setState({
        failureSummary: null,
        capacityProfile: null,
        failureSummaryIssue: null,
        capacityProfileIssue: null,
        loading: false,
      })
      return
    }

    const controller = new AbortController()
    setState({
      failureSummary: null,
      capacityProfile: null,
      failureSummaryIssue: null,
      capacityProfileIssue: null,
      loading: true,
    })
    const failure = failureID
      ? loadDiagnosticArtifact(
          report.run.id,
          failureID,
          'failure-summary.json',
          decodeEvaluationFailureSummary,
          controller.signal,
        )
      : Promise.resolve<DiagnosticArtifactResult<EvaluationFailureSummary>>({
          value: null,
          issue: null,
        })
    const capacity = capacityID
      ? loadDiagnosticArtifact(
          report.run.id,
          capacityID,
          'capacity-profile.json',
          (value) =>
            decodeEvaluationCapacityProfile(
              value,
              report.run.capacity_slo,
              report.run.capacity_load_protocol,
            ),
          controller.signal,
        )
      : Promise.resolve<DiagnosticArtifactResult<EvaluationCapacityProfile>>({
          value: null,
          issue: null,
        })

    void Promise.all([failure, capacity]).then(([failureResult, capacityResult]) => {
      if (controller.signal.aborted) return
      setState({
        failureSummary: failureResult.value,
        capacityProfile: capacityResult.value,
        failureSummaryIssue: failureResult.issue,
        capacityProfileIssue: capacityResult.issue,
        loading: false,
      })
    })

    return () => controller.abort()
  }, [report])

  return state
}
