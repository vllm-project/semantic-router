import type { EvaluationCapacityProfile } from './evaluationCapacityReport'
import type { EvaluationFailureSummary } from './evaluationReport'

export type EvaluationDiagnosticArtifactIssueKind = 'invalid' | 'unavailable'

export interface EvaluationDiagnosticArtifactIssue {
  kind: EvaluationDiagnosticArtifactIssueKind
  artifactName: string
  message: string
}

export interface EvaluationReportDiagnosticsState {
  failureSummary: EvaluationFailureSummary | null
  capacityProfile: EvaluationCapacityProfile | null
  failureSummaryIssue: EvaluationDiagnosticArtifactIssue | null
  capacityProfileIssue: EvaluationDiagnosticArtifactIssue | null
  loading: boolean
}
